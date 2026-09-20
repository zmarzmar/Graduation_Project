'use client'

import { useCallback } from 'react'
import { GUEST_LOGIN_MESSAGE, isGuestUsageLimitError } from '@/lib/api'
import type { AgentEvent, NodeName } from '../types/agent-run'
import type { StreamMode } from '@/store/analysis-store'
import { useAnalysisStore } from '@/store/analysis-store'
import { useAuthStore } from '@/store/auth-store'

// 실행 상태(isRunning)는 store에 있어 페이지를 나가도 유지된다. 취소 핸들도 같은 수명(모듈)에 두어
// 페이지 복귀 후에도 진행 중인 스트림을 취소할 수 있게 한다. (직렬화 불가 객체라 store에는 넣지 않는다)
const controllers = new Map<StreamMode, AbortController>()

export function useAgentStream(mode: StreamMode) {
  const { streams, setStreamState, resetStream } = useAnalysisStore()
  const { openModal } = useAuthStore()
  const { nodeStatuses, nodeLogs, nodeDurations, result, isRunning, cancelled, error, pdfFallbackRequest } = streams[mode]

  const reset = useCallback(() => {
    resetStream(mode)
  }, [mode, resetStream])

  const cancel = useCallback(() => {
    controllers.get(mode)?.abort()
  }, [mode])

  const processStream = useCallback(
    async (response: Response) => {
      if (!response.ok) {
        const text = await response.text()
        // FastAPI 오류는 {"detail": "..."} 형식 — 사용자에게는 메시지만 보여준다
        let detail = text
        try {
          const parsed = JSON.parse(text)
          if (typeof parsed?.detail === 'string') detail = parsed.detail
        } catch {
          // JSON이 아니면 원문 그대로 표시
        }
        setStreamState(mode, { error: `요청 실패 (${response.status}): ${detail}`, isRunning: false })
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        setStreamState(mode, { error: '스트림을 읽을 수 없습니다.', isRunning: false })
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const event: AgentEvent = JSON.parse(line.slice(6))

              if (event.event === 'node_start' && event.node) {
                const state = useAnalysisStore.getState().streams[mode]
                const node = event.node as NodeName
                setStreamState(mode, {
                  nodeStatuses: { ...state.nodeStatuses, [node]: 'running' },
                  nodeDurations: { ...state.nodeDurations, [node]: null },
                })
              } else if (event.event === 'log' && event.node && event.message) {
                const node = event.node as NodeName
                const cur = useAnalysisStore.getState().streams[mode].nodeLogs
                setStreamState(mode, { nodeLogs: { ...cur, [node]: [...cur[node], event.message!] } })
              } else if (event.event === 'node_done' && event.node) {
                const status = event.error ? 'error' : 'done'
                const state = useAnalysisStore.getState().streams[mode]
                const node = event.node as NodeName
                setStreamState(mode, {
                  nodeStatuses: { ...state.nodeStatuses, [node]: status },
                  ...(typeof event.elapsed_ms === 'number'
                    ? { nodeDurations: { ...state.nodeDurations, [node]: event.elapsed_ms } }
                    : {}),
                })
              } else if (event.event === 'complete') {
                const currentError = useAnalysisStore.getState().streams[mode].error
                setStreamState(mode, { result: event.result ?? null, ...(currentError ? {} : { error: null }) })
              } else if (event.event === 'error') {
                setStreamState(mode, { error: event.message ?? '알 수 없는 오류가 발생했습니다.' })
              } else if (event.event === 'pdf_fallback_required') {
                const state = useAnalysisStore.getState().streams[mode]
                const nextStatuses = { ...state.nodeStatuses, analyzer: 'error' as const }
                const message = event.message ?? 'PDF를 다운로드하지 못했습니다. 초록만으로 분석을 진행할까요?'
                setStreamState(mode, {
                  nodeStatuses: nextStatuses,
                  pdfFallbackRequest: { message, reason: event.reason },
                })
              }
            } catch {
              // JSON 파싱 실패 라인은 무시
            }
          }
        }
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') {
          // 취소 시 running 상태인 노드를 pending으로 되돌림
          const cur = useAnalysisStore.getState().streams[mode].nodeStatuses
          const next = { ...cur }
          for (const key of Object.keys(next) as NodeName[]) {
            if (next[key] === 'running') next[key] = 'pending'
          }
          setStreamState(mode, { cancelled: true, nodeStatuses: next })
        } else {
          throw e
        }
      } finally {
        reader.releaseLock()
        setStreamState(mode, { isRunning: false })
      }
    },
    [mode, setStreamState],
  )

  const startStream = useCallback(
    async (fetchFn: (signal: AbortSignal) => Promise<Response>) => {
      // 같은 모드가 이미 실행 중이면 시작하지 않는다 — 덮어쓰면 기존 요청의 취소 핸들을 잃는다.
      if (controllers.has(mode)) return

      const controller = new AbortController()
      controllers.set(mode, controller)

      reset()
      setStreamState(mode, { isRunning: true, pdfFallbackRequest: null })

      try {
        const response = await fetchFn(controller.signal)
        await processStream(response)
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') {
          setStreamState(mode, { cancelled: true, isRunning: false })
        } else if (isGuestUsageLimitError(e)) {
          openModal('login', GUEST_LOGIN_MESSAGE)
          setStreamState(mode, { error: GUEST_LOGIN_MESSAGE, isRunning: false })
        } else {
          setStreamState(mode, { error: e instanceof Error ? e.message : '네트워크 오류가 발생했습니다.', isRunning: false })
        }
      } finally {
        // 응답 전에 실패한 경우까지 정리한다. 현재 실행의 컨트롤러만 지운다.
        if (controllers.get(mode) === controller) controllers.delete(mode)
      }
    },
    [reset, processStream, openModal, mode, setStreamState],
  )

  // 스트림 없이 캐시된 결과를 바로 표시한다
  const showCachedResult = useCallback(
    (cachedResult: import('@/lib/types/agent-run').AgentResult) => {
      resetStream(mode)
      setStreamState(mode, { result: cachedResult })
    },
    [mode, resetStream, setStreamState],
  )

  return {
    nodeStatuses,
    nodeLogs,
    nodeDurations,
    result,
    isRunning,
    cancelled,
    error,
    pdfFallbackRequest,
    startStream,
    cancel,
    reset,
    showCachedResult,
  }
}
