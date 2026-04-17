'use client'

import { useCallback, useRef } from 'react'
import type { AgentEvent, NodeName } from '../types/agent-run'
import type { StreamMode } from '@/store/analysis-store'
import { useAnalysisStore } from '@/store/analysis-store'

export function useAgentStream(mode: StreamMode) {
  const { streams, setStreamState, resetStream } = useAnalysisStore()
  const { nodeStatuses, nodeLogs, result, isRunning, cancelled, error } = streams[mode]

  // abortRef는 컴포넌트 로컬 — 페이지 이탈 후 복귀 시 취소 버튼은 동작 안 하지만 스트림은 계속 실행됨
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    resetStream(mode)
  }, [mode, resetStream])

  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const processStream = useCallback(
    async (response: Response) => {
      if (!response.ok) {
        const text = await response.text()
        setStreamState(mode, { error: `요청 실패 (${response.status}): ${text}`, isRunning: false })
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
                const cur = useAnalysisStore.getState().streams[mode].nodeStatuses
                setStreamState(mode, { nodeStatuses: { ...cur, [event.node as NodeName]: 'running' } })
              } else if (event.event === 'log' && event.node && event.message) {
                const node = event.node as NodeName
                const cur = useAnalysisStore.getState().streams[mode].nodeLogs
                setStreamState(mode, { nodeLogs: { ...cur, [node]: [...cur[node], event.message!] } })
              } else if (event.event === 'node_done' && event.node) {
                const status = event.error ? 'error' : 'done'
                const cur = useAnalysisStore.getState().streams[mode].nodeStatuses
                setStreamState(mode, { nodeStatuses: { ...cur, [event.node as NodeName]: status } })
              } else if (event.event === 'complete') {
                const currentError = useAnalysisStore.getState().streams[mode].error
                setStreamState(mode, { result: event.result ?? null, ...(currentError ? {} : { error: null }) })
              } else if (event.event === 'error') {
                setStreamState(mode, { error: event.message ?? '알 수 없는 오류가 발생했습니다.' })
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
      reset()
      setStreamState(mode, { isRunning: true })

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const response = await fetchFn(controller.signal)
        await processStream(response)
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') {
          setStreamState(mode, { cancelled: true, isRunning: false })
        } else {
          setStreamState(mode, { error: e instanceof Error ? e.message : '네트워크 오류가 발생했습니다.', isRunning: false })
        }
      }
    },
    [reset, processStream, mode, setStreamState],
  )

  return { nodeStatuses, nodeLogs, result, isRunning, cancelled, error, startStream, cancel, reset }
}
