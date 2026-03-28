'use client'

import { useCallback, useRef, useState } from 'react'
import type { AgentEvent, AgentResult, NodeLogs, NodeName, NodeStatuses } from '../types/agent-run'

const INITIAL_NODE_STATUSES: NodeStatuses = {
  planner: 'pending',
  researcher: 'pending',
  trend_analyzer: 'pending',
  analyzer: 'pending',
  coder: 'pending',
  reviewer: 'pending',
}

const INITIAL_NODE_LOGS: NodeLogs = {
  planner: [],
  researcher: [],
  trend_analyzer: [],
  analyzer: [],
  coder: [],
  reviewer: [],
}

export function useAgentStream() {
  const [nodeStatuses, setNodeStatuses] = useState<NodeStatuses>(INITIAL_NODE_STATUSES)
  const [nodeLogs, setNodeLogs] = useState<NodeLogs>(INITIAL_NODE_LOGS)
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [result, setResult] = useState<AgentResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [cancelled, setCancelled] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 현재 진행 중인 요청을 취소하기 위한 AbortController 참조
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setNodeStatuses(INITIAL_NODE_STATUSES)
    setNodeLogs(INITIAL_NODE_LOGS)
    setEvents([])
    setResult(null)
    setError(null)
    setCancelled(false)
  }, [])

  /** 진행 중인 분석을 취소한다 */
  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const processStream = useCallback(async (response: Response) => {
    if (!response.ok) {
      const text = await response.text()
      setError(`요청 실패 (${response.status}): ${text}`)
      setIsRunning(false)
      return
    }

    const reader = response.body?.getReader()
    if (!reader) {
      setError('스트림을 읽을 수 없습니다.')
      setIsRunning(false)
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
            setEvents((prev) => [...prev, event])

            if (event.event === 'node_start' && event.node) {
              setNodeStatuses((prev) => ({ ...prev, [event.node as NodeName]: 'running' }))
            } else if (event.event === 'log' && event.node && event.message) {
              const node = event.node as NodeName
              setNodeLogs((prev) => ({ ...prev, [node]: [...prev[node], event.message!] }))
            } else if (event.event === 'node_done' && event.node) {
              const status = event.error ? 'error' : 'done'
              setNodeStatuses((prev) => ({ ...prev, [event.node as NodeName]: status }))
            } else if (event.event === 'complete') {
              setResult(event.result ?? null)
            } else if (event.event === 'error') {
              setError(event.message ?? '알 수 없는 오류가 발생했습니다.')
            }
          } catch {
            // JSON 파싱 실패 라인은 무시
          }
        }
      }
    } catch (e) {
      // AbortError는 사용자가 직접 취소한 것 — 오류로 처리하지 않음
      if (e instanceof Error && e.name === 'AbortError') {
        setCancelled(true)
        // 진행된 노드 상태 중 'running'인 것을 'pending'으로 되돌림
        setNodeStatuses((prev) => {
          const next = { ...prev }
          for (const key of Object.keys(next) as NodeName[]) {
            if (next[key] === 'running') next[key] = 'pending'
          }
          return next
        })
      } else {
        throw e
      }
    } finally {
      reader.releaseLock()
      setIsRunning(false)
    }
  }, [])

  /**
   * 에이전트 스트리밍을 시작한다.
   * fetchFn은 AbortSignal을 받아 fetch Promise를 반환하는 함수여야 한다.
   * 예: (signal) => runSearchAgent("LoRA", signal)
   */
  const startStream = useCallback(
    async (fetchFn: (signal: AbortSignal) => Promise<Response>) => {
      reset()
      setIsRunning(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const response = await fetchFn(controller.signal)
        await processStream(response)
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') {
          setCancelled(true)
          setIsRunning(false)
        } else {
          setError(e instanceof Error ? e.message : '네트워크 오류가 발생했습니다.')
          setIsRunning(false)
        }
      }
    },
    [reset, processStream],
  )

  return { nodeStatuses, nodeLogs, events, result, isRunning, cancelled, error, startStream, cancel, reset }
}
