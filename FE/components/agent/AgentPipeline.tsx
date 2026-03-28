'use client'

import { CheckCircle, Circle, Loader2, XCircle } from 'lucide-react'
import type { AgentMode, NodeLogs, NodeName, NodeStatuses } from '@/lib/types/agent-run'

interface AgentPipelineProps {
  nodeStatuses: NodeStatuses
  nodeLogs: NodeLogs
  mode: AgentMode
}

interface NodeConfig {
  name: NodeName
  label: string
  description: string
}

const ALL_NODES: NodeConfig[] = [
  { name: 'planner',        label: 'Planner',        description: '계획 수립' },
  { name: 'researcher',     label: 'Researcher',     description: '논문 수집' },
  { name: 'trend_analyzer', label: 'TrendAnalyzer',  description: '트렌드 분석' },
  { name: 'analyzer',       label: 'Analyzer',       description: '논문 분석' },
  { name: 'coder',          label: 'Coder',          description: '코드 생성' },
  { name: 'reviewer',       label: 'Reviewer',       description: '코드 검증' },
]

// trend 모드는 TrendAnalyzer 사용, Coder/Reviewer 없음
const TREND_NODES: NodeName[] = ['planner', 'researcher', 'trend_analyzer']

function NodeIcon({ status }: { status: string }) {
  switch (status) {
    case 'running': return <Loader2 className="h-4 w-4 animate-spin text-blue-500 flex-shrink-0" />
    case 'done':    return <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
    case 'error':   return <XCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
    default:        return <Circle className="h-4 w-4 text-gray-300 flex-shrink-0" />
  }
}

function nodeBorderColor(status: string): string {
  switch (status) {
    case 'running': return 'border-blue-400'
    case 'done':    return 'border-green-400'
    case 'error':   return 'border-red-400'
    default:        return 'border-gray-200'
  }
}

function nodeHeaderBg(status: string): string {
  switch (status) {
    case 'running': return 'bg-blue-50'
    case 'done':    return 'bg-green-50'
    case 'error':   return 'bg-red-50'
    default:        return 'bg-gray-50'
  }
}

export function AgentPipeline({ nodeStatuses, nodeLogs, mode }: AgentPipelineProps) {
  // trend 모드는 Coder/Reviewer 숨김
  const visibleNodes = mode === 'trend'
    ? ALL_NODES.filter((n) => TREND_NODES.includes(n.name))
    : ALL_NODES

  const hasAnyActivity = visibleNodes.some((n) => nodeStatuses[n.name] !== 'pending')

  if (!hasAnyActivity) return null

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">파이프라인 진행 상황</h3>
      <div className="flex flex-col gap-2">
        {visibleNodes.map((node) => {
          const status = nodeStatuses[node.name]
          const logs = nodeLogs[node.name]
          const isActive = status !== 'pending'

          if (!isActive) return null

          return (
            <div
              key={node.name}
              className={`rounded-lg border-2 overflow-hidden transition-all duration-300 ${nodeBorderColor(status)}`}
            >
              {/* 노드 헤더 */}
              <div className={`flex items-center gap-2 px-3 py-2 ${nodeHeaderBg(status)}`}>
                <NodeIcon status={status} />
                <span className="text-sm font-semibold text-gray-800">{node.label}</span>
                <span className="text-xs text-gray-500">— {node.description}</span>
              </div>

              {/* 로그 메시지 */}
              {logs.length > 0 && (
                <div className="px-3 py-2 bg-gray-950 font-mono">
                  {logs.map((msg, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs text-gray-300 leading-relaxed">
                      <span className="text-gray-600 select-none mt-0.5">›</span>
                      <span>{msg}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
