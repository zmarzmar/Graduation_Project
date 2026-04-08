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
  { name: 'planner',        label: 'Planner',       description: '계획 수립' },
  { name: 'researcher',     label: 'Researcher',    description: '논문 수집' },
  { name: 'trend_analyzer', label: 'TrendAnalyzer', description: '트렌드 분석' },
  { name: 'analyzer',       label: 'Analyzer',      description: '논문 분석' },
  { name: 'coder',          label: 'Coder',         description: '코드 생성' },
  { name: 'reviewer',       label: 'Reviewer',      description: '코드 검증' },
]

const SEARCH_NODES: NodeName[]  = ['planner', 'researcher']
const ANALYZE_NODES: NodeName[] = ['analyzer', 'coder', 'reviewer']
const PDF_NODES: NodeName[]     = ['planner', 'analyzer', 'coder', 'reviewer']
const TREND_NODES: NodeName[]   = ['planner', 'researcher', 'trend_analyzer']

function NodeIcon({ status }: { status: string }) {
  switch (status) {
    case 'running': return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
    case 'done':    return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'error':   return <XCircle className="h-4 w-4 text-red-500" />
    default:        return <Circle className="h-4 w-4 text-gray-300" />
  }
}

export function AgentPipeline({ nodeStatuses, nodeLogs, mode }: AgentPipelineProps) {
  const visibleNodes =
    mode === 'trend'   ? ALL_NODES.filter((n) => TREND_NODES.includes(n.name))
    : mode === 'search'  ? ALL_NODES.filter((n) => SEARCH_NODES.includes(n.name))
    : mode === 'analyze' ? ALL_NODES.filter((n) => ANALYZE_NODES.includes(n.name))
    : mode === 'pdf'     ? ALL_NODES.filter((n) => PDF_NODES.includes(n.name))
    : ALL_NODES

  const hasAnyActivity = visibleNodes.some((n) => nodeStatuses[n.name] !== 'pending')
  if (!hasAnyActivity) return null

  // 로그가 있는 모든 활성 노드를 순서대로 표시
  const logNodes = visibleNodes.filter(
    (n) => nodeStatuses[n.name] !== 'pending' && nodeLogs[n.name].length > 0
  )

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 space-y-4">
      <h3 className="text-sm font-semibold text-gray-700">파이프라인 진행 상황</h3>

      {/* 수평 스텝 바 */}
      <div className="flex items-center">
        {visibleNodes.map((node, idx) => {
          const status = nodeStatuses[node.name]
          const isActive = status === 'running'
          const isDone   = status === 'done'
          const isError  = status === 'error'

          return (
            <div key={node.name} className="flex items-center flex-1 min-w-0">
              {/* 노드 */}
              <div className="flex flex-col items-center gap-1 flex-shrink-0">
                {/* 아이콘 원 */}
                <div className={`flex h-9 w-9 items-center justify-center rounded-full border-2 transition-all duration-300 ${
                  isActive ? 'border-blue-400 bg-blue-50'
                  : isDone  ? 'border-green-400 bg-green-50'
                  : isError ? 'border-red-400 bg-red-50'
                  : 'border-gray-200 bg-gray-50'
                }`}>
                  <NodeIcon status={status} />
                </div>
                {/* 레이블 */}
                <div className="text-center">
                  <p className={`text-[11px] font-semibold leading-tight ${
                    isActive ? 'text-blue-600'
                    : isDone  ? 'text-green-600'
                    : isError ? 'text-red-600'
                    : 'text-gray-400'
                  }`}>
                    {node.label}
                  </p>
                  <p className="text-[10px] text-gray-400 leading-tight">{node.description}</p>
                </div>
              </div>

              {/* 화살표 연결선 (마지막 노드 제외) */}
              {idx < visibleNodes.length - 1 && (
                <div className="flex-1 flex items-center px-1 mb-5">
                  <div className={`h-0.5 flex-1 transition-all duration-500 ${
                    isDone ? 'bg-green-300' : 'bg-gray-200'
                  }`} />
                  <svg
                    className={`h-3 w-3 flex-shrink-0 transition-colors duration-500 ${
                      isDone ? 'text-green-300' : 'text-gray-200'
                    }`}
                    viewBox="0 0 12 12"
                    fill="currentColor"
                  >
                    <path d="M2 6h7M6 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* 노드별 로그 — 활성화된 순서대로 누적 표시 */}
      {logNodes.length > 0 && (
        <div className="space-y-2">
          {logNodes.map((node) => {
            const status = nodeStatuses[node.name]
            const logs = nodeLogs[node.name]
            return (
              <div key={node.name} className={`rounded-lg border-2 overflow-hidden transition-all duration-300 ${
                status === 'running' ? 'border-blue-400'
                : status === 'done'  ? 'border-green-400'
                : 'border-red-400'
              }`}>
                {/* 로그 헤더 */}
                <div className={`flex items-center gap-2 px-3 py-2 ${
                  status === 'running' ? 'bg-blue-50'
                  : status === 'done'  ? 'bg-green-50'
                  : 'bg-red-50'
                }`}>
                  <NodeIcon status={status} />
                  <span className="text-sm font-semibold text-gray-800">{node.label}</span>
                  <span className="text-xs text-gray-500">— {node.description}</span>
                </div>

                {/* 로그 메시지 */}
                <div className="px-3 py-2 bg-gray-950 font-mono">
                  {logs.map((msg, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs text-gray-300 leading-relaxed">
                      <span className="text-gray-600 select-none mt-0.5">›</span>
                      <span>{msg}</span>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
