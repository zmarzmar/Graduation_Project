'use client'

import { useRef, useState } from 'react'
import { Search, Upload, TrendingUp, Play, RotateCcw, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AgentPipeline } from '@/components/agent/AgentPipeline'
import { ResultsPanel } from '@/components/agent/ResultsPanel'
import { useAgentStream } from '@/lib/hooks/useAgentStream'
import { runSearchAgent, runPdfAgent, runTrendAgent } from '@/lib/api'
import type { AgentMode } from '@/lib/types/agent-run'

const MODES: { id: AgentMode; label: string; icon: React.ReactNode; description: string }[] = [
  {
    id: 'search',
    label: '키워드 검색',
    icon: <Search className="h-4 w-4" />,
    description: 'arXiv + Semantic Scholar 검색 후 코드 재현',
  },
  {
    id: 'pdf',
    label: 'PDF 업로드',
    icon: <Upload className="h-4 w-4" />,
    description: '논문 PDF를 직접 업로드해 분석',
  },
  {
    id: 'trend',
    label: '트렌드 브리핑',
    icon: <TrendingUp className="h-4 w-4" />,
    description: 'HuggingFace 기반 최신 트렌드 리포트',
  },
]

const SEARCH_EXAMPLES = ['LoRA fine-tuning', 'diffusion model', 'attention mechanism', 'graph neural network', 'retrieval augmented generation']
const TREND_EXAMPLES = ['large language model', 'vision transformer', 'reinforcement learning', 'multimodal', 'agent']

export default function HomePage() {
  const [mode, setMode] = useState<AgentMode>('search')
  const [query, setQuery] = useState('LoRA fine-tuning')
  const [topic, setTopic] = useState('large language model')
  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 파이프라인에 실제로 사용된 모드 — 탭 전환과 무관하게 유지
  const [pipelineMode, setPipelineMode] = useState<AgentMode>('search')

  const { nodeStatuses, nodeLogs, result, isRunning, cancelled, error, startStream, cancel, reset } =
    useAgentStream()

  const canRun =
    !isRunning &&
    ((mode === 'search' && query.trim().length > 0) ||
      (mode === 'pdf' && file !== null) ||
      (mode === 'trend' && topic.trim().length > 0))

  function handleRun() {
    setPipelineMode(mode)
    if (mode === 'search') {
      startStream((signal) => runSearchAgent(query.trim(), signal))
    } else if (mode === 'pdf' && file) {
      startStream((signal) => runPdfAgent(file, signal))
    } else if (mode === 'trend') {
      startStream((signal) => runTrendAgent(topic.trim(), signal))
    }
  }

  function handleReset() {
    reset()
    setQuery('LoRA fine-tuning')
    setTopic('large language model')
    setFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const showPipeline = isRunning || cancelled || result !== null || error !== null

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* 타이틀 */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">AI Research Analyst</h1>
        <p className="mt-1 text-sm text-gray-500">
          최신 AI 논문을 분석하고 PyTorch 코드 스켈레톤을 자동으로 재현합니다.
        </p>
      </div>

      {/* 모드 선택 — 탭 전환 시 파이프라인 유지 */}
      <div className="flex gap-2">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`flex flex-1 flex-col items-center gap-1 rounded-lg border-2 p-3 text-sm transition-all ${
              mode === m.id
                ? 'border-blue-500 bg-blue-50 text-blue-700'
                : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
            }`}
          >
            {m.icon}
            <span className="font-medium">{m.label}</span>
            <span className="text-[10px] text-center opacity-70">{m.description}</span>
          </button>
        ))}
      </div>

      {/* 입력 영역 */}
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        {mode === 'search' && (
          <div className="space-y-2">
            <Input
              placeholder="키워드를 입력하세요"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && canRun && handleRun()}
              disabled={isRunning}
            />
            <div className="flex flex-wrap gap-1.5">
              {SEARCH_EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setQuery(ex)}
                  disabled={isRunning}
                  className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
                    query === ex
                      ? 'border-blue-400 bg-blue-50 text-blue-700'
                      : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === 'pdf' && (
          <div className="space-y-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              disabled={isRunning}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-gray-500 file:mr-3 file:rounded-md file:border-0 file:bg-blue-50 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-blue-600 hover:file:bg-blue-100"
            />
            {file && (
              <p className="text-xs text-gray-500">
                선택된 파일: {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </div>
        )}

        {mode === 'trend' && (
          <div className="space-y-2">
            <Input
              placeholder="트렌드 주제를 입력하세요"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && canRun && handleRun()}
              disabled={isRunning}
            />
            <div className="flex flex-wrap gap-1.5">
              {TREND_EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setTopic(ex)}
                  disabled={isRunning}
                  className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
                    topic === ex
                      ? 'border-blue-400 bg-blue-50 text-blue-700'
                      : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* 버튼 영역 */}
        <div className="mt-3 flex gap-2">
          {/* 분석 시작 버튼 */}
          <Button onClick={handleRun} disabled={!canRun} className="gap-2">
            {isRunning ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                분석 중...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                분석 시작
              </>
            )}
          </Button>

          {/* 취소 버튼 — 분석 중일 때만 표시 */}
          {isRunning && (
            <Button variant="destructive" onClick={cancel} className="gap-2">
              <Square className="h-4 w-4" />
              취소
            </Button>
          )}

          {/* 초기화 버튼 — 결과/오류/취소 상태일 때 표시 */}
          {!isRunning && (result || error || cancelled) && (
            <Button variant="outline" onClick={handleReset} className="gap-2">
              <RotateCcw className="h-4 w-4" />
              초기화
            </Button>
          )}
        </div>
      </div>

      {/* 취소 메시지 */}
      {cancelled && !isRunning && (
        <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-800">
          분석이 취소됐습니다. 완료된 노드까지의 결과는 아래에서 확인할 수 있습니다.
        </div>
      )}

      {/* 오류 메시지 */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* 파이프라인 진행 상황 — pipelineMode 기준으로 표시 */}
      {showPipeline && (
        <AgentPipeline nodeStatuses={nodeStatuses} nodeLogs={nodeLogs} mode={pipelineMode} />
      )}

      {/* 결과 패널 */}
      {result && <ResultsPanel result={result} />}
    </div>
  )
}
