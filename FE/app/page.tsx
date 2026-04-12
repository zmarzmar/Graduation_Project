'use client'

import { useRef, useState } from 'react'
import { Search, Upload, TrendingUp, Play, RotateCcw, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AgentPipeline } from '@/components/agent/AgentPipeline'
import { ResultsPanel } from '@/components/agent/ResultsPanel'
import { useAgentStream } from '@/lib/hooks/useAgentStream'
import { useAnalysisStore } from '@/store/analysis-store'
import { runSearchAgent, runPdfAgent, runTrendAgent, runAnalyzeAgent } from '@/lib/api'
import type { ArxivPaper, AgentMode } from '@/lib/types/agent-run'

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
  // 페이지 이탈 후 복귀 시 상태를 유지하기 위해 Zustand 스토어 사용
  const {
    mode, setMode,
    query, setQuery,
    topic, setTopic,
    searchPipelineMode, setSearchPipelineMode,
    searchedPapers, setSearchedPapers,
  } = useAnalysisStore()

  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 모드별 독립 스트림 — 탭 전환 및 페이지 이탈 후에도 각자의 상태 유지
  const searchStream = useAgentStream('search')
  const pdfStream = useAgentStream('pdf')
  const trendStream = useAgentStream('trend')

  // 현재 탭에 해당하는 스트림
  const activeStream = mode === 'search' ? searchStream : mode === 'pdf' ? pdfStream : trendStream

  // 검색 탭 파이프라인 모드 — 검색 후 분석 시작하면 'analyze'로 전환
  const pipelineMode: AgentMode = mode === 'search' ? searchPipelineMode : mode === 'pdf' ? 'pdf' : 'trend'

  const canRun =
    !activeStream.isRunning &&
    ((mode === 'search' && query.trim().length > 0) ||
      (mode === 'pdf' && file !== null) ||
      (mode === 'trend' && topic.trim().length > 0))

  function handleRun() {
    if (mode === 'search') {
      setSearchedPapers([])
      setSearchPipelineMode('search')
      searchStream.startStream((signal) => runSearchAgent(query.trim(), signal))
    } else if (mode === 'pdf' && file) {
      pdfStream.startStream((signal) => runPdfAgent(file, signal))
    } else if (mode === 'trend') {
      trendStream.startStream((signal) => runTrendAgent(topic.trim(), signal))
    }
  }

  function handleAnalyze(paper: ArxivPaper) {
    if (searchStream.result?.papers.length) {
      setSearchedPapers(searchStream.result.papers)
    }
    setSearchPipelineMode('analyze')
    searchStream.startStream((signal) => runAnalyzeAgent(paper, query.trim(), signal))
  }

  function handleReset() {
    activeStream.reset()
    if (mode === 'search') {
      setSearchedPapers([])
      setSearchPipelineMode('search')
      setQuery('LoRA fine-tuning')
    } else if (mode === 'pdf') {
      setFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    } else if (mode === 'trend') {
      setTopic('large language model')
    }
  }

  const showPipeline =
    activeStream.isRunning ||
    activeStream.cancelled ||
    activeStream.result !== null ||
    activeStream.error !== null

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* 타이틀 */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">AI Research Analyst</h1>
        <p className="mt-1 text-sm text-gray-500">
          최신 AI 논문을 분석하고 PyTorch 코드 스켈레톤을 자동으로 재현합니다.
        </p>
      </div>

      {/* 모드 선택 */}
      <div className="flex gap-2">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id as AgentMode)}
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
              disabled={activeStream.isRunning}
            />
            <div className="flex flex-wrap gap-1.5">
              {SEARCH_EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setQuery(ex)}
                  disabled={activeStream.isRunning}
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
              disabled={activeStream.isRunning}
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
              disabled={activeStream.isRunning}
            />
            <div className="flex flex-wrap gap-1.5">
              {TREND_EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setTopic(ex)}
                  disabled={activeStream.isRunning}
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
          <Button onClick={handleRun} disabled={!canRun} className="gap-2">
            {activeStream.isRunning ? (
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

          {activeStream.isRunning && (
            <Button variant="destructive" onClick={activeStream.cancel} className="gap-2">
              <Square className="h-4 w-4" />
              취소
            </Button>
          )}

          {!activeStream.isRunning && (activeStream.result || activeStream.error || activeStream.cancelled) && (
            <Button variant="outline" onClick={handleReset} className="gap-2">
              <RotateCcw className="h-4 w-4" />
              초기화
            </Button>
          )}
        </div>
      </div>

      {/* Empty State — 분석 시작 전에만 표시 */}
      {!showPipeline && (
        <div className="space-y-3">
          <p className="text-sm font-semibold text-gray-500">어떻게 작동하나요?</p>
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg border border-gray-200 bg-white p-4 space-y-2">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                  <Search className="h-4 w-4" />
                </div>
                <p className="text-sm font-semibold text-gray-800">키워드 검색</p>
              </div>
              <p className="text-xs leading-relaxed text-gray-500 break-keep">
                키워드를 입력하면 여러 논문 DB에서 관련 논문을 수집하고 PyTorch 코드를 자동 재현합니다.
              </p>
            </div>
            <div className="rounded-lg border border-gray-200 bg-white p-4 space-y-2">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-purple-100 text-purple-600">
                  <Upload className="h-4 w-4" />
                </div>
                <p className="text-sm font-semibold text-gray-800">PDF 업로드</p>
              </div>
              <p className="text-xs leading-relaxed text-gray-500 break-keep">
                논문 PDF를 업로드하면 전문을 분석해 핵심 수식·강점·한계를 정리하고 코드를 재현합니다.
              </p>
            </div>
            <div className="rounded-lg border border-gray-200 bg-white p-4 space-y-2">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-green-100 text-green-600">
                  <TrendingUp className="h-4 w-4" />
                </div>
                <p className="text-sm font-semibold text-gray-800">트렌드 브리핑</p>
              </div>
              <p className="text-xs leading-relaxed text-gray-500 break-keep">
                관심 분야를 입력하면 HuggingFace · arXiv 기반 최신 논문 Top 10과 키워드 리포트를 생성합니다.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 취소 메시지 */}
      {activeStream.cancelled && !activeStream.isRunning && (
        <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-800">
          분석이 취소됐습니다. 완료된 노드까지의 결과는 아래에서 확인할 수 있습니다.
        </div>
      )}

      {/* 오류 메시지 */}
      {activeStream.error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {activeStream.error}
        </div>
      )}

      {/* 파이프라인 진행 상황 */}
      {showPipeline && (
        <AgentPipeline
          nodeStatuses={activeStream.nodeStatuses}
          nodeLogs={activeStream.nodeLogs}
          mode={pipelineMode}
        />
      )}

      {/* 결과 패널 */}
      {activeStream.result && (
        <ResultsPanel
          result={activeStream.result}
          searchedPapers={mode === 'search' ? searchedPapers : undefined}
          onAnalyze={mode === 'search' && !searchStream.isRunning ? handleAnalyze : undefined}
        />
      )}
    </div>
  )
}
