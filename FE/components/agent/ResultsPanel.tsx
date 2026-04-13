'use client'

import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { BlockMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import type { AgentResult, ArxivPaper, TrendAnalysis } from '@/lib/types/agent-run'

/** LaTeX 렌더링 실패 시 plain text로 fallback */
function FormulaBlock({ latex }: { latex: string }) {
  try {
    return <BlockMath math={latex} />
  } catch {
    return <code className="font-mono text-xs text-gray-700">{latex}</code>
  }
}

interface ResultsPanelProps {
  result: AgentResult
  searchedPapers?: ArxivPaper[]  // 검색 모드에서 수집된 전체 논문 목록 (분석 후에도 유지)
  onAnalyze?: (paper: ArxivPaper) => void
}

function PaperCard({ paper, summary, onAnalyze }: { paper: ArxivPaper; summary?: string; onAnalyze?: (paper: ArxivPaper) => void }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-sm font-semibold text-blue-600 hover:underline"
          >
            {paper.title}
            <ExternalLink className="h-3 w-3 flex-shrink-0" />
          </a>
          <p className="mt-1 text-xs text-gray-500">
            {paper.authors.slice(0, 3).join(', ')}
            {paper.authors.length > 3 && ' 외'}
          </p>
          {paper.published_at && (
            <p className="mt-0.5 text-xs text-gray-400">
              {new Date(paper.published_at).getFullYear()}년
            </p>
          )}
        </div>
        <div className="flex flex-shrink-0 flex-col items-end gap-1">
          {paper.citation_count != null && (
            <Badge variant="secondary" className="text-xs">
              인용 {paper.citation_count}
            </Badge>
          )}
          {'upvotes' in paper && paper.upvotes != null && (
            <Badge variant="secondary" className="text-xs">
              ↑ {paper.upvotes}
            </Badge>
          )}
        </div>
      </div>

      {/* AI 한 줄 요약 (트렌드 모드) */}
      {summary && (
        <p className="mt-2 rounded-md bg-blue-50 px-3 py-1.5 text-xs text-blue-700">
          💡 {summary}
        </p>
      )}

      {/* 기존 TL;DR (S2 제공) */}
      {!summary && paper.tldr && (
        <p className="mt-2 text-xs text-gray-600 italic">"{paper.tldr}"</p>
      )}

      {/* 초록 토글 + 분석 버튼 */}
      <div className="mt-2 flex items-center justify-between">
        {paper.abstract ? (
          <button
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            초록 {expanded ? '접기' : '보기'}
          </button>
        ) : (
          <span className="text-xs text-gray-300">초록 없음</span>
        )}
        {onAnalyze && (
          <button
            onClick={() => onAnalyze(paper)}
            className="rounded-md bg-blue-500 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-blue-600"
          >
            이 논문 분석하기
          </button>
        )}
      </div>
      {expanded && paper.abstract && (
        <p className="mt-2 text-xs leading-relaxed text-gray-600">{paper.abstract}</p>
      )}
    </div>
  )
}

function TrendAnalysisPanel({ analysis }: { analysis: TrendAnalysis }) {
  return (
    <div className="space-y-4">
      {/* 트렌드 키워드 */}
      {analysis.trending_keywords.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-gray-700">🔥 이번 주 트렌드 키워드</h4>
          <div className="flex flex-wrap gap-2">
            {analysis.trending_keywords.map((kw, i) => (
              <div
                key={i}
                className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2"
              >
                <span className="block text-sm font-semibold text-blue-700">{kw.keyword}</span>
                <span className="block text-xs text-blue-500">{kw.explanation}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 논문별 한 줄 요약 */}
      {analysis.paper_summaries.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-gray-700">📋 논문별 한 줄 요약</h4>
          <div className="space-y-2">
            {analysis.paper_summaries.map((item, i) => (
              <div key={i} className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2">
                <p className="text-xs font-medium text-gray-700 line-clamp-1">{item.title}</p>
                <p className="mt-0.5 text-xs text-gray-500">→ {item.summary}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export function ResultsPanel({ result, searchedPapers, onAnalyze }: ResultsPanelProps) {
  const isTrend = result.mode === 'trend'
  const hasCoding = !!result.generated_code
  const hasTrendAnalysis = isTrend && !!result.trend_analysis
  const hasAnalysis = !isTrend && !!(
    result.paper_summary ||
    (result.paper_review && Object.keys(result.paper_review).length > 0) ||
    result.key_formulas?.length
  )

  type TabId = 'papers' | 'analysis' | 'trend' | 'code' | 'review'
  const tabs: TabId[] = [
    'papers',
    ...(hasAnalysis ? ['analysis' as TabId] : []),
    ...(hasTrendAnalysis ? ['trend' as TabId] : []),
    ...(hasCoding ? ['code' as TabId, 'review' as TabId] : []),
  ]
  const labels: Record<TabId, string> = {
    papers: (result.mode === 'pdf' || result.mode === 'analyze') ? '참고 논문' : '수집 논문',
    analysis: '논문 분석',
    trend: '트렌드 분석',
    code: '생성 코드',
    review: '코드 리뷰',
  }

  const [activeTab, setActiveTab] = useState<TabId>('papers')
  type SortKey = 'newest' | 'oldest' | 'citations'
  const [sortKey, setSortKey] = useState<SortKey>('newest')

  // 분석 완료 시 논문 분석 탭으로 자동 전환
  useEffect(() => {
    if (hasAnalysis) setActiveTab('analysis')
  }, [hasAnalysis])

  // 수집 논문 탭에 표시할 논문 목록 — 분석 후에도 검색 결과 유지
  const basePapers = searchedPapers && searchedPapers.length > 0 ? searchedPapers : result.papers

  // 정렬 적용
  const displayPapers = [...basePapers].sort((a, b) => {
    if (sortKey === 'newest') return new Date(b.published_at).getTime() - new Date(a.published_at).getTime()
    if (sortKey === 'oldest') return new Date(a.published_at).getTime() - new Date(b.published_at).getTime()
    // 인용수 — null은 뒤로
    const ca = a.citation_count ?? -1
    const cb = b.citation_count ?? -1
    return cb - ca
  })

  // 트렌드 분석 결과에서 제목으로 한 줄 요약 매핑
  const summaryMap = new Map(
    (result.trend_analysis?.paper_summaries ?? []).map((s) => [s.title, s.summary])
  )

  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      {/* 탭 헤더 */}
      <div className="flex border-b border-gray-200">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab(tab)}
          >
            {labels[tab]}
            {tab === 'papers' && (
              <span className="ml-1.5 rounded-full bg-gray-100 px-1.5 py-0.5 text-xs text-gray-600">
                {displayPapers.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* 탭 콘텐츠 */}
      <div className="p-4">
        {activeTab === 'papers' && (
          <div className="space-y-3">
            {displayPapers.length > 0 && (
              <div className="flex items-center gap-1.5">
                {(['newest', 'oldest', 'citations'] as SortKey[]).map((key) => (
                  <button
                    key={key}
                    onClick={() => setSortKey(key)}
                    className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
                      sortKey === key
                        ? 'border-blue-400 bg-blue-50 text-blue-700'
                        : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300'
                    }`}
                  >
                    {key === 'newest' ? '최신순' : key === 'oldest' ? '오래된순' : '인용수'}
                  </button>
                ))}
              </div>
            )}
            {displayPapers.length === 0 ? (
              <p className="text-sm text-gray-400">수집된 논문이 없습니다.</p>
            ) : (
              displayPapers.map((paper, i) => (
                <PaperCard
                  key={paper.arxiv_id || `${paper.title}-${i}`}
                  paper={paper}
                  summary={summaryMap.get(paper.title)}
                  onAnalyze={onAnalyze}
                />
              ))
            )}
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-5">
            {/* 분석 대상 논문 */}
            {result.papers[0] && (
              <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3">
                <p className="text-xs font-medium text-blue-500 mb-0.5">분석 논문</p>
                <p className="text-sm font-semibold text-blue-900 leading-snug">{result.papers[0].title}</p>
                <p className="mt-1 text-xs text-blue-600">
                  {result.papers[0].authors.slice(0, 3).join(', ')}
                  {result.papers[0].authors.length > 3 && ' 외'}
                </p>
              </div>
            )}

            {/* 논문 요약 */}
            {result.paper_summary && (
              <div>
                <h4 className="mb-2 text-sm font-semibold text-gray-700">📄 논문 요약</h4>
                <p className="rounded-lg bg-gray-50 p-3 text-sm leading-relaxed text-gray-700">
                  {result.paper_summary}
                </p>
              </div>
            )}

            {/* 강점 / 한계 / 의의 */}
            {result.paper_review && (
              <div className="space-y-3">
                {result.paper_review.strengths?.length > 0 && (
                  <div>
                    <h4 className="mb-1.5 text-sm font-semibold text-gray-700">✅ 강점</h4>
                    <ul className="space-y-1">
                      {result.paper_review.strengths.map((s, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                          <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-green-400" />
                          {s}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.paper_review.limitations?.length > 0 && (
                  <div>
                    <h4 className="mb-1.5 text-sm font-semibold text-gray-700">⚠️ 한계점</h4>
                    <ul className="space-y-1">
                      {result.paper_review.limitations.map((l, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                          <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-yellow-400" />
                          {l}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.paper_review.significance && (
                  <div>
                    <h4 className="mb-1.5 text-sm font-semibold text-gray-700">🎯 의의</h4>
                    <p className="text-sm leading-relaxed text-gray-600">{result.paper_review.significance}</p>
                  </div>
                )}
              </div>
            )}

            {/* 핵심 수식 */}
            {result.key_formulas && result.key_formulas.length > 0 && (
              <div>
                <h4 className="mb-2 text-sm font-semibold text-gray-700">🔢 핵심 수식</h4>
                <div className="space-y-2">
                  {result.key_formulas.map((f, i) => (
                    <div key={i} className="rounded-lg border border-gray-100 bg-gray-50 p-3">
                      <p className="text-xs font-semibold text-gray-700">{f.name}</p>
                      <div className="mt-2 overflow-x-auto rounded bg-white py-2 text-center">
                        <FormulaBlock latex={f.latex} />
                      </div>
                      {f.description && (
                        <p className="mt-1.5 text-xs text-gray-500">{f.description}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trend' && result.trend_analysis && (
          <TrendAnalysisPanel analysis={result.trend_analysis} />
        )}

        {activeTab === 'code' && result.generated_code && (
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs text-gray-500">
                반복 횟수: {result.iteration_count ?? 1}회
              </span>
              <Button
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={() => navigator.clipboard.writeText(result.generated_code!)}
              >
                복사
              </Button>
            </div>
            <SyntaxHighlighter
              language="python"
              style={oneDark}
              customStyle={{ borderRadius: '0.5rem', fontSize: '0.75rem', margin: 0 }}
              showLineNumbers
            >
              {result.generated_code}
            </SyntaxHighlighter>
          </div>
        )}

        {activeTab === 'review' && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              {result.review_passed ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500" />
              )}
              <span className="text-sm font-semibold">
                {result.review_passed ? '검증 통과' : '검증 미통과'}
              </span>
            </div>
            {result.review_feedback && (
              <p className="whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-sm leading-relaxed text-gray-700">
                {result.review_feedback}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
