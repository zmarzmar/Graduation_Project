'use client'

import { useState } from 'react'
import { CheckCircle, XCircle, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { AgentResult, ArxivPaper, TrendAnalysis } from '@/lib/types/agent-run'

interface ResultsPanelProps {
  result: AgentResult
}

function PaperCard({ paper, summary }: { paper: ArxivPaper; summary?: string }) {
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

      {/* 초록 토글 */}
      <button
        className="mt-2 flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        초록 {expanded ? '접기' : '보기'}
      </button>
      {expanded && (
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

export function ResultsPanel({ result }: ResultsPanelProps) {
  const isTrend = result.mode === 'trend'
  const hasCoding = !!result.generated_code
  const hasTrendAnalysis = isTrend && !!result.trend_analysis

  type TabId = 'papers' | 'trend' | 'code' | 'review'
  const tabs: TabId[] = [
    'papers',
    ...(hasTrendAnalysis ? ['trend' as TabId] : []),
    ...(hasCoding ? ['code' as TabId, 'review' as TabId] : []),
  ]
  const labels: Record<TabId, string> = {
    papers: '수집 논문',
    trend: '트렌드 분석',
    code: '생성 코드',
    review: '리뷰',
  }

  const [activeTab, setActiveTab] = useState<TabId>('papers')

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
                {result.papers.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* 탭 콘텐츠 */}
      <div className="p-4">
        {activeTab === 'papers' && (
          <div className="space-y-3">
            {result.papers.length === 0 ? (
              <p className="text-sm text-gray-400">수집된 논문이 없습니다.</p>
            ) : (
              result.papers.map((paper) => (
                <PaperCard
                  key={paper.arxiv_id || paper.title}
                  paper={paper}
                  summary={summaryMap.get(paper.title)}
                />
              ))
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
            <pre className="overflow-x-auto rounded-lg bg-gray-900 p-4 text-xs leading-relaxed text-gray-100">
              <code>{result.generated_code}</code>
            </pre>
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
