'use client'

import { useEffect, useState } from 'react'
import { BookOpen, Clock, Code2, User, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { BlockMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import { getSearchHistory, getAnalysisHistory, getAnalysisDetail, deleteSearchHistory, deleteAllSearchHistory, deleteAnalysisHistory, deleteAllAnalysisHistory } from '@/lib/api'
import type { SearchHistoryItem, AnalysisHistoryItem, AnalysisDetail } from '@/lib/api'

/** LaTeX 렌더링 실패 시 plain text로 fallback */
function FormulaBlock({ latex }: { latex: string }) {
  try {
    return <BlockMath math={latex} />
  } catch {
    return <code className="font-mono text-xs text-gray-700">{latex}</code>
  }
}

const MODE_LABEL: Record<string, string> = {
  search: '키워드 검색',
  pdf: 'PDF 업로드',
  trend: '트렌드 브리핑',
}

function formatDate(iso: string) {
  const d = new Date(iso)
  return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, '0')}.${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

function EmptyState({ message, sub }: { message: string; sub: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <p className="text-sm font-medium text-gray-500">{message}</p>
      <p className="mt-1 text-xs text-gray-400">{sub}</p>
    </div>
  )
}

function AnalysisAccordion({ item, onDelete }: { item: AnalysisHistoryItem; onDelete: (id: number) => void }) {
  const [open, setOpen] = useState(false)
  const [detail, setDetail] = useState<AnalysisDetail | null>(null)
  const [fetching, setFetching] = useState(false)

  async function handleToggle() {
    if (!open && !detail) {
      setFetching(true)
      try {
        const data = await getAnalysisDetail(item.id)
        setDetail(data)
      } catch { /* 무시 */ }
      finally { setFetching(false) }
    }
    setOpen((v) => !v)
  }

  return (
    <div className="border-b border-gray-100 last:border-0">
      <button className="w-full py-3 text-left" onClick={handleToggle}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {item.paper_title ? (
              <p className="truncate text-sm font-medium text-gray-800">{item.paper_title}</p>
            ) : (
              <p className="text-sm font-medium text-gray-800">{item.query}</p>
            )}
            <div className="mt-0.5 flex items-center gap-2 text-xs text-gray-400">
              <span>{MODE_LABEL[item.mode] ?? item.mode}</span>
              {item.has_code && (
                <>
                  <span>·</span>
                  <span className="flex items-center gap-1"><BookOpen className="h-3 w-3" />코드 생성됨</span>
                </>
              )}
              {item.has_code && (
                <>
                  <span>·</span>
                  {item.review_passed ? (
                    <span className="flex items-center gap-1 text-green-500"><CheckCircle className="h-3 w-3" />검증 통과</span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-400"><XCircle className="h-3 w-3" />검증 미통과</span>
                  )}
                </>
              )}
            </div>
          </div>
          <div className="flex flex-shrink-0 items-center gap-2">
            <span className="text-xs text-gray-400">{formatDate(item.created_at)}</span>
            {open ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(item.id) }}
              className="text-gray-300 hover:text-red-400"
            >
              <XCircle className="h-4 w-4" />
            </button>
          </div>
        </div>
      </button>

      {open && (
        <div className="pb-4 space-y-4">
          {fetching && <p className="text-xs text-gray-400">불러오는 중...</p>}
          {detail && (
            <>
              {detail.paper_summary && (
                <div>
                  <p className="mb-1 text-xs font-semibold text-gray-600">📄 논문 요약</p>
                  <p className="rounded-lg bg-gray-50 p-3 text-xs leading-relaxed text-gray-700">{detail.paper_summary}</p>
                </div>
              )}
              {(detail.paper_review?.strengths?.length ?? 0) > 0 && (
                <div>
                  <p className="mb-1 text-xs font-semibold text-gray-600">✅ 강점</p>
                  <ul className="space-y-1">
                    {detail.paper_review!.strengths!.map((s, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                        <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-green-400" />{s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {(detail.paper_review?.limitations?.length ?? 0) > 0 && (
                <div>
                  <p className="mb-1 text-xs font-semibold text-gray-600">⚠️ 한계점</p>
                  <ul className="space-y-1">
                    {detail.paper_review!.limitations!.map((l, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                        <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-yellow-400" />{l}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {detail.key_formulas?.length > 0 && (
                <div>
                  <p className="mb-2 text-xs font-semibold text-gray-600">🔢 핵심 수식</p>
                  <div className="space-y-2">
                    {detail.key_formulas.map((f, i) => (
                      <div key={i} className="rounded-lg border border-gray-100 bg-gray-50 p-3">
                        <p className="text-xs font-medium text-gray-700">{f.name}</p>
                        <div className="mt-2 overflow-x-auto rounded bg-white py-2 text-center">
                          <FormulaBlock latex={f.latex} />
                        </div>
                        {f.description && (
                          <p className="mt-1 text-xs text-gray-500">{f.description}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {detail.generated_code && (
                <div>
                  <p className="mb-1 text-xs font-semibold text-gray-600">💻 생성 코드</p>
                  <pre className="overflow-x-auto rounded-lg bg-gray-900 p-3 text-xs leading-relaxed text-gray-100 max-h-60">
                    <code>{detail.generated_code}</code>
                  </pre>
                </div>
              )}
              {detail.review_feedback && (
                <div>
                  <p className="mb-1 text-xs font-semibold text-gray-600">🔍 코드 리뷰</p>
                  <p className="rounded-lg bg-gray-50 p-3 text-xs leading-relaxed text-gray-700 whitespace-pre-wrap">{detail.review_feedback}</p>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default function MyPage() {
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([])
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getSearchHistory(), getAnalysisHistory()])
      .then(([search, analysis]) => {
        setSearchHistory(search)
        setAnalysisHistory(analysis)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  async function handleDeleteSearch(id: number) {
    await deleteSearchHistory(id)
    setSearchHistory((prev) => prev.filter((item) => item.id !== id))
  }

  async function handleDeleteAllSearch() {
    if (!confirm('검색 기록을 모두 삭제하시겠어요?')) return
    await deleteAllSearchHistory()
    setSearchHistory([])
  }

  async function handleDeleteAnalysis(id: number) {
    await deleteAnalysisHistory(id)
    setAnalysisHistory((prev) => prev.filter((item) => item.id !== id))
  }

  async function handleDeleteAllAnalysis() {
    if (!confirm('분석 히스토리를 모두 삭제하시겠어요?')) return
    await deleteAllAnalysisHistory()
    setAnalysisHistory([])
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">마이페이지</h1>
        <p className="mt-1 text-sm text-gray-500">내 정보와 분석 기록을 확인하세요.</p>
      </div>

      {/* 내 정보 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <User className="h-4 w-4" />
            내 정보
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-blue-100 text-blue-600">
              <User className="h-8 w-8" />
            </div>
            <div className="space-y-1 text-sm text-gray-500">
              <p>로그인 후 이용 가능합니다.</p>
              <p className="text-xs">회원가입 기능은 추후 지원 예정입니다.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 최근 검색 기록 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              최근 검색 기록
              {searchHistory.length > 0 && (
                <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
                  {searchHistory.length}
                </span>
              )}
            </div>
            {searchHistory.length > 0 && (
              <button onClick={handleDeleteAllSearch} className="text-xs text-red-400 hover:text-red-600">
                전체 삭제
              </button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="py-4 text-center text-sm text-gray-400">불러오는 중...</p>
          ) : searchHistory.length === 0 ? (
            <EmptyState message="검색 기록이 없습니다." sub="키워드 검색이나 PDF 업로드로 논문을 분석해보세요." />
          ) : (
            <div className="divide-y divide-gray-100">
              {searchHistory.map((item) => (
                <div key={item.id} className="flex items-center justify-between py-3">
                  <div>
                    <p className="text-sm font-medium text-gray-800">{item.query}</p>
                    <p className="mt-0.5 text-xs text-gray-400">
                      {MODE_LABEL[item.mode] ?? item.mode} · 논문 {item.result_count}편
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">{formatDate(item.created_at)}</span>
                    <button onClick={() => handleDeleteSearch(item.id)} className="text-gray-300 hover:text-red-400">
                      <XCircle className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 분석 히스토리 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            <div className="flex items-center gap-2">
              <Code2 className="h-4 w-4" />
              분석 히스토리
              {analysisHistory.length > 0 && (
                <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
                  {analysisHistory.length}
                </span>
              )}
            </div>
            {analysisHistory.length > 0 && (
              <button onClick={handleDeleteAllAnalysis} className="text-xs text-red-400 hover:text-red-600">
                전체 삭제
              </button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="py-4 text-center text-sm text-gray-400">불러오는 중...</p>
          ) : analysisHistory.length === 0 ? (
            <EmptyState message="분석 기록이 없습니다." sub="분석이 완료되면 생성된 코드와 결과가 여기에 저장됩니다." />
          ) : (
            <div>
              {analysisHistory.map((item) => (
                <AnalysisAccordion key={item.id} item={item} onDelete={handleDeleteAnalysis} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
