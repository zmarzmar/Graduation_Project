'use client'

import { useEffect, useState } from 'react'
import { BookOpen, Clock, Code2, User, CheckCircle, XCircle, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { BlockMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { getSearchHistory, getAnalysisHistory, getAnalysisDetail, deleteSearchHistory, deleteAllSearchHistory, deleteAnalysisHistory, deleteAllAnalysisHistory, getMyInfo } from '@/lib/api'
import type { SearchHistoryItem, AnalysisHistoryItem, AnalysisDetail, UserInfo, SearchHistoryPaper } from '@/lib/api'

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
  analyze: '논문 선택 분석',
}

function formatDate(iso: string) {
  const d = new Date(iso)
  return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, '0')}.${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

function ConfirmDialog({
  open,
  title,
  description,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  description: string
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <Dialog open={open} onOpenChange={(v) => !v && onCancel()}>
      <DialogContent showCloseButton={false} className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>취소</Button>
          <Button variant="destructive" onClick={onConfirm}>삭제</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function EmptyState({ message, sub }: { message: string; sub: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <p className="text-sm font-medium text-gray-500">{message}</p>
      <p className="mt-1 text-xs text-gray-400">{sub}</p>
    </div>
  )
}

function SearchAccordion({ item, onDelete }: { item: SearchHistoryItem; onDelete: (id: number) => void }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border-b border-gray-100 last:border-0">
      <div className="flex items-start gap-3 py-3">
        <div className="flex-1 min-w-0 cursor-pointer" onClick={() => setOpen((v) => !v)}>
          <p className="truncate text-sm font-medium text-gray-800">{item.query}</p>
          <p className="mt-0.5 text-xs text-gray-400">
            {MODE_LABEL[item.mode] ?? item.mode} · 논문 {item.result_count}편
          </p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-2 cursor-pointer" onClick={() => setOpen((v) => !v)}>
          <span className="text-xs text-gray-400">{formatDate(item.created_at)}</span>
          {open ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
        </div>
        <button onClick={() => onDelete(item.id)} className="flex-shrink-0 text-gray-300 hover:text-red-400">
          <XCircle className="h-4 w-4" />
        </button>
      </div>

      {open && item.papers.length > 0 && (
        <div className="pb-4 space-y-2">
          {item.papers.map((paper: SearchHistoryPaper, i: number) => (
            <div key={i} className="flex items-start justify-between gap-2 rounded-lg bg-gray-50 px-3 py-2">
              <div className="min-w-0">
                <p className="truncate text-xs font-medium text-gray-700">{paper.title}</p>
                {paper.authors.length > 0 && (
                  <p className="mt-0.5 truncate text-xs text-gray-400">{paper.authors.join(', ')}</p>
                )}
              </div>
              {paper.url && (
                <a href={paper.url} target="_blank" rel="noopener noreferrer" className="flex-shrink-0 text-gray-400 hover:text-blue-500">
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              )}
            </div>
          ))}
        </div>
      )}
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
      {/* button 중첩 방지 — 토글 영역과 삭제 버튼을 div로 분리 */}
      <div className="flex items-start gap-3 py-3">
        <div className="flex-1 min-w-0 cursor-pointer" onClick={handleToggle}>
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
        <div className="flex flex-shrink-0 items-center gap-2 cursor-pointer" onClick={handleToggle}>
          <span className="text-xs text-gray-400">{formatDate(item.created_at)}</span>
          {open ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
        </div>
        <button
          onClick={() => onDelete(item.id)}
          className="flex-shrink-0 text-gray-300 hover:text-red-400"
        >
          <XCircle className="h-4 w-4" />
        </button>
      </div>

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
                  <div className="max-h-60 overflow-auto rounded-lg">
                    <SyntaxHighlighter
                      language="python"
                      style={oneDark}
                      customStyle={{ borderRadius: '0.5rem', fontSize: '0.75rem', margin: 0, maxHeight: '15rem' }}
                      showLineNumbers
                    >
                      {detail.generated_code}
                    </SyntaxHighlighter>
                  </div>
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
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null)
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([])
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [confirmDialog, setConfirmDialog] = useState<{ type: 'search' | 'analysis' } | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([getMyInfo(), getSearchHistory(), getAnalysisHistory()])
      .then(([user, search, analysis]) => {
        setUserInfo(user)
        setSearchHistory(search)
        setAnalysisHistory(analysis)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  async function handleDeleteSearch(id: number) {
    try {
      await deleteSearchHistory(id)
      setSearchHistory((prev) => prev.filter((item) => item.id !== id))
    } catch {
      setDeleteError('삭제에 실패했습니다. 다시 시도해주세요.')
    }
  }

  async function handleDeleteAllSearch() {
    setConfirmDialog({ type: 'search' })
  }

  async function handleDeleteAnalysis(id: number) {
    try {
      await deleteAnalysisHistory(id)
      setAnalysisHistory((prev) => prev.filter((item) => item.id !== id))
    } catch {
      setDeleteError('삭제에 실패했습니다. 다시 시도해주세요.')
    }
  }

  async function handleDeleteAllAnalysis() {
    setConfirmDialog({ type: 'analysis' })
  }

  async function handleConfirmDelete() {
    try {
      if (confirmDialog?.type === 'search') {
        await deleteAllSearchHistory()
        setSearchHistory([])
      } else if (confirmDialog?.type === 'analysis') {
        await deleteAllAnalysisHistory()
        setAnalysisHistory([])
      }
    } catch {
      setDeleteError('삭제에 실패했습니다. 다시 시도해주세요.')
    }
    setConfirmDialog(null)
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <ConfirmDialog
        open={confirmDialog !== null}
        title={confirmDialog?.type === 'search' ? '검색 기록 전체 삭제' : '분석 히스토리 전체 삭제'}
        description={confirmDialog?.type === 'search' ? '모든 검색 기록이 삭제됩니다. 이 작업은 되돌릴 수 없습니다.' : '모든 분석 히스토리가 삭제됩니다. 이 작업은 되돌릴 수 없습니다.'}
        onConfirm={handleConfirmDelete}
        onCancel={() => setConfirmDialog(null)}
      />
      <div>
        <h1 className="text-2xl font-bold text-gray-900">마이페이지</h1>
        <p className="mt-1 text-sm text-gray-500">내 정보와 분석 기록을 확인하세요.</p>
      </div>

      {deleteError && (
        <div className="flex items-center justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <span>{deleteError}</span>
          <button onClick={() => setDeleteError(null)} className="ml-4 text-red-400 hover:text-red-600">✕</button>
        </div>
      )}

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
            <div className="flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-600">
              <User className="h-8 w-8" />
            </div>
            {userInfo ? (
              <div className="space-y-1">
                <p className="text-base font-semibold text-gray-900">
                  {userInfo.full_name ?? userInfo.username}
                  <span className="ml-2 text-sm font-normal text-gray-400">@{userInfo.username}</span>
                </p>
                <p className="text-sm text-gray-500">{userInfo.email}</p>
                {userInfo.affiliation && (
                  <p className="text-sm text-gray-500">{userInfo.affiliation}</p>
                )}
              </div>
            ) : (
              <div className="space-y-1 text-sm text-gray-500">
                <p>로그인 후 이용 가능합니다.</p>
                <p className="text-xs">회원가입 기능은 추후 지원 예정입니다.</p>
              </div>
            )}
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
            <div>
              {searchHistory.map((item) => (
                <SearchAccordion key={item.id} item={item} onDelete={handleDeleteSearch} />
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
