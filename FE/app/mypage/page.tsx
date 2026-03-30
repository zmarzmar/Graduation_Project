'use client'

import { useEffect, useState } from 'react'
import { BookOpen, Clock, Code2, User, CheckCircle, XCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getSearchHistory, getAnalysisHistory } from '@/lib/api'
import type { SearchHistoryItem, AnalysisHistoryItem } from '@/lib/api'

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
          <CardTitle className="flex items-center gap-2 text-base">
            <Clock className="h-4 w-4" />
            최근 검색 기록
            {searchHistory.length > 0 && (
              <span className="ml-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
                {searchHistory.length}
              </span>
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
                  <span className="text-xs text-gray-400">{formatDate(item.created_at)}</span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 분석 히스토리 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Code2 className="h-4 w-4" />
            분석 히스토리
            {analysisHistory.length > 0 && (
              <span className="ml-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
                {analysisHistory.length}
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="py-4 text-center text-sm text-gray-400">불러오는 중...</p>
          ) : analysisHistory.length === 0 ? (
            <EmptyState message="분석 기록이 없습니다." sub="분석이 완료되면 생성된 코드와 결과가 여기에 저장됩니다." />
          ) : (
            <div className="divide-y divide-gray-100">
              {analysisHistory.map((item) => (
                <div key={item.id} className="py-3">
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
                            <span className="flex items-center gap-1">
                              <BookOpen className="h-3 w-3" />
                              코드 생성됨
                            </span>
                          </>
                        )}
                        <span>·</span>
                        {item.review_passed ? (
                          <span className="flex items-center gap-1 text-green-500">
                            <CheckCircle className="h-3 w-3" />
                            검증 통과
                          </span>
                        ) : item.has_code ? (
                          <span className="flex items-center gap-1 text-red-400">
                            <XCircle className="h-3 w-3" />
                            검증 미통과
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <span className="flex-shrink-0 text-xs text-gray-400">{formatDate(item.created_at)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
