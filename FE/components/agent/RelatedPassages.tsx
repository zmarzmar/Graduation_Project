'use client'

import { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { getRelatedPassages, retryWhileIndexing } from '@/lib/api'
import type { RelatedItem } from '@/lib/api'
import { useAuthStore } from '@/store/auth-store'

interface RelatedPassagesProps {
  /** 커밋된 분석 기록 id. 저장에 실패했으면 null, BE가 값을 주지 않았으면 undefined */
  analysisId: number | null | undefined
  hasDocument: boolean
}

type LoadState =
  | { status: 'idle' }
  | { status: 'loading'; indexing: boolean }
  | { status: 'found'; items: RelatedItem[] }
  | { status: 'unavailable'; message: string }

function ItemView({ item }: { item: RelatedItem }) {
  const [open, setOpen] = useState<number | null>(null)

  return (
    <li className="space-y-1.5">
      <p className="text-xs text-gray-700">
        <span className="mr-1.5 rounded bg-gray-100 px-1.5 py-0.5 text-[11px] text-gray-500">
          {item.kind === 'formula' ? '수식' : '요약'}
        </span>
        {item.label}
      </p>
      {item.passages.length === 0 ? (
        <p className="text-xs text-gray-400">비슷한 구절을 찾지 못했어요.</p>
      ) : (
        <div className="flex flex-wrap items-center gap-1.5">
          {item.passages.map((passage, index) => (
            <button
              key={`${passage.chunk_index}-${index}`}
              type="button"
              aria-expanded={open === index}
              onClick={() => setOpen(open === index ? null : index)}
              className={`rounded-full border px-2 py-0.5 text-xs transition-colors ${
                open === index
                  ? 'border-gray-400 bg-gray-100 text-gray-800'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
              }`}
            >
              p.{passage.page}
            </button>
          ))}
        </div>
      )}
      {open !== null && item.passages[open] && (
        <blockquote className="max-h-48 overflow-y-auto rounded-lg border-l-2 border-gray-300 bg-gray-50 px-3 py-2 text-xs leading-relaxed text-gray-700">
          <p className="mb-1 font-medium text-gray-500">{item.passages[open].page}쪽의 비슷한 구절</p>
          <p className="whitespace-pre-wrap break-words">{item.passages[open].text}</p>
        </blockquote>
      )}
    </li>
  )
}

/** 관련 원문 목록. 분석이나 사용자가 바뀌면 key가 바뀌어 통째로 다시 만들어진다 — 진행 중인 요청과 대기 타이머는 그때 정리된다 */
function RelatedList({ analysisId }: { analysisId: number }) {
  const logout = useAuthStore((state) => state.logout)
  const [state, setState] = useState<LoadState>({ status: 'idle' })
  const controllerRef = useRef<AbortController | null>(null)

  // 화면 이탈·분석 변경·로그아웃(언마운트) 시 요청과 대기 타이머를 취소한다
  useEffect(() => () => controllerRef.current?.abort(), [])

  async function handleLoad() {
    const controller = new AbortController()
    controllerRef.current = controller
    setState({ status: 'loading', indexing: false })
    try {
      const response = await retryWhileIndexing(
        () => getRelatedPassages(analysisId, controller.signal),
        controller.signal,
        () => {
          if (!controller.signal.aborted) setState({ status: 'loading', indexing: true })
        },
      )
      // 취소 뒤에 도착한 응답은 반영하지 않는다
      if (controller.signal.aborted) return

      if (response.status === 'unauthorized') {
        logout() // 토큰 만료 — 로그인 안내로 보낸다
      } else if (response.status === 'found') {
        setState({ status: 'found', items: response.items })
      } else if (response.status === 'still_indexing') {
        setState({ status: 'unavailable', message: '논문 검색 준비가 오래 걸리고 있어요. 잠시 후 다시 시도해 주세요.' })
      } else if (response.status === 'no_document') {
        setState({ status: 'unavailable', message: '이 기록에는 보관된 논문 원문이 없어요. 논문을 다시 분석하면 볼 수 있어요.' })
      } else {
        setState({ status: 'unavailable', message: '분석 기록을 찾을 수 없어요. 삭제됐거나 지금 로그인한 계정의 기록이 아니에요.' })
      }
    } catch (error) {
      if (controller.signal.aborted) return
      setState({ status: 'unavailable', message: error instanceof Error ? error.message : '요청을 처리하지 못했습니다.' })
    }
  }

  if (state.status === 'idle') {
    return (
      <Button variant="outline" size="sm" className="text-xs" onClick={handleLoad}>
        관련 원문 찾기
      </Button>
    )
  }
  if (state.status === 'loading') {
    return (
      <p className="flex items-center gap-2 text-xs text-gray-500" aria-live="polite">
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
        {state.indexing ? '논문 검색을 준비하는 중이에요. 처음에는 조금 더 걸려요...' : '비슷한 구절을 찾는 중이에요...'}
      </p>
    )
  }
  if (state.status === 'unavailable') {
    return (
      <div className="space-y-2">
        <p className="text-xs text-gray-500">{state.message}</p>
        <Button variant="outline" size="sm" className="text-xs" onClick={handleLoad}>
          다시 시도
        </Button>
      </div>
    )
  }
  if (state.items.length === 0) {
    return <p className="text-xs text-gray-400">이 분석에는 관련 원문을 찾을 요약이나 수식이 없어요.</p>
  }
  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        요약 문장·수식과 가장 비슷한 구절을 논문에서 검색한 결과예요. 구절이 그 내용을 뒷받침하는지는 확인되지 않았어요 —
        관련이 없는 구절이 나올 수 있어요.
      </p>
      <ul className="space-y-3">
        {state.items.map((item) => (
          <ItemView key={item.id} item={item} />
        ))}
      </ul>
    </div>
  )
}

/** 분석 결과의 요약·수식과 비슷한 원문 구절을 보여준다. 홈 분석 결과와 마이페이지 분석 상세가 함께 쓴다 */
export function RelatedPassages({ analysisId, hasDocument }: RelatedPassagesProps) {
  const user = useAuthStore((state) => state.user)

  // 쓸 수 없는 이유는 논문 Q&A 쪽에서 자세히 안내한다 — 여기서는 한 줄만 둔다
  if (!user || analysisId == null || !hasDocument) {
    return (
      <p className="text-xs text-gray-400">
        관련 원문은 로그인한 상태로 분석해 논문 원문이 보관된 기록에서 볼 수 있어요.
      </p>
    )
  }
  return <RelatedList key={`${user.id}:${analysisId}`} analysisId={analysisId} />
}
