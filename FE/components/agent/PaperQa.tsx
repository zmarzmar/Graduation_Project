'use client'

import { useEffect, useRef, useState } from 'react'
import { Info, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { FormulaBlock } from '@/components/ui/formula-block'
import { Input } from '@/components/ui/input'
import { MathText } from '@/components/ui/math-text'
import { askPaper, getQaMessages, QA_HISTORY_TURNS, QA_QUESTION_MAX_LENGTH, QA_QUESTION_MIN_LENGTH } from '@/lib/api'
import type { QaAnswer, QaHistoryTurn } from '@/lib/api'
import type { KeyFormula, QaUnavailableReason } from '@/lib/types/agent-run'
import { useAuthStore } from '@/store/auth-store'
import { CitationChip } from './CitationChip'
import { CitationQuote } from './CitationQuote'
import { WebAnswer } from './WebAnswer'
import { groupByPage } from '@/lib/citations'

// 색인 중(202) 자동 재시도 상한 — BE의 색인 시간 제한(60초)을 3초 간격으로 덮는 횟수
const MAX_INDEXING_RETRIES = 20
const MIN_RETRY_SECONDS = 1
const MAX_RETRY_SECONDS = 10

interface PaperQaProps {
  /** 커밋된 분석 기록 id. 저장에 실패했으면 null, BE가 값을 주지 않았으면 undefined */
  analysisId: number | null | undefined
  hasDocument: boolean
  unavailableReason?: QaUnavailableReason | null
}

interface QaEntry {
  id: number
  question: string
  status: 'pending' | 'indexing' | 'done' | 'error'
  result?: QaAnswer
  error?: string
}

const NO_DOCUMENT_MESSAGES: Record<string, string> = {
  guest: '로그인하기 전에 분석한 결과라 논문 원문이 보관되지 않았어요. 로그인한 상태로 논문을 다시 분석하면 질문할 수 있어요.',
  no_text: '초록만으로 분석한 결과라 질문에 쓸 논문 원문이 없어요. PDF 전문으로 다시 분석하면 질문할 수 있어요.',
  document_store_failed: '분석 기록은 저장됐지만 논문 원문을 보관하지 못했어요. 논문을 다시 분석하면 질문할 수 있어요.',
}
const NO_DOCUMENT_DEFAULT = '이 기록에는 보관된 논문 원문이 없어요. 논문을 다시 분석하면 질문할 수 있어요.'

/** 취소할 수 있는 대기 — 취소되면 타이머를 지우고 AbortError로 끝난다 */
function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(new DOMException('Aborted', 'AbortError'))
      },
      { once: true },
    )
  })
}

function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-600">
      <Info className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden />
      <div>{children}</div>
    </div>
  )
}

/** 수식 질문에 함께 오는 분석 결과의 핵심 수식 — '논문 분석' 탭과 같은 모양. 인용으로 확인한 답변과는 칸을 나눈다 */
function RelatedFormulas({ formulas }: { formulas: KeyFormula[] }) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-500">🔢 분석 결과의 핵심 수식 · 논문 원문 인용으로 확인한 것은 아니에요</p>
      {formulas.map((f, i) => (
        <div key={i} className="rounded-lg border border-gray-100 bg-gray-50 p-3">
          {f.name && <p className="text-xs font-semibold text-gray-700">{f.name}</p>}
          <div className="mt-2 overflow-x-auto rounded bg-white py-2 text-center">
            <FormulaBlock latex={f.latex} />
          </div>
          {f.description && (
            <p className="mt-1.5 text-xs text-gray-500">
              <MathText text={f.description} />
            </p>
          )}
        </div>
      ))}
    </div>
  )
}

function AnswerView({ result }: { result: QaAnswer }) {
  // 펼친 출처 — 문장별 화면에서는 '문장 번호:출처 번호', 이전 응답의 화면에서는 출처 번호
  const [openCitation, setOpenCitation] = useState<string | null>(null)
  const toggle = (key: string) => setOpenCitation(openCitation === key ? null : key)

  if (!result.answerable) {
    const web = result.web_answer ? <WebAnswer answer={result.web_answer} /> : null
    // 수식 질문이라 분석 결과의 핵심 수식이 함께 오면, '근거 없음'을 크게 띄우지 않고 수식으로 안내한다
    if (result.related_formulas && result.related_formulas.length > 0) {
      return (
        <div className="space-y-2">
          <p className="text-xs text-gray-500">논문 원문에서 인용으로 확인한 문장 답변은 없어요. 아래 분석 결과의 핵심 수식을 참고해 주세요.</p>
          {web}
        </div>
      )
    }
    // 근거 부족은 오류가 아니라 정상 결과다
    return (
      <div className="space-y-2">
        <Notice>
          <p>{result.answer}</p>
          {result.dropped_claims > 0 && (
            <p className="mt-1 text-xs text-gray-500">답변 초안의 출처를 논문에서 확인하지 못해 표시하지 않았어요.</p>
          )}
        </Notice>
        {web}
      </div>
    )
  }

  const droppedNotice = result.dropped_claims > 0 && (
    <p className="text-xs text-yellow-700">
      출처를 논문에서 확인하지 못한 문장 {result.dropped_claims}개를 답변에서 제외했어요.
    </p>
  )

  // 문장별 연결이 있으면 문장마다 그 문장의 출처를 붙인다 — 어느 구절을 어느 문장과 대조할지 보인다
  if (result.claims && result.claims.length > 0) {
    return (
      <div className="space-y-2">
        {/* 한 문단으로 이어 읽히게 — 문장마다 끝에 그 문장의 출처 칩을 붙인다 */}
        <p className="text-sm leading-7 text-gray-800">
          {result.claims.map((claim, claimIndex) => (
            <span key={claimIndex}>
              {/* 답변 안의 인라인 수식(\(...\), $...$)을 KaTeX로 그린다 — 출처 검증은 문장 단위 그대로 */}
              <MathText text={claim.text} className="whitespace-pre-wrap" />{' '}
              {groupByPage(claim.citations).map((group) => (
                <span key={group.page} className="mr-1 inline-block align-middle">
                  <CitationChip
                    page={group.page}
                    open={openCitation === `${claimIndex}:${group.page}`}
                    onToggle={() => toggle(`${claimIndex}:${group.page}`)}
                  />
                </span>
              ))}{' '}
            </span>
          ))}
        </p>
        {result.claims.map((claim, claimIndex) =>
          groupByPage(claim.citations).map(
            (group) =>
              openCitation === `${claimIndex}:${group.page}` &&
              group.quotes.map((citation, index) => (
                <CitationQuote key={`${claimIndex}:${group.page}:${index}`} citation={citation} />
              )),
          ),
        )}
        {droppedNotice}
      </div>
    )
  }

  // 문장별 연결이 없는 이전 BE 응답 — 답변 전체 아래에 출처를 모아 보여준다
  return (
    <div className="space-y-2">
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-gray-800">{result.answer}</p>
      {droppedNotice}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-gray-400">출처</span>
        {result.citations.map((citation, index) => (
          <CitationChip
            key={index}
            page={citation.page}
            open={openCitation === String(index)}
            onToggle={() => toggle(String(index))}
          />
        ))}
      </div>
      {result.citations.map(
        (citation, index) => openCitation === String(index) && <CitationQuote key={index} citation={citation} />,
      )}
    </div>
  )
}

/** 질문·답변 목록. 분석이나 사용자가 바뀌면 key가 바뀌어 통째로 다시 만들어진다 — 진행 중인 요청과 대기 타이머는 그때 정리된다 */
function QaThread({ analysisId, noDocumentMessage }: { analysisId: number; noDocumentMessage: string }) {
  const logout = useAuthStore((state) => state.logout)
  const [entries, setEntries] = useState<QaEntry[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  // 질문 중에 서버가 알려준 상태 — 원문 없음(409)·기록 없음(404)
  const [blocked, setBlocked] = useState<'no_document' | 'not_found' | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const nextIdRef = useRef(1)
  // 저장된 Q&A 불러오기 — 끝나야 질문할 수 있다 (불러온 기록이 이어서 이해하기 문맥에 들어가야 하고, 늦게 온 기록이
  // 새 질문을 덮지 않게). 실패하면 다시 불러오기를 누르게 한다
  const [loadState, setLoadState] = useState<'loading' | 'ok' | 'error'>('loading')
  const [loadAttempt, setLoadAttempt] = useState(0)

  // 화면 이탈·분석 변경·로그아웃(언마운트) 시 요청과 대기 타이머를 취소한다
  useEffect(() => () => controllerRef.current?.abort(), [])

  useEffect(() => {
    const controller = new AbortController()
    getQaMessages(analysisId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return
        if (result.status === 'unauthorized') {
          logout()
          return
        }
        if (result.status === 'not_found') {
          setBlocked('not_found')
          return
        }
        // 저장된 행은 음수 id — 이 화면에서 새로 묻는 항목(양수)과 key가 겹치지 않게
        setEntries(result.messages.map((m) => ({ id: -m.id, question: m.question, status: 'done', result: m.response })))
        setLoadState('ok')
      })
      .catch(() => {
        if (!controller.signal.aborted) setLoadState('error')
      })
    return () => controller.abort()
  }, [analysisId, logout, loadAttempt])

  const question = input.trim()
  const canAsk =
    !busy && loadState === 'ok' && question.length >= QA_QUESTION_MIN_LENGTH && question.length <= QA_QUESTION_MAX_LENGTH

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    if (!canAsk) return

    const controller = new AbortController()
    controllerRef.current = controller
    const id = nextIdRef.current++
    const update = (patch: Partial<QaEntry>) =>
      setEntries((prev) => prev.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)))

    // 직전 대화(답을 받은 것만, 불러온 기록 포함, 최대 6개)를 함께 보낸다 — "수식 알려줘" 같은 후속 질문이 같은 주제를 찾게 한다
    const history: QaHistoryTurn[] = entries
      .filter((entry) => entry.status === 'done' && entry.result)
      .slice(-QA_HISTORY_TURNS)
      // BE도 자르지만 보낼 때부터 맞춘다 (BE HistoryTurn: 질문 500자, 답변 2,000자)
      .map((entry) => ({
        question: entry.question.slice(0, QA_QUESTION_MAX_LENGTH),
        // 웹 설명도 보낸다 — "둘 중 어느 쪽?" 같은 후속 질문이 웹 답을 가리킬 수 있다 (서버는 참고용으로만 쓴다)
        answer: (entry.result?.answerable
          ? entry.result.answer
          : entry.result?.web_answer
            ? `[논문 밖 웹 설명] ${entry.result.web_answer.text}`
            : ''
        ).slice(0, 2000),
      }))
    setEntries((prev) => [...prev, { id, question, status: 'pending' }])
    setInput('')
    setBusy(true)

    try {
      for (let attempt = 0; ; attempt++) {
        const response = await askPaper(analysisId, question, history, controller.signal)
        // 취소 뒤에 도착한 응답은 반영하지 않는다 (취소가 응답 수신과 겹칠 수 있다)
        if (controller.signal.aborted) return

        if (response.status === 'indexing') {
          if (attempt >= MAX_INDEXING_RETRIES) {
            update({ status: 'error', error: '논문 검색 준비가 오래 걸리고 있어요. 잠시 후 다시 질문해 주세요.' })
            break
          }
          update({ status: 'indexing' })
          const seconds = Math.min(Math.max(response.retryAfterSeconds, MIN_RETRY_SECONDS), MAX_RETRY_SECONDS)
          await wait(seconds * 1000, controller.signal)
          continue
        }
        if (response.status === 'unauthorized') {
          // 토큰 만료 — 로그아웃 처리해 로그인 안내로 보낸다. 안내를 이 화면의 상태로 두면 같은 계정으로
          // 다시 로그인해도(사용자 id가 같아 화면이 다시 만들어지지 않는다) 사라지지 않는다.
          logout()
          return
        }
        if (response.status === 'answered') {
          update({ status: 'done', result: response.data })
        } else {
          setBlocked(response.status)
          setEntries((prev) => prev.filter((entry) => entry.id !== id))
        }
        break
      }
    } catch (error) {
      if (controller.signal.aborted) return
      update({ status: 'error', error: error instanceof Error ? error.message : '질문을 처리하지 못했습니다.' })
    }
    if (controller.signal.aborted) return
    controllerRef.current = null
    setBusy(false)
  }

  if (blocked === 'no_document') return <Notice>{noDocumentMessage}</Notice>
  if (blocked === 'not_found') {
    return <Notice>분석 기록을 찾을 수 없어요. 삭제됐거나 지금 로그인한 계정의 기록이 아니에요.</Notice>
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-500">
        질문과 답은 이 분석 기록에 저장돼 마이페이지에서 다시 볼 수 있고, 기록을 삭제하면 함께 지워져요. 직전 질문 여섯 개까지 이어서
        이해해요 (예: &ldquo;수식 알려줘&rdquo;). 수식을 물으면 분석 결과의 핵심 수식도 함께 보여 줘요.
        답변은 논문에서 인용 구절을 확인한 문장만 보여 줍니다 — 구절이 그 문장을 뒷받침하는지는 문장 옆의 출처를 펼쳐 직접 확인해 주세요.
        논문에서 근거를 찾지 못하면 웹 검색 설명을 따로 구분해 보여 줘요.
      </p>

      {loadState === 'loading' && (
        <p className="flex items-center gap-2 text-xs text-gray-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          이전 질문을 불러오는 중이에요...
        </p>
      )}
      {loadState === 'error' && (
        <p className="flex items-center gap-2 text-xs text-yellow-700">
          이전 질문을 불러오지 못했어요.
          <button
            type="button"
            className="underline underline-offset-2"
            onClick={() => {
              setLoadState('loading')
              setLoadAttempt((n) => n + 1)
            }}
          >
            다시 불러오기
          </button>
        </p>
      )}

      {entries.length > 0 && (
        <ul className="space-y-4" aria-live="polite">
          {entries.map((entry) => (
            <li key={entry.id} className="space-y-2">
              <p className="rounded-lg bg-gray-100 px-3 py-2 text-sm font-medium text-gray-800">Q. {entry.question}</p>
              {(entry.status === 'pending' || entry.status === 'indexing') && (
                <p className="flex items-center gap-2 text-xs text-gray-500">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                  {entry.status === 'indexing'
                    ? '논문 검색을 준비하는 중이에요. 첫 질문은 조금 더 걸려요...'
                    : '논문에서 근거를 찾는 중이에요...'}
                </p>
              )}
              {entry.status === 'done' && entry.result && <AnswerView result={entry.result} />}
              {entry.status === 'done' && entry.result?.related_formulas && entry.result.related_formulas.length > 0 && (
                <RelatedFormulas formulas={entry.result.related_formulas} />
              )}
              {entry.status === 'done' && entry.result?.saved === false && (
                <p className="text-xs text-yellow-700">이 답변은 저장되지 않았어요 — 이 화면을 떠나면 다시 볼 수 없어요.</p>
              )}
              {entry.status === 'error' && <p className="text-sm text-red-600">{entry.error}</p>}
            </li>
          ))}
        </ul>
      )}

      <form onSubmit={handleSubmit} className="space-y-1">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="이 논문에 대해 질문해 보세요"
            aria-label="논문에 대한 질문"
            disabled={busy}
            className="text-sm"
          />
          <Button type="submit" size="sm" disabled={!canAsk}>
            질문
          </Button>
        </div>
        {question.length > QA_QUESTION_MAX_LENGTH && (
          <p className="text-xs text-red-600">
            질문은 {QA_QUESTION_MAX_LENGTH}자까지 쓸 수 있어요 (지금 {question.length}자).
          </p>
        )}
      </form>
    </div>
  )
}

/** 분석한 논문에 질문하는 화면. 홈 분석 결과와 마이페이지 분석 상세가 함께 쓴다 */
export function PaperQa({ analysisId, hasDocument, unavailableReason }: PaperQaProps) {
  const user = useAuthStore((state) => state.user)
  const isAuthReady = useAuthStore((state) => state.isInitialized)
  const openModal = useAuthStore((state) => state.openModal)
  const noDocumentMessage = NO_DOCUMENT_MESSAGES[unavailableReason ?? ''] ?? NO_DOCUMENT_DEFAULT

  if (!isAuthReady) return null

  // 순서: 로그인 여부 → 기록 저장 여부 → 원문 보유 여부
  if (!user) {
    return (
      <Notice>
        <p>논문 Q&amp;A는 로그인한 뒤에 분석한 논문에서 쓸 수 있어요.</p>
        <Button size="sm" variant="outline" className="mt-2 text-xs" onClick={() => openModal('login')}>
          로그인
        </Button>
      </Notice>
    )
  }
  if (analysisId == null && unavailableReason !== 'guest') {
    return (
      <Notice>
        {unavailableReason === 'save_failed'
          ? '분석 결과를 기록에 저장하지 못해 질문할 수 없어요. 논문을 다시 분석해 주세요.'
          : '분석 기록을 확인하지 못해 질문할 수 없어요. 마이페이지의 분석 기록에서 다시 시도해 주세요.'}
      </Notice>
    )
  }
  if (!hasDocument || analysisId == null) return <Notice>{noDocumentMessage}</Notice>

  return <QaThread key={`${user.id}:${analysisId}`} analysisId={analysisId} noDocumentMessage={noDocumentMessage} />
}
