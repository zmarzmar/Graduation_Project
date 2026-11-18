'use client'

import { useState } from 'react'
import { MathText } from '@/components/ui/math-text'
import type { SummaryClaim } from '@/lib/types/agent-run'
import { CitationChip } from './CitationChip'
import { CitationQuote } from './CitationQuote'

interface CitedSummaryProps {
  summary: string
  /** 문장별 원문 인용. 없으면(초록 기반·이전 기록) 요약 문단만 보여준다 */
  claims?: SummaryClaim[]
  className?: string
}

/** 논문 요약 — 문장마다 원문 인용 칩을 붙이고, 원문에서 인용을 찾지 못한 문장은 '출처 미확인'으로 표시한다 */
export function CitedSummary({ summary, claims, className = 'text-sm' }: CitedSummaryProps) {
  const [openCitation, setOpenCitation] = useState<string | null>(null)
  const toggle = (key: string) => setOpenCitation(openCitation === key ? null : key)

  if (!claims || claims.length === 0) {
    return (
      <p className={`rounded-lg bg-gray-50 p-3 leading-relaxed text-gray-700 ${className}`}>
        <MathText text={summary} />
      </p>
    )
  }

  const unverified = claims.filter((claim) => !claim.verified).length
  return (
    <div className="space-y-2">
      <p className={`rounded-lg bg-gray-50 p-3 leading-7 text-gray-700 ${className}`}>
        {claims.map((claim, claimIndex) => (
          <span key={claimIndex}>
            <MathText text={claim.text} />{' '}
            {claim.citations.map((citation, index) => (
              <span key={index} className="mr-1 inline-block align-middle">
                <CitationChip
                  page={citation.page}
                  open={openCitation === `${claimIndex}:${index}`}
                  onToggle={() => toggle(`${claimIndex}:${index}`)}
                />
              </span>
            ))}
            {!claim.verified && (
              <span className="mr-1 inline-block rounded-full border border-yellow-200 bg-yellow-50 px-2 py-0.5 align-middle text-xs text-yellow-700">
                출처 미확인
              </span>
            )}{' '}
          </span>
        ))}
      </p>
      {claims.map((claim, claimIndex) =>
        claim.citations.map(
          (citation, index) =>
            openCitation === `${claimIndex}:${index}` && (
              <CitationQuote key={`${claimIndex}:${index}`} citation={citation} />
            ),
        ),
      )}
      {/* 확인한 것은 '인용이 원문에 있다'까지다 — 그 인용이 문장을 뒷받침하는지는 직접 대조해 보게 한다 */}
      <p className="text-xs text-gray-500">
        쪽 번호를 누르면 그 문장의 원문 인용을 볼 수 있어요. 인용이 논문에 실제로 있는지만 확인했으니, 문장과 직접 대조해 보세요.
        {unverified > 0 && ` '출처 미확인'은 원문에서 인용을 찾지 못한 문장이에요.`}
      </p>
    </div>
  )
}
