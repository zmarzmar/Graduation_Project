import { Globe } from 'lucide-react'
import { MathText } from '@/components/ui/math-text'
import type { QaWebAnswer } from '@/lib/api'

/** 논문에서 근거를 찾지 못한 질문의 웹 검색 일반 설명 — 인용을 확인한 논문 답변과 칸을 나누고, 출처 링크를 함께 보여준다 */
export function WebAnswer({ answer }: { answer: QaWebAnswer }) {
  return (
    <div className="space-y-2 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
      <p className="flex items-center gap-1.5 text-xs font-medium text-gray-600">
        <Globe className="h-3.5 w-3.5" aria-hidden />
        논문 밖 설명 · 웹 검색 — 논문에서 확인한 내용이 아니에요. 아래 출처에서 직접 확인해 주세요.
      </p>
      {/* 웹 설명에도 \(\sqrt{d_k}\) 같은 인라인 수식이 섞여 온다 — 원문 LaTeX가 보이지 않게 그린다 */}
      <MathText text={answer.text} className="block whitespace-pre-wrap text-sm leading-7 text-gray-700" />
      <ul className="space-y-0.5 text-xs">
        {answer.sources.map((source) => (
          <li key={source.url}>
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all text-blue-600 underline-offset-2 hover:underline"
            >
              {source.title}
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}
