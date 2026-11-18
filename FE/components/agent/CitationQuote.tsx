/** 펼친 인용 구절 — 서버가 원문에 실제로 있음을 확인한 구절이다 (논문 Q&A·논문 요약 공용) */
export function CitationQuote({ citation }: { citation: { page: number; quote: string } }) {
  return (
    <blockquote className="rounded-lg border-l-2 border-blue-300 bg-blue-50 px-3 py-2 text-xs leading-relaxed text-gray-700">
      <p className="mb-1 font-medium text-blue-600">{citation.page}쪽에서 인용</p>
      <p className="whitespace-pre-wrap break-words">{citation.quote}</p>
    </blockquote>
  )
}
