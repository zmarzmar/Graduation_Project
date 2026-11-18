/** 출처 쪽 번호 칩 — 누르면 인용 구절을 펼친다 (논문 Q&A·논문 요약 공용) */
export function CitationChip({ page, open, onToggle }: { page: number; open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      aria-expanded={open}
      onClick={onToggle}
      className={`rounded-full border px-2 py-0.5 text-xs transition-colors ${
        open ? 'border-blue-400 bg-blue-50 text-blue-700' : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
      }`}
    >
      p.{page}
    </button>
  )
}
