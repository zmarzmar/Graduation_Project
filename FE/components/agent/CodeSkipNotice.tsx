import { Info } from 'lucide-react'

/** Coder가 구현 대상이 아니라고 판단해 코드를 만들지 않았을 때의 안내. 홈 결과와 마이페이지 상세가 함께 쓴다. */
export function CodeSkipNotice({ reason }: { reason: string }) {
  return (
    <div className="flex items-start gap-1.5 rounded-lg bg-gray-50 p-3 text-sm text-gray-700">
      <Info className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-500" aria-hidden />
      <div>
        <p className="font-medium">구현 초안 대상이 아닌 논문으로 판단해 코드를 만들지 않았습니다.</p>
        <p className="mt-1 text-gray-600">{reason}</p>
        <p className="mt-1 text-xs text-gray-500">AI(LLM)의 판단이므로 틀릴 수 있습니다.</p>
      </div>
    </div>
  )
}
