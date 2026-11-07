import { AlertTriangle } from 'lucide-react'

/** 생성 코드는 그대로 실행할 결과물이 아니라 참고용 초안이라는 경고. 홈 결과와 마이페이지 상세가 함께 쓴다. */
export function CodeDraftNotice() {
  return (
    <p className="mb-2 flex items-start gap-1.5 rounded-lg bg-amber-50 p-2.5 text-xs text-amber-800">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" aria-hidden />
      <span>
        AI가 논문을 읽고 만든 <strong>참고용 구현 초안</strong>입니다. 실행 검사는 작은 입력으로 오류 없이 돌아가는지만
        확인하므로, 통과했더라도 논문과 다르게 구현된 부분이 있을 수 있습니다. 그대로 사용하지 말고 논문과 대조하며 참고하세요.
      </span>
    </p>
  )
}
