import { CheckCircle, HelpCircle, XCircle } from 'lucide-react'
import { execStatusLabel, isReviewSkipped } from '@/lib/exec-check'

const DESCRIPTION: Record<string, string> = {
  passed: '작은 입력으로 smoke_test를 실행해 오류 없이 끝났습니다. smoke_test도 AI가 쓴 것이고, 통과가 논문대로 구현됐다는 뜻은 아닙니다.',
  failed: '실행 오류를 정해진 횟수(3회) 안에 고치지 못했습니다. 마지막 코드는 실행 중 아래 오류로 멈췄습니다.',
  test_error: 'smoke_test의 반환 형식이 규칙과 달라 실행 결과를 판단하지 못했습니다.',
  unavailable: '실행 환경을 쓸 수 없어 실행 검사 없이 LLM 검토만 했습니다.',
  none: '이 기록은 실행 검사를 하지 않았습니다.',
}

/** 생성 코드 실행 검사 결과. 홈 결과와 마이페이지 상세가 함께 쓴다. */
export function ExecCheckStatus({ execStatus, execDetail }: { execStatus?: string; execDetail?: string }) {
  const failed = isReviewSkipped(execStatus)
  const key = failed ? 'failed' : execStatus && execStatus in DESCRIPTION ? execStatus : 'none'
  const Icon = execStatus === 'passed' ? CheckCircle : failed ? XCircle : HelpCircle
  const color = execStatus === 'passed' ? 'text-green-500' : failed ? 'text-red-500' : 'text-gray-400'

  return (
    <div className="mb-2 rounded-lg border border-gray-100 p-2.5 text-xs text-gray-600">
      <div className="flex items-center gap-1.5">
        <Icon className={`h-4 w-4 ${color}`} aria-hidden />
        <span className="font-semibold text-gray-800">{execStatusLabel(execStatus)}</span>
        <span className="text-gray-400">· 격리된 외부 샌드박스(E2B, 네트워크 차단)에서 실행</span>
      </div>
      <p className="mt-1">{DESCRIPTION[key]}</p>
      {execDetail && execStatus !== 'passed' && (
        <pre className="mt-1.5 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-[11px] text-gray-700">{execDetail}</pre>
      )}
    </div>
  )
}
