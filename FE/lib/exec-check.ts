// 생성 코드 실행 검사 상태 (BE agents/exec_check.py와 1:1)

export type ExecStatus = '' | 'passed' | 'code_error' | 'timeout' | 'oom' | 'test_error' | 'unavailable'

const CODE_ISSUES: string[] = ['code_error', 'timeout', 'oom']

/** 마지막 코드가 실행에 실패해 Reviewer(LLM)를 거치지 않았는지 — 이때 판정은 "LLM 검토 안 함"이다 (미통과와 구분) */
export function isReviewSkipped(execStatus: string | undefined): boolean {
  return CODE_ISSUES.includes(execStatus ?? '')
}

export function execStatusLabel(execStatus: string | undefined): string {
  switch (execStatus) {
    case 'passed': return '실행 통과'
    case 'code_error':
    case 'timeout':
    case 'oom': return '실행 실패'
    case 'test_error': return '실행 판단 불가'
    case 'unavailable': return '실행 확인 못 함'
    default: return '실행 검사 안 함'
  }
}
