import type { StreamMode } from '@/store/analysis-store'

// 실행 상태(isRunning)는 store에 있어 페이지를 나가도 유지된다. 취소 핸들도 같은 수명(모듈)에 두어
// 페이지 복귀 후에도 진행 중인 스트림을 취소할 수 있게 한다. (직렬화 불가 객체라 store에는 넣지 않는다)
export const streamControllers = new Map<StreamMode, AbortController>()

/** 진행 중인 모든 에이전트 스트림을 취소한다 — 로그아웃 때 이전 사용자의 요청이 결과를 다시 채우지 않게 */
export function abortAllStreams(): void {
  for (const controller of streamControllers.values()) controller.abort()
  streamControllers.clear()
}
