'use client'

import { Info } from 'lucide-react'
import { useAuthStore } from '@/store/auth-store'

/** 로그인 상태에서 분석하면 논문 추출 텍스트가 서버에 저장된다는 안내. 게스트 분석은 저장하지 않으므로 표시하지 않는다. */
export function DocumentStorageNotice() {
  const user = useAuthStore((state) => state.user)
  if (!user) return null

  return (
    <p className="mt-2 flex items-start gap-1.5 text-xs text-gray-500">
      <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" aria-hidden />
      <span>
        로그인 상태에서 분석한 논문의 추출 텍스트는 논문 Q&amp;A 제공을 위해 서버에 저장됩니다. 같은 문서를 사용하는
        분석 기록을 모두 삭제하면 저장된 원문도 삭제됩니다.
      </span>
    </p>
  )
}
