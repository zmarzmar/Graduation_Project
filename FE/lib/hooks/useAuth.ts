'use client'

import { useCallback } from 'react'
import { getMe, login as apiLogin, register as apiRegister } from '@/lib/api'
import { useAuthStore } from '@/store/auth-store'

export function useAuth() {
  const { user, token, openModal, closeModal, setUser, setToken, logout } = useAuthStore()

  /** 앱 시작 시 localStorage 토큰으로 유저 정보 복원 */
  const initAuth = useCallback(async () => {
    const stored = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    if (!stored) return
    try {
      setToken(stored)
      const me = await getMe()
      setUser(me)
    } catch {
      // 토큰 만료 — 조용히 로그아웃
      logout()
    }
  }, [setToken, setUser, logout])

  /** 로그인 후 callback 실행이 필요한 액션에 사용 */
  const requireAuth = useCallback(
    (callback?: () => void) => {
      if (user) {
        callback?.()
      } else {
        openModal('login')
      }
    },
    [user, openModal],
  )

  const login = useCallback(
    async (email: string, password: string) => {
      const { access_token } = await apiLogin(email, password)
      setToken(access_token)
      const me = await getMe()
      setUser(me)
      closeModal()
    },
    [setToken, setUser, closeModal],
  )

  const register = useCallback(
    async (email: string, password: string, username: string) => {
      const { access_token } = await apiRegister(email, password, username)
      setToken(access_token)
      const me = await getMe()
      setUser(me)
      closeModal()
    },
    [setToken, setUser, closeModal],
  )

  return { user, token, isLoggedIn: !!user, requireAuth, login, register, logout, initAuth, openModal }
}
