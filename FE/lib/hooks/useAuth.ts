'use client'

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { getMe, login as apiLogin, register as apiRegister } from '@/lib/api'
import { useAuthStore } from '@/store/auth-store'

export function useAuth() {
  const router = useRouter()
  const { user, token, isInitialized, openModal, closeModal, setUser, setToken, setInitialized, logout } = useAuthStore()

  /** 앱 시작 시 localStorage 토큰으로 유저 정보 복원 */
  const initAuth = useCallback(async () => {
    const stored = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    if (!stored) {
      setInitialized(true)
      return
    }
    try {
      setToken(stored)
      const me = await getMe()
      setUser(me)
    } catch {
      // 토큰 만료 — 조용히 로그아웃
      logout()
    } finally {
      setInitialized(true)
    }
  }, [setToken, setUser, setInitialized, logout])

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
    async (identifier: string, password: string) => {
      const { access_token } = await apiLogin(identifier, password)
      setToken(access_token)
      const me = await getMe()
      setUser(me)
      setInitialized(true)
      closeModal()
      if (me.is_admin) router.push('/admin')
    },
    [setToken, setUser, setInitialized, closeModal, router],
  )

  const register = useCallback(
    async (email: string, password: string, username: string, fullName?: string) => {
      const { access_token } = await apiRegister(email, password, username, fullName)
      setToken(access_token)
      const me = await getMe()
      setUser(me)
      setInitialized(true)
      closeModal()
    },
    [setToken, setUser, setInitialized, closeModal],
  )

  return { user, token, isLoggedIn: !!user, isAuthReady: isInitialized, requireAuth, login, register, logout, initAuth, openModal }
}
