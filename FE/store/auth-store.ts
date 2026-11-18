import { create } from 'zustand'
import { abortAllStreams } from '@/lib/stream-controllers'
import { useAnalysisStore } from '@/store/analysis-store'

export interface AuthUser {
  id: number
  email: string
  username: string
  full_name: string | null
  affiliation: string | null
  preferred_framework: string[] | null
  is_admin: boolean
}

interface AuthState {
  user: AuthUser | null
  token: string | null
  isInitialized: boolean
  isModalOpen: boolean
  modalDefaultTab: 'login' | 'register'
  modalMessage: string | null

  setUser: (user: AuthUser | null) => void
  setToken: (token: string | null) => void
  setInitialized: (isInitialized: boolean) => void
  openModal: (tab?: 'login' | 'register', message?: string) => void
  closeModal: () => void
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isInitialized: false,
  isModalOpen: false,
  modalDefaultTab: 'login',
  modalMessage: null,

  setUser: (user) => set({ user }),
  setToken: (token) => {
    if (token) {
      localStorage.setItem('auth_token', token)
    } else {
      localStorage.removeItem('auth_token')
    }
    set({ token })
  },
  setInitialized: (isInitialized) => set({ isInitialized }),
  openModal: (tab = 'login', message) => set({ isModalOpen: true, modalDefaultTab: tab, modalMessage: message ?? null }),
  closeModal: () => set({ isModalOpen: false, modalMessage: null }),
  logout: () => {
    localStorage.removeItem('auth_token')
    // 이전 사용자의 분석 결과(파일명·요약·코드)가 다음 사람에게 보이지 않게 — 한 기기를 여럿이 쓰는 시연 (2026-11-18 운영 확인)
    abortAllStreams()
    useAnalysisStore.getState().resetAll()
    set({ user: null, token: null, isInitialized: true })
  },
}))
