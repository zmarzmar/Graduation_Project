import { create } from 'zustand'

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
    set({ user: null, token: null, isInitialized: true })
  },
}))
