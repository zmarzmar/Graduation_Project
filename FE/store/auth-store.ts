import { create } from 'zustand'

export interface AuthUser {
  id: number
  email: string
  username: string
  full_name: string | null
  affiliation: string | null
  preferred_framework: string | null
}

interface AuthState {
  user: AuthUser | null
  token: string | null
  isInitialized: boolean
  isModalOpen: boolean
  modalDefaultTab: 'login' | 'register'

  setUser: (user: AuthUser | null) => void
  setToken: (token: string | null) => void
  setInitialized: (isInitialized: boolean) => void
  openModal: (tab?: 'login' | 'register') => void
  closeModal: () => void
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isInitialized: false,
  isModalOpen: false,
  modalDefaultTab: 'login',

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
  openModal: (tab = 'login') => set({ isModalOpen: true, modalDefaultTab: tab }),
  closeModal: () => set({ isModalOpen: false }),
  logout: () => {
    localStorage.removeItem('auth_token')
    set({ user: null, token: null, isInitialized: true })
  },
}))
