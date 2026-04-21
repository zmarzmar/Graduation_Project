'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useAuthStore } from '@/store/auth-store'
import { useAuth } from '@/lib/hooks/useAuth'

type Tab = 'login' | 'register'

export function AuthModal() {
  const { isModalOpen, modalDefaultTab, closeModal } = useAuthStore()
  const { login, register } = useAuth()

  const [tab, setTab] = useState<Tab>(modalDefaultTab)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [username, setUsername] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // 탭 전환 시 입력값 초기화
  const switchTab = (t: Tab) => {
    setTab(t)
    setEmail('')
    setPassword('')
    setUsername('')
    setError('')
  }

  // 모달 열릴 때 기본 탭으로 초기화
  const handleOpenChange = (open: boolean) => {
    if (open) {
      setTab(modalDefaultTab)
      setEmail('')
      setPassword('')
      setUsername('')
      setError('')
    } else {
      closeModal()
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (tab === 'login') {
        await login(email, password)
      } else {
        await register(email, password, username)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '오류가 발생했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={isModalOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>PaperPilot</DialogTitle>
        </DialogHeader>

        {/* 탭 */}
        <div className="flex border-b mb-4">
          {(['login', 'register'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => switchTab(t)}
              className={`flex-1 py-2 text-sm font-medium transition-colors ${
                tab === t
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {t === 'login' ? '로그인' : '회원가입'}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          {tab === 'register' && (
            <Input
              placeholder="유저명 (2-50자)"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              minLength={2}
              maxLength={50}
              disabled={loading}
            />
          )}
          <Input
            type="email"
            placeholder="이메일"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            disabled={loading}
          />
          <Input
            type="password"
            placeholder={tab === 'register' ? '비밀번호 (8자 이상)' : '비밀번호'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={tab === 'register' ? 8 : undefined}
            disabled={loading}
          />

          {error && <p className="text-sm text-red-500">{error}</p>}

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? '처리 중...' : tab === 'login' ? '로그인' : '회원가입'}
          </Button>
        </form>

        <p className="text-center text-sm text-gray-500 mt-2">
          {tab === 'login' ? '계정이 없으신가요?' : '이미 계정이 있으신가요?'}{' '}
          <button
            onClick={() => switchTab(tab === 'login' ? 'register' : 'login')}
            className="text-blue-600 hover:underline font-medium"
          >
            {tab === 'login' ? '회원가입' : '로그인'}
          </button>
        </p>
      </DialogContent>
    </Dialog>
  )
}
