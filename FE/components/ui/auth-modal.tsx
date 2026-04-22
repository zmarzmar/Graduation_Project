'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
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
  const [passwordConfirm, setPasswordConfirm] = useState('')
  const [username, setUsername] = useState('')
  const [fullName, setFullName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const resetForm = () => {
    setEmail('')
    setPassword('')
    setPasswordConfirm('')
    setUsername('')
    setFullName('')
    setError('')
  }

  const switchTab = (t: Tab) => {
    setTab(t)
    resetForm()
  }

  const handleOpenChange = (open: boolean) => {
    if (open) {
      setTab(modalDefaultTab)
      resetForm()
    } else {
      closeModal()
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (tab === 'register' && password !== passwordConfirm) {
      setError('비밀번호가 일치하지 않습니다.')
      return
    }

    setLoading(true)
    try {
      if (tab === 'login') {
        await login(email, password)
      } else {
        await register(email, password, username, fullName || undefined)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '오류가 발생했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={isModalOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader className="pb-2">
          <DialogTitle className="text-xl">
            {tab === 'login' ? '로그인' : '회원가입'}
          </DialogTitle>
          <DialogDescription>
            {tab === 'login'
              ? 'PaperPilot에 오신 걸 환영합니다'
              : '계정을 만들고 분석 기록을 저장하세요'}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-3 pt-1">
          {tab === 'register' && (
            <>
              <Input
                placeholder="이름 (선택)"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                disabled={loading}
              />
              <Input
                placeholder="유저명 (2-50자)"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={2}
                maxLength={50}
                disabled={loading}
              />
            </>
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
          {tab === 'register' && (
            <Input
              type="password"
              placeholder="비밀번호 확인"
              value={passwordConfirm}
              onChange={(e) => setPasswordConfirm(e.target.value)}
              required
              disabled={loading}
            />
          )}

          {error && <p className="text-sm text-red-500">{error}</p>}

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? '처리 중...' : tab === 'login' ? '로그인' : '회원가입'}
          </Button>
        </form>

        <div className="relative my-3">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t" />
          </div>
          <div className="relative flex justify-center text-xs text-gray-400">
            <span className="bg-white px-2">
              {tab === 'login' ? '계정이 없으신가요?' : '이미 계정이 있으신가요?'}
            </span>
          </div>
        </div>

        <Button
          type="button"
          variant="outline"
          className="w-full"
          onClick={() => switchTab(tab === 'login' ? 'register' : 'login')}
          disabled={loading}
        >
          {tab === 'login' ? '회원가입' : '로그인'}
        </Button>
      </DialogContent>
    </Dialog>
  )
}
