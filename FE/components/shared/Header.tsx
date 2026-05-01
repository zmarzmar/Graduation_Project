'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BookOpen, LogOut, Search, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/lib/hooks/useAuth';

export function Header() {
  const pathname = usePathname();
  const { user, isLoggedIn, isAuthReady, logout, openModal } = useAuth();

  const isActive = (path: string) => {
    if (path === '/') return pathname === '/';
    return pathname.startsWith(path);
  };

  return (
    <header className="sticky top-0 z-50 border-b bg-white/80 backdrop-blur-md">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <BookOpen className="h-6 w-6 text-blue-600" />
            <div>
              <h1 className="text-xl font-bold">AI Research Analyst</h1>
              <p className="text-xs text-gray-500">AI 논문 자동 분석 및 코드 재현 Deep Research Agent</p>
            </div>
          </Link>

          <div className="flex items-center gap-2">
            <nav className="flex items-center gap-1">
              <Link
                href="/"
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
                  isActive('/') && !isActive('/mypage')
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Search className="h-4 w-4" />
                <span>논문 분석</span>
              </Link>
              <Link
                href="/mypage"
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
                  isActive('/mypage')
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <User className="h-4 w-4" />
                <span>마이페이지</span>
              </Link>
            </nav>

            {/* 로그인 상태에 따라 다른 UI */}
            <div className="flex items-center gap-2 pl-2 border-l ml-1">
              {!isAuthReady ? (
                <div className="h-8 w-28" aria-hidden="true" />
              ) : isLoggedIn ? (
                <>
                  <span className="text-sm text-gray-600 font-medium">
                    {user?.username}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={logout}
                    className="text-gray-500 hover:text-red-500"
                  >
                    <LogOut className="h-4 w-4" />
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => openModal('login')}
                    className="text-gray-600"
                  >
                    로그인
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => openModal('register')}
                  >
                    회원가입
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
