'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BookOpen, Search, User } from 'lucide-react';

export function Header() {
  const pathname = usePathname();

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
        </div>
      </div>
    </header>
  );
}
