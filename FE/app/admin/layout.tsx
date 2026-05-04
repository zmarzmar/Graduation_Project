'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Users, FileText, Activity, Lock } from 'lucide-react';
import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/lib/hooks/useAuth';

const navItems = [
  {
    title: '개요',
    href: '/admin',
    icon: LayoutDashboard,
  },
  {
    title: '사용자 관리',
    href: '/admin/users',
    icon: Users,
  },
  {
    title: '논문 DB',
    href: '/admin/papers',
    icon: FileText,
  },
  {
    title: '시스템 모니터링',
    href: '/admin/system',
    icon: Activity,
  },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const { user, isAuthReady, isLoggedIn, openModal } = useAuth();

  useEffect(() => {
    if (isAuthReady && !isLoggedIn) openModal('login');
  }, [isAuthReady, isLoggedIn, openModal]);

  if (!isAuthReady) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6 text-sm text-gray-500">
        권한을 확인하는 중입니다.
      </div>
    );
  }

  if (!isLoggedIn) {
    return (
      <div className="mx-auto max-w-md rounded-lg border border-gray-200 bg-white p-6 text-center">
        <Lock className="mx-auto h-8 w-8 text-gray-400" />
        <h1 className="mt-3 text-lg font-semibold text-gray-900">관리자 로그인이 필요합니다.</h1>
        <p className="mt-1 text-sm text-gray-500">관리자 계정으로 로그인하면 이 화면을 볼 수 있습니다.</p>
        <Button className="mt-4" onClick={() => openModal('login')}>로그인</Button>
      </div>
    );
  }

  if (!user?.is_admin) {
    return (
      <div className="mx-auto max-w-md rounded-lg border border-gray-200 bg-white p-6 text-center">
        <Lock className="mx-auto h-8 w-8 text-red-400" />
        <h1 className="mt-3 text-lg font-semibold text-gray-900">관리자 권한이 없습니다.</h1>
        <p className="mt-1 text-sm text-gray-500">관리자 계정으로 다시 로그인해주세요.</p>
      </div>
    );
  }

  return (
    <div className="flex gap-6">
      <aside className="w-64 flex-shrink-0">
        <div className="sticky top-6 space-y-1">
          <h2 className="px-3 mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            관리자 메뉴
          </h2>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                    isActive
                      ? 'bg-purple-50 text-purple-700 font-medium'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{item.title}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </aside>

      <div className="flex-1 min-w-0">
        {children}
      </div>
    </div>
  );
}
