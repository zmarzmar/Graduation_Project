'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FlaskConical, Home, Settings } from 'lucide-react';

export function Header() {
  const pathname = usePathname();

  const isActive = (path: string) => {
    if (path === '/') return pathname === '/';
    return pathname.startsWith(path);
  };

  return (
    <header className="border-b bg-white">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <FlaskConical className="h-6 w-6 text-blue-600" />
            <div>
              <h1 className="text-xl font-bold">Bio Protocol Generator</h1>
              <p className="text-xs text-gray-500">Auto-generated experimental protocols from PubMed</p>
            </div>
          </Link>

          <nav className="flex items-center gap-1">
            <Link
              href="/"
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
                isActive('/') && !isActive('/admin')
                  ? 'bg-blue-50 text-blue-700 font-medium'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Home className="h-4 w-4" />
              <span>대시보드</span>
            </Link>
            <Link
              href="/admin"
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
                isActive('/admin')
                  ? 'bg-purple-50 text-purple-700 font-medium'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Settings className="h-4 w-4" />
              <span>관리자</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}
