'use client';

import Link from 'next/link';
import { Users } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function UsersPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">사용자 관리</h1>
          <p className="mt-1 text-sm text-gray-500">관리자 사용자 조회 API 연결 전 화면입니다.</p>
        </div>
        <Link href="/admin" className="rounded-md bg-gray-100 px-3 py-2 text-sm text-gray-700 hover:bg-gray-200">
          관리자 홈
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Users className="h-4 w-4" />
            사용자 데이터 없음
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-600">
            기존 mock 사용자 목록은 삭제했습니다. 실제 사용자 목록은 백엔드 관리자 API가 추가되면 표시됩니다.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
