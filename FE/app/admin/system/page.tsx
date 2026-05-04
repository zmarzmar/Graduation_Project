'use client';

import Link from 'next/link';
import { Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function SystemPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">시스템 모니터링</h1>
          <p className="mt-1 text-sm text-gray-500">운영 지표 API 연결 전 화면입니다.</p>
        </div>
        <Link href="/admin" className="rounded-md bg-gray-100 px-3 py-2 text-sm text-gray-700 hover:bg-gray-200">
          관리자 홈
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="h-4 w-4" />
            시스템 지표 없음
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-600">
            기존 mock 시스템 지표는 삭제했습니다. 실제 운영 지표는 백엔드 관리자 API가 추가되면 표시됩니다.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
