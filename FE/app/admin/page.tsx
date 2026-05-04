'use client';

import Link from 'next/link';
import { Activity, Database, FileText, Users } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const sections = [
  {
    title: '사용자 관리',
    description: '실제 사용자 관리 API가 연결되면 이 화면에서 계정을 관리합니다.',
    href: '/admin/users',
    icon: Users,
  },
  {
    title: '논문 DB',
    description: '저장된 논문 조회 API가 연결되면 이 화면에서 논문 데이터를 확인합니다.',
    href: '/admin/papers',
    icon: FileText,
  },
  {
    title: '시스템 모니터링',
    description: '운영 지표 API가 연결되면 이 화면에서 서비스 상태를 확인합니다.',
    href: '/admin/system',
    icon: Activity,
  },
];

export default function AdminPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">관리자 대시보드</h1>
        <p className="mt-1 text-sm text-gray-500">
          관리자 권한이 있는 계정만 접근할 수 있습니다.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Database className="h-4 w-4" />
            운영 데이터 연동 상태
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-600">
            더미 데이터는 제거했습니다. 사용자, 논문, 시스템 지표는 별도 관리자 API가 추가되면 실제 데이터로 표시됩니다.
          </p>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {sections.map((section) => {
          const Icon = section.icon;
          return (
            <Link key={section.href} href={section.href}>
              <Card className="h-full transition-shadow hover:shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Icon className="h-4 w-4 text-purple-600" />
                    {section.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-gray-600">{section.description}</p>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
