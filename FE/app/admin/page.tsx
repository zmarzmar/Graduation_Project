'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { Activity, Database, FileText, Users } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAdminSystemSummary, type AdminSystemSummary } from '@/lib/api';

const sections = [
  {
    title: '사용자 관리',
    description: '실제 DB 사용자 목록과 계정 상태를 확인합니다.',
    href: '/admin/users',
    icon: Users,
  },
  {
    title: '논문 DB',
    description: '저장된 논문 메타데이터와 인용 정보를 확인합니다.',
    href: '/admin/papers',
    icon: FileText,
  },
  {
    title: '시스템 모니터링',
    description: 'API, DB 상태와 주요 데이터 개수를 확인합니다.',
    href: '/admin/system',
    icon: Activity,
  },
];

export default function AdminPage() {
  const [summary, setSummary] = useState<AdminSystemSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getAdminSystemSummary()
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : '관리자 요약 조회 실패');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

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
          {loading ? (
            <p className="text-sm text-gray-500">관리자 데이터를 불러오는 중입니다.</p>
          ) : error ? (
            <p className="text-sm text-red-600">{error}</p>
          ) : summary ? (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              <div>
                <p className="text-xs text-gray-500">API</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{summary.status}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">DB</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{summary.database}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">사용자</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{summary.counts.users}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">논문</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{summary.counts.papers}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">분석</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{summary.counts.analysis_results}</p>
              </div>
            </div>
          ) : null}
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
