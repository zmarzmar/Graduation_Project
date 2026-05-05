'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { Activity, Database } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAdminSystemSummary, type AdminSystemSummary } from '@/lib/api';

const countLabels: Record<keyof AdminSystemSummary['counts'], string> = {
  users: '전체 사용자',
  active_users: '활성 사용자',
  admin_users: '관리자',
  papers: '저장 논문',
  search_history: '검색 기록',
  analysis_results: '분석 결과',
};

export default function SystemPage() {
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
        if (!cancelled) setError(err instanceof Error ? err.message : '시스템 요약 조회 실패');
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">시스템 모니터링</h1>
          <p className="mt-1 text-sm text-gray-500">API, DB 연결 상태와 주요 데이터 개수를 확인합니다.</p>
        </div>
        <Link href="/admin" className="rounded-md bg-gray-100 px-3 py-2 text-sm text-gray-700 hover:bg-gray-200">
          관리자 홈
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="h-4 w-4" />
            시스템 상태
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-sm text-gray-500">시스템 상태를 불러오는 중입니다.</p>
          ) : error ? (
            <p className="text-sm text-red-600">{error}</p>
          ) : summary ? (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="rounded-lg border border-gray-100 p-4">
                <p className="text-xs text-gray-500">API 상태</p>
                <p className="mt-1 text-lg font-semibold text-green-600">{summary.status}</p>
              </div>
              <div className="rounded-lg border border-gray-100 p-4">
                <p className="text-xs text-gray-500">DB 상태</p>
                <p className="mt-1 text-lg font-semibold text-green-600">{summary.database}</p>
              </div>
              <div className="rounded-lg border border-gray-100 p-4">
                <p className="text-xs text-gray-500">API 버전</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{summary.version}</p>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {summary && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Database className="h-4 w-4" />
              데이터 요약
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              {(Object.entries(summary.counts) as Array<[keyof AdminSystemSummary['counts'], number]>).map(([key, value]) => (
                <div key={key} className="rounded-lg border border-gray-100 p-4">
                  <p className="text-xs text-gray-500">{countLabels[key]}</p>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">{value}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
