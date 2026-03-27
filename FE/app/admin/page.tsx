'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Users, FileText, Activity, Database } from 'lucide-react';
import Link from 'next/link';
import { getUserStats } from '@/lib/mock-data/users';
import { getDatabaseStats } from '@/lib/mock-data/papers';
import { getSystemHealth } from '@/lib/mock-data/system';

export default function AdminPage() {
  const userStats = getUserStats();
  const dbStats = getDatabaseStats();
  const systemHealth = getSystemHealth();

  const stats = [
    {
      title: '총 사용자',
      value: userStats.totalUsers,
      description: `활성: ${userStats.activeUsers}명`,
      icon: Users,
      href: '/admin/users',
      color: 'text-blue-600',
    },
    {
      title: '저장된 논문',
      value: dbStats.totalPapers,
      description: `전체 텍스트: ${dbStats.papersWithFullText}개`,
      icon: FileText,
      href: '/admin/papers',
      color: 'text-green-600',
    },
    {
      title: '시스템 상태',
      value: systemHealth.overall === 'healthy' ? '정상' : '경고',
      description: `서비스: ${systemHealth.services.filter(s => s.status === 'healthy').length}/${systemHealth.services.length} 정상`,
      icon: Activity,
      href: '/admin/system',
      color: systemHealth.overall === 'healthy' ? 'text-green-600' : 'text-yellow-600',
    },
    {
      title: '데이터베이스',
      value: `${(dbStats.totalSize / (1024 * 1024 * 1024)).toFixed(1)}GB`,
      description: `${dbStats.totalPapers}개 논문`,
      icon: Database,
      href: '/admin/papers',
      color: 'text-purple-600',
    },
  ];

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-2">관리자 대시보드</h1>
        <p className="text-gray-600">시스템 관리 및 모니터링</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Link key={stat.title} href={stat.href}>
              <Card className="hover:shadow-lg transition-shadow cursor-pointer">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
                  <Icon className={`h-4 w-4 ${stat.color}`} />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <p className="text-xs text-gray-600 mt-1">{stat.description}</p>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>최근 활동</CardTitle>
            <CardDescription>시스템 로그 및 이벤트</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {systemHealth.recentLogs.slice(0, 5).map((log) => (
                <div key={log.id} className="flex items-start gap-3 text-sm">
                  <div
                    className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                      log.level === 'info'
                        ? 'bg-blue-500'
                        : log.level === 'warning'
                        ? 'bg-yellow-500'
                        : log.level === 'error'
                        ? 'bg-red-500'
                        : 'bg-purple-500'
                    }`}
                  />
                  <div className="flex-1">
                    <p className="font-medium">{log.message}</p>
                    <p className="text-xs text-gray-500">
                      {log.source} • {log.timestamp.toLocaleTimeString('ko-KR')}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API 사용 현황</CardTitle>
            <CardDescription>주요 엔드포인트 호출 통계</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {systemHealth.apiUsage.slice(0, 5).map((api) => (
                <div key={api.endpoint} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{api.endpoint}</span>
                    <span className="text-gray-600">{api.calls.toLocaleString()} 호출</span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>평균 {api.avgResponseTime}ms</span>
                    <span className={api.errorRate > 1 ? 'text-red-600' : 'text-green-600'}>
                      오류율 {api.errorRate}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Link href="/admin/users">
          <Card className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                사용자 관리
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                사용자 계정, 권한, 활동 기록을 관리합니다.
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/admin/papers">
          <Card className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                논문 데이터베이스
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                저장된 논문을 검색하고 관리합니다.
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/admin/system">
          <Card className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                시스템 모니터링
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                서버 상태, 성능, 로그를 모니터링합니다.
              </p>
            </CardContent>
          </Card>
        </Link>
      </div>
    </div>
  );
}
