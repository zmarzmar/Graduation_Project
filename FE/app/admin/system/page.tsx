'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Activity,
  Cpu,
  HardDrive,
  Network,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import { getSystemHealth } from '@/lib/mock-data/system';
import Link from 'next/link';

export default function SystemPage() {
  const [systemHealth, setSystemHealth] = useState(getSystemHealth());

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemHealth(getSystemHealth());
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(2)} GB`;
  };

  const formatBytesPerSecond = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB/s`;
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}일 ${hours}시간 ${minutes}분`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'degraded':
        return 'text-yellow-600 bg-yellow-100';
      case 'down':
      case 'critical':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'info':
        return 'bg-blue-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'error':
        return 'bg-red-500';
      case 'critical':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  const metrics = systemHealth.metrics;

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">시스템 모니터링</h1>
          <p className="text-gray-600">서버 상태 및 성능 지표</p>
        </div>
        <Link
          href="/admin"
          className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
        >
          ← 관리자 홈
        </Link>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>시스템 상태</CardTitle>
              <CardDescription>전체 시스템 헬스 체크</CardDescription>
            </div>
            <div className={`px-4 py-2 rounded-full font-medium ${getStatusColor(systemHealth.overall)}`}>
              {systemHealth.overall === 'healthy' ? '정상' : systemHealth.overall === 'degraded' ? '경고' : '위험'}
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU 사용률</CardTitle>
            <Cpu className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.cpu.usage.toFixed(1)}%</div>
            <p className="text-xs text-gray-600 mt-1">{metrics.cpu.cores} 코어</p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${metrics.cpu.usage}%` }}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">메모리</CardTitle>
            <Activity className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.memory.percentage}%</div>
            <p className="text-xs text-gray-600 mt-1">
              {formatBytes(metrics.memory.used)} / {formatBytes(metrics.memory.total)}
            </p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-600 h-2 rounded-full transition-all"
                style={{ width: `${metrics.memory.percentage}%` }}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">디스크</CardTitle>
            <HardDrive className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.disk.percentage}%</div>
            <p className="text-xs text-gray-600 mt-1">
              {formatBytes(metrics.disk.used)} / {formatBytes(metrics.disk.total)}
            </p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-600 h-2 rounded-full transition-all"
                style={{ width: `${metrics.disk.percentage}%` }}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">네트워크</CardTitle>
            <Network className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">수신:</span>
                <span className="font-medium">
                  {formatBytesPerSecond(metrics.network.inbound)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">송신:</span>
                <span className="font-medium">
                  {formatBytesPerSecond(metrics.network.outbound)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>서비스 상태</CardTitle>
          <CardDescription>각 서비스의 실시간 상태 및 응답 시간</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {systemHealth.services.map((service) => (
              <div key={service.name} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center gap-3">
                  {service.status === 'healthy' ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-yellow-600" />
                  )}
                  <div>
                    <p className="font-medium">{service.name}</p>
                    <p className="text-xs text-gray-600">
                      가동 시간: {formatUptime(service.uptime)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`px-3 py-1 rounded-full text-xs font-medium mb-1 ${getStatusColor(service.status)}`}>
                    {service.status === 'healthy' ? '정상' : service.status === 'degraded' ? '저하' : '중단'}
                  </div>
                  {service.responseTime && (
                    <p className="text-xs text-gray-600">{service.responseTime}ms</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>API 사용 통계</CardTitle>
            <CardDescription>엔드포인트별 호출 및 성능 지표</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {systemHealth.apiUsage.map((api) => (
                <div key={`${api.endpoint}-${api.method}`} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{api.endpoint}</span>
                    <span className="text-xs text-gray-600 px-2 py-1 bg-gray-100 rounded">
                      {api.method}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <span className="text-gray-600">호출:</span>{' '}
                      <span className="font-medium">{api.calls.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">응답:</span>{' '}
                      <span className="font-medium">{api.avgResponseTime}ms</span>
                    </div>
                    <div>
                      <span className="text-gray-600">오류율:</span>{' '}
                      <span className={`font-medium ${api.errorRate > 1 ? 'text-red-600' : 'text-green-600'}`}>
                        {api.errorRate}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>시스템 로그</CardTitle>
            <CardDescription>최근 시스템 이벤트 및 오류</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {systemHealth.recentLogs.map((log) => (
                <div key={log.id} className="flex items-start gap-3 text-sm">
                  <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${getLogLevelColor(log.level)}`} />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium line-clamp-2">{log.message}</p>
                    <p className="text-xs text-gray-500">
                      {log.source} • {log.timestamp.toLocaleTimeString('ko-KR')}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
