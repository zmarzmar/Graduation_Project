'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useProtocols } from '@/lib/hooks/useProtocols';
import { useAgents } from '@/lib/hooks/useAgents';
import { ProtocolStatus } from '@/lib/types/protocol';
import { AgentStatus } from '@/lib/types/agent';
import { FileText, CheckCircle, XCircle, Activity, TrendingUp } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';

export function StatsOverview() {
  const { data: protocolsData } = useProtocols({ pageSize: 1000 }); // Get all for stats
  const { data: agents } = useAgents();

  const protocols = protocolsData?.protocols || [];
  const totalProtocols = protocols.length;
  const completedProtocols = protocols.filter((p) => p.status === ProtocolStatus.COMPLETED).length;
  const failedProtocols = protocols.filter((p) => p.status === ProtocolStatus.FAILED).length;
  const activeAgents =
    agents?.filter(
      (a) =>
        a.status !== AgentStatus.IDLE &&
        a.status !== AgentStatus.COMPLETED &&
        a.status !== AgentStatus.ERROR
    ).length || 0;

  const successRate = totalProtocols > 0 ? ((completedProtocols / totalProtocols) * 100).toFixed(1) : '0';

  const isLoading = !protocolsData || !agents;

  const stats = [
    {
      title: 'Total Protocols',
      value: totalProtocols,
      icon: FileText,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      title: 'Completed',
      value: completedProtocols,
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      title: 'Failed',
      value: failedProtocols,
      icon: XCircle,
      color: 'text-red-600',
      bgColor: 'bg-red-100',
    },
    {
      title: 'Active Agents',
      value: activeAgents,
      icon: Activity,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      title: 'Success Rate',
      value: `${successRate}%`,
      icon: TrendingUp,
      color: 'text-emerald-600',
      bgColor: 'bg-emerald-100',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
      {stats.map((stat) => (
        <Card key={stat.title}>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">{stat.title}</p>
                {isLoading ? (
                  <Skeleton className="h-8 w-16 mt-2" />
                ) : (
                  <p className="text-2xl font-bold mt-2">{stat.value}</p>
                )}
              </div>
              <div className={`${stat.bgColor} p-3 rounded-lg`}>
                <stat.icon className={`h-6 w-6 ${stat.color}`} />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
