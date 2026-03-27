import { Agent, AgentStatus, AgentType } from '@/lib/types/agent';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle, Loader2, Pause } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface AgentCardProps {
  agent: Agent;
}

function getStatusColor(status: AgentStatus): string {
  switch (status) {
    case AgentStatus.IDLE:
      return 'bg-gray-100 text-gray-800 border-gray-300';
    case AgentStatus.SEARCHING_PAPERS:
    case AgentStatus.ANALYZING_PAPERS:
    case AgentStatus.GENERATING_PROTOCOL:
      return 'bg-blue-100 text-blue-800 border-blue-300';
    case AgentStatus.VALIDATING:
      return 'bg-purple-100 text-purple-800 border-purple-300';
    case AgentStatus.COMPLETED:
      return 'bg-green-100 text-green-800 border-green-300';
    case AgentStatus.ERROR:
      return 'bg-red-100 text-red-800 border-red-300';
    case AgentStatus.PAUSED:
      return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-300';
  }
}

function getStatusIcon(status: AgentStatus) {
  switch (status) {
    case AgentStatus.ERROR:
      return <AlertCircle className="h-4 w-4" />;
    case AgentStatus.COMPLETED:
      return <CheckCircle className="h-4 w-4" />;
    case AgentStatus.PAUSED:
      return <Pause className="h-4 w-4" />;
    case AgentStatus.IDLE:
      return null;
    default:
      return <Loader2 className="h-4 w-4 animate-spin" />;
  }
}

function formatAgentType(type: AgentType): string {
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function formatStatus(status: AgentStatus): string {
  return status
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function AgentCard({ agent }: AgentCardProps) {
  const isActive = agent.progress !== undefined && agent.progress < 100;

  return (
    <Card className={`${isActive ? 'ring-2 ring-blue-400' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base font-semibold">{agent.name}</CardTitle>
            <p className="text-xs text-gray-500 mt-1">{formatAgentType(agent.type)}</p>
          </div>
          <Badge variant="outline" className={`${getStatusColor(agent.status)} flex items-center gap-1`}>
            {getStatusIcon(agent.status)}
            {formatStatus(agent.status)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {agent.currentTask && (
          <div className="text-sm text-gray-700">
            <p className="font-medium text-xs text-gray-500 mb-1">Current Task:</p>
            <p className="line-clamp-2">{agent.currentTask}</p>
          </div>
        )}

        {agent.progress !== undefined && (
          <div>
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
              <span>Progress</span>
              <span>{agent.progress}%</span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all duration-500"
                style={{ width: `${agent.progress}%` }}
              />
            </div>
          </div>
        )}

        {agent.metadata && (
          <div className="grid grid-cols-2 gap-2 pt-2 border-t text-xs">
            {agent.metadata.papersProcessed !== undefined && (
              <div>
                <p className="text-gray-500">Papers Processed</p>
                <p className="font-semibold">{agent.metadata.papersProcessed}</p>
              </div>
            )}
            {agent.metadata.protocolsGenerated !== undefined && (
              <div>
                <p className="text-gray-500">Protocols Generated</p>
                <p className="font-semibold">{agent.metadata.protocolsGenerated}</p>
              </div>
            )}
            {agent.metadata.averageTime !== undefined && (
              <div>
                <p className="text-gray-500">Avg Time</p>
                <p className="font-semibold">{agent.metadata.averageTime}s</p>
              </div>
            )}
            {agent.metadata.errorCount !== undefined && (
              <div>
                <p className="text-gray-500">Errors</p>
                <p className="font-semibold">{agent.metadata.errorCount}</p>
              </div>
            )}
          </div>
        )}

        {agent.error && (
          <div className="bg-red-50 border border-red-200 rounded p-2 text-xs">
            <p className="font-semibold text-red-800">Error:</p>
            <p className="text-red-700">{agent.error.message}</p>
          </div>
        )}

        <p className="text-xs text-gray-400">
          Updated {formatDistanceToNow(new Date(agent.lastUpdated), { addSuffix: true })}
        </p>
      </CardContent>
    </Card>
  );
}
