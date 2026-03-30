import { useQuery } from '@tanstack/react-query';
import { Agent } from '../types/agent';
import { useDashboardStore } from '../store/dashboard-store';

export function useAgents() {
  const realtimeEnabled = useDashboardStore((state) => state.realtimeEnabled);

  return useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: async () => {
      const response = await fetch('/api/agents');
      if (!response.ok) {
        throw new Error('Failed to fetch agents');
      }
      const data = await response.json();
      return data.agents;
    },
    refetchInterval: realtimeEnabled ? 5000 : false, // Poll every 5 seconds if realtime enabled
    staleTime: 3000,
  });
}
