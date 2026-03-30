import { useQuery } from '@tanstack/react-query';
import { Protocol, ProtocolStatus } from '../types/protocol';
import { useDashboardStore } from '../store/dashboard-store';

interface UseProtocolsOptions {
  statuses?: ProtocolStatus[];
  search?: string;
  page?: number;
  pageSize?: number;
}

export function useProtocols(options: UseProtocolsOptions = {}) {
  const { statuses, search, page = 1, pageSize = 25 } = options;

  return useQuery<{ protocols: Protocol[]; total: number }>({
    queryKey: ['protocols', statuses, search, page, pageSize],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (statuses && statuses.length > 0) {
        params.append('statuses', statuses.join(','));
      }
      if (search) {
        params.append('search', search);
      }
      params.append('page', page.toString());
      params.append('pageSize', pageSize.toString());

      const response = await fetch(`/api/protocols?${params}`);
      if (!response.ok) {
        throw new Error('Failed to fetch protocols');
      }
      return response.json();
    },
    staleTime: 10000,
  });
}

export function useProtocol(id: string) {
  return useQuery<Protocol>({
    queryKey: ['protocol', id],
    queryFn: async () => {
      const response = await fetch(`/api/protocols/${id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch protocol');
      }
      const data = await response.json();
      return data.protocol;
    },
    enabled: !!id,
  });
}
