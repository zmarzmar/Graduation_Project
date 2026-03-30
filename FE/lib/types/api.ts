import { Agent } from './agent';
import { Protocol } from './protocol';

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface AgentsResponse {
  agents: Agent[];
  timestamp: string;
}

export interface ProtocolsResponse {
  protocols: Protocol[];
  total: number;
  page: number;
  pageSize: number;
}

export interface ProtocolDetailResponse {
  protocol: Protocol;
  relatedProtocols?: Protocol[];
}

export interface DashboardStats {
  totalProtocols: number;
  completedProtocols: number;
  failedProtocols: number;
  activeAgents: number;
  successRate: number;
  averageGenerationTime: string;
}
