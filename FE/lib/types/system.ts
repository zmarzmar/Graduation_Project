export interface SystemMetrics {
  cpu: {
    usage: number; // percentage
    cores: number;
  };
  memory: {
    used: number; // in bytes
    total: number; // in bytes
    percentage: number;
  };
  disk: {
    used: number; // in bytes
    total: number; // in bytes
    percentage: number;
  };
  network: {
    inbound: number; // bytes per second
    outbound: number; // bytes per second
  };
}

export interface APIUsage {
  endpoint: string;
  method: string;
  calls: number;
  avgResponseTime: number; // milliseconds
  errorRate: number; // percentage
  lastCalled?: Date;
}

export interface SystemLog {
  id: string;
  timestamp: Date;
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  source: string;
  details?: Record<string, unknown>;
}

export interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  uptime: number; // seconds
  lastCheck: Date;
  responseTime?: number; // milliseconds
}

export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'critical';
  services: ServiceStatus[];
  metrics: SystemMetrics;
  apiUsage: APIUsage[];
  recentLogs: SystemLog[];
}
