import { SystemHealth, SystemLog, APIUsage } from '../types/system';

export const generateSystemMetrics = () => {
  // Simulate varying metrics
  const cpuUsage = 45 + Math.random() * 20; // 45-65%
  const memoryUsed = 8 * 1024 * 1024 * 1024 + Math.random() * 2 * 1024 * 1024 * 1024; // 8-10GB
  const diskUsed = 150 * 1024 * 1024 * 1024; // 150GB

  return {
    cpu: {
      usage: Math.round(cpuUsage * 10) / 10,
      cores: 8,
    },
    memory: {
      used: Math.round(memoryUsed),
      total: 16 * 1024 * 1024 * 1024, // 16GB
      percentage: Math.round((memoryUsed / (16 * 1024 * 1024 * 1024)) * 100),
    },
    disk: {
      used: diskUsed,
      total: 512 * 1024 * 1024 * 1024, // 512GB
      percentage: Math.round((diskUsed / (512 * 1024 * 1024 * 1024)) * 100),
    },
    network: {
      inbound: Math.round(1024 * 1024 * (0.5 + Math.random())), // 0.5-1.5 MB/s
      outbound: Math.round(512 * 1024 * (0.3 + Math.random())), // 0.3-0.8 MB/s
    },
  };
};

export const mockAPIUsage: APIUsage[] = [
  {
    endpoint: '/api/protocols',
    method: 'GET',
    calls: 3456,
    avgResponseTime: 125,
    errorRate: 0.5,
    lastCalled: new Date('2025-02-02T10:35:00'),
  },
  {
    endpoint: '/api/agents',
    method: 'GET',
    calls: 2891,
    avgResponseTime: 89,
    errorRate: 0.2,
    lastCalled: new Date('2025-02-02T10:34:00'),
  },
  {
    endpoint: '/api/protocols/:id',
    method: 'GET',
    calls: 1567,
    avgResponseTime: 156,
    errorRate: 1.2,
    lastCalled: new Date('2025-02-02T10:30:00'),
  },
  {
    endpoint: '/api/papers/search',
    method: 'POST',
    calls: 890,
    avgResponseTime: 234,
    errorRate: 2.1,
    lastCalled: new Date('2025-02-02T10:28:00'),
  },
  {
    endpoint: '/api/users',
    method: 'GET',
    calls: 456,
    avgResponseTime: 67,
    errorRate: 0.3,
    lastCalled: new Date('2025-02-02T10:25:00'),
  },
];

export const mockSystemLogs: SystemLog[] = [
  {
    id: 'log-1',
    timestamp: new Date('2025-02-02T10:35:12'),
    level: 'info',
    message: 'Protocol generation completed successfully',
    source: 'ProtocolGenerator',
    details: { protocolId: 'protocol-12', duration: 4523 },
  },
  {
    id: 'log-2',
    timestamp: new Date('2025-02-02T10:34:45'),
    level: 'info',
    message: 'Agent status updated',
    source: 'SystemOrchestrator',
    details: { agentId: 'agent-1', status: 'searching' },
  },
  {
    id: 'log-3',
    timestamp: new Date('2025-02-02T10:33:28'),
    level: 'warning',
    message: 'High memory usage detected',
    source: 'SystemMonitor',
    details: { usage: 87.3, threshold: 85 },
  },
  {
    id: 'log-4',
    timestamp: new Date('2025-02-02T10:32:15'),
    level: 'error',
    message: 'Failed to fetch full text from PubMed',
    source: 'PubMedRetriever',
    details: { pmid: '12345678', error: 'PMC ID not found' },
  },
  {
    id: 'log-5',
    timestamp: new Date('2025-02-02T10:31:03'),
    level: 'info',
    message: 'User login successful',
    source: 'AuthService',
    details: { userId: 'user-2', email: 'researcher1@bioprotocol.org' },
  },
  {
    id: 'log-6',
    timestamp: new Date('2025-02-02T10:29:47'),
    level: 'warning',
    message: 'Rate limit approaching for PubMed API',
    source: 'PubMedRetriever',
    details: { current: 9500, limit: 10000 },
  },
  {
    id: 'log-7',
    timestamp: new Date('2025-02-02T10:28:22'),
    level: 'info',
    message: 'Database backup completed',
    source: 'DatabaseService',
    details: { size: '2.3GB', duration: 145 },
  },
  {
    id: 'log-8',
    timestamp: new Date('2025-02-02T10:26:10'),
    level: 'critical',
    message: 'Database connection lost',
    source: 'DatabaseService',
    details: { error: 'Connection timeout', retrying: true },
  },
];

export const getSystemHealth = (): SystemHealth => {
  const metrics = generateSystemMetrics();

  return {
    overall: 'healthy',
    services: [
      {
        name: 'API Server',
        status: 'healthy',
        uptime: 864532, // ~10 days
        lastCheck: new Date(),
        responseTime: 45,
      },
      {
        name: 'Database',
        status: 'healthy',
        uptime: 2592000, // 30 days
        lastCheck: new Date(),
        responseTime: 23,
      },
      {
        name: 'PubMed API',
        status: 'healthy',
        uptime: 8640000, // 100 days
        lastCheck: new Date(),
        responseTime: 156,
      },
      {
        name: 'LLM Service',
        status: 'degraded',
        uptime: 86400, // 1 day
        lastCheck: new Date(),
        responseTime: 890,
      },
      {
        name: 'Agent Orchestrator',
        status: 'healthy',
        uptime: 432000, // 5 days
        lastCheck: new Date(),
        responseTime: 34,
      },
    ],
    metrics,
    apiUsage: mockAPIUsage,
    recentLogs: mockSystemLogs.slice(0, 5),
  };
};
