import { Agent, AgentType, AgentStatus } from '../types/agent';

export const mockAgents: Agent[] = [
  {
    id: 'agent-1',
    name: 'PubMed Retriever',
    type: AgentType.PAPER_RETRIEVAL,
    status: AgentStatus.SEARCHING_PAPERS,
    currentTask: 'Searching PubMed for "CRISPR gene editing protocols"',
    progress: 67,
    startedAt: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
    lastUpdated: new Date().toISOString(),
    metadata: {
      papersProcessed: 156,
      errorCount: 2,
      averageTime: 45,
    },
  },
  {
    id: 'agent-2',
    name: 'Protocol Analyzer',
    type: AgentType.PAPER_ANALYZER,
    status: AgentStatus.ANALYZING_PAPERS,
    currentTask: 'Analyzing 12 papers for Western Blot protocol extraction',
    progress: 40,
    startedAt: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
    lastUpdated: new Date().toISOString(),
    metadata: {
      papersProcessed: 423,
      errorCount: 8,
      averageTime: 120,
    },
  },
  {
    id: 'agent-3',
    name: 'Protocol Generator',
    type: AgentType.PROTOCOL_GENERATOR,
    status: AgentStatus.IDLE,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
    metadata: {
      protocolsGenerated: 78,
      errorCount: 3,
      averageTime: 180,
    },
  },
  {
    id: 'agent-4',
    name: 'Quality Validator',
    type: AgentType.QUALITY_VALIDATOR,
    status: AgentStatus.IDLE,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
    metadata: {
      protocolsGenerated: 78,
      errorCount: 1,
      averageTime: 60,
    },
  },
  {
    id: 'agent-5',
    name: 'System Orchestrator',
    type: AgentType.ORCHESTRATOR,
    status: AgentStatus.COMPLETED,
    currentTask: 'Coordinated protocol generation for query "PCR amplification"',
    progress: 100,
    startedAt: new Date(Date.now() - 1000 * 60 * 20).toISOString(),
    lastUpdated: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
    metadata: {
      protocolsGenerated: 156,
      errorCount: 5,
      averageTime: 300,
    },
  },
];

// Simulate real-time updates
export function simulateAgentUpdate(agent: Agent): Agent {
  const now = new Date().toISOString();

  // If agent is in progress, increment progress
  if (agent.progress !== undefined && agent.progress < 100) {
    const newProgress = Math.min(100, agent.progress + Math.random() * 10);

    // Status transitions
    let newStatus = agent.status;
    if (newProgress >= 100) {
      newStatus = AgentStatus.COMPLETED;
    }

    return {
      ...agent,
      progress: Math.floor(newProgress),
      status: newStatus,
      lastUpdated: now,
    };
  }

  // If completed, maybe reset to idle after some time
  if (agent.status === AgentStatus.COMPLETED) {
    const timeSinceComplete = Date.now() - new Date(agent.lastUpdated).getTime();
    if (timeSinceComplete > 30000) { // 30 seconds
      return {
        ...agent,
        status: AgentStatus.IDLE,
        currentTask: undefined,
        progress: undefined,
        lastUpdated: now,
      };
    }
  }

  return agent;
}
