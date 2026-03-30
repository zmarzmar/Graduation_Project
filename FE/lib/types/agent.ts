export enum AgentStatus {
  IDLE = 'idle',
  SEARCHING_PAPERS = 'searching_papers',
  ANALYZING_PAPERS = 'analyzing_papers',
  GENERATING_PROTOCOL = 'generating_protocol',
  VALIDATING = 'validating',
  COMPLETED = 'completed',
  ERROR = 'error',
  PAUSED = 'paused',
}

export enum AgentType {
  PAPER_RETRIEVAL = 'paper_retrieval',
  PAPER_ANALYZER = 'paper_analyzer',
  PROTOCOL_GENERATOR = 'protocol_generator',
  QUALITY_VALIDATOR = 'quality_validator',
  ORCHESTRATOR = 'orchestrator',
}

export interface AgentError {
  message: string;
  timestamp: string;
  recoverable: boolean;
}

export interface AgentMetadata {
  papersProcessed?: number;
  protocolsGenerated?: number;
  errorCount?: number;
  averageTime?: number;
}

export interface Agent {
  id: string;
  name: string;
  type: AgentType;
  status: AgentStatus;
  currentTask?: string;
  progress?: number; // 0-100
  startedAt?: string;
  lastUpdated: string;
  metadata?: AgentMetadata;
  error?: AgentError;
}
