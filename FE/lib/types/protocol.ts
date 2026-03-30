export enum ProtocolStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  VALIDATING = 'validating',
}

export interface ProtocolPaper {
  pmid: string;
  pmcid?: string;
  title: string;
  authors: string[];
  journal?: string;
  year?: number;
  relevanceScore: number; // 0-1
  citedSections: string[]; // Which protocol sections cite this paper
}

export interface ProtocolStep {
  stepNumber: number;
  title: string;
  description: string;
  duration: string;
  materials: string[];
  warnings?: string[];
  references: string[]; // PMIDs
}

export interface ProtocolContent {
  materials: string[];
  steps: ProtocolStep[];
  notes: string[];
  estimatedTime: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

export interface ProtocolMetadata {
  version: number;
  papersAnalyzed: number;
  confidenceScore?: number; // 0-1
  tags: string[];
}

export interface Protocol {
  id: string;
  title: string;
  description: string;
  status: ProtocolStatus;
  query: string; // Original user query
  createdAt: string;
  completedAt?: string;
  generatedBy: string; // Agent ID
  papers: ProtocolPaper[];
  content?: ProtocolContent;
  metadata: ProtocolMetadata;
}
