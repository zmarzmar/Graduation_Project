export interface PaperMetadata {
  pmid: string;
  pmcid?: string;
  title: string;
  authors: string[];
  journal: string;
  publicationDate: string;
  doi?: string;
  abstract: string;
  keywords: string[];
  citationCount: number;
}

export interface StoredPaper extends PaperMetadata {
  id: string;
  fullTextAvailable: boolean;
  downloadedAt: Date;
  fileSize?: number; // in bytes
  usedInProtocols: string[]; // protocol IDs
  accessCount: number;
  lastAccessedAt?: Date;
}

export interface PaperSearchResult {
  papers: StoredPaper[];
  total: number;
  page: number;
  pageSize: number;
}

export interface DatabaseStats {
  totalPapers: number;
  papersWithFullText: number;
  totalSize: number; // in bytes
  mostAccessedPapers: StoredPaper[];
  recentPapers: StoredPaper[];
}
