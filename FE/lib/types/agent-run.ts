// BE의 AgentState / SSE 이벤트와 1:1 대응하는 타입 정의

export type AgentMode = 'pdf' | 'search' | 'trend' | 'analyze'
export type NodeName = 'planner' | 'researcher' | 'trend_analyzer' | 'analyzer' | 'coder' | 'reviewer'
export type NodeStatus = 'pending' | 'running' | 'done' | 'error'

/** BE PaperResult 모델과 동일한 구조 */
export interface ArxivPaper {
  arxiv_id: string
  title: string
  authors: string[]
  abstract: string
  url: string
  pdf_url: string
  published_at: string
  categories: string[]
  citation_count: number | null
  influential_citation_count: number | null
  tldr: string | null
  s2_paper_id: string | null
  upvotes?: number // HuggingFace 트렌드 모드 전용
}

/** 트렌드 분석 결과 */
export interface TrendAnalysis {
  paper_summaries: Array<{
    title: string
    summary: string
  }>
  trending_keywords: Array<{
    keyword: string
    explanation: string
  }>
}

/** 핵심 수식 */
export interface KeyFormula {
  name: string
  latex: string
  description: string
}

/** 논문 리뷰 */
export interface PaperReview {
  strengths: string[]
  limitations: string[]
  significance: string
}

/** 에이전트 파이프라인 최종 결과물 */
export interface AgentResult {
  papers: ArxivPaper[]
  paper_summary?: string
  paper_review?: PaperReview
  key_formulas?: KeyFormula[]
  generated_code?: string
  review_feedback?: string
  review_passed?: boolean
  iteration_count?: number
  mode?: string
  trend_analysis?: TrendAnalysis
}

/** BE SSE 이벤트 구조 */
export interface AgentEvent {
  event: 'node_start' | 'node_done' | 'log' | 'complete' | 'error'
  node?: NodeName
  plan_summary?: string
  papers_count?: number
  iteration?: number
  review_passed?: boolean
  review_feedback?: string
  error?: string
  result?: AgentResult
  message?: string
}

export type NodeStatuses = Record<NodeName, NodeStatus>

/** 노드별 실시간 로그 메시지 목록 */
export type NodeLogs = Record<NodeName, string[]>
