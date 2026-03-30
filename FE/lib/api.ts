// BE API 호출 전용 파일 — 컴포넌트에서 직접 fetch 금지
import type { ArxivPaper } from './types/agent-run'

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1').replace(/\/$/, '')

export interface PaperSearchResponse {
  query: string
  total: number
  papers: ArxivPaper[]
}

/** arXiv 논문 검색 */
export async function searchPapers(
  query: string,
  maxResults = 5,
): Promise<PaperSearchResponse> {
  const params = new URLSearchParams({
    search: query,
    max_results: String(maxResults),
  })
  const res = await fetch(`${API_BASE}/papers/search?${params}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`논문 검색 실패 (${res.status}): ${text}`)
  }
  return res.json()
}

/** 키워드 검색 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runSearchAgent(query: string, signal?: AbortSignal): Promise<Response> {
  return fetch(`${API_BASE}/agent/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  })
}

/** PDF 업로드 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runPdfAgent(file: File, signal?: AbortSignal): Promise<Response> {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${API_BASE}/agent/pdf`, { method: 'POST', body: form, signal })
}

/** 트렌드 브리핑 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runTrendAgent(topic: string, signal?: AbortSignal): Promise<Response> {
  return fetch(`${API_BASE}/agent/trend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
    signal,
  })
}

/** 선택한 논문 분석 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runAnalyzeAgent(paper: ArxivPaper, query: string, signal?: AbortSignal): Promise<Response> {
  return fetch(`${API_BASE}/agent/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper, query }),
    signal,
  })
}

export interface SearchHistoryItem {
  id: number
  query: string
  mode: string
  result_count: number
  created_at: string
}

export interface AnalysisHistoryItem {
  id: number
  query: string
  mode: string
  paper_title: string | null
  paper_authors: string[] | null
  review_passed: boolean
  has_code: boolean
  created_at: string
}

/** 최근 검색 기록 조회 */
export async function getSearchHistory(): Promise<SearchHistoryItem[]> {
  const res = await fetch(`${API_BASE}/mypage/search-history`)
  if (!res.ok) throw new Error('검색 기록 조회 실패')
  return res.json()
}

/** 분석 히스토리 조회 */
export async function getAnalysisHistory(): Promise<AnalysisHistoryItem[]> {
  const res = await fetch(`${API_BASE}/mypage/analysis-history`)
  if (!res.ok) throw new Error('분석 히스토리 조회 실패')
  return res.json()
}

/** 서버 헬스 체크 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  const base = API_BASE.replace('/api/v1', '')
  const res = await fetch(`${base}/health`)
  if (!res.ok) throw new Error('서버 응답 없음')
  return res.json()
}
