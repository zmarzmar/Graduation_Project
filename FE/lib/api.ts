// BE API 호출 전용 파일 — 컴포넌트에서 직접 fetch 금지
import type { ArxivPaper } from './types/agent-run'

// NEXT_PUBLIC_API_URL은 /api/v1까지 포함한 전체 경로로 설정 (예: http://localhost:8000/api/v1)
const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1').replace(/\/$/, '')

/** 인증 필요 API 전용 fetch — Authorization 헤더 자동 주입 */
function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  })
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export interface AuthTokenResponse {
  access_token: string
  token_type: string
}

export interface AuthUser {
  id: number
  email: string
  username: string
  full_name: string | null
  affiliation: string | null
  preferred_framework: string | null
}

/** 회원가입 — JWT 즉시 반환 */
export async function register(email: string, password: string, username: string, full_name?: string): Promise<AuthTokenResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, username, full_name }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail ?? '회원가입 실패')
  }
  return res.json()
}

/** 로그인 — JWT 반환 */
export async function login(email: string, password: string): Promise<AuthTokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail ?? '로그인 실패')
  }
  return res.json()
}

/** 현재 유저 정보 조회 (토큰 검증용) */
export async function getMe(): Promise<AuthUser> {
  const res = await authFetch(`${API_BASE}/auth/me`)
  if (!res.ok) throw new Error('인증 만료')
  return res.json()
}

export interface PaperSearchResponse {
  query: string
  total: number
  papers: ArxivPaper[]
}

/** arXiv 논문 검색 */
export async function searchPapers(
  query: string,
  maxResults = 3,
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
  return authFetch(`${API_BASE}/agent/search`, {
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
  return authFetch(`${API_BASE}/agent/pdf`, { method: 'POST', body: form, signal })
}

/** 트렌드 브리핑 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runTrendAgent(topic: string, signal?: AbortSignal): Promise<Response> {
  return authFetch(`${API_BASE}/agent/trend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
    signal,
  })
}

/** 선택한 논문 분석 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runAnalyzeAgent(paper: ArxivPaper, query: string, signal?: AbortSignal): Promise<Response> {
  return authFetch(`${API_BASE}/agent/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper, query }),
    signal,
  })
}

export interface SearchHistoryPaper {
  title: string
  authors: string[]
  arxiv_id: string
  url: string
}

export interface SearchHistoryItem {
  id: number
  query: string
  mode: string
  result_count: number
  papers: SearchHistoryPaper[]
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
  const res = await authFetch(`${API_BASE}/mypage/search-history`)
  if (!res.ok) throw new Error('검색 기록 조회 실패')
  return res.json()
}

/** 분석 히스토리 조회 */
export async function getAnalysisHistory(): Promise<AnalysisHistoryItem[]> {
  const res = await authFetch(`${API_BASE}/mypage/analysis-history`)
  if (!res.ok) throw new Error('분석 히스토리 조회 실패')
  return res.json()
}

export interface AnalysisDetail extends AnalysisHistoryItem {
  paper_summary: string
  paper_review: { strengths?: string[]; limitations?: string[]; significance?: string }
  key_formulas: { name: string; latex: string; description: string }[]
  generated_code: string
  review_feedback: string
  iteration_count: number
}

/** 분석 결과 상세 조회 */
export async function getAnalysisDetail(id: number): Promise<AnalysisDetail> {
  const res = await authFetch(`${API_BASE}/mypage/analysis-history/${id}`)
  if (!res.ok) throw new Error('분석 상세 조회 실패')
  return res.json()
}

/** 검색 기록 개별 삭제 */
export async function deleteSearchHistory(id: number): Promise<void> {
  const res = await authFetch(`${API_BASE}/mypage/search-history/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('검색 기록 삭제 실패')
}

/** 검색 기록 전체 삭제 */
export async function deleteAllSearchHistory(): Promise<void> {
  const res = await authFetch(`${API_BASE}/mypage/search-history`, { method: 'DELETE' })
  if (!res.ok) throw new Error('검색 기록 전체 삭제 실패')
}

/** 분석 히스토리 개별 삭제 */
export async function deleteAnalysisHistory(id: number): Promise<void> {
  const res = await authFetch(`${API_BASE}/mypage/analysis-history/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('분석 기록 삭제 실패')
}

/** 분석 히스토리 전체 삭제 */
export async function deleteAllAnalysisHistory(): Promise<void> {
  const res = await authFetch(`${API_BASE}/mypage/analysis-history`, { method: 'DELETE' })
  if (!res.ok) throw new Error('분석 기록 전체 삭제 실패')
}

export interface UserInfo {
  id: number
  username: string
  full_name: string | null
  email: string
  affiliation: string | null
  preferred_framework: string | null
}

/** 내 정보 조회 */
export async function getMyInfo(): Promise<UserInfo> {
  const res = await authFetch(`${API_BASE}/mypage/me`)
  if (!res.ok) throw new Error('내 정보 조회 실패')
  return res.json()
}

/** 서버 헬스 체크 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  const base = API_BASE.replace('/api/v1', '')
  const res = await fetch(`${base}/health`)
  if (!res.ok) throw new Error('서버 응답 없음')
  return res.json()
}
