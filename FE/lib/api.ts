// BE API 호출 전용 파일 — 컴포넌트에서 직접 fetch 금지
import type { ArxivPaper } from './types/agent-run'

// NEXT_PUBLIC_API_URL은 백엔드 origin 기준으로 설정한다. 기존 /api/v1 포함 값도 호환한다.
const API_VERSION_PATH = '/api/v1'
const API_ORIGIN = normalizeApiOrigin(process.env.NEXT_PUBLIC_API_URL)
const API_BASE = `${API_ORIGIN}${API_VERSION_PATH}`
const GUEST_FREE_USAGE_KEY = 'guest_free_usage_used'

export const GUEST_LOGIN_MESSAGE = '무료 체험을 사용했어요. 계속하려면 로그인해 주세요.'

export class GuestUsageLimitError extends Error {
  constructor() {
    super(GUEST_LOGIN_MESSAGE)
    this.name = 'GuestUsageLimitError'
  }
}

function normalizeApiOrigin(rawUrl?: string): string {
  const url = (rawUrl?.trim() || 'http://localhost:8000').replace(/\/+$/, '')
  return url.endsWith(API_VERSION_PATH) ? url.slice(0, -API_VERSION_PATH.length) : url
}

export function isGuestUsageLimitError(error: unknown): error is GuestUsageLimitError {
  return error instanceof GuestUsageLimitError
}

function hasAuthToken(): boolean {
  return typeof window !== 'undefined' && Boolean(localStorage.getItem('auth_token'))
}

function assertGuestCanUseAgent(): void {
  if (typeof window === 'undefined' || hasAuthToken()) return

  if (sessionStorage.getItem(GUEST_FREE_USAGE_KEY) === 'true') {
    throw new GuestUsageLimitError()
  }

  sessionStorage.setItem(GUEST_FREE_USAGE_KEY, 'true')
}

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
  preferred_framework: string[] | null
  is_admin: boolean
}

/** 구버전 BE 호환: preferred_framework가 string으로 올 경우 배열로 변환 */
function normalizePreferredFramework<T extends { preferred_framework?: string[] | string | null }>(
  data: T,
): Omit<T, 'preferred_framework'> & { preferred_framework: string[] | null } {
  return {
    ...data,
    preferred_framework:
      typeof data.preferred_framework === 'string'
        ? data.preferred_framework ? [data.preferred_framework] : null
        : data.preferred_framework ?? null,
  }
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
export async function login(identifier: string, password: string): Promise<AuthTokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ identifier, password }),
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
  return normalizePreferredFramework(await res.json()) as AuthUser
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
  assertGuestCanUseAgent()
  return authFetch(`${API_BASE}/agent/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  })
}

/** PDF 업로드 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runPdfAgent(file: File, signal?: AbortSignal): Promise<Response> {
  assertGuestCanUseAgent()
  const form = new FormData()
  form.append('file', file)
  return authFetch(`${API_BASE}/agent/pdf`, { method: 'POST', body: form, signal })
}

/** 트렌드 브리핑 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runTrendAgent(topic: string, signal?: AbortSignal): Promise<Response> {
  assertGuestCanUseAgent()
  return authFetch(`${API_BASE}/agent/trend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
    signal,
  })
}

export interface AnalyzeAgentOptions {
  allowAbstractFallback?: boolean
  consumeGuestUsage?: boolean
}

/** 선택한 논문 분석 에이전트 실행 — SSE 스트리밍 응답 반환 */
export function runAnalyzeAgent(
  paper: ArxivPaper,
  query: string,
  signal?: AbortSignal,
  options: AnalyzeAgentOptions = {},
): Promise<Response> {
  if (options.consumeGuestUsage !== false) {
    assertGuestCanUseAgent()
  }
  return authFetch(`${API_BASE}/agent/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      paper,
      query,
      allow_abstract_fallback: options.allowAbstractFallback ?? false,
    }),
    signal,
  })
}

/** 오늘 날짜 기준 AI 트렌드 키워드 5개 반환 */
export async function fetchDailyKeywords(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/papers/daily-keywords`)
  if (!res.ok) throw new Error('daily-keywords fetch failed')
  const data: { keywords: string[] } = await res.json()
  return data.keywords
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
  preferred_framework: string[] | null
  bio: string | null
  github_url: string | null
  research_interests: string[] | null
}

export interface UpdateUserInfoPayload {
  full_name?: string | null
  affiliation?: string | null
  preferred_framework?: string[] | null
  bio?: string | null
  github_url?: string | null
  research_interests?: string[] | null
}

/** 내 정보 조회 */
export async function getMyInfo(): Promise<UserInfo> {
  const res = await authFetch(`${API_BASE}/mypage/me`)
  if (!res.ok) throw new Error('내 정보 조회 실패')
  return normalizePreferredFramework(await res.json()) as UserInfo
}

/** 내 정보 수정 */
export async function updateMyInfo(payload: UpdateUserInfoPayload): Promise<UserInfo> {
  const res = await authFetch(`${API_BASE}/mypage/me`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('내 정보 수정 실패')
  return normalizePreferredFramework(await res.json()) as UserInfo
}

export interface AdminUser {
  id: number
  email: string
  username: string
  full_name: string | null
  affiliation: string | null
  preferred_framework: string[] | null
  is_active: boolean
  is_admin: boolean
  created_at: string
  last_login_at: string | null
}

export interface AdminPaper {
  id: number
  arxiv_id: string
  title: string
  authors: string[]
  url: string
  pdf_url: string
  published_at: string | null
  categories: string[]
  citation_count: number | null
  created_at: string
}

export interface AdminSystemSummary {
  status: string
  version: string
  database: string
  counts: {
    users: number
    active_users: number
    admin_users: number
    papers: number
    search_history: number
    analysis_results: number
  }
}

/** 관리자 사용자 목록 조회 */
export async function getAdminUsers(): Promise<AdminUser[]> {
  const res = await authFetch(`${API_BASE}/admin/users`)
  if (!res.ok) throw new Error('관리자 사용자 목록 조회 실패')
  const data: Array<AdminUser & { preferred_framework?: string[] | string | null }> = await res.json()
  return data.map((user) => normalizePreferredFramework(user) as AdminUser)
}

/** 관리자 논문 목록 조회 */
export async function getAdminPapers(): Promise<AdminPaper[]> {
  const res = await authFetch(`${API_BASE}/admin/papers`)
  if (!res.ok) throw new Error('관리자 논문 목록 조회 실패')
  return res.json()
}

/** 관리자 시스템 요약 조회 */
export async function getAdminSystemSummary(): Promise<AdminSystemSummary> {
  const res = await authFetch(`${API_BASE}/admin/system`)
  if (!res.ok) throw new Error('관리자 시스템 요약 조회 실패')
  return res.json()
}

/** 서버 헬스 체크 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  const res = await fetch(`${API_ORIGIN}/health`)
  if (!res.ok) throw new Error('서버 응답 없음')
  return res.json()
}
