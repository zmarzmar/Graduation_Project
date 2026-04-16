// 논문 분석 탭 상태를 전역으로 유지 — 페이지 이탈 후 복귀 시 상태 보존

import { create } from 'zustand'
import type { AgentMode, AgentResult, ArxivPaper, NodeLogs, NodeName, NodeStatuses } from '@/lib/types/agent-run'

export type StreamMode = 'search' | 'pdf' | 'trend'

const INITIAL_NODE_STATUSES: NodeStatuses = {
  planner: 'pending',
  researcher: 'pending',
  trend_analyzer: 'pending',
  analyzer: 'pending',
  coder: 'pending',
  reviewer: 'pending',
}

const INITIAL_NODE_LOGS: NodeLogs = {
  planner: [],
  researcher: [],
  trend_analyzer: [],
  analyzer: [],
  coder: [],
  reviewer: [],
}

export interface StreamSnapshot {
  nodeStatuses: NodeStatuses
  nodeLogs: NodeLogs
  result: AgentResult | null
  isRunning: boolean
  cancelled: boolean
  error: string | null
}

const INITIAL_STREAM: StreamSnapshot = {
  nodeStatuses: { ...INITIAL_NODE_STATUSES },
  nodeLogs: { ...INITIAL_NODE_LOGS },
  result: null,
  isRunning: false,
  cancelled: false,
  error: null,
}

interface AnalysisStore {
  // UI 상태
  mode: AgentMode
  query: string
  topic: string
  searchPipelineMode: AgentMode
  searchedPapers: ArxivPaper[]

  // 모드별 스트림 상태 (isRunning 제외 — 페이지 이탈 시 진행 중인 스트림은 복원 불가)
  streams: Record<StreamMode, StreamSnapshot>

  setMode: (mode: AgentMode) => void
  setQuery: (query: string) => void
  setTopic: (topic: string) => void
  setSearchPipelineMode: (mode: AgentMode) => void
  setSearchedPapers: (papers: ArxivPaper[]) => void
  setStreamState: (mode: StreamMode, patch: Partial<StreamSnapshot>) => void
  resetStream: (mode: StreamMode) => void
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  mode: 'search',
  query: 'LoRA fine-tuning',
  topic: 'large language model',
  searchPipelineMode: 'search',
  searchedPapers: [],
  streams: {
    search: { ...INITIAL_STREAM },
    pdf: { ...INITIAL_STREAM },
    trend: { ...INITIAL_STREAM },
  },

  setMode: (mode) => set({ mode }),
  setQuery: (query) => set({ query }),
  setTopic: (topic) => set({ topic }),
  setSearchPipelineMode: (searchPipelineMode) => set({ searchPipelineMode }),
  setSearchedPapers: (searchedPapers) => set({ searchedPapers }),

  setStreamState: (mode, patch) =>
    set((state) => ({
      streams: { ...state.streams, [mode]: { ...state.streams[mode], ...patch } },
    })),

  resetStream: (mode) =>
    set((state) => ({
      streams: {
        ...state.streams,
        [mode]: { ...INITIAL_STREAM },
      },
    })),
}))
