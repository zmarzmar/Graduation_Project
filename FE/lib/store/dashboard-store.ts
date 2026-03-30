import { create } from 'zustand';
import { ProtocolStatus } from '../types/protocol';

interface DashboardStore {
  // View preferences
  viewMode: 'table' | 'card';
  setViewMode: (mode: 'table' | 'card') => void;

  // Filter states
  statusFilter: ProtocolStatus[];
  setStatusFilter: (statuses: ProtocolStatus[]) => void;

  searchQuery: string;
  setSearchQuery: (query: string) => void;

  dateRange: { from?: Date; to?: Date };
  setDateRange: (range: { from?: Date; to?: Date }) => void;

  // Real-time simulation
  realtimeEnabled: boolean;
  toggleRealtime: () => void;

  // Selected items
  selectedProtocolId?: string;
  setSelectedProtocolId: (id?: string) => void;

  selectedAgentId?: string;
  setSelectedAgentId: (id?: string) => void;

  // Reset filters
  resetFilters: () => void;
}

export const useDashboardStore = create<DashboardStore>((set) => ({
  // View preferences
  viewMode: 'table',
  setViewMode: (mode) => set({ viewMode: mode }),

  // Filter states
  statusFilter: [],
  setStatusFilter: (statuses) => set({ statusFilter: statuses }),

  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),

  dateRange: {},
  setDateRange: (range) => set({ dateRange: range }),

  // Real-time simulation
  realtimeEnabled: true,
  toggleRealtime: () => set((state) => ({ realtimeEnabled: !state.realtimeEnabled })),

  // Selected items
  selectedProtocolId: undefined,
  setSelectedProtocolId: (id) => set({ selectedProtocolId: id }),

  selectedAgentId: undefined,
  setSelectedAgentId: (id) => set({ selectedAgentId: id }),

  // Reset filters
  resetFilters: () =>
    set({
      statusFilter: [],
      searchQuery: '',
      dateRange: {},
    }),
}));
