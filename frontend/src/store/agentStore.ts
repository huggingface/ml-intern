/**
 * Agent store — manages UI state that is NOT handled by the Vercel AI SDK.
 *
 * Message state (messages, streaming, tool calls) is now managed by useChat().
 * This store only handles:
 *  - Connection / processing flags
 *  - Panel state (right panel — single-artifact pattern)
 *  - Plan state
 *  - User info / error banners
 *  - Edited scripts (for hf_jobs code editing)
 */
import { create } from 'zustand';
import type { User } from '@/types/agent';

export interface PlanItem {
  id: string;
  content: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface PanelSection {
  content: string;
  language: string;
}

export interface PanelData {
  title: string;
  script?: PanelSection;
  output?: PanelSection;
  parameters?: Record<string, unknown>;
}

export type PanelView = 'script' | 'output';

export interface LLMHealthError {
  error: string;
  errorType: 'auth' | 'credits' | 'rate_limit' | 'network' | 'unknown';
  model: string;
}

export type ActivityStatus =
  | { type: 'idle' }
  | { type: 'thinking' }
  | { type: 'tool'; toolName: string }
  | { type: 'waiting-approval' }
  | { type: 'streaming' };

interface AgentStore {
  // Global UI flags
  isProcessing: boolean;
  isConnected: boolean;
  activityStatus: ActivityStatus;
  user: User | null;
  error: string | null;
  llmHealthError: LLMHealthError | null;

  // Right panel (single-artifact pattern)
  panelData: PanelData | null;
  panelView: PanelView;
  panelEditable: boolean;

  // Plan
  plan: PlanItem[];

  // Edited scripts (tool_call_id -> edited content)
  editedScripts: Record<string, string>;

  // Job URLs (tool_call_id -> job URL) for HF jobs
  jobUrls: Record<string, string>;

  // Actions
  setProcessing: (isProcessing: boolean) => void;
  setConnected: (isConnected: boolean) => void;
  setActivityStatus: (status: ActivityStatus) => void;
  setUser: (user: User | null) => void;
  setError: (error: string | null) => void;
  setLlmHealthError: (error: LLMHealthError | null) => void;

  setPanel: (data: PanelData, view?: PanelView, editable?: boolean) => void;
  setPanelView: (view: PanelView) => void;
  setPanelOutput: (output: PanelSection) => void;
  updatePanelScript: (content: string) => void;
  lockPanel: () => void;
  clearPanel: () => void;

  setPlan: (plan: PlanItem[]) => void;

  setEditedScript: (toolCallId: string, content: string) => void;
  getEditedScript: (toolCallId: string) => string | undefined;
  clearEditedScripts: () => void;

  setJobUrl: (toolCallId: string, jobUrl: string) => void;
  getJobUrl: (toolCallId: string) => string | undefined;
}

export const useAgentStore = create<AgentStore>()((set, get) => ({
  isProcessing: false,
  isConnected: false,
  activityStatus: { type: 'idle' },
  user: null,
  error: null,
  llmHealthError: null,

  panelData: null,
  panelView: 'script',
  panelEditable: false,

  plan: [],

  editedScripts: {},
  jobUrls: {},

  // ── Global flags ──────────────────────────────────────────────────

  setProcessing: (isProcessing) => {
    const current = get().activityStatus;
    const preserveStatus = current.type === 'waiting-approval';
    set({ isProcessing, ...(!isProcessing && !preserveStatus ? { activityStatus: { type: 'idle' } } : {}) });
  },
  setConnected: (isConnected) => set({ isConnected }),
  setActivityStatus: (status) => set({ activityStatus: status }),
  setUser: (user) => set({ user }),
  setError: (error) => set({ error }),
  setLlmHealthError: (error) => set({ llmHealthError: error }),

  // ── Panel (single-artifact) ───────────────────────────────────────

  setPanel: (data, view, editable) => set({
    panelData: data,
    panelView: view ?? (data.script ? 'script' : 'output'),
    panelEditable: editable ?? false,
  }),

  setPanelView: (view) => set({ panelView: view }),

  setPanelOutput: (output) => set((state) => ({
    panelData: state.panelData ? { ...state.panelData, output } : null,
  })),

  updatePanelScript: (content) => set((state) => ({
    panelData: state.panelData?.script
      ? { ...state.panelData, script: { ...state.panelData.script, content } }
      : state.panelData,
  })),

  lockPanel: () => set({ panelEditable: false }),

  clearPanel: () => set({ panelData: null, panelView: 'script', panelEditable: false }),

  // ── Plan ──────────────────────────────────────────────────────────

  setPlan: (plan) => set({ plan }),

  // ── Edited scripts ────────────────────────────────────────────────

  setEditedScript: (toolCallId, content) => {
    set((state) => ({
      editedScripts: { ...state.editedScripts, [toolCallId]: content },
    }));
  },

  getEditedScript: (toolCallId) => get().editedScripts[toolCallId],

  clearEditedScripts: () => set({ editedScripts: {} }),

  // ── Job URLs ────────────────────────────────────────────────────────

  setJobUrl: (toolCallId, jobUrl) => {
    set((state) => ({
      jobUrls: { ...state.jobUrls, [toolCallId]: jobUrl },
    }));
  },

  getJobUrl: (toolCallId) => get().jobUrls[toolCallId],
}));
