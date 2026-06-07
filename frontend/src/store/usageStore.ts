import { create } from 'zustand';
import { apiFetch } from '@/utils/api';

export interface UsageBucket {
  session_id?: string | null;
  window_start?: string | null;
  window_end?: string | null;
  timezone?: string | null;
  total_usd: number;
  inference_usd: number;
  hf_jobs_estimated_usd: number;
  llm_calls: number;
  hf_jobs_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  total_tokens: number;
  hf_jobs_billable_seconds_estimate: number;
}

export interface HfAccountUsageBucket {
  window_start?: string | null;
  window_end?: string | null;
  timezone?: string | null;
  total_usd: number;
  inference_providers_usd: number;
  hf_jobs_usd: number;
  inference_provider_requests: number;
  hf_jobs_minutes: number;
}

export interface HfInferenceProvidersCredits {
  included_usd: number;
  used_usd: number;
  remaining_included_usd: number;
  limit_usd: number;
  remaining_limit_usd: number;
  num_requests: number;
  period_start?: string | null;
  period_end?: string | null;
}

export interface HfAccountUsage {
  source: 'hf_billing_usage_v2';
  available: boolean;
  error?: string | null;
  current_session: HfAccountUsageBucket | null;
  today: HfAccountUsageBucket | null;
  month: HfAccountUsageBucket | null;
  inference_providers_credits: HfInferenceProvidersCredits | null;
}

export interface UsageResponse {
  source: 'app_telemetry';
  currency: 'USD';
  generated_at: string;
  timezone: string;
  session: UsageBucket | null;
  today: UsageBucket;
  month: UsageBucket;
  hf_account?: HfAccountUsage | null;
  links: Record<string, string>;
}

type UsageEventType = 'llm_call' | 'hf_job_complete';

interface UsageStore {
  usage: UsageResponse | null;
  isLoading: boolean;
  error: string | null;
  fetchUsage: (sessionId?: string | null) => Promise<void>;
  applyUsageEvent: (
    sessionId: string,
    eventType: UsageEventType,
    data: Record<string, unknown>,
  ) => void;
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function intValue(value: unknown): number {
  return Math.trunc(numberValue(value));
}

function roundUsd(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function usageUrl(sessionId?: string | null): string {
  const params = new URLSearchParams();
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  params.set('tz', timezone);
  params.set('include_rollups', 'false');
  if (sessionId) params.set('session_id', sessionId);
  return `/api/usage?${params.toString()}`;
}

function applyEventToBucket(
  bucket: UsageBucket | null,
  eventType: UsageEventType,
  data: Record<string, unknown>,
): UsageBucket | null {
  if (!bucket) return null;
  const next = { ...bucket };

  if (eventType === 'llm_call') {
    const prompt = intValue(data.prompt_tokens);
    const completion = intValue(data.completion_tokens);
    const cacheRead = intValue(data.cache_read_tokens);
    const cacheCreation = intValue(data.cache_creation_tokens);
    const total =
      intValue(data.total_tokens) || prompt + completion + cacheRead + cacheCreation;
    next.llm_calls += 1;
    next.inference_usd = roundUsd(next.inference_usd + numberValue(data.cost_usd));
    next.prompt_tokens += prompt;
    next.completion_tokens += completion;
    next.cache_read_tokens += cacheRead;
    next.cache_creation_tokens += cacheCreation;
    next.total_tokens += total;
  } else if (eventType === 'hf_job_complete') {
    next.hf_jobs_count += 1;
    next.hf_jobs_estimated_usd = roundUsd(
      next.hf_jobs_estimated_usd + numberValue(data.estimated_cost_usd),
    );
    next.hf_jobs_billable_seconds_estimate +=
      intValue(data.billable_seconds_estimate) || intValue(data.wall_time_s);
  }

  next.total_usd = roundUsd(next.inference_usd + next.hf_jobs_estimated_usd);
  return next;
}

export const useUsageStore = create<UsageStore>()((set, get) => ({
  usage: null,
  isLoading: false,
  error: null,

  fetchUsage: async (sessionId?: string | null) => {
    const current = get().usage;
    set({
      usage:
        sessionId && current?.session?.session_id !== sessionId ? null : current,
      isLoading: true,
      error: null,
    });
    try {
      let response = await apiFetch(usageUrl(sessionId));
      if (response.status === 404 && sessionId) {
        response = await apiFetch(usageUrl());
      }
      if (!response.ok) {
        throw new Error(response.statusText || 'Failed to load usage');
      }
      const usage = (await response.json()) as UsageResponse;
      set({ usage, isLoading: false, error: null });
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load usage',
      });
    }
  },

  applyUsageEvent: (sessionId, eventType, data) => {
    const current = get().usage;
    if (!current) return;
    set({
      usage: {
        ...current,
        session:
          current.session?.session_id === sessionId
            ? applyEventToBucket(current.session, eventType, data)
            : current.session,
        today: applyEventToBucket(current.today, eventType, data) ?? current.today,
        month: applyEventToBucket(current.month, eventType, data) ?? current.month,
      },
    });
  },
}));
