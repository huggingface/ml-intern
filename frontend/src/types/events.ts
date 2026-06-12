/**
 * Event types from the agent backend
 */

export type EventType =
  | 'ready'
  | 'processing'
  | 'assistant_message'
  | 'assistant_chunk'
  | 'assistant_stream_end'
  | 'tool_call'
  | 'tool_output'
  | 'tool_log'
  | 'approval_required'
  | 'tool_state_change'
  | 'llm_call'
  | 'hf_job_complete'
  | 'sandbox_create'
  | 'sandbox_destroy'
  | 'session_update'
  | 'turn_complete'
  | 'compacted'
  | 'error'
  | 'shutdown'
  | 'interrupted'
  | 'undo_complete'
  | 'plan_update';

export interface AgentEvent {
  event_type: EventType;
  data?: Record<string, unknown>;
  seq?: number;
}
