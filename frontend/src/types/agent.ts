/**
 * Agent-related types
 */

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp: string;
  toolName?: string;
  toolCallId?: string;
  trace?: TraceLog[];
  approval?: {
    status: 'pending' | 'approved' | 'rejected';
    batch: ApprovalBatch;
    decisions?: ToolApproval[];
  };
  toolOutput?: string;
}

export interface ToolCall {
  id: string;
  tool: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  output?: string;
}

export interface ToolApproval {
  toolCallId: string;
  approved: boolean;
  feedback?: string;
}

export interface ApprovalBatch {
  tools: Array<{
    tool: string;
    arguments: Record<string, unknown>;
    tool_call_id: string;
  }>;
  count: number;
}

export interface TraceLog {
  id: string;
  type: 'call' | 'output';
  text: string;
  tool: string;
  timestamp: string;
  completed?: boolean;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
}
