/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
}

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
}

export interface ToolApproval {
  tool_call_id: string;
  approved: boolean;
  feedback?: string | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
}
