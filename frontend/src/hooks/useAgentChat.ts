/**
 * Central hook wiring the Vercel AI SDK's useChat with our custom
 * WebSocketChatTransport. Replaces the old useAgentWebSocket + agentStore
 * message management.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useChat } from '@ai-sdk/react';
import type { UIMessage } from 'ai';
import { WebSocketChatTransport, type SideChannelCallbacks } from '@/lib/ws-chat-transport';
import { loadMessages, saveMessages } from '@/lib/chat-message-store';
import { apiFetch } from '@/utils/api';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import { logger } from '@/utils/logger';

interface UseAgentChatOptions {
  sessionId: string | null;
  onReady?: () => void;
  onError?: (error: string) => void;
  onSessionDead?: (sessionId: string) => void;
}

export function useAgentChat({ sessionId, onReady, onError, onSessionDead }: UseAgentChatOptions) {
  const callbacksRef = useRef({ onReady, onError, onSessionDead });
  callbacksRef.current = { onReady, onError, onSessionDead };

  const {
    setProcessing,
    setConnected,
    setActivityStatus,
    setError,
    setPanel,
    setPanelOutput,
  } = useAgentStore();

  const { setRightPanelOpen, setLeftSidebarOpen } = useLayoutStore();
  const { setSessionActive } = useSessionStore();

  // ── Build side-channel callbacks (stable ref) ────────────────────
  const sideChannel = useMemo<SideChannelCallbacks>(
    () => ({
      onReady: () => {
        setConnected(true);
        setProcessing(false);
        if (sessionId) setSessionActive(sessionId, true);
        callbacksRef.current.onReady?.();
      },
      onShutdown: () => {
        setConnected(false);
        setProcessing(false);
      },
      onError: (error: string) => {
        setError(error);
        setProcessing(false);
        callbacksRef.current.onError?.(error);
      },
      onProcessing: () => {
        setProcessing(true);
        setActivityStatus({ type: 'thinking' });
      },
      onProcessingDone: () => {
        setProcessing(false);
      },
      onUndoComplete: () => {
        setProcessing(false);
        // Remove the last turn (user msg + assistant response) from useChat state
        const setMsgs = chatActionsRef.current.setMessages;
        const msgs = chatActionsRef.current.messages;
        if (setMsgs && msgs.length > 0) {
          let lastUserIdx = -1;
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'user') { lastUserIdx = i; break; }
          }
          const updated = lastUserIdx > 0 ? msgs.slice(0, lastUserIdx) : [];
          setMsgs(updated);
          if (sessionId) saveMessages(sessionId, updated);
        }
      },
      onCompacted: (oldTokens: number, newTokens: number) => {
        logger.log(`Context compacted: ${oldTokens} → ${newTokens} tokens`);
      },
      onPlanUpdate: (plan) => {
        useAgentStore.getState().setPlan(plan as Array<{ id: string; content: string; status: 'pending' | 'in_progress' | 'completed' }>);
        if (!useLayoutStore.getState().isRightPanelOpen) {
          setRightPanelOpen(true);
        }
      },
      onToolLog: (tool: string, log: string) => {
        if (tool === 'hf_jobs') {
          const state = useAgentStore.getState();
          const existingOutput = state.panelData?.output?.content || '';
          const newContent = existingOutput
            ? existingOutput + '\n' + log
            : '--- Job execution started ---\n' + log;

          setPanelOutput({ content: newContent, language: 'text' });

          if (!useLayoutStore.getState().isRightPanelOpen) {
            setRightPanelOpen(true);
          }
        }
      },
      onConnectionChange: (connected: boolean) => {
        setConnected(connected);
      },
      onSessionDead: (deadSessionId: string) => {
        logger.warn(`Session ${deadSessionId} dead, removing`);
        callbacksRef.current.onSessionDead?.(deadSessionId);
      },
      onApprovalRequired: (tools) => {
        if (!tools.length) return;
        setActivityStatus({ type: 'waiting-approval' });
        const firstTool = tools[0];
        const args = firstTool.arguments as Record<string, string | undefined>;

        if (firstTool.tool === 'hf_jobs' && args.script) {
          setPanel(
            { title: 'Script', script: { content: args.script, language: 'python' }, parameters: firstTool.arguments as Record<string, unknown> },
            'script',
            true,
          );
        } else if (firstTool.tool === 'hf_repo_files' && args.content) {
          const filename = args.path || 'file';
          setPanel({
            title: filename.split('/').pop() || 'Content',
            script: { content: args.content, language: filename.endsWith('.py') ? 'python' : 'text' },
            parameters: firstTool.arguments as Record<string, unknown>,
          });
        } else {
          setPanel({
            title: firstTool.tool,
            output: { content: JSON.stringify(firstTool.arguments, null, 2), language: 'json' },
          }, 'output');
        }

        setRightPanelOpen(true);
        setLeftSidebarOpen(false);
      },
      onToolCallPanel: (toolName: string, args: Record<string, unknown>) => {
        if (toolName === 'hf_jobs' && args.operation && args.script) {
          setPanel(
            { title: 'Script', script: { content: String(args.script), language: 'python' }, parameters: args },
            'script',
          );
          setRightPanelOpen(true);
          setLeftSidebarOpen(false);
        } else if (toolName === 'hf_repo_files' && args.operation === 'upload' && args.content) {
          setPanel({
            title: `File Upload: ${String(args.path || 'unnamed')}`,
            script: { content: String(args.content), language: String(args.path || '').endsWith('.py') ? 'python' : 'text' },
            parameters: args,
          });
          setRightPanelOpen(true);
          setLeftSidebarOpen(false);
        }
      },
      onToolOutputPanel: (toolName: string, _toolCallId: string, output: string, success: boolean) => {
        if (toolName === 'hf_jobs' && output) {
          setPanelOutput({ content: output, language: 'markdown' });
          if (!success) useAgentStore.getState().setPanelView('output');
        }
      },
      onStreaming: () => {
        setActivityStatus({ type: 'streaming' });
      },
      onToolRunning: (toolName: string) => {
        setActivityStatus({ type: 'tool', toolName });
      },
    }),
    // Zustand setters are stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId],
  );

  // ── Create transport (single stable instance for the lifetime of this hook) ──
  const transportRef = useRef<WebSocketChatTransport | null>(null);
  if (!transportRef.current) {
    transportRef.current = new WebSocketChatTransport({ sideChannel });
  }

  // Keep side-channel callbacks in sync (they capture sessionId)
  useEffect(() => {
    transportRef.current?.updateSideChannel(sideChannel);
  }, [sideChannel]);

  // Connect / disconnect WebSocket when session changes
  useEffect(() => {
    transportRef.current?.connectToSession(sessionId);
    return () => {
      transportRef.current?.connectToSession(null);
    };
  }, [sessionId]);

  // ── Restore persisted messages for this session ─────────────────
  const initialMessages = useMemo(
    () => (sessionId ? loadMessages(sessionId) : []),
    [sessionId],
  );

  // ── Ref for chat actions (used by sideChannel callbacks created before chat) ──
  const chatActionsRef = useRef<{
    setMessages: ((msgs: UIMessage[]) => void) | null;
    messages: UIMessage[];
  }>({ setMessages: null, messages: [] });

  // ── useChat from Vercel AI SDK ───────────────────────────────────
  const chat = useChat({
    id: sessionId || '__no_session__',
    messages: initialMessages,
    transport: transportRef.current!,
    experimental_throttle: 80,
    onFinish: ({ messages, isAbort, isError }) => {
      if (isAbort || isError) return;
      if (sessionId && messages.length > 0) {
        saveMessages(sessionId, messages);
      }
    },
    onError: (error) => {
      logger.error('useChat error:', error);
      setError(error.message);
      setProcessing(false);
    },
  });

  // Keep chatActionsRef in sync every render
  chatActionsRef.current.setMessages = chat.setMessages;
  chatActionsRef.current.messages = chat.messages;

  // ── Persist messages on every user send (onFinish covers assistant turns) ──
  const prevLenRef = useRef(initialMessages.length);
  useEffect(() => {
    if (!sessionId || chat.messages.length === 0) return;
    if (chat.messages.length !== prevLenRef.current) {
      prevLenRef.current = chat.messages.length;
      saveMessages(sessionId, chat.messages);
    }
  }, [sessionId, chat.messages]);

  // ── Undo last turn (calls backend + syncs useChat + localStorage) ──
  const undoLastTurn = useCallback(async () => {
    if (!sessionId) return;
    try {
      const res = await apiFetch(`/api/undo/${sessionId}`, { method: 'POST' });
      if (!res.ok) {
        logger.error('Undo API returned', res.status);
        return;
      }
    } catch (e) {
      logger.error('Undo failed:', e);
    }
    // Backend will also send undo_complete, but we apply optimistically
    // so the UI updates immediately.
  }, [sessionId]);

  // ── Convenience: approve tools via transport ─────────────────────
  const approveTools = useCallback(
    async (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null }>) => {
      if (!sessionId || !transportRef.current) return false;
      const ok = await transportRef.current.approveTools(sessionId, approvals);
      if (ok) {
        const hasApproved = approvals.some(a => a.approved);
        if (hasApproved) setProcessing(true);
      }
      return ok;
    },
    [sessionId, setProcessing],
  );

  return {
    messages: chat.messages,
    sendMessage: chat.sendMessage,
    status: chat.status,
    undoLastTurn,
    approveTools,
    transport: transportRef.current,
  };
}
