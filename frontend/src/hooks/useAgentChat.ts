/**
 * Central hook wiring the Vercel AI SDK's useChat with our SSE-based
 * ChatTransport.
 *
 * In the per-session architecture, each session mounts its own instance
 * of this hook. The `isActive` flag controls whether side-channel
 * callbacks update the global UI stores (agentStore / layoutStore) or
 * only per-session metadata (sessionStore.needsAttention).
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useChat } from '@ai-sdk/react';
import { type UIMessage, lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';
import { SSEChatTransport, type SideChannelCallbacks } from '@/lib/sse-chat-transport';
import { loadMessages, saveMessages } from '@/lib/chat-message-store';
import { llmMessagesToUIMessages } from '@/lib/convert-llm-messages';
import { apiFetch } from '@/utils/api';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import { logger } from '@/utils/logger';

interface UseAgentChatOptions {
  sessionId: string;
  isActive: boolean;
  onReady?: () => void;
  onError?: (error: string) => void;
  onSessionDead?: (sessionId: string) => void;
}

export function useAgentChat({ sessionId, isActive, onReady, onError, onSessionDead }: UseAgentChatOptions) {
  const callbacksRef = useRef({ onReady, onError, onSessionDead });
  callbacksRef.current = { onReady, onError, onSessionDead };

  const isActiveRef = useRef(isActive);
  isActiveRef.current = isActive;

  const {
    setProcessing,
    setConnected,
    setActivityStatus,
    setError,
    setPanel,
    setPanelOutput,
  } = useAgentStore();

  const { setRightPanelOpen, setLeftSidebarOpen } = useLayoutStore();
  const { setSessionActive, setNeedsAttention } = useSessionStore();

  // -- Build side-channel callbacks (stable ref) --------------------------
  const sideChannel = useMemo<SideChannelCallbacks>(
    () => ({
      onReady: () => {
        if (isActiveRef.current) {
          setConnected(true);
          setProcessing(false);
        }
        setSessionActive(sessionId, true);
        callbacksRef.current.onReady?.();
      },
      onShutdown: () => {
        if (isActiveRef.current) {
          setConnected(false);
          setProcessing(false);
        }
      },
      onError: (error: string) => {
        if (isActiveRef.current) {
          setError(error);
          setProcessing(false);
        }
        callbacksRef.current.onError?.(error);
      },
      onProcessing: () => {
        if (isActiveRef.current) {
          setProcessing(true);
          setActivityStatus({ type: 'thinking' });
        }
      },
      onProcessingDone: () => {
        if (isActiveRef.current) {
          setProcessing(false);
        }
      },
      onUndoComplete: () => {
        // Undo is handled client-side in undoLastTurn(). With SSE, undo_complete
        // events are discarded (no subscriber listening between turns).
        if (isActiveRef.current) setProcessing(false);
      },
      onCompacted: (oldTokens: number, newTokens: number) => {
        logger.log(`Context compacted: ${oldTokens} -> ${newTokens} tokens`);
      },
      onPlanUpdate: (plan) => {
        if (!isActiveRef.current) return;
        useAgentStore.getState().setPlan(plan as Array<{ id: string; content: string; status: 'pending' | 'in_progress' | 'completed' }>);
        if (!useLayoutStore.getState().isRightPanelOpen) {
          setRightPanelOpen(true);
        }
      },
      onToolLog: (tool: string, log: string) => {
        if (!isActiveRef.current) return;
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
        if (isActiveRef.current) setConnected(connected);
      },
      onSessionDead: (deadSessionId: string) => {
        logger.warn(`Session ${deadSessionId} dead, removing`);
        callbacksRef.current.onSessionDead?.(deadSessionId);
      },
      onApprovalRequired: (tools) => {
        if (!tools.length) return;
        setNeedsAttention(sessionId, true);
        if (!isActiveRef.current) return;

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
        if (!isActiveRef.current) return;
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
        if (!isActiveRef.current) return;
        if (toolName === 'hf_jobs' && output) {
          setPanelOutput({ content: output, language: 'markdown' });
          if (!success) useAgentStore.getState().setPanelView('output');
        }
      },
      onStreaming: () => {
        if (isActiveRef.current) setActivityStatus({ type: 'streaming' });
      },
      onToolRunning: (toolName: string, description?: string) => {
        if (isActiveRef.current) setActivityStatus({ type: 'tool', toolName, description });
      },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId],
  );

  // -- Create transport (one per session, stable for lifetime) ------------
  const transportRef = useRef<SSEChatTransport | null>(null);
  if (!transportRef.current) {
    transportRef.current = new SSEChatTransport(sessionId, sideChannel);
  }

  // Keep side-channel callbacks in sync
  useEffect(() => {
    transportRef.current?.updateSideChannel(sideChannel);
  }, [sideChannel]);

  // Destroy transport on unmount
  useEffect(() => {
    return () => {
      transportRef.current?.destroy();
      transportRef.current = null;
    };
  }, []);

  // -- Restore persisted messages for this session ------------------------
  const initialMessages = useMemo(
    () => loadMessages(sessionId),
    [sessionId],
  );

  // -- Ref for chat actions (used by sideChannel callbacks) ---------------
  const chatActionsRef = useRef<{
    setMessages: ((msgs: UIMessage[]) => void) | null;
    messages: UIMessage[];
  }>({ setMessages: null, messages: [] });

  // -- useChat from Vercel AI SDK -----------------------------------------
  const chat = useChat({
    id: sessionId,
    messages: initialMessages,
    transport: transportRef.current!,
    experimental_throttle: 80,
    // After all approval responses are set, auto-send to continue the agent loop.
    // Without this, addToolApprovalResponse only updates the UI — it won't trigger
    // sendMessages on the transport.
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    onError: (error) => {
      logger.error('useChat error:', error);
      if (isActiveRef.current) {
        setError(error.message);
        setProcessing(false);
      }
    },
  });

  // Keep chatActionsRef in sync every render
  chatActionsRef.current.setMessages = chat.setMessages;
  chatActionsRef.current.messages = chat.messages;

  // -- Hydrate from backend on mount (page refresh recovery) --------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [msgsRes, infoRes] = await Promise.all([
          apiFetch(`/api/session/${sessionId}/messages`),
          apiFetch(`/api/session/${sessionId}`),
        ]);
        if (cancelled) return;

        let pendingIds: Set<string> | undefined;
        if (infoRes.ok) {
          const info = await infoRes.json();
          if (info.pending_approval && Array.isArray(info.pending_approval)) {
            pendingIds = new Set(
              info.pending_approval.map((t: { tool_call_id: string }) => t.tool_call_id)
            );
            if (pendingIds.size > 0) {
              setNeedsAttention(sessionId, true);
            }
          }
        }

        if (msgsRes.ok) {
          const data = await msgsRes.json();
          if (cancelled || !Array.isArray(data) || data.length === 0) return;
          const uiMsgs = llmMessagesToUIMessages(data, pendingIds);
          if (uiMsgs.length > 0) {
            chat.setMessages(uiMsgs);
            saveMessages(sessionId, uiMsgs);
          }
        }
      } catch {
        /* backend unreachable -- localStorage fallback is fine */
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Persist messages ---------------------------------------------------
  const prevLenRef = useRef(initialMessages.length);
  useEffect(() => {
    if (chat.messages.length === 0) return;
    if (chat.messages.length !== prevLenRef.current) {
      prevLenRef.current = chat.messages.length;
      saveMessages(sessionId, chat.messages);
    }
  }, [sessionId, chat.messages]);

  // -- Undo last turn (REST call + client-side message removal) -----------
  // With SSE there's no persistent connection to receive the undo_complete
  // event, so we handle message removal on the frontend after a successful
  // REST call to the backend.
  const undoLastTurn = useCallback(async () => {
    try {
      const res = await apiFetch(`/api/undo/${sessionId}`, { method: 'POST' });
      if (!res.ok) {
        logger.error('Undo API returned', res.status);
        return;
      }
      // Remove the last user turn + assistant response from the UI
      const msgs = chatActionsRef.current.messages;
      const setMsgs = chatActionsRef.current.setMessages;
      if (setMsgs && msgs.length > 0) {
        let lastUserIdx = -1;
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === 'user') { lastUserIdx = i; break; }
        }
        const updated = lastUserIdx > 0 ? msgs.slice(0, lastUserIdx) : [];
        setMsgs(updated);
        saveMessages(sessionId, updated);
      }
      if (isActiveRef.current) setProcessing(false);
    } catch (e) {
      logger.error('Undo failed:', e);
    }
  }, [sessionId, setProcessing]);

  // -- Approve tools ------------------------------------------------------
  const approveTools = useCallback(
    async (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null }>) => {
      // Store edited scripts so the transport can read them when sendMessages is called
      for (const a of approvals) {
        if (a.edited_script) {
          useAgentStore.getState().setEditedScript(a.tool_call_id, a.edited_script);
        }
      }

      // Update SDK tool state — this triggers sendMessages() via the transport
      for (const a of approvals) {
        chat.addToolApprovalResponse({
          id: `approval-${a.tool_call_id}`,
          approved: a.approved,
          reason: a.approved ? undefined : (a.feedback || 'Rejected by user'),
        });
      }

      setNeedsAttention(sessionId, false);
      const hasApproved = approvals.some(a => a.approved);
      if (hasApproved && isActiveRef.current) setProcessing(true);
      return true;
    },
    [sessionId, chat, setProcessing, setNeedsAttention],
  );

  // -- Stop (abort SSE stream + interrupt backend agent loop) ---------------
  const stop = useCallback(() => {
    chat.stop();
    apiFetch(`/api/interrupt/${sessionId}`, { method: 'POST' }).catch(() => {});
  }, [sessionId, chat]);

  return {
    messages: chat.messages,
    sendMessage: chat.sendMessage,
    stop,
    status: chat.status,
    undoLastTurn,
    approveTools,
  };
}
