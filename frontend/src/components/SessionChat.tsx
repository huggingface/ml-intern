/**
 * Per-session chat component.
 *
 * Each session renders its own SessionChat. The hook (useAgentChat) always
 * runs — processing events — but only the active session renders visible
 * UI (MessageList + ChatInput).
 */
import { useCallback, useEffect, useRef } from 'react';
import { useAgentChat } from '@/hooks/useAgentChat';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import MessageList from '@/components/Chat/MessageList';
import ChatInput from '@/components/Chat/ChatInput';
import { apiFetch } from '@/utils/api';
import { logger } from '@/utils/logger';

interface SessionChatProps {
  sessionId: string;
  isActive: boolean;
  onSessionDead: (sessionId: string) => void;
}

export default function SessionChat({ sessionId, isActive, onSessionDead }: SessionChatProps) {
  const { isConnected, isProcessing, setProcessing, activityStatus } = useAgentStore();
  const { updateSessionTitle } = useSessionStore();

  const { messages, sendMessage, stop, undoLastTurn, approveTools } = useAgentChat({
    sessionId,
    isActive,
    onReady: () => logger.log(`Session ${sessionId} ready`),
    onError: (error) => logger.error(`Session ${sessionId} error:`, error),
    onSessionDead,
  });

  // When this session becomes active, sync ALL global agentStore state
  // so the UI correctly reflects this session's current state.
  const prevActiveRef = useRef(isActive);
  useEffect(() => {
    if (isActive && !prevActiveRef.current) {
      const store = useAgentStore.getState();

      // SSE transport has no persistent connection — always connected
      store.setConnected(true);

      // Check if this session has pending approvals in its messages
      const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
      const hasPendingApproval = lastAssistant?.parts.some(
        (p) => p.type === 'dynamic-tool' && p.state === 'approval-requested'
      ) ?? false;
      const hasApprovedRunning = lastAssistant?.parts.some(
        (p) => p.type === 'dynamic-tool' && p.state === 'approval-responded'
      ) ?? false;

      if (hasPendingApproval) {
        store.setActivityStatus({ type: 'waiting-approval' });
        store.setProcessing(false);

        // Restore panel for the first pending tool
        const pendingTool = lastAssistant!.parts.find(
          (p) => p.type === 'dynamic-tool' && p.state === 'approval-requested'
        );
        if (pendingTool && pendingTool.type === 'dynamic-tool') {
          const args = pendingTool.input as Record<string, string | undefined>;
          if (pendingTool.toolName === 'hf_jobs' && args?.script) {
            store.setPanel(
              { title: 'Script', script: { content: args.script, language: 'python' }, parameters: pendingTool.input as Record<string, unknown> },
              'script',
              true,
            );
          } else if (pendingTool.toolName === 'hf_repo_files' && args?.content) {
            const filename = args.path || 'file';
            store.setPanel({
              title: filename.split('/').pop() || 'Content',
              script: { content: args.content, language: filename.endsWith('.py') ? 'python' : 'text' },
              parameters: pendingTool.input as Record<string, unknown>,
            });
          } else {
            store.setPanel({
              title: pendingTool.toolName,
              output: { content: JSON.stringify(pendingTool.input, null, 2), language: 'json' },
            }, 'output');
          }
          useLayoutStore.getState().setRightPanelOpen(true);
        }
      } else if (hasApprovedRunning) {
        // Tools were approved but still executing — show processing state
        store.setActivityStatus({ type: 'tool', toolName: 'running' });
        store.setProcessing(true);
      } else {
        // Check if any tools are still running (non-approval tools like bash, read, etc.)
        const runningTool = lastAssistant?.parts.find(
          (p) => p.type === 'dynamic-tool' && (p.state === 'input-available' || p.state === 'input-streaming')
        );
        if (runningTool && runningTool.type === 'dynamic-tool') {
          const desc = (runningTool.input as Record<string, unknown>)?.description as string | undefined;
          store.setActivityStatus({ type: 'tool', toolName: runningTool.toolName, description: desc });
          store.setProcessing(true);
        } else {
          store.setActivityStatus({ type: 'idle' });
          store.setProcessing(false);
        }
      }
    }
    prevActiveRef.current = isActive;
  }, [isActive, messages]);

  const handleSendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isProcessing) return;

      setProcessing(true);
      sendMessage({ text: text.trim(), metadata: { createdAt: new Date().toISOString() } });

      // Auto-title the session from the first user message
      const isFirstMessage = messages.filter((m) => m.role === 'user').length <= 1;
      if (isFirstMessage) {
        apiFetch('/api/title', {
          method: 'POST',
          body: JSON.stringify({ session_id: sessionId, text: text.trim() }),
        })
          .then((res) => res.json())
          .then((data) => {
            if (data.title) updateSessionTitle(sessionId, data.title);
          })
          .catch(() => {
            const raw = text.trim();
            updateSessionTitle(sessionId, raw.length > 40 ? raw.slice(0, 40) + '\u2026' : raw);
          });
      }
    },
    [sessionId, sendMessage, messages, updateSessionTitle, isProcessing, setProcessing],
  );

  // Don't render UI for background sessions — hooks still run
  if (!isActive) return null;

  return (
    <>
      <MessageList
        messages={messages}
        isProcessing={isProcessing}
        approveTools={approveTools}
        onUndoLastTurn={undoLastTurn}
      />
      <ChatInput
        onSend={handleSendMessage}
        onStop={stop}
        isProcessing={isProcessing}
        disabled={!isConnected || activityStatus.type === 'waiting-approval'}
        placeholder={activityStatus.type === 'waiting-approval' ? 'Approve or reject pending tools first...' : undefined}
      />
    </>
  );
}
