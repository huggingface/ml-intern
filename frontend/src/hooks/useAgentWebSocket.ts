import { useCallback, useEffect, useRef } from 'react';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import type { AgentEvent } from '@/types/events';
import type { Message, TraceLog } from '@/types/agent';

const WS_RECONNECT_DELAY = 1000;
const WS_MAX_RECONNECT_DELAY = 30000;

interface UseAgentWebSocketOptions {
  sessionId: string | null;
  onReady?: () => void;
  onError?: (error: string) => void;
}

export function useAgentWebSocket({
  sessionId,
  onReady,
  onError,
}: UseAgentWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectDelayRef = useRef(WS_RECONNECT_DELAY);

  const {
    addMessage,
    updateMessage,
    setProcessing,
    setConnected,
    setPendingApprovals,
    setError,
    addTraceLog,
    updateTraceLog,
    clearTraceLogs,
    setPanelContent,
    setPlan,
    traceLogs,
    setCurrentTurnMessageId,
    updateCurrentTurnTrace,
  } = useAgentStore();

  const { setRightPanelOpen, setLeftSidebarOpen } = useLayoutStore();

  const { setSessionActive } = useSessionStore();

  const handleEvent = useCallback(
    (event: AgentEvent) => {
      if (!sessionId) return;

      switch (event.event_type) {
        case 'ready':
          setConnected(true);
          setProcessing(false);
          setSessionActive(sessionId, true);
          onReady?.();
          break;

        case 'processing':
          setProcessing(true);
          clearTraceLogs();
          setCurrentTurnMessageId(null); // Start a new turn
          break;

        case 'assistant_message': {
          const content = (event.data?.content as string) || '';
          const currentTrace = useAgentStore.getState().traceLogs;
          const currentTurnMsgId = useAgentStore.getState().currentTurnMessageId;

          if (currentTurnMsgId) {
            // Update existing message - append content and update trace
            const messages = useAgentStore.getState().getMessages(sessionId);
            const existingMsg = messages.find(m => m.id === currentTurnMsgId);

            if (existingMsg) {
              const newContent = existingMsg.content ? existingMsg.content + '\n\n' + content : content;
              updateMessage(sessionId, currentTurnMsgId, {
                content: newContent,
                trace: currentTrace.length > 0 ? [...currentTrace] : undefined,
              });
            }
          } else {
            // Create new message
            const messageId = `msg_${Date.now()}`;
            const message: Message = {
              id: messageId,
              role: 'assistant',
              content,
              timestamp: new Date().toISOString(),
              trace: currentTrace.length > 0 ? [...currentTrace] : undefined,
            };
            addMessage(sessionId, message);
            setCurrentTurnMessageId(messageId);
          }
          break;
        }

        case 'tool_call': {
          const toolName = (event.data?.tool as string) || 'unknown';
          const args = (event.data?.arguments as Record<string, any>) || {};

          // Don't display plan_tool in trace logs (it shows up elsewhere in the UI)
          if (toolName !== 'plan_tool') {
            const log: TraceLog = {
              id: `tool_${Date.now()}`,
              type: 'call',
              text: `Agent is executing ${toolName}...`,
              tool: toolName,
              timestamp: new Date().toISOString(),
              completed: false,
            };
            addTraceLog(log);
            // Update the current turn message's trace in real-time
            updateCurrentTurnTrace(sessionId);
          }

          // Auto-expand Right Panel for specific tools
          if (toolName === 'hf_jobs' && (args.operation === 'run' || args.operation === 'scheduled run') && args.script) {
            setPanelContent({
              title: 'Compute Job Script',
              content: args.script,
              language: 'python'
            });
            setRightPanelOpen(true);
            setLeftSidebarOpen(false);
                    } else if (toolName === 'hf_repo_files' && args.operation === 'upload' && args.content) {
                      setPanelContent({
                        title: `File Upload: ${args.path || 'unnamed'} `,
                        content: args.content,
                        parameters: args,
                        language: args.path?.endsWith('.py') ? 'python' : undefined
                      });
                      setRightPanelOpen(true);
                      setLeftSidebarOpen(false);
                    }

          console.log('Tool call:', toolName, args);
          break;
        }

        case 'tool_output': {
          const toolName = (event.data?.tool as string) || 'unknown';
          const output = (event.data?.output as string) || '';
          const success = event.data?.success as boolean;

          // Mark the corresponding trace log as completed
          updateTraceLog(toolName, { completed: true });
          // Update the current turn message's trace in real-time
          updateCurrentTurnTrace(sessionId);

          // Special handling for hf_jobs - update the approval message with output
          if (toolName === 'hf_jobs') {
            const messages = useAgentStore.getState().getMessages(sessionId);
            const lastApprovalMsg = [...messages].reverse().find(m => m.approval);

            if (lastApprovalMsg) {
                const currentOutput = lastApprovalMsg.toolOutput || '';
                const newOutput = currentOutput ? currentOutput + '\n\n' + output : output;

                useAgentStore.getState().updateMessage(sessionId, lastApprovalMsg.id, {
                    toolOutput: newOutput
                });
                console.log('Updated approval message with tool output:', toolName);
            } else {
                console.warn('Received hf_jobs output but no approval message found to update.');
            }
          }

          // Don't create message bubbles for tool outputs - they only show in trace logs
          console.log('Tool output:', toolName, success);
          break;
        }

        case 'tool_log': {
          const toolName = (event.data?.tool as string) || 'unknown';
          const log = (event.data?.log as string) || '';

          if (toolName === 'hf_jobs') {
            const currentPanel = useAgentStore.getState().panelContent;
            
            // If we are already showing logs, append
            // If we are showing "Compute Job Script", overwrite/switch to logs
            // Otherwise, initialize
            
            let newContent = log;
            if (currentPanel?.title === 'Job Logs') {
              newContent = currentPanel.content + '\n' + log;
            } else if (currentPanel?.title === 'Compute Job Script') {
               // We were showing the script, now logs start.
               // Maybe we want to clear and start showing logs.
               newContent = '--- Starting execution ---\n' + log;
            }

            setPanelContent({
              title: 'Job Logs',
              content: newContent,
              language: 'text'
            });
            
            if (!useLayoutStore.getState().isRightPanelOpen) {
                 setRightPanelOpen(true);
            }
          }
          break;
        }

        case 'plan_update': {
          const plan = (event.data?.plan as any[]) || [];
          setPlan(plan);
          if (!useLayoutStore.getState().isRightPanelOpen) {
            setRightPanelOpen(true);
          }
          break;
        }

        case 'approval_required': {
          const tools = event.data?.tools as Array<{
            tool: string;
            arguments: Record<string, unknown>;
            tool_call_id: string;
          }>;
          const count = (event.data?.count as number) || 0;
          
          // Create a persistent message for the approval request
          const message: Message = {
            id: `msg_approval_${Date.now()}`,
            role: 'assistant',
            content: '', // Content is handled by the approval UI
            timestamp: new Date().toISOString(),
            approval: {
                status: 'pending',
                batch: { tools, count }
            }
          };
          addMessage(sessionId, message);
          
          // We don't set pendingApprovals in the global store anymore as the message handles the UI
          setPendingApprovals(null);
          setProcessing(false);
          break;
        }

        case 'turn_complete':
          setProcessing(false);
          setCurrentTurnMessageId(null); // Clear the current turn
          break;

        case 'compacted': {
          const oldTokens = event.data?.old_tokens as number;
          const newTokens = event.data?.new_tokens as number;
          console.log(`Context compacted: ${oldTokens} -> ${newTokens} tokens`);
          break;
        }

        case 'error': {
          const errorMsg = (event.data?.error as string) || 'Unknown error';
          setError(errorMsg);
          setProcessing(false);
          onError?.(errorMsg);
          break;
        }

        case 'shutdown':
          setConnected(false);
          setProcessing(false);
          break;

        case 'interrupted':
          setProcessing(false);
          break;

        case 'undo_complete':
          // Could remove last messages from store
          break;

        default:
          console.log('Unknown event:', event);
      }
    },
    // Zustand setters are stable, so we don't need them in deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId, onReady, onError]
  );

  const connect = useCallback(() => {
    if (!sessionId) return;
    
    // Don't connect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Connect directly to backend (Vite doesn't proxy WebSockets)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // In development, connect directly to backend port 7860
    // In production, use the same host
    const isDev = import.meta.env.DEV;
    const host = isDev ? '127.0.0.1:7860' : window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/${sessionId}`;

    console.log('Connecting to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      reconnectDelayRef.current = WS_RECONNECT_DELAY;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as AgentEvent;
        handleEvent(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed', event.code, event.reason);
      setConnected(false);

      // Only reconnect if it wasn't a normal closure and session still exists
      if (event.code !== 1000 && sessionId) {
        // Attempt to reconnect with exponential backoff
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = window.setTimeout(() => {
          reconnectDelayRef.current = Math.min(
            reconnectDelayRef.current * 2,
            WS_MAX_RECONNECT_DELAY
          );
          connect();
        }, reconnectDelayRef.current);
      }
    };

    wsRef.current = ws;
  }, [sessionId, handleEvent]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  // Connect when sessionId changes (with a small delay to ensure session is ready)
  useEffect(() => {
    if (!sessionId) {
      disconnect();
      return;
    }

    // Small delay to ensure session is fully created on backend
    const timeoutId = setTimeout(() => {
      connect();
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Heartbeat
  useEffect(() => {
    const interval = setInterval(sendPing, 30000);
    return () => clearInterval(interval);
  }, [sendPing]);

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
    connect,
    disconnect,
  };
}
