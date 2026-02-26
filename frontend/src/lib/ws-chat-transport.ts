/**
 * Custom ChatTransport that bridges our WebSocket-based backend protocol
 * to the Vercel AI SDK's UIMessageChunk streaming interface.
 *
 * The backend stays unchanged — this adapter translates WebSocket events
 * into the chunk types that useChat() expects.
 */
import type { ChatTransport, UIMessage, UIMessageChunk, ChatRequestOptions } from 'ai';
import { apiFetch, getWebSocketUrl } from '@/utils/api';
import { logger } from '@/utils/logger';
import type { AgentEvent } from '@/types/events';
import { useAgentStore } from '@/store/agentStore';

// ---------------------------------------------------------------------------
// Side-channel callback interface (non-chat events forwarded to the store)
// ---------------------------------------------------------------------------
export interface SideChannelCallbacks {
  onReady: () => void;
  onShutdown: () => void;
  onError: (error: string) => void;
  onProcessing: () => void;
  onProcessingDone: () => void;
  onUndoComplete: () => void;
  onCompacted: (oldTokens: number, newTokens: number) => void;
  onPlanUpdate: (plan: Array<{ id: string; content: string; status: string }>) => void;
  onToolLog: (tool: string, log: string) => void;
  onConnectionChange: (connected: boolean) => void;
  onSessionDead: (sessionId: string) => void;
  /** Called when approval_required arrives — lets the store manage panels */
  onApprovalRequired: (tools: Array<{ tool: string; arguments: Record<string, unknown>; tool_call_id: string }>) => void;
  /** Called when a tool_call arrives with panel-relevant args */
  onToolCallPanel: (tool: string, args: Record<string, unknown>) => void;
  /** Called when tool_output arrives with panel-relevant data */
  onToolOutputPanel: (tool: string, toolCallId: string, output: string, success: boolean) => void;
  /** Called when assistant text starts streaming */
  onStreaming: () => void;
  /** Called when a tool starts running (non-plan) */
  onToolRunning: (toolName: string) => void;
}

// ---------------------------------------------------------------------------
// Transport options
// ---------------------------------------------------------------------------
export interface WebSocketChatTransportOptions {
  sideChannel: SideChannelCallbacks;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const WS_RECONNECT_DELAY = 1000;
const WS_MAX_RECONNECT_DELAY = 30000;
const WS_MAX_RETRIES = 5;
const WS_PING_INTERVAL = 30000;

let partIdCounter = 0;
function nextPartId(prefix: string): string {
  return `${prefix}-${Date.now()}-${++partIdCounter}`;
}

// ---------------------------------------------------------------------------
// Transport implementation
// ---------------------------------------------------------------------------
export class WebSocketChatTransport implements ChatTransport<UIMessage> {
  private ws: WebSocket | null = null;
  private currentSessionId: string | null = null;
  private sideChannel: SideChannelCallbacks;

  private streamController: ReadableStreamDefaultController<UIMessageChunk> | null = null;
  private streamGeneration = 0;
  private abortedGeneration = 0;
  private textPartId: string | null = null;
  private awaitingProcessing = false;

  private connectTimeout: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = WS_RECONNECT_DELAY;
  private retries = 0;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private boundVisibilityHandler: (() => void) | null = null;
  private wasHidden = false;

  constructor({ sideChannel }: WebSocketChatTransportOptions) {
    this.sideChannel = sideChannel;
    this.setupVisibilityHandler();
  }

  private setupVisibilityHandler(): void {
    this.boundVisibilityHandler = () => {
      if (document.visibilityState === 'hidden') {
        this.wasHidden = true;
        return;
      }

      if (document.visibilityState === 'visible' && this.currentSessionId) {
        const wsState = this.ws?.readyState;
        if (wsState !== WebSocket.OPEN && wsState !== WebSocket.CONNECTING) {
          logger.log('Tab visible: WS is dead, reconnecting immediately');
          this.retries = 0;
          this.reconnectDelay = WS_RECONNECT_DELAY;
          this.createWebSocket(this.currentSessionId);

          if (this.wasHidden) {
            const store = useAgentStore.getState();
            if (store.isProcessing) {
              logger.log('Tab visible after WS drop: resetting stale processing state');
              store.setProcessing(false);
              this.closeActiveStream();
            }
          }
        } else if (wsState === WebSocket.OPEN) {
          this.ws!.send(JSON.stringify({ type: 'ping' }));
        }
        this.wasHidden = false;
      }
    };
    document.addEventListener('visibilitychange', this.boundVisibilityHandler);
  }

  /** Update side-channel callbacks (e.g. when sessionId changes). */
  updateSideChannel(sideChannel: SideChannelCallbacks): void {
    this.sideChannel = sideChannel;
  }

  // ── Public API ──────────────────────────────────────────────────────

  /** Connect (or reconnect) to a session's WebSocket. */
  connectToSession(sessionId: string | null): void {
    if (this.connectTimeout) {
      clearTimeout(this.connectTimeout);
      this.connectTimeout = null;
    }
    this.disconnectWebSocket();
    this.currentSessionId = sessionId;
    if (sessionId) {
      this.retries = 0;
      this.reconnectDelay = WS_RECONNECT_DELAY;
      this.connectTimeout = setTimeout(() => {
        this.connectTimeout = null;
        if (this.currentSessionId === sessionId) {
          this.createWebSocket(sessionId);
        }
      }, 100);
    }
  }

  /** Approve / reject tools. Called directly from the UI. */
  async approveTools(
    sessionId: string,
    approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null }>,
  ): Promise<boolean> {
    try {
      const res = await apiFetch('/api/approve', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, approvals }),
      });
      return res.ok;
    } catch (e) {
      logger.error('Approval request failed:', e);
      return false;
    }
  }

  /** Clean up everything. */
  destroy(): void {
    if (this.connectTimeout) {
      clearTimeout(this.connectTimeout);
      this.connectTimeout = null;
    }
    if (this.boundVisibilityHandler) {
      document.removeEventListener('visibilitychange', this.boundVisibilityHandler);
      this.boundVisibilityHandler = null;
    }
    this.disconnectWebSocket();
    this.closeActiveStream();
  }

  // ── ChatTransport interface ─────────────────────────────────────────

  async sendMessages(
    options: {
      trigger: 'submit-message' | 'regenerate-message';
      chatId: string;
      messageId: string | undefined;
      messages: UIMessage[];
      abortSignal: AbortSignal | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const sessionId = options.chatId;

    // Close any previously active stream (e.g. user sent new msg during approval)
    this.closeActiveStream();

    // Track generation to protect against late cancel from a stale stream
    const gen = ++this.streamGeneration;
    logger.log(`sendMessages: gen=${gen}, awaitingProcessing=${this.awaitingProcessing}, abortedGen=${this.abortedGeneration}`);

    // Wire up abort signal to interrupt the backend and close the stream
    if (options.abortSignal) {
      const onAbort = () => {
        if (this.streamGeneration !== gen) return;
        logger.log(`Stream aborted by user (gen=${gen})`);
        this.interruptBackend(sessionId);
        this.endTextPart();
        if (this.streamController) {
          this.enqueue({ type: 'finish-step' });
          this.enqueue({ type: 'finish', finishReason: 'stop' });
          this.closeActiveStream();
        }
        this.awaitingProcessing = true;
        this.abortedGeneration = this.streamGeneration;
        logger.log(`Abort complete: awaitingProcessing=true, abortedGen=${this.abortedGeneration}`);
        this.sideChannel.onProcessingDone();
      };
      if (options.abortSignal.aborted) {
        onAbort();
      } else {
        options.abortSignal.addEventListener('abort', onAbort, { once: true });
      }
    }

    // Create the stream BEFORE the POST so WebSocket events arriving
    // while the HTTP request is in-flight are captured immediately.
    const stream = new ReadableStream<UIMessageChunk>({
      start: (controller) => {
        this.streamController = controller;
        this.textPartId = null;
      },
      cancel: () => {
        if (this.streamGeneration === gen) {
          this.streamController = null;
          this.textPartId = null;
        }
      },
    });

    // Extract the latest user text from the messages array
    const lastUserMsg = [...options.messages].reverse().find(m => m.role === 'user');
    const text = lastUserMsg
      ? lastUserMsg.parts
          .filter((p): p is Extract<typeof p, { type: 'text' }> => p.type === 'text')
          .map(p => p.text)
          .join('')
      : '';

    // POST to the existing backend endpoint
    try {
      await apiFetch('/api/submit', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, text }),
      });
    } catch (e) {
      logger.error('Submit failed:', e);
      this.enqueue({ type: 'error', errorText: 'Failed to send message' });
      this.closeActiveStream();
    }

    return stream;
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    return null;
  }

  /** Ask the backend to interrupt the current generation. Fire-and-forget. */
  private interruptBackend(sessionId: string): void {
    apiFetch(`/api/interrupt/${sessionId}`, { method: 'POST' }).catch((e) =>
      logger.warn('Interrupt request failed:', e),
    );
  }

  // ── WebSocket lifecycle ─────────────────────────────────────────────

  private createWebSocket(sessionId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      return;
    }

    const wsUrl = getWebSocketUrl(sessionId);
    logger.log('WS transport connecting:', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      logger.log('WS transport connected');
      this.sideChannel.onConnectionChange(true);
      this.reconnectDelay = WS_RECONNECT_DELAY;
      this.retries = 0;
      this.startPing();
    };

    ws.onmessage = (evt) => {
      try {
        const raw = JSON.parse(evt.data);
        if (raw.type === 'pong') return;
        this.handleEvent(raw as AgentEvent);
      } catch (e) {
        logger.error('WS parse error:', e);
      }
    };

    ws.onerror = (err) => logger.error('WS error:', err);

    ws.onclose = (evt) => {
      logger.log('WS closed', evt.code, evt.reason);
      this.sideChannel.onConnectionChange(false);
      this.stopPing();

      const noRetry = [1000, 4001, 4003, 4004];
      if (evt.code === 4004 && sessionId) {
        this.sideChannel.onSessionDead(sessionId);
        return;
      }
      if (!noRetry.includes(evt.code) && this.currentSessionId === sessionId) {
        this.retries += 1;
        if (this.retries > WS_MAX_RETRIES) {
          logger.warn('WS max retries reached');
          this.sideChannel.onSessionDead(sessionId);
          return;
        }
        this.reconnectTimeout = setTimeout(() => {
          this.reconnectDelay = Math.min(this.reconnectDelay * 2, WS_MAX_RECONNECT_DELAY);
          this.createWebSocket(sessionId);
        }, this.reconnectDelay);
      }
    };

    this.ws = ws;
  }

  private disconnectWebSocket(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.sideChannel.onConnectionChange(false);
  }

  private startPing(): void {
    this.stopPing();
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, WS_PING_INTERVAL);
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  // ── Stream helpers ──────────────────────────────────────────────────

  private closeActiveStream(): void {
    if (this.streamController) {
      try {
        this.streamController.close();
      } catch {
        // already closed
      }
      this.streamController = null;
      this.textPartId = null;
    }
  }

  private enqueue(chunk: UIMessageChunk): void {
    try {
      this.streamController?.enqueue(chunk);
    } catch {
      // stream already closed
    }
  }

  private endTextPart(): void {
    if (this.textPartId) {
      this.enqueue({ type: 'text-end', id: this.textPartId });
      this.textPartId = null;
    }
  }

  // ── Event → UIMessageChunk mapping ──────────────────────────────────

  private static readonly STREAM_EVENTS = new Set([
    'assistant_chunk', 'assistant_stream_end', 'assistant_message',
    'tool_call', 'tool_output', 'approval_required', 'tool_state_change',
    'turn_complete', 'error',
  ]);

  private handleEvent(event: AgentEvent): void {
    // After an abort, ignore stale stream events until the next 'processing'
    if (this.awaitingProcessing && WebSocketChatTransport.STREAM_EVENTS.has(event.event_type)) {
      logger.log(`Filtering stale "${event.event_type}" (gen=${this.streamGeneration}, aborted=${this.abortedGeneration})`);
      return;
    }

    switch (event.event_type) {
      // ── Side-channel only events ────────────────────────────────
      case 'ready':
        this.sideChannel.onReady();
        break;

      case 'shutdown':
        this.sideChannel.onShutdown();
        this.closeActiveStream();
        break;

      case 'interrupted':
        // Don't close the stream here — the abort handler already did, and
        // a new stream for the next user message may already exist.
        // Closing here would destroy the NEWER stream, causing the next
        // response to be silently dropped.
        this.sideChannel.onProcessingDone();
        break;

      case 'undo_complete':
        this.endTextPart();
        this.closeActiveStream();
        this.sideChannel.onUndoComplete();
        break;

      case 'compacted':
        this.sideChannel.onCompacted(
          (event.data?.old_tokens as number) || 0,
          (event.data?.new_tokens as number) || 0,
        );
        break;

      case 'plan_update':
        this.sideChannel.onPlanUpdate(
          (event.data?.plan as Array<{ id: string; content: string; status: string }>) || [],
        );
        break;

      case 'tool_log':
        this.sideChannel.onToolLog(
          (event.data?.tool as string) || '',
          (event.data?.log as string) || '',
        );
        break;

      // ── Chat stream events ──────────────────────────────────────
      case 'processing':
        if (this.awaitingProcessing) {
          if (this.streamGeneration <= this.abortedGeneration) {
            logger.log(`Ignoring stale "processing" (gen=${this.streamGeneration} <= aborted=${this.abortedGeneration})`);
            break;
          }
          logger.log(`Accepting "processing" for new generation (gen=${this.streamGeneration}, aborted=${this.abortedGeneration})`);
          this.awaitingProcessing = false;
        }
        this.sideChannel.onProcessing();
        if (this.streamController) {
          this.enqueue({
            type: 'start',
            messageMetadata: { createdAt: new Date().toISOString() },
          });
          this.enqueue({ type: 'start-step' });
        }
        break;

      case 'assistant_chunk': {
        const delta = (event.data?.content as string) || '';
        if (!delta || !this.streamController) break;

        if (!this.textPartId) {
          this.textPartId = nextPartId('text');
          this.enqueue({ type: 'text-start', id: this.textPartId });
          this.sideChannel.onStreaming();
        }
        this.enqueue({ type: 'text-delta', id: this.textPartId, delta });
        break;
      }

      case 'assistant_stream_end':
        this.endTextPart();
        break;

      case 'assistant_message': {
        const content = (event.data?.content as string) || '';
        if (!content || !this.streamController) break;
        const id = nextPartId('text');
        this.enqueue({ type: 'text-start', id });
        this.enqueue({ type: 'text-delta', id, delta: content });
        this.enqueue({ type: 'text-end', id });
        break;
      }

      case 'tool_call': {
        if (!this.streamController) break;
        const toolName = (event.data?.tool as string) || 'unknown';
        const toolCallId = (event.data?.tool_call_id as string) || '';
        const args = (event.data?.arguments as Record<string, unknown>) || {};

        if (toolName === 'plan_tool') break;

        this.endTextPart();
        this.enqueue({ type: 'tool-input-start', toolCallId, toolName, dynamic: true });
        this.enqueue({ type: 'tool-input-available', toolCallId, toolName, input: args, dynamic: true });

        this.sideChannel.onToolRunning(toolName);
        this.sideChannel.onToolCallPanel(toolName, args as Record<string, unknown>);
        break;
      }

      case 'tool_output': {
        if (!this.streamController) break;
        const toolCallId = (event.data?.tool_call_id as string) || '';
        const output = (event.data?.output as string) || '';
        const success = event.data?.success as boolean;
        const toolName = (event.data?.tool as string) || '';

        if (toolName === 'plan_tool' || toolCallId.startsWith('plan_tool')) break;

        if (success) {
          this.enqueue({ type: 'tool-output-available', toolCallId, output, dynamic: true });
        } else {
          this.enqueue({ type: 'tool-output-error', toolCallId, errorText: output, dynamic: true });
        }

        this.sideChannel.onToolOutputPanel(toolName, toolCallId, output, success);
        break;
      }

      case 'approval_required': {
        const tools = event.data?.tools as Array<{
          tool: string;
          arguments: Record<string, unknown>;
          tool_call_id: string;
        }>;
        if (!tools || !this.streamController) break;

        this.endTextPart();

        for (const t of tools) {
          this.enqueue({ type: 'tool-input-start', toolCallId: t.tool_call_id, toolName: t.tool, dynamic: true });
          this.enqueue({ type: 'tool-input-available', toolCallId: t.tool_call_id, toolName: t.tool, input: t.arguments, dynamic: true });
          this.enqueue({ type: 'tool-approval-request', approvalId: `approval-${t.tool_call_id}`, toolCallId: t.tool_call_id });
        }

        this.sideChannel.onApprovalRequired(tools);
        this.sideChannel.onProcessingDone();
        break;
      }

      case 'tool_state_change': {
        const tcId = (event.data?.tool_call_id as string) || '';
        const state = (event.data?.state as string) || '';
        const jobUrl = (event.data?.jobUrl as string) || undefined;

        if (tcId.startsWith('plan_tool')) break;

        if (jobUrl && tcId) {
          useAgentStore.getState().setJobUrl(tcId, jobUrl);
        }

        if (this.streamController && (state === 'rejected' || state === 'abandoned')) {
          this.enqueue({ type: 'tool-output-denied', toolCallId: tcId });
        }
        break;
      }

      case 'turn_complete':
        this.endTextPart();
        if (this.streamController) {
          this.enqueue({ type: 'finish-step' });
          this.enqueue({ type: 'finish', finishReason: 'stop' });
          this.closeActiveStream();
        }
        this.sideChannel.onProcessingDone();
        break;

      case 'error': {
        const errorMsg = (event.data?.error as string) || 'Unknown error';
        this.sideChannel.onError(errorMsg);
        if (this.streamController) {
          this.enqueue({ type: 'error', errorText: errorMsg });
        }
        this.sideChannel.onProcessingDone();
        break;
      }

      default:
        logger.log('WS transport: unknown event', event);
    }
  }
}
