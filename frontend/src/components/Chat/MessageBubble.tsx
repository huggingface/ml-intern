import UserMessage from './UserMessage';
import AssistantMessage from './AssistantMessage';
import type { UIMessage } from 'ai';

interface MessageBubbleProps {
  message: UIMessage;
  isLastTurn?: boolean;
  onUndoTurn?: () => void;
  isProcessing?: boolean;
  isStreaming?: boolean;
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null }>) => Promise<boolean>;
}

export default function MessageBubble({
  message,
  isLastTurn = false,
  onUndoTurn,
  isProcessing = false,
  isStreaming = false,
  approveTools,
}: MessageBubbleProps) {
  if (message.role === 'user') {
    return (
      <UserMessage
        message={message}
        isLastTurn={isLastTurn}
        onUndoTurn={onUndoTurn}
        isProcessing={isProcessing}
      />
    );
  }

  if (message.role === 'assistant') {
    return (
      <AssistantMessage
        message={message}
        isStreaming={isStreaming}
        approveTools={approveTools}
      />
    );
  }

  return null;
}
