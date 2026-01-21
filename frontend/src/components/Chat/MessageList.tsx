import { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import { useSessionStore } from '@/store/sessionStore';
import MessageBubble from './MessageBubble';
import type { Message } from '@/types/agent';

interface MessageListProps {
  messages: Message[];
  isProcessing: boolean;
}

const TechnicalIndicator = () => (
  <Box
    component="span"
    sx={{
      color: 'primary.main',
      fontFamily: 'monospace',
      fontWeight: 'bold',
      fontSize: '1.2rem',
      lineHeight: 0,
      display: 'inline-block',
      verticalAlign: 'middle',
      width: '1em',
      letterSpacing: '-3px',
      transform: 'scale(0.6) translateY(-2px)',
      '&::after': {
        content: '""',
        animation: 'dots 2s steps(4, end) infinite',
      },
      '@keyframes dots': {
        '0%': { content: '""' },
        '25%': { content: '"."' },
        '50%': { content: '".."' },
        '75%, 100%': { content: '"..."' },
      },
    }}
  />
);

export default function MessageList({ messages, isProcessing }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const { activeSessionId } = useSessionStore();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  return (
    <Box
      sx={{
        flex: 1,
        overflow: 'auto',
        p: 2,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box sx={{ maxWidth: 'md', mx: 'auto', width: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {messages.length === 0 && !isProcessing ? (
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
            }}
          >
            <Typography color="text.secondary" sx={{ fontFamily: 'monospace' }}>
              Awaiting input…
            </Typography>
          </Box>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}
        
        {isProcessing && (
          <Box sx={{ width: '100%', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1, px: 0.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                Thinking
              </Typography>
              <TechnicalIndicator />
            </Box>
          </Box>
        )}

        {activeSessionId && (
          // ApprovalFlow is now handled within messages
          null
        )}
        
        <div ref={bottomRef} />
      </Box>
    </Box>
  );
}