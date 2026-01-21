import { Box, Paper, Typography, Chip } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ApprovalFlow from './ApprovalFlow';
import type { Message } from '@/types/agent';

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isTool = message.role === 'tool';
  const isAssistant = message.role === 'assistant';

  if (message.approval) {
    return (
        <Box sx={{ width: '100%', maxWidth: '880px', mx: 'auto', my: 2 }}>
            <ApprovalFlow message={message} />
        </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        width: '100%',
        maxWidth: '880px',
        mx: 'auto',
      }}
    >
      <Paper
        elevation={0}
        className={`message ${isUser ? 'user' : isAssistant ? 'assistant' : ''}`}
        sx={{
          p: '14px 18px',
          margin: '10px 0',
          width: isTool ? '100%' : 'auto',
          maxWidth: '100%',
          borderRadius: 'var(--radius-lg)',
          borderTopLeftRadius: isAssistant ? '6px' : undefined,
          lineHeight: 1.45,
          boxShadow: 'var(--shadow-1)',
          border: '1px solid rgba(255,255,255,0.03)',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.015), transparent)',
        }}
      >
        {isTool && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Tool
            </Typography>
            {message.toolName && (
              <Chip
                label={message.toolName}
                size="small"
                variant="outlined"
                sx={{ ml: 1, height: 20, fontSize: '0.7rem' }}
              />
            )}
          </Box>
        )}

        <Box
          sx={{
            '& p': { m: 0, color: isUser ? 'var(--text)' : 'var(--text)' }, // User might want different text color? Defaults to --text
            '& pre': {
              bgcolor: 'rgba(0,0,0,0.5)',
              p: 1.5,
              borderRadius: 1,
              overflow: 'auto',
              fontSize: '0.85rem',
              border: '1px solid rgba(255,255,255,0.05)',
            },
            '& code': {
              bgcolor: 'rgba(255,255,255,0.05)',
              px: 0.5,
              py: 0.25,
              borderRadius: 0.5,
              fontSize: '0.85rem',
              fontFamily: '"JetBrains Mono", monospace',
            },
            '& pre code': {
              bgcolor: 'transparent',
              p: 0,
            },
            '& a': {
              color: 'var(--accent-yellow)',
              textDecoration: 'none',
              '&:hover': {
                textDecoration: 'underline',
              },
            },
            '& ul, & ol': {
              pl: 2,
              my: 1,
            },
            '& table': {
              borderCollapse: 'collapse',
              width: '100%',
              my: 2,
              fontSize: '0.875rem',
            },
            '& th': {
              borderBottom: '1px solid',
              borderColor: 'rgba(255,255,255,0.1)',
              textAlign: 'left',
              p: 1,
              bgcolor: 'rgba(255,255,255,0.02)',
            },
            '& td': {
              borderBottom: '1px solid',
              borderColor: 'rgba(255,255,255,0.05)',
              p: 1,
            },
          }}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </Box>

        {/* Persisted Trace Logs - Now at the bottom */}
        {message.trace && message.trace.length > 0 && (
          <Box
            sx={{
              bgcolor: 'rgba(0,0,0,0.3)',
              borderRadius: 1,
              p: 1.5,
              border: 1,
              borderColor: 'rgba(255,255,255,0.05)',
              width: '100%',
              mt: 2,
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {message.trace.map((log) => {
                // Extract tool name from text "Agent is executing {toolName}..."
                const match = log.text.match(/Agent is executing (.+)\.\.\./);
                const toolName = match ? match[1] : log.tool;

                return (
                  <Typography
                    key={log.id}
                    variant="caption"
                    component="div"
                    sx={{
                      color: 'var(--muted-text)',
                      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
                      fontSize: '0.75rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                    }}
                  >
                    <span style={{ color: log.completed ? '#FDB022' : 'inherit' }}>*</span>
                    <span>Agent is executing </span>
                    <span style={{
                      fontWeight: 600,
                      color: 'rgba(255, 255, 255, 0.9)',
                    }}>
                      {toolName}
                    </span>
                    <span>...</span>
                  </Typography>
                );
              })}
            </Box>
          </Box>
        )}

        <Typography className="meta" variant="caption" sx={{ display: 'block', textAlign: 'right', mt: 1, fontSize: '11px', opacity: 0.5 }}>
            {new Date(message.timestamp).toLocaleTimeString()}
        </Typography>
      </Paper>
    </Box>
  );
}
