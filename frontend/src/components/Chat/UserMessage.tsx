import { Box, Stack, Typography, IconButton, Tooltip } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import type { UIMessage } from 'ai';
import type { MessageMeta } from '@/types/agent';

interface UserMessageProps {
  message: UIMessage;
  isLastTurn?: boolean;
  onUndoTurn?: () => void;
  isProcessing?: boolean;
}

function extractText(message: UIMessage): string {
  return message.parts
    .filter((p): p is Extract<typeof p, { type: 'text' }> => p.type === 'text')
    .map(p => p.text)
    .join('');
}

export default function UserMessage({
  message,
  isLastTurn = false,
  onUndoTurn,
  isProcessing = false,
}: UserMessageProps) {
  const showUndo = isLastTurn && !isProcessing && !!onUndoTurn;
  const text = extractText(message);
  const meta = message.metadata as MessageMeta | undefined;
  const timeStr = meta?.createdAt
    ? new Date(meta.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;
  return (
    <Stack
      direction="row"
      spacing={1.5}
      justifyContent="flex-end"
      alignItems="flex-start"
      sx={{
        '& .undo-btn': {
          opacity: 0,
          transition: 'opacity 0.15s ease',
        },
        '&:hover .undo-btn': {
          opacity: 1,
        },
      }}
    >
      {showUndo && (
        <Box className="undo-btn" sx={{ display: 'flex', alignItems: 'center', mt: 0.75 }}>
          <Tooltip title="Remove this turn" placement="left">
            <IconButton
              onClick={onUndoTurn}
              size="small"
              sx={{
                width: 24,
                height: 24,
                color: 'var(--muted-text)',
                '&:hover': {
                  color: 'var(--accent-red)',
                  bgcolor: 'rgba(244,67,54,0.08)',
                },
              }}
            >
              <CloseIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      <Box
        sx={{
          maxWidth: { xs: '88%', md: '72%' },
          bgcolor: 'var(--surface)',
          borderRadius: 1.5,
          borderTopRightRadius: 4,
          px: { xs: 1.5, md: 2.5 },
          py: 1.5,
          border: '1px solid var(--border)',
        }}
      >
        <Typography
          variant="body1"
          sx={{
            fontSize: '0.925rem',
            lineHeight: 1.65,
            color: 'var(--text)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {text}
        </Typography>

        {timeStr && (
          <Typography
            variant="caption"
            sx={{ color: 'var(--muted-text)', mt: 0.5, display: 'block', textAlign: 'right', fontSize: '0.7rem' }}
          >
            {timeStr}
          </Typography>
        )}
      </Box>
    </Stack>
  );
}
