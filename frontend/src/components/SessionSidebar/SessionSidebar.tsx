import { useCallback, useState } from 'react';
import {
  Alert,
  Box,
  IconButton,
  Typography,
  CircularProgress,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';

interface SessionSidebarProps {
  onClose?: () => void;
}

/** Small coloured dot for connection status */
const StatusDot = ({ connected }: { connected: boolean }) => (
  <Box
    sx={{
      width: 6,
      height: 6,
      borderRadius: '50%',
      bgcolor: connected ? 'var(--accent-green)' : 'var(--accent-red)',
      boxShadow: connected ? '0 0 4px rgba(76,175,80,0.4)' : 'none',
      flexShrink: 0,
    }}
  />
);

export default function SessionSidebar({ onClose }: SessionSidebarProps) {
  const { sessions, activeSessionId, createSession, deleteSession, switchSession } =
    useSessionStore();
  const { isConnected, setPlan, clearPanel } =
    useAgentStore();
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [capacityError, setCapacityError] = useState<string | null>(null);

  // ── Handlers ──────────────────────────────────────────────────────

  const handleNewSession = useCallback(async () => {
    if (isCreatingSession) return;
    setIsCreatingSession(true);
    setCapacityError(null);
    try {
      const response = await apiFetch('/api/session', { method: 'POST' });
      if (response.status === 503) {
        const data = await response.json();
        setCapacityError(data.detail || 'Server is at capacity.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
      onClose?.();
    } catch {
      setCapacityError('Failed to create session.');
    } finally {
      setIsCreatingSession(false);
    }
  }, [isCreatingSession, createSession, setPlan, clearPanel, onClose]);

  const handleDelete = useCallback(
    async (sessionId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      try {
        await apiFetch(`/api/session/${sessionId}`, { method: 'DELETE' });
        deleteSession(sessionId);
      } catch {
        // Delete locally even if backend fails (session may already be gone)
        deleteSession(sessionId);
      }
    },
    [deleteSession],
  );

  const handleSelect = useCallback(
    (sessionId: string) => {
      switchSession(sessionId);
      setPlan([]);
      clearPanel();
      onClose?.();
    },
    [switchSession, setPlan, clearPanel, onClose],
  );

  const formatTime = (d: string) =>
    new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // ── Render ────────────────────────────────────────────────────────

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--panel)',
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────── */}
      <Box sx={{ px: 1.75, pt: 2, pb: 0 }}>
        <Typography
          variant="caption"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.65rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          Recent chats
        </Typography>
      </Box>

      {/* ── Capacity error ─────────────────────────────────────────── */}
      {capacityError && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setCapacityError(null)}
          sx={{
            m: 1,
            fontSize: '0.7rem',
            py: 0.25,
            '& .MuiAlert-message': { py: 0 },
            borderColor: '#FF9D00',
            color: 'var(--text)',
          }}
        >
          {capacityError}
        </Alert>
      )}

      {/* ── Session list ───────────────────────────────────────────── */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          py: 1,
          // Thinner scrollbar
          '&::-webkit-scrollbar': { width: 4 },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: 'var(--scrollbar-thumb)',
            borderRadius: 2,
          },
        }}
      >
        {sessions.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              px: 3,
              gap: 1.5,
            }}
          >
            <ChatBubbleOutlineIcon
              sx={{ fontSize: 28, color: 'var(--muted-text)', opacity: 0.25 }}
            />
            <Typography
              variant="caption"
              sx={{
                color: 'var(--muted-text)',
                opacity: 0.5,
                textAlign: 'center',
                lineHeight: 1.5,
                fontSize: '0.72rem',
              }}
            >
              No sessions yet
            </Typography>
          </Box>
        ) : (
          [...sessions].reverse().map((session, index) => {
            const num = sessions.length - index;
            const isSelected = session.id === activeSessionId;

            return (
              <Box
                key={session.id}
                onClick={() => handleSelect(session.id)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 1.5,
                  py: 0.875,
                  mx: 0.75,
                  borderRadius: '10px',
                  cursor: 'pointer',
                  transition: 'background-color 0.12s ease',
                  bgcolor: isSelected
                    ? 'var(--hover-bg)'
                    : 'transparent',
                  '&:hover': {
                    bgcolor: 'var(--hover-bg)',
                  },
                  '& .delete-btn': {
                    opacity: 0,
                    transition: 'opacity 0.12s',
                  },
                  '&:hover .delete-btn': {
                    opacity: 1,
                  },
                }}
              >
                <ChatBubbleOutlineIcon
                  sx={{
                    fontSize: 15,
                    color: isSelected ? 'var(--text)' : 'var(--muted-text)',
                    opacity: isSelected ? 0.8 : 0.4,
                    flexShrink: 0,
                  }}
                />

                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: isSelected ? 600 : 400,
                      color: 'var(--text)',
                      fontSize: '0.84rem',
                      lineHeight: 1.4,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {session.title.startsWith('Chat ') ? `Session ${String(num).padStart(2, '0')}` : session.title}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'var(--muted-text)',
                      fontSize: '0.65rem',
                      lineHeight: 1.2,
                    }}
                  >
                    {formatTime(session.createdAt)}
                  </Typography>
                </Box>

                <IconButton
                  className="delete-btn"
                  size="small"
                  onClick={(e) => handleDelete(session.id, e)}
                  sx={{
                    color: 'var(--muted-text)',
                    width: 26,
                    height: 26,
                    flexShrink: 0,
                    '&:hover': { color: 'var(--accent-red)', bgcolor: 'rgba(244,67,54,0.08)' },
                  }}
                >
                  <DeleteOutlineIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </Box>
            );
          })
        )}
      </Box>

      {/* ── Footer: New Session + status ──────────────────────────── */}
      <Divider sx={{ opacity: 0.5 }} />
      <Box
        sx={{
          px: 1.5,
          py: 1.5,
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          flexShrink: 0,
        }}
      >
        <Box
          component="button"
          onClick={handleNewSession}
          disabled={isCreatingSession}
          sx={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 0.75,
            width: '100%',
            px: 1.5,
            py: 1.25,
            border: 'none',
            borderRadius: '10px',
            bgcolor: '#FF9D00',
            color: '#000',
            fontSize: '0.85rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.12s ease',
            '&:hover': {
              bgcolor: '#FFB340',
            },
            '&:disabled': {
              opacity: 0.5,
              cursor: 'not-allowed',
            },
          }}
        >
          {isCreatingSession ? (
            <>
              <CircularProgress size={12} sx={{ color: '#000' }} />
              Creating...
            </>
          ) : (
            <>
              <AddIcon sx={{ fontSize: 16 }} />
              New Session
            </>
          )}
        </Box>

        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 0.5,
          }}
        >
          <StatusDot connected={isConnected} />
          <Typography
            variant="caption"
            sx={{ color: 'var(--muted-text)', fontSize: '0.62rem', letterSpacing: '0.02em' }}
          >
            {sessions.length} session{sessions.length !== 1 ? 's' : ''} &middot; Backend {isConnected ? 'online' : 'offline'}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
