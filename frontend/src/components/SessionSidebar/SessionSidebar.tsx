import { useCallback, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Typography,
  CircularProgress,
  Divider,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  TextField,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';

interface SessionSidebarProps {
  onClose?: () => void;
}

export default function SessionSidebar({ onClose }: SessionSidebarProps) {
  const { sessions, activeSessionId, createSession, deleteSession, switchSession, updateSessionTitle } =
    useSessionStore();
  const { setPlan, clearPanel } =
    useAgentStore();
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [capacityError, setCapacityError] = useState<string | null>(null);

  // -- Handlers -----------------------------------------------------------

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

  // -- Delete with dialog confirmation ------------------------------------
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null);
  const [renameSessionId, setRenameSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');

  const handleDeleteClick = useCallback(
    (sessionId: string, e?: React.MouseEvent) => {
      e?.stopPropagation();
      setMenuAnchorEl(null);
      setMenuSessionId(null);
      setConfirmDeleteId(sessionId);
    },
    [],
  );

  const handleMenuOpen = useCallback((sessionId: string, e: React.MouseEvent<HTMLElement>) => {
    e.stopPropagation();
    setMenuAnchorEl(e.currentTarget);
    setMenuSessionId(sessionId);
  }, []);

  const handleMenuClose = useCallback(() => {
    setMenuAnchorEl(null);
    setMenuSessionId(null);
  }, []);

  const openRenameDialog = useCallback(() => {
    if (!menuSessionId) return;
    const session = sessions.find((s) => s.id === menuSessionId);
    setRenameSessionId(menuSessionId);
    setRenameValue((session?.title || '').trim());
    handleMenuClose();
  }, [menuSessionId, sessions, handleMenuClose]);

  const handleRenameSave = useCallback(() => {
    if (!renameSessionId) return;
    const trimmed = renameValue.trim();
    if (!trimmed) return;
    const clean = trimmed.length > 60 ? `${trimmed.slice(0, 60).trimEnd()}…` : trimmed;
    updateSessionTitle(renameSessionId, clean);
    setRenameSessionId(null);
    setRenameValue('');
  }, [renameSessionId, renameValue, updateSessionTitle]);

  const handleDeleteConfirm = useCallback(async () => {
    if (!confirmDeleteId || isDeleting) return;
    const sessionId = confirmDeleteId;
    setIsDeleting(true);

    const isLastSession = sessions.length === 1;

    useAgentStore.getState().clearSessionState(sessionId);
    try {
      await apiFetch(`/api/session/${sessionId}`, { method: 'DELETE' });
      deleteSession(sessionId);
    } catch {
      deleteSession(sessionId);
    }

    // If this was the last session, create a new one
    if (isLastSession) {
      try {
        const response = await apiFetch('/api/session', { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          createSession(data.session_id);
          setPlan([]);
          clearPanel();
        }
      } catch (error) {
        console.error('Failed to create new session after deleting last one:', error);
      }
    }

    setIsDeleting(false);
    setConfirmDeleteId(null);
  }, [deleteSession, confirmDeleteId, isDeleting, sessions, createSession, setPlan, clearPanel]);

  const handleSelect = useCallback(
    (sessionId: string) => {
      switchSession(sessionId);
      // Per-session state (plan, panel, activity) is restored automatically
      // by SessionChat's useEffect when isActive flips to true.
      onClose?.();
    },
    [switchSession, onClose],
  );

  const formatTime = (d: string) =>
    new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // -- Render -------------------------------------------------------------

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--panel)',
      }}
    >
      {/* -- Header -------------------------------------------------------- */}
      <Box sx={{ px: 1.75, pt: 2, pb: 0 }}>
        <Typography
          variant="caption"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.68rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}
        >
          Recent chats
        </Typography>
      </Box>

      {/* -- Capacity error ------------------------------------------------ */}
      {capacityError && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setCapacityError(null)}
          sx={{
            m: 1,
            fontSize: '0.76rem',
            py: 0.25,
            '& .MuiAlert-message': { py: 0 },
            borderColor: '#FF9D00',
            color: 'var(--text)',
          }}
        >
          {capacityError}
        </Alert>
      )}

      {/* -- Session list -------------------------------------------------- */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          py: 1,
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
                  mb: 0.2,
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
                  '& .more-btn': {
                    opacity: 0,
                    transition: 'opacity 0.12s',
                  },
                  '&:hover .more-btn': {
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

                {/* Attention badge — pulsing dot when background session needs approval */}
                {session.needsAttention && !isSelected && (
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: 'var(--accent-yellow)',
                      flexShrink: 0,
                      animation: 'pulse 2s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%, 100%': { opacity: 1, transform: 'scale(1)' },
                        '50%': { opacity: 0.5, transform: 'scale(0.8)' },
                      },
                    }}
                  />
                )}

                <IconButton
                  className="more-btn"
                  size="small"
                  onClick={(e) => handleMenuOpen(session.id, e)}
                  sx={{
                    color: 'var(--muted-text)',
                    width: 26,
                    height: 26,
                    flexShrink: 0,
                    '&:hover': { color: 'var(--text)', bgcolor: 'var(--hover-bg)' },
                  }}
                >
                  <MoreVertIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </Box>
            );
          })
        )}
      </Box>

      {/* -- Footer: New Task + status ------------------------------------- */}
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
            borderRadius: '12px',
            bgcolor: '#FF9D00',
            color: '#000',
            fontSize: '0.88rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'transform 0.16s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.16s ease',
            '&:hover': {
              bgcolor: '#FFB340',
              transform: 'translateY(-1px)',
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
              Creating session...
            </>
          ) : (
            <>
              <AddIcon sx={{ fontSize: 16 }} />
              New Task
            </>
          )}
        </Box>

      </Box>
      {/* Delete confirmation dialog */}
      <Dialog
        open={!!confirmDeleteId}
        onClose={() => !isDeleting && setConfirmDeleteId(null)}
        slotProps={{
          backdrop: { sx: { backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' } },
        }}
        PaperProps={{
          sx: {
            bgcolor: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-1)',
            maxWidth: 340,
            mx: 2,
          },
        }}
      >
        <DialogTitle
          sx={{
            color: 'var(--text)',
            fontWeight: 700,
            fontSize: '0.95rem',
            pb: 0,
            pt: 2.5,
            px: 3,
          }}
        >
          Delete conversation?
        </DialogTitle>
        <DialogContent sx={{ px: 3, pt: 1 }}>
          <DialogContentText
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              lineHeight: 1.6,
            }}
          >
            This permanently removes this conversation and all related history.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
          <Button
            onClick={() => setConfirmDeleteId(null)}
            size="small"
            disabled={isDeleting}
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              px: 2,
              '&:hover': { bgcolor: 'var(--hover-bg)' },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            size="small"
            disabled={isDeleting}
            startIcon={isDeleting ? <CircularProgress size={16} sx={{ color: '#fff' }} /> : undefined}
            sx={{
              fontSize: '0.82rem',
              px: 2.5,
              bgcolor: 'var(--accent-red)',
              color: '#fff',
              boxShadow: 'none',
              '&:hover': {
                bgcolor: 'var(--accent-red)',
                filter: 'brightness(1.15)',
                boxShadow: 'none',
              },
              '&.Mui-disabled': {
                bgcolor: 'var(--accent-red)',
                color: '#fff',
                opacity: 0.7,
              },
            }}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={!!renameSessionId}
        onClose={() => setRenameSessionId(null)}
        PaperProps={{
          sx: {
            bgcolor: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-1)',
            maxWidth: 420,
            mx: 2,
          },
        }}
      >
        <DialogTitle
          sx={{
            color: 'var(--text)',
            fontWeight: 700,
            fontSize: '0.95rem',
            pb: 0.5,
            pt: 2.5,
            px: 3,
          }}
        >
          Rename conversation
        </DialogTitle>
        <DialogContent sx={{ px: 3, pt: 1 }}>
          <TextField
            autoFocus
            fullWidth
            size="small"
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleRenameSave();
              }
            }}
            placeholder="Enter a new title"
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
          <Button
            onClick={() => setRenameSessionId(null)}
            size="small"
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              px: 2,
              '&:hover': { bgcolor: 'var(--hover-bg)' },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleRenameSave}
            variant="contained"
            size="small"
            disabled={!renameValue.trim()}
            sx={{
              fontSize: '0.82rem',
              px: 2.5,
              bgcolor: 'var(--accent-yellow)',
              color: '#1b1b1b',
              boxShadow: 'none',
              '&:hover': {
                bgcolor: 'var(--accent-yellow)',
                filter: 'brightness(1.08)',
                boxShadow: 'none',
              },
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
        slotProps={{
          paper: {
            sx: {
              bgcolor: 'var(--panel)',
              border: '1px solid var(--border)',
              minWidth: 170,
            },
          },
        }}
      >
        <MenuItem onClick={openRenameDialog}>
          <ListItemIcon>
            <EditOutlinedIcon sx={{ fontSize: 16, color: 'var(--muted-text)' }} />
          </ListItemIcon>
          <ListItemText primary="Rename" />
        </MenuItem>
        <MenuItem
          onClick={() => {
            if (menuSessionId) handleDeleteClick(menuSessionId);
          }}
        >
          <ListItemIcon>
            <DeleteOutlineIcon sx={{ fontSize: 16, color: 'var(--accent-red)' }} />
          </ListItemIcon>
          <ListItemText
            primary="Delete"
            primaryTypographyProps={{ sx: { color: 'var(--accent-red)' } }}
          />
        </MenuItem>
      </Menu>
    </Box>
  );
}
