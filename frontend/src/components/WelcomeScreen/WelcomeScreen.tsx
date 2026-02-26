import { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { isInIframe, triggerLogin } from '@/hooks/useAuth';

/** HF brand orange */
const HF_ORANGE = '#FF9D00';

export default function WelcomeScreen() {
  const { createSession } = useSessionStore();
  const { setPlan, clearPanel, user } = useAgentStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inIframe = isInIframe();
  const isAuthenticated = user?.authenticated;
  const isDevUser = user?.username === 'dev';

  const handleStart = useCallback(async () => {
    if (isCreating) return;

    // Not authenticated and not dev → need to login
    if (!isAuthenticated && !isDevUser) {
      // In iframe: can't redirect (cookies blocked) — user needs to open in new tab
      // This shouldn't happen because we show a different button in iframe
      // But just in case:
      if (inIframe) return;
      triggerLogin();
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      const response = await apiFetch('/api/session', { method: 'POST' });
      if (response.status === 503) {
        const data = await response.json();
        setError(data.detail || 'Server is at capacity. Please try again later.');
        return;
      }
      if (response.status === 401) {
        triggerLogin();
        return;
      }
      if (!response.ok) {
        setError('Failed to create session. Please try again.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
    } catch {
      // Redirect may throw — ignore
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, createSession, setPlan, clearPanel, isAuthenticated, isDevUser, inIframe]);

  // Build the direct Space URL for the "open in new tab" link
  const spaceHost = typeof window !== 'undefined'
    ? window.location.hostname.includes('.hf.space')
      ? window.location.origin
      : `https://smolagents-ml-agent.hf.space`
    : '';

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--body-gradient)',
        py: 8,
      }}
    >
      {/* HF Logo */}
      <Box
        component="img"
        src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
        alt="Hugging Face"
        sx={{ width: 96, height: 96, mb: 3, display: 'block' }}
      />

      {/* Title */}
      <Typography
        variant="h2"
        sx={{
          fontWeight: 800,
          color: 'var(--text)',
          mb: 1.5,
          letterSpacing: '-0.02em',
          fontSize: { xs: '2rem', md: '2.8rem' },
        }}
      >
        HF Agent
      </Typography>

      {/* Description */}
      <Typography
        variant="body1"
        sx={{
          color: 'var(--muted-text)',
          maxWidth: 520,
          mb: 5,
          lineHeight: 1.8,
          fontSize: '0.95rem',
          textAlign: 'center',
          px: 2,
          '& strong': { color: 'var(--text)', fontWeight: 600 },
        }}
      >
        A general-purpose AI agent for <strong>machine learning engineering</strong>.
        It browses <strong>Hugging Face documentation</strong>, manages{' '}
        <strong>repositories</strong>, launches <strong>training jobs</strong>,
        and explores <strong>datasets</strong> — all through natural conversation.
      </Typography>

      {/* Action button — depends on context */}
      {inIframe && !isAuthenticated && !isDevUser ? (
        // In iframe + not logged in → link to open Space directly
        <Button
          variant="contained"
          size="large"
          component="a"
          href={spaceHost}
          target="_blank"
          rel="noopener noreferrer"
          endIcon={<OpenInNewIcon />}
          sx={{
            px: 5,
            py: 1.5,
            fontSize: '1rem',
            fontWeight: 700,
            textTransform: 'none',
            borderRadius: '12px',
            bgcolor: HF_ORANGE,
            color: '#000',
            boxShadow: '0 4px 24px rgba(255, 157, 0, 0.3)',
            textDecoration: 'none',
            '&:hover': {
              bgcolor: '#FFB340',
              boxShadow: '0 6px 32px rgba(255, 157, 0, 0.45)',
            },
          }}
        >
          Open HF Agent
        </Button>
      ) : !isAuthenticated && !isDevUser ? (
        // Direct access + not logged in → sign in button
        <Button
          variant="contained"
          size="large"
          onClick={() => triggerLogin()}
          sx={{
            px: 5,
            py: 1.5,
            fontSize: '1rem',
            fontWeight: 700,
            textTransform: 'none',
            borderRadius: '12px',
            bgcolor: HF_ORANGE,
            color: '#000',
            boxShadow: '0 4px 24px rgba(255, 157, 0, 0.3)',
            '&:hover': {
              bgcolor: '#FFB340',
              boxShadow: '0 6px 32px rgba(255, 157, 0, 0.45)',
            },
          }}
        >
          Sign in with Hugging Face
        </Button>
      ) : (
        // Authenticated or dev → start session
        <Button
          variant="contained"
          size="large"
          onClick={handleStart}
          disabled={isCreating}
          startIcon={
            isCreating ? <CircularProgress size={20} color="inherit" /> : null
          }
          sx={{
            px: 5,
            py: 1.5,
            fontSize: '1rem',
            fontWeight: 700,
            textTransform: 'none',
            borderRadius: '12px',
            bgcolor: HF_ORANGE,
            color: '#000',
            boxShadow: '0 4px 24px rgba(255, 157, 0, 0.3)',
            '&:hover': {
              bgcolor: '#FFB340',
              boxShadow: '0 6px 32px rgba(255, 157, 0, 0.45)',
            },
            '&.Mui-disabled': {
              bgcolor: 'rgba(255, 157, 0, 0.35)',
              color: 'rgba(0,0,0,0.45)',
            },
          }}
        >
          {isCreating ? 'Initializing...' : 'Start Session'}
        </Button>
      )}

      {/* Error */}
      {error && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setError(null)}
          sx={{
            mt: 3,
            maxWidth: 400,
            fontSize: '0.8rem',
            borderColor: HF_ORANGE,
            color: 'var(--text)',
          }}
        >
          {error}
        </Alert>
      )}

      {/* Footnote */}
      <Typography
        variant="caption"
        sx={{ mt: 5, color: 'var(--muted-text)', opacity: 0.5, fontSize: '0.7rem' }}
      >
        Conversations are stored locally in your browser.
      </Typography>
    </Box>
  );
}
