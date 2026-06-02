/**
 * Shown above the composer when the active session's premium usage is being
 * billed to the user's own HF account — i.e. they're past the subsidized daily
 * allowance and the session flipped to `premium_user_billed`. Dismissible once
 * per day so we stay transparent about billing without nagging every session.
 */
import { useState } from 'react';
import { Box, Link, Typography } from '@mui/material';

const DISMISS_KEY = 'ml-intern:billing-banner-dismissed';
const today = () => new Date().toISOString().slice(0, 10);

export default function BillingBanner() {
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(DISMISS_KEY) === today();
    } catch {
      return false;
    }
  });

  if (dismissed) return null;

  const dismiss = () => {
    try {
      localStorage.setItem(DISMISS_KEY, today());
    } catch {
      /* ignore storage failures */
    }
    setDismissed(true);
  };

  return (
    <Box sx={{ maxWidth: '880px', mx: 'auto', width: '100%', px: { xs: 0, sm: 1, md: 2 }, mb: 1 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          p: '8px 12px',
          borderRadius: 'var(--radius-md)',
          bgcolor: 'var(--accent-yellow-weak)',
          border: '1px solid var(--border)',
        }}
      >
        <Typography
          variant="caption"
          sx={{ flex: 1, color: 'var(--text)', fontSize: '0.78rem', lineHeight: 1.5 }}
        >
          You've used today's subsidized premium sessions — this session's usage is billed to your{' '}
          <Link
            href="https://huggingface.co/settings/billing"
            target="_blank"
            rel="noopener noreferrer"
            sx={{ color: 'inherit', textDecoration: 'underline' }}
          >
            Hugging Face account
          </Link>{' '}
          through Hugging Face Inference Providers.
        </Typography>
        <Box
          component="button"
          onClick={dismiss}
          aria-label="Dismiss billing notice"
          sx={{
            border: 'none',
            background: 'none',
            cursor: 'pointer',
            color: 'var(--muted-text)',
            fontSize: '0.95rem',
            lineHeight: 1,
            p: 0.5,
            '&:hover': { color: 'var(--text)' },
          }}
        >
          ✕
        </Box>
      </Box>
    </Box>
  );
}
