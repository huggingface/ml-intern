import { useEffect, useMemo, useState } from 'react';
import { Alert, AlertTitle, Box, Button, Link, Typography } from '@mui/material';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { inferenceCreditCta, isInferenceCreditError } from '@/utils/inferenceBilling';

interface ChatErrorBannerProps {
  error: string;
  sessionId: string;
  model?: string | null;
  onDismiss: () => void;
}

const DISCUSSIONS_URL = 'https://huggingface.co/spaces/smolagents/ml-intern/discussions';

export default function ChatErrorBanner({ error, sessionId, model, onDismiss }: ChatErrorBannerProps) {
  const [copied, setCopied] = useState(false);
  const [reportedAt, setReportedAt] = useState(() => new Date().toISOString());
  const userPlan = useAgentStore((s) => s.user?.plan);
  const isCreditError = isInferenceCreditError(error);
  const creditCta = isCreditError ? inferenceCreditCta(userPlan) : null;

  useEffect(() => {
    setReportedAt(new Date().toISOString());
    setCopied(false);
  }, [error]);

  const details = useMemo(
    () => [
      'ML Intern message failure',
      `time: ${reportedAt}`,
      `session: ${sessionId}`,
      `model: ${model || 'unknown'}`,
      `error: ${error}`,
    ].join('\n'),
    [error, model, reportedAt, sessionId],
  );

  const copyDetails = async () => {
    try {
      await navigator.clipboard.writeText(details);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const trackProClick = () => {
    if (userPlan === 'pro') return;
    void apiFetch(`/api/pro-click/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify({ source: 'inference_credit_error', target: 'hf_pro' }),
    }).catch(() => {});
  };

  return (
    <Box sx={{ maxWidth: 880, mx: 'auto', width: '100%', px: { xs: 0, sm: 1, md: 2 }, mb: 1 }}>
      <Alert
        severity="error"
        variant="outlined"
        onClose={onDismiss}
        sx={{
          bgcolor: 'rgba(224, 90, 79, 0.08)',
          borderColor: 'rgba(224, 90, 79, 0.4)',
          color: 'var(--text)',
          '& .MuiAlert-icon': { color: 'var(--accent-red)' },
        }}
        action={
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <Button color="inherit" size="small" onClick={copyDetails} sx={{ textTransform: 'none' }}>
              {copied ? 'Copied' : 'Copy details'}
            </Button>
            <Button color="inherit" size="small" onClick={onDismiss} sx={{ textTransform: 'none' }}>
              Dismiss
            </Button>
          </Box>
        }
      >
        <AlertTitle sx={{ fontWeight: 700, fontSize: '0.86rem' }}>
          {creditCta?.title ?? 'Message failed'}
        </AlertTitle>
        <Typography variant="body2" sx={{ fontSize: '0.8rem', lineHeight: 1.5 }}>
          {creditCta ? (
            creditCta.message
          ) : (
            <>
              The backend could not process the last message. Retry after a moment. If it keeps
              happening,{' '}
              <Link
                href={DISCUSSIONS_URL}
                target="_blank"
                rel="noopener noreferrer"
                color="inherit"
                underline="always"
              >
                open a discussion
              </Link>{' '}
              with the copied details.
            </>
          )}
        </Typography>
        {creditCta && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
            <Button
              component="a"
              href={creditCta.primaryHref}
              target="_blank"
              rel="noopener noreferrer"
              color="inherit"
              size="small"
              variant="outlined"
              onClick={trackProClick}
              sx={{ textTransform: 'none' }}
            >
              {creditCta.primaryLabel}
            </Button>
            {creditCta.secondaryHref && creditCta.secondaryLabel && (
              <Button
                component="a"
                href={creditCta.secondaryHref}
                target="_blank"
                rel="noopener noreferrer"
                color="inherit"
                size="small"
                sx={{ textTransform: 'none' }}
              >
                {creditCta.secondaryLabel}
              </Button>
            )}
          </Box>
        )}
        <Typography
          variant="caption"
          component="pre"
          sx={{
            display: 'block',
            mt: 1,
            mb: 0,
            p: 1,
            maxHeight: 96,
            overflow: 'auto',
            bgcolor: 'var(--code-bg)',
            borderRadius: '6px',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: '0.72rem',
          }}
        >
          {error}
        </Typography>
      </Alert>
    </Box>
  );
}
