import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Divider,
  Link,
  Popover,
  Tooltip,
  Typography,
} from '@mui/material';
import PaidOutlinedIcon from '@mui/icons-material/PaidOutlined';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { useSessionStore } from '@/store/sessionStore';
import { type UsageBucket, useUsageStore } from '@/store/usageStore';

function formatUsd(value: number | undefined, compact = false): string {
  const amount = value ?? 0;
  if (amount > 0 && amount < 0.01) return '<$0.01';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: compact || amount >= 1 ? 2 : 4,
  }).format(amount);
}

function formatCount(value: number | undefined): string {
  return new Intl.NumberFormat('en-US').format(value ?? 0);
}

function UsageSection({ title, bucket }: { title: string; bucket: UsageBucket | null }) {
  return (
    <Box sx={{ py: 1 }}>
      <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700 }}>
        {title}
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr auto', columnGap: 2, rowGap: 0.5, mt: 0.75 }}>
        <Typography variant="body2" color="text.secondary">Total</Typography>
        <Typography variant="body2" sx={{ fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
          {formatUsd(bucket?.total_usd)}
        </Typography>
        <Typography variant="body2" color="text.secondary">Inference Providers</Typography>
        <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>
          {formatUsd(bucket?.inference_usd)}
        </Typography>
        <Typography variant="body2" color="text.secondary">HF Jobs estimated</Typography>
        <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>
          {formatUsd(bucket?.hf_jobs_estimated_usd)}
        </Typography>
        <Typography variant="body2" color="text.secondary">Calls / jobs</Typography>
        <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>
          {formatCount(bucket?.llm_calls)} / {formatCount(bucket?.hf_jobs_count)}
        </Typography>
        <Typography variant="body2" color="text.secondary">Tokens</Typography>
        <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>
          {formatCount(bucket?.total_tokens)}
        </Typography>
      </Box>
    </Box>
  );
}

export default function UsageMeter() {
  const activeSessionId = useSessionStore((state) => state.activeSessionId);
  const { usage, isLoading, error, fetchUsage } = useUsageStore();
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

  useEffect(() => {
    void fetchUsage(activeSessionId);
  }, [activeSessionId, fetchUsage]);

  const sessionTotal = usage?.session?.total_usd ?? 0;
  const links = useMemo(() => usage?.links ?? {}, [usage?.links]);
  const open = Boolean(anchorEl);

  return (
    <>
      <Tooltip title="Usage">
        <Button
          size="small"
          variant="outlined"
          startIcon={isLoading ? <CircularProgress size={14} /> : <PaidOutlinedIcon fontSize="small" />}
          onClick={(event) => setAnchorEl(event.currentTarget)}
          sx={{
            minWidth: { xs: 58, sm: 84 },
            height: 32,
            px: { xs: 0.75, sm: 1 },
            borderColor: 'divider',
            color: 'text.secondary',
            fontVariantNumeric: 'tabular-nums',
            '& .MuiButton-startIcon': { mr: { xs: 0.25, sm: 0.5 } },
            '&:hover': { borderColor: 'primary.main', color: 'primary.main' },
          }}
        >
          {formatUsd(sessionTotal, true)}
        </Button>
      </Tooltip>
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: {
              width: 320,
              maxWidth: 'calc(100vw - 24px)',
              p: 2,
              border: '1px solid',
              borderColor: 'divider',
            },
          },
        }}
      >
        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
          Usage
        </Typography>
        <Typography variant="caption" color="text.secondary">
          ML Intern-attributed spend, not your full HF invoice.
        </Typography>

        {error ? (
          <Typography variant="body2" color="error" sx={{ mt: 1.5 }}>
            {error}
          </Typography>
        ) : (
          <>
            <UsageSection title="Current session" bucket={usage?.session ?? null} />
            <Divider />
            <UsageSection title="Today" bucket={usage?.today ?? null} />
            <Divider />
            <UsageSection title="This month" bucket={usage?.month ?? null} />
          </>
        )}

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, pt: 1 }}>
          {links.hf_billing && (
            <Link href={links.hf_billing} target="_blank" rel="noopener noreferrer" underline="hover" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, fontSize: '0.75rem' }}>
              HF billing <OpenInNewIcon sx={{ fontSize: 12 }} />
            </Link>
          )}
          {links.inference_providers_usage && (
            <Link href={links.inference_providers_usage} target="_blank" rel="noopener noreferrer" underline="hover" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, fontSize: '0.75rem' }}>
              Inference usage <OpenInNewIcon sx={{ fontSize: 12 }} />
            </Link>
          )}
          {links.jobs_pricing && (
            <Link href={links.jobs_pricing} target="_blank" rel="noopener noreferrer" underline="hover" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, fontSize: '0.75rem' }}>
              Jobs pricing <OpenInNewIcon sx={{ fontSize: 12 }} />
            </Link>
          )}
        </Box>
      </Popover>
    </>
  );
}
