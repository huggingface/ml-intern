import { type ReactNode, useEffect, useMemo, useState } from 'react';
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
import {
  type HfAccountUsageBucket,
  type HfInferenceProvidersCredits,
  type UsageBucket,
  useUsageStore,
} from '@/store/usageStore';

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

function UsageRow({
  label,
  value,
  strong = false,
}: {
  label: string;
  value: string;
  strong?: boolean;
}) {
  return (
    <>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography
        variant="body2"
        sx={{ fontWeight: strong ? 700 : 400, fontVariantNumeric: 'tabular-nums' }}
      >
        {value}
      </Typography>
    </>
  );
}

function UsageGrid({ children }: { children: ReactNode }) {
  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        columnGap: 2,
        rowGap: 0.5,
        mt: 0.75,
      }}
    >
      {children}
    </Box>
  );
}

function AccountUsageSection({
  title,
  account,
  telemetry,
}: {
  title: string;
  account: HfAccountUsageBucket | null | undefined;
  telemetry: UsageBucket | null | undefined;
}) {
  return (
    <Box sx={{ py: 1 }}>
      <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700 }}>
        {title}
      </Typography>
      <UsageGrid>
        <UsageRow
          label="Inference Providers"
          value={formatUsd(account?.inference_providers_usd ?? telemetry?.inference_usd)}
          strong
        />
        <UsageRow
          label={account ? 'HF Jobs' : 'HF Jobs estimated'}
          value={formatUsd(account?.hf_jobs_usd ?? telemetry?.hf_jobs_estimated_usd)}
        />
      </UsageGrid>
    </Box>
  );
}

function CreditsSection({ credits }: { credits: HfInferenceProvidersCredits | null | undefined }) {
  if (!credits) return null;
  return (
    <>
      <Divider />
      <Box sx={{ py: 1 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700 }}>
          Inference credits
        </Typography>
        <UsageGrid>
          <UsageRow
            label="Included remaining"
            value={formatUsd(credits.remaining_included_usd)}
            strong
          />
          <UsageRow
            label="Used / included"
            value={`${formatUsd(credits.used_usd)} / ${formatUsd(credits.included_usd)}`}
          />
          {credits.limit_usd > 0 && (
            <UsageRow
              label="Spend limit remaining"
              value={formatUsd(credits.remaining_limit_usd)}
            />
          )}
        </UsageGrid>
      </Box>
    </>
  );
}

export default function UsageMeter() {
  const activeSessionId = useSessionStore((state) => state.activeSessionId);
  const { usage, isLoading, error, fetchUsage } = useUsageStore();
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

  useEffect(() => {
    void fetchUsage(activeSessionId);
  }, [activeSessionId, fetchUsage]);

  const sessionTotal =
    usage?.hf_account?.current_session?.total_usd ?? usage?.session?.total_usd ?? 0;
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
              maxHeight: 'calc(100vh - 24px)',
              overflowY: 'auto',
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
          Current session billing and Inference Providers credits.
        </Typography>

        {error ? (
          <Typography variant="body2" color="error" sx={{ mt: 1.5 }}>
            {error}
          </Typography>
        ) : (
          <>
            <AccountUsageSection
              title="Current session"
              account={usage?.hf_account?.current_session ?? null}
              telemetry={usage?.session ?? null}
            />
            <CreditsSection credits={usage?.hf_account?.inference_providers_credits} />
            {usage?.hf_account?.available && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', pt: 0.5 }}>
                Session billing is inferred from HF account usage since session start.
              </Typography>
            )}
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
