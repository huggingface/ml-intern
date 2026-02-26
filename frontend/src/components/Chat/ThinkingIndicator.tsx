import { Box, Typography } from '@mui/material';

/** Pulsing dots shown while the agent is processing. */
export default function ThinkingIndicator() {
  return (
    <Box sx={{ pt: 0.75 }}>
      <Typography
        variant="caption"
        sx={{
          fontWeight: 700,
          fontSize: '0.72rem',
          color: 'var(--muted-text)',
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          display: 'flex',
          alignItems: 'center',
          gap: 0.75,
        }}
      >
        Thinking
        <Box
          component="span"
          sx={{
            display: 'inline-flex',
            gap: '3px',
            '& span': {
              width: 4,
              height: 4,
              borderRadius: '50%',
              bgcolor: 'primary.main',
              animation: 'dotPulse 1.4s ease-in-out infinite',
            },
            '& span:nth-of-type(2)': { animationDelay: '0.2s' },
            '& span:nth-of-type(3)': { animationDelay: '0.4s' },
            '@keyframes dotPulse': {
              '0%, 80%, 100%': { opacity: 0.25, transform: 'scale(0.8)' },
              '40%': { opacity: 1, transform: 'scale(1)' },
            },
          }}
        >
          <span />
          <span />
          <span />
        </Box>
      </Typography>
    </Box>
  );
}
