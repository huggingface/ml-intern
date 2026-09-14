import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { Box, Button, Paper, Typography } from '@mui/material';

const HUGGINGCHAT_URL = 'https://huggingface.co/chat/';

export default function RetirementPage() {
  return (
    <Box
      component="main"
      sx={{
        width: '100%',
        minHeight: '100dvh',
        display: 'grid',
        placeItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        px: { xs: 2, sm: 4 },
        py: { xs: 4, sm: 6 },
        background: 'var(--body-gradient)',
        '&::before': {
          content: '""',
          position: 'absolute',
          width: { xs: 280, sm: 480 },
          height: { xs: 280, sm: 480 },
          borderRadius: '50%',
          background: 'rgba(255, 157, 0, 0.12)',
          filter: 'blur(80px)',
          transform: 'translate(-45%, -45%)',
          pointerEvents: 'none',
        },
      }}
    >
      <Paper
        elevation={0}
        sx={{
          width: '100%',
          maxWidth: 680,
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          px: { xs: 3, sm: 7 },
          py: { xs: 5, sm: 7 },
          textAlign: 'center',
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: { xs: 3, sm: 4 },
          boxShadow: 'var(--shadow-1)',
        }}
      >
        <Box
          component="img"
          src="/smolagents.webp"
          alt="ML Intern"
          sx={{ width: { xs: 72, sm: 88 }, height: { xs: 72, sm: 88 }, mb: 3 }}
        />

        <Typography
          variant="overline"
          sx={{
            mb: 1.5,
            color: 'primary.main',
            fontSize: '0.75rem',
            fontWeight: 800,
            letterSpacing: '0.16em',
          }}
        >
          ML Intern
        </Typography>

        <Typography
          component="h1"
          sx={{
            maxWidth: 560,
            color: 'text.primary',
            fontSize: { xs: '2rem', sm: '3rem' },
            fontWeight: 800,
            letterSpacing: '-0.035em',
            lineHeight: 1.08,
          }}
        >
          ML Intern has moved to HuggingChat
        </Typography>

        <Typography
          variant="body1"
          sx={{
            maxWidth: 440,
            mt: 2.5,
            color: 'text.secondary',
            fontSize: { xs: '1rem', sm: '1.1rem' },
            lineHeight: 1.7,
          }}
        >
          Continue with ML Intern in HuggingChat.
        </Typography>

        <Button
          component="a"
          href={HUGGINGCHAT_URL}
          target="_blank"
          rel="noopener noreferrer"
          variant="contained"
          size="large"
          endIcon={<OpenInNewIcon />}
          sx={{
            mt: 4,
            px: { xs: 3, sm: 4 },
            py: 1.4,
            color: 'primary.contrastText',
            fontSize: '1rem',
            boxShadow: '0 8px 28px rgba(255, 157, 0, 0.28)',
          }}
        >
          Open ML Intern in HuggingChat
        </Button>
      </Paper>
    </Box>
  );
}
