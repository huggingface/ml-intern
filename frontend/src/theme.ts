import { createTheme, type ThemeOptions } from '@mui/material/styles';

// ── Shared tokens ────────────────────────────────────────────────
const sharedTypography: ThemeOptions['typography'] = {
  fontFamily: '"Manrope", "Segoe UI", sans-serif',
  fontSize: 15.5,
  h1: { fontFamily: '"Sora", "Manrope", sans-serif', fontWeight: 700, letterSpacing: '-0.03em' },
  h2: { fontFamily: '"Sora", "Manrope", sans-serif', fontWeight: 700, letterSpacing: '-0.025em' },
  h3: { fontFamily: '"Sora", "Manrope", sans-serif', fontWeight: 650, letterSpacing: '-0.02em' },
  h4: { fontFamily: '"Sora", "Manrope", sans-serif', fontWeight: 650, letterSpacing: '-0.015em' },
  h5: { fontFamily: '"Sora", "Manrope", sans-serif', fontWeight: 600 },
  h6: { fontFamily: '"Sora", "Manrope", sans-serif', fontWeight: 600 },
  body1: { lineHeight: 1.65 },
  body2: { lineHeight: 1.6 },
  button: {
    fontFamily: '"Manrope", "Segoe UI", sans-serif',
    textTransform: 'none' as const,
    fontWeight: 600,
    letterSpacing: '0.01em',
  },
};

const sharedComponents: ThemeOptions['components'] = {
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: '10px',
        fontWeight: 600,
        transition: 'transform 0.16s cubic-bezier(0.22, 1, 0.36, 1), background 0.18s ease, box-shadow 0.18s ease',
        '&:hover': { transform: 'translateY(-1px) scale(1.01)' },
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: { backgroundImage: 'none' },
    },
  },
};

const sharedShape: ThemeOptions['shape'] = { borderRadius: 12 };

// ── Dark palette ─────────────────────────────────────────────────
const darkVars = {
  '--bg': 'oklch(0.15 0.01 80)',
  '--panel': 'oklch(0.19 0.012 80)',
  '--surface': 'oklch(0.22 0.014 80)',
  '--text': 'oklch(0.92 0.01 85)',
  '--muted-text': 'oklch(0.72 0.012 82)',
  '--accent-yellow': 'oklch(0.76 0.17 72)',
  '--accent-yellow-weak': 'color-mix(in oklch, oklch(0.76 0.17 72) 15%, transparent)',
  '--accent-green': 'oklch(0.74 0.12 154)',
  '--accent-red': 'oklch(0.66 0.17 28)',
  '--shadow-1': '0 6px 18px rgba(2,6,12,0.55)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(255,157,0,0.12)',
  '--border': 'rgba(255,255,255,0.07)',
  '--border-hover': 'rgba(255,255,255,0.18)',
  '--code-bg': 'rgba(12,12,14,0.68)',
  '--tool-bg': 'rgba(12,12,14,0.48)',
  '--tool-border': 'rgba(255,255,255,0.08)',
  '--hover-bg': 'rgba(255,255,255,0.07)',
  '--composer-bg': 'rgba(255,255,255,0.02)',
  '--msg-gradient': 'linear-gradient(180deg, rgba(255,255,255,0.03), transparent)',
  '--body-gradient': 'radial-gradient(1000px 520px at 50% -10%, rgba(255,157,0,0.10), transparent 55%), linear-gradient(180deg, oklch(0.15 0.01 80), oklch(0.12 0.008 80))',
  '--scrollbar-thumb': 'oklch(0.35 0.01 80)',
  '--success-icon': '#FDB022',
  '--error-icon': '#F87171',
  '--clickable-text': 'rgba(255, 255, 255, 0.9)',
  '--clickable-underline': 'rgba(255,255,255,0.3)',
  '--code-panel-bg': 'oklch(0.14 0.008 80)',
  '--tab-active-bg': 'rgba(255,255,255,0.08)',
  '--tab-active-border': 'rgba(255,255,255,0.1)',
  '--tab-hover-bg': 'rgba(255,255,255,0.05)',
  '--tab-close-hover': 'rgba(255,255,255,0.1)',
  '--plan-bg': 'rgba(0,0,0,0.2)',
} as const;

// ── Light palette ────────────────────────────────────────────────
const lightVars = {
  '--bg': 'oklch(0.985 0.004 80)',
  '--panel': 'oklch(0.962 0.006 80)',
  '--surface': 'oklch(0.94 0.008 80)',
  '--text': 'oklch(0.23 0.02 80)',
  '--muted-text': 'oklch(0.48 0.015 80)',
  '--accent-yellow': 'oklch(0.72 0.16 72)',
  '--accent-yellow-weak': 'color-mix(in oklch, oklch(0.72 0.16 72) 14%, transparent)',
  '--accent-green': 'oklch(0.66 0.14 154)',
  '--accent-red': 'oklch(0.62 0.2 28)',
  '--shadow-1': '0 4px 12px rgba(0,0,0,0.08)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(255,157,0,0.15)',
  '--border': 'rgba(25,25,28,0.10)',
  '--border-hover': 'rgba(25,25,28,0.22)',
  '--code-bg': 'rgba(0,0,0,0.04)',
  '--tool-bg': 'rgba(0,0,0,0.03)',
  '--tool-border': 'rgba(0,0,0,0.08)',
  '--hover-bg': 'rgba(18,18,20,0.05)',
  '--composer-bg': 'rgba(18,18,20,0.03)',
  '--msg-gradient': 'linear-gradient(180deg, rgba(25,25,25,0.03), transparent)',
  '--body-gradient': 'radial-gradient(1100px 560px at 50% -15%, rgba(255,157,0,0.11), transparent 56%), linear-gradient(180deg, oklch(0.985 0.004 80), oklch(0.95 0.006 80))',
  '--scrollbar-thumb': 'oklch(0.79 0.01 80)',
  '--success-icon': '#FF9D00',
  '--error-icon': '#DC2626',
  '--clickable-text': 'rgba(0, 0, 0, 0.85)',
  '--clickable-underline': 'rgba(0,0,0,0.25)',
  '--code-panel-bg': 'oklch(0.94 0.006 80)',
  '--tab-active-bg': 'rgba(0,0,0,0.06)',
  '--tab-active-border': 'rgba(0,0,0,0.1)',
  '--tab-hover-bg': 'rgba(0,0,0,0.04)',
  '--tab-close-hover': 'rgba(0,0,0,0.08)',
  '--plan-bg': 'rgba(0,0,0,0.03)',
} as const;

// ── Shared CSS baseline (scrollbar, code, brand-logo) ────────────
function makeCssBaseline(vars: Record<string, string>) {
  return {
    styleOverrides: {
      ':root': vars,
      body: {
        background: 'var(--body-gradient)',
        color: 'var(--text)',
        fontFeatureSettings: '"calt" 1, "ss01" 1',
        scrollbarWidth: 'thin' as const,
        '&::-webkit-scrollbar': { width: '8px', height: '8px' },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: 'var(--scrollbar-thumb)',
          borderRadius: '2px',
        },
        '&::-webkit-scrollbar-track': { backgroundColor: 'transparent' },
      },
      'code, pre': {
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace',
      },
      '.brand-logo': {
        position: 'relative' as const,
        padding: '6px',
        borderRadius: '8px',
        '&::after': {
          content: '""',
          position: 'absolute' as const,
          inset: '-6px',
          borderRadius: '10px',
          background: 'var(--accent-yellow-weak)',
          zIndex: -1,
          pointerEvents: 'none' as const,
        },
      },
    },
  };
}

function makeDrawer() {
  return {
    styleOverrides: {
      paper: {
        backgroundColor: 'var(--panel)',
        borderRight: '1px solid var(--border)',
      },
    },
  };
}

function makeTextField() {
  return {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 'var(--radius-md)',
          '& fieldset': { borderColor: 'var(--border)' },
          '&:hover fieldset': { borderColor: 'var(--border-hover)' },
          '&.Mui-focused fieldset': {
            borderColor: 'var(--accent-yellow)',
            borderWidth: '1px',
            boxShadow: 'var(--focus)',
          },
        },
      },
    },
  };
}

// ── Theme builders ───────────────────────────────────────────────
export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#FF9D00', light: '#FFB740', dark: '#E08C00', contrastText: '#161616' },
    secondary: { main: '#FF9D00' },
    background: { default: '#121315', paper: '#17191c' },
    text: { primary: '#E8EBEF', secondary: '#AFB6BF' },
    divider: 'rgba(255,255,255,0.08)',
    success: { main: '#2FCC71' },
    error: { main: '#E05A4F' },
    warning: { main: '#FF9D00' },
    info: { main: '#58A6FF' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(darkVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#E68B00', light: '#F2A21B', dark: '#B66A00', contrastText: '#1E1E1E' },
    secondary: { main: '#E08C00' },
    background: { default: '#FDFCF9', paper: '#F6F3ED' },
    text: { primary: '#262320', secondary: '#65605A' },
    divider: 'rgba(18,18,18,0.10)',
    success: { main: '#16A34A' },
    error: { main: '#DC2626' },
    warning: { main: '#FF9D00' },
    info: { main: '#2563EB' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(lightVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

// Keep default export for backwards compat
export default darkTheme;
