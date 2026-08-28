import { createTheme } from '@mui/material/styles';

/**
 * The documentation site borrows the engine UI's Tokyo Night palette so the
 * two read as one project, and adds a light mode because documentation gets
 * read in daylight and printed.
 */
const shared = {
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif',
    h1: { fontWeight: 700, letterSpacing: '-0.03em' },
    h2: { fontWeight: 700, letterSpacing: '-0.02em' },
    h3: { fontWeight: 600, letterSpacing: '-0.02em' },
    h6: { fontWeight: 600, letterSpacing: '-0.01em' },
    overline: { letterSpacing: '0.12em', fontWeight: 600 },
    button: { textTransform: 'none', fontWeight: 600 },
  },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiCssBaseline: {
      styleOverrides: {
        html: { scrollBehavior: 'smooth' },
        // Anchored headings must clear the fixed app bar when jumped to.
        ':target': { scrollMarginTop: '84px' },
      },
    },
  },
};

const palettes = {
  dark: {
    mode: 'dark',
    primary: { main: '#7aa2f7' },
    secondary: { main: '#bb9af7' },
    success: { main: '#9ece6a' },
    warning: { main: '#e0af68' },
    error: { main: '#f7768e' },
    background: { default: '#16161e', paper: '#1a1b26' },
    divider: 'rgba(255,255,255,0.09)',
    text: { primary: '#c8d3f5', secondary: '#8d99c4' },
  },
  light: {
    mode: 'light',
    primary: { main: '#2e5fd0' },
    secondary: { main: '#7c4dcc' },
    success: { main: '#3f7d1f' },
    warning: { main: '#9a6700' },
    error: { main: '#c0384f' },
    background: { default: '#f6f7fb', paper: '#ffffff' },
    divider: 'rgba(16,20,40,0.12)',
    text: { primary: '#1b2033', secondary: '#5a628a' },
  },
};

/** Colours the markdown renderer needs but MUI has no slot for. */
export const codeTokens = {
  dark: {
    surface: '#11121b',
    border: 'rgba(255,255,255,0.09)',
    plain: '#a9b1d6',
    keyword: '#bb9af7',
    string: '#9ece6a',
    number: '#ff9e64',
    comment: '#565f89',
    title: '#7aa2f7',
    attr: '#e0af68',
    meta: '#7dcfff',
  },
  light: {
    surface: '#f2f4fa',
    border: 'rgba(16,20,40,0.10)',
    plain: '#2b3050',
    keyword: '#7c3fbf',
    string: '#2f7a2f',
    number: '#b1560f',
    comment: '#767e9e',
    title: '#2255c0',
    attr: '#8a5a00',
    meta: '#0f6f93',
  },
};

export function buildTheme(mode) {
  return createTheme({ ...shared, palette: palettes[mode] || palettes.dark });
}

export default buildTheme;
