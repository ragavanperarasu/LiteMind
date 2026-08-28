import { createTheme } from '@mui/material/styles';

/** Dark by default: this is a tool that runs long jobs and gets stared at. */
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#7aa2f7' },
    secondary: { main: '#bb9af7' },
    success: { main: '#9ece6a' },
    warning: { main: '#e0af68' },
    background: { default: '#16161e', paper: '#1a1b26' },
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif',
    h6: { fontWeight: 600, letterSpacing: '-0.01em' },
    overline: { letterSpacing: '0.12em', fontWeight: 600 },
  },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiCard: {
      styleOverrides: {
        root: { border: '1px solid rgba(255,255,255,0.08)' },
      },
    },
  },
});

export default theme;
