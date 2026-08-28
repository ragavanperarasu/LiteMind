import {
  AppBar,
  Box,
  Button,
  IconButton,
  Stack,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import GitHubIcon from '@mui/icons-material/GitHub';
import { routes } from '../lib/router.js';
import { site } from '../lib/site.js';

/** The mark from the favicon, so the tab and the header agree. */
function Logo() {
  return (
    <Box component="svg" viewBox="0 0 32 32" sx={{ width: 28, height: 28, flexShrink: 0 }}>
      <circle cx="9" cy="16" r="3" fill="currentColor" />
      <circle cx="22" cy="9" r="2.4" fill="#e0af68" />
      <circle cx="23" cy="17" r="2.4" fill="#bb9af7" />
      <circle cx="21" cy="24" r="2.4" fill="currentColor" opacity="0.3" />
      <g stroke="currentColor" strokeWidth="1.2" opacity="0.6">
        <path d="M11.6 14.8 19.7 9.6M12 16h8.6M11.7 17.4 19 23" />
      </g>
    </Box>
  );
}

export default function TopBar({ mode, onToggleMode, onOpenSearch, onOpenNav }) {
  const theme = useTheme();
  const compact = useMediaQuery(theme.breakpoints.down('md'));
  const isApple = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);

  return (
    <AppBar
      position="fixed"
      elevation={0}
      color="transparent"
      sx={{
        borderBottom: 1,
        borderColor: 'divider',
        backdropFilter: 'blur(12px)',
        bgcolor: (t) =>
          t.palette.mode === 'dark' ? 'rgba(22,22,30,0.82)' : 'rgba(255,255,255,0.86)',
        zIndex: (t) => t.zIndex.drawer + 1,
      }}
    >
      <Toolbar sx={{ gap: 1, minHeight: { xs: 60, md: 64 } }}>
        {compact && (
          <IconButton edge="start" onClick={onOpenNav} aria-label="Open navigation">
            <MenuRoundedIcon />
          </IconButton>
        )}

        <Stack
          component="a"
          href={routes.home}
          direction="row"
          alignItems="center"
          spacing={1.25}
          sx={{ textDecoration: 'none', color: 'primary.main', mr: 2 }}
        >
          <Logo />
          <Box sx={{ lineHeight: 1.1 }}>
            <Typography sx={{ fontWeight: 700, fontSize: '1.05rem', color: 'text.primary' }}>
              {site.name}
            </Typography>
            {!compact && (
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {site.tagline}
              </Typography>
            )}
          </Box>
        </Stack>

        <Box sx={{ flex: 1 }} />

        <Button
          onClick={onOpenSearch}
          startIcon={<SearchRoundedIcon />}
          sx={{
            color: 'text.secondary',
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
            px: 1.5,
            minWidth: { xs: 40, sm: 210 },
            justifyContent: 'flex-start',
            fontWeight: 500,
          }}
        >
          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>Search</Box>
          <Box sx={{ flex: 1 }} />
          <Box
            sx={{
              display: { xs: 'none', sm: 'block' },
              fontSize: '0.72rem',
              px: 0.75,
              py: '1px',
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
            }}
          >
            {isApple ? '⌘K' : 'Ctrl K'}
          </Box>
        </Button>

        <Tooltip title={mode === 'dark' ? 'Switch to light' : 'Switch to dark'}>
          <IconButton onClick={onToggleMode} aria-label="Toggle colour scheme">
            {mode === 'dark' ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}
          </IconButton>
        </Tooltip>

        <Tooltip title="Source on GitHub">
          <IconButton
            component="a"
            href={site.repository}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Source on GitHub"
          >
            <GitHubIcon />
          </IconButton>
        </Tooltip>
      </Toolbar>
    </AppBar>
  );
}
