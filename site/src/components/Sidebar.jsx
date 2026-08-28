import {
  Box,
  Chip,
  Divider,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material';
import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import GridViewRoundedIcon from '@mui/icons-material/GridViewRounded';
import HubRoundedIcon from '@mui/icons-material/HubRounded';
import PersonRoundedIcon from '@mui/icons-material/PersonRounded';
import { sections } from '../lib/content.js';
import { routes } from '../lib/router.js';
import { site } from '../lib/site.js';

const shortcuts = [
  { label: 'Home', href: routes.home, icon: HomeRoundedIcon, match: (r) => r.view === 'home' },
  {
    label: 'All pages',
    href: routes.index,
    icon: GridViewRoundedIcon,
    match: (r) => r.view === 'index',
  },
  {
    label: 'Model specification',
    href: routes.model,
    icon: HubRoundedIcon,
    match: (r) => r.view === 'model',
  },
  {
    label: 'Developer',
    href: routes.developer,
    icon: PersonRoundedIcon,
    match: (r) => r.view === 'developer',
  },
];

export default function Sidebar({ route, onNavigate }) {
  return (
    <Box sx={{ py: 2, px: 1.5, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <List dense disablePadding>
        {shortcuts.map(({ label, href, icon: Icon, match }) => {
          const active = match(route);
          return (
            <ListItemButton
              key={href}
              component="a"
              href={href}
              onClick={onNavigate}
              selected={active}
              sx={{ borderRadius: 2, mb: 0.25, gap: 1.25, px: 1.5 }}
            >
              <Icon fontSize="small" sx={{ color: active ? 'primary.main' : 'text.secondary' }} />
              <ListItemText
                primary={label}
                slotProps={{ primary: { fontWeight: active ? 650 : 500, fontSize: '0.9rem' } }}
              />
            </ListItemButton>
          );
        })}
      </List>

      <Divider sx={{ my: 1.5 }} />

      <Box sx={{ flex: 1, overflowY: 'auto', pr: 0.5 }}>
        {sections.map((section) => (
          <Box key={section.label} sx={{ mb: 2 }}>
            <Typography
              variant="overline"
              sx={{ px: 1.5, color: 'text.secondary', fontSize: '0.68rem' }}
            >
              {section.label}
            </Typography>
            <List dense disablePadding sx={{ mt: 0.5 }}>
              {section.pages.map((page) => {
                const active = route.view === 'doc' && route.id === page.id;
                return (
                  <ListItemButton
                    key={page.id}
                    component="a"
                    href={routes.doc(page.id)}
                    onClick={onNavigate}
                    selected={active}
                    sx={{
                      borderRadius: 2,
                      mb: 0.25,
                      px: 1.5,
                      gap: 1.25,
                      borderLeft: '2px solid',
                      borderLeftColor: active ? 'primary.main' : 'transparent',
                    }}
                  >
                    <Chip
                      label={page.number}
                      size="small"
                      sx={{
                        height: 20,
                        minWidth: 28,
                        fontSize: '0.68rem',
                        fontWeight: 600,
                        fontVariantNumeric: 'tabular-nums',
                        bgcolor: active ? 'primary.main' : 'action.hover',
                        color: active ? 'primary.contrastText' : 'text.secondary',
                      }}
                    />
                    <ListItemText
                      primary={page.title}
                      slotProps={{
                        primary: {
                          fontWeight: active ? 650 : 500,
                          fontSize: '0.875rem',
                          color: active ? 'text.primary' : 'text.secondary',
                        },
                      }}
                    />
                  </ListItemButton>
                );
              })}
            </List>
          </Box>
        ))}
      </Box>

      <Divider sx={{ mb: 1.5 }} />
      <Stack spacing={0.25} sx={{ px: 1.5, pb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {site.license} licensed · C++20 · no GPU
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Built against DeepSeek-V2-Lite
        </Typography>
      </Stack>
    </Box>
  );
}
