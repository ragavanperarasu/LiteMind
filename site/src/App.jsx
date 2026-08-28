import { useCallback, useEffect, useMemo, useState } from 'react';
import { Box, Container, CssBaseline, Drawer, Toolbar } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { buildTheme } from './theme.js';
import MarkdownStyles from './components/MarkdownStyles.jsx';
import TopBar from './components/TopBar.jsx';
import Sidebar from './components/Sidebar.jsx';
import Footer from './components/Footer.jsx';
import SearchDialog from './components/SearchDialog.jsx';
import Home from './views/Home.jsx';
import DocsIndex from './views/DocsIndex.jsx';
import DocPage from './views/DocPage.jsx';
import ModelSpec from './views/ModelSpec.jsx';
import Developer from './views/Developer.jsx';
import NotFound from './views/NotFound.jsx';
import { docsById } from './lib/content.js';
import { useRoute } from './lib/router.js';

const DRAWER_WIDTH = 288;
const MODE_KEY = 'litemind-docs-theme';

function initialMode() {
  const stored = window.localStorage.getItem(MODE_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export default function App() {
  const route = useRoute();
  const [mode, setMode] = useState(initialMode);
  const [navOpen, setNavOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const theme = useMemo(() => buildTheme(mode), [mode]);

  const toggleMode = useCallback(() => {
    setMode((current) => {
      const next = current === 'dark' ? 'light' : 'dark';
      window.localStorage.setItem(MODE_KEY, next);
      return next;
    });
  }, []);

  // Ctrl+K / Cmd+K, and / when the reader is not already typing.
  useEffect(() => {
    const onKeyDown = (event) => {
      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && key === 'k') {
        event.preventDefault();
        setSearchOpen(true);
      } else if (
        key === '/' &&
        !event.metaKey &&
        !event.ctrlKey &&
        !/^(input|textarea)$/i.test(document.activeElement?.tagName || '')
      ) {
        event.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // A route change on a phone means the reader picked something; get out of
  // the way.
  useEffect(() => setNavOpen(false), [route.view, route.id]);

  useEffect(() => {
    const doc = route.view === 'doc' ? docsById.get(route.id) : null;
    document.title = doc
      ? `${doc.rawTitle} — LiteMind`
      : route.view === 'model'
        ? 'Model specification — LiteMind'
        : route.view === 'developer'
          ? 'Developer — LiteMind'
          : 'LiteMind — mixture-of-experts inference on the CPU';
  }, [route]);

  const content = (() => {
    if (route.view === 'home') return <Home />;
    if (route.view === 'index') return <DocsIndex />;
    if (route.view === 'model') return <ModelSpec />;
    if (route.view === 'developer') return <Developer />;
    if (route.view === 'doc') {
      const doc = docsById.get(route.id);
      return doc ? (
        <DocPage doc={doc} anchor={route.anchor} />
      ) : (
        <NotFound path={`docs/${route.id}`} />
      );
    }
    return <NotFound path={route.path} />;
  })();

  const drawer = <Sidebar route={route} onNavigate={() => setNavOpen(false)} />;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <MarkdownStyles />

      <TopBar
        mode={mode}
        onToggleMode={toggleMode}
        onOpenSearch={() => setSearchOpen(true)}
        onOpenNav={() => setNavOpen(true)}
      />

      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
        aria-label="Documentation"
      >
        <Drawer
          variant="temporary"
          open={navOpen}
          onClose={() => setNavOpen(false)}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { width: DRAWER_WIDTH, boxSizing: 'border-box' },
          }}
        >
          <Toolbar />
          {drawer}
        </Drawer>

        <Drawer
          variant="permanent"
          open
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
              borderRight: 1,
              borderColor: 'divider',
              bgcolor: 'background.default',
            },
          }}
        >
          <Toolbar />
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          ml: { md: `${DRAWER_WIDTH}px` },
          minHeight: '100vh',
          bgcolor: 'background.default',
        }}
      >
        <Toolbar />
        <Container maxWidth="lg" sx={{ py: { xs: 4, md: 6 } }}>
          {content}
          <Footer />
        </Container>
      </Box>

      <SearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} />
    </ThemeProvider>
  );
}
