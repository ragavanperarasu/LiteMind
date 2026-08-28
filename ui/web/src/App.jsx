import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Stack from '@mui/material/Stack';
import Toolbar from '@mui/material/Toolbar';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import ChatIcon from '@mui/icons-material/ChatBubbleOutline';
import HubIcon from '@mui/icons-material/Hub';
import MemoryIcon from '@mui/icons-material/Memory';
import SettingsIcon from '@mui/icons-material/Settings';
import SpeedIcon from '@mui/icons-material/Speed';

import ChatView from './views/ChatView.jsx';
import ModelView from './views/ModelView.jsx';
import RoutingView from './views/RoutingView.jsx';
import UsageView from './views/UsageView.jsx';
import SettingsDialog from './components/SettingsDialog.jsx';
import { fetchHealth, fetchSettings, fetchUsage, streamGeneration } from './api.js';

const DRAWER_WIDTH = 232;

const SECTIONS = [
  { id: 'chat', label: 'Chat', icon: <ChatIcon />, caption: 'Ask the model' },
  { id: 'model', label: 'Model details', icon: <MemoryIcon />, caption: 'Architecture and cost' },
  { id: 'usage', label: 'Usage', icon: <SpeedIcon />, caption: 'CPU, memory, parameters' },
  { id: 'routing', label: 'Expert routing', icon: <HubIcon />, caption: 'Which experts ran' },
];

const emptyRouting = { counts: null, currentSet: new Set(), tokens: 0, touched: 0 };

export default function App() {
  const [section, setSection] = useState('chat');
  const [health, setHealth] = useState(null);
  const [settings, setSettings] = useState(null);
  const [usage, setUsage] = useState(null);
  const [model, setModel] = useState(null);
  const [memory, setMemory] = useState(null);
  const [plan, setPlan] = useState(null);
  const [routing, setRouting] = useState(emptyRouting);
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const abortRef = useRef(null);
  // Counts are mutated per token and read once per render; keeping the array in
  // a ref avoids rebuilding a 1,664-entry copy for every token that arrives.
  const countsRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch((cause) => setHealth({ ready: false, error: cause.message }));
    fetchSettings().then(setSettings).catch(() => setSettings({}));
  }, []);

  // Poll usage only while it is on screen, or while a prompt is running and the
  // numbers are actually changing.
  useEffect(() => {
    const wanted = section === 'usage' || running;
    if (!wanted) return undefined;
    const tick = () => fetchUsage().then(setUsage).catch(() => {});
    tick();
    const timer = setInterval(tick, 1500);
    return () => clearInterval(timer);
  }, [section, running]);

  const submit = useCallback(async () => {
    const text = prompt.trim();
    if (text === '' || running) return;

    setPrompt('');
    setError(null);
    setRunning(true);
    setPlan(null);
    setRouting(emptyRouting);
    countsRef.current = null;

    setMessages((current) => [
      ...current,
      { role: 'user', text },
      { role: 'assistant', text: '', pending: true },
    ]);

    const updateAnswer = (change) =>
      setMessages((current) => {
        const next = [...current];
        const last = next.length - 1;
        next[last] = { ...next[last], ...change(next[last]) };
        return next;
      });

    const controller = new AbortController();
    abortRef.current = controller;

    // The ready event arrives in this same stream, so the model state variable
    // is still whatever it was when this callback was created - null on the
    // first prompt. The shape of the routing grid comes from the event itself.
    let shape = model;

    try {
      await streamGeneration({
        prompt: text,
        // Routing is only worth its bandwidth when there is somewhere to show it.
        settings: { routing: true },
        signal: controller.signal,
        onEvent: (event) => {
          switch (event.event) {
            case 'ready':
              shape = event;
              setModel(event);
              break;
            case 'memory':
              setMemory(event);
              break;
            case 'plan':
              setPlan(event);
              updateAnswer(() => ({ stage: 'Generating…' }));
              break;
            case 'token':
              updateAnswer((message) => ({ text: message.text + event.text, pending: false }));
              break;
            case 'routing': {
              const total = (shape?.moe_layers ?? 0) * (shape?.routed_experts ?? 0);
              if (!countsRef.current && total > 0) countsRef.current = new Int32Array(total);
              const counts = countsRef.current;
              const perLayer = shape?.routed_experts ?? 0;
              const active = new Set();
              const topK = event.experts.length / (shape?.moe_layers || 1);
              event.experts.forEach((expert, index) => {
                const layer = Math.floor(index / topK);
                const cell = layer * perLayer + expert;
                active.add(cell);
                if (counts && cell < counts.length) counts[cell] += 1;
              });
              setRouting((current) => ({
                counts: counts ? Array.from(counts) : null,
                currentSet: active,
                tokens: current.tokens + 1,
                touched: counts ? counts.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0) : 0,
              }));
              break;
            }
            case 'done':
              updateAnswer((message) => ({
                text: event.text || message.text, pending: false, stats: event,
              }));
              break;
            case 'cancelled':
              updateAnswer(() => ({ pending: false, error: 'Stopped.' }));
              break;
            case 'error':
              updateAnswer(() => ({ pending: false, error: event.message }));
              break;
            default:
              break;
          }
        },
      });
    } catch (cause) {
      if (cause.name !== 'AbortError') setError(cause.message);
      updateAnswer(() => ({ pending: false }));
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [prompt, running, model]);

  const stop = () => {
    abortRef.current?.abort();
    fetch('/api/cancel', { method: 'POST' }).catch(() => {});
  };

  const blocked = Boolean(health && !health.ready);
  const lastStats = useMemo(
    () => [...messages].reverse().find((message) => message.stats)?.stats ?? null,
    [messages],
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH, flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH, boxSizing: 'border-box',
            borderRight: '1px solid', borderColor: 'divider',
            bgcolor: 'background.paper',
          },
        }}
      >
        <Toolbar sx={{ px: 2 }}>
          <Stack>
            <Typography variant="h6" sx={{ lineHeight: 1.2 }}>LiteMind</Typography>
            <Typography variant="caption" color="text.secondary">
              MoE inference on the CPU
            </Typography>
          </Stack>
        </Toolbar>
        <Divider />
        <List sx={{ px: 1, py: 1.5 }}>
          {SECTIONS.map(({ id, label, icon, caption }) => (
            <ListItemButton
              key={id}
              selected={section === id}
              onClick={() => setSection(id)}
              sx={{ borderRadius: 1.5, mb: 0.5 }}
            >
              <ListItemIcon sx={{ minWidth: 38, color: section === id ? 'primary.main' : undefined }}>
                {icon}
              </ListItemIcon>
              <ListItemText
                primary={label}
                secondary={caption}
                slotProps={{
                  primary: { fontSize: 14, fontWeight: section === id ? 600 : 400 },
                  secondary: { fontSize: 11 },
                }}
              />
            </ListItemButton>
          ))}
        </List>
        <Box sx={{ flexGrow: 1 }} />
        <Divider />
        <Box sx={{ p: 2 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Box sx={{
              width: 8, height: 8, borderRadius: '50%',
              bgcolor: blocked ? 'error.main' : running ? 'warning.main' : 'success.main',
            }} />
            <Typography variant="caption" color="text.secondary">
              {blocked ? 'engine unavailable' : running ? 'generating' : 'idle'}
            </Typography>
          </Stack>
          {lastStats && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              last: {lastStats.tokens_per_second.toFixed(2)} tok/s
            </Typography>
          )}
        </Box>
      </Drawer>

      <Box sx={{ flexGrow: 1, minWidth: 0 }}>
        <AppBar
          position="sticky" elevation={0} color="transparent"
          sx={{ borderBottom: '1px solid', borderColor: 'divider', backdropFilter: 'blur(8px)' }}
        >
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              {SECTIONS.find((entry) => entry.id === section)?.label}
            </Typography>
            {model && (
              <Stack direction="row" spacing={1} sx={{ mr: 1 }}>
                <Chip size="small" variant="outlined"
                      label={`${model.layers} layers · ${model.routed_experts} experts`} />
              </Stack>
            )}
            <Tooltip title="Settings">
              <IconButton onClick={() => setSettingsOpen(true)}><SettingsIcon /></IconButton>
            </Tooltip>
          </Toolbar>
          {running && <LinearProgress />}
        </AppBar>

        <Box sx={{ p: 3 }}>
          {blocked && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {health.error ?? 'The engine or the checkpoint is missing.'}{' '}
              {health.executable
                ? `Model directory: ${health.modelDirectory}`
                : 'Build it with scripts/build.ps1.'}
            </Alert>
          )}
          {health?.settingsError && (
            <Alert severity="warning" sx={{ mb: 2 }}>{health.settingsError}</Alert>
          )}

          {section === 'chat' && (
            <ChatView
              messages={messages} prompt={prompt} setPrompt={setPrompt}
              running={running} blocked={blocked} error={error}
              onSubmit={submit} onStop={stop}
            />
          )}
          {section === 'model' && (
            <ModelView model={model} plan={plan}
                       generatedTokens={running ? null : lastStats?.generated_tokens ?? null} />
          )}
          {section === 'usage' && (
            <UsageView usage={usage} memory={memory} model={model} plan={plan} />
          )}
          {section === 'routing' && (
            <RoutingView model={model} routing={routing} live={running} />
          )}
        </Box>
      </Box>

      <SettingsDialog
        open={settingsOpen} settings={settings}
        onClose={() => setSettingsOpen(false)} onSaved={setSettings}
      />
    </Box>
  );
}
