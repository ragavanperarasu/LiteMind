import { useCallback, useEffect, useRef, useState } from 'react';
import AppBar from '@mui/material/AppBar';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid2';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Toolbar from '@mui/material/Toolbar';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import SettingsIcon from '@mui/icons-material/Settings';
import SendIcon from '@mui/icons-material/Send';
import StopIcon from '@mui/icons-material/Stop';

import ModelCard from './components/ModelCard.jsx';
import PlanCard from './components/PlanCard.jsx';
import SettingsDialog from './components/SettingsDialog.jsx';
import { fetchHealth, fetchSettings, streamGeneration } from './api.js';
import { seconds } from './format.js';

const EMPTY_RUN = { text: '', plan: null, stats: null, error: null };

export default function App() {
  const [health, setHealth] = useState(null);
  const [settings, setSettings] = useState(null);
  const [model, setModel] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [run, setRun] = useState(EMPTY_RUN);
  const [running, setRunning] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const abortRef = useRef(null);
  const answerRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch((cause) => setHealth({ ready: false, error: cause.message }));
    fetchSettings().then(setSettings).catch(() => setSettings({}));
  }, []);

  // Follow the text as it streams, but only while the reader is already at the
  // bottom - yanking the view back while someone is reading earlier output is
  // worse than letting it scroll away.
  useEffect(() => {
    const element = answerRef.current;
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
    if (atBottom) element.scrollTop = element.scrollHeight;
  }, [run.text]);

  const submit = useCallback(async () => {
    const text = prompt.trim();
    if (text === '' || running) return;

    setRun(EMPTY_RUN);
    setRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamGeneration({
        prompt: text,
        settings: {},
        signal: controller.signal,
        onEvent: (event) => {
          switch (event.event) {
            case 'ready':
              setModel(event);
              break;
            case 'plan':
              setRun((current) => ({ ...current, plan: event }));
              break;
            case 'token':
              setRun((current) => ({ ...current, text: current.text + event.text }));
              break;
            case 'done':
              setRun((current) => ({ ...current, stats: event, text: event.text || current.text }));
              break;
            case 'cancelled':
              setRun((current) => ({ ...current, error: 'Stopped.' }));
              break;
            case 'error':
              setRun((current) => ({ ...current, error: event.message }));
              break;
            default:
              break;
          }
        },
      });
    } catch (cause) {
      if (cause.name !== 'AbortError') {
        setRun((current) => ({ ...current, error: cause.message }));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [prompt, running]);

  const stop = () => {
    abortRef.current?.abort();
    fetch('/api/cancel', { method: 'POST' }).catch(() => {});
  };

  const onKeyDown = (event) => {
    // Enter sends; Shift+Enter is a newline, since prompts are sometimes long.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const blocked = health && !health.ready;

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="sticky" elevation={0} color="transparent"
              sx={{ borderBottom: '1px solid', borderColor: 'divider', backdropFilter: 'blur(8px)' }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            LiteMind
            <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 1.5 }}>
              DeepSeek-V2 mixture-of-experts on the CPU
            </Typography>
          </Typography>
          {model && (
            <Chip size="small" variant="outlined" sx={{ mr: 1 }}
                  label={`${model.layers} layers · ${model.routed_experts} experts`} />
          )}
          <Tooltip title="Settings">
            <IconButton onClick={() => setSettingsOpen(true)}><SettingsIcon /></IconButton>
          </Tooltip>
        </Toolbar>
        {running && <LinearProgress />}
      </AppBar>

      <Container maxWidth="xl" sx={{ py: 3 }}>
        {blocked && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {health.error ?? 'The engine or the checkpoint is missing.'}{' '}
            {health.executable ? `Model directory: ${health.modelDirectory}` : 'Build it with scripts/build.ps1.'}
          </Alert>
        )}
        {health?.settingsError && (
          <Alert severity="warning" sx={{ mb: 2 }}>{health.settingsError}</Alert>
        )}

        <Grid container spacing={2}>
          <Grid size={{ xs: 12, md: 8 }}>
            <Stack spacing={2}>
              <Card>
                <CardContent>
                  <TextField
                    fullWidth multiline minRows={2} maxRows={8}
                    placeholder="Ask something…"
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    onKeyDown={onKeyDown}
                    disabled={running || blocked}
                  />
                  <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} alignItems="center">
                    <Typography variant="caption" color="text.secondary" sx={{ flexGrow: 1 }}>
                      Enter to send · Shift+Enter for a new line
                    </Typography>
                    {running ? (
                      <Button onClick={stop} startIcon={<StopIcon />} color="warning" variant="outlined">
                        Stop
                      </Button>
                    ) : (
                      <Button onClick={submit} startIcon={<SendIcon />} variant="contained"
                              disabled={blocked || prompt.trim() === ''}>
                        Send
                      </Button>
                    )}
                  </Stack>
                </CardContent>
              </Card>

              <Card sx={{ minHeight: 320 }}>
                <CardContent>
                  <Typography variant="overline" color="text.secondary">Answer</Typography>
                  {run.error && <Alert severity="error" sx={{ mt: 1 }}>{run.error}</Alert>}
                  <Box
                    ref={answerRef}
                    sx={{
                      mt: 1, maxHeight: '52vh', overflowY: 'auto',
                      whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                      fontSize: '0.95rem', lineHeight: 1.7,
                    }}
                  >
                    {run.text || (
                      <Typography variant="body2" color="text.disabled">
                        {running ? 'Reading the prompt…' : 'Nothing yet.'}
                      </Typography>
                    )}
                  </Box>
                  {run.stats && (
                    <Stack direction="row" spacing={1} sx={{ mt: 2 }} flexWrap="wrap" useFlexGap>
                      <Chip size="small" label={`${run.stats.generated_tokens} tokens`} />
                      <Chip size="small" label={`${run.stats.tokens_per_second.toFixed(2)} tok/s`} />
                      <Chip size="small" variant="outlined"
                            label={`prefill ${seconds(run.stats.prefill_seconds)}`} />
                      <Chip size="small" variant="outlined"
                            label={`decode ${seconds(run.stats.decode_seconds)}`} />
                      <Chip size="small" variant="outlined" color="secondary"
                            label={run.stats.stop_reason} />
                    </Stack>
                  )}
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          <Grid size={{ xs: 12, md: 4 }}>
            <Stack spacing={2}>
              <ModelCard model={model} />
              <PlanCard plan={run.plan} generatedTokens={run.stats?.generated_tokens ?? null} />
            </Stack>
          </Grid>
        </Grid>
      </Container>

      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSaved={setSettings}
      />
    </Box>
  );
}
