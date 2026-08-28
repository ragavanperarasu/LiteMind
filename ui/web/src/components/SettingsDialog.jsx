import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import FormControlLabel from '@mui/material/FormControlLabel';
import Grid from '@mui/material/Grid2';
import Stack from '@mui/material/Stack';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';

import { saveSettings } from '../api.js';

const NUMBERS = [
  { key: 'max_tokens', label: 'Max new tokens', help: 'Upper bound on the reply length.' },
  { key: 'context', label: 'Context', help: 'Prompt plus reply. Sizes the key/value cache.' },
  { key: 'threads', label: 'Threads', help: '0 uses every hardware thread.' },
  { key: 'temperature', label: 'Temperature', step: '0.05', help: '0 is greedy, which loops on long replies.' },
  { key: 'top_k', label: 'Top-k', help: 'Candidates kept before sampling. 0 disables.' },
  { key: 'top_p', label: 'Top-p', step: '0.01', help: 'Nucleus cut. 1 disables.' },
  { key: 'repeat_penalty', label: 'Repeat penalty', step: '0.05', help: 'Above 1 discourages repetition.' },
  {
    key: 'expert_cache_gb',
    label: 'Expert cache (GB)',
    help: '0 leaves residency to the page cache. A budget under ~2.6 GB evicts every expert before reuse.',
  },
];

/** Edits litemind.json, which is also the file the command line reads. */
export default function SettingsDialog({ open, settings, onClose, onSaved }) {
  const [draft, setDraft] = useState(settings ?? {});
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setDraft(settings ?? {});
      setError(null);
    }
  }, [open, settings]);

  const set = (key) => (event) => {
    const raw = event.target.value;
    // Empty stays empty rather than collapsing to zero, so a half-typed number
    // does not silently become a real setting.
    setDraft((current) => ({ ...current, [key]: raw === '' ? '' : Number(raw) }));
  };

  const commit = async () => {
    setSaving(true);
    setError(null);
    try {
      const cleaned = Object.fromEntries(
        Object.entries(draft).filter(([, value]) => value !== '' && !Number.isNaN(value)),
      );
      onSaved(await saveSettings(cleaned));
      onClose();
    } catch (cause) {
      setError(cause.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Settings</DialogTitle>
      <DialogContent dividers>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Written to <code>litemind.json</code> at the repository root — the same file the command
          line reads. Comment keys in it are preserved.
        </Typography>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <Grid container spacing={2}>
          {NUMBERS.map(({ key, label, step, help }) => (
            <Grid key={key} size={{ xs: 12, sm: 6 }}>
              <TextField
                fullWidth
                size="small"
                type="number"
                label={label}
                helperText={help}
                value={draft[key] ?? ''}
                onChange={set(key)}
                slotProps={{ htmlInput: { step: step ?? '1' } }}
              />
            </Grid>
          ))}
        </Grid>
        <Stack sx={{ mt: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={draft.chat !== false}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, chat: event.target.checked }))
                }
              />
            }
            label="Apply the checkpoint's chat template"
          />
          <Typography variant="caption" color="text.secondary" sx={{ ml: 6, mt: -0.5 }}>
            Off sends prompts raw. A base checkpoint has no template, so this does nothing there.
          </Typography>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={commit} variant="contained" disabled={saving}>
          {saving ? 'Saving…' : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
