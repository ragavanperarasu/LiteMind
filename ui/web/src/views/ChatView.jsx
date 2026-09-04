import { useEffect, useRef } from 'react';
import Alert from '@mui/material/Alert';
import Avatar from '@mui/material/Avatar';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import Switch from '@mui/material/Switch';
import Tooltip from '@mui/material/Tooltip';
import PersonIcon from '@mui/icons-material/Person';
import MemoryIcon from '@mui/icons-material/Memory';
import SendIcon from '@mui/icons-material/Send';
import StopIcon from '@mui/icons-material/Stop';
import HistoryIcon from '@mui/icons-material/History';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';

import { seconds } from '../format.js';

/**
 * The conversation.
 *
 * The engine still keeps nothing between requests - each prompt is a fresh
 * process reading a fresh prompt. What makes the transcript a conversation is
 * that the finished exchanges are sent again with every new question, so the
 * memory lives here, in the page, and is only as good as what fits in the
 * context window. The counter says how much is actually being carried, because
 * a conversation that has quietly dropped its opening reads like a model that
 * stopped paying attention.
 */
export default function ChatView({
  messages, prompt, setPrompt, running, blocked, onSubmit, onStop, error,
  remember, onToggleRemember, rememberedTurns, onClear, droppedTurns,
}) {
  const endRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 120;
    if (atBottom) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <Stack sx={{ height: 'calc(100vh - 132px)' }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ pb: 1 }}>
        <Tooltip title={
          remember
            ? 'Finished exchanges are sent again with each new question, so the model can refer back to them.'
            : 'Each question is answered on its own, with nothing carried over.'
        }>
          <Chip
            size="small"
            icon={<HistoryIcon />}
            variant={remember ? 'filled' : 'outlined'}
            color={remember ? 'primary' : 'default'}
            label={
              !remember
                ? 'Memory off'
                : rememberedTurns === 0
                  ? 'Memory on'
                  : `Remembering ${rememberedTurns} ${rememberedTurns === 1 ? 'exchange' : 'exchanges'}`
            }
          />
        </Tooltip>
        {droppedTurns > 0 && (
          <Tooltip title="The conversation outgrew the context window, so the oldest exchanges were left out of the last prompt. Raise --context in settings to keep more.">
            <Chip size="small" variant="outlined" color="warning"
                  label={`${droppedTurns} dropped to fit the context`} />
          </Tooltip>
        )}
        <Box sx={{ flexGrow: 1 }} />
        <Typography variant="caption" color="text.secondary">Remember</Typography>
        <Switch size="small" checked={remember} onChange={onToggleRemember} disabled={running} />
        <Tooltip title="Start a new conversation">
          <span>
            <IconButton size="small" onClick={onClear} disabled={running || messages.length === 0}>
              <DeleteSweepIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
      <Divider sx={{ mb: 2 }} />

      <Box ref={scrollRef} sx={{ flexGrow: 1, overflowY: 'auto', pr: 1 }}>
        {messages.length === 0 && (
          <Stack alignItems="center" justifyContent="center" sx={{ height: '100%', opacity: 0.6 }}>
            <MemoryIcon sx={{ fontSize: 44, mb: 1 }} />
            <Typography variant="h6">Ask something</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, maxWidth: 460, textAlign: 'center' }}>
              {remember
                ? 'Each finished exchange is sent again with your next question, so you can follow one up with another. Turn Remember off to ask each question on its own.'
                : 'Each question is answered on its own, with nothing carried over from the last one.'}
            </Typography>
          </Stack>
        )}

        <Stack spacing={2}>
          {messages.map((message, index) => (
            <Stack
              key={index}
              direction="row"
              spacing={1.5}
              sx={{ flexDirection: message.role === 'user' ? 'row-reverse' : 'row' }}
            >
              <Avatar sx={{
                width: 30, height: 30,
                bgcolor: message.role === 'user' ? 'secondary.main' : 'primary.main',
              }}>
                {message.role === 'user'
                  ? <PersonIcon sx={{ fontSize: 18 }} />
                  : <MemoryIcon sx={{ fontSize: 18 }} />}
              </Avatar>
              <Paper
                variant="outlined"
                sx={{
                  p: 1.5, maxWidth: '76%',
                  bgcolor: message.role === 'user' ? 'action.selected' : 'background.paper',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.7,
                }}
              >
                {message.text || (message.pending && (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={13} />
                    <Typography variant="body2" color="text.secondary">
                      {message.stage ?? 'Reading the prompt…'}
                    </Typography>
                  </Stack>
                ))}
                {message.stats && (
                  <Stack direction="row" spacing={0.75} sx={{ mt: 1.5 }} flexWrap="wrap" useFlexGap>
                    <Chip size="small" label={`${message.stats.generated_tokens} tokens`} />
                    <Chip size="small" label={`${message.stats.tokens_per_second.toFixed(2)} tok/s`} />
                    <Chip size="small" variant="outlined"
                          label={`prefill ${seconds(message.stats.prefill_seconds)}`} />
                    <Chip size="small" variant="outlined" color="secondary"
                          label={message.stats.stop_reason} />
                  </Stack>
                )}
                {message.error && (
                  <Alert severity="error" sx={{ mt: 1 }}>{message.error}</Alert>
                )}
              </Paper>
            </Stack>
          ))}
        </Stack>
        <div ref={endRef} />
      </Box>

      {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}

      <Paper variant="outlined" sx={{ p: 1.5, mt: 2 }}>
        <TextField
          fullWidth multiline minRows={1} maxRows={8} variant="standard"
          placeholder="Ask something…"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={running || blocked}
          slotProps={{ input: { disableUnderline: true } }}
        />
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ flexGrow: 1 }}>
            Enter to send · Shift+Enter for a new line
          </Typography>
          {running ? (
            <Button onClick={onStop} startIcon={<StopIcon />} color="warning" variant="outlined" size="small">
              Stop
            </Button>
          ) : (
            <Button onClick={onSubmit} startIcon={<SendIcon />} variant="contained" size="small"
                    disabled={blocked || prompt.trim() === ''}>
              Send
            </Button>
          )}
        </Stack>
      </Paper>
    </Stack>
  );
}
