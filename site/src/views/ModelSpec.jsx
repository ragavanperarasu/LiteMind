import { Box, Card, Divider, Link, Stack, Typography } from '@mui/material';
import { alpha } from '@mui/material/styles';
import modelInfo from '../../../docs/model-info.json';
import { site, sourceUrl } from '../lib/site.js';

/**
 * The `_note` fields are prose with the occasional `backticked` identifier in
 * them. Split rather than render markdown, so nothing in the JSON can inject
 * markup.
 */
function Ticked({ text }) {
  return String(text)
    .split('`')
    .map((part, index) =>
      index % 2 === 1 ? (
        <Box
          key={index}
          component="code"
          sx={{
            fontFamily: '"JetBrains Mono", ui-monospace, monospace',
            fontSize: '0.86em',
            px: 0.6,
            py: 0.15,
            borderRadius: 1,
            bgcolor: (t) => alpha(t.palette.primary.main, 0.13),
          }}
        >
          {part}
        </Box>
      ) : (
        <span key={index}>{part}</span>
      ),
    );
}

/** `expert_executions_per_token` reads as a field name; this is for humans. */
function humanise(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\bgib\b/i, 'GiB')
    .replace(/\bmib\b/i, 'MiB')
    .replace(/\bkib\b/i, 'KiB')
    .replace(/\bkv\b/i, 'KV')
    .replace(/^./, (character) => character.toUpperCase());
}

function format(value) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString('en-US') : String(value);
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

function Row({ name, value }) {
  return (
    <Stack
      direction={{ xs: 'column', sm: 'row' }}
      spacing={{ xs: 0.25, sm: 2 }}
      sx={{ px: 3, py: 1.5, alignItems: { sm: 'baseline' } }}
    >
      <Typography variant="body2" color="text.secondary" sx={{ minWidth: { sm: 260 } }}>
        {humanise(name)}
      </Typography>
      <Typography
        variant="body2"
        sx={{
          fontFamily: '"JetBrains Mono", ui-monospace, monospace',
          fontVariantNumeric: 'tabular-nums',
          fontWeight: 500,
          wordBreak: 'break-word',
        }}
      >
        {format(value)}
      </Typography>
    </Stack>
  );
}

function Group({ name, data, depth = 0 }) {
  const note = data._note;
  const entries = Object.entries(data).filter(([key]) => !key.startsWith('_'));
  const leaves = entries.filter(([, value]) => typeof value !== 'object' || value === null);
  const nested = entries.filter(([, value]) => typeof value === 'object' && value !== null);

  const body = (
    <>
      {note && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ px: 3, pt: depth === 0 ? 0 : 2, pb: 2, lineHeight: 1.65, maxWidth: '72ch' }}
        >
          <Ticked text={note} />
        </Typography>
      )}
      <Stack divider={<Divider />}>
        {leaves.map(([key, value]) => (
          <Row key={key} name={key} value={value} />
        ))}
      </Stack>
      {nested.map(([key, value]) => (
        <Box key={key} sx={{ mt: 2, mx: 3, mb: 1, borderLeft: 2, borderColor: 'divider' }}>
          <Typography variant="overline" color="text.secondary" sx={{ px: 2 }}>
            {humanise(key)}
          </Typography>
          <Box sx={{ mx: -3 }}>
            <Group name={key} data={value} depth={depth + 1} />
          </Box>
        </Box>
      ))}
    </>
  );

  if (depth > 0) return body;

  return (
    <Card variant="outlined" sx={{ borderColor: 'divider', overflow: 'hidden' }}>
      <Box sx={{ px: 3, py: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography sx={{ fontWeight: 650 }}>{humanise(name)}</Typography>
      </Box>
      <Box sx={{ py: 2 }}>{body}</Box>
    </Card>
  );
}

/** One MoE layer: 64 experts, 6 of them lit. The picture the whole project rests on. */
function RoutingFigure() {
  const chosen = new Set([3, 11, 24, 37, 48, 59]);
  return (
    <Card variant="outlined" sx={{ borderColor: 'divider', p: 3 }}>
      <Typography variant="overline" color="text.secondary">
        One mixture-of-experts layer
      </Typography>
      <Box
        sx={{
          mt: 2,
          display: 'grid',
          gridTemplateColumns: 'repeat(16, 1fr)',
          gap: 1,
          maxWidth: 420,
        }}
      >
        {Array.from({ length: 64 }, (_, index) => {
          const on = chosen.has(index);
          return (
            <Box
              key={index}
              sx={{
                aspectRatio: '1',
                borderRadius: '50%',
                bgcolor: (t) =>
                  on ? t.palette.warning.main : alpha(t.palette.text.primary, 0.13),
                boxShadow: (t) => (on ? `0 0 10px ${alpha(t.palette.warning.main, 0.7)}` : 'none'),
              }}
            />
          );
        })}
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 2.5, maxWidth: '60ch' }}>
        Six of sixty-four experts run for a given token, plus two shared experts that always run.
        Multiply by 26 layers: <b>156 expert executions per token</b>, drawn from a pool of{' '}
        <b>1,664</b>. That ratio is why 29.3 GiB of weights fit on a machine that cannot hold them.
      </Typography>
    </Card>
  );
}

export default function ModelSpec() {
  const groups = Object.entries(modelInfo).filter(([key]) => !key.startsWith('_'));

  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Reference
      </Typography>
      <Typography
        component="h1"
        sx={{ mt: 1, fontSize: 'clamp(2rem, 1.5rem + 2vw, 2.8rem)', fontWeight: 700, letterSpacing: '-0.03em' }}
      >
        Model specification
      </Typography>
      <Typography color="text.secondary" sx={{ mt: 1.5, maxWidth: '68ch', lineHeight: 1.7 }}>
        <Ticked text={modelInfo._about} /> The source is{' '}
        <Link href={sourceUrl('docs/model-info.json')} target="_blank" rel="noopener noreferrer">
          docs/model-info.json
        </Link>
        , and the weights are{' '}
        <Link href={site.model} target="_blank" rel="noopener noreferrer">
          deepseek-ai/DeepSeek-V2-Lite
        </Link>
        .
      </Typography>

      <Box sx={{ mt: 4 }}>
        <RoutingFigure />
      </Box>

      <Stack spacing={3} sx={{ mt: 3 }}>
        {groups.map(([key, value]) => (
          <Group key={key} name={key} data={value} />
        ))}
      </Stack>
    </Box>
  );
}
