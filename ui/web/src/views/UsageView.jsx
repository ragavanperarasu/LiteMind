import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Grid from '@mui/material/Grid2';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';

import { bytes, parameters, withSeparators } from '../format.js';

const Meter = ({ label, value, caption, colour = 'primary' }) => (
  <Box sx={{ mb: 2.5 }}>
    <Stack direction="row" justifyContent="space-between" alignItems="baseline">
      <Typography variant="body2" color="text.secondary">{label}</Typography>
      <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
        {(value * 100).toFixed(0)}%
      </Typography>
    </Stack>
    <LinearProgress
      variant="determinate"
      value={Math.min(100, value * 100)}
      color={colour}
      sx={{ height: 8, borderRadius: 4, mt: 0.5 }}
    />
    {caption && (
      <Typography variant="caption" color="text.secondary">{caption}</Typography>
    )}
  </Box>
);

const Stat = ({ label, value, hint }) => (
  <Tooltip title={hint ?? ''} arrow placement="top" disableHoverListener={!hint}>
    <Box sx={{ minWidth: 120 }}>
      <Typography variant="h6" sx={{ fontVariantNumeric: 'tabular-nums' }}>{value}</Typography>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
    </Box>
  </Tooltip>
);

/**
 * What the machine and the model are costing.
 *
 * The host figures are the whole machine, not this process: the engine's memory
 * is a mapping of a file, so most of what it "uses" is page cache the operating
 * system owns and will reclaim on demand. Attributing that to one process would
 * be a number that looks precise and means nothing.
 */
export default function UsageView({ usage, memory, model, plan }) {
  const hostMemory = usage ? usage.usedMemory / usage.totalMemory : 0;

  return (
    <Grid container spacing={2}>
      <Grid size={{ xs: 12, md: 6 }}>
        <Card sx={{ height: '100%' }}>
          <CardContent>
            <Typography variant="overline" color="text.secondary">Host</Typography>
            <Typography variant="body2" sx={{ mt: 0.5, mb: 2 }}>
              {usage?.model ?? '—'}
            </Typography>

            <Meter
              label="CPU"
              value={usage?.cpu ?? 0}
              caption={`${usage?.coreCount ?? 0} logical cores`}
              colour={(usage?.cpu ?? 0) > 0.85 ? 'warning' : 'primary'}
            />
            <Meter
              label="Memory"
              value={hostMemory}
              caption={usage ? `${bytes(usage.usedMemory)} of ${bytes(usage.totalMemory)} in use` : ''}
              colour={hostMemory > 0.9 ? 'warning' : 'success'}
            />

            {usage?.cores?.length > 0 && (
              <>
                <Typography variant="caption" color="text.secondary">Per core</Typography>
                <Stack direction="row" spacing={0.5} sx={{ mt: 0.75 }} flexWrap="wrap" useFlexGap>
                  {usage.cores.map((load, index) => (
                    <Tooltip key={index} title={`core ${index}: ${(load * 100).toFixed(0)}%`} arrow>
                      <Box
                        sx={{
                          width: 16, height: 40, borderRadius: 0.5,
                          bgcolor: 'rgba(255,255,255,0.06)',
                          display: 'flex', alignItems: 'flex-end', overflow: 'hidden',
                        }}
                      >
                        <Box sx={{
                          width: '100%', height: `${Math.max(3, load * 100)}%`,
                          bgcolor: load > 0.85 ? 'warning.main' : 'primary.main',
                          transition: 'height 240ms ease',
                        }} />
                      </Box>
                    </Tooltip>
                  ))}
                </Stack>
              </>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, md: 6 }}>
        <Card sx={{ height: '100%' }}>
          <CardContent>
            <Typography variant="overline" color="text.secondary">Checkpoint</Typography>
            {!memory ? (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Run a prompt to read the engine's own figures.
              </Typography>
            ) : (
              <Stack spacing={1.2} sx={{ mt: 1 }}>
                <Row label="Mapped from disk" value={bytes(memory.mapped_bytes)}
                     hint={`${memory.shards} shard(s), ${withSeparators(memory.tensors)} tensors`} />
                <Row label="Always hot" value={bytes(memory.hot_bytes)}
                     hint="Embeddings, attention, norms, shared experts, output head. Touched every token." />
                <Row label="Routed experts" value={bytes(memory.routed_expert_bytes)}
                     hint="Streamed on demand. This is what the sparsity keeps off the critical path." />
                <Row label="Key/value cache" value={bytes(memory.kv_cache_bytes)} />
                <Row
                  label="Expert budget"
                  value={memory.expert_budget_bytes ? bytes(memory.expert_budget_bytes) : 'page cache'}
                  hint="0 leaves residency to the operating system, which is fastest when RAM is plentiful."
                />
                {memory.expert_budget_bytes > 0 && (
                  <Row label="Resident now"
                       value={`${bytes(memory.resident_expert_bytes)} · ${memory.resident_experts} experts`} />
                )}
              </Stack>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid size={12}>
        <Card>
          <CardContent>
            <Typography variant="overline" color="text.secondary">Parameters</Typography>
            {!plan && !model ? (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Run a prompt to see the split.
              </Typography>
            ) : (
              <>
                <Stack direction="row" spacing={4} sx={{ mt: 1.5 }} flexWrap="wrap" useFlexGap>
                  <Stat label="Total" value={parameters(plan?.total_parameters)} />
                  <Stat label="Active per token" value={parameters(plan?.active_parameters)}
                        hint="Everything except the routed experts, plus the six chosen in each layer." />
                  <Stat
                    label="Fraction active"
                    value={plan ? `${((plan.active_parameters / plan.total_parameters) * 100).toFixed(1)}%` : '—'}
                    hint="The whole reason a 15.7 billion parameter model runs on a laptop."
                  />
                  <Stat label="Experts per token" value={withSeparators(plan?.experts_per_token)} />
                  <Stat label="One expert" value={bytes(plan?.expert_bytes)} />
                </Stack>

                {plan && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="caption" color="text.secondary">
                      Active (coloured) against dormant, per token
                    </Typography>
                    <Box sx={{
                      mt: 0.75, height: 22, borderRadius: 1, overflow: 'hidden', display: 'flex',
                      bgcolor: 'rgba(255,255,255,0.06)',
                    }}>
                      <Box sx={{
                        width: `${(plan.active_parameters / plan.total_parameters) * 100}%`,
                        bgcolor: 'primary.main',
                      }} />
                    </Box>
                  </Box>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

const Row = ({ label, value, hint }) => (
  <Stack direction="row" justifyContent="space-between" alignItems="baseline" spacing={2}>
    <Tooltip title={hint ?? ''} arrow placement="left" disableHoverListener={!hint}>
      <Typography variant="body2" color="text.secondary"
                  sx={hint ? { borderBottom: '1px dotted', borderColor: 'divider', cursor: 'help' } : null}>
        {label}
      </Typography>
    </Tooltip>
    <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>
      {value}
    </Typography>
  </Stack>
);
