import { useMemo } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import { withSeparators } from '../format.js';

/**
 * The routing map: every routed expert in the model, one dot each.
 *
 * A column is a mixture-of-experts layer and a row is an expert within it, so
 * the whole pool is on screen at once - 26 x 64 = 1,664 dots for
 * DeepSeek-V2-Lite. Six dots light up per column per token, which is the
 * sparsity the entire project rests on, made literal: almost all of this grid
 * stays dark while the model answers.
 *
 * Brightness accumulates over the reply, so the picture also shows which
 * experts this particular prompt kept coming back to.
 */

const DOT = 9;
const RADIUS = 3;
const MARGIN = { left: 46, top: 28, right: 12, bottom: 8 };

export default function RoutingView({ model, routing, live }) {
  const layers = model?.moe_layers ?? 0;
  const perLayer = model?.routed_experts ?? 0;
  const topK = layers ? (model?.experts_per_token ?? 0) / layers : 0;

  const width = MARGIN.left + layers * DOT + MARGIN.right;
  const height = MARGIN.top + perLayer * DOT + MARGIN.bottom;

  // The busiest single expert sets the scale, so the contrast stays useful
  // whether the reply was four tokens or four hundred.
  const peak = useMemo(
    () => (routing.counts ? routing.counts.reduce((a, b) => Math.max(a, b), 0) : 0),
    [routing.counts],
  );

  const dots = useMemo(() => {
    if (!layers || !perLayer) return null;
    const nodes = [];
    for (let layer = 0; layer < layers; layer += 1) {
      for (let expert = 0; expert < perLayer; expert += 1) {
        const count = routing.counts?.[layer * perLayer + expert] ?? 0;
        const active = routing.current?.[layer] === expert
          || routing.currentSet?.has(layer * perLayer + expert);
        // A used expert never falls all the way back to the unused colour, so
        // "touched once" stays distinguishable from "never touched".
        const intensity = peak > 0 && count > 0 ? 0.25 + 0.75 * (count / peak) : 0;
        nodes.push(
          <circle
            key={`${layer}-${expert}`}
            cx={MARGIN.left + layer * DOT + DOT / 2}
            cy={MARGIN.top + expert * DOT + DOT / 2}
            r={active ? RADIUS + 1.4 : RADIUS}
            fill={
              active
                ? '#e0af68'
                : intensity > 0
                  ? `rgba(122, 162, 247, ${intensity})`
                  : 'rgba(255,255,255,0.07)'
            }
            style={active ? { filter: 'drop-shadow(0 0 4px #e0af68)' } : undefined}
          >
            <title>{`layer ${layer} · expert ${expert} · used ${count}×`}</title>
          </circle>,
        );
      }
    }
    return nodes;
  }, [layers, perLayer, routing.counts, routing.currentSet, routing.current, peak]);

  if (!model) {
    return (
      <Typography color="text.secondary">
        Run a prompt first — the map is drawn from the checkpoint the engine reports.
      </Typography>
    );
  }

  return (
    <Stack spacing={2}>
      <Card>
        <CardContent>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
            <Typography variant="overline" color="text.secondary" sx={{ flexGrow: 1 }}>
              Expert routing
            </Typography>
            <Chip size="small" variant="outlined" label={`${layers} layers`} />
            <Chip size="small" variant="outlined" label={`${perLayer} experts each`} />
            <Chip size="small" color="primary" variant="outlined"
                  label={`${withSeparators(layers * perLayer)} total`} />
            <Chip size="small" color="warning" variant="outlined"
                  label={`${topK} chosen per layer`} />
            {live && <Chip size="small" color="success" label="live" />}
          </Stack>

          <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
            One dot per routed expert. A column is a layer; {topK} of its {perLayer} dots light up
            for each token. Everything still dark is weight the model never read.
          </Typography>

          <Box sx={{ mt: 2, overflowX: 'auto' }}>
            <svg width={width} height={height} role="img" aria-label="Expert routing map">
              {Array.from({ length: Math.ceil(layers / 5) }, (_, index) => index * 5).map((layer) => (
                <text
                  key={layer}
                  x={MARGIN.left + layer * DOT + DOT / 2}
                  y={MARGIN.top - 10}
                  fill="rgba(255,255,255,0.4)"
                  fontSize="9"
                  textAnchor="middle"
                >
                  {layer}
                </text>
              ))}
              {Array.from({ length: Math.ceil(perLayer / 16) }, (_, index) => index * 16).map((expert) => (
                <text
                  key={expert}
                  x={MARGIN.left - 8}
                  y={MARGIN.top + expert * DOT + DOT / 2 + 3}
                  fill="rgba(255,255,255,0.4)"
                  fontSize="9"
                  textAnchor="end"
                >
                  {expert}
                </text>
              ))}
              <text x={4} y={14} fill="rgba(255,255,255,0.55)" fontSize="10">expert ↓ / layer →</text>
              {dots}
            </svg>
          </Box>

          <Stack direction="row" spacing={2} sx={{ mt: 2 }} alignItems="center" flexWrap="wrap" useFlexGap>
            <Legend colour="rgba(255,255,255,0.07)" label="never used" />
            <Legend colour="rgba(122, 162, 247, 0.35)" label="used occasionally" />
            <Legend colour="rgba(122, 162, 247, 1)" label="used often" />
            <Legend colour="#e0af68" label="this token" />
          </Stack>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="overline" color="text.secondary">Coverage</Typography>
          <Stack direction="row" spacing={3} sx={{ mt: 1 }} flexWrap="wrap" useFlexGap>
            <Figure label="Tokens routed" value={withSeparators(routing.tokens)} />
            <Figure
              label="Experts touched"
              value={`${withSeparators(routing.touched)} of ${withSeparators(layers * perLayer)}`}
            />
            <Figure
              label="Pool reached"
              value={`${(layers * perLayer ? (routing.touched / (layers * perLayer)) * 100 : 0).toFixed(1)}%`}
            />
            <Figure label="Busiest expert" value={`${withSeparators(peak)} uses`} />
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  );
}

const Legend = ({ colour, label }) => (
  <Stack direction="row" spacing={0.75} alignItems="center">
    <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: colour }} />
    <Typography variant="caption" color="text.secondary">{label}</Typography>
  </Stack>
);

const Figure = ({ label, value }) => (
  <Box>
    <Typography variant="h6" sx={{ fontVariantNumeric: 'tabular-nums' }}>{value}</Typography>
    <Typography variant="caption" color="text.secondary">{label}</Typography>
  </Box>
);
