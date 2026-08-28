import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';

import { bytes, parameters, withSeparators } from '../format.js';

const Row = ({ label, value, hint }) => (
  <Stack direction="row" justifyContent="space-between" alignItems="baseline" spacing={2}>
    <Tooltip title={hint ?? ''} placement="left" arrow disableHoverListener={!hint}>
      <Typography
        variant="body2"
        color="text.secondary"
        sx={hint ? { borderBottom: '1px dotted', borderColor: 'divider', cursor: 'help' } : null}
      >
        {label}
      </Typography>
    </Tooltip>
    <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>
      {value}
    </Typography>
  </Stack>
);

/**
 * What this prompt costs.
 *
 * The counts here are easy to misread, so the card states the pool size and the
 * reuse factor next to the execution count. An execution is not a load: 79,248
 * executions draw on 1,664 distinct experts, each reused about forty-eight
 * times, which is the whole reason the model fits on a laptop.
 */
export default function PlanCard({ plan, generatedTokens }) {
  if (!plan) return null;

  const distinct = plan.distinct_experts || 1;
  const plannedReuse = plan.expert_activations / distinct;

  // Once a reply has finished, what it actually did is more informative than
  // the ceiling that was printed before it started.
  const actualPasses = generatedTokens == null ? null : plan.prompt_tokens + generatedTokens;
  const actualActivations = actualPasses == null ? null : actualPasses * plan.experts_per_token;

  return (
    <Card>
      <CardContent>
        <Typography variant="overline" color="text.secondary">This prompt</Typography>
        <Stack spacing={1.2} sx={{ mt: 1 }}>
          <Row label="Prompt tokens" value={withSeparators(plan.prompt_tokens)} />
          <Row
            label="Forward passes"
            value={
              actualPasses == null
                ? `up to ${withSeparators(plan.forward_passes)}`
                : withSeparators(actualPasses)
            }
            hint="One per token: the prompt is read a token at a time, then each new token is generated the same way."
          />

          <Divider sx={{ my: 0.5 }} />

          <Row
            label="Experts per token"
            value={`${plan.moe_layers} × ${plan.experts_per_token / plan.moe_layers} = ${plan.experts_per_token}`}
            hint="Each mixture-of-experts layer has its own pool and chooses independently."
          />
          <Row
            label="Distinct experts"
            value={withSeparators(distinct)}
            hint="Every routed expert in the model. Executions reuse these; nothing else is ever loaded."
          />
          <Row
            label="Expert executions"
            value={
              actualActivations == null
                ? `up to ${withSeparators(plan.expert_activations)}`
                : withSeparators(actualActivations)
            }
            hint="Executions, not loads. The same experts run over and over."
          />
          <Row
            label="Reuse per expert"
            value={`~${Math.max(1, Math.round(
              (actualActivations ?? plan.expert_activations) / distinct,
            ))}×`}
            hint="Why streaming works: after the first touch these come from the page cache, not the SSD."
          />

          <Divider sx={{ my: 0.5 }} />

          <Row label="One expert" value={bytes(plan.expert_bytes)} />
          <Row
            label="Weight reads"
            value={
              actualActivations == null
                ? `up to ${bytes(plan.weight_traffic_bytes)}`
                : bytes(actualActivations * plan.expert_bytes)
            }
            hint="Logical reads. Most are served from RAM, not the disk."
          />

          <Divider sx={{ my: 0.5 }} />

          <Row label="Parameters" value={parameters(plan.total_parameters)} />
          <Row
            label="Active per token"
            value={`${parameters(plan.active_parameters)}  (${(
              (plan.active_parameters / plan.total_parameters) * 100
            ).toFixed(1)}%)`}
            hint="The sparsity the whole design rests on."
          />
        </Stack>
        {generatedTokens == null && plannedReuse > 0 && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1.5, display: 'block' }}>
            Ceilings, assuming the reply runs to the token limit. Actual figures replace these when it finishes.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}
