import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Grid from '@mui/material/Grid2';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';

import PlanCard from '../components/PlanCard.jsx';
import { withSeparators } from '../format.js';

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

/** The architecture, as the engine reports it rather than as documented. */
export default function ModelView({ model, plan, generatedTokens }) {
  if (!model) {
    return (
      <Typography color="text.secondary">
        Run a prompt — these figures come from the engine when it loads the checkpoint.
      </Typography>
    );
  }

  return (
    <Grid container spacing={2}>
      <Grid size={{ xs: 12, md: 6 }}>
        <Card sx={{ height: '100%' }}>
          <CardContent>
            <Typography variant="overline" color="text.secondary">Architecture</Typography>
            <Stack direction="row" spacing={1} sx={{ mt: 1, mb: 1.5 }} flexWrap="wrap" useFlexGap>
              <Chip size="small" color="primary" variant="outlined" label={model.model_type} />
              <Chip
                size="small"
                color={model.chat_template ? 'success' : 'default'}
                variant="outlined"
                label={model.chat_template ? 'instruction-tuned' : 'base checkpoint'}
              />
            </Stack>
            <Stack spacing={1.2}>
              <Row label="Layers" value={model.layers}
                   hint="One dense feed-forward layer, then mixture-of-experts for the rest." />
              <Row label="Mixture-of-experts layers" value={model.moe_layers} />
              <Row label="Hidden size" value={withSeparators(model.hidden_size)} />
              <Row label="Attention heads" value={model.attention_heads} />
              <Row label="Vocabulary" value={withSeparators(model.vocab_size)} />
              <Row label="Context" value={withSeparators(model.context_length)}
                   hint="Prompt plus reply. Sizes the key/value cache." />
              <Divider sx={{ my: 0.5 }} />
              <Row label="Routed experts per layer" value={model.routed_experts} />
              <Row label="Chosen per layer" value={model.experts_per_token / model.moe_layers}
                   hint="The router scores all of them and keeps this many." />
              <Row label="Shared experts" value={model.shared_experts}
                   hint="Run for every token regardless of routing." />
              <Row
                label="Routed experts in model"
                value={withSeparators(model.moe_layers * model.routed_experts)}
                hint="The whole pool. Expert executions reuse these; nothing else is ever loaded."
              />
              <Row label="Load time" value={`${model.load_seconds.toFixed(2)} s`}
                   hint="The weights are memory mapped, so this is not a read of the whole file." />
            </Stack>
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, md: 6 }}>
        <PlanCard plan={plan} generatedTokens={generatedTokens} />
      </Grid>
    </Grid>
  );
}
