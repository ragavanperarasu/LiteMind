import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Skeleton from '@mui/material/Skeleton';

import { withSeparators } from '../format.js';

const Row = ({ label, value }) => (
  <Stack direction="row" justifyContent="space-between" alignItems="baseline" spacing={2}>
    <Typography variant="body2" color="text.secondary">{label}</Typography>
    <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>
      {value}
    </Typography>
  </Stack>
);

/** What the checkpoint is, from the engine's own ready event. */
export default function ModelCard({ model }) {
  return (
    <Card>
      <CardContent>
        <Typography variant="overline" color="text.secondary">Model</Typography>
        {!model ? (
          <Stack spacing={1} sx={{ mt: 1 }}>
            {[...Array(6)].map((_, index) => <Skeleton key={index} height={22} />)}
          </Stack>
        ) : (
          <Stack spacing={1.2} sx={{ mt: 1 }}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Chip size="small" label={model.model_type} color="primary" variant="outlined" />
              {model.chat_template && (
                <Chip size="small" label="chat template" color="success" variant="outlined" />
              )}
            </Stack>
            <Divider sx={{ my: 0.5 }} />
            <Row label="Layers" value={`${model.layers} (${model.moe_layers} MoE)`} />
            <Row label="Hidden size" value={withSeparators(model.hidden_size)} />
            <Row label="Attention heads" value={model.attention_heads} />
            <Row
              label="Experts per layer"
              value={`${model.experts_per_token} of ${model.routed_experts}`}
            />
            <Row label="Shared experts" value={model.shared_experts} />
            <Row label="Vocabulary" value={withSeparators(model.vocab_size)} />
            <Row label="Context" value={withSeparators(model.context_length)} />
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}
