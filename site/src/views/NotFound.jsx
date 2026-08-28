import { Box, Button, Typography } from '@mui/material';
import { routes } from '../lib/router.js';

export default function NotFound({ path }) {
  return (
    <Box sx={{ py: 8, textAlign: 'center' }}>
      <Typography sx={{ fontSize: '3rem', fontWeight: 700, letterSpacing: '-0.03em' }}>
        No such page
      </Typography>
      <Typography color="text.secondary" sx={{ mt: 1.5, mb: 4 }}>
        Nothing is routed at <code>#/{path}</code>.
      </Typography>
      <Button component="a" href={routes.index} variant="contained">
        See every page
      </Button>
    </Box>
  );
}
