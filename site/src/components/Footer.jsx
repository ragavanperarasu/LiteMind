import { Box, Divider, Link, Stack, Typography } from '@mui/material';
import { site, sourceUrl } from '../lib/site.js';
import { routes } from '../lib/router.js';

export default function Footer() {
  return (
    <Box component="footer" sx={{ mt: 10 }}>
      <Divider />
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        justifyContent="space-between"
        spacing={2}
        sx={{ py: 4 }}
      >
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 650 }}>
            {site.name}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {site.license} licensed. Built against{' '}
            <Link href={site.model} target="_blank" rel="noopener noreferrer">
              DeepSeek-V2-Lite
            </Link>
            , whose weights carry their own licence.
          </Typography>
        </Box>
        <Stack direction="row" spacing={3} sx={{ flexWrap: 'wrap' }}>
          <Link href={routes.doc('00-setup')} variant="body2" underline="hover" color="text.secondary">
            Setup
          </Link>
          <Link href={routes.model} variant="body2" underline="hover" color="text.secondary">
            Model spec
          </Link>
          <Link
            href={sourceUrl('docs')}
            target="_blank"
            rel="noopener noreferrer"
            variant="body2"
            underline="hover"
            color="text.secondary"
          >
            Markdown source
          </Link>
          <Link
            href={site.repository}
            target="_blank"
            rel="noopener noreferrer"
            variant="body2"
            underline="hover"
            color="text.secondary"
          >
            GitHub
          </Link>
        </Stack>
      </Stack>
    </Box>
  );
}
