import { Box, Card, Chip, Link, Stack, Typography } from '@mui/material';
import { docs, sections } from '../lib/content.js';
import { routes } from '../lib/router.js';

export default function DocsIndex() {
  const total = docs.reduce((sum, doc) => sum + doc.minutes, 0);

  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Contents
      </Typography>
      <Typography
        component="h1"
        sx={{ mt: 1, fontSize: 'clamp(2rem, 1.5rem + 2vw, 2.8rem)', fontWeight: 700, letterSpacing: '-0.03em' }}
      >
        Every page
      </Typography>
      <Typography color="text.secondary" sx={{ mt: 1.5, mb: 5, maxWidth: '62ch' }}>
        {docs.length} pages, about {total} minutes of reading in total. They are written to be read
        in order the first time, and used as reference afterwards.
      </Typography>

      <Stack spacing={5}>
        {sections.map((section) => (
          <Box key={section.label}>
            <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2 }}>
              <Typography variant="overline" color="text.secondary">
                {section.label}
              </Typography>
              <Box sx={{ flex: 1, height: '1px', bgcolor: 'divider' }} />
            </Stack>

            <Stack spacing={1.5}>
              {section.pages.map((page) => (
                <Card
                  key={page.id}
                  variant="outlined"
                  sx={{
                    borderColor: 'divider',
                    transition: 'border-color .15s, transform .15s',
                    '&:hover': { borderColor: 'primary.main', transform: 'translateY(-1px)' },
                  }}
                >
                  <Link
                    href={routes.doc(page.id)}
                    underline="none"
                    sx={{ display: 'block', p: 2.5, color: 'inherit' }}
                  >
                    <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 0.75 }}>
                      <Chip
                        label={page.number}
                        size="small"
                        sx={{ height: 22, minWidth: 32, fontWeight: 700, fontSize: '0.7rem' }}
                      />
                      <Typography sx={{ fontWeight: 650, fontSize: '1.02rem' }}>
                        {page.title}
                      </Typography>
                      <Box sx={{ flex: 1 }} />
                      <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                        {page.minutes} min
                      </Typography>
                    </Stack>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        lineHeight: 1.6,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}
                    >
                      {page.summary}
                    </Typography>
                  </Link>
                </Card>
              ))}
            </Stack>
          </Box>
        ))}
      </Stack>
    </Box>
  );
}
