import { useEffect } from 'react';
import { Box, Button, Chip, Divider, Link, Stack, Typography } from '@mui/material';
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import { neighbours, sections } from '../lib/content.js';
import { routes } from '../lib/router.js';
import { sourceUrl } from '../lib/site.js';
import Markdown from '../components/Markdown.jsx';
import TableOfContents from '../components/TableOfContents.jsx';

function sectionOf(id) {
  const found = sections.find((section) => section.pages.some((page) => page.id === id));
  return found ? found.label : 'Documentation';
}

export default function DocPage({ doc, anchor }) {
  const { previous, next } = neighbours(doc.id);

  useEffect(() => {
    // The anchor is part of the route, so this runs on a fresh page and on a
    // jump within one. The layout has to exist first, hence the frame.
    const frame = window.requestAnimationFrame(() => {
      if (!anchor) {
        window.scrollTo({ top: 0, behavior: 'auto' });
        return;
      }
      const target = document.getElementById(anchor);
      if (target) {
        const top = target.getBoundingClientRect().top + window.scrollY - 84;
        window.scrollTo({ top, behavior: 'auto' });
      } else {
        window.scrollTo({ top: 0, behavior: 'auto' });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [doc.id, anchor]);

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) 232px' },
        gap: { xs: 0, lg: 6 },
        alignItems: 'start',
      }}
    >
      <Box sx={{ minWidth: 0 }}>
        <Stack direction="row" alignItems="center" spacing={1.25} sx={{ mb: 2.5 }}>
          <Chip
            label={`Page ${doc.number}`}
            size="small"
            sx={{ height: 22, fontSize: '0.7rem', fontWeight: 600 }}
          />
          <Typography variant="caption" color="text.secondary">
            {sectionOf(doc.id)} · {doc.minutes} min read
          </Typography>
        </Stack>

        <Markdown source={doc.markdown} docId={doc.id} />

        <Divider sx={{ mt: 6, mb: 3 }} />

        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="center"
          sx={{ mb: 3, flexWrap: 'wrap', gap: 1 }}
        >
          <Link
            href={sourceUrl(`docs/${doc.file}`)}
            target="_blank"
            rel="noopener noreferrer"
            underline="hover"
            variant="body2"
            sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75 }}
          >
            <EditRoundedIcon sx={{ fontSize: 16 }} />
            Edit this page on GitHub
          </Link>
          <Typography variant="caption" color="text.secondary">
            docs/{doc.file}
          </Typography>
        </Stack>

        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          sx={{ '& > *': { flex: 1 } }}
        >
          {previous ? (
            <Button
              component="a"
              href={routes.doc(previous.id)}
              variant="outlined"
              color="inherit"
              startIcon={<ArrowBackRoundedIcon />}
              sx={{ justifyContent: 'flex-start', py: 1.5, borderColor: 'divider' }}
            >
              <Box sx={{ textAlign: 'left', minWidth: 0 }}>
                <Typography variant="caption" color="text.secondary" display="block">
                  Previous
                </Typography>
                <Typography variant="body2" fontWeight={600} noWrap>
                  {previous.title}
                </Typography>
              </Box>
            </Button>
          ) : (
            <Box />
          )}
          {next ? (
            <Button
              component="a"
              href={routes.doc(next.id)}
              variant="outlined"
              color="inherit"
              endIcon={<ArrowForwardRoundedIcon />}
              sx={{ justifyContent: 'flex-end', py: 1.5, borderColor: 'divider' }}
            >
              <Box sx={{ textAlign: 'right', minWidth: 0 }}>
                <Typography variant="caption" color="text.secondary" display="block">
                  Next
                </Typography>
                <Typography variant="body2" fontWeight={600} noWrap>
                  {next.title}
                </Typography>
              </Box>
            </Button>
          ) : (
            <Box />
          )}
        </Stack>
      </Box>

      <Box sx={{ display: { xs: 'none', lg: 'block' } }}>
        <TableOfContents doc={doc} />
      </Box>
    </Box>
  );
}
