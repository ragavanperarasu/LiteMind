import { useEffect, useState } from 'react';
import { Box, Link, Stack, Typography } from '@mui/material';
import { routes } from '../lib/router.js';

/**
 * The heading nearest the top of the viewport, recomputed on scroll. An
 * IntersectionObserver would fire on entry and exit rather than telling us
 * which heading the reader is currently under, which is the question here.
 */
function useActiveHeading(headings) {
  const [active, setActive] = useState(null);

  useEffect(() => {
    if (headings.length === 0) return undefined;
    let frame = 0;

    const measure = () => {
      frame = 0;
      let current = headings[0].id;
      for (const heading of headings) {
        const element = document.getElementById(heading.id);
        if (element && element.getBoundingClientRect().top <= 120) current = heading.id;
      }
      // At the very bottom the last heading may never reach the line.
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 8) {
        current = headings[headings.length - 1].id;
      }
      setActive(current);
    };

    const onScroll = () => {
      if (frame === 0) frame = window.requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [headings]);

  return active;
}

export default function TableOfContents({ doc }) {
  const active = useActiveHeading(doc.headings);
  if (doc.headings.length < 2) return null;

  return (
    <Box component="nav" aria-label="On this page" sx={{ position: 'sticky', top: 96 }}>
      <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: '0.68rem' }}>
        On this page
      </Typography>
      <Stack
        spacing={0.25}
        sx={{ mt: 1, borderLeft: 1, borderColor: 'divider', maxHeight: '70vh', overflowY: 'auto' }}
      >
        {doc.headings.map((heading) => {
          const current = heading.id === active;
          return (
            <Link
              key={heading.id}
              href={routes.doc(doc.id, heading.id)}
              underline="none"
              sx={{
                pl: heading.level === 3 ? 3 : 1.75,
                pr: 1,
                py: 0.4,
                ml: '-1px',
                borderLeft: '2px solid',
                borderColor: current ? 'primary.main' : 'transparent',
                color: current ? 'text.primary' : 'text.secondary',
                fontWeight: current ? 600 : 400,
                fontSize: heading.level === 3 ? '0.78rem' : '0.82rem',
                lineHeight: 1.45,
                '&:hover': { color: 'text.primary' },
              }}
            >
              {heading.text}
            </Link>
          );
        })}
      </Stack>
    </Box>
  );
}
