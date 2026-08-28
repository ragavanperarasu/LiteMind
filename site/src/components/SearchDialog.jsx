import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Chip,
  Dialog,
  InputBase,
  List,
  ListItemButton,
  Stack,
  Typography,
} from '@mui/material';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import { docs } from '../lib/content.js';
import { routes } from '../lib/router.js';

const LIMIT = 8;

/** The whole corpus is under 2,000 lines, so a linear scan is instant. */
function search(query) {
  const needle = query.trim().toLowerCase();
  if (needle.length < 2) return [];

  const results = [];
  for (const doc of docs) {
    const haystack = doc.text.toLowerCase();
    const titleHit = doc.rawTitle.toLowerCase().includes(needle);
    const heading = doc.headings.find((item) => item.text.toLowerCase().includes(needle));
    const at = haystack.indexOf(needle);
    if (!titleHit && !heading && at < 0) continue;

    let hits = 0;
    for (let index = haystack.indexOf(needle); index >= 0; index = haystack.indexOf(needle, index + needle.length)) {
      hits += 1;
    }

    results.push({
      doc,
      heading: heading || null,
      snippet: at >= 0 ? excerpt(doc.text, at, needle.length) : null,
      score: (titleHit ? 100 : 0) + (heading ? 40 : 0) + Math.min(hits, 10),
    });
  }

  return results.sort((a, b) => b.score - a.score).slice(0, LIMIT);
}

function excerpt(text, at, length) {
  const start = Math.max(0, at - 55);
  const end = Math.min(text.length, at + length + 85);
  return {
    before: (start > 0 ? '…' : '') + text.slice(start, at),
    match: text.slice(at, at + length),
    after: text.slice(at + length, end) + (end < text.length ? '…' : ''),
  };
}

export default function SearchDialog({ open, onClose }) {
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef(null);
  const results = useMemo(() => search(query), [query]);

  useEffect(() => {
    if (open) {
      setQuery('');
      setCursor(0);
    }
  }, [open]);

  useEffect(() => setCursor(0), [query]);

  const go = (result) => {
    window.location.hash = routes.doc(result.doc.id, result.heading ? result.heading.id : null);
    onClose();
  };

  const onKeyDown = (event) => {
    if (results.length === 0) return;
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setCursor((value) => (value + 1) % results.length);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setCursor((value) => (value - 1 + results.length) % results.length);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      go(results[cursor]);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="sm"
      slotProps={{ paper: { sx: { position: 'fixed', top: 72, m: 0, borderRadius: 3 } } }}
      onTransitionEnter={() => inputRef.current?.focus()}
    >
      <Stack
        direction="row"
        alignItems="center"
        spacing={1.5}
        sx={{ px: 2.5, py: 1.75, borderBottom: 1, borderColor: 'divider' }}
      >
        <SearchRoundedIcon sx={{ color: 'text.secondary' }} />
        <InputBase
          inputRef={inputRef}
          autoFocus
          fullWidth
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Search the documentation…"
          sx={{ fontSize: '1.02rem' }}
        />
        <Chip label="Esc" size="small" variant="outlined" sx={{ height: 22, fontSize: '0.7rem' }} />
      </Stack>

      {query.trim().length < 2 ? (
        <Box sx={{ px: 2.5, py: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Type at least two characters. Try <b>expert cache</b>, <b>YaRN</b>, <b>tok/s</b> or{' '}
            <b>prerequisites</b>.
          </Typography>
        </Box>
      ) : results.length === 0 ? (
        <Box sx={{ px: 2.5, py: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Nothing in the {docs.length} pages matches “{query.trim()}”.
          </Typography>
        </Box>
      ) : (
        <List disablePadding sx={{ maxHeight: '58vh', overflowY: 'auto', py: 1 }}>
          {results.map((result, index) => (
            <ListItemButton
              key={result.doc.id}
              selected={index === cursor}
              onMouseEnter={() => setCursor(index)}
              onClick={() => go(result)}
              sx={{ px: 2.5, py: 1.25, display: 'block' }}
            >
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography sx={{ fontWeight: 650, fontSize: '0.92rem' }}>
                  {result.doc.title}
                </Typography>
                {result.heading && (
                  <Typography variant="caption" color="primary.main">
                    › {result.heading.text}
                  </Typography>
                )}
              </Stack>
              {result.snippet && (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mt: 0.25, fontSize: '0.82rem' }}
                >
                  {result.snippet.before}
                  <Box component="mark" sx={{ bgcolor: 'transparent', color: 'warning.main', fontWeight: 650 }}>
                    {result.snippet.match}
                  </Box>
                  {result.snippet.after}
                </Typography>
              )}
            </ListItemButton>
          ))}
        </List>
      )}
    </Dialog>
  );
}
