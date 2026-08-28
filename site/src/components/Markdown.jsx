import { useEffect, useMemo, useRef } from 'react';
import Box from '@mui/material/Box';
import { renderMarkdown } from '../lib/markdown.js';

/**
 * Rendered markdown arrives as a string of HTML, so the copy buttons have to be
 * attached to the DOM afterwards rather than declared in JSX.
 */
function attachCopyButtons(root, timers) {
  root.querySelectorAll('pre.code-block').forEach((pre) => {
    const wrap = document.createElement('div');
    wrap.className = 'code-wrap';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        button.textContent = 'Copied';
        button.dataset.copied = 'true';
      } catch {
        // Clipboard access is refused outside a secure context. Say so rather
        // than leaving a button that looks broken.
        button.textContent = 'Select and copy';
      }
      timers.push(
        window.setTimeout(() => {
          button.textContent = 'Copy';
          delete button.dataset.copied;
        }, 1600),
      );
    });
    wrap.appendChild(button);
  });
}

export default function Markdown({ source, docId, sx }) {
  const container = useRef(null);
  const html = useMemo(() => renderMarkdown(source, docId), [source, docId]);

  useEffect(() => {
    const root = container.current;
    if (!root) return undefined;
    const timers = [];
    attachCopyButtons(root, timers);
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [html]);

  return (
    <Box
      ref={container}
      className="markdown-body"
      sx={sx}
      // The source is this repository's own markdown, inlined at build time.
      // Nothing user-supplied reaches this renderer.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
