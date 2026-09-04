import GlobalStyles from '@mui/material/GlobalStyles';
import { useTheme, alpha } from '@mui/material/styles';
import { codeTokens } from '../theme.js';

const MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

/**
 * Rendered markdown is injected as HTML, so it cannot carry MUI's `sx`. These
 * are the styles for it, written once against the active theme so light and
 * dark stay in step.
 */
export default function MarkdownStyles() {
  const theme = useTheme();
  const code = codeTokens[theme.palette.mode];
  const { palette } = theme;

  return (
    <GlobalStyles
      styles={{
        '.markdown-body': {
          color: palette.text.primary,
          fontSize: '1rem',
          lineHeight: 1.72,
          overflowWrap: 'break-word',
        },
        '.markdown-body > *:first-of-type': { marginTop: 0 },
        '.markdown-body p, .markdown-body li': { maxWidth: '74ch' },

        '.markdown-body h1': {
          fontSize: 'clamp(1.9rem, 1.4rem + 1.6vw, 2.6rem)',
          fontWeight: 700,
          letterSpacing: '-0.03em',
          lineHeight: 1.15,
          margin: '0 0 28px',
        },
        '.markdown-body h2': {
          fontSize: '1.45rem',
          fontWeight: 700,
          letterSpacing: '-0.02em',
          margin: '56px 0 16px',
          paddingTop: 20,
          borderTop: `1px solid ${palette.divider}`,
        },
        // A `---` just before a heading would draw the same line twice.
        // The charts are authored at a fixed width; they scale down rather than
      // pushing the page sideways on a phone.
      '.markdown-body .doc-figure': {
        margin: '28px 0',
        padding: 0,
        overflowX: 'auto',
      },
      '.markdown-body .doc-figure svg': {
        display: 'block',
        width: '100%',
        maxWidth: 760,
        // Below this the 10px annotations shrink to nothing, so on a phone the
        // figure scrolls inside its own box rather than becoming unreadable.
        minWidth: 620,
        height: 'auto',
      },
      '.markdown-body .missing-figure': {
        color: theme.palette.error.main,
        fontFamily: 'monospace',
        fontSize: '0.85rem',
      },

      '.markdown-body hr + h2': { borderTop: 0, paddingTop: 0, marginTop: 24 },
        '.markdown-body h3': {
          fontSize: '1.1rem',
          fontWeight: 650,
          letterSpacing: '-0.01em',
          margin: '36px 0 12px',
        },
        '.markdown-body h4': { fontSize: '1rem', fontWeight: 650, margin: '28px 0 10px' },

        '.markdown-body .heading-anchor': {
          marginLeft: 10,
          color: palette.text.secondary,
          opacity: 0,
          textDecoration: 'none',
          fontWeight: 400,
          transition: 'opacity .15s',
        },
        '.markdown-body h2:hover .heading-anchor, .markdown-body h3:hover .heading-anchor': {
          opacity: 0.65,
        },
        '.markdown-body .heading-anchor:hover': { opacity: 1, color: palette.primary.main },

        '.markdown-body p': { margin: '0 0 18px' },
        '.markdown-body ul, .markdown-body ol': { margin: '0 0 18px', paddingLeft: 26 },
        '.markdown-body li': { margin: '0 0 8px' },
        '.markdown-body li > ul, .markdown-body li > ol': { margin: '8px 0 0' },
        '.markdown-body strong': { fontWeight: 650, color: palette.text.primary },
        '.markdown-body hr': {
          border: 0,
          borderTop: `1px solid ${palette.divider}`,
          margin: '40px 0',
        },

        '.markdown-body a': {
          color: palette.primary.main,
          textDecoration: 'none',
          borderBottom: `1px solid ${alpha(palette.primary.main, 0.35)}`,
        },
        '.markdown-body a:hover': { borderBottomColor: palette.primary.main },

        '.markdown-body blockquote': {
          margin: '0 0 20px',
          padding: '4px 0 4px 20px',
          borderLeft: `3px solid ${alpha(palette.primary.main, 0.5)}`,
          color: palette.text.secondary,
        },

        // Inline code
        '.markdown-body :not(pre) > code': {
          fontFamily: MONO,
          fontSize: '0.85em',
          padding: '0.15em 0.42em',
          borderRadius: 6,
          background: alpha(palette.primary.main, palette.mode === 'dark' ? 0.13 : 0.09),
          color: palette.mode === 'dark' ? '#c0caf5' : '#233',
          border: `1px solid ${code.border}`,
          whiteSpace: 'nowrap',
        },

        // Fenced blocks. Most of them are ASCII diagrams and terminal
        // transcripts, so the box has to be honest about width and scroll.
        '.code-wrap': { position: 'relative', margin: '0 0 22px' },
        '.markdown-body pre.code-block': {
          margin: 0,
          padding: '18px 20px',
          background: code.surface,
          border: `1px solid ${code.border}`,
          borderRadius: 10,
          overflowX: 'auto',
          fontFamily: MONO,
          fontSize: '0.83rem',
          lineHeight: 1.62,
          color: code.plain,
          tabSize: 4,
        },
        '.markdown-body pre.code-block code': {
          fontFamily: 'inherit',
          background: 'none',
          padding: 0,
          border: 0,
          whiteSpace: 'pre',
        },
        '.code-wrap .copy-button': {
          position: 'absolute',
          top: 8,
          right: 8,
          padding: '4px 10px',
          fontFamily: 'inherit',
          fontSize: '0.72rem',
          fontWeight: 600,
          letterSpacing: '0.03em',
          color: palette.text.secondary,
          background: alpha(palette.background.paper, 0.92),
          border: `1px solid ${code.border}`,
          borderRadius: 7,
          cursor: 'pointer',
          opacity: 0,
          transition: 'opacity .15s, color .15s',
        },
        '.code-wrap:hover .copy-button, .code-wrap .copy-button:focus-visible': { opacity: 1 },
        '.code-wrap .copy-button:hover': { color: palette.primary.main },
        '.code-wrap .copy-button[data-copied="true"]': { opacity: 1, color: palette.success.main },

        // Tables carry most of the reference material in these docs.
        '.table-scroll': {
          overflowX: 'auto',
          margin: '0 0 24px',
          border: `1px solid ${palette.divider}`,
          borderRadius: 10,
        },
        '.markdown-body table': { borderCollapse: 'collapse', width: '100%', fontSize: '0.92rem' },
        '.markdown-body th': {
          textAlign: 'left',
          fontWeight: 650,
          padding: '11px 16px',
          background: alpha(palette.text.primary, palette.mode === 'dark' ? 0.05 : 0.035),
          borderBottom: `1px solid ${palette.divider}`,
          whiteSpace: 'nowrap',
        },
        '.markdown-body td': {
          padding: '11px 16px',
          borderBottom: `1px solid ${palette.divider}`,
          verticalAlign: 'top',
        },
        '.markdown-body tbody tr:last-child td': { borderBottom: 0 },
        '.markdown-body td:empty': { padding: 0 },

        // Syntax highlighting, themed here rather than by importing one of
        // highlight.js's stylesheets, so the toggle switches both at once.
        '.hljs-keyword, .hljs-built_in, .hljs-literal, .hljs-selector-tag': { color: code.keyword },
        '.hljs-string, .hljs-regexp, .hljs-addition': { color: code.string },
        '.hljs-number, .hljs-symbol, .hljs-bullet': { color: code.number },
        '.hljs-comment, .hljs-quote': { color: code.comment, fontStyle: 'italic' },
        '.hljs-title, .hljs-title\\.function_, .hljs-section, .hljs-name': { color: code.title },
        '.hljs-attr, .hljs-attribute, .hljs-variable, .hljs-template-variable': { color: code.attr },
        '.hljs-meta, .hljs-type, .hljs-params': { color: code.meta },
        '.hljs-emphasis': { fontStyle: 'italic' },
        '.hljs-strong': { fontWeight: 600 },

        // Scrollbars, so the many horizontally scrolling blocks look intentional.
        '*::-webkit-scrollbar': { width: 10, height: 10 },
        '*::-webkit-scrollbar-thumb': {
          background: alpha(palette.text.secondary, 0.3),
          borderRadius: 8,
          border: `2px solid transparent`,
          backgroundClip: 'content-box',
        },
        '*::-webkit-scrollbar-thumb:hover': { background: alpha(palette.text.secondary, 0.5) },
        '*::-webkit-scrollbar-track': { background: 'transparent' },
      }}
    />
  );
}
