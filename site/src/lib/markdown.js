import MarkdownIt from 'markdown-it';
import anchor from 'markdown-it-anchor';
import hljs from 'highlight.js/lib/core';
import bash from 'highlight.js/lib/languages/bash';
import cpp from 'highlight.js/lib/languages/cpp';
import json from 'highlight.js/lib/languages/json';
import powershell from 'highlight.js/lib/languages/powershell';
import { slugify } from './slug.js';
import { sourceUrl } from './site.js';

// Only the four languages the documentation actually fences. Registering the
// whole library would triple the bundle for nothing.
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('cpp', cpp);
hljs.registerLanguage('json', json);
hljs.registerLanguage('powershell', powershell);

const md = new MarkdownIt({
  // The docs are plain markdown. Disabling raw HTML means a stray `<path>` in
  // prose renders as text instead of being swallowed as a tag.
  html: false,
  linkify: true,
  breaks: false,
  highlight(code, language) {
    const body =
      language && hljs.getLanguage(language)
        ? hljs.highlight(code, { language, ignoreIllegals: true }).value
        : md.utils.escapeHtml(code);
    const label = language ? ` data-language="${md.utils.escapeHtml(language)}"` : '';
    // Returning a complete <pre> tells markdown-it to use this verbatim.
    return `<pre class="code-block"${label}><code class="hljs">${body}</code></pre>`;
  },
});

md.use(anchor, {
  slugify,
  level: [2, 3],
  permalink: anchor.permalink.linkInsideHeader({
    symbol: '#',
    class: 'heading-anchor',
    placement: 'after',
    ariaHidden: false,
  }),
});

const DOC_LINK = /^(\d{2}-[a-z0-9-]+)\.md(?:#(.*))?$/;

/**
 * The same markdown has to work in two places. On GitHub `[the kernels](../src/Gemm.cpp)`
 * resolves against the repository; here it has to become an absolute link to
 * GitHub, and a link to a sibling page has to become a route in this app.
 */
function rewrite(href, env) {
  if (!href) return { href: '#/', external: false };
  if (/^(https?:|mailto:)/i.test(href)) return { href, external: true };

  if (href.startsWith('#')) {
    // An in-page anchor: keep the reader on this page, at that heading.
    const anchorId = href.slice(1);
    return { href: env.docId ? `#/docs/${env.docId}/${anchorId}` : href, external: false };
  }

  const doc = DOC_LINK.exec(href);
  if (doc) {
    return { href: `#/docs/${doc[1]}${doc[2] ? `/${doc[2]}` : ''}`, external: false };
  }

  if (href === 'README.md' || href === './README.md') return { href: '#/docs', external: false };
  if (href === 'model-info.json') return { href: '#/model', external: false };

  // Anything else points out of docs/ and into the repository itself.
  if (href.startsWith('../')) return { href: sourceUrl(href.slice(3)), external: true };
  if (href.endsWith('.pptx')) return { href: sourceUrl(`docs/${href}`), external: true };
  return { href: sourceUrl(`docs/${href}`), external: true };
}

md.renderer.rules.link_open = (tokens, index, options, env, self) => {
  const token = tokens[index];
  const { href, external } = rewrite(token.attrGet('href'), env);
  token.attrSet('href', href);
  if (external) {
    token.attrSet('target', '_blank');
    token.attrSet('rel', 'noopener noreferrer');
    token.attrJoin('class', 'external-link');
  }
  return self.renderToken(tokens, index, options);
};

// Wide tables scroll inside their own box rather than widening the page.
md.renderer.rules.table_open = () => '<div class="table-scroll"><table>';
md.renderer.rules.table_close = () => '</table></div>';

// Markdown has no headerless table, so the docs write `| | |` and get an empty
// header row. Rendered, that is a blank grey band above the table.
const EMPTY_HEADER = /<thead>\s*<tr>\s*(?:<th[^>]*>\s*<\/th>\s*)+<\/tr>\s*<\/thead>/g;

/** Render one document. `docId` lets in-page anchors resolve to a route. */
export function renderMarkdown(markdown, docId) {
  return md.render(markdown, { docId }).replace(EMPTY_HEADER, '');
}

/** Inline markdown only - for a JSON `_note` field that contains backticks. */
export function renderInline(text) {
  return md.renderInline(String(text), {});
}

export default renderMarkdown;
