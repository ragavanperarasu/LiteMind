import { slugify } from './slug.js';

/**
 * The website has no content of its own: it renders the same markdown that
 * lives in docs/ and is read on GitHub. Vite inlines the files at build time,
 * so the published site is static and there is never a second copy to keep in
 * step.
 */
const sources = import.meta.glob('../../../docs/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
});

/** `## Heading` lines, but not the ones inside a fenced code block. */
function extractHeadings(markdown) {
  const headings = [];
  let fenced = false;
  for (const line of markdown.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) {
      fenced = !fenced;
      continue;
    }
    if (fenced) continue;
    const match = /^(#{2,3})\s+(.+?)\s*$/.exec(line);
    if (!match) continue;
    // markdown-it-anchor slugifies the *rendered* heading, in which a link
    // has collapsed to its label. Slugify the same thing or the contents
    // links point at nothing.
    const flattened = match[2].replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
    const text = flattened
      .replace(/`/g, '')
      .replace(/\*\*/g, '')
      // "Json - src/Json.cpp" reads as "Json" in a 230px rail.
      .replace(/\s+[\u2014-]\s+\S+\.(?:cpp|hpp|py|json|ps1|mjs|jsx)$/, '')
      .trim();
    headings.push({ level: match[1].length, text, id: slugify(flattened) });
  }
  return headings;
}

/** Plain-ish text for the search index: no fences, no table pipes, no markup. */
function searchableText(markdown) {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[`*_>|]/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[ \t]+/g, ' ');
}

/**
 * The opening paragraph, used as the blurb on the index. Several pages open
 * with a source-file byline or a code block rather than prose, so this takes
 * the first block that actually reads like a sentence.
 */
function firstParagraph(markdown) {
  const blocks = [];
  let fenced = false;
  let current = [];
  for (const line of markdown.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) {
      fenced = !fenced;
      if (current.length > 0) blocks.push(current);
      current = [];
      continue;
    }
    if (fenced) continue;
    if (line.trim() === '') {
      if (current.length > 0) blocks.push(current);
      current = [];
      continue;
    }
    current.push(line.trim());
  }
  if (current.length > 0) blocks.push(current);

  for (const block of blocks) {
    const head = block[0];
    if (/^(#|\||>|[-*+]\s|\d+\.\s)/.test(head)) continue;
    const text = block
      .join(' ')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/[`*_]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    // A byline such as "src/Tokenizer.cpp - include/Tokenizer.hpp" is short
    // and has no sentence in it.
    if (text.split(' ').length >= 10 && text.includes('.')) return text;
  }
  return '';
}

/**
 * A few pages open with a code block or a source byline rather than a lead
 * paragraph, so there is nothing sensible to lift. Their blurbs are written
 * here instead of bending the extraction above out of shape.
 */
const blurbs = {
  '10-cli-and-config':
    'One command for every run, what litemind.json holds, every flag, and which of the two wins.',
  '11-testing':
    'Nine executables, no framework. A synthetic model that runs in seconds, and logits checked digit by digit against the reference implementation.',
  '14-web-ui':
    'The JSON event protocol the engine speaks, the Node backend that spawns it, and the React interface that draws the routing as it happens.',
};

function build(path, markdown) {
  const file = path.slice(path.lastIndexOf('/') + 1);
  const id = file.replace(/\.md$/, '');
  const firstHeading = /^#\s+(.+)$/m.exec(markdown);
  const rawTitle = firstHeading ? firstHeading[1].trim() : id;
  // Pages are titled "4. Attention" or "14 \u00b7 The web interface"; the
  // sidebar wants the number and the words in separate columns. The chip
  // always uses the two-digit file prefix so the column lines up.
  const numbered = /^\d+\s*[.\u00b7]\s*(.+)$/.exec(rawTitle);
  const words = searchableText(markdown).split(/\s+/).filter(Boolean).length;
  const summary = blurbs[id] || firstParagraph(markdown);

  return {
    id,
    file,
    order: Number.parseInt(id.slice(0, 2), 10),
    number: id.slice(0, 2),
    title: numbered ? numbered[1] : rawTitle,
    rawTitle,
    markdown,
    headings: extractHeadings(markdown),
    summary,
    text: searchableText(markdown),
    words,
    minutes: Math.max(1, Math.round(words / 220)),
  };
}

export const docs = Object.entries(sources)
  .filter(([path]) => !path.endsWith('/README.md'))
  .map(([path, markdown]) => build(path, markdown))
  .sort((a, b) => a.order - b.order);

export const docsById = new Map(docs.map((doc) => [doc.id, doc]));

/**
 * Reading order is the file order; browsing order is by subject. The sidebar
 * uses these groups, which is why a couple of numbers sit out of sequence.
 */
const groups = [
  { label: 'Start here', ids: ['00-setup', '01-overview'] },
  { label: 'Reading the checkpoint', ids: ['02-loading', '03-tokenizer'] },
  {
    label: 'The forward pass',
    ids: ['04-attention', '05-rope', '06-mixture-of-experts', '07-expert-cache', '08-kernels'],
  },
  { label: 'Producing text', ids: ['09-sampling', '13-chat-template'] },
  { label: 'Running it', ids: ['10-cli-and-config', '14-web-ui'] },
  { label: 'Evidence', ids: ['11-testing', '12-performance'] },
];

export const sections = groups
  .map((group) => ({
    label: group.label,
    pages: group.ids.map((id) => docsById.get(id)).filter(Boolean),
  }))
  .filter((section) => section.pages.length > 0);

// Any page added to docs/ later still appears, even before it is filed into a
// group above, rather than silently vanishing from the sidebar.
const grouped = new Set(sections.flatMap((section) => section.pages.map((page) => page.id)));
const ungrouped = docs.filter((doc) => !grouped.has(doc.id));
if (ungrouped.length > 0) sections.push({ label: 'More', pages: ungrouped });

/** Sidebar order, which is what "previous" and "next" should follow. */
export const flatOrder = sections.flatMap((section) => section.pages);

export function neighbours(id) {
  const index = flatOrder.findIndex((doc) => doc.id === id);
  if (index < 0) return { previous: null, next: null };
  return {
    previous: index > 0 ? flatOrder[index - 1] : null,
    next: index < flatOrder.length - 1 ? flatOrder[index + 1] : null,
  };
}
