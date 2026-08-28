/**
 * Renders every route in Node and asserts the things a bundler cannot see:
 * that no page throws, that the markdown loaded at all, that every
 * table-of-contents entry has a heading with that id, and that every link the
 * renderer emits is either absolute or a route this app serves.
 *
 * Run with `npm run check`. Exits non-zero on any failure.
 */
globalThis.window = {
  location: { hash: '', pathname: '/' },
  localStorage: { getItem: () => null, setItem: () => {} },
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  addEventListener() {},
  removeEventListener() {},
  requestAnimationFrame: () => 0,
  cancelAnimationFrame() {},
  innerHeight: 900,
  scrollY: 0,
};
globalThis.document = {
  getElementById: () => null,
  documentElement: { style: {} },
  activeElement: { tagName: 'BODY' },
  createElement: () => ({ style: {}, setAttribute() {}, appendChild() {} }),
  head: { appendChild() {} },
  body: { scrollHeight: 2000, appendChild() {} },
  querySelectorAll: () => [],
  title: '',
};
Object.defineProperty(globalThis, 'navigator', {
  value: { platform: 'Win32', clipboard: {} },
  configurable: true,
});

import { renderToString } from 'react-dom/server';
import React from 'react';
import App from '../src/App.jsx';
import { docs, sections, flatOrder } from '../src/lib/content.js';
import { parseRoute } from '../src/lib/router.js';
import { renderMarkdown } from '../src/lib/markdown.js';

const failures = [];

function render(hash, label) {
  window.location.hash = hash;
  try {
    const html = renderToString(React.createElement(App));
    return html;
  } catch (error) {
    failures.push(`${label}: ${error.message}`);
    return '';
  }
}

console.log(`docs loaded: ${docs.length}`);
console.log(`sections: ${sections.map((s) => `${s.label}(${s.pages.length})`).join(', ')}`);
console.log(`sidebar order: ${flatOrder.map((d) => d.number).join(' ')}`);

const home = render('#/', 'home');
console.log(`home  ${home.length} chars, hero: ${home.includes('15.7-billion-parameter')}`);
const index = render('#/docs', 'index');
console.log(`index ${index.length} chars, lists all: ${docs.every((d) => index.includes(d.title))}`);
const model = render('#/model', 'model');
console.log(`model ${model.length} chars, has 1,664: ${model.includes('1,664')}`);
const dev = render('#/developer', 'developer');
console.log(`dev   ${dev.length} chars, all links: ${
  ['https://ragavan.vercel.app/', 'https://www.linkedin.com/in/ragavandevp', 'https://gct.ac.in/',
   'https://play.google.com/store/apps/details?id=com.mygcthub', 'https://mygct.org/',
   'https://store.mygct.org/', 'https://slides.mygct.org/'].every((u) => dev.includes(u))
}`);
const missing = render('#/nope', 'missing');
console.log(`404   ${missing.length} chars, says so: ${missing.includes('No such page')}`);

for (const doc of docs) {
  const html = render(`#/docs/${doc.id}`, doc.id);
  const rendered = renderMarkdown(doc.markdown, doc.id);
  const anchors = doc.headings.filter((h) => !rendered.includes(`id="${h.id}"`));
  console.log(
    `  ${doc.id.padEnd(22)} ${String(html.length).padStart(7)} chars · ` +
      `${doc.headings.length} headings · ${anchors.length} unanchored · ` +
      `${(rendered.match(/<table>/g) || []).length} tables · ` +
      `${(rendered.match(/pre class="code-block"/g) || []).length} code blocks`,
  );
  if (anchors.length) failures.push(`${doc.id}: no anchor for ${anchors.map((a) => a.id).join(', ')}`);
}

// Every link the renderer produces must be absolute or a route that exists.
const ids = new Set(docs.map((d) => d.id));
const known = new Set(['#/', '#/docs', '#/model', '#/developer']);
for (const doc of docs) {
  const html = renderMarkdown(doc.markdown, doc.id);
  for (const match of html.matchAll(/href="([^"]+)"/g)) {
    const href = match[1];
    if (/^https?:/.test(href) || known.has(href)) continue;
    const route = parseRoute(href);
    if (route.view === 'doc' && ids.has(route.id)) continue;
    if (href.startsWith('#') && route.view === 'doc') {
      failures.push(`${doc.id}: link to missing page ${href}`);
      continue;
    }
    failures.push(`${doc.id}: unresolved href ${href}`);
  }
}

if (docs.length === 0) failures.push('no markdown loaded: check the glob in src/lib/content.js');

if (failures.length === 0) {
  console.log('\nOK: no failures');
} else {
  console.error(`\nFAILURES:\n${failures.join('\n')}`);
  process.exitCode = 1;
}
