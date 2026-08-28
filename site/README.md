# The documentation website

A static React site that renders the markdown in [`docs/`](../docs) as a
browsable, searchable website, published to GitHub Pages at
**https://ragavanperarasu.github.io/LiteMind/**.

There is no second copy of the documentation. Vite inlines `docs/*.md` at build
time, so editing a page in `docs/` and pushing to `main` republishes the site
with it. The markdown still reads correctly on GitHub — links such as
`[src/Gemm.cpp](../src/Gemm.cpp)` are rewritten to absolute GitHub URLs while
the site is being rendered, and links between pages become routes.

## Running it

```bash
cd site
npm install
npm run dev        # http://localhost:5175
```

| Script | What it does |
|---|---|
| `npm run dev` | Development server with hot reload |
| `npm run check` | Renders every route in Node and fails on a broken page or link |
| `npm run build` | Static bundle into `site/dist` |
| `npm run preview` | Serves that bundle at http://localhost:4173 |

`npm run build` bakes the base path `/LiteMind/` into every asset URL, because
that is where a GitHub project site lives. To serve it from a domain root
instead, build with `BASE_PATH=/ npm run build`.

## What `npm run check` catches

A bundler is happy with a page that throws at render time, with markdown that
silently failed to load, and with a contents entry pointing at an anchor no
heading ever produced. `scripts/render-check.jsx` renders all
routes in Node with `react-dom/server` and asserts:

- no route throws
- the markdown loaded at all — an empty glob is otherwise invisible
- every table-of-contents id matches a heading id the renderer emitted
- every link the renderer produces is either absolute or a route that exists

It runs in CI before the deploy, so a broken link cannot reach the published
site. It found two real bugs the first time it ran: the markdown glob was one
directory off, and the five headings in
[page 2](../docs/02-loading.md) that contain links were being slugified
differently by the contents and by the anchor plugin.

## Publishing

[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) builds and
deploys on every push to `main` that touches `docs/`, `site/` or the workflow
itself. It needs one setting on the repository, once:

**Settings → Pages → Build and deployment → Source → GitHub Actions.**

There is no `gh-pages` branch and nothing is committed to the repository by the
workflow.

## How it is put together

| File | Responsibility |
|---|---|
| `src/lib/content.js` | Loads `docs/*.md`, extracts titles, headings, blurbs, reading time |
| `src/lib/markdown.js` | markdown-it: highlighting, heading anchors, link rewriting |
| `src/lib/router.js` | Hash routing — `#/docs/04-attention/the-kv-cache` |
| `src/components/` | App bar, sidebar, search dialog, contents rail, markdown host |
| `src/views/` | Home, index, a documentation page, model spec, 404 |

Routing is by hash on purpose. GitHub Pages serves static files and knows
nothing about client-side routes, so `/docs/04-attention` would 404 on reload
or on a shared link. Nothing after a `#` is ever sent to the server.
