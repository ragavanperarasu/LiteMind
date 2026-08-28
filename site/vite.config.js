import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

// GitHub Pages serves a project site from https://<user>.github.io/<repo>/, so
// every asset URL needs that prefix baked in at build time. Override it with
// BASE_PATH=/ when serving from a domain root or a plain local file server.
const defaultBase = '/LiteMind/';

/**
 * GitHub Pages serves 404.html for any path it has no file for. Every real
 * route here lives after the #, so the only useful answer is to bounce back to
 * the app root and let it route. Generated rather than kept in public/ so the
 * base path can never drift out of step with the one above.
 */
function notFoundRedirect(base) {
  return {
    name: 'litemind-404',
    apply: 'build',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: '404.html',
        source: `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>LiteMind - redirecting</title>
    <script>location.replace('${base}#/' + location.hash.replace(/^#\\/?/, ''));</script>
  </head>
  <body>
    <p>Redirecting to the <a href="${base}">LiteMind documentation</a>.</p>
  </body>
</html>
`,
      });
    },
  };
}

const base = process.env.BASE_PATH || defaultBase;

export default defineConfig({
  base,
  plugins: [react(), notFoundRedirect(base)],
  server: {
    port: 5175,
    // The markdown this site renders lives in ../docs, outside the project
    // root. Vite refuses to serve files above the root unless told otherwise.
    fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
});
