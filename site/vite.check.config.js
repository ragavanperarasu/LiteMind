import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

/**
 * `npm run check` builds scripts/render-check.jsx for Node and runs it. The
 * normal build cannot fail on a page that throws while rendering, or on a
 * table-of-contents link that points at an anchor no heading produced; this
 * can. `noExternal` bundles MUI rather than leaving Node to resolve it.
 */
export default defineConfig({
  plugins: [react()],
  server: { fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] } },
  ssr: { noExternal: true },
  build: {
    ssr: 'scripts/render-check.jsx',
    outDir: 'check-out',
    emptyOutDir: true,
    minify: false,
  },
});
