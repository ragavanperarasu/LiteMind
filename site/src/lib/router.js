import { useEffect, useState } from 'react';

/**
 * Hash routing, deliberately. GitHub Pages serves static files and knows
 * nothing about client-side routes, so a real path such as /docs/04-attention
 * would 404 on reload or on a shared link. Everything after the # never
 * reaches the server.
 */
export function parseRoute(hash) {
  const path = String(hash || '').replace(/^#/, '').replace(/^\/+/, '');
  const parts = path.split('/').filter(Boolean);

  if (parts.length === 0) return { view: 'home' };
  if (parts[0] === 'model') return { view: 'model' };
  if (parts[0] === 'developer') return { view: 'developer' };
  if (parts[0] === 'docs') {
    if (parts.length === 1) return { view: 'index' };
    return { view: 'doc', id: parts[1], anchor: parts[2] || null };
  }
  return { view: 'missing', path };
}

export function useRoute() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));

  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);

  return route;
}

export function navigate(to) {
  const target = to.startsWith('#') ? to : `#${to}`;
  if (window.location.hash === target) return;
  window.location.hash = target;
}

export const routes = {
  home: '#/',
  index: '#/docs',
  model: '#/model',
  developer: '#/developer',
  doc: (id, anchor) => `#/docs/${id}${anchor ? `/${anchor}` : ''}`,
};
