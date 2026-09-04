import http from 'node:http';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { findExecutable, generate, readSettings, withoutComments } from './litemind.mjs';
import { sampleUsage } from './usage.mjs';

/**
 * The HTTP front end.
 *
 * Written against node:http with no packages, because the engine it serves has
 * no dependencies either and the two lines of routing here do not justify
 * breaking that. The browser bundle is a different matter - Material UI is a
 * dependency by definition - but nothing on this side is.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(here, '..', '..');
const webRoot = path.join(here, '..', 'web', 'dist');
const port = Number(process.env.PORT ?? 5174);

/**
 * One generation at a time.
 *
 * The engine already spreads a single token across every core; running two
 * prompts at once would halve the threads available to each and make both
 * slower than running them in turn. A second request waits rather than
 * competing.
 */
let active = null;

const send = (response, status, body, type = 'application/json') => {
  const payload = type === 'application/json' ? JSON.stringify(body) : body;
  response.writeHead(status, {
    'Content-Type': type === 'application/json' ? 'application/json; charset=utf-8' : type,
    'Cache-Control': 'no-store',
  });
  response.end(payload);
};

const readBody = (request) =>
  new Promise((resolve, reject) => {
    let text = '';
    request.on('data', (chunk) => {
      text += chunk;
      // A prompt is text typed by a person; anything this large is a mistake or
      // an attempt at one, and is refused before it is parsed.
      if (text.length > 1_000_000) {
        reject(new Error('The request body is too large.'));
        request.destroy();
      }
    });
    request.on('end', () => resolve(text));
    request.on('error', reject);
  });

/**
 * Checks a conversation sent by the browser before it reaches the engine.
 *
 * The engine refuses a malformed transcript rather than repairing one, and it
 * is better to say why here than to let a 500 come back from a process the
 * reader cannot see. The cap is on turns rather than bytes because the body
 * limit already bounds the size, and a conversation this long has outgrown the
 * context anyway.
 */
const MAX_HISTORY_MESSAGES = 200;

function validateHistory(value) {
  if (value === undefined || value === null) return { history: [] };
  if (!Array.isArray(value)) return { error: 'The conversation must be an array.' };
  if (value.length > MAX_HISTORY_MESSAGES) {
    return { error: `The conversation is longer than ${MAX_HISTORY_MESSAGES} messages.` };
  }
  const history = [];
  for (const [index, message] of value.entries()) {
    if (!message || typeof message !== 'object') {
      return { error: `Message ${index} of the conversation is not an object.` };
    }
    const { role, content } = message;
    if (role !== 'user' && role !== 'assistant' && role !== 'system') {
      return { error: `Message ${index} has an unknown role: ${JSON.stringify(role)}.` };
    }
    if (typeof content !== 'string') {
      return { error: `Message ${index} has no text.` };
    }
    // Only the two fields the engine reads are forwarded; anything the
    // interface keeps alongside a turn - timings, stop reasons - is its own.
    history.push({ role, content });
  }
  return { history };
}

function describeEnvironment() {
  const executable = findExecutable(repositoryRoot);
  let settings = {};
  let settingsError = null;
  try {
    settings = withoutComments(readSettings(repositoryRoot));
  } catch (cause) {
    settingsError = cause.message;
  }
  const modelDirectory = path.resolve(repositoryRoot, settings.model ?? 'models');
  return {
    executable,
    settings,
    settingsError,
    modelDirectory,
    ready: Boolean(executable) && existsSync(path.join(modelDirectory, 'config.json')),
  };
}

async function handleGenerate(request, response) {
  const { executable, settings, modelDirectory, ready } = describeEnvironment();

  if (!ready) {
    send(response, 503, {
      error: !executable
        ? 'LiteMind is not built. Run scripts/build.ps1 first.'
        : `No config.json in ${modelDirectory}.`,
    });
    return;
  }
  if (active) {
    send(response, 409, { error: 'A prompt is already running. Wait for it to finish.' });
    return;
  }

  let body;
  try {
    body = JSON.parse(await readBody(request));
  } catch (cause) {
    send(response, 400, { error: `Malformed request: ${cause.message}` });
    return;
  }

  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';
  if (prompt === '') {
    send(response, 400, { error: 'The prompt is empty.' });
    return;
  }

  const { history, error: historyError } = validateHistory(body.history);
  if (historyError) {
    send(response, 400, { error: historyError });
    return;
  }

  // Server-sent events: the reply arrives a token at a time over seconds, and
  // this is the one streaming transport that needs nothing on either side.
  response.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const write = (event) => response.write(`data: ${JSON.stringify(event)}\n\n`);

  const run = generate({
    executable,
    repositoryRoot,
    modelDirectory,
    prompt,
    history,
    settings: { ...settings, ...withoutComments(body.settings ?? {}) },
    onEvent: write,
  });
  active = run;

  // A closed tab should not leave a process burning eight cores for another
  // four hundred tokens nobody will read.
  request.on('close', () => {
    if (active === run && !response.writableEnded) run.cancel();
  });

  await run.finished;
  active = null;
  response.end();
}

function handleSettingsWrite(request, response) {
  readBody(request)
    .then((text) => {
      const incoming = JSON.parse(text);
      const file = path.join(repositoryRoot, 'litemind.json');
      // The file is edited by hand as well, and its comment keys are the
      // documentation. They are preserved rather than written over.
      const existing = existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : {};
      const merged = { ...existing, ...withoutComments(incoming) };
      writeFileSync(file, `${JSON.stringify(merged, null, 2)}\n`, 'utf8');
      send(response, 200, withoutComments(merged));
    })
    .catch((cause) => send(response, 400, { error: cause.message }));
}

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
  '.woff2': 'font/woff2',
};

function serveStatic(url, response) {
  if (!existsSync(webRoot)) {
    send(
      response,
      503,
      '<h1>The interface is not built.</h1><p>Run <code>npm install &amp;&amp; npm run build</code> in <code>ui/web</code>, or use the Vite dev server.</p>',
      'text/html',
    );
    return;
  }
  const requested = url === '/' ? '/index.html' : url;
  const file = path.join(webRoot, path.normalize(requested).replace(/^(\.\.[/\\])+/, ''));
  // Anything outside the build directory is a traversal attempt, and single
  // page routing means an unknown path is a route, not a file.
  const target = file.startsWith(webRoot) && existsSync(file) && !file.endsWith(path.sep)
    ? file
    : path.join(webRoot, 'index.html');
  send(response, 200, readFileSync(target), CONTENT_TYPES[path.extname(target)] ?? 'application/octet-stream');
}

const server = http.createServer((request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);

  if (url.pathname === '/api/health') {
    const environment = describeEnvironment();
    send(response, 200, {
      ready: environment.ready,
      executable: environment.executable,
      modelDirectory: environment.modelDirectory,
      settingsError: environment.settingsError,
      busy: Boolean(active),
    });
    return;
  }
  if (url.pathname === '/api/usage') {
    send(response, 200, { ...sampleUsage(), busy: Boolean(active) });
    return;
  }
  if (url.pathname === '/api/settings' && request.method === 'GET') {
    const environment = describeEnvironment();
    send(response, environment.settingsError ? 500 : 200,
      environment.settingsError ? { error: environment.settingsError } : environment.settings);
    return;
  }
  if (url.pathname === '/api/settings' && request.method === 'PUT') {
    handleSettingsWrite(request, response);
    return;
  }
  if (url.pathname === '/api/generate' && request.method === 'POST') {
    handleGenerate(request, response).catch((cause) => {
      if (!response.headersSent) send(response, 500, { error: cause.message });
      else response.end();
    });
    return;
  }
  if (url.pathname === '/api/cancel' && request.method === 'POST') {
    active?.cancel();
    send(response, 200, { cancelled: Boolean(active) });
    return;
  }
  if (url.pathname.startsWith('/api/')) {
    send(response, 404, { error: `No such endpoint: ${url.pathname}` });
    return;
  }
  serveStatic(url.pathname, response);
});

server.listen(port, () => {
  const environment = describeEnvironment();
  console.log(`LiteMind UI on http://localhost:${port}`);
  console.log(`  engine: ${environment.executable ?? 'NOT BUILT - run scripts/build.ps1'}`);
  console.log(`  model:  ${environment.modelDirectory}${environment.ready ? '' : '  (no config.json)'}`);
});
