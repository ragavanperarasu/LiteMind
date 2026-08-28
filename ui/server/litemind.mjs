import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

/**
 * Locating and driving the LiteMind executable.
 *
 * The engine is run once per request rather than kept alive between them. That
 * sounds wasteful for a 29.3 GiB checkpoint and is not: the weights are memory
 * mapped, so a load costs about a third of a second, and the expert pages stay
 * in the operating system's cache across processes. A fresh process also means
 * a cancelled request leaves nothing behind to reset.
 */

const EXECUTABLE_NAMES = ['LiteMind.exe', 'LiteMind'];

/** Finds the built executable, newest build directory first. */
export function findExecutable(repositoryRoot) {
  const candidates = [];
  for (const directory of ['build/bin', 'build', 'build/Release/bin']) {
    for (const name of EXECUTABLE_NAMES) {
      candidates.push(path.join(repositoryRoot, directory, name));
    }
  }
  return candidates.find(existsSync) ?? null;
}

/** Reads litemind.json, falling back to an empty object when it is absent. */
export function readSettings(repositoryRoot) {
  const file = path.join(repositoryRoot, 'litemind.json');
  if (!existsSync(file)) return {};
  try {
    return JSON.parse(readFileSync(file, 'utf8'));
  } catch (cause) {
    throw new Error(`litemind.json is not valid JSON: ${cause.message}`);
  }
}

/** Strips the documentation keys, which exist for a human editing the file. */
export function withoutComments(settings) {
  return Object.fromEntries(
    Object.entries(settings).filter(([key]) => !key.startsWith('_')),
  );
}

/**
 * Splits a byte stream into whole JSON objects, one per line.
 *
 * A chunk boundary falls wherever the pipe buffer happens to end, which is
 * regularly in the middle of a line and, for multi-byte characters, in the
 * middle of a character. Decoding per chunk would corrupt any reply that is not
 * pure ASCII, so the decoder is kept across chunks and told the stream is not
 * finished.
 */
export function createEventParser(onEvent, onMalformed) {
  const decoder = new TextDecoder('utf-8');
  let pending = '';

  return {
    push(chunk) {
      pending += decoder.decode(chunk, { stream: true });
      let newline;
      while ((newline = pending.indexOf('\n')) !== -1) {
        const line = pending.slice(0, newline).trim();
        pending = pending.slice(newline + 1);
        if (line === '') continue;
        try {
          onEvent(JSON.parse(line));
        } catch {
          // The engine writes nothing but events to stdout, so a line that does
          // not parse is a real fault worth surfacing, not noise to swallow.
          onMalformed?.(line);
        }
      }
    },
    flush() {
      pending += decoder.decode();
      const line = pending.trim();
      pending = '';
      if (line === '') return;
      try {
        onEvent(JSON.parse(line));
      } catch {
        onMalformed?.(line);
      }
    },
  };
}

/**
 * Runs one prompt, calling onEvent for each event the engine emits.
 *
 * Returns a handle with cancel(), which kills the process. Generation is not
 * interruptible from inside - the decode loop only checks its own stop
 * conditions - so cancelling means ending the process that is doing the work.
 */
export function generate({ executable, repositoryRoot, modelDirectory, prompt, settings, onEvent }) {
  const args = [modelDirectory, '-p', prompt, '--json'];

  const maxTokens = Number(settings.max_tokens);
  if (Number.isFinite(maxTokens) && maxTokens > 0) args.push('-n', String(maxTokens));

  const context = Number(settings.context);
  if (Number.isFinite(context) && context > 0) args.push('--context', String(context));

  const threads = Number(settings.threads);
  if (Number.isFinite(threads) && threads > 0) args.push('--threads', String(threads));

  const temperature = Number(settings.temperature);
  if (Number.isFinite(temperature) && temperature > 0) {
    args.push('--temperature', String(temperature));
  }

  const expertCache = Number(settings.expert_cache_gb);
  if (Number.isFinite(expertCache) && expertCache > 0) {
    args.push('--expert-cache', String(expertCache));
  }

  // Every sampling control is passed explicitly rather than left to the engine
  // to read from litemind.json. That lookup is relative to the working
  // directory, so relying on it would make the reply depend on where the server
  // happened to be started - the interface and the command line would sample
  // differently from the same settings, which is the sort of difference that
  // gets mistaken for a fault in the model.
  const topK = Number(settings.top_k);
  if (Number.isFinite(topK) && topK >= 0) args.push('--top-k', String(topK));

  const topP = Number(settings.top_p);
  if (Number.isFinite(topP) && topP > 0) args.push('--top-p', String(topP));

  const repeatPenalty = Number(settings.repeat_penalty);
  if (Number.isFinite(repeatPenalty) && repeatPenalty > 0) {
    args.push('--repeat-penalty', String(repeatPenalty));
  }

  const seed = Number(settings.seed);
  if (Number.isFinite(seed) && seed > 0) args.push('--seed', String(seed));

  if (settings.chat === false) args.push('--no-chat');

  // Routing is 156 numbers a token, so it is asked for rather than assumed.
  if (settings.routing) args.push('--routing');

  // No shell: the prompt is passed as one argv entry, so quoting and shell
  // metacharacters in it mean nothing.
  const child = spawn(executable, args, { shell: false, cwd: repositoryRoot });

  const parser = createEventParser(onEvent, (line) => {
    onEvent({ event: 'error', message: `Unreadable output from the engine: ${line}` });
  });

  child.stdout.on('data', (chunk) => parser.push(chunk));

  let stderrText = '';
  child.stderr.on('data', (chunk) => {
    stderrText += chunk.toString('utf8');
  });

  let cancelled = false;
  const finished = new Promise((resolve) => {
    child.on('error', (cause) => {
      onEvent({ event: 'error', message: `Could not start the engine: ${cause.message}` });
      resolve();
    });
    child.on('close', (code) => {
      parser.flush();
      if (cancelled) {
        onEvent({ event: 'cancelled' });
      } else if (code !== 0) {
        onEvent({
          event: 'error',
          message: stderrText.trim() || `The engine exited with code ${code}.`,
        });
      }
      resolve();
    });
  });

  return {
    finished,
    cancel() {
      cancelled = true;
      child.kill();
    },
  };
}
