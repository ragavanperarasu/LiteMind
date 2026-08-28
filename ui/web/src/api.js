/**
 * Talking to the Node backend.
 *
 * Generation is streamed with server-sent events over a POST, which rules out
 * EventSource - it only issues GET requests - so the response body is read and
 * split here instead. The framing is the same either way.
 */

export async function fetchHealth() {
  const response = await fetch('/api/health');
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}

export async function fetchUsage() {
  const response = await fetch('/api/usage');
  if (!response.ok) throw new Error(`Usage read failed: ${response.status}`);
  return response.json();
}

export async function fetchSettings() {
  const response = await fetch('/api/settings');
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `Could not read settings: ${response.status}`);
  return body;
}

export async function saveSettings(settings) {
  const response = await fetch('/api/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `Could not save settings: ${response.status}`);
  return body;
}

/**
 * Runs a prompt, calling onEvent for each event as it arrives.
 *
 * signal aborts the request, which closes the connection; the backend notices
 * and kills the engine, so cancelling actually stops the work rather than just
 * hiding it.
 */
export async function streamGeneration({ prompt, settings, signal, onEvent }) {
  const response = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, settings }),
    signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ?? `The engine could not be started (${response.status}).`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let pending = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // stream: true keeps a multi-byte character intact when it straddles the
    // boundary between two network chunks.
    pending += decoder.decode(value, { stream: true });

    let split;
    while ((split = pending.indexOf('\n\n')) !== -1) {
      const frame = pending.slice(0, split);
      pending = pending.slice(split + 2);
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        onEvent(JSON.parse(line.slice(6)));
      }
    }
  }
}
