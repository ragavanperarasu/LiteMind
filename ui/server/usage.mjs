import os from 'node:os';

/**
 * Host CPU and memory.
 *
 * os.cpus() reports time accumulated since boot, not a rate, so a single
 * reading says nothing about load right now. Utilisation is the difference
 * between two readings, which means keeping the previous one.
 */

let previous = null;

const totals = () =>
  os.cpus().map(({ times }) => ({
    idle: times.idle,
    total: times.user + times.nice + times.sys + times.idle + times.irq,
  }));

/** Per-core utilisation since the last call, and the memory in use. */
export function sampleUsage() {
  const current = totals();
  const cores = current.map((core, index) => {
    const before = previous?.[index];
    if (!before) return 0;
    const idle = core.idle - before.idle;
    const total = core.total - before.total;
    // Two samples taken in the same millisecond have no elapsed time to divide
    // by, and would otherwise read as a spike or a NaN.
    return total <= 0 ? 0 : Math.min(1, Math.max(0, 1 - idle / total));
  });
  previous = current;

  const totalMemory = os.totalmem();
  const freeMemory = os.freemem();

  return {
    cores,
    // The first call has no previous reading, so it reports zero rather than
    // guessing. The caller polls, so the second reading is a moment away.
    cpu: cores.length ? cores.reduce((sum, value) => sum + value, 0) / cores.length : 0,
    coreCount: cores.length,
    model: os.cpus()[0]?.model?.trim() ?? 'unknown',
    totalMemory,
    freeMemory,
    usedMemory: totalMemory - freeMemory,
    platform: `${os.platform()} ${os.release()}`,
    uptime: os.uptime(),
  };
}
