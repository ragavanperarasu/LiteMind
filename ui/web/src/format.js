/** Shared number formatting, so the same quantity reads the same everywhere. */

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

export const withSeparators = (value) => Number(value ?? 0).toLocaleString('en-US');

/** Bytes at whatever scale keeps the number readable. */
export function bytes(value) {
  const amount = Number(value ?? 0);
  if (amount >= 1024 * GIB) return `${(amount / (1024 * GIB)).toFixed(2)} TiB`;
  if (amount >= GIB) return `${(amount / GIB).toFixed(2)} GiB`;
  if (amount >= MIB) return `${(amount / MIB).toFixed(1)} MiB`;
  return `${withSeparators(Math.round(amount / 1024))} KiB`;
}

/** Parameter counts, which span from thousands to billions across checkpoints. */
export function parameters(value) {
  const amount = Number(value ?? 0);
  if (amount >= 1e9) return `${(amount / 1e9).toFixed(2)} B`;
  if (amount >= 1e6) return `${(amount / 1e6).toFixed(1)} M`;
  if (amount >= 1e3) return `${(amount / 1e3).toFixed(1)} K`;
  return withSeparators(amount);
}

export const seconds = (value) => `${Number(value ?? 0).toFixed(1)} s`;
