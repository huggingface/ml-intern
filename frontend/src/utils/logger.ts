/**
 * Lightweight logger that silences verbose output in production.
 *
 * - `log` / `debug` are only emitted when `import.meta.env.DEV` is true.
 * - `warn` and `error` always go through so real issues surface in prod.
 */

const isDev = import.meta.env.DEV;

/* eslint-disable no-console */
export const logger = {
  /** Debug-level log — DEV only. */
  log: (...args: unknown[]) => {
    if (isDev) console.log(...args);
  },
  /** Debug-level log — DEV only. */
  debug: (...args: unknown[]) => {
    if (isDev) console.debug(...args);
  },
  /** Warning — always emitted. */
  warn: console.warn.bind(console),
  /** Error — always emitted. */
  error: console.error.bind(console),
};
