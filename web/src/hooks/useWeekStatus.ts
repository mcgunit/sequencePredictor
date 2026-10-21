import { useEffect, useState } from 'react';
import { getAdminWeek, type WeekView } from '../api';
import { weekStatus, type WeekStatus } from '../pages/admin/week';

// This week's status for the sidebar's dot, shared with the This week page: one request at a time (the page and the
// sidebar share an answer on its way), and every answer the page loads moves the dot too. undefined: not known (not
// asked yet, or the last ask failed - no dot rather than a wrong one). While a reader is mounted the status is asked
// again every REFRESH_MS while the tab is visible, and when the tab comes back into view or focus (at most once a
// minute), so a weekend failure shows on an admin page left open.
const REFRESH_MS = 10 * 60 * 1000;
const MIN_GAP_MS = 60 * 1000;
let status: WeekStatus | undefined;
let pending: Promise<WeekView> | null = null;
let askedAt = 0; // when the last ask started (ms); 0: never asked
const readers = new Set<(s: WeekStatus | undefined) => void>();
const tell = () => readers.forEach((r) => r(status));

export function fetchWeek(): Promise<WeekView> {
  if (!pending) {
    askedAt = Date.now();
    pending = getAdminWeek()
      .then(
        (w) => {
          status = weekStatus(w);
          tell();
          return w;
        },
        (e: unknown) => {
          status = undefined;
          tell();
          throw e;
        },
      )
      .finally(() => {
        pending = null;
      });
  }
  return pending;
}

// One timer and one pair of listeners however many readers watch.
let watchers = 0;
let timer: ReturnType<typeof setInterval> | null = null;
const visible = () => typeof document === 'undefined' || document.visibilityState !== 'hidden';
const ask = (minGap: number) => {
  if (visible() && Date.now() - askedAt >= minGap) fetchWeek().catch(() => undefined);
};
const onWake = () => ask(MIN_GAP_MS);
function watch() {
  watchers += 1;
  if (watchers > 1) return;
  timer = setInterval(() => ask(REFRESH_MS - MIN_GAP_MS), REFRESH_MS);
  document.addEventListener('visibilitychange', onWake);
  window.addEventListener('focus', onWake);
}
function unwatch() {
  watchers -= 1;
  if (watchers > 0) return;
  if (timer) clearInterval(timer);
  timer = null;
  document.removeEventListener('visibilitychange', onWake);
  window.removeEventListener('focus', onWake);
}

export function useWeekStatus(enabled: boolean): WeekStatus | undefined {
  const [s, setS] = useState<WeekStatus | undefined>(status);
  useEffect(() => {
    readers.add(setS);
    setS(status);
    if (enabled) {
      if (!askedAt) fetchWeek().catch(() => undefined);
      watch();
    }
    return () => {
      readers.delete(setS);
      if (enabled) unwatch();
    };
  }, [enabled]);
  return s;
}
