import { useCallback, useEffect, useState } from 'react';
import { getAdminNotifications, type AlertsView } from '../api';

// The administrator's notifications (notifications.js, 16 Nov 2026), shared by the sidebar's bell, the dialog after
// signing in and the Notifications page: one request at a time, every answer moves all three. While a reader is
// mounted the list is asked again every REFRESH_MS while the tab is visible, and when the tab comes back into view
// or focus (at most once a minute) - a failure during the night shows on a page left open. undefined: not known.
const REFRESH_MS = 5 * 60 * 1000;
const MIN_GAP_MS = 60 * 1000;
let view: AlertsView | undefined;
let failed: Error | null = null;
let pending: Promise<AlertsView> | null = null;
let askedAt = 0;
const readers = new Set<() => void>();
const tell = () => readers.forEach((r) => r());

export function fetchNotifications(): Promise<AlertsView> {
  if (!pending) {
    askedAt = Date.now();
    pending = getAdminNotifications()
      .then(
        (v) => {
          view = v;
          failed = null;
          return v;
        },
        (e: Error) => {
          failed = e;
          throw e;
        },
      )
      .finally(() => {
        pending = null;
        tell();
      });
  }
  return pending;
}

let watchers = 0;
let timer: ReturnType<typeof setInterval> | null = null;
const visible = () => typeof document === 'undefined' || document.visibilityState !== 'hidden';
const ask = (minGap: number) => {
  if (visible() && Date.now() - askedAt >= minGap) fetchNotifications().catch(() => undefined);
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

// when: 'never' (not an administrator), 'once' (the sidebar and the dialog: unless already known), 'now' (the page)
export function useNotifications(when: 'never' | 'once' | 'now'): { view: AlertsView | undefined; error: Error | null; refresh: () => Promise<void> } {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (when === 'never') return undefined;
    const reader = () => setTick((t) => t + 1);
    readers.add(reader);
    if (when === 'now' || (view === undefined && !failed && !askedAt)) fetchNotifications().catch(() => undefined);
    watch();
    return () => {
      readers.delete(reader);
      unwatch();
    };
  }, [when]);
  const refresh = useCallback(() => fetchNotifications().then(() => undefined, () => undefined), []);
  return { view: when === 'never' ? undefined : view, error: failed, refresh };
}
