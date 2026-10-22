import { useCallback, useEffect, useRef, useState } from 'react';

// A view read again every `everyMs` while the tab is visible, and at once when the tab comes back into view after
// that long; refresh() reads it now (after an action). One request at a time. A failed read keeps the last answer on
// screen with the error beside it, so a restart of the server does not blank the page.
export interface Poll<T> {
  data: T | null;
  error: Error | null;
  at: number | null; // when the shown answer arrived (ms)
  busy: boolean;
  refresh: () => Promise<void>;
}

export function usePoll<T>(load: () => Promise<T>, everyMs: number): Poll<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [at, setAt] = useState<number | null>(null);
  const [busy, setBusy] = useState(true);
  const loadRef = useRef(load);
  loadRef.current = load;
  const alive = useRef(true);
  const pending = useRef<Promise<void> | null>(null);
  const askedAt = useRef(0);

  const refresh = useCallback((): Promise<void> => {
    if (pending.current) return pending.current;
    askedAt.current = Date.now();
    setBusy(true);
    pending.current = loadRef
      .current()
      .then(
        (d) => {
          if (!alive.current) return;
          setData(d);
          setError(null);
          setAt(Date.now());
        },
        (e: Error) => {
          if (alive.current) setError(e);
        },
      )
      .finally(() => {
        pending.current = null;
        if (alive.current) setBusy(false);
      });
    return pending.current;
  }, []);

  useEffect(() => {
    alive.current = true;
    const visible = () => document.visibilityState !== 'hidden';
    void refresh();
    const timer = setInterval(() => {
      if (visible()) void refresh();
    }, everyMs);
    const onWake = () => {
      if (visible() && Date.now() - askedAt.current >= everyMs) void refresh();
    };
    document.addEventListener('visibilitychange', onWake);
    window.addEventListener('focus', onWake);
    return () => {
      alive.current = false;
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onWake);
      window.removeEventListener('focus', onWake);
    };
  }, [everyMs, refresh]);

  return { data, error, at, busy, refresh };
}
