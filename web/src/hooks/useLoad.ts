import { useEffect, useState } from 'react';

export type Load<T> = { kind: 'loading' } | { kind: 'error'; error: Error } | { kind: 'ok'; data: T };

// One fetch per change of its keys, the newest answer winning; retry() asks again. A key change shows
// "loading" again only when there is nothing to show yet, so a switch (All / Top 5) keeps the old rows
// on screen until the new ones arrive.
export function useLoad<T>(load: () => Promise<T>, keys: unknown[]): Load<T> & { retry: () => void; busy: boolean } {
  const [state, setState] = useState<Load<T>>({ kind: 'loading' });
  const [busy, setBusy] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let alive = true;
    setBusy(true);
    setState((s) => (s.kind === 'ok' ? s : { kind: 'loading' }));
    load()
      .then((data) => alive && setState({ kind: 'ok', data }))
      .catch((error: Error) => alive && setState({ kind: 'error', error }))
      .finally(() => alive && setBusy(false));
    return () => {
      alive = false;
    };
    // the caller names what the fetch depends on in keys
  }, [...keys, attempt]);

  return { ...state, busy, retry: () => setAttempt((n) => n + 1) };
}
