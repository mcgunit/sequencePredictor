import { useCallback, useEffect, useState } from 'react';
import { getWhatsNew, type WhatsNewView } from '../api';

// What's new, shared by the sidebar's unseen count and the page: one request at a time, and every answer the page
// reads (after "mark as seen") moves the count too. The sidebar asks once per visit; the page asks again when it opens.
let view: WhatsNewView | undefined;
let failed: Error | null = null;
let pending: Promise<WhatsNewView> | null = null;
const readers = new Set<() => void>();
const tell = () => readers.forEach((r) => r());

export function fetchWhatsNew(): Promise<WhatsNewView> {
  if (!pending) {
    pending = getWhatsNew()
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

// when: 'never' (not signed in yet), 'once' (the sidebar: unless already known), 'now' (the page: read it again)
export function useWhatsNew(when: 'never' | 'once' | 'now'): { view: WhatsNewView | undefined; error: Error | null; refresh: () => Promise<void> } {
  const [, setTick] = useState(0);
  useEffect(() => {
    const reader = () => setTick((t) => t + 1);
    readers.add(reader);
    if (when === 'now' || (when === 'once' && view === undefined && !failed)) fetchWhatsNew().catch(() => undefined);
    return () => {
      readers.delete(reader);
    };
  }, [when]);
  const refresh = useCallback(() => fetchWhatsNew().then(() => undefined, () => undefined), []);
  return { view, error: failed, refresh };
}
