import { useCallback, useRef, useState } from 'react';
import { ApiError, postJson, type Done } from '../api';
import type { ToastMessage } from '../components/admin/Toast';
import { csrfToken, refreshMe } from './useMe';

const STALE_TOKEN = /form token/i;

// Sends a page's actions with the session's form token: busy while one is on its way, the refusal in the server's own
// words (for the dialog that asked), the success as a toast, then after() - the page reads its JSON again, so what it
// shows is the state the action left. A refusal for a stale token (the session was re-issued in another tab) asks
// /api/me for the current one, so trying again works. errorToast: a page without a dialog (its buttons sit in a long
// list, far from the page head on a phone) also shows a refusal as a toast, these words before it.
export interface Sender {
  busy: boolean;
  error: string | null;
  toast: ToastMessage | null;
  send: <T extends Done = Done>(url: string, body?: Record<string, unknown>) => Promise<T | null>;
  reset: () => void; // a dialog opens: no old refusal in it
  dismiss: () => void;
}

export function useSend(after?: () => unknown, { errorToast }: { errorToast?: string } = {}): Sender {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastMessage | null>(null);
  const afterRef = useRef(after);
  afterRef.current = after;
  const counter = useRef(0);

  const send = useCallback(async <T extends Done = Done>(url: string, body: Record<string, unknown> = {}): Promise<T | null> => {
    const token = csrfToken();
    setError(null);
    if (!token) {
      void refreshMe();
      setError('The form token is not loaded yet - try again in a moment.');
      return null;
    }
    setBusy(true);
    try {
      const done = await postJson<T>(url, token, body);
      counter.current += 1;
      setToast({ id: counter.current, text: done.message });
      void afterRef.current?.();
      return done;
    } catch (e) {
      const message = (e as Error).message || 'The action failed.';
      if (e instanceof ApiError && e.status === 403 && STALE_TOKEN.test(message)) void refreshMe();
      setError(message);
      if (errorToast !== undefined) {
        counter.current += 1;
        setToast({ id: counter.current, text: `${errorToast}${message}` });
      }
      // refused by the state it found (409): what the page shows is behind, so read it again
      if (e instanceof ApiError && e.status === 409) void afterRef.current?.();
      return null;
    } finally {
      setBusy(false);
    }
  }, [errorToast]);

  return {
    busy,
    error,
    toast,
    send,
    reset: useCallback(() => setError(null), []),
    dismiss: useCallback(() => setToast(null), []),
  };
}
