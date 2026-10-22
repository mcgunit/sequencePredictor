import { useEffect, useState } from 'react';
import { ApiError, getMe, type Me } from '../api';

// Who is reading, shared by the sidebar and the admin guard. undefined while the answer is on its way. A
// signed-out answer (a 401) reads as user null; any other failure (a network error, a 502 during a restart) is
// kept as failed and never as "not the admin": the next reader to mount asks again, and retryMe() asks now.
export type MeState = Me & { failed?: Error };

let state: MeState | undefined;
let pending = false;
const readers = new Set<(s: MeState | undefined) => void>();
const tell = () => readers.forEach((r) => r(state));

function ask() {
  if (pending) return;
  pending = true;
  getMe()
    .then(
      (me): MeState => me,
      (error: Error): MeState => (error instanceof ApiError && error.status === 401 ? { user: null } : { user: null, failed: error }),
    )
    .then((s) => {
      state = s;
      pending = false;
      tell();
    });
}

// The session's form token for the actions (postJson), or null while it is not known (asked, failed, signed out).
export function csrfToken(): string | null {
  return state && !state.failed ? state.csrf ?? null : null;
}

// A password change re-issues the session with a new form token (POST /api/account/password answers it): from then on
// every action sends that one.
export function setCsrf(csrf: string) {
  if (!state || state.failed) return;
  state = { ...state, csrf };
  tell();
}

// After an action was refused for a stale form token (a password changed in another browser tab re-issued the session):
// ask for the current one without putting the readers back to "loading", so the page and its dialog stay as they are.
// A failed ask keeps what was known.
export function refreshMe(): Promise<void> {
  return getMe().then(
    (me) => {
      state = me;
      tell();
    },
    () => undefined,
  );
}

// Ask again after a failure (the guard's "Try again"); every reader shows "loading" until the answer.
export function retryMe() {
  state = undefined;
  tell();
  ask();
}

export function useMe(): MeState | undefined {
  const [me, setMe] = useState<MeState | undefined>(state);
  useEffect(() => {
    readers.add(setMe);
    if (state === undefined || state.failed) ask();
    else setMe(state);
    return () => {
      readers.delete(setMe);
    };
  }, []);
  return me;
}
