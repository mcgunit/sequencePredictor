import { useCallback } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { isGame, isScope, type Scope } from '../api';

// The admin pages keep what a reading may want to link to in the URL: ?scope=all|top (the All / Top 5
// switch) and the open drawer (?game=lotto, ?row=...). The switch replaces the history entry; opening a
// drawer adds one, so Back closes it, and closing a drawer the page opened goes back to where it was.
// Going back drops a switch made inside the drawer from the URL, so the last choice is also kept in this
// browser and is the default when the URL names none. Every change keeps the URL's hash (#q0, a card the menu
// scrolled to): dropping it and getting it back on Back would scroll the page to that card again.
const SCOPE_KEY = 'admin.scope';

function storedScope(): Scope {
  try {
    const v = window.localStorage.getItem(SCOPE_KEY);
    return isScope(v) ? v : 'all';
  } catch {
    return 'all';
  }
}

export function useScope(): [Scope, (s: Scope) => void] {
  const [params] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const raw = params.get('scope');
  const scope: Scope = isScope(raw) ? raw : storedScope();
  const set = useCallback(
    (s: Scope) => {
      try {
        window.localStorage.setItem(SCOPE_KEY, s);
      } catch {
        /* private window: the URL still carries it */
      }
      const next = new URLSearchParams(params);
      next.set('scope', s);
      navigate({ search: `?${next}`, hash: location.hash }, { replace: true, state: location.state });
    },
    [params, navigate, location.hash, location.state],
  );
  return [scope, set];
}

export interface DrawerParam {
  game: string | null; // a known game, or null when no drawer is open
  extra: string | null; // a second key (the phase-shift row)
  open: (game: string, extra?: string) => void;
  close: () => void;
}

export function useDrawerParam(extraKey?: string): DrawerParam {
  const [params] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const raw = params.get('game');
  const game = isGame(raw) ? raw : null;
  const extra = extraKey ? params.get(extraKey) : null;
  const open = useCallback(
    (g: string, x?: string) => {
      const next = new URLSearchParams(params);
      next.set('game', g);
      if (extraKey) {
        if (x) next.set(extraKey, x);
        else next.delete(extraKey);
      }
      navigate({ search: `?${next}`, hash: location.hash }, { state: { drawer: true } });
    },
    [params, navigate, location.hash, extraKey],
  );
  const close = useCallback(() => {
    if ((location.state as { drawer?: boolean } | null)?.drawer) {
      navigate(-1);
      return;
    }
    const next = new URLSearchParams(params);
    next.delete('game');
    if (extraKey) next.delete(extraKey);
    navigate({ search: `?${next}`, hash: location.hash }, { replace: true });
  }, [location.state, location.hash, navigate, params, extraKey]);
  return { game, extra, open, close };
}

// A drawer named by any query key (This week's ?run=predictor@2026-10-17T07:00:01Z), with the same history rules as
// the game drawer: opening adds an entry (Back closes it), closing one the page opened goes back. The keys in clear
// are dropped on opening, so two drawers (?game=, ?run=) are never named at once.
export interface ParamDrawer {
  value: string | null;
  open: (value: string) => void;
  close: () => void;
}

export function useParamDrawer(key: string, clear: string[] = []): ParamDrawer {
  const [params] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const value = params.get(key) || null;
  const dropped = clear.join(',');
  const open = useCallback(
    (v: string) => {
      const next = new URLSearchParams(params);
      dropped.split(',').filter(Boolean).forEach((k) => next.delete(k));
      next.set(key, v);
      navigate({ search: `?${next}`, hash: location.hash }, { state: { drawer: true } });
    },
    [params, navigate, location.hash, key, dropped],
  );
  const close = useCallback(() => {
    if ((location.state as { drawer?: boolean } | null)?.drawer) {
      navigate(-1);
      return;
    }
    const next = new URLSearchParams(params);
    next.delete(key);
    navigate({ search: `?${next}`, hash: location.hash }, { replace: true });
  }, [location.state, location.hash, navigate, params, key]);
  return { value, open, close };
}

// A switch kept in the URL (?by=game) and, like the All / Top 5 switch, remembered in this browser as the default
// when the URL names none. Switching replaces the history entry.
export function useSwitchParam<T extends string>(key: string, values: readonly T[], fallback: T): [T, (v: T) => void] {
  const [params] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const stored = `admin.${key}`;
  const raw = params.get(key);
  let remembered: string | null = null;
  try {
    remembered = window.localStorage.getItem(stored);
  } catch {
    remembered = null;
  }
  const pick = (v: string | null): T | null => (v !== null && (values as readonly string[]).includes(v) ? (v as T) : null);
  const value = pick(raw) ?? pick(remembered) ?? fallback;
  const set = useCallback(
    (v: T) => {
      try {
        window.localStorage.setItem(stored, v);
      } catch {
        /* private window: the URL still carries it */
      }
      const next = new URLSearchParams(params);
      next.set(key, v);
      navigate({ search: `?${next}`, hash: location.hash }, { replace: true, state: location.state });
    },
    [params, navigate, location.hash, location.state, key, stored],
  );
  return [value, set];
}
