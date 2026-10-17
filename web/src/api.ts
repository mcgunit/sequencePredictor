// Typed fetch helpers for the markets API, served by server.js next to this app.
// The backend may leave any field null or empty. getMarket() fills in the
// containers (arrays, the status block, the notes) so that components only have
// to guard single values, which the format helpers do.

export type Market = 'crypto' | 'shares';
export type Game = 'day' | 'week';
export type Call = 'down' | 'flat';
// 'sold': a week position the guard sold at today's close (after that close); 'too late': a day position after its close
export type OpenStatus = 'buy now' | 'buy at open' | 'held' | 'too late' | 'sold' | 'out';

export interface Me {
  user: { name: string; admin: boolean } | null;
}

export interface Quotes {
  at: string | null;
  quotes: Record<string, number>;
}

export interface DayResult {
  date: string;
  pnl: number;
}

export interface WeekStatus {
  since: string | null;
  days: number;
  liveTotal: number | null;
  liveMarket: number | null;
  ahead: number | null;
  positions: number;
  won: number;
}

export interface CoinResult {
  symbol: string;
  pnl: number;
  positions: number;
}

export interface Status {
  week: WeekStatus | null; // the weekly positions' own record; the figures below count daily and weekly together
  dailyTotal: number | null;
  weeklyTotal: number | null;
  bestCoin: CoinResult | null;
  worstCoin: CoinResult | null;
  since: string | null;
  days: number;
  liveTotal: number | null;
  liveMarket: number | null;
  ahead: number | null;
  positions: number;
  won: number;
  lost: number;
  winRate: number | null; // 0..1
  daysWon: number;
  daysPlayed: number; // days with at least one position
  daysIdle: number; // days on which nothing was bought
  bestDay: DayResult | null;
  worstDay: DayResult | null;
  openToBuy: number;
  openHeld: number;
  paperNote: string;
}

export interface DayGame {
  date: string;
  stateText: string;
  stateTitle?: string;
  closeText: string;
  model: string | null;
  modelCount: number;
  ruleName: string;
  entryNote: string;
}

export interface WeekGame {
  endDay: string;
  guarded?: boolean; // the week rows follow the guard's state for today (MarketSettle.guarded_week)
  stateText: string;
  stateTitle?: string;
  closeText: string;
  model: string | null;
  modelCount: number;
  ruleName: string;
}

export interface Band {
  low: number | null;
  high: number | null;
  bin: number | null;
  lowPct: number | null;
  highPct: number | null;
}

export interface Guard {
  // buy/hold/sell/out under the guard; weekend: sold at the week's close whatever the daily call
  state: 'buy' | 'hold' | 'sell' | 'weekend' | 'out';
  text: string;
  reason: string;
  booked?: boolean; // false while the guard rule has no paper book: information, not an order
}

export interface OpenPosition {
  symbol: string;
  game: Game;
  status: OpenStatus;
  statusText: string;
  buyText?: string; // when to buy, short: "now", "today 15:30", or the day a held position was bought
  openRate: number | null;
  openRateNote: string | null;
  band: Band | null;
  profitPct: number | null;
  profitPctClose: number | null;
  sellText: string;
  sellNote?: string | null; // under the Sell time: "at the latest · the guard can sell sooner"
  guard: Guard | null;
  since?: string | null; // week rows: "since Mon 12 Oct", or "sold Tue 13 Oct · −1.5% (−1.74 USDT)" for a position the guard sold
  realized?: number | null; // money of this week's positions the guard already sold (null when one is not known)
  earlier?: string | null; // a week row held again after a guard sale this week: "earlier: sold Tue 13 Oct · −1.5% (−1.74 USDT)"
  action?: 'hold' | 'sell' | 'buy' | 'out'; // the guard's action for the running day
  modelsUp: { up: number; of: number } | null;
}

export interface NoneCall {
  symbol: string;
  call: Call;
}

export interface LogEntry {
  date: string;
  symbol: string;
  game: Game;
  pnl: number;
}

export interface DailyPoint {
  date: string;
  pnl: number;
  market: number | null;
}

export interface CumulativePoint {
  date: string;
  plan: number;
  market: number | null;
}

export interface PerInstrument {
  symbol: string;
  pnl: number;
  positions: number;
  won: number;
}

export interface ClosedPosition {
  closed: string;
  symbol: string;
  game: Game;
  model: string | null;
  movePct: number | null;
  pnl: number;
  exit: string;
  sold?: string; // a guarded week position: the day it was sold (closed is its week's close)
}

export interface MarketView {
  market: Market;
  title: string;
  unit: string;
  noun: string;
  instruments: string[];
  generatedAt: string | null;
  now: string;
  status: Status;
  day: DayGame | null;
  week: WeekGame | null;
  open: OpenPosition[];
  none: { day: NoneCall[]; week: NoneCall[] };
  log: LogEntry[];
  daily: DailyPoint[];
  cumulative: CumulativePoint[];
  perCoin: PerInstrument[];
  closed: ClosedPosition[];
  notes: { entry: string; sell: string; guard: string; entryText?: string; sellText?: string; guardText?: string; closeDay?: string };
}

export class ApiError extends Error {
  constructor(
    public readonly url: string,
    public readonly status: number | null,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getJson<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { credentials: 'same-origin', cache: 'no-store', headers: { Accept: 'application/json' } });
  } catch (e) {
    throw new ApiError(url, null, `GET ${url} failed: ${(e as Error).message || 'network error'}`);
  }
  // Signed out: the server answers the API with 401 (auth.js); an older server redirects to the login page instead.
  if (res.status === 401 || (res.redirected && new URL(res.url).pathname === '/login')) {
    window.location.assign(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
    throw new ApiError(url, 401, 'signed out - going to the login page');
  }
  if (!res.ok) {
    let detail = '';
    try {
      const body = (await res.json()) as { error?: unknown };
      if (body && typeof body.error === 'string') detail = body.error;
    } catch {
      /* no JSON body */
    }
    throw new ApiError(url, res.status, detail ? `${detail} (GET ${url}: ${res.status})` : `GET ${url} answered ${res.status} ${res.statusText}`.trim());
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError(url, res.status, `GET ${url} did not answer with JSON`);
  }
}

export function isMarket(s: string | undefined): s is Market {
  return s === 'crypto' || s === 'shares';
}

export const TITLES: Record<Market, string> = { crypto: 'Crypto', shares: 'Shares' };

export async function getMe(): Promise<Me> {
  const me = await getJson<Partial<Me> | null>('/api/me');
  return { user: me?.user && typeof me.user.name === 'string' ? me.user : null };
}

export async function getQuotes(market: Market): Promise<Quotes> {
  const q = await getJson<Partial<Quotes> | null>(`/api/quotes/${market}`);
  const quotes: Record<string, number> = {};
  for (const [sym, v] of Object.entries(q?.quotes ?? {})) if (typeof v === 'number' && Number.isFinite(v) && v > 0) quotes[sym] = v;
  return { at: typeof q?.at === 'string' ? q.at : null, quotes };
}

export async function getMarket(market: Market): Promise<MarketView> {
  return normalize(await getJson<Partial<MarketView> | null>(`/api/markets/${market}`), market);
}

// Fill the containers so the components can map over them without checks.
const arr = <T,>(v: T[] | null | undefined): T[] => (Array.isArray(v) ? v : []);
const num = (v: unknown, fallback: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);
const opt = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);

function normalize(raw: Partial<MarketView> | null | undefined, market: Market): MarketView {
  const r = raw ?? {};
  const s: Partial<Status> = r.status ?? {};
  return {
    market: isMarket(r.market) ? r.market : market,
    title: str(r.title, TITLES[market]),
    unit: str(r.unit),
    noun: str(r.noun, market === 'crypto' ? 'coin' : 'share'),
    instruments: arr(r.instruments).filter((x): x is string => typeof x === 'string'),
    generatedAt: typeof r.generatedAt === 'string' ? r.generatedAt : null,
    now: str(r.now, new Date().toISOString()),
    status: {
      week: s.week && typeof s.week === 'object'
        ? { since: typeof s.week.since === 'string' ? s.week.since : null, days: num(s.week.days, 0), liveTotal: opt(s.week.liveTotal), liveMarket: opt(s.week.liveMarket),
            ahead: opt(s.week.ahead), positions: num(s.week.positions, 0), won: num(s.week.won, 0) }
        : null,
      dailyTotal: opt(s.dailyTotal),
      weeklyTotal: opt(s.weeklyTotal),
      bestCoin: s.bestCoin && typeof s.bestCoin.symbol === 'string' && opt(s.bestCoin.pnl) !== null ? s.bestCoin : null,
      worstCoin: s.worstCoin && typeof s.worstCoin.symbol === 'string' && opt(s.worstCoin.pnl) !== null ? s.worstCoin : null,
      since: typeof s.since === 'string' ? s.since : null,
      days: num(s.days, 0),
      liveTotal: opt(s.liveTotal),
      liveMarket: opt(s.liveMarket),
      ahead: opt(s.ahead),
      positions: num(s.positions, 0),
      won: num(s.won, 0),
      lost: num(s.lost, 0),
      winRate: opt(s.winRate),
      daysWon: num(s.daysWon, 0),
      daysPlayed: num(s.daysPlayed, num(s.days, 0)),
      daysIdle: num(s.daysIdle, 0),
      bestDay: s.bestDay && opt(s.bestDay.pnl) !== null ? s.bestDay : null,
      worstDay: s.worstDay && opt(s.worstDay.pnl) !== null ? s.worstDay : null,
      openToBuy: num(s.openToBuy, 0),
      openHeld: num(s.openHeld, 0),
      paperNote: str(s.paperNote),
    },
    day: r.day ?? null,
    week: r.week ?? null,
    open: arr(r.open),
    none: { day: arr(r.none?.day), week: arr(r.none?.week) },
    log: arr(r.log),
    daily: arr(r.daily),
    cumulative: arr(r.cumulative),
    perCoin: arr(r.perCoin),
    closed: arr(r.closed),
    notes: { entry: str(r.notes?.entry), sell: str(r.notes?.sell), guard: str(r.notes?.guard), entryText: str(r.notes?.entryText), sellText: str(r.notes?.sellText), guardText: str(r.notes?.guardText), closeDay: str(r.notes?.closeDay) },
  };
}
