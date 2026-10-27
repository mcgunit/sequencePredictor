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
  // open: open access (no login configured), where there is nobody to sign out
  user: { name: string; admin: boolean; open?: boolean } | null;
  // the session's form token, which every POST of the app sends back as _csrf (system.js); renewed by a password change
  csrf?: string | null;
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
  guarded?: boolean; // weeks of the guarded book are among them
  guardFrom?: string | null; // the first week the guard counts; before it the plain week
}

export interface CoinResult {
  symbol: string;
  pnl: number;
  positions: number;
}

export interface Status {
  week: WeekStatus | null; // the weekly positions' own record; the figures below count daily and weekly together
  entryNote: string | null; // crypto: positions booked from the 00:00 UTC close, before the tickets exist
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
  byBook?: boolean; // the model is followed for its paper book; false: no book has minTrades positions yet, the best exact rate
  minTrades?: number;
  stake?: number | null; // the paper stake a position: a sized row's position is size x stake
  entryNote: string;
}

export interface WeekGame {
  endDay: string;
  guarded?: boolean; // the week rows follow the guard's state for today (MarketSettle.guarded_week)
  guardOrders?: boolean; // the guard gives this week's orders; false: the plain week with the guard as information
  stateText: string;
  stateTitle?: string;
  closeText: string;
  model: string | null;
  modelCount: number;
  ruleName: string;
  byBook?: boolean; // as DayGame: false while no weekly book has minTrades positions
  minTrades?: number;
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
  size?: number | null; // a sized row's position (the RL Position Model): a multiple of the stake; null for every other row
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
    // the answer's code where the page must react to it (lottery.js: 'before-first', 'not-ranked')
    public readonly code: string | null = null,
    // the server's own words, without the request (a user page says these, the admin pages the whole message)
    public readonly detail: string | null = null,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function getJson<T>(url: string): Promise<T> {
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
    let code: string | null = null;
    try {
      const body = (await res.json()) as { error?: unknown; code?: unknown };
      if (body && typeof body.error === 'string') detail = body.error;
      if (body && typeof body.code === 'string' && body.code) code = body.code;
    } catch {
      /* no JSON body */
    }
    throw new ApiError(url, res.status, detail ? `${detail} (GET ${url}: ${res.status})` : `GET ${url} answered ${res.status} ${res.statusText}`.trim(), code, detail || null);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError(url, res.status, `GET ${url} did not answer with JSON`);
  }
}

// An action (system.js): a JSON POST carrying the session's form token as _csrf. The answer is {ok, message, ...}; a
// refusal is {error} with its status (400 a rule, 403 not allowed or a stale token, 404 nothing by that name, 409 the
// state refused it, 500 the system did), and its message is the server's own words, ready to show.
export interface Done {
  ok: true;
  message: string;
}
export async function postJson<T extends Done = Done>(url: string, csrf: string, body: Record<string, unknown>): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ ...body, _csrf: csrf }),
    });
  } catch (e) {
    throw new ApiError(url, null, `Could not reach the server (${(e as Error).message || 'network error'}). Nothing was sent.`);
  }
  if (res.status === 401) {
    window.location.assign(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
    throw new ApiError(url, 401, 'signed out - going to the login page');
  }
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  const answer = (data && typeof data === 'object' ? data : {}) as { ok?: unknown; message?: unknown; error?: unknown };
  if (!res.ok) throw new ApiError(url, res.status, typeof answer.error === 'string' ? answer.error : `The server answered ${res.status} ${res.statusText}`.trim());
  if (answer.ok !== true) throw new ApiError(url, res.status, 'The server did not answer with JSON.');
  return { ...(answer as object), ok: true, message: typeof answer.message === 'string' ? answer.message : 'done' } as T;
}

export function isMarket(s: string | undefined): s is Market {
  return s === 'crypto' || s === 'shares';
}

export const TITLES: Record<Market, string> = { crypto: 'Crypto', shares: 'Shares' };

export async function getMe(): Promise<Me> {
  const me = await getJson<Partial<Me> | null>('/api/me');
  const u = me?.user;
  return {
    user: u && typeof u.name === 'string' ? { name: u.name, admin: u.admin === true, open: u.open === true } : null,
    csrf: typeof me?.csrf === 'string' && me.csrf ? me.csrf : null,
  };
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
            ahead: opt(s.week.ahead), positions: num(s.week.positions, 0), won: num(s.week.won, 0),
            guarded: s.week.guarded === true, guardFrom: typeof s.week.guardFrom === 'string' ? s.week.guardFrom : null }
        : null,
      dailyTotal: opt(s.dailyTotal),
      weeklyTotal: opt(s.weeklyTotal),
      entryNote: typeof s.entryNote === 'string' ? s.entryNote : null,
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

// --- Admin (/api/admin/*, admin.js) ---------------------------------------------
// The administrator's views: the four Sunday controls, the best model per game with each game's ranking, the
// phase-shift card and the Randomness watch. A signed-in user who is not the administrator gets a 403 (an ApiError
// with status 403). The getters fill every container and turn each figure into a number or null, so the pages
// only guard single values; null means "not measured", never 0.

export type Scope = 'all' | 'top';
export type GroupName = 'Lottery' | 'Markets';

export const LOTTERY_GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
export const MARKET_GAMES = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];
export const GAMES = [...LOTTERY_GAMES, ...MARKET_GAMES];
export const isGame = (s: unknown): s is string => typeof s === 'string' && GAMES.includes(s);
export const groupOf = (game: string): GroupName => (MARKET_GAMES.includes(game) ? 'Markets' : 'Lottery');
export const isScope = (s: unknown): s is Scope => s === 'all' || s === 'top';

export interface Range {
  min: number;
  max: number;
}

export interface Interval {
  mean: number | null;
  lo: number | null;
  hi: number | null;
  verdict: string; // 'better' | 'worse' | 'no difference'
}

// Q0 (27 Oct 2026): the daily report's row test (src/RowChance.py) - each row on its live draws against the fair draw
// (lottery, exact) or its own tickets shuffled within blocks of 7 live days (markets), Holm over the card, and a fixed-length
// verdict on new draws for a row that read p < 0.05. The null band (NullControls.py) is context only.
// A control the band is not built from: its record is not there, or there but unreadable (caught half written)
export interface MissingControl {
  mode: string;
  unreadable: boolean;
}
export interface Q0Band {
  ceiling: number | null;
  from: string; // 'synthetic' | 'shuffled': the control the band comes from
  rows: number | null;
  seeds: number | null;
  history: number | null; // fair-history length, in the game's unit
  metric: string;
  at: string | null;
  modes: { mode: string; at: string | null }[]; // each control record's own time (a band half rewritten on a Sunday mixes weeks)
  missing: MissingControl[]; // not empty: the band comes from one control only
}
// A followed row's fresh record: from the Sunday it first read p < 0.05 (since), judged once on verdictLength new live
// draws - running (the figures so far, no p), stopped (four weeks without a new live draw), confirmed (p < 0.01) or not
export type FreshState = 'running' | 'stopped' | 'confirmed' | 'not confirmed';
export interface Q0Fresh {
  state: FreshState;
  since: string | null; // the flag Sunday
  flagDraws: number | null; // the live draws before it, that read p < 0.05
  flagP: number | null;
  flagMean: number | null;
  flagNull: number | null;
  verdictLength: number | null; // the new live draws the verdict is taken on
  draws: number; // new live draws so far (may pass verdictLength: the verdict reads the first ones)
  last: string | null; // the newest of them
  verdictAt: string | null; // the draw the verdict was taken on
  decidedOn: string | null; // the report's day that took it
  p: number | null; // the verdict's (null while running)
  mean: number | null; // the judged new draws' hits per draw (so far, while running)
  null: number | null;
}
// One row's test on one line: its own hits per draw on its live draws, the null (fair or shuffled), its p and the
// corrected p (Holm over the card's family; null outside it)
export interface Q0Line {
  row: string;
  mean: number | null;
  live: number; // the live draws tested
  notLive: number; // back-built or late (and a second row of one name on a draw)
  null: number | null;
  p: number | null;
  pHolm: number | null;
  inFamily: boolean; // at least minDraws live draws
  recovered: number | null; // lottery: live draws whose ticket was rewritten after the draw, scored as it stood at the close
  fresh: Q0Fresh | null;
}
export interface Q0Test {
  method: 'exact' | 'permutation';
  alpha: number;
  verdictAlpha: number;
  minDraws: number | null;
  permutations: number | null;
  block: number | null;
  family: number; // rows with at least minDraws live draws
  liveMax: number; // the most live draws of any row (the week games: too few until minDraws)
  under: number; // raw p < alpha
  expected: number; // what chance puts under alpha
  holds: number; // corrected p < alpha
  followed: number; // running + stopped
  running: number;
  stopped: number;
  confirmed: number;
  notConfirmed: number;
  checkedThrough: string | null;
  stored: string | null;
}
export interface Q0 {
  ranking: 'hits' | 'profit';
  inTile: boolean; // false for the payout games: tested on hits, information only
  test: Q0Test | null;
  error: string | null; // why there is no test
  lowest: Q0Line | null; // the family row with the lowest p
  rows: Q0Line[]; // the rows the card names: under alpha, followed or judged
  confirmed: string[];
  followed: string[];
  leader: { row: string; draws: number | null } | null; // payout games: the profit-bound leader
  allRows: number;
  minDraws: number;
  band: Q0Band | null; // context only
  bandUnreadable: boolean; // no band: a NullControls record is there but did not load
}
// failed: a comparison scored fewer than 2 seeds, so the run has no band (never read as within)
export type Q2Status = 'above' | 'within' | 'too few draws' | 'above band, not shuffled' | 'failed';
export interface Q2 {
  status: Q2Status;
  rule: number; // the verdict version it was read under: 1 = mean + 2 sd of 3 seeds (to 25 Oct 2026), 2 = bounds from the seed count
  real: number | null;
  threshold: number | null;
  thresholdSe: number | null; // the threshold's own standard error (rule 2 only)
  shuffled: number | null;
  shuffledThreshold: number | null; // the shuffled control's own bound (rule 2 only)
  bandK: number | null; // k of the null bound, mean + k sd: from the seed count (rule 2 only)
  seeds: number | null;
  windowsTest: number | null;
  fewWindows: boolean;
  minWindows: number | null; // the test windows the record was judged by: fewer read "too few draws"
  window: number | null;
  drawsCap: number | null; // the --draws the run asked for: at most the newest that many
  drawsUsed: number | null; // the draws the windows actually came from (short histories use fewer)
  quantum: boolean;
  at: string | null;
}
export interface FcVariant {
  key: string; // lr gb qk qv svm
  verdict: string; // 'fits noise' | 'inconclusive' | 'noise ignored' | 'failed'
  share: number | null;
  auc: [number | null, number | null]; // held-out AUC without noise, with noise
  above: string[];
}
export interface Fc {
  fits: number;
  inconclusive: number;
  ignored: number;
  failed: number;
  columns: number | null;
  chanceShare: number | null;
  tableRows: number | null;
  lockboxRows: number | null;
  behind: number | null; // draws the table lags its game's day files by
  positional: boolean; // the table is built per position (pick3, Joker+, the market games)
  repeats: number | null;
  noiseColumns: number | null;
  columnsAbove: number;
  at: string | null;
  variants: FcVariant[];
}
export interface MarketRows {
  window: number | null;
  requested: number | null;
  firstDay: string | null;
  lastDay: string | null;
  lockboxWithheld: number | null;
  reference: string;
  rowsTotal: number;
  rowsScored: number;
  baselines: number;
  betterThanUniform: string[];
  betterThanReference: string[];
  noDifferenceVsReference: number;
  referenceVsUniform: Interval | null;
  // against the frequencies (MarketRows.py, 26 Oct 2026; null / empty on a report from before): frequency is the baseline
  // row every row is held against, frequencyVsUniform what knowing the frequencies alone scores against uniform
  frequency: string | null;
  betterThanFrequency: string[];
  referenceVsFrequency: Interval | null;
  frequencyVsUniform: Interval | null;
  errors: number;
  hmmFeatures: string | null;
  at: string | null;
}
// Week over week (data/readings/<sunday>.json): what one control read, as one comparable string, and its run of weeks.
// weeks: the weeks in a row, ending at the record's own Sunday, the stored readings read the same; changed: whether the
// week before read differently (null: no reading of that week - the first week on record); last: what it read then,
// only when changed (null: no record that week); behind: the record is older than the newest part's (a run in progress);
// noRunBefore: the week before has a reading, but this control's record in it was an older week's (it did not run then).
// Q0 only: live - no reading of the record's own Sunday yet, so the state is the report on disk; sinceReading - the
// reading of its Sunday is the state, and today's report reads differently (rows followed or confirmed added, dropped).
export interface SinceReading {
  now: string; // the state against the live ranking
  added: string[];
  dropped: string[];
  rankingAt: string | null; // the ranking the reading was built on
}
export interface PartHistory {
  state: string;
  sunday: string | null;
  weeks: number;
  changed: boolean | null;
  last: string | null;
  behind: boolean;
  noRunBefore: boolean;
  live: boolean;
  sinceReading: SinceReading | null;
  // Q2 only: the week before was read under another verdict rule (its rule and seeds), so the run of weeks starts again
  ruleChanged: { rule: number; seeds: number | null } | null;
}
export interface GameHistory {
  q0: PartHistory | null;
  q2: PartHistory | null;
  fc: PartHistory | null;
  rows: PartHistory | null;
}
export interface ReadingsMeta {
  stored: number;
  sundays: string[]; // the last 8 stored, oldest first
  newestRecords: string | null; // the Sunday of the newest control record on disk
  unreadable: string[];
  rule: string;
}
export interface GameControls {
  game: string;
  unit: string;
  q0: Q0;
  q2: Q2 | null;
  fc: Fc | null;
  rows: MarketRows | null;
  errors: string[]; // the parts whose record could not be read ('q0', 'q2', 'fc', 'rows')
  history: GameHistory; // the overview only; every part null in a drawer
}
export interface ControlsView {
  builtAt: string | null;
  rankingAt: string | null;
  readings: ReadingsMeta | null;
  meta: {
    q0: { rule: string; test: string; under: string; holds: string; followed: string; headline: string; band: string; rows: Range | null; history: Range | null; seeds: Record<string, number[]> };
    q2: { metric: string; axis: [number, number]; fewWindows: number | null; rule: string }; // fewWindows: the power rule, controlRules.json
    fc: { share: string; chanceShare: string; auc: string; variants: Record<string, string> };
    rows: { score: string; pnl: string };
  };
  tiles: ControlTiles;
  groups: { name: GroupName; games: GameControls[] }[];
}
export interface ControlTiles {
  // Q0 (27 Oct 2026): the headline is the rows confirmed on new draws in the hit-ranked games; the rest are this week's figures
  q0: {
    confirmed: number;
    confirmedRows: { game: string; row: string }[];
    holds: number;
    under: number;
    expected: number;
    family: number;
    followed: number;
    stopped: number;
    notConfirmed: number;
    games: number; // hit-ranked games with a family
    hitRankedGames: number;
    tooFew: { game: string; liveMax: number; minDraws: number | null }[]; // tested, no row with enough live draws yet
    payout: { games: string[]; under: number; expected: number; holds: number; followed: number; confirmed: number; notConfirmed: number };
    noTest: { game: string; why: string | null }[];
    at: string | null; // the report that ran the tests
    band: { games: number; from: string | null; at: string | null; oneControl: { game: string; from: string; missing: MissingControl[] }[] }; // context
  };
  // missing: games with no record yet; unreadable: games whose record is there but could not be read - both out of the counts
  q2: {
    above: number;
    of: number;
    aboveGames: string[];
    tooFewGames: string[];
    failedGames: string[]; // a run where a comparison scored fewer than 2 seeds: no reading
    seeds: Record<string, number[]>;
    missing: string[];
    unreadable: string[];
    from: string | null;
    at: string | null;
  };
  fc: { fits: number; inconclusive: number; ignored: number; failed: number; variants: number; games: number; missing: string[]; unreadable: string[]; from: string | null; at: string | null };
  rows: {
    beatReference: number;
    games: number;
    referenceAboveUniform: number;
    frequencyGames: number; // the games whose report holds the reference against the frequencies (26 Oct 2026 on)
    referenceAboveFrequency: number;
    windows: { game: string; window: number | null; unit: string }[];
    missing: string[]; // every market game without a usable report, unreadable ones included
    unreadable: string[];
    from: string | null;
    at: string | null;
  };
}

// GET /api/admin/controls/:game - the drawer: the same objects in full
export interface Q0Mode {
  best: number | null;
  sd: number | null;
  ceiling: number | null;
  rowMean: number | null;
  random: number | null;
  selectionGain: number | null;
  rows: number | null;
  seeds: number | null;
  history: number | null;
  winners: string[];
  at: string | null;
  perRow: Record<string, number | null>;
}
export interface Q2Full extends Q2 {
  label: string;
  windowsTrain: number | null;
  features: number | null;
  realMean: number | null;
  realSd: number | null;
  nullMean: number | null;
  nullSd: number | null;
  shuffledMean: number | null;
  shuffledSd: number | null;
  thresholdExact: number | null;
  shuffledK: number | null; // k of the shuffled control's bound (rule 2 only)
  seedsScored: { real: number | null; null: number | null; shuffled: number | null } | null; // seeds whose suite scored (rule 2 only)
  perSeed: { real: number[]; null: number[]; shuffled: number[] };
  aboveBand: boolean;
  survivesShuffled: boolean;
}
export interface FcVariantFull {
  key: string;
  label: string;
  verdict: string;
  error: string | null;
  columns: number | null;
  noiseShare: number | null;
  chanceShare: number | null;
  ratio: number | null;
  t: number | null;
  band: number | null;
  above: string[];
  aucWithout: number | null;
  aucWith: number | null;
  cost: number | null;
  seconds: number | null;
  rule: string | null; // only when it differs from the record's rule
}
export interface RowsTableRow {
  row: string;
  kind: string; // 'market' | 'base' | 'meta' | 'baseline' ...
  reference: boolean;
  days: number | null;
  logScore: number | null;
  se: number | null;
  vsReference: Interval | null;
  vsUniform: Interval | null;
  frequency: boolean; // the baseline the vs-frequencies column is taken against
  vsFrequency: Interval | null;
  exact: number | null;
  adjacent: number | null;
  direction: number | null;
}
export interface Regime {
  row: string;
  date: string | null;
  label: string;
  regimes: number | null;
  probability: number | null;
  volatilityRank: number | null;
  history: number | null;
}
export interface ControlsGameView {
  game: string;
  group: GroupName;
  unit: string;
  builtAt: string | null;
  rankingAt: string | null;
  q0: Q0 & { table: Q0Line[]; modes: Record<string, Q0Mode> }; // table: every family row and every followed or judged row
  q2: Q2Full | null;
  fc: (Omit<Fc, 'variants'> & { rule: string | null; variants: FcVariantFull[] }) | null;
  // failed: the rows that failed in the report (admin.js rowsFull): why one scored no day, or how many days one raised on;
  // floor: a probability under it is scored as it (the report's probability_floor); smoothing: the add-one the baselines'
  // frequencies were counted with (null on a report from before 26 Oct 2026)
  rows: (MarketRows & { uniform: number | null; floor: number | null; smoothing: number | null; table: RowsTableRow[]; failed: { row: string; reason: string | null; days: number | null }[] }) | null;
  regimes: Regime[] | null;
  errors: string[];
}

// GET /api/admin/models and /api/admin/models/:game
export interface Measure {
  unit: string; // 'hits per draw' | 'EUR per bet'
  text: string;
  capEur: number | null;
  bound: string | null;
  against: string | null; // keno: a fair ticket (not money won); pick3, jokerplus: the payout table
}
// a game's Q0 row tests on the Models page (27 Oct 2026, in place of the band): nothing is greyed
export interface ModelsChance {
  test: Q0Test | null;
  error: string | null;
  inTile: boolean; // false for the payout games: information only
}
export interface SinceBest {
  row: string;
  value: number | null;
  draws: number | null;
  tooFew: boolean;
}
// the forward record from the design freeze; none: no record for this game yet
export interface SinceSummary {
  date: string | null;
  label: string;
  none: boolean;
  minDraws: number | null;
  rows: number | null;
  best: SinceBest | null;
}
export interface BestModel {
  row: string;
  value: number | null; // the ranking's value: the live record's when the game ranks on it (rankedOn 'live'), its figures below the same
  draws: number | null; // every stored draw
  young: boolean;
  allValue: number | null; // every stored draw's value beside a live one (null when the ranking reads every draw)
  liveDraws: number | null; // the live draws the value is over (null when the ranking reads every draw)
  profitPerBet: number | null;
  bets: number | null;
  avgHits: number | null;
  live: number | null;
  recovered: number | null;
  chance: Q0Line | null; // its Q0 row test (null: not tested)
}
// a market game: the row its users' page (Crypto, Shares) follows, by best book - not the History top five
export interface Follows {
  row: string;
  page: string; // 'Crypto' | 'Shares'
  book: string; // 'daily' | 'week'
  byBook: boolean; // false: no book long enough yet, the best exact rate stands in
  rule: string; // the book's rule, 'daily round trip', 'open to close' ...
}
export interface ModelsGameSummary {
  game: string;
  group: GroupName;
  unit: string;
  metric: string; // 'avg_hits' | 'profit_bound'
  measure: Measure;
  minDraws: number; // the ranking's guard: on live draws when it ranks on the live record
  rankedOn: 'live' | 'all'; // 'live' (lottery from 27 Oct 2026): only tickets saved before each draw's sales close
  rows: number;
  rankedRows: number;
  best: BestModel | null;
  chance: ModelsChance;
  since: SinceSummary | null;
  top: string[];
  learning: Record<string, LearningCheck>; // the best row's and the top five's, by name; a row without one is not in it
  follows: Follows | null;
  combination: Combination | null; // the best set (no top-ten list here: that is in the game's drawer)
}
// A deep-learning row's newest learning check (src/LearningCheck.py, 25 Oct 2026): 'counts-only' is the "no better
// than the frequencies" label; the losses are cross-entropy summed over the heads, without the L2 term. A label only.
export interface LearningCheck {
  verdict: 'learns' | 'counts-only';
  valLoss: number | null;
  baselineLoss: number | null; // the training split's class frequencies per position, smoothed as the row's loss
  margin: number | null; // what an output that ignores its input gets below the baseline on these windows
  sameTicket: number | null; // share of the validation windows on the most common argmax ticket
  inputBlind: boolean; // sameTicket 95% or more
  windows: number | null;
  date: string | null; // the day file that carries it: the newest draw the check's data runs to
}
export interface ModelsView {
  builtAt: string | null;
  generatedAt: string | null;
  window: string;
  since: { date: string | null; label: string } | null;
  topN: number;
  games: ModelsGameSummary[];
}
export interface RankRow {
  rank: number;
  row: string;
  top: boolean;
  value: number | null; // the ranking's value: the live record's when the game ranks on it; its figures below the same
  allValue: number | null; // every stored draw's value beside a live one (null when the ranking reads every draw)
  liveDraws: number | null;
  profitPerBet: number | null;
  bets: number | null;
  avgHits: number | null;
  bestDay: number | null;
  draws: number | null; // every stored draw
  live: number | null;
  recovered: number | null;
  young: boolean;
  chance: Q0Line | null; // its Q0 row test (null: not tested)
  since: { value: number | null; draws: number | null; young: boolean } | null | undefined; // undefined: the game has no since record
  bySize: { size: number; profitPerBet: number | null; bets: number | null }[] | null;
  learning: LearningCheck | null; // null: not a deep-learning row, or none since 25 Oct 2026
}
export interface MultiPick {
  draw: number;
  extra: number;
  minHits: number | null;
  perExtraChance: number | null;
  chance: Record<string, number | null>;
  grids: number | null;
  price: number | null;
  rowsWithExtras: number;
  draws: Range | null;
  rows: { row: string; draws: number | null; avgHits6: number | null; avgHits9: number | null; win6: number | null; win9: number | null; extraHits: (number | null)[] }[];
}
export interface ComboLine {
  members: string[];
  how: string;
  size: number | null;
  value: number | null;
  draws: number | null;
  profitPerBet: number | null;
  winDayRate: number | null;
  avgHits: number | null;
  bestDay: number | null;
}
export interface Combination {
  metric: string; // 'profit_per_draw' | 'avg_best_hits'
  measure: { unit: string; text: string; capped: boolean | null };
  candidateRows: number;
  evaluated: number | null;
  best: ComboLine | null;
  note: string | null;
  bestSingle: { row: string; value: number | null; draws: number | null } | null;
  beatsBestSingle: boolean | null;
  control: { shuffles: number | null; bestMean: number | null; bestP95: number | null; pValue: number | null; consistentWithLuck: boolean } | { error: string } | null;
  ranking: ComboLine[];
}
// A lottery game's liveness as the daily report judged it (src/Liveness.py, 27 Oct 2026)
export interface GameLiveness {
  checked: string | null; // 'stamp+git' | 'stamp only' (git could not be read) | 'unchecked' (no evidence: not ranked on it)
  error: string | null;
  draws: number | null; // scored draws judged
  allLive: number | null; // draws whose every row was live as stored
  allCounted: number | null; // draws with no row back-built or late (the recovered ones in)
  rowsLive: number | null;
  rowsRecovered: number | null;
  rowsNotLive: number | null;
  newestNotLive: string | null;
  rewriteAudit: { rows: number | null; hitsAtClose: number | null; hitsStored: number | null };
  unscheduled: string[];
}
export interface ModelsGameView {
  game: string;
  group: GroupName;
  unit: string;
  builtAt: string | null;
  generatedAt: string | null;
  window: string;
  rankedOn: 'live' | 'all';
  metric: string;
  measure: Measure;
  minDraws: number; // the ranking's guard: on live draws when it ranks on the live record
  allMinDraws: number; // every stored draw's guard
  liveness: GameLiveness | null;
  scope: Scope;
  shown: number;
  of: number;
  top: string[];
  follows: Follows | null;
  chance: ModelsChance; // the game's Q0 row tests (27 Oct 2026, in place of the band)
  since: { date: string | null; label: string; none: boolean; minDraws: number | null; bestModel: string | null; live: string | null } | null;
  rows: RankRow[];
  multiPick: MultiPick | null;
  combination: Combination | null;
}

// GET /api/admin/phase-shift and /api/admin/phase-shift/:game
export interface LagMeta {
  lags: string;
  runsKept: number | null;
  trailShown: number | null;
  tracked: string;
  z: string;
  control: string;
}
export interface PhaseShiftSummary {
  game: string;
  group: GroupName;
  unit: string;
  rows: number;
  tracked: number;
  topTracked: number;
  newestRun: string | null; // the newest run counted (a scan that added no draw is not a run)
  scanned: string | null; // the newest scan (the report's run date)
  runs: Range | null;
  commonLags: { lag: number; rows: number; tracked: number }[]; // over every row's most frequent peak
}
export interface PhaseShiftView {
  builtAt: string | null;
  generatedAt: string | null;
  meta: LagMeta;
  games: PhaseShiftSummary[];
}
// one step of the peak trail: a run of runs that agreed on one lag; avg hits and z are those of its newest run
export interface TrailStep {
  lag: number | null;
  runs: number;
  from: string | null;
  to: string | null;
  avgHits: number | null;
  z: number | null;
}
export interface LagRow {
  row: string;
  top: boolean;
  // run: the run the peak dates from; scanned: the newest scan; unchanged: that scan added no draw, so it is no run
  peak: { lag: number | null; avgHits: number | null; z: number | null; n: number | null; profileMean: number | null; run: string | null; scanned: string | null; unchanged: boolean };
  runs: number;
  mostFrequent: { lag: number | null; runs: number; share: number | null };
  tracked: boolean;
  trail: TrailStep[];
  earlierSteps: number;
}
export interface PhaseShiftGameView {
  game: string;
  group: GroupName;
  unit: string;
  builtAt: string | null;
  generatedAt: string | null;
  meta: LagMeta;
  scope: Scope;
  shown: number;
  of: number;
  top: string[];
  topMissing: string[]; // top rows with no lag record
  rows: LagRow[];
}

// GET /api/admin/randomness and /api/admin/randomness/:game
export interface RandomnessMeta {
  tripwires: { entropyBelow: number | null; klHistoryAbove: number | null; aeBelow: number | null };
  window: string;
  ae: string;
  modelKl: string;
  q2: string;
  fewWindows: number | null; // Q2's power rule (controlRules.json): a record on fewer held-out windows reads "too few draws"
}
export interface RandomnessLine {
  game: string;
  group: GroupName;
  unit: string;
  status: 'watch' | 'normal';
  reasons: string[]; // 'entropy' | 'kl' | 'ae' | 'q2'
  window: number | null;
  drawsTotal: number | null;
  entropy: number | null;
  klHistory: number | null;
  klUniform: number | null;
  ae: { minZ: number | null; latestZ: number | null; date: string | null; alert: boolean; window: number | null } | null;
  q2: (Q2 & { label: string; realSd: number | null; nullMean: number | null; nullSd: number | null }) | null;
  trend: { end: string | null; entropy: number | null; klHistory: number | null; klUniform: number | null }[];
  modelRows: number;
}
export interface RandomnessView {
  builtAt: string | null;
  generatedAt: string | null;
  meta: RandomnessMeta;
  games: RandomnessLine[];
}
export interface RandomnessGameView extends RandomnessLine {
  builtAt: string | null;
  generatedAt: string | null;
  meta: RandomnessMeta;
  scope: Scope;
  shown: number;
  of: number;
  top: string[];
  topMissing: string[]; // top rows with no KL row over the window
  models: { row: string; kl: number | null; top: boolean }[];
}

// GET /api/admin/week - This week: the weekend's runs, today's predictor, the tuners, the controls matrix, since Monday.
// Instants are ISO UTC (the page prints Brussels time); days 'YYYY-MM-DD' are Brussels days; durations in minutes.
export interface Usual {
  median: number | null;
  min: number | null;
  max: number | null;
  n: number; // earlier successful runs
  from: string | null;
  to: string | null;
  basis: string; // 'earlier runs in the job log' | 'earlier runs in the scheduler history'
}
// new: absent from every earlier slice of the same job the logs still hold; null: nothing earlier to compare with
export interface WarningSig {
  sig: string;
  n: number;
  new: boolean | null;
}
export interface RunLine {
  key: string;
  job: string;
  start: string | null;
  end: string | null;
  minutes: number | null;
  result: string; // 'ok' | 'exit N' | 'unknown'
  note: string | null;
  usual: Usual | null;
  overUsual: boolean | null;
  vsMedian: number | null;
  warnings: WarningSig[] | null; // null: the run's log slice is no longer in its log
  noiseFiltered: number | null;
  comparedWith: { runs: number; from: string | null; to: string | null } | null;
}
export interface HeldFromLastWeekend {
  weekend: string;
  replaced: number;
  kept: number;
  replacedAgain: number;
  reverted: number;
  notDecided: number;
}
export interface TuningTiles {
  n: number;
  replaced: number;
  kept: number;
  reverted: number;
  marginZero: number;
  replacedUnder1Se: number;
  replacedUnder1SeVsBetterReference: number;
  under1SeTuners: string[];
  unattributed: number;
  notThisWeekend: number;
  games: number;
  from: string | null;
  to: string | null;
  heldFromLastWeekend: HeldFromLastWeekend | null;
}
export interface WeekTuner {
  tuner: string;
  job: string | null;
  n: number;
  replaced: number;
  kept: number;
  reverted: number;
  games: number | null;
  marginZero: number;
  marginOneSe: number | null;
  marginSeCounts: Record<string, number>; // records by their margin's k in standard errors ('1', '2.5'); {} from an older server
  replacedUnder1Se: number;
}
export interface MatrixGame {
  game: string;
  group: GroupName;
  unit: string;
  errors: string[];
  // Q0 (27 Oct 2026): the row test's cell; band - the NullControls records' times, for what that run wrote only
  q0: {
    test: Omit<Q0Test, 'alpha' | 'verdictAlpha' | 'permutations' | 'block' | 'running' | 'checkedThrough' | 'stored'> | null;
    error: string | null;
    confirmedRows: string[];
    followedRows: string[];
    inTile: boolean;
    ranking: 'hits' | 'profit';
    at: string | null; // the report that ran the tests
    band: { at: string | null; modes: { mode: string; at: string | null }[] } | null;
    bandUnreadable: boolean; // no band: a NullControls record is there but did not load
    history: PartHistory | null;
  } | null;
  q2: { status: Q2Status; real: number | null; threshold: number | null; seeds: number | null; windowsTest: number | null; fewWindows: boolean; minWindows: number | null; at: string | null; history: PartHistory | null } | null;
  fc: { fits: number; inconclusive: number; ignored: number; failed: number; variants: number; at: string | null; history: PartHistory | null } | null;
  rows: { beat: number; scored: number; reference: string; window: number | null; referenceVsUniform: string | null; referenceVsFrequency: string | null; at: string | null; history: PartHistory | null } | null;
  gates: { replaced: number; kept: number; reverted: number; notThisWeekend: number } | null;
}
export interface CouncilStatus {
  error: string | null; // the health probe failed or did not answer in time
  name: string;
  status: string;
  healthy: boolean | null;
  summary: string;
  since: string | null;
  restarts: number | null;
  lastExit: { code: number | null; signal: string | null; at: string | null } | null;
  lastError: string | null;
  checkedAt: string | null;
}
export interface WeekView {
  builtAt: string | null;
  rankingAt: string | null;
  weekend: { saturday: string | null; from: string | null; until: string | null };
  tiles: {
    weekend: {
      runs: number;
      ok: number;
      expected: number;
      missing: { key: string; day: string | null }[];
      running: { key: string; start: string | null } | null;
      notOk: { key: string; result: string; note: string | null }[];
      from: string | null;
      to: string | null;
      hours: number | null;
      slowest: { key: string; minutes: number | null; usual: Usual | null } | null;
      newWarnings: { key: string; sig: string; n: number }[];
    };
    today: { day: string | null; run: RunLine | null; running: { start: string | null } | null };
    tuners: (TuningTiles & { weekend: string | null; byTuner: WeekTuner[] }) | null;
  } & ControlTiles;
  runs: RunLine[];
  matrix: { fewWindows: number | null; games: MatrixGame[] }; // fewWindows: Q2's power rule, "too few draws" under this many test windows
  readings: ReadingsMeta | null;
  since: {
    from: string | null;
    predictorRuns: RunLine[];
    queue: { key: string; name: string; queuedAt: string | null; manual: boolean }[];
    running: { key: string; name: string; start: string | null } | null;
    lock: { held: boolean; pid: number | null; script: string | null; key: string | null } | null; // null: not known here
    council: CouncilStatus | null; // null: not known here
    signIns: { failed: number; locked: number; ok: number; last7Days: { failed: number; locked: number }; scanned: number; oldestScanned: string | null } | null;
  };
}

// GET /api/admin/tuning and /api/admin/tuning/:game - the champion/challenger gate's decisions of the newest weekend
export interface TunerSummary {
  tuner: string; // 'statistical' | 'boosting' | 'RL ticket' | 'ensemble' | 'quantum' | 'keno sub-selections' | 'unattributed'
  job: string | null;
  n: number;
  replaced: number;
  kept: number;
  reverted: number;
  marginZero: number;
  marginOneSe: number;
  marginSeCounts: Record<string, number>; // records by their margin's k in standard errors ('1', '2.5'); {} from an older server
  replacedUnder1Se: number;
  replacedUnder1SeVsBetterReference: number;
  games: number;
  kinds: string[];
  aboveFair: number; // records whose capped-profit score is measured above a fair ticket (keno)
  fairPerBet: Range | null; // that fair ticket's capped value per bet
  windowDays: Range | null;
  windowByUnit: Record<string, Range>; // the gate window per unit: draws, days, trading days, weeks
  runTrials: Range | null;
  studyTrials: Range | null;
  run: { start: string | null; end: string | null; minutes: number | null; result: string } | null;
  replacedRows: Record<string, string[]>; // game -> rows
  revertedRows: Record<string, string[]>;
}
export interface GameGates {
  game: string;
  group: GroupName;
  n: number;
  replaced: number;
  kept: number;
  reverted: number;
  marginZero: number;
  notThisWeekend: number;
  noFile: boolean;
  replacedRows: string[];
  revertedRows: string[];
}
export interface TuningView {
  builtAt: string | null;
  weekend: string | null;
  tiles: TuningTiles;
  byTuner: TunerSummary[];
  byGame: GameGates[];
  unattributed: { game: string; row: string; weekend: string | null; family: string | null; why: string }[];
  history: { weekends: string[]; commits: number; error: string | null; source: string };
  meta: Record<string, string>;
}
export interface GateSide {
  score: number | null;
  se: number | null;
  days: number | null; // backtest days the score covers (can be fewer than the window)
  fairPerBet: number | null; // keno: the score is capped profit above a fair ticket of this capped value per bet, not money
  profitPerBet: number | null; // keno: the raw, uncapped profit per bet in EUR
}
// a weekend's decision: 'replaced' | 'kept' | 'reverted'; null: that weekend did not decide the row; 'not read': git could not be read
export type GateLetter = string | null;
export interface GateRow {
  row: string;
  tuner: string;
  by: string | null; // how it was attributed: 'record' | 'family + commit' | 'family + run'
  decision: string;
  thisWeekend: boolean;
  weekend: string | null;
  date: string | null;
  kind: string | null;
  windowDays: number | null;
  // the lockbox days no score of the decision counts (since 26 Oct 2026; null before): out of windowDays ('window', a row
  // tuner scores the other days), or out of the quantum tuner's score table before its holdout was cut ('table')
  lockboxDays: number | null;
  lockboxFrom: 'window' | 'table' | null;
  runTrials: number | null;
  studyTrials: number | null;
  default: GateSide;
  incumbent: GateSide;
  challenger: GateSide;
  margin: number | null;
  marginSe: number | null;
  // the k standard errors the tuner asked for (since 27 Oct 2026; null before): with k > 0 and no challenger SE the
  // gate kept the incumbent for want of one
  marginK: number | null;
  // the margin of the revert's default-over-incumbent comparison (since 27 Oct 2026): 0 = none (statistical, RL,
  // ensemble - the default is not picked from trials); null on an older record, whose revert took the margin
  revertMargin: number | null;
  // the RL Position row: whether the served sizing sits out, and why (null on every other row)
  sitOut: boolean | null;
  sitOutReason: string | null;
  bar: number | null;
  gainSe: number | null;
  gainBestSe: number | null;
  gainBarSe: number | null;
  history: GateLetter[];
  why: string | null;
  family: string | null;
}
export interface TuningGameView {
  game: string;
  group: GroupName;
  unit: string;
  builtAt: string | null;
  weekend: string | null;
  n: number;
  replaced: number;
  kept: number;
  reverted: number;
  notThisWeekend: number;
  weekends: string[];
  historyError: string | null;
  rows: GateRow[];
}

export const getAdminControls = async (): Promise<ControlsView> => normControls(await getJson<unknown>('/api/admin/controls'));
export const getAdminControlsGame = async (game: string): Promise<ControlsGameView> =>
  normControlsGame(await getJson<unknown>(`/api/admin/controls/${encodeURIComponent(game)}`), game);
export const getAdminModels = async (): Promise<ModelsView> => normModels(await getJson<unknown>('/api/admin/models'));
export const getAdminModelsGame = async (game: string, scope: Scope): Promise<ModelsGameView> =>
  normModelsGame(await getJson<unknown>(`/api/admin/models/${encodeURIComponent(game)}?scope=${scope}`), game, scope);
export const getAdminPhaseShift = async (): Promise<PhaseShiftView> => normPhaseShift(await getJson<unknown>('/api/admin/phase-shift'));
export const getAdminPhaseShiftGame = async (game: string, scope: Scope): Promise<PhaseShiftGameView> =>
  normPhaseShiftGame(await getJson<unknown>(`/api/admin/phase-shift/${encodeURIComponent(game)}?scope=${scope}`), game, scope);
export const getAdminRandomness = async (): Promise<RandomnessView> => normRandomness(await getJson<unknown>('/api/admin/randomness'));
export const getAdminRandomnessGame = async (game: string, scope: Scope): Promise<RandomnessGameView> =>
  normRandomnessGame(await getJson<unknown>(`/api/admin/randomness/${encodeURIComponent(game)}?scope=${scope}`), game, scope);
export const getAdminWeek = async (): Promise<WeekView> => normWeek(await getJson<unknown>('/api/admin/week'));
export const getAdminTuning = async (): Promise<TuningView> => normTuning(await getJson<unknown>('/api/admin/tuning'));
export const getAdminTuningGame = async (game: string): Promise<TuningGameView> => normTuningGame(await getJson<unknown>(`/api/admin/tuning/${encodeURIComponent(game)}`), game);

// --- normalisation of the admin views ---
type Raw = Record<string, unknown>;
const o = (v: unknown): Raw => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Raw) : {});
const has = (v: unknown): boolean => Boolean(v) && typeof v === 'object' && !Array.isArray(v);
const nums = (v: unknown): number[] => arr(v as number[]).filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
const strs = (v: unknown): string[] => arr(v as string[]).filter((x): x is string => typeof x === 'string');
const ostr = (v: unknown): string | null => (typeof v === 'string' && v ? v : null);
const bool = (v: unknown): boolean => v === true;
const unitOf = (v: unknown, game: string): string => str(v, MARKET_GAMES.includes(game) ? 'days' : 'draws');
const groupName = (v: unknown, game: string): GroupName => (v === 'Lottery' || v === 'Markets' ? v : groupOf(game));
const rangeOf = (v: unknown): Range | null => {
  const r = o(v);
  return opt(r.min) !== null && opt(r.max) !== null ? { min: r.min as number, max: r.max as number } : null;
};
const intervalOf = (v: unknown): Interval | null => {
  if (!has(v)) return null;
  const r = o(v);
  return { mean: opt(r.mean), lo: opt(r.lo), hi: opt(r.hi), verdict: str(r.verdict) };
};
const seedsOf = (v: unknown): Record<string, number[]> => Object.fromEntries(Object.entries(o(v)).map(([k, x]) => [k, nums(x)]));
const scopeOf = (v: unknown, fallback: Scope): Scope => (isScope(v) ? v : fallback);
const missingOf = (v: unknown): MissingControl[] => arr(v as unknown[]).filter(has).map((x) => ({ mode: str(o(x).mode), unreadable: bool(o(x).unreadable) }));
const modesOf = (v: unknown): { mode: string; at: string | null }[] => arr(v as unknown[]).filter(has).map((x) => ({ mode: str(o(x).mode), at: ostr(o(x).at) }));

const FRESH_STATES: FreshState[] = ['running', 'stopped', 'confirmed', 'not confirmed'];
// a fresh state this page does not know is no fresh record: never a verdict it cannot read
function q0FreshOf(v: unknown): Q0Fresh | null {
  const f = o(v);
  if (!FRESH_STATES.includes(f.state as FreshState)) return null;
  return {
    state: f.state as FreshState, since: ostr(f.since), flagDraws: opt(f.flagDraws), flagP: opt(f.flagP), flagMean: opt(f.flagMean), flagNull: opt(f.flagNull),
    verdictLength: opt(f.verdictLength), draws: num(f.draws, 0), last: ostr(f.last), verdictAt: ostr(f.verdictAt), decidedOn: ostr(f.decidedOn),
    p: opt(f.p), mean: opt(f.mean), null: opt(f.null),
  };
}
function q0LineOf(v: unknown): Q0Line | null {
  if (!has(v)) return null;
  const r = o(v);
  return {
    row: str(r.row), mean: opt(r.mean), live: num(r.live, 0), notLive: num(r.notLive, 0), null: opt(r.null), p: opt(r.p), pHolm: opt(r.pHolm), inFamily: bool(r.inFamily),
    recovered: opt(r.recovered), fresh: has(r.fresh) ? q0FreshOf(r.fresh) : null,
  };
}
const q0LinesOf = (v: unknown): Q0Line[] => arr(v as unknown[]).map(q0LineOf).filter((x): x is Q0Line => x !== null);
function q0TestOf(v: unknown): Q0Test | null {
  if (!has(v)) return null;
  const t = o(v);
  return {
    method: t.method === 'permutation' ? 'permutation' : 'exact', alpha: num(t.alpha, 0.05), verdictAlpha: num(t.verdictAlpha, 0.01), minDraws: opt(t.minDraws),
    permutations: opt(t.permutations), block: opt(t.block), family: num(t.family, 0), liveMax: num(t.liveMax, 0), under: num(t.under, 0), expected: num(t.expected, 0),
    holds: num(t.holds, 0), followed: num(t.followed, 0), running: num(t.running, 0), stopped: num(t.stopped, 0), confirmed: num(t.confirmed, 0),
    notConfirmed: num(t.notConfirmed, 0), checkedThrough: ostr(t.checkedThrough), stored: ostr(t.stored),
  };
}
function q0Of(v: unknown): Q0 {
  const r = o(v);
  const b = o(r.band);
  return {
    ranking: r.ranking === 'profit' ? 'profit' : 'hits',
    inTile: r.inTile !== false,
    test: q0TestOf(r.test),
    error: ostr(r.error),
    lowest: q0LineOf(r.lowest),
    rows: q0LinesOf(r.rows),
    confirmed: strs(r.confirmed),
    followed: strs(r.followed),
    leader: has(r.leader) ? { row: str(o(r.leader).row), draws: opt(o(r.leader).draws) } : null,
    allRows: num(r.allRows, 0),
    minDraws: num(r.minDraws, 0),
    band: has(r.band)
      ? { ceiling: opt(b.ceiling), from: str(b.from), rows: opt(b.rows), seeds: opt(b.seeds), history: opt(b.history), metric: str(b.metric, 'hits per draw'), at: ostr(b.at), modes: modesOf(b.modes), missing: missingOf(b.missing) }
      : null,
    bandUnreadable: bool(r.bandUnreadable),
  };
}
const Q2_STATUS: Q2Status[] = ['above', 'within', 'too few draws', 'above band, not shuffled', 'failed'];
// a status this page does not know is no reading: 'failed', never a clean 'within'
const q2StatusOf = (v: unknown): Q2Status => (Q2_STATUS.includes(v as Q2Status) ? (v as Q2Status) : 'failed');
function q2Of(v: unknown): Q2 | null {
  if (!has(v)) return null;
  const r = o(v);
  return {
    status: q2StatusOf(r.status),
    rule: num(r.rule, 1),
    real: opt(r.real),
    threshold: opt(r.threshold),
    thresholdSe: opt(r.thresholdSe),
    shuffled: opt(r.shuffled),
    shuffledThreshold: opt(r.shuffledThreshold),
    bandK: opt(r.bandK),
    seeds: opt(r.seeds),
    windowsTest: opt(r.windowsTest),
    fewWindows: bool(r.fewWindows),
    minWindows: opt(r.minWindows),
    window: opt(r.window),
    drawsCap: opt(r.drawsCap),
    drawsUsed: opt(r.drawsUsed),
    quantum: bool(r.quantum),
    at: ostr(r.at),
  };
}
function fcOf(v: unknown): Fc | null {
  if (!has(v)) return null;
  const r = o(v);
  return {
    fits: num(r.fits, 0),
    inconclusive: num(r.inconclusive, 0),
    ignored: num(r.ignored, 0),
    failed: num(r.failed, 0),
    columns: opt(r.columns),
    chanceShare: opt(r.chanceShare),
    tableRows: opt(r.tableRows),
    lockboxRows: opt(r.lockboxRows),
    behind: opt(r.behind),
    positional: bool(r.positional),
    repeats: opt(r.repeats),
    noiseColumns: opt(r.noiseColumns),
    columnsAbove: num(r.columnsAbove, 0),
    at: ostr(r.at),
    variants: arr(r.variants as unknown[]).map((x) => {
      const w = o(x);
      const auc = arr(w.auc as unknown[]);
      return { key: str(w.key), verdict: str(w.verdict), share: opt(w.share), auc: [opt(auc[0]), opt(auc[1])], above: strs(w.above) };
    }),
  };
}
function rowsOf(v: unknown): MarketRows | null {
  if (!has(v)) return null;
  const r = o(v);
  return {
    window: opt(r.window),
    requested: opt(r.requested),
    firstDay: ostr(r.firstDay),
    lastDay: ostr(r.lastDay),
    lockboxWithheld: opt(r.lockboxWithheld),
    reference: str(r.reference, 'GARCH Model'),
    rowsTotal: num(r.rowsTotal, 0),
    rowsScored: num(r.rowsScored, 0),
    baselines: num(r.baselines, 0),
    betterThanUniform: strs(r.betterThanUniform),
    betterThanReference: strs(r.betterThanReference),
    noDifferenceVsReference: num(r.noDifferenceVsReference, 0),
    referenceVsUniform: intervalOf(r.referenceVsUniform),
    frequency: ostr(r.frequency),
    betterThanFrequency: strs(r.betterThanFrequency),
    referenceVsFrequency: intervalOf(r.referenceVsFrequency),
    frequencyVsUniform: intervalOf(r.frequencyVsUniform),
    errors: num(r.errors, 0),
    hmmFeatures: ostr(r.hmmFeatures),
    at: ostr(r.at),
  };
}
function partHistoryOf(v: unknown): PartHistory | null {
  if (!has(v)) return null;
  const r = o(v);
  const since = o(r.sinceReading);
  return {
    state: str(r.state), sunday: ostr(r.sunday), weeks: num(r.weeks, 1), changed: typeof r.changed === 'boolean' ? r.changed : null, last: ostr(r.last), behind: bool(r.behind),
    noRunBefore: bool(r.noRunBefore), live: bool(r.live),
    sinceReading: has(r.sinceReading) ? { now: str(since.now), added: strs(since.added), dropped: strs(since.dropped), rankingAt: ostr(since.rankingAt) } : null,
    ruleChanged: r.ruleChanged === true ? { rule: num(r.lastRule, 1), seeds: opt(r.lastSeeds) } : null,
  };
}
function gameHistoryOf(v: unknown): GameHistory {
  const r = o(v);
  return { q0: partHistoryOf(r.q0), q2: partHistoryOf(r.q2), fc: partHistoryOf(r.fc), rows: partHistoryOf(r.rows) };
}
function readingsOf(v: unknown): ReadingsMeta | null {
  if (!has(v)) return null;
  const r = o(v);
  return { stored: num(r.stored, 0), sundays: strs(r.sundays), newestRecords: ostr(r.newestRecords), unreadable: strs(r.unreadable), rule: str(r.rule) };
}
function gameControlsOf(v: unknown): GameControls {
  const r = o(v);
  const game = str(r.game);
  return { game, unit: unitOf(r.unit, game), q0: q0Of(r.q0), q2: q2Of(r.q2), fc: fcOf(r.fc), rows: rowsOf(r.rows), errors: strs(r.errors), history: gameHistoryOf(r.history) };
}

// The four controls' tiles: the Controls page's, and the same objects on This week
function controlTilesOf(v: unknown): ControlTiles {
  const t = o(v);
  const [q0, q2, fc, rows] = [o(t.q0), o(t.q2), o(t.fc), o(t.rows)];
  const pay = o(q0.payout);
  const band = o(q0.band);
  return {
    q0: {
      confirmed: num(q0.confirmed, 0),
      confirmedRows: arr(q0.confirmedRows as unknown[]).filter(has).map((x) => ({ game: str(o(x).game), row: str(o(x).row) })),
      holds: num(q0.holds, 0),
      under: num(q0.under, 0),
      expected: num(q0.expected, 0),
      family: num(q0.family, 0),
      followed: num(q0.followed, 0),
      stopped: num(q0.stopped, 0),
      notConfirmed: num(q0.notConfirmed, 0),
      games: num(q0.games, 0),
      hitRankedGames: num(q0.hitRankedGames, 0),
      tooFew: arr(q0.tooFew as unknown[]).filter(has).map((x) => ({ game: str(o(x).game), liveMax: num(o(x).liveMax, 0), minDraws: opt(o(x).minDraws) })),
      payout: { games: strs(pay.games), under: num(pay.under, 0), expected: num(pay.expected, 0), holds: num(pay.holds, 0), followed: num(pay.followed, 0),
        confirmed: num(pay.confirmed, 0), notConfirmed: num(pay.notConfirmed, 0) },
      noTest: arr(q0.noTest as unknown[]).filter(has).map((x) => ({ game: str(o(x).game), why: ostr(o(x).why) })),
      at: ostr(q0.at),
      band: { games: num(band.games, 0), from: ostr(band.from), at: ostr(band.at),
        oneControl: arr(band.oneControl as unknown[]).filter(has).map((x) => ({ game: str(o(x).game), from: str(o(x).from), missing: missingOf(o(x).missing) })) },
    },
    q2: {
      above: num(q2.above, 0),
      of: num(q2.of, 0),
      aboveGames: strs(q2.aboveGames),
      tooFewGames: strs(q2.tooFewGames),
      failedGames: strs(q2.failedGames),
      seeds: seedsOf(q2.seeds),
      missing: strs(q2.missing),
      unreadable: strs(q2.unreadable),
      from: ostr(q2.from),
      at: ostr(q2.at),
    },
    fc: {
      fits: num(fc.fits, 0),
      inconclusive: num(fc.inconclusive, 0),
      ignored: num(fc.ignored, 0),
      failed: num(fc.failed, 0),
      variants: num(fc.variants, 0),
      games: num(fc.games, 0),
      missing: strs(fc.missing),
      unreadable: strs(fc.unreadable),
      from: ostr(fc.from),
      at: ostr(fc.at),
    },
    rows: {
      beatReference: num(rows.beatReference, 0),
      games: num(rows.games, 0),
      referenceAboveUniform: num(rows.referenceAboveUniform, 0),
      frequencyGames: num(rows.frequencyGames, 0),
      referenceAboveFrequency: num(rows.referenceAboveFrequency, 0),
      windows: arr(rows.windows as unknown[]).filter(has).map((x) => ({ game: str(o(x).game), window: opt(o(x).window), unit: str(o(x).unit, 'days') })),
      missing: strs(rows.missing),
      unreadable: strs(rows.unreadable),
      from: ostr(rows.from),
      at: ostr(rows.at),
    },
  };
}

function normControls(raw: unknown): ControlsView {
  const r = o(raw);
  const m = o(r.meta);
  const axis = nums(o(m.q2).axis);
  return {
    builtAt: ostr(r.builtAt),
    rankingAt: ostr(r.rankingAt),
    readings: readingsOf(r.readings),
    meta: {
      q0: { rule: str(o(m.q0).rule), test: str(o(m.q0).test), under: str(o(m.q0).under), holds: str(o(m.q0).holds), followed: str(o(m.q0).followed),
        headline: str(o(m.q0).headline), band: str(o(m.q0).band), rows: rangeOf(o(m.q0).rows), history: rangeOf(o(m.q0).history), seeds: seedsOf(o(m.q0).seeds) },
      q2: { metric: str(o(m.q2).metric), axis: axis.length === 2 ? [axis[0], axis[1]] : [0.5, 1], fewWindows: opt(o(m.q2).fewWindows), rule: str(o(m.q2).rule) },
      fc: { share: str(o(m.fc).share), chanceShare: str(o(m.fc).chanceShare), auc: str(o(m.fc).auc), variants: Object.fromEntries(Object.entries(o(o(m.fc).variants)).map(([k, x]) => [k, str(x, k)])) },
      rows: { score: str(o(m.rows).score), pnl: str(o(m.rows).pnl) },
    },
    tiles: controlTilesOf(r.tiles),
    groups: arr(r.groups as unknown[]).map((g) => ({
      name: o(g).name === 'Markets' ? 'Markets' : 'Lottery',
      games: arr(o(g).games as unknown[]).filter(has).map(gameControlsOf),
    })),
  };
}

function normControlsGame(raw: unknown, game: string): ControlsGameView {
  const r = o(raw);
  const base = gameControlsOf({ ...r, game: str(r.game, game) });
  const modes: Record<string, Q0Mode> = {};
  Object.entries(o(o(r.q0).modes)).forEach(([name, x]) => {
    const m = o(x);
    modes[name] = {
      best: opt(m.best), sd: opt(m.sd), ceiling: opt(m.ceiling), rowMean: opt(m.rowMean), random: opt(m.random), selectionGain: opt(m.selectionGain),
      rows: opt(m.rows), seeds: opt(m.seeds), history: opt(m.history), winners: strs(m.winners), at: ostr(m.at),
      perRow: Object.fromEntries(Object.entries(o(m.perRow)).map(([k, y]) => [k, opt(y)])),
    };
  });
  let q2: Q2Full | null = null;
  const q2Base = q2Of(r.q2);
  if (q2Base) {
    const x = o(r.q2);
    const ps = o(x.perSeed);
    q2 = {
      ...q2Base, label: str(x.label), windowsTrain: opt(x.windowsTrain), features: opt(x.features),
      realMean: opt(x.realMean), realSd: opt(x.realSd), nullMean: opt(x.nullMean), nullSd: opt(x.nullSd), shuffledMean: opt(x.shuffledMean), shuffledSd: opt(x.shuffledSd),
      thresholdExact: opt(x.thresholdExact), shuffledK: opt(x.shuffledK),
      seedsScored: has(x.seedsScored) ? { real: opt(o(x.seedsScored).real), null: opt(o(x.seedsScored).null), shuffled: opt(o(x.seedsScored).shuffled) } : null,
      perSeed: { real: nums(ps.real), null: nums(ps.null), shuffled: nums(ps.shuffled) },
      aboveBand: bool(x.aboveBand), survivesShuffled: bool(x.survivesShuffled),
    };
  }
  const fcBase = fcOf(r.fc);
  const fc = fcBase
    ? {
        ...fcBase,
        rule: ostr(o(r.fc).rule),
        variants: arr(o(r.fc).variants as unknown[]).map((x) => {
          const w = o(x);
          return {
            key: str(w.key), label: str(w.label, str(w.key)), verdict: str(w.verdict), error: ostr(w.error), columns: opt(w.columns),
            noiseShare: opt(w.noiseShare), chanceShare: opt(w.chanceShare), ratio: opt(w.ratio), t: opt(w.t), band: opt(w.band), above: strs(w.above),
            aucWithout: opt(w.aucWithout), aucWith: opt(w.aucWith), cost: opt(w.cost), seconds: opt(w.seconds), rule: ostr(w.rule),
          };
        }),
      }
    : null;
  const rowsBase = rowsOf(r.rows);
  const rows = rowsBase
    ? {
        ...rowsBase,
        uniform: opt(o(r.rows).uniform),
        floor: opt(o(r.rows).floor),
        smoothing: opt(o(r.rows).smoothing),
        failed: arr(o(r.rows).failed as unknown[]).filter(has).map((x) => ({ row: str(o(x).row), reason: ostr(o(x).reason), days: opt(o(x).days) })),
        table: arr(o(r.rows).table as unknown[]).map((x) => {
          const w = o(x);
          return {
            row: str(w.row), kind: str(w.kind), reference: bool(w.reference), days: opt(w.days), logScore: opt(w.logScore), se: opt(w.se),
            vsReference: intervalOf(w.vsReference), vsUniform: intervalOf(w.vsUniform), frequency: bool(w.frequency), vsFrequency: intervalOf(w.vsFrequency),
            exact: opt(w.exact), adjacent: opt(w.adjacent), direction: opt(w.direction),
          };
        }),
      }
    : null;
  return {
    game: base.game,
    group: groupName(r.group, base.game),
    unit: base.unit,
    builtAt: ostr(r.builtAt),
    rankingAt: ostr(r.rankingAt),
    q0: { ...base.q0, table: q0LinesOf(o(r.q0).table), modes },
    q2,
    fc,
    rows,
    regimes: Array.isArray(r.regimes)
      ? r.regimes.map((x) => {
          const w = o(x);
          return { row: str(w.row), date: ostr(w.date), label: str(w.label), regimes: opt(w.regimes), probability: opt(w.probability), volatilityRank: opt(w.volatilityRank), history: opt(w.history) };
        })
      : null,
    errors: base.errors,
  };
}

function measureOf(v: unknown): Measure {
  const m = o(v);
  return { unit: str(m.unit, 'hits per draw'), text: str(m.text), capEur: opt(m.capEur), bound: ostr(m.bound), against: ostr(m.against) };
}
function modelsChanceOf(v: unknown): ModelsChance {
  const c = o(v);
  return { test: q0TestOf(c.test), error: ostr(c.error), inTile: c.inTile !== false };
}
function followsOf(v: unknown): Follows | null {
  if (!has(v)) return null;
  const f = o(v);
  return { row: str(f.row), page: str(f.page), book: str(f.book), byBook: bool(f.byBook), rule: str(f.rule) };
}
function sinceSummaryOf(v: unknown): SinceSummary | null {
  if (!has(v)) return null;
  const s = o(v);
  const best = o(s.best);
  return {
    date: ostr(s.date), label: str(s.label), none: bool(s.none), minDraws: opt(s.minDraws), rows: opt(s.rows),
    best: has(s.best) ? { row: str(best.row), value: opt(best.value), draws: opt(best.draws), tooFew: bool(best.tooFew) } : null,
  };
}
// a verdict other than the two LearningCheck.py writes reads as no check at all
function learningCheckOf(v: unknown): LearningCheck | null {
  const l = o(v);
  if (l.verdict !== 'learns' && l.verdict !== 'counts-only') return null;
  return {
    verdict: l.verdict, valLoss: opt(l.valLoss), baselineLoss: opt(l.baselineLoss), margin: opt(l.margin), sameTicket: opt(l.sameTicket),
    inputBlind: bool(l.inputBlind), windows: opt(l.windows), date: ostr(l.date),
  };
}

function normModels(raw: unknown): ModelsView {
  const r = o(raw);
  const since = o(r.since);
  return {
    builtAt: ostr(r.builtAt),
    generatedAt: ostr(r.generatedAt),
    window: str(r.window, 'every scored draw'),
    since: has(r.since) ? { date: ostr(since.date), label: str(since.label) } : null,
    topN: num(r.topN, 5),
    games: arr(r.games as unknown[]).filter(has).map((x) => {
      const g = o(x);
      const game = str(g.game);
      const b = o(g.best);
      return {
        game, group: groupName(g.group, game), unit: unitOf(g.unit, game), metric: str(g.metric, 'avg_hits'), measure: measureOf(g.measure),
        minDraws: num(g.minDraws, 0), rankedOn: g.rankedOn === 'live' ? 'live' : 'all', rows: num(g.rows, 0), rankedRows: num(g.rankedRows, 0),
        best: has(g.best)
          ? { row: str(b.row), value: opt(b.value), draws: opt(b.draws), young: bool(b.young), allValue: opt(b.allValue), liveDraws: opt(b.liveDraws), profitPerBet: opt(b.profitPerBet),
              bets: opt(b.bets), avgHits: opt(b.avgHits), live: opt(b.live), recovered: opt(b.recovered), chance: q0LineOf(b.chance) }
          : null,
        chance: modelsChanceOf(g.chance),
        since: sinceSummaryOf(g.since),
        top: strs(g.top),
        learning: Object.fromEntries(Object.entries(o(g.learning)).flatMap(([k, x]) => {
          const l = learningCheckOf(x);
          return l ? [[k, l] as const] : [];
        })),
        follows: followsOf(g.follows),
        combination: combinationOf(g.combination),
      };
    }),
  };
}

function comboLineOf(v: unknown): ComboLine {
  const c = o(v);
  return {
    members: strs(c.members), how: str(c.how), size: opt(c.size), value: opt(c.value), draws: opt(c.draws),
    profitPerBet: opt(c.profitPerBet), winDayRate: opt(c.winDayRate), avgHits: opt(c.avgHits), bestDay: opt(c.bestDay),
  };
}
function combinationOf(v: unknown): Combination | null {
  if (!has(v)) return null;
  const c = o(v);
  const m = o(c.measure);
  const ctl = o(c.control);
  const single = o(c.bestSingle);
  return {
    metric: str(c.metric),
    measure: { unit: str(m.unit), text: str(m.text), capped: typeof m.capped === 'boolean' ? m.capped : null },
    candidateRows: num(c.candidateRows, 0),
    evaluated: opt(c.evaluated),
    best: has(c.best) ? comboLineOf(c.best) : null,
    note: ostr(c.note),
    bestSingle: has(c.bestSingle) ? { row: str(single.row), value: opt(single.value), draws: opt(single.draws) } : null,
    beatsBestSingle: typeof c.beatsBestSingle === 'boolean' ? c.beatsBestSingle : null,
    control: !has(c.control)
      ? null
      : typeof ctl.error === 'string'
        ? { error: ctl.error }
        : { shuffles: opt(ctl.shuffles), bestMean: opt(ctl.bestMean), bestP95: opt(ctl.bestP95), pValue: opt(ctl.pValue), consistentWithLuck: bool(ctl.consistentWithLuck) },
    ranking: arr(c.ranking as unknown[]).map(comboLineOf),
  };
}
function multiPickOf(v: unknown): MultiPick | null {
  if (!has(v)) return null;
  const m = o(v);
  return {
    draw: num(m.draw, 6), extra: num(m.extra, 3), minHits: opt(m.minHits), perExtraChance: opt(m.perExtraChance),
    chance: Object.fromEntries(Object.entries(o(m.chance)).map(([k, x]) => [k, opt(x)])), grids: opt(m.grids), price: opt(m.price),
    rowsWithExtras: num(m.rowsWithExtras, 0), draws: rangeOf(m.draws),
    rows: arr(m.rows as unknown[]).map((x) => {
      const w = o(x);
      return { row: str(w.row), draws: opt(w.draws), avgHits6: opt(w.avgHits6), avgHits9: opt(w.avgHits9), win6: opt(w.win6), win9: opt(w.win9), extraHits: arr(w.extraHits as unknown[]).map(opt) };
    }),
  };
}

function normModelsGame(raw: unknown, game: string, scope: Scope): ModelsGameView {
  const r = o(raw);
  const g = str(r.game, game);
  const s = o(r.since);
  return {
    game: g, group: groupName(r.group, g), unit: unitOf(r.unit, g), builtAt: ostr(r.builtAt), generatedAt: ostr(r.generatedAt), window: str(r.window, 'every scored draw'),
    rankedOn: r.rankedOn === 'live' ? 'live' : 'all', metric: str(r.metric, 'avg_hits'), measure: measureOf(r.measure), minDraws: num(r.minDraws, 0),
    allMinDraws: num(r.allMinDraws, num(r.minDraws, 0)), liveness: gameLivenessOf(r.liveness),
    scope: scopeOf(r.scope, scope), shown: num(r.shown, 0), of: num(r.of, 0), top: strs(r.top), follows: followsOf(r.follows),
    chance: modelsChanceOf(r.chance),
    since: has(r.since) ? { date: ostr(s.date), label: str(s.label), none: bool(s.none), minDraws: opt(s.minDraws), bestModel: ostr(s.bestModel), live: ostr(s.live) } : null,
    rows: arr(r.rows as unknown[]).filter(has).map((x, i) => {
      const w = o(x);
      const since = w.since === undefined ? undefined : has(w.since) ? { value: opt(o(w.since).value), draws: opt(o(w.since).draws), young: bool(o(w.since).young) } : null;
      return {
        rank: num(w.rank, i + 1), row: str(w.row), top: bool(w.top), value: opt(w.value), allValue: opt(w.allValue), liveDraws: opt(w.liveDraws), profitPerBet: opt(w.profitPerBet), bets: opt(w.bets),
        avgHits: opt(w.avgHits), bestDay: opt(w.bestDay), draws: opt(w.draws), live: opt(w.live), recovered: opt(w.recovered), young: bool(w.young), chance: q0LineOf(w.chance), since,
        bySize: Array.isArray(w.bySize) ? w.bySize.map((y) => ({ size: num(o(y).size, 0), profitPerBet: opt(o(y).profitPerBet), bets: opt(o(y).bets) })) : null,
        learning: learningCheckOf(w.learning),
      };
    }),
    multiPick: multiPickOf(r.multiPick),
    combination: combinationOf(r.combination),
  };
}

function gameLivenessOf(v: unknown): GameLiveness | null {
  if (!has(v)) return null;
  const l = o(v);
  const a = o(l.rewriteAudit);
  return { checked: ostr(l.checked), error: ostr(l.error), draws: opt(l.draws), allLive: opt(l.allLive), allCounted: opt(l.allCounted), rowsLive: opt(l.rowsLive),
    rowsRecovered: opt(l.rowsRecovered), rowsNotLive: opt(l.rowsNotLive), newestNotLive: ostr(l.newestNotLive),
    rewriteAudit: { rows: opt(a.rows), hitsAtClose: opt(a.hitsAtClose), hitsStored: opt(a.hitsStored) }, unscheduled: strs(l.unscheduled) };
}

function lagMetaOf(v: unknown): LagMeta {
  const m = o(v);
  return { lags: str(m.lags), runsKept: opt(m.runsKept), trailShown: opt(m.trailShown), tracked: str(m.tracked), z: str(m.z), control: str(m.control) };
}
function normPhaseShift(raw: unknown): PhaseShiftView {
  const r = o(raw);
  return {
    builtAt: ostr(r.builtAt), generatedAt: ostr(r.generatedAt), meta: lagMetaOf(r.meta),
    games: arr(r.games as unknown[]).filter(has).map((x) => {
      const g = o(x);
      const game = str(g.game);
      return {
        game, group: groupName(g.group, game), unit: unitOf(g.unit, game), rows: num(g.rows, 0), tracked: num(g.tracked, 0), topTracked: num(g.topTracked, 0),
        newestRun: ostr(g.newestRun), scanned: ostr(g.scanned), runs: rangeOf(g.runs),
        commonLags: arr(g.commonLags as unknown[]).map((y) => ({ lag: num(o(y).lag, 0), rows: num(o(y).rows, 0), tracked: num(o(y).tracked, 0) })),
      };
    }),
  };
}
function normPhaseShiftGame(raw: unknown, game: string, scope: Scope): PhaseShiftGameView {
  const r = o(raw);
  const g = str(r.game, game);
  return {
    game: g, group: groupName(r.group, g), unit: unitOf(r.unit, g), builtAt: ostr(r.builtAt), generatedAt: ostr(r.generatedAt), meta: lagMetaOf(r.meta),
    scope: scopeOf(r.scope, scope), shown: num(r.shown, 0), of: num(r.of, 0), top: strs(r.top), topMissing: strs(r.topMissing),
    rows: arr(r.rows as unknown[]).filter(has).map((x) => {
      const w = o(x);
      const p = o(w.peak);
      const f = o(w.mostFrequent);
      return {
        row: str(w.row), top: bool(w.top),
        peak: { lag: opt(p.lag), avgHits: opt(p.avgHits), z: opt(p.z), n: opt(p.n), profileMean: opt(p.profileMean), run: ostr(p.run), scanned: ostr(p.scanned), unchanged: bool(p.unchanged) },
        runs: num(w.runs, 0), mostFrequent: { lag: opt(f.lag), runs: num(f.runs, 0), share: opt(f.share) }, tracked: bool(w.tracked),
        // a step travels as [lag, runs, from, to, avgHits, z] (meta.trailFields)
        trail: arr(w.trail as unknown[]).filter(Array.isArray).map((s) => {
          const t = s as unknown[];
          return { lag: opt(t[0]), runs: num(t[1], 1), from: ostr(t[2]), to: ostr(t[3]), avgHits: opt(t[4]), z: opt(t[5]) };
        }),
        earlierSteps: num(w.earlierSteps, 0),
      };
    }),
  };
}

function randomnessMetaOf(v: unknown): RandomnessMeta {
  const m = o(v);
  const t = o(m.tripwires);
  return { tripwires: { entropyBelow: opt(t.entropyBelow), klHistoryAbove: opt(t.klHistoryAbove), aeBelow: opt(t.aeBelow) }, window: str(m.window), ae: str(m.ae), modelKl: str(m.modelKl), q2: str(m.q2), fewWindows: opt(m.fewWindows) };
}
function randomnessLineOf(v: unknown, fallbackGame = ''): RandomnessLine {
  const r = o(v);
  const game = str(r.game, fallbackGame);
  const ae = o(r.ae);
  const q2 = q2Of(r.q2);
  const q = o(r.q2);
  return {
    game, group: groupName(r.group, game), unit: unitOf(r.unit, game), status: r.status === 'watch' ? 'watch' : 'normal', reasons: strs(r.reasons),
    window: opt(r.window), drawsTotal: opt(r.drawsTotal), entropy: opt(r.entropy), klHistory: opt(r.klHistory), klUniform: opt(r.klUniform),
    ae: has(r.ae) ? { minZ: opt(ae.minZ), latestZ: opt(ae.latestZ), date: ostr(ae.date), alert: bool(ae.alert), window: opt(ae.window) } : null,
    q2: q2 ? { ...q2, label: str(q.label), realSd: opt(q.realSd), nullMean: opt(q.nullMean), nullSd: opt(q.nullSd) } : null,
    trend: arr(r.trend as unknown[]).map((x) => ({ end: ostr(o(x).end), entropy: opt(o(x).entropy), klHistory: opt(o(x).klHistory), klUniform: opt(o(x).klUniform) })),
    modelRows: num(r.modelRows, 0),
  };
}
function normRandomness(raw: unknown): RandomnessView {
  const r = o(raw);
  return { builtAt: ostr(r.builtAt), generatedAt: ostr(r.generatedAt), meta: randomnessMetaOf(r.meta), games: arr(r.games as unknown[]).filter(has).map((x) => randomnessLineOf(x)) };
}
function normRandomnessGame(raw: unknown, game: string, scope: Scope): RandomnessGameView {
  const r = o(raw);
  return {
    ...randomnessLineOf(r, game),
    builtAt: ostr(r.builtAt), generatedAt: ostr(r.generatedAt), meta: randomnessMetaOf(r.meta),
    scope: scopeOf(r.scope, scope), shown: num(r.shown, 0), of: num(r.of, 0), top: strs(r.top), topMissing: strs(r.topMissing),
    models: arr(r.models as unknown[]).map((x) => ({ row: str(o(x).row), kl: opt(o(x).kl), top: bool(o(x).top) })),
  };
}

// --- This week and the tuning gates ---
const DECISION_LETTERS = ['replaced', 'kept', 'reverted', 'not read'];
function usualOf(v: unknown): Usual | null {
  if (!has(v)) return null;
  const u = o(v);
  return { median: opt(u.median), min: opt(u.min), max: opt(u.max), n: num(u.n, 0), from: ostr(u.from), to: ostr(u.to), basis: str(u.basis) };
}
function runLineOf(v: unknown): RunLine {
  const r = o(v);
  const c = o(r.comparedWith);
  return {
    key: str(r.key), job: str(r.job, str(r.key)), start: ostr(r.start), end: ostr(r.end), minutes: opt(r.minutes), result: str(r.result, 'unknown'), note: ostr(r.note),
    usual: usualOf(r.usual), overUsual: typeof r.overUsual === 'boolean' ? r.overUsual : null, vsMedian: opt(r.vsMedian),
    warnings: Array.isArray(r.warnings)
      ? r.warnings.filter(has).map((x) => ({ sig: str(o(x).sig), n: num(o(x).n, 0), new: typeof o(x).new === 'boolean' ? (o(x).new as boolean) : null }))
      : null,
    noiseFiltered: opt(r.noiseFiltered),
    comparedWith: has(r.comparedWith) ? { runs: num(c.runs, 0), from: ostr(c.from), to: ostr(c.to) } : null,
  };
}
function tuningTilesOf(v: unknown): TuningTiles {
  const t = o(v);
  const h = o(t.heldFromLastWeekend);
  return {
    n: num(t.n, 0), replaced: num(t.replaced, 0), kept: num(t.kept, 0), reverted: num(t.reverted, 0), marginZero: num(t.marginZero, 0),
    replacedUnder1Se: num(t.replacedUnder1Se, 0), replacedUnder1SeVsBetterReference: num(t.replacedUnder1SeVsBetterReference, 0), under1SeTuners: strs(t.under1SeTuners),
    unattributed: num(t.unattributed, 0), notThisWeekend: num(t.notThisWeekend, 0), games: num(t.games, 0), from: ostr(t.from), to: ostr(t.to),
    heldFromLastWeekend: has(t.heldFromLastWeekend)
      ? { weekend: str(h.weekend), replaced: num(h.replaced, 0), kept: num(h.kept, 0), replacedAgain: num(h.replacedAgain, 0), reverted: num(h.reverted, 0), notDecided: num(h.notDecided, 0) }
      : null,
  };
}
const historyIn = (v: unknown) => partHistoryOf(o(v).history);
function matrixGameOf(v: unknown): MatrixGame {
  const r = o(v);
  const game = str(r.game);
  const [q0, q2, fc, rows, gates] = [o(r.q0), o(r.q2), o(r.fc), o(r.rows), o(r.gates)];
  return {
    game, group: groupName(r.group, game), unit: unitOf(r.unit, game), errors: strs(r.errors),
    q0: has(r.q0)
      ? (() => {
          const t = q0TestOf(q0.test);
          const b = o(q0.band);
          return {
            test: t, error: ostr(q0.error), confirmedRows: strs(q0.confirmedRows), followedRows: strs(q0.followedRows), inTile: q0.inTile !== false,
            ranking: q0.ranking === 'profit' ? ('profit' as const) : ('hits' as const), at: ostr(q0.at),
            band: has(q0.band) ? { at: ostr(b.at), modes: modesOf(b.modes) } : null, bandUnreadable: bool(q0.bandUnreadable), history: historyIn(r.q0),
          };
        })()
      : null,
    q2: has(r.q2)
      ? { status: q2StatusOf(q2.status), real: opt(q2.real), threshold: opt(q2.threshold), seeds: opt(q2.seeds),
          windowsTest: opt(q2.windowsTest), fewWindows: bool(q2.fewWindows), minWindows: opt(q2.minWindows), at: ostr(q2.at), history: historyIn(r.q2) }
      : null,
    fc: has(r.fc)
      ? { fits: num(fc.fits, 0), inconclusive: num(fc.inconclusive, 0), ignored: num(fc.ignored, 0), failed: num(fc.failed, 0), variants: num(fc.variants, 0), at: ostr(fc.at), history: historyIn(r.fc) }
      : null,
    rows: has(r.rows)
      ? { beat: num(rows.beat, 0), scored: num(rows.scored, 0), reference: str(rows.reference, 'GARCH Model'), window: opt(rows.window), referenceVsUniform: ostr(rows.referenceVsUniform),
          referenceVsFrequency: ostr(rows.referenceVsFrequency),
          at: ostr(rows.at), history: historyIn(r.rows) }
      : null,
    gates: has(r.gates) ? { replaced: num(gates.replaced, 0), kept: num(gates.kept, 0), reverted: num(gates.reverted, 0), notThisWeekend: num(gates.notThisWeekend, 0) } : null,
  };
}
function councilOf(v: unknown): CouncilStatus | null {
  if (!has(v)) return null;
  const c = o(v);
  const e = o(c.lastExit);
  return {
    error: ostr(c.error), name: str(c.name, 'Council API'), status: str(c.status), healthy: typeof c.healthy === 'boolean' ? c.healthy : null, summary: str(c.summary),
    since: ostr(c.since), restarts: opt(c.restarts), lastExit: has(c.lastExit) ? { code: opt(e.code), signal: ostr(e.signal), at: ostr(e.at) } : null,
    lastError: ostr(c.lastError), checkedAt: ostr(c.checkedAt),
  };
}

function normWeek(raw: unknown): WeekView {
  const r = o(raw);
  const w = o(r.weekend);
  const t = o(r.tiles);
  const we = o(t.weekend);
  const today = o(t.today);
  const tuners = o(t.tuners);
  const since = o(r.since);
  const lock = o(since.lock);
  const sign = o(since.signIns);
  const slowest = o(we.slowest);
  return {
    builtAt: ostr(r.builtAt),
    rankingAt: ostr(r.rankingAt),
    weekend: { saturday: ostr(w.saturday), from: ostr(w.from), until: ostr(w.until) },
    tiles: {
      weekend: {
        runs: num(we.runs, 0), ok: num(we.ok, 0), expected: num(we.expected, 0),
        missing: arr(we.missing as unknown[]).filter(has).map((x) => ({ key: str(o(x).key), day: ostr(o(x).day) })),
        running: has(we.running) ? { key: str(o(we.running).key), start: ostr(o(we.running).start) } : null,
        notOk: arr(we.notOk as unknown[]).filter(has).map((x) => ({ key: str(o(x).key), result: str(o(x).result), note: ostr(o(x).note) })),
        from: ostr(we.from), to: ostr(we.to), hours: opt(we.hours),
        slowest: has(we.slowest) ? { key: str(slowest.key), minutes: opt(slowest.minutes), usual: usualOf(slowest.usual) } : null,
        newWarnings: arr(we.newWarnings as unknown[]).filter(has).map((x) => ({ key: str(o(x).key), sig: str(o(x).sig), n: num(o(x).n, 0) })),
      },
      today: { day: ostr(today.day), run: has(today.run) ? runLineOf(today.run) : null, running: has(today.running) ? { start: ostr(o(today.running).start) } : null },
      tuners: has(t.tuners)
        ? {
            ...tuningTilesOf(t.tuners),
            weekend: ostr(tuners.weekend),
            byTuner: arr(tuners.byTuner as unknown[]).filter(has).map((x) => {
              const b = o(x);
              return { tuner: str(b.tuner), job: ostr(b.job), n: num(b.n, 0), replaced: num(b.replaced, 0), kept: num(b.kept, 0), reverted: num(b.reverted, 0), games: opt(b.games),
                marginZero: num(b.marginZero, 0), marginOneSe: opt(b.marginOneSe), marginSeCounts: countsOf(b.marginSeCounts), replacedUnder1Se: num(b.replacedUnder1Se, 0) };
            }),
          }
        : null,
      ...controlTilesOf(t),
    },
    runs: arr(r.runs as unknown[]).filter(has).map(runLineOf),
    matrix: { fewWindows: opt(o(r.matrix).fewWindows), games: arr(o(r.matrix).games as unknown[]).filter(has).map(matrixGameOf) },
    readings: readingsOf(r.readings),
    since: {
      from: ostr(since.from),
      predictorRuns: arr(since.predictorRuns as unknown[]).filter(has).map(runLineOf),
      queue: arr(since.queue as unknown[]).filter(has).map((x) => ({ key: str(o(x).key), name: str(o(x).name, str(o(x).key)), queuedAt: ostr(o(x).queuedAt), manual: bool(o(x).manual) })),
      running: has(since.running) ? { key: str(o(since.running).key), name: str(o(since.running).name), start: ostr(o(since.running).start) } : null,
      lock: has(since.lock) ? { held: bool(lock.held), pid: opt(lock.pid), script: ostr(lock.script), key: ostr(lock.key) } : null,
      council: councilOf(since.council),
      signIns: has(since.signIns)
        ? { failed: num(sign.failed, 0), locked: num(sign.locked, 0), ok: num(sign.ok, 0),
            last7Days: { failed: num(o(sign.last7Days).failed, 0), locked: num(o(sign.last7Days).locked, 0) }, scanned: num(sign.scanned, 0), oldestScanned: ostr(sign.oldestScanned) }
        : null,
    },
  };
}

const rowsByGame = (v: unknown): Record<string, string[]> => Object.fromEntries(Object.entries(o(v)).map(([g, x]) => [g, strs(x)]));
// {k: records} of a tuner's margin chips: counts only, a non-count dropped
const countsOf = (v: unknown): Record<string, number> =>
  Object.fromEntries(Object.entries(o(v)).filter((e): e is [string, number] => typeof e[1] === 'number' && Number.isFinite(e[1]) && e[1] > 0));
function normTuning(raw: unknown): TuningView {
  const r = o(raw);
  const h = o(r.history);
  return {
    builtAt: ostr(r.builtAt),
    weekend: ostr(r.weekend),
    tiles: tuningTilesOf(r.tiles),
    byTuner: arr(r.byTuner as unknown[]).filter(has).map((x) => {
      const b = o(x);
      const run = o(b.run);
      return {
        tuner: str(b.tuner), job: ostr(b.job), n: num(b.n, 0), replaced: num(b.replaced, 0), kept: num(b.kept, 0), reverted: num(b.reverted, 0),
        marginZero: num(b.marginZero, 0), marginOneSe: num(b.marginOneSe, 0), marginSeCounts: countsOf(b.marginSeCounts), replacedUnder1Se: num(b.replacedUnder1Se, 0),
        replacedUnder1SeVsBetterReference: num(b.replacedUnder1SeVsBetterReference, 0), games: num(b.games, 0), kinds: strs(b.kinds),
        aboveFair: num(b.aboveFair, 0), fairPerBet: rangeOf(b.fairPerBet),
        windowDays: rangeOf(b.windowDays), runTrials: rangeOf(b.runTrials), studyTrials: rangeOf(b.studyTrials),
        windowByUnit: Object.fromEntries(Object.entries(o(b.windowByUnit)).map(([u, x]) => [u, rangeOf(x)]).filter((e): e is [string, Range] => e[1] !== null)),
        run: has(b.run) ? { start: ostr(run.start), end: ostr(run.end), minutes: opt(run.minutes), result: str(run.result, 'unknown') } : null,
        replacedRows: rowsByGame(b.replacedRows), revertedRows: rowsByGame(b.revertedRows),
      };
    }),
    byGame: arr(r.byGame as unknown[]).filter(has).map((x) => {
      const g = o(x);
      const game = str(g.game);
      return {
        game, group: groupName(g.group, game), n: num(g.n, 0), replaced: num(g.replaced, 0), kept: num(g.kept, 0), reverted: num(g.reverted, 0), marginZero: num(g.marginZero, 0),
        notThisWeekend: num(g.notThisWeekend, 0), noFile: bool(g.noFile), replacedRows: strs(g.replacedRows), revertedRows: strs(g.revertedRows),
      };
    }),
    unattributed: arr(r.unattributed as unknown[]).filter(has).map((x) => ({ game: str(o(x).game), row: str(o(x).row), weekend: ostr(o(x).weekend), family: ostr(o(x).family), why: str(o(x).why) })),
    history: { weekends: strs(h.weekends), commits: num(h.commits, 0), error: ostr(h.error), source: str(h.source) },
    meta: Object.fromEntries(Object.entries(o(r.meta)).map(([k, x]) => [k, str(x)])),
  };
}
const sideOf = (v: unknown): GateSide => ({ score: opt(o(v).score), se: opt(o(v).se), days: opt(o(v).days), fairPerBet: opt(o(v).fairPerBet), profitPerBet: opt(o(v).profitPerBet) });
function normTuningGame(raw: unknown, game: string): TuningGameView {
  const r = o(raw);
  const g = str(r.game, game);
  return {
    game: g, group: groupName(r.group, g), unit: unitOf(r.unit, g), builtAt: ostr(r.builtAt), weekend: ostr(r.weekend),
    n: num(r.n, 0), replaced: num(r.replaced, 0), kept: num(r.kept, 0), reverted: num(r.reverted, 0), notThisWeekend: num(r.notThisWeekend, 0),
    weekends: strs(r.weekends), historyError: ostr(r.historyError),
    rows: arr(r.rows as unknown[]).filter(has).map((x) => {
      const w = o(x);
      return {
        row: str(w.row), tuner: str(w.tuner, 'unattributed'), by: ostr(w.by), decision: str(w.decision), thisWeekend: w.thisWeekend !== false, weekend: ostr(w.weekend), date: ostr(w.date),
        kind: ostr(w.kind), windowDays: opt(w.windowDays), runTrials: opt(w.runTrials), studyTrials: opt(w.studyTrials),
        lockboxDays: opt(w.lockboxDays), lockboxFrom: w.lockboxFrom === 'window' || w.lockboxFrom === 'table' ? w.lockboxFrom : null,
        default: sideOf(w.default), incumbent: sideOf(w.incumbent), challenger: sideOf(w.challenger),
        margin: opt(w.margin), marginSe: opt(w.marginSe), marginK: opt(w.marginK), revertMargin: opt(w.revertMargin),
        sitOut: typeof w.sitOut === 'boolean' ? w.sitOut : null, sitOutReason: ostr(w.sitOutReason),
        bar: opt(w.bar), gainSe: opt(w.gainSe), gainBestSe: opt(w.gainBestSe), gainBarSe: opt(w.gainBarSe),
        history: arr(w.history as unknown[]).map((d) => (typeof d === 'string' && DECISION_LETTERS.includes(d) ? d : null)),
        why: ostr(w.why), family: ostr(w.family),
      };
    }),
  };
}

// --- System pages (system.js): Jobs & services, People, Account, What's new ----------------------------------------
// Jobs & services and People are the administrator's (a JSON 403 for a user); Account and What's new every signed-in
// reader's own. Instants are ISO UTC (the pages print Brussels time), durations minutes to one decimal. The actions
// (postJson) send the form token GET /api/me serves.

export type ScheduleMode = 'on' | 'dry' | 'off';
// a run as the tiles and the schedule name it: its log id (the run drawer's ?run=), its start, how long, how it ended
export interface RunShort {
  id: string;
  key: string;
  start: string | null;
  minutes: number | null;
  result: string; // 'ok' | 'exit N' | 'unknown' (killed, or ended without an exit code)
}
export interface JobLine {
  key: string;
  name: string;
  description: string;
  command: string;
  log: string | null; // the log file's name
  plan: string | null; // 'weeklyTuning' | 'weeklyControls' | null (the daily predictor)
  daily: boolean;
  nextAt: string | null; // the daily job's next slot
  trigger: string | null; // a weekly job: "when Saturday's predictor finishes"
  usual: Usual | null; // its earlier successful runs, a weekly job's from its own log
  last: RunShort | null; // the newest run that ran
  canRun: boolean;
  refusal: string | null; // why Run now is refused now (the schedule is off, already running or queued)
}
export type RunKind = 'run' | 'dry' | 'blocked';
export interface RunItem {
  id: string;
  key: string;
  name: string;
  kind: RunKind; // dry: the schedule only recorded it; blocked: it waited 6 h on the lock
  start: string | null;
  end: string | null;
  minutes: number | null;
  result: string; // 'ok' | 'exit N' | 'unknown' | 'dry run' | 'waited on the lock' (it stayed queued)
  note: string | null;
  reason: string | null;
  usual: Usual | null; // the runs of this job before this one (kind run)
  overUsual: boolean | null;
  vsMedian: number | null;
  warnings: { signatures: number; new: number } | null;
  sliceGone: boolean; // a run whose log slice has rotated away: its warnings cannot be read
  sha: string | null; // the commit it ran at (git short sha)
}
export interface Running {
  id: string; // its log id, for the run drawer
  runId: string; // what Stop names
  key: string;
  name: string;
  reason: string | null;
  start: string | null;
  minutes: number | null; // elapsed at builtAt
  pid: number | null;
  sha: string | null; // the commit it runs at
  usual: Usual | null;
  vsMedian: number | null;
  overUsual: boolean | null;
  log: string[];
}
export interface QueueItem {
  key: string;
  name: string;
  reason: string | null;
  queuedAt: string | null;
  waitedMinutes: number | null;
  manual: boolean;
}
export interface ServiceLine {
  key: string;
  name: string;
  description: string;
  status: string; // 'running' | 'external' | 'restarting' | 'stopped' | 'disabled' | 'failed'
  healthy: boolean;
  summary: string | null;
  command: string | null; // what it runs, without absolute paths
  cwd: string | null; // where: relative to the checkout (src/llmCouncil)
  pid: number | null;
  port: number | null;
  since: string | null;
  upMinutes: number | null;
  restarts: number;
  lastExit: { code: number | null; signal: string | null; at: string | null } | null;
  lastError: string | null;
  checkedAt: string | null;
  actions: { start: boolean; stop: boolean; restart: boolean };
  log: { lines: string[]; healthDropped: number; file: string | null; scanned: { lines: number; kb: number; whole: boolean } | null } | null;
}
export interface JobsView {
  builtAt: string | null;
  mode: { mode: ScheduleMode; reason: string } | null;
  lock: { held: boolean; pid: number | null; script: string | null; key: string | null; ours: boolean } | null;
  running: Running | null;
  queue: QueueItem[];
  jobs: JobLine[];
  runs: { kept: number | null; shown: number; ran: number; items: RunItem[] };
  today: { day: string | null; key: string; run: RunShort | null; running: boolean } | null;
  lastFailure: (RunShort & { name: string | null; note: string | null }) | null;
  services: ServiceLine[] | null;
  servicesError: string | null;
  errors: string[];
}
export interface LogWarning {
  sig: string;
  n: number;
  new: boolean | null; // null: no earlier run of the job in the log to compare with
  line: string | null; // its newest line
}
export interface RunLog {
  id: string;
  key: string;
  job: string;
  log: string | null;
  start: string | null;
  end: string | null;
  minutes: number | null;
  result: string; // 'ok' | 'exit N' | 'unknown' | 'running'
  note: string | null;
  sha: string | null; // the commit it ran at: the kept run's, or its start marker's
  inKeptRuns: boolean;
  running: boolean;
  warnings: LogWarning[];
  noiseFiltered: number | null;
  comparedWith: { runs: number; from: string | null; to: string | null } | null;
  lines: { shown: number; of: number | null; whole: boolean; readKb: number };
  tail: string[];
  error: string | null;
}

export interface AuditEntry {
  at: string | null;
  event: string;
  label: string;
  user: string | null;
  address: string | null;
  device: string | null; // "Chrome on Windows"
  detail: string | null;
}
export interface PersonLine {
  name: string;
  role: 'admin' | 'user';
  source: string; // 'environment' | 'users.json'
  created: string | null;
  passwordChangedAt: string | null;
  signIns: number | null;
  lastSignIn: { at: string | null; address: string | null; device: string | null; from: 'activity' | 'account' } | null;
}
export interface PeopleView {
  builtAt: string | null;
  auth: { enabled: boolean; writable: boolean; minPasswordLength: number; userName: string };
  tiles: {
    users: number;
    failed: number;
    lockouts: number; // an address and name whose failures reached the sign-in limit
    refusedWhileLocked: number; // sign-ins tried while locked (login.locked)
    newDeviceOrAddress: number;
    days: number;
    from: string | null;
    to: string | null;
    scanned: { events: number; oldest: string | null; all: boolean; complete: boolean };
  };
  users: PersonLine[];
  exceptions: { shown: number; of: number; compared: string; items: (AuditEntry & { kind: string })[] };
  activity: { shown: number; of: number | null; oldest: string | null; items: AuditEntry[] };
}
export interface AccountView {
  name: string;
  role: 'admin' | 'user';
  open: boolean;
  source: string; // 'open access' | 'environment' | 'users.json'
  created: string | null;
  passwordChangedAt: string | null;
  password: { canChange: boolean; why: string | null; minLength: number };
  lastSignIn: { at: string | null; address: string | null; device: string | null } | null;
  activity: { shown: number; max: number; items: AuditEntry[] };
}
export interface Note {
  id: string;
  date: string | null;
  level: 'major' | 'minor';
  audience: 'all' | 'admin';
  title: string;
  body: string[];
  link: { href: string; label: string } | null;
  seen: boolean | null; // null: open access, no account to record it against
}
export interface WhatsNewView {
  open: boolean;
  audience: 'admin' | 'all';
  unseen: number | null;
  of: number;
  pending: { kind: 'tour' | 'news'; ids: string[] } | null;
  tour: { done: boolean | null; title: string; intro: string; steps: { title: string; body: string }[] };
  items: Note[];
}

export const MIN_PASSWORD = 8; // auth.js MIN_PASSWORD_LENGTH, when an answer does not name it

const runShortOf = (v: unknown): RunShort | null => {
  if (!has(v)) return null;
  const r = o(v);
  return { id: str(r.id), key: str(r.key), start: ostr(r.start), minutes: opt(r.minutes), result: str(r.result, 'unknown') };
};
const entryOf = (v: unknown): AuditEntry => {
  const e = o(v);
  return { at: ostr(e.at), event: str(e.event), label: str(e.label, str(e.event)), user: ostr(e.user), address: ostr(e.address), device: ostr(e.device), detail: ostr(e.detail) };
};
const signInOf = (v: unknown) => {
  if (!has(v)) return null;
  const s = o(v);
  return { at: ostr(s.at), address: ostr(s.address), device: ostr(s.device) };
};
const MODES: ScheduleMode[] = ['on', 'dry', 'off'];
const RUN_KINDS: RunKind[] = ['run', 'dry', 'blocked'];

function normJobs(raw: unknown): JobsView {
  const r = o(raw);
  const m = o(r.mode);
  const lock = o(r.lock);
  const cur = o(r.running);
  const runs = o(r.runs);
  const today = o(r.today);
  const fail = o(r.lastFailure);
  return {
    builtAt: ostr(r.builtAt),
    mode: has(r.mode) ? { mode: MODES.includes(m.mode as ScheduleMode) ? (m.mode as ScheduleMode) : 'dry', reason: str(m.reason) } : null,
    lock: has(r.lock) ? { held: bool(lock.held), pid: opt(lock.pid), script: ostr(lock.script), key: ostr(lock.key), ours: bool(lock.ours) } : null,
    running: has(r.running)
      ? {
          id: str(cur.id), runId: str(cur.runId), key: str(cur.key), name: str(cur.name, str(cur.key)), reason: ostr(cur.reason), start: ostr(cur.start),
          minutes: opt(cur.minutes), pid: opt(cur.pid), sha: ostr(cur.sha), usual: usualOf(cur.usual), vsMedian: opt(cur.vsMedian),
          overUsual: typeof cur.overUsual === 'boolean' ? cur.overUsual : null, log: strs(cur.log),
        }
      : null,
    queue: arr(r.queue as unknown[]).filter(has).map((x) => {
      const q = o(x);
      return { key: str(q.key), name: str(q.name, str(q.key)), reason: ostr(q.reason), queuedAt: ostr(q.queuedAt), waitedMinutes: opt(q.waitedMinutes), manual: bool(q.manual) };
    }),
    jobs: arr(r.jobs as unknown[]).filter(has).map((x) => {
      const j = o(x);
      return {
        key: str(j.key), name: str(j.name, str(j.key)), description: str(j.description), command: str(j.command), log: ostr(j.log), plan: ostr(j.plan), daily: bool(j.daily),
        nextAt: ostr(j.nextAt), trigger: ostr(j.trigger), usual: usualOf(j.usual), last: runShortOf(j.last), canRun: bool(j.canRun), refusal: ostr(j.refusal),
      };
    }),
    runs: {
      kept: opt(runs.kept), shown: num(runs.shown, 0), ran: num(runs.ran, 0),
      items: arr(runs.items as unknown[]).filter(has).map((x) => {
        const h = o(x);
        const w = o(h.warnings);
        const kind = RUN_KINDS.includes(h.kind as RunKind) ? (h.kind as RunKind) : 'run';
        return {
          id: str(h.id), key: str(h.key), name: str(h.name, str(h.key)), kind, start: ostr(h.start), end: ostr(h.end), minutes: opt(h.minutes),
          result: str(h.result, 'unknown'), note: ostr(h.note), reason: ostr(h.reason), usual: usualOf(h.usual),
          overUsual: typeof h.overUsual === 'boolean' ? h.overUsual : null, vsMedian: opt(h.vsMedian),
          warnings: has(h.warnings) ? { signatures: num(w.signatures, 0), new: num(w.new, 0) } : null,
          sliceGone: kind === 'run' && 'warnings' in h && h.warnings === null,
          sha: ostr(h.sha),
        };
      }),
    },
    today: has(r.today) ? { day: ostr(today.day), key: str(today.key, 'predictor'), run: runShortOf(today.run), running: bool(today.running) } : null,
    lastFailure: has(r.lastFailure) ? { ...(runShortOf(r.lastFailure) as RunShort), name: ostr(fail.name), note: ostr(fail.note) } : null,
    services: Array.isArray(r.services)
      ? r.services.filter(has).map((x) => {
          const s = o(x);
          const e = o(s.lastExit);
          const a = o(s.actions);
          const l = o(s.log);
          const sc = o(l.scanned);
          return {
            key: str(s.key), name: str(s.name, str(s.key)), description: str(s.description), status: str(s.status, 'unknown'), healthy: bool(s.healthy), summary: ostr(s.summary),
            command: ostr(s.command), cwd: ostr(s.cwd),
            pid: opt(s.pid), port: opt(s.port), since: ostr(s.since), upMinutes: opt(s.upMinutes), restarts: num(s.restarts, 0),
            lastExit: has(s.lastExit) ? { code: opt(e.code), signal: ostr(e.signal), at: ostr(e.at) } : null,
            lastError: ostr(s.lastError), checkedAt: ostr(s.checkedAt),
            actions: { start: bool(a.start), stop: bool(a.stop), restart: bool(a.restart) },
            log: has(s.log)
              ? { lines: strs(l.lines), healthDropped: num(l.healthDropped, 0), file: ostr(l.file), scanned: has(l.scanned) ? { lines: num(sc.lines, 0), kb: num(sc.kb, 0), whole: bool(sc.whole) } : null }
              : null,
          };
        })
      : null,
    servicesError: ostr(r.servicesError),
    errors: strs(r.errors),
  };
}

function normRunLog(raw: unknown, id: string): RunLog {
  const r = o(raw);
  const l = o(r.lines);
  const c = o(r.comparedWith);
  return {
    id: str(r.id, id), key: str(r.key), job: str(r.job, str(r.key)), log: ostr(r.log), start: ostr(r.start), end: ostr(r.end), minutes: opt(r.minutes),
    result: str(r.result, 'unknown'), note: ostr(r.note), sha: ostr(r.sha), inKeptRuns: bool(r.inKeptRuns), running: bool(r.running),
    warnings: arr(r.warnings as unknown[]).filter(has).map((x) => {
      const w = o(x);
      return { sig: str(w.sig), n: num(w.n, 0), new: typeof w.new === 'boolean' ? w.new : null, line: ostr(w.line) };
    }),
    noiseFiltered: opt(r.noiseFiltered),
    comparedWith: has(r.comparedWith) ? { runs: num(c.runs, 0), from: ostr(c.from), to: ostr(c.to) } : null,
    lines: { shown: num(l.shown, 0), of: opt(l.of), whole: bool(l.whole), readKb: num(l.readKb, 0) },
    tail: strs(r.tail),
    error: ostr(r.error),
  };
}

function normPeople(raw: unknown): PeopleView {
  const r = o(raw);
  const a = o(r.auth);
  const t = o(r.tiles);
  const sc = o(t.scanned);
  const ex = o(r.exceptions);
  const ac = o(r.activity);
  return {
    builtAt: ostr(r.builtAt),
    auth: { enabled: bool(a.enabled), writable: bool(a.writable), minPasswordLength: num(a.minPasswordLength, MIN_PASSWORD), userName: str(a.userName, '3-32 letters, digits, dots, underscores or dashes') },
    tiles: {
      users: num(t.users, 0), failed: num(t.failed, 0), lockouts: num(t.lockouts, 0), refusedWhileLocked: num(t.refusedWhileLocked, 0),
      newDeviceOrAddress: num(t.newDeviceOrAddress, 0), days: num(t.days, 7),
      from: ostr(t.from), to: ostr(t.to), scanned: { events: num(sc.events, 0), oldest: ostr(sc.oldest), all: bool(sc.all), complete: bool(sc.complete) },
    },
    users: arr(r.users as unknown[]).filter(has).map((x) => {
      const u = o(x);
      const s = o(u.lastSignIn);
      return {
        name: str(u.name), role: u.role === 'admin' ? 'admin' : 'user', source: str(u.source), created: ostr(u.created), passwordChangedAt: ostr(u.passwordChangedAt),
        signIns: opt(u.signIns), lastSignIn: has(u.lastSignIn) ? { ...(signInOf(u.lastSignIn) as { at: string | null; address: string | null; device: string | null }), from: s.from === 'account' ? 'account' : 'activity' } : null,
      };
    }),
    exceptions: { shown: num(ex.shown, 0), of: num(ex.of, 0), compared: str(ex.compared), items: arr(ex.items as unknown[]).filter(has).map((x) => ({ ...entryOf(x), kind: str(o(x).kind) })) },
    activity: { shown: num(ac.shown, 0), of: opt(ac.of), oldest: ostr(ac.oldest), items: arr(ac.items as unknown[]).filter(has).map(entryOf) },
  };
}

function normAccount(raw: unknown): AccountView {
  const r = o(raw);
  const p = o(r.password);
  const ac = o(r.activity);
  return {
    name: str(r.name), role: r.role === 'admin' ? 'admin' : 'user', open: bool(r.open), source: str(r.source), created: ostr(r.created), passwordChangedAt: ostr(r.passwordChangedAt),
    password: { canChange: bool(p.canChange), why: ostr(p.why), minLength: num(p.minLength, MIN_PASSWORD) },
    lastSignIn: signInOf(r.lastSignIn),
    activity: { shown: num(ac.shown, 0), max: num(ac.max, 15), items: arr(ac.items as unknown[]).filter(has).map(entryOf) },
  };
}

function normWhatsNew(raw: unknown): WhatsNewView {
  const r = o(raw);
  const t = o(r.tour);
  const p = o(r.pending);
  return {
    open: bool(r.open), audience: r.audience === 'admin' ? 'admin' : 'all', unseen: opt(r.unseen), of: num(r.of, 0),
    pending: has(r.pending) ? { kind: p.kind === 'tour' ? 'tour' : 'news', ids: strs(p.ids) } : null,
    tour: {
      done: typeof t.done === 'boolean' ? t.done : null, title: str(t.title, 'Welcome'), intro: str(t.intro),
      steps: arr(t.steps as unknown[]).filter(has).map((x) => ({ title: str(o(x).title), body: str(o(x).body) })),
    },
    items: arr(r.items as unknown[]).filter(has).map((x) => {
      const n = o(x);
      const l = o(n.link);
      return {
        id: str(n.id), date: ostr(n.date), level: n.level === 'major' ? 'major' : 'minor', audience: n.audience === 'admin' ? 'admin' : 'all', title: str(n.title), body: strs(n.body),
        link: has(n.link) && typeof l.href === 'string' ? { href: l.href, label: str(l.label, l.href) } : null, seen: typeof n.seen === 'boolean' ? n.seen : null,
      };
    }),
  };
}

export const getAdminJobs = async (): Promise<JobsView> => normJobs(await getJson<unknown>('/api/admin/jobs'));
export const getAdminRunLog = async (id: string): Promise<RunLog> => normRunLog(await getJson<unknown>(`/api/admin/jobs/run/${encodeURIComponent(id)}/log`), id);
export const getAdminPeople = async (all: boolean): Promise<PeopleView> => normPeople(await getJson<unknown>(`/api/admin/people${all ? '?limit=all' : ''}`));
export const getAccount = async (): Promise<AccountView> => normAccount(await getJson<unknown>('/api/account'));
export const getWhatsNew = async (): Promise<WhatsNewView> => normWhatsNew(await getJson<unknown>('/api/whats-new'));

// The actions' addresses (system.js), named once.
export const ACTIONS = {
  run: '/api/admin/jobs/run', // {key}
  cancel: '/api/admin/jobs/cancel', // {key}
  stop: '/api/admin/jobs/stop', // {runId}
  service: (name: string, action: 'start' | 'stop' | 'restart') => `/api/admin/services/${encodeURIComponent(name)}/${action}`,
  addUser: '/api/admin/people/add', // {username, password}
  setPassword: '/api/admin/people/password', // {username, password}
  deleteUser: '/api/admin/people/delete', // {username}
  ownPassword: '/api/account/password', // {current, password, confirm} -> csrf: the new session's token
  seen: '/api/whats-new/seen', // {ids?, all?, tour?}
  replay: '/api/whats-new/replay',
};
