// The Workbench's API (admin batch 4, workbench.js): every past draw of a game, one draw, and the market workbench.
// Typed and normalised like api.ts: every container is filled and every figure is a number or null, so the
// components map over them without checks. The users' Lottery (next batch) reads the same shapes from its own
// routes, with only the rows in its scope sent: the normalisers are exported for its getters.
import { MARKET_GAMES, getJson, groupOf, isScope, type GroupName, type Scope } from './api';

type Raw = Record<string, unknown>;
const o = (v: unknown): Raw => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Raw) : {});
const has = (v: unknown): boolean => Boolean(v) && typeof v === 'object' && !Array.isArray(v);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const opt = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const num = (v: unknown, fallback: number): number => opt(v) ?? fallback;
const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const ostr = (v: unknown): string | null => (typeof v === 'string' && v ? v : null);
const bool = (v: unknown): boolean => v === true;
const nums = (v: unknown): number[] => arr(v).filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
const optNums = (v: unknown): (number | null)[] => arr(v).map(opt);
const strs = (v: unknown): string[] => arr(v).filter((x): x is string => typeof x === 'string');
const pairs = (v: unknown): [number, number][] =>
  arr(v)
    .filter((p): p is [number, number] => Array.isArray(p) && opt(p[0]) !== null && opt(p[1]) !== null)
    .map((p) => [p[0], p[1]]);

// --- the shapes of a game ------------------------------------------------------------------------------------------
export type Family = 'lotto' | 'specials' | 'jokerplus' | 'keno' | 'pick3' | 'market';
const FAMILIES: Family[] = ['lotto', 'specials', 'jokerplus', 'keno', 'pick3', 'market'];
export interface TicketInfo {
  currency: string;
  price: number | null;
  sizes: [number, number] | null; // keno: the played sizes, 5 to 10
  ranking: string | null; // keno: what the 20-number row is
  triple: number | null; // pick3: the price of a triple
  bets: string[]; // pick3: the bet types a ticket plays
  note: string | null;
}
export interface Shape {
  family: Family;
  specials: string[]; // ['Star 1','Star 2'] | ['Dream'] | ['Viking'] | ['Sign'] | []
  bonus: boolean; // lotto: the 7th drawn number supplements
  positional: boolean; // pick3, jokerplus, markets: scored by slot
  money: boolean; // keno, pick3, jokerplus
  slots: string[]; // markets: the instrument per slot
  ticket: TicketInfo | null;
}
export function shapeOf(v: unknown, game: string): Shape {
  const s = o(v);
  const fam = FAMILIES.includes(s.family as Family) ? (s.family as Family) : MARKET_GAMES.includes(game) ? 'market' : 'lotto';
  const t = o(s.ticket);
  const sizes = nums(t.sizes);
  return {
    family: fam, specials: strs(s.specials), bonus: bool(s.bonus), positional: bool(s.positional), money: bool(s.money), slots: strs(s.slots),
    ticket: has(s.ticket)
      ? { currency: str(t.currency, 'EUR'), price: opt(t.price), sizes: sizes.length === 2 ? [sizes[0], sizes[1]] : null, ranking: ostr(t.ranking),
          triple: opt(t.triple), bets: strs(t.bets), note: ostr(t.note) }
      : null,
  };
}

export interface DrawResult {
  mains: number[];
  specials: number[];
  bonus: number[];
  signName: string | null; // jokerplus: the sign by name
}
export function resultOf(v: unknown): DrawResult | null {
  if (!has(v)) return null;
  const r = o(v);
  return { mains: nums(r.mains), specials: nums(r.specials), bonus: nums(r.bonus), signName: ostr(r.signName) };
}

// The best ticket of a draw or a month, over the rows in scope: lotto and the special games N (M), Joker+ L/R (Z), the
// markets N slots; keno and pick3 have none (their figure is the profit).
export interface Best {
  mains: number | null;
  specials: number | null;
  left: number | null;
  right: number | null;
  sign: number | null;
  by: string;
}
export function bestOf(v: unknown): Best | null {
  if (!has(v)) return null;
  const b = o(v);
  return { mains: opt(b.mains), specials: opt(b.specials), left: opt(b.left), right: opt(b.right), sign: opt(b.sign), by: str(b.by) };
}

// A version of the History ranking: when it was saved (committed) and generated, by which metric, from how many draws
export interface RankVersion {
  committedAt: string | null;
  generatedAt: string | null;
  metric: string;
  metricText: string;
  minDraws: number | null;
}
export function versionOf(v: unknown, metrics: Record<string, string> = {}): RankVersion | null {
  if (!has(v)) return null;
  const r = o(v);
  const metric = str(r.metric);
  // metricKey: a version ranked on the live record names its words apart ('avg_hits:live'), so older versions keep theirs
  const key = str(r.metricKey, metric);
  return { committedAt: ostr(r.committedAt), generatedAt: ostr(r.generatedAt), metric, metricText: str(r.metricText, metrics[key] ?? metric.replace(/_/g, ' ')), minDraws: opt(r.minDraws) };
}

// --- GET /api/admin/history/:game ------------------------------------------------------------------------------------
export interface DayFigures {
  rows: number;
  tickets: number;
  profit: number | null; // money games, net, EUR
  stake: number | null;
  best: Best | null;
  multi: number | null; // lotto: the best hits among a row's nine numbers, when a row had extras
}
export function figuresOf(v: unknown): DayFigures | null {
  if (!has(v)) return null;
  const f = o(v);
  if (opt(f.rows) === null) return null;
  return { rows: num(f.rows, 0), tickets: num(f.tickets, 0), profit: opt(f.profit), stake: opt(f.stake), best: bestOf(f.best), multi: opt(f.multi) };
}
export interface DayLine {
  date: string;
  weekday: string;
  result: DrawResult | null;
  figures: DayFigures | null; // null: no rows in scope (Top before a ranking existed)
  version: number | null; // index into History.ranking.versions
  none: string | null; // why there is no Top: 'before-first' | 'not-ranked' | 'no-ranked-row' | 'unavailable'
  // the admin: the rows in scope not live as stored, by status (late, recovered, duplicate, unchecked; 27 Oct 2026)
  liveMarks: Record<string, number> | null;
  // the users: a ticket of the five was made after the draw's sales close - shown greyed, out of the totals
  afterClose: boolean;
}
export interface Month {
  month: string; // YYYY-MM
  draws: number;
  counted: number; // draws with rows in scope (users: and not greyed); the totals are over these only
  afterClose: number; // the users: draws shown greyed, their tickets made after the sales close, out of the totals
  rows: number; // summed over the counted draws (row-draws), not a count of rows
  rowsMin: number | null; // the rows of one counted draw, fewest and most
  rowsMax: number | null;
  tickets: number;
  profit: number | null;
  stake: number | null;
  best: Best | null;
  multi: number | null;
  days: DayLine[] | null; // null: not the month this answer opened
}
export interface History {
  game: string;
  group: GroupName;
  unit: string;
  builtAt: string | null;
  scope: Scope;
  topN: number;
  shape: Shape;
  draws: number;
  counted: number;
  month: string;
  months: Month[];
  ranking: {
    first: { committedAt: string | null; generatedAt: string | null } | null; // the first saved ranking of any game
    firstOfGame: { committedAt: string | null; generatedAt: string | null } | null; // the first that ranks this game
    error: string | null; // a version could not be read: every draw (no first) or the draws it decides have no Top
    cutoff: { hour: number; zone: string; weekGames: string[]; text: string };
    versions: RankVersion[];
    metrics: Record<string, string>;
  };
  // the draws' tickets judged against their sales close (lotteryLiveness.json); null for the markets and before 27 Oct 2026.
  // afterClose: the users' line for a greyed draw
  liveness: { checked: string | null; afterClose: string | null } | null;
}
const countsOf = (v: unknown): Record<string, number> | null => {
  const out = Object.fromEntries(Object.entries(o(v)).filter(([, n]) => opt(n) !== null).map(([k, n]) => [k, n as number]));
  return Object.keys(out).length ? out : null;
};
function dayLineOf(v: unknown): DayLine {
  const d = o(v);
  const r = o(d.ranking);
  return {
    date: str(d.date), weekday: str(d.weekday), result: resultOf(d.result), figures: figuresOf(d),
    version: opt(r.v), none: ostr(r.none), liveMarks: countsOf(d.liveMarks), afterClose: bool(d.afterClose),
  };
}
export function normHistory(raw: unknown, game: string, scope: Scope): History {
  const r = o(raw);
  const g = str(r.game, game);
  const rk = o(r.ranking);
  const metrics = Object.fromEntries(Object.entries(o(rk.metrics)).map(([k, x]) => [k, str(x, k)]));
  const cut = o(rk.cutoff);
  return {
    game: g, group: r.group === 'Lottery' || r.group === 'Markets' ? r.group : groupOf(g), unit: str(r.unit, MARKET_GAMES.includes(g) ? 'days' : 'draws'),
    builtAt: ostr(r.builtAt), scope: isScope(r.scope) ? r.scope : scope, topN: num(r.topN, 5), shape: shapeOf(r.shape, g),
    draws: num(r.draws, 0), counted: num(r.counted, 0), month: str(r.month),
    months: arr(r.months).filter(has).map((x) => {
      const m = o(x);
      return {
        month: str(m.month), draws: num(m.draws, 0), counted: num(m.counted, 0), afterClose: num(m.afterClose, 0), rows: num(m.rows, 0), rowsMin: opt(m.rowsMin), rowsMax: opt(m.rowsMax), tickets: num(m.tickets, 0),
        profit: opt(m.profit), stake: opt(m.stake), best: bestOf(m.best), multi: opt(m.multi),
        days: Array.isArray(m.days) ? m.days.map(dayLineOf) : null,
      };
    }),
    ranking: {
      first: has(rk.first) ? { committedAt: ostr(o(rk.first).committedAt), generatedAt: ostr(o(rk.first).generatedAt) } : null,
      firstOfGame: has(rk.firstOfGame) ? { committedAt: ostr(o(rk.firstOfGame).committedAt), generatedAt: ostr(o(rk.firstOfGame).generatedAt) } : null,
      error: ostr(rk.error),
      cutoff: { hour: num(cut.hour, 15), zone: str(cut.zone, 'Europe/Brussels'), weekGames: strs(cut.weekGames), text: str(cut.text) },
      versions: arr(rk.versions).map((x) => versionOf(x, metrics)).filter((x): x is RankVersion => x !== null),
      metrics,
    },
    liveness: has(r.liveness) ? { checked: ostr(o(r.liveness).checked), afterClose: ostr(o(r.liveness).afterClose) } : null,
  };
}

// --- GET /api/admin/draw/:game/:date ---------------------------------------------------------------------------------
export interface Ticket {
  mains: number[];
  specials: number[] | null; // the special-column games only; [] = a mains-only row
  signName: string | null;
  kind: 'ticket' | 'ranking' | null; // keno: a 5-10 ticket, or the row's 20-number ranking (not played)
  mark: string; // one letter per main cell ('' on the coming draw): m hit, b lotto bonus, l / r Joker+ runs, - none
  markSpecials: string; // s a special hit, z Joker+'s sign
  markMember: string; // pick3: the old page's membership marks
  hits: { mains: number | null; specials: number | null; left: number | null; right: number | null; sign: number | null; positional: number | null; membership: number | null } | null;
  stake: number | null;
  profit: number | null;
  bets: { straight: number; consolation: number; box: number; frontPair: number; backPair: number } | null;
}
// What the live record makes of a stored row of a draw (src/Liveness.py, lotteryLiveness.json, 27 Oct 2026; the admin
// only): late - back-built, or the row added after the close, with when it was first saved; recovered - rewritten after
// the draw, the live record scoring the ticket that stood at the close (atClose, scored against this draw); duplicate -
// a second row of one name on the draw; unchecked - git could not be read and no earlier check covers it.
export interface LiveMark {
  status: 'late' | 'recovered' | 'duplicate' | 'unchecked';
  reason: string | null;
  close: string | null; // the draw's sales close
  firstSaved: string | null; // late: the draw's earliest late row's first save (one per draw in the file)
  delayHours: number | null;
  lateRows: number | null; // late: the late rows the file names on the draw (null: every row) - over 1, firstSaved is the earliest's
  savedAt: string | null; // recovered: when the ticket that stood at the close was committed
  commit: string | null;
  atClose: Row | null;
}
export interface Row {
  name: string;
  rank: number | null; // place in the ranking the draw is read against
  top: boolean; // in that draw's point-in-time five (the coming draw: today's)
  tickets: Ticket[];
  multiPick: { numbers: number[]; mark: string; nine: number | null } | null; // lotto: the 7th-9th numbers
  liveMark: LiveMark | null; // null: live as stored (or not judged)
}
const LIVE_MARKS: LiveMark['status'][] = ['late', 'recovered', 'duplicate', 'unchecked'];
function liveMarkOf(v: unknown): LiveMark | null {
  if (!has(v)) return null;
  const m = o(v);
  if (!LIVE_MARKS.includes(m.status as LiveMark['status'])) return null;
  return { status: m.status as LiveMark['status'], reason: ostr(m.reason), close: ostr(m.close), firstSaved: ostr(m.firstSaved), delayHours: opt(m.delayHours),
    lateRows: opt(m.lateRows), savedAt: ostr(m.savedAt), commit: ostr(m.commit), atClose: has(m.atClose) ? rowOf(m.atClose) : null };
}
function ticketOf(v: unknown): Ticket {
  const t = o(v);
  const h = o(t.hits);
  const b = o(t.bets);
  return {
    mains: nums(t.mains), specials: Array.isArray(t.specials) ? nums(t.specials) : null, signName: ostr(t.signName),
    kind: t.kind === 'ticket' || t.kind === 'ranking' ? t.kind : null,
    mark: str(t.mark), markSpecials: str(t.markSpecials), markMember: str(t.markMember),
    hits: has(t.hits)
      ? { mains: opt(h.mains), specials: opt(h.specials), left: opt(h.left), right: opt(h.right), sign: opt(h.sign), positional: opt(h.positional), membership: opt(h.membership) }
      : null,
    stake: opt(t.stake), profit: opt(t.profit),
    bets: has(t.bets) ? { straight: num(b.straight, 0), consolation: num(b.consolation, 0), box: num(b.box, 0), frontPair: num(b.frontPair, 0), backPair: num(b.backPair, 0) } : null,
  };
}
export function rowOf(v: unknown): Row {
  const r = o(v);
  const mp = o(r.multiPick);
  return {
    name: str(r.name), rank: opt(r.rank), top: bool(r.top), tickets: arr(r.tickets).filter(has).map(ticketOf),
    multiPick: has(r.multiPick) && nums(mp.numbers).length ? { numbers: nums(mp.numbers), mark: str(mp.mark), nine: opt(mp.nine) } : null,
    liveMark: liveMarkOf(r.liveMark),
  };
}

export interface Votes {
  rows: number; // the shown rows the counts are over
  kind: 'numbers' | 'slots';
  counts: [number, number][]; // numbers: value -> how many shown rows' first ticket carry it
  drawn: number[];
  slots: { drawn: number | null; counts: [number, number][] }[];
  stored: { from: 'file' | 'previous file'; values: [number, number][] } | null; // the file's weighted vote of every row (All only)
}
function votesOf(v: unknown): Votes {
  const r = o(v);
  const s = o(r.stored);
  return {
    rows: num(r.rows, 0), kind: r.kind === 'slots' ? 'slots' : 'numbers', counts: pairs(r.counts), drawn: nums(r.drawn),
    slots: arr(r.slots).filter(has).map((x) => ({ drawn: opt(o(x).drawn), counts: pairs(o(x).counts) })),
    stored: has(r.stored) ? { from: s.from === 'previous file' ? 'previous file' : 'file', values: pairs(s.values) } : null,
  };
}

export interface MarketDayModel {
  exact: number | null;
  adjacent: number | null;
  direction: number | null;
  positions: number | null;
  trades: number | null;
  pnl: number | null; // price units
  pnlCash: number | null; // money
  sizes: (number | null)[] | null; // the RL Position Model's size per slot
}
export interface MarketDay {
  k: number;
  chance: { exact: number | null; adjacent: number | null; direction: number | null };
  best: string | null;
  exactMean: number | null; // over every row of the day: All only
  instruments: { symbol: string; ret: number | null; bin: number | null; edges: number[]; before: number | null }[]; // before: the previous close, a bin's prices
  models: Record<string, MarketDayModel>;
}
// A bin's move interval under a day's edges: [low, high], null at an open end (markets.js binInterval); a bin is that
// interval of prices from the close before, close x e^edge (the moves are log returns)
export function binInterval(bin: number, edges: number[]): [number | null, number | null] {
  return [bin > 0 ? (edges[bin - 1] ?? null) : null, bin < edges.length ? (edges[bin] ?? null) : null];
}
function chanceOf(v: unknown) {
  const c = o(v);
  return { exact: opt(c.exact), adjacent: opt(c.adjacent), direction: opt(c.direction) };
}
function marketDayOf(v: unknown): MarketDay | null {
  if (!has(v)) return null;
  const m = o(v);
  return {
    k: num(m.k, 10), chance: chanceOf(m.chance), best: ostr(m.best), exactMean: opt(m.exactMean),
    instruments: arr(m.instruments).filter(has).map((x) => ({ symbol: str(o(x).symbol), ret: opt(o(x).ret), bin: opt(o(x).bin), edges: nums(o(x).edges), before: opt(o(x).before) })),
    models: Object.fromEntries(
      Object.entries(o(m.models)).map(([name, x]) => {
        const w = o(x);
        return [name, { exact: opt(w.exact), adjacent: opt(w.adjacent), direction: opt(w.direction), positions: opt(w.positions), trades: opt(w.trades),
          pnl: opt(w.pnl), pnlCash: opt(w.pnlCash), sizes: Array.isArray(w.sizes) ? optNums(w.sizes) : null }];
      }),
    ),
  };
}

export interface DrawScope {
  scope: Scope;
  rows: number; // rows shown
  of: number; // of the table's rows
  tickets: number;
  top: string[]; // the point-in-time five ([] when none)
  cutoff: string | null; // the instant the ranking was read at
  ranking: RankVersion | null;
  none: { code: string; text: string } | null;
}
// A lottery draw's sales close (src/drawClose.json, 27 Oct 2026): Brussels time, the instant, whether its weekday is in
// the schedule (not: the game's earliest close stands in), and whether the coming tickets were made at or after it
export interface SalesClose {
  time: string;
  at: string | null;
  scheduled: boolean;
  afterClose: boolean;
}
export function salesCloseOf(v: unknown): SalesClose | null {
  if (!has(v)) return null;
  const c = o(v);
  return { time: str(c.time), at: ostr(c.at), scheduled: c.scheduled !== false, afterClose: bool(c.afterClose) };
}
export interface Coming {
  date: string | null; // the next draw on the game's own schedule
  madeAt: string | null;
  close: SalesClose | null; // a lottery draw's sales close; null for the markets
  scope: { scope: Scope; rows: number; of: number; top: string[]; ranking: (RankVersion & { today: true }) | null; none: { code: string; text: string } | null };
  rows: Row[];
  votes: Votes;
}
export interface MultiPickLegendData {
  extra: number;
  gridPrice: number | null;
  perExtra: number | null;
  minHits: number | null;
  sizes: { size: number; grids: number | null; price: number | null; chance: number | null }[];
}
export interface Draw {
  game: string;
  group: GroupName;
  unit: string;
  builtAt: string | null;
  date: string;
  weekday: string;
  file: string;
  prev: string | null;
  next: string | null;
  newest: boolean;
  shape: Shape;
  result: DrawResult | null;
  scope: DrawScope;
  rows: Row[];
  figures: DayFigures | null;
  multiPick: MultiPickLegendData | null;
  votes: Votes;
  market: MarketDay | null;
  coming: Coming | null;
  scoredIn: string | null;
  // the admin: the shown rows not live as stored, by status, and whether the users' rule greys the draw
  liveness: { checked: string | null; marks: Record<string, number> | null; afterClose: boolean; afterCloseText: string | null } | null;
  // the users: a ticket of the five was made after the draw's sales close - shown, out of the totals
  afterClose: boolean;
  afterCloseText: string | null;
}
export const noneOf = (v: unknown) => (has(v) ? { code: str(o(v).code), text: str(o(v).text) } : null);
// the pieces of a draw the users' Lottery (lottery.ts) also reads on its own: whose rows a past draw's table shows, whose
// the coming draw's, and lotto's multi-pick legend
export function drawScopeOf(v: unknown, scope: Scope): DrawScope {
  const s = o(v);
  return {
    scope: isScope(s.scope) ? s.scope : scope, rows: num(s.rows, 0), of: num(s.of, 0), tickets: num(s.tickets, 0), top: strs(s.top),
    cutoff: ostr(s.cutoff), ranking: versionOf(s.ranking), none: noneOf(s.none),
  };
}
export function comingScopeOf(v: unknown, scope: Scope): Coming['scope'] {
  const cs = o(v);
  const todayRanking = versionOf(cs.ranking);
  return { scope: isScope(cs.scope) ? cs.scope : scope, rows: num(cs.rows, 0), of: num(cs.of, 0), top: strs(cs.top),
    ranking: todayRanking ? { ...todayRanking, today: true } : null, none: noneOf(cs.none) };
}
export function multiPickLegendOf(v: unknown): MultiPickLegendData | null {
  if (!has(v)) return null;
  const mp = o(v);
  return { extra: num(mp.extra, 3), gridPrice: opt(mp.gridPrice), perExtra: opt(mp.perExtra), minHits: opt(mp.minHits),
    sizes: arr(mp.sizes).filter(has).map((x) => ({ size: num(o(x).size, 0), grids: opt(o(x).grids), price: opt(o(x).price), chance: opt(o(x).chance) })) };
}
export function normDraw(raw: unknown, game: string, date: string, scope: Scope): Draw {
  const r = o(raw);
  const g = str(r.game, game);
  const c = o(r.coming);
  return {
    game: g, group: r.group === 'Lottery' || r.group === 'Markets' ? r.group : groupOf(g), unit: str(r.unit, MARKET_GAMES.includes(g) ? 'days' : 'draws'),
    builtAt: ostr(r.builtAt), date: str(r.date, date), weekday: str(r.weekday), file: str(r.file),
    prev: ostr(r.prev), next: ostr(r.next), newest: bool(r.newest), shape: shapeOf(r.shape, g), result: resultOf(r.result),
    scope: drawScopeOf(r.scope, scope),
    rows: arr(r.rows).filter(has).map(rowOf),
    figures: figuresOf(r.figures),
    multiPick: multiPickLegendOf(r.multiPick),
    votes: votesOf(r.votes),
    market: marketDayOf(r.market),
    coming: has(r.coming)
      ? {
          date: ostr(c.date), madeAt: ostr(c.madeAt), close: salesCloseOf(c.close),
          scope: comingScopeOf(c.scope, scope),
          rows: arr(c.rows).filter(has).map(rowOf),
          votes: votesOf(c.votes),
        }
      : null,
    scoredIn: ostr(r.scoredIn),
    liveness: has(r.liveness)
      ? { checked: ostr(o(r.liveness).checked), marks: countsOf(o(r.liveness).marks), afterClose: bool(o(r.liveness).afterClose), afterCloseText: ostr(o(r.liveness).afterCloseText) }
      : null,
    afterClose: bool(r.afterClose) || (has(r.liveness) && bool(o(r.liveness).afterClose)),
    afterCloseText: ostr(r.afterCloseText) ?? (has(r.liveness) ? ostr(o(r.liveness).afterCloseText) : null),
  };
}

// --- GET /api/admin/market/:market/workbench ---------------------------------------------------------------------------
export interface Call {
  bin: number | null;
  size: number | null; // the RL Position Model's size
  up: boolean;
  word: 'up' | 'flat' | 'down';
  price: number | null;
  band: { low: number | null; high: number | null };
  move: { low: number | null; high: number | null };
}
function callOf(v: unknown): Call | null {
  if (!has(v)) return null;
  const c = o(v);
  const word = c.word === 'up' || c.word === 'down' ? c.word : 'flat';
  return { bin: opt(c.bin), size: opt(c.size), up: bool(c.up), word, price: opt(c.price),
    band: { low: opt(o(c.band).low), high: opt(o(c.band).high) }, move: { low: opt(o(c.move).low), high: opt(o(c.move).high) } };
}
export interface BookLine {
  total: number | null;
  positions: number | null;
  stake: number | null;
  settled: number | null; // the picked model's own settled days (plan) or weeks (week card)
  firstDay: string | null; // its first and last (a week by its end)
  lastDay: string | null;
  market: number | null; // buying everything over the same days
  ahead: number | null;
  exact: number | null; // picked by exact rate instead of a book
}
export interface RecordMix {
  firstLive: string | null;
  backBuilt: number;
  live: number;
  late: number;
  lastLive: string | null;
}
export interface SoFar {
  liveDays: number;
  positions: number | null;
  firstLive: string | null;
  lastLive: string | null;
  total: number | null;
  market: number | null;
  ahead: number | null;
}
export interface Plan {
  day: string;
  calendarKnown: boolean;
  madeOn: string | null;
  madeAt: string | null;
  openAt: string | null;
  closeAt: string | null;
  closed: boolean;
  opened: boolean;
  status: string;
  model: string;
  byBook: boolean;
  rule: string;
  minTrades: number;
  models: number;
  book: BookLine;
  record: RecordMix | null;
  soFar: SoFar | null;
  rows: { symbol: string; base: number | null; call: Call | null; order: string; up: number; of: number }[];
}
export interface WeekCard {
  weekEnd: string;
  firstDay: string | null;
  madeOn: string | null;
  madeAt: string | null;
  openAt: string | null;
  closeAt: string | null;
  started: boolean;
  closed: boolean;
  hoursLeft: number | null;
  status: string;
  settledWeeks: number | null;
  model: string;
  byBook: boolean;
  rule: string;
  minTrades: number;
  models: number;
  book: BookLine;
  guard: { from: string | null; applies: boolean };
  record: RecordMix | null;
  soFar: SoFar | null;
  rows: {
    symbol: string;
    base: number | null;
    call: Call | null;
    soFar: { closes: number; last: number | null; lastDate: string | null; move: number | null; where: string } | null;
    order: string;
    up: number;
    of: number;
  }[];
}
export type FollowedPoint = [string, string | null, 'book' | 'exact' | null, number, number | null, number | null]; // date, model, pickedBy, positions, pnl, market
export interface Followed {
  rule: string;
  week: boolean;
  minTrades: number;
  firstLive: string | null;
  liveDays: number;
  positions: number | null;
  total: number | null;
  market: number | null;
  ahead: number | null;
  live: FollowedPoint[]; // oldest first
  late: { n: number; total: number | null; market: number | null; points: FollowedPoint[] };
  backBuilt: { n: number; total: number | null; market: number | null; points: FollowedPoint[] };
}
export type BookRule = 'daily' | 'hold' | 'open' | 'daily_ls' | 'hold_ls' | 'open_ls';
const BOOK_RULES: BookRule[] = ['daily', 'hold', 'open', 'daily_ls', 'hold_ls', 'open_ls'];
export interface Books {
  days: number;
  first: string | null;
  last: string | null;
  stake: number | null;
  feePerLeg: number | null;
  currency: string;
  minDays: number;
  seasonedRows: number | null;
  rules: BookRule[];
  ruleText: { daily: string; hold: string; open: string | null; short: string };
  headline: {
    daily: { best: { model: string; total: number | null } | null; market: number | null } | null;
    hold: { best: { model: string; total: number | null } | null; market: number | null } | null;
    open: { best: { model: string; total: number | null } | null; market: number | null } | null;
    longShort: { best: { model: string; total: number | null } | null; bought: number | null; sold: number | null } | null;
  };
  best: string | null;
}
export interface BookFigures {
  trades: number | null;
  winRate: number | null;
  total: number | null;
  perTrade: number | null;
}
export interface ModelRow {
  name: string;
  days: number | null;
  firstDay: string | null;
  lastDay: string | null;
  sized: boolean;
  exact: number | null;
  adjacent: number | null;
  direction: number | null;
  aboveExact: boolean;
  aboveAdjacent: boolean;
  aboveDirection: boolean;
  daily: BookFigures | null;
  hold: BookFigures | null;
  open: BookFigures | null;
  longShort: (BookFigures & { hold: number | null; open: number | null }) | null;
}
export interface GuardWeek {
  date: string;
  model: string | null;
  positions: number | null;
  guard: number | null;
  plain: number | null;
  market: number | null;
  userPages: string;
  afterTheFact: boolean;
  live: boolean;
}
export interface GuardCard {
  rule: string;
  forwardFrom: string | null;
  shownFrom: string | null;
  live: { weeks: number; guard: number | null; plain: number | null; market: number | null; afterTheFact: number };
  weeks: GuardWeek[]; // newest first
}
export interface RegimeLine {
  row: string;
  date: string | null;
  regimes: number | null;
  label: string;
  template: number | null;
  probability: number | null;
  volatilityRank: number | null;
  features: string[];
  expected: { symbol: string; value: number | null }[];
  readings: number | null;
}
export interface Workbench {
  market: string;
  builtAt: string | null;
  title: string;
  noun: string;
  unit: string;
  calendar: string;
  weekGame: string | null;
  k: number;
  chance: { exact: number | null; adjacent: number | null; direction: number | null };
  fee: number | null;
  generatedAt: string | null;
  newestGameDay: string | null;
  madeOn: string | null;
  madeAt: string | null;
  scoredDays: number | null;
  stake: number | null;
  feePerLeg: number | null;
  currency: string;
  best: string | null;
  instruments: { symbol: string; name: string; position: number; quote: string; active: boolean; lastClose: number | null; lastDate: string | null; calls: number }[];
  judged: { day: string | null; at: string | null; status: string; madeOn: string | null } | null;
  plan: Plan | null;
  week: WeekCard | null;
  followed: { day: Followed | null; week: Followed | null };
  books: Books | null;
  models: ModelRow[];
  accuracy: { chance: number | null; days: { date: string; mean: number | null; best: number | null; bestModel: string | null }[] };
  days: { date: string; best: string | null; exactMean: number | null; instruments: { symbol: string; ret: number | null; bin: number | null }[] }[];
  guard: GuardCard | null;
  regimes: RegimeLine[];
  example: {
    date: string;
    instruments: { symbol: string; ret: number | null; bin: number | null; edges: number[]; close: { before: number | null; on: number | null } }[];
    best: { name: string; bins: (number | null)[]; exact: number | null; adjacent: number | null; direction: number | null; positions: number | null } | null;
  } | null;
}
function bookLineOf(v: unknown): BookLine {
  const b = o(v);
  return { total: opt(b.total), positions: opt(b.positions), stake: opt(b.stake), settled: opt(b.settledDays) ?? opt(b.settledWeeks) ?? opt(b.scoredDays),
    firstDay: ostr(b.firstDay), lastDay: ostr(b.lastDay), market: opt(b.market), ahead: opt(b.ahead), exact: opt(b.exact) };
}
function recordMixOf(v: unknown): RecordMix | null {
  if (!has(v)) return null;
  const r = o(v);
  return { firstLive: ostr(r.firstLive), backBuilt: num(r.backBuilt, 0), live: num(r.live, 0), late: num(r.late, 0), lastLive: ostr(r.lastLive) };
}
function soFarOf(v: unknown): SoFar | null {
  if (!has(v)) return null;
  const s = o(v);
  return { liveDays: num(s.liveDays, 0), positions: opt(s.positions), firstLive: ostr(s.firstLive), lastLive: ostr(s.lastLive), total: opt(s.total), market: opt(s.market), ahead: opt(s.ahead) };
}
function pointOf(v: unknown): FollowedPoint | null {
  if (!Array.isArray(v)) return null;
  const by = v[2] === 'book' || v[2] === 'exact' ? v[2] : null;
  return [str(v[0]), ostr(v[1]), by, num(v[3], 0), opt(v[4]), opt(v[5])];
}
const pointsOf = (v: unknown): FollowedPoint[] => arr(v).map(pointOf).filter((x): x is FollowedPoint => x !== null && x[0] !== '');
function followedOf(v: unknown): Followed | null {
  if (!has(v)) return null;
  const f = o(v);
  const part = (x: unknown) => ({ n: num(o(x).n, 0), total: opt(o(x).total), market: opt(o(x).market), points: pointsOf(o(x).points) });
  return {
    rule: str(f.rule), week: bool(f.week), minTrades: num(f.minTrades, 0), firstLive: ostr(f.firstLive), liveDays: num(f.liveDays, 0), positions: opt(f.positions),
    total: opt(f.total), market: opt(f.market), ahead: opt(f.ahead), live: pointsOf(f.live), late: part(f.late), backBuilt: part(f.backBuilt),
  };
}
function bookFiguresOf(v: unknown): BookFigures | null {
  if (!has(v)) return null;
  const b = o(v);
  return { trades: opt(b.trades), winRate: opt(b.winRate), total: opt(b.total), perTrade: opt(b.perTrade) };
}
function headOf(v: unknown) {
  if (!has(v)) return null;
  const h = o(v);
  return { best: has(h.best) ? { model: str(o(h.best).model), total: opt(o(h.best).total) } : null, market: opt(h.market) };
}
export function normWorkbench(raw: unknown, market: string): Workbench {
  const r = o(raw);
  const b = o(r.books);
  const hl = o(b.headline);
  const ls = o(hl.longShort);
  const p = o(r.plan);
  const w = o(r.week);
  const g = o(r.guard);
  const acc = o(r.accuracy);
  const ex = o(r.example);
  const exBest = o(ex.best);
  return {
    market: str(r.market, market), builtAt: ostr(r.builtAt), title: str(r.title), noun: str(r.noun, 'instrument'), unit: str(r.unit), calendar: str(r.calendar),
    weekGame: ostr(r.weekGame), k: num(r.k, 10), chance: chanceOf(r.chance), fee: opt(r.fee), generatedAt: ostr(r.generatedAt), newestGameDay: ostr(r.newestGameDay),
    madeOn: ostr(r.madeOn), madeAt: ostr(r.madeAt), scoredDays: opt(r.scoredDays), stake: opt(r.stake), feePerLeg: opt(r.feePerLeg), currency: str(r.currency, str(r.unit)),
    best: ostr(r.best),
    instruments: arr(r.instruments).filter(has).map((x, i) => {
      const n = o(x);
      return { symbol: str(n.symbol), name: str(n.name), position: num(n.position, i), quote: str(n.quote), active: n.active !== false, lastClose: opt(n.lastClose), lastDate: ostr(n.lastDate), calls: num(n.calls, 0) };
    }),
    judged: has(r.judged) ? { day: ostr(o(r.judged).day), at: ostr(o(r.judged).at), status: str(o(r.judged).status), madeOn: ostr(o(r.judged).madeOn) } : null,
    plan: has(r.plan)
      ? {
          day: str(p.day), calendarKnown: p.calendarKnown !== false, madeOn: ostr(p.madeOn), madeAt: ostr(p.madeAt), openAt: ostr(p.openAt), closeAt: ostr(p.closeAt),
          closed: bool(p.closed), opened: bool(p.opened), status: str(p.status), model: str(p.model), byBook: bool(p.byBook), rule: str(p.rule), minTrades: num(p.minTrades, 10),
          models: num(p.models, 0), book: bookLineOf(p.book), record: recordMixOf(p.record), soFar: soFarOf(p.soFar),
          rows: arr(p.rows).filter(has).map((x) => {
            const q = o(x);
            return { symbol: str(q.symbol), base: opt(q.base), call: callOf(q.call), order: str(q.order), up: num(q.up, 0), of: num(q.of, 0) };
          }),
        }
      : null,
    week: has(r.week)
      ? {
          weekEnd: str(w.weekEnd), firstDay: ostr(w.firstDay), madeOn: ostr(w.madeOn), madeAt: ostr(w.madeAt), openAt: ostr(w.openAt), closeAt: ostr(w.closeAt),
          started: bool(w.started), closed: bool(w.closed), hoursLeft: opt(w.hoursLeft), status: str(w.status), settledWeeks: opt(w.settledWeeks),
          model: str(w.model), byBook: bool(w.byBook), rule: str(w.rule), minTrades: num(w.minTrades, 5), models: num(w.models, 0), book: bookLineOf(w.book),
          guard: { from: ostr(o(w.guard).from), applies: bool(o(w.guard).applies) }, record: recordMixOf(w.record), soFar: soFarOf(w.soFar),
          rows: arr(w.rows).filter(has).map((x) => {
            const q = o(x);
            const sf = o(q.soFar);
            return {
              symbol: str(q.symbol), base: opt(q.base), call: callOf(q.call),
              soFar: has(q.soFar) ? { closes: num(sf.closes, 0), last: opt(sf.last), lastDate: ostr(sf.lastDate), move: opt(sf.move), where: str(sf.where) } : null,
              order: str(q.order), up: num(q.up, 0), of: num(q.of, 0),
            };
          }),
        }
      : null,
    followed: { day: followedOf(o(r.followed).day), week: followedOf(o(r.followed).week) },
    books: has(r.books)
      ? {
          days: num(b.days, 0), first: ostr(b.first), last: ostr(b.last), stake: opt(b.stake), feePerLeg: opt(b.feePerLeg), currency: str(b.currency), minDays: num(b.minDays, 10),
          seasonedRows: opt(b.seasonedRows), rules: strs(b.rules).filter((x): x is BookRule => BOOK_RULES.includes(x as BookRule)),
          ruleText: { daily: str(o(b.ruleText).daily), hold: str(o(b.ruleText).hold), open: ostr(o(b.ruleText).open), short: str(o(b.ruleText).short) },
          headline: {
            daily: headOf(hl.daily), hold: headOf(hl.hold), open: headOf(hl.open),
            longShort: has(hl.longShort) ? { best: has(ls.best) ? { model: str(o(ls.best).model), total: opt(o(ls.best).total) } : null, bought: opt(ls.bought), sold: opt(ls.sold) } : null,
          },
          best: ostr(b.best),
        }
      : null,
    models: arr(r.models).filter(has).map((x) => {
      const m = o(x);
      const l = o(m.longShort);
      return {
        name: str(m.name), days: opt(m.days), firstDay: ostr(m.firstDay), lastDay: ostr(m.lastDay), sized: bool(m.sized),
        exact: opt(m.exact), adjacent: opt(m.adjacent), direction: opt(m.direction), aboveExact: bool(m.aboveExact), aboveAdjacent: bool(m.aboveAdjacent), aboveDirection: bool(m.aboveDirection),
        daily: bookFiguresOf(m.daily), hold: bookFiguresOf(m.hold), open: bookFiguresOf(m.open),
        longShort: has(m.longShort) ? { ...(bookFiguresOf(m.longShort) as BookFigures), hold: opt(l.hold), open: opt(l.open) } : null,
      };
    }),
    accuracy: {
      chance: opt(acc.chance),
      days: arr(acc.days)
        .filter((d): d is unknown[] => Array.isArray(d) && typeof d[0] === 'string')
        .map((d) => ({ date: d[0] as string, mean: opt(d[1]), best: opt(d[2]), bestModel: ostr(d[3]) })),
    },
    days: arr(r.days).filter(has).map((x) => {
      const d = o(x);
      return { date: str(d.date), best: ostr(d.best), exactMean: opt(d.exactMean), instruments: arr(d.instruments).filter(has).map((i) => ({ symbol: str(o(i).symbol), ret: opt(o(i).ret), bin: opt(o(i).bin) })) };
    }),
    guard: has(r.guard)
      ? {
          rule: str(g.rule), forwardFrom: ostr(g.forwardFrom), shownFrom: ostr(g.shownFrom),
          live: { weeks: num(o(g.live).weeks, 0), guard: opt(o(g.live).guard), plain: opt(o(g.live).plain), market: opt(o(g.live).market), afterTheFact: num(o(g.live).afterTheFact, 0) },
          weeks: arr(g.weeks).filter(has).map((x) => {
            const q = o(x);
            return { date: str(q.date), model: ostr(q.model), positions: opt(q.positions), guard: opt(q.guard), plain: opt(q.plain), market: opt(q.market),
              userPages: str(q.userPages), afterTheFact: bool(q.afterTheFact), live: bool(q.live) };
          }),
        }
      : null,
    regimes: arr(r.regimes).filter(has).map((x) => {
      const q = o(x);
      return {
        row: str(q.row), date: ostr(q.date), regimes: opt(q.regimes), label: str(q.label), template: opt(q.template), probability: opt(q.probability),
        volatilityRank: opt(q.volatilityRank), features: strs(q.features), expected: arr(q.expected).filter(has).map((e) => ({ symbol: str(o(e).symbol), value: opt(o(e).value) })),
        readings: opt(q.readings),
      };
    }),
    example: has(r.example)
      ? {
          date: str(ex.date),
          instruments: arr(ex.instruments).filter(has).map((i) => {
            const q = o(i);
            return { symbol: str(q.symbol), ret: opt(q.ret), bin: opt(q.bin), edges: nums(q.edges), close: { before: opt(o(q.close).before), on: opt(o(q.close).on) } };
          }),
          best: has(ex.best)
            ? { name: str(exBest.name), bins: optNums(exBest.bins), exact: opt(exBest.exact), adjacent: opt(exBest.adjacent), direction: opt(exBest.direction), positions: opt(exBest.positions) }
            : null,
        }
      : null,
  };
}

// GET .../workbench/books/:rule: one paper book's cumulative money per model, aligned to the dates
export interface BookSeries {
  rule: BookRule;
  dates: string[];
  currency: string;
  stake: number | null;
  best: string | null;
  market: (number | null)[];
  sold: (number | null)[] | null;
  order: string[];
  series: Record<string, (number | null)[]>;
}
export function normBooks(raw: unknown, rule: BookRule): BookSeries {
  const r = o(raw);
  return {
    rule, dates: strs(r.dates), currency: str(r.currency), stake: opt(r.stake), best: ostr(r.best), market: optNums(r.market),
    sold: Array.isArray(r.sold) ? optNums(r.sold) : null, order: strs(r.order),
    series: Object.fromEntries(Object.entries(o(r.series)).map(([k, v]) => [k, optNums(v)])),
  };
}

// GET .../workbench/instrument/:symbol: a year of closes, the tracked period's bin edges and every model's bin per day
export interface InstrumentSeries {
  symbol: string;
  name: string;
  quote: string;
  active: boolean;
  lastClose: number | null;
  lastDate: string | null;
  nextBase: number | null;
  nextDate: string | null;
  nextDay: string | null;
  k: number;
  dates: string[];
  closes: (number | null)[];
  moves: (number | null)[];
  from: number; // index of the first tracked day
  edges: (number[] | null)[]; // dates[from..]
  order: string[];
  course: Record<string, (number | null)[]>; // each model's bin, dates[from..]
  next: { model: string; bin: number | null; size: number | null; up: boolean; price: number | null; low: number | null; high: number | null }[];
  best: string | null;
}
export function normInstrument(raw: unknown, symbol: string): InstrumentSeries {
  const r = o(raw);
  return {
    symbol: str(r.symbol, symbol), name: str(r.name), quote: str(r.quote), active: r.active !== false, lastClose: opt(r.lastClose), lastDate: ostr(r.lastDate),
    nextBase: opt(r.nextBase), nextDate: ostr(r.nextDate), nextDay: ostr(r.nextDay), k: num(r.k, 10),
    dates: strs(r.dates), closes: optNums(r.closes), moves: optNums(r.moves), from: num(r.from, 0),
    edges: arr(r.edges).map((e) => (Array.isArray(e) ? nums(e) : null)), order: strs(r.order),
    course: Object.fromEntries(Object.entries(o(r.course)).map(([k, v]) => [k, optNums(v)])),
    next: arr(r.next).filter(has).map((x) => {
      const q = o(x);
      return { model: str(q.model), bin: opt(q.bin), size: opt(q.size), up: bool(q.up), price: opt(q.price), low: opt(q.low), high: opt(q.high) };
    }),
    best: ostr(r.best),
  };
}

const enc = encodeURIComponent;
export const getHistory = async (game: string, scope: Scope, month?: string): Promise<History> =>
  normHistory(await getJson<unknown>(`/api/admin/history/${enc(game)}?scope=${scope}${month ? `&month=${enc(month)}` : ''}`), game, scope);
export const getDraw = async (game: string, date: string, scope: Scope): Promise<Draw> =>
  normDraw(await getJson<unknown>(`/api/admin/draw/${enc(game)}/${enc(date)}?scope=${scope}`), game, date, scope);
export const getWorkbench = async (market: string): Promise<Workbench> => normWorkbench(await getJson<unknown>(`/api/admin/market/${enc(market)}/workbench`), market);
export const getBooks = async (market: string, rule: BookRule): Promise<BookSeries> =>
  normBooks(await getJson<unknown>(`/api/admin/market/${enc(market)}/workbench/books/${rule}`), rule);
export const getInstrument = async (market: string, symbol: string): Promise<InstrumentSeries> =>
  normInstrument(await getJson<unknown>(`/api/admin/market/${enc(market)}/workbench/instrument/${enc(symbol)}`), symbol);
