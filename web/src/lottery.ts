// The users' Lottery and Home (lottery.js, batch 5): every signed-in reader gets the user view - the administrator too.
// A game's coming draw shows today's five rows (the History ranking's top five), a past draw the five ranked before it,
// and the draws before the game's first saved ranking are left out; every figure is over the rows sent. The shapes are
// the Workbench's (workbench.ts), so its normalisers read them; this file adds the Lottery's cards, the history's whole
// total and the line on the draws left out, and Home's one answer.
import { getJson } from './api';
import {
  bestOf,
  comingScopeOf,
  drawScopeOf,
  figuresOf,
  multiPickLegendOf,
  normDraw,
  normHistory,
  resultOf,
  rowOf,
  shapeOf,
  versionOf,
  noneOf,
  type Best,
  type Coming,
  type DayFigures,
  type Draw,
  type DrawResult,
  type DrawScope,
  type History,
  type MultiPickLegendData,
  type RankVersion,
  type Row,
  type Shape,
} from './workbench';

type Raw = Record<string, unknown>;
const o = (v: unknown): Raw => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Raw) : {});
const has = (v: unknown): boolean => Boolean(v) && typeof v === 'object' && !Array.isArray(v);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const opt = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const num = (v: unknown, fallback: number): number => opt(v) ?? fallback;
const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const ostr = (v: unknown): string | null => (typeof v === 'string' && v ? v : null);

// The users' games, in the server's whitelist order (lottery.js GAMES); the markets have their own pages
export const USER_GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
export const isUserGame = (s: unknown): s is string => typeof s === 'string' && USER_GAMES.includes(s);

// The coming draw: its date on the game's own schedule, its sales close (src/drawClose.json, 27 Oct 2026: "13:00",
// Brussels time) and when its tickets were made - afterClose when that was at or after the close: then they are for
// reading, not for playing, and afterCloseText says so
export interface NextDraw {
  date: string | null;
  weekday: string | null;
  time: string | null; // the sales close, Brussels time
  past: boolean; // the date has gone and no newer run has scored it yet
  today: boolean; // the date is today's in Brussels
  madeAt: string | null;
  closeAt: string | null;
  afterClose: boolean;
  afterCloseText: string | null;
}
function nextOf(v: unknown): NextDraw | null {
  if (!has(v)) return null;
  const n = o(v);
  return { date: ostr(n.date), weekday: ostr(n.weekday), time: ostr(n.time), past: n.past === true, today: n.today === true, madeAt: ostr(n.madeAt),
    closeAt: ostr(n.closeAt), afterClose: n.afterClose === true, afterCloseText: ostr(n.afterCloseText) };
}

// A pipeline job holding process.lock (pipeline.js), on /api/lottery and /api/home: the job in the old banner's words and
// since when; null when nothing runs
export interface Running {
  what: string;
  since: string | null;
}
function runningOf(v: unknown): Running | null {
  if (!has(v)) return null;
  const r = o(v);
  return { what: str(r.what, 'A pipeline job is running'), since: ostr(r.since) };
}

// --- GET /api/lottery ---------------------------------------------------------------------------------------------------
// One row of the last draw's five, with its own figures (the best match without "by": it is the row's own)
export interface LastRow {
  name: string;
  rank: number | null;
  tickets: number;
  profit: number | null;
  stake: number | null;
  best: Best | null;
  multi: number | null;
}
export interface LastDraw {
  date: string;
  weekday: string;
  result: DrawResult | null;
  scope: DrawScope | null; // the five ranked before it; null when it is left out
  figures: DayFigures | null; // over those five together
  rows: LastRow[];
  leftOut: string | null; // "no ranking of jokerplus had been saved before 15:00 Brussels time that day (the first was saved on …)"
  none: { code: string; text: string } | null; // no top 5 and not left out: the saved rankings could not be read just now
  afterClose: boolean; // a ticket of its five was made after the draw's sales close: shown, out of the totals
}
export interface GameCard {
  game: string;
  unit: string;
  empty: string | null; // no stored draw yet: nothing else is filled
  shape: Shape;
  next: NextDraw | null;
  coming: { scope: Coming['scope']; rows: Row[] }; // today's five, in ranking order
  multiPick: MultiPickLegendData | null; // lotto: the system play's sizes, prices and chances, once per game
  last: LastDraw | null;
}
export interface LotteryOverview {
  builtAt: string | null;
  topN: number;
  firstSaved: { committedAt: string | null; generatedAt: string | null } | null;
  games: GameCard[];
  running: Running | null;
}
const savedOf = (v: unknown) => (has(v) ? { committedAt: ostr(o(v).committedAt), generatedAt: ostr(o(v).generatedAt) } : null);
function lastOf(v: unknown): LastDraw | null {
  if (!has(v)) return null;
  const l = o(v);
  return {
    date: str(l.date), weekday: str(l.weekday), result: resultOf(l.result),
    scope: has(l.scope) ? drawScopeOf(l.scope, 'top') : null, figures: figuresOf(l.figures),
    rows: arr(l.rows).filter(has).map((x) => {
      const r = o(x);
      return { name: str(r.name), rank: opt(r.rank), tickets: num(r.tickets, 0), profit: opt(r.profit), stake: opt(r.stake), best: bestOf(r.best), multi: opt(r.multi) };
    }),
    leftOut: ostr(l.leftOut),
    none: noneOf(l.none),
    afterClose: l.afterClose === true,
  };
}
function cardOf(v: unknown): GameCard {
  const c = o(v);
  const game = str(c.game);
  const coming = o(c.coming);
  return {
    game, unit: str(c.unit, 'draws'), empty: ostr(c.empty), shape: shapeOf(c.shape, game), next: nextOf(c.next),
    coming: { scope: comingScopeOf(coming.scope, 'top'), rows: arr(coming.rows).filter(has).map(rowOf) },
    multiPick: multiPickLegendOf(c.multiPick), last: lastOf(c.last),
  };
}
export function normOverview(raw: unknown): LotteryOverview {
  const r = o(raw);
  return { builtAt: ostr(r.builtAt), topN: num(r.topN, 5), firstSaved: savedOf(r.firstSaved), games: arr(r.games).filter(has).map(cardOf), running: runningOf(r.running) };
}

// --- GET /api/lottery/:game ---------------------------------------------------------------------------------------------
// The Workbench's history with the scope Top, from the game's first saved ranking on, plus the whole history's totals over
// the same rows and the line on the draws left out
export interface HistoryTotal {
  from: string | null;
  to: string | null;
  draws: number;
  counted: number;
  afterClose: number; // draws shown greyed, their tickets made after the sales close: out of every total
  rows: number; // row-draws, summed over the counted draws: not a count of rows
  tickets: number;
  profit: number | null;
  stake: number | null;
  best: Best | null;
  multi: number | null;
}
export interface LeftOut {
  draws: number;
  firstSaved: string | null;
  firstCounted: string | null;
  text: string;
}
export interface UserHistory extends History {
  total: HistoryTotal | null;
  leftOut: LeftOut | null;
}
export function normUserHistory(raw: unknown, game: string): UserHistory {
  const r = o(raw);
  const t = o(r.total);
  const l = o(r.leftOut);
  return {
    ...normHistory(raw, game, 'top'),
    total: has(r.total)
      ? { from: ostr(t.from), to: ostr(t.to), draws: num(t.draws, 0), counted: num(t.counted, 0), afterClose: num(t.afterClose, 0), rows: num(t.rows, 0), tickets: num(t.tickets, 0),
          profit: opt(t.profit), stake: opt(t.stake), best: bestOf(t.best), multi: opt(t.multi) }
      : null,
    leftOut: has(r.leftOut) ? { draws: num(l.draws, 0), firstSaved: ostr(l.firstSaved), firstCounted: ostr(l.firstCounted), text: str(l.text) } : null,
  };
}

// --- GET /api/home ------------------------------------------------------------------------------------------------------
export interface HomeGame {
  game: string;
  unit: string;
  empty: string | null;
  shape: Shape;
  next: NextDraw | null;
  top: Row | null; // the first of today's five, with its coming tickets
  rows: number; // today's five shown on the Lottery page
  ranking: (RankVersion & { today: true }) | null;
  none: { code: string; text: string } | null;
  last: { date: string | null; rows: number; figures: DayFigures | null; none: { code: string; text: string } | null; leftOut: string | null; afterClose: boolean } | null;
}
export interface MarketHeadline {
  market: string;
  title: string;
  unit: string;
  empty: string | null;
  // following the plan: the day and week positions together, live days only, against buying everything over the same days
  plan: {
    since: string | null;
    through: string | null;
    days: number;
    positions: number;
    total: number | null;
    market: number | null;
    ahead: number | null;
    paperNote: string;
    entryNote: string | null;
  } | null;
  today: { date: string | null; state: string | null; toBuy: number; held: number } | null;
}
export interface Home {
  builtAt: string | null;
  lottery: HomeGame[];
  crypto: MarketHeadline;
  shares: MarketHeadline;
  running: Running | null;
}
function homeGameOf(v: unknown): HomeGame {
  const g = o(v);
  const game = str(g.game);
  const l = o(g.last);
  const ranking = versionOf(g.ranking);
  return {
    game, unit: str(g.unit, 'draws'), empty: ostr(g.empty), shape: shapeOf(g.shape, game), next: nextOf(g.next),
    top: has(g.top) ? rowOf(g.top) : null, rows: num(g.rows, 0), ranking: ranking ? { ...ranking, today: true } : null, none: noneOf(g.none),
    last: has(g.last) ? { date: ostr(l.date), rows: num(l.rows, 0), figures: figuresOf(l.figures), none: noneOf(l.none), leftOut: ostr(l.leftOut), afterClose: l.afterClose === true } : null,
  };
}
const TITLE: Record<string, string> = { crypto: 'Crypto', shares: 'Shares' };
function headlineOf(v: unknown, market: string): MarketHeadline {
  const m = o(v);
  const p = o(m.plan);
  const t = o(m.today);
  return {
    market, title: str(m.title, TITLE[market] ?? market), unit: str(m.unit), empty: has(v) ? ostr(m.empty) : 'no record yet',
    plan: has(m.plan)
      ? { since: ostr(p.since), through: ostr(p.through), days: num(p.days, 0), positions: num(p.positions, 0), total: opt(p.total), market: opt(p.market),
          ahead: opt(p.ahead), paperNote: str(p.paperNote), entryNote: ostr(p.entryNote) }
      : null,
    today: has(m.today) ? { date: ostr(t.date), state: ostr(t.state), toBuy: num(t.toBuy, 0), held: num(t.held, 0) } : null,
  };
}
export function normHome(raw: unknown): Home {
  const r = o(raw);
  return {
    builtAt: ostr(r.builtAt), lottery: arr(r.lottery).filter(has).map(homeGameOf), crypto: headlineOf(r.crypto, 'crypto'), shares: headlineOf(r.shares, 'shares'),
    running: runningOf(r.running),
  };
}

const enc = encodeURIComponent;
export const getLottery = async (): Promise<LotteryOverview> => normOverview(await getJson<unknown>('/api/lottery'));
export const getUserHistory = async (game: string, month?: string): Promise<UserHistory> =>
  normUserHistory(await getJson<unknown>(`/api/lottery/${enc(game)}${month ? `?month=${enc(month)}` : ''}`), game);
export const getUserDraw = async (game: string, date: string): Promise<Draw> => normDraw(await getJson<unknown>(`/api/lottery/${enc(game)}/${enc(date)}`), game, date, 'top');
export const getHome = async (): Promise<Home> => normHome(await getJson<unknown>('/api/home'));
