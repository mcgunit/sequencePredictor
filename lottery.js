// lottery.js - the users' Lottery and Home (23 Oct 2026, batch 5: the users' side of the app). Read-only JSON for every
// signed-in reader; the administrator gets the same view as a user here (the Workbench has everything else):
//
//   GET /api/lottery                    per lottery game: the coming draw (its date, today's five rows and their
//                                       tickets with the game's extras) and the last scored draw (its result and how
//                                       the five ranked first BEFORE it did, row by row)
//   GET /api/lottery/:game?month=       the game's draws from its first saved ranking on, by month, newest first (one
//                                       month's draws per answer, by default the newest): per draw the result, the best
//                                       match and the money over the five ranked first before that draw; each month's
//                                       totals and the whole history's over the same rows; the draws before the first
//                                       saved ranking left out, and one line saying so
//   GET /api/lottery/:game/:date        one draw (YYYY-MM-DD): its result and the five rows of its point-in-time ranking
//                                       with their hits; on the newest day also the coming draw's five by today's ranking
//   GET /api/home                       the Home page in one answer: per lottery game the next draw and its top pick,
//                                       per market (crypto, shares) the plan's headline and today's positions, counted
//
// The owner's rules (21-23 Oct 2026). A user sees the top five rows of each game's History ranking and nothing else:
// for the coming draw today's five (ranking.js usersTop, the five /lottery showed), for a past draw the five ranked
// first before it (rankhistory.js topAt, the 15:00 Brussels cut-off) - never today's five applied to a past draw. A
// draw without a ranking of its game saved before its cut-off (the first: 25 Aug 2026; Joker+ 7 Sep at 18:56, so its
// 7 Sep draw too) is left out, never read against a rebuilt ranking, and the lines say so by that rule. Every figure - hits, best match, money - is over those five rows only, and a row outside them is never
// sent, nor a vote of every row: the views are workbench.js's with the scope Top (draws.js scores them through its
// `keep` filter), and each answer is checked against the five once more before it leaves (onlyTop: the five are read
// again from the ranking, and a row outside them is a 500, never a leak). Controls and model internals stay in the
// admin section.
//
// Home's two markets are api.js marketView's status - Following the plan against buying everything over the same
// days, with its window - and today's positions, counted: the figures of /app/crypto and /app/shares in about 1 KB
// instead of two whole views (27 KB on 23 Oct). One small answer for a phone rather than three routes combined in the
// page (the choice of 23 Oct, batch 5).
//
// Caching: the day files are read once per change (workbench.js, keyed by modification time and size), the History
// ranking once per change (ranking.js), its git versions once per sha (rankhistory.js, memory and a gitignored cache
// file), the market records once per change (here). A history answer and the overview are kept until one of their
// sources changes: their stamp is HEAD and the versions read, the report's and every counted day file's modification
// time and size.
'use strict';

const fs = require('fs');
const path = require('path');
const admin = require('./admin');
const api = require('./api');
const brussels = require('./brussels');
const draws = require('./draws');
const markets = require('./markets');
const rankings = require('./ranking');
const rankhistory = require('./rankhistory');
const workbench = require('./workbench');

const GAMES = admin.LOTTERY_GAMES;                     // the users' Lottery: the markets have their own pages (api.js)
const TOP_N = rankings.TOP_MODELS_SHOWN;
const MARKETS = Object.keys(markets.MARKETS);          // crypto, shares
const DAY = /^\d{4}-\d{2}-\d{2}$/;
const MONTH = /^\d{4}-\d{2}$/;

// --- small helpers ---------------------------------------------------------------------------------------------
const list = (v) => (Array.isArray(v) ? v : []);
const nameOf = (m) => (m && m.name) || 'not known';
const namesOf = (rows) => [...new Set(list(rows).filter((m) => m && typeof m === 'object' && Array.isArray(m.predictions)).map(nameOf))];
const isoNow = (now) => now.toISOString().replace(/\.\d{3}Z$/, 'Z');
const round2 = (x) => Math.round(x * 100) / 100;
// a day as the user routes take it: YYYY-MM-DD and a real date (the admin routes also read 2026-10-6)
const strictDay = (text) => (DAY.test(String(text)) && workbench.isoDay(text) === String(text) ? String(text) : null);
const stampOf = (file) => { try { const s = fs.statSync(file); return `${s.mtimeMs}:${s.size}`; } catch (e) { return 'missing'; } };
// a ranking version as a user gets it: when it was saved and generated (its sha stays in the admin section)
const savedOf = (v) => (v ? { committedAt: v.committedAt || null, generatedAt: v.generatedAt || null } : null);
const extraOf = (game) => ({ group: 'Lottery', unit: admin.UNIT[game] });
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

// An answer kept until its stamp changes (see the header). A few keys per game: the overview, the home page, a history
// per month asked; an answer that is an error is not kept.
const memo = new Map();   // key -> {stamp, body}
function remember(key, stamp, build) {
  const hit = memo.get(key);
  if (hit && hit.stamp === stamp) return hit.body;
  const body = build();
  if (body && !body.error) memo.set(key, { stamp, body });
  return body;
}
// what the ranking history is made of now: HEAD, the versions read, the versions git could not read
const histStamp = (hist) => `${hist.head}:${hist.versions.length}:${hist.unread}:${hist.error || ''}`;

// The market records (1 to 1.7 MB each), parsed once per change of the file; null when missing or not a record.
const records = new Map();   // file -> {stamp, record}
function recordOf(dir, game) {
  const file = path.join(dir, `${game}.json`);
  const stamp = stampOf(file);
  const hit = records.get(file);
  if (hit && hit.stamp === stamp) return hit.record;
  let record = null;
  try { record = markets.loadMarket(dir, game); } catch (e) { record = hit ? hit.record : null; }
  records.set(file, { stamp, record });
  return record;
}

// --- where a game's history starts -----------------------------------------------------------------------------
// The first saved ranking that ranks the game (rankhistory firstOf: Joker+ entered after the first of all, 25 Aug
// 2026), and the test of a draw: its cut-off (15:00 Brussels on its day) comes after that ranking was saved, so the
// ranking it is read against exists. null when no readable version ranks the game. time: when it was saved, Brussels
// wall time (Joker+'s first, 7 Sep 2026 18:56, came after that day's cut-off: its first counted draw is 8 Sep). unsure:
// a version older than it could not be read (a timeout; asked again after 30 s) and may rank the game too, so the
// draws before it are not said to be left out: they could not be checked just now.
function startOf(hist, game) {
  const first = hist.firstOf(game);
  const at = first ? Date.parse(first.committedAt) : NaN;
  if (!Number.isFinite(at)) return null;
  const p = brussels.parts(at);
  return { first: savedOf(first), day: brussels.dayOf(at), time: `${String(p.hour).padStart(2, '0')}:${String(p.minute).padStart(2, '0')}`,
    unsure: list(hist.unreadAt).some((u) => u < at), counts: (date) => { const c = rankhistory.cutoffOf(game, date); return c !== null && c > at; } };
}
// the ranking history is not whole where the game's would start: git could not be read at all, or a version it could
// not read comes before the game's first readable one (or no readable one ranks the game)
const unreadStart = (hist, start) => Boolean((hist.error && !hist.versions.length) || (start ? start.unsure : list(hist.unreadAt).length));
const UNREAD = 'the ranking history could not be read right now';
// no start: the history is not whole (503: asked again shortly), or no saved ranking ranks the game yet
const noStart = (hist, game) => (unreadStart(hist, null)
  ? { error: UNREAD, status: 503 }
  : { error: `no ranking of ${game} has been saved yet, so its draws are left out`, code: 'not-ranked', status: 404 });
// why a draw before the game's first counted one is left out, by the rule itself: no ranking of the game had been
// saved before the cut-off on its day (the first saved ranking can come later that same day: Joker+ on 7 Sep)
const cutoffText = `${rankhistory.CUTOFF_HOUR}:00 Brussels time`;
const savedText = (start) => `the first was saved on ${start.day} at ${start.time}`;

// A ranking git could not read, in a reader's words: its reason (a timeout, git's exit code) is the server log's and
// the admin pages' (rankhistory.js keeps it in the history's error and the 'unavailable' line).
const READ_ERROR = 'a saved ranking could not be read just now';
const plainScope = (s) => (s && s.none && s.none.code === 'unavailable'
  ? { ...s, none: { ...s.none, text: 'the ranking saved before this draw could not be read just now' } } : s);

// --- the last check before an answer leaves --------------------------------------------------------------------------
// Every row sent is one of the five the rules allow - read again here from the ranking, not from the view - and every
// figure is over the rows sent. A failure is a bug: it throws, the route answers 500 and logs it; nothing is sent.
function check(cond, what) { if (!cond) throw new Error(`lottery.js: ${what}`); }
function onlyTop(view, allowed, comingAllowed) {
  const where = `${view.game} ${view.date}`;
  const within = (rows, set) => rows.length <= TOP_N && rows.every((r) => set.has(r.name));
  check(within(list(view.rows), allowed), `${where}: a row outside the five ranked before the draw`);
  check(view.scope && view.scope.scope === 'top' && list(view.scope.top).every((n) => allowed.has(n)), `${where}: a scope that is not the draw's five`);
  check(!view.figures || (view.figures.rows === view.rows.length && (!view.figures.best || allowed.has(view.figures.best.by))), `${where}: figures over other rows`);
  check(view.votes && view.votes.stored === undefined && view.votes.rows === view.rows.length, `${where}: a vote of other rows`);
  check(!view.market, `${where}: a market section`);
  if (view.coming) {
    check(comingAllowed && within(list(view.coming.rows), comingAllowed), `${where}: a coming row outside today's five`);
    check(view.coming.votes && view.coming.votes.stored === undefined && view.coming.votes.rows === view.coming.rows.length, `${where}: a vote of other coming rows`);
  }
}
// the five a draw allows (the ranking before it) and, on the newest day, the five the coming draw allows (today's)
function allowedFor(dirs, game, file, date, hist, report, newest) {
  const day = workbench.readFull(path.join(dirs.databaseDir, game, file)) || { currentPrediction: [], newPrediction: [] };
  const t = hist.topAt(game, date, namesOf(day.currentPrediction));
  return { day, allowed: new Set(t.top || []), comingAllowed: newest ? new Set(rankings.usersTop(report, game, namesOf(day.newPrediction))) : null };
}
// a history's draws: at most five rows each, and the best ticket one of the draw's five. files: the day files counted.
function checkHistory(dirs, game, view, hist, files) {
  const fileOf = new Map(files.map((f) => [f.date, f.file]));
  view.months.forEach((m) => {
    check(m.rowsMax === null || m.rowsMax <= TOP_N, `${game} ${m.month}: more than ${TOP_N} rows on a draw`);
    list(m.days).forEach((d) => {
      check(fileOf.has(d.date), `${game} ${d.date}: a draw that is left out`);
      if (d.rows === undefined) return;
      const day = workbench.readDay(path.join(dirs.databaseDir, game, fileOf.get(d.date))) || { currentPrediction: [] };
      const top = new Set(hist.topAt(game, d.date, namesOf(day.currentPrediction)).top || []);
      check(d.rows <= TOP_N && d.rows <= top.size && (!d.best || top.has(d.best.by)), `${game} ${d.date}: figures over a row outside the draw's five`);
    });
  });
}

// --- GET /api/lottery/:game/:date ------------------------------------------------------------------------------------
// The admin draw view with the scope Top (workbench.drawView), checked, without the day file's name; prev skips the
// draws left out. A draw before the game's first saved ranking is a 404 that says so (code 'before-first').
function drawOf(dirs, game, date, hist, report, now = new Date()) {
  const files = workbench.dayFilesOf(path.join(dirs.databaseDir, game));
  const at = files.findIndex((f) => f.date === date);
  if (at < 0) return { error: 'no draw of this game on that date', status: 404 };
  const start = startOf(hist, game);
  if (!start) return noStart(hist, game);
  if (!start.counts(date)) {
    if (start.unsure) return { error: UNREAD, status: 503 };
    return { error: `no ranking of ${game} had been saved before ${cutoffText} on this draw's day (${savedText(start)}), so it is left out`, code: 'before-first', status: 404 };
  }
  const view = workbench.drawView(dirs, game, date, 'top', hist, report, extraOf(game), now);
  if (view.error) return view;
  const { allowed, comingAllowed } = allowedFor(dirs, game, files[at].file, date, hist, report, at === 0);
  onlyTop(view, allowed, comingAllowed);
  const { file, ...out } = view;
  if (out.prev && !start.counts(out.prev)) out.prev = null;
  return { ...out, scope: plainScope(out.scope) };
}

// --- GET /api/lottery/:game ------------------------------------------------------------------------------------------
// The admin history with the scope Top (workbench.historyView) over the draws from the game's first saved ranking on;
// plus the whole history's totals over the same rows (total) and the line on the draws left out (leftOut).
function totalOf(game, months, kept) {
  const family = draws.familyOf(game);
  const sum = (key) => months.reduce((s, m) => s + (m[key] || 0), 0);
  let best = null; let multi = null;
  months.forEach((m) => {
    best = draws.betterBest(family, best, m.best);
    if (m.multi !== undefined && m.multi !== null) multi = Math.max(multi === null ? 0 : multi, m.multi);
  });
  const out = { from: kept[kept.length - 1].date, to: kept[0].date, draws: sum('draws'), counted: sum('counted'), rows: sum('rows'), tickets: sum('tickets'), best };
  if (draws.MONEY_GAMES.includes(game)) { out.profit = round2(sum('profit')); out.stake = round2(sum('stake')); }
  if (multi !== null) out.multi = multi;
  return out;
}
// unavailable: a version that may be the game's first could not be read (start.unsure): the earlier draws are not shown
// just now, and not said to be left out
function leftOutOf(game, start, files, kept) {
  const n = files.length - kept.length;
  const first = kept.length ? kept[kept.length - 1].date : null;
  const is = n === 1 ? 'is' : 'are';
  if (n && start.unsure) {
    return { draws: n, firstSaved: null, firstCounted: first, unavailable: true,
      text: `${plural(n, 'earlier draw')} ${is} not shown just now: a saved ranking that may count ${n === 1 ? 'it' : 'them'} could not be read; it is read again shortly.` };
  }
  const text = n
    ? `${plural(n, 'earlier draw')} ${is} left out: no ranking of ${game} had been saved before ${cutoffText} on ${n === 1 ? 'its' : 'their'} day (${savedText(start)}${first ? `; the first counted draw is ${first}` : ''}).`
    : `Every draw since the first saved ranking of ${game} (${start.day}).`;
  return { draws: n, firstSaved: start.day, firstCounted: first, text };
}
function historyOf(dirs, game, hist, month, now = new Date()) {
  const dir = path.join(dirs.databaseDir, game);
  const files = workbench.dayFilesOf(dir);
  if (!files.length) return { error: 'no stored draw for this game yet', status: 404 };
  const start = startOf(hist, game);
  if (!start) return noStart(hist, game);
  const kept = files.filter((f) => start.counts(f.date));
  const leftOut = leftOutOf(game, start, files, kept);
  if (!kept.length) {
    if (month !== null) return { error: 'no counted draw in that month', status: 404 };
    return { game, ...extraOf(game), builtAt: isoNow(now), scope: 'top', topN: TOP_N, shape: draws.shapeOf(game), draws: 0, counted: 0, month: null, months: [],
      ranking: { first: savedOf(hist.first), firstOfGame: start.first, error: hist.error ? READ_ERROR : null, versions: [], metrics: {} }, total: null, leftOut };
  }
  const stamp = [histStamp(hist), ...kept.map((f) => `${f.file}@${stampOf(path.join(dir, f.file))}`)].join('|');
  return remember(`history:${dir}:${month || ''}`, stamp, () => {
    const view = workbench.historyView(dirs, game, 'top', hist, extraOf(game), month, now, { include: start.counts });
    if (view.error) return view.status === 404 && month !== null ? { error: 'no counted draw in that month', status: 404 } : view;
    checkHistory(dirs, game, view, hist, kept);
    return { ...view, ranking: { ...view.ranking, first: savedOf(view.ranking.first), firstOfGame: savedOf(view.ranking.firstOfGame), error: view.ranking.error ? READ_ERROR : null },
      total: totalOf(game, view.months, kept), leftOut };
  });
}

// --- GET /api/lottery ------------------------------------------------------------------------------------------------
// One game's card: the coming draw - its date on the game's own schedule (no time of day: the records hold the date
// only), today's five rows in their order with their tickets - and the last scored draw: its result, the five ranked
// before it, each row's figures (draws.dayFigures over that row alone) and the five's together. past: the coming
// draw's date has gone and no newer run has scored it yet.
const byRank = (a, b) => (a.rank === null ? 99 : a.rank) - (b.rank === null ? 99 : b.rank);
function rowFigures(game, day, name) {
  const { rows, best, ...f } = draws.dayFigures(game, day, (n) => n === name);
  if (!best) return { ...f, best: null };
  const { by, ...b } = best;
  return { ...f, best: b };
}
function cardOf(dirs, game, hist, report, now) {
  const files = workbench.dayFilesOf(path.join(dirs.databaseDir, game));
  const base = { game, unit: admin.UNIT[game] };
  if (!files.length) return { ...base, empty: 'no stored draw yet' };
  const view = workbench.drawView(dirs, game, files[0].date, 'top', hist, report, {}, now);
  if (view.error) return { ...base, empty: view.error };
  const { day, allowed, comingAllowed } = allowedFor(dirs, game, files[0].file, files[0].date, hist, report, true);
  onlyTop(view, allowed, comingAllowed);
  const start = startOf(hist, game);
  const c = view.coming;
  const last = { date: view.date, weekday: view.weekday, result: view.result };
  if (start && start.counts(view.date)) {
    Object.assign(last, { scope: plainScope(view.scope), figures: view.figures,
      rows: view.rows.slice().sort(byRank).map((r) => ({ name: r.name, rank: r.rank, ...rowFigures(game, day, r.name) })) });
  } else if (unreadStart(hist, start)) {
    // the history is not whole here: not "left out", nor "not saved yet" - the rankings could not be read just now
    last.none = { code: 'unavailable', text: 'the saved rankings could not be read just now' };
  } else {
    last.leftOut = start ? `no ranking of ${game} had been saved before ${cutoffText} that day (${savedText(start)})` : 'no ranking of this game has been saved yet';
  }
  return {
    ...base, shape: view.shape,
    next: { date: c.date, weekday: c.date ? draws.weekdayOf(c.date) : null, time: null, past: Boolean(c.date && c.date < brussels.dayOf(now.getTime())), madeAt: c.madeAt },
    // every coming row is one of today's five: their `top` flag (the admin's All marks them) is left out
    coming: { scope: c.scope, rows: c.rows.slice().sort(byRank).map(({ top, ...r }) => r) },
    multiPick: view.multiPick,
    last,
  };
}
function overviewOf(dirs, hist, report, now = new Date()) {
  const stamp = [histStamp(hist), stampOf(path.join(dirs.databaseDir, 'modelPerformance.json')), brussels.dayOf(now.getTime()),
    ...GAMES.map((g) => { const f = workbench.dayFilesOf(path.join(dirs.databaseDir, g)); return f.length ? `${f.length}:${f[0].file}@${stampOf(path.join(dirs.databaseDir, g, f[0].file))}` : '-'; })].join('|');
  return remember(`overview:${dirs.databaseDir}`, stamp, () => ({
    builtAt: isoNow(now), topN: TOP_N, firstSaved: savedOf(hist.first),
    games: GAMES.map((g) => cardOf(dirs, g, hist, report, now)),
  }));
}

// --- GET /api/home ---------------------------------------------------------------------------------------------------
// Lottery: per game the next draw, today's top pick (the first of today's five, its tickets) and the last draw's
// figures over the five ranked before it. Crypto and Shares: api.js marketView's headline - following the plan (the
// day and week positions together, live days only) against buying everything over the same days, its window (since,
// through, days) - and today's positions: to buy (now or at the open) and held.
function marketHeadline(dirs, market, now) {
  const record = recordOf(dirs.marketsDir, market);
  if (!record) return { market, empty: 'no record yet' };
  const v = api.marketView(market, record, recordOf(dirs.marketsDir, markets.WEEK_GAMES[market]), now);
  if (!v) return { market, empty: 'no record yet' };
  const s = v.status;
  return {
    market, title: v.title, unit: v.unit,
    plan: { since: s.since, through: v.daily.length ? v.daily[v.daily.length - 1].date : null, days: s.days, positions: s.positions,
      total: s.liveTotal, market: s.liveMarket, ahead: s.ahead, paperNote: s.paperNote, entryNote: s.entryNote },
    today: { date: v.day ? v.day.date : null, state: v.day ? v.day.stateText : null, toBuy: s.openToBuy, held: s.openHeld },
  };
}
function homeOf(dirs, hist, report, now = new Date()) {
  const overview = overviewOf(dirs, hist, report, now);
  return {
    builtAt: isoNow(now),
    lottery: overview.games.map((g) => (g.empty ? g : {
      game: g.game, unit: g.unit, shape: g.shape, next: g.next,
      top: g.coming.rows[0] || null, rows: g.coming.scope.rows, ranking: g.coming.scope.ranking, none: g.coming.scope.none,
      last: { date: g.last.date, rows: g.last.rows ? g.last.rows.length : 0, figures: g.last.figures || null, none: g.last.none || (g.last.scope ? g.last.scope.none : null), leftOut: g.last.leftOut || null },
    })),
    ...Object.fromEntries(MARKETS.map((m) => [m, marketHeadline(dirs, m, now)])),
  };
}

// --- routes ----------------------------------------------------------------------------------------------------------
// options: as admin.install (root, dataDir or databaseDir / marketsDir), rankCacheFile (the rank history's cache: the
// server's gitignored config/ by default, the file admin.js uses too)
function install(app, options = {}) {
  const dirs = admin.resolveDirs(options);
  const rankCacheFile = options.rankCacheFile || path.join(__dirname, 'config', 'rankhistory-cache.json');
  const json = (res, body, status = 200) => { res.set('Cache-Control', 'no-store'); res.status(status).json(body); };
  const answerLater = (res, build) => Promise.resolve().then(build).then(
    (body) => (body && body.error ? json(res, { error: body.error, ...(body.code ? { code: body.code } : {}) }, body.status || 404) : json(res, body)),
    (e) => {
      console.error(`[${new Date().toISOString()}] lottery.js:`, e);
      return json(res, { error: 'could not build this view' }, 500);
    });
  // auth.middleware answers a signed-out /api request with its 401 first; this one holds without it (open access is a
  // reader too: auth.middleware signs it in as the administrator)
  const signedIn = (req, res, next) => (req.user ? next() : json(res, { error: 'login required' }, 401));
  const history = () => rankhistory.load(dirs.root, { cacheFile: rankCacheFile });
  const report = () => rankings.loadReport(dirs.databaseDir);
  const gameOf = (req) => (GAMES.includes(req.params.game) ? req.params.game : null);

  app.use('/api/lottery', signedIn);
  app.use('/api/home', signedIn);
  app.get('/api/lottery', (req, res) => answerLater(res, async () => overviewOf(dirs, await history(), report())));
  app.get('/api/lottery/:game', (req, res) => answerLater(res, async () => {
    const game = gameOf(req);
    if (!game) return { error: 'unknown game', status: 404 };
    const month = req.query.month === undefined || req.query.month === '' ? null : String(req.query.month);
    if (month !== null && !MONTH.test(month)) return { error: 'the month is YYYY-MM', status: 400 };
    return historyOf(dirs, game, await history(), month);
  }));
  app.get('/api/lottery/:game/:date', (req, res) => answerLater(res, async () => {
    const game = gameOf(req);
    if (!game) return { error: 'unknown game', status: 404 };
    const date = strictDay(req.params.date);
    if (!date) return { error: 'the date is YYYY-MM-DD', status: 400 };
    return drawOf(dirs, game, date, await history(), report());
  }));
  app.get('/api/home', (req, res) => answerLater(res, async () => homeOf(dirs, await history(), report())));
  // anything else under /api/lottery and /api/home is a JSON 404, never a page
  app.use('/api/lottery', (req, res) => json(res, { error: 'not found' }, 404));
  app.use('/api/home', (req, res) => json(res, { error: 'not found' }, 404));
}

module.exports = { GAMES, strictDay, startOf, onlyTop, drawOf, historyOf, overviewOf, cardOf, homeOf, marketHeadline, install };
