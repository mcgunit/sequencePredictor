// workbench.js - the admin section's Workbench (22 Oct 2026, admin batch 4: the owner's decision of 21 Oct to replace
// the old pages with the app, everything detailed in the admin section). Read-only JSON under /api/admin, behind
// admin.js's guard - admin.install mounts these routes before its catch-all 404:
//
//   GET /api/admin/history/:game                        one game's draws by month, newest first (the old
//                                                       /database/:folder page), ?scope=all|top
//   GET /api/admin/draw/:game/:date                     one draw (the old /database/:folder/:file page): the drawn
//                                                       row, every row's tickets scored, the vote under them, a
//                                                       market day's moves; the coming draw on the newest day
//   GET /api/admin/market/:market/workbench             a market's workbench in sections (the old /markets/:market
//                                                       page, minus what /app and the other admin pages show)
//   GET /api/admin/market/:market/workbench/books/:rule        one paper book's cumulative series per model
//   GET /api/admin/market/:market/workbench/instrument/:symbol one instrument's price chart: closes, moves, every
//                                                              model's bins and the next calls
//
// The owner's rule, no hindsight: Top on a past draw is the five rows ranked first BEFORE that draw (rankhistory.js,
// production's git history of the History ranking), never today's five; a draw before the first saved ranking (25 Aug
// 2026) says so and has no Top rows. The coming draw's Top is today's five (ranking.js usersTop: the five /lottery
// shows a user). Every figure is over the rows the scope keeps and says how many (draws.js builds them with a `keep`
// filter, so the users' Lottery can reuse them and never send a hidden row).
//
// The scoring is draws.js's, the market figures markets.js's (describeMarket, pickPlan, pickWeek, the close and open
// moments); this module reads the files, picks the rows and shapes the answers. Each source file is parsed once per
// change (keyed by modification time and size); a file caught half written keeps its last good read.
'use strict';

const fs = require('fs');
const path = require('path');
const draws = require('./draws');
const markets = require('./markets');
const rankings = require('./ranking');
const rankhistory = require('./rankhistory');

const SCOPES = ['all', 'top'];
const MARKETS = Object.keys(markets.MARKETS);                       // crypto, shares
const BOOK_RULES = ['daily', 'hold', 'open', 'daily_ls', 'hold_ls', 'open_ls'];
const DATE = /^(\d{4})-(\d{1,2})-(\d{1,2})$/;
const SYMBOL = /^[A-Za-z0-9._^-]{1,20}$/;
const DAYS_IN_DETAIL = 30;                                          // MarketSettle's day records: the newest 30 settled days

// --- small helpers ---------------------------------------------------------------------------------------------
const num = (x) => { const v = Number(x); return x === null || x === undefined || x === '' || typeof x === 'boolean' || !Number.isFinite(v) ? null : v; };
const round = (digits) => (x) => (num(x) === null ? null : Math.round(num(x) * 10 ** digits) / 10 ** digits);
const r2 = round(2);
const r4 = round(4);
const r6 = round(6);
const list = (v) => (Array.isArray(v) ? v : []);
const isoMs = (ms) => (Number.isFinite(ms) ? new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null);
const isoDate = (d) => (d instanceof Date && !Number.isNaN(d.getTime()) ? isoMs(d.getTime()) : null);
function isoUtc(value) {
  if (value === null || value === undefined || value === '') return null;
  const text = String(value);
  const date = new Date(/[zZ]$|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isNaN(date.getTime()) ? null : date.toISOString().replace(/\.\d{3}Z$/, 'Z');
}
const nameOf = (m) => (m && m.name) || 'not known';
const rowsOf = (v) => list(v).filter((m) => m && typeof m === 'object' && Array.isArray(m.predictions));
const namesOf = (rows) => [...new Set(rowsOf(rows).map(nameOf))];
// "2026-9-5" or "2026-09-05" -> "2026-09-05", or null
function isoDay(text) {
  const m = DATE.exec(String(text || ''));
  if (!m) return null;
  const iso = `${m[1]}-${m[2].padStart(2, '0')}-${m[3].padStart(2, '0')}`;
  const d = new Date(`${iso}T00:00:00Z`);
  return !Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === iso ? iso : null;
}

// --- reading, once per change ------------------------------------------------------------------------------------
const fileCache = new Map();   // key -> {stamp, value}
function stampOf(file) { try { const s = fs.statSync(file); return `${s.mtimeMs}:${s.size}`; } catch (e) { return null; } }
// build()'s value, rebuilt when the files change; a build that throws (a file caught half written) keeps the last
// good value, or null
function cached(key, files, build) {
  const stamps = files.map(stampOf);
  if (stamps[0] === null) { fileCache.delete(key); return null; }
  const stamp = stamps.join('|');
  const hit = fileCache.get(key);
  if (hit && hit.stamp === stamp) return hit.value;
  let value;
  try { value = build(); } catch (e) { value = hit ? hit.value : null; }
  fileCache.set(key, { stamp, value });
  return value;
}
const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf-8'));
// a day file as the history reads it: the drawn row and the scored rows (the raw probability arrays left out)
const readDay = (file) => cached(`day:${file}`, [file], () => { const d = readJson(file); return { realResult: list(d.realResult), currentPrediction: rowsOf(d.currentPrediction) }; });
// a day file in full, for the view of one draw
const readFull = (file) => cached(`full:${file}`, [file], () => {
  const d = readJson(file);
  return { realResult: list(d.realResult), currentPrediction: rowsOf(d.currentPrediction), newPrediction: rowsOf(d.newPrediction),
    currentNumberFrequency: d.currentNumberFrequency, numberFrequency: d.numberFrequency, madeAt: typeof d.madeAt === 'string' ? d.madeAt : null };
});
// a game's day files, newest first: [{file, date}]
function dayFilesOf(dir) {
  let names;
  try { names = fs.readdirSync(dir); } catch (e) { return []; }
  return names.map((file) => ({ file, date: draws.isoOfFile(file) })).filter((f) => f.date).sort((a, b) => (a.date < b.date ? 1 : (a.date > b.date ? -1 : 0)));
}
// a market game's record (data/markets/<game>.json) described, and its instruments in slot order
function marketView(marketsDir, game) {
  const file = path.join(marketsDir, `${game}.json`);
  return cached(`market:${file}`, [file], () => { const r = markets.loadMarket(marketsDir, game); return r ? markets.describeMarket(r) : null; });
}
const slotsOf = (view) => (view ? view.instruments.slice().sort((a, b) => (a.position || 0) - (b.position || 0)).map((i) => i.symbol) : null);
// an instrument's close before a day and on it (null where its closes do not reach): the close before is the base a
// bin's edges turn into prices from
function closeAround(view, symbol, date) {
  const inst = view.instruments.find((i) => i.symbol === symbol);
  const at = inst ? inst.closes.findIndex((p) => p[0] === date) : -1;
  return { before: at > 0 ? inst.closes[at - 1][1] : null, on: at >= 0 ? inst.closes[at][1] : null };
}
// the workbench's view of a market: its record with the week game, the rows report and the regime log
function workbenchSource(dirs, market) {
  const week = markets.WEEK_GAMES[market];
  const files = [path.join(dirs.marketsDir, `${market}.json`), path.join(dirs.marketsDir, `${week}.json`),
    path.join(dirs.controlsDir, 'markets', `${market}-rows.json`), path.join(dirs.marketsDir, `${market}-regimes.json`)];
  return cached(`workbench:${files[0]}`, files, () => {
    const record = markets.loadMarket(dirs.marketsDir, market);
    if (!record) return null;
    const weekRecord = markets.loadMarket(dirs.marketsDir, week);
    return markets.describeMarket(record, { rows: markets.loadRows(dirs.controlsDir, market), regimes: markets.loadRegimes(dirs.marketsDir, market),
      week: weekRecord ? markets.describeMarket(weekRecord) : null });
  });
}

// --- the ranking a draw is read against ----------------------------------------------------------------------------
// what a version of the History ranking is, for the line over a table: its commit, its generation, its metric
const versionOut = (r) => ({ committedAt: r.version.committedAt, generatedAt: r.version.generatedAt, metric: r.metric, metricText: r.metricText, minDraws: r.minDraws });
const cutoffRule = () => ({ hour: rankhistory.CUTOFF_HOUR, zone: 'Europe/Brussels', weekGames: rankhistory.WEEK_GAMES,
  text: `the newest ranking saved before ${rankhistory.CUTOFF_HOUR}:00 Brussels time on the draw's day (the week games: on the week's Monday)` });
// the scope of a past draw: {keep, scope line} from the point-in-time top (rankhistory topAt)
function scopeOfDraw(scope, t, presentCount) {
  const top = t.top ? new Set(t.top) : null;
  const keep = scope === 'top' ? (top ? (n) => top.has(n) : null) : () => true;
  const line = { scope, of: presentCount, top: t.top || [], cutoff: t.cutoff || null,
    ranking: t.ranking ? versionOut(t.ranking) : null, none: t.none ? { code: t.none, text: t.text } : null };
  return { keep, top, line };
}

// --- GET /api/admin/history/:game ------------------------------------------------------------------------------------
// The game's months newest first with their totals, and the draws of one month (?month=YYYY-MM, by default the newest:
// the page opens on it, the others load when opened - keno's 348 days at once are about 90 KB): per draw the date, the
// drawn row, the rows and tickets in scope, the money games' profit and stake, the best ticket (draws.dayFigures).
// A month's totals are over the same rows as its draws. With Top a draw's rows are its point-in-time top five; a draw
// without a recorded ranking has no figures and says why (ranking.none). A draw names the ranking version it was read
// against by its index in `ranking.versions` (the versions the sent draws use). include (optional, date -> boolean):
// the draws this history holds - the users' Lottery (lottery.js) leaves out the draws before the game's first saved
// ranking, so they are not read, listed or counted at all.
function historyView(dirs, game, scope, hist, extra = {}, month = null, now = new Date(), { include = null } = {}) {
  const dir = path.join(dirs.databaseDir, game);
  const files = dayFilesOf(dir).filter((f) => !include || include(f.date));
  if (!files.length) return { error: include ? 'no draw of this game in this history yet' : 'no stored draw for this game', status: 404 };
  const entries = files.map(({ file, date }) => {
    const day = readDay(path.join(dir, file)) || { realResult: [], currentPrediction: [] };
    const t = hist.topAt(game, date, namesOf(day.currentPrediction));
    return { date, day, keep: scopeOfDraw(scope, t, 0).keep, ranking: t };
  });
  const months = draws.historyMonths(game, entries);
  const open = month === null ? months[0].month : month;
  if (!months.some((m) => m.month === open)) return { error: 'no stored draw in that month', status: 404 };
  const versions = [];
  const metrics = {};
  const index = new Map();
  const ref = (t) => {
    if (t.none) return { none: t.none };
    const sha = t.ranking.version.sha;
    if (!index.has(sha)) {
      index.set(sha, versions.length);
      const v = versionOut(t.ranking);
      metrics[v.metric] = v.metricText;
      delete v.metricText;
      versions.push(v);
    }
    return { v: index.get(sha) };
  };
  return {
    game, ...extra, builtAt: isoDate(now), scope, topN: rankings.TOP_MODELS_SHOWN,
    // a market game's slots name its instruments in order, as on the draw view (the history's result lines read by slot)
    shape: draws.shapeOf(game, { slots: draws.MARKET_GAMES.includes(game) ? slotsOf(marketView(dirs.marketsDir, game)) : null }),
    draws: entries.length, counted: months.reduce((s, m) => s + m.counted, 0), month: open,
    months: months.map((m) => {
      const { days, ...totals } = m;
      return m.month === open ? { ...totals, days: days.map((d) => ({ ...d, ranking: ref(d.ranking) })) } : { ...totals, days: null };
    }),
    // first: the first saved ranking of any game (25 Aug 2026); firstOfGame: the first that ranks this game (a game can
    // have entered it later); error: a version git could not read (the draws it decides have no Top, ranking.none)
    ranking: { first: hist.first, firstOfGame: hist.firstOf ? hist.firstOf(game) : null, error: hist.error, cutoff: cutoffRule(), versions, metrics },
  };
}

// --- GET /api/admin/draw/:game/:date ---------------------------------------------------------------------------------
// One draw. rows: the scored table in the file's order, each row with its tickets (draws.scoreRow), its position in
// the ranking the draw is read against (rank, null when not ranked then) and whether it was in that top five; figures:
// the history's day line over the same rows; votes: the vote of the shown rows' first tickets (per slot for the
// positional games) and, with All, the file's own weighted vote of every row; market: a market day's moves with the
// close before it (a bin's price range) and every shown row's exact / direction / money and, for a sized row, its size
// per slot (the newest 30 settled days; older days have bins and hits only). coming: the
// newest day's next prediction with today's ranking; an older day names the draw its next prediction was scored in.
function drawView(dirs, game, date, scope, hist, report, extra = {}, now = new Date()) {
  const dir = path.join(dirs.databaseDir, game);
  const files = dayFilesOf(dir);
  const at = files.findIndex((f) => f.date === date);
  if (at < 0) return { error: 'no draw of this game on that date', status: 404 };
  const day = readFull(path.join(dir, files[at].file));
  if (!day) return { error: 'this draw\'s file could not be read', status: 500 };
  const market = draws.MARKET_GAMES.includes(game) ? marketView(dirs.marketsDir, game) : null;
  const slots = slotsOf(market);
  const split = draws.splitRealResult(day.realResult, game);
  const present = namesOf(day.currentPrediction);
  const t = hist.topAt(game, date, present);
  const { keep, top, line } = scopeOfDraw(scope, t, present.length);
  const order = t.ranking ? t.ranking.names : null;
  const shown = keep ? day.currentPrediction.filter((m) => keep(nameOf(m))) : [];
  const rankOf = (names, name) => { const i = names ? names.indexOf(name) : -1; return i >= 0 ? i + 1 : null; };
  const rows = shown.map((m) => ({ ...draws.scoreRow(game, m, day.realResult, split), rank: rankOf(order, nameOf(m)), top: Boolean(top && top.has(nameOf(m))) }));
  const figures = keep ? draws.dayFigures(game, day, keep) : null;
  // the stored vote: this file's, or - missing in 35 old vikinglotto files - the previous file's numberFrequency, the same vote
  let stored = day.currentNumberFrequency && typeof day.currentNumberFrequency === 'object' && Object.keys(day.currentNumberFrequency).length ? day.currentNumberFrequency : null;
  let from = 'file';
  if (!stored && files[at + 1]) {
    const prev = readFull(path.join(dir, files[at + 1].file));
    if (prev && prev.numberFrequency && typeof prev.numberFrequency === 'object' && Object.keys(prev.numberFrequency).length) { stored = prev.numberFrequency; from = 'previous file'; }
  }
  const out = {
    game, ...extra, builtAt: isoDate(now), date, weekday: draws.weekdayOf(date), file: files[at].file,
    prev: files[at + 1] ? files[at + 1].date : null, next: at > 0 ? files[at - 1].date : null, newest: at === 0,
    shape: draws.shapeOf(game, { slots }), result: draws.resultOf(game, day.realResult),
    scope: { ...line, rows: shown.length, tickets: figures ? figures.tickets : 0 },
    rows, figures,
    multiPick: day.currentPrediction.concat(day.newPrediction).some((m) => Array.isArray(m.multiPick) && m.multiPick.length) ? draws.multiPickLegend(game) : null,
    votes: draws.votesOf(game, shown, day.realResult, scope === 'all' ? stored : null, from),
    market: null, coming: null,
  };
  if (market) {
    const d = market.days.find((x) => x.date === date);
    if (d) {
      const kept = new Set(shown.map(nameOf));
      // best: the day's best row if it is shown; exactMean: the mean exact rate over every row of the day, so only with All
      out.market = { k: market.k, chance: market.chance, best: d.best && kept.has(d.best) ? d.best : null, exactMean: scope === 'all' ? r4(d.exactMean) : null,
        instruments: d.instruments.map((i) => ({ symbol: i.symbol, ret: r6(i.ret), bin: i.bin, edges: i.edges.map(r6), before: closeAround(market, i.symbol, d.date).before })),
        models: Object.fromEntries(d.models.filter((m) => kept.has(m.name)).map((m) => [m.name, { exact: m.exact, adjacent: m.adjacent, direction: m.direction,
          positions: m.positions, trades: m.trades, pnl: r6(m.pnl), pnlCash: r2(m.pnlCash), sizes: m.sizes }])) };
    }
  }
  if (at === 0) {
    // the coming draw: today's ranking (the working tree's report), as /lottery shows it
    const nextRows = day.newPrediction;
    const nextNames = namesOf(nextRows);
    const today = rankings.rankingOf(report, game);
    const todayTop = rankings.usersTop(report, game, nextNames);
    const info = report && report.games ? report.games[game] : null;
    const keepNext = scope === 'top' ? new Set(todayTop) : null;
    const comingRows = nextRows.filter((m) => !keepNext || keepNext.has(nameOf(m)));
    const empty = draws.splitRealResult([], game);
    out.coming = {
      date: draws.nextDrawDay(files.map((f) => f.file), files[at].file), madeAt: isoUtc(day.madeAt),
      scope: { scope, rows: comingRows.length, of: nextNames.length, top: todayTop,
        ranking: today ? { today: true, generatedAt: isoUtc(report.generatedAt), metric: info ? info.metric : null, metricText: today.metric, minDraws: info ? num(info.minDrawsForRanking) : null } : null,
        none: scope === 'top' && !todayTop.length ? { code: today ? 'no-ranked-row' : 'not-ranked', text: today ? 'no row of this table is in today\'s ranking' : 'the game is not in today\'s ranking' } : null },
      rows: comingRows.map((m) => ({ ...draws.scoreRow(game, m, [], empty, split.mains), rank: rankOf(today ? today.names : null, nameOf(m)), top: todayTop.includes(nameOf(m)) })),
      votes: draws.votesOf(game, comingRows, [], scope === 'all' ? day.numberFrequency : null, 'file', split.mains),
    };
  } else {
    out.scoredIn = files[at - 1].date;
  }
  return out;
}

// --- the market workbench ------------------------------------------------------------------------------------------------
// the books' definition of an up call: a bin in the upper half, or a sized row's size above 0 (markets.js, api.js)
const isUp = (p, k) => (p.size !== null && p.size !== undefined ? p.size > 0 : (p.bin !== null && p.bin >= k / 2));
// the word of a call that is not up; a sized row that is not up is flat whatever its bin
const callWord = (p, k) => (p.size !== null && p.size !== undefined ? 'flat' : (p.bin !== null && p.bin < k / 2 - 1 ? 'down' : 'flat'));
// a book against buying everything under the same rule: the difference of the two figures as printed (to the cent)
const aheadOf = (book, bench) => (num(book) === null || num(bench) === null ? null : r2(r2(book) - r2(bench)));
const lastTotal = (points) => (points && points.length ? points[points.length - 1].total : null);
const ticketModels = (instruments) => { const names = new Set(); instruments.forEach((i) => i.next.forEach((p) => names.add(p.model))); return names.size; };
// one call as the plan tables read it: the bin, a sized row's size, up / flat / down, and its band as prices and as
// moves from the base close
function callOf(call, base, k) {
  if (!call) return null;
  const move = (v) => (v === null || base === null || base <= 0 ? null : r4(v / base - 1));
  return { bin: call.bin, size: call.size, up: isUp(call, k), word: isUp(call, k) ? 'up' : callWord(call, k), price: call.price,
    band: { low: call.low, high: call.high }, move: { low: move(call.low), high: move(call.high) } };
}
// what a followed record is made of: back-built days before the first live one, live days, late days after it
function recordOf(f) {
  if (!f || !f.points.length) return null;
  const before = f.points.filter((p) => !p.live && (!f.firstLive || p.date < f.firstLive));
  const late = f.points.filter((p) => !p.live && f.firstLive && p.date > f.firstLive);
  const live = f.points.filter((p) => p.live);
  return { firstLive: f.firstLive, backBuilt: before.length, live: live.length, late: late.length, lastLive: live.length ? live[live.length - 1].date : null };
}
// The picked model's own book (a younger row's book covers fewer days than the market's record): its settled days,
// first and last, buying everything under the same rule over those days, and what those days are made of in the
// followed record (back-built, live, late). The followed record's own figures are soFarOf's.
function ownBookOf(model, bench, f) {
  const first = model.firstDay || null;
  const last = model.lastDay || null;
  const inside = (date) => (!first || date >= first) && (!last || date <= last);
  const at = bench.findIndex((p) => inside(p.date));
  let end = at;
  while (at >= 0 && end + 1 < bench.length && inside(bench[end + 1].date)) end += 1;
  // the running totals' difference, so a book over every day ends on the market's own total
  const market = at < 0 ? null : bench[end].total - (at > 0 ? bench[at - 1].total : 0);
  return { days: model.days, firstDay: first, lastDay: last, market, record: f ? recordOf({ ...f, points: f.points.filter((p) => inside(p.date)) }) : null };
}
// "Following this card so far": what doing what the card said made, live days only, against buying everything
function soFarOf(f) {
  if (!f) return null;
  const rec = recordOf(f);
  return { liveDays: f.liveDays, positions: f.livePositions, firstLive: f.firstLive, lastLive: rec ? rec.lastLive : null,
    total: r2(f.liveTotal), market: r2(f.liveMarket), ahead: aheadOf(f.liveTotal, f.liveMarket) };
}

// Today's plan (C2): the model followed and why, its book and the record behind it, following the card so far, and per
// active instrument its call, band, the order it means now and how many models call it up. order: 'no-call' (the
// model has no call for it), 'no-buy' (not up), 'too-late' (the day has closed), 'buy-now' (crypto, or a share after
// the open), 'buy-at-open' (a share before the open); the sell is at the close, whatever the price.
function planOf(market, view, now) {
  const pick = markets.pickPlan(view, market);
  const day = markets.nextGameDay(market, view.madeOn);
  if (!day || !pick.model) return null;
  const t = view.trading;
  const bench = t ? (market === 'crypto' ? t.benchmark : (t.open ? t.open.benchmark : [])) : [];
  const closeAt = markets.closeMoment(market, day);
  const openAt = markets.openMoment(market, day);
  const closed = Boolean(closeAt && now >= closeAt);
  const f = t ? t.followed : null;
  const own = ownBookOf(pick.model, bench, f);
  const rows = view.instruments.filter((i) => i.active).map((inst) => {
    const base = inst.nextBase !== null && inst.nextBase !== undefined ? inst.nextBase : inst.lastClose;
    const call = inst.next.find((p) => p.model === pick.model.name) || null;
    const c = callOf(call, base, view.k);
    let order;
    if (!c) order = 'no-call';
    else if (!c.up) order = 'no-buy';
    else if (closed) order = 'too-late';
    else if (market === 'crypto' || (openAt && now >= openAt)) order = 'buy-now';
    else order = 'buy-at-open';
    return { symbol: inst.symbol, base, call: c, order, up: inst.next.filter((p) => isUp(p, view.k)).length, of: inst.next.length };
  });
  return {
    // calendarKnown: a share day in a year NYSE has published its holidays for; otherwise the day is "normally" this one
    day, calendarKnown: market === 'crypto' || [...markets.NYSE_CLOSED].some((d) => d.slice(0, 4) === day.slice(0, 4)),
    madeOn: view.madeOn, madeAt: isoUtc(view.madeAt), openAt: isoDate(openAt), closeAt: isoDate(closeAt), closed, opened: Boolean(openAt && now >= openAt),
    status: markets.dayStatus(market, day, now),
    model: pick.model.name, byBook: pick.byBook, rule: pick.ruleName, minTrades: markets.MIN_PLAN_TRADES, models: ticketModels(view.instruments) || view.models.length,
    // the book, its days and its market are the picked model's own window; record: what those days are
    book: pick.byBook
      ? { total: r2(pick.model[pick.key]), positions: pick.model[pick.tradesKey], stake: t ? t.stake : null, settledDays: own.days, firstDay: own.firstDay, lastDay: own.lastDay,
        market: r2(own.market), ahead: aheadOf(pick.model[pick.key], own.market) }
      : { exact: r4(pick.model.exact), scoredDays: own.days, firstDay: own.firstDay, lastDay: own.lastDay },
    record: own.record, soFar: soFarOf(f), rows,
  };
}

// The week card (C3; "Week game" in the admin, not to be confused with the This week page): the week game's model and
// why, its weekly book, following the card so far, and per active instrument its call, the band at the week's close,
// how the week stands so far on the daily closes, and the order now. order: 'no-call', 'no-position' (not up),
// 'closed' (the week has closed), 'before-start' (a buy at the week's start), 'first-day' (a buy now, the week's
// first day), 'running' (the book's position is open; a buy now runs only to the week's close).
function weekOf(market, view, now) {
  const w = view.week;
  if (!w || !w.madeOn || markets.nextGameDay(market, w.madeOn) === null) return null;
  const first = markets.nextGameDay(market, w.madeOn);
  const endDay = markets.weekEnd(market, w.madeOn);
  const closeAt = endDay ? markets.closeMoment(market, endDay) : null;
  const openAt = first ? markets.openMoment(market, first) : null;
  const pick = markets.pickWeek(w, market);
  if (!pick.model || !endDay || !closeAt || !openAt) return null;
  const closed = now >= closeAt;
  const started = now >= openAt;
  const bench = w.trading ? (market === 'crypto' ? w.trading.benchmark : (w.trading.open ? w.trading.open.benchmark : [])) : [];
  const f = w.trading ? w.trading.followed : null;
  const own = ownBookOf(pick.model, bench, f);
  const rows = w.instruments.filter((i) => i.active).map((inst) => {
    const base = inst.nextBase !== null && inst.nextBase !== undefined ? inst.nextBase : inst.lastClose;
    const call = inst.next.find((p) => p.model === pick.model.name) || null;
    const c = callOf(call, base, w.k);
    const daily = view.instruments.find((i) => i.symbol === inst.symbol);
    const since = daily ? daily.closes.filter((p) => p[0] > w.madeOn) : [];
    let soFar = null;
    if (since.length && base) {
      const last = since[since.length - 1];
      soFar = { closes: since.length, last: last[1], lastDate: last[0], move: r4(last[1] / base - 1),
        where: !c ? null : (c.band.low !== null && last[1] < c.band.low ? 'below' : (c.band.high !== null && last[1] > c.band.high ? 'above' : 'inside')) };
    }
    let order;
    if (!c) order = 'no-call';
    else if (!c.up) order = 'no-position';
    else if (closed) order = 'closed';
    else if (!started) order = 'before-start';
    else if (now - openAt < 86400000) order = 'first-day';
    else order = 'running';
    return { symbol: inst.symbol, base, call: c, soFar, order, up: inst.next.filter((p) => isUp(p, w.k)).length, of: inst.next.length };
  });
  return {
    weekEnd: endDay, firstDay: first, madeOn: w.madeOn, madeAt: isoUtc(w.madeAt), openAt: isoDate(openAt), closeAt: isoDate(closeAt), started, closed,
    hoursLeft: closed ? 0 : r2((closeAt - now) / 3600000), status: markets.dayStatusWeek(now, openAt, closeAt), settledWeeks: w.scoredDays,
    model: pick.model.name, byBook: pick.byBook, rule: pick.ruleName, minTrades: markets.MIN_WEEK_TRADES, models: ticketModels(w.instruments) || w.models.length,
    book: pick.byBook
      ? { total: r2(pick.model[pick.key]), positions: pick.model[pick.tradesKey], stake: w.trading ? w.trading.stake : null, settledWeeks: own.days, firstDay: own.firstDay, lastDay: own.lastDay,
        market: r2(own.market), ahead: aheadOf(pick.model[pick.key], own.market) }
      : { exact: r4(pick.model.exact), settledWeeks: own.days, firstDay: own.firstDay, lastDay: own.lastDay },
    // the user pages follow the week with the guard from this week on (markets.GUARD_SHOWN_FROM); the card shows the plain call
    guard: { from: markets.GUARD_SHOWN_FROM, applies: first >= markets.GUARD_SHOWN_FROM },
    record: own.record, soFar: soFarOf(f), rows,
  };
}

// Following the plan (C5 - C7), one game: the live days (oldest first) with their row, positions and money against
// buying everything - the client adds the running totals -, and apart the days too late to follow and the back-built
// start, each with its totals (not in the live total).
const FOLLOWED_COLUMNS = ['date', 'model', 'pickedBy', 'positions', 'pnl', 'market'];
function followedOf(f, { rule, week }) {
  if (!f || !f.points.length) return null;
  // four decimals: the page adds the running totals, which then end on the record's own total as the old page's did
  // (it summed the unrounded results; cents would drift by one or two over a month)
  const pts = (arr) => arr.map((p) => [p.date, p.model, p.pickedBy, p.positions, r4(p.pnl), r4(p.market)]);
  const group = (arr) => ({ n: arr.length, total: r2(arr.reduce((s, p) => s + p.pnl, 0)), market: r2(arr.reduce((s, p) => s + (p.market || 0), 0)), points: pts(arr) });
  const live = f.points.filter((p) => p.live);
  return {
    rule, week, minTrades: f.minTrades, firstLive: f.firstLive, liveDays: f.liveDays, positions: f.livePositions,
    total: r2(f.liveTotal), market: r2(f.liveMarket), ahead: aheadOf(f.liveTotal, f.liveMarket),
    columns: FOLLOWED_COLUMNS, live: pts(live),
    late: group(f.points.filter((p) => !p.live && f.firstLive && p.date > f.firstLive)),
    backBuilt: group(f.points.filter((p) => !p.live && (!f.firstLive || p.date < f.firstLive))),
  };
}

// The paper books' headline (C8): per rule the best book among the rows with markets.MIN_RANK_DAYS settled days (a
// sized row never shorts, so it cannot head the long-and-short book) against the market under the same rule. The
// series are behind /books/:rule.
function booksOf(view) {
  const t = view.trading;
  if (!t || !t.dates.length) return null;
  const sized = new Set(view.instruments.flatMap((i) => i.next.filter((p) => p.size !== null).map((p) => p.model)));
  const seasoned = view.models.filter((m) => m.days !== null && m.days >= markets.MIN_RANK_DAYS);
  const bestBy = (key, skip = () => false) => {
    const m = seasoned.filter((x) => x[key] !== null && !skip(x)).sort((a, b) => b[key] - a[key])[0];
    return m ? { model: m.name, total: r2(m[key]) } : null;
  };
  const rules = ['daily', 'hold'];
  if (t.open) rules.push('open');
  if (t.short && t.short.daily) { rules.push('daily_ls', 'hold_ls'); if (t.open && t.short.open) rules.push('open_ls'); }
  return {
    days: t.dates.length, first: t.dates[0], last: t.dates[t.dates.length - 1], stake: t.stake, feePerLeg: t.feePerLeg, currency: t.currency,
    minDays: markets.MIN_RANK_DAYS, seasonedRows: seasoned.length, rules,
    ruleText: { daily: t.rule, hold: t.holdRule, open: t.openRule || null, short: t.shortRule || null },
    headline: {
      daily: { best: bestBy('pnlCash'), market: r2(lastTotal(t.benchmark)) },
      hold: { best: bestBy('holdTotal'), market: r2(lastTotal(t.hold.benchmark)) },
      open: t.open ? { best: bestBy('openTotal'), market: r2(lastTotal(t.open.benchmark)) } : null,
      longShort: t.short && t.short.daily ? { best: bestBy('shortTotal', (m) => sized.has(m.name)), bought: r2(lastTotal(t.benchmark)), sold: r2(lastTotal(t.short.daily.sold)) } : null,
    },
    best: view.best,
  };
}
// One book's series (C8 chart): every model's cumulative money aligned to the book's dates (null before its first day),
// the market under the same rule, and for a long-and-short book what selling everything gave.
function bookSeries(view, rule) {
  const t = view.trading;
  if (!t || !t.dates.length) return null;
  const base = rule.replace(/_ls$/, '');
  const shorts = rule.endsWith('_ls');
  const book = base === 'daily' ? t : (base === 'hold' ? t.hold : t.open);
  if (!book || (shorts && !(t.short && t.short[base]))) return null;
  const aligned = (points) => { const by = Object.fromEntries((points || []).map((p) => [p.date, p.total])); return t.dates.map((d) => (by[d] === undefined ? null : r2(by[d]))); };
  const source = shorts ? t.short[base].models : book.models;
  const order = view.models.map((m) => m.name).filter((m) => source[m]);
  Object.keys(source).forEach((m) => { if (!order.includes(m)) order.push(m); });
  return { rule, dates: t.dates, currency: t.currency, stake: t.stake, best: view.best && order.includes(view.best) ? view.best : (order[0] || null),
    market: aligned(book.benchmark), sold: shorts ? aligned(t.short[base].sold) : null, order, series: Object.fromEntries(order.map((m) => [m, aligned(source[m])])) };
}

// One instrument's chart (C9): the closes a year back, the move of each day, the bin edges and every model's bin from
// the first tracked day on (`from`, an index into dates), and the next calls with their prices. A bin's interval is
// [edges[bin - 1], edges[bin]] (open at the ends); its price: the previous close x exp(the interval's middle; an end
// bin's edge moved out by half the neighbouring bin's width) - markets.js marketRepresentative.
function instrumentOf(market, view, symbol) {
  const inst = view.instruments.find((i) => i.symbol === symbol);
  if (!inst) return null;
  const models = Object.keys(inst.course);
  let from = inst.closes.length;
  models.forEach((m) => { const i = inst.course[m].findIndex((b) => b !== null); if (i >= 0 && i < from) from = i; });
  const order = view.models.map((m) => m.name).filter((m) => models.includes(m)).concat(models.filter((m) => !view.models.some((x) => x.name === m)));
  return {
    market, symbol: inst.symbol, name: inst.name, quote: inst.quote, active: inst.active, lastClose: inst.lastClose, lastDate: inst.lastDate,
    nextBase: inst.nextBase, nextDate: inst.nextDate, nextDay: markets.nextGameDay(market, view.madeOn), k: view.k,
    dates: inst.closes.map((p) => p[0]), closes: inst.closes.map((p) => p[1]), moves: inst.moves.map(r6),
    from, edges: inst.edges.slice(from).map((e) => (e ? e.map(r6) : null)),
    order, course: Object.fromEntries(order.map((m) => [m, inst.course[m].slice(from)])),
    next: inst.next.map((p) => ({ model: p.model, bin: p.bin, size: p.size, up: isUp(p, view.k), price: p.price, low: p.low, high: p.high })),
    best: view.best && order.includes(view.best) ? view.best : (order[0] || null),
  };
}

// The models table (C12): accuracy against chance and every book, per row; the hover-only figures of the old table
// (trades, win rate and money per position under each rule) are fields here. sized: the RL Position Model, long only.
function modelsOf(view) {
  const sized = new Set(view.instruments.flatMap((i) => i.next.filter((p) => p.size !== null).map((p) => p.model)));
  return view.models.map((m) => ({
    name: m.name, days: m.days, firstDay: m.firstDay, lastDay: m.lastDay, sized: sized.has(m.name),
    exact: r4(m.exact), adjacent: r4(m.adjacent), direction: r4(m.direction), aboveExact: m.aboveExact, aboveAdjacent: m.aboveAdjacent, aboveDirection: m.aboveDirection,
    daily: { trades: m.trades, winRate: r4(m.winRate), total: r2(m.pnlCash), perTrade: r2(m.pnlCashPerTrade) },
    hold: { trades: m.holdTrades, winRate: r4(m.holdWinRate), total: r2(m.holdTotal), perTrade: r2(m.holdPerTrade) },
    open: m.openTotal === null ? null : { trades: m.openTrades, winRate: r4(m.openWinRate), total: r2(m.openTotal), perTrade: r2(m.openPerTrade) },
    longShort: m.shortTotal === null ? null : { trades: m.shortTrades, winRate: r4(m.shortWinRate), total: r2(m.shortTotal), perTrade: r2(m.shortPerTrade),
      hold: r2(m.shortHoldTotal), open: r2(m.shortOpenTotal) },
  }));
}

// The guard on the week positions (C15): week by week, the guarded book against the plain week and buying everything,
// with what the user pages counted (the plain week before markets.GUARD_SHOWN_FROM, the guard from it, only live
// weeks), and the weeks booked after the fact (before the guard ran forward).
function guardOf(view) {
  const g = view.week && view.week.trading ? view.week.trading.guarded : null;
  if (!g || !g.points.some((p) => p.guarded)) return null;
  const plainBy = {};
  list(view.week.trading.followed && view.week.trading.followed.points).forEach((p) => { plainBy[p.date] = p; });
  const guarded = g.points.filter((p) => p.guarded);
  const weeks = guarded.slice().reverse().map((p) => {
    const plain = plainBy[p.date] || null;
    const before = p.date < markets.GUARD_SHOWN_FROM;
    const counted = before ? Boolean(plain && plain.live) : p.live;
    return { date: p.date, model: p.model, positions: p.positions, guard: r2(p.pnl), plain: plain ? r2(plain.pnl) : null, market: r2(p.market),
      userPages: counted ? (before ? 'plain week' : 'guard') : 'not counted', afterTheFact: Boolean(g.forwardFrom && p.date < g.forwardFrom), live: p.live };
  });
  // the live totals from the unrounded weeks, rounded once (as the old page, Following the plan and the week card;
  // a sum of the cent-rounded weeks drifts by a cent)
  const live = guarded.filter((p) => p.live);
  const sum = (f) => r2(live.reduce((s, p) => s + (num(f(p)) || 0), 0));
  return { rule: g.rule, forwardFrom: g.forwardFrom, shownFrom: markets.GUARD_SHOWN_FROM,
    live: { weeks: live.length, guard: sum((p) => p.pnl), plain: sum((p) => (plainBy[p.date] ? plainBy[p.date].pnl : null)), market: sum((p) => p.market),
      afterTheFact: live.filter((p) => g.forwardFrom && p.date < g.forwardFrom).length }, weeks };
}

// The worked example (C11): the newest settled day per instrument - its move, its bin, its edges, the close before and
// on it - and the best row of that day's ticket.
function exampleOf(view) {
  const ex = view.days[0];
  if (!ex) return null;
  const best = ex.models.find((m) => m.name === ex.best) || ex.models[0] || null;
  return { date: ex.date, instruments: ex.instruments.map((i) => ({ symbol: i.symbol, ret: r6(i.ret), bin: i.bin, edges: i.edges.map(r6), close: closeAround(view, i.symbol, ex.date) })),
    best: best ? { name: best.name, bins: best.bins, exact: best.exact, adjacent: best.adjacent, direction: best.direction, positions: best.positions } : null };
}

// GET /api/admin/market/:market/workbench: everything the market workbench draws, in sections, but the books' series
// and the price charts (their own routes).
function workbenchOf(market, view, now = new Date()) {
  const meta = markets.MARKETS[market];
  const t = view.trading;
  const nextDay = markets.nextGameDay(market, view.madeOn);
  const close = nextDay ? markets.closeMoment(market, nextDay) : null;
  return {
    market, builtAt: isoDate(now), title: meta.title, noun: meta.noun, unit: meta.unit, calendar: meta.calendar, weekGame: markets.WEEK_GAMES[market],
    k: view.k, chance: view.chance, fee: view.fee, generatedAt: isoUtc(view.generatedAt), newestGameDay: view.newestGameDay, madeOn: view.madeOn, madeAt: isoUtc(view.madeAt),
    scoredDays: view.scoredDays, stake: t ? t.stake : null, feePerLeg: t ? t.feePerLeg : null, currency: t ? t.currency : meta.unit, best: view.best,
    instruments: view.instruments.map((i) => ({ symbol: i.symbol, name: i.name, position: i.position, quote: i.quote, active: i.active, lastClose: i.lastClose, lastDate: i.lastDate, calls: i.next.length })),
    // when the newest ticket's calls are judged: said once for every instrument
    judged: nextDay ? { day: nextDay, at: isoDate(close), status: markets.dayStatus(market, nextDay, now), madeOn: view.madeOn } : null,
    plan: planOf(market, view, now),
    week: weekOf(market, view, now),
    followed: {
      day: followedOf(t ? t.followed : null, { rule: market === 'crypto' ? 'daily round trip' : 'open to close', week: false }),
      week: followedOf(view.week && view.week.trading ? view.week.trading.followed : null, { rule: market === 'crypto' ? 'weekly round trip' : 'Monday open to Friday close', week: true }),
    },
    books: booksOf(view),
    models: modelsOf(view),
    accuracy: { chance: view.chance.exact, columns: ['date', 'mean', 'best', 'bestModel'],
      days: view.daily.map((d) => [d.date, r4(d.exact_mean), r4(d.best_exact), d.best_model || null]) },
    days: view.days.slice(0, DAYS_IN_DETAIL).map((d) => ({ date: d.date, best: d.best, exactMean: r4(d.exactMean),
      instruments: d.instruments.map((i) => ({ symbol: i.symbol, ret: r6(i.ret), bin: i.bin })) })),
    guard: guardOf(view),
    regimes: view.regimes.map((r) => ({ row: r.row, date: r.date, regimes: r.regimes, label: r.label, template: r.template, probability: r4(r.probability),
      volatilityRank: r.volatilityRank, features: r.features, expected: r.expected.map((e) => ({ symbol: e.symbol, value: r6(e.value) })), readings: r.history })),
    example: exampleOf(view),
  };
}

// --- routes ------------------------------------------------------------------------------------------------------------
// ctx: from admin.install - dirs (resolveDirs), answerLater (its JSON answer of a promise), games ({GAMES, UNIT,
// groupOf}), rankCacheFile (the rank history's cache, gitignored config/ by default)
function install(app, ctx) {
  const { dirs, answerLater, games } = ctx;
  const gameOf = (req) => (games.GAMES.includes(req.params.game) ? req.params.game : null);
  const scopeOf = (req) => (req.query.scope === undefined || req.query.scope === '' ? 'all' : String(req.query.scope));
  const badRequest = (req) => {
    if (!gameOf(req)) return { error: 'unknown game', status: 404 };
    if (!SCOPES.includes(scopeOf(req))) return { error: 'scope is all or top', status: 400 };
    return null;
  };
  const extraOf = (game) => ({ group: games.groupOf(game), unit: games.UNIT[game] });
  const history = () => rankhistory.load(dirs.root, { cacheFile: ctx.rankCacheFile });
  const marketOf = (req) => (MARKETS.includes(req.params.market) ? req.params.market : null);
  const source = (req) => {
    const market = marketOf(req);
    if (!market) return { error: 'unknown market', status: 404 };
    const view = workbenchSource(dirs, market);
    return view ? { market, view } : { error: 'no market record yet', status: 404 };
  };

  app.get('/api/admin/history/:game', (req, res) => answerLater(res, async () => {
    const bad = badRequest(req);
    if (bad) return bad;
    const month = req.query.month === undefined || req.query.month === '' ? null : String(req.query.month);
    if (month !== null && !/^\d{4}-\d{2}$/.test(month)) return { error: 'the month is YYYY-MM', status: 400 };
    return historyView(dirs, gameOf(req), scopeOf(req), await history(), extraOf(gameOf(req)), month);
  }));
  app.get('/api/admin/draw/:game/:date', (req, res) => answerLater(res, async () => {
    const bad = badRequest(req);
    if (bad) return bad;
    const date = isoDay(req.params.date);
    if (!date) return { error: 'the date is YYYY-MM-DD', status: 400 };
    return drawView(dirs, gameOf(req), date, scopeOf(req), await history(), rankings.loadReport(dirs.databaseDir), extraOf(gameOf(req)));
  }));
  app.get('/api/admin/market/:market/workbench', (req, res) => answerLater(res, async () => {
    const s = source(req);
    return s.error ? s : workbenchOf(s.market, s.view);
  }));
  app.get('/api/admin/market/:market/workbench/books/:rule', (req, res) => answerLater(res, async () => {
    const s = source(req);
    if (s.error) return s;
    if (!BOOK_RULES.includes(req.params.rule)) return { error: `the rule is one of ${BOOK_RULES.join(', ')}`, status: 404 };
    return bookSeries(s.view, req.params.rule) || { error: 'this market has no such book', status: 404 };
  }));
  app.get('/api/admin/market/:market/workbench/instrument/:symbol', (req, res) => answerLater(res, async () => {
    const s = source(req);
    if (s.error) return s;
    if (!SYMBOL.test(req.params.symbol)) return { error: 'unknown instrument', status: 404 };
    return instrumentOf(s.market, s.view, req.params.symbol) || { error: 'unknown instrument', status: 404 };
  }));
}

module.exports = { SCOPES, MARKETS, BOOK_RULES, isoDay, dayFilesOf, readDay, readFull, historyView, drawView, planOf, weekOf, followedOf, booksOf, bookSeries, instrumentOf,
  modelsOf, guardOf, exampleOf, workbenchOf, workbenchSource, install };
