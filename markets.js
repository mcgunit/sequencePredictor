// markets.js - the Crypto and Shares records (README roadmap item 4) read for the JSON API.
//
// Predictor.py tracks the two markets as positional games, and after each
// run src/MarketSettle.py settles every stored day against the real returns
// and writes data/markets/<market>.json: closes with, per date, the game
// day's edges and return and every model's predicted bin, the next day's
// predictions as prices, per-model accuracy against chance and the daily
// accuracy series. Two more records sit next to it (phase M3):
// data/controls/markets/<market>-rows.json, the weekly MarketRows.py report
// that scores the market rows under a proper scoring rule against GARCH, and
// data/markets/<market>-regimes.json, the Regime HMM rows' reading of which
// regime the market is in.
//
// Pure functions only, no routes: the loaders and view models (describeMarket
// and friends), the book the plan and the week card follow (pickPlan,
// pickWeek) and the market clock (when a call is judged) - for the app's
// market pages (api.js), the admin workbench (workbench.js), admin.js and
// lottery.js, and test/markets.test.js.
'use strict';

const fs = require('fs');
const path = require('path');

const MARKETS = {
  crypto: { title: 'Crypto predictor', noun: 'coin', unit: 'USDT', calendar: 'every day (UTC)' },
  shares: { title: 'Shares predictor', noun: 'share', unit: 'EUR', calendar: 'every trading day (Euronext Brussels and Amsterdam)' },
};
// The week games (README roadmap item 4, M5): one draw per week, cut from the
// same bars (MarketGame.weekly_returns), settled and exported like a market
// (data/markets/<week game>.json) and shown on the market's pages as the
// week card - the multi-day forecast a hold can be planned on.
const WEEK_GAMES = { crypto: 'cryptoweek', shares: 'sharesweek' };
const MIN_WEEK_TRADES = 5;   // positions a weekly book needs before the week card follows it
const MIN_PLAN_TRADES = 10;  // positions a daily book needs before Today's plan follows it
const MIN_RANK_DAYS = 10;    // MarketSettle.MIN_RANK_DAYS: a row with fewer settled days ranks after the others and cannot head a book

// Which book the plan follows (Today's plan, and the user page's positions, api.js): the best book under the rule a
// reader can follow - open to close for shares, the daily round trip for crypto - among the rows with MIN_PLAN_TRADES
// positions, else the best exact rate. {model, byBook, key, tradesKey, ruleName}.
function pickPlan(view, market) {
  const key = market === 'crypto' ? 'pnlCash' : 'openTotal';
  const tradesKey = market === 'crypto' ? 'trades' : 'openTrades';
  const ranked = view.models.filter((m) => m[key] !== null && m[tradesKey] !== null && m[tradesKey] >= MIN_PLAN_TRADES).sort((a, b) => b[key] - a[key]);
  return { model: ranked[0] || view.models.find((m) => m.name === view.best) || null, byBook: ranked.length > 0, key, tradesKey,
           ruleName: market === 'crypto' ? 'daily round trip' : 'open to close' };
}
// The same for the week game's card: its best weekly book from MIN_WEEK_TRADES positions on.
function pickWeek(w, market) {
  const key = market === 'crypto' ? 'pnlCash' : 'openTotal';
  const tradesKey = market === 'crypto' ? 'trades' : 'openTrades';
  const ranked = w.models.filter((m) => m[key] !== null && m[tradesKey] !== null && m[tradesKey] >= MIN_WEEK_TRADES).sort((a, b) => b[key] - a[key]);
  return { model: ranked[0] || w.models.find((m) => m.name === w.best) || w.models[0] || null, byBook: ranked.length > 0, key, tradesKey,
           ruleName: market === 'crypto' ? 'weekly round trip' : 'Monday open to Friday close' };
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return null; }
}

function loadMarket(dir, market) {
  const record = readJson(path.join(dir, `${market}.json`));
  return record && record.market === market && Array.isArray(record.instruments) && Array.isArray(record.models) ? record : null;
}

function loadRows(controlsDir, market) {
  const record = readJson(path.join(controlsDir, 'markets', `${market}-rows.json`));
  return record && record.market === market && Array.isArray(record.rows) ? record : null;
}

function loadRegimes(dir, market) {
  const record = readJson(path.join(dir, `${market}-regimes.json`));
  return record && typeof record === 'object' ? record : null;
}

const num = (x) => {
  const v = Number(x);
  return x === null || x === undefined || !Number.isFinite(v) ? null : v;
};

// The weekly report's rows, ready to print: log-scores, the interval and
// verdict against GARCH, against uniform and against the column frequencies,
// hit rates and the fixed rule's P&L with the real returns. Sorted best
// log-score first, rows without a probability last. Since 26 Oct 2026 the
// baselines carry probabilities too (random 1 in K, the frequencies counted
// over the days before), so they sort among the rows, and vsFrequency says
// whether a row beats what knowing the bins' frequencies alone scores; a
// report from before has no vsFrequency (null) and baselines without a score.
// The summary lists (betterThanReference, betterThanFrequency) name model
// rows only: a baseline beating GARCH is read from its own row.
function describeRows(record) {
  if (!record) return null;
  const uniform = num(record.uniform_log_score);
  const interval = (x) => (x && num(x.mean) !== null ? { mean: num(x.mean), lo: num(x.lo), hi: num(x.hi), verdict: String(x.verdict || ''), days: num(x.days) } : null);
  const frequency = record.frequency_reference ? String(record.frequency_reference) : null;
  const rows = (record.rows || []).map((r) => ({
    name: String(r.name), kind: String(r.kind || 'base'), days: num(r.days), scoredDays: num(r.scored_days),
    logScore: num(r.log_score), se: num(r.log_score_se),
    vsReference: interval(r.vs_reference), vsUniform: interval(r.vs_uniform), vsFrequency: interval(r.vs_frequency),
    exact: num(r.exact_rate), adjacent: num(r.adjacent_rate), direction: num(r.direction_rate),
    trades: num(r.trades), pnl: num(r.pnl_return), pnlPerTrade: num(r.pnl_per_trade),
    isReference: String(r.name) === String(record.reference),
    isFrequency: frequency !== null && String(r.name) === frequency,
  }));
  rows.sort((a, b) => {
    if ((a.logScore === null) !== (b.logScore === null)) return a.logScore === null ? 1 : -1;
    if (a.logScore !== null && b.logScore !== null && a.logScore !== b.logScore) return b.logScore - a.logScore;
    return (b.exact || 0) - (a.exact || 0);
  });
  // the model rows, in the sorted order (the summary lists below name them best log-score first, as before)
  const models = rows.filter((r) => r.kind !== 'baseline');
  const reference = rows.find((r) => r.isReference) || null;
  return {
    market: record.market, generatedAt: record.generated_at || null, days: num(record.days_scored), requested: num(record.days_requested),
    firstDay: record.first_day || null, lastDay: record.last_day || null, uniform, reference: String(record.reference || ''),
    referenceRow: reference, floor: num(record.probability_floor), rows, lockboxWithheld: num(record.lockbox_days_withheld) || 0,
    hmmFeatures: record.hmm_features ? String(record.hmm_features) : null,   // the feature set the three Regime HMM rows ran on (MarketRows, 14 Oct 2026)
    // the baseline vsFrequency is taken against, its row, and the add-one the baselines' counts were smoothed with
    frequency, frequencyRow: rows.find((r) => r.isFrequency) || null, frequencySmoothing: num(record.frequency_smoothing),
    betterThanReference: models.filter((r) => r.vsReference && r.vsReference.verdict === 'better').map((r) => r.name),
    betterThanFrequency: models.filter((r) => r.vsFrequency && r.vsFrequency.verdict === 'better').map((r) => r.name),
    referenceAboveUniform: !!(reference && reference.vsUniform && reference.vsUniform.verdict === 'better'),
    referenceAboveFrequency: !!(reference && reference.vsFrequency && reference.vsFrequency.verdict === 'better'),
    errors: record.errors && typeof record.errors === 'object' ? record.errors : {},
  };
}

// The newest reading of each Regime HMM row: which regime, how sure, how many.
function describeRegimes(record, symbols) {
  if (!record) return [];
  return Object.keys(record).filter((k) => Array.isArray(record[k]) && record[k].length).sort().map((row) => {
    const latest = record[row][record[row].length - 1];
    return {
      row, date: latest.date || null, regimes: num(latest.regimes), label: String(latest.label || ''), template: latest.template === null || latest.template === undefined ? null : num(latest.template),
      probability: num(latest.probability), volatilityRank: num(latest.volatility_rank),
      features: Array.isArray(latest.features) ? latest.features.map(String) : null,   // the row's feature set (MarketModels' regime log, 14 Oct 2026)
      expected: Array.isArray(latest.expected_return) ? latest.expected_return.map((v, i) => ({ symbol: symbols && symbols[i] ? symbols[i] : `#${i + 1}`, value: num(v) })) : [],
      volatility: Array.isArray(latest.volatility) ? latest.volatility.map((v) => num(v)) : [],
      history: record[row].length,
    };
  });
}

// One view model per market: models with their rates read against chance,
// instruments with what the charts need, the next-day table, the series.
// `extras` = { rows, regimes }: the weekly report and the regime log.
function describeMarket(record, extras) {
  const chance = record.chance || { exact: 0.1, adjacent: 0.28, direction: 0.5 };
  const models = (record.models || []).map((m) => {
    const exact = num(m.exact_rate); const adjacent = num(m.adjacent_rate); const direction = num(m.direction_rate);
    return {
      name: String(m.name), days: num(m.days), positions: num(m.positions), trades: num(m.trades),
      exact, adjacent, direction,
      aboveExact: exact !== null && exact > chance.exact,
      aboveAdjacent: adjacent !== null && adjacent > chance.adjacent,
      aboveDirection: direction !== null && direction > chance.direction,
      pnlTotal: num(m.pnl_total), pnlPerTrade: num(m.pnl_per_trade),
      pnlCash: num(m.pnl_cash_total), pnlCashPerTrade: num(m.pnl_cash_per_trade), wins: num(m.wins), winRate: num(m.win_rate),
      holdTotal: num(m.hold_total), holdTrades: num(m.hold_trades), holdWinRate: num(m.hold_win_rate), holdPerTrade: num(m.hold_per_trade),
      openTotal: num(m.open_total), openTrades: num(m.open_trades), openWinRate: num(m.open_win_rate), openPerTrade: num(m.open_per_trade),
      shortTotal: num(m.short_total), shortTrades: num(m.short_trades), shortWinRate: num(m.short_win_rate), shortPerTrade: num(m.short_per_trade),
      shortHoldTotal: num(m.short_hold_total), shortOpenTotal: num(m.short_open_total),
      firstDay: m.first_day || null, lastDay: m.last_day || null,
    };
  });
  const next = record.next || null;
  const instruments = (record.instruments || []).filter((i) => i && typeof i === 'object').map((i) => {
    const nextFor = next && next.instruments && next.instruments[i.symbol] ? next.instruments[i.symbol] : null;
    // closes with a malformed point dropped; the aligned series (edges, moves,
    // course) follow the same indices, or are blank when their length disagrees
    const rawCloses = Array.isArray(i.closes) ? i.closes : [];
    const keep = rawCloses.map((p, k) => (Array.isArray(p) && p.length === 2 && num(p[1]) !== null ? k : -1)).filter((k) => k >= 0);
    const aligned = (arr) => (Array.isArray(arr) && arr.length === rawCloses.length ? keep.map((k) => arr[k]) : keep.map(() => null));
    const course = {};
    if (i.course && typeof i.course === 'object') {
      Object.keys(i.course).forEach((model) => { course[model] = aligned(i.course[model]).map((b) => (b === null || b === undefined ? null : num(b))); });
    }
    return {
      symbol: String(i.symbol), name: i.name || String(i.symbol), position: num(i.position), quote: i.quote || '',
      active: i.active !== false,
      lastClose: num(i.last_close), lastDate: i.last_date || null,
      closes: keep.map((k) => rawCloses[k]),
      edges: aligned(i.edges).map((e) => (Array.isArray(e) ? e.map(num) : null)),
      moves: aligned(i.moves).map(num),
      course,
      next: nextFor ? (nextFor.predictions || []).map((p) => ({
        model: String(p.model), bin: num(p.bin), direction: num(p.direction), price: num(p.price), low: num(p.low), high: num(p.high),
        size: num(p.size),   // a sized row (RL Position Model): the position is its size, not its bin
      })) : [],
      // the close the next-day call stands on: the newest GAME day's, which a
      // newer bar of this instrument alone does not replace (MarketSettle.next_day_predictions)
      nextBase: nextFor ? num(nextFor.last_close) : null, nextDate: nextFor ? nextFor.last_date || null : null,
    };
  });
  return {
    market: record.market, generatedAt: record.generated_at || null, k: num(record.k) || 10, fee: num(record.fee),
    chance, newestGameDay: record.newest_game_day || null,
    madeOn: next ? next.made_on || null : null,
    madeAt: next ? next.made_at || null : null,      // the ticket's write time (MarketSettle.next_day_predictions)
    models, drawn: Array.isArray(record.drawn_models) ? record.drawn_models.map(String) : [],
    best: record.best_model ? String(record.best_model) : (Array.isArray(record.drawn_models) && record.drawn_models.length ? String(record.drawn_models[0]) : (models.length ? models[0].name : null)),
    instruments, daily: Array.isArray(record.daily) ? record.daily : [],
    scoredDays: models.length ? Math.max(...models.map((m) => m.days || 0)) : 0,
    rows: describeRows(extras && extras.rows ? extras.rows : null),
    regimes: describeRegimes(extras && extras.regimes ? extras.regimes : null, instruments.map((i) => i.symbol)),
    week: extras && extras.week && typeof extras.week === 'object' && extras.week.market ? extras.week : null,   // the week game's own view (describeMarket of its record)
    days: describeDays(record.days),
    trading: describeTrading(record.trading),
  };
}

// The paper-trading book (MarketSettle.ledger): per model the money made per
// settled day and the running total, the market benchmark, stake and fees.
function describeTrading(t) {
  if (!t || typeof t !== 'object') return null;
  const point = (p) => (Array.isArray(p) && p.length === 3 && typeof p[0] === 'string' ? { date: p[0], pnl: num(p[1]), total: num(p[2]) } : null);
  const series = (arr) => (Array.isArray(arr) ? arr.map(point).filter(Boolean) : []);
  const seriesMap = (obj) => { const out = {}; if (obj && typeof obj === 'object') Object.keys(obj).forEach((m) => { out[m] = series(obj[m]); }); return out; };
  const hold = t.hold && typeof t.hold === 'object' ? t.hold : {};
  const open = t.open && typeof t.open === 'object' ? t.open : null;   // the open-to-close book: shares only (MarketSettle.OPEN_RULE_MARKETS)
  return {
    stake: num(t.stake), feePerLeg: num(t.fee_per_leg), currency: t.currency ? String(t.currency) : '', rule: t.rule ? String(t.rule) : '', holdRule: t.hold_rule ? String(t.hold_rule) : '',
    // how a position is sized and charged (MarketSettle.cost_terms, 31 Oct 2026): the shares' whole shares at Bolero's
    // costs, crypto's 0.1% a leg; null on older records (crypto's terms)
    costs: t.costs && typeof t.costs === 'object' ? { kind: String(t.costs.kind || ''), wholeUnits: t.costs.whole_units === true, text: t.costs.text ? String(t.costs.text) : '' } : null,
    openRule: t.open_rule ? String(t.open_rule) : '',
    shortRule: t.short_rule ? String(t.short_rule) : '',
    dates: Array.isArray(t.dates) ? t.dates.map(String) : [], models: seriesMap(t.models), benchmark: series(t.benchmark),
    hold: { models: seriesMap(hold.models), benchmark: series(hold.benchmark) },
    open: open ? { models: seriesMap(open.models), benchmark: series(open.benchmark) } : null,
    // what following the page's own plan made (MarketSettle.followed_plan): per settled day the row the plan followed, picked from the
    // record before that day, its positions and money that day, and whether the day's ticket was out before the day closed (live)
    followed: describeFollowed(t.followed),
    // the week games' guarded book (MarketSettle.guarded_week, 17 Oct 2026): the week positions under the daily guard
    guarded: describeGuarded(t.guarded),
    // the crypto order book (src/MarketOrders.py, 30 Oct 2026): what the day and week cards tell a reader to do, booked at
    // the card's publication with a sell limit; crypto and cryptoweek only, null elsewhere and on a record exported before it
    orders: describeOrders(t.orders),
    // the same rules with shorts on (MarketSettle.ledger "short"): money per model per rule; the market lines above are their benchmark
    short: t.short && typeof t.short === 'object' ? Object.fromEntries(['daily', 'hold', 'open'].filter((r) => t.short[r] && typeof t.short[r] === 'object').map((r) => [r, { models: seriesMap(t.short[r].models), sold: series(t.short[r].benchmark) }])) : null,
  };
}

// The first week the user pages (api.js) count the guarded week book: the first full week the page showed the guard
// (owner, 20 Oct 2026: count "what the page said"); before it they count the plain week book.
const GUARD_SHOWN_FROM = '2026-10-26';
// MarketSettle.GUARD_FORWARD_FROM: the first week the guard ran forward; for a record exported before it carried forward_from
const GUARD_FORWARD_FROM = '2026-10-19';

// The guarded week book (MarketSettle.guarded_week). Points are [week close, week row, positions, money, total, live,
// market, by_symbol, trades, guarded]; guarded false is a week from before the daily record (the plain week).
function describeGuarded(g) {
  if (!g || typeof g !== 'object' || !Array.isArray(g.points)) return null;
  const points = g.points.filter((p) => Array.isArray(p) && p.length >= 8 && typeof p[0] === 'string').map((p) => ({
    date: p[0], model: p[1] === null || p[1] === undefined ? null : String(p[1]), positions: num(p[2]) || 0, pnl: num(p[3]) || 0,
    live: p[5] === true, market: num(p[6]), guarded: p[9] !== false,
  }));
  return { rule: g.rule ? String(g.rule) : '', forwardFrom: typeof g.forward_from === 'string' ? g.forward_from : GUARD_FORWARD_FROM, points };
}

// The first day (a day card's UTC day; a week by its first day, the Monday) the user pages show the ORDER BOOK's card
// and count its record (src/MarketOrders.py, design orderbook-DESIGN-30oct.md section 10); days and weeks before it keep
// the old record (as GUARD_SHOWN_FROM splits the week record). null: a shadow on the Admin market page only.
// Switched by the owner's decision of 31 Oct 2026 to Mon 2 Nov 2026 - the first full week after the deploy, when the
// publication times are recorded instead of estimated - with no shadow period, because the pages had no users yet.
// Until a card for that day (or the week starting that day) is on the page, the user page stays exactly as before
// (api.marketView gates on the card's day, not on the clock).
const ORDERS_SHOWN_FROM = '2026-11-02';
// MarketOrders.ORDER_VARIANT_LOG's primary, pre-registered on 30 Oct 2026: the limit at the predicted bin's lower edge.
// A record exported before it carried "primary" uses it; the user line books per card the variant that card showed.
const ORDER_VARIANT_PRIMARY = 'edge';
const ORDER_STATUSES = ['closed', 'open', 'pending', 'skip', 'not followable'];
const ORDER_EXITS = ['limit', 'early', 'next card', 'fallback', 'close'];

// The order book (MarketOrders.orders_record; its module docstring has the exact shape). Read tolerantly like the
// guarded book: a malformed order, point or entry is dropped, a missing block is empty or null, nothing is thrown.
function describeOrders(o) {
  if (!o || typeof o !== 'object' || Array.isArray(o)) return null;
  const obj = (v) => (v && typeof v === 'object' && !Array.isArray(v) ? v : null);
  const str = (v) => (typeof v === 'string' && v ? v : null);
  const counts = (v) => Object.fromEntries(Object.entries(obj(v) || {}).filter(([, n]) => num(n) !== null).map(([k, n]) => [String(k), num(n)]));
  const stats = (s) => {
    const x = obj(s) || {};
    return { orders: num(x.orders) || 0, closed: num(x.closed) || 0, open: num(x.open) || 0, pending: num(x.pending) || 0, filled: num(x.filled) || 0,
      early: num(x.early) || 0, timeStop: num(x.time_stop) || 0, skips: counts(x.skips), notFollowable: num(x.not_followable) || 0,
      total: num(x.total), wins: num(x.wins) || 0, estimated: num(x.estimated) || 0, rolls: num(x.rolls) || 0 };
  };
  const order = (r) => {
    if (!obj(r) || !str(r.interval) || !ISO_DAY.test(r.interval) || !str(r.symbol)) return null;
    const status = ORDER_STATUSES.includes(r.status) ? r.status : null;
    if (!status) return null;
    const days = Array.isArray(r.days) ? r.days.filter((d) => typeof d === 'string' && ISO_DAY.test(d)) : [];
    return {
      interval: r.interval, days: days.length ? days : [r.interval], symbol: String(r.symbol), model: str(r.model), bin: num(r.bin), scale: num(r.scale),
      variant: str(r.variant), kind: ['main', 'late', 'reference', 'newcomer'].includes(r.kind) ? r.kind : 'main', status, reason: str(r.reason),
      publishedAt: str(r.published_at), estimated: r.estimated === true, baseClose: num(r.base_close), limit: num(r.limit),
      limits: Array.isArray(r.limits) ? r.limits.map(num).filter((v) => v !== null) : [],
      entryAt: str(r.entry_at), entryPrice: num(r.entry_price), limitPct: num(r.limit_pct),
      exitKind: ORDER_EXITS.includes(r.exit_kind) ? r.exit_kind : null, exitAt: str(r.exit_at), exitPrice: num(r.exit_price),
      move: num(r.move), money: status === 'closed' ? num(r.money) : null, rolls: num(r.rolls) || 0, provisional: r.provisional === true, live: r.live !== false,
    };
  };
  const book = (b) => {
    const x = obj(b);
    if (!x) return null;
    const orders = (Array.isArray(x.orders) ? x.orders : []).map(order).filter(Boolean);
    return { orders, stats: stats(x.stats), refused: x.refused === undefined ? null : counts(x.refused) };
  };
  const books = (v) => Object.fromEntries(Object.entries(obj(v) || {}).map(([k, b]) => [String(k), book(b)]).filter(([, b]) => b));
  const c = obj(o.constants) || {};
  const bench = obj(o.benchmark) || {};
  const rec = obj(o.record) || {};
  const coverage = {};
  Object.entries(obj(o.coverage) || {}).forEach(([sym, v]) => {
    const x = obj(v);
    if (x) coverage[String(sym)] = { first1h: str(x['1h_first']), last1h: str(x['1h_last']), bars1h: num(x['1h_bars']), bars1m: num(x['1m_bars']) };
  });
  // a roll card's newcomer entry (31 Oct 2026): the reader who did not buy on the earlier card
  const newcomerOf = (v) => {
    const x = obj(v);
    if (!x || !['buy now', 'no order'].includes(x.status)) return null;
    return { status: x.status, state: str(x.state), limit: num(x.limit), entryPrice: num(x.entry_price), limitPct: num(x.limit_pct), reason: str(x.reason), provisional: x.provisional === true };
  };
  const nowOf = (n) => {
    const x = obj(n);
    if (!x || !str(x.interval)) return null;
    const symbols = {};
    Object.entries(obj(x.symbols) || {}).forEach(([sym, v]) => {
      const s = obj(v);
      if (!s || !str(s.status)) return;
      symbols[String(sym)] = { status: s.status, call: str(s.call), bin: num(s.bin), limit: num(s.limit), entryPrice: num(s.entry_price), limitPct: num(s.limit_pct),
        reason: str(s.reason), provisional: s.provisional === true, heldSince: str(s.held_since), state: str(s.state), exitKind: ORDER_EXITS.includes(s.exit_kind) ? s.exit_kind : null,
        exitAt: str(s.exit_at), money: num(s.money), newcomer: newcomerOf(s.newcomer),
        // earlier newcomer entries held into this card or sold at it: [first day, entry price]
        newcomersHeld: (Array.isArray(s.newcomers_held) ? s.newcomers_held : []).filter((p) => Array.isArray(p) && typeof p[0] === 'string' && ISO_DAY.test(p[0]))
          .map((p) => ({ since: p[0], entryPrice: num(p[1]) })) };
    });
    const late = {};
    Object.entries(obj(x.late) || {}).forEach(([sym, v]) => {
      const s = obj(v);
      if (s) late[String(sym)] = { shown: s.shown === true, reason: str(s.reason), price: num(s.price), gapPct: num(s.gap_pct), limit: num(s.limit) };
    });
    return { interval: x.interval, firstDay: str(x.first_day), madeOn: str(x.made_on), publishedAt: str(x.published_at), estimated: x.estimated === true, model: str(x.model),
      variant: str(x.variant), deadlineFallback: str(x.deadline_fallback), runningDay: str(x.running_day), dayModel: str(x.day_model), symbols, late };
  };
  return {
    market: str(o.market), game: o.game === 'week' ? 'week' : 'day', generatedAt: str(o.generated_at), since: str(o.since), rule: str(o.rule) || '', lateRule: str(o.late_rule) || '',
    newcomerRule: str(o.newcomer_rule) || '',
    constants: { feePerLeg: num(c.fee_per_leg), minGap: num(c.min_gap), fillThrough: num(c.fill_through), publishLagFallbackMin: num(c.publish_lag_fallback_min),
      deadlineFallbackHours: num(c.deadline_fallback_hours), lateMinGap: num(c.late_min_gap), lateMinLeftHours: num(c.late_min_left_hours), stake: num(c.stake), fixedGap: num(c.fixed_gap) },
    primary: str(o.primary) || ORDER_VARIANT_PRIMARY,
    variants: Object.fromEntries(Object.entries(obj(o.variants) || {}).filter(([, t]) => typeof t === 'string').map(([k, t]) => [String(k), t])),
    variantLog: (Array.isArray(o.variant_log) ? o.variant_log : []).filter((p) => Array.isArray(p) && p.length >= 2 && typeof p[0] === 'string' && typeof p[1] === 'string')
      .map((p) => ({ date: p[0], variant: p[1], reason: typeof p[2] === 'string' ? p[2] : '' })),
    review: str(o.review) || '',
    books: books(o.books), reference: books(o.reference), late: obj(o.late) ? books(o.late) : null,
    // the day's newcomer ledger (31 Oct 2026): a roll card's entry for the reader who did not buy before; null on a week or an older record
    newcomer: obj(o.newcomer) ? books(o.newcomer) : null,
    benchmark: { points: (Array.isArray(bench.points) ? bench.points : []).filter((p) => Array.isArray(p) && p.length >= 3 && typeof p[0] === 'string' && num(p[2]) !== null)
      .map((p) => ({ interval: p[0], coins: num(p[1]) || 0, money: num(p[2]), closed: p[3] !== false })), stats: stats(bench.stats) },
    record: { shown: (Array.isArray(rec.shown) ? rec.shown : []).filter((p) => Array.isArray(p) && typeof p[0] === 'string' && typeof p[1] === 'string')
      .map((p) => ({ interval: p[0], variant: p[1] })), stats: stats(rec.stats) },
    publications: (Array.isArray(o.publications) ? o.publications : []).filter((p) => Array.isArray(p) && p.length >= 3 && typeof p[0] === 'string' && typeof p[1] === 'string')
      .map((p) => ({ madeOn: p[0], interval: p[1], publishedAt: str(p[2]), estimated: p[3] === true })),
    coverage, now: nowOf(o.now),
  };
}

// The followed-plan record (MarketSettle.followed_plan). Points are
// [date, model, picked_by, positions, pnl, total, live, market_pnl, by_symbol, moves]; a
// malformed point is dropped, never thrown.
function describeFollowed(f) {
  if (!f || typeof f !== 'object' || !Array.isArray(f.points)) return null;
  const points = f.points.filter((p) => Array.isArray(p) && p.length >= 8 && typeof p[0] === 'string').map((p) => ({
    date: p[0], model: p[1] === null || p[1] === undefined ? null : String(p[1]), pickedBy: p[2] === null || p[2] === undefined ? null : String(p[2]),
    positions: num(p[3]) || 0, pnl: num(p[4]) || 0, total: num(p[5]), live: p[6] === true, market: num(p[7]),
    // the followed row's money per instrument that day (MarketSettle.followed_plan, 15 Oct 2026); {} on older records
    bySymbol: p.length >= 9 && p[8] && typeof p[8] === 'object' ? Object.fromEntries(Object.entries(p[8]).map(([k, v]) => [String(k), num(v)])) : {},
    // the same positions' moves in percent (31 Oct 2026); {} on older records, where the move is read back from the money
    moveBySymbol: p.length >= 10 && p[9] && typeof p[9] === 'object' ? Object.fromEntries(Object.entries(p[9]).map(([k, v]) => [String(k), num(v)])) : {},
  }));
  return {
    rule: f.rule === 'open' ? 'open' : 'daily', minTrades: num(f.min_trades), firstLive: typeof f.first_live === 'string' ? f.first_live : null,
    total: num(f.total), marketTotal: num(f.market_total), liveTotal: num(f.live_total), liveMarket: num(f.live_market),
    liveDays: num(f.live_days) || 0, livePositions: num(f.live_positions) || 0, points,
  };
}

// The settled days in market terms (MarketSettle.day_records): per instrument
// the real return, its bin and the edges it was cut with; per model the bins
// it played and the day's hits. Malformed entries are dropped, never thrown.
function describeDays(days) {
  if (!Array.isArray(days)) return [];
  return days.filter((d) => d && typeof d.date === 'string' && Array.isArray(d.instruments)).map((d) => ({
    date: d.date,
    instruments: d.instruments.filter((i) => i && i.symbol !== undefined).map((i) => ({
      symbol: String(i.symbol), ret: num(i.return), bin: num(i.bin), edges: Array.isArray(i.edges) ? i.edges.map(num) : [],
    })),
    models: (Array.isArray(d.models) ? d.models : []).filter((m) => m && typeof m === 'object' && m.name !== undefined).map((m) => ({
      name: String(m.name), bins: Array.isArray(m.bins) ? m.bins.map((b) => (b === null ? null : num(b))) : [],
      sizes: Array.isArray(m.sizes) ? m.sizes.map((v) => (v === null || v === undefined ? null : num(v))) : null,   // a sized row's size per slot (RL Position Model)
      exact: num(m.exact), adjacent: num(m.adjacent), direction: num(m.direction), positions: num(m.positions), pnl: num(m.pnl), pnlCash: num(m.pnl_cash), trades: num(m.trades),
    })),
    best: d.best || null, exactMean: num(d.exact_mean),
  }));
}

// --- when a call is judged ------------------------------------------------
// A ticket is made by the daily run (crypto 09:00, shares 18:30 Belgian time) after the newest
// candle has closed, for the game day after that candle; the day file carries
// the newest candle's date (made_on) and its write time (made_at), so the day
// a ticket is for is the next game day, and the moment it is judged is that
// day's close. The reader asked (4 Oct 2026) when a position has to be closed:
// these helpers give that moment, and say it in Belgian time. Crypto's day is
// the UTC day (close at 00:00 UTC, which is the next calendar day in Belgium);
// a share's day is the Euronext session of Brussels and Amsterdam (09:00-17:30
// Belgian time, 31 Oct 2026: the five shares moved there from New York), and
// every instant is computed for its date, never hard-coded in UTC.
// Which daily run brings a market's next call and results (jobs.js): crypto's at 09:00, the shares' at 18:30 on
// weekdays since 31 Oct 2026, so their card is up before Euronext's 09:00 open.
const RUN_NAME = { crypto: 'the morning run', shares: 'the evening run' };
const runName = (market) => RUN_NAME[market] || 'the daily run';
const BRUSSELS = 'Europe/Brussels';
const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;
const ISO_DAY = /^(\d{4})-(\d{2})-(\d{2})$/;
const isoDay = (ms) => new Date(ms).toISOString().slice(0, 10);
const parseDay = (text) => {
  const m = ISO_DAY.exec(String(text || ''));
  if (!m) return null;
  const ms = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  return isoDay(ms) === m[0] ? ms : null;        // Date.UTC rolls 2026-02-30 over to March; refuse it instead
};
const zoneParts = (date, zone) => {
  const parts = {};
  new Intl.DateTimeFormat('en-GB', { timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
                                     weekday: 'long', hour12: false, timeZoneName: 'short' }).formatToParts(date).forEach((p) => { parts[p.type] = p.value; });
  parts.hour = parts.hour === '24' ? '00' : parts.hour;
  return parts;
};
// Euronext's harmonised calendar for its cash markets (live.euronext.com, "Trading hours & holidays"): closed on
// New Year's Day, Good Friday, Easter Monday, 1 May, 25 and 26 December; 24 and 31 December are half days. The rule
// matches every session of the five shares from 2016 to October 2026 (no bar on a rule day, none missing on another
// weekday); EURONEXT_PUBLISHED_YEARS are the years Euronext has published a calendar for - another year gets the
// rule and the word "normally" (workbench.js calendarKnown). Half days close at 14:00 (continuous trading; the
// closing auction follows, as at 17:30 on a full day).
const EURONEXT_PUBLISHED_YEARS = new Set(['2026']);
const EURONEXT_OPEN = [9, 0];
const EURONEXT_CLOSE = [17, 30];
const EURONEXT_HALF_CLOSE = [14, 0];
function easterSunday(year) {      // the Gregorian computus (anonymous algorithm), as a UTC midnight in ms
  const a = year % 19; const b = Math.floor(year / 100); const c = year % 100; const d = Math.floor(b / 4); const e = b % 4;
  const f = Math.floor((b + 8) / 25); const g = Math.floor((b - f + 1) / 3); const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4); const k = c % 4; const l = (32 + 2 * e + 2 * i - h - k) % 7; const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31); const day = ((h + l - 7 * m + 114) % 31) + 1;
  return Date.UTC(year, month - 1, day);
}
const weekday = (ms) => ![0, 6].includes(new Date(ms).getUTCDay());
function euronextClosed(year) {
  const easter = easterSunday(year);
  return [Date.UTC(year, 0, 1), easter - 2 * DAY_MS, easter + DAY_MS, Date.UTC(year, 4, 1), Date.UTC(year, 11, 25), Date.UTC(year, 11, 26)]
    .filter(weekday).map(isoDay);
}
const EURONEXT_CLOSED = new Set();
const EURONEXT_HALF_DAYS = new Set();
for (let year = 2016; year <= 2040; year += 1) {
  euronextClosed(year).forEach((d) => EURONEXT_CLOSED.add(d));
  [Date.UTC(year, 11, 24), Date.UTC(year, 11, 31)].filter(weekday).forEach((ms) => EURONEXT_HALF_DAYS.add(isoDay(ms)));
}
function nextGameDay(market, madeOn) {
  const start = parseDay(madeOn);
  if (start === null) return null;
  if (market === 'crypto') return isoDay(start + DAY_MS);
  let next = start + DAY_MS;
  while (!weekday(next) || EURONEXT_CLOSED.has(isoDay(next))) next += DAY_MS;   // the next session
  return isoDay(next);
}
// hh:mm Belgian time on an ISO day, as an instant (Brussels is UTC+1 or UTC+2; the clock-change nights fall on Sundays)
function brusselsAt(start, [hour, minute]) {
  for (const offset of [1, 2]) {
    const candidate = new Date(start + (hour - offset) * HOUR_MS + minute * 60000);
    const b = zoneParts(candidate, BRUSSELS);
    if (Number(b.hour) === hour && Number(b.minute) === minute) return candidate;
  }
  return null;
}
function closeMoment(market, day) {
  const start = parseDay(day);
  if (start === null) return null;
  if (market === 'crypto') return new Date(start + DAY_MS);              // 00:00 UTC of the following day
  return brusselsAt(start, EURONEXT_HALF_DAYS.has(day) ? EURONEXT_HALF_CLOSE : EURONEXT_CLOSE);   // 17:30 Belgian time, 14:00 on a half day
}
// when the game day begins: 00:00 UTC for crypto, the 09:00 Euronext open for shares
function openMoment(market, day) {
  const start = parseDay(day);
  if (start === null) return null;
  if (market === 'crypto') return new Date(start);
  return brusselsAt(start, EURONEXT_OPEN);
}
function brusselsText(date, withDate = false) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;
  const b = zoneParts(date, BRUSSELS);
  const time = `${b.hour}:${b.minute} Belgian time`;
  return withDate ? `${time} on ${b.weekday.slice(0, 3)} ${b.day}/${b.month}` : time;
}
const weekdayOf = (day) => {
  const start = parseDay(day);
  return start === null ? null : zoneParts(new Date(start + 12 * HOUR_MS), 'UTC').weekday;
};
// the last day of the week a week-game ticket is for: made after `madeOn` (a Sunday for crypto, the week's last session for shares),
// the ticket covers the following ISO week - its Sunday for crypto, its last session (normally Friday, holidays skipped) for shares
function weekEnd(market, madeOn) {
  const start = parseDay(madeOn);
  if (start === null) return null;
  let monday = start + DAY_MS;
  while (new Date(monday).getUTCDay() !== 1) monday += DAY_MS;
  if (market === 'crypto') return isoDay(monday + 6 * DAY_MS);
  for (let k = 4; k >= 0; k -= 1) {
    const day = isoDay(monday + k * DAY_MS);
    if (!EURONEXT_CLOSED.has(day)) return day;
  }
  return isoDay(monday + 4 * DAY_MS);
}
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
// how far into the week, as the week's texts say it: hours under a day ("0.5 h in", where "0 days in" read wrong
// on a share week's first afternoon), days after it ("1 day in", "3.4 days in")
function weekElapsedText(now, openAt) {
  const ms = now - openAt;
  if (ms < DAY_MS) return `${Math.round(ms / HOUR_MS * 10) / 10} h in`;
  return `${plural(Math.round(ms / DAY_MS * 10) / 10, 'day', 'days')} in`;
}
function dayStatusWeek(now, openAt, closeAt) {
  if (!(now instanceof Date) || Number.isNaN(now.getTime())) return null;
  if (now < openAt) return `the week has not started yet - it starts at ${brusselsText(openAt, true)}`;
  if (now < closeAt) return `the week is running, ${weekElapsedText(now, openAt)}; it closes at ${brusselsText(closeAt, true)}`;
  return `the week has closed (${brusselsText(closeAt, true)})`;
}
// where the predicted day stands at `now`
function dayStatus(market, day, now) {
  const open = openMoment(market, day);
  const close = closeMoment(market, day);
  if (!open || !close || !(now instanceof Date) || Number.isNaN(now.getTime())) return null;
  const what = market === 'crypto' ? `the UTC day ${day}` : `the Euronext session of ${weekdayOf(day)} ${day}`;
  if (now < open) {
    return market === 'crypto' ? `${what} has not started yet` : `${what} has not opened yet - it opens at ${brusselsText(open)} and closes at ${brusselsText(close)}`;
  }
  if (now < close) {
    const hours = Math.round((now - open) / HOUR_MS * 10) / 10;
    return `${what} is running now, ${hours} hours in; it closes at ${brusselsText(close)}`;
  }
  return `${what} has already closed (${brusselsText(close)}); the verdict appears here after ${market === 'crypto' ? "the next morning's run" : "that evening's run"}`;
}

module.exports = { MARKETS, runName, loadMarket, loadRows, loadRegimes, describeMarket, describeRows, describeRegimes, describeDays, describeTrading,
  pickPlan, pickWeek, MIN_PLAN_TRADES, MIN_WEEK_TRADES, dayStatusWeek, describeFollowed, describeGuarded, GUARD_SHOWN_FROM,
  describeOrders, ORDERS_SHOWN_FROM, ORDER_VARIANT_PRIMARY,
  nextGameDay, closeMoment, openMoment, brusselsText, dayStatus, EURONEXT_CLOSED, EURONEXT_HALF_DAYS, EURONEXT_PUBLISHED_YEARS, WEEK_GAMES, weekEnd, MIN_RANK_DAYS };
