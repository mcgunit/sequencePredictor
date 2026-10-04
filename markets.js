// markets.js - the Crypto and Shares pages (README roadmap item 4, phase M2).
//
// Predictor.py tracks the two markets as positional games, and after each
// run src/MarketSettle.py settles every stored day against the real returns
// and writes data/markets/<market>.json: closes with, per date, the game
// day's edges and return and every model's predicted bin, the next day's
// predictions as prices, per-model accuracy against chance and the daily
// accuracy series. These pages draw that file and
// nothing else - no database, no Python at request time.
//
// Two more records join the page (phase M3): data/controls/markets/<market>-rows.json,
// the weekly MarketRows.py report that scores the market rows under a proper
// scoring rule against GARCH, and data/markets/<market>-regimes.json, the
// Regime HMM rows' reading of which regime the market is in.
//
// Pure functions (loadMarket, describeMarket, formatting) are exported for
// test/markets.test.js; install() adds the two routes.
'use strict';

const fs = require('fs');
const path = require('path');

const MARKETS = {
  crypto: { title: 'Crypto predictor', noun: 'coin', unit: 'USDT', calendar: 'every day (UTC)' },
  shares: { title: 'Shares predictor', noun: 'share', unit: 'USD', calendar: 'every trading day (New York)' },
};

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
const pct = (x, digits = 1) => (num(x) === null ? '-' : `${(num(x) * 100).toFixed(digits)}%`);
const money = (x, digits = 4) => (num(x) === null ? '-' : (num(x) >= 0 ? '+' : '') + num(x).toFixed(digits));
const price = (x) => {
  const v = num(x);
  if (v === null) return '-';
  return v >= 1000 ? v.toFixed(0) : (v >= 10 ? v.toFixed(2) : v.toFixed(4));
};

// The weekly report's rows, ready to print: log-scores, the interval and
// verdict against GARCH and against uniform, hit rates and the fixed rule's
// P&L with the real returns. Sorted best log-score first, rows without a
// probability (the baselines) last.
function describeRows(record) {
  if (!record) return null;
  const uniform = num(record.uniform_log_score);
  const interval = (x) => (x && num(x.mean) !== null ? { mean: num(x.mean), lo: num(x.lo), hi: num(x.hi), verdict: String(x.verdict || ''), days: num(x.days) } : null);
  const rows = (record.rows || []).map((r) => ({
    name: String(r.name), kind: String(r.kind || 'base'), days: num(r.days), scoredDays: num(r.scored_days),
    logScore: num(r.log_score), se: num(r.log_score_se),
    vsReference: interval(r.vs_reference), vsUniform: interval(r.vs_uniform),
    exact: num(r.exact_rate), adjacent: num(r.adjacent_rate), direction: num(r.direction_rate),
    trades: num(r.trades), pnl: num(r.pnl_return), pnlPerTrade: num(r.pnl_per_trade),
    isReference: String(r.name) === String(record.reference),
  }));
  rows.sort((a, b) => {
    if ((a.logScore === null) !== (b.logScore === null)) return a.logScore === null ? 1 : -1;
    if (a.logScore !== null && b.logScore !== null && a.logScore !== b.logScore) return b.logScore - a.logScore;
    return (b.exact || 0) - (a.exact || 0);
  });
  const reference = rows.find((r) => r.isReference) || null;
  return {
    market: record.market, generatedAt: record.generated_at || null, days: num(record.days_scored), requested: num(record.days_requested),
    firstDay: record.first_day || null, lastDay: record.last_day || null, uniform, reference: String(record.reference || ''),
    referenceRow: reference, floor: num(record.probability_floor), rows, lockboxWithheld: num(record.lockbox_days_withheld) || 0,
    betterThanReference: rows.filter((r) => r.vsReference && r.vsReference.verdict === 'better').map((r) => r.name),
    referenceAboveUniform: !!(reference && reference.vsUniform && reference.vsUniform.verdict === 'better'),
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
    openRule: t.open_rule ? String(t.open_rule) : '',
    shortRule: t.short_rule ? String(t.short_rule) : '',
    dates: Array.isArray(t.dates) ? t.dates.map(String) : [], models: seriesMap(t.models), benchmark: series(t.benchmark),
    hold: { models: seriesMap(hold.models), benchmark: series(hold.benchmark) },
    open: open ? { models: seriesMap(open.models), benchmark: series(open.benchmark) } : null,
    // the same rules with shorts on (MarketSettle.ledger "short"): money per model per rule; the market lines above are their benchmark
    short: t.short && typeof t.short === 'object' ? Object.fromEntries(['daily', 'hold', 'open'].filter((r) => t.short[r] && typeof t.short[r] === 'object').map((r) => [r, { models: seriesMap(t.short[r].models), sold: series(t.short[r].benchmark) }])) : null,
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
      exact: num(m.exact), adjacent: num(m.adjacent), direction: num(m.direction), positions: num(m.positions), pnl: num(m.pnl), pnlCash: num(m.pnl_cash), trades: num(m.trades),
    })),
    best: d.best || null, exactMean: num(d.exact_mean),
  }));
}

// A bin's return interval under the day's edges: [low, high] with null at the
// open ends (MarketGame.bin_interval).
function binInterval(bin, edges) {
  if (bin === null || !edges.length) return [null, null];
  return [bin > 0 ? edges[bin - 1] : null, bin < edges.length ? edges[bin] : null];
}
const signedPct = (x, digits = 1) => {
  if (num(x) === null) return '-';
  const text = (num(x) * 100).toFixed(digits);
  if (Number(text) === 0) return `${Math.abs(Number(text)).toFixed(digits)}%`;   // never "-0.0%"
  return `${num(x) >= 0 ? '+' : ''}${text}%`;
};
function intervalText(bin, edges, digits = 1) {
  const [low, high] = binInterval(bin, edges);
  if (low === null && high === null) return '-';
  if (low === null) return `below ${signedPct(high, digits)}`;
  if (high === null) return `above ${signedPct(low, digits)}`;
  return `${signedPct(low, digits)} to ${signedPct(high, digits)}`;
}
// The bin a move would fall in under a day's edges (MarketGame.bin_of).
const binOfMove = (move, edges) => edges.filter((e) => e <= move).length;
// The day file behind a game day, as Predictor.py names it (no zero padding).
function gameViewLink(market, date) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(date));
  return m ? `/database/${market}/${m[1]}-${Number(m[2])}-${Number(m[3])}.json` : `/database/${market}`;
}

// --- page -------------------------------------------------------------------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// JSON for an inline <script>: a '<' in a model name must not end the script block
const jsonScript = (value) => JSON.stringify(value).replace(/</g, '\\u003c');

// --- when a call is judged ------------------------------------------------
// A ticket is made by the daily run (09:00 Belgian time) after the newest
// candle has closed, for the game day after that candle; the day file carries
// the newest candle's date (made_on) and its write time (made_at), so the day
// a ticket is for is the next game day, and the moment it is judged is that
// day's close. The reader asked (4 Oct 2026) when a position has to be closed
// to meet the chart's "next" point: these helpers put that moment on the page
// in Belgian time. Crypto's day is the UTC day (close at 00:00 UTC, which is
// the next calendar day in Belgium); a share's day is the New York session
// (09:30-16:00 there - the Belgian hours differ in the few weeks a year when
// only one side of the Atlantic has changed its clocks, so every hour on the
// page is computed for its date, never hard-coded). Reviewed by three
// readers on 4 Oct 2026.
const BRUSSELS = 'Europe/Brussels';
const NEW_YORK = 'America/New_York';
const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;
const ISO_DAY = /^(\d{4})-(\d{2})-(\d{2})$/;
const ISO_INSTANT = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})$/;
const isoDay = (ms) => new Date(ms).toISOString().slice(0, 10);
const parseDay = (text) => {
  const m = ISO_DAY.exec(String(text || ''));
  if (!m) return null;
  const ms = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  return isoDay(ms) === m[0] ? ms : null;        // Date.UTC rolls 2026-02-30 over to March; refuse it instead
};
const parseInstant = (text) => {
  if (!ISO_INSTANT.test(String(text || ''))) return null;   // an offset-less or date-only string would be read in the server's zone
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? null : date;
};
const zoneParts = (date, zone) => {
  const parts = {};
  new Intl.DateTimeFormat('en-GB', { timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
                                     weekday: 'long', hour12: false, timeZoneName: 'short' }).formatToParts(date).forEach((p) => { parts[p.type] = p.value; });
  parts.hour = parts.hour === '24' ? '00' : parts.hour;
  return parts;
};
// NYSE Group's published calendar (ir.theice.com, "Holiday and Early Closings
// Calendar"): the days the exchange is closed and the days it closes at 13:00
// New York time, for the years it has announced. A year outside the table
// gets the weekday rule and the word "normally" on the page.
const NYSE_CLOSED = new Set([
  '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
  '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31', '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
]);
const NYSE_EARLY_CLOSE = new Set(['2026-11-27', '2026-12-24', '2027-11-26']);
const NYSE_YEARS = new Set([...NYSE_CLOSED].map((d) => d.slice(0, 4)));
const calendarKnown = (market, day) => market === 'crypto' || (typeof day === 'string' && NYSE_YEARS.has(day.slice(0, 4)));
function nextGameDay(market, madeOn) {
  const start = parseDay(madeOn);
  if (start === null) return null;
  if (market === 'crypto') return isoDay(start + DAY_MS);
  let next = start + DAY_MS;
  while ([0, 6].includes(new Date(next).getUTCDay()) || NYSE_CLOSED.has(isoDay(next))) next += DAY_MS;   // the next session
  return isoDay(next);
}
function closeMoment(market, day) {
  const start = parseDay(day);
  if (start === null) return null;
  if (market === 'crypto') return new Date(start + DAY_MS);              // 00:00 UTC of the following day
  // 16:00 New York on that date (13:00 on an early-close day): the UTC hour depends on New York's clocks
  const closeHour = NYSE_EARLY_CLOSE.has(day) ? '13' : '16';
  for (const hour of [20, 21, 17, 18, 19, 22]) {
    const candidate = new Date(start + hour * HOUR_MS);
    if (zoneParts(candidate, NEW_YORK).hour === closeHour) return candidate;
  }
  return null;
}
// when the game day begins: 00:00 UTC for crypto, the 09:30 New York open for shares (6.5 h before the close, 3.5 h on an early-close day)
function openMoment(market, day) {
  const start = parseDay(day);
  if (start === null) return null;
  if (market === 'crypto') return new Date(start);
  const close = closeMoment(market, day);
  return close ? new Date(close.getTime() - (NYSE_EARLY_CLOSE.has(day) ? 3.5 : 6.5) * HOUR_MS) : null;
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
// the moment a call is judged, as a short clause and as a full sentence part
function closeShort(market, day) {
  const moment = closeMoment(market, day);
  if (!moment) return null;
  if (market === 'crypto') {
    const b = zoneParts(moment, BRUSSELS);
    return `${b.hour}:${b.minute} Belgian time in the night from ${weekdayOf(day)} to ${b.weekday} (00:00 UTC on ${isoDay(moment.getTime())})`;
  }
  const early = NYSE_EARLY_CLOSE.has(day);
  return `${brusselsText(moment)} on ${weekdayOf(day)} ${day} (${early ? '13:00 New York time, an early close' : '16:00 New York time'})`;
}
function closeText(market, day) {
  const short = closeShort(market, day);
  if (!short) return null;
  return market === 'crypto' ? `the close of ${day}, which is ${short}` : `the New York close of ${day}, which is ${short}`;
}
function openText(market, day) {
  const moment = openMoment(market, day);
  if (!moment || market === 'crypto') return null;
  return `${brusselsText(moment)} on ${weekdayOf(day)} ${day} (09:30 New York time)`;
}
// when the newest ticket was written (the day file's time), in Belgian time
function appearedText(madeAt) {
  const date = parseInstant(madeAt);
  return date ? brusselsText(date, true) : null;
}
// hours from the start of the predicted day to the ticket, one decimal; null when unknown or negative
function hoursInto(madeAt, market, day) {
  const made = parseInstant(madeAt);
  const start = openMoment(market, day);
  if (!made || !start || made < start) return null;
  return Math.round((made - start) / HOUR_MS * 10) / 10;
}
// where the predicted day stands at the moment the page is rendered
function dayStatus(market, day, now) {
  const open = openMoment(market, day);
  const close = closeMoment(market, day);
  if (!open || !close || !(now instanceof Date) || Number.isNaN(now.getTime())) return null;
  const what = market === 'crypto' ? `the UTC day ${day}` : `the New York session of ${weekdayOf(day)} ${day}`;
  if (now < open) {
    return market === 'crypto' ? `${what} has not started yet` : `${what} has not opened yet - it opens at ${brusselsText(open)} and closes at ${brusselsText(close)}`;
  }
  if (now < close) {
    const hours = Math.round((now - open) / HOUR_MS * 10) / 10;
    return `${what} is running now, ${hours} hours in; it closes at ${brusselsText(close)}`;
  }
  return `${what} has already closed (${brusselsText(close)}); the verdict appears here after the next morning's run`;
}
const COLOURS = ['#e67e22', '#8e44ad', '#16a085', '#2980b9', '#d35400', '#27ae60', '#7f8c8d', '#f39c12', '#1abc9c', '#9b59b6',
  '#34495e', '#e84393', '#00a8ff', '#44bd32', '#8c7ae6', '#e1b12c'];

// The chart client, one copy per page: a pure model builder (what the tests
// run in Node) and the glue that draws it. Three views per instrument:
//   lines  (default - the owner's preferred picture) the close as a line and,
//          per switched-on model, a dashed line through the prices its bins
//          stood for; a "roughly flat" call makes this a copy of the close
//          line one day late, which the explainer card says in words;
//   bars   the day's call as a bar from the previous game day's close to the
//          price its bin stood for (green up, red down) over a paler bar for
//          the bin's whole interval - the same calls drawn so that "flat"
//          reads as a short bar instead of a lag;
//   moves  the real move per day as a bar (percent) and, per model, the bin's
//          interval as a paler floating bar with the call's middle as a dot -
//          the view in which skill, or its absence, is visible.
// A bin becomes a return the way MarketGame does it: the interval's midpoint,
// and for the two open bins the edge moved outward by half the neighbouring
// bin's width (representative_return); the base of a day is close x exp(-move),
// the previous GAME day's close.
const CHART_CLIENT_JS = `
window.marketCharts = window.marketCharts || {}; window.marketData = window.marketData || {}; window.marketState = window.marketState || {};
function marketInterval(bin, edges) { return [bin > 0 ? edges[bin - 1] : null, bin < edges.length ? edges[bin] : null]; }
function marketRepresentative(bin, edges) {
  var iv = marketInterval(bin, edges), lo = iv[0], hi = iv[1];
  if (lo !== null && hi !== null) return 0.5 * (lo + hi);
  var width = edges.length > 1 ? (lo === null ? edges[1] - edges[0] : edges[edges.length - 1] - edges[edges.length - 2]) : 0;
  return lo === null ? edges[0] - 0.5 * width : edges[edges.length - 1] + 0.5 * width;
}
function marketAlpha(hex, a) { var n = parseInt(hex.slice(1), 16); return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')'; }
function marketPrice(v) { return v >= 1000 ? v.toFixed(0) : (v >= 10 ? v.toFixed(2) : v.toFixed(4)); }
var MARKET_REAL = 'rgba(44,62,80,0.75)';
function marketChartModel(id, view, enabled) {
  var d = window.marketData[id]; var n = d.labels.length; var last = n - 1;   // the last label is 'next'
  var datasets = [];
  if (d.kind === 'ledger') {   // the paper-trading book under one rule: cumulative money per model, the market as the dashed grey line
    var RULE_WORDS = { daily: ['market, bought every day', 'daily round trip'], hold: ['market, buy and hold', 'held while up'], open: ['market, open to close every day', 'open to close'],
                       daily_ls: ['market, bought every day', 'daily round trip, long and short', 'market, sold every day'], hold_ls: ['market, buy and hold', 'held while the call holds, long and short', 'market, sold and held'], open_ls: ['market, open to close every day', 'open to close, long and short', 'market, sold at the open every day'] };
    var base = String(view).replace('_ls', '');
    var rule = d.rules[view] ? view : (d.rules[base] ? base : 'daily'); var book = d.rules[rule]; var words = RULE_WORDS[rule] || RULE_WORDS.daily;
    datasets.push({ type: 'line', label: words[0], data: book.benchmark, borderColor: '#7f8c8d', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0, tension: 0, order: 1, legendColour: '#7f8c8d' });
    if (book.sold && words[2]) datasets.push({ type: 'line', label: words[2], data: book.sold, borderColor: '#b0b7bc', borderDash: [2, 4], borderWidth: 1.5, pointRadius: 0, tension: 0, order: 1, legendColour: '#b0b7bc' });
    enabled.forEach(function (model) {
      var colour = d.colours[model] || '#7f8c8d';
      datasets.push({ type: 'line', label: model, data: book.series[model] || [], borderColor: colour, backgroundColor: colour, borderWidth: 2, pointRadius: 0, tension: 0, order: 0, legendColour: colour, model: model });
    });
    return { datasets: datasets, yTitle: 'cumulative P&L, ' + d.currency + ' (' + d.stake + ' per position, ' + words[1] + ')', percent: false, money: true };
  }
  var call = function (model, i) {   // {base, rep, lo, hi} in return space for label i, or null
    if (i === last) {
      var nx = d.next[model]; if (!nx || nx.price === null || d.lastClose === null) return null;
      return { base: d.lastClose, rep: Math.log(nx.price / d.lastClose), lo: nx.low === null ? null : Math.log(nx.low / d.lastClose), hi: nx.high === null ? null : Math.log(nx.high / d.lastClose) };
    }
    var bins = d.course[model]; var bin = bins ? bins[i] : null; var edges = d.edges[i];
    if (bin === null || bin === undefined || !edges || d.closes[i] === null || d.moves[i] === null) return null;
    var iv = marketInterval(bin, edges);
    return { base: d.closes[i] * Math.exp(-d.moves[i]), rep: marketRepresentative(bin, edges), lo: iv[0], hi: iv[1] };
  };
  // an open-ended bin ("everything below -3.1%") runs to the edge of what the chart shows: the extent of the moves and calls drawn, padded
  var lo = Infinity, hi = -Infinity;
  var see = function (v) { if (v !== null && v !== undefined && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } };
  d.moves.forEach(see);
  enabled.forEach(function (model) { for (var i = 0; i < n; i++) { var c = call(model, i); if (c) { see(c.rep); see(c.lo); see(c.hi); } } });
  if (!isFinite(lo)) { lo = -0.01; hi = 0.01; }
  var pad = 0.1 * (hi - lo) || 0.01; var floor = lo - pad, ceil = hi + pad;
  var bandOf = function (c) { return [c.lo === null ? floor : c.lo, c.hi === null ? ceil : c.hi]; };
  var bar = function (extra) { return Object.assign({ type: 'bar', grouped: false, categoryPercentage: 0.9 }, extra); };   // grouped:false - every bar centred on its date, so a call sits on its band
  if (view === 'lines') {   // the course: the close and, per model, a dashed line through the prices its bins stood for
    datasets.push({ type: 'line', label: d.symbol + ' close', data: d.closes, borderColor: '#2c3e50', borderWidth: 2, pointRadius: 0, tension: 0.1, order: 0, legendColour: '#2c3e50' });
    enabled.forEach(function (model) {
      var colour = d.colours[model] || '#7f8c8d'; var line = [];
      for (var i = 0; i < n; i++) { var c = call(model, i); line.push(c ? c.base * Math.exp(c.rep) : null); }
      datasets.push({ type: 'line', label: model, data: line, borderColor: colour, backgroundColor: colour, borderDash: [3, 3], borderWidth: 1, pointRadius: 2, tension: 0.1, order: 1, legendColour: colour, model: model });
    });
    return { datasets: datasets, yTitle: d.quote, percent: false };
  }
  if (view === 'bars') {   // the calls: a bar per day from the previous game day's close to the bin's price, over the bin's interval
    datasets.push({ type: 'line', label: d.symbol + ' close', data: d.closes, borderColor: '#2c3e50', borderWidth: 2, pointRadius: 0, tension: 0.1, order: 0, legendColour: '#2c3e50' });
    enabled.forEach(function (model) {
      var colour = d.colours[model] || '#7f8c8d'; var band = [], step = [], fills = [];
      for (var i = 0; i < n; i++) {
        var c = call(model, i);
        if (!c) { band.push(null); step.push(null); fills.push('rgba(0,0,0,0)'); continue; }
        var b = bandOf(c);
        step.push([c.base, c.base * Math.exp(c.rep)]); fills.push(c.rep >= 0 ? '#27ae60' : '#c0392b');
        band.push([c.base * Math.exp(b[0]), c.base * Math.exp(b[1])]);
      }
      datasets.push(bar({ label: model + ' - bin interval', data: band, backgroundColor: marketAlpha(colour, 0.22), borderWidth: 0, order: 3, legendHidden: true, barPercentage: 0.95 }));
      datasets.push(bar({ label: model, data: step, backgroundColor: fills, borderColor: colour, borderWidth: 1, order: 2, barPercentage: 0.45, legendColour: colour, model: model }));
    });
    return { datasets: datasets, yTitle: d.quote, percent: false };
  }
  var real = d.moves.map(function (m) { return m === null ? null : m * 100; });
  datasets.push(bar({ label: 'real move', data: real, backgroundColor: MARKET_REAL, order: 2, barPercentage: 0.5, legendColour: MARKET_REAL }));
  enabled.forEach(function (model) {
    var colour = d.colours[model] || '#7f8c8d'; var band = [], dots = [];
    for (var i = 0; i < n; i++) {
      var c = call(model, i);
      if (!c) { band.push(null); dots.push(null); continue; }
      var b = bandOf(c);
      dots.push(c.rep * 100); band.push([b[0] * 100, b[1] * 100]);
    }
    datasets.push(bar({ label: model + ' - bin interval', data: band, backgroundColor: marketAlpha(colour, 0.22), borderWidth: 0, order: 3, legendHidden: true, barPercentage: 0.95 }));
    datasets.push({ type: 'line', label: model, data: dots, borderColor: colour, backgroundColor: colour, showLine: false, pointRadius: 3, pointHoverRadius: 5, order: 1, legendColour: colour, model: model });
  });
  return { datasets: datasets, yTitle: '% move, close to close', percent: true };
}
function marketEnabled(id) {
  return Array.prototype.slice.call(document.querySelectorAll('input[data-chart="' + id + '"]')).filter(function (b) { return b.checked; }).map(function (b) { return b.getAttribute('data-model'); });
}
function marketTooltip(percent, money) {
  var one = function (v) { return percent ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : (money ? (v >= 0 ? '+' : '') + v.toFixed(2) : marketPrice(v)); };
  return { filter: function (item) { return item.raw !== null && item.raw !== undefined && !item.dataset.legendHidden; },
    callbacks: { label: function (item) { var r = item.raw; return item.dataset.label + ': ' + (Array.isArray(r) ? one(r[0]) + ' to ' + one(r[1]) : one(r)); } } };
}
function marketLegend(id) {
  return {
    labels: { boxWidth: 12,
      generateLabels: function (chart) {   // a bar's default swatch is its first element's colour, which is the transparent "no call" of the window's first day
        var items = Chart.defaults.plugins.legend.labels.generateLabels(chart);
        items.forEach(function (it) { var ds = chart.data.datasets[it.datasetIndex]; if (ds && ds.legendColour) { it.fillStyle = ds.legendColour; it.strokeStyle = ds.legendColour; } });
        return items;
      },
      filter: function (item, data) { return !data.datasets[item.datasetIndex].legendHidden; } },
    onClick: function (e, item, legend) {   // a model's legend entry switches the model off (band and call together); the close and the real move toggle as usual
      var ds = legend.chart.data.datasets[item.datasetIndex];
      if (ds && ds.model) {
        document.querySelectorAll('input[data-chart="' + id + '"]').forEach(function (b) { if (b.getAttribute('data-model') === ds.model) b.checked = false; });
        marketRender(id);
      } else { Chart.defaults.plugins.legend.onClick.call(this, e, item, legend); }
    } };
}
function marketRender(id) {
  var d = window.marketData[id]; if (!d) return;
  var state = window.marketState[id];
  if (!state) {   // open on the newest month: a year of 1 px bars is the close line alone
    var count = d.labels.length;
    state = window.marketState[id] = { view: d.kind === 'ledger' ? 'daily' : 'lines', min: count > 31 ? d.labels[count - 31] : undefined, max: count > 31 ? d.labels[count - 1] : undefined };
  }
  var old = window.marketCharts[id];
  if (old) { try { state.min = old.options.scales.x.min; state.max = old.options.scales.x.max; old.destroy(); } catch (e) { /* a dead chart is replaced anyway */ } }
  var model = marketChartModel(id, state.view, marketEnabled(id));
  var scales = { x: { ticks: { maxTicksLimit: 10 }, stacked: false }, y: { title: { display: true, text: model.yTitle }, stacked: false } };
  if (!model.percent) scales.y.beginAtZero = false;   // the bar controller's scale override would otherwise start the price axis at 0
  if (state.min !== undefined) scales.x.min = state.min;
  if (state.max !== undefined) scales.x.max = state.max;
  if (model.percent || model.money) scales.y.grid = { color: function (ctx) { return ctx.tick && ctx.tick.value === 0 ? '#2c3e50' : 'rgba(0,0,0,0.08)'; } };
  var zoom = window.ChartZoom ? { zoom: { wheel: { enabled: true }, pinch: { enabled: true }, drag: { enabled: false }, mode: 'x' }, pan: { enabled: true, mode: 'x' } } : undefined;
  window.marketCharts[id] = new Chart(document.getElementById(id).getContext('2d'), { type: 'line', data: { labels: d.labels, datasets: model.datasets },
    options: { maintainAspectRatio: false, spanGaps: true, interaction: { mode: 'index', intersect: false },
      plugins: { legend: marketLegend(id), tooltip: marketTooltip(model.percent, model.money), zoom: zoom }, scales: scales } });
  document.querySelectorAll('button[data-view-for="' + id + '"]').forEach(function (b) {
    var current = String(state.view), shorts = current.indexOf('_ls') >= 0, baseView = current.replace('_ls', '');
    var on = b.hasAttribute('data-shorts') ? ((b.getAttribute('data-shorts') === 'on') === shorts) : b.getAttribute('data-view') === baseView;
    b.style.background = on ? '#2c3e50' : 'white'; b.style.color = on ? 'white' : '#2c3e50';
  });
}
function marketView(id, view) { window.marketState[id] = window.marketState[id] || { view: 'lines' }; window.marketState[id].view = view; marketRender(id); }
// the book's rule buttons keep the shorts switch, and the switch keeps the rule
function marketRule(id, rule) { var s = window.marketState[id] || { view: 'daily' }; marketView(id, rule + (String(s.view).indexOf('_ls') >= 0 ? '_ls' : '')); }
function marketShorts(id, on) { var s = window.marketState[id] || { view: 'daily' }; marketView(id, String(s.view).replace('_ls', '') + (on ? '_ls' : '')); }
function marketToggle(id) { marketRender(id); }
function marketModels(id, which) {
  var d = window.marketData[id];
  document.querySelectorAll('input[data-chart="' + id + '"]').forEach(function (b) { b.checked = which === 'all' ? true : (which === 'none' ? false : b.getAttribute('data-model') === d.best); });
  marketRender(id);
}
function marketRange(id, days) {
  var chart = window.marketCharts[id]; if (!chart) return;
  var labels = chart.data.labels; var x = chart.options.scales.x; var state = window.marketState[id] || (window.marketState[id] = { view: 'lines' });
  if (typeof chart.resetZoom === 'function') chart.resetZoom('none');
  if (days > 0 && labels.length > days) { x.min = labels[labels.length - 1 - days]; x.max = labels[labels.length - 1]; } else { delete x.min; delete x.max; }
  state.min = x.min; state.max = x.max; chart.update();
}
function marketReset(id) { marketRange(id, 0); }
`;

const CHART_SCRIPTS = `<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<style>.container { max-width: 1400px; } .chart-tools { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:8px; font-size:0.85em; }
.chart-tools button { padding:4px 10px; border:1px solid #ccd1d6; border-radius:4px; background:white; color:#2c3e50; cursor:pointer; } .chart-tools button:hover { background:#f1f3f5; }
.chart-tools .hint { color:#7f8c8d; margin-left:auto; } .chart-tools .sep { color:#ccd1d6; margin:0 4px; }
.chart-models { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:0.82em; margin:4px 0 10px; color:#2c3e50; } .chart-models label { cursor:pointer; white-space:nowrap; }
.chart-models .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; } .chart-models a { color:#2980b9; margin-right:8px; }
.bg-section { border-top:1px solid #eee; padding:4px 0; } .bg-section:first-child { border-top:none; } .bg-section summary { cursor:pointer; padding:8px 0; list-style-position:inside; }
.bg-section summary b { font-size:1.05em; } .bg-section summary .card-meta { margin-left:10px; } .bg-body { padding:2px 0 12px; }
.plan td, .plan th { vertical-align:top; } .plan td.order { text-align:left; font-size:0.92em; }</style>
<script>
  try { if (window.ChartZoom) Chart.register(window.ChartZoom); } catch (e) { /* zoom stays off, the chart still draws */ }
${CHART_CLIENT_JS}
</script>`;
const RANGES = [['1M', 30], ['3M', 91], ['6M', 182], ['1Y', 365], ['All', 0]];
const MIN_RANK_DAYS = 10;   // MarketSettle.MIN_RANK_DAYS: a row with fewer settled days ranks after the others and cannot head a book

function page(market, view, header, footer, user, now = new Date()) {
  const meta = MARKETS[market];
  let html = header(meta.title, user);
  html += CHART_SCRIPTS;
  html += `<h1>${esc(meta.title)}</h1>`;
  if (!view) {
    html += `<p style="color:#7f8c8d;">No market record yet. It appears after the first daily run that includes the <code>${esc(market)}</code>
      game (the predictor fetches the bars, cuts the return bins, predicts, and the settlement writes <code>data/markets/${esc(market)}.json</code>).</p>`;
    return html + footer();
  }
  const c = view.chance;
  html += `<p style="color:#7f8c8d; margin-top:-12px;">Same models as the lottery games, same daily tracking, same controls - a predictor, not a trading bot.
    Each ${esc(meta.noun)}'s next-day return is cut into ${view.k} equiprobable bins fitted on its own past (see README, roadmap item 4), and every model
    predicts one bin per ${esc(meta.noun)}, ${esc(meta.calendar)}. A bin is a return interval, so it is drawn as a predicted price on the chart, with the price band it stands for in the table beneath.
    Chance is ${pct(c.exact, 0)} for the exact bin, ${pct(c.adjacent, 0)} for the adjacent bin and ${pct(c.direction, 0)} for the direction.
    <b>Paper trading</b> scores every up call (a predicted bin in the upper half) in money${view.trading && view.trading.short ? ' - and, with the shorts switch, every down call as a paper short, which is never an order' : ''}, ${view.trading && view.trading.stake !== null ? view.trading.stake : 100} ${esc(meta.unit)} a position and a ${view.trading && view.trading.feePerLeg !== null ? pct(view.trading.feePerLeg, 2) : '0.10%'} fee on each leg,
    ${view.trading && view.trading.open ? 'three ways (Paper trading card); <b>Today\'s plan</b> turns the one a reader can follow - buy at the open, sell at the close - into orders' : 'two ways (Paper trading card); <b>Today\'s plan</b> turns the calls into orders'}.
    Results settle the morning after, when the day's candle has closed. ${market === 'crypto'
      ? 'A crypto day is the UTC day, so "the day\'s close" is 00:00 UTC - 02:00 Belgian time in the night that follows (01:00 in winter); that close decides every call.'
      : 'A share\'s day is the New York session, so "the day\'s close" is 16:00 New York time - 22:00 Belgian time in most weeks; that close decides every call.'}</p>`;

  // The page, since 4 Oct 2026 (the owner found seven cards heavy): the plan
  // (the orders that match how a call is scored), the paper book, a chart
  // per instrument, and ONE collapsed Background card holding the six
  // supporting sections as <details> - the explainer, the models table,
  // accuracy per day, day by day, the proper score, the regime reading. Each
  // section is still built as a card below and converted by asDetail, so
  // their content and tests are unchanged.
  let bg = '', paper = '', charts = '';
  const asDetail = (card, open = false) => {
    const m = /^<div class="card(?: expanded)?"><div class="card-header" onclick="toggleCard\(this\)"><div><span class="card-title">([\s\S]*?)<\/span>\s*<span class="card-meta"[^>]*>([\s\S]*?)<\/span><\/div><div class="card-icon">▼<\/div><\/div>\s*<div class="card-body"([^>]*)>([\s\S]*)<\/div><\/div>$/.exec(card.trim());
    if (!m) return card;
    return `<details class="bg-section"${open ? ' open' : ''}><summary><b>${m[1]}</b><span class="card-meta">${m[2]}</span></summary><div class="bg-body"${m[3]}>${m[4]}</div></details>`;
  };

  // how a day becomes a draw - for a newcomer, in three steps, with the newest
  // settled day as the worked example and a strip per instrument showing what
  // a bin looks like. Wording reviewed by two reader personas on 1 Oct 2026.
  const example = view.days.length ? view.days[0] : null;
  const half = view.k / 2;
  const first = example && example.instruments.length ? example.instruments[0] : null;
  const closeBefore = (symbol, date) => {
    const inst = view.instruments.find((i) => i.symbol === symbol);
    if (!inst) return null;
    const idx = inst.closes.findIndex((p) => p[0] === date);
    return idx > 0 ? inst.closes[idx - 1][1] : null;
  };
  const closeOn = (symbol, date) => {
    const inst = view.instruments.find((i) => i.symbol === symbol);
    const point = inst ? inst.closes.find((p) => p[0] === date) : null;
    return point ? point[1] : null;
  };
  const candle = first ? { before: closeBefore(first.symbol, example.date), on: closeOn(first.symbol, example.date) } : null;
  const best = example ? (example.models.find((m) => m.name === example.best) || example.models[0] || null) : null;
  const dayWord = meta.calendar.replace(/ \(.*\)$/, '').replace(/^every /, '');      // "day" / "trading day"
  const nouns = `${esc(meta.noun)}s`;
  const appeared = appearedText(view.madeAt);
  const forDay = nextGameDay(market, view.madeOn);
  const forJudged = forDay ? closeText(market, forDay) : null;
  const hoursOld = forDay ? hoursInto(view.madeAt, 'crypto', forDay) : null;
  const closeNote = market === 'crypto'
    ? `A crypto day runs midnight to midnight UTC - 02:00 Belgian time in summer, 01:00 in winter - so when the ticket went up the day it predicts was already ${hoursOld !== null ? `${hoursOld} hours` : 'eight to ten hours'} old${hoursOld !== null ? '' : ', depending on the season and on how long the run took'}; none of those hours is used, because the models read closed daily candles only, and the day's close is still unknown`
    : `The New York session runs 09:30-16:00 there, 15:30-22:00 Belgian time in most weeks (an hour earlier in the few weeks a year when only one side of the Atlantic has changed its clocks), so a trading day's ticket is on this page hours before New York's regular session opens, and a Monday's ticket is made on the Saturday, from Friday's close`;

  // step 1: two closes, one number
  let step1 = `<p><b>1. Two closes, one number.</b> On a price chart each ${esc(dayWord)} is drawn as one candle, and a candle is four prices: where the price opened, the highest
    and lowest it reached, and where it closed. Only the last one, the <b>close</b>, is used here. The ${esc(dayWord)}'s <b>move</b> is how much the close changed since the close before it, in percent`;
  if (candle && candle.before !== null && candle.on !== null) {
    const diff = candle.on - candle.before;
    // the difference in the close's own precision (price() would print 337 as 337.00)
    const diffText = candle.before >= 1000 ? Math.round(Math.abs(diff)).toString() : (candle.before >= 10 ? Math.abs(diff).toFixed(2) : Math.abs(diff).toFixed(4));
    step1 += `: ${esc(first.symbol)}'s previous close was ${price(candle.before)}; on ${esc(example.date)} it closed at ${price(candle.on)}, ${diffText} ${diff < 0 ? 'lower' : 'higher'}, a move of <b>${signedPct(first.ret, 2)}</b>`;
  }
  step1 += `. Nothing that happened within the ${esc(dayWord)} is used. Each new candle therefore adds exactly one number to the ${esc(meta.noun)}'s record: the move from the close before it to its own close.</p>`;

  // step 2: the number goes into one of ten bins
  let compare = '';
  if (example && example.instruments.length > 1) {
    const two = Math.log(1.02);
    const placed = example.instruments.filter((i) => i.edges.length === view.k - 1).map((i) => ({ symbol: i.symbol, bin: binOfMove(two, i.edges) }));
    if (placed.length > 1) {
      const top = placed.reduce((a, b) => (b.bin > a.bin ? b : a));
      const bottom = placed.reduce((a, b) => (b.bin < a.bin ? b : a));
      if (top.bin !== bottom.bin) {
        compare = ` Each ${esc(meta.noun)} has its own bins because the same move is a different event on each: on the strips below a +2% ${esc(dayWord)} lands in ${esc(top.symbol)}'s bin ${top.bin} but only in ${esc(bottom.symbol)}'s bin ${bottom.bin}.`;
      }
    }
  }
  if (!compare) compare = ` Each ${esc(meta.noun)} has its own bins, because the same move is a different event on each - ordinary for a volatile one, large for a calm one.`;
  const step2 = `<p><b>2. The number goes into one of ${view.k} bins.</b> A bin is a <b>range of daily moves</b>, and every ${esc(meta.noun)} has its own ${view.k} ranges, cut from its own past:
    take every daily move the ${esc(meta.noun)} has had on the days all the ${nouns} of this game traded side by side - years of them - sort them from the biggest fall to the biggest rise, and
    cut the sorted list into ${view.k} equal piles. Each pile is one bin: bin 0 holds the ${esc(meta.noun)}'s worst tenth of days, bin ${view.k - 1} its best tenth, bins ${half - 1} and ${half} the days that
    barely moved. The ${view.k - 1} cut points are the <b>bin edges</b>. They shift a little as days are added, and for any given day they are cut from the days before it only, so a day never
    helps draw the lines that put it in its bin. Every bin therefore holds about one day in ${view.k}, like one face of a ${view.k}-sided die: a model that picks a bin blindly lands in the
    right one about one time in ${view.k}, and that is the bar every model has to beat.${compare}</p>`;

  // the strips: what a bin looks like, per instrument, with the real move and the best ticket
  let strips = '';
  if (example) {
    const rows = example.instruments.map((inst, slot) => {
      const predicted = best && best.bins[slot] !== undefined ? best.bins[slot] : null;
      const cells = Array.from({ length: view.k }, (_, b) => {
        const actual = inst.bin === b;
        const played = predicted === b;
        const near = played && inst.bin !== null && Math.abs(b - inst.bin) === 1;   // pale green means one bin off, as in the ticket cells below
        const style = actual ? 'background:#2c3e50; color:white; font-weight:bold;' : (near ? 'background:#d5f5e3;' : 'background:white;');
        const border = played ? 'border:2px solid #27ae60;' : 'border:1px solid #ccd1d6;';
        const title = `bin ${b}: ${intervalText(b, inst.edges, 2)}${actual ? ' - the day fell here' : ''}${played ? ` - ${best.name} predicted this` : ''}`;
        return `<div style="flex:1; min-width:0; text-align:center; padding:3px 0; font-size:0.8em; ${style} ${border}" title="${esc(title)}">${b}</div>`;
      }).join('');
      const labels = Array.from({ length: view.k }, (_, b) => `<div style="flex:1; min-width:0; text-align:right; font-size:0.65em; color:#7f8c8d; transform:translateX(50%); white-space:nowrap;">${b < inst.edges.length ? esc(signedPct(inst.edges[b])) : ''}</div>`).join('');
      return `<div style="display:flex; align-items:flex-start; gap:10px; margin:6px 0 14px;">
        <div style="min-width:150px; font-size:0.9em; padding-top:3px;"><b>${esc(inst.symbol)}</b> <span style="color:${(inst.ret || 0) >= 0 ? '#27ae60' : '#c0392b'};">${signedPct(inst.ret, 2)}</span> &rarr; bin <b>${inst.bin === null ? '-' : inst.bin}</b></div>
        <div style="flex:1; min-width:0;"><div style="display:flex; gap:1px;">${cells}</div><div style="display:flex; gap:1px; margin-right:0;">${labels}</div></div></div>`;
    }).join('');
    const ends = first && first.edges.length === view.k - 1
      ? ` - ${esc(first.symbol)}'s bin ${half - 1} runs from ${esc(signedPct(first.edges[half - 2]))} to ${esc(signedPct(first.edges[half - 1]))}, while bin 0 is everything below ${esc(signedPct(first.edges[0]))} and bin ${view.k - 1} everything above ${esc(signedPct(first.edges[view.k - 2]))}, so the two end boxes have no outer edge`
      : '';
    strips = `<p style="margin:14px 0 2px;"><b>Worked example - ${esc(example.date)}, the newest settled day</b> (settled: its closes are in and every model's prediction for it has been scored against them).
      Each strip is one ${esc(meta.noun)}'s ${view.k} bins, with the cut point between two bins written under their boundary. The boxes are drawn the same width because each holds a tenth of
      the ${esc(meta.noun)}'s past days, not because the ranges are equal${ends}. <b>Dark box</b>: where the day actually landed${best ? `. <b>Green outline</b>: what ${esc(best.name)} - one of the
      models, the one with the most exact hits that day - had predicted beforehand. When both are the same box it is a hit and the dark box carries the green outline; a predicted box next to the dark one is filled pale green (one off); further away it stays white.
      That prediction is its ticket, explained in step 3, where the same hit is shown as a solid green digit (green outlines mark the prediction only in these strips; elsewhere a solid green cell is a hit, and the charts draw each model in its own colour, the actual close in dark navy)` : ''}.</p>${rows}`;
  } else {
    strips = `<p style="color:#7f8c8d; font-size:0.9em;">The worked example - the newest day's moves on the ${view.k} bins, one strip per ${esc(meta.noun)} - appears here after the first settled day.</p>`;
  }

  // step 3: the draw, the ticket, the hit
  let step3 = `<p><b>3. The bins side by side are the draw.</b> Every ${esc(dayWord)} is one draw - a <i>game day</i>, for ${market === 'crypto' ? 'crypto simply one UTC day' : 'shares one New York session'} - with one slot per ${esc(meta.noun)}`;
  if (example) {
    const drawText = example.instruments.map((i) => (i.bin === null ? '?' : i.bin)).join(' ');
    step3 += `: the draw of ${esc(example.date)} reads <b style="letter-spacing:2px;">${esc(drawText)}</b>, which is what the <a href="${gameViewLink(market, example.date)}">game view</a> shows as digits`;
  }
  step3 += `. A model's <b>ticket</b> is one bin per ${esc(meta.noun)}, made by the daily run, which starts at 09:00 Belgian time after the previous candle has closed and puts the ticket on this page when it finishes${appeared ? ` (the newest ticket went up at ${esc(appeared)})` : ''}`;
  if (forDay && forJudged) {
    step3 += `. That ticket is for ${market === 'crypto' ? `the UTC day ${esc(forDay)}` : `the New York session of ${esc(weekdayOf(forDay))} ${esc(forDay)}`} and is judged at ${esc(forJudged)} - that close is the number every call on it is scored against`;
  }
  step3 += `. ${closeNote}`;
  if (best) {
    const cells = best.bins.map((b, i) => {
      const actual = example.instruments[i] ? example.instruments[i].bin : null;
      const hit = b !== null && actual !== null && b === actual;
      const near = !hit && b !== null && actual !== null && Math.abs(b - actual) === 1;
      return `<span style="display:inline-block; min-width:1.4em; text-align:center; padding:1px 4px; margin-right:2px; border-radius:3px; ${hit ? 'background:#2ecc71; color:white;' : (near ? 'background:#d5f5e3;' : 'background:#eee;')}">${b === null ? '?' : b}</span>`;
    }).join('');
    const settled = best.positions === null ? example.instruments.length : best.positions;
    const neighbours = (best.adjacent === null || best.exact === null) ? null : best.adjacent - best.exact;
    step3 += `. ${esc(best.name)} had played ${cells} after the previous close - ${best.exact === null ? '-' : best.exact} of ${settled} ${nouns} in the right bin (green), ${neighbours === null ? '-' : neighbours} more in a neighbouring bin (pale green), ${best.direction === null ? '-' : best.direction} on the right side of zero`;
  }
  const two = example && example.instruments.length > 1 ? example.instruments.slice(0, 2).map((i) => i.symbol) : [`one ${esc(meta.noun)}`, 'another'];
  step3 += `. A <b>hit</b> is the right bin for the right ${esc(meta.noun)}, nothing else: a 7 played on ${esc(two[0])} is judged against ${esc(two[0])}'s own bin, and ${esc(two[1])} landing in 7 does not help it.
    One bin off counts as <b>adjacent</b> (nearly right). <b>Direction</b> is whether the middle of the predicted bin has the same sign as the real move - up or down: that matters for the bin
    that straddles zero (bin ${half - 1} on most days), which counts as down when its middle is below zero, and a day that closes exactly unchanged is a miss for every model.
    Chance is 1 in ${view.k} for the exact bin, ${pct(c.adjacent, 0)} for adjacent (a blind guess spread over all ${view.k} bins: an inner bin is within one by luck ${Math.round(300 / view.k)}% of the time, an end bin ${Math.round(200 / view.k)}%)
    and about ${pct(c.direction, 0)} for direction.</p>`;

  // why bins, and that a bin is still a price range
  let why = `<p><b>Why bins and not the price itself?</b> A price is never hit exactly, so "right" would need a tolerance, say &plusmn;1% - and with a fixed band a model that always says
    "unchanged" scores a hit on every day that moves less than the band. That score would measure how calm the ${esc(meta.noun)} was, not the model, and the same &plusmn;1% would mean one thing
    for a volatile ${esc(meta.noun)} and another for a calm one. With each ${esc(meta.noun)}'s own bins any answer - always bin ${half - 1}, always bin ${view.k - 1} - is right by luck about one time in ${view.k},
    because every bin has held one day in ${view.k} of its past. About, not exactly: the edges come from the past and the day being scored is a new one, so in a calm stretch the middle bins
    come up more often than one in ${view.k}, in a wild stretch the outer ones. That tilt - how big the move will be, not which way - is the one thing about a day's move known to be somewhat
    predictable, and the GARCH row, a model that forecasts only that, sits in the table to show how much of a score is just that tilt. So the question this project asks - do these models beat
    luck, and beat a plain volatility model? - has a clean answer, read against the same controls as the lottery games.</p>
    <p style="margin-bottom:0;"><b>A bin is still a price range</b>`;
  if (first && candle && candle.before !== null && first.bin !== null && first.edges.length) {
    const [lo, hi] = binInterval(first.bin, first.edges);
    const low = lo === null ? null : candle.before * Math.exp(lo);
    const high = hi === null ? null : candle.before * Math.exp(hi);
    const straddles = lo !== null && hi !== null && lo < 0 && hi > 0;
    const middle = straddles ? (lo + hi) / 2 : null;
    why += `: ${esc(first.symbol)}'s bin ${first.bin} on ${esc(example.date)} ran from ${esc(intervalText(first.bin, first.edges, 2))}${straddles ? ` (this bin straddles zero; its middle sits a hair ${middle < 0 ? 'below' : 'above'} zero, so for direction it counts as ${middle < 0 ? 'down' : 'up'})` : ''}, which from the close of ${price(candle.before)} means a close
      ${low === null ? `below about ${price(high)}` : (high === null ? `above about ${price(low)}` : `between about ${price(low)} and ${price(high)}`)}`;
  }
  why += `. That is why the charts on this page can draw a predicted bin as a price, and the table under each chart lists the band.</p>`;

  bg += asDetail(`<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">How a day becomes a draw</span>
    <span class="card-meta" style="margin-left:10px;">read this first: what a bin is, what a ticket is, what a hit is</span></div><div class="card-icon">▼</div></div>
    <div class="card-body" style="max-width:1080px;">${step1}${step2}${strips}${step3}${why}
    <p style="color:#7f8c8d; font-size:0.85em; margin:10px 0 0;">Strictly, every move and edge on this page is a log return - the kind that adds up day over day - rather than a chart's plain percentage change;
    for ordinary days the two are the same number, at the &plusmn;3% edges they differ by a few hundredths of a percent, at &plusmn;6% by about two tenths.
    The <a href="/database/${esc(market)}">game view</a> shows these same digits in the lottery layout, scored the same way (the right bin for the right ${esc(meta.noun)}).${view.days.length ? ' The <i>Day by day</i> section is that history translated back into moves.' : ''}</p></div></div>`, true);

  // models
  const hasOpenCol = view.models.some((m) => m.openTotal !== null);
  const hasShortCol = view.models.some((m) => m.shortTotal !== null);
  // the sized rows (the RL Position Model): their next-day calls carry a size; they never short
  const sizedModels = new Set(view.instruments.flatMap((i) => i.next.filter((p) => p.size !== null).map((p) => p.model)));
  let modelsCard = `<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Models over ${view.scoredDays} scored day(s)</span>
    <span class="card-meta" style="margin-left:10px;">accuracy against chance, paper P&amp;L of the fixed rule</span></div><div class="card-icon">▼</div></div>
    <div class="card-body"><div class="table-wrapper"><table>
    <tr><th style="text-align:left;">Model</th><th>Days</th><th title="predicted bin equals the actual bin; chance ${pct(c.exact, 0)}">Exact</th>
    <th title="within one bin; chance ${pct(c.adjacent, 0)}">Adjacent</th><th title="sign of the predicted return equals the real one; chance ${pct(c.direction, 0)}">Direction</th>
    <th title="positions taken (predicted bin in the upper half)">Trades</th><th title="positions that made money after both fees">Win rate</th>
    <th title="money made by the positions taken: the stake bought at the previous close, sold at the day's close, a fee on each leg">P&amp;L ${esc(meta.unit)}</th><th>per trade</th>
    <th title="the same calls, but a position is kept while the calls stay up and sold at the close of the last up day: one fee pair per run">P&amp;L holding</th>${hasOpenCol
      ? '<th title="the same calls, but the stake is bought at the session\'s open and sold at its close: the rule a reader can follow">P&amp;L open-close</th>' : ''}${hasShortCol
      ? `<th title="the daily rule with shorts on: an upper-half bin bought, a lower-half bin sold first and bought back at the close - paper only; the hold${hasOpenCol ? ' and open-to-close' : ''} variant${hasOpenCol ? 's' : ''} on hover">P&amp;L long &amp; short (daily)</th>` : ''}</tr>`;
  if (!view.models.length) modelsCard += `<tr><td colspan="${10 + (hasOpenCol ? 1 : 0) + (hasShortCol ? 1 : 0)}" style="color:#aaa;">no settled day yet - the first predictions settle tomorrow morning</td></tr>`;
  view.models.forEach((m) => {
    const mark = (above, text) => `<span style="${above ? 'color:#27ae60; font-weight:bold;' : ''}">${text}</span>`;
    modelsCard += `<tr><td style="text-align:left; font-weight:bold;">${esc(m.name)}</td><td>${m.days === null ? '-' : m.days}</td>
      <td>${mark(m.aboveExact, pct(m.exact))}</td><td>${mark(m.aboveAdjacent, pct(m.adjacent))}</td><td>${mark(m.aboveDirection, pct(m.direction))}</td>
      <td>${m.trades === null ? '-' : m.trades}</td><td>${pct(m.winRate, 0)}</td>
      <td style="color:${(m.pnlCash || 0) >= 0 ? '#27ae60' : '#c0392b'}; font-weight:bold;">${money(m.pnlCash, 2)}</td><td>${money(m.pnlCashPerTrade, 2)}</td>
      <td style="color:${(m.holdTotal || 0) >= 0 ? '#27ae60' : '#c0392b'};" title="${m.holdTrades === null ? '' : `${m.holdTrades} position(s), win rate ${pct(m.holdWinRate, 0)}, ${money(m.holdPerTrade, 2)} per position`}">${money(m.holdTotal, 2)}</td>${hasOpenCol
        ? `<td style="color:${(m.openTotal || 0) >= 0 ? '#27ae60' : '#c0392b'};" title="${m.openTrades === null ? '' : `${m.openTrades} position(s), win rate ${pct(m.openWinRate, 0)}, ${money(m.openPerTrade, 2)} per position`}">${money(m.openTotal, 2)}</td>` : ''}${hasShortCol
        ? `<td style="color:${(m.shortTotal || 0) >= 0 ? '#27ae60' : '#c0392b'};" title="${m.shortTrades === null ? '' : `${sizedModels.has(m.name) ? 'a sized row: longs only, never shorts - its long-and-short books are its long-only books; ' : ''}${m.shortTrades} position(s), win rate ${pct(m.shortWinRate, 0)}, ${money(m.shortPerTrade, 2)} per position; holding ${money(m.shortHoldTotal, 2)}${m.shortOpenTotal !== null ? `, open to close ${money(m.shortOpenTotal, 2)}` : ''}`}">${money(m.shortTotal, 2)}${sizedModels.has(m.name) ? ' <span style="color:#7f8c8d;">(long only)</span>' : ''}</td>` : ''}</tr>`;
  });
  modelsCard += `</table></div><p style="color:#7f8c8d; font-size:0.85em;">Green: above chance. With ${view.instruments.length} ${esc(meta.noun)}s a day and a
    handful of days, a rate above chance is noise more often than not; read the rows over months, and against the null band the controls give the lottery rows.
    P&amp;L is paper money: ${view.trading && view.trading.stake !== null ? view.trading.stake : 100} ${esc(meta.unit)} per position, ${view.trading && view.trading.feePerLeg !== null ? pct(view.trading.feePerLeg, 2) : '0.10%'} on each leg - see the Paper trading card.</p></div></div>`;
  bg += asDetail(modelsCard, true);

  // the paper-trading book: cumulative money per model against the market
  if (view.trading && view.trading.dates.length) {
    const t = view.trading;
    const bookId = `ledger-${esc(market)}`;
    const bookModels = view.models.map((m) => m.name).filter((m) => t.models[m]);
    Object.keys(t.models).forEach((m) => { if (!bookModels.includes(m)) bookModels.push(m); });
    const bookColours = Object.fromEntries(bookModels.map((m, i) => [m, COLOURS[i % COLOURS.length]]));
    const bookBest = view.best && bookModels.includes(view.best) ? view.best : (bookModels[0] || null);
    const byDate = (points) => Object.fromEntries(points.map((p) => [p.date, p.total]));
    const aligned = (points) => { const idx = byDate(points); return t.dates.map((d) => (idx[d] === undefined ? null : idx[d])); };
    const ruleData = (book) => ({ benchmark: aligned(book.benchmark), series: Object.fromEntries(bookModels.map((m) => [m, aligned(book.models[m] || [])])) });
    const hasOpen = !!t.open;
    const hasShort = !!(t.short && t.short.daily);
    const shortRule = (rule, bench) => ({ benchmark: aligned(bench.benchmark), sold: aligned(t.short[rule].sold || []), series: Object.fromEntries(bookModels.map((m) => [m, aligned(t.short[rule].models[m] || [])])) });
    const bookData = {
      kind: 'ledger', symbol: 'book', quote: t.currency, labels: t.dates, currency: t.currency, stake: t.stake,
      rules: { daily: ruleData(t), hold: ruleData(t.hold), ...(hasOpen ? { open: ruleData(t.open) } : {}),
        ...(hasShort ? { daily_ls: shortRule('daily', t), hold_ls: shortRule('hold', t.hold) } : {}), ...(hasShort && hasOpen && t.short.open ? { open_ls: shortRule('open', t.open) } : {}) },
      colours: bookColours, best: bookBest,
    };
    // the headline books: rows with MIN_RANK_DAYS settled days (a one-day row can lead by luck), and no sized row as the long-and-short best (it never shorts)
    const seasoned = view.models.filter((m) => m.days !== null && m.days >= MIN_RANK_DAYS);
    const richest = seasoned.filter((m) => m.pnlCash !== null).sort((a, b) => b.pnlCash - a.pnlCash)[0] || null;
    const richestHold = seasoned.filter((m) => m.holdTotal !== null).sort((a, b) => b.holdTotal - a.holdTotal)[0] || null;
    const richestOpen = hasOpen ? (seasoned.filter((m) => m.openTotal !== null).sort((a, b) => b.openTotal - a.openTotal)[0] || null) : null;
    const marketTotal = t.benchmark.length ? t.benchmark[t.benchmark.length - 1].total : null;
    const marketHold = t.hold.benchmark.length ? t.hold.benchmark[t.hold.benchmark.length - 1].total : null;
    const marketOpen = hasOpen && t.open.benchmark.length ? t.open.benchmark[t.open.benchmark.length - 1].total : null;
    const richestShort = hasShort ? (seasoned.filter((m) => m.shortTotal !== null && !sizedModels.has(m.name)).sort((a, b) => b.shortTotal - a.shortTotal)[0] || null) : null;
    const soldTotal = hasShort && t.short.daily.sold.length ? t.short.daily.sold[t.short.daily.sold.length - 1].total : null;
    const bookSwitches = bookModels.map((m) => `<label><input type="checkbox" data-chart="${bookId}" data-model="${esc(m)}"${m === bookBest ? ' checked' : ''} onchange="marketToggle('${bookId}')">
      <span class="swatch" style="background:${bookColours[m]};"></span>${esc(m)}</label>`).join('');
    paper += `<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Paper trading</span>
      <span class="card-meta" style="margin-left:10px;">${t.stake === null ? '' : `${t.stake} ${esc(t.currency)} per position, `}${t.feePerLeg === null ? '' : `${pct(t.feePerLeg, 2)} a leg, `}${t.dates.length} settled day(s)${richest ? ` - best book ${money(richest.pnlCash, 2)} ${esc(t.currency)} (${esc(richest.name)})` : ''}${marketTotal === null ? '' : `, the market ${money(marketTotal, 2)}`}${richestHold ? `; holding: ${money(richestHold.holdTotal, 2)} (${esc(richestHold.name)})` : ''}${marketHold === null ? '' : `, buy-and-hold ${money(marketHold, 2)}`}${richestOpen ? `; open to close: ${money(richestOpen.openTotal, 2)} (${esc(richestOpen.name)})` : ''}${marketOpen === null ? '' : `, the market ${money(marketOpen, 2)}`}${richestShort ? `; long and short (daily): ${richestShort.shortTotal < 0 ? 'best ' : ''}${money(richestShort.shortTotal, 2)} (${esc(richestShort.name)})${richestShort.shortTotal < 0 ? ' - no long-and-short book is in the money' : ''}, buying everything ${money(marketTotal, 2)}${soldTotal === null ? '' : `, selling everything ${money(soldTotal, 2)}`}` : ''}</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="margin-top:0; color:#555;">The same calls, ${hasOpen ? 'three' : 'two'} ways of trading them${hasShort ? ', each long only or long and short (the switch)' : ''}, <b>${t.stake === null ? '-' : t.stake} ${esc(t.currency)}</b> a position and ${t.feePerLeg === null ? '-' : pct(t.feePerLeg, 2)} fee on the buy and on the sell.
      <b>Daily round trip</b>: every ${esc(meta.noun)} a model calls up (bin 5 or higher) is bought at the previous close (${market === 'crypto' ? '00:00 UTC, in the night before the ticket goes up' : 'the previous session\'s close'}) and sold at the day's own close; a flat or down call sits out${hasShort ? ' (long only) or is a paper short (the switch)' : ''}.
      <b>Hold while up</b>: the position is kept while the next day's call is up again and sold at the close of the last up day - one fee pair per run, the stake compounds; a position still open on the newest day is valued at its close.${hasOpen
        ? ` <b>Open to close</b>: bought at the session's own open (09:30 New York, 15:30 Belgian time in most weeks) and sold at its close - the one rule a reader of this page can follow, since the ticket is up hours before New York opens; <i>Today's plan</i> turns it into orders.` : ''}
      The grey dashed line is the <b>market</b> under the same rule - every ${esc(meta.noun)} every day, or everything bought on the first day and held - which a model has to beat before its book means anything.
      The daily and hold books buy at the previous close, ${market === 'crypto' ? 'eight to ten hours before the ticket is on this page' : 'with the gap to the next open still ahead'}, which nobody reading this can do: they score the call on the daily candle, not what a reader could have made from it. Paper money: no slippage, no funding, fills at the open or the close.${hasShort
        ? ` <b>Shorts</b> (the switch): with shorts on, a bin in the lower half is a short - sold at the previous close and bought back at the day's close (daily), kept while the call stays down (hold)${hasOpen ? ', sold at the open and bought back at the close (open to close)' : ''} - with the same stake and fees; the sized RL row never shorts. Paper only: selling what you do not own needs a margin or derivatives account, so <i>Today's plan</i> turns no short into an order. In the long-and-short views the second dashed line is what <b>selling</b> everything gave, next to the buying line: a short pays the market's drift and both fees, and a call in bin 4 - a hair below zero - cannot cover two fee legs by construction, so read a long-and-short book against both lines.` : ''}</p>
      <div class="chart-tools"><button type="button" data-view-for="${bookId}" data-view="daily" onclick="marketRule('${bookId}', 'daily')" title="every up call bought at the previous close and sold at the day's close">Daily round trip</button>
        <button type="button" data-view-for="${bookId}" data-view="hold" onclick="marketRule('${bookId}', 'hold')" title="a position kept while the call holds its direction (up; with shorts on also down), closed at the close of the last day of the run">Hold while up</button>${hasOpen
          ? `<button type="button" data-view-for="${bookId}" data-view="open" onclick="marketRule('${bookId}', 'open')" title="every up call bought at the session's open and sold at its close - the rule a reader can follow">Open to close</button>` : ''}<span class="sep">|</span>${hasShort
          ? `<button type="button" data-view-for="${bookId}" data-shorts="off" onclick="marketShorts('${bookId}', false)" title="down calls sit out">Long only</button>
        <button type="button" data-view-for="${bookId}" data-shorts="on" onclick="marketShorts('${bookId}', true)" title="a lower-half bin is a short: sold first, bought back later - paper only">Long and short</button><span class="sep">|</span>` : ''}
        ${RANGES.map(([label, days]) => `<button type="button" onclick="marketRange('${bookId}', ${days})">${label}</button>`).join('')}
        <button type="button" onclick="marketReset('${bookId}')" style="margin-left:6px;">Reset zoom</button><span class="hint">cumulative ${esc(t.currency)} per model; scroll or pinch to zoom, drag to pan</span></div>
      <div class="chart-models"><span style="color:#7f8c8d;">Models: <a href="#" onclick="marketModels('${bookId}', 'best'); return false;">best</a><a href="#" onclick="marketModels('${bookId}', 'all'); return false;">all</a><a href="#" onclick="marketModels('${bookId}', 'none'); return false;">none</a></span>${bookSwitches}</div>
      <div style="height:320px;"><canvas id="${bookId}"></canvas></div>
      <script>window.marketData['${bookId}'] = ${jsonScript(bookData)}; marketRender('${bookId}');</script>
      <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:0;">The models table in the Background card has each model's total, win rate and money per position under every rule${view.days.length ? '; its <i>Day by day</i> section shows the money each day' : ''}. A yardstick, not a result.</p></div></div>`;
  }

  // daily accuracy
  if (view.daily.length) {
    bg += asDetail(`<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Accuracy per day</span>
      <span class="card-meta" style="margin-left:10px;">mean exact rate over every model, the best model's, and chance</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><div style="height:220px;"><canvas id="daily-${esc(market)}"></canvas></div>
      <script>new Chart(document.getElementById('daily-${esc(market)}').getContext('2d'), { type: 'line', data: {
        labels: ${jsonScript(view.daily.map((d) => d.date))},
        datasets: [
          { label: 'mean exact rate', data: ${JSON.stringify(view.daily.map((d) => num(d.exact_mean)))}, borderColor: '#2980b9', tension: 0.2, pointRadius: 2 },
          { label: 'best model that day', data: ${JSON.stringify(view.daily.map((d) => num(d.best_exact)))}, borderColor: '#27ae60', tension: 0.2, pointRadius: 2 },
          { label: 'chance', data: ${JSON.stringify(view.daily.map(() => c.exact))}, borderColor: '#95a5a6', borderDash: [4, 4], pointRadius: 0 }
        ] }, options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } } } });</script></div></div>`);
  }

  // day by day: the settled days in market terms, each expandable to its models
  if (view.days.length) {
    const dayBlocks = view.days.map((d) => {
      const symbols = d.instruments.map((i) => i.symbol);     // a day's own slots: the instrument set can change between days
      const instrumentCells = d.instruments.map((i) => `<span style="display:inline-block; min-width:110px; margin-right:8px;"><b>${esc(i.symbol)}</b>
        <span style="color:${(i.ret || 0) >= 0 ? '#27ae60' : '#c0392b'};">${signedPct(i.ret)}</span> <span style="color:#7f8c8d;">bin ${i.bin === null ? '-' : i.bin}</span></span>`).join('');
      const modelRows = d.models.map((m) => {
        const cells = m.bins.map((b, i) => {
          const inst = d.instruments[i];
          const actual = inst ? inst.bin : null;
          const hit = b !== null && actual !== null && b === actual;
          const near = !hit && b !== null && actual !== null && Math.abs(b - actual) === 1;
          const arrow = b === null ? '' : (b >= half ? '<span style="color:#27ae60;">&#9650;</span>' : '<span style="color:#c0392b;">&#9660;</span>');
          return `<td style="${hit ? 'background:#2ecc71; color:white;' : (near ? 'background:#d5f5e3;' : '')}" title="${inst ? esc(intervalText(b, inst.edges)) : ''}${b === null ? '' : (b >= half ? ' - rule: long' : ' - rule: flat')}">${b === null ? '-' : b} ${arrow}</td>`;
        }).join('');
        return `<tr><td style="text-align:left; font-weight:bold;">${esc(m.name)}</td>${cells}<td>${m.exact === null ? '-' : m.exact}/${m.positions === null ? '-' : m.positions}</td>
          <td>${m.direction === null ? '-' : m.direction}/${m.positions === null ? '-' : m.positions}</td><td style="color:${((m.pnlCash !== null ? m.pnlCash : m.pnl) || 0) >= 0 ? '#27ae60' : '#c0392b'};">${m.pnlCash !== null ? money(m.pnlCash, 2) : money(m.pnl)}</td></tr>`;
      }).join('');
      return `<details style="border-bottom:1px solid #eee; padding:8px 0;"><summary style="cursor:pointer; display:flex; flex-wrap:wrap; align-items:center; gap:6px;">
        <b style="min-width:100px;">${esc(d.date)}</b> ${instrumentCells}
        <span style="margin-left:auto; color:#7f8c8d; font-size:0.9em;">best ${esc(d.best || '-')} · mean exact ${pct(d.exactMean, 0)} · <a href="${gameViewLink(market, d.date)}">game view</a></span></summary>
        <div class="table-wrapper" style="margin-top:8px;"><table><tr><th style="text-align:left;">Model (ticket made after the previous close)</th>${symbols.map((sym) => `<th>${esc(sym)}</th>`).join('')}<th title="right bin in the right slot">Exact</th><th title="predicted bin on the same side of zero as the real return">Direction</th><th title="paper money that day: the stake per long position, a fee on each leg">P&amp;L ${esc(meta.unit)}</th></tr>${modelRows}</table></div></details>`;
    }).join('');
    bg += asDetail(`<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Day by day</span>
      <span class="card-meta" style="margin-left:10px;">the newest ${view.days.length} settled day(s) as returns and bins; open a day for every model's ticket</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="color:#7f8c8d; font-size:0.9em; margin-top:0;">Each line is one draw: per ${esc(meta.noun)} the real close-to-close return and the bin it fell in. Inside, every model's bin per ${esc(meta.noun)}
      with the position the fixed P&amp;L rule takes on it (&#9650; long, bin in the upper half; &#9660; lower half - sits out in the long-only books, a paper short with the shorts switch on) - the <i>Direction</i> column is scored on the sign of the bin's own return,
      which is read off the hover interval; green is the right bin, pale green one bin off. Chance is 1 in ${view.k} per ${esc(meta.noun)}.</p>${dayBlocks}</div></div>`);
  }

  // the rows under a proper score (the weekly MarketRows.py report)
  if (view.rows) {
    const rep = view.rows;
    const signed = (x, d = 3) => (num(x) === null ? '-' : (num(x) >= 0 ? '+' : '') + num(x).toFixed(d));
    const verdictMark = (iv) => {
      if (!iv) return '<span style="color:#aaa;">-</span>';
      const colour = iv.verdict === 'better' ? '#27ae60' : (iv.verdict === 'worse' ? '#c0392b' : '#7f8c8d');
      return `<span style="color:${colour};${iv.verdict === 'better' ? ' font-weight:bold;' : ''}" title="mean difference in log-score per day, 95% paired bootstrap interval over ${iv.days === null ? '?' : iv.days} days">${signed(iv.mean)} [${signed(iv.lo)}, ${signed(iv.hi)}] ${esc(iv.verdict)}</span>`;
    };
    const kindNote = { market: 'market row', base: 'lottery row', baseline: 'baseline' };
    let table = '';
    rep.rows.forEach((r) => {
      table += `<tr><td style="text-align:left; font-weight:bold;">${esc(r.name)}${r.isReference ? ' <span style="color:#7f8c8d; font-weight:normal;">(reference)</span>' : ''}<br><span style="color:#aaa; font-size:0.8em;">${esc(kindNote[r.kind] || r.kind)}</span></td>
        <td>${r.logScore === null ? '<span style="color:#aaa;">no probabilities</span>' : `<b>${r.logScore.toFixed(3)}</b>${r.se === null ? '' : ` <span style="color:#aaa;">± ${r.se.toFixed(3)}</span>`}`}</td>
        <td>${r.isReference ? '<span style="color:#7f8c8d;">reference</span>' : verdictMark(r.vsReference)}</td><td>${verdictMark(r.vsUniform)}</td>
        <td>${pct(r.exact)}</td><td>${pct(r.adjacent)}</td><td>${pct(r.direction)}</td><td>${r.trades === null ? '-' : r.trades}</td>
        <td style="color:${(r.pnl || 0) >= 0 ? '#27ae60' : '#c0392b'};">${money(r.pnl)}</td></tr>`;
    });
    const headline = rep.betterThanReference.length
      ? `<b style="color:#27ae60;">${esc(rep.betterThanReference.join(', '))}</b> carr${rep.betterThanReference.length === 1 ? 'ies' : 'y'} information beyond GARCH over this window (the interval's lower bound is above zero).`
      : `<b>No row carries information beyond GARCH</b> over this window: every interval against the reference includes zero or lies below it.`;
    const garchLine = rep.referenceRow
      ? (rep.referenceAboveUniform ? `GARCH itself is above the uniform forecast (${rep.referenceRow.logScore.toFixed(3)} against ${rep.uniform.toFixed(3)}), which is volatility clustering - the known predictable component.`
        : `GARCH itself is not distinguishable from the uniform forecast here (${rep.referenceRow.logScore === null ? '-' : rep.referenceRow.logScore.toFixed(3)} against ${rep.uniform.toFixed(3)}).`)
      : '';
    bg += asDetail(`<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Rows under a proper score</span>
      <span class="card-meta" style="margin-left:10px;">mean log-score of the probability each row gave the bin that happened, ${rep.days === null ? '?' : rep.days} game days${rep.firstDay ? ` ${esc(rep.firstDay)} to ${esc(rep.lastDay || '')}` : ''}, against GARCH</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="color:#7f8c8d; font-size:0.9em;">${headline} ${garchLine} A row is judged here before its hit rates or paper P&amp;L are read: the log-score rewards honest
      probabilities and punishes overconfidence, so a row cannot win it by betting on the same bin every day. The uniform forecast scores ${rep.uniform === null ? '-' : rep.uniform.toFixed(3)}
      (log of 1/${view.k}); a probability under ${rep.floor === null ? '-' : rep.floor} is scored as ${rep.floor === null ? '-' : rep.floor}.${rep.lockboxWithheld ? ` ${rep.lockboxWithheld} lockbox day(s) withheld.` : ''}</p>
      <div class="table-wrapper"><table><tr><th style="text-align:left;">Row</th><th title="mean over days of the mean over instruments of log p(actual bin); higher is better">Log-score</th>
      <th title="difference to the GARCH row, 95% paired bootstrap interval over days">vs GARCH</th><th title="difference to the uniform forecast">vs uniform</th><th>Exact</th><th>Adjacent</th><th>Direction</th><th>Trades</th><th title="fixed rule with the real returns and the fee, in units of price (0.01 = 1%)">P&amp;L</th></tr>${table}</table></div>
      <p style="color:#7f8c8d; font-size:0.85em;">Walk-forward, every day refitted on the past only (MarketRows.py, Sundays). "Better" and "worse" are read off the interval, not the point estimate. The two ablation rows
      say what the Regime HMM's regimes are made of: if the full row is not better than <i>ZeroMean</i>, the regimes carry variance only; if it is not better than <i>Single</i>, there are no regimes worth the name.${Object.keys(rep.errors).length ? ` Rows that failed on some days: ${esc(Object.keys(rep.errors).join(', '))}.` : ''}
      Generated ${esc(rep.generatedAt || '-')}.</p></div></div>`);
  }

  // the regime reading (the Regime HMM rows' templates)
  if (view.regimes.length) {
    let table = '';
    view.regimes.forEach((r) => {
      table += `<tr><td style="text-align:left; font-weight:bold;">${esc(r.row)}</td><td>${esc(r.date || '-')}</td><td>${r.regimes === null ? '-' : r.regimes}</td>
        <td>${r.template === null ? '-' : `T${r.template}`}${r.label ? ` <span style="color:#7f8c8d;">(${esc(r.label)}${r.volatilityRank !== null && r.regimes !== null && r.regimes > 1 ? `, ${r.volatilityRank} of ${r.regimes} by volatility` : ''})</span>` : ''}</td>
        <td>${pct(r.probability, 0)}</td><td style="text-align:left; font-size:0.85em;">${r.expected.map((e) => `${esc(e.symbol)} ${e.value === null ? '-' : (e.value * 100).toFixed(2) + '%'}`).join(', ')}</td></tr>`;
    });
    bg += asDetail(`<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Regime reading</span>
      <span class="card-meta" style="margin-left:10px;">which regime the Regime HMM rows believe the market is in, after the newest game day</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><div class="table-wrapper"><table><tr><th style="text-align:left;">Row</th><th>After</th><th>Regimes</th><th title="a persistent template matched across the daily refits by the Wasserstein distance between the regime Gaussians; calm / normal / turbulent by the regime's volatility against the market's">Template</th><th>Probability</th><th style="text-align:left;">Expected next-day return</th></tr>${table}</table></div>
      <p style="color:#7f8c8d; font-size:0.85em;">A regime is a Gaussian over the ${esc(meta.noun)}s' returns and their recent volatility and momentum; the label is the volatility of the dominant regime against the market's, the template number keeps its identity across refits.
      The expected return is the mixture's mean per ${esc(meta.noun)} - a reading, not a recommendation.</p></div></div>`);
  }

  // instruments: the data object per chart, the switches, the canvas
  const modelNames = view.models.map((m) => m.name);
  view.instruments.forEach((inst) => {
    inst.course && Object.keys(inst.course).forEach((m) => { if (!modelNames.includes(m)) modelNames.push(m); });
  });
  const colourFor = (i) => {   // the palette first, then a golden-angle hue as hex (the client's alpha helper parses hex)
    if (i < COLOURS.length) return COLOURS[i];
    const h = (i * 137.508) % 360, sat = 0.55, lig = 0.42;
    const f = (nn) => { const k = (nn + h / 30) % 12; const c = lig - sat * Math.min(lig, 1 - lig) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); return Math.round(c * 255).toString(16).padStart(2, '0'); };
    return `#${f(0)}${f(8)}${f(4)}`;
  };
  const colours = Object.fromEntries(modelNames.map((m, i) => [m, colourFor(i)]));
  const bestModel = view.best && modelNames.includes(view.best) ? view.best : (modelNames[0] || null);
  const nextDay = nextGameDay(market, view.madeOn);                 // the game day the newest ticket is for
  const judged = nextDay ? closeText(market, nextDay) : null;      // and the moment it is judged, in Belgian time
  const judgedShort = nextDay ? closeShort(market, nextDay) : null;
  const opens = nextDay ? openText(market, nextDay) : null;        // shares: the regular session's open that day
  const status = nextDay ? dayStatus(market, nextDay, now) : null; // where that day stands as the page is rendered
  const nextDayText = nextDay ? (calendarKnown(market, nextDay) ? esc(nextDay) : `the next session, normally ${esc(nextDay)}`) : null;
  const hasOpenBook = !!(view.trading && view.trading.open);
  charts += `<details class="chart-help"><summary style="cursor:pointer; font-weight:bold; padding:6px 0;">How to read the charts</summary>
      <p style="color:#7f8c8d; font-size:0.85em; margin:4px 0 10px;"><b>Price lines</b>: the close as a line and, for each model switched on, a dashed line through the prices its bins stood for, day by day
        (the previous game day's close moved by the bin's middle return); the last dashed point, <i>next</i>, is the call for ${nextDayText || 'the next trading day'} - the candle after the last one drawn, not "tomorrow"${nextDay ? `; it was made after the close of ${esc(view.madeOn)}` : ''}${judged ? ` and is judged at ${esc(judged)}` : ''}.
        A model calling "roughly flat" every day draws the close line one day late - that is the call, not a lag (the Background card explains why). <b>Price bars</b>: the same calls as a bar
        per day from the previous game day's close to the price its bin stood for (green up, red down) over a paler bar for the bin's whole interval, so a flat call reads as a stubby bar; a hit is the
        close landing inside the pale bar on the same date, and the two open-ended bins run to the edge of what the chart shows. <b>Moves</b>: the real move per day as a dark bar, in percent,
        with each model's interval as a paler bar and its middle as a dot - a hit is the dark bar ending inside the model's band. Each chart opens on the newest month; a year of closes is behind it
        (the range buttons, or zoom and pan), and the calls start where the tracking started. The model ticked by default has the best exact rate over the scored days; <i>Today's plan</i> follows the best book instead, so the two can differ.</p></details>`;
  view.instruments.forEach((inst) => {
    const id = `chart-${esc(market)}-${esc(inst.symbol)}`;
    const labels = inst.closes.map((p) => p[0]).concat(['next']);
    const data = {
      symbol: inst.symbol, quote: inst.quote, labels, closes: inst.closes.map((p) => p[1]).concat([null]),
      lastClose: inst.nextBase !== null && inst.nextBase !== undefined ? inst.nextBase : inst.lastClose,   // the newest game day's close, the base of the next-day call
      edges: inst.edges.concat([null]), moves: inst.moves.concat([null]),
      course: Object.fromEntries(Object.keys(inst.course).map((m) => [m, inst.course[m].concat([null])])),
      next: Object.fromEntries(inst.next.map((p) => [p.model, { bin: p.bin, price: p.price, low: p.low, high: p.high }])),
      colours, best: bestModel,
    };
    const switches = modelNames.map((m) => `<label><input type="checkbox" data-chart="${id}" data-model="${esc(m)}"${m === bestModel ? ' checked' : ''} onchange="marketToggle('${id}')">
      <span class="swatch" style="background:${colours[m]};"></span>${esc(m)}${m === bestModel ? ' <span style="color:#7f8c8d;">(best exact rate over the scored days)</span>' : ''}</label>`).join('');
    const nextRows = inst.next.map((p) => `<tr><td style="text-align:left;">${esc(p.model)}</td><td>${p.bin === null ? '-' : p.bin}</td>
      <td>${p.size !== null ? (p.size > 0 ? `<span style="color:#27ae60;">long &times;${p.size}</span>` : 'flat (size 0)') : (p.bin !== null && p.bin >= view.k / 2 ? '<span style="color:#27ae60;">long</span>' : (p.bin !== null ? '<span style="color:#7f8c8d;">flat - short (paper) with shorts on</span>' : '-'))}</td>
      <td>${price(p.price)}</td><td>${p.low === null ? 'below ' + price(p.high) : (p.high === null ? 'above ' + price(p.low) : `${price(p.low)} - ${price(p.high)}`)}</td></tr>`).join('');
    charts += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">${esc(inst.symbol)} - ${esc(inst.name)}</span>
      <span class="card-meta" style="margin-left:10px;">last close ${price(inst.lastClose)} ${esc(inst.quote)} on ${esc(inst.lastDate || '-')}${inst.active ? '' : ' (inactive)'}</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><div class="chart-tools">
        <button type="button" data-view-for="${id}" data-view="lines" onclick="marketView('${id}', 'lines')" title="the close and each model's predicted course as a dashed line">Price lines</button>
        <button type="button" data-view-for="${id}" data-view="bars" onclick="marketView('${id}', 'bars')" title="each day's call as a bar from the previous close to the predicted price, over the bin's interval">Price bars</button>
        <button type="button" data-view-for="${id}" data-view="moves" onclick="marketView('${id}', 'moves')" title="the real move per day against each model's interval, in percent">Moves</button><span class="sep">|</span>
        ${RANGES.map(([label, days]) => `<button type="button" onclick="marketRange('${id}', ${days})">${label}</button>`).join('')}
        <button type="button" onclick="marketReset('${id}')" style="margin-left:6px;">Reset zoom</button>
        <span class="hint">scroll or pinch to zoom, drag to pan</span></div>
      <div class="chart-models"><span style="color:#7f8c8d;">Models: <a href="#" onclick="marketModels('${id}', 'best'); return false;">best</a><a href="#" onclick="marketModels('${id}', 'all'); return false;">all</a><a href="#" onclick="marketModels('${id}', 'none'); return false;">none</a></span>${switches}</div>
      <div style="height:420px;"><canvas id="${id}"></canvas></div>
      <script>window.marketData['${id}'] = ${jsonScript(data)}; marketRender('${id}');</script>
      ${nextRows ? `<details class="all-calls"><summary style="cursor:pointer; color:#2980b9; font-size:0.9em;">Every model's call for ${nextDayText || 'the next day'} (${inst.next.length})</summary>
      <div class="table-wrapper"><table><tr><th style="text-align:left;">Model</th><th>Bin</th><th title="long when the bin is in the upper half (5 or higher); a lower-half bin is a short with the shorts switch on - paper only">Position</th><th title="the bin's middle return applied to the last close">Price drawn on the chart (bin middle)</th><th title="the bin as prices: a hit is the close landing anywhere inside">Band</th></tr>${nextRows}</table></div></details>` : '<p style="color:#aaa;">no prediction for the next day yet</p>'}
      ${judged ? `<p style="color:#7f8c8d; font-size:0.85em; margin:6px 0 0;"><b>Judged at</b> ${esc(judged)}.${status ? ` Right now ${esc(status)}.` : ''} The call is scored close-to-close: right when the close lands in the band, whatever happened in between.
        What a reader can do with it is in <i>Today's plan</i> at the top${hasOpenBook ? ' and in the Open to close book' : ''}.</p>` : ''}
      </div></div>`;
  });

  // --- today's plan: the best book's calls as the two orders that match how the page scores them ---
  // Reviewed on 4 Oct 2026 by an owner stand-in (a Belgian retail investor)
  // and a code reviewer: the orders are said in broker words, the sell is
  // at the close and not a limit at the band, "up" is the books' definition
  // (the upper half), a book needs MIN_PLAN_TRADES positions before it can
  // be followed, and a reader who holds from an earlier day is told what to
  // do on a down call.
  let plan = '';
  const MIN_PLAN_TRADES = 10;
  const planKey = market === 'crypto' ? 'pnlCash' : 'openTotal';   // the rule a reader can follow: open to close for shares, the daily book for crypto
  const planTradesKey = market === 'crypto' ? 'trades' : 'openTrades';
  const planRuleName = market === 'crypto' ? 'daily round trip' : 'open to close';
  const ranked = view.models.filter((m) => m[planKey] !== null && m[planTradesKey] !== null && m[planTradesKey] >= MIN_PLAN_TRADES).sort((a, b) => b[planKey] - a[planKey]);
  const planModel = ranked[0] || view.models.find((m) => m.name === view.best) || null;
  const planByBook = ranked.length > 0;
  if (nextDay && planModel) {
    const t = view.trading;
    const planBench = t ? (market === 'crypto' ? t.benchmark : (t.open ? t.open.benchmark : [])) : [];
    const planMarket = planBench.length ? planBench[planBench.length - 1].total : null;
    const closeAt = closeMoment(market, nextDay);
    const openAt = openMoment(market, nextDay);
    const closed = closeAt && now >= closeAt;
    const closeClock = closeAt ? brusselsText(closeAt) : null;
    const nounCap = meta.noun[0].toUpperCase() + meta.noun.slice(1);
    const rows = view.instruments.filter((i) => i.active).map((inst) => {
      const base = inst.nextBase !== null && inst.nextBase !== undefined ? inst.nextBase : inst.lastClose;
      const call = inst.next.find((p) => p.model === planModel.name) || null;
      const isUp = (p) => (p.size !== null && p.size !== undefined ? p.size > 0 : (p.bin !== null && p.bin >= view.k / 2));   // the books' definition: a bin in the upper half, or a sized row's size
      const sized = call && call.size !== null && call.size !== undefined;
      const sizeNote = sized && call.size > 0 ? `position size &times;${call.size} (${Math.round(call.size * (t && t.stake !== null ? t.stake : 100))} ${esc(meta.unit)} at the paper stake) - ` : '';
      const up = !!call && isUp(call);
      const ups = inst.next.filter(isUp).length;
      const movePct = (v) => (v === null || base === null || base <= 0 ? null : (v / base - 1) * 100);
      const moveText = call ? (call.low === null ? `below ${signedPct(movePct(call.high) / 100, 1)}` : (call.high === null ? `above ${signedPct(movePct(call.low) / 100, 1)}` : `${signedPct(movePct(call.low) / 100, 1)} to ${signedPct(movePct(call.high) / 100, 1)}`)) : '';
      const band = call ? `${call.low === null ? `below ${price(call.high)}` : (call.high === null ? `above ${price(call.low)}` : `${price(call.low)} - ${price(call.high)}`)}<br><span style="color:#7f8c8d; font-size:0.85em;">bin ${call.bin}: ${moveText}</span>` : '-';
      const holdNote = `if you still hold it from an earlier day, sell at the close${closeClock ? `, ${esc(closeClock)}` : ''}`;
      let buy, sell;
      if (!call) { buy = `no call from this model for this ${esc(meta.noun)}`; sell = '-'; }
      else if (!up) { buy = sized ? `no new buy - the model sizes this ${esc(meta.noun)} at 0 today (its vote bin is ${call.bin})` : `no new buy - the call is ${call.bin < view.k / 2 - 1 ? 'down' : 'flat to down'} (bin ${call.bin})${view.trading && view.trading.short ? `; with shorts on, the ${planRuleName} book shorts it on paper, which needs a margin or derivatives account - no order here` : ''}`; sell = holdNote; }
      else if (closed) { buy = `too late for this ticket - the day closed at ${esc(closeClock)}; the next ticket is here after the next run, about 10:30`; sell = holdNote; }
      else if (market === 'crypto') {
        buy = `${sizeNote}a market order now - the ticket went up at ${esc(appeared || 'about 10:30')}; ${esc(status || '')}`;
        sell = `a market order at ${esc(judgedShort)}, whatever the price then is`;
      } else if (openAt && now >= openAt) {
        buy = `${sizeNote}a market order now, at the current price - the open (${esc(opens)}) is past; ${esc(status || '')}`;
        sell = `a market order in the last minutes before ${esc(judgedShort)}, or an at-the-close order if your broker offers one on US shares`;
      } else {
        buy = `${sizeNote}a market order placed before ${esc(opens)} - it fills at the open`;
        sell = `a market order in the last minutes before ${esc(judgedShort)}, or an at-the-close order if your broker offers one on US shares`;
      }
      const call_ = !call ? '-' : (up ? `<span style="color:#27ae60; font-weight:bold;">up &#9650;${sized ? ` &times;${call.size}` : ''}</span>` : (sized ? '<span style="color:#c0392b;">size 0 &#9660;</span>' : `<span style="color:#c0392b;">${call.bin < view.k / 2 - 1 ? 'down' : 'flat'} &#9660;</span>`));
      return `<tr><td style="text-align:left; font-weight:bold;">${esc(inst.symbol)}<br><span style="color:#7f8c8d; font-weight:normal; font-size:0.85em;">last close ${price(base)}</span></td>
        <td>${call_}</td><td>${band}</td><td class="order">${buy}</td><td class="order">${sell}</td><td>${ups} of ${inst.next.length}</td></tr>`;
    }).join('');
    const picked = planByBook
      ? `the best book under the <b>${planRuleName}</b> rule over the settled days: ${money(planModel[planKey], 2)} ${esc(meta.unit)} from ${planModel[planTradesKey]} position(s)${planMarket === null ? '' : `, the market under the same rule ${money(planMarket, 2)}`}`
      : `the best exact rate over the scored days - no book has ${MIN_PLAN_TRADES} positions yet under the ${planRuleName} rule`;
    plan = `<div class="card expanded"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Today's plan</span>
      <span class="card-meta" style="margin-left:10px;">for ${nextDayText}${status ? ` - ${esc(status)}` : ''}</span></div><div class="card-icon">▼</div></div>
      <div class="card-body"><p style="margin-top:0; color:#555;">Follows <b>${esc(planModel.name)}</b>, ${picked}; the other models' books are in the Background card's models table${planByBook && market !== 'crypto' ? ' (column <i>P&amp;L open-close</i>)' : ''}.
      Picked on ${view.scoredDays} settled day(s) of paper money; the pick can change tomorrow. Per ${esc(meta.noun)}: its call, the band the close must land in for the call to be right, and the two orders that match how this page scores it, in Belgian time for this date.
      <b>The sell is at the close, whatever the price</b> - it is not a limit order at the band; the band only says when the model counts as right.${appeared ? ` This plan is from the ticket of ${esc(appeared)}${market === 'crypto'
        ? '; every morning\'s run (about 10:30, sometimes later) replaces it.'
        : `; it is the plan for the whole of ${esc(nextDay)} and does not change before that session - the next ticket comes the morning after its close.`}` : ''}</p>
      <div class="table-wrapper"><table class="plan"><tr><th style="text-align:left;">${esc(nounCap)}</th><th title="up = the predicted bin is in the upper half (5 or higher), the position the books take">Call</th><th title="the predicted bin as prices: the call is right when the close lands anywhere inside">Band (the predicted bin as prices)</th>
      <th style="text-align:left;">Buy</th><th style="text-align:left;">Sell</th><th title="how many of the models with a call for this date have a bin in the upper half">Models calling up</th></tr>${rows}</table></div>
      <p style="color:#7f8c8d; font-size:0.85em; margin:8px 0 0;"><b>Up</b> means the predicted bin is in the upper half (5 to 9), the one position the long-only books take${view.trading && view.trading.short ? '; with the shorts switch on, the paper books also short the lower half - paper only, no order' : ''}. The two bins around zero can be "up" here and a hair negative by their middle.
      The <i>RL Position Model</i> is the one row that sizes its positions - 0 to 2 times the stake, learned from what the other rows said on past days and the money it made - so its call reads &times;0.5 to &times;2, and a size of 0 is a sit-out whatever its vote bin says.
      <b>Stake and fees</b>: the books use ${t && t.stake !== null ? t.stake : 100} ${esc(meta.unit)} a position and ${t && t.feePerLeg !== null ? pct(t.feePerLeg, 2) : '0.10%'} a leg; your broker's commission replaces that fee and decides whether a move as small as the band says can pay.
      <b>Holding instead of selling at the close</b>: keep the position while the next morning's call (here after about 10:30) is up again, and sell at the close of the first day it is not. There is no multi-day forecast yet - the models predict one day ahead - so a hold is decided one morning at a time;
      your result is then open-to-close on the first day and close-to-close after it, for which the <i>Hold while up</i> book is the closest yardstick, not an exact match.
      ${market === 'crypto' ? 'The daily book these orders follow buys at 00:00 UTC, which nobody reading this can do: your entry is later and at another price, so your result and the book\'s differ by the day\'s first hours.' : 'These orders follow the <i>Open to close</i> book, the one a reader can follow; the daily and hold books buy at the previous close, which nobody reading this can do.'}
      This describes when the page's condition is checked - it is not advice.</p></div></div>`;
  }

  html += plan + paper + charts;
  if (bg) {
    const sectionTitles = [...bg.matchAll(/<summary><b>([^<]*)<\/b>/g)].map((m) => m[1].replace(/ over \d+ scored day\(s\)$/, ''));
    html += `<div class="card"><div class="card-header" onclick="toggleCard(this)"><div><span class="card-title">Background</span>
      <span class="card-meta" style="margin-left:10px;">${esc(sectionTitles.join(' - ').toLowerCase())} - open a section</span></div><div class="card-icon">▼</div></div>
      <div class="card-body">${bg}</div></div>`;
  }
  html += `<p style="color:#7f8c8d; font-size:0.85em; margin-top:20px;">Record generated ${esc(view.generatedAt || '-')}${appearedText(view.generatedAt) ? ` (${esc(appearedText(view.generatedAt))})` : ''}; newest game day ${esc(view.newestGameDay || '-')}.
    The same rows are tracked, slot by slot, on the <a href="/database/${esc(market)}">game view</a> like every lottery game.</p>`;
  return html + footer();
}

function install(app, { header, footer, dataDir, controlsDir }) {
  Object.keys(MARKETS).forEach((market) => {
    app.get(`/markets/${market}`, (req, res) => {
      const record = loadMarket(dataDir, market);
      const extras = { rows: controlsDir ? loadRows(controlsDir, market) : null, regimes: loadRegimes(dataDir, market) };
      res.send(page(market, record ? describeMarket(record, extras) : null, header, footer, req.user));
    });
  });
}

module.exports = { MARKETS, COLOURS, CHART_CLIENT_JS, loadMarket, loadRows, loadRegimes, describeMarket, describeRows, describeRegimes, describeDays, describeTrading, binInterval,
  nextGameDay, closeMoment, openMoment, closeText, closeShort, openText, brusselsText, appearedText, hoursInto, dayStatus, NYSE_CLOSED, NYSE_EARLY_CLOSE,
  intervalText, binOfMove, gameViewLink, signedPct, page, install, pct, money, price };
