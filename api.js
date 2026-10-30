// api.js - the JSON API behind the React user interface (web/, served under /app;
// 15 Oct 2026, the UI redesign). Three routes, all behind the same session as
// the pages (auth.middleware runs first):
//
//   GET /api/me               who is signed in, and the session's form token
//                             (csrf) the app's JSON POSTs send back (system.js)
//   GET /api/markets/:market  the user view of a market (crypto, shares): the
//                             followed plan as positions, open and closed, with
//                             the day-by-day and per-instrument results
//   GET /api/quotes/:market   live prices for the open positions (Binance for
//                             crypto, Nasdaq's delayed quote for shares), cached
//                             a minute; {at: null, quotes: {}} when unavailable
//
// The view is built from the same records and the same picks as the server-
// rendered pages (markets.js: describeMarket, pickPlan, pickWeek), so the two
// never disagree on which model is followed or what it made. Every figure is
// paper money; the notes say where the book's entry differs from a reader's.
'use strict';

const path = require('path');
const fs = require('fs');
const markets = require('./markets');
const shell = require('./shell');   // the few server-rendered pages' look (the app-not-built answer)

const { MARKETS, WEEK_GAMES, loadMarket, describeMarket, nextGameDay, closeMoment, openMoment, weekEnd, dayStatus, dayStatusWeek,
        pickPlan, pickWeek, GUARD_SHOWN_FROM, ORDERS_SHOWN_FROM } = markets;

const BRUSSELS = 'Europe/Brussels';
const DAY_MS = 86400000;
// markets.GUARD_SHOWN_FROM: the first week the user pages count the guard (MarketSettle.guarded_week), the first full
// week the page showed it (owner, 20 Oct 2026: count "what the page said"). Weeks closing before it count the plain
// week book - held to the week's end - which is what the page told followers then; the guard's own book for those
// weeks is on the Admin market page only (the weeks before MarketSettle.GUARD_FORWARD_FROM, 19 Oct, booked after the
// fact). The week of 19 Oct shows the guard as information.

// --- clocks in Belgian time, short ------------------------------------------
const fmtClock = new Intl.DateTimeFormat('en-GB', { timeZone: BRUSSELS, hour: '2-digit', minute: '2-digit', hour12: false });
const fmtDay = new Intl.DateTimeFormat('en-CA', { timeZone: BRUSSELS, year: 'numeric', month: '2-digit', day: '2-digit' });
const fmtWeekday = new Intl.DateTimeFormat('en-GB', { timeZone: BRUSSELS, weekday: 'short' });
const clock = (date) => fmtClock.format(date);
const dayKey = (date) => fmtDay.format(date);

const fmtDayMonth = new Intl.DateTimeFormat('en-GB', { timeZone: BRUSSELS, day: 'numeric', month: 'short' });
// "Mon 12 Oct" for a game day given as YYYY-MM-DD
const dayText = (iso) => { const d = new Date(`${iso}T12:00:00Z`); return Number.isNaN(d.getTime()) ? '—' : `${fmtWeekday.format(d)} ${fmtDayMonth.format(d)}`; };
// "today 22:00", "tonight 02:00" (before 06:00 on the next day), "Mon 12 Oct" for a past day, "Mon 19 Oct 02:00" for a later one.
function when(date, now) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '—';
  const d = dayKey(date), n = dayKey(now);
  const hhmm = clock(date);
  if (d === n) return `today ${hhmm}`;
  if (d === dayKey(new Date(now.getTime() + DAY_MS)) && Number(hhmm.slice(0, 2)) < 6) return `tonight ${hhmm}`;
  const dayText = `${fmtWeekday.format(date)} ${fmtDayMonth.format(date)}`;
  return date < now ? dayText : `${dayText} ${hhmm}`;
}

const num = (x) => { const v = Number(x); return x === null || x === undefined || !Number.isFinite(v) ? null : v; };
const round2 = (x) => (num(x) === null ? null : Math.round(num(x) * 100) / 100);
const isUp = (p, k) => (p.size !== null && p.size !== undefined ? p.size > 0 : (p.bin !== null && p.bin >= k / 2));   // the books' definition
// the word of a call that is not up; a sized row that is not up is flat whatever its bin (MarketSettle.call_word)
const callWord = (p, k) => (p.size !== null && p.size !== undefined ? 'flat' : (p.bin !== null && p.bin < k / 2 - 1 ? 'down' : 'flat'));
// "+1.23" / "−1.23", as the app's format.money writes it
const MINUS = '\u2212';
const signed = (v, d = 2) => { const r = Number(v.toFixed(d)); return r > 0 ? `+${Math.abs(r).toFixed(d)}` : r < 0 ? `${MINUS}${Math.abs(r).toFixed(d)}` : Math.abs(r).toFixed(d); };
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const base = (inst) => (inst.nextBase !== null && inst.nextBase !== undefined ? inst.nextBase : inst.lastClose);

// --- the positions of the followed record ------------------------------------
// The one row that sizes its positions (src/RLPositionModel.ROW_NAME): its money is size x the move, so its move
// cannot be read back from the money.
const SIZED_ROWS = new Set(['RL Position Model']);

// One entry per instrument per live day, from the followed points' per-symbol
// money (MarketSettle.followed_plan). movePct is the position's move, read
// back from the money of an unsized row: under both rules money = stake x
// ratio x (1 - fee) - stake x (1 + fee) (MarketSettle.cash_pnl /
// open_close_pnl), so ratio - 1 = (money + 2 x fee x stake) / (stake x (1 -
// fee)). Null for the sized row. The exit is the day's own close.
function positionsOf(market, view, followed, game) {
  if (!followed || !view.trading) return [];
  const stake = view.trading.stake, fee = view.trading.feePerLeg || 0;
  const out = [];
  followed.points.filter((p) => p.live).forEach((p) => {
    const closeAt = game === 'day' ? closeMoment(market, p.date) : null;
    const exit = game === 'day' ? `close ${closeAt ? clock(closeAt) : ''}${market === 'crypto' ? ' next day' : ''}`.trim() : 'week end';
    Object.entries(p.bySymbol || {}).forEach(([symbol, pnl]) => {
      if (pnl === null) return;
      const movePct = !SIZED_ROWS.has(p.model) && stake ? round2(((pnl + 2 * fee * stake) / (stake * (1 - fee))) * 100) : null;
      out.push({ closed: p.date, symbol, game, model: p.model, movePct, pnl: round2(pnl), exit, raw: pnl });
    });
  });
  return out;
}

// The week positions of the guarded week book (MarketSettle.guarded_week): one entry per position of a
// live week, dated the week's close like the week's market - a position the guard sold mid-week belongs to its
// week and is booked when the week closes, so no past day changes afterwards. The exit says when it was bought
// and sold. movePct as in positionsOf, null for the sized row.
function guardedPositions(market, guarded, stake, fee) {
  if (!guarded || !Array.isArray(guarded.points)) return [];
  const out = [];
  guarded.points.filter((p) => Array.isArray(p) && p[5] === true).forEach((p) => {
    (Array.isArray(p[8]) ? p[8] : []).forEach((t) => {
      const [symbol, entryDay, exitDay, money, reason] = t;
      const pnl = num(money);
      if (pnl === null) return;
      const closeAt = closeMoment(market, exitDay);
      const bought = entryDay ? ` · bought ${dayText(entryDay)}` : '';
      const exit = reason === 'guard'
        ? `guard: ${dayText(exitDay)}, close ${closeAt ? clock(closeAt) : ''}${market === 'crypto' ? ' next day' : ''}${bought}`
        : `week end${bought}`;
      const movePct = !SIZED_ROWS.has(p[1]) && stake ? round2(((pnl + 2 * (fee || 0) * stake) / (stake * (1 - (fee || 0)))) * 100) : null;
      out.push({ closed: p[0], symbol: String(symbol), game: 'week', model: p[1] || null, movePct, pnl: round2(pnl), exit, raw: pnl, sold: exitDay });
    });
  });
  return out;
}

// One week position under the guard, from the rule's action for the running day (MarketSettle.guarded_now):
// hold, sell (at the running day's close: a flat or down daily call - the guard - or the week's last day), buy (on
// an up daily call: the week's start, a first up call later, or a re-entry after a sale), out. After the running
// day's close the row says what the book did that day until the next run brings the next call: a buy is held, a
// guard sale is sold. `lag`: the week record is a run behind the daily one, so its running day has closed - the
// row is the book as of that close (marketView adds today's daily call as information).
function guardedRow(market, inst, call, gs, gNow, w, wClose, now, daily, unit, { lag = false, endDay = null, nextDay = null } = {}) {
  const sym = inst.symbol;
  const crypto = market === 'crypto';
  const running = gNow.running_day;
  const rClose = closeMoment(market, running);
  const rOpen = openMoment(market, running);
  const closed = lag || Boolean(rClose && now >= rClose);
  const sessionOpen = Boolean(rOpen && now >= rOpen);
  const trades = Array.isArray(gs.trades) ? gs.trades : [];
  const lastTrade = trades.length ? trades[trades.length - 1] : null;
  const held = gs.action === 'hold' || gs.action === 'sell';
  const entry = held ? gs.entry : (gs.action === 'buy' ? running : null);
  // the entry the position is measured from: the book's price; a buy whose price is not known yet (a share bought
  // at today's open, before it) from the last close; an out row has none
  // the closes of both records by date, the daily record's where both have one: a week record ahead of the daily
  // one holds the newer close
  const byDay = new Map();
  [Array.isArray(inst.closes) ? inst.closes : [], daily && Array.isArray(daily.closes) ? daily.closes : []]
    .forEach((cs) => cs.forEach((c) => { if (Array.isArray(c) && num(c[1]) !== null) byDay.set(c[0], c); }));
  const allCloses = [...byDay.values()].sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0));
  let b = num(gs.entry_price);
  let bNote = crypto ? 'bought at the close before the entry day (the paper book)' : 'bought at the entry day\'s open';
  if (gs.action === 'out') { b = null; bNote = 'no position held'; }
  else if (b === null && gs.action === 'buy') {
    // the newest close before the running day, from whichever record is newer (the daily record's base otherwise)
    const before = allCloses.filter((c) => c[0] < running).slice(-1)[0] || null;
    b = before ? num(before[1]) : (daily ? base(daily) : null);
    bNote = crypto ? 'last close (the paper book buys at it)'
      : (closed ? `bought at the open of ${dayText(running)}; shown from the close before it until ${lag ? 'the week record catches up' : 'the morning run brings the open'}`
        : `last close; the book buys at the open of ${dayText(running)}`);
  }
  else if (b === null) { b = base(inst); bNote = 'the week base (the entry price is not in the record)'; }
  const pctOf = (v) => (v === null || b === null || b <= 0 ? null : round2((v / b - 1) * 100));
  // the newest daily close on or after the entry day: the move at the last close
  const closes = entry ? allCloses.filter((c) => c[0] >= entry) : [];
  const lastClose = closes.length ? num(closes[closes.length - 1][1]) : null;
  const sellAtWeekEnd = gNow.last_day && rClose ? rClose : wClose;
  const callText = gs.call ? `daily call ${gs.call}` : 'no daily call';
  // "today's" while the running day is today; its date otherwise (a share week read on the weekend before it)
  const todays = dayKey(now) === running ? 'today\'s' : `${dayText(running)}'s`;
  const today = dayKey(now) === running ? 'today' : dayText(running);
  const row = {
    symbol: sym, game: 'week', status: 'held', statusText: '', buyText: '—',
    openRate: b, openRateNote: bNote,
    band: { low: call.low, high: call.high, bin: call.bin, lowPct: pctOf(call.low), highPct: pctOf(call.high) },
    profitPct: null, profitPctClose: held && lastClose !== null ? pctOf(lastClose) : null,
    sellText: when(sellAtWeekEnd, now), sellAt: sellAtWeekEnd ? sellAtWeekEnd.toISOString() : null, sellNote: null,
    guard: { state: 'hold', text: callText, booked: true, reason: '' },
    since: null, modelsUp: { up: inst.next.filter((p) => isUp(p, w.k)).length, of: inst.next.length }, size: null,
    realized: num(gs.realized), action: gs.action,
  };
  // this week's earlier sales of the coin (guarded_now's trades): their day, move and money, until the week closes
  const realized = num(gs.realized);
  const moneyText = realized !== null ? `${signed(realized)} ${unit}` : null;
  let soldText = null;
  if (trades.length === 1) {
    const move = num(lastTrade[4]);
    const movePart = move !== null ? `${signed((Math.exp(move) - 1) * 100, 1)}%` : null;
    soldText = [`sold ${dayText(lastTrade[1])}`, [movePart, moneyText ? `(${moneyText})` : null].filter(Boolean).join(' ')].filter(Boolean).join(' · ');
  } else if (trades.length > 1) {
    soldText = [`sold ${trades.length}× this week`, moneyText].filter(Boolean).join(' · ');
  }
  // the guard can sell before the week's close only when a game day whose close is still to come lies before the
  // week's last day: the day after the running day, or - when the week record lags - today, or the day after it
  // once today's close has passed
  const nextClose = lag && nextDay ? closeMoment(market, nextDay) : null;
  const after = lag && nextDay ? (nextClose && now >= nextClose ? nextGameDay(market, nextDay) : nextDay) : nextGameDay(market, running);
  const sooner = Boolean(!gNow.last_day && after && endDay && after < endDay);
  const waiting = () => {
    row.guard.text = 'next call after the morning run';
    row.guard.reason = `${dayText(running)}'s daily call for ${sym} was ${gs.call || 'missing'}; the next one comes with the morning run`;
  };
  const latest = () => { if (!gNow.last_day) row.sellNote = sooner ? 'at the latest · the guard can sell sooner' : 'the week\'s close'; };
  if (held) {
    row.buyText = gs.entry ? dayText(gs.entry) : '—';
    row.since = gs.entry ? `since ${dayText(gs.entry)}` : null;
    row.statusText = `held since ${gs.entry ? dayText(gs.entry) : 'the week\'s start'}`;
    if (gs.action === 'hold') {
      row.guard.reason = gs.call ? `${todays} daily call for ${sym} is ${gs.call}, so the position is kept` : `no daily call for ${sym} ${today}: a held position is kept`;
      latest();
      if (closed) waiting();
    } else if (gs.reason === 'week end') {
      row.sellText = when(rClose, now);
      row.sellAt = rClose ? rClose.toISOString() : null;
      row.guard = { state: 'weekend', text: callText, booked: true,
                    reason: `the week ends at this close: sold ${when(rClose, now)} whatever ${todays} daily call (${gs.call || 'none'})` };
    } else {
      row.sellAt = rClose ? rClose.toISOString() : null;
      row.guard = { state: 'sell', text: callText, booked: true,
                    reason: `${todays} daily call for ${sym} is ${gs.call || 'missing'}, so the position is sold at the close (${when(rClose, now)})` };
      if (closed) {
        // the game day, as the closed positions name it, once the close is not today's ("Thu 15 Oct, close 02:00 next day")
        const soldAt = rClose && dayKey(rClose) !== dayKey(now) ? `${dayText(running)}${crypto ? `, close ${clock(rClose)} next day` : ''}` : when(rClose, now);
        row.status = 'sold'; row.sellText = '—';
        row.statusText = lag ? `sold at the close by the guard (${soldAt}); its money is booked when the week closes`
          : `sold at the close by the guard (${soldAt}); its move shows after the morning run, its money when the week closes`;
      } else {
        row.sellText = when(rClose, now);
        row.statusText += `; the guard sells it at the close (${when(rClose, now)})`;
      }
    }
  } else if (gs.action === 'buy') {
    row.guard = { state: 'buy', text: callText, booked: true, reason: `${todays} daily call for ${sym} is up, so the position is bought` };
    if (gNow.last_day) row.sellNote = 'week end'; else latest();
    if (closed) {
      // the book bought it on the running day and holds it: only the next daily call can sell it
      row.status = 'held'; row.buyText = dayText(running); row.since = `since ${dayText(running)}`;
      row.statusText = `bought ${dayText(running)}; still held${lag ? '' : ' - the next daily call comes with the morning run'}`;
      row.guard.state = 'hold';
      waiting();
    } else if (crypto || sessionOpen) {
      row.status = 'buy now'; row.buyText = 'now';
      row.statusText = gs.reason === 're-enter' ? `back in: ${todays} daily call is up again; a market order now`
        : gs.reason === 'first up call' ? 'the first up daily call this week: a market order now' : 'a market order now';
    } else {
      row.status = 'buy at open'; row.buyText = when(rOpen, now);
      row.statusText = `a market order placed before ${clock(rOpen)}, filled at the open`;
    }
  } else {
    row.status = 'out'; row.sellText = '—'; row.sellAt = null;
    row.guard = { state: 'out', text: callText, booked: true,
                  reason: gs.call ? `${todays} daily call for ${sym} is ${gs.call}, so it stays out` : `no daily call for ${sym} ${today}, so it stays out` };
    if (closed) waiting();
    // on the week's last day nothing is left to come back on: the week ends at this close
    const ends = `the week ends at the close (${when(rClose, now)})`;
    row.since = soldText || (gNow.last_day ? 'not bought this week' : 'waiting for an up call');
    row.statusText = trades.length
      ? `sold ${dayText(gs.sold_on || lastTrade[1])} by the guard; ${gNow.last_day ? `out for the rest of the week - ${ends}` : 'back in on the next up daily call'}. Its money is booked when the week closes.`
      : (gNow.last_day ? `not bought this week: no up daily call; ${ends}`
        : (Array.isArray(gNow.settled_days) && gNow.settled_days.length ? 'not bought yet this week: no up daily call so far; in on the first one'
          : `not bought yet: ${todays} daily call is ${gs.call || 'missing'}; in on the first up daily call this week`));
  }
  // a coin the guard sold earlier this week and bought back: the earlier sale stays visible until the week closes
  if (row.status !== 'out' && soldText) row.earlier = `earlier: ${soldText}`;
  return row;
}

// --- the order book (src/MarketOrders.py, 30 Oct 2026) ------------------------------
// The crypto day and week cards as orders a reader can place: buy at market when the card appears, a sell limit at X,
// sold at the limit, at market when the next card of the same game appears (a fallback clock time at the latest) or -
// a week position - at market when a daily card turns the coin flat or down. Shown on the user page from
// ORDERS_SHOWN_FROM on (markets.js; null: Admin only, the shadow), with the record split there as GUARD_SHOWN_FROM
// splits the week record: the old books before, the order book from it. Every count a labelled line says is counted
// from the same rows the tables show (orderSoFar), and each caveat sits on the line whose number it qualifies.
const HOUR_MS = 3600000;
const EUR_NOTE = 'Prices are Binance USDT; on a EUR exchange, place the limit at the same percentage above your own buy price.';
// an order price: grouped thousands, more decimals for a smaller price (a limit needs the exchange's precision)
function priceText(v) {
  const x = num(v);
  if (x === null) return '—';
  const a = Math.abs(x);
  const d = a >= 10000 ? 0 : a >= 10 ? 2 : a >= 1 ? 4 : 5;
  return x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}
const pctText = (v, d = 2) => (num(v) === null ? '—' : `${signed(num(v), d)}%`);
const dateOf = (iso) => { const d = iso ? new Date(iso) : null; return d && !Number.isNaN(d.getTime()) ? d : null; };
// "Thu 15 Oct 17:00" in Brussels time, for a moment that is not now
const stampText = (date) => (date ? `${fmtWeekday.format(date)} ${fmtDayMonth.format(date)} ${clock(date)}` : '—');
const SKIP_TEXT = {
  'target below fees': (s, minGap) => `No order: the call${s.bin !== null ? ` (bin ${s.bin})` : ''} aims at less than the ${pctText(minGap * 100).replace('+', '')} above the last close that two fees and a margin need.`,
  'already reached': (s) => `No order: the price was already above the limit${s.limit !== null ? ` ${priceText(s.limit)}` : ''} when the card appeared; the move happened before the card.`,
  'too close to the limit': (s, minGap) => `No order: the price was within ${pctText(minGap * 100).replace('+', '')} of the limit${s.limit !== null ? ` ${priceText(s.limit)}` : ''} when the card appeared: two fees would eat the gain.`,
  'no target': () => 'No order: the call is the open-ended lowest bin, which has no lower edge to place a limit at.',
  "published after the interval's end": () => 'No order: the card appeared after its day had ended.',
};
// the limit's distance from a price as a reader computes it (simple percent; the record's limit_pct is a log move)
const limitPctOf = (limit, from, logPct) => (num(limit) !== null && num(from) !== null && from > 0 ? round2((limit / from - 1) * 100) : (num(logPct) !== null ? round2((Math.exp(logPct / 100) - 1) * 100) : null));
const skipText = (s, minGap) => (SKIP_TEXT[s.reason] ? SKIP_TEXT[s.reason](s, minGap) : `No order: ${s.reason || 'no reason recorded'}.`);
const EXIT_WORDS = { limit: 'limit filled', early: 'daily call not up, at market', 'next card': 'next card, at market', fallback: 'no next card, at market (fallback)', close: '00:00 UTC close' };
// a closed order's exit, as the closed positions name it: "limit filled Thu 15 Oct 17:00", "next card, at market Fri 16 Oct 11:02"
function orderExitText(q) {
  const at = dateOf(q.exitAt);
  const word = EXIT_WORDS[q.exitKind] || 'sold';
  return `${word}${at ? ` ${stampText(at)}` : ''}${q.rolls ? ` · rolled ${q.rolls}×` : ''}${q.estimated ? ' · time estimated' : ''}`;
}
// the variant each card showed (record.shown); the primary where an interval is not listed
const shownVariant = (o, interval) => { const s = o.record.shown.find((x) => x.interval === interval); return s ? s.variant : o.primary; };

// The order book's closed positions from `from` on, as the user record counts them: the main book of the variant each
// card showed, live orders only. A day position is booked on the day it closed (a rolled one spans days), a week
// position on its week's close.
function orderPositions(o, game, keep) {
  if (!o || !keep) return [];
  const out = [];
  Object.entries(o.books).forEach(([variant, book]) => book.orders.forEach((q) => {
    if (q.kind !== 'main' || q.status !== 'closed' || !q.live || q.money === null || !keep(q.interval) || shownVariant(o, q.interval) !== variant) return;
    const movePct = q.move !== null ? round2((Math.exp(q.move) - 1) * 100) : null;
    out.push({ closed: game === 'day' ? q.days[q.days.length - 1] : q.interval, symbol: q.symbol, game, model: q.model, movePct, pnl: round2(q.money), exit: orderExitText(q),
      raw: q.money, rule: 'orders', exitKind: q.exitKind, estimated: q.estimated, entryAt: q.entryAt, entryPrice: q.entryPrice, exitPrice: q.exitPrice, rolls: q.rolls,
      sold: q.exitAt ? dayKey(new Date(q.exitAt)) : undefined });
  }));
  return out;
}
// buying every coin at each card's publication, sold at the same deadline (the order book's benchmark), from `from` on
const orderBench = (o, keep) => (o && keep ? o.benchmark.points.filter((p) => keep(p.interval) && p.closed) : []);
// a week interval is its close (the Sunday); it counts under the order book when its first day is on or after `from`
const prevDayOf = (day) => { const d = new Date(`${day}T12:00:00Z`); return Number.isNaN(d.getTime()) ? '' : new Date(d.getTime() - DAY_MS).toISOString().slice(0, 10); };
const weekFirst = (sunday) => { const d = new Date(`${sunday}T12:00:00Z`); return Number.isNaN(d.getTime()) ? '' : new Date(d.getTime() - 6 * DAY_MS).toISOString().slice(0, 10); };

// "Following this card so far" for one game: the counts of the order positions (the closed table's rows), their money
// against the benchmark, and the old record's positions before `from`.
function orderSoFar(positions, bench, game, from, old, cons, unit, { newcomer = false } = {}) {
  const count = (kind) => positions.filter((q) => q.exitKind === kind).length;
  const days = new Set(positions.map((q) => q.closed)).size;
  const total = positions.reduce((s, q) => s + q.raw, 0);
  const market = bench.reduce((s, p) => s + p.money, 0);
  const estimated = positions.filter((q) => q.estimated).length;
  const s = { from, orders: positions.length, days, filled: count('limit'), nextCard: count('next card'), fallback: count('fallback'), early: count('early'),
    total: round2(total), market: bench.length ? round2(market) : null, estimated, before: old };
  const lag = cons && cons.publishLagFallbackMin !== null ? cons.publishLagFallbackMin : 15;
  const parts = [`${s.filled} filled at the limit`];
  if (game === 'week') parts.push(`${s.early} sold on a flat or down daily call`, `${s.nextCard + s.fallback} sold at the week's end`);
  else parts.push(`${s.nextCard} sold at market when the next card appeared`);
  if (game === 'day' && s.fallback) parts.push(`${s.fallback} sold at the fallback time (no next card)`);
  const what = game === 'week' ? `${plural(s.orders, 'order', 'orders')} over ${plural(days, 'week', 'weeks')}` : `${plural(s.orders, 'order', 'orders')} on ${plural(days, 'day', 'days')}`;
  const head = s.orders
    ? `${what}: ${parts.join(', ')}; ${signed(total)} ${unit} against ${s.market !== null ? signed(market) : '—'} buying every coin at the ${game === 'week' ? "week card's" : "card's"} time.`
    : `no order closed yet since ${dayText(from)}.`;
  const caveat = s.orders
    ? ` Paper money, bought at the price when each card appeared${estimated ? ` (${estimated === s.orders ? (estimated === 1 ? 'its card time' : 'every card time') : `${estimated} of them with a card time`} estimated at ${lag} min after the ticket: publication was not recorded before 30 Oct)` : ''}; 0.1% fee per leg.`
    : '';
  const before = old && old.positions
    ? ` Before ${dayText(from)}, under the old rule (${game === 'week' ? 'the week book bought at Monday 00:00 UTC' : 'bought at 00:00 UTC, before the ticket was up'}): ${plural(old.positions, 'position', 'positions')}, ${signed(old.total)} ${unit}.`
    : '';
  s.text = `${head}${caveat}${before}${game === 'week' && s.orders ? ' Late entries are not in this line; see below.' : ''}`
    + `${game === 'day' && newcomer && s.orders ? ' Newcomer entries are not in this line; see below.' : ''}`;
  return s;
}

// The day's newcomer ledger from `keep` on (31 Oct 2026; MarketOrders newcomer_entry): every newcomer entry a roll card
// showed, in the variant that card showed, live ones only - its own labelled line, never part of "following this card
// so far". `rows` are the ledger's positions (closed, open, pending) the line counts; `notPlaced` the entries the
// fresh-order tests refused (counted, never positions).
function newcomerRecordOf(o, keep, unit) {
  if (!o || !o.newcomer || !keep) return null;
  const rows = [];
  let notPlaced = 0;
  Object.entries(o.newcomer).forEach(([variant, book]) => book.orders.forEach((q) => {
    if (q.kind !== 'newcomer' || !q.live || !keep(q.interval) || shownVariant(o, q.interval) !== variant) return;
    if (['closed', 'open', 'pending'].includes(q.status)) rows.push(q);
    else if (q.status === 'skip') notPlaced += 1;
  }));
  const closed = rows.filter((q) => q.status === 'closed');
  const total = round2(closed.reduce((s, q) => s + (q.money || 0), 0));
  const r = { shown: rows.length, closed: closed.length, open: rows.length - closed.length, filled: closed.filter((q) => q.exitKind === 'limit').length, total, notPlaced,
    rows: rows.map((q) => ({ interval: q.interval, symbol: q.symbol, status: q.status, entryPrice: q.entryPrice, limit: q.limit, exitKind: q.exitKind, money: q.money === null ? null : round2(q.money) })) };
  r.text = `Newcomer entries so far (a separate record): ${plural(r.shown, 'shown', 'shown')}${r.open ? ` (${r.open} still open)` : ''}, ${r.closed ? `${signed(total)} ${unit} on the closed ones` : 'none closed yet'}, `
    + `each counted from the price when that day's card appeared${notPlaced ? `; ${plural(notPlaced, 'entry', 'entries')} not placed (under the fee margin)` : ''}.`;
  return r;
}

// The day card's order rows from MarketOrders.orders_now. Every line is written from the READER's side (owner, 31 Oct
// 2026): never the book's state as a bare status ("held", "hold (rolled)" read as "you hold it"), but one branch per
// reader situation, each labelled "If you bought ..." / "If you did not buy ...", and a branch invites only what a
// ledger books. A roll (D5) continues yesterday's position; the reader who did not buy yesterday is told to buy now -
// the NEWCOMER entry, booked in its own ledger (MarketOrders newcomer_entry, owner 31 Oct 2026), never in "following
// this card so far" - or, when that entry is not placed, why, in the reader's words. A call that is not up is a chip,
// as before. `status` stays the book's code (the client maps it to a chip; order.chip is the reader's words).
const branch = (who, text, extra = {}) => ({ who, text, ...extra });
const textOf = (branches) => branches.map((b) => (b.who ? `${b.who}: ${b.text}` : b.text)).join(' ');
function dayOrderRows(o, view, now, unit) {
  const n = o.now;
  const variant = n.variant || o.primary;
  const book = o.books[variant] || null;
  const pub = dateOf(n.publishedAt);
  const fallback = dateOf(n.deadlineFallback);
  const close = closeMoment('crypto', n.interval);
  const minGap = o.constants.minGap !== null ? o.constants.minGap : 0.003;
  const minGapText = pctText(minGap * 100).replace('+', '');
  const rows = [], none = [];
  const recordBought = pub && now - pub > HOUR_MS;
  const deadline = fallback ? `if it has not filled when tomorrow's card appears (by ${when(fallback, now)} at the latest), sell at market` : 'if it has not filled when tomorrow\'s card appears, sell at market';
  view.instruments.filter((i) => i.active).forEach((inst) => {
    const sym = inst.symbol;
    const s = n.symbols[sym];
    if (!s) return;
    if (s.status === 'no order' && s.call !== 'up') { none.push({ symbol: sym, call: s.call === 'down' ? 'down' : 'flat' }); return; }
    const call = inst.next.find((p) => p.model === n.model) || null;
    const q = book ? book.orders.find((x) => x.kind === 'main' && x.symbol === sym && x.days.includes(n.interval)) || null : null;
    // the record's exit counts once it has happened (a record read back in time - a test, a preview - knows later fills)
    const exited = Boolean(q && q.status === 'closed' && dateOf(q.exitAt) && dateOf(q.exitAt) <= now);
    const entry = q && q.entryPrice !== null ? q.entryPrice : s.entryPrice;
    const pctOf = (v) => (v === null || entry === null || entry <= 0 ? null : round2((v / entry - 1) * 100));
    const limitPct = limitPctOf(s.limit, entry, s.limitPct);
    const order = { variant, limit: s.limit, limitPct, entryPrice: entry, entryAt: q ? q.entryAt : null, publishedAt: n.publishedAt, estimated: n.estimated,
      deadlineAt: fallback ? fallback.toISOString() : null, deadlineText: fallback ? when(fallback, now) : null, exitRule: 'at the limit, or at market when tomorrow\'s card appears',
      skipReason: null, heldSince: s.heldSince, rolls: q ? q.rolls : 0, belowEntry: false, exitKind: exited ? q.exitKind : null,
      exitAt: exited ? q.exitAt : null, exitPrice: exited ? q.exitPrice : null, money: exited ? round2(q.money) : null, provisional: s.provisional, chip: '', branches: [], text: '',
      newcomer: null };
    const row = {
      symbol: sym, game: 'day', status: 'buy now', statusText: '', buyText: 'now',
      openRate: entry, openRateNote: `the price when the card appeared${pub ? `, ${clock(pub)}` : ''}${n.estimated ? ' (time estimated)' : ''}: the record's entry`,
      band: call ? { low: call.low, high: call.high, bin: call.bin, lowPct: pctOf(call.low), highPct: pctOf(call.high) } : null,
      profitPct: null, profitPctClose: null, sellText: 'at the limit', sellAt: fallback ? fallback.toISOString() : null,
      sellNote: fallback ? `or at market at the next card, ${when(fallback, now)} at the latest` : 'or at market at the next card',
      guard: null, since: null, modelsUp: { up: inst.next.filter((p) => isUp(p, view.k)).length, of: inst.next.length }, size: null, order,
    };
    const since = s.heldSince ? dayText(s.heldSince) : 'yesterday';
    // who holds the coin into this card: the main position bought on `since`'s card, and newcomer entries of later
    // roll cards. "yesterday" when the position was bought on the card just before this one (the owner's words)
    const yesterday = Boolean(s.heldSince && s.heldSince === prevDayOf(n.interval));
    const held = s.newcomersHeld || [];
    const heldWho = yesterday
      ? `If you bought ${sym} yesterday${entry !== null ? ` (at about ${priceText(entry)} ${unit})` : ''}`
      : `If you bought ${sym} on an earlier card and still have it (the record bought it on ${since}'s card${entry !== null ? ` at about ${priceText(entry)} ${unit}` : ''}`
        + `${held.length ? `; newcomer entries ${held.map((h) => `on ${dayText(h.since)}'s card${h.entryPrice !== null ? ` at about ${priceText(h.entryPrice)} ${unit}` : ''}`).join(' and ')}` : ''})`;
    const freshWho = yesterday ? `If you did not buy ${sym} yesterday` : `If you do not have ${sym} from an earlier card`;
    const noPosition = () => { row.buyText = '—'; row.sellText = '—'; row.sellAt = null; row.sellNote = null; row.openRate = null; row.openRateNote = 'no position'; };
    if (s.status === 'no order') {
      row.status = 'no order'; noPosition();
      order.skipReason = s.reason; order.chip = 'No order';
      order.branches = [branch('', skipText(s, minGap))];
    } else if (s.status === 'sell now') {
      row.status = 'sell now'; row.buyText = '—'; row.sellText = 'now'; row.sellNote = 'at market';
      order.chip = 'If bought: sell now';
      const sellWho = held.length ? heldWho : `If you bought ${sym} on ${since}'s card`;
      // the position ends at this card: called up again but the new limit is too close (D5 refused), or not called up
      const upAgain = s.call === 'up';
      order.branches = [
        branch(sellWho, upAgain
          ? `sell at market now - cancel your limit first. Today's card calls ${sym} up again, but its new limit would be less than ${minGapText} above the price, which two fees would eat.`
          : `sell at market now - cancel your limit first. Today's card does not call ${sym} up (${s.call || 'no call'}).`),
        branch(held.length ? freshWho : `If you did not buy ${sym} on ${since}'s card`, upAgain
          ? 'no order today: for the same reason, a new buy at this card would not clear the fees.'
          : 'no order today: today\'s call is not up.'),
      ];
    } else if (exited) {
      row.status = 'sold'; row.buyText = '—'; row.sellText = '—'; row.sellNote = null;
      const move = q.move !== null ? pctText((Math.exp(q.move) - 1) * 100) : null;
      const how = q.exitKind === 'limit' ? `your limit filled ${stampText(dateOf(q.exitAt))} at ${priceText(q.exitPrice)} ${unit}` : `the record sold at market ${stampText(dateOf(q.exitAt))} at ${priceText(q.exitPrice)} ${unit}`;
      order.chip = q.exitKind === 'limit' ? 'Limit filled' : 'Sold';
      // a rolled position: the holders from an earlier card and today's newcomers share the limit, each with its own price
      const rolled = s.status === 'hold (rolled)';
      order.branches = [
        rolled
          ? branch(`If you bought ${sym} on an earlier card or on this card`, `${how} - nothing more to do.`)
          : branch(`If you bought ${sym} on this card`, `${how}${move ? `, ${move} from the card's price` : ''} (${signed(q.money)} ${unit} on the record) - nothing more to do.`),
        branch(`If you have not bought ${sym}`, `no order now: the limit has already been reached. Wait for tomorrow's card.`),
      ];
    } else if (s.status === 'hold (rolled)') {
      row.status = 'hold (rolled)';
      order.belowEntry = s.limit !== null && entry !== null && s.limit < entry;
      const nc = s.newcomer;
      const ncBuy = Boolean(nc && nc.status === 'buy now');
      row.buyText = ncBuy ? 'now' : '—';
      // the entry shown is the earlier card's (the position continues), not today's card price
      row.openRateNote = `the price when ${since}'s card appeared: the record's entry${ncBuy && nc.entryPrice !== null ? `; a newcomer buys at today's card price, about ${priceText(nc.entryPrice)}` : ''}`;
      order.chip = ncBuy ? 'Keep, or buy now' : 'If bought: keep';
      const keep = branch(heldWho,
        `keep it; replace your sell limit with ${priceText(s.limit)} ${unit} (${pctText(limitPct)} from that buy price); ${deadline}.`
        + (order.belowEntry ? ` This limit is below your buy price: a fill sells at a loss of about ${pctText(limitPct).replace(/^[+−-]/, '')}.` : ''),
        { belowEntry: order.belowEntry });
      let fresh;
      if (ncBuy) {
        const ncPct = limitPctOf(nc.limit, nc.entryPrice, nc.limitPct);
        order.newcomer = { limit: nc.limit, limitPct: ncPct, entryPrice: nc.entryPrice, skipReason: null, provisional: nc.provisional };
        fresh = branch(freshWho, `buy now at market (about ${priceText(nc.entryPrice)} ${unit}${recordBought ? `; the newcomer record bought at ${clock(pub)} at that price` : ''}), `
          + `sell limit ${priceText(nc.limit)} ${unit} (${pctText(ncPct)} from the card's price); ${deadline}. A separate record counts these newcomer entries, not "following this card so far".`
          + (nc.provisional ? ' The record\'s price at this card is provisional; its money settles on the next update.' : ''));
      } else if (nc) {
        order.newcomer = { limit: nc.limit, limitPct: null, entryPrice: nc.entryPrice, skipReason: nc.reason, provisional: nc.provisional };
        fresh = branch(freshWho, skipText({ ...nc, bin: s.bin }, minGap).replace(/^No order: /, 'no order today: '));
      } else {
        // a record from before the newcomer ledger (31 Oct 2026): it booked no entry, so the card invites none
        fresh = branch(freshWho, 'no new order today: today\'s card continues that position, it does not open a new one.');
      }
      order.branches = [keep, fresh];
    } else {
      // the card's buy: a market order while its day runs; after the day's close the record keeps it until the next card
      const dayOver = Boolean(close && now >= close);
      row.status = dayOver ? 'held' : 'buy now';
      row.buyText = dayOver ? '—' : 'now';
      const keep = branch(`If you bought ${sym} on this card`, `keep your sell limit at ${priceText(s.limit)} ${unit}; ${deadline}.`);
      if (dayOver) {
        order.chip = 'If bought: keep';
        order.branches = [keep, branch(`If you have not bought ${sym}`, 'no order now: this card\'s day has ended. Wait for tomorrow\'s card.')];
      } else {
        order.chip = 'Buy now';
        order.branches = [
          branch(`If you have not bought ${sym} today`, `buy now at market (about ${priceText(entry)} ${unit}${recordBought ? `; the record bought at ${clock(pub)} at that price` : ''}). `
            + `Place a sell limit at ${priceText(s.limit)} ${unit} (${pctText(limitPct)} from the card's price); ${deadline}.`),
          keep,
        ];
      }
    }
    // a sale or roll at this card booked on the last stored minute (MarketOrders "provisional"): its money comes later
    if (s.provisional && (s.status === 'sell now' || s.status === 'hold (rolled)') && order.branches[0]) {
      order.branches[0].text += ' The record\'s price at this card is provisional; its money settles on the next update.';
    }
    order.text = textOf(order.branches);
    row.statusText = order.text;
    rows.push(row);
  });
  return { rows, none };
}

// The week card's order rows (MarketOrders.orders_now, week), from the reader's side as the day card: a coin bought on
// the week card (the week call and Monday's daily call up, D4) keeps its limit and is sold at market on a flat or down
// daily card (D2), never bought back (D3); "if you have not bought it this week" is a branch only while the running day's
// card shows the late entry (D7: today's card, the price at least 1% below the limit, 24 h left) - its own record.
function weekOrderRows(o, w, nextDay, now, unit) {
  const n = o.now;
  const variant = n.variant || o.primary;
  const book = o.books[variant] || null;
  const lateBook = o.late ? o.late[variant] || null : null;
  const pub = dateOf(n.publishedAt);
  const fallback = dateOf(n.deadlineFallback);
  const firstClose = n.firstDay ? closeMoment('crypto', n.firstDay) : null;
  const minGap = o.constants.minGap !== null ? o.constants.minGap : 0.003;
  const lateGap = o.constants.lateMinGap !== null ? o.constants.lateMinGap : 0.01;
  const leftMs = o.constants.lateMinLeftHours !== null ? o.constants.lateMinLeftHours * HOUR_MS : 24 * HOUR_MS;
  const rows = [], none = [], late = [];
  const nextCard = fallback ? `when next Monday's card appears (by ${when(fallback, now)} at the latest)` : "when next Monday's card appears";
  const lateToday = Boolean(n.runningDay && n.runningDay === nextDay && n.runningDay !== n.firstDay && fallback && fallback - now >= leftMs);
  const weekCard = (extra) => `this week's card${pub || extra ? ` (${[pub ? stampText(pub) : '', extra || ''].filter(Boolean).join(', ')})` : ''}`;
  const lateGapText = `${round2(lateGap * 100)}%`;
  w.instruments.filter((i) => i.active).forEach((inst) => {
    const sym = inst.symbol;
    const s = n.symbols[sym];
    if (!s) return;
    if (s.status === 'no order' && s.call !== 'up') { none.push({ symbol: sym, call: s.call === 'down' ? 'down' : 'flat' }); return; }
    const call = inst.next.find((p) => p.model === n.model) || null;
    const q = book ? book.orders.find((x) => x.kind === 'main' && x.symbol === sym && x.interval === n.interval) || null : null;
    const entry = q && q.entryPrice !== null ? q.entryPrice : s.entryPrice;
    const pctOf = (v) => (v === null || entry === null || entry <= 0 ? null : round2((v / entry - 1) * 100));
    const limitPct = limitPctOf(s.limit, entry, s.limitPct);
    const exitAt0 = dateOf(s.exitAt || (q && q.status === 'closed' ? q.exitAt : null));
    const closedNow = Boolean((s.state === 'closed' || (q && q.status === 'closed')) && exitAt0 && exitAt0 <= now);
    const order = { variant, limit: s.limit, limitPct, entryPrice: entry, entryAt: q ? q.entryAt : null, publishedAt: n.publishedAt, estimated: n.estimated,
      deadlineAt: fallback ? fallback.toISOString() : null, deadlineText: fallback ? when(fallback, now) : null,
      exitRule: "at the limit, at market on a flat or down daily card, or at market when next Monday's card appears", skipReason: null, heldSince: null, rolls: 0, belowEntry: false,
      exitKind: closedNow ? s.exitKind || (q ? q.exitKind : null) : null, exitAt: closedNow ? exitAt0.toISOString() : null,
      exitPrice: closedNow && q ? q.exitPrice : null, money: closedNow ? (s.money !== null ? round2(s.money) : (q ? round2(q.money) : null)) : null, provisional: false,
      chip: '', branches: [], text: '' };
    const row = {
      symbol: sym, game: 'week', status: 'buy now', statusText: '', buyText: 'now',
      openRate: entry, openRateNote: `the price when the week card appeared${pub ? `, ${stampText(pub)}` : ''}${n.estimated ? ' (time estimated)' : ''}: the record's entry`,
      band: call ? { low: call.low, high: call.high, bin: call.bin, lowPct: pctOf(call.low), highPct: pctOf(call.high) } : null,
      profitPct: null, profitPctClose: null, sellText: 'at the limit', sellAt: fallback ? fallback.toISOString() : null,
      sellNote: fallback ? `or at market on a flat or down daily card, or at the next week card (${when(fallback, now)} at the latest)` : 'or at market on a flat or down daily card, or at the next week card',
      guard: null, since: null, modelsUp: { up: inst.next.filter((p) => isUp(p, w.k)).length, of: inst.next.length }, size: null, order,
    };
    const sellRule = `Sell at market if a daily card turns ${sym} flat or down (cancel the limit first); if it has not filled ${nextCard}, sell at market.`;
    const noPosition = () => { row.buyText = '—'; row.sellText = '—'; row.sellAt = null; row.sellNote = null; row.openRate = null; row.openRateNote = 'no position'; };
    // the late entry on today's card, as a branch for a reader who has not bought the coin this week
    const l = n.late[sym] || null;
    let lateBranch = null;
    if (l && l.shown && lateToday && l.price !== null && l.limit !== null) {
      const entry1 = { symbol: sym, price: l.price, limit: l.limit, gapPct: l.gapPct, minGapPct: lateGap * 100, day: n.runningDay };
      lateBranch = branch(`If you have not bought ${sym} this week`,
        `still good to enter now - today's daily call is up and the price is ${pctText(l.gapPct).replace('+', '')} below the limit. Buy at market (about ${priceText(l.price)} ${unit}), `
        + `sell limit ${priceText(l.limit)} ${unit}, sell at market if a daily card turns ${sym} flat or down, and sell at market ${nextCard} at the latest.`.replace(' at the latest) at the latest.', ' at the latest).'),
        { late: { limit: l.limit, minGapPct: lateGap * 100 } });
      entry1.text = `${lateBranch.who}: ${lateBranch.text}`;
      late.push(entry1);
    }
    const LATE_WHY = { 'daily call not up': "today's daily call is not up", 'price within 1 % of the limit': `the price is within ${lateGapText} of the limit`,
      'less than 24 h left': 'less than 24 h are left before the orders end' };
    const notLate = (monday) => {
      const why = [monday ? "Monday's daily call was not up" : '', l && l.reason ? LATE_WHY[l.reason] || l.reason : ''].filter(Boolean);
      return branch(`If you have not bought ${sym} this week`, `no order today${why.length ? `: ${why.join(', and ')}` : ''}. `
        + `If a later daily call is up while the price is at least ${lateGapText} below the limit and a day or more is left, that day's card says "still good to enter".`);
    };
    // a reader who took an earlier late entry of this week: the same limit, early exit and deadline
    const lateHeld = lateBook ? lateBook.orders.some((x) => x.kind === 'late' && x.symbol === sym && x.interval === n.interval && x.status === 'open') : false;
    const lateKeep = lateHeld ? branch(`If you bought ${sym} on an earlier "still good to enter" line this week`, `keep it; your sell limit stays at ${priceText(s.limit)} ${unit}. ${sellRule}`) : null;
    if (s.status === 'no order') {
      row.status = 'no order'; noPosition();
      order.skipReason = s.reason; order.chip = 'No order';
      order.branches = [branch('', skipText(s, minGap))];
    } else if (s.status === 'not bought') {
      row.status = 'not bought'; noPosition();
      order.skipReason = s.reason;
      order.chip = lateBranch ? 'Still good to enter' : 'No order today';
      order.branches = [lateBranch || notLate(s.reason === 'daily call not up'), lateKeep].filter(Boolean);
    } else if (closedNow) {
      row.status = 'sold'; row.buyText = '—'; row.sellText = '—'; row.sellNote = null;
      const exitDay = order.exitAt ? fmtWeekday.format(dateOf(order.exitAt)) : '';
      const move = q && q.move !== null ? pctText((Math.exp(q.move) - 1) * 100, 1) : null;
      const money = order.money !== null ? ` (${signed(order.money)} ${unit} on the record)` : '';
      const how = order.exitKind === 'early' ? `it was sold at market on ${exitDay}'s daily card (its call was not up)`
        : order.exitKind === 'limit' ? `your limit filled ${stampText(dateOf(order.exitAt))}` : `it was sold at market ${stampText(dateOf(order.exitAt))}`;
      order.chip = order.exitKind === 'limit' ? 'Limit filled' : 'Sold';
      order.branches = [branch(`If you bought ${sym} on ${weekCard()}`, `${how}${move ? `, ${move}` : ''}${money}. Do not buy it back this week.`), lateBranch || lateKeep].filter(Boolean);
    } else if (s.status === 'sell now') {
      row.status = 'sell now'; row.buyText = '—'; row.sellText = 'now'; row.sellNote = 'at market';
      order.chip = 'If bought: sell now';
      order.branches = [branch(`If you hold ${sym} from ${weekCard()}`, `sell at market now (today's daily call is not up) - cancel your limit first. Do not buy it back this week.`),
        branch(`If you have not bought ${sym} this week`, 'no order.')];
    } else {
      const first = Boolean(firstClose && now < firstClose);
      row.status = first ? 'buy now' : 'held';
      row.buyText = first ? 'now' : '—';
      const keep = branch(`If you bought ${sym} on ${weekCard(entry !== null ? `at about ${priceText(entry)} ${unit}` : '')}${lateHeld ? ' or on an earlier "still good to enter" line' : ''}`,
        `keep it; your sell limit stays at ${priceText(s.limit)} ${unit} (${pctText(limitPct)} from that price). ${sellRule}`);
      if (first) {
        order.chip = 'Buy now';
        order.branches = [branch(`If you have not bought ${sym} this week`, `buy now at market (about ${priceText(entry)} ${unit}). Place a sell limit at ${priceText(s.limit)} ${unit} (${pctText(limitPct)}). ${sellRule}`)];
      } else {
        order.chip = lateBranch ? 'Still good to enter' : 'If bought: keep';
        order.branches = [keep, lateBranch || notLate(false)];
      }
    }
    order.text = textOf(order.branches);
    row.statusText = order.text;
    rows.push(row);
  });
  return { rows, none, late };
}

// --- the view ------------------------------------------------------------------
// `guardFrom`: the first week the guard counts (GUARD_SHOWN_FROM; the tests move it). `ordersFrom`: the first day (a
// week: its close) the order book's card and record count (ORDERS_SHOWN_FROM; null keeps the old card; the tests set it).
function marketView(market, record, weekRecord, now = new Date(), { guardFrom = GUARD_SHOWN_FROM, ordersFrom = ORDERS_SHOWN_FROM } = {}) {
  const meta = MARKETS[market];
  if (!meta || !record) return null;
  const view = describeMarket(record, { week: weekRecord ? describeMarket(weekRecord) : null });
  const k = view.k;
  const t = view.trading;
  const f = t ? t.followed : null;
  const w = view.week;
  const wf = w && w.trading ? w.trading.followed : null;
  // today's plan: the followed day model's calls as positions
  const plan = pickPlan(view, market);
  const nextDay = nextGameDay(market, view.madeOn);
  // the order book (crypto): the day and week records' trading.orders, counted from ordersFrom on - once a card of
  // ordersFrom or later is on the page (the day card for that day, or the week card whose week starts on it). Before
  // that the user page is exactly the old one (owner, 31 Oct 2026: the switch to 2 Nov changes nothing until then); the
  // gate is the card's day, not the clock, so a late run on the switch day keeps the old page until its card is out.
  const weekStart = w && w.madeOn ? nextGameDay(market, w.madeOn) : null;
  const ordersOn = Boolean(ordersFrom && ((nextDay && nextDay >= ordersFrom) || (weekStart && weekStart >= ordersFrom)));
  const dayBook = market === 'crypto' && ordersOn && t && t.orders && t.orders.game === 'day' ? t.orders : null;
  const weekBook = market === 'crypto' && ordersOn && w && w.trading && w.trading.orders && w.trading.orders.game === 'week' ? w.trading.orders : null;
  const orderDay = dayBook ? (date) => date >= ordersFrom : null;
  const orderWeek = weekBook ? (date) => weekFirst(date) >= ordersFrom : null;
  const closeAt = nextDay ? closeMoment(market, nextDay) : null;
  const openAt = nextDay ? openMoment(market, nextDay) : null;
  const dayClosed = Boolean(closeAt && now >= closeAt);
  const sessionOpen = Boolean(openAt && now >= openAt);
  const dayWord = market === 'crypto' ? 'day' : 'session';
  const dayLabel = nextDay ? fmtDayMonth.format(new Date(`${nextDay}T12:00:00Z`)) : '';
  const hoursIn = openAt && sessionOpen ? Math.round(((Math.min(now, closeAt || now) - openAt) / 3600000) * 10) / 10 : null;
  const dayState = !nextDay ? '' : (dayClosed ? `${dayWord} ${dayLabel} closed` : (sessionOpen ? `${dayWord} ${dayLabel} running, ${hoursIn} h in` : `${dayWord} ${dayLabel} not started`));
  const open = [];
  const none = { day: [], week: [] };
  const dailyCall = {};   // symbol -> the followed day model's call, for the week guard
  // the day card from the order book: its state is for this card's day (a record whose order state is for another day
  // keeps the old card rather than show orders for a day it is not about)
  const dayOrders = dayBook && dayBook.now && nextDay && dayBook.now.interval === nextDay && nextDay >= ordersFrom ? dayBook.now : null;
  let dayLate = null;
  if (nextDay && plan.model) {
    view.instruments.filter((i) => i.active).forEach((inst) => {
      const call = inst.next.find((p) => p.model === plan.model.name) || null;
      if (call) dailyCall[inst.symbol] = call;
      const ups = inst.next.filter((p) => isUp(p, k)).length;
      if (!call || dayOrders) return;
      if (!isUp(call, k)) { none.day.push({ symbol: inst.symbol, call: callWord(call, k) }); return; }
      const b = base(inst);
      const pctOf = (v) => (v === null || b === null || b <= 0 ? null : round2((v / b - 1) * 100));
      // buyText: when to buy, short - "now", "today 15:30" - next to the sell time in the table
      let status, statusText, buyText;
      if (dayClosed) { status = 'too late'; statusText = `closed at ${clock(closeAt)}; the result comes with the morning run`; buyText = '—'; }
      else if (market === 'crypto' || sessionOpen) { status = 'buy now'; statusText = 'a market order now'; buyText = 'now'; }
      else { status = 'buy at open'; statusText = `a market order placed before ${clock(openAt)}, filled at the open`; buyText = when(openAt, now); }
      open.push({
        symbol: inst.symbol, game: 'day', status, statusText, buyText,
        openRate: b, openRateNote: market === 'crypto' ? 'last close (the paper book buys at 00:00 UTC)' : 'last close; the book buys at the open',
        band: { low: call.low, high: call.high, bin: call.bin, lowPct: pctOf(call.low), highPct: pctOf(call.high) },
        profitPct: null, profitPctClose: null,
        sellText: dayClosed ? '—' : when(closeAt, now),
        sellAt: closeAt ? closeAt.toISOString() : null,
        guard: null, since: null,
        modelsUp: { up: ups, of: inst.next.length },
        size: call.size !== null && call.size !== undefined ? call.size : null,
      });
    });
  }
  if (dayOrders) {
    const d = dayOrderRows(dayBook, view, now, meta.unit);
    open.push(...d.rows);
    none.day.push(...d.none);
  }

  // this week: the followed week model's calls as positions under the daily guard (MarketSettle.guarded_week's
  // `now`) from the week the guard counts (guardFrom); before it, or on a record without the guard block, the plain
  // week with today's daily call as information
  const guarded = weekRecord && weekRecord.trading && weekRecord.trading.guarded && typeof weekRecord.trading.guarded === 'object' ? weekRecord.trading.guarded : null;
  let week = null;
  let weekOrders = null;    // this week's rows are the order book's (weekOrderRows)
  let guardShown = false;   // this week's rows follow the guard
  if (w && w.madeOn && nextGameDay(market, w.madeOn)) {
    const pick = pickWeek(w, market);
    const firstSession = nextGameDay(market, w.madeOn);
    const endDay = weekEnd(market, w.madeOn);
    const wClose = endDay ? closeMoment(market, endDay) : null;
    const wOpen = firstSession ? openMoment(market, firstSession) : null;
    const started = Boolean(wOpen && now >= wOpen);
    const wClosed = Boolean(wClose && now >= wClose);
    // the entry window: the week's first day, as MarketSettle.entry_instant counts a crypto week (the ticket is up
    // hours after Monday 00:00 UTC) and as the page says "a market order now" for a share week's first session
    const firstClose = firstSession ? closeMoment(market, firstSession) : null;
    const entering = Boolean(started && firstClose && now < firstClose);
    if (pick.model && endDay && wClose && wOpen) {
      week = { endDay, startText: when(wOpen, now), stateText: `week ${fmtDayMonth.format(wOpen)} → ${fmtDayMonth.format(new Date(`${endDay}T12:00:00Z`))}`,
               stateTitle: dayStatusWeek(now, wOpen, wClose) || '', closeText: when(wClose, now),
               model: pick.model.name, modelCount: ticketModels(w.instruments) || w.models.length, ruleName: pick.ruleName, byBook: pick.byBook,
               minTrades: markets.MIN_WEEK_TRADES };
      // the guard's state applies only when it is for this week and for the day the daily rows are for: a week
      // record that lags the daily one (its export failed or was skipped) would apply yesterday's guard to today
      const useGuard = Boolean(guarded) && firstSession >= guardFrom;
      guardShown = useGuard;
      const gWeek = useGuard && guarded.now && guarded.now.first_day === firstSession && guarded.now.symbols ? guarded.now : null;
      // the week record for the daily rows' day - or ahead of them (the daily export failed): its state is the newer one
      const gNow = gWeek && nextDay && gWeek.running_day >= nextDay ? gWeek : null;
      // a week record a run behind the daily one: its state is still the book as of its running day's close (sales and
      // coins never bought stay true), shown with today's daily call as information until the next run
      const gLag = !gNow && gWeek && nextDay && gWeek.running_day < nextDay ? gWeek : null;
      week.guarded = Boolean(gNow);
      week.guardOrders = useGuard;   // the guard gives this week's orders (not information only)
      if ((gNow || gLag) && (gNow || gLag).week_model) { week.model = (gNow || gLag).week_model; }
      // the week card from the order book, from the first week starting on or after ordersFrom: its state is for this week
      const wNow = weekBook && weekBook.now && weekBook.now.firstDay === firstSession && firstSession >= ordersFrom ? weekBook.now : null;
      if (wNow) {
        const wr = weekOrderRows(weekBook, w, nextDay, now, meta.unit);
        weekOrders = wr;
        week.orders = true; week.guarded = false; week.guardOrders = false; guardShown = false;
        if (wNow.model) week.model = wNow.model;
        open.push(...wr.rows);
        none.week.push(...wr.none);
      }
      w.instruments.filter((i) => !wNow && i.active).forEach((inst) => {
        const call = inst.next.find((p) => p.model === week.model) || null;
        if (!call) return;
        if (!isUp(call, w.k)) { none.week.push({ symbol: inst.symbol, call: callWord(call, w.k) }); return; }
        if (wClosed) return;
        const gState = gNow || gLag;
        const gs = gState ? gState.symbols[inst.symbol] : null;
        const daily = view.instruments.find((i) => i.symbol === inst.symbol);
        if (gNow && gs) { open.push(guardedRow(market, inst, call, gs, gNow, w, wClose, now, daily, meta.unit, { endDay, nextDay })); return; }
        const todays = dailyCall[inst.symbol] || null;
        // Today's daily call for the coin, shown next to the week position as information: a record without the
        // guard block (before 17 Oct 2026), or a guard state that is not for today (the week record lags the daily
        // one). A sell the rows cannot vouch for would be an order nobody can check, so the Sell column keeps the
        // week's end and the help says so.
        let guard = null;
        if (started && todays && !dayClosed) {
          const pending = useGuard
            ? 'the guard\'s state for today is not in the week record yet - it comes with the next run'
            : (guarded ? `this page counts the guard from the week of ${dayText(guardFrom)}, so this week the sell stays at the week's end`
              : 'it is not in the record yet, so the sell here stays at the week\'s end');
          guard = isUp(todays, k)
            ? (nextDay === endDay
              ? { state: 'weekend', text: 'daily call up', booked: false, reason: `today is the week's last day: the guard rule sells ${inst.symbol} at the close whatever the daily call (up, bin ${todays.bin})` }
              : { state: 'hold', text: 'daily call up', booked: false, reason: `today's daily call for ${inst.symbol} is up (bin ${todays.bin}); the guard rule would hold` })
            : { state: 'sell', text: `daily call ${callWord(todays, k)}`, booked: false, reason: `today's daily call for ${inst.symbol} is ${callWord(todays, k)} (bin ${todays.bin}); the guard rule would sell at the close - ${pending}` };
        }
        // a coin the book does not hold: on an up call the rule would buy it, otherwise it stays out
        const notHeld = () => {
          if (!guard) return false;
          const up = isUp(todays, k);
          guard.state = up ? 'buy' : 'out';
          guard.reason = `today's daily call for ${inst.symbol} is ${up ? 'up' : callWord(todays, k)} (bin ${todays.bin}); the guard rule would ${up ? 'buy it' : 'leave it out'} - the guard's state for today is not in the week record yet - it comes with the next run`;
          return up;
        };
        if (gLag && gs) {
          const row = guardedRow(market, inst, call, gs, gLag, w, wClose, now, daily, meta.unit, { lag: true, endDay, nextDay });
          if (row.status !== 'held' && notHeld() && row.status === 'out' && !(gs.trades || []).length) row.since = 'daily call up today: in by the rule';
          row.guard = guard;
          let runs = 0;
          for (let d = gLag.running_day; d && d < nextDay && runs < 30; d = nextGameDay(market, d)) runs += 1;
          const behind = runs > 1 ? `${runs} runs behind` : 'a run behind';
          row.statusText = `${row.statusText.replace(/\.$/, '')}. The week record is ${behind} the daily one: this is the book as of the close of ${dayText(gLag.running_day)}; today's guard state comes with the next run.`;
          open.push(row);
          return;
        }
        if (useGuard) {
          // a guard block without this week's state (guarded_now None: the daily record has no ticket for this week's
          // days yet): the guard buys nothing on a day without a daily call, so the plain week's entry is never shown
          const up = notHeld();
          open.push({
            symbol: inst.symbol, game: 'week', status: 'out', buyText: '—',
            statusText: guard ? 'the guard\'s state for this week is not in the week record yet; today\'s daily call is shown as information'
              : 'not bought yet this week: no daily call for this week in the record yet; the guard buys on the first up daily call',
            openRate: null, openRateNote: 'no position held',
            band: { low: call.low, high: call.high, bin: call.bin, lowPct: null, highPct: null },
            profitPct: null, profitPctClose: null, sellText: '—', sellAt: null, sellNote: null,
            guard, since: up ? 'daily call up today: in by the rule' : 'waiting for an up call',
            modelsUp: { up: inst.next.filter((p) => isUp(p, w.k)).length, of: inst.next.length }, size: null,
          });
          return;
        }
        const b = base(inst);
        const since = daily ? daily.closes.filter((p) => p[0] > w.madeOn) : [];
        const lastClose = since.length ? since[since.length - 1][1] : null;
        const profitPctClose = lastClose !== null && b ? round2((lastClose / b - 1) * 100) : null;
        const pctOf = (v) => (v === null || b === null || b <= 0 ? null : round2((v / b - 1) * 100));
        let status, statusText, buyText;
        if (!started) { status = market === 'crypto' ? 'buy now' : 'buy at open'; statusText = market === 'crypto' ? `a market order at the week's start, ${when(wOpen, now)}` : `a market order placed before ${clock(wOpen)}, filled at the week's first open`; }
        else if (entering) { status = market === 'crypto' || sessionOpen ? 'buy now' : 'buy at open'; statusText = market === 'crypto' ? `a market order now; the book entered at the week's start, ${when(wOpen, now)}` : (sessionOpen ? 'a market order now, at the current price; the book bought at the open' : `a market order placed before ${clock(wOpen)}, filled at the open`); }
        else { status = 'held'; statusText = `held since ${when(wOpen, now)}`; }
        // when to buy: at the week's start before it, now during its first day, and once held the day it was bought
        buyText = !started ? when(wOpen, now) : (entering ? (market === 'crypto' || sessionOpen ? 'now' : when(wOpen, now)) : when(wOpen, now));
        open.push({
          symbol: inst.symbol, game: 'week', status, statusText, buyText,
          openRate: b, openRateNote: market === 'crypto' ? 'the week base (the book buys at the week\'s start)' : 'the week base; the book buys at the first open',
          band: { low: call.low, high: call.high, bin: call.bin, lowPct: pctOf(call.low), highPct: pctOf(call.high) },
          profitPct: null, profitPctClose,
          sellText: when(wClose, now), sellAt: wClose.toISOString(),
          guard, since: `since ${when(wOpen, now)}`,
          modelsUp: { up: inst.next.filter((p) => isUp(p, w.k)).length, of: inst.next.length },
          size: null,
        });
      });
    }
  }

  // the record: what following the whole table made - the daily positions and the weekly ones together,
  // as a reader who does every row would have it (15 Oct 2026: one headline, not one per horizon). A
  // week's result counts on the day it settles, next to that day's daily result; the market under the
  // same rules is added the same way.
  // the day positions as the page told them: the old daily book before ordersFrom (every day while it is null), the
  // order book from it
  const oldDay = (date) => !orderDay || !orderDay(date);
  const dayOrderPositions = orderPositions(dayBook, 'day', orderDay);
  const dayPositions = positionsOf(market, view, f, 'day').filter((q) => oldDay(q.closed)).concat(dayOrderPositions);
  // the weekly positions as the page told them: the plain week book for weeks closing before guardFrom (or every
  // week of a record without the guard block), the guarded book from guardFrom on, the order book from ordersFrom on
  const oldWeek = (date) => !orderWeek || !orderWeek(date);
  const plainWeek = (date) => (!guarded || date < guardFrom) && oldWeek(date);
  const guardWeekPositions = guarded
    ? guardedPositions(market, guarded, w && w.trading ? w.trading.stake : null, w && w.trading ? w.trading.feePerLeg : 0).filter((q) => !plainWeek(q.closed) && oldWeek(q.closed))
    : [];
  const weekOrderPositions = orderPositions(weekBook, 'week', orderWeek);
  const weekPositions = (w ? positionsOf(market, w, wf, 'week').filter((q) => plainWeek(q.closed)) : []).concat(guardWeekPositions, weekOrderPositions);
  const allPositions = dayPositions.concat(weekPositions);
  const closed = allPositions.slice().sort((a, b) => (a.closed < b.closed ? 1 : a.closed > b.closed ? -1 : 0));   // raw stripped at the end
  const byDate = {};
  const addPoints = (followed, keep = () => true) => (followed ? followed.points.filter((p) => p.live && keep(p.date)) : []).forEach((p) => {
    const e = byDate[p.date] || (byDate[p.date] = { date: p.date, pnl: 0, market: 0, hasMarket: false, positions: 0 });
    e.pnl += p.pnl;
    e.positions += p.positions || 0;
    if (p.market !== null && p.market !== undefined) { e.market += p.market; e.hasMarket = true; }
  });
  addPoints(f, oldDay);
  addPoints(wf, plainWeek);
  // the guarded weeks: each position's money and the week's market on the week's close (guardedPositions)
  const guardPts = guarded ? (guarded.points || []).filter((p) => Array.isArray(p) && p[5] === true && !plainWeek(p[0]) && oldWeek(p[0])) : [];
  // the order book's positions on the day they closed, its benchmark (every coin at the card) on the card's interval
  const dayBench = orderBench(dayBook, orderDay);
  const weekBench = orderBench(weekBook, orderWeek);
  dayOrderPositions.concat(weekOrderPositions).forEach((q) => {
    const e = byDate[q.closed] || (byDate[q.closed] = { date: q.closed, pnl: 0, market: 0, hasMarket: false, positions: 0 });
    e.pnl += q.raw; e.positions += 1;
  });
  dayBench.concat(weekBench).forEach((p) => {
    const e = byDate[p.interval] || (byDate[p.interval] = { date: p.interval, pnl: 0, market: 0, hasMarket: false, positions: 0 });
    e.market += p.money; e.hasMarket = true;
  });
  guardWeekPositions.forEach((q) => {
    const e = byDate[q.closed] || (byDate[q.closed] = { date: q.closed, pnl: 0, market: 0, hasMarket: false, positions: 0 });
    e.pnl += q.raw; e.positions += 1;
  });
  guardPts.filter((p) => num(p[6]) !== null).forEach((p) => {
    const e = byDate[p[0]] || (byDate[p[0]] = { date: p[0], pnl: 0, market: 0, hasMarket: false, positions: 0 });
    e.market += num(p[6]); e.hasMarket = true;
  });
  const days = Object.values(byDate).sort((a, b) => (a.date < b.date ? -1 : 1));
  let runPlan = 0, runMarket = 0;
  const daily = [], cumulative = [];
  days.forEach((p) => {
    runPlan += p.pnl; runMarket += p.market;
    daily.push({ date: p.date, pnl: round2(p.pnl), market: p.hasMarket ? round2(p.market) : null });
    cumulative.push({ date: p.date, plan: round2(runPlan), market: round2(runMarket) });
  });
  const perCoinMap = {};
  allPositions.forEach((q) => {
    const e = perCoinMap[q.symbol] || (perCoinMap[q.symbol] = { symbol: q.symbol, pnl: 0, positions: 0, won: 0 });
    e.pnl += q.raw; e.positions += 1; e.won += q.raw > 0 ? 1 : 0;
  });
  const perCoin = Object.values(perCoinMap).map((e) => ({ ...e, pnl: round2(e.pnl) })).sort((a, b) => b.pnl - a.pnl);
  const strip = (rows) => rows.map(({ raw, ...q }) => q);
  const won = allPositions.filter((q) => q.raw > 0).length;
  // a day counts towards won/lost only when something was bought: a day whose calls were all down or flat
  // took no position and is neither (owner, 17 Oct 2026)
  const played = days.filter((p) => p.positions > 0);
  const best = played.reduce((m, p) => (m === null || p.pnl > m.pnl ? p : m), null);
  const worst = played.reduce((m, p) => (m === null || p.pnl < m.pnl ? p : m), null);
  const sum = (a, b) => (a === null && b === null ? null : (a || 0) + (b || 0));
  // the live days counted: the old daily book's before ordersFrom, the order book's from it (its positions' days and
  // the benchmark's cards); without the order book the followed record's own totals, as before
  const oldDayPts = f ? f.points.filter((p) => p.live && oldDay(p.date)) : [];
  const orderDays = [...new Set(dayOrderPositions.map((q) => q.closed).concat(dayBench.map((p) => p.interval)))].sort();
  const dailyTotal = !dayBook ? (f && f.liveDays ? f.liveTotal : null)
    : (oldDayPts.length || orderDays.length ? oldDayPts.reduce((a, p) => a + (p.pnl || 0), 0) + dayOrderPositions.reduce((a, q) => a + q.raw, 0) : null);
  const dailyMarket = !dayBook ? (f && f.liveDays ? f.liveMarket : null)
    : (oldDayPts.length || orderDays.length ? oldDayPts.reduce((a, p) => a + (p.market || 0), 0) + dayBench.reduce((a, p) => a + p.money, 0) : null);
  const daySince = !dayBook ? (f && f.liveDays ? f.firstLive : null) : ([oldDayPts.length ? oldDayPts[0].date : null, orderDays[0] || null].filter(Boolean).sort()[0] || null);
  // the live weeks counted: the plain book's before guardFrom, the guarded book's from it, the order book's from ordersFrom
  const plainPts = wf ? wf.points.filter((p) => p.live && plainWeek(p.date)) : [];
  const orderWeeks = [...new Set(weekOrderPositions.map((q) => q.closed).concat(weekBench.map((p) => p.interval)))].sort();
  const weekCount = plainPts.length + guardPts.length + orderWeeks.length;
  const weeklyTotal = weekCount ? plainPts.reduce((a, p) => a + (p.pnl || 0), 0) + guardPts.reduce((a, p) => a + (num(p[3]) || 0), 0) + weekOrderPositions.reduce((a, q) => a + q.raw, 0) : null;
  const weeklyMarket = weekCount ? plainPts.reduce((a, p) => a + (p.market || 0), 0) + guardPts.reduce((a, p) => a + (num(p[6]) || 0), 0) + weekBench.reduce((a, p) => a + p.money, 0) : null;
  const weekSince = [plainPts.length ? plainPts[0].date : null, guardPts.length ? guardPts[0][0] : null, orderWeeks[0] || null].filter(Boolean).sort()[0] || null;
  const liveTotal = sum(dailyTotal, weeklyTotal);
  const liveMarket = sum(dailyMarket, weeklyMarket);
  const since = [daySince, weekSince].filter(Boolean).sort()[0] || null;
  const bestCoin = perCoin.length ? perCoin[0] : null, worstCoin = perCoin.length ? perCoin[perCoin.length - 1] : null;
  const orders = dayBook || weekBook ? ordersView({ dayBook, weekBook, dayOrders, weekOrders, week, view, w, plan, ordersFrom, now, unit: meta.unit,
    dayPositions, weekPositions, dayOrderPositions, weekOrderPositions, dayBench, weekBench, orderDay, orderWeek, nextDay }) : null;

  return {
    market, title: meta.title.replace(' predictor', ''), unit: meta.unit, noun: meta.noun,
    instruments: view.instruments.filter((i) => i.active).map((i) => i.symbol),
    generatedAt: view.generatedAt || null, now: now.toISOString(),
    status: {
      since, days: days.length, liveTotal: round2(liveTotal), liveMarket: round2(liveMarket),
      ahead: liveTotal !== null && liveMarket !== null ? round2(liveTotal - liveMarket) : null,
      // the two horizons the headline adds up; crypto's entry caveat
      dailyTotal: round2(dailyTotal), weeklyTotal: round2(weeklyTotal),
      entryNote: market === 'crypto'
        ? (orders ? `From ${dayText(ordersFrom)} the crypto positions are the cards' orders, bought at the price when each card appeared; before it they were booked from the 00:00 UTC close before their first day - before the tickets existed.`
          : 'Crypto positions, daily and weekly, are booked from the 00:00 UTC close before their first day - before the tickets exist; a real entry is later and at another price.') : null,
      positions: allPositions.length, won, lost: allPositions.length - won,
      winRate: allPositions.length ? won / allPositions.length : null,
      daysWon: played.filter((p) => p.pnl > 0).length, daysPlayed: played.length, daysIdle: days.length - played.length,
      bestDay: best ? { date: best.date, pnl: round2(best.pnl) } : null, worstDay: worst ? { date: worst.date, pnl: round2(worst.pnl) } : null,
      bestCoin: bestCoin ? { symbol: bestCoin.symbol, pnl: bestCoin.pnl, positions: bestCoin.positions } : null,
      worstCoin: worstCoin && worstCoin !== bestCoin ? { symbol: worstCoin.symbol, pnl: worstCoin.pnl, positions: worstCoin.positions } : null,
      openToBuy: open.filter((q) => q.status === 'buy now' || q.status === 'buy at open').length,
      openHeld: open.filter((q) => q.status === 'held' || q.status === 'hold (rolled)').length,
      // the order card's other rows: sell now, and coins with no order (counted, never positions)
      openToSell: open.filter((q) => q.status === 'sell now').length,
      // a roll row whose newcomer entry is a buy (31 Oct 2026): kept by a holder, bought by a reader who did not buy before
      openNewcomer: open.filter((q) => q.status === 'hold (rolled)' && q.order && q.order.newcomer && q.order.newcomer.skipReason === null).length,
      noOrder: open.filter((q) => q.status === 'no order' || q.status === 'not bought').length,
      // the order book's counts on the labelled lines (null before ordersFrom or without the order book)
      orders: orders ? { from: ordersFrom, day: orders.day ? orders.day.soFar : null, week: orders.week ? orders.week.soFar : null } : null,
      paperNote: `Paper money, ${t && t.stake !== null ? t.stake : 100} ${meta.unit} a position, ${t && t.feePerLeg !== null ? (t.feePerLeg * 100).toFixed(2) : '0.10'}% a leg.`,
      // the weekly positions' own record, kept for whoever needs it apart
      // guarded: weeks of the guarded book are among them; guardFrom: the first week the guard counts
      week: w || guarded ? { since: weekSince, days: weekCount, liveTotal: round2(weeklyTotal), liveMarket: round2(weeklyMarket),
                             ahead: weeklyTotal !== null && weeklyMarket !== null ? round2(weeklyTotal - weeklyMarket) : null,
                             positions: weekPositions.length, won: weekPositions.filter((q) => q.raw > 0).length,
                             guarded: guardPts.length > 0, guardFrom: guarded ? guardFrom : null } : null,
    },
    day: nextDay ? { date: nextDay, stateText: dayState, stateTitle: dayStatus(market, nextDay, now) || '', closeText: closeAt ? when(closeAt, now) : '—',
                     model: plan.model ? plan.model.name : null, modelCount: ticketModels(view.instruments) || view.models.length,
                     ruleName: plan.ruleName, byBook: plan.byBook, madeAt: view.madeAt || null,
                     // the positions a book needs before the plan follows it (the best exact rate until then); the paper
                     // stake, which a sized row's position is a multiple of
                     minTrades: markets.MIN_PLAN_TRADES, stake: t && t.stake !== null ? t.stake : null,
                     entryNote: market === 'crypto' ? 'The daily paper book buys at 00:00 UTC, before the ticket exists; a real entry is later and at another price.'
                                                    : 'The book buys at the open and sells at the close, which a reader can follow.' } : null,
    week,
    open, none,
    // the order cards' labelled lines and the late entries (null while ORDERS_SHOWN_FROM is null)
    orders,
    log: closed.slice(0, 10).map((q) => ({ date: q.closed, symbol: q.symbol, game: q.game, pnl: q.pnl })),
    daily, cumulative, perCoin, closed: strip(closed),
    notes: {
      entry: market === 'crypto' ? 'paper book enters at 00:00 UTC' : 'book buys at the open, sells at the close',
      entryText: market === 'crypto' ? 'The paper books buy at 00:00 UTC - a day position at the close before its day, a week position at the close before its first day - before the tickets exist; a real entry is later and at another price.' : 'The book buys at the open and sells at the close, which a reader can follow.',
      sell: 'sells are market orders at the time shown',
      sellText: 'Every sell is a market order at the time shown, whatever the price; the band only says when the call counts as right.',
      guard: !guarded ? 'daily call shown; the guard rule is not booked yet'
        : (!guardShown ? `guard: information this week, orders from ${dayText(guardFrom)}`
          : (week && !week.guarded ? 'guard: today\'s state comes with the next run' : 'guard: a week position is held while the daily call is up')),
      guardText: guarded
        ? 'The guard: a coin the weekly model calls up is bought on the first day of the week whose daily call for it is up - not on a day whose daily call is flat or down, nor on a day without a daily call - and held until a daily call turns flat or down; it is then sold at that day\'s close, bought again on the next up call, and sold at the week\'s close at the latest. Each week position is booked when its week closes, dated the week\'s close; until then a position the guard sold shows as Out with its move and money.'
          + ` The weekly results on this page count the guard from the week of ${dayText(guardFrom)}, the first full week the page showed it; earlier weeks count the plain week (held to the week's end), which is what the page said then. The guard's own book for the earlier weeks is on the Admin market page, the weeks before the week of ${dayText((w && w.trading && w.trading.guarded && w.trading.guarded.forwardFrom) || '2026-10-19')} booked there after the fact.`
          + (!guardShown ? ' This week the guard is shown as information: the sell stays at the week\'s end.'
            : (week && !week.guarded ? ' Today\'s guard state is not in the week record yet: the week rows show the daily call as information until the next run.' : ''))
        : 'The guard rule - sell a week position at the close of the first day whose daily call is flat or down, re-enter on the next up call - is shown as information next to each week position until its paper book exists; until then the sell of a week position stays at the week\'s end, as the record books it.',
      closeDay: closeAt ? `${market === 'crypto' ? 'A day position sells at the UTC close, ' : 'A day position sells at the New York close, '}${clock(closeAt)} Belgian time.` : '',
    },
  };
}

// The order cards' labelled lines (the plan-card rule, 13 Oct 2026): one model's calls - whose and why, its own record,
// not a reader's; following this card so far - the order positions the closed table lists, against buying every coin
// at the card's time, and the old record before ordersFrom; the sell. The week adds the late entries and their own
// record, never part of "following this card so far".
function ordersView(a) {
  const { dayBook, weekBook, dayOrders, weekOrders, week, view, w, plan, ordersFrom, now, unit } = a;
  const oldOf = (rows) => { const old = rows.filter((q) => q.rule !== 'orders'); return { positions: old.length, total: round2(old.reduce((s, q) => s + q.raw, 0)) }; };
  const modelLine = (name, v, models, minTrades, weekly) => {
    const m = v ? v.models.find((x) => x.name === name) : null;
    const key = 'pnlCash', trades = 'trades';
    const byBook = m && m[key] !== null && m[trades] !== null && m[trades] >= minTrades;
    return `${name || 'No model'}, one of the ${models} models on ${weekly ? 'the week ticket' : 'this ticket'}, `
      + (byBook ? `followed because its ${weekly ? 'weekly' : 'daily'} book is the best of those with ${minTrades}+ positions (its own record, not what a reader made: ${signed(m[key])} ${unit} over ${m[trades]} positions).`
        : `followed for its exact rate while no ${weekly ? 'weekly' : 'daily'} book has ${minTrades} positions (its own record, not what a reader made).`);
  };
  const out = { from: ordersFrom, day: null, week: null };
  if (dayBook) {
    const soFar = orderSoFar(a.dayOrderPositions, a.dayBench, 'day', ordersFrom, oldOf(a.dayPositions), dayBook.constants, unit, { newcomer: Boolean(dayBook.newcomer) });
    const newcomerRecord = newcomerRecordOf(dayBook, a.orderDay, unit);
    const n = dayOrders;
    const pub = n ? dateOf(n.publishedAt) : null;
    const fallback = n ? dateOf(n.deadlineFallback) : null;
    out.day = {
      soFar, variant: n ? n.variant || dayBook.primary : dayBook.primary, card: Boolean(n), interval: n ? n.interval : null, publishedAt: n ? n.publishedAt : null, estimated: n ? n.estimated : false,
      newcomerRecord: newcomerRecord ? { shown: newcomerRecord.shown, closed: newcomerRecord.closed, open: newcomerRecord.open, total: newcomerRecord.total, notPlaced: newcomerRecord.notPlaced } : undefined,
      lines: {
        model: n ? modelLine(n.model, view, ticketModels(view.instruments) || view.models.length, markets.MIN_PLAN_TRADES, false) : (plan.model ? modelLine(plan.model.name, view, ticketModels(view.instruments) || view.models.length, markets.MIN_PLAN_TRADES, false) : ''),
        soFar: soFar.text,
        today: n ? `${dayKey(now) === n.interval ? 'Today' : 'The card for'}, ${dayText(n.interval)} (UTC day; the card appeared ${pub ? `${clock(pub)} Brussels` : 'at a time not recorded'}${n.estimated ? ', estimated at 15 min after the ticket' : ''}):` : '',
        sell: `At the limit, or at market when tomorrow's card appears, whichever comes first${fallback ? ` (${when(fallback, now)} at the latest)` : ''}. Cancel the limit before you sell at market. If tomorrow's card calls the same coin up again, keep it and only replace the limit; the card will say so, and says when the new limit is below your entry price. A reader who did not buy before such a card gets its own buy line (a newcomer entry). ${EUR_NOTE}`,
        ...(newcomerRecord ? { newcomerRecord: newcomerRecord.text } : {}),
      },
    };
  }
  if (weekBook) {
    const soFar = orderSoFar(a.weekOrderPositions, a.weekBench, 'week', ordersFrom, oldOf(a.weekPositions), weekBook.constants, unit);
    const n = weekBook.now && weekOrders ? weekBook.now : null;
    const fallback = n ? dateOf(n.deadlineFallback) : null;
    const pub = n ? dateOf(n.publishedAt) : null;
    // the late ledger: every late showing from ordersFrom on, in the variant its week card showed
    const lateRows = [];
    Object.entries(weekBook.late || {}).forEach(([variant, book]) => book.orders.forEach((q) => {
      if (q.kind === 'late' && q.live && a.orderWeek(q.interval) && shownVariant(weekBook, q.interval) === variant && ['closed', 'open', 'pending'].includes(q.status)) lateRows.push(q);
    }));
    const lateClosed = lateRows.filter((q) => q.status === 'closed');
    const lateRecord = { shown: lateRows.length, closed: lateClosed.length, open: lateRows.length - lateClosed.length, total: round2(lateClosed.reduce((s, q) => s + (q.money || 0), 0)) };
    const endText = week && week.endDay ? dayText(week.endDay) : '';
    out.week = {
      soFar, variant: n ? n.variant || weekBook.primary : weekBook.primary, card: Boolean(n), interval: n ? n.interval : null, publishedAt: n ? n.publishedAt : null, estimated: n ? n.estimated : false,
      late: weekOrders ? weekOrders.late : [], lateRecord,
      lines: {
        model: w ? modelLine(n ? n.model : (week ? week.model : null), w, ticketModels(w.instruments) || w.models.length, markets.MIN_WEEK_TRADES, true) : '',
        soFar: soFar.text,
        today: n && n.firstDay ? `Week ${dayText(n.firstDay)} - ${endText}${pub ? ` (the week card appeared ${stampText(pub)} Brussels${n.estimated ? ', estimated' : ''}` : ' ('}`
          + `${week && week.closeText ? `; the week's close is ${when(closeMoment('crypto', n.interval), now)}` : ''}; the orders end when next Monday's card appears):` : '',
        sell: `At the limit; at market if a daily card turns the coin flat or down (cancel the limit first); at market when next Monday's card appears${fallback ? ` (${when(fallback, now)} at the latest)` : ''}. One round trip per coin a week: a coin sold is not bought back this week. ${EUR_NOTE}`,
        lateRecord: `Late entries so far (a separate record): ${plural(lateRecord.shown, 'shown', 'shown')}${lateRecord.open ? ` (${lateRecord.open} still open)` : ''}, ${lateRecord.closed ? `${signed(lateRecord.total)} ${unit} on the closed ones` : 'none closed yet'}, each counted from the price when that day's card appeared.`,
      },
    };
  }
  return out;
}

function ticketModels(instruments) {
  const names = new Set();
  (instruments || []).forEach((i) => (i.next || []).forEach((p) => names.add(p.model)));
  return names.size;
}

// --- live quotes ---------------------------------------------------------------
const QUOTE_TTL_MS = 60000;
const quoteCache = {};   // market -> {at, quotes, fetched}
const NASDAQ_HEADERS = { 'User-Agent': 'Mozilla/5.0', Accept: 'application/json, text/plain, */*', 'Accept-Language': 'en-US,en;q=0.9',
                         Origin: 'https://www.nasdaq.com', Referer: 'https://www.nasdaq.com/' };

async function fetchJson(url, headers) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  try {
    const res = await fetch(url, { headers, signal: controller.signal });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchQuotes(market, symbols) {
  const quotes = {};
  if (!symbols.length) return quotes;
  if (market === 'crypto') {
    const pairs = symbols.map((s) => `${s}USDT`);
    const data = await fetchJson(`https://api.binance.com/api/v3/ticker/price?symbols=${encodeURIComponent(JSON.stringify(pairs))}`);
    (Array.isArray(data) ? data : []).forEach((q) => {
      const sym = String(q.symbol || '').replace(/USDT$/, '');
      const v = num(q.price);
      if (symbols.includes(sym) && v !== null && v > 0) quotes[sym] = v;
    });
  } else if (market === 'shares') {
    await Promise.all(symbols.map(async (s) => {
      const data = await fetchJson(`https://api.nasdaq.com/api/quote/${encodeURIComponent(s)}/info?assetclass=stocks`, NASDAQ_HEADERS);
      const raw = data && data.data && data.data.primaryData ? data.data.primaryData.lastSalePrice : null;
      // an unknown or delisted symbol answers 200 with data null: no price is no quote (never 0)
      const v = typeof raw === 'string' && raw.trim() ? num(raw.replace(/[$,]/g, '')) : null;
      if (v !== null && v > 0) quotes[s] = v;
    }));
  }
  return quotes;
}

async function quotesFor(market, symbols) {
  const cached = quoteCache[market];
  if (cached && Date.now() - cached.fetched < QUOTE_TTL_MS) return { at: cached.at, quotes: cached.quotes };
  const quotes = await fetchQuotes(market, symbols);
  const at = Object.keys(quotes).length ? new Date().toISOString() : null;
  quoteCache[market] = { at, quotes, fetched: Date.now() };
  return { at, quotes };
}

// --- routes --------------------------------------------------------------------
function install(app, { dataDir }) {
  const json = (res, body, status = 200) => { res.set('Cache-Control', 'no-store'); res.status(status).json(body); };
  // open: open access (no login configured) - there is nobody to sign out, and the app says so. csrf: the session's
  // form token, which every JSON POST of the app sends back as _csrf (system.js).
  // A cross-origin page cannot read this answer, so it cannot learn the token.
  app.get('/api/me', (req, res) => {
    json(res, { user: req.user ? { name: req.user.name, admin: req.user.role === 'admin', open: req.user.open === true } : null,
      csrf: req.user && req.user.csrf ? String(req.user.csrf) : null });
  });
  app.get('/api/markets/:market', (req, res) => {
    const market = req.params.market;
    if (!MARKETS[market]) return json(res, { error: 'unknown market' }, 404);
    const record = loadMarket(dataDir, market);
    if (!record) return json(res, { error: 'no record yet' }, 404);
    json(res, marketView(market, record, loadMarket(dataDir, WEEK_GAMES[market]), new Date()));
  });
  app.get('/api/quotes/:market', async (req, res) => {
    const market = req.params.market;
    if (!MARKETS[market]) return json(res, { error: 'unknown market' }, 404);
    const record = loadMarket(dataDir, market);
    const symbols = record ? record.instruments.filter((i) => i.active !== false).map((i) => String(i.symbol)) : [];
    json(res, await quotesFor(market, symbols));
  });
}

// The React app (web/dist) under /app, with the SPA fallback; a missing build
// says how to make one instead of a 404.
function installApp(app) {
  const express = require('express');
  const dist = path.join(__dirname, 'web', 'dist');
  app.use('/app', express.static(dist, { index: false, maxAge: '1h' }));
  app.get(['/app', '/app/*'], (req, res) => {
    const index = path.join(dist, 'index.html');
    if (fs.existsSync(index)) { res.set('Cache-Control', 'no-store'); return res.sendFile(index); }
    // in the sign-in look (shell.js)
    res.status(503).send(shell.page('Not built', '<h1>The app is not built</h1><p class="sub">Run <code>npm run build:web</code> in the checkout and restart the server.</p>'));
  });
}

module.exports = { marketView, positionsOf, when, fetchQuotes, quotesFor, install, installApp, ticketModels,
  // the order book's pieces, for the Admin shadow (workbench.js ordersOf: what the user card would say)
  dayOrderRows, weekOrderRows, orderPositions, orderBench, orderSoFar, newcomerRecordOf, orderExitText, weekFirst, priceText };
