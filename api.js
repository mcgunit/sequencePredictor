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
        pickPlan, pickWeek, GUARD_SHOWN_FROM } = markets;

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

// --- the view ------------------------------------------------------------------
// `guardFrom`: the first week the guard counts (GUARD_SHOWN_FROM; the tests move it).
function marketView(market, record, weekRecord, now = new Date(), { guardFrom = GUARD_SHOWN_FROM } = {}) {
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
  if (nextDay && plan.model) {
    view.instruments.filter((i) => i.active).forEach((inst) => {
      const call = inst.next.find((p) => p.model === plan.model.name) || null;
      if (call) dailyCall[inst.symbol] = call;
      const ups = inst.next.filter((p) => isUp(p, k)).length;
      if (!call) return;
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

  // this week: the followed week model's calls as positions under the daily guard (MarketSettle.guarded_week's
  // `now`) from the week the guard counts (guardFrom); before it, or on a record without the guard block, the plain
  // week with today's daily call as information
  const guarded = weekRecord && weekRecord.trading && weekRecord.trading.guarded && typeof weekRecord.trading.guarded === 'object' ? weekRecord.trading.guarded : null;
  let week = null;
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
      w.instruments.filter((i) => i.active).forEach((inst) => {
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
  const dayPositions = positionsOf(market, view, f, 'day');
  // the weekly positions as the page told them: the plain week book for weeks closing before guardFrom (or every
  // week of a record without the guard block), the guarded book from guardFrom on
  const plainWeek = (date) => !guarded || date < guardFrom;
  const guardWeekPositions = guarded
    ? guardedPositions(market, guarded, w && w.trading ? w.trading.stake : null, w && w.trading ? w.trading.feePerLeg : 0).filter((q) => !plainWeek(q.closed))
    : [];
  const weekPositions = (w ? positionsOf(market, w, wf, 'week').filter((q) => plainWeek(q.closed)) : []).concat(guardWeekPositions);
  const allPositions = dayPositions.concat(weekPositions);
  const closed = allPositions.slice().sort((a, b) => (a.closed < b.closed ? 1 : a.closed > b.closed ? -1 : 0));   // raw stripped at the end
  const byDate = {};
  const addPoints = (followed, keep = () => true) => (followed ? followed.points.filter((p) => p.live && keep(p.date)) : []).forEach((p) => {
    const e = byDate[p.date] || (byDate[p.date] = { date: p.date, pnl: 0, market: 0, hasMarket: false, positions: 0 });
    e.pnl += p.pnl;
    e.positions += p.positions || 0;
    if (p.market !== null && p.market !== undefined) { e.market += p.market; e.hasMarket = true; }
  });
  addPoints(f);
  addPoints(wf, plainWeek);
  // the guarded weeks: each position's money and the week's market on the week's close (guardedPositions)
  const guardPts = guarded ? (guarded.points || []).filter((p) => Array.isArray(p) && p[5] === true && !plainWeek(p[0])) : [];
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
  const dailyTotal = f && f.liveDays ? f.liveTotal : null;
  // the live weeks counted: the plain book's before guardFrom, the guarded book's from it
  const plainPts = wf ? wf.points.filter((p) => p.live && plainWeek(p.date)) : [];
  const weekCount = plainPts.length + guardPts.length;
  const weeklyTotal = weekCount ? plainPts.reduce((a, p) => a + (p.pnl || 0), 0) + guardPts.reduce((a, p) => a + (num(p[3]) || 0), 0) : null;
  const weeklyMarket = weekCount ? plainPts.reduce((a, p) => a + (p.market || 0), 0) + guardPts.reduce((a, p) => a + (num(p[6]) || 0), 0) : null;
  const weekSince = [plainPts.length ? plainPts[0].date : null, guardPts.length ? guardPts[0][0] : null].filter(Boolean).sort()[0] || null;
  const liveTotal = sum(dailyTotal, weeklyTotal);
  const liveMarket = sum(f && f.liveDays ? f.liveMarket : null, weeklyMarket);
  const since = [f && f.liveDays ? f.firstLive : null, weekSince].filter(Boolean).sort()[0] || null;
  const bestCoin = perCoin.length ? perCoin[0] : null, worstCoin = perCoin.length ? perCoin[perCoin.length - 1] : null;

  return {
    market, title: meta.title.replace(' predictor', ''), unit: meta.unit, noun: meta.noun,
    instruments: view.instruments.filter((i) => i.active).map((i) => i.symbol),
    generatedAt: view.generatedAt || null, now: now.toISOString(),
    status: {
      since, days: days.length, liveTotal: round2(liveTotal), liveMarket: round2(liveMarket),
      ahead: liveTotal !== null && liveMarket !== null ? round2(liveTotal - liveMarket) : null,
      // the two horizons the headline adds up; crypto's entry caveat
      dailyTotal: round2(dailyTotal), weeklyTotal: round2(weeklyTotal),
      entryNote: market === 'crypto' ? 'Crypto positions, daily and weekly, are booked from the 00:00 UTC close before their first day - before the tickets exist; a real entry is later and at another price.' : null,
      positions: allPositions.length, won, lost: allPositions.length - won,
      winRate: allPositions.length ? won / allPositions.length : null,
      daysWon: played.filter((p) => p.pnl > 0).length, daysPlayed: played.length, daysIdle: days.length - played.length,
      bestDay: best ? { date: best.date, pnl: round2(best.pnl) } : null, worstDay: worst ? { date: worst.date, pnl: round2(worst.pnl) } : null,
      bestCoin: bestCoin ? { symbol: bestCoin.symbol, pnl: bestCoin.pnl, positions: bestCoin.positions } : null,
      worstCoin: worstCoin && worstCoin !== bestCoin ? { symbol: worstCoin.symbol, pnl: worstCoin.pnl, positions: worstCoin.positions } : null,
      openToBuy: open.filter((q) => q.status === 'buy now' || q.status === 'buy at open').length,
      openHeld: open.filter((q) => q.status === 'held').length,
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

module.exports = { marketView, positionsOf, when, fetchQuotes, quotesFor, install, installApp, ticketModels };
