// api.js - the JSON API behind the React user interface (web/, served under /app;
// 15 Oct 2026, the UI redesign). Three routes, all behind the same session as
// the pages (auth.middleware runs first):
//
//   GET /api/me               who is signed in
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

const { MARKETS, WEEK_GAMES, loadMarket, describeMarket, nextGameDay, closeMoment, openMoment, weekEnd, dayStatus, dayStatusWeek,
        pickPlan, pickWeek } = markets;

const BRUSSELS = 'Europe/Brussels';
const DAY_MS = 86400000;

// --- clocks in Belgian time, short ------------------------------------------
const fmtClock = new Intl.DateTimeFormat('en-GB', { timeZone: BRUSSELS, hour: '2-digit', minute: '2-digit', hour12: false });
const fmtDay = new Intl.DateTimeFormat('en-CA', { timeZone: BRUSSELS, year: 'numeric', month: '2-digit', day: '2-digit' });
const fmtWeekday = new Intl.DateTimeFormat('en-GB', { timeZone: BRUSSELS, weekday: 'short' });
const clock = (date) => fmtClock.format(date);
const dayKey = (date) => fmtDay.format(date);

const fmtDayMonth = new Intl.DateTimeFormat('en-GB', { timeZone: BRUSSELS, day: 'numeric', month: 'short' });
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
const callWord = (p, k) => (p.bin !== null && p.bin < k / 2 - 1 ? 'down' : 'flat');
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

// --- the view ------------------------------------------------------------------
function marketView(market, record, weekRecord, now = new Date()) {
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
      let status, statusText;
      if (dayClosed) { status = 'too late'; statusText = `closed at ${clock(closeAt)}; the result comes with the morning run`; }
      else if (market === 'crypto' || sessionOpen) { status = 'buy now'; statusText = 'a market order now'; }
      else { status = 'buy at open'; statusText = `a market order placed before ${clock(openAt)}, filled at the open`; }
      open.push({
        symbol: inst.symbol, game: 'day', status, statusText,
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

  // this week: the followed week model's calls as held positions, guarded by today's daily call
  let week = null;
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
      week = { endDay, startText: when(wOpen, now), stateText: `week ${fmtDayMonth.format(wOpen)} → ${fmtDayMonth.format(wClose)}`,
               stateTitle: dayStatusWeek(now, wOpen, wClose) || '', closeText: when(wClose, now),
               model: pick.model.name, modelCount: ticketModels(w.instruments) || w.models.length, ruleName: pick.ruleName, byBook: pick.byBook };
      w.instruments.filter((i) => i.active).forEach((inst) => {
        const call = inst.next.find((p) => p.model === pick.model.name) || null;
        if (!call) return;
        if (!isUp(call, w.k)) { none.week.push({ symbol: inst.symbol, call: callWord(call, w.k) }); return; }
        if (wClosed) return;
        const b = base(inst);
        const daily = view.instruments.find((i) => i.symbol === inst.symbol);
        const since = daily ? daily.closes.filter((p) => p[0] > w.madeOn) : [];
        const lastClose = since.length ? since[since.length - 1][1] : null;
        const profitPctClose = lastClose !== null && b ? round2((lastClose / b - 1) * 100) : null;
        const todays = dailyCall[inst.symbol] || null;
        // Today's daily call for the coin, shown next to the week position. The guard rule the owner chose (sell at the
        // close of the first day whose daily call is flat or down, re-enter on the next up call) is information here
        // until MarketSettle books it: a sell the record does not apply would make the record a lie, so the Sell
        // column keeps the week's end and the help says so.
        let guard = null;
        if (started && todays && !dayClosed) {
          guard = isUp(todays, k)
            ? { state: 'hold', text: 'daily call up', booked: false, reason: `today's daily call for ${inst.symbol} is up (bin ${todays.bin}); the guard rule would hold` }
            : { state: 'sell', text: `daily call ${callWord(todays, k)}`, booked: false, reason: `today's daily call for ${inst.symbol} is ${callWord(todays, k)} (bin ${todays.bin}); the guard rule would sell at the close - it is not in the record yet, so the sell here stays at the week's end` };
        }
        const pctOf = (v) => (v === null || b === null || b <= 0 ? null : round2((v / b - 1) * 100));
        let status, statusText;
        if (!started) { status = market === 'crypto' ? 'buy now' : 'buy at open'; statusText = market === 'crypto' ? `a market order at the week's start, ${when(wOpen, now)}` : `a market order placed before ${clock(wOpen)}, filled at the week's first open`; }
        else if (entering) { status = market === 'crypto' || sessionOpen ? 'buy now' : 'buy at open'; statusText = market === 'crypto' ? `a market order now; the book entered at the week's start, ${when(wOpen, now)}` : (sessionOpen ? 'a market order now, at the current price; the book bought at the open' : `a market order placed before ${clock(wOpen)}, filled at the open`); }
        else { status = 'held'; statusText = `held since ${when(wOpen, now)}`; }
        open.push({
          symbol: inst.symbol, game: 'week', status, statusText,
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

  // the record: positions, days, totals
  const dayPositions = positionsOf(market, view, f, 'day');
  const weekPositions = w ? positionsOf(market, w, wf, 'week') : [];
  const closed = dayPositions.concat(weekPositions).sort((a, b) => (a.closed < b.closed ? 1 : a.closed > b.closed ? -1 : 0));   // raw stripped at the end
  const livePoints = f ? f.points.filter((p) => p.live) : [];
  let runPlan = 0, runMarket = 0;
  const daily = [], cumulative = [];
  livePoints.forEach((p) => {
    runPlan += p.pnl; runMarket += p.market || 0;
    daily.push({ date: p.date, pnl: round2(p.pnl), market: round2(p.market) });
    cumulative.push({ date: p.date, plan: round2(runPlan), market: round2(runMarket) });
  });
  const perCoinMap = {};
  dayPositions.forEach((q) => {
    const e = perCoinMap[q.symbol] || (perCoinMap[q.symbol] = { symbol: q.symbol, pnl: 0, positions: 0, won: 0 });
    e.pnl += q.raw; e.positions += 1; e.won += q.raw > 0 ? 1 : 0;
  });
  const perCoin = Object.values(perCoinMap).map((e) => ({ ...e, pnl: round2(e.pnl) })).sort((a, b) => b.pnl - a.pnl);
  const strip = (rows) => rows.map(({ raw, ...q }) => q);
  const won = dayPositions.filter((q) => q.raw > 0).length;
  const weekWon = weekPositions.filter((q) => q.raw > 0).length;
  const best = livePoints.reduce((m, p) => (m === null || p.pnl > m.pnl ? p : m), null);
  const worst = livePoints.reduce((m, p) => (m === null || p.pnl < m.pnl ? p : m), null);
  const liveTotal = f ? f.liveTotal : null, liveMarket = f ? f.liveMarket : null;

  return {
    market, title: meta.title.replace(' predictor', ''), unit: meta.unit, noun: meta.noun,
    instruments: view.instruments.filter((i) => i.active).map((i) => i.symbol),
    generatedAt: view.generatedAt || null, now: now.toISOString(),
    status: {
      since: f ? f.firstLive : null, days: f ? f.liveDays : 0, liveTotal: round2(liveTotal), liveMarket: round2(liveMarket),
      ahead: liveTotal !== null && liveMarket !== null ? round2(liveTotal - liveMarket) : null,
      positions: dayPositions.length, won, lost: dayPositions.length - won,
      winRate: dayPositions.length ? won / dayPositions.length : null,
      daysWon: livePoints.filter((p) => p.pnl > 0).length,
      bestDay: best ? { date: best.date, pnl: round2(best.pnl) } : null, worstDay: worst ? { date: worst.date, pnl: round2(worst.pnl) } : null,
      openToBuy: open.filter((q) => q.status === 'buy now' || q.status === 'buy at open').length,
      openHeld: open.filter((q) => q.status === 'held').length,
      paperNote: `Paper money, ${t && t.stake !== null ? t.stake : 100} ${meta.unit} a position, ${t && t.feePerLeg !== null ? (t.feePerLeg * 100).toFixed(2) : '0.10'}% a leg. The figures of the plan, the positions, the days and the charts are the day game's; the week game is its own line.`,
      // the week game's own record (the strip's figures above are the day game's)
      week: wf ? { since: wf.firstLive, days: wf.liveDays, liveTotal: round2(wf.liveTotal), liveMarket: round2(wf.liveMarket),
                   ahead: wf.liveTotal !== null && wf.liveMarket !== null ? round2(wf.liveTotal - wf.liveMarket) : null,
                   positions: weekPositions.length, won: weekWon } : null,
    },
    day: nextDay ? { date: nextDay, stateText: dayState, stateTitle: dayStatus(market, nextDay, now) || '', closeText: closeAt ? when(closeAt, now) : '—',
                     model: plan.model ? plan.model.name : null, modelCount: ticketModels(view.instruments) || view.models.length,
                     ruleName: plan.ruleName, byBook: plan.byBook, madeAt: view.madeAt || null,
                     entryNote: market === 'crypto' ? 'The daily paper book buys at 00:00 UTC, before the ticket exists; a real entry is later and at another price.'
                                                    : 'The book buys at the open and sells at the close, which a reader can follow.' } : null,
    week,
    open, none,
    log: closed.slice(0, 10).map((q) => ({ date: q.closed, symbol: q.symbol, game: q.game, pnl: q.pnl })),
    daily, cumulative, perCoin, closed: strip(closed),
    notes: {
      entry: market === 'crypto' ? 'paper book enters at 00:00 UTC' : 'book buys at the open, sells at the close',
      entryText: market === 'crypto' ? 'The daily paper book buys at 00:00 UTC, before the ticket exists; a real entry is later and at another price.' : 'The book buys at the open and sells at the close, which a reader can follow.',
      sell: 'sells are market orders at the time shown',
      sellText: 'Every sell is a market order at the time shown, whatever the price; the band only says when the call counts as right.',
      guard: 'daily call shown; the guard rule is not booked yet',
      guardText: 'The guard rule the owner chose - sell a week position at the close of the first day whose daily call is flat or down, re-enter on the next up call - is shown as information next to each week position until its paper book exists; until then the sell of a week position stays at the week\'s end, as the record books it.',
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
  app.get('/api/me', (req, res) => {
    json(res, { user: req.user ? { name: req.user.name, admin: req.user.role === 'admin' } : null });
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
function installApp(app, { header, footer }) {
  const express = require('express');
  const dist = path.join(__dirname, 'web', 'dist');
  app.use('/app', express.static(dist, { index: false, maxAge: '1h' }));
  app.get(['/app', '/app/*'], (req, res) => {
    const index = path.join(dist, 'index.html');
    if (fs.existsSync(index)) { res.set('Cache-Control', 'no-store'); return res.sendFile(index); }
    res.status(503).send(header('App', req.user) + '<h1>The app is not built</h1><p>Run <code>npm run build:web</code> in the checkout and restart the server.</p>' + footer());
  });
}

module.exports = { marketView, positionsOf, when, fetchQuotes, quotesFor, install, installApp, ticketModels };
