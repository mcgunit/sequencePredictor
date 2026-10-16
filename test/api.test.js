'use strict';
// The JSON API's view of a market (api.js): the followed plan as positions,
// the record per day and per instrument, the clocks - on a hand-made record
// and, when the checkout has one, on production's own files (shape only).
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const api = require('../api');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- clocks -------------------------------------------------------------------
const now = new Date('2026-10-15T18:00:00Z');   // 20:00 Belgian time on Thursday 15 Oct
ok(api.when(new Date('2026-10-16T00:00:00Z'), now) === 'tonight 02:00', `the UTC close reads tonight 02:00: ${api.when(new Date('2026-10-16T00:00:00Z'), now)}`);
ok(api.when(new Date('2026-10-15T20:00:00Z'), now) === 'today 22:00', 'a close later today reads today hh:mm');
ok(api.when(new Date('2026-10-19T00:00:00Z'), now) === 'Mon 19 Oct 02:00' && api.when(new Date('2026-10-12T00:00:00Z'), now) === 'Mon 12 Oct', `a later day reads with its date and time, a past one with its date: ${api.when(new Date('2026-10-19T00:00:00Z'), now)} / ${api.when(new Date('2026-10-12T00:00:00Z'), now)}`);
ok(api.when(null, now) === '—' && api.when(new Date('x'), now) === '—', 'no date, no text');

// --- a hand-made crypto record and its week game -------------------------------
const pred = (model, bin, low, high, size) => ({ model, bin, low, high, direction: bin >= 5 ? 1 : -1, ...(size === undefined ? {} : { size }) });
const record = {
  market: 'crypto', generated_at: '2026-10-15T08:52:01+00:00', newest_game_day: '2026-10-14', k: 10,
  chance: { exact: 0.1, adjacent: 0.28, direction: 0.5 },
  instruments: [
    { symbol: 'BTC', active: true, last_close: 81682, closes: [['2026-10-12', 83000], ['2026-10-13', 82500], ['2026-10-14', 81682]] },
    { symbol: 'ETH', active: true, last_close: 2421.6, closes: [['2026-10-12', 2500], ['2026-10-13', 2450], ['2026-10-14', 2421.6]] },
    { symbol: 'OLD', active: false, last_close: 1, closes: [] },
  ],
  models: [
    { name: 'GARCH Model', days: 42, positions: 84, exact_rate: 0.12, trades: 30, pnl_cash_total: 23.17, hold_total: 1, open_total: null, open_trades: null, first_day: '2026-09-01', last_day: '2026-10-14' },
    { name: 'LightGBM Model', days: 42, positions: 84, exact_rate: 0.15, trades: 28, pnl_cash_total: 21.98, hold_total: 1, open_total: null, open_trades: null, first_day: '2026-09-01', last_day: '2026-10-14' },
    { name: 'Young Model', days: 3, positions: 6, exact_rate: 0.5, trades: 2, pnl_cash_total: 9, hold_total: 1, open_total: null, open_trades: null, first_day: '2026-10-12', last_day: '2026-10-14' },
  ],
  best_model: 'Young Model',
  next: { made_on: '2026-10-14', made_at: '2026-10-15T08:54:36+00:00', for: 'the next trading day', instruments: {
    BTC: { base: 81682, predictions: [pred('GARCH Model', 6, 82085, 82600), pred('LightGBM Model', 3, 80500, 81300), pred('Young Model', 7, 82600, 83500)] },
    ETH: { base: 2421.6, predictions: [pred('GARCH Model', 3, 2389.5, 2408.6), pred('LightGBM Model', 5, 2421, 2440), pred('Young Model', 4, 2410, 2421)] },
  } },
  trading: { stake: 100, fee_per_leg: 0.001, currency: 'USDT', rule: 'daily', dates: ['2026-10-12', '2026-10-13', '2026-10-14'],
    models: { 'GARCH Model': [['2026-10-12', -1, -1], ['2026-10-13', 3.8, 2.8], ['2026-10-14', -1.99, 0.81]] },
    benchmark: [['2026-10-12', -19.49, -19.49], ['2026-10-13', 6.67, -12.82], ['2026-10-14', -12.24, -25.06]],
    hold: { models: {}, benchmark: [] },
    followed: { rule: 'daily', min_trades: 10, first_live: '2026-10-13', total: 0.81, market_total: -25.06, live_total: 1.81, live_market: -5.57, live_days: 2, live_positions: 4,
      points: [
        ['2026-10-12', 'LightGBM Model', 'book', 2, -1, -1, false, -19.49, { BTC: -2.5, ETH: 1.5 }],
        ['2026-10-13', 'GARCH Model', 'book', 3, 3.8, 2.8, true, 6.67, { BTC: 1.41, ETH: 1.58, OLD: 0.81 }],
        ['2026-10-14', 'GARCH Model', 'book', 1, -1.99, 0.81, true, -12.24, { BTC: -1.99 }],
      ] },
  },
  days: [],
};
const weekRecord = {
  market: 'cryptoweek', generated_at: '2026-10-15T08:52:05+00:00', newest_game_day: '2026-10-11', k: 10,
  instruments: [
    { symbol: 'BTC', active: true, last_close: 83773, closes: [['2026-10-11', 83773]] },
    { symbol: 'ETH', active: true, last_close: 2536.9, closes: [['2026-10-11', 2536.9]] },
  ],
  models: [{ name: 'PoissonMonteCarlo Model', days: 31, positions: 155, exact_rate: 0.2, trades: 75, pnl_cash_total: 123, hold_total: 1, open_total: null, open_trades: null, first_day: '2026-03-15', last_day: '2026-10-11' }],
  best_model: 'PoissonMonteCarlo Model',
  next: { made_on: '2026-10-11', made_at: '2026-10-12T09:00:00+00:00', for: 'the next week', instruments: {
    BTC: { base: 83773, predictions: [pred('PoissonMonteCarlo Model', 1, 76548, 80180)] },
    ETH: { base: 2536.9, predictions: [pred('PoissonMonteCarlo Model', 6, 2595, 2667)] },
  } },
  trading: { stake: 100, fee_per_leg: 0.001, currency: 'USDT', rule: 'daily', dates: ['2026-10-04', '2026-10-11'],
    models: { 'PoissonMonteCarlo Model': [['2026-10-04', 1, 1], ['2026-10-11', -24.82, -23.82]] }, benchmark: [['2026-10-04', 4.9, 4.9], ['2026-10-11', -31.98, -27.08]],
    hold: { models: {}, benchmark: [] },
    followed: { rule: 'daily', min_trades: 5, first_live: '2026-10-11', total: -23.82, market_total: -27.08, live_total: -24.82, live_market: -31.98, live_days: 1, live_positions: 2,
      points: [['2026-10-04', 'GARCH Model', 'book', 2, 1, 1, false, 4.9, { BTC: 0.5, ETH: 0.5 }], ['2026-10-11', 'GARCH Model', 'book', 2, -24.82, -23.82, true, -31.98, { BTC: -10, ETH: -14.82 }]] },
  },
  days: [],
};

const v = api.marketView('crypto', record, weekRecord, now);
ok(v && v.market === 'crypto' && v.unit === 'USDT' && v.instruments.join() === 'BTC,ETH', 'the view names the market and its active instruments');

// the day positions: GARCH's up call becomes a buy-now row, its down call a chip; the week's up call a held row guarded by today's daily call
ok(v.day && v.day.model === 'GARCH Model' && v.day.byBook && v.day.modelCount === 3 && v.day.ruleName === 'daily round trip', `the day card follows the best book: ${JSON.stringify(v.day)}`);
const btc = v.open.find((q) => q.symbol === 'BTC' && q.game === 'day');
ok(btc && btc.status === 'buy now' && btc.sellText === 'tonight 02:00' && btc.openRate === 81682 && btc.band.bin === 6 && btc.band.lowPct === 0.49 && btc.modelsUp.up === 2 && btc.modelsUp.of === 3,
   `BTC is a buy-now day position sold tonight: ${JSON.stringify(btc)}`);
ok(v.none.day.length === 1 && v.none.day[0].symbol === 'ETH' && v.none.day[0].call === 'down', `ETH's down call is a no-position chip: ${JSON.stringify(v.none.day)}`);
const eth = v.open.find((q) => q.symbol === 'ETH' && q.game === 'week');
ok(eth && eth.status === 'held' && eth.since === 'since Mon 12 Oct' && eth.openRate === 2536.9 && eth.profitPctClose === -4.54 && eth.guard && eth.guard.state === 'sell' && eth.guard.booked === false && eth.sellText === 'Mon 19 Oct 02:00',
   `ETH's week position is held; today's daily call is down, shown as information, the sell stays at the week's end: ${JSON.stringify(eth)}`);
ok(v.none.week.length === 1 && v.none.week[0].symbol === 'BTC' && v.none.week[0].call === 'down', 'BTC has no week position');
ok(v.week && v.week.model === 'PoissonMonteCarlo Model' && v.week.endDay === '2026-10-18' && v.week.closeText === 'Mon 19 Oct 02:00' && v.week.stateText === 'week 12 Oct → 19 Oct' && v.day.stateText === 'day 15 Oct running, 18 h in', `the cards: ${JSON.stringify(v.week)} ${v.day.stateText}`);
ok(v.status.openToBuy === 1 && v.status.openHeld === 1, 'one to buy, one held');

// the record: positions from the live points' per-symbol money, movePct back from the money, newest first, the week's positions with their exit
ok(v.status.since === '2026-10-13' && v.status.days === 2 && v.status.liveTotal === 1.81 && v.status.liveMarket === -5.57 && v.status.ahead === 7.38 && v.status.week && v.status.week.days === 1 && v.status.week.liveTotal === -24.82 && v.status.week.positions === 2 && v.status.week.won === 0, `the status strip: ${JSON.stringify(v.status)}`);
ok(v.status.positions === 4 && v.status.won === 3 && v.status.lost === 1 && v.status.daysWon === 1 && v.status.bestDay.date === '2026-10-13' && v.status.worstDay.pnl === -1.99, `positions and days: ${JSON.stringify(v.status)}`);
ok(v.closed.length === 6 && v.closed[0].closed === '2026-10-14' && v.closed[0].symbol === 'BTC' && v.closed[0].pnl === -1.99 && v.closed[0].movePct === -1.79 && v.closed[0].exit === 'close 02:00 next day' && v.closed[0].raw === undefined,
   `the newest closed position: ${JSON.stringify(v.closed[0])}`);
ok(v.closed.filter((q) => q.game === 'week').length === 2 && v.closed.find((q) => q.game === 'week' && q.symbol === 'ETH').pnl === -14.82 && v.closed.find((q) => q.game === 'week').exit === 'week end',
   'the settled live week gives one closed position per instrument');
ok(!v.closed.some((q) => q.closed === '2026-10-12'), 'back-built days give no positions');
ok(v.perCoin.length === 3 && v.perCoin[0].symbol === 'ETH' && v.perCoin[0].pnl === 1.58 && v.perCoin.find((c) => c.symbol === 'BTC').positions === 2, `per coin: ${JSON.stringify(v.perCoin)}`);
ok(v.daily.length === 2 && v.cumulative[1].plan === 1.81 && v.cumulative[1].market === -5.57, `daily and cumulative: ${JSON.stringify(v.cumulative)}`);
ok(v.log.length === 6 && v.log[0].symbol === 'BTC' && v.log[0].game === 'day', 'the log is the newest positions');
ok(v.notes.guardText.includes('flat or down') && v.notes.guard.includes('not booked') && v.day.entryNote.includes('00:00 UTC'), 'the notes carry the rules');

// after the close the day row says so; a record without a followed block still renders
const late = api.marketView('crypto', record, weekRecord, new Date('2026-10-16T01:00:00Z'));
ok(late.open.find((q) => q.game === 'day').status === 'too late' && late.open.find((q) => q.game === 'week').sellText === 'Mon 19 Oct 02:00' && late.open.find((q) => q.game === 'week').guard === null, 'after the close: too late, and the daily call waits for the next day');
// the week's first day: a buy, not a held position (the ticket is up hours after the week's start)
const monday = api.marketView('crypto', { ...record, next: { ...record.next, made_on: '2026-10-11' } }, weekRecord, new Date('2026-10-12T09:30:00Z'));
const ethMon = monday.open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethMon && ethMon.status === 'buy now' && ethMon.statusText.includes('market order now') && ethMon.sellText === 'Mon 19 Oct 02:00' && monday.status.openToBuy >= 1,
   `Monday morning the week position is a buy now: ${JSON.stringify(ethMon)}`);
const sunday = api.marketView('crypto', record, weekRecord, new Date('2026-10-11T12:00:00Z'));
ok(sunday.open.find((q) => q.game === 'week' && q.symbol === 'ETH').status === 'buy now' && sunday.open.find((q) => q.game === 'week').statusText.includes("week's start"), 'before the week starts: a market order at the start');
const bare = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: undefined } }, null, now);
ok(bare.status.days === 0 && bare.status.liveTotal === null && bare.closed.length === 0 && bare.week === null && bare.open.length === 1, 'no followed record, no week: the open positions still render');
ok(api.marketView('nope', record, null, now) === null && api.marketView('crypto', null, null, now) === null, 'an unknown market or no record is null');
// an older point without the per-symbol money gives no positions but keeps the day
const old = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: { ...record.trading.followed, points: record.trading.followed.points.map((p) => p.slice(0, 8)) } } }, null, now);
ok(old.status.days === 2 && old.status.positions === 0 && old.daily.length === 2, 'older points without per-symbol money: days yes, positions none');

const exact = (ret) => 100 * Math.exp(ret) * 0.999 - 100 * 1.001;   // MarketSettle.cash_pnl for a 100 stake at 0.10% a leg
const moved = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: { ...record.trading.followed, points: [['2026-10-14', 'GARCH Model', 'book', 2, 0, 0, true, 0, { BTC: exact(-0.05), ETH: exact(0.1) }]] } } }, null, now);
ok(moved.closed.find((q) => q.symbol === 'BTC').movePct === -4.88 && moved.closed.find((q) => q.symbol === 'ETH').movePct === 10.52, `the move reads back exactly: ${JSON.stringify(moved.closed.map((q) => q.movePct))}`);
const sizedView = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: { ...record.trading.followed, points: [['2026-10-14', 'RL Position Model', 'book', 1, 2, 2, true, 0, { BTC: 2 }]] } } }, null, now);
ok(sizedView.closed[0].movePct === null && sizedView.closed[0].pnl === 2, 'the sized row has no move, only money');

// --- shares: the open-to-close rule and its clocks --------------------------------
const sharesRecord = { ...record, market: 'shares', instruments: record.instruments.map((i) => ({ ...i, symbol: i.symbol === 'BTC' ? 'NVDA' : (i.symbol === 'ETH' ? 'AAPL' : i.symbol) })),
  next: { ...record.next, made_on: '2026-10-14', instruments: { NVDA: record.next.instruments.BTC, AAPL: record.next.instruments.ETH } },
  models: record.models.map((m) => ({ ...m, open_total: m.pnl_cash_total, open_trades: m.trades })),
  trading: { ...record.trading, currency: 'USD', open: { models: record.trading.models, benchmark: record.trading.benchmark }, followed: { ...record.trading.followed, rule: 'open' } } };
const before = api.marketView('shares', sharesRecord, null, new Date('2026-10-15T09:00:00Z'));   // 11:00 Belgian, before New York opens
const nvda = before.open.find((q) => q.symbol === 'NVDA');
ok(before.day.ruleName === 'open to close' && nvda.status === 'buy at open' && nvda.statusText.includes('15:30') && nvda.sellText === 'today 22:00' && before.notes.closeDay.includes('22:00'), `shares before the open: ${JSON.stringify(nvda)}`);
const during = api.marketView('shares', sharesRecord, null, new Date('2026-10-15T15:00:00Z'));
ok(during.open.find((q) => q.symbol === 'NVDA').status === 'buy now' && during.closed[0].exit === 'close 22:00', 'shares during the session: buy now, closed positions exit at 22:00');

// --- production's own files, shape only (read-only) -------------------------------
const prodDir = '/root/sequencePredictor/data/markets';
if (fs.existsSync(path.join(prodDir, 'crypto.json'))) {
  ['crypto', 'shares'].forEach((market) => {
    const rec = JSON.parse(fs.readFileSync(path.join(prodDir, `${market}.json`)));
    const wk = fs.existsSync(path.join(prodDir, `${market}week.json`)) ? JSON.parse(fs.readFileSync(path.join(prodDir, `${market}week.json`))) : null;
    const pv = api.marketView(market, rec, wk, new Date());
    const text = JSON.stringify(pv);
    ok(pv && pv.status && Array.isArray(pv.open) && Array.isArray(pv.closed) && !/undefined|NaN/.test(text) && pv.status.days >= 0,
       `${market}: production's record renders (${pv.open.length} open, ${pv.closed.length} closed, ${pv.status.days} days)`);
  });
}

console.log(`api.js: ${checks} checks passed`);
