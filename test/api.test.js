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
ok(btc && btc.status === 'buy now' && btc.buyText === 'now' && btc.sellText === 'tonight 02:00' && btc.openRate === 81682 && btc.band.bin === 6 && btc.band.lowPct === 0.49 && btc.modelsUp.up === 2 && btc.modelsUp.of === 3,
   `BTC is a buy-now day position sold tonight: ${JSON.stringify(btc)}`);
ok(v.none.day.length === 1 && v.none.day[0].symbol === 'ETH' && v.none.day[0].call === 'down', `ETH's down call is a no-position chip: ${JSON.stringify(v.none.day)}`);
const eth = v.open.find((q) => q.symbol === 'ETH' && q.game === 'week');
ok(eth && eth.status === 'held' && eth.buyText === 'Mon 12 Oct' && eth.since === 'since Mon 12 Oct' && eth.openRate === 2536.9 && eth.profitPctClose === -4.54 && eth.guard && eth.guard.state === 'sell' && eth.guard.booked === false && eth.sellText === 'Mon 19 Oct 02:00',
   `ETH's week position is held; today's daily call is down, shown as information, the sell stays at the week's end: ${JSON.stringify(eth)}`);
ok(v.none.week.length === 1 && v.none.week[0].symbol === 'BTC' && v.none.week[0].call === 'down', 'BTC has no week position');
ok(v.week && v.week.model === 'PoissonMonteCarlo Model' && v.week.endDay === '2026-10-18' && v.week.closeText === 'Mon 19 Oct 02:00' && v.week.stateText === 'week 12 Oct → 18 Oct' && v.day.stateText === 'day 15 Oct running, 18 h in', `the cards: ${JSON.stringify(v.week)} ${v.day.stateText}`);
ok(v.status.openToBuy === 1 && v.status.openHeld === 1, 'one to buy, one held');

// the record: positions from the live points' per-symbol money, movePct back from the money, newest first, the week's positions with their exit
// the headline is what following the whole table made: the daily positions (1.81 over 13-14 Oct) and the weekly ones
// (-24.82, the week ending 11 Oct) together, against the market under the same rules (-5.57 and -31.98)
ok(v.status.since === '2026-10-11' && v.status.days === 3 && v.status.liveTotal === -23.01 && v.status.liveMarket === -37.55 && v.status.ahead === 14.54
   && v.status.dailyTotal === 1.81 && v.status.weeklyTotal === -24.82 && v.status.week && v.status.week.days === 1 && v.status.week.positions === 2 && v.status.week.won === 0,
   `the status strip: ${JSON.stringify(v.status)}`);
ok(v.status.positions === 6 && v.status.won === 3 && v.status.lost === 3 && v.status.daysWon === 1 && v.status.daysPlayed === 3 && v.status.daysIdle === 0 && v.status.bestDay.date === '2026-10-13' && v.status.worstDay.pnl === -24.82 && v.status.worstDay.date === '2026-10-11'
   && v.status.bestCoin.symbol === 'OLD' && v.status.worstCoin.symbol === 'ETH' && v.status.worstCoin.pnl === -13.24, `positions, days and coins: ${JSON.stringify(v.status)}`);
ok(v.closed.length === 6 && v.closed[0].closed === '2026-10-14' && v.closed[0].symbol === 'BTC' && v.closed[0].pnl === -1.99 && v.closed[0].movePct === -1.79 && v.closed[0].exit === 'close 02:00 next day' && v.closed[0].raw === undefined,
   `the newest closed position: ${JSON.stringify(v.closed[0])}`);
ok(v.closed.filter((q) => q.game === 'week').length === 2 && v.closed.find((q) => q.game === 'week' && q.symbol === 'ETH').pnl === -14.82 && v.closed.find((q) => q.game === 'week').exit === 'week end',
   'the settled live week gives one closed position per instrument');
ok(!v.closed.some((q) => q.closed === '2026-10-12'), 'back-built days give no positions');
ok(v.perCoin.length === 3 && v.perCoin[0].symbol === 'OLD' && v.perCoin.find((c) => c.symbol === 'ETH').pnl === -13.24 && v.perCoin.find((c) => c.symbol === 'BTC').positions === 3, `per coin, daily and weekly positions together: ${JSON.stringify(v.perCoin)}`);
ok(v.daily.length === 3 && v.daily[0].date === '2026-10-11' && v.daily[0].pnl === -24.82 && v.cumulative[2].plan === -23.01 && v.cumulative[2].market === -37.55, `daily and cumulative, the week on the day it settled: ${JSON.stringify(v.cumulative)}`);
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
ok(old.status.days === 2 && old.status.positions === 0 && old.daily.length === 2 && old.status.liveTotal === 1.81, 'older points without per-symbol money: days yes, positions none');
// a day on which nothing was bought is neither won nor lost
const idle = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: { ...record.trading.followed, live_days: 3,
  points: [...record.trading.followed.points, ['2026-10-15', 'GARCH Model', 'book', 0, 0, 0.81, true, 1.2, {}]] } } }, null, now);
ok(idle.status.days === 3 && idle.status.daysPlayed === 2 && idle.status.daysIdle === 1 && idle.status.daysWon === 1, `a day without a position: ${JSON.stringify(idle.status)}`);

const exact = (ret) => 100 * Math.exp(ret) * 0.999 - 100 * 1.001;   // MarketSettle.cash_pnl for a 100 stake at 0.10% a leg
const moved = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: { ...record.trading.followed, points: [['2026-10-14', 'GARCH Model', 'book', 2, 0, 0, true, 0, { BTC: exact(-0.05), ETH: exact(0.1) }]] } } }, null, now);
ok(moved.closed.find((q) => q.symbol === 'BTC').movePct === -4.88 && moved.closed.find((q) => q.symbol === 'ETH').movePct === 10.52, `the move reads back exactly: ${JSON.stringify(moved.closed.map((q) => q.movePct))}`);
const sizedView = api.marketView('crypto', { ...record, trading: { ...record.trading, followed: { ...record.trading.followed, points: [['2026-10-14', 'RL Position Model', 'book', 1, 2, 2, true, 0, { BTC: 2 }]] } } }, null, now);
ok(sizedView.closed[0].movePct === null && sizedView.closed[0].pnl === 2, 'the sized row has no move, only money');

// --- the guarded week (MarketSettle.guarded_week): the week rows and the weekly record follow the guard ----------
// these fixtures are weeks of 5-18 Oct 2026, before the guard counts on the page (GUARD_SHOWN_FROM 26 Oct): move the switch
const G = { guardFrom: '2026-10-05' };
const guardedWeek = { ...weekRecord, trading: { ...weekRecord.trading, guarded: {
  rule: 'guard', first_live: '2026-10-11', live_total: -3.5, live_market: -31.98, live_weeks: 1, live_positions: 2, total: -3.5,
  points: [['2026-10-04', 'GARCH Model', 1, 1, 1, false, 4.9, { BTC: 1 }, [['BTC', '2026-09-28', '2026-10-04', 1, 'week end']], true],
           ['2026-10-11', 'GARCH Model', 2, -3.5, -2.5, true, -31.98, { BTC: -1.5, ETH: -2 }, [['BTC', '2026-10-05', '2026-10-07', -1.5, 'guard'], ['ETH', '2026-10-05', '2026-10-11', -2, 'week end']], true]],
  now: { running_day: '2026-10-15', first_day: '2026-10-12', last_day: false, week_model: 'PoissonMonteCarlo Model', day_model: 'GARCH Model', settled_days: ['2026-10-12', '2026-10-13', '2026-10-14'],
    symbols: { ETH: { action: 'sell', reason: 'guard', entry: '2026-10-13', entry_price: 2500, call: 'down', sold_on: null, realized: null, trades: [] } } } } } };
const gv = api.marketView('crypto', record, guardedWeek, now, G);
const ethG = gv.open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethG && ethG.status === 'held' && ethG.buyText === 'Tue 13 Oct' && ethG.openRate === 2500 && ethG.sellText === 'tonight 02:00' && ethG.guard.booked === true && ethG.guard.state === 'sell'
   && ethG.profitPctClose === -3.14 && !ethG.sellNote && ethG.guard.reason === "today's daily call for ETH is down, so the position is sold at the close (tonight 02:00)"
   && gv.week.guarded === true && gv.notes.guard.includes('held while the daily call is up'),
   `under the guard a held week position whose daily call is down is sold tonight, measured from its entry at the last close: ${JSON.stringify(ethG)}`);
const btcClosed = gv.closed.find((q) => q.game === 'week' && q.symbol === 'BTC');
ok(gv.status.weeklyTotal === -3.5 && gv.status.week.guarded === true && gv.closed.filter((q) => q.game === 'week').length === 2
   && btcClosed.exit === 'guard: Wed 7 Oct, close 02:00 next day · bought Mon 5 Oct' && btcClosed.closed === '2026-10-11' && btcClosed.sold === '2026-10-07'
   && gv.closed.find((q) => q.game === 'week' && q.symbol === 'ETH').exit === 'week end · bought Mon 5 Oct',
   `the weekly record is the guarded book's, each position dated its week's close with the days it was held: ${JSON.stringify(gv.closed.filter((q) => q.game === 'week'))}`);
const gDaily = Object.fromEntries(gv.daily.map((d) => [d.date, d]));
// the page counts what it said (owner, 20 Oct 2026): before GUARD_SHOWN_FROM (26 Oct) the plain week, the guard as information
const pv = api.marketView('crypto', record, guardedWeek, now);
const ethPlain = pv.open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(pv.week.guarded === false && ethPlain.status === 'held' && ethPlain.sellText === 'Mon 19 Oct 02:00' && ethPlain.guard.booked === false
   && ethPlain.guard.reason.includes('counts the guard from the week of Mon 26 Oct') && pv.notes.guard.includes('information this week')
   && pv.status.weeklyTotal === -24.82 && pv.status.week.guarded === false && pv.status.week.guardFrom === '2026-10-26'
   && pv.closed.filter((q) => q.game === 'week').map((q) => `${q.symbol} ${q.pnl} ${q.exit}`).join() === 'BTC -10 week end,ETH -14.82 week end',
   `before the guard is on the page the week rows and the record are the plain week: ${JSON.stringify(pv.status.week)} / ${JSON.stringify(ethPlain)}`);
// the switch-over week: its first session is guardFrom itself (26 Oct for crypto and shares) - the rows follow the guard,
// and the week closing before it stays the plain week
const sv = api.marketView('crypto', record, guardedWeek, now, { guardFrom: '2026-10-12' });
ok(sv.week.guarded === true && sv.week.guardOrders === true && pv.week.guardOrders === false && sv.open.find((q) => q.game === 'week' && q.symbol === 'ETH').sellText === 'tonight 02:00'
   && sv.status.weeklyTotal === -24.82 && sv.status.week.guarded === false && !sv.notes.guardText.includes('information'),
   `the week starting on guardFrom follows the guard; the week closing before it is the plain week: ${JSON.stringify(sv.status.week)}`);
ok(pv.notes.guardText.includes('the weeks before the week of Mon 19 Oct booked there after the fact'), `the after-the-fact caveat names its weeks: ${pv.notes.guardText}`);
// a record with both: plain weeks before the switch, guarded weeks from it, added up in one weekly figure
const bothWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, followed: { ...guardedWeek.trading.followed,
  points: guardedWeek.trading.followed.points.map((q) => (q[0] === '2026-10-04' ? [...q.slice(0, 6), true, ...q.slice(7)] : q)) } } };
const bv = api.marketView('crypto', record, bothWeek, now, { guardFrom: '2026-10-08' });
ok(bv.status.weeklyTotal === -2.5 && bv.status.week.liveMarket === -27.08 && bv.status.week.days === 2 && bv.status.week.since === '2026-10-04' && bv.status.week.guarded === true
   && bv.closed.filter((q) => q.game === 'week').map((q) => `${q.closed} ${q.symbol} ${q.pnl}`).sort().join() === '2026-10-04 BTC 0.5,2026-10-04 ETH 0.5,2026-10-11 BTC -1.5,2026-10-11 ETH -2',
   `the plain week before the switch and the guard from it: ${JSON.stringify(bv.status.week)}`);
ok(gDaily['2026-10-07'] === undefined && gDaily['2026-10-11'].pnl === -3.5 && gDaily['2026-10-11'].market === -31.98,
   `a guarded week's money and its market land together on the week's close, never on a mid-week day: ${JSON.stringify(gv.daily)}`);
const outWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now,
  symbols: { ETH: { action: 'out', reason: 'guard', entry: null, entry_price: null, call: 'flat', sold_on: '2026-10-13', realized: -1.7, trades: [['2026-10-12', '2026-10-13', 'guard', -1.7, -0.015, 2536.9, 2499.1]] } } } } } };
const ethOut = api.marketView('crypto', record, outWeek, now, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethOut && ethOut.status === 'out' && ethOut.sellText === '—' && ethOut.since === 'sold Tue 13 Oct · \u22121.5% (\u22121.70 USDT)' && ethOut.profitPctClose === null
   && ethOut.openRate === null && ethOut.openRateNote === 'no position held' && ethOut.guard.state === 'out' && ethOut.guard.reason === "today's daily call for ETH is flat, so it stays out"
   && ethOut.statusText.includes('back in on the next up daily call'),
   `a position the guard sold earlier this week shows as out, its sale under Buy and no entry: ${JSON.stringify(ethOut)}`);
const twiceWeek = { ...outWeek, trading: { ...outWeek.trading, guarded: { ...outWeek.trading.guarded, now: { ...outWeek.trading.guarded.now,
  symbols: { ETH: { ...outWeek.trading.guarded.now.symbols.ETH, realized: -2.4, trades: [['2026-10-12', '2026-10-12', 'guard', -1.7, -0.015, 2536.9, 2499.1], ['2026-10-13', '2026-10-14', 'guard', -0.7, -0.005, 2499.1, 2486.6]] } } } } } };
ok(api.marketView('crypto', record, twiceWeek, now, G).open.find((q) => q.game === 'week').since === 'sold 2× this week · \u22122.40 USDT', 'two sales this week read as one line with their money');
// a coin bought back after a guard sale keeps the sale in view until the week closes
const backWeek = { ...outWeek, trading: { ...outWeek.trading, guarded: { ...outWeek.trading.guarded, now: { ...outWeek.trading.guarded.now,
  symbols: { ETH: { ...outWeek.trading.guarded.now.symbols.ETH, action: 'hold', reason: null, entry: '2026-10-14', entry_price: 2450, call: 'up', sold_on: null } } } } } };
const ethBack = api.marketView('crypto', record, backWeek, now, G).open.find((q) => q.game === 'week');
ok(ethBack.status === 'held' && ethBack.buyText === 'Wed 14 Oct' && ethBack.earlier === 'earlier: sold Tue 13 Oct · \u22121.5% (\u22121.70 USDT)', `a re-bought coin shows its earlier sale: ${JSON.stringify(ethBack)}`);
// a sale whose money is not known is never shown as 0.00 (guarded_now sends realized null)
const unknownWeek = { ...outWeek, trading: { ...outWeek.trading, guarded: { ...outWeek.trading.guarded, now: { ...outWeek.trading.guarded.now,
  symbols: { ETH: { ...outWeek.trading.guarded.now.symbols.ETH, realized: null, trades: [['2026-10-12', '2026-10-13', 'guard', null, null, 2536.9, null]] } } } } } };
ok(api.marketView('crypto', record, unknownWeek, now, G).open.find((q) => q.game === 'week').since === 'sold Tue 13 Oct', 'an unknown sale reads without a figure');
const buyWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now,
  symbols: { ETH: { action: 'buy', reason: 're-enter', entry: '2026-10-15', entry_price: 2421.6, call: 'up', sold_on: null, realized: -1.7, trades: [] } } } } } };
const ethBuy = api.marketView('crypto', record, buyWeek, now, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethBuy && ethBuy.status === 'buy now' && ethBuy.buyText === 'now' && ethBuy.sellText === 'Mon 19 Oct 02:00' && ethBuy.sellNote === 'at the latest · the guard can sell sooner'
   && ethBuy.statusText.includes('back in') && ethBuy.guard.state === 'buy' && ethBuy.guard.reason === "today's daily call for ETH is up, so the position is bought",
   `a re-entry on an up call is a buy now, the week's close the latest sell: ${JSON.stringify(ethBuy)}`);
// after the running day's close (07:00 Belgian on Friday): a buy is held, a guard sale is sold, until the morning run
const overnight = new Date('2026-10-16T05:00:00Z');
const ethHeld = api.marketView('crypto', record, buyWeek, overnight, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethHeld && ethHeld.status === 'held' && ethHeld.buyText === 'Thu 15 Oct' && ethHeld.since === 'since Thu 15 Oct' && ethHeld.sellText === 'Mon 19 Oct 02:00'
   && ethHeld.guard.state === 'hold' && ethHeld.guard.text === 'next call after the morning run' && ethHeld.statusText.includes('still held'),
   `a position bought on the running day is still held after its close: ${JSON.stringify(ethHeld)}`);
const ethSold = api.marketView('crypto', record, guardedWeek, overnight, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethSold && ethSold.status === 'sold' && ethSold.buyText === 'Tue 13 Oct' && ethSold.sellText === '—' && ethSold.statusText.includes('money when the week closes'),
   `a guard sale after the close reads as sold, not too late: ${JSON.stringify(ethSold)}`);
// the week's last day: sold at the week's close whatever the daily call
const sundayRecord = { ...record, next: { ...record.next, made_on: '2026-10-17' } };
const endWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now, running_day: '2026-10-18', last_day: true,
  symbols: { ETH: { action: 'sell', reason: 'week end', entry: '2026-10-13', entry_price: 2500, call: 'up', sold_on: null, realized: null, trades: [] } } } } } };
const ethEnd = api.marketView('crypto', sundayRecord, endWeek, new Date('2026-10-18T10:00:00Z'), G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethEnd && ethEnd.status === 'held' && ethEnd.sellText === 'tonight 02:00' && !ethEnd.sellNote && ethEnd.guard.state === 'weekend' && ethEnd.guard.text === 'daily call up',
   `the week's last day sells at the week's close, not by the guard: ${JSON.stringify(ethEnd)}`);
const endOut = { ...endWeek, trading: { ...endWeek.trading, guarded: { ...endWeek.trading.guarded, now: { ...endWeek.trading.guarded.now,
  symbols: { ETH: { action: 'out', reason: 'call', entry: null, entry_price: null, call: 'flat', sold_on: null, realized: null, trades: [] } } } } } };
const ethEndOut = api.marketView('crypto', sundayRecord, endOut, new Date('2026-10-18T10:00:00Z'), G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethEndOut.since === 'not bought this week' && ethEndOut.statusText.includes('the week ends at the close') && !ethEndOut.statusText.includes('first one'),
   `on the last day an out coin promises no re-entry: ${JSON.stringify(ethEndOut)}`);
// the day before the last: the guard's next chance to sell is the week's own close
const saturday = { ...record, next: { ...record.next, made_on: '2026-10-16' } };
const satWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now, running_day: '2026-10-17',
  symbols: { ETH: { action: 'hold', reason: null, entry: '2026-10-13', entry_price: 2500, call: 'up', sold_on: null, realized: null, trades: [] } } } } } };
const ethSat = api.marketView('crypto', saturday, satWeek, new Date('2026-10-17T10:00:00Z'), G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethSat.sellText === 'Mon 19 Oct 02:00' && ethSat.sellNote === 'the week\'s close', `on the day before the last the guard cannot sell sooner: ${JSON.stringify(ethSat)}`);
const staleWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now, first_day: '2026-10-05' } } } };
ok(api.marketView('crypto', record, staleWeek, now, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH').guard.booked === false, 'a guard state for another week is not used: the plain week with the call as information');
// a week record a day behind the daily one: its guard state is yesterday's, so it is not applied
const lagWeek = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now, running_day: '2026-10-14' } } } };
const lagView = api.marketView('crypto', record, lagWeek, now, G);
const ethLag = lagView.open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(lagView.week.guarded === false && ethLag.status === 'sold' && ethLag.buyText === 'Tue 13 Oct' && ethLag.statusText.includes('a run behind')
   && ethLag.guard.booked === false && ethLag.guard.reason.includes('not in the week record yet') && !ethLag.guard.reason.includes('stays at the week')
   && lagView.notes.guard.includes('next run') && lagView.status.week.guarded === true && lagView.status.openHeld === 0,
   `a week record a run behind shows the book as of its running day's close, today's call as information: ${JSON.stringify(ethLag)} / ${lagView.notes.guard}`);
const lagOut = { ...lagWeek, trading: { ...lagWeek.trading, guarded: { ...lagWeek.trading.guarded, now: { ...lagWeek.trading.guarded.now, symbols: outWeek.trading.guarded.now.symbols } } } };
const ethLagOut = api.marketView('crypto', record, lagOut, now, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethLagOut.status === 'out' && ethLagOut.since.startsWith('sold Tue 13 Oct'), `a coin the lagging record has sold stays out, never the plain week's Held: ${JSON.stringify(ethLagOut)}`);
const lagUp = { ...lagOut, trading: { ...lagOut.trading, guarded: { ...lagOut.trading.guarded, now: { ...lagOut.trading.guarded.now,
  symbols: { BTC: { action: 'out', reason: 'call', entry: null, entry_price: null, call: 'flat', sold_on: null, realized: null, trades: [] } } } } } };
const btcLagWeek = { ...lagUp, next: { ...lagUp.next, instruments: { ...lagUp.next.instruments, BTC: { base: 83773, predictions: [pred('PoissonMonteCarlo Model', 7, 85000, 86000)] } } } };
const btcLag = api.marketView('crypto', record, btcLagWeek, now, G).open.find((q) => q.game === 'week' && q.symbol === 'BTC');
ok(btcLag && btcLag.status === 'out' && btcLag.guard.state === 'buy' && btcLag.guard.reason.includes('would buy it') && btcLag.statusText.includes('first one. The week record is a run behind')
   && btcLag.since === 'daily call up today: in by the rule',
   `a lagging record's out coin on today's up call: the rule would buy it, and the texts read as sentences: ${JSON.stringify(btcLag)}`);
// a week record ahead of the daily one (the daily export failed): its state is the newer one and is applied
const aheadWeek = { ...outWeek, trading: { ...outWeek.trading, guarded: { ...outWeek.trading.guarded, now: { ...outWeek.trading.guarded.now, running_day: '2026-10-16' } } } };
const aheadView = api.marketView('crypto', record, aheadWeek, now, G);
const ethAhead = aheadView.open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(aheadView.week.guarded === true && ethAhead.status === 'out' && aheadView.status.openHeld === 0, `a week record ahead of the daily one keeps the guard's rows: ${JSON.stringify(ethAhead)}`);
// ...and its closes are used when they are newer than the daily record's: the move at the last close
const aheadHold = { ...aheadWeek, instruments: aheadWeek.instruments.map((i) => (i.symbol === 'ETH' ? { ...i, closes: [...i.closes, ['2026-10-15', 2460]] } : i)),
  trading: { ...aheadWeek.trading, guarded: { ...aheadWeek.trading.guarded, now: { ...aheadWeek.trading.guarded.now,
    symbols: { ETH: { action: 'hold', reason: null, entry: '2026-10-15', entry_price: 2400, call: 'up', sold_on: null, realized: null, trades: [] } } } } } };
ok(api.marketView('crypto', record, aheadHold, now, G).open.find((q) => q.game === 'week').profitPctClose === 2.5, 'a week record ahead brings the newer close for the move');
// a guard block without this week's state (no daily ticket for the week yet): nothing is bought, never the plain week's entry
const noState = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: null } } };
const noStateView = api.marketView('crypto', record, noState, new Date('2026-10-16T05:00:00Z'), G);
ok(noStateView.open.filter((q) => q.game === 'week').every((q) => q.status === 'out') && noStateView.status.openToBuy === 0 && noStateView.status.openHeld === 0,
   `without the guard's state for the week the week rows are out: ${JSON.stringify(noStateView.open.filter((q) => q.game === 'week'))}`);
// a crypto sale shown after a later day began names its game day, as the closed positions do
const lag2 = { ...guardedWeek, trading: { ...guardedWeek.trading, guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now, running_day: '2026-10-13' } } } };
const ethLag2 = api.marketView('crypto', record, lag2, now, G).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethLag2.status === 'sold' && ethLag2.statusText.includes('(Tue 13 Oct, close 02:00 next day)') && ethLag2.statusText.includes('2 runs behind'), `a past crypto sale names its game day: ${ethLag2.statusText}`);
// the plain week on its last day: an up daily call does not hold, the week ends at the close
const sundayUp = { ...sundayRecord, next: { ...sundayRecord.next, instruments: { ...sundayRecord.next.instruments, ETH: { base: 2421.6, predictions: [pred('GARCH Model', 7, 2440, 2460)] } } } };
const ethPlainEnd = api.marketView('crypto', sundayUp, weekRecord, new Date('2026-10-18T10:00:00Z')).open.find((q) => q.game === 'week' && q.symbol === 'ETH');
ok(ethPlainEnd && ethPlainEnd.guard && ethPlainEnd.guard.state === 'weekend' && ethPlainEnd.guard.reason.includes('last day'), `on the week's last day the information is the week end: ${JSON.stringify(ethPlainEnd && ethPlainEnd.guard)}`);
// a sized daily row that is not up reads flat, whatever its bin (MarketSettle.call_word)
const sizedDay = { ...record, next: { ...record.next, instruments: { ...record.next.instruments, ETH: { base: 2421.6, predictions: [pred('GARCH Model', 1, 2300, 2350, 0)] } } } };
ok(api.marketView('crypto', sizedDay, null, now).none.day.find((c) => c.symbol === 'ETH').call === 'flat', 'a sized row at size 0 is flat, not down');

// --- shares: the open-to-close rule and its clocks --------------------------------
const sharesRecord = { ...record, market: 'shares', instruments: record.instruments.map((i) => ({ ...i, symbol: i.symbol === 'BTC' ? 'NVDA' : (i.symbol === 'ETH' ? 'AAPL' : i.symbol) })),
  next: { ...record.next, made_on: '2026-10-14', instruments: { NVDA: record.next.instruments.BTC, AAPL: record.next.instruments.ETH } },
  models: record.models.map((m) => ({ ...m, open_total: m.pnl_cash_total, open_trades: m.trades })),
  trading: { ...record.trading, currency: 'USD', open: { models: record.trading.models, benchmark: record.trading.benchmark }, followed: { ...record.trading.followed, rule: 'open' } } };
const before = api.marketView('shares', sharesRecord, null, new Date('2026-10-15T09:00:00Z'));   // 11:00 Belgian, before New York opens
const nvda = before.open.find((q) => q.symbol === 'NVDA');
ok(before.day.ruleName === 'open to close' && nvda.status === 'buy at open' && nvda.buyText === 'today 15:30' && nvda.statusText.includes('15:30') && nvda.sellText === 'today 22:00' && before.notes.closeDay.includes('22:00'), `shares before the open: ${JSON.stringify(nvda)}`);
const during = api.marketView('shares', sharesRecord, null, new Date('2026-10-15T15:00:00Z'));
ok(during.open.find((q) => q.symbol === 'NVDA').status === 'buy now' && during.closed[0].exit === 'close 22:00', 'shares during the session: buy now, closed positions exit at 22:00');
// a guarded share week: a first up call mid-week buys at the open, measured until then from the last close
const sharesWeek = { ...weekRecord, market: 'sharesweek', instruments: [{ symbol: 'NVDA', active: true, last_close: 229.28, closes: [['2026-10-09', 229.28]] }],
  next: { ...weekRecord.next, made_on: '2026-10-09', instruments: { NVDA: { base: 229.28, predictions: [pred('PoissonMonteCarlo Model', 7, 235, 240)] } } },
  trading: { ...weekRecord.trading, rule: 'open', guarded: { ...guardedWeek.trading.guarded, now: { ...guardedWeek.trading.guarded.now,
    symbols: { NVDA: { action: 'buy', reason: 'first up call', entry: '2026-10-15', entry_price: null, call: 'up', sold_on: null, realized: null, trades: [] } } } } } };
const nvdaW = api.marketView('shares', sharesRecord, sharesWeek, new Date('2026-10-15T09:00:00Z'), G).open.find((q) => q.game === 'week' && q.symbol === 'NVDA');
const nvdaOut = api.marketView('shares', sharesRecord, { ...sharesWeek, trading: { ...sharesWeek.trading, guarded: { ...sharesWeek.trading.guarded, now: { ...sharesWeek.trading.guarded.now, settled_days: [],
  symbols: { NVDA: { action: 'out', reason: 'call', entry: null, entry_price: null, call: 'flat', sold_on: null, realized: null, trades: [] } } } } } }, new Date('2026-10-15T09:00:00Z'), G).open.find((q) => q.game === 'week');
ok(nvdaOut.statusText === "not bought yet: today's daily call is flat; in on the first up daily call this week", `before anything settled the out text names today's call: ${nvdaOut.statusText}`);
ok(nvdaW && nvdaW.status === 'buy at open' && nvdaW.buyText === 'today 15:30' && nvdaW.openRate === 81682 && nvdaW.openRateNote === 'last close; the book buys at the open of Thu 15 Oct',
   `a share bought at today's open is measured from the last close, not the week base: ${JSON.stringify(nvdaW)}`);
// a lagging share week: the daily record already has the running day's close; the entry shown is the close before it
const sharesAhead = { ...sharesRecord, next: { ...sharesRecord.next, made_on: '2026-10-15' },
  instruments: sharesRecord.instruments.map((i) => (i.symbol === 'NVDA' ? { ...i, last_close: 80000, closes: [...i.closes, ['2026-10-15', 80000]] } : i)) };
sharesAhead.next = { ...sharesAhead.next, instruments: { ...sharesAhead.next.instruments, NVDA: { ...sharesAhead.next.instruments.NVDA, base: 80000 } } };
const nvdaLag = api.marketView('shares', sharesAhead, sharesWeek, new Date('2026-10-16T09:00:00Z'), G).open.find((q) => q.game === 'week' && q.symbol === 'NVDA');
ok(nvdaLag && nvdaLag.status === 'held' && nvdaLag.openRate === 81682 && nvdaLag.openRateNote.includes('the week record catches up'),
   `a lagging share buy is measured from the close before its day, not its own close: ${JSON.stringify(nvdaLag)}`);
const nvdaNight = api.marketView('shares', sharesRecord, sharesWeek, new Date('2026-10-15T21:00:00Z'), G).open.find((q) => q.game === 'week' && q.symbol === 'NVDA');
ok(nvdaNight.status === 'held' && nvdaNight.openRateNote.startsWith('bought at the open of Thu 15 Oct'), `after the close the note says the share was bought: ${JSON.stringify(nvdaNight)}`);

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
