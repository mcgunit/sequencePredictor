// The Crypto and Shares records (markets.js): what the view models, the books the plan and the week card
// follow and the market clock must keep true. Runs on a fixture written to a temp dir, never on data/.
//   node test/markets.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const markets = require('../markets');

let passed = 0;
function ok(condition, message) { assert.ok(condition, message); passed += 1; }

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'markets-'));
const record = {
  market: 'crypto', generated_at: '2026-10-02T07:15:00+00:00', k: 10, fee: 0.001,
  chance: { exact: 0.1, adjacent: 0.28, direction: 0.5 }, newest_game_day: '2026-10-01',
  instruments: [
    { symbol: 'BTC', name: 'Bitcoin', position: 0, quote: 'USDT', active: true, last_close: 84500, last_date: '2026-10-02',
      closes: [['2026-09-29', 83500], ['2026-09-30', 83663], ['2026-10-01', 84000], ['bad', null]],
      edges: [null, [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025], [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025], null],
      moves: [null, 0.00195, 0.004, null],
      course: { 'Markov Model': [null, 4, 6, null], 'Odd Model': [null, null, 9, null] } },
    { symbol: 'ETH', name: 'Ethereum', position: 1, quote: 'USDT', active: false, last_close: 3000, last_date: '2026-10-01', closes: [], edges: [1, 2], moves: 'x', course: { 'Markov Model': [1] } },
  ],
  models: [
    { name: 'Markov Model', days: 12, positions: 60, exact_rate: 0.15, adjacent_rate: 0.3, direction_rate: 0.45, trades: 25, pnl_total: 0.0123, pnl_per_trade: 0.000492, first_day: '2026-09-20', last_day: '2026-10-01',
      pnl_cash_total: 12.34, pnl_cash_per_trade: 0.4936, wins: 14, win_rate: 0.56, hold_total: 15.5, hold_trades: 9, hold_wins: 6, hold_win_rate: 0.6667, hold_per_trade: 1.7222 },
    { name: 'Odd Model', days: 3, positions: 15, exact_rate: null, adjacent_rate: 0.2, direction_rate: 0.6, trades: 0, pnl_total: 0, pnl_per_trade: null },
  ],
  drawn_models: ['Markov Model'], best_model: 'Markov Model',
  trading: { stake: 100, fee_per_leg: 0.001, currency: 'USDT', rule: 'long when up', dates: ['2026-09-30', '2026-10-01'],
    models: { 'Markov Model': [['2026-09-30', 5.5, 5.5], ['2026-10-01', 6.84, 12.34]], 'Odd Model': [['2026-09-30', null, null], ['2026-10-01', 0, 0]] },
    benchmark: [['2026-09-30', -1.2, -1.2], ['2026-10-01', 3.0, 1.8]],
    hold_rule: 'kept while up',
    hold: { models: { 'Markov Model': [['2026-09-30', 5.6, 5.6], ['2026-10-01', 9.9, 15.5]] }, benchmark: [['2026-09-30', -1.1, -1.1], ['2026-10-01', 3.2, 2.1]] } },
  next: { made_on: '2026-10-01', made_at: '2026-10-02T07:15:00+00:00', for: 'the next trading day', instruments: {
    BTC: { last_close: 84000, last_date: '2026-10-01', predictions: [
      { model: 'Markov Model', bin: 7, direction: 1, price: 84900, low: 84500, high: 85300 },
      { model: 'Odd Model', bin: 0, direction: -1, price: 80000, low: null, high: 81000 } ] } } },
  daily: [{ date: '2026-09-30', models: 2, exact_mean: 0.1, direction_mean: 0.5, best_exact: 0.2, best_model: 'Markov Model', pnl_mean: 0 },
          { date: '2026-10-01', models: 2, exact_mean: 0.2, direction_mean: 0.4, best_exact: 0.4, best_model: 'Markov Model', pnl_mean: 0.001 }],
  days: [
    { date: '2026-10-01', best: 'Markov Model', exact_mean: 0.25,
      instruments: [{ symbol: 'BTC', return: 0.004, bin: 6, edges: [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025] },
                    { symbol: 'ETH', return: -0.041, bin: 0, edges: [-0.04, -0.025, -0.012, -0.005, 0.0, 0.004, 0.011, 0.02, 0.03] }],
      models: [{ name: 'Markov Model', bins: [6, 3], exact: 1, adjacent: 1, direction: 1, positions: 2, pnl: 0.003, pnl_cash: 0.2, trades: 1 },
               { name: 'Odd Model', bins: [1, null], exact: 0, adjacent: 0, direction: 0, positions: 2, pnl: 0, trades: 0 }] },
    { date: '2026-09-30', best: 'Odd Model', exact_mean: 0, instruments: [{ symbol: 'BTC', return: -0.002, bin: 4, edges: [] }, { symbol: 'ETH', return: 0.001, bin: 5, edges: [] }], models: [] },
    { date: 'bad' },
  ],
};
fs.writeFileSync(path.join(dir, 'crypto.json'), JSON.stringify(record));
fs.writeFileSync(path.join(dir, 'shares.json'), '{');

// the weekly MarketRows.py report and the regime log (phase M3)
const controls = fs.mkdtempSync(path.join(os.tmpdir(), 'controls-'));
fs.mkdirSync(path.join(controls, 'markets'));
const interval = (mean, lo, hi, verdict) => ({ mean, lo, hi, verdict, days: 40 });
const rowsRecord = {
  market: 'crypto', generated_at: '2026-10-04T12:00:00+00:00', days_requested: 40, days_scored: 40, first_day: '2026-08-21', last_day: '2026-09-29',
  k: 10, uniform_log_score: -2.302585, probability_floor: 0.0001, reference: 'GARCH Model', lockbox_days_withheld: 0,
  errors: { 'Chronos Model': 'no forecast succeeded before the backtest', 'XGBoost Model': 2 },
  rows: [
    { name: 'Regime HMM Model', kind: 'market', days: 40, scored_days: 40, log_score: -2.45, log_score_se: 0.03,
      vs_reference: interval(-0.24, -0.4, -0.1, 'worse'), vs_uniform: interval(-0.15, -0.3, -0.02, 'worse'),
      exact_rate: 0.09, adjacent_rate: 0.27, direction_rate: 0.48, trades: 90, pnl_return: -0.02, pnl_per_trade: -0.0002 },
    { name: 'random', kind: 'baseline', days: 40, scored_days: 0, log_score: null, log_score_se: null, vs_reference: null, vs_uniform: null,
      exact_rate: 0.1, adjacent_rate: 0.28, direction_rate: 0.5, trades: 100, pnl_return: 0.0, pnl_per_trade: 0.0 },
    { name: 'GARCH Model', kind: 'market', days: 40, scored_days: 40, log_score: -2.21, log_score_se: 0.02,
      vs_reference: null, vs_uniform: interval(0.09, 0.05, 0.13, 'better'),
      exact_rate: 0.12, adjacent_rate: 0.3, direction_rate: 0.5, trades: 100, pnl_return: 0.01, pnl_per_trade: 0.0001 },
    { name: 'Markov Model', kind: 'base', days: 40, scored_days: 40, log_score: -2.3, log_score_se: 0.02,
      vs_reference: interval(-0.09, -0.2, 0.02, 'no difference'), vs_uniform: interval(0.0, -0.05, 0.05, 'no difference'),
      exact_rate: 0.1, adjacent_rate: 0.28, direction_rate: 0.49, trades: 95, pnl_return: -0.005, pnl_per_trade: null },
  ],
};
fs.writeFileSync(path.join(controls, 'markets', 'crypto-rows.json'), JSON.stringify(rowsRecord));
fs.writeFileSync(path.join(controls, 'markets', 'shares-rows.json'), JSON.stringify({ market: 'crypto', rows: [] }));
const regimeRecord = { updated: '2026-10-01', 'Regime HMM Model': [
  { date: '2026-09-30', row: 'Regime HMM Model', regimes: 2, dominant: 0, volatility_rank: 1, label: 'calm', probability: 0.6, weights: [0.6, 0.4], expected_return: [0.001, 0.0], volatility: [0.02, 0.03], template: 1 },
  { date: '2026-10-01', row: 'Regime HMM Model', regimes: 2, dominant: 1, volatility_rank: 2, label: 'turbulent', probability: 0.97, weights: [0.03, 0.97], expected_return: [0.0012, -0.002], volatility: [0.04, 0.05], template: 3 } ],
  'Regime HMM Single Model': [{ date: '2026-10-01', row: 'Regime HMM Single Model', regimes: 1, dominant: 0, volatility_rank: 1, label: 'normal', probability: 1, weights: [1], expected_return: [0.0003, 0.0001], volatility: [0.03, 0.04], template: 1 }] };
fs.writeFileSync(path.join(dir, 'crypto-regimes.json'), JSON.stringify(regimeRecord));

// --- loading ------------------------------------------------------------------
ok(markets.loadMarket(dir, 'crypto') && markets.loadMarket(dir, 'crypto').market === 'crypto', 'a market record loads');
ok(markets.loadMarket(dir, 'shares') === null, 'a broken file is no record');
ok(markets.loadMarket(path.join(dir, 'nowhere'), 'crypto') === null, 'a missing folder is no record');
fs.writeFileSync(path.join(dir, 'shares.json'), JSON.stringify({ market: 'crypto', instruments: [], models: [] }));
ok(markets.loadMarket(dir, 'shares') === null, 'a record naming another market is refused');

// --- the view model -----------------------------------------------------------
const view = markets.describeMarket(record);
ok(view.k === 10 && view.fee === 0.001 && view.scoredDays === 12 && view.newestGameDay === '2026-10-01' && view.madeOn === '2026-10-01',
  'the view carries the game parameters and the newest days');
const markov = view.models[0];
ok(markov.name === 'Markov Model' && markov.aboveExact && markov.aboveAdjacent && !markov.aboveDirection,
  'rates are read against chance: 15% exact and 30% adjacent are above, 45% direction is not');
ok(view.models[1].exact === null && !view.models[1].aboveExact && view.models[1].pnlPerTrade === null, 'missing rates read as null, never above chance');
ok(markov.pnlCash === 12.34 && markov.winRate === 0.56 && markov.wins === 14 && view.models[1].pnlCash === null && markov.holdTotal === 15.5 && markov.holdTrades === 9 && view.models[1].holdTotal === null,
  'the money fields of both rules pass through, missing ones read as null');
ok(view.trading.hold.models['Markov Model'][1].total === 15.5 && view.trading.hold.benchmark[1].total === 2.1 && markets.describeTrading({ models: {} }).hold.benchmark.length === 0,
  'the hold book describes next to the daily one');
ok(view.trading && view.trading.stake === 100 && view.trading.currency === 'USDT' && view.trading.dates.length === 2 && view.trading.models['Markov Model'][1].total === 12.34
  && view.trading.models['Odd Model'][0].total === null && view.trading.benchmark[1].total === 1.8 && markets.describeTrading(null) === null && markets.describeTrading({ models: 7 }).dates.length === 0,
  'the trading book describes per model and for the market, and tolerates a missing or odd record');
const btc = view.instruments[0];
ok(btc.closes.length === 3 && btc.lastClose === 84500 && btc.nextBase === 84000 && btc.nextDate === '2026-10-01', 'a malformed close point is dropped; the next-day base is the newest game day\'s close, not the newer bar');
ok(btc.edges.length === 3 && btc.edges[0] === null && btc.edges[2].length === 9 && btc.moves.join(',') === ',0.00195,0.004'
  && btc.course['Markov Model'].join(',') === ',4,6' && btc.course['Odd Model'][2] === 9 && view.best === 'Markov Model',
  'edges, moves and every model\'s bins follow the kept closes; the best model is named');
ok(view.instruments[1].edges.length === 0 && view.instruments[1].moves.length === 0 && view.instruments[1].course['Markov Model'].length === 0,
  'series whose length disagrees with the closes are blanked, never misaligned');
ok(btc.next.length === 2 && btc.next[0].price === 84900 && btc.next[1].low === null && btc.next[1].high === 81000,
  'the next-day predictions carry price and open interval');
ok(view.instruments[1].next.length === 0 && view.instruments[1].active === false, 'an instrument without a next-day entry has none, inactive is kept');
ok(view.drawn.length === 1 && view.daily.length === 2, 'drawn models and the daily series pass through');
ok(view.days.length === 2 && view.days[0].date === '2026-10-01' && view.days[0].instruments[1].bin === 0 && view.days[0].models[1].bins[1] === null,
  'the day records pass through, a malformed day is dropped, a missing bin stays null');
ok(markets.describeDays(null).length === 0 && markets.describeDays([{ date: '2026-01-01', instruments: [] }]).length === 1, 'day records describe safely');
ok(markets.describeDays([{ date: '2026-01-01', instruments: [null, { symbol: 'A' }], models: [null, 7, { bins: [1] }, { name: 'X', bins: [1] }] }])[0].models.length === 1
  && markets.describeDays([{ date: '2026-01-01', instruments: [null, { symbol: 'A' }], models: [] }])[0].instruments.length === 1,
  'malformed instrument and model entries are dropped, never thrown');
ok(markets.describeMarket({ market: 'crypto', instruments: [null, { symbol: 'BTC' }], models: [] }).instruments.length === 1, 'a null instrument in the record is dropped');

const bare = markets.describeMarket({ market: 'shares', instruments: [], models: [] });
ok(bare.scoredDays === 0 && bare.chance.exact === 0.1 && bare.madeOn === null && bare.daily.length === 0, 'an empty record describes with defaults');

// --- the report and the regime log ------------------------------------------
ok(markets.loadRows(controls, 'crypto') && markets.loadRows(controls, 'crypto').reference === 'GARCH Model', 'the rows report loads');
ok(markets.loadRows(controls, 'shares') === null && markets.loadRows(path.join(controls, 'nowhere'), 'crypto') === null, 'a report naming another market, or none, is no report');
ok(markets.loadRegimes(dir, 'crypto') && markets.loadRegimes(dir, 'shares') === null, 'the regime log loads when it exists');
const rep = markets.describeRows(rowsRecord);
ok(rep.rows.map((r) => r.name).join(',') === 'GARCH Model,Markov Model,Regime HMM Model,random', 'rows sort best log-score first, rows without probabilities last');
ok(rep.referenceRow && rep.referenceRow.isReference && rep.referenceAboveUniform && rep.betterThanReference.length === 0,
  'the reference is found, it is above uniform, and no row beats it in this record');
ok(rep.rows[2].vsReference.verdict === 'worse' && rep.rows[1].vsReference.verdict === 'no difference' && rep.rows[3].logScore === null && rep.uniform < -2.3,
  'intervals and verdicts pass through, uniform is log 1/10');
ok(markets.describeRows(null) === null && markets.describeRows({ market: 'shares', rows: [] }).rows.length === 0, 'no report, or an empty one, describes safely');
const better = markets.describeRows({ ...rowsRecord, rows: rowsRecord.rows.map((r) => (r.name === 'Markov Model' ? { ...r, vs_reference: interval(0.1, 0.02, 0.2, 'better') } : r)) });
ok(better.betterThanReference.join(',') === 'Markov Model', 'a row whose interval against GARCH lies above zero is named');
const regimes = markets.describeRegimes(regimeRecord, ['BTC', 'ETH']);
// the HMM rows' feature set travels with the results (14 Oct 2026): the report's hmm_features and the regime log's features
ok(rep.hmmFeatures === null && markets.describeRows({ ...rowsRecord, hmm_features: 'returns' }).hmmFeatures === 'returns' && regimes[0].features === null
   && markets.describeRegimes({ 'Regime HMM Model': [{ date: '2026-10-01', row: 'Regime HMM Model', regimes: 2, features: ['returns', 'vol'] }] }, ['BTC'])[0].features.join() === 'returns,vol',
   'the feature set is read from the report and the regime log when present');
ok(regimes.length === 2 && regimes[0].row === 'Regime HMM Model' && regimes[0].date === '2026-10-01' && regimes[0].template === 3 && regimes[0].label === 'turbulent'
  && regimes[0].expected[0].symbol === 'BTC' && regimes[0].expected[1].value === -0.002 && regimes[0].history === 2,
  'the newest reading per row, with the instruments named');
ok(markets.describeRegimes(null, []).length === 0 && markets.describeRegimes({ updated: 'x' }, []).length === 0, 'no readings, no rows');
const full = markets.describeMarket(record, { rows: rowsRecord, regimes: regimeRecord });
ok(full.rows && full.rows.rows.length === 4 && full.regimes.length === 2 && full.regimes[0].expected[1].symbol === 'ETH', 'the view carries the report and the readings');
ok(markets.describeMarket(record).rows === null && markets.describeMarket(record).regimes.length === 0, 'without extras the view has neither');
fs.rmSync(dir, { recursive: true, force: true });
fs.rmSync(controls, { recursive: true, force: true });
// --- when a call is judged (4 Oct 2026): the day a ticket is for and its close in Belgian time ---
ok(markets.nextGameDay('crypto', '2026-10-02') === '2026-10-03' && markets.nextGameDay('crypto', '2026-10-31') === '2026-11-01' && markets.nextGameDay('crypto', '2026-12-31') === '2027-01-01', 'a crypto ticket is for the next calendar day');
ok(markets.nextGameDay('shares', '2026-10-02') === '2026-10-05' && markets.nextGameDay('shares', '2026-10-03') === '2026-10-05' && markets.nextGameDay('shares', '2026-10-04') === '2026-10-05' && markets.nextGameDay('shares', '2026-10-05') === '2026-10-06',
   'a shares ticket is for the next weekday - Friday, Saturday and Sunday all point at Monday');
ok(markets.nextGameDay('shares', '2026-11-25') === '2026-11-27' && markets.nextGameDay('shares', '2026-12-24') === '2026-12-28' && markets.nextGameDay('shares', '2027-12-23') === '2027-12-27'
   && markets.nextGameDay('shares', '2026-07-02') === '2026-07-06', 'New York holidays are skipped: Thanksgiving, Christmas, the observed days');
const perYear = (year) => [...markets.NYSE_CLOSED].filter((d) => d.startsWith(year)).length;
ok(perYear('2026') === 10 && perYear('2027') === 10 && markets.NYSE_CLOSED.size === 20 && [...markets.NYSE_CLOSED].every((d) => ![0, 6].includes(new Date(d).getUTCDay())), 'the closure table has ten weekdays in each announced year');
ok(markets.nextGameDay('crypto', 'soon') === null && markets.nextGameDay('crypto', '2026-02-30') === null && markets.nextGameDay('crypto', '2026-13-01') === null && markets.closeMoment('shares', null) === null, 'malformed and impossible dates give null, not a roll-over');
// the moments in Belgian time (brusselsText, as dayStatus and dayStatusWeek say them)
const belgian = (moment) => markets.brusselsText(moment);
ok(markets.brusselsText(new Date('2026-10-03T08:29:00Z'), true) === '10:29 Belgian time on Sat 03/10' && markets.brusselsText(new Date('bad')) === null && markets.brusselsText('2026-10-03') === null,
   `a moment in Belgian time, with its date: ${markets.brusselsText(new Date('2026-10-03T08:29:00Z'), true)}`);
ok(markets.closeMoment('crypto', '2026-10-03').toISOString() === '2026-10-04T00:00:00.000Z' && belgian(markets.closeMoment('crypto', '2026-10-03')) === '02:00 Belgian time'
   && belgian(markets.closeMoment('crypto', '2026-12-10')) === '01:00 Belgian time', 'a crypto day closes at 00:00 UTC: 02:00 Belgian time in summer, 01:00 in winter');
ok(belgian(markets.closeMoment('crypto', '2026-10-24')) === '02:00 Belgian time' && belgian(markets.closeMoment('crypto', '2026-10-25')) === '01:00 Belgian time'
   && belgian(markets.closeMoment('crypto', '2027-03-27')) === '01:00 Belgian time', 'the Belgian clock-change nights are computed, not assumed');
ok(belgian(markets.closeMoment('shares', '2026-10-05')) === '22:00 Belgian time' && belgian(markets.closeMoment('shares', '2026-12-10')) === '22:00 Belgian time',
   `New York closes at 22:00 Belgian time in summer and in winter: ${belgian(markets.closeMoment('shares', '2026-10-05'))}`);
ok(belgian(markets.closeMoment('shares', '2026-10-28')) === '21:00 Belgian time' && belgian(markets.closeMoment('shares', '2027-03-16')) === '21:00 Belgian time' && belgian(markets.closeMoment('shares', '2026-03-09')) === '21:00 Belgian time',
   `between the two clock changes New York closes at 21:00 Belgian time: ${belgian(markets.closeMoment('shares', '2026-10-28'))}`);
ok(markets.closeMoment('shares', '2026-10-05').toISOString() === '2026-10-05T20:00:00.000Z' && markets.closeMoment('shares', '2026-12-10').toISOString() === '2026-12-10T21:00:00.000Z'
   && markets.closeMoment('shares', '2026-03-09').toISOString() === '2026-03-09T20:00:00.000Z', 'the New York close as an instant, also in the US-only summer-time week');
ok(belgian(markets.closeMoment('shares', '2026-11-27')) === '19:00 Belgian time' && belgian(markets.closeMoment('shares', '2026-12-24')) === '19:00 Belgian time',
   `an early close is 13:00 New York time: ${belgian(markets.closeMoment('shares', '2026-11-27'))}`);
ok(belgian(markets.openMoment('shares', '2026-10-05')) === '15:30 Belgian time' && belgian(markets.openMoment('shares', '2026-10-28')) === '14:30 Belgian time'
   && belgian(markets.openMoment('shares', '2026-11-27')) === '15:30 Belgian time', `the New York open is computed per date, also on an early-close day: ${belgian(markets.openMoment('shares', '2026-10-28'))}`);
ok(markets.openMoment('crypto', '2026-10-03').toISOString() === '2026-10-03T00:00:00.000Z', 'a crypto day opens at 00:00 UTC');
ok(markets.dayStatus('crypto', '2026-10-03', new Date('2026-10-03T09:00:00Z')) === 'the UTC day 2026-10-03 is running now, 9 hours in; it closes at 02:00 Belgian time'
   && markets.dayStatus('crypto', '2026-10-03', new Date('2026-10-04T05:00:00Z')).startsWith('the UTC day 2026-10-03 has already closed (02:00 Belgian time)')
   && markets.dayStatus('shares', '2026-10-05', new Date('2026-10-03T09:00:00Z')) === 'the New York session of Monday 2026-10-05 has not opened yet - it opens at 15:30 Belgian time and closes at 22:00 Belgian time'
   && markets.dayStatus('shares', '2026-10-05', new Date('2026-10-05T15:00:00Z')).includes('is running now, 1.5 hours in')
   && markets.dayStatus('shares', '2026-10-05', new Date('2026-10-05T21:00:00Z')).includes('has already closed') && markets.dayStatus('crypto', 'bad', new Date()) === null, 'the day status follows the clock');

// the open-to-close book for shares (4 Oct 2026), and the same rules with shorts on
const sharesRecord = { ...record, market: 'shares',
  models: [{ ...record.models[0], open_total: 9.5, open_trades: 12, open_wins: 7, open_win_rate: 0.5833, open_per_trade: 0.7917 }, record.models[1]],
  trading: { ...record.trading, currency: 'USD', open_rule: 'open to close',
    open: { models: { 'Markov Model': [['2026-09-30', 3.0, 3.0], ['2026-10-01', 6.5, 9.5]] }, benchmark: [['2026-09-30', 1.0, 1.0], ['2026-10-01', 2.0, 3.0]] } } };
// with shorts on (MarketSettle.ledger "short"): the same rules, a lower-half bin sold first; the fixture's shares record carries them
const shortedRecord = { ...sharesRecord,
  models: [{ ...sharesRecord.models[0], short_total: 4.25, short_trades: 20, short_win_rate: 0.55, short_per_trade: 0.2125, short_hold_total: 3.1, short_open_total: -1.2 }, sharesRecord.models[1]],
  trading: { ...sharesRecord.trading, short_rule: 'with shorts on',
    short: { daily: { models: { 'Markov Model': [['2026-09-30', 2.0, 2.0], ['2026-10-01', 2.25, 4.25]] }, benchmark: [['2026-09-30', -1.5, -1.5], ['2026-10-01', -2.6, -4.1]] }, hold: { models: { 'Markov Model': [['2026-09-30', 1.0, 1.0], ['2026-10-01', 2.1, 3.1]] } },
             open: { models: { 'Markov Model': [['2026-09-30', -0.5, -0.5], ['2026-10-01', -0.7, -1.2]] } } } } };
const sharesView = markets.describeMarket(sharesRecord);
ok(sharesView.trading.open && sharesView.trading.open.models['Markov Model'].length === 2 && sharesView.trading.openRule === 'open to close' && sharesView.models[0].openTotal === 9.5
   && view.trading.open === null && view.models[0].openTotal === null, 'the open-to-close book is read when present and absent otherwise');

// position sizes (the RL Position Model): the view carries the size next to the bin
const sizedRecord = { ...record, next: { ...record.next, instruments: { ...record.next.instruments,
  BTC: { ...record.next.instruments.BTC, predictions: record.next.instruments.BTC.predictions.map((p) => (p.model === 'Markov Model' ? { ...p, size: 1.5 } : p))
    .concat([{ model: 'RL Position Model', bin: 6, direction: 1, price: 84800, low: 84300, high: 85100, size: 0 }]) } } } };
const sizedView = markets.describeMarket(sizedRecord);
ok(sizedView.instruments[0].next[0].size === 1.5 && sizedView.instruments[0].next[2].model === 'RL Position Model' && sizedView.instruments[0].next[2].size === 0, 'a sized call carries its size, a size of 0 included');
ok(markets.describeMarket(record).instruments[0].next[0].size === null, 'a plain row has no size');

// shorts in the paper books
const shortedView = markets.describeMarket(shortedRecord);
ok(shortedView.trading.short && Object.keys(shortedView.trading.short).sort().join() === 'daily,hold,open' && shortedView.trading.short.daily.models['Markov Model'][1].total === 4.25 && shortedView.trading.short.daily.sold[1].total === -4.1 && shortedView.trading.short.hold.sold.length === 0
   && shortedView.models[0].shortTotal === 4.25 && shortedView.models[0].shortHoldTotal === 3.1 && view.trading.short === null && view.models[0].shortTotal === null, 'the long-and-short books are read when present');

// the followed-plan record (MarketSettle.followed_plan, 13 Oct 2026): live days apart from the back-built ones
const followed = markets.describeMarket({ ...sharesRecord, trading: { ...sharesRecord.trading, followed: { rule: 'open', min_trades: 10, first_live: '2026-10-01', total: 1.5, market_total: -2, live_total: 1.5, live_market: -2, live_days: 1, live_positions: 2,
  points: [['2026-09-30', 'Markov Model', 'book', 2, 0, 0, false, 1.1], ['2026-10-01', 'Markov Model', 'exact', 2, 1.5, 1.5, true, -2, { BTC: 1.5 }], ['short', 1], 'bad'] } } }).trading.followed;
ok(followed && followed.rule === 'open' && followed.minTrades === 10 && followed.firstLive === '2026-10-01' && followed.liveTotal === 1.5 && followed.liveMarket === -2 && followed.liveDays === 1 && followed.livePositions === 2
   && followed.points.length === 2 && followed.points[0].live === false && followed.points[1].live === true && followed.points[1].pickedBy === 'exact' && followed.points[1].market === -2
   && followed.points[1].bySymbol.BTC === 1.5 && Object.keys(followed.points[0].bySymbol).length === 0 && view.trading.followed === null,
   `the followed record keeps its live days, totals and per-instrument money; a malformed point is dropped: ${JSON.stringify(followed)}`);
const backBuilt = markets.describeFollowed({ rule: 'daily', first_live: null, points: [['2026-09-30', null, null, 2, 0.5, 0.5, false, 1.1]] });
ok(backBuilt.rule === 'daily' && backBuilt.firstLive === null && backBuilt.liveDays === 0 && backBuilt.points[0].model === null && markets.describeFollowed({ rule: 'x', points: 7 }) === null
   && markets.describeFollowed({ rule: 'x', points: [] }).rule === 'daily', 'a record with no live day yet has no first live day; a record without points is none');

// the week game (M5): its view rides along with the daily one; the week's clock and its guarded book
const weekRecord = { ...record, market: 'cryptoweek', newest_game_day: '2026-09-27',
  models: [{ ...record.models[0], days: 12, trades: 25, pnl_cash_total: 7.5 }, record.models[1]],
  instruments: [{ ...record.instruments[0], last_close: 83000, last_date: '2026-09-27', closes: [['2026-09-20', 82000], ['2026-09-27', 83000]], edges: [null, null], moves: [null, null], course: {} }, record.instruments[1]],
  next: { made_on: '2026-09-27', made_at: '2026-09-28T07:40:00+00:00', for: 'the next week', instruments: {
    BTC: { last_close: 83000, last_date: '2026-09-27', predictions: [{ model: 'Markov Model', bin: 7, direction: 1, price: 85000, low: 84200, high: 85900, size: null }, { model: 'Odd Model', bin: 3, direction: -1, price: 82500, low: 82000, high: 82900, size: null }] } } },
  trading: { ...record.trading, dates: ['2026-09-20', '2026-09-27'] } };
const weekView = markets.describeMarket(weekRecord);
const withWeek = markets.describeMarket(record, { week: weekView });
ok(withWeek.week && withWeek.week.market === 'cryptoweek' && markets.describeMarket(record).week === null && markets.describeMarket(record, { week: { nonsense: true } }).week === null, 'the week view rides along when it is a market view');

// the books Today's plan and the week card follow (pickPlan, pickWeek, read by api.js, admin.js and workbench.js): the
// best book under the market's rule from MIN_PLAN_TRADES / MIN_WEEK_TRADES positions on; before that, the best exact rate
const planOf = (rec, market) => markets.pickPlan(markets.describeMarket(rec), market);
const weekOf = (rec, market) => markets.pickWeek(markets.describeMarket(rec), market);
const cryptoPlan = planOf({ ...record, models: [record.models[0], { ...record.models[1], trades: 30, pnl_cash_total: 20 }] }, 'crypto');
ok(markets.MIN_PLAN_TRADES === 10 && markets.MIN_WEEK_TRADES === 5 && cryptoPlan.model.name === 'Odd Model' && cryptoPlan.byBook === true && cryptoPlan.key === 'pnlCash'
   && cryptoPlan.ruleName === 'daily round trip', 'crypto: the plan follows the best daily book with 10 positions, not the best exact rate');
const youngPlan = planOf({ ...record, models: [{ ...record.models[0], trades: 9 }, { ...record.models[1], trades: 9, pnl_cash_total: 20 }] }, 'crypto');
ok(youngPlan.model.name === 'Markov Model' && youngPlan.byBook === false, 'no daily book with 10 positions yet: the plan follows the best exact rate');
const sharesPlan = planOf(sharesRecord, 'shares');
ok(sharesPlan.model.name === 'Markov Model' && sharesPlan.byBook === true && sharesPlan.key === 'openTotal' && sharesPlan.ruleName === 'open to close'
   && planOf({ ...sharesRecord, models: [{ ...sharesRecord.models[0], open_trades: 3 }, sharesRecord.models[1]] }, 'shares').byBook === false,
   'shares: the plan follows the open-to-close book, not the daily one, and not before it has 10 positions');
const cryptoWeek = markets.pickWeek(weekView, 'crypto');
ok(cryptoWeek.model.name === 'Markov Model' && cryptoWeek.byBook === true && cryptoWeek.ruleName === 'weekly round trip', 'crypto: the week card follows the weekly book');
const sharesWeekOf = (oddTrades) => weekOf({ ...weekRecord, market: 'sharesweek',
  models: [{ ...weekRecord.models[0], open_total: 3.2, open_trades: 4 }, { ...weekRecord.models[1], open_total: 9, open_trades: oddTrades }] }, 'shares');
ok(sharesWeekOf(5).model.name === 'Odd Model' && sharesWeekOf(5).byBook === true && sharesWeekOf(5).key === 'openTotal' && sharesWeekOf(5).ruleName === 'Monday open to Friday close'
   && sharesWeekOf(4).model.name === 'Markov Model' && sharesWeekOf(4).byBook === false,
   'shares: the week card follows the Monday open to Friday close book from 5 positions on, the best exact rate before');
ok(markets.dayStatusWeek(new Date('2026-10-19T14:00:00Z'), new Date('2026-10-19T13:30:00Z'), new Date('2026-10-23T20:00:00Z')).startsWith('the week is running, 0.5 h in'), 'a week under a day old counts hours, not 0 days');
ok(markets.dayStatusWeek(new Date('2026-10-20T13:40:00Z'), new Date('2026-10-19T13:30:00Z'), new Date('2026-10-23T20:00:00Z')).startsWith('the week is running, 1 day in'), 'one day in reads 1 day');
{
  const g = markets.describeGuarded({ rule: 'guard', points: [['2026-10-11', 'GARCH Model', 2, -3.5, -3.5, true, -31.98, {}, [], true], ['2026-08-30', 'X', 1, 1, 1, false, 2, {}, [], false], 'bad'] });
  ok(g && g.points.length === 2 && g.forwardFrom === '2026-10-19' && g.points[0].live === true && g.points[0].market === -31.98 && g.points[1].guarded === false
     && markets.describeGuarded(null) === null && markets.GUARD_SHOWN_FROM === '2026-10-26',
     `the guarded book is read for the Admin card, forward_from defaulting to 19 Oct: ${JSON.stringify(g)}`);
}
ok(markets.weekEnd('crypto', '2026-10-04') === '2026-10-11' && markets.weekEnd('shares', '2026-10-02') === '2026-10-09' && markets.weekEnd('shares', '2026-12-18') === '2026-12-24' && markets.weekEnd('shares', '2026-11-20') === '2026-11-27' && markets.weekEnd('crypto', 'bad') === null,
   'the week ends on the next Sunday for crypto and on the week\'s last session for shares, holidays skipped');

console.log(`markets.js: ${passed} checks passed`);
