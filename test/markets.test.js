// The Crypto and Shares pages (markets.js): what the view model must keep true
// and that the page renders from a settlement export. Runs on a fixture
// written to a temp dir, never on data/.
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
    { symbol: 'BTC', name: 'Bitcoin', position: 0, quote: 'USDT', active: true, last_close: 84000, last_date: '2026-10-01',
      closes: [['2026-09-29', 83500], ['2026-09-30', 83663], ['2026-10-01', 84000], ['bad', null]],
      predicted_course: { 'Markov Model': [['2026-09-30', 83400], ['2026-10-01', 84100]] } },
    { symbol: 'ETH', name: 'Ethereum', position: 1, quote: 'USDT', active: false, last_close: 3000, last_date: '2026-10-01', closes: [], predicted_course: {} },
  ],
  models: [
    { name: 'Markov Model', days: 12, positions: 60, exact_rate: 0.15, adjacent_rate: 0.3, direction_rate: 0.45, trades: 25, pnl_total: 0.0123, pnl_per_trade: 0.000492, first_day: '2026-09-20', last_day: '2026-10-01' },
    { name: 'Odd Model', days: 3, positions: 15, exact_rate: null, adjacent_rate: 0.2, direction_rate: 0.6, trades: 0, pnl_total: 0, pnl_per_trade: null },
  ],
  drawn_models: ['Markov Model'],
  next: { made_on: '2026-10-01', for: 'the next trading day', instruments: {
    BTC: { last_close: 84000, last_date: '2026-10-01', predictions: [
      { model: 'Markov Model', bin: 7, direction: 1, price: 84900, low: 84500, high: 85300 },
      { model: 'Odd Model', bin: 0, direction: -1, price: 80000, low: null, high: 81000 } ] } } },
  daily: [{ date: '2026-09-30', models: 2, exact_mean: 0.1, direction_mean: 0.5, best_exact: 0.2, best_model: 'Markov Model', pnl_mean: 0 },
          { date: '2026-10-01', models: 2, exact_mean: 0.2, direction_mean: 0.4, best_exact: 0.4, best_model: 'Markov Model', pnl_mean: 0.001 }],
  days: [
    { date: '2026-10-01', best: 'Markov Model', exact_mean: 0.25,
      instruments: [{ symbol: 'BTC', return: 0.004, bin: 6, edges: [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025] },
                    { symbol: 'ETH', return: -0.041, bin: 0, edges: [-0.04, -0.025, -0.012, -0.005, 0.0, 0.004, 0.011, 0.02, 0.03] }],
      models: [{ name: 'Markov Model', bins: [6, 3], exact: 1, adjacent: 1, direction: 1, positions: 2, pnl: 0.003, trades: 1 },
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
  k: 10, uniform_log_score: -2.302585, probability_floor: 0.0001, reference: 'GARCH Model', lockbox_days_withheld: 0, errors: {},
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
const btc = view.instruments[0];
ok(btc.closes.length === 3 && btc.lastClose === 84000, 'a malformed close point is dropped');
ok(btc.next.length === 2 && btc.next[0].price === 84900 && btc.next[1].low === null && btc.next[1].high === 81000,
  'the next-day predictions carry price and open interval');
ok(view.instruments[1].next.length === 0 && view.instruments[1].active === false, 'an instrument without a next-day entry has none, inactive is kept');
ok(view.drawn.length === 1 && view.daily.length === 2, 'drawn models and the daily series pass through');
ok(view.days.length === 2 && view.days[0].date === '2026-10-01' && view.days[0].instruments[1].bin === 0 && view.days[0].models[1].bins[1] === null,
  'the day records pass through, a malformed day is dropped, a missing bin stays null');
ok(markets.intervalText(0, view.days[0].instruments[1].edges) === 'below -4.0%' && markets.intervalText(9, view.days[0].instruments[0].edges) === 'above +2.5%'
  && markets.intervalText(6, view.days[0].instruments[0].edges) === '+0.3% to +0.9%' && markets.intervalText(4, []) === '-' && markets.intervalText(null, [0.1]) === '-',
  'a bin reads as its return interval, open at the ends');
ok(markets.signedPct(-0.00004) === '0.0%' && markets.signedPct(0.0123) === '+1.2%' && markets.signedPct(-0.0123, 2) === '-1.23%' && markets.signedPct(null) === '-', 'signed percentages never read -0.0%');
ok(markets.binInterval(0, [1, 2])[0] === null && markets.binInterval(2, [1, 2])[1] === null && markets.binInterval(1, [1, 2]).join(',') === '1,2', 'bin intervals');
ok(markets.gameViewLink('crypto', '2026-10-01') === '/database/crypto/2026-10-1.json' && markets.gameViewLink('shares', 'bad') === '/database/shares',
  'the game view link uses the day file name without zero padding');
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
ok(regimes.length === 2 && regimes[0].row === 'Regime HMM Model' && regimes[0].date === '2026-10-01' && regimes[0].template === 3 && regimes[0].label === 'turbulent'
  && regimes[0].expected[0].symbol === 'BTC' && regimes[0].expected[1].value === -0.002 && regimes[0].history === 2,
  'the newest reading per row, with the instruments named');
ok(markets.describeRegimes(null, []).length === 0 && markets.describeRegimes({ updated: 'x' }, []).length === 0, 'no readings, no rows');
const full = markets.describeMarket(record, { rows: rowsRecord, regimes: regimeRecord });
ok(full.rows && full.rows.rows.length === 4 && full.regimes.length === 2 && full.regimes[0].expected[1].symbol === 'ETH', 'the view carries the report and the readings');
ok(markets.describeMarket(record).rows === null && markets.describeMarket(record).regimes.length === 0, 'without extras the view has neither');
// --- formatting ---------------------------------------------------------------
ok(markets.pct(0.1234) === '12.3%' && markets.pct(null) === '-' && markets.pct(0.5, 0) === '50%', 'percentages');
ok(markets.money(0.0123) === '+0.0123' && markets.money(-0.5, 2) === '-0.50' && markets.money(undefined) === '-', 'signed money');
ok(markets.price(84000.4) === '84000' && markets.price(83.456) === '83.46' && markets.price(2.34567) === '2.3457' && markets.price(null) === '-', 'prices by magnitude');

// --- the page renders ---------------------------------------------------------
const header = (title, user) => `<html><title>${title}</title><body>${user ? user.name : ''}`;
const footer = () => '</body></html>';
const html = markets.page('crypto', view, header, footer, { name: 'ann' });
ok(html.includes('<title>Crypto predictor</title>') && html.includes('ann'), 'the page uses the shared header');
ok(html.includes('Markov Model') && html.includes('15.0%') && html.includes('+0.0123'), 'the models table shows rates and P&L');
ok(html.includes('chart-crypto-BTC') && html.includes('daily-crypto') && (html.match(/new Chart\(/g) || []).length === 3, 'one chart per instrument plus the daily chart');
ok(html.includes('84900') && html.includes('84500 - 85300') && html.includes('below 81000'), 'the next-day table shows the price, a closed and an open interval');
ok(html.includes('(inactive)'), 'an inactive instrument is marked');
ok(html.includes('How a day becomes a draw') && html.includes('Worked example - 2026-10-01') && html.includes('letter-spacing:2px;">6 0<') && html.includes('/database/crypto/2026-10-1.json')
  && html.includes('Markov Model</b>') === false && html.includes('Markov Model had played') && html.includes('1 of 2 coins in the right bin (green), 0 more in a neighbouring bin')
  && html.includes('Every day is one draw') && !html.includes('Every every'),
  'the explainer uses the newest day as a worked example with the draw and the best ticket, and opens with a readable sentence');
ok(markets.page('shares', markets.describeMarket({ ...record, market: 'shares' }), header, footer, null).includes('Every trading day is one draw'), 'the shares explainer names the trading day');
ok(html.includes('Day by day') && html.includes('the newest 2 settled day(s)') && html.includes('+0.3% to +0.9%') && html.includes('below -4.0%') && html.includes('1/2</td>')
  && (html.match(/<details/g) || []).length === 2 && html.includes('background:#2ecc71; color:white;" title="+0.3% to +0.9% - rule: long">6'),
  'the day-by-day card lists each day as returns and bins, with every model\'s ticket inside');
const twoSets = markets.page('crypto', markets.describeMarket({ ...record, days: [record.days[0], { date: '2026-09-29', best: null, exact_mean: null,
  instruments: [{ symbol: 'BTC', return: 0.01, bin: 7, edges: [] }, { symbol: 'ETH', return: 0.0, bin: 4, edges: [] }, { symbol: 'SOL', return: -0.02, bin: 1, edges: [] }],
  models: [{ name: 'Markov Model', bins: [7, 4, 2], exact: 2, adjacent: 3, direction: 2, positions: 3, pnl: 0.0, trades: 1 }] }] }), header, footer, null);
ok(twoSets.includes('<th>SOL</th>') && (twoSets.match(/<th>BTC<\/th>/g) || []).length === 2, 'each day heads its model table with its own instruments');
const noDays = markets.page('crypto', markets.describeMarket({ ...record, days: [] }), header, footer, null);
ok(noDays.includes('How a day becomes a draw') && !noDays.includes('Worked example') && !noDays.includes('card-title">Day by day') && !noDays.includes('<i>Day by day</i>'),
  'without settled days the explainer is generic and there is no day-by-day card');
const fullHtml = markets.page('crypto', full, header, footer, null);
ok(fullHtml.includes('Rows under a proper score') && fullHtml.includes('No row carries information beyond GARCH') && fullHtml.includes('(reference)')
  && fullHtml.includes('no probabilities') && fullHtml.includes('-0.240 [-0.400, -0.100] worse') && fullHtml.includes('GARCH itself is above the uniform forecast'),
  'the score card names the reference, the verdicts and the intervals');
ok(fullHtml.includes('Regime reading') && fullHtml.includes('T3') && fullHtml.includes('turbulent') && fullHtml.includes('2 of 2 by volatility') && fullHtml.includes('ETH -0.20%'),
  'the regime card shows the template, its label and the expected returns');
const betterRows = { ...rowsRecord, rows: rowsRecord.rows.map((r) => (r.name === 'Markov Model' ? { ...r, vs_reference: interval(0.1, 0.02, 0.2, 'better') } : r)) };
ok(markets.page('crypto', markets.describeMarket(record, { rows: betterRows }), header, footer, null).includes('Markov Model</b> carries information beyond GARCH'),
  'a better row is named in the headline');
ok(!html.includes('Rows under a proper score') && !html.includes('Regime reading'), 'without a report or a log there is neither card');
const empty = markets.page('shares', null, header, footer, null);
ok(empty.includes('No market record yet') && empty.includes('data/markets/shares.json'), 'without a record the page says what will fill it');
const noModels = markets.page('shares', markets.describeMarket({ market: 'shares', instruments: [], models: [], chance: { exact: 0.1, adjacent: 0.28, direction: 0.5 } }), header, footer, null);
ok(noModels.includes('no settled day yet'), 'without settled days the models table says so');
const routes = {};
markets.install({ get: (p, h) => { routes[p] = h; } }, { header, footer, dataDir: dir, controlsDir: controls });
ok(Object.keys(routes).sort().join(',') === '/markets/crypto,/markets/shares', 'two routes are installed');
let sent = '';
routes['/markets/crypto']({ user: null }, { send: (h) => { sent = h; } });
ok(sent.includes('Crypto predictor') && sent.includes('chart-crypto-BTC') && sent.includes('Rows under a proper score') && sent.includes('Regime reading'),
  'the route renders the record, the report and the readings on disk');

fs.rmSync(dir, { recursive: true, force: true });
fs.rmSync(controls, { recursive: true, force: true });
console.log(`markets.js: ${passed} checks passed`);
