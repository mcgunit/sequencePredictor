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
ok(html.includes('Markov Model') && html.includes('15.0%') && html.includes('+12.34') && html.includes('>56%<') && html.includes('P&amp;L USDT') && !html.includes('+0.0123'),
  'the models table shows rates, the win rate and the money, not the fraction');
ok(html.includes('P&amp;L holding') && html.includes('>+15.50</td>') && html.includes('9 position(s), win rate 67%'), 'the models table carries the hold total with its details on hover');
ok(html.includes(`data-view-for="ledger-crypto" data-view="daily"`) && html.includes(`data-view="hold"`) && html.includes('holding: +15.50 (Markov Model), buy-and-hold +2.10'), 'the book card toggles the two rules and names both bests');
ok(html.includes('card-title">Paper trading') && html.includes('best book +12.34 USDT (Markov Model), the market +1.80') && html.includes("marketRender('ledger-crypto')")
  && html.includes(`data-chart="ledger-crypto" data-model="Markov Model" checked`) && html.includes(`data-chart="ledger-crypto" data-model="Odd Model" onchange`)
  && html.includes('100 USDT</b> as bought at the previous close ('),
  'the paper-trading card names the rule, the best book against the market, and switches every model with the best on');
const bookData = JSON.parse(html.match(/window\.marketData\['ledger-crypto'\] = (\{.*?\}); marketRender/s)[1]);
ok(bookData.kind === 'ledger' && bookData.labels.join(',') === '2026-09-30,2026-10-01' && bookData.rules.daily.benchmark.join(',') === '-1.2,1.8' && bookData.rules.daily.series['Markov Model'].join(',') === '5.5,12.34'
  && bookData.rules.daily.series['Odd Model'][0] === null && bookData.rules.hold.series['Markov Model'].join(',') === '5.6,15.5' && bookData.rules.hold.benchmark.join(',') === '-1.1,2.1'
  && bookData.rules.hold.series['Odd Model'].join(',') === ',' && bookData.best === 'Markov Model',
  'the book data is aligned to the settled days with the running totals of both rules');
ok(!markets.page('crypto', markets.describeMarket({ ...record, trading: undefined }), header, footer, null).includes('card-title">Paper trading'), 'without a book there is no paper-trading card');
ok(html.includes("marketRender('chart-crypto-BTC')") && html.includes("marketRender('chart-crypto-ETH')") && html.includes('daily-crypto')
  && (html.match(/new Chart\(/g) || []).length === 2, 'one chart per instrument through the client, plus the daily chart');
ok(html.includes('chartjs-plugin-zoom') && html.includes('hammer.min.js') && html.includes('Chart.register(window.ChartZoom)') && html.includes("wheel: { enabled: true }")
  && html.includes(`onclick="marketRange('chart-crypto-BTC', 30)">1M<`) && html.includes(`onclick="marketReset('chart-crypto-BTC')"`) && html.includes('height:420px')
  && html.includes("marketRender('chart-crypto-BTC')") && html.includes('.container { max-width: 1400px; }'),
  'the instrument charts zoom and pan, have range buttons and a reset, are taller, and the page is wider');
ok(html.includes(`data-view-for="chart-crypto-BTC" data-view="lines"`) && html.includes(`data-view="bars"`) && html.includes(`data-view="moves"`)
  && html.includes(`data-chart="chart-crypto-BTC" data-model="Markov Model" checked`) && html.includes(`data-chart="chart-crypto-BTC" data-model="Odd Model" onchange`)
  && !html.includes(`data-model="Odd Model" checked`) && html.includes('(best over the scored days)') && html.includes(`marketModels('chart-crypto-BTC', 'all')`),
  'three views to toggle, a switch per model with only the best one on, and best/all/none links');
const dataMatch = html.match(/window\.marketData\['chart-crypto-BTC'\] = (\{.*?\}); marketRender/s);
const embedded = JSON.parse(dataMatch[1]);
ok(embedded.labels.join(',') === '2026-09-29,2026-09-30,2026-10-01,next' && embedded.closes[3] === null && embedded.course['Markov Model'].join(',') === ',4,6,'
  && embedded.next['Markov Model'].price === 84900 && embedded.best === 'Markov Model' && embedded.colours['Markov Model'] === markets.COLOURS[0] && embedded.lastClose === 84000,
  'the embedded chart data is aligned to the labels with a trailing next slot standing on the newest game day\'s close');
const many = markets.describeMarket({ ...record, models: Array.from({ length: 20 }, (_, i) => ({ name: `M${i}`, days: 1, exact_rate: 0.1 })), drawn_models: ['M0'], best_model: 'M0' });
const manyHtml = markets.page('crypto', many, header, footer, null);
const manyData = JSON.parse(manyHtml.match(/window\.marketData\['chart-crypto-BTC'\] = (\{.*?\}); marketRender/s)[1]);
ok(new Set(Object.values(manyData.colours)).size === Object.keys(manyData.colours).length && Object.values(manyData.colours).every((c) => /^#[0-9a-f]{6}$/.test(c)),
  'more models than the palette still get distinct hex colours');
ok((markets.CHART_CLIENT_JS.match(/type: 'bar'/g) || []).length === 1 && markets.CHART_CLIENT_JS.includes("type: 'bar', grouped: false"),
  'every bar dataset goes through the one helper that centres bars on their date');

// --- the chart client runs in Node against a stubbed Chart ---------------------
const vm = require('vm');
const sandbox = { window: {}, document: { querySelectorAll: () => [], getElementById: () => ({ getContext: () => ({}) }) }, Math, Array, Object, parseInt, isFinite, console };
sandbox.Chart = function (ctx, config) { this.config = config; this.data = config.data; this.options = config.options; sandbox.lastConfig = config; };
sandbox.Chart.defaults = { plugins: { legend: { labels: { generateLabels: (chart) => chart.data.datasets.map((ds, i) => ({ text: ds.label, datasetIndex: i, fillStyle: 'x' })) }, onClick: () => {} } } };
vm.createContext(sandbox);
vm.runInContext(markets.CHART_CLIENT_JS, sandbox);
const w = Object.assign(sandbox, { marketData: sandbox.window.marketData, marketState: sandbox.window.marketState });   // functions are globals of the context; data and state live on window
const edgesFx = [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025];
ok(Math.abs(w.marketRepresentative(6, edgesFx) - 0.006) < 1e-12 && Math.abs(w.marketRepresentative(9, edgesFx) - 0.03) < 1e-12
  && Math.abs(w.marketRepresentative(0, edgesFx) + 0.035) < 1e-12 && w.marketInterval(0, edgesFx)[0] === null && w.marketInterval(9, edgesFx)[1] === null,
  'the client turns a bin into a return exactly as MarketGame.representative_return does, open bins included');
w.marketData.t = embedded;
const linesModel = w.marketChartModel('t', 'lines', ['Markov Model']);
ok(linesModel.datasets.length === 2 && linesModel.datasets[1].type === 'line' && linesModel.datasets[1].borderDash.join(',') === '3,3' && linesModel.datasets[1].model === 'Markov Model'
  && linesModel.datasets[1].data[0] === null && Math.abs(linesModel.datasets[1].data[2] - 84000 * Math.exp(-0.004) * Math.exp(0.006)) < 1e-6 && Math.abs(linesModel.datasets[1].data[3] - 84900) < 1e-9,
  'lines view: the dashed predicted course through the prices the bins stood for, the next-day price last');
const priceModel = w.marketChartModel('t', 'bars', ['Markov Model']);
ok(priceModel.datasets.length === 3 && priceModel.datasets[0].type === 'line' && priceModel.datasets[1].legendHidden === true && priceModel.datasets[2].label === 'Markov Model'
  && priceModel.datasets[1].grouped === false && priceModel.datasets[2].grouped === false && priceModel.datasets[2].model === 'Markov Model' && priceModel.datasets[2].legendColour === markets.COLOURS[0]
  && priceModel.datasets[2].data[0] === null && Math.abs(priceModel.datasets[2].data[2][0] - 84000 * Math.exp(-0.004)) < 1e-6
  && Math.abs(priceModel.datasets[2].data[2][1] - 84000 * Math.exp(-0.004) * Math.exp(0.006)) < 1e-6 && priceModel.datasets[2].backgroundColor[2] === '#27ae60'
  && Math.abs(priceModel.datasets[2].data[3][0] - 84000) < 1e-9 && Math.abs(priceModel.datasets[2].data[3][1] - 84900) < 1e-9
  && Math.abs(priceModel.datasets[1].data[3][0] - 84500) < 1e-9 && Math.abs(priceModel.datasets[1].data[3][1] - 85300) < 1e-9,
  'price view: centred bars from the previous game day\'s close to the bin\'s price, green when up, the band over the bin\'s interval, the next slot on the newest game day\'s close');
const movesModel = w.marketChartModel('t', 'moves', ['Markov Model', 'Odd Model']);
ok(movesModel.datasets.length === 5 && movesModel.datasets[0].label === 'real move' && Math.abs(movesModel.datasets[0].data[2] - 0.4) < 1e-9 && movesModel.datasets[0].data[3] === null
  && movesModel.datasets[0].backgroundColor === 'rgba(44,62,80,0.75)' && movesModel.datasets[2].type === 'line' && movesModel.datasets[2].showLine === false && Math.abs(movesModel.datasets[2].data[2] - 0.6) < 1e-9
  && Math.abs(movesModel.datasets[1].data[2][0] - 0.3) < 1e-9 && Math.abs(movesModel.datasets[1].data[2][1] - 0.9) < 1e-9
  && movesModel.datasets[4].data[2] !== null && movesModel.datasets[4].data[1] === null && movesModel.percent === true,
  'moves view: the real move as one dark bar in percent, each model\'s band as a floating bar and its middle as a dot');
// Odd Model played bin 9, an open-ended bin: its band runs from the edge to the top of what the chart shows, beyond the bin's middle
const openBand = movesModel.datasets[3].data[2];
ok(Math.abs(openBand[0] - 2.5) < 1e-9 && openBand[1] > 3.0 && Math.abs(movesModel.datasets[4].data[2] - 3.0) < 1e-9, 'an open-ended bin\'s band reaches the edge of the chart, not a half-width stub');
w.marketRender('t');
const cfg = sandbox.lastConfig;
ok(cfg && cfg.data.datasets.length === 1 && cfg.options.scales.y.title.text === 'USDT' && cfg.options.scales.y.beginAtZero === false && cfg.options.plugins.zoom === undefined
  && cfg.options.scales.x.min === undefined && typeof cfg.options.plugins.tooltip.filter === 'function' && typeof cfg.options.plugins.legend.labels.generateLabels === 'function'
  && w.marketState.t.view === 'lines',
  'rendering with no model on draws the close alone in the default lines view; the price axis does not start at zero; four labels open unranged; tooltip and legend helpers are wired');
const tip = cfg.options.plugins.tooltip;
ok(tip.filter({ raw: null, dataset: {} }) === false && tip.filter({ raw: [1, 2], dataset: { legendHidden: true } }) === false && tip.filter({ raw: 5, dataset: {} }) === true
  && tip.callbacks.label({ raw: [83663.4, 84168.9], dataset: { label: 'M' } }) === 'M: 83663 to 84169' && tip.callbacks.label({ raw: 84000.4, dataset: { label: 'BTC close' } }) === 'BTC close: 84000',
  'tooltips skip empty rows and hidden bands and print floating bars as a readable range');
w.marketState.t = { view: 'moves' }; w.marketRender('t');
const tipPct = sandbox.lastConfig.options.plugins.tooltip;
ok(tipPct.callbacks.label({ raw: [0.3, 0.8999999], dataset: { label: 'M' } }) === 'M: +0.30% to +0.90%' && sandbox.lastConfig.options.scales.y.beginAtZero === undefined,
  'in the moves view tooltips read in percent and zero stays on the axis');
w.marketData.book = bookData;
const bookModel = w.marketChartModel('book', 'daily', ['Markov Model']);
ok(bookModel.datasets.length === 2 && bookModel.datasets[0].label === 'market, bought every day' && bookModel.datasets[0].data.join(',') === '-1.2,1.8'
  && bookModel.datasets[1].model === 'Markov Model' && bookModel.datasets[1].data.join(',') === '5.5,12.34' && bookModel.money === true && bookModel.yTitle.includes('USDT'),
  'the book chart: the market dashed, each switched-on model\'s running total as a line');
const holdModel = w.marketChartModel('book', 'hold', ['Markov Model']);
ok(holdModel.datasets[0].label === 'market, buy and hold' && holdModel.datasets[0].data.join(',') === '-1.1,2.1' && holdModel.datasets[1].data.join(',') === '5.6,15.5' && holdModel.yTitle.includes('held while up')
  && w.marketChartModel('book', 'lines', ['Markov Model']).datasets[1].data.join(',') === '5.5,12.34',
  'the hold rule draws its own book and benchmark; an unknown rule falls back to the daily one');
w.marketRender('book');
ok(w.marketState.book.view === 'daily', 'a book opens on the daily rule');
ok(sandbox.lastConfig.options.plugins.tooltip.callbacks.label({ raw: 12.345, dataset: { label: 'M' } }) === 'M: +12.35' && sandbox.lastConfig.options.scales.y.grid !== undefined,
  'book tooltips read as signed money and the zero line is drawn');
w.marketData.long = Object.assign({}, embedded, { labels: Array.from({ length: 40 }, (_, i) => `d${i}`), closes: Array.from({ length: 40 }, () => 1), edges: Array.from({ length: 40 }, () => null),
  moves: Array.from({ length: 40 }, () => null), course: {} });
w.marketRender('long');
ok(sandbox.lastConfig.options.scales.x.min === 'd9' && sandbox.lastConfig.options.scales.x.max === 'd39', 'a longer history opens on its newest month');
ok(html.includes('84900') && html.includes('84500 - 85300') && html.includes('below 81000'), 'the next-day table shows the price, a closed and an open interval');
ok(html.includes('(inactive)'), 'an inactive instrument is marked');
ok(html.includes('How a day becomes a draw') && html.includes('1. Two closes, one number') && html.includes('a candle is four prices')
  && html.includes("BTC's previous close was 83663; on 2026-10-01 it closed at 84000, 337 higher, a move of <b>+0.40%</b>")
  && html.includes('2. The number goes into one of 10 bins') && html.includes('Each pile is one bin') && html.includes("a +2% day lands in BTC's bin 8 but only in ETH's bin 7")
  && html.includes('Worked example - 2026-10-01, the newest settled day</b> (settled:') && html.includes('letter-spacing:2px;">6 0<') && html.includes('/database/crypto/2026-10-1.json')
  && html.includes('Markov Model had played') && html.includes('1 of 2 coins in the right bin (green), 0 more in a neighbouring bin')
  && html.includes('Every day is one draw') && !html.includes('Every every') && html.includes('a 7 played on BTC is judged against BTC\'s own bin, and ETH landing in 7')
  && html.includes("bin 6 on 2026-10-01 ran from +0.30% to +0.90%, which from the close of 83663 means a close\n      between about 83914 and 84419"),
  'the explainer walks through closes, bins and draw with the newest day, the best ticket and a price range');
ok((html.match(/the day fell here/g) || []).length === 2 && html.includes('Markov Model predicted this') && html.includes('<b>Green outline</b>: what Markov Model')
  && html.includes('transform:translateX(50%)') && html.includes('>+0.3%</div>') && html.includes("BTC's bin 4 runs from -0.4% to 0.0%, while bin 0 is everything below -3.0%")
  && html.includes('an inner bin is within one by luck 30% of the time, an end bin 20%') && html.includes('log return'),
  'each instrument gets a strip of ten bins with its edges, the real move dark and the predicted bin outlined, and the caveats are stated');
ok(markets.binOfMove(Math.log(1.02), [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025]) === 8 && markets.binOfMove(-1, [0]) === 0 && markets.binOfMove(1, [0]) === 1
  && markets.intervalText(6, [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025], 2) === '+0.30% to +0.90%', 'a move finds its bin under the edges; intervals take a precision');
ok(markets.page('shares', markets.describeMarket({ ...record, market: 'shares' }), header, footer, null).includes('Every trading day is one draw'), 'the shares explainer names the trading day');
ok(html.includes('Day by day') && html.includes('the newest 2 settled day(s)') && html.includes('+0.3% to +0.9%') && html.includes('title="bin 0: below -4.00% - the day fell here"') && html.includes('1/2</td>')
  && (html.match(/<details/g) || []).length === 2 && html.includes('background:#2ecc71; color:white;" title="+0.3% to +0.9% - rule: long">6'),
  'the day-by-day card lists each day as returns and bins, with every model\'s ticket inside');
const twoSets = markets.page('crypto', markets.describeMarket({ ...record, days: [record.days[0], { date: '2026-09-29', best: null, exact_mean: null,
  instruments: [{ symbol: 'BTC', return: 0.01, bin: 7, edges: [] }, { symbol: 'ETH', return: 0.0, bin: 4, edges: [] }, { symbol: 'SOL', return: -0.02, bin: 1, edges: [] }],
  models: [{ name: 'Markov Model', bins: [7, 4, 2], exact: 2, adjacent: 3, direction: 2, positions: 3, pnl: 0.0, trades: 1 }] }] }), header, footer, null);
ok(twoSets.includes('<th>SOL</th>') && (twoSets.match(/<th>BTC<\/th>/g) || []).length === 2, 'each day heads its model table with its own instruments');
const noDays = markets.page('crypto', markets.describeMarket({ ...record, days: [] }), header, footer, null);
ok(noDays.includes('How a day becomes a draw') && !noDays.includes('Worked example -') && noDays.includes('appears here after the first settled day') && !noDays.includes('card-title">Day by day') && !noDays.includes('<i>Day by day</i>'),
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
// when a call is judged (4 Oct 2026): the day a ticket is for and its close in Belgian time
ok(markets.nextGameDay('crypto', '2026-10-02') === '2026-10-03' && markets.nextGameDay('crypto', '2026-10-31') === '2026-11-01' && markets.nextGameDay('crypto', '2026-12-31') === '2027-01-01', 'a crypto ticket is for the next calendar day');
ok(markets.nextGameDay('shares', '2026-10-02') === '2026-10-05' && markets.nextGameDay('shares', '2026-10-03') === '2026-10-05' && markets.nextGameDay('shares', '2026-10-04') === '2026-10-05' && markets.nextGameDay('shares', '2026-10-05') === '2026-10-06',
   'a shares ticket is for the next weekday - Friday, Saturday and Sunday all point at Monday');
ok(markets.nextGameDay('shares', '2026-11-25') === '2026-11-27' && markets.nextGameDay('shares', '2026-12-24') === '2026-12-28' && markets.nextGameDay('shares', '2027-12-23') === '2027-12-27'
   && markets.nextGameDay('shares', '2026-07-02') === '2026-07-06', 'New York holidays are skipped: Thanksgiving, Christmas, the observed days');
const perYear = (year) => [...markets.NYSE_CLOSED].filter((d) => d.startsWith(year)).length;
ok(perYear('2026') === 10 && perYear('2027') === 10 && markets.NYSE_CLOSED.size === 20 && [...markets.NYSE_CLOSED].every((d) => ![0, 6].includes(new Date(d).getUTCDay())), 'the closure table has ten weekdays in each announced year');
ok(markets.nextGameDay('crypto', 'soon') === null && markets.nextGameDay('crypto', '2026-02-30') === null && markets.nextGameDay('crypto', '2026-13-01') === null && markets.closeText('shares', null) === null, 'malformed and impossible dates give null, not a roll-over');
ok(markets.appearedText('2026-10-03T08:29:00+00:00') === '10:29 Belgian time on Sat 03/10' && markets.appearedText('2026-10-03T08:29:00Z') === '10:29 Belgian time on Sat 03/10', `the ticket time in Belgian time: ${markets.appearedText('2026-10-03T08:29:00+00:00')}`);
ok(markets.appearedText('2026-10-03T08:29:00') === null && markets.appearedText('2026-10-03') === null && markets.appearedText('garbage 2026') === null && markets.appearedText(null) === null, 'an offset-less, date-only or malformed time is not shown');
ok(markets.closeText('crypto', '2026-10-03') === 'the close of 2026-10-03, which is 02:00 Belgian time in the night from Saturday to Sunday (00:00 UTC on 2026-10-04)', `crypto summer close: ${markets.closeText('crypto', '2026-10-03')}`);
ok(markets.closeText('crypto', '2026-12-10') === 'the close of 2026-12-10, which is 01:00 Belgian time in the night from Thursday to Friday (00:00 UTC on 2026-12-11)', `crypto winter close: ${markets.closeText('crypto', '2026-12-10')}`);
ok(markets.closeShort('crypto', '2026-10-24').startsWith('02:00 Belgian time in the night from Saturday to Sunday') && markets.closeShort('crypto', '2026-10-25').startsWith('01:00 Belgian time in the night from Sunday to Monday')
   && markets.closeShort('crypto', '2027-03-27').startsWith('01:00 Belgian time in the night from Saturday to Sunday'), 'the Belgian clock-change nights are computed, not assumed');
ok(markets.closeText('shares', '2026-10-05') === 'the New York close of 2026-10-05, which is 22:00 Belgian time on Monday 2026-10-05 (16:00 New York time)' && markets.closeShort('shares', '2026-12-10').startsWith('22:00 Belgian time on Thursday'),
   `New York closes at 22:00 Belgian time in summer and in winter: ${markets.closeText('shares', '2026-10-05')}`);
ok(markets.closeShort('shares', '2026-10-28').startsWith('21:00 Belgian time') && markets.closeShort('shares', '2027-03-16').startsWith('21:00 Belgian time') && markets.closeShort('shares', '2026-03-09').startsWith('21:00 Belgian time'),
   `between the two clock changes New York closes at 21:00 Belgian time: ${markets.closeShort('shares', '2026-10-28')}`);
ok(markets.closeMoment('shares', '2026-10-05').toISOString() === '2026-10-05T20:00:00.000Z' && markets.closeMoment('shares', '2026-12-10').toISOString() === '2026-12-10T21:00:00.000Z'
   && markets.closeMoment('shares', '2026-03-09').toISOString() === '2026-03-09T20:00:00.000Z', 'the New York close as an instant, also in the US-only summer-time week');
ok(markets.closeText('shares', '2026-11-27') === 'the New York close of 2026-11-27, which is 19:00 Belgian time on Friday 2026-11-27 (13:00 New York time, an early close)'
   && markets.closeShort('shares', '2026-12-24').startsWith('19:00 Belgian time'), `an early close is said: ${markets.closeText('shares', '2026-11-27')}`);
ok(markets.openText('shares', '2026-10-05') === '15:30 Belgian time on Monday 2026-10-05 (09:30 New York time)' && markets.openText('shares', '2026-10-28').startsWith('14:30 Belgian time')
   && markets.openText('shares', '2026-11-27').startsWith('15:30 Belgian time') && markets.openText('crypto', '2026-10-05') === null, `the New York open is computed per date: ${markets.openText('shares', '2026-10-28')}`);
ok(markets.openMoment('crypto', '2026-10-03').toISOString() === '2026-10-03T00:00:00.000Z' && markets.hoursInto('2026-10-03T08:29:00+00:00', 'crypto', '2026-10-03') === 8.5
   && markets.hoursInto('2026-12-10T09:38:00+00:00', 'crypto', '2026-12-10') === 9.6 && markets.hoursInto('2026-10-02T23:00:00+00:00', 'crypto', '2026-10-03') === null && markets.hoursInto('bad', 'crypto', '2026-10-03') === null,
   'hours into the predicted day, from the ticket time');
ok(markets.dayStatus('crypto', '2026-10-03', new Date('2026-10-03T09:00:00Z')) === 'the UTC day 2026-10-03 is running now, 9 hours in; it closes at 02:00 Belgian time'
   && markets.dayStatus('crypto', '2026-10-03', new Date('2026-10-04T05:00:00Z')).startsWith('the UTC day 2026-10-03 has already closed (02:00 Belgian time)')
   && markets.dayStatus('shares', '2026-10-05', new Date('2026-10-03T09:00:00Z')) === 'the New York session of Monday 2026-10-05 has not opened yet - it opens at 15:30 Belgian time and closes at 22:00 Belgian time'
   && markets.dayStatus('shares', '2026-10-05', new Date('2026-10-05T15:00:00Z')).includes('is running now, 1.5 hours in')
   && markets.dayStatus('shares', '2026-10-05', new Date('2026-10-05T21:00:00Z')).includes('has already closed') && markets.dayStatus('crypto', 'bad', new Date()) === null, 'the day status follows the clock');
const at = new Date('2026-10-02T09:00:00Z');   // 11:00 Belgian time on the day the fixture's ticket is for
const timed = markets.page('crypto', view, header, footer, null, at);
ok(timed.includes('is the call for 2026-10-02 - the candle after the last one drawn, not "tomorrow"; it was made after the close of 2026-10-01 and is judged at the close of 2026-10-02, which is 02:00 Belgian time in the night from Friday to Saturday (00:00 UTC on 2026-10-03)')
   && timed.includes('The day being predicted - 2026-10-02, per model') && timed.includes('<b>When it is judged:</b> the close of 2026-10-02, which is 02:00 Belgian time'), 'the chart names the day the next call is for and when it is judged');
ok(timed.includes('Right now the UTC day 2026-10-02 is running now, 9 hours in; it closes at 02:00 Belgian time.'), 'the judged line says where the day stands as the page is read');
ok(timed.includes('sells at 02:00 Belgian time in the night from Friday to Saturday (00:00 UTC on 2026-10-03): the model is right if BTC then closes inside its interval, whatever was paid')
   && timed.includes('another price than the 84000 USDT the paper book starts from') && timed.includes('no position opened after reading this page matches it exactly'), 'the judged line says when a reader sells to be compared like the model, and that a hit is not a gain');
ok(timed.includes('Price drawn on the chart (bin middle)') && timed.includes('Right if the close lands in') && timed.includes('so an interval can start a little below the last close and still count as up'), 'the next-day table says which column decides a hit');
ok(timed.includes('the newest ticket went up at 09:15 Belgian time on Fri 02/10') && timed.includes('That ticket is for the UTC day 2026-10-02 and is judged at the close of 2026-10-02, which is 02:00 Belgian time in the night from Friday to Saturday')
   && timed.includes('the day it predicts was already 7.3 hours old'), 'the explainer names the ticket\'s time, its day and its close; the hours are computed');
ok(timed.includes('a <i>game day</i>, for crypto simply one UTC day') && timed.includes('Results settle the morning after, when the day\'s candle has closed. A crypto day is the UTC day, so'), 'the intro and step 3 put the day on a clock and define a game day');
ok(timed.includes('<b>Dark box</b>: where the day actually landed') && timed.includes('a predicted box next to the dark one is filled pale green (one off); further away it stays white') && timed.includes('green outlines mark the prediction only in these strips'), 'the strips say what their colours mean');
ok(timed.includes('the book counts\n      <b>100 USDT</b> as bought at the previous close (00:00 UTC') && timed.includes('eight to ten hours before the ticket is on this page') && !timed.includes('ten hours into'), 'the paper card puts both legs on the clock and does not overstate the hours');
ok(timed.includes('Record generated 2026-10-02T07:15:00+00:00 (09:15 Belgian time on Fri 02/10)'), 'the footer gives the export time in Belgian time too');
const sharesHtml = markets.page('shares', markets.describeMarket({ ...record, market: 'shares' }), header, footer, null, at);
ok(sharesHtml.includes('is the call for 2026-10-02 - the candle after the last one drawn, not "tomorrow"; it was made after the close of 2026-10-01 and is judged at the New York close of 2026-10-02, which is 22:00 Belgian time on Friday 2026-10-02 (16:00 New York time)')
   && sharesHtml.includes('a reader can buy from 15:30 Belgian time on Friday 2026-10-02 (09:30 New York time), when New York\'s regular session opens; pre-market trading exists before it')
   && sharesHtml.includes('the New York session of Friday 2026-10-02 has not opened yet - it opens at 15:30 Belgian time'), 'the shares page names the New York close and the computed open');
ok(sharesHtml.includes('15:30-22:00 Belgian time in most weeks (an hour earlier in the few weeks a year') && !sharesHtml.includes('21:00 in winter') && !sharesHtml.includes('at the earliest')
   && sharesHtml.includes('the previous session\'s close - the gap from that close to the next open (overnight, or a weekend for a Monday ticket)'), 'the shares explainer and fill note are right about the hours and the gap');
ok(markets.page('shares', markets.describeMarket({ ...record, market: 'shares', next: { ...record.next, made_on: '2028-01-03' } }), header, footer, null, at).includes('is the call for the next session, normally 2028-01-04'),
   'a year the exchange has not announced is said to be normal, not certain');
const badMade = markets.page('crypto', markets.describeMarket({ ...record, next: { ...record.next, made_on: 'bad', made_at: 'bad' } }), header, footer, null, at);
ok(!badMade.includes('made after the close of bad') && !badMade.includes('When it is judged') && !badMade.includes('went up at'), 'a made_on or made_at that does not parse is dropped, not echoed');
ok(!markets.page('crypto', markets.describeMarket({ ...record, next: undefined }), header, footer, null, at).includes('When it is judged'), 'without a next-day call there is no judged-at line');

console.log(`markets.js: ${passed} checks passed`);
