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
    { name: 'Markov Model', days: 12, positions: 60, exact_rate: 0.15, adjacent_rate: 0.3, direction_rate: 0.45, trades: 25, pnl_total: 0.0123, pnl_per_trade: 0.000492, first_day: '2026-09-20', last_day: '2026-10-01' },
    { name: 'Odd Model', days: 3, positions: 15, exact_rate: null, adjacent_rate: 0.2, direction_rate: 0.6, trades: 0, pnl_total: 0, pnl_per_trade: null },
  ],
  drawn_models: ['Markov Model'], best_model: 'Markov Model',
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
ok(html.includes('Markov Model') && html.includes('15.0%') && html.includes('+0.0123'), 'the models table shows rates and P&L');
ok(html.includes("marketRender('chart-crypto-BTC')") && html.includes("marketRender('chart-crypto-ETH')") && html.includes('daily-crypto')
  && (html.match(/new Chart\(/g) || []).length === 2, 'one chart per instrument through the client, plus the daily chart');
ok(html.includes('chartjs-plugin-zoom') && html.includes('hammer.min.js') && html.includes('Chart.register(window.ChartZoom)') && html.includes("wheel: { enabled: true }")
  && html.includes(`onclick="marketRange('chart-crypto-BTC', 30)">1M<`) && html.includes(`onclick="marketReset('chart-crypto-BTC')"`) && html.includes('height:420px')
  && html.includes("marketRender('chart-crypto-BTC')") && html.includes('.container { max-width: 1400px; }'),
  'the instrument charts zoom and pan, have range buttons and a reset, are taller, and the page is wider');
ok(html.includes(`data-view-for="chart-crypto-BTC" data-view="price"`) && html.includes(`data-view="moves"`)
  && html.includes(`data-chart="chart-crypto-BTC" data-model="Markov Model" checked`) && html.includes(`data-chart="chart-crypto-BTC" data-model="Odd Model" onchange`)
  && !html.includes(`data-model="Odd Model" checked`) && html.includes('(best over the scored days)') && html.includes(`marketModels('chart-crypto-BTC', 'all')`),
  'two views to toggle, a switch per model with only the best one on, and best/all/none links');
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
const priceModel = w.marketChartModel('t', 'price', ['Markov Model']);
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
  && cfg.options.scales.x.min === undefined && typeof cfg.options.plugins.tooltip.filter === 'function' && typeof cfg.options.plugins.legend.labels.generateLabels === 'function',
  'rendering with no model on draws the close alone; the price axis does not start at zero; four labels open unranged; tooltip and legend helpers are wired');
const tip = cfg.options.plugins.tooltip;
ok(tip.filter({ raw: null, dataset: {} }) === false && tip.filter({ raw: [1, 2], dataset: { legendHidden: true } }) === false && tip.filter({ raw: 5, dataset: {} }) === true
  && tip.callbacks.label({ raw: [83663.4, 84168.9], dataset: { label: 'M' } }) === 'M: 83663 to 84169' && tip.callbacks.label({ raw: 84000.4, dataset: { label: 'BTC close' } }) === 'BTC close: 84000',
  'tooltips skip empty rows and hidden bands and print floating bars as a readable range');
w.marketState.t = { view: 'moves' }; w.marketRender('t');
const tipPct = sandbox.lastConfig.options.plugins.tooltip;
ok(tipPct.callbacks.label({ raw: [0.3, 0.8999999], dataset: { label: 'M' } }) === 'M: +0.30% to +0.90%' && sandbox.lastConfig.options.scales.y.beginAtZero === undefined,
  'in the moves view tooltips read in percent and zero stays on the axis');
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
ok((html.match(/the day fell here/g) || []).length === 2 && html.includes('Markov Model predicted this') && html.includes('the green outline is the bin that Markov Model')
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
console.log(`markets.js: ${passed} checks passed`);
