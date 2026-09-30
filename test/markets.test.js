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
};
fs.writeFileSync(path.join(dir, 'crypto.json'), JSON.stringify(record));
fs.writeFileSync(path.join(dir, 'shares.json'), '{');

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
const bare = markets.describeMarket({ market: 'shares', instruments: [], models: [] });
ok(bare.scoredDays === 0 && bare.chance.exact === 0.1 && bare.madeOn === null && bare.daily.length === 0, 'an empty record describes with defaults');

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
const empty = markets.page('shares', null, header, footer, null);
ok(empty.includes('No market record yet') && empty.includes('data/markets/shares.json'), 'without a record the page says what will fill it');
const noModels = markets.page('shares', markets.describeMarket({ market: 'shares', instruments: [], models: [], chance: { exact: 0.1, adjacent: 0.28, direction: 0.5 } }), header, footer, null);
ok(noModels.includes('no settled day yet'), 'without settled days the models table says so');
const routes = {};
markets.install({ get: (p, h) => { routes[p] = h; } }, { header, footer, dataDir: dir });
ok(Object.keys(routes).sort().join(',') === '/markets/crypto,/markets/shares', 'two routes are installed');
let sent = '';
routes['/markets/crypto']({ user: null }, { send: (h) => { sent = h; } });
ok(sent.includes('Crypto predictor') && sent.includes('chart-crypto-BTC'), 'the route renders the record on disk');

fs.rmSync(dir, { recursive: true, force: true });
console.log(`markets.js: ${passed} checks passed`);
