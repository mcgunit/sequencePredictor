'use strict';
// The old pages' access rules (21 Oct 2026, the owner's decision): a user sees the top five rows of the History
// ranking and nothing else - no "Show all", the hidden rows not even sent, the day and month figures counted over
// those rows only, no controls, no detailed cards, and the market workbench sends them to the app - while the
// administrator sees every row with the switch and every card. server.js is the entry point, so it is loaded here
// from a copy in a temp dir (its data folder is the hand-made one next to it) with express, auth, jobs, services
// and the other side-effect modules stubbed: no port is opened and no job is started. Its routes are then called
// directly as each reader.
//   node test/pages.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const REPO = path.join(__dirname, '..');
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- hand-made records: lotto ranked A..E on top, F and G below; G has the best day ------------------------
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pages-'));
const data = path.join(root, 'data');
['database/lotto', 'database/eurodreams', 'database/crypto', 'controls/results', 'markets'].forEach((d) => fs.mkdirSync(path.join(data, d), { recursive: true }));
fs.mkdirSync(path.join(root, 'config'));
const write = (rel, obj) => fs.writeFileSync(path.join(data, rel), JSON.stringify(obj));
const NAMES = ['A Model', 'B Model', 'C Model', 'D Model', 'E Model', 'F Model', 'G Model'];
const row = (name, i) => ({ name, draws: 40, avg_hits: 1.5 - i / 10, profit_bound: null, avg_special_hits: 0, best_hits: 3, profit_total: null, profit_per_bet: null, bets: 0, profit_by_size: null, multi_pick: null });
write('database/modelPerformance.json', { generatedAt: '2026-10-21T08:48:01', games: { lotto: { metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'A Model', models: NAMES.map(row) } } });
// the drawn row 1-6 (+ bonus 7): G holds all six, every top row two
const ticket = (name) => (name === 'G Model' ? [1, 2, 3, 4, 5, 6] : [1, 2, 40, 41, 42, 43]);
const day = { realResult: [1, 2, 3, 4, 5, 6, 7], currentPrediction: NAMES.map((name) => ({ name, predictions: [ticket(name)] })),
  newPrediction: NAMES.map((name) => ({ name, predictions: [[10, 11, 12, 13, 14, 15]], multiPick: [20, 21, 22] })), numberFrequency: { 10: 0.5, 44: 0.5 }, currentNumberFrequency: { 1: 1 } };
write('database/lotto/2026-10-20.json', day);
// a market game view: its notes link to the reader's market page
write('database/crypto/2026-10-20.json', { realResult: [1, 2], currentPrediction: [{ name: 'A Model', predictions: [[1, 3]] }], newPrediction: [{ name: 'A Model', predictions: [[4, 5]] }],
  numberFrequency: {}, currentNumberFrequency: {} });
// a game the report has no ranking for: a user still gets five rows, said to be the table's first five
const OTHERS = ['P Model', 'Q Model', 'R Model', 'S Model', 'T Model', 'U Model', 'V Model'];
write('database/eurodreams/2026-10-20.json', { realResult: [1, 2, 3, 4, 5, 6, 1], currentPrediction: [], numberFrequency: {},
  newPrediction: OTHERS.map((name) => ({ name, predictions: [[10, 11, 12, 13, 14, 1]] })) });
write('controls/results/lotto-synthetic.json', { report: { rows_compared: 8, row_mean: 0.78, best_row_mean: 0.95, best_row_sd_across_seeds: 0.08, selection_gain: 0.1,
  random_baseline: 0.79, per_row_mean: {}, game: 'lotto', mode: 'synthetic', seeds: [1, 2, 3], days: 120, metric: 'hits per draw', winner_per_seed: [], generated_at: '2026-10-18T10:11:55' }, per_seed: [] });

// --- server.js from a copy, its side-effect modules stubbed ---------------------------------------------------
fs.copyFileSync(path.join(REPO, 'server.js'), path.join(root, 'server.js'));
const gets = [];
const noop = () => {};
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const stubs = {
  express: Object.assign(() => ({ use: noop, set: noop, post: noop, get(p, ...h) { gets.push([p, h]); }, listen: () => ({ on: noop, close: noop }) }),
    { urlencoded: () => noop, static: () => noop, json: () => noop }),
  './config': { CONFIG_DIR: path.join(root, 'config'), INTERFACE: '127.0.0.1', PORT: 0, TRUST_PROXY: false, ADMIN_USER: null },
  './auth': { middleware: (q, s, n) => n(), install: noop, misconfigured: () => false, enabled: () => true, escapeHtml: esc, requireAdmin: () => (q, s, n) => n(), csrfOk: () => false },
  './council': { install: noop },
  './services': { install: noop, startAll: noop, stopAll: noop, section: () => '' },
  './jobs': { install: noop, section: () => '', start: noop, stop: noop, JOBS: [] },
  './whatsnew': { install: noop, dialog: () => '', navLink: () => '' },
};
const original = Module.prototype.require;
Module.prototype.require = function stubbed(id) {
  if (Object.prototype.hasOwnProperty.call(stubs, id)) return stubs[id];
  // the copy's own modules are the checkout's
  if (id.startsWith('./') && this.filename === path.join(root, 'server.js')) return original.call(this, path.join(REPO, id));
  return original.apply(this, arguments);
};
const listeners = ['SIGTERM', 'SIGINT', 'exit', 'uncaughtException', 'unhandledRejection'].map((e) => [e, process.listeners(e).slice()]);
try {
  require(path.join(root, 'server.js'));
} finally {
  Module.prototype.require = original;
  // drop the handlers the server registers for itself: this process is a test
  listeners.forEach(([e, before]) => process.listeners(e).filter((l) => !before.includes(l)).forEach((l) => process.removeListener(e, l)));
}

const READERS = { admin: { name: 'admin', role: 'admin', open: false, csrf: 'x' }, user: { name: 'reader', role: 'user', open: false, csrf: 'x' } };
function render(route, params, who) {
  const layer = gets.find(([p]) => (Array.isArray(p) ? p.includes(route) : p === route));
  assert.ok(layer, `no route ${route}`);
  const out = { html: '', redirect: null, status: 200 };
  const res = { send: (x) => { out.html = x; }, status(c) { out.status = c; return this; }, redirect: (a, b) => { out.redirect = b === undefined ? a : b; }, set() { return this; }, json: noop, sendFile: noop };
  layer[1][layer[1].length - 1]({ user: READERS[who], params, query: {}, session: {}, originalUrl: route, get: () => '' }, res);
  return out;
}
const rowsOf = (html, name) => (html.match(new RegExp(`>${name}</td>`, 'g')) || []).length;
const TOP = NAMES.slice(0, 5);

// --- a user ------------------------------------------------------------------------------------------------
// The old lottery History pages send a past draw's every row and its file's vote, so they are the administrator's
// (23 Oct 2026, the owner: no row outside the five ranked before a draw is ever sent to a user): a user is sent to the
// same draw, game or list in the app (lottery.js), and nothing of the draw is sent.
const uDay = render('/database/:folder/:file', { folder: 'lotto', file: '2026-10-20.json' }, 'user');
ok(uDay.redirect === '/app/lottery/lotto/2026-10-20' && uDay.html === '' && NAMES.every((n) => !uDay.html.includes(n)) && !uDay.html.includes('currentNumberFrequency'),
  `a user's old day view: sent to the draw in the app, no row and no vote sent (${uDay.redirect})`);
ok(render('/database/:folder/:file', { folder: 'eurodreams', file: '2026-10-20.json' }, 'user').redirect === '/app/lottery/eurodreams/2026-10-20',
  'a user\'s old day view of another game: the same draw in the app');
const uFolder = render('/database/:folder', { folder: 'lotto' }, 'user');
ok(uFolder.redirect === '/app/lottery/lotto' && uFolder.html === '', `a user's old month view: sent to the game's counted draws in the app (${uFolder.redirect})`);
const uIndex = render('/database', {}, 'user');
ok(uIndex.redirect === '/app/lottery' && uIndex.html === '', `a user's old History: sent to the app's Lottery (${uIndex.redirect})`);
ok(render('/database/:folder/:file', { folder: 'lotto', file: '2026-10-21.json' }, 'user').status === 404 && render('/database/:folder', { folder: 'bingo' }, 'user').status === 404,
  'a user: a draw or a folder that is not there is still a 404');
const uLottery = render('/lottery', {}, 'user').html;
ok(TOP.every((n) => rowsOf(uLottery, n) === 1) && rowsOf(uLottery, 'G Model') === 0 && !uLottery.includes('Show all 7 models') && uLottery.includes('top 5 of 7 model rows'),
  'a user\'s new predictions: the top five rows, no switch');
ok(!uLottery.includes('chart-lotto') && uLottery.includes('the top 5 models\' tickets for the next draw'), 'a user\'s new predictions: no chart of every row\'s vote, help in the user\'s terms');
const uHome = render('/', {}, 'user').html;
ok(uHome.includes('<b>lotto</b>') && uHome.includes('top 5 of 7 model rows') && !uHome.includes('every model\'s ticket'), 'a user\'s home: the top five rows of each game, as /lottery counts them');
const uMarket = render('/database/:folder', { folder: 'crypto' }, 'user').html + render('/database/:folder/:file', { folder: 'crypto', file: '2026-10-20.json' }, 'user').html;
ok(!uMarket.includes('href="/markets/') && uMarket.includes('href="/app/crypto"') && !uMarket.includes('Week ahead') && !uMarket.includes('three steps'),
  'a user\'s market game view links to the app page and promises none of the workbench\'s cards');
ok(uLottery.includes('href="/app/crypto"') && !uLottery.includes('href="/markets/crypto"'), 'a user\'s navbar points Crypto at the app');
ok(uLottery.includes('The first 5 rows of this table: no row of it is in the History ranking yet') && OTHERS.slice(0, 5).every((n) => rowsOf(uLottery, n) === 1) && rowsOf(uLottery, 'V Model') === 0,
  'a game without a ranking: the first five rows, said as such, never every row');
ok(render('/markets/crypto', {}, 'user').redirect === '/app/crypto', 'a user is sent from the market workbench to the app');

// --- the administrator ---------------------------------------------------------------------------------------
const aDay = render('/database/:folder/:file', { folder: 'lotto', file: '2026-10-20.json' }, 'admin').html;
ok(NAMES.every((n) => rowsOf(aDay, n) === 2) && aDay.includes('Show all 7 models'), 'the administrator\'s day view: every row, with the switch');
ok((aDay.match(/a user sees the top 5 of the/g) || []).length === 1 && (aDay.match(/applied to a past draw \(hindsight\); a user sees every row of a past draw/g) || []).length === 1
  && aDay.includes('chart-detail') && aDay.includes('History</a> page tracks whether'),
  'the administrator\'s day view: the coming draw\'s switch says what a user gets, the past draw\'s says it is hindsight');
ok(render('/database/:folder', { folder: 'crypto' }, 'admin').html.includes('href="/markets/crypto"'), 'the administrator\'s market game view keeps the workbench link');
ok(render('/', {}, 'admin').html.includes('7 model rows') && render('/', {}, 'admin').html.includes('every model\'s ticket'), 'the administrator\'s home counts every row');
const aFolder = render('/database/:folder', { folder: 'lotto' }, 'admin').html;
ok(aFolder.includes('Match: 6') && !aFolder.includes('top 5 models ·'), 'the administrator\'s figures count every row');
const aIndex = render('/database', {}, 'admin').html;
ok(aIndex.includes('Best model per game') && aIndex.includes('Null band'), 'the administrator\'s History keeps the ranking card and the band');
ok(render('/lottery', {}, 'admin').html.includes('href="/markets/crypto"'), 'the administrator\'s navbar keeps the workbench');

fs.rmSync(root, { recursive: true, force: true });
console.log(`pages: ${checks} checks passed`);
