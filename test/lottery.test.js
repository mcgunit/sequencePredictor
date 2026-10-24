'use strict';
// The users' Lottery and Home (lottery.js) and the Council's per-reader routes (council.js), batch 5 (23 Oct 2026). The
// owner's rules: a user gets the top five rows of each game's History ranking - for the coming draw today's five, for a
// past draw the five ranked first BEFORE it (rankhistory.js), never today's five applied to a past draw -, the draws
// before the game's first saved ranking are left out, every figure is over those five only, and no other row is ever
// sent. Here, on a temp git repository whose ranking changed: the point-in-time five where today's differ and give
// another figure, the left-out draws, the money and hits over the five only, the same view for the administrator, the
// whitelist and the strict date, the last check that turns a row outside the five into a 500; Home's market headline
// against api.js; the Council's endpoints for the administrator only and a run answered to its asker only (a stand-in
// for the Council API on a free port - the real one is never called). When this checkout runs next to production:
// production's draws read through the API on a copy of its files and a scratch root whose .git is production's
// (read-only: git log and show; the cache file in the temp dir), the five and the figures of a few draws against an
// independent computation - git read directly, the money by the pipeline's own scorer (src/Helpers.py).
//   node test/lottery.test.js
const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const Module = require('module');
const { execFileSync } = require('child_process');
const api = require('../api');
const draws = require('../draws');
const gates = require('../gates');
const lottery = require('../lottery');
const markets = require('../markets');
const rankhistory = require('../rankhistory');
const workbench = require('../workbench');

const REPO = path.join(__dirname, '..');
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lottery-'));
const cleanUp = () => fs.rmSync(tmp, { recursive: true, force: true });

// --- a stand-in for express: app.use(prefix, fn), app.get / app.post(pattern, ...fns), in order -------------------------
function fakeApp() {
  const stack = [];
  const add = (method) => (pattern, ...fns) => { stack.push({ method, pattern, fns }); };
  return { stack, use(prefix, fn) { stack.push({ prefix, fns: [fn] }); }, get: add('GET'), post: add('POST') };
}
function matchRoute(pattern, pathname) {
  const p = pattern.split('/'); const a = pathname.split('/');
  if (p.length !== a.length) return null;
  const params = {};
  for (let k = 0; k < p.length; k += 1) {
    if (p[k].startsWith(':')) params[p[k].slice(1)] = decodeURIComponent(a[k]);
    else if (p[k] !== a[k]) return null;
  }
  return params;
}
function request(app, url, user, { method = 'GET', body = undefined } = {}) {
  return new Promise((resolve) => {
    const [pathname, qs] = url.split('?');
    const res = { statusCode: 200, headers: {}, body: undefined,
      status(c) { this.statusCode = c; return this; }, set(k, v) { this.headers[k] = v; return this; },
      json(b) { this.body = b; resolve(this); return this; }, send(b) { this.body = b; resolve(this); return this; } };
    // _body: express.json leaves a body it is told was parsed alone (the stand-in has no stream)
    const req = { method, user, body, _body: true, headers: {}, query: Object.fromEntries(new URLSearchParams(qs || '')), params: {}, path: pathname };
    let i = 0;
    const next = () => {
      while (i < app.stack.length) {
        const layer = app.stack[i++];
        if (layer.prefix !== undefined) {
          if (pathname === layer.prefix || pathname.startsWith(`${layer.prefix}/`)) return layer.fns[0](req, res, next);
        } else if (layer.method === method) {
          const params = matchRoute(layer.pattern, pathname);
          if (params) {
            req.params = params;
            let j = 0;
            const step = () => (j < layer.fns.length ? layer.fns[j++](req, res, step) : next());
            return step();
          }
        }
      }
      res.statusCode = 404; res.body = 'fell through'; resolve(res);
      return undefined;
    };
    next();
  });
}
const ADMIN = { name: 'admin', role: 'admin', open: false, csrf: 'tok-admin' };
const USER = { name: 'reader', role: 'user', open: false, csrf: 'tok-reader' };
const OPEN = { name: 'open access', role: 'admin', open: true, csrf: 'tok-open' };
// the stand-in Council API once it listens: closed on a failure too, or the process would wait on it
let liveStandIn = null;
const withoutBuiltAt = (body) => JSON.stringify(body, (k, v) => (k === 'builtAt' ? undefined : v));
const mentions = (body, name) => JSON.stringify(body).includes(JSON.stringify(name));

// ===================================================================================================================
// 1. a temp repository: the History ranking committed three times, today's in the working tree
// ===================================================================================================================
const repo = path.join(tmp, 'repo');
const db = path.join(repo, 'data', 'database');
['lotto', 'keno', 'jokerplus'].forEach((g) => fs.mkdirSync(path.join(db, g), { recursive: true }));
const gitIn = (args, when) => String(execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', ...args],
  { cwd: repo, stdio: 'pipe', env: { PATH: process.env.PATH, HOME: process.env.HOME || '/root', GIT_AUTHOR_DATE: when, GIT_COMMITTER_DATE: when } }));
const REPORT = path.join(db, 'modelPerformance.json');
// Z is ranked last everywhere: it is never one of anybody's five, and its name must never be sent
const NAMES = ['A Model', 'B Model', 'C Model', 'D Model', 'E Model', 'F Model', 'G Model', 'Z Model'];
const V1 = ['A Model', 'B Model', 'C Model', 'D Model', 'E Model', 'F Model', 'G Model', 'Z Model'];      // five: A-E
const V2 = ['G Model', 'F Model', 'A Model', 'B Model', 'C Model', 'D Model', 'E Model', 'Z Model'];      // five: G F A B C
const TODAY = ['C Model', 'B Model', 'G Model', 'A Model', 'F Model', 'E Model', 'D Model', 'Z Model'];   // five: C B G A F (uncommitted)
const gameOf = (order, metric) => ({ metric, minDrawsForRanking: 10, models: order.map((name, i) => ({ name, draws: 20, [metric]: 2 - i / 10 })) });
// Joker+ enters the ranking with the second version (as on production, after the first of all)
const reportOf = (order, generatedAt, joker) => JSON.stringify({ generatedAt, games: { lotto: gameOf(order, 'avg_hits'), keno: gameOf(order, 'profit_bound'),
  ...(joker ? { jokerplus: gameOf(order, 'profit_per_bet') } : {}) } });
const commitReport = (order, when, joker) => {
  fs.writeFileSync(REPORT, reportOf(order, when.replace(/Z$/, ''), joker));
  gitIn(['add', '-A'], when);
  gitIn(['commit', '-q', '-m', 'saving last predictions'], when);
};

// the day files. lotto: G holds all six on every draw, A-F two, Z five; keno: G and Z win 5 of 5 (+149 each), the others
// lose (-1 a ticket); Joker+: G hits every digit and the sign
const lottoTicket = (name) => (name === 'G Model' ? [1, 2, 3, 4, 5, 6] : (name === 'Z Model' ? [1, 2, 3, 4, 5, 40] : [1, 2, 40, 41, 42, 43]));
const KENO_REAL = Array.from({ length: 20 }, (_, i) => i + 1);
const kenoRow = (name) => ({ name, predictions: [KENO_REAL.slice(0, 19).concat([70]), name === 'G Model' || name === 'Z Model' ? [1, 2, 3, 4, 5] : [61, 62, 63, 64, 65]] });
const jokerTicket = (name) => (name === 'G Model' ? [1, 2, 3, 4, 5, 6, 4] : [9, 9, 9, 9, 9, 6, 0]);
function writeDays() {
  const lottoDay = (file, extra = {}) => fs.writeFileSync(path.join(db, 'lotto', file), JSON.stringify({ realResult: [1, 2, 3, 4, 5, 6, 7],
    currentPrediction: NAMES.map((name) => ({ name, predictions: [lottoTicket(name)], multiPick: [7, 8, 9] })), currentNumberFrequency: { 1: 0.5, 44: 0.5 },
    newPrediction: NAMES.map((name) => ({ name, predictions: [[10, 11, 12, 13, 14, 15]], multiPick: [16, 17, 18] })), numberFrequency: { 10: 1 }, ...extra }));
  lottoDay('2026-10-3.json');     // before the first saved ranking: left out
  lottoDay('2026-10-6.json');     // V1: A-E (G, the six-hit row, not among them; today's five hold it)
  lottoDay('2026-10-7.json');     // V2: G F A B C
  lottoDay('2026-10-10.json', { madeAt: '2026-10-11T07:10:00+00:00' });   // the newest: V2 for its draw, today's report for the coming one
  const kenoDay = (file) => fs.writeFileSync(path.join(db, 'keno', file), JSON.stringify({ realResult: KENO_REAL,
    currentPrediction: NAMES.map(kenoRow), newPrediction: NAMES.map((name) => ({ name, predictions: [[21, 22, 23, 24, 25]] })), currentNumberFrequency: {}, numberFrequency: {} }));
  kenoDay('2026-10-6.json'); kenoDay('2026-10-7.json');
  const jokerDay = (file) => fs.writeFileSync(path.join(db, 'jokerplus', file), JSON.stringify({ realResult: [1, 2, 3, 4, 5, 6, 4],
    currentPrediction: NAMES.map((name) => ({ name, predictions: [jokerTicket(name)] })), newPrediction: NAMES.map((name) => ({ name, predictions: [[0, 0, 0, 0, 0, 0, 1]] })),
    currentNumberFrequency: {}, numberFrequency: {} }));
  jokerDay('2026-10-6.json');      // before Joker+ was first ranked (7 Oct): left out
  jokerDay('2026-10-7.json');
}

// a hand-made crypto record and its week game (as test/api.test.js), for Home's market headline
function writeMarket() {
  const mdir = path.join(repo, 'data', 'markets');
  fs.mkdirSync(mdir, { recursive: true });
  const pred = (model, bin, low, high) => ({ model, bin, low, high, direction: bin >= 5 ? 1 : -1 });
  const record = {
    market: 'crypto', generated_at: '2026-10-15T08:52:01+00:00', newest_game_day: '2026-10-14', k: 10, chance: { exact: 0.1, adjacent: 0.28, direction: 0.5 },
    instruments: [{ symbol: 'BTC', active: true, last_close: 81682, closes: [['2026-10-12', 83000], ['2026-10-13', 82500], ['2026-10-14', 81682]] },
      { symbol: 'ETH', active: true, last_close: 2421.6, closes: [['2026-10-12', 2500], ['2026-10-13', 2450], ['2026-10-14', 2421.6]] }],
    models: [{ name: 'GARCH Model', days: 42, positions: 84, exact_rate: 0.12, trades: 30, pnl_cash_total: 23.17, hold_total: 1, open_total: null, open_trades: null, first_day: '2026-09-01', last_day: '2026-10-14' }],
    best_model: 'GARCH Model',
    next: { made_on: '2026-10-14', made_at: '2026-10-15T08:54:36+00:00', instruments: { BTC: { base: 81682, predictions: [pred('GARCH Model', 6, 82085, 82600)] },
      ETH: { base: 2421.6, predictions: [pred('GARCH Model', 3, 2389.5, 2408.6)] } } },
    trading: { stake: 100, fee_per_leg: 0.001, currency: 'USDT', rule: 'daily', dates: ['2026-10-12', '2026-10-13', '2026-10-14'],
      models: { 'GARCH Model': [['2026-10-12', -1, -1], ['2026-10-13', 3.8, 2.8], ['2026-10-14', -1.99, 0.81]] },
      benchmark: [['2026-10-12', -19.49, -19.49], ['2026-10-13', 6.67, -12.82], ['2026-10-14', -12.24, -25.06]], hold: { models: {}, benchmark: [] },
      followed: { rule: 'daily', min_trades: 10, first_live: '2026-10-13', total: 0.81, market_total: -25.06, live_total: 1.81, live_market: -5.57, live_days: 2, live_positions: 4,
        points: [['2026-10-12', 'GARCH Model', 'book', 2, -1, -1, false, -19.49, { BTC: -2.5, ETH: 1.5 }],
          ['2026-10-13', 'GARCH Model', 'book', 3, 3.8, 2.8, true, 6.67, { BTC: 1.41, ETH: 1.58 }],
          ['2026-10-14', 'GARCH Model', 'book', 1, -1.99, 0.81, true, -12.24, { BTC: -1.99 }]] } },
    days: [],
  };
  fs.writeFileSync(path.join(mdir, 'crypto.json'), JSON.stringify(record));
}

async function partOne() {
  gitIn(['init', '-q'], '2026-10-01T00:00:00Z');
  fs.writeFileSync(path.join(repo, 'README'), 'x');
  gitIn(['add', 'README'], '2026-10-01T00:00:00Z');
  gitIn(['commit', '-q', '-m', 'start'], '2026-10-01T00:00:00Z');
  commitReport(V1, '2026-10-05T07:30:00Z', false);   // 09:30 Brussels, Monday: the first saved ranking
  commitReport(V2, '2026-10-07T07:30:00Z', true);    // Wednesday morning: Joker+ ranked for the first time
  fs.writeFileSync(REPORT, reportOf(TODAY, '2026-10-12T09:00:00', true));   // today's: the working tree, not committed
  writeDays();
  writeMarket();
  const app = fakeApp();
  lottery.install(app, { root: repo, dataDir: path.join(repo, 'data'), rankCacheFile: path.join(tmp, 'rank-cache.json') });
  const get = async (url, user = USER) => request(app, url, user);

  // --- signed out: 401; every route answers a user and the administrator alike ----------------------------------------
  const routes = ['/api/lottery', '/api/lottery/lotto', '/api/lottery/lotto?month=2026-10', '/api/lottery/lotto/2026-10-06', '/api/lottery/lotto/2026-10-10',
    '/api/lottery/keno', '/api/lottery/keno/2026-10-06', '/api/lottery/jokerplus', '/api/home'];
  for (const url of routes) {
    const out = await get(url, null);   // null: undefined would take the default reader
    ok(out.statusCode === 401 && out.body.error === 'login required' && out.headers['Cache-Control'] === 'no-store', `${url}: 401 signed out`);
    const u = await get(url, USER);
    const a = await get(url, ADMIN);
    const o = await get(url, OPEN);
    ok(u.statusCode === 200 && u.headers['Cache-Control'] === 'no-store' && !/NaN|undefined/.test(JSON.stringify(u.body)), `${url}: 200 for a user (${u.statusCode} ${JSON.stringify(u.body).slice(0, 160)})`);
    ok(withoutBuiltAt(a.body) === withoutBuiltAt(u.body) && withoutBuiltAt(o.body) === withoutBuiltAt(u.body), `${url}: the administrator (and open access) get the same user view`);
    ok(!mentions(u.body, 'Z Model'), `${url}: the row ranked in nobody's five is never sent`);
  }

  // --- the whitelist and the strict date ------------------------------------------------------------------------------
  const bad = [['/api/lottery/bingo', 404], ['/api/lottery/crypto', 404], ['/api/lottery/cryptoweek/2026-10-06', 404], ['/api/lottery/..', 404], ['/api/lottery/LOTTO', 404],
    ['/api/lottery/lotto?month=2026-1', 400], ['/api/lottery/lotto?month=2026-10-01', 400], ['/api/lottery/lotto?month=2025-01', 404], ['/api/lottery/lotto?month=2026-09', 404],
    ['/api/lottery/lotto/2026-10-6', 400], ['/api/lottery/lotto/2026-13-01', 400], ['/api/lottery/lotto/2026-02-30', 400], ['/api/lottery/lotto/..', 400],
    ['/api/lottery/lotto/2026-10-04', 404], ['/api/lottery/bingo/2026-10-06', 404], ['/api/lottery/lotto/2026-10-06/x', 404], ['/api/lottery/lotto/2026-10-03', 404], ['/api/home/x', 404]];
  for (const [url, status] of bad) {
    const r = await get(url);
    ok(r.statusCode === status && typeof r.body.error === 'string' && r.headers['Cache-Control'] === 'no-store', `${url}: ${status} with its reason (${r.statusCode} ${r.body && r.body.error})`);
  }
  const early = (await get('/api/lottery/lotto/2026-10-03')).body;
  ok(early.code === 'before-first' && /2026-10-05/.test(early.error) && early.rows === undefined && early.result === undefined,
    `a draw before the first saved ranking is left out, said why, and nothing of it sent: ${early.error}`);

  // --- the coming draw: today's five; the last scored draw: the five ranked before it ------------------------------------
  const overview = (await get('/api/lottery')).body;
  ok(overview.games.map((g) => g.game).join() === lottery.GAMES.join() && overview.topN === 5 && overview.firstSaved.committedAt === '2026-10-05T07:30:00Z'
    && overview.firstSaved.sha === undefined, 'the overview: every lottery game, in order; the first saved ranking without its sha');
  const lotto = overview.games.find((g) => g.game === 'lotto');
  ok(lotto.coming.rows.map((r) => r.name).join() === TODAY.slice(0, 5).join() && lotto.coming.scope.top.join() === TODAY.slice(0, 5).join()
    && lotto.coming.scope.ranking.today === true && lotto.coming.scope.of === 8 && lotto.coming.rows.every((r) => r.top === undefined),
    `the coming draw: today's five in their order (${lotto.coming.rows.map((r) => r.name).join(', ')})`);
  ok(lotto.next.date === '2026-10-13' && lotto.next.weekday === 'Tue' && lotto.next.time === null && lotto.next.madeAt === '2026-10-11T07:10:00Z' && lotto.next.past === true && lotto.next.today === false
    && JSON.stringify(lotto.coming.rows[0].multiPick) === '{"numbers":[16,17,18]}' && JSON.stringify(lotto.coming.rows[0].tickets[0].mains) === '[10,11,12,13,14,15]'
    && lotto.multiPick.sizes.length === 4, 'the next draw on the game\'s schedule (no time: the records hold the date), its rows with the multi-pick 7th-9th and the legend once');
  ok(lotto.last.date === '2026-10-10' && lotto.last.scope.top.join() === V2.slice(0, 5).join() && lotto.last.rows.map((r) => r.name).join() === V2.slice(0, 5).join()
    && lotto.last.figures.best.mains === 6 && lotto.last.figures.rows === 5 && lotto.last.rows[0].best.mains === 6 && lotto.last.rows[0].multi === 6
    && lotto.last.rows[1].best.mains === 2 && lotto.last.result.bonus[0] === 7,
    'the last scored draw: the five ranked before it (not today\'s), each row\'s figures and the five\'s together');
  ok(!mentions(lotto, 'D Model') && !mentions(lotto, 'E Model'), 'the lotto card names no row outside the coming five and the last draw\'s five');
  const joker = overview.games.find((g) => g.game === 'jokerplus');
  ok(joker.last.rows.length === 5 && joker.last.figures.best.left === 6 && joker.coming.rows[0].tickets[0].signName === 'Stier' && joker.shape.specials[0] === 'Sign',
    'Joker+: the sign by its name on the coming tickets; the last draw over its five');
  const keno = overview.games.find((g) => g.game === 'keno');
  ok(keno.last.rows.length === 5 && keno.last.figures.profit === 149 - 4 && keno.last.figures.stake === 5 && keno.last.figures.tickets === 5
    && keno.last.rows.find((r) => r.name === 'G Model').profit === 149 && keno.last.rows.find((r) => r.name === 'A Model').profit === -1,
    `keno: the money of the last draw over its five (G wins 149, four lose 1 each), the 20-number ranking not counted: ${JSON.stringify(keno.last.figures)}`);
  ok(['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'pick3'].every((g) => overview.games.find((x) => x.game === g).empty !== undefined || g === 'lotto'),
    'a game without a stored draw says so');

  // --- a past draw: the five ranked before it, never today's -------------------------------------------------------------
  const d6 = (await get('/api/lottery/lotto/2026-10-06')).body;
  const hindsight = TODAY.slice(0, 5);
  ok(d6.rows.map((r) => r.name).join() === V1.slice(0, 5).join() && d6.rows.map((r) => r.name).sort().join() !== hindsight.slice().sort().join()
    && d6.scope.ranking.committedAt === '2026-10-05T07:30:00Z' && d6.scope.cutoff === '2026-10-06T13:00:00Z' && d6.scope.rows === 5 && d6.scope.of === 8,
    `6 Oct: the five ranked before it (the 5 Oct version: A-E), not today's (${hindsight.join(', ')})`);
  ok(d6.figures.best.mains === 2 && d6.figures.rows === 5 && d6.votes.rows === 5 && d6.votes.stored === undefined && !mentions(d6, 'G Model') && !mentions(d6, 'F Model'),
    'its figures over those five: best 2 - today\'s five would say 6 (G was not ranked among the five then); G and F are not even named');
  ok(d6.prev === null && d6.next === '2026-10-07' && d6.file === undefined && d6.coming === null && d6.scoredIn === '2026-10-07',
    'no way back to the draw left out; the day file\'s name stays in the admin section');
  const d10 = (await get('/api/lottery/lotto/2026-10-10')).body;
  ok(d10.rows.map((r) => r.name).sort().join() === V2.slice(0, 5).sort().join() && d10.coming.rows.map((r) => r.name).sort().join() === TODAY.slice(0, 5).sort().join()
    && d10.coming.scope.ranking.today === true && d10.coming.votes.stored === undefined && !mentions(d10, 'D Model') && !mentions(d10, 'E Model'),
    'the newest day: its own draw on the ranking before it, the coming draw on today\'s - and nothing of D and E, in neither five');

  // --- the history: from the first ranked draw, each draw over its own five, the months over the same rows ---------------
  const h = (await get('/api/lottery/lotto')).body;
  const oct = h.months[0];
  const line = (date) => oct.days.find((x) => x.date === date);
  ok(h.draws === 3 && h.counted === 3 && h.months.length === 1 && oct.draws === 3 && oct.days.map((x) => x.date).join() === '2026-10-10,2026-10-07,2026-10-06'
    && h.leftOut.draws === 1 && h.leftOut.firstCounted === '2026-10-06' && h.leftOut.firstSaved === '2026-10-05' && /1 earlier draw is left out/.test(h.leftOut.text),
    `the history starts at the first ranked draw; the draw before it is left out and the line says so: ${h.leftOut.text}`);
  ok(line('2026-10-06').best.mains === 2 && line('2026-10-07').best.mains === 6 && line('2026-10-06').rows === 5 && oct.best.mains === 6 && oct.rows === 15 && oct.rowsMax === 5
    && h.total.best.mains === 6 && h.total.counted === 3 && h.total.from === '2026-10-06' && h.total.to === '2026-10-10' && h.total.multi === 6,
    'each draw over its own five (6 Oct: 2, the five then; 7 Oct: 6, G among them), the month and the whole history over the same rows');
  ok(h.ranking.versions.length === 2 && h.ranking.first.sha === undefined && h.ranking.firstOfGame.committedAt === '2026-10-05T07:30:00Z' && h.scope === 'top' && h.topN === 5,
    'the history names the ranking versions it read, without their sha');
  const kh = (await get('/api/lottery/keno')).body;
  ok(kh.draws === 2 && kh.months[0].profit === (-5) + (149 - 4) && kh.months[0].stake === 10 && kh.months[0].tickets === 10 && kh.total.profit === 140 && kh.total.stake === 10
    && kh.months[0].days.find((x) => x.date === '2026-10-06').profit === -5,
    `keno: money over each draw's five only - 6 Oct -5 (A-E lose; G's and Z's +149 are not theirs), 7 Oct +145 - every row would say ${-6 + 149 * 2} and ${-6 + 149 * 2}`);
  const kd = (await get('/api/lottery/keno/2026-10-06')).body;
  ok(kd.figures.profit === -5 && kd.figures.stake === 5 && kd.rows.every((r) => r.tickets[0].kind === 'ranking' && r.tickets[0].profit === null && r.tickets[1].profit === -1),
    'keno, one draw: each row\'s ranking of 20 not played, its 5-ticket scored');
  const jh = (await get('/api/lottery/jokerplus')).body;
  ok(jh.draws === 1 && jh.leftOut.draws === 1 && jh.leftOut.firstSaved === '2026-10-07' && jh.ranking.firstOfGame.committedAt === '2026-10-07T07:30:00Z'
    && (await get('/api/lottery/jokerplus/2026-10-06')).body.code === 'before-first',
    'Joker+ entered the ranking after the first of all: its history starts at its own first saved ranking');
  // the lines say why by the rule itself - no ranking saved before 15:00 Brussels time on the draw's day - and when the
  // first was saved (on production Joker+'s came at 18:56 on 7 Sep, after that day's cut-off: its 7 Sep draw is left out)
  const jd = (await get('/api/lottery/jokerplus/2026-10-06')).body;
  ok(jh.leftOut.text === '1 earlier draw is left out: no ranking of jokerplus had been saved before 15:00 Brussels time on its day (the first was saved on 2026-10-07 at 09:30; the first counted draw is 2026-10-07).'
    && jd.error === 'no ranking of jokerplus had been saved before 15:00 Brussels time on this draw\'s day (the first was saved on 2026-10-07 at 09:30), so it is left out',
    `the left-out line and a left-out draw give the cut-off rule and the first save's time: ${jh.leftOut.text} / ${jd.error}`);

  // --- a ranking history git could not read whole: never "not saved yet", never a later start, never git's own words ---
  const dirs = { root: repo, databaseDir: db, marketsDir: path.join(repo, 'data', 'markets') };
  const full = await rankhistory.load(repo, { cacheFile: path.join(tmp, 'rank-cache.json') });
  const todays = require('../ranking').loadReport(db);
  const noGit = rankhistory.makeHistory([], { error: 'git is not installed here' });
  const blind = lottery.overviewOf(dirs, noGit, todays);
  const blindHome = lottery.homeOf(dirs, noGit, todays);
  const lottoCard = blind.games.find((g) => g.game === 'lotto');
  ok(lottoCard.last.leftOut === undefined && lottoCard.last.none.code === 'unavailable' && !/saved yet|git/.test(JSON.stringify(blind))
    && blindHome.lottery.find((g) => g.game === 'lotto').last.none.code === 'unavailable' && !/saved yet|git/.test(JSON.stringify(blindHome.lottery))
    && lottery.historyOf(dirs, 'lotto', noGit, null).status === 503,
    `no ranking history at all: the cards and Home say the rankings could not be read just now, not that none was saved (${lottoCard.last.none.text})`);
  const [v1, v2] = full.versions;
  const firstUnread = rankhistory.makeHistory([v2], { head: 'first-unread', error: 'git took over 20 s', unread: [v1.at] });
  const fh = lottery.historyOf(dirs, 'lotto', firstUnread, null);
  ok(fh.leftOut.unavailable === true && fh.leftOut.firstSaved === null && /not shown just now/.test(fh.leftOut.text) && fh.draws === 2
    && lottery.drawOf(dirs, 'lotto', '2026-10-06', firstUnread, todays).status === 503 && !/git/.test(JSON.stringify(fh)),
    `the first version unread: the draws before the next one are not said to be left out, nor read on a later start (${fh.leftOut.text})`);
  const middleUnread = rankhistory.makeHistory([v1], { head: 'middle-unread', error: 'git exited with code 128', unread: [v2.at] });
  const mh = lottery.historyOf(dirs, 'lotto', middleUnread, null);
  const md = lottery.drawOf(dirs, 'lotto', '2026-10-07', middleUnread, todays);
  ok(mh.draws === 3 && mh.counted === 1 && mh.total.counted === 1 && mh.ranking.error === 'a saved ranking could not be read just now'
    && md.scope.none.code === 'unavailable' && md.rows.length === 0 && !/git/.test(JSON.stringify(mh) + JSON.stringify(md)),
    `a version git could not read: its draws have no top 5, and a user reads why without git's words (${md.scope.none.text})`);

  // --- the last check: a row outside the five is a 500, never sent ------------------------------------------------------
  const realDraw = workbench.drawView;
  const realHistory = workbench.historyView;
  const logged = console.error;
  console.error = () => {};
  try {
    workbench.drawView = (...args) => { const v = realDraw(...args); v.rows.push({ name: 'Z Model', tickets: [], rank: 8, top: false }); return v; };
    const leak = await get('/api/lottery/lotto/2026-10-06');
    ok(leak.statusCode === 500 && leak.body.error === 'could not build this view' && !mentions(leak.body, 'Z Model'), 'a draw view with a row outside the five: 500, nothing sent');
    workbench.drawView = (...args) => { const v = realDraw(...args); if (v.coming) v.coming.rows.push({ name: 'Z Model', tickets: [] }); return v; };
    // the overview is kept until a source changes: today's report is touched, so it is built again
    fs.utimesSync(REPORT, new Date(), new Date(Date.now() + 5000));
    ok((await get('/api/lottery/lotto/2026-10-10')).statusCode === 500 && (await get('/api/lottery')).statusCode === 500, 'a coming row outside today\'s five: 500, on the draw and the overview');
    workbench.drawView = (...args) => { const v = realDraw(...args); v.votes.stored = { from: 'file', values: [[1, 1]] }; return v; };
    ok((await get('/api/lottery/lotto/2026-10-06')).statusCode === 500, 'a vote of every row: 500');
    workbench.drawView = realDraw;
    workbench.historyView = (...args) => { const v = realHistory(...args); v.months[0].days[2].best.by = 'G Model'; return v; };
    fs.utimesSync(path.join(db, 'lotto', '2026-10-6.json'), new Date(), new Date(Date.now() + 5000));   // a counted day file changed: built again
    ok((await get('/api/lottery/lotto?month=2026-10')).statusCode === 500, 'a history line whose best ticket is not one of the draw\'s five: 500');
  } finally { workbench.drawView = realDraw; workbench.historyView = realHistory; console.error = logged; }
  ok((await get('/api/lottery/lotto/2026-10-06')).statusCode === 200, 'and back to normal');

  // --- Home: the lottery's next draws and top picks, the markets' headlines -----------------------------------------------
  const home = (await get('/api/home')).body;
  const hl = home.lottery.find((g) => g.game === 'lotto');
  ok(hl.top.name === 'C Model' && hl.rows === 5 && hl.next.date === '2026-10-13' && hl.last.date === '2026-10-10' && hl.last.figures.best.mains === 6 && hl.last.rows === 5
    && hl.ranking.today === true && !mentions(home.lottery, 'Z Model'), 'Home, lottery: the next draw, the first of today\'s five, the last draw over the five before it');
  const record = markets.loadMarket(path.join(repo, 'data', 'markets'), 'crypto');
  const view = api.marketView('crypto', record, null, new Date());
  ok(home.crypto.plan.total === view.status.liveTotal && home.crypto.plan.market === view.status.liveMarket && home.crypto.plan.ahead === view.status.ahead
    && home.crypto.plan.since === view.status.since && home.crypto.plan.days === view.status.days && home.crypto.plan.through === '2026-10-14'
    && home.crypto.today.toBuy === view.status.openToBuy && home.crypto.today.held === view.status.openHeld && home.crypto.unit === 'USDT',
    `Home, crypto: the plan against buying everything with its window, today's positions - api.js marketView's own figures: ${JSON.stringify(home.crypto.plan)}`);
  ok(home.shares.empty === 'no record yet' && JSON.stringify(home).length < 10 * 1024, `Home: a market without a record says so; the answer is small (${JSON.stringify(home).length} bytes)`);

  // --- the coming draw on its own day: today (the old pages' "for the draw of today, ..."), not past --------------------
  const onTheDay = new Date('2026-10-13T08:00:00Z');   // 10:00 Brussels on the draw's day
  const dayCard = lottery.overviewOf(dirs, full, todays, onTheDay).games.find((g) => g.game === 'lotto');
  const dayHome = lottery.homeOf(dirs, full, todays, onTheDay).lottery.find((g) => g.game === 'lotto');
  ok(dayCard.next.date === '2026-10-13' && dayCard.next.today === true && dayCard.next.past === false && dayHome.next.today === true
    && lottery.overviewOf(dirs, full, todays, new Date('2026-10-12T21:59:00Z')).games.find((g) => g.game === 'lotto').next.today === false,
    'the coming draw on its own Brussels day is today, not past; the evening before (23:59 Brussels) it is neither');

  // --- a run in progress: both answers say which job and since when (pipeline.js), never its pid or the lock's path -----
  ok(overview.running === null && home.running === null, 'no process.lock: running is null on /api/lottery and /api/home');
  const lock = path.join(repo, 'process.lock');
  const since = new Date('2026-10-12T05:00:00Z');
  fs.writeFileSync(lock, `${process.pid}\n`);   // this test's own process: alive, and no pipeline script on its command line
  fs.utimesSync(lock, since, since);
  try {
    for (const url of ['/api/lottery', '/api/home']) {
      const u = (await get(url)).body;
      const a = (await get(url, ADMIN)).body;
      ok(u.running && Object.keys(u.running).join() === 'what,since' && u.running.what === 'A pipeline job is running' && u.running.since === '2026-10-12T05:00:00Z'
        && JSON.stringify(a.running) === JSON.stringify(u.running) && !/process\.lock|\/proc/.test(JSON.stringify(u.running)),
        `${url}: running says the job and since when, the same for a user and the administrator, no pid or path (${JSON.stringify(u.running)})`);
    }
    fs.writeFileSync(lock, '');
    ok((await get('/api/lottery')).body.running === null, 'an empty lock: nothing running');
  } finally { fs.rmSync(lock, { force: true }); }
  const after = (await get('/api/lottery')).body;
  ok(after.running === null && withoutBuiltAt({ ...after, running: undefined }) === withoutBuiltAt({ ...overview, running: undefined }),
    'the lock gone: running is null again, and the rest of the answer is what it was');
}

// ===================================================================================================================
// 2. the Council: the members for the administrator only, a run answered to its asker only
// ===================================================================================================================
async function partTwo() {
  const seen = [];
  // the runs in the order they are asked: alice's answered, then two that fail api.py's preflight (the model boxes off)
  const ids = ['abcabcabcabc', 'badbadbadbad', 'cafecafecafe'];
  let askCount = 0;
  // what api.py says about its endpoints: their names and LAN addresses (orchestrator.check_endpoints, health.check)
  const DOWN = 'EndpointsUnavailable: Member One: http://10.0.0.5:8080 unreachable: <urlopen error [Errno 113] No route to host>; Head: modelbox:8081 returned HTTP 503';
  const failed = (id) => ({ id, state: 'failed', question: 'q', progress: { phase: 'starting', seats: { 'Member One': { state: 'failed', error: 'http://10.0.0.5:8080 unreachable' } }, head: null }, error: DOWN });
  const standIn = http.createServer((req, res) => {
    seen.push(req.url);
    const send = (status, body) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };
    if (req.url === '/endpoints') return send(200, { members: [{ name: 'Member One', lab: 'x', temperature: 0.7 }], head: { name: 'Head', lab: 'y' }, status: { ready: true, members: [] } });
    if (req.url === '/health') {
      return send(200, { ok: true, busy: false, status: { checked: '2026-10-23T10:00:00+00:00', ready: false, reachable: 0, total: 1,
        members: [{ name: 'Member One', ok: false, detail: 'http://10.0.0.5:8080 unreachable' }], head: { name: 'Head', ok: false, detail: 'modelbox:8081 returned HTTP 503' } } });
    }
    if (req.url === '/ask' && req.method === 'POST') { req.resume(); const id = ids[Math.min(askCount, ids.length - 1)]; askCount += 1; return req.on('end', () => send(202, { id, state: 'queued' })); }
    if (req.url === '/job/abcabcabcabc') {
      return send(200, { id: 'abcabcabcabc', state: 'done', question: 'q', result: {
        members: [{ name: 'Member One', lab: 'Lab X', ok: true, answer: 'a', seconds: 1 }, { name: 'Member Two', lab: 'Lab Z', ok: false, error: 'request to http://10.0.0.7:8082/v1/chat/completions failed' }],
        head: { name: 'Head', lab: 'Lab Y', ok: true, answer: 'yes', order: ['Member One'] } } });
    }
    if (req.url === '/job/badbadbadbad' || req.url === '/job/cafecafecafe') return send(200, failed(req.url.slice(5)));
    return send(404, { error: 'no such job' });
  });
  await new Promise((resolve) => standIn.listen(0, '127.0.0.1', resolve));
  liveStandIn = standIn;
  process.env.COUNCIL_API_HOST = '127.0.0.1';
  process.env.COUNCIL_API_PORT = String(standIn.address().port);
  process.env.COUNCIL_ACCESS = 'user';
  // the session store in the temp dir (never config/), the form token as the session's; express only where it is installed
  const store = require('../councilSessions').createStore(path.join(tmp, 'council-sessions'));
  let haveExpress = true;
  try { require.resolve('express'); } catch (e) { haveExpress = false; }
  const stubs = { './auth': { csrfOk: (req) => Boolean(req.user && req.body && req.body._csrf === req.user.csrf) }, './councilSessions': store };
  if (!haveExpress) stubs.express = { json: () => (req, res, next) => next() };
  const councilFile = path.join(REPO, 'council.js');
  const original = Module.prototype.require;
  Module.prototype.require = function stubbed(id) {
    if (this.filename === councilFile && Object.prototype.hasOwnProperty.call(stubs, id)) return stubs[id];
    return original.apply(this, arguments);
  };
  const app = fakeApp();
  let council;
  try {
    council = require('../council');
    council.install(app, { header: () => '', footer: () => '', escapeHtml: (t) => t });
  } finally { Module.prototype.require = original; }
  const alice = { name: 'alice', role: 'user', open: false, csrf: 'tok-a' };
  const bob = { name: 'bob', role: 'user', open: false, csrf: 'tok-b' };
  const carol = { name: 'carol', role: 'user', open: false, csrf: 'tok-c' };

  const eu = await request(app, '/council/api/endpoints', alice);
  ok(eu.statusCode === 403 && eu.body.error === 'admin only' && eu.headers['Cache-Control'] === 'no-store' && !seen.includes('/endpoints'), 'the members and their endpoints: 403 for a user, the Council API not asked');
  const ea = await request(app, '/council/api/endpoints', ADMIN);
  const eo = await request(app, '/council/api/endpoints', OPEN);
  ok(ea.statusCode === 200 && ea.body.members[0].name === 'Member One' && eo.statusCode === 200, 'the administrator (and open access) get them');

  ok((await request(app, '/council/api/ask', alice, { method: 'POST', body: { question: 'Will it rain?' } })).statusCode === 403, 'asking without the form token: 403');
  const asked = await request(app, '/council/api/ask', alice, { method: 'POST', body: { question: 'Will it rain?', _csrf: 'tok-a' } });
  ok(asked.statusCode === 202 && asked.body.id === 'abcabcabcabc' && store.ID.test(asked.body.session), 'a user asks: the run is filed in a session of theirs');
  // the server's own tracker polls the run until it is done (council.js track): wait for it, so the counts below are ours
  for (let i = 0; i < 100 && store.load('alice', asked.body.session).turns[0].state === 'pending'; i += 1) await new Promise((r) => setTimeout(r, 20));
  ok(store.load('alice', asked.body.session).turns[0].state === 'done', 'the tracker filed the answer in the asker\'s session');
  const jobA = await request(app, '/council/api/job/abcabcabcabc', alice);
  const asks = seen.length;
  const jobB = await request(app, '/council/api/job/abcabcabcabc', bob);
  const jobAdmin = await request(app, '/council/api/job/abcabcabcabc', ADMIN);
  ok(jobA.statusCode === 200 && jobA.body.state === 'done' && jobB.statusCode === 404 && jobAdmin.statusCode === 404 && seen.length === asks && !mentions(jobB.body, 'Will it rain?'),
    'the run is answered to its asker only: another user, and the administrator, get a 404 and the Council API is not asked');
  ok((await request(app, '/council/api/job/ABC', alice)).statusCode === 400 && (await request(app, '/council/api/job/0123456789ab', alice)).statusCode === 404,
    'a job id that is not hex: 400; one that is not the caller\'s: 404');
  // a run filed before a restart (not in this process's memory): looked up in the caller's own sessions
  const filed = store.appendTurn('carol', null, { jobId: 'feedfeedfeed', question: 'earlier', context: null });
  store.updateTurn('carol', filed.session.id, filed.turn.id, { state: 'done', result: { head: { answer: 'kept' } } });
  const before = seen.length;
  const jobC = await request(app, '/council/api/job/feedfeedfeed', carol);
  ok(jobC.statusCode === 404 && seen.length === before + 1 && (await request(app, '/council/api/job/feedfeedfeed', bob)).statusCode === 404 && seen.length === before + 1,
    'a run of an earlier process: its asker\'s request goes to the Council API, another\'s does not');
  // sessions: always the caller's own
  const listA = (await request(app, '/council/api/sessions', alice)).body.sessions;
  const listB = (await request(app, '/council/api/sessions', bob)).body.sessions;
  ok(listA.length === 1 && listB.length === 0 && (await request(app, `/council/api/sessions/${listA[0].id}`, bob)).statusCode === 404
    && (await request(app, `/council/api/sessions/${listA[0].id}`, alice)).body.session.turns[0].question === 'Will it rain?',
    'sessions: each reader lists and opens their own only');
  const del = await request(app, `/council/api/sessions/${listA[0].id}/delete`, bob, { method: 'POST', body: { _csrf: 'tok-b' } });
  ok(del.body.removed === false && (await request(app, '/council/api/sessions', alice)).body.sessions.length === 1, 'another reader cannot delete it');
  // what a user is sent of a run: no member's lab, no endpoint's address - in the job and in the filed turn alike
  const leaks = (body) => { const text = JSON.stringify(body); return ['"lab"', 'Lab X', 'Lab Y', '10.0.0.', 'modelbox', 'http://', ':8080', ':8081'].filter((x) => text.includes(x)); };
  const doneSession = (await request(app, `/council/api/sessions/${listA[0].id}`, alice)).body.session;
  ok(leaks(jobA.body).length === 0 && leaks(doneSession).length === 0 && jobA.body.result.members.length === 2 && jobA.body.result.head.answer === 'yes'
    && doneSession.turns[0].result.members[1].error === 'request to an endpoint failed',
    `a user's answered run: the members and the head without their labs, a member's error without its address (${leaks(jobA.body).concat(leaks(doneSession)).join(', ')})`);
  const dave = { name: 'dave', role: 'user', open: false, csrf: 'tok-d' };
  const davesRun = await request(app, '/council/api/ask', dave, { method: 'POST', body: { question: 'Is the council in?', _csrf: 'tok-d' } });
  for (let i = 0; i < 100 && store.load('dave', davesRun.body.session).turns[0].state === 'pending'; i += 1) await new Promise((r) => setTimeout(r, 20));
  const davesJob = await request(app, `/council/api/job/${davesRun.body.id}`, dave);
  const davesSession = (await request(app, `/council/api/sessions/${davesRun.body.session}`, dave)).body.session;
  ok(davesRun.body.id === 'badbadbadbad' && davesJob.body.state === 'failed' && davesSession.turns[0].state === 'failed'
    && leaks(davesJob.body).length === 0 && leaks(davesSession).length === 0
    && davesJob.body.error === davesSession.turns[0].error && /not sitting/.test(davesJob.body.error),
    `a user's run that failed api.py's preflight: the council was not sitting, in the job and the session, without the endpoints (${davesJob.body.error})`);
  const adminsRun = await request(app, '/council/api/ask', ADMIN, { method: 'POST', body: { question: 'And for me?', _csrf: 'tok-admin' } });
  for (let i = 0; i < 100 && store.load('admin', adminsRun.body.session).turns[0].state === 'pending'; i += 1) await new Promise((r) => setTimeout(r, 20));
  const adminsJob = await request(app, `/council/api/job/${adminsRun.body.id}`, ADMIN);
  ok(adminsJob.body.error === DOWN && (await request(app, `/council/api/sessions/${adminsRun.body.session}`, ADMIN)).body.session.turns[0].error === DOWN,
    'the administrator\'s own run keeps api.py\'s words, the endpoints in them');
  // whether the council can sit now: every reader, only ready and busy
  const st = await request(app, '/council/api/status', alice);
  ok(st.statusCode === 200 && JSON.stringify(Object.keys(st.body).sort()) === '["busy","ready"]' && st.body.ready === false && st.body.busy === false
    && st.headers['Cache-Control'] === 'no-store' && leaks(st.body).length === 0 && !mentions(st.body, 'Member One'),
    `the council's status for a user: ready and busy, nobody's name or address (${JSON.stringify(st.body)})`);
  ok((await request(app, `/council/api/sessions/${listA[0].id}/delete`, alice, { method: 'POST', body: {} })).statusCode === 403
    && (await request(app, `/council/api/sessions/${listA[0].id}/delete`, alice, { method: 'POST', body: { _csrf: 'tok-a' } })).body.removed === true, 'its owner can, with the form token');
  return { standIn, council, store, haveExpress };
}

// ===================================================================================================================
// 3. production, read-only: a copy of its lottery files, a scratch root whose .git is production's
// ===================================================================================================================
const PROD_ROOT = '/root/sequencePredictor';
const GIT_REPORT = 'data/database/modelPerformance.json';
// 15:00 Brussels on a day, found by asking Intl which UTC hour is 15:00 there (independent of brussels.js)
function cutoffMs(date) {
  const fmt = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/Brussels', hour: '2-digit', hour12: false });
  for (const h of [12, 13, 14]) { const ms = Date.parse(`${date}T${String(h).padStart(2, '0')}:00:00Z`); if (fmt.format(ms) === '15') return ms; }
  throw new Error(`no 15:00 Brussels on ${date}`);
}
// the five ranked before a draw, read from git directly: the newest version committed before the cut-off that parses
// (a version that does not parse was never shown); its order, the names present in the draw's table, the first five
function independentTop(root, commits, game, date, present) {
  const cut = cutoffMs(date);
  for (const c of commits.filter((x) => x.at < cut).sort((a, b) => b.at - a.at)) {
    let report;
    try { report = JSON.parse(execFileSync('git', ['-C', root, 'show', `${c.sha}:${GIT_REPORT}`], { maxBuffer: 64 * 1024 * 1024 }).toString()); } catch (e) { continue; }
    const g = report && report.games && report.games[game];
    return g ? g.models.map((m) => String(m.name)).filter((n) => present.includes(n)).slice(0, 5) : [];
  }
  return [];
}
// the best ticket written fresh: mains hits (membership; Joker+ the runs from either end), then the supplement
function independentBest(game, rows, real) {
  const specials = { euromillions: 2, eurodreams: 1, vikinglotto: 1 }[game] || 0;
  const mainsReal = specials ? real.slice(0, real.length - specials) : (game === 'lotto' ? real.slice(0, 6) : real.slice(0, 6));
  const specReal = specials ? real.slice(-specials) : [];
  const bonus = game === 'lotto' ? real.slice(6) : [];
  let best = null;
  rows.forEach((r) => r.predictions.forEach((t) => {
    let m; let s;
    if (game === 'jokerplus') {
      let l = 0; while (l < 6 && Number(t[l]) === Number(real[l])) l += 1;
      let rr = 0; if (l < 6) while (rr < 6 - l && Number(t[5 - rr]) === Number(real[5 - rr])) rr += 1;
      m = l + rr; s = t.length === 7 && real.length === 7 && Number(t[6]) === Number(real[6]) ? 1 : 0;
    } else {
      const tm = specials && t.length > mainsReal.length ? t.slice(0, t.length - specials) : t;
      const ts = specials && t.length > mainsReal.length ? t.slice(-specials) : [];
      m = tm.filter((n) => mainsReal.includes(n)).length;
      s = ts.filter((n) => specReal.includes(n)).length + tm.filter((n) => bonus.includes(n)).length;
    }
    if (!best || m > best[0] || (m === best[0] && s > best[1])) best = [m, s];
  }));
  return best;
}
async function partThree() {
  if (!fs.existsSync(path.join(PROD_ROOT, GIT_REPORT)) || !fs.existsSync(path.join(PROD_ROOT, '.git'))) return false;
  const root = path.join(tmp, 'prod');
  fs.mkdirSync(path.join(root, 'data', 'database'), { recursive: true });
  fs.symlinkSync(path.join(PROD_ROOT, '.git'), path.join(root, '.git'));
  lottery.GAMES.forEach((g) => fs.cpSync(path.join(PROD_ROOT, 'data', 'database', g), path.join(root, 'data', 'database', g), { recursive: true }));
  fs.copyFileSync(path.join(PROD_ROOT, GIT_REPORT), path.join(root, GIT_REPORT));
  fs.cpSync(path.join(PROD_ROOT, 'data', 'markets'), path.join(root, 'data', 'markets'), { recursive: true });
  const app = fakeApp();
  lottery.install(app, { root, dataDir: path.join(root, 'data'), rankCacheFile: path.join(tmp, 'prod-rank-cache.json') });
  const sizes = [];
  const get = async (url, user = USER) => {
    const r = await request(app, url, user);
    const text = JSON.stringify(r.body);
    sizes.push([url, text.length]);
    ok(r.statusCode === 200 && !/NaN|undefined/.test(text), `production ${url}: ${r.statusCode} ${r.statusCode !== 200 ? text.slice(0, 200) : ''}`);
    ok(text.length < 40 * 1024, `production ${url}: ${text.length} bytes, under 40 KB`);
    return r.body;
  };
  const commits = execFileSync('git', ['-C', root, 'log', '--format=%H %ct', '--', GIT_REPORT]).toString().trim().split('\n')
    .map((l) => l.split(' ')).map(([sha, ct]) => ({ sha, at: Number(ct) * 1000 }));
  const firstSaved = Math.min(...commits.map((c) => c.at));
  const report = JSON.parse(fs.readFileSync(path.join(root, GIT_REPORT), 'utf-8'));
  const readDay = (game, file) => JSON.parse(fs.readFileSync(path.join(root, 'data', 'database', game, file), 'utf-8'));
  const names = (rows) => [...new Set((rows || []).map((m) => m.name || 'not known'))];

  // the coming draw and the last one: today's five computed from the report by hand; nothing of the other rows sent
  const overview = await get('/api/lottery');
  ok(withoutBuiltAt(overview) === withoutBuiltAt((await request(app, '/api/lottery', ADMIN)).body), 'production: the administrator gets the user view');
  for (const card of overview.games) {
    const files = workbench.dayFilesOf(path.join(root, 'data', 'database', card.game));
    const day = readDay(card.game, files[0].file);
    const todayFive = report.games[card.game] ? report.games[card.game].models.map((m) => String(m.name)).filter((n) => names(day.newPrediction).includes(n)).slice(0, 5) : [];
    ok(card.coming.rows.map((r) => r.name).sort().join() === todayFive.slice().sort().join() && card.coming.rows.length <= 5,
      `production ${card.game}: the coming draw's rows are today's five (${todayFive.join(', ')})`);
    const allowed = new Set([...todayFive, ...(card.last.scope ? card.last.scope.top : [])]);
    const outside = [...new Set([...names(day.currentPrediction), ...names(day.newPrediction)])].filter((n) => !allowed.has(n));
    ok(outside.length > 0 && outside.every((n) => !mentions(card, n)), `production ${card.game}: none of the ${outside.length} other rows is named on the card`);
  }

  // the history: the draws from the first saved ranking on; a few draws against the independent computation
  const toScore = [];
  const compared = [];
  for (const game of lottery.GAMES) {
    const files = workbench.dayFilesOf(path.join(root, 'data', 'database', game));
    const h = await get(`/api/lottery/${game}`);
    const listed = h.months.flatMap((m) => (m.days || []).map((d) => d.date));
    ok(h.draws + h.leftOut.draws === files.length && listed.every((d) => cutoffMs(d) > firstSaved) && h.total.counted <= h.draws
      && files.filter((f) => cutoffMs(f.date) <= firstSaved).length <= h.leftOut.draws,
      `production ${game}: ${h.draws} draws from ${h.leftOut.firstCounted}, ${h.leftOut.draws} left out - none before the first saved ranking listed`);
    ok(h.months.every((m) => m.rowsMax === null || m.rowsMax <= 5), `production ${game}: at most five rows a draw`);
    const pick = [h.total.to, h.leftOut.firstCounted, listed[Math.floor(listed.length / 2)]].filter(Boolean);
    for (const date of [...new Set(pick)]) {
      const d = await get(`/api/lottery/${game}/${date}`);
      const file = files.find((f) => f.date === date).file;
      const day = readDay(game, file);
      const five = independentTop(root, commits, game, date, names(day.currentPrediction));
      ok(d.rows.map((r) => r.name).sort().join() === five.slice().sort().join() && d.scope.top.slice().sort().join() === five.slice().sort().join(),
        `production ${game} ${date}: the five ranked before the draw, read from git by hand: ${five.join(', ')}`);
      // the other rows: every name of the day file outside the draw's five (and, on the newest day, today's five)
      const sent = new Set([...five, ...(d.coming ? d.coming.scope.top : [])]);
      const outside = [...new Set([...names(day.currentPrediction), ...(d.coming ? names(day.newPrediction) : [])])].filter((n) => !sent.has(n));
      ok(outside.every((n) => !mentions(d, n)), `production ${game} ${date}: none of the other rows is named`);
      if (date === h.leftOut.firstCounted) ok(d.prev === null, `production ${game} ${date}: the first counted draw has no way back`);
      const rows = day.currentPrediction.filter((m) => five.includes(m.name));
      const line = h.months.flatMap((m) => m.days || []).find((x) => x.date === date) || (await get(`/api/lottery/${game}?month=${date.slice(0, 7)}`)).months.find((m) => m.month === date.slice(0, 7)).days.find((x) => x.date === date);
      if (['keno', 'pick3', 'jokerplus'].includes(game)) {
        const tickets = rows.flatMap((m) => m.predictions.filter((t) => game !== 'keno' || (t.length >= 5 && t.length <= 10)));
        const at = toScore.length;
        tickets.forEach((t) => toScore.push({ game, ticket: t, real: day.realResult }));
        const stake = tickets.reduce((s, t) => s + (game === 'keno' ? 1 : (game === 'pick3' ? (new Set(t).size === 1 ? 3 : 4) : 1.5)), 0);
        compared.push({ game, date, at, n: tickets.length, stake, api: d.figures, line });
      }
      if (game !== 'keno' && game !== 'pick3') {
        const best = independentBest(game, rows, day.realResult);
        const b = d.figures.best;
        const got = game === 'jokerplus' ? [b.left + b.right, b.sign] : [b.mains, b.specials === undefined ? 0 : b.specials];
        ok(best && got[0] === best[0] && got[1] === best[1] && line.best && JSON.stringify(line.best) === JSON.stringify(b),
          `production ${game} ${date}: the best ticket over the five, counted by hand: ${JSON.stringify(best)} = ${JSON.stringify(got)}, as the history line`);
      }
    }
  }
  // the money of the sampled draws by the pipeline's own scorer (src/Helpers.py), the stake by the rules written out
  const py = 'import sys, json\nsys.path.insert(0, sys.argv[1])\nfrom src.Helpers import Helpers\nh = Helpers()\nout = []\n'
    + 'for it in json.load(sys.stdin):\n    f = {"keno": h.keno_ticket_profit, "pick3": h.pick3_ticket_profit, "jokerplus": h.jokerplus_ticket_profit}[it["game"]]\n'
    + '    out.append(f(it["ticket"], it["real"]))\nprint(json.dumps(out))\n';
  const profits = JSON.parse(execFileSync('python3', ['-c', py, REPO], { input: JSON.stringify(toScore), cwd: REPO, maxBuffer: 16 * 1024 * 1024 }).toString());
  compared.forEach((c) => {
    const sum = Math.round(profits.slice(c.at, c.at + c.n).reduce((s, p) => s + (p === null ? 0 : p), 0) * 100) / 100;
    ok(c.api.profit === sum && c.api.stake === c.stake && c.api.tickets === c.n && c.line.profit === sum && c.line.stake === c.stake,
      `production ${c.game} ${c.date}: the five's money ${c.api.profit} over ${c.n} tickets, stake ${c.api.stake} = Helpers.py ${sum}, stake ${c.stake}`);
  });
  console.log(`lottery on production: ${compared.length} money draws against Helpers.py; the largest answer ${sizes.sort((a, b) => b[1] - a[1])[0].join(' at ')} bytes`);
  return true;
}

// --- the same routes through express itself, when it can be loaded (NODE_PATH) ----------------------------------------
async function throughExpress(council) {
  let express = null;
  try { express = require('express'); } catch (e) { return false; }
  const real = express();
  const who = { admin: ADMIN, user: USER, alice: { name: 'alice', role: 'user', open: false, csrf: 'tok-a' } };
  real.use((req, res, next) => { req.user = who[req.get('x-test-user')]; next(); });
  lottery.install(real, { root: repo, dataDir: path.join(repo, 'data'), rankCacheFile: path.join(tmp, 'rank-cache.json') });
  if (council) council.install(real, { header: () => '', footer: () => '', escapeHtml: (t) => t });
  await new Promise((resolve) => {
    const server = real.listen(0, '127.0.0.1', async () => {
      const base = `http://127.0.0.1:${server.address().port}`;
      try {
        const get = async (url, user) => { const r = await fetch(base + url, { headers: user ? { 'x-test-user': user } : {} }); return { status: r.status, body: await r.json() }; };
        ok((await get('/api/lottery/lotto/2026-10-06')).status === 401 && (await get('/api/lottery/lotto/2026-10-06', 'user')).body.rows.length === 5, 'express: 401 signed out, the five for a user');
        ok((await get('/api/lottery/..%2F..%2Fetc', 'user')).status === 404 && (await get('/api/lottery/lotto/2026-10-06%2F..', 'user')).status === 400
          && (await get('/api/lottery/lotto/2026-10-06/x', 'user')).status === 404 && (await get('/api/lottery/lotto?month=2026-10&month=2026-11', 'user')).status === 400,
          'express: an encoded path is a name, never a path; a repeated month is refused; anything else under /api/lottery is a JSON 404');
        if (council) {
          ok((await get('/council/api/endpoints', 'user')).status === 403 && (await get('/council/api/endpoints', 'admin')).status === 200
            && (await get('/council/api/job/abcabcabcabc', 'user')).status === 404 && (await get('/council/api/job/abcabcabcabc', 'alice')).status === 200,
            'express: the endpoints for the administrator only, a run for its asker only');
          const st = await get('/council/api/status', 'user');
          ok(st.status === 200 && JSON.stringify(st.body) === '{"ready":false,"busy":false}', `express: the council's status for a user (${JSON.stringify(st.body)})`);
        }
      } catch (e) { console.error(e); process.exitCode = 1; } finally { server.close(); resolve(); }
    });
  });
  return true;
}

let standIn = null;
partOne().then(() => {
  rankhistory.reset();
  gates.forgetHeads();
  return partTwo();
}).then(async (c) => {
  standIn = c.standIn;
  const prod = await partThree();
  const viaExpress = await throughExpress(c.haveExpress ? c.council : null);
  return { prod, viaExpress };
}).then(({ prod, viaExpress }) => {
  if (standIn) standIn.close();
  cleanUp();
  console.log(`lottery: ${checks} checks passed${prod ? ' (production included)' : ''}${viaExpress ? ' (express included)' : ' (express not installed here: the route checks ran on the stand-in)'}`);
}).catch((e) => {
  console.error(e);
  process.exitCode = 1;
  if (liveStandIn) { liveStandIn.close(); liveStandIn.closeAllConnections(); }
  cleanUp();
});
