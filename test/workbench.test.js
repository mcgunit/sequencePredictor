'use strict';
// The admin Workbench (workbench.js), the scoring it shares with the old pages (draws.js) and the History ranking as
// it stood before each draw (rankhistory.js): every family's ticket scored on hand-made rows, the keno profit net as
// the History ranking counts it, the git history of a temp repository read once per sha and cached in a file, the
// cut-off of a draw, a version that does not parse skipped, the routes guarded and whitelisted, a point-in-time top
// five that is not today's when the ranking changed, the coming draw on today's ranking, the market workbench's
// sections on a hand-made record - and, when this checkout runs next to production, the figures of the old pages
// (server.js /database/:folder and /database/:folder/:file, rendered from a copy of production's files) against the
// API's, every view's size, and the market workbench against the old /markets pages. Production is only read: its
// files are copied to a temp dir, its git is read with log and show, the cache file goes to the temp dir.
//   node test/workbench.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { execFileSync } = require('child_process');
const admin = require('../admin');
const draws = require('../draws');
const gates = require('../gates');
const markets = require('../markets');
const rankhistory = require('../rankhistory');
const workbench = require('../workbench');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'workbench-'));
const cleanUp = () => fs.rmSync(tmp, { recursive: true, force: true });

// --- a tiny stand-in for express (as test/admin.test.js): app.use(prefix, fn) and app.get(pattern, fn), in order ------
function fakeApp() {
  const stack = [];
  return { stack, use(prefix, fn) { stack.push({ prefix, fn }); }, get(pattern, fn) { stack.push({ pattern, fn }); } };
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
function request(app, url, user) {
  return new Promise((resolve) => {
    const [pathname, qs] = url.split('?');
    const res = { statusCode: 200, headers: {}, body: undefined,
      status(c) { this.statusCode = c; return this; }, set(k, v) { this.headers[k] = v; return this; }, json(b) { this.body = b; resolve(this); return this; } };
    const req = { user, query: Object.fromEntries(new URLSearchParams(qs || '')), params: {}, path: pathname };
    let i = 0;
    const next = () => {
      while (i < app.stack.length) {
        const layer = app.stack[i++];
        if (layer.prefix !== undefined) {
          if (pathname === layer.prefix || pathname.startsWith(`${layer.prefix}/`)) return layer.fn(req, res, next);
        } else {
          const params = matchRoute(layer.pattern, pathname);
          if (params) { req.params = params; return layer.fn(req, res, next); }
        }
      }
      res.statusCode = 404; res.body = 'fell through'; resolve(res);
      return undefined;
    };
    next();
  });
}
const ADMIN = { name: 'admin', role: 'admin', open: false };
const USER = { name: 'reader', role: 'user', open: false };

// ===================================================================================================================
// 1. draws.js: one ticket per family
// ===================================================================================================================
{
  const lotto = draws.scoreRow('lotto', { name: 'L', predictions: [[1, 2, 7, 40, 41, 42]], multiPick: [3, 44, 45] }, [1, 2, 3, 4, 5, 6, 7]);
  ok(lotto.tickets[0].mark === 'mmb---' && lotto.tickets[0].hits.mains === 2 && lotto.tickets[0].hits.specials === 1 && lotto.tickets[0].specials === undefined
    && lotto.multiPick.mark === 'm--' && lotto.multiPick.nine === 3, `lotto: two mains, the bonus amber ("2 (1)"), the multi-pick's third number a hit, 3 of nine: ${JSON.stringify(lotto)}`);
  const em = draws.scoreTicket('euromillions', [1, 2, 10, 11, 12, 7, 9], [1, 2, 3, 4, 5, 6, 7]);
  const emOnly = draws.scoreTicket('euromillions', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]);
  ok(em.mark === 'mm---' && em.markSpecials === 's-' && em.hits.mains === 2 && em.hits.specials === 1 && JSON.stringify(emOnly.specials) === '[]' && emOnly.hits.mains === 5,
    'euromillions: a star lights only among the stars; a mains-only row has no special cells');
  const jp = draws.scoreTicket('jokerplus', [1, 2, 9, 9, 5, 6, 4], [1, 2, 3, 4, 5, 6, 4]);
  ok(jp.mark === 'll--rr' && jp.markSpecials === 'z' && jp.hits.left === 2 && jp.hits.right === 2 && jp.hits.sign === 1 && jp.signName === 'Leeuw'
    && jp.profit === 2 * 5 + 1.5 - 1.5 && jp.stake === 1.5 && jp.profit === draws.jokerplusTicketProfit([1, 2, 9, 9, 5, 6, 4], [1, 2, 3, 4, 5, 6, 4]),
    `Joker+: runs from either end and the sign, by its name; the money as the payout table: ${JSON.stringify(jp)}`);
  const real20 = Array.from({ length: 20 }, (_, i) => i + 1);
  const k5 = draws.scoreTicket('keno', [1, 2, 3, 4, 5], real20);
  const k10 = draws.scoreTicket('keno', [21, 22, 23, 24, 25, 26, 27, 28, 29, 30], real20);
  const k6 = draws.scoreTicket('keno', [1, 2, 30, 31, 32, 33], real20);
  const kRank = draws.scoreTicket('keno', real20.slice(0, 19).concat([70]), real20);
  ok(k5.profit === 149 && k10.profit === 2 && k6.profit === -1 && k5.stake === 1 && kRank.kind === 'ranking' && kRank.profit === null && kRank.stake === 0 && kRank.hits.mains === 19,
    'keno: net per ticket as Helpers.keno_ticket_profit (5 of 5 pays 150, a profit of 149; 0 of 10 pays 3, a profit of 2), the 20-number row a ranking, not played');
  ok(draws.calculateProfit([1, 1, 2, 3, 4], real20, 'keno') === draws.kenoTicketProfit([1, 1, 2, 3, 4], real20) && draws.kenoTicketProfit([1, 1, 2, 3, 4], real20) === 5 - 1
    && draws.calculateProfit([1, 2, 3], real20, 'keno') === 0, 'keno: a number counted once (a 5-ticket with 1 twice: 4 hits pay 5, not 5 hits 150), a size without a tracked profit 0');
  const p3 = draws.scoreTicket('pick3', [1, 2, 2], [1, 2, 2]);
  const p3box = draws.scoreTicket('pick3', [2, 2, 1], [1, 2, 2]);
  const p3triple = draws.scoreTicket('pick3', [2, 2, 2], [1, 2, 2]);
  ok(p3.profit === 500 + 160 + 50 + 50 - 4 && JSON.stringify(p3.bets) === '{"straight":500,"consolation":0,"box":160,"frontPair":50,"backPair":50}'
    && p3box.profit === 160 - 4 && p3box.hits.positional === 1 && p3box.hits.membership === 3 && p3box.mark === '-m-' && p3box.markMember === 'mmm'
    && p3triple.stake === 3 && p3triple.profit === 1 + 50 - 3, 'pick3: every bet on its own and cumulated, the cells by position, the old membership hits kept apart');
  const mk = draws.scoreTicket('crypto', [3, 7, 5], [3, 5, 7]);
  ok(mk.mark === 'm--' && mk.hits.mains === 1 && mk.profit === undefined, 'a market: the right bin in the right slot only, no money on the draw');
  const coming = draws.scoreTicket('euromillions', [1, 2, 3, 4, 5, 6, 7], [], draws.splitRealResult([], 'euromillions'), [9, 9, 9, 9, 9]);
  const comingRl = draws.scoreTicket('euromillions', [1, 2, 3, 4, 5], [], draws.splitRealResult([], 'euromillions'), [9, 9, 9, 9, 9]);
  ok(coming.mark === undefined && JSON.stringify(coming.specials) === '[6,7]' && JSON.stringify(comingRl.specials) === '[]',
    'the coming draw: no hits, a row split against the newest drawn row (a mains-only RL row keeps its five mains)');

  // one day's figures and a month's, over the kept rows
  const day = { realResult: [1, 2, 3, 4, 5, 6, 0], currentPrediction: [
    { name: 'A', predictions: [[1, 2, 9, 9, 9, 6, 0]] }, { name: 'B', predictions: [[1, 2, 3, 4, 9, 9, 5]] }, { name: 'C', predictions: [[9, 9, 9, 9, 9, 9, 0]] }] };
  const all = draws.dayFigures('jokerplus', day);
  const ab = draws.dayFigures('jokerplus', day, (n) => n !== 'B');
  ok(all.rows === 3 && all.tickets === 3 && all.stake === 4.5 && all.profit === (5 + 2 + 1.5 - 1.5) + (200 - 1.5) + (1.5 - 1.5) && all.best.left === 4 && all.best.by === 'B'
    && ab.rows === 2 && ab.best.left === 2 && ab.best.right === 1 && ab.best.sign === 1 && ab.profit === 7,
    `a day's figures over the kept rows only, the best line L/R (Z): ${JSON.stringify(all)} / ${JSON.stringify(ab)}`);
  const months = draws.historyMonths('jokerplus', [
    { date: '2026-10-02', day, keep: () => true }, { date: '2026-10-01', day, keep: null, ranking: { none: 'before-first' } }, { date: '2026-09-30', day, keep: (n) => n === 'C' }]);
  ok(months.length === 2 && months[0].draws === 2 && months[0].counted === 1 && months[0].profit === all.profit && months[0].days[1].rows === undefined
    && months[0].days[1].ranking.none === 'before-first' && months[1].profit === 0 && months[1].best.left === 0 && months[0].days[0].result.signName === 'Ram',
    'a month: its totals over the draws with rows in scope only; a draw without a ranking has no figures and says why');
  const lv = draws.votesOf('pick3', [{ name: 'A', predictions: [[1, 2, 3]] }, { name: 'B', predictions: [[1, 5, 3]] }], [1, 2, 9], { 1: 0.5, 2: 0.25, x: 1 });
  ok(lv.kind === 'slots' && JSON.stringify(lv.slots[0]) === '{"drawn":1,"counts":[[1,2]]}' && JSON.stringify(lv.slots[1].counts) === '[[2,1],[5,1]]' && lv.stored.values.length === 2,
    'the vote of a positional game per slot, with the digit drawn there; the stored vote kept apart');
  ok(draws.isoOfFile('2026-9-5.json') === '2026-09-05' && draws.isoOfFile('2026-2-30.json') === null && draws.fileOfIso('2026-09-05') === '2026-9-5.json'
    && draws.nextDrawDay(['2026-10-14.json', '2026-10-17.json', '2026-10-21.json'], '2026-10-21.json') === '2026-10-24', 'day files: ISO days both ways, the next draw on the game\'s schedule');
}

// ===================================================================================================================
// 2. rankhistory.js on a temp git repository
// ===================================================================================================================
const repo = path.join(tmp, 'repo');
fs.mkdirSync(path.join(repo, 'data', 'database', 'lotto'), { recursive: true });
const gitIn = (args, when) => String(execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', ...args],
  { cwd: repo, stdio: 'pipe', env: { PATH: process.env.PATH, HOME: process.env.HOME || '/root', GIT_AUTHOR_DATE: when, GIT_COMMITTER_DATE: when } }));
const REPORT = path.join(repo, 'data', 'database', 'modelPerformance.json');
const NAMES = ['A Model', 'B Model', 'C Model', 'D Model', 'E Model', 'F Model', 'G Model'];
const reportOf = (order, generatedAt) => JSON.stringify({ generatedAt, games: { lotto: { metric: 'avg_hits', minDrawsForRanking: 10,
  models: order.map((n, i) => ({ name: n, draws: 20, avg_hits: 2 - i / 10 })) } } });
const commitReport = (order, when, text = null) => {
  fs.writeFileSync(REPORT, text === null ? reportOf(order, when.replace(/Z$/, '')) : text);
  gitIn(['add', '-A'], when);
  gitIn(['commit', '-q', '-m', 'saving last predictions'], when);
  return gitIn(['rev-parse', 'HEAD'], when).trim();
};
const V1 = ['A Model', 'B Model', 'C Model', 'D Model', 'E Model', 'F Model', 'G Model'];      // top: A-E
const V2 = ['G Model', 'F Model', 'A Model', 'B Model', 'C Model', 'D Model', 'E Model'];      // top: G F A B C
const V3 = ['F Model', 'G Model', 'E Model', 'D Model', 'C Model', 'B Model', 'A Model'];      // the hand run in the evening
const V5 = ['D Model', 'E Model', 'F Model', 'G Model', 'A Model', 'B Model', 'C Model'];      // HEAD: top D E F G A
const TODAY = ['C Model', 'B Model', 'G Model', 'A Model', 'F Model', 'E Model', 'D Model'];   // the working tree, uncommitted
const cacheFile = path.join(tmp, 'config', 'rankhistory-cache.json');
async function partTwo() {
  gitIn(['init', '-q'], '2026-10-01T00:00:00Z');
  fs.writeFileSync(path.join(repo, 'README'), 'x');
  gitIn(['add', 'README'], '2026-10-01T00:00:00Z');
  gitIn(['commit', '-q', '-m', 'start'], '2026-10-01T00:00:00Z');
  const s1 = commitReport(V1, '2026-10-05T07:30:00Z');    // 09:30 Brussels, Monday
  commitReport(V2, '2026-10-07T07:30:00Z');               // Wednesday morning
  const s3 = commitReport(V3, '2026-10-07T16:30:00Z');    // 18:30 Brussels: a hand run, after the cut-off of that day
  const s4 = commitReport(null, '2026-10-08T07:30:00Z', '{"generatedAt": "2026-10-08T07:29:00", "games": {"lotto": {"metric": "avg_hits", "x": NaN}}}');
  ok(rankhistory.cutoffOf('lotto', '2026-10-07') === Date.parse('2026-10-07T13:00:00Z') && rankhistory.cutoffOf('lotto', '2026-11-04') === Date.parse('2026-11-04T14:00:00Z')
    && rankhistory.cutoffOf('cryptoweek', '2026-10-11') === Date.parse('2026-10-05T13:00:00Z') && rankhistory.cutoffOf('sharesweek', '2026-10-09') === Date.parse('2026-10-05T13:00:00Z')
    && rankhistory.cutoffOf('lotto', '2026-02-30') === null, 'the cut-off: 15:00 Brussels on the draw\'s day (summer and winter time), on the week\'s Monday for a week game');
  let h = await rankhistory.load(repo, { cacheFile });
  ok(h.error === null && h.versions.length === 3 && h.skipped === 1 && h.first.committedAt === '2026-10-05T07:30:00Z' && h.first.sha === s1,
    `the versions of the report, oldest first; the one that does not parse skipped: ${h.versions.length} read, ${h.skipped} skipped`);
  const before = h.topAt('lotto', '2026-10-04', NAMES);
  ok(before.none === 'before-first' && /2026-10-05/.test(before.text) && before.top === undefined, `a draw before the first saved ranking has none, and says so: ${before.text}`);
  ok(h.topAt('lotto', '2026-10-05', NAMES).top.join() === V1.slice(0, 5).join() && h.topAt('lotto', '2026-10-06', NAMES).top.join() === V1.slice(0, 5).join(),
    'the morning run of a draw\'s day counts for that day');
  ok(h.topAt('lotto', '2026-10-07', NAMES).top.join() === V2.slice(0, 5).join() && h.topAt('lotto', '2026-10-08', NAMES).top.join() === V3.slice(0, 5).join()
    && h.topAt('lotto', '2026-10-08', NAMES).ranking.version.sha === s3,
    'a hand run in the evening counts from the next day on; the next morning\'s version does not parse, so the one before it stands (as ranking.js kept it)');
  ok(h.topAt('lotto', '2026-10-06', NAMES.filter((n) => n !== 'A Model')).top.join() === 'B Model,C Model,D Model,E Model,F Model'
    && h.topAt('lotto', '2026-10-06', ['X Model']).none === 'no-ranked-row' && h.topAt('euromillions', '2026-10-06', NAMES).none === 'not-ranked',
    'the top five among the names in that day\'s table; none ranked, or a game not in the ranking yet, said as such');
  const cached = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));
  ok(Object.keys(cached.versions).length === 4 && cached.versions[s4].skip === 'not a report' && cached.versions[s1].games.lotto.names.join() === V1.join(),
    'the cache file: every version reduced, keyed by sha, the unreadable one as a fact');
  // a new commit: read incrementally once HEAD moves
  const s5 = commitReport(V5, '2026-10-12T07:30:00Z');
  gates.forgetHeads();
  h = await rankhistory.load(repo, { cacheFile });
  ok(h.versions.length === 4 && h.versions[3].sha === s5 && Object.keys(JSON.parse(fs.readFileSync(cacheFile, 'utf-8')).versions).length === 5,
    'a new commit is read once HEAD moves, and added to the cache file');
  // the cache file is read before git: a version there is not read again
  rankhistory.reset();
  const doctored = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));
  doctored.versions[s1].games.lotto.names = ['Z Model'].concat(V1);
  doctored.versions[s1].games.lotto.values.unshift(3);
  doctored.versions[s1].games.lotto.draws.unshift(20);
  const doctoredFile = path.join(tmp, 'doctored.json');
  fs.writeFileSync(doctoredFile, JSON.stringify(doctored));
  h = await rankhistory.load(repo, { cacheFile: doctoredFile });
  ok(h.topAt('lotto', '2026-10-06', ['Z Model', ...NAMES]).top[0] === 'Z Model', 'a version in the cache file is taken from it, not from git');
  // an entry that parses but has the wrong shape (a hand edit, reduce() changed without a version bump) is read again
  rankhistory.reset();
  const broken = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));
  broken.versions[s1] = { generatedAt: '2026-10-05T07:29:00Z' };
  const brokenFile = path.join(tmp, 'broken.json');
  fs.writeFileSync(brokenFile, JSON.stringify(broken));
  h = await rankhistory.load(repo, { cacheFile: brokenFile });
  ok(h.error === null && h.versions.length === 4 && h.topAt('lotto', '2026-10-06', NAMES).top.join() === V1.slice(0, 5).join()
    && JSON.parse(fs.readFileSync(brokenFile, 'utf-8')).versions[s1].games.lotto.names.join() === V1.join(), 'a cache entry of the wrong shape is read again from git, and the file mended');
  // a version git could not read: the draws it decides have no Top (never the version before it), and it is not asked
  // again on every request
  rankhistory.reset();
  gates.forgetHeads();
  const realGit = gates.git;
  let shows = 0;
  gates.git = (root, args) => {
    if (args[0] === 'show') { shows += 1; if (args[1].startsWith(s3)) return Promise.reject(Object.assign(new Error('timeout'), { killed: true, code: null })); }
    return realGit(root, args);
  };
  try {
    h = await rankhistory.load(repo, { cacheFile: null });
    const gap = h.topAt('lotto', '2026-10-08', NAMES);
    ok(typeof h.error === 'string' && h.versions.length === 3 && gap.none === 'unavailable' && gap.top === undefined
      && h.topAt('lotto', '2026-10-07', NAMES).top.join() === V2.slice(0, 5).join() && h.topAt('lotto', '2026-10-12', NAMES).ranking.version.sha === s5,
      `an unread version leaves the draws it decides without a Top, the others read as usual: ${gap.text}`);
    shows = 0;
    const again = await rankhistory.load(repo, { cacheFile: null });
    ok(shows === 0 && again.topAt('lotto', '2026-10-08', NAMES).none === 'unavailable', 'the unread version is not asked again on the next request');
  } finally { gates.git = realGit; }
  rankhistory.reset();
  gates.forgetHeads();
  const none = await rankhistory.load(path.join(tmp, 'no-repo'), { cacheFile: null });
  ok(none.versions.length === 0 && typeof none.error === 'string' && none.topAt('lotto', '2026-10-06', NAMES).none === 'unavailable',
    `no git history: no Top, said why in a fixed phrase: ${none.error}`);
  rankhistory.reset();
  gates.forgetHeads();
  // today's report: the working tree, not yet committed
  fs.writeFileSync(REPORT, reportOf(TODAY, '2026-10-12T09:00:00'));
}

// ===================================================================================================================
// 3. the routes, on the temp repository's files
// ===================================================================================================================
const ticket = (name) => (name === 'G Model' ? [1, 2, 3, 4, 5, 6] : [1, 2, 40, 41, 42, 43]);
function writeLottoDays() {
  const dir = path.join(repo, 'data', 'database', 'lotto');
  const dayFile = (date, extra = {}) => fs.writeFileSync(path.join(dir, date), JSON.stringify({ realResult: [1, 2, 3, 4, 5, 6, 7],
    currentPrediction: NAMES.map((name) => ({ name, predictions: [ticket(name)], multiPick: [7, 8, 9] })), currentNumberFrequency: { 1: 0.5, 44: 0.5 },
    newPrediction: NAMES.map((name) => ({ name, predictions: [[10, 11, 12, 13, 14, 15]] })), numberFrequency: { 10: 1 }, ...extra }));
  dayFile('2026-10-3.json');    // before the first ranking
  dayFile('2026-10-6.json');    // V1: A-E, G not among them
  dayFile('2026-10-7.json');    // V2: G among them
  dayFile('2026-10-10.json', { madeAt: '2026-10-11T07:10:00+00:00' });   // the newest: V3 for its draw, today's report for the coming one
}
async function partThree() {
  writeLottoDays();
  const app = fakeApp();
  admin.install(app, { root: repo, dataDir: path.join(repo, 'data'), rankCacheFile: cacheFile });
  const urls = ['/api/admin/history/lotto', '/api/admin/history/lotto?scope=top', '/api/admin/draw/lotto/2026-10-06', '/api/admin/draw/lotto/2026-10-06?scope=top',
    '/api/admin/market/crypto/workbench', '/api/admin/market/crypto/workbench/books/daily', '/api/admin/market/crypto/workbench/instrument/BTC'];
  for (const url of urls) {
    const signedOut = await request(app, url, undefined);
    const u = await request(app, url, USER);
    ok(signedOut.statusCode === 401 && u.statusCode === 403 && u.body.error === 'admin only' && u.headers['Cache-Control'] === 'no-store', `${url}: 401 signed out, 403 for a user`);
  }
  for (const url of urls.slice(0, 4)) {
    const a = await request(app, url, ADMIN);
    ok(a.statusCode === 200 && a.headers['Cache-Control'] === 'no-store' && !/NaN|undefined/.test(JSON.stringify(a.body)), `${url}: 200 for the administrator`);
  }
  const bad = [['/api/admin/history/bingo', 404], ['/api/admin/history/..', 404], ['/api/admin/history/lotto?scope=some', 400], ['/api/admin/history/lotto?month=2026-1', 400],
    ['/api/admin/history/lotto?month=2025-01', 404], ['/api/admin/history/keno', 404], ['/api/admin/draw/bingo/2026-10-06', 404], ['/api/admin/draw/lotto/2026-13-01', 400],
    ['/api/admin/draw/lotto/..', 400], ['/api/admin/draw/lotto/2026-10-04', 404], ['/api/admin/draw/lotto/2026-10-06?scope=x', 400], ['/api/admin/market/bingo/workbench', 404],
    ['/api/admin/market/crypto/workbench', 404], ['/api/admin/market/lotto/workbench', 404]];
  for (const [url, status] of bad) {
    const r = await request(app, url, ADMIN);
    ok(r.statusCode === status && typeof r.body.error === 'string', `${url}: ${status} with its reason (${r.body && r.body.error})`);
  }
  ok((await request(app, '/api/admin/draw/lotto/2026-10-6', ADMIN)).body.date === '2026-10-06', 'a date without its zeros is the same day');

  // the point-in-time top five, never today's
  const todayTop = TODAY.slice(0, 5);
  const top6 = (await request(app, '/api/admin/draw/lotto/2026-10-06?scope=top', ADMIN)).body;
  ok(top6.rows.map((r) => r.name).join() === V1.slice(0, 5).join() && top6.rows.map((r) => r.name).sort().join() !== todayTop.slice().sort().join()
    && top6.scope.ranking.committedAt === '2026-10-05T07:30:00Z' && top6.scope.cutoff === '2026-10-06T13:00:00Z' && top6.scope.rows === 5 && top6.scope.of === 7,
    `Top on a past draw: the five ranked first before it (the 5 Oct version), not today's five: ${top6.rows.map((r) => r.name).join(', ')}`);
  ok(top6.figures.best.mains === 2 && top6.figures.rows === 5 && top6.votes.stored === undefined && top6.votes.rows === 5 && top6.scoredIn === '2026-10-07' && top6.coming === null,
    'its figures over those five only (G, the six-hit row, was not among them); no vote of every row; an older day names the draw its next ticket was scored in');
  const all6 = (await request(app, '/api/admin/draw/lotto/2026-10-06', ADMIN)).body;
  ok(all6.rows.length === 7 && all6.figures.best.mains === 6 && all6.figures.best.by === 'G Model' && all6.rows.find((r) => r.name === 'G Model').rank === 7
    && all6.rows.filter((r) => r.top).length === 5 && all6.votes.stored.values.length === 2 && all6.multiPick.sizes.length === 4 && all6.rows[0].multiPick.nine === 2,
    'All: every row, each with its place in that ranking and whether it was in that top five; the stored vote; the multi-pick legend once');
  const top7 = (await request(app, '/api/admin/draw/lotto/2026-10-07?scope=top', ADMIN)).body;
  ok(top7.rows.map((r) => r.name).join() === 'A Model,B Model,C Model,F Model,G Model' && top7.figures.best.mains === 6, 'the next draw: the ranking of its own day');
  const top3 = (await request(app, '/api/admin/draw/lotto/2026-10-03?scope=top', ADMIN)).body;
  ok(top3.rows.length === 0 && top3.figures === null && top3.scope.none.code === 'before-first' && top3.scope.rows === 0, 'before the first saved ranking: no Top rows and no figures, said why');
  const newest = (await request(app, '/api/admin/draw/lotto/2026-10-10?scope=top', ADMIN)).body;
  ok(newest.newest && newest.rows.map((r) => r.name).join() === 'C Model,D Model,E Model,F Model,G Model' && newest.coming.scope.top.join() === todayTop.join()
    && newest.coming.rows.map((r) => r.name).sort().join() === todayTop.slice().sort().join() && newest.coming.scope.ranking.today === true
    && newest.coming.date === '2026-10-13' && newest.coming.madeAt === '2026-10-11T07:10:00Z' && newest.coming.votes.stored === undefined,
    'the newest day: its own draw on the ranking of its day, the coming draw on today\'s (the working tree\'s report, not HEAD\'s)');

  // the history: by month, the newest open; Top over each draw's own five
  const hTop = (await request(app, '/api/admin/history/lotto?scope=top', ADMIN)).body;
  const oct = hTop.months[0];
  ok(hTop.draws === 4 && hTop.counted === 3 && oct.draws === 4 && oct.counted === 3 && oct.best.mains === 6 && oct.rows === 15
    && oct.days.find((d) => d.date === '2026-10-06').best.mains === 2 && oct.days.find((d) => d.date === '2026-10-03').ranking.none === 'before-first'
    && oct.days.find((d) => d.date === '2026-10-03').rows === undefined && hTop.ranking.versions[oct.days.find((d) => d.date === '2026-10-06').ranking.v].committedAt === '2026-10-05T07:30:00Z',
    `history, Top: each draw over its own top five, the draw before the first ranking without figures, the month over the counted draws: ${JSON.stringify(oct.days.map((d) => [d.date, d.best && d.best.mains]))}`);
  const hAll = (await request(app, '/api/admin/history/lotto', ADMIN)).body;
  ok(hAll.counted === 4 && hAll.months[0].rows === 28 && hAll.months[0].days.every((d) => d.best.mains === 6 && d.multi === 6 && d.ranking) && hAll.month === '2026-10'
    && hAll.ranking.metrics.avg_hits === 'average hits', 'history, All: every row of every draw');

  // the controls drawer names the market rows that failed and why (C16 of the parity: only a count was sent)
  fs.mkdirSync(path.join(repo, 'data', 'controls', 'markets'), { recursive: true });
  fs.writeFileSync(path.join(repo, 'data', 'controls', 'markets', 'crypto-rows.json'), JSON.stringify({ market: 'crypto', reference: 'GARCH Model', uniform_log_score: -2.3026,
    rows: [{ name: 'GARCH Model', kind: 'market', log_score: -2.29 }], errors: { 'Chronos Model': 'no forecast on any day', 'TimesFM Model': 3 } }));
  const drawer = (await request(app, '/api/admin/controls/crypto', ADMIN)).body;
  ok(drawer.rows.errors === 2 && JSON.stringify(drawer.rows.failed) === '[{"row":"Chronos Model","reason":"no forecast on any day"},{"row":"TimesFM Model","days":3}]',
    `the controls drawer: the failed market rows by name, with the days they failed or why they scored none: ${JSON.stringify(drawer.rows.failed)}`);

  // the market workbench on a hand-made record
  await marketPart(app);
}

async function marketPart(app) {
  const mdir = path.join(repo, 'data', 'markets');
  fs.mkdirSync(mdir, { recursive: true });
  const edges = [-0.03, -0.02, -0.01, -0.004, 0.0, 0.003, 0.009, 0.015, 0.025];
  const fpoint = (date, model, live, pnl, market) => [date, model, 'book', 1, pnl, 0, live, market, {}];
  const record = {
    market: 'crypto', generated_at: '2026-10-05T07:15:00+00:00', k: 10, fee: 0.001, chance: { exact: 0.1, adjacent: 0.28, direction: 0.5 }, newest_game_day: '2026-10-04',
    instruments: [{ symbol: 'BTC', name: 'Bitcoin', position: 0, quote: 'USDT', active: true, last_close: 84000, last_date: '2026-10-04',
      closes: [['2026-10-02', 83000], ['2026-10-03', 83500], ['2026-10-04', 84000]], edges: [null, edges, edges], moves: [null, 0.006, 0.006],
      course: { 'Up Model': [null, 7, 6], 'Down Model': [null, null, 1] } }],
    models: [{ name: 'Up Model', days: 12, positions: 12, exact_rate: 0.2, adjacent_rate: 0.3, direction_rate: 0.6, trades: 12, pnl_cash_total: 5, pnl_cash_per_trade: 0.42, win_rate: 0.5,
      hold_total: 4, hold_trades: 5, hold_win_rate: 0.6, hold_per_trade: 0.8 },
    { name: 'Down Model', days: 3, positions: 3, exact_rate: 0, adjacent_rate: 0, direction_rate: 0.3, trades: 0, pnl_cash_total: 0, hold_total: 0 }],
    drawn_models: ['Up Model'], best_model: 'Up Model',
    trading: { stake: 100, fee_per_leg: 0.001, currency: 'USDT', rule: 'long when up', hold_rule: 'kept while up', dates: ['2026-10-03', '2026-10-04'],
      models: { 'Up Model': [['2026-10-03', 2, 2], ['2026-10-04', 3, 5]] }, benchmark: [['2026-10-03', 1, 1], ['2026-10-04', -2, -1]],
      hold: { models: { 'Up Model': [['2026-10-03', 1, 1], ['2026-10-04', 3, 4]] }, benchmark: [['2026-10-03', 1, 1], ['2026-10-04', 1, 2]] },
      followed: { rule: 'daily', min_trades: 10, first_live: '2026-10-02', total: 4, market_total: 1, live_total: 3, live_market: -1, live_days: 2, live_positions: 2,
        points: [fpoint('2026-10-01', 'Up Model', false, 1, 2), fpoint('2026-10-02', 'Up Model', true, 2, 1), fpoint('2026-10-03', 'Up Model', false, 5, 0), fpoint('2026-10-04', 'Up Model', true, 1, -2)] } },
    next: { made_on: '2026-10-04', made_at: '2026-10-05T07:15:00+00:00', instruments: { BTC: { last_close: 84000, last_date: '2026-10-04', predictions: [
      { model: 'Up Model', bin: 7, direction: 1, price: 84900, low: 84500, high: 85300 }, { model: 'Down Model', bin: 0, direction: -1, price: 80000, low: null, high: 81000 }] } } },
    daily: [{ date: '2026-10-04', exact_mean: 0.5, best_exact: 1, best_model: 'Up Model' }],
    days: [{ date: '2026-10-04', best: 'Up Model', exact_mean: 0.5, instruments: [{ symbol: 'BTC', return: 0.006, bin: 6, edges }],
      models: [{ name: 'Up Model', bins: [6], exact: 1, adjacent: 1, direction: 1, positions: 1, pnl: 0.005, pnl_cash: 0.5, trades: 1 },
        { name: 'Down Model', bins: [1], exact: 0, adjacent: 0, direction: 0, positions: 1, pnl: 0, pnl_cash: 0, trades: 0 },
        { name: 'RL Position Model', bins: [3], sizes: [0.5], exact: 0, adjacent: 0, direction: 0, positions: 1, pnl: 0.003, pnl_cash: 0.2, trades: 1 }] }],
  };
  fs.writeFileSync(path.join(mdir, 'crypto.json'), JSON.stringify(record));
  const view = markets.describeMarket(record, { week: null });
  const w = workbench.workbenchOf('crypto', view, new Date('2026-10-05T10:00:00Z'));
  ok(w.plan.day === '2026-10-05' && w.plan.model === 'Up Model' && w.plan.byBook === true && w.plan.book.total === 5 && w.plan.book.market === -1 && w.plan.book.ahead === 6
    && w.plan.rows[0].order === 'buy-now' && w.plan.rows[0].call.word === 'up' && w.plan.rows[0].up === 1 && w.plan.rows[0].of === 2 && w.plan.rows[0].call.move.low === 0.006,
    `Today's plan: the model followed by its book, the book against buying everything, the call and the order now: ${JSON.stringify(w.plan.rows[0])}`);
  ok(w.plan.record.backBuilt === 1 && w.plan.record.live === 2 && w.plan.record.late === 1 && w.plan.soFar.liveDays === 2 && w.plan.soFar.total === 3 && w.plan.soFar.ahead === 4,
    'the plan\'s record: back-built, live and late days, counted as the followed table counts them');
  const late = workbench.planOf('crypto', view, new Date('2026-10-06T01:00:00Z'));
  ok(late.closed && late.rows[0].order === 'too-late', 'after the day\'s close the plan says too late');
  const f = w.followed.day;
  ok(f.live.length === f.liveDays && f.live.map((p) => p[0]).join() === '2026-10-02,2026-10-04' && f.backBuilt.n === 1 && f.backBuilt.total === 1 && f.late.n === 1 && f.late.total === 5
    && f.columns.join() === 'date,model,pickedBy,positions,pnl,market' && w.followed.week === null, 'Following the plan: the live days, the late and the back-built apart with their totals');
  ok(w.books.rules.join() === 'daily,hold' && w.books.headline.daily.best.model === 'Up Model' && w.books.headline.daily.market === -1 && w.books.headline.open === null
    && w.models.length === 2 && w.models[0].daily.total === 5 && w.models[0].hold.trades === 5 && w.accuracy.days.length === 1 && w.days[0].instruments[0].bin === 6
    && w.example.best.name === 'Up Model' && w.example.instruments[0].close.before === 83500 && w.guard === null && w.week === null && w.judged.day === '2026-10-05',
    'the paper books\' headline (a 3-day row cannot head a book), the models, accuracy, the days, the worked example');
  const books = (await request(app, '/api/admin/market/crypto/workbench/books/daily', ADMIN)).body;
  ok(books.dates.length === 2 && books.series['Up Model'].join() === '2,5' && books.market.join() === '1,-1' && books.order.join() === 'Up Model'
    && (await request(app, '/api/admin/market/crypto/workbench/books/open', ADMIN)).statusCode === 404
    && (await request(app, '/api/admin/market/crypto/workbench/books/weekly', ADMIN)).statusCode === 404, 'a book\'s series; a rule this market has not, or no rule at all, a 404');
  const btc = (await request(app, '/api/admin/market/crypto/workbench/instrument/BTC', ADMIN)).body;
  ok(btc.dates.length === 3 && btc.from === 1 && btc.edges.length === 2 && btc.course['Up Model'].join() === '7,6' && btc.course['Down Model'][0] === null
    && btc.next.find((p) => p.model === 'Up Model').up === true && btc.best === 'Up Model' && (await request(app, '/api/admin/market/crypto/workbench/instrument/ETH', ADMIN)).statusCode === 404,
    'an instrument: closes and moves, the edges and every model\'s bin from the first tracked day, the next calls');
  const day = (await request(app, '/api/admin/market/crypto/workbench', ADMIN)).body;
  ok(day.market === 'crypto' && day.plan.model === 'Up Model' && !/NaN|undefined/.test(JSON.stringify(day)), 'the workbench route');
  // a settled day's draw (the old Day by day, C14): the close before the day, which turns a bin's edges into prices, and
  // the sized row's size per slot, which its cells read instead of the bin's half
  const ddir = path.join(repo, 'data', 'database', 'crypto');
  fs.mkdirSync(ddir, { recursive: true });
  fs.writeFileSync(path.join(ddir, '2026-10-4.json'), JSON.stringify({ realResult: [6],
    currentPrediction: [{ name: 'Up Model', predictions: [[6]] }, { name: 'RL Position Model', predictions: [[3]] }], newPrediction: [] }));
  const drawn = (await request(app, '/api/admin/draw/crypto/2026-10-04?scope=all', ADMIN)).body;
  const dm = drawn.market;
  ok(dm && dm.instruments[0].before === 83500 && dm.instruments[0].edges.length === 9 && dm.models['RL Position Model'].sizes.join() === '0.5'
    && dm.models['Up Model'].sizes === null && dm.models['Down Model'] === undefined && !/NaN|undefined/.test(JSON.stringify(drawn)),
    `a market draw: the close before the day for the bins' prices, the sized row's size per slot, only the rows shown: ${JSON.stringify(dm)}`);
}

// ===================================================================================================================
// 4. production, read-only: the old pages' figures against the API, sizes, the market workbench against the old page
// ===================================================================================================================
const PROD_ROOT = '/root/sequencePredictor';
// server.js from a copy, its side-effect modules stubbed (as test/pages.test.js); its data folder is a copy of production's
function oldRenderer(root) {
  fs.copyFileSync(path.join(__dirname, '..', 'server.js'), path.join(root, 'server.js'));
  fs.mkdirSync(path.join(root, 'config'), { recursive: true });
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
  const REPO = path.join(__dirname, '..');
  Module.prototype.require = function stubbed(id) {
    if (Object.prototype.hasOwnProperty.call(stubs, id)) return stubs[id];
    if (id.startsWith('./') && this.filename === path.join(root, 'server.js')) return original.call(this, path.join(REPO, id));
    return original.apply(this, arguments);
  };
  const listeners = ['SIGTERM', 'SIGINT', 'exit', 'uncaughtException', 'unhandledRejection'].map((e) => [e, process.listeners(e).slice()]);
  try { require(path.join(root, 'server.js')); } finally {
    Module.prototype.require = original;
    listeners.forEach(([e, before]) => process.listeners(e).filter((l) => !before.includes(l)).forEach((l) => process.removeListener(e, l)));
  }
  return (route, params) => {
    const layer = gets.find(([p]) => (Array.isArray(p) ? p.includes(route) : p === route));
    let html = '';
    const res = { send: (x) => { html = x; }, status() { return this; }, redirect: noop, set() { return this; }, json: noop, sendFile: noop };
    layer[1][layer[1].length - 1]({ user: { name: 'admin', role: 'admin', open: false, csrf: 'x' }, params, query: {}, session: {}, originalUrl: route, get: () => '' }, res);
    return html;
  };
}
const text = (html) => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
// the old folder page: per day file its line, per month its header
function oldFolder(html) {
  const days = {};
  for (const m of html.matchAll(/<a href="\/database\/[^/"]+\/([^"]+)"[^>]*>📄 [^<]*<\/a>\s*<span[^>]*>([\s\S]*?)<\/span>\s*<\/li>/g)) days[draws.isoOfFile(m[1])] = text(m[2]);
  const months = {};
  for (const m of html.matchAll(/<span class="card-title">(\d{4}-\d{2})<\/span><span[^>]*>([^<]*)<\/span>/g)) months[m[1]] = text(m[2]);
  return { days, months };
}
// the API's day line and month header in the old page's words
const fmtProfit = (game, x) => (game === 'jokerplus' ? Number(x).toFixed(2) : String(x));
function newDayLine(game, d) {
  const b = d.best || { mains: 0, specials: 0, left: 0, right: 0, sign: 0 };
  const supplement = draws.SPECIAL_COLUMN_COUNTS[game] || game === 'lotto';
  const match = game === 'jokerplus' ? `Match: ${b.left}/${b.right} (${b.sign})` : `Match: ${b.mains}${supplement ? ` (${b.specials})` : ''}`;
  const money = game === 'keno' || game === 'pick3' || game === 'jokerplus';
  const stat = money ? (game === 'jokerplus' ? `${match} · ${fmtProfit(game, d.profit)} €` : `${fmtProfit(game, d.profit)} €`) : match;
  return stat + (d.multi !== undefined ? ` · of 9: ${d.multi}` : '');
}
function newMonthLine(game, m) {
  const b = m.best || { mains: 0, specials: 0, left: 0, right: 0, sign: 0 };
  const supplement = draws.SPECIAL_COLUMN_COUNTS[game] || game === 'lotto';
  const best = game === 'jokerplus' ? `Best Match: ${b.left}/${b.right} (${b.sign})` : `Best Match: ${b.mains}${supplement ? ` (${b.specials})` : ''}`;
  if (game === 'jokerplus') return `Total: ${fmtProfit(game, m.profit)} € · ${best}`;
  if (game === 'keno' || game === 'pick3') return `Total: ${fmtProfit(game, m.profit)} €`;
  return best;
}
// the old draw page's Analysis table: per data row its cells' text
function oldAnalysis(html) {
  const start = html.indexOf('Analysis of Prediction');
  const section = html.slice(start, html.indexOf('Next Draw Prediction', start));
  const tableAt = section.indexOf('<table border="1">');
  if (tableAt < 0) return [];   // "No predictions." (a game's first day)
  const table = section.slice(tableAt, section.indexOf('</table>', tableAt));
  return table.split('<tr').slice(1).filter((tr) => !/<th/.test(tr)).map((tr) => [...tr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((m) => text(m[1])));
}
function newAnalysis(game, body) {
  const out = [];
  body.rows.forEach((r) => r.tickets.forEach((t, j) => {
    const cells = game === 'jokerplus' ? [...t.mains, ...(t.specials.length ? [t.signName] : [])] : [...t.mains, ...(t.specials || [])];
    let hits;
    if (game === 'jokerplus') hits = `${t.hits.left}/${t.hits.right} (${t.hits.sign})`;
    else if (game === 'pick3') hits = `${t.hits.membership}`;
    else if (t.hits.specials !== undefined) hits = `${t.hits.mains} (${t.hits.specials})${j === 0 && r.multiPick && r.multiPick.nine !== undefined ? ` · ${r.multiPick.nine}` : ''}`;
    else hits = `${t.hits.mains}`;
    const extras = j === 0 && r.multiPick ? r.multiPick.numbers : null;
    out.push({ name: r.name, cells: cells.map(String), extras, hits, profit: t.profit === undefined ? undefined : `${fmtProfit(game, t.profit === null ? 0 : t.profit)} €` });
  }));
  return out;
}

async function partFour() {
  if (!fs.existsSync(path.join(PROD_ROOT, 'data', 'database', 'modelPerformance.json'))) return;
  const GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'keno', 'pick3', 'crypto', 'cryptoweek', 'shares', 'sharesweek'];
  const root = path.join(tmp, 'prod');
  // a copy of production's files: the old renderer and the API read the same bytes, whatever the daily run does meanwhile
  GAMES.forEach((g) => fs.cpSync(path.join(PROD_ROOT, 'data', 'database', g), path.join(root, 'data', 'database', g), { recursive: true }));
  fs.copyFileSync(path.join(PROD_ROOT, 'data', 'database', 'modelPerformance.json'), path.join(root, 'data', 'database', 'modelPerformance.json'));
  fs.cpSync(path.join(PROD_ROOT, 'data', 'markets'), path.join(root, 'data', 'markets'), { recursive: true });
  if (fs.existsSync(path.join(PROD_ROOT, 'data', 'controls', 'markets'))) fs.cpSync(path.join(PROD_ROOT, 'data', 'controls', 'markets'), path.join(root, 'data', 'controls', 'markets'), { recursive: true });
  const render = oldRenderer(root);
  const app = fakeApp();
  // the API on the copy; the ranking from production's git (read-only), its cache in the temp dir
  admin.install(app, { root: PROD_ROOT, dataDir: path.join(root, 'data'), rankCacheFile: path.join(tmp, 'prod-cache.json') });
  const sizes = [];
  const get = async (url) => {
    const r = await request(app, url, ADMIN);
    const body = JSON.stringify(r.body);
    sizes.push([url, body.length]);
    ok(r.statusCode === 200 && !/NaN|undefined/.test(body), `production ${url}: ${r.statusCode} ${r.statusCode !== 200 ? body.slice(0, 200) : ''}`);
    ok(body.length < 60 * 1024, `production ${url}: ${body.length} bytes, under 60 KB`);
    return r.body;
  };
  let compared = { days: 0, months: 0, tickets: 0 };
  for (const game of GAMES) {
    // the history: the newest three months, day by day and month by month, as the old page prints them
    const old = oldFolder(render('/database/:folder', { folder: game }));
    const first = await get(`/api/admin/history/${game}`);
    for (const m of first.months.slice(0, 3)) {
      const body = m.days ? first : await get(`/api/admin/history/${game}?month=${m.month}`);
      const month = body.months.find((x) => x.month === m.month);
      ok(newMonthLine(game, month) === old.months[m.month], `${game} ${m.month}: the month header "${old.months[m.month]}" = "${newMonthLine(game, month)}"`);
      compared.months += 1;
      month.days.forEach((d) => {
        assert.strictEqual(newDayLine(game, d), old.days[d.date], `${game} ${d.date}: the day line`);
        compared.days += 1; checks += 1;
      });
      ok(month.draws === month.days.length && month.counted === month.draws && month.rows === month.days.reduce((s, d) => s + d.rows, 0),
        `${game} ${m.month}: the month counts its draws and rows (All)`);
    }
    // the draws: the newest and two older ones, row by row and ticket by ticket
    const files = workbench.dayFilesOf(path.join(root, 'data', 'database', game));
    for (const f of [files[0], files[Math.min(5, files.length - 1)], files[Math.min(40, files.length - 1)]]) {
      const body = await get(`/api/admin/draw/${game}/${f.date}`);
      const oldRows = oldAnalysis(render('/database/:folder/:file', { folder: game, file: f.file }));
      const newRows = newAnalysis(game, body);
      ok(oldRows.length === newRows.length, `${game} ${f.date}: ${oldRows.length} tickets on the old page, ${newRows.length} in the API`);
      newRows.forEach((n, i) => {
        const o = oldRows[i];
        const money = n.profit !== undefined;
        const hitsAt = o.length - (money ? 2 : 1);
        const cellsOld = o.slice(2, 2 + n.cells.length);
        assert.deepStrictEqual([o[0], cellsOld, o[hitsAt], money ? o[o.length - 1] : undefined], [n.name, n.cells, n.hits, n.profit], `${game} ${f.date} ticket ${i}`);
        if (n.extras) assert.deepStrictEqual(o.slice(2 + n.cells.length, 2 + n.cells.length + 3), n.extras.map(String), `${game} ${f.date} ticket ${i}: the multi-pick cells`);
        compared.tickets += 1; checks += 1;
      });
      const top = await get(`/api/admin/draw/${game}/${f.date}?scope=top`);
      ok(top.rows.length === (top.scope.none ? 0 : top.scope.top.length) && top.scope.top.length <= 5 && top.rows.every((r) => r.top && top.scope.top.includes(r.name)) && top.scope.rows === top.rows.length
        && (!top.figures || top.figures.rows === top.rows.length), `${game} ${f.date}, Top: ${top.rows.length} rows, ${top.scope.none ? top.scope.none.text : `ranked on ${top.scope.ranking.committedAt}`}`);
      if (top.scope.ranking) ok(Date.parse(top.scope.ranking.committedAt) < Date.parse(top.scope.cutoff), `${game} ${f.date}: the version read was committed before the cut-off`);
    }
    const hTop = await get(`/api/admin/history/${game}?scope=top`);
    ok(hTop.counted <= hTop.draws && hTop.months[0].days.every((d) => (d.ranking.none ? d.rows === undefined : d.rows <= 5)), `${game}: Top counts ${hTop.counted} of ${hTop.draws} draws`);
  }
  // the first saved ranking is 25 Aug 2026: nothing before it has a Top
  const kenoAug = await get('/api/admin/history/keno?scope=top&month=2026-08');
  ok(kenoAug.months.find((m) => m.month === '2026-08').days.filter((d) => d.date < '2026-08-25').every((d) => d.ranking.none === 'before-first')
    && kenoAug.ranking.first.committedAt === '2026-08-25T05:21:45Z', 'production: no ranking recorded before 25 Aug 2026, said per draw');
  const jokerTop = await get('/api/admin/history/jokerplus?scope=top');
  ok(jokerTop.ranking.first.committedAt === '2026-08-25T05:21:45Z' && jokerTop.ranking.firstOfGame && jokerTop.ranking.firstOfGame.committedAt.startsWith('2026-09-07'),
    `production: Joker+ entered the ranking after the first saved one, and the history says when: ${JSON.stringify(jokerTop.ranking.firstOfGame)}`);
  console.log(`workbench on production: ${compared.days} day lines, ${compared.months} months, ${compared.tickets} tickets equal to the old pages`);

  // the market workbench against the old /markets pages
  for (const market of ['crypto', 'shares']) {
    const w = await get(`/api/admin/market/${market}/workbench`);
    const html = render(`/markets/${market}`, {});
    if (w.plan) ok(html.includes(`The table shows <b>${w.plan.model}</b>'s calls`), `${market}: Today's plan follows ${w.plan.model}, as the old page`);
    if (w.week) ok(html.includes(`<b>${w.week.model}</b>'s calls for the week`), `${market}: the week card follows ${w.week.model}, as the old page`);
    const fd = w.followed.day;
    if (fd && fd.liveDays) {
      const m = /Followed since <b>([^<]+)<\/b> \((\d+) [a-z]+, (\d+) positions?\):\s*<b[^>]*>([+-][\d.]+)/.exec(html);
      ok(m && m[1] === fd.firstLive && Number(m[2]) === fd.liveDays && Number(m[3]) === fd.positions && Number(m[4]) === fd.total && fd.live.length === fd.liveDays
        && w.plan.soFar.liveDays === fd.live.length, `${market}: Following the plan as the old page: ${m && m[0]}`);
    }
    const modelsHtml = html.slice(html.indexOf('Models over '), html.indexOf('</table>', html.indexOf('Models over ')));
    ok(modelsHtml.startsWith(`Models over ${w.scoredDays} scored day(s)`) && (modelsHtml.match(/<tr><td style="text-align:left; font-weight:bold;">/g) || []).length === w.models.length,
      `${market}: the models table, ${w.models.length} rows over ${w.scoredDays} scored days, as the old page`);
    for (const rule of w.books.rules) {
      const b = await get(`/api/admin/market/${market}/workbench/books/${rule}`);
      const head = w.books.headline[rule.replace(/_ls$/, '') === rule ? rule : 'longShort'];
      ok(b.dates.length === w.books.days && Object.values(b.series).every((s) => s.length === b.dates.length)
        && (rule.endsWith('_ls') || head.market === b.market[b.market.length - 1]), `${market} ${rule}: the series aligned to the book's dates, its market line ends at the headline's`);
    }
    for (const inst of w.instruments) {
      const c = await get(`/api/admin/market/${market}/workbench/instrument/${inst.symbol}`);
      ok(c.edges.length === c.dates.length - c.from && Object.values(c.course).every((s) => s.length === c.dates.length - c.from) && c.closes.length === c.dates.length,
        `${market} ${inst.symbol}: closes, edges and bins aligned`);
    }
    if (w.guard) ok(w.guard.live.weeks === w.guard.weeks.filter((x) => x.live).length, `${market}: the guard's totals count its live weeks`);
    if (w.guard && w.followed.week && w.guard.weeks.filter((x) => x.live).map((x) => x.date).sort().join() === w.followed.week.live.map((p) => p[0]).join()) {
      ok(w.guard.live.plain === w.followed.week.total, `${market}: the guard's plain week total over the live weeks is Following the plan's (${w.guard.live.plain})`);
    }
    ok(w.days.length <= 30 && w.accuracy.days.length > 0, `${market}: day by day and accuracy`);
    const d0 = await get(`/api/admin/draw/${market}/${w.days[0].date}`);
    ok(d0.market && d0.market.instruments.map((i) => i.bin).join() === d0.result.mains.join() && Object.keys(d0.market.models).length === d0.rows.length,
      `${market} ${w.days[0].date}: the draw's market section - the bins are the drawn row, every shown row's money`);
  }
  const biggest = sizes.sort((a, b) => b[1] - a[1])[0];
  console.log(`workbench on production: ${sizes.length} views, the largest ${biggest[0]} at ${biggest[1]} bytes`);
}

// --- the same routes through express itself, when it can be loaded (NODE_PATH) ----------------------------------------
async function throughExpress() {
  let express = null;
  try { express = require('express'); } catch (e) { return false; }
  const real = express();
  real.use((req, res, next) => { const who = req.get('x-test-user'); req.user = who === 'admin' ? ADMIN : (who === 'user' ? USER : undefined); next(); });
  admin.install(real, { root: repo, dataDir: path.join(repo, 'data'), rankCacheFile: cacheFile });
  await new Promise((resolve) => {
    const server = real.listen(0, '127.0.0.1', async () => {
      const base = `http://127.0.0.1:${server.address().port}`;
      try {
        const get = async (url, who) => { const r = await fetch(base + url, { headers: who ? { 'x-test-user': who } : {} }); return { status: r.status, body: await r.json() }; };
        ok((await get('/api/admin/draw/lotto/2026-10-06?scope=top', 'user')).status === 403 && (await get('/api/admin/history/lotto')).status === 401, 'express: 403 for a user, 401 signed out');
        const r = await get('/api/admin/draw/lotto/2026-10-06?scope=top', 'admin');
        ok(r.status === 200 && r.body.rows.length === 5, 'express: the scope read from the query');
        ok((await get('/api/admin/history/..%2F..%2Fetc', 'admin')).status === 404 && (await get('/api/admin/draw/lotto/2026-10-06%2F..', 'admin')).status === 400,
          'express: an encoded path is a name, never a path');
      } catch (e) { console.error(e); process.exitCode = 1; } finally { server.close(); resolve(); }
    });
  });
  return true;
}

partTwo().then(partThree).then(partFour).then(throughExpress).then((viaExpress) => {
  cleanUp();
  console.log(`workbench: ${checks} checks passed${viaExpress ? ' (express included)' : ' (express not installed here: the route checks ran on the stand-in)'}`);
}).catch((e) => {
  console.error(e);
  process.exitCode = 1;
  cleanUp();
});
