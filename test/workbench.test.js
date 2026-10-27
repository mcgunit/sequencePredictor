'use strict';
// The admin Workbench (workbench.js), the scoring it reads (draws.js) and the History ranking as
// it stood before each draw (rankhistory.js): every family's ticket scored on hand-made rows, the keno profit net as
// the History ranking counts it, the git history of a temp repository read once per sha and cached in a file, the
// cut-off of a draw, a version that does not parse skipped, the routes guarded and whitelisted, a point-in-time top
// five that is not today's when the ranking changed, the coming draw on today's ranking, the market workbench's
// sections on a hand-made record - and, when this checkout runs next to production, every view on a copy of
// production's files (its size, the month counts, the Top as it stood before each draw) and the market workbench
// against markets.js's own view. Production is only read: its
// files are copied to a temp dir, its git is read with log and show, the cache file goes to the temp dir.
//   node test/workbench.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const admin = require('../admin');
const brussels = require('../brussels');
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
  // the sales close when it comes first (src/drawClose.json, 27 Oct 2026): keno, pick3 and Joker+ close at 13:00 on Sundays -
  // 18 Oct summer time, 25 Oct (the clocks went back that night) and 1 Nov winter time; every other draw closes after 15:00
  ok(rankhistory.cutoffOf('keno', '2026-10-18') === Date.parse('2026-10-18T11:00:00Z') && rankhistory.cutoffOf('pick3', '2026-10-25') === Date.parse('2026-10-25T12:00:00Z')
    && rankhistory.cutoffOf('jokerplus', '2026-11-01') === Date.parse('2026-11-01T12:00:00Z') && rankhistory.cutoffOf('keno', '2026-10-24') === Date.parse('2026-10-24T13:00:00Z')
    && rankhistory.cutoffOf('vikinglotto', '2026-10-21') === Date.parse('2026-10-21T13:00:00Z') && rankhistory.cutoffOf('crypto', '2026-10-25') === Date.parse('2026-10-25T14:00:00Z'),
    'the cut-off is the sales close when that is earlier: Sunday keno, pick3 and Joker+ at 13:00 Brussels; Saturday keno, vikinglotto and the markets keep 15:00');
  // a report ranked on the live record (rankedOn 'live', Helpers.py from 27 Oct 2026) is reduced with its live values
  const liveReduced = rankhistory.reduce(JSON.stringify({ generatedAt: '2026-10-28T08:00:00', games: {
    keno: { metric: 'profit_bound', minDrawsForRanking: 10, rankedOn: 'live', live: { minDrawsForRanking: 9, models: [] }, models: [
      { name: 'H Model', draws: 260, profit_bound: 0.09, live: { draws: 244, profit_bound: 0.111 } }, { name: 'Q Model', draws: 53, profit_bound: 0.099, live: { draws: 49, profit_bound: 0.001 } }] },
    lotto: { metric: 'avg_hits', minDrawsForRanking: 10, models: [{ name: 'A Model', draws: 20, avg_hits: 1.2, live: { draws: 12, avg_hits: 0.9 } }] } } }));
  const kr = liveReduced.games.keno;
  ok(kr.rankedOn === 'live' && kr.names.join() === 'H Model,Q Model' && kr.values.join() === '0.111,0.001' && kr.draws.join() === '244,49' && kr.minDraws === 9
    && liveReduced.games.lotto.rankedOn === 'all' && liveReduced.games.lotto.values[0] === 1.2 && liveReduced.games.lotto.draws[0] === 20,
    'reduce: a version ranked on the live record keeps its order with the live values, draws and minimum; one before the switch its every-draw ones');
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

  // the live record's markers (data/database/lotteryLiveness.json, src/Liveness.py): 6 Oct back-built (every row first saved
  // after the close), 7 Oct B's row added after the close and G's ticket rewritten after the draw, 10 Oct a second row
  // of C. A row's late and duplicate marks over the draws are its draws minus its live draws in the History ranking - A,
  // D, E, F, G 1 (6 Oct), B 2 (6 and 7 Oct), C 2 (6 Oct, and its second row on 10 Oct counts as a draw and is not
  // live) -; G's recovered 7 Oct is in its live record (live.recovered 1), scored as it stood at the close
  const liveFile = path.join(repo, 'data', 'database', workbench.LIVENESS_FILE);
  const tenth = path.join(repo, 'data', 'database', 'lotto', '2026-10-10.json');
  const tenthText = fs.readFileSync(tenth, 'utf-8');
  fs.writeFileSync(liveFile, JSON.stringify({ games: { lotto: { checked: 'stamp+git', checkedThrough: '2026-10-10', draws: {
    '2026-10-06': { close: '2026-10-06T17:00:00Z', ticketFile: '2026-10-3.json', late: 'all', lateReason: 'first saved after the close', firstSaved: '2026-10-08T07:00:00Z', delayHours: 38 },
    '2026-10-07': { close: '2026-10-07T17:00:00Z', ticketFile: '2026-10-6.json', late: ['B Model'], lateReason: 'row added after the close', firstSaved: '2026-10-07T20:00:00Z', delayHours: 3,
      recovered: { commit: '0123abcd', savedAt: '2026-10-07T08:00:00Z', rows: { 'G Model': { predictions: [[1, 2, 40, 41, 42, 43]], multiPick: [7, 8, 9] } } } },
    '2026-10-10': { close: '2026-10-10T17:00:00Z', ticketFile: '2026-10-7.json', duplicates: { 'C Model': 1 } } } } } }));
  const tenthDay = JSON.parse(tenthText);
  tenthDay.currentPrediction.push({ name: 'C Model', predictions: [[1, 2, 3, 41, 42, 43]] });
  fs.writeFileSync(tenth, JSON.stringify(tenthDay));
  try {
    const markers = {};
    const recovered = {};
    for (const date of ['2026-10-03', '2026-10-06', '2026-10-07', '2026-10-10']) {
      const d = (await request(app, `/api/admin/draw/lotto/${date}`, ADMIN)).body;
      d.rows.filter((r) => r.liveMark).forEach((r) => {
        const into = r.liveMark.status === 'recovered' ? recovered : markers;
        into[r.name] = (into[r.name] || 0) + 1;
      });
    }
    ok(JSON.stringify(markers) === JSON.stringify({ 'A Model': 1, 'B Model': 2, 'C Model': 2, 'D Model': 1, 'E Model': 1, 'F Model': 1, 'G Model': 1 })
      && JSON.stringify(recovered) === '{"G Model":1}',
      `the draws' not-live markers per row are its draws minus its live draws, its recovered ones apart: ${JSON.stringify(markers)} ${JSON.stringify(recovered)}`);
    const d7 = (await request(app, '/api/admin/draw/lotto/2026-10-07', ADMIN)).body;
    const g7 = d7.rows.find((r) => r.name === 'G Model').liveMark;
    const b7 = d7.rows.find((r) => r.name === 'B Model').liveMark;
    ok(g7.status === 'recovered' && g7.commit === '0123abcd' && g7.savedAt === '2026-10-07T08:00:00Z' && g7.atClose.tickets[0].hits.mains === 2 && g7.ticket === undefined
      && b7.status === 'late' && b7.reason === 'row added after the close' && b7.delayHours === 3 && b7.close === '2026-10-07T17:00:00Z'
      && JSON.stringify(d7.liveness.marks) === '{"late":1,"recovered":1}' && d7.liveness.afterClose === true,
      'a draw: the rewritten row with the ticket that stood at the close, scored; the late row with why, when and how late; the counts by status');
    // whether the users' history greys a draw is read on its five whatever the admin's scope: 6 Oct's late rows include
    // the five (A-E), so All says so too
    const all6Live = (await request(app, '/api/admin/draw/lotto/2026-10-06', ADMIN)).body;
    ok(all6Live.liveness.afterClose === true && JSON.stringify(all6Live.liveness.marks) === '{"late":7}', 'All: a draw whose five were late is greyed for the users, its seven rows marked');
    const d10 = (await request(app, '/api/admin/draw/lotto/2026-10-10', ADMIN)).body;
    const cs = d10.rows.filter((r) => r.name === 'C Model');
    ok(cs.length === 2 && cs[0].liveMark === null && cs[1].liveMark.status === 'duplicate', 'a second row of one name on a draw is the not-live one, the first is judged as usual');
    const hl = (await request(app, '/api/admin/history/lotto', ADMIN)).body.months[0].days;
    const hlTop = (await request(app, '/api/admin/history/lotto?scope=top', ADMIN)).body.months[0].days;
    ok(JSON.stringify(hl.find((d) => d.date === '2026-10-06').liveMarks) === '{"late":7}' && JSON.stringify(hl.find((d) => d.date === '2026-10-07').liveMarks) === '{"late":1,"recovered":1}'
      && JSON.stringify(hl.find((d) => d.date === '2026-10-10').liveMarks) === '{"duplicate":1}' && hl.find((d) => d.date === '2026-10-03').liveMarks === undefined
      && JSON.stringify(hlTop.find((d) => d.date === '2026-10-07').liveMarks) === '{"late":1,"recovered":1}' && JSON.stringify(hlTop.find((d) => d.date === '2026-10-06').liveMarks) === '{"late":5}',
      'the history: per draw the not-live rows the scope keeps, by status (Top: only the five\'s)');
  } finally {
    fs.rmSync(liveFile, { force: true });
    fs.writeFileSync(tenth, tenthText);
  }
  // ... and late rows outside the five (7 Oct: D and E), or a draw before the first saved ranking (3 Oct: no five, the
  // users' history never lists it), grey nothing for the users, whatever All marks; with several late rows named the
  // file's one first save is the earliest's, and each mark says how many share it (lateRows)
  fs.writeFileSync(liveFile, JSON.stringify({ games: { lotto: { checked: 'stamp+git', checkedThrough: '2026-10-10', draws: {
    '2026-10-03': { close: '2026-10-03T17:00:00Z', ticketFile: '2026-9-30.json', late: 'all', lateReason: 'first saved after the close' },
    '2026-10-07': { close: '2026-10-07T17:00:00Z', ticketFile: '2026-10-6.json', late: ['D Model', 'E Model'], lateReason: 'row added after the close',
      firstSaved: '2026-10-07T20:00:00Z', delayHours: 3 } } } } }));
  try {
    const d3 = (await request(app, '/api/admin/draw/lotto/2026-10-03', ADMIN)).body;
    const d7 = (await request(app, '/api/admin/draw/lotto/2026-10-07', ADMIN)).body;
    const d7Top = (await request(app, '/api/admin/draw/lotto/2026-10-07?scope=top', ADMIN)).body;
    ok(JSON.stringify(d3.liveness.marks) === '{"late":7}' && d3.liveness.afterClose === false
      && JSON.stringify(d7.liveness.marks) === '{"late":2}' && d7.liveness.afterClose === false && d7Top.liveness.marks === null && d7Top.liveness.afterClose === false
      && d7.rows.filter((r) => r.liveMark).every((r) => r.liveMark.lateRows === 2) && d3.rows.every((r) => r.liveMark.lateRows === null),
      `All marks a late row outside the five, and a draw before the first saved ranking, without saying the users' history greys them: ${JSON.stringify([d3.liveness, d7.liveness])}`);
  } finally {
    fs.rmSync(liveFile, { force: true });
  }

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
// 4. production, read-only: every view on a copy of its files, sizes, the market workbench against markets.js
// ===================================================================================================================
const PROD_ROOT = '/root/sequencePredictor';
async function partFour() {
  if (!fs.existsSync(path.join(PROD_ROOT, 'data', 'database', 'modelPerformance.json'))) return;
  const GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'keno', 'pick3', 'crypto', 'cryptoweek', 'shares', 'sharesweek'];
  const root = path.join(tmp, 'prod');
  // a copy of production's files: the API and markets.js read the same bytes, whatever the daily run does meanwhile
  GAMES.forEach((g) => fs.cpSync(path.join(PROD_ROOT, 'data', 'database', g), path.join(root, 'data', 'database', g), { recursive: true }));
  fs.copyFileSync(path.join(PROD_ROOT, 'data', 'database', 'modelPerformance.json'), path.join(root, 'data', 'database', 'modelPerformance.json'));
  const prodLive = path.join(PROD_ROOT, 'data', 'database', workbench.LIVENESS_FILE);
  if (fs.existsSync(prodLive)) fs.copyFileSync(prodLive, path.join(root, 'data', 'database', workbench.LIVENESS_FILE));
  fs.cpSync(path.join(PROD_ROOT, 'data', 'markets'), path.join(root, 'data', 'markets'), { recursive: true });
  if (fs.existsSync(path.join(PROD_ROOT, 'data', 'controls', 'markets'))) fs.cpSync(path.join(PROD_ROOT, 'data', 'controls', 'markets'), path.join(root, 'data', 'controls', 'markets'), { recursive: true });
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
  for (const game of GAMES) {
    // the history: the newest three months, month by month
    const first = await get(`/api/admin/history/${game}`);
    for (const m of first.months.slice(0, 3)) {
      const body = m.days ? first : await get(`/api/admin/history/${game}?month=${m.month}`);
      const month = body.months.find((x) => x.month === m.month);
      ok(month.draws === month.days.length && month.counted === month.draws && month.rows === month.days.reduce((s, d) => s + d.rows, 0),
        `${game} ${m.month}: the month counts its draws and rows (All)`);
    }
    // the draws: the newest and two older ones, All and Top
    const files = workbench.dayFilesOf(path.join(root, 'data', 'database', game));
    for (const f of [files[0], files[Math.min(5, files.length - 1)], files[Math.min(40, files.length - 1)]]) {
      await get(`/api/admin/draw/${game}/${f.date}`);
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
  // the sales-close cut-off (27 Oct 2026) changes no point-in-time Top: no ranking was ever saved between a Sunday's 13:00
  // close and 15:00 - every lottery draw reads the same version under the old 15:00 rule and the new one
  const hist = await rankhistory.load(PROD_ROOT, { cacheFile: path.join(tmp, 'prod-cache.json') });
  let sameVersion = 0;
  const moved = [];
  for (const game of GAMES.filter((g) => !['crypto', 'cryptoweek', 'shares', 'sharesweek'].includes(g))) {
    for (const f of workbench.dayFilesOf(path.join(root, 'data', 'database', game))) {
      const old = hist.rankingAt(game, brussels.instant(f.date, 15, 0));
      const now = hist.rankingAt(game, rankhistory.cutoffOf(game, f.date));
      if ((old.version ? old.version.sha : old.none) === (now.version ? now.version.sha : now.none)) sameVersion += 1; else moved.push(`${game} ${f.date}`);
    }
  }
  ok(moved.length === 0 && sameVersion > 1000, `production: every lottery draw (${sameVersion}) reads the same ranking version under the sales-close cut-off as under 15:00 (${moved.join(', ')})`);
  // once production has a liveness file: each row's not-live markers over its draws are its draws minus its live draws
  if (fs.existsSync(prodLive)) {
    const report = JSON.parse(fs.readFileSync(path.join(root, 'data', 'database', 'modelPerformance.json'), 'utf-8'));
    for (const game of ['lotto', 'vikinglotto', 'jokerplus']) {
      const info = report.games[game];
      if (!info || !info.models.some((m) => m.live_draws !== undefined)) continue;
      const marks = {};
      const recovered = {};
      const dirs = admin.resolveDirs({ root, dataDir: path.join(root, 'data') });
      const files = workbench.dayFilesOf(path.join(root, 'data', 'database', game));
      for (const f of files) {
        workbench.drawView(dirs, game, f.date, 'all', hist, report).rows.filter((r) => r.liveMark).forEach((r) => {
          const into = r.liveMark.status === 'recovered' ? recovered : marks;
          into[r.name] = (into[r.name] || 0) + 1;
        });
      }
      const off = info.models.filter((m) => (marks[m.name] || 0) !== m.draws - m.live_draws || (recovered[m.name] || 0) !== ((m.live && m.live.recovered) || 0))
        .map((m) => `${m.name} ${marks[m.name] || 0} vs ${m.draws - m.live_draws}`);
      ok(off.length === 0, `production ${game}: every row's not-live markers are its draws minus its live draws, its recovered ones its live.recovered (${off.slice(0, 3).join('; ')})`);
    }
  }
  const jokerTop = await get('/api/admin/history/jokerplus?scope=top');
  ok(jokerTop.ranking.first.committedAt === '2026-08-25T05:21:45Z' && jokerTop.ranking.firstOfGame && jokerTop.ranking.firstOfGame.committedAt.startsWith('2026-09-07'),
    `production: Joker+ entered the ranking after the first saved one, and the history says when: ${JSON.stringify(jokerTop.ranking.firstOfGame)}`);

  // the market workbench against markets.js's reading of the same copy (describeMarket with the week game, the rows
  // report and the regime log; pickPlan, pickWeek), as the retired /markets pages read it
  const marketsDir = path.join(root, 'data', 'markets');
  for (const market of ['crypto', 'shares']) {
    const w = await get(`/api/admin/market/${market}/workbench`);
    const weekRecord = markets.loadMarket(marketsDir, markets.WEEK_GAMES[market]);
    const rec = markets.describeMarket(markets.loadMarket(marketsDir, market), { rows: markets.loadRows(path.join(root, 'data', 'controls'), market),
      regimes: markets.loadRegimes(marketsDir, market), week: weekRecord ? markets.describeMarket(weekRecord) : null });
    if (w.plan) ok(w.plan.model === markets.pickPlan(rec, market).model.name, `${market}: Today's plan follows ${w.plan.model}, as markets.pickPlan`);
    if (w.week) ok(rec.week && w.week.model === markets.pickWeek(rec.week, market).model.name, `${market}: the week card follows ${w.week.model}, as markets.pickWeek`);
    const fd = w.followed.day;
    if (fd && fd.liveDays) {
      const f = rec.trading && rec.trading.followed;
      ok(f && fd.firstLive === f.firstLive && f.points.filter((p) => p.live).length === fd.liveDays && fd.positions === f.livePositions && fd.total === Number(f.liveTotal.toFixed(2))
        && fd.live.length === fd.liveDays && w.plan.soFar.liveDays === fd.live.length, `${market}: Following the plan as the record: since ${fd.firstLive}, ${fd.liveDays} live, ${fd.positions} positions, ${fd.total}`);
    }
    ok(w.scoredDays === rec.scoredDays && w.models.length === rec.models.length,
      `${market}: the models table, ${w.models.length} rows over ${w.scoredDays} scored days, as the record`);
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
