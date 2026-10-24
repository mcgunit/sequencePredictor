'use strict';
// The admin section's JSON (admin.js) and the shared History ranking (ranking.js): the guard (401 signed out, 403 for
// a user, through for the administrator), every route's shape on hand-made records written to a temp dir, the rules
// the builders apply (Q0 hits with hits, the live split, the keno bound against a fair ticket, the combination's p
// value, the lag trail, the tripwires, the top-five scope), the cache by file change, the ranking against the one
// server.js used before it moved, and - when the checkout runs next to production - every view on production's own
// files, read-only, with its size.
//   node test/admin.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const admin = require('../admin');
const rankings = require('../ranking');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- a tiny stand-in for express: app.use(prefix, fn) and app.get(pattern, fn), run in order --------------------
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
function dispatch(app, url, user, done = () => {}) {
  const [pathname, qs] = url.split('?');
  const res = { statusCode: 200, headers: {}, body: undefined,
    status(c) { this.statusCode = c; return this; }, set(k, v) { this.headers[k] = v; return this; }, json(b) { this.body = b; done(this); return this; } };
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
    res.statusCode = 404; res.body = 'fell through';
    done(res);
    return undefined;
  };
  next();
  return res;
}
const request = (app, url, user) => dispatch(app, url, user);
// the routes that wait on git or a probe answer later: resolved when they do
const requestAsync = (app, url, user) => new Promise((resolve) => { dispatch(app, url, user, resolve); });
const ADMIN = { name: 'admin', role: 'admin', open: false };
const USER = { name: 'reader', role: 'user', open: false };
const OPEN = { name: 'open access', role: 'admin', open: true };

// --- the guard alone ------------------------------------------------------------------
{
  const res = () => ({ statusCode: 200, headers: {}, status(c) { this.statusCode = c; return this; }, set(k, v) { this.headers[k] = v; return this; }, json(b) { this.body = b; return this; } });
  let passed = 0;
  const r1 = res(); admin.guard({ user: USER }, r1, () => { passed += 1; });
  ok(r1.statusCode === 403 && r1.body.error === 'admin only' && r1.headers['Cache-Control'] === 'no-store' && passed === 0, 'a signed-in user gets the JSON 403');
  const r2 = res(); admin.guard({}, r2, () => { passed += 1; });
  ok(r2.statusCode === 401 && r2.body.error === 'login required' && passed === 0, 'no session at all is a 401, never data');
  admin.guard({ user: ADMIN }, res(), () => { passed += 1; });
  admin.guard({ user: OPEN }, res(), () => { passed += 1; });
  ok(passed === 2, 'the administrator and open access go through');
}

// --- hand-made records ---------------------------------------------------------------------
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'admin-'));
const dataDir = path.join(tmp, 'data');
['database', 'controls/results', 'controls/discrimination', 'controls/features', 'controls/markets', 'markets'].forEach((d) => fs.mkdirSync(path.join(dataDir, d), { recursive: true }));
const write = (rel, obj) => fs.writeFileSync(path.join(dataDir, rel), JSON.stringify(obj));

const lottoRow = (name, draws, avgHits, bestHits, extra = {}) => ({ name, draws, avg_hits: avgHits, profit_bound: null, avg_special_hits: null, best_hits: bestHits, profit_total: null, profit_per_bet: null, bets: 0, profit_by_size: null, multi_pick: null, ...extra });
const lagHistory = (lags) => lags.map((lag, i) => ({ run: `2026-10-${String(i + 1).padStart(2, '0')}`, lag, avg_hits: 1 + i / 100, n: 80 + i, z: 2 + i / 10, profile_mean: 0.8, metric: 'main_hits' }));
const lag = (lags) => { const h = lagHistory(lags); const counts = {}; h.forEach((r) => { counts[r.lag] = (counts[r.lag] || 0) + 1; }); const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return { peak: h[h.length - 1], history: h, runs: h.length, lag_counts: counts, consensus_lag: Number(best[0]), consensus_runs: best[1], best_lag: h[h.length - 1].lag }; };
const report = {
  generatedAt: '2026-10-21T08:48:01', since: { date: '2026-09-27', label: 'September 2026 design freeze' },
  games: {
    lotto: {
      metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'A Model',
      models: [
        lottoRow('A Model', 14, 1.3, 3, { multi_pick: { draws: 4, avg_hits: 1.5, base_avg_hits: 1.25, win_rate: 0.25, base_win_rate: 0, extra_hit_rates: [0.25, 0, 0] } }),
        lottoRow('B Model', 30, 1.2, 4), lottoRow('C Model', 90, 1.0, 3), lottoRow('D Model', 90, 0.9, 3), lottoRow('E Model', 90, 0.85, 2),
        lottoRow('F Model', 90, 0.8, 2, { multi_pick: { draws: 4, avg_hits: 1, base_avg_hits: 1, win_rate: 0, base_win_rate: 0, extra_hit_rates: [0, 0, 0] } }),
        lottoRow('Young Model', 3, 2, 3),
      ],
      multi_pick: { extra: 3, draw: 6, min_hits: 3, grid_price: 1.5, per_extra_chance: 0.13333, chance: { 6: { win: 0.02383, grids: 1, price: 1.5 }, 9: { win: 0.08395, grids: 84, price: 126 } }, rows_with_extras: 2 },
      since: { date: '2026-09-27', label: 'September 2026 design freeze', metric: 'avg_hits', minDrawsForRanking: 6, bestModel: 'B Model',
        models: [{ name: 'B Model', draws: 6, avg_hits: 2, best_hits: 4 }, { name: 'A Model', draws: 6, avg_hits: 1.5, best_hits: 3 }] },
      combinations: { metric: 'avg_best_hits', minDrawsForRanking: 10, candidateRows: ['A Model', 'B Model', 'C Model'], evaluated: 4,
        best: { members: ['A Model', 'B Model'], size: 2, how: 'pair', draws: 14, avg_hits: 1.1, avg_best_hits: 1.9, best_hits: 3, value: 1.9 },
        ranking: [{ members: ['A Model', 'B Model'], size: 2, how: 'pair', draws: 14, avg_hits: 1.1, avg_best_hits: 1.9, best_hits: 3, value: 1.9 }],
        bestSingle: { name: 'A Model', value: 1.3, draws: 14 }, beatsBestSingle: true, control: { shuffles: 100, observed: 1.9, best_mean: 1.77, best_p95: 2, p_value: 0.188 } },
      lagAnalysis: { 'A Model': lag([5, 5, 5, 7, 5]), 'Young Model': lag([3, 9]), 'Gone Model': lag([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        // scanned on 21 Oct with no new draw (the same n): no run, its peak dates from the newest recorded run
        'Bygone Model': (() => { const l = lag([4, 4, 4]); return { ...l, peak: { ...l.peak, run: '2026-10-21' } }; })() },
      anomalyWatch: { date: '2026-10-17', latest_z: 1.0459, min_z_recent: -1.224, alert: false },
      randomnessWatch: { window: 60, draws_total: 98, kl_vs_history: 0.0189, kl_vs_uniform: 0.0669, entropy_norm: 0.94,
        trend: [{ end_date: '2026-09-12', kl_vs_history: 0.02, kl_vs_uniform: 0.08, entropy_norm: 0.97 }, { end_date: '2026-10-17', kl_vs_history: 0.0189, kl_vs_uniform: 0.0669, entropy_norm: 0.94 }],
        model_kl_vs_real: { 'C Model': 0.3, 'A Model': 0.1, 'Z Model': 0.2 } },
    },
    keno: {
      metric: 'profit_bound', minDrawsForRanking: 10, bestModel: 'K1 Model',
      models: [
        { name: 'K1 Model', draws: 48, avg_hits: 6.02, profit_bound: 0.117, best_hits: 11, profit_total: -22, profit_per_bet: -0.195, bets: 113,
          profit_by_size: { 5: { bets: 13, profit_total: -11, profit_per_bet: -0.846 }, 10: { bets: 48, profit_total: -16, profit_per_bet: -0.333 } } },
        { name: 'K2 Model', draws: 300, avg_hits: 6.3, profit_bound: 0.04, best_hits: 12, profit_total: 80, profit_per_bet: 0.15, bets: 560, profit_by_size: null },
      ],
      combinations: { metric: 'profit_per_draw', minDrawsForRanking: 10, candidateRows: ['K1 Model', 'K2 Model'], evaluated: 1,
        best: { members: ['K1 Model', 'K2 Model'], size: 2, how: 'pair', draws: 48, profit_per_bet: 1.2, profit_per_draw: 44.4, win_day_rate: 0.24, avg_hits: 6, best_hits: 11, value: 44.4 },
        ranking: [{ members: ['K1 Model', 'K2 Model'], size: 2, how: 'pair', draws: 48, profit_per_bet: 1.2, profit_per_draw: 44.4, win_day_rate: 0.24, avg_hits: 6, best_hits: 11, value: 44.4 }],
        bestSingle: { name: 'K2 Model', value: 21.3, draws: 48 }, beatsBestSingle: true, control: { shuffles: 100, best_mean: 33.8, best_p95: 99.7, p_value: 0.297 } },
    },
    crypto: {
      metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'L Model',
      models: [lottoRow('L Model', 20, 1.25, 3), lottoRow('M Model', 50, 0.94, 3), lottoRow('N Model', 50, 0.5, 2)],
      since: { date: '2026-09-27', label: 'September 2026 design freeze', metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'L Model',
        models: [{ name: 'L Model', draws: 20, avg_hits: 1.25 }, { name: 'M Model', draws: 20, avg_hits: 0.9 }] },
      anomalyWatch: { date: '2026-10-17', latest_z: -3.5, min_z_recent: -3.5, alert: true },
    },
  },
};
write('database/modelPerformance.json', report);
// crypto's newest day file has no M Model row: the users' top rows are L and N (a ranked row missing from today's
// table is passed over, as /lottery does); an older day still has it
fs.mkdirSync(path.join(dataDir, 'database', 'crypto'));
write('database/crypto/2026-10-9.json', { newPrediction: ['L Model', 'M Model', 'N Model'].map((name) => ({ name, predictions: [[1]] })) });
write('database/crypto/2026-10-20.json', { newPrediction: ['N Model', 'L Model'].map((name) => ({ name, predictions: [[1]] })) });
const q0Report = (game, mode, best, sd, perRow, at) => ({ report: { rows_compared: 8, row_mean: 0.78, best_row_mean: best, selection_gain: 0.1, random_baseline: 0.79, best_row_sd_across_seeds: sd,
  per_row_mean: perRow, game, mode, seeds: [1, 2, 3], days: 120, metric: 'hits per draw', winner_per_seed: ['C Model', 'C Model', 'D Model'], generated_at: at }, per_seed: [] });
write('controls/results/lotto-synthetic.json', q0Report('lotto', 'synthetic', 0.95, 0.0878, { 'C Model': 0.7, 'D Model': 0.68 }, '2026-10-18T10:11:55'));
write('controls/results/lotto-shuffled.json', q0Report('lotto', 'shuffled', 0.8778, 0.0293, { 'C Model': 0.75 }, '2026-10-18T10:23:00'));
write('controls/results/keno-synthetic.json', q0Report('keno', 'synthetic', 6.0, 0.08, { 'K2 Model': 5.9 }, '2026-10-18T16:17:37'));
write('controls/results/crypto-shuffled.json', q0Report('crypto', 'shuffled', 0.7, 0.05, { 'M Model': 0.642 }, '2026-10-18T22:09:59'));
write('controls/discrimination/lotto-w10.json', { game: 'lotto', window: 10, draws: 1000, repetitions: 3, seeds: [1, 2, 3], quantum: true, generated_at: '2026-10-18T08:48:16',
  shape: { windows_train: 1486, windows_test: 496, features: 45 },
  verdict: { metric: 'best AUC', real_max_of_suite_mean: 0.5902, real_max_of_suite_sd: 0.0139, null_max_of_suite_mean: 0.509, null_max_of_suite_sd: 0.0306, shuffled_max_of_suite_mean: 0.5419,
    shuffled_max_of_suite_sd: 0.0205, null_threshold_2sd: 0.5703, per_seed_real: [0.58, 0.61, 0.58], per_seed_null: [0.52, 0.48, 0.53], per_seed_shuffled: [0.55, 0.52, 0.56],
    above_null_band: true, survives_marginals_control: true, too_few_draws: false, windows_test: 496 } });
write('controls/discrimination/crypto-w10.json', { game: 'crypto', window: 10, draws: 1000, repetitions: 5, seeds: [1, 2, 3, 4, 5], quantum: true, generated_at: '2026-10-18T08:56:23',
  shape: { windows_train: 300, windows_test: 106, features: 50 },
  verdict: { real_max_of_suite_mean: 0.959, real_max_of_suite_sd: 0.01, null_max_of_suite_mean: 0.7, null_max_of_suite_sd: 0.036, shuffled_max_of_suite_mean: 0.942,
    null_threshold_2sd: 0.772, above_null_band: true, survives_marginals_control: true, too_few_draws: false, windows_test: 106 } });
const variant = (share, fits, above) => ({ noise: { share_of_train_importance: share, noise_to_real_ratio: fits ? 1.03 : 0.05, train_t: fits ? 2.5 : 0.4, heldout_band: 0.0082 },
  columns: Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`c${i}`, 0])), real_above_noise_band: above, heldout_auc_without_noise: 0.5012, heldout_auc_with_noise_mean: 0.4987,
  heldout_cost_of_noise: 0.0025, fits_noise: fits, inconclusive: false, seconds: 12.3, rule: 'the rule' });
write('controls/features/lotto.json', { game: 'lotto', generated_at: '2026-10-18T09:10:00+00:00', repeats: 3, noise_columns: 3,
  table: { days: 275, behind_file_by: 1, lockbox_days_withheld: 25, positional: false }, variants: { logistic: variant(0.3117, true, []), quantum_kernel: variant(0.19, false, ['C Model']) } });
write('controls/markets/crypto-rows.json', { market: 'crypto', generated_at: '2026-10-18T09:54:09+00:00', days_requested: 250, days_scored: 231, first_day: '2026-03-01', last_day: '2026-10-17',
  uniform_log_score: -2.3026, probability_floor: 0.001, reference: 'GARCH Model', lockbox_days_withheld: 19, hmm_features: 'returns', errors: {},
  rows: [
    { name: 'GARCH Model', kind: 'market', days: 231, log_score: -2.21, log_score_se: 0.0137, vs_uniform: { mean: 0.0923, lo: 0.0647, hi: 0.1183, days: 231, verdict: 'better' }, vs_reference: null, exact_rate: 0.15, pnl_return: -0.14 },
    { name: 'Chronos Model', kind: 'foundation', days: 231, log_score: -2.25, log_score_se: 0.02, vs_uniform: { mean: 0.05, lo: 0.01, hi: 0.09, verdict: 'better' }, vs_reference: { mean: -0.04, lo: -0.08, hi: 0.0, verdict: 'worse' }, exact_rate: 0.12 },
    { name: 'Uniform', kind: 'baseline', days: 231, log_score: null, exact_rate: 0.1 },
  ] });
write('markets/crypto-regimes.json', { updated: '2026-10-20', 'Regime HMM Model': [{ date: '2026-10-20', regimes: 5, features: ['returns'], volatility_rank: 1, label: 'calm', probability: 0.8985, expected_return: [0.001], volatility: [0.02] }] });

// --- the ranking: ranking.js against server.js's own reader before it moved (HEAD 25dd4e03) ------------------
function oldRankingFor(dataPath, game) {
  try {
    const old = JSON.parse(fs.readFileSync(path.join(dataPath, 'modelPerformance.json'), 'utf-8'));
    const info = old.games && old.games[game];
    if (!info || !Array.isArray(info.models) || !info.models.length) return null;
    return { names: info.models.map((m) => String(m.name)), metric: info.metric === 'profit_bound' ? 'capped profit per bet (lower bound)' : (info.metric === 'profit_per_bet' ? 'profit per bet' : 'average hits') };
  } catch (e) { return null; }
}
// generateTable's old pick of the shown rows: the first five ranked names present in the day's table
const oldTop = (ranking, present) => (ranking ? ranking.names.filter((n) => present.includes(n)).slice(0, 5) : []);
function sameRanking(databaseDir, games, label) {
  games.forEach((g) => {
    const now = rankings.rankingFor(databaseDir, g);
    assert.deepStrictEqual(now, oldRankingFor(databaseDir, g), `${label} ${g}: the shared ranking is the old one`);
    const present = now ? now.names.filter((n, i) => i % 3 !== 1) : [];   // a day's table missing some ranked rows
    assert.deepStrictEqual(rankings.topNames(now, rankings.TOP_MODELS_SHOWN, present), oldTop(now, present), `${label} ${g}: the same top five of a day's table`);
    assert.deepStrictEqual(rankings.topNames(now), now ? now.names.slice(0, 5) : [], `${label} ${g}: the game's top five`);
    checks += 3;
  });
}
const db = path.join(dataDir, 'database');
sameRanking(db, ['lotto', 'keno', 'crypto', 'vikinglotto'], 'fixture');
ok(rankings.TOP_MODELS_SHOWN === 5, 'users see five rows');
ok(rankings.rankingFor(path.join(tmp, 'nowhere'), 'lotto') === null, 'no report, no ranking');

// the cache: the same parsed object until the file changes, a new one after
const first = rankings.loadReport(db);
ok(rankings.loadReport(db) === first, 'the report is parsed once while the file is unchanged');
const later = { ...report, generatedAt: '2026-10-22T08:00:00' };
fs.writeFileSync(path.join(db, 'modelPerformance.json'), JSON.stringify(later));
fs.utimesSync(path.join(db, 'modelPerformance.json'), new Date(), new Date(Date.now() + 5000));
ok(rankings.loadReport(db) !== first && rankings.loadReport(db).generatedAt === '2026-10-22T08:00:00', 'a changed file is read again');
write('database/modelPerformance.json', report);
fs.utimesSync(path.join(db, 'modelPerformance.json'), new Date(), new Date(Date.now() + 10000));

// --- the routes on the hand-made records -----------------------------------------------------
const app = fakeApp();
admin.install(app, { dataDir, root: tmp });
const routes = ['/api/admin/controls', '/api/admin/controls/lotto', '/api/admin/models', '/api/admin/models/lotto', '/api/admin/models/lotto?scope=top',
  '/api/admin/phase-shift', '/api/admin/phase-shift/lotto', '/api/admin/randomness', '/api/admin/randomness/lotto?scope=all'];
routes.forEach((url) => {
  const r = request(app, url, USER);
  ok(r.statusCode === 403 && r.body.error === 'admin only', `${url}: 403 for a user`);
  const a = request(app, url, ADMIN);
  ok(a.statusCode === 200 && a.headers['Cache-Control'] === 'no-store' && !/NaN|undefined/.test(JSON.stringify(a.body)), `${url}: 200 for the administrator`);
});
ok(request(app, '/api/admin/models/bingo', ADMIN).statusCode === 404 && request(app, '/api/admin/controls/..', ADMIN).statusCode === 404, 'an unknown game is a 404');
ok(request(app, '/api/admin/models/lotto?scope=some', ADMIN).statusCode === 400, 'a scope other than all or top is a 400');
ok(request(app, '/api/admin/models/eurodreams', ADMIN).body.error === 'no ranking for this game yet', 'a game the report does not have yet is a 404 that says so');
const lost = request(app, '/api/admin/nothing', ADMIN);
ok(lost.statusCode === 404 && lost.body.error === 'not found', 'any other /api/admin path is a JSON 404');
ok(request(app, '/api/admin/nothing', USER).statusCode === 403, 'and a user never learns which paths exist');

// controls: grouped, every game, the Q0 rules
const c = request(app, '/api/admin/controls', ADMIN).body;
ok(c.groups.map((g) => g.name).join() === 'Lottery,Markets' && c.groups[0].games.length === 7 && c.groups[1].games.length === 4, 'controls: 11 games, grouped Lottery / Markets');
const cg = Object.fromEntries(c.groups.flatMap((g) => g.games).map((g) => [g.game, g]));
ok(cg.lotto.q0.band.ceiling === 1.126 && cg.lotto.q0.band.from === 'synthetic' && cg.lotto.q0.band.seeds === 3 && cg.lotto.q0.band.history === 120 && cg.lotto.q0.band.at === '2026-10-18T10:11:55Z',
  `Q0: the band, its control, seeds, history and its own record's time: ${JSON.stringify(cg.lotto.q0.band)}`);
ok(cg.lotto.q0.above.n === 2 && cg.lotto.q0.above.rows.map((r) => r.row).join() === 'A Model,B Model' && cg.lotto.q0.rankedRows === 6 && cg.lotto.q0.above.longest === 30,
  'Q0: the ranked rows (minimum draws) above the band, the longest record among them');
ok(cg.lotto.unit === 'draws' && cg.crypto.unit === 'days' && cg.cryptoweek.unit === 'weeks' && cg.shares.unit === 'trading days', 'every game says what one of its draws is');
ok(cg.keno.q0.ranking === 'profit' && cg.keno.q0.above.n === 1 && cg.keno.q0.above.inTile === false && cg.keno.q0.leader.row === 'K1 Model' && cg.keno.q0.above.rows[0].hits === 6.3,
  'Q0: a payout game is compared hits with hits, names its profit-bound leader without a euro figure and stays out of the headline count');
const L = cg.crypto.q0.above.rows.find((r) => r.row === 'M Model');
ok(L && L.live === 20 && L.notLive === 30 && L.own === 0.642 && cg.crypto.q0.above.longestLive === 20, `Q0: a market row's live and back-built draws, and its own control level: ${JSON.stringify(L)}`);
ok(cg.lotto.q0.above.rows[0].live === undefined, 'Q0: a lottery row has no live split (every draw was predicted before it)');
ok(c.tiles.q0.rowsAbove === 4 && c.tiles.q0.rowsAbove === c.groups.flatMap((g) => g.games).filter((g) => g.q0.ranking === 'hits').reduce((s, g) => s + g.q0.above.n, 0)
  && c.tiles.q0.alternatives.allGamesOnHits.rowsAbove === 5 && c.tiles.q0.alternatives.bestRowOnly.games.join() === 'lotto,crypto', `Q0 tile: the count equals the rows of the cards (${c.tiles.q0.rowsAbove})`);
ok(cg.lotto.q2.status === 'above' && cg.lotto.q2.seeds === 3 && cg.lotto.q2.windowsTest === 496 && cg.crypto.q2.fewWindows === true && c.tiles.q2.aboveFewWindows[0].game === 'crypto',
  'Q2: status, seeds and test windows; an above on fewer than 250 windows is named');
ok(cg.lotto.q2.drawsUsed === 1000 && cg.crypto.q2.drawsUsed === 212 && cg.crypto.q2.drawsCap === 1000,
  `Q2: the draws the windows came from, (train + test) / 2 + window - 1, apart from the 1000 asked: ${cg.crypto.q2.drawsUsed}`);
ok(c.tiles.q0.noBand.includes('euromillions') && !c.tiles.q0.noBand.includes('lotto') && c.tiles.rows.missing.join() === 'shares,cryptoweek,sharesweek',
  'the tiles name the games with no band and the market games with no rows report');
ok(cg.lotto.fc.fits === 1 && cg.lotto.fc.ignored === 1 && cg.lotto.fc.chanceShare === 0.25 && cg.lotto.fc.tableRows === 275 && cg.lotto.fc.lockboxRows === 25 && cg.lotto.fc.variants[0].key === 'lr' && cg.lotto.fc.variants[0].share === 0.3117,
  'FC: the verdict counts, the chance share 3 / (9 + 3), the table and lockbox rows in draws');
ok(cg.crypto.rows.window === 231 && cg.crypto.rows.rowsScored === 1 && cg.crypto.rows.betterThanUniform.join() === 'Chronos Model' && cg.crypto.rows.betterThanReference.length === 0
  && cg.crypto.rows.referenceVsUniform.mean === 0.0923 && cg.lotto.rows === null, 'market rows: the window, the rows against uniform and GARCH; a lottery game has none');
const drawer = request(app, '/api/admin/controls/crypto', ADMIN).body;
ok(drawer.q0.modes.shuffled.perRow['M Model'] === 0.642 && drawer.q2.perSeed.real.length === 0 && drawer.rows.table.length === 3 && drawer.regimes[0].label === 'calm', 'the drawer: the modes, the rows table and the regime reading');
// the Market rows help names the probability floor, each row's kind and the Regime HMM feature set (the old markets page's C16 / C17)
ok(drawer.rows.floor === 0.001 && drawer.rows.table.map((x) => x.kind).join() === 'market,foundation,baseline' && drawer.rows.hmmFeatures === 'returns' && drawer.q2.quantum === true,
  `the drawer: the probability floor, each row's kind, the HMM feature set and the Q2 quantum flag: ${JSON.stringify([drawer.rows.floor, drawer.rows.table.map((x) => x.kind)])}`);
const lottoDrawer = request(app, '/api/admin/controls/lotto', ADMIN).body;
ok(lottoDrawer.q2.perSeed.real.length === 3 && lottoDrawer.fc.variants[1].above[0] === 'C Model' && lottoDrawer.rows === null && lottoDrawer.regimes === null, 'the lottery drawer: per seed, the variants in full, no market parts');

// models: the summary and one game
const m = request(app, '/api/admin/models', ADMIN).body;
const mk = m.games.find((g) => g.game === 'keno');
ok(mk.best.value === 0.117 && mk.best.profitPerBet === -0.195 && /fair ticket/.test(mk.measure.against) && mk.band.greys === false,
  'keno: the bound comes with the raw profit per bet and what it is measured against; the hits band greys nothing');
const ml = m.games.find((g) => g.game === 'lotto');
ok(ml.top.join() === 'A Model,B Model,C Model,D Model,E Model' && ml.since.best.row === 'B Model' && ml.best.inBand === false && m.topN === 5, 'the summary: the five users see, the best since the freeze');
ok(m.games.find((g) => g.game === 'crypto').top.join() === 'L Model,N Model' && request(app, '/api/admin/models/crypto?scope=top', ADMIN).body.rows.map((r) => r.row).join() === 'L Model,N Model',
  'the top rows are those in the newest day\'s table: a ranked row missing from it is passed over, as on the users\' pages');
const mc = m.games.find((g) => g.game === 'crypto');
const cm = request(app, '/api/admin/models/crypto', ADMIN).body;
ok(mc.best.live === 20 && cm.rows.find((r) => r.row === 'M Model').live === 20 && cm.rows.find((r) => r.row === 'N Model').live === 0 && ml.best.live === undefined,
  'a market row\'s live draws next to all its draws; none for a lottery row');
const all = request(app, '/api/admin/models/lotto', ADMIN).body;
const top = request(app, '/api/admin/models/lotto?scope=top', ADMIN).body;
ok(all.scope === 'all' && all.shown === 7 && all.of === 7 && top.scope === 'top' && top.shown === 5 && top.rows.every((r) => r.top) && top.of === 7, 'scope: every row, or the top five only');
ok(all.rows.find((r) => r.row === 'Young Model').young === true && all.rows.find((r) => r.row === 'D Model').inBand === true && all.rows.find((r) => r.row === 'A Model').inBand === false,
  'the young and in-band flags of the old ranking');
ok(all.multiPick.rows.length === 2 && top.multiPick.rows.length === 1 && all.multiPick.chance[9] === 0.08395, 'multi-pick: the rows with extras, filtered by the scope; the exact chance');
ok(all.combination.control.pValue === 0.188 && all.combination.control.consistentWithLuck === true && all.combination.beatsBestSingle === null, 'combination: the p value travels with the value; "beats" only for profit');
const kg = request(app, '/api/admin/models/keno', ADMIN).body;
ok(kg.rows[0].bySize.length === 2 && kg.rows[0].bySize[0].size === 5 && kg.combination.measure.capped === false && kg.combination.beatsBestSingle === true, 'keno: per ticket size, the uncapped combination');

// phase shift and randomness
const ps = request(app, '/api/admin/phase-shift/lotto', ADMIN).body;
const pa = ps.rows.find((r) => r.row === 'A Model');
ok(ps.rows[0].row === 'A Model' && pa.tracked === true && pa.mostFrequent.lag === 5 && pa.mostFrequent.runs === 4 && JSON.stringify(pa.trail.map((s) => [s[0], s[1]])) === '[[5,3],[7,1],[5,1]]',
  `phase shift: in ranking order, the tracked rule, the run-length trail: ${JSON.stringify(pa.trail)}`);
const gone = ps.rows.find((r) => r.row === 'Gone Model');
ok(gone.trail.length === 8 && gone.earlierSteps === 3 && gone.tracked === false && ps.rows[ps.rows.length - 1].row === 'Gone Model', 'phase shift: the newest 8 steps; rows outside the ranking last');
ok(request(app, '/api/admin/phase-shift/lotto?scope=top', ADMIN).body.rows.map((r) => r.row).join() === 'A Model', 'phase shift: the top scope');
const bygone = ps.rows.find((r) => r.row === 'Bygone Model');
ok(bygone.peak.unchanged === true && bygone.peak.run === '2026-10-03' && bygone.peak.scanned === '2026-10-21' && pa.peak.unchanged === false && pa.peak.run === '2026-10-05',
  `phase shift: a scan with no new draw is no run - the peak dates from the newest recorded run: ${JSON.stringify(bygone.peak)}`);
const psl = request(app, '/api/admin/phase-shift', ADMIN).body.games.find((g) => g.game === 'lotto');
ok(JSON.stringify(psl.commonLags) === JSON.stringify([{ lag: 4, rows: 1, tracked: 1 }, { lag: 5, rows: 1, tracked: 1 }, { lag: 1, rows: 1, tracked: 0 }]) && psl.newestRun === '2026-10-11' && psl.scanned === '2026-10-21',
  `phase shift: the common lags count every row's most frequent peak, with the tracked ones apart: ${JSON.stringify(psl.commonLags)}`);
const rw = request(app, '/api/admin/randomness', ADMIN).body;
const rl = rw.games.find((g) => g.game === 'lotto'); const rc = rw.games.find((g) => g.game === 'crypto');
ok(rl.status === 'watch' && rl.reasons.join() === 'entropy,q2' && rl.q2.real === 0.59 && rc.reasons.join() === 'ae,q2' && rc.window === null && rw.games.length === 2,
  'randomness: which tripwire is up, the Q2 verdict next to it; a market game with the autoencoder only');
const rlg = request(app, '/api/admin/randomness/lotto?scope=top', ADMIN).body;
ok(rlg.models.map((r) => r.row).join() === 'A Model,C Model' && rlg.of === 3 && rlg.topMissing.join() === 'B Model,D Model,E Model',
  'randomness: the model KL rows, lowest first, the top scope, and the top rows with no KL row named');
// the readings the old cards' help gave: what the autoencoder's z is of and which way is the alarm; z near 1 is a flat lag profile
ok(/reconstruction NLL/.test(rw.meta.ae) && /strongly negative/.test(rw.meta.ae) && /predictability spike/.test(rw.meta.ae) && /near 1 the profile is flat/.test(ps.meta.z)
  && /number frequency rather than timing/.test(request(app, '/api/admin/phase-shift', ADMIN).body.meta.z), 'the meta texts: the autoencoder z read as the old card did, z near 1 a flat profile');

// a game whose combination search found no set: the note and how many established rows were tried (the old "(N established rows)")
const noSet = JSON.parse(JSON.stringify(report));
noSet.games.lotto.combinations = { metric: 'avg_best_hits', minDrawsForRanking: 10, candidateRows: ['A Model'], evaluated: 0, note: 'fewer than two established rows', best: null };
const noSetCombo = admin.modelsGame(noSet, 'lotto').combination;
ok(noSetCombo.best === null && noSetCombo.note === 'fewer than two established rows' && noSetCombo.candidateRows === 1 && all.combination.candidateRows === 3,
  `combination: no set found keeps its note and the established rows tried: ${JSON.stringify(noSetCombo)}`);

// a market row's live draws: the report's own count over the whole record; the since record stands in only while
// the freeze is on or before the day the markets went live
const withLive = JSON.parse(JSON.stringify(report));
withLive.games.crypto.models.forEach((x, i) => { x.live_draws = [12, 33, 0][i]; });
ok(admin.modelsGame(withLive, 'crypto').rows.map((r) => r.live).join() === '12,33,0', 'live draws from the report\'s live_draws');
const laterFreeze = JSON.parse(JSON.stringify(report));
laterFreeze.games.crypto.since.date = '2026-11-02';
ok(admin.modelsGame(laterFreeze, 'crypto').rows.every((r) => r.live === undefined), 'a later freeze without live_draws: no live split rather than a wrong one');

// one broken record costs its own cell, never the page
const broken = admin.readAll(admin.resolveDirs({ dataDir }));
broken.crypto = { ...broken.crypto, rows: { market: 'crypto', rows: [null] } };
const bc = admin.controlsOverview(broken, report).groups[1].games.find((g) => g.game === 'crypto');
ok(bc.errors.join() === 'rows' && bc.rows === null && bc.q0.band !== null, 'a broken rows record: the overview still builds, the part is named');
// a record caught half written (the writers do not replace files atomically) reads "unreadable", not "no record
// yet"; a band left with one control says so
const cutDir = path.join(tmp, 'cut');
fs.cpSync(path.join(dataDir, 'controls'), cutDir, { recursive: true });
['results/lotto-synthetic.json', 'discrimination/lotto-w10.json', 'features/lotto.json', 'markets/crypto-rows.json'].forEach((f) => {
  fs.writeFileSync(path.join(cutDir, f), fs.readFileSync(path.join(cutDir, f), 'utf-8').slice(0, 40));
});
const cutDirs = admin.resolveDirs({ dataDir, controlsDir: cutDir });
const cut = admin.controlsOverview(admin.readAll(cutDirs), report);
const cutGames = Object.fromEntries(cut.groups.flatMap((g) => g.games).map((g) => [g.game, g]));
ok(cutGames.lotto.errors.join() === 'q2,fc' && cutGames.crypto.errors.join() === 'rows' && cut.tiles.q2.unreadable.join() === 'lotto' && cut.tiles.fc.unreadable.join() === 'lotto'
  && cut.tiles.rows.unreadable.join() === 'crypto' && cut.tiles.q2.missing.includes('euromillions') && !cut.tiles.q2.missing.includes('lotto'),
  `a half-written record is named unreadable, apart from the games with no record yet: ${JSON.stringify(cutGames.lotto.errors)}`);
ok(JSON.stringify(cutGames.lotto.q0.band.missing) === '[{"mode":"synthetic","unreadable":true}]' && cutGames.lotto.q0.band.from === 'shuffled'
  && JSON.stringify(cg.keno.q0.band.missing) === '[{"mode":"shuffled","unreadable":false}]' && cg.lotto.q0.band.missing.length === 0
  && cut.tiles.q0.oneControl.map((x) => x.game).join() === 'lotto,keno,crypto',
  'a band from one control names the other and why it is not there; the tile names the games');
const cutDrawer = admin.controlsDrawer('lotto', admin.readControls(cutDirs, 'lotto'), report);
ok(cutDrawer.errors.join() === 'q2,fc' && cutDrawer.q0.band.missing[0].unreadable === true, 'the drawer: the same parts unreadable, the band from one control');
ok(admin.modelsGame(report, 'lotto', 'all', admin.readControls(cutDirs, 'lotto')).band.missing[0].mode === 'synthetic', 'the Models band says it comes from one control');
const brokenQ2 = { ...broken.lotto, q2: { ...broken.lotto.q2, verdict: { ...broken.lotto.q2.verdict, per_seed_real: 5 } } };
ok(admin.controlsDrawer('lotto', brokenQ2, report).q2.perSeed.real.length === 0, 'a per-seed field that is not a list reads as none');
const brokenSince = JSON.parse(JSON.stringify(report));
brokenSince.games.lotto.since.models = { a: 1 };
ok(admin.modelsSummary(brokenSince).games.length === 3 && admin.modelsGame(brokenSince, 'lotto').rows.length === 7, 'a since record whose models are not a list');

// a report caught half written keeps the last good one
const good = rankings.loadReport(db);
fs.writeFileSync(path.join(db, 'modelPerformance.json'), '{"games": {"lotto"');
fs.utimesSync(path.join(db, 'modelPerformance.json'), new Date(), new Date(Date.now() + 20000));
ok(rankings.loadReport(db) === good && rankings.rankingFor(db, 'lotto').names.length === 7, 'a half-written report: the last good ranking stays');
write('database/modelPerformance.json', report);
fs.utimesSync(path.join(db, 'modelPerformance.json'), new Date(), new Date(Date.now() + 30000));

// --- phase P2: week over week, the runs, the tuning gates -----------------------------------------------------
const readings = require('../readings');
const gates = require('../gates');
const { execFileSync } = require('child_process');
const brussels = require('../brussels');

// Brussels days and the instants they start at, whatever this process's zone
ok(brussels.dayOf(Date.parse('2026-10-18T22:30:00Z')) === '2026-10-19' && brussels.lastWeekday(Date.parse('2026-10-18T22:30:00Z'), 0) === '2026-10-18'
  && brussels.instant('2026-10-26') === Date.parse('2026-10-25T23:00:00Z') && brussels.instant('2026-10-25', 9) === Date.parse('2026-10-25T08:00:00Z')
  && brussels.isDay('2026-10-18') && !brussels.isDay('2026-02-30'), 'Brussels days: Monday 00:30 is Monday, the clocks go back on 25 Oct');
ok(admin.sundayOf('2026-10-18T22:45:39Z') === '2026-10-18' && admin.sundayOf('2026-10-20T06:00:00Z') === '2026-10-18' && admin.sundayOf('nonsense') === null,
  'a control record belongs to the Sunday on or before it: a Monday or Tuesday catch-up is still that weekend\'s');

// stored readings: 4 and 11 Oct, built by the same builder from these records moved to their own week (every record
// time a week or two earlier) and changed by hand; a broken file and a file that is not a Sunday's next to them
const readingsDir = path.join(dataDir, 'readings');
const moveBack = (x, days) => {
  if (!x || typeof x !== 'object') return;
  Object.keys(x).forEach((k) => {
    if (k === 'at' && typeof x[k] === 'string') x[k] = new Date(Date.parse(x[k]) - days * 86400000).toISOString().replace(/\.\d{3}Z$/, 'Z');
    else moveBack(x[k], days);
  });
};
const storedWeek = (sunday, change) => {
  const r = admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report, { sunday, now: new Date(`${sunday}T23:00:00Z`) });
  moveBack(r.games, (Date.parse('2026-10-18') - Date.parse(sunday)) / 86400000);
  change(r.games);
  return readings.writeReading(r, { outDir: readingsDir });
};
storedWeek('2026-10-11', (g) => { g.lotto.q0.above = g.lotto.q0.above.filter((r) => r.row === 'A Model'); g.lotto.q2.status = 'within'; g.crypto.rows = null; });
storedWeek('2026-10-04', (g) => { g.lotto.q0.above = []; });
fs.writeFileSync(path.join(readingsDir, '2026-10-25.json'), '{"version": 1, "sunday": "2026-10-');
fs.writeFileSync(path.join(readingsDir, '2026-10-20.json'), JSON.stringify({ version: 1, sunday: '2026-10-20', games: {} }));
const stored = admin.loadReadings(readingsDir);
ok(stored.sundays.join() === '2026-10-04,2026-10-11' && stored.unreadable.join() === '2026-10-20.json,2026-10-25.json', 'the stored readings, and the files that are not one');

const ch = request(app, '/api/admin/controls', ADMIN).body;
const hg = Object.fromEntries(ch.groups.flatMap((g) => g.games).map((g) => [g.game, g]));
ok(JSON.stringify(hg.lotto.history.q0) === '{"state":"above","sunday":"2026-10-18","weeks":2,"changed":false,"live":true}',
  `Q0 above for the 2nd week (4 Oct read none above); no reading of 18 Oct yet, so against the live ranking: ${JSON.stringify(hg.lotto.history.q0)}`);
ok(hg.lotto.q0.above.rows.map((r) => `${r.row}:${r.weeks}`).join() === 'A Model:2,B Model:1', 'each row above the band says for how many weeks in a row');
ok(JSON.stringify(hg.lotto.history.q2) === '{"state":"above","sunday":"2026-10-18","weeks":1,"changed":true,"last":"within"}',
  'a change from last week carries what it read then - the only thing the page colours');
ok(hg.lotto.history.fc.state === 'lr fit · qk ignored' && hg.lotto.history.fc.weeks === 3 && hg.lotto.history.fc.changed === false && hg.lotto.history.fc.last === undefined,
  'a state that holds every week: its count of weeks, no change, nothing to colour');
ok(hg.crypto.history.rows.state === 'beat GARCH: none · GARCH vs uniform: better' && hg.crypto.history.rows.changed === true && hg.crypto.history.rows.last === null
  && hg.keno.history.q0.weeks === 3 && hg.euromillions.history.q0 === null && hg.lotto.history.rows === undefined,
  'a record where last week had none is a change (last: null); a game without records has no history; a lottery game no rows');
ok(ch.readings.stored === 2 && ch.readings.unreadable.length === 2 && ch.readings.newestRecords === '2026-10-18', 'the page says how many weeks it reads, and which files it could not');
ok(JSON.stringify(admin.streakOf({ bySunday: new Map([['2026-10-04', stored.bySunday.get('2026-10-04')]]) }, 'lotto', 'q2', admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report).games.lotto))
  === '{"state":"above","sunday":"2026-10-18","weeks":1,"changed":null}', 'a missing week ends the run and leaves nothing to compare: weeks 1, changed null');
const none = admin.withHistory(admin.controlsOverview(admin.readAll(admin.resolveDirs({ dataDir })), report), admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report),
  admin.loadReadings(path.join(tmp, 'no-readings')));
ok(none.readings.stored === 0 && none.groups[0].games[0].history.q0.weeks === 1 && none.groups[0].games[0].history.q0.changed === null, 'with no reading at all: the first week on record');
// the controls are overwritten while they run: a part whose record is from an earlier week than the newest says so
const midRun = admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report);
midRun.games.lotto.q2 = { ...midRun.games.lotto.q2, at: '2026-10-25T08:48:16Z' };
const mh = admin.historyOf(midRun, stored);
ok(mh.newest === '2026-10-25' && mh.games.lotto.q2.sunday === '2026-10-25' && mh.games.lotto.q0.behind === true && mh.games.lotto.q2.behind === undefined,
  'mid-run on a Sunday: the cells still from last week are marked behind');
// Q0 rewrites a game's synthetic record hours before its shuffled one: half rewritten, the band is still last week's
ok(midRun.games.lotto.q0.at === '2026-10-18T10:11:55Z', 'a reading\'s Q0 time is the older of its two control records');
const halfQ0 = JSON.parse(JSON.stringify(midRun));
halfQ0.games.lotto.q0.modes.synthetic.at = '2026-10-25T10:11:55Z';
const hh = admin.historyOf(halfQ0, stored);
ok(hh.games.lotto.q0.sunday === '2026-10-18' && hh.games.lotto.q0.behind === true, `a band half rewritten is behind, not this week's: ${JSON.stringify(hh.games.lotto.q0)}`);

// a stored week whose reading holds an older record of a control (it failed or skipped the game that Sunday, so last
// week's record was still on disk) is no reading of that control: it ends the run, it is not one more week
const staleStored = { ...stored, bySunday: new Map([...stored.bySunday].map(([day, r]) => [day, JSON.parse(JSON.stringify(r))])) };
const r11 = staleStored.bySunday.get('2026-10-11');
r11.games.lotto.q2 = { ...staleStored.bySunday.get('2026-10-04').games.lotto.q2 };
r11.games.lotto.fc = { ...staleStored.bySunday.get('2026-10-04').games.lotto.fc };
r11.games.lotto.q0 = JSON.parse(JSON.stringify(staleStored.bySunday.get('2026-10-04').games.lotto.q0));
r11.games.lotto.q0.above = [{ row: 'A Model' }];
const cur18 = admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report);
const sf = admin.streakOf(staleStored, 'lotto', 'fc', cur18.games.lotto);
ok(JSON.stringify(sf) === '{"state":"lr fit · qk ignored","sunday":"2026-10-18","weeks":1,"changed":null,"noRunBefore":true}',
  `a stale copy is not counted twice: weeks 1, nothing to compare, said: ${JSON.stringify(sf)}`);
ok(admin.streakOf(staleStored, 'lotto', 'q2', cur18.games.lotto).changed === null && admin.streakOf(staleStored, 'lotto', 'q0', cur18.games.lotto).weeks === 1,
  'the same for Q2 and Q0');
const staleView = admin.withHistory(admin.controlsOverview(admin.readAll(admin.resolveDirs({ dataDir })), report), cur18, staleStored);
ok(staleView.groups[0].games[0].q0.above.rows.map((r) => `${r.row}:${r.weeks}`).join() === 'A Model:1,B Model:1', 'a row above in a stale Q0 copy gets no extra week');

// Q0 once the reading of its own Sunday is stored: the state and the rows' weeks are that reading's (that Sunday's
// ranking), never today's; where today's ranking reads the band differently is said apart (sinceReading)
const r18 = admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report, { sunday: '2026-10-18' });
r18.games.lotto.q0.above = r18.games.lotto.q0.above.filter((r) => r.row === 'A Model');
r18.rankingAt = '2026-10-18T08:47:31Z';
const withOwn = { ...stored, bySunday: new Map([...stored.bySunday, ['2026-10-18', r18]]) };
const own = admin.streakOf(withOwn, 'lotto', 'q0', cur18.games.lotto);
ok(JSON.stringify(own) === '{"state":"above","sunday":"2026-10-18","weeks":2,"changed":false,"sinceReading":{"now":"above","added":["B Model"],"dropped":[],"rankingAt":"2026-10-18T08:47:31Z"}}',
  `Q0 reads its own Sunday's reading, the live ranking's extra row said apart: ${JSON.stringify(own)}`);
r18.games.lotto.q0.above = [];
const flipped = admin.streakOf(withOwn, 'lotto', 'q0', cur18.games.lotto);
ok(flipped.state === 'none above' && flipped.changed === true && flipped.last === 'above' && flipped.sinceReading.now === 'above' && flipped.sinceReading.added.join() === 'A Model,B Model',
  'a later ranking never turns a stored "none above" into "above": the change is the readings\' own');
r18.games.lotto.q0.above = [{ row: 'A Model' }];
const ownView = admin.withHistory(admin.controlsOverview(admin.readAll(admin.resolveDirs({ dataDir })), report), cur18, withOwn);
ok(ownView.groups[0].games[0].q0.above.rows.map((r) => `${r.row}:${r.weeks}`).join() === 'A Model:2,B Model:null',
  'a row above only against the live ranking has no weeks; the others count from the reading of their Sunday');

// a market row's live count gets its own "≤": rows of one length can differ in live days
const lives = admin.q0Of('crypto', { synthetic: { best_row_mean: 1, best_row_sd_across_seeds: 0.1, generated_at: '2026-10-18T10:00:00', rows_compared: 5, seeds: [1, 2, 3], days: 100 } },
  { metric: 'avg_hits', minDrawsForRanking: 10, models: [{ name: 'A Model', avg_hits: 2, draws: 50, live_draws: 20 }, { name: 'B Model', avg_hits: 2.1, draws: 50, live_draws: 17 }] });
ok(lives.above.sameLength === true && lives.above.sameLive === false && lives.above.longestLive === 20 && lives.band.modes.length === 1 && lives.band.modes[0].at === '2026-10-18T10:00:00Z',
  'Q0: the same record length, different live counts; the band names each control record\'s time');

// the runs: three jobs, the scheduler's history and their logs
const TEST_JOBS = [
  { key: 'predictor', name: 'Daily predictor', log: 'predictor.log', daily: { hour: 9, minute: 0 } },
  { key: 'hyperoptStatistics', name: 'Weekly tuning: statistical models', log: 'hyperoptStatistics.log', plan: 'weeklyTuning' },
  { key: 'nullControls', name: 'Weekly controls: null histories (Q0)', log: 'nullControls.log', plan: 'weeklyControls' },
];
const Z = (iso) => Date.parse(iso);
const runOf = (key, startIso, minutes, exit = 0, extra = {}) => ({ at: Z(startIso), finishedAt: Z(startIso) + minutes * 60000, key, name: key, reason: 'test', seconds: minutes * 60, exit, sha: 'abc12345', ...extra });
const history = [
  ...[12, 13, 14, 15, 16].map((d, i) => runOf('predictor', `2026-10-${d}T07:00:00Z`, [100, 110, 120, 130, 140][i])),
  runOf('predictor', '2026-10-17T07:00:00Z', 121), runOf('hyperoptStatistics', '2026-10-17T09:00:00Z', 300),
  runOf('predictor', '2026-10-18T07:00:00Z', 100), runOf('nullControls', '2026-10-18T10:00:00Z', 760, 1),
  runOf('predictor', '2026-10-19T07:00:00Z', 98), runOf('predictor', '2026-10-20T07:00:00Z', 112), runOf('predictor', '2026-10-21T07:00:00Z', 108),
  { at: Z('2026-10-17T12:00:00Z'), key: 'hyperoptStatistics', name: 'x', reason: 'r', dry: true },
  { at: Z('2026-10-18T01:00:00Z'), finishedAt: Z('2026-10-18T07:00:00Z'), key: 'nullControls', name: 'x', reason: 'r', seconds: 21600, exit: null, blocked: true },
].reverse();
fs.mkdirSync(path.join(tmp, 'config'));
fs.writeFileSync(path.join(tmp, 'config', 'scheduler.json'), JSON.stringify({ version: 1, history, current: null, queue: [{ key: 'predictor', reason: 'by hand', queuedAt: Z('2026-10-21T17:00:00Z'), manual: true }] }));
fs.mkdirSync(path.join(tmp, 'log'));
const slice = (name, start, lines, status = 0, seconds = 60) => [`=== scheduler: ${name} started ${start} abc12345 UTC ===`, ...lines,
  ...(status === null ? [] : [`=== scheduler: ${name} finished status=${status} after ${seconds}s ===`])].join('\n');
const writeLog = (file, parts) => fs.writeFileSync(path.join(tmp, 'log', file), `${parts.join('\n')}\n`);
writeLog('hyperoptStatistics.log.1', [slice('Weekly tuning: statistical models', '2026-10-03 08:38:29', ['Traceback (most recent call last):', 'ValueError: boom'], 0, 6000)]);
writeLog('hyperoptStatistics.log', [
  slice('Weekly tuning: statistical models', '2026-10-10 08:52:22', ['LOCKBOX: 2 days withheld', 'LOCKBOX: again', '/x/foo.py:12: UserWarning: careful'], 0, 7200),
  slice('Weekly tuning: statistical models', '2026-10-17 09:00:01', ['LOCKBOX: 1 day withheld', '/src/MarketModels.py:448: RuntimeWarning: invalid value encountered in divide',
    'Traceback (most recent call last):', '  the gate: standard error 0.0883', 'TF-TRT Warning: Could not find TensorRT'], 0, 18000)]);
writeLog('nullControls.log', [slice('Weekly controls: null histories (Q0)', '2026-10-18 10:00:01', ['/src/MarketModels.py:448: RuntimeWarning: invalid value'], 1, 45600)]);
writeLog('predictor.log', ['2026-10-12', '2026-10-16', '2026-10-17', '2026-10-18', '2026-10-19', '2026-10-20', '2026-10-21']
  .map((d) => slice('Daily predictor', `${d} 07:00:02`, ['saving_lib.py:1: UserWarning: keras', ...(d === '2026-10-21' ? ['Traceback (most recent call last):'] : [])], 0, 6000)));
ok(admin.slicesOf('=== scheduler: J started 2026-10-18 10:00:01 UTC ===\nZeroDivisionError: x\n=== scheduler: J started 2026-10-18 11:00:01 abc UTC ===\nstill running').map((s) => `${s.status}:${Object.keys(s.sig)}`).join('|')
  === 'null:ZeroDivisionError|null:', 'a slice without its finished marker (killed, or still running) is kept, without a status');

// the tuning gates: lotto's bestParams in a git repository of the tuners' save commits, and an uncommitted working tree
const gate = (date, decision, chScore, inc, def, { margin = 0, se = 0.08, kind = 'hits', days = 90, tuner = null } = {}) => ({
  date, ...(tuner ? { tuner } : {}), window_days: days, study_best: 1, margin, decision, run_trials: 15, study_trials: 105, reason: 'the numbers again', trial: 7,
  challenger: chScore, incumbent: inc, default: def,
  challenger_tuning: { score: chScore, kind, days, score_se: se }, incumbent_tuning: inc === null ? null : { score: inc, kind, days, score_se: se },
  default_tuning: def === null ? null : { score: def, kind, days: days - 3, score_se: se } });
const bestParams = (records) => fs.writeFileSync(path.join(tmp, 'bestParams_lotto.json'), JSON.stringify({ someKnob: 1, tuningGate: records }, null, 2));
const git = (args, when) => execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', ...args],
  { cwd: tmp, stdio: 'pipe', env: { PATH: process.env.PATH, HOME: process.env.HOME || '/root', GIT_AUTHOR_DATE: when, GIT_COMMITTER_DATE: when } });
const commit = (subject, when, records) => { bestParams(records); git(['add', 'bestParams_lotto.json'], when); git(['commit', '-q', '--allow-empty', '-m', subject], when); };
git(['init', '-q'], '2026-10-01T00:00:00Z');
const markov10 = gate('2026-10-10T10:00:00+00:00', 'replaced', 0.8, 0.6, 0.65);
const xgb10 = gate('2026-10-10T20:00:00+00:00', 'kept', 0.5, 0.7, 0.6, { margin: 0.08 });
commit('Saving latest statistical hyperopt', '2026-10-10T12:00:00Z', { 'Markov Model': markov10 });
commit('Saving latest boosting hyperopt', '2026-10-10T22:00:00Z', { 'Markov Model': markov10, 'XGBoost Model': xgb10 });
commit('saving last predictions', '2026-10-11T09:00:00Z', { 'Markov Model': markov10, 'XGBoost Model': xgb10 });
const markov17 = gate('2026-10-17T10:00:00+00:00', 'kept', 0.7, 0.75, 0.65);
const mystery = gate('2026-10-17T10:30:00+00:00', 'replaced', 0.9, 0.5, 0.85);
commit('Saving latest statistical hyperopt', '2026-10-17T12:00:00Z', { 'Markov Model': markov17, 'XGBoost Model': xgb10, 'Mystery Model': mystery });
// the boosting tuner did not commit this weekend: its XGBoost record first rides in the RL ticket tuner's commit
const xgb17 = gate('2026-10-17T23:00:00+00:00', 'replaced', 0.9, 0.7, 0.6, { margin: 0.08 });
const rl17 = gate('2026-10-17T23:30:00+00:00', 'replaced', 0.71, 0.70, 0.5);
commit('Saving latest RL ticket hyperopt', '2026-10-18T01:00:00Z', { 'Markov Model': markov17, 'XGBoost Model': xgb17, 'Mystery Model': mystery, 'RL Ticket Model': rl17 });
bestParams({ 'Markov Model': markov17, 'XGBoost Model': xgb17, 'Mystery Model': mystery, 'RL Ticket Model': rl17,
  'LaplaceMonteCarlo Model': gate('2026-10-17T10:15:00+00:00', 'reverted', 0.6, 0.55, 0.62),   // in no commit: the statistical run's window confirms it
  'QuantumVQC Model': gate('2026-10-18T02:00:00+00:00', 'reverted', 0.5, 0.4, 0.6, { margin: 0.08, tuner: 'HyperoptQuantum' }),
  'ClassicalSVM Model': gate('2026-10-18T02:10:00+00:00', 'kept', 0.5, 0.6, 0.55, { margin: 0.08, tuner: 'Elsewhere.py' }),
  'PoissonMarkov Model': gate('2026-10-03T10:00:00+00:00', 'kept', 0.5, 0.6, 0.55) });
// a checkout with the same files but no git
const bare = fs.mkdtempSync(path.join(os.tmpdir(), 'admin-bare-'));
fs.copyFileSync(path.join(tmp, 'bestParams_lotto.json'), path.join(bare, 'bestParams_lotto.json'));

ok(gates.familyOf('Markov Model') === 'statistical' && gates.familyOf('Mystery Model') === null && gates.tunerOfScript('HyperoptRLTicket') === 'RL ticket'
  && gates.tunerOfScript('/x/HyperoptBoost.py') === 'boosting' && gates.tunerOfSubject('Saving latest subset ensemble hyperopt') === 'ensemble'
  && gates.tunerOfSubject('Saving latest deep learning hyperopt') === null && gates.weekendOf(Z('2026-10-18T04:03:29Z')) === '2026-10-17',
  'the tuners by row, by script and by commit subject; a Sunday-night decision belongs to Saturday\'s weekend');
const d = gates.describe('Mystery Model', mystery);
ok(d.bar === 0.85 && Math.abs(d.gainSe - 5) < 1e-9 && Math.abs(d.gainBestSe - 0.625) < 1e-9 && d.gainBarSe === d.gainBestSe && d.marginSe === 0 && d.default.days === 87 && d.weekend === '2026-10-17',
  `one decision's numbers: the bar is the better reference plus the margin, the gains in the challenger's SE: ${JSON.stringify(d)}`);
ok(gates.attribute('Markov Model', markov17, { carrier: { tuner: 'statistical' } }).by === 'family + commit'
  && gates.attribute('XGBoost Model', xgb17, { carrier: { tuner: 'RL ticket' } }).tuner === 'unattributed'
  && gates.attribute('Markov Model', markov17, { run: { tuner: 'statistical' } }).by === 'family + run'
  && gates.attribute('Markov Model', markov17).tuner === 'unattributed' && gates.attribute('Mystery Model', mystery, { carrier: { tuner: 'statistical' } }).tuner === 'unattributed'
  && gates.attribute('Anything', { tuner: 'HyperoptQuantum' }).by === 'record',
  'attribution: the family confirmed by a commit or a run, the record\'s own tuner, otherwise unattributed - never the time alone');
ok(gates.runOf(Z('2026-10-17T14:00:30Z'), history)?.tuner === 'statistical' && gates.runOf(Z('2026-10-17T14:02:00Z'), history) === null && gates.runOf(Z('2026-10-17T10:00:00Z'), [])  === null,
  'a run\'s window holds a decision; a minute of slack only past its end');

async function phaseTwo() {
  // the routes that answer later: the guard first, then 200
  const app2 = fakeApp();
  const events = [{ at: '2026-10-21T10:00:00Z', event: 'login.failed', user: 'mallory', ip: '203.0.113.9' }, { at: '2026-10-20T10:00:00Z', event: 'login.ok', user: 'admin', ip: '198.51.100.7' },
    { at: '2026-10-16T10:00:00Z', event: 'login.locked', user: 'mallory', ip: '203.0.113.9' }];
  admin.install(app2, { dataDir, root: tmp, jobs: { JOBS: TEST_JOBS, status: () => ({ lock: { pid: 4242, script: 'Predictor.py', key: 'predictor' }, current: null, queue: [] }) },
    services: { status: async () => [{ key: 'councilApi', name: 'Council API', status: 'running', healthy: true, summary: 'models ready (3/3 members)', since: Z('2026-10-20T20:16:00Z'), restarts: 0, lastExit: null, lastError: null }] },
    audit: { recent: () => events } });
  for (const url of ['/api/admin/week', '/api/admin/tuning', '/api/admin/tuning/lotto']) {
    const u = await requestAsync(app2, url, USER);
    ok(u.statusCode === 403 && u.body.error === 'admin only', `${url}: 403 for a user`);
    const a = await requestAsync(app2, url, ADMIN);
    ok(a.statusCode === 200 && a.headers['Cache-Control'] === 'no-store' && !/NaN|undefined/.test(JSON.stringify(a.body)), `${url}: 200 for the administrator`);
  }
  ok((await requestAsync(app2, '/api/admin/tuning/bingo', ADMIN)).statusCode === 404 && (await requestAsync(app2, '/api/admin/tuning/keno', ADMIN)).body.error === 'no bestParams file for this game',
    'tuning: an unknown game, and a game without a bestParams file, are 404s that say so');

  // --- the tuning gates
  const t = (await requestAsync(app2, '/api/admin/tuning', ADMIN)).body;
  ok(t.weekend === '2026-10-17' && t.tiles.n === 7 && t.tiles.replaced === 3 && t.tiles.kept === 2 && t.tiles.reverted === 2 && t.tiles.notThisWeekend === 1 && t.tiles.marginZero === 4,
    `tuning: this weekend's decisions; a record from an earlier weekend is not counted as kept: ${JSON.stringify(t.tiles)}`);
  ok(t.tiles.replacedUnder1Se === 1 && t.tiles.replacedUnder1SeVsBetterReference === 2 && t.tiles.under1SeTuners.join() === 'RL ticket',
    'tuning: replacements that gained under one SE, over the incumbent and over the better reference');
  ok(JSON.stringify(t.tiles.heldFromLastWeekend) === '{"weekend":"2026-10-10","replaced":1,"kept":1,"replacedAgain":0,"reverted":0,"notDecided":0}',
    'tuning: what this weekend did with last weekend\'s replacements');
  const tun = Object.fromEntries(t.byTuner.map((x) => [x.tuner, x]));
  ok(tun.statistical.n === 2 && tun.boosting.n === 0 && tun['RL ticket'].n === 1 && tun.quantum.n === 1 && tun.unattributed.n === 3 && t.byTuner.reduce((s, x) => s + x.n, 0) === t.tiles.n,
    `tuning: by tuner, the unattributed apart, the counts adding up to the tile: ${t.byTuner.map((x) => `${x.tuner} ${x.n}`).join(', ')}`);
  ok(tun.statistical.run.minutes === 300 && tun.statistical.run.result === 'ok' && tun.statistical.revertedRows.lotto.join() === 'LaplaceMonteCarlo Model' && tun.boosting.run === null,
    'tuning: each tuner\'s run of that weekend, its replaced and reverted rows by game');
  ok(t.unattributed.map((x) => x.row).sort().join() === 'ClassicalSVM Model,Mystery Model,PoissonMarkov Model,XGBoost Model'
    && /RL ticket tuner/.test(t.unattributed.find((x) => x.row === 'XGBoost Model').why), 'tuning: every unattributed record with why');
  ok(t.history.weekends.join() === '2026-10-17,2026-10-10' && t.history.error === null && t.history.commits === 4, 'tuning: the weekends read from git');
  const lotto2 = t.byGame.find((x) => x.game === 'lotto');
  ok(lotto2.n === 7 && lotto2.notThisWeekend === 1 && t.byGame.find((x) => x.game === 'keno').noFile === true && t.byGame.reduce((s, x) => s + x.replaced, 0) === t.tiles.replaced,
    'tuning: by game, the games without a file named');
  const tl = (await requestAsync(app2, '/api/admin/tuning/lotto', ADMIN)).body;
  const by = Object.fromEntries(tl.rows.map((r) => [r.row, r]));
  ok(tl.rows.length === 8 && tl.weekends.join() === '2026-10-17,2026-10-10' && by['Markov Model'].history.join() === 'kept,replaced' && by['XGBoost Model'].history.join() === 'replaced,kept'
    && JSON.stringify(by['Mystery Model'].history) === '["replaced",null]' && JSON.stringify(by['PoissonMarkov Model'].history) === '[null,null]',
    'one game: each row\'s last weekends (null: not decided that weekend)');
  ok(by['Markov Model'].by === 'family + commit' && by['LaplaceMonteCarlo Model'].by === 'family + run' && by['QuantumVQC Model'].by === 'record' && by['QuantumVQC Model'].script === 'HyperoptQuantum'
    && by['ClassicalSVM Model'].tuner === 'unattributed' && by['PoissonMarkov Model'].thisWeekend === false && by['PoissonMarkov Model'].weekend === '2026-10-03',
    'one game: who decided each row and how that is known; a stale record says its own weekend');
  const m = by['Mystery Model'];
  ok(m.challenger.score === 0.9 && m.challenger.se === 0.08 && m.default.days === 87 && m.bar === 0.85 && m.gainSe === 5 && m.gainBestSe === 0.63 && m.margin === 0 && m.kind === 'hits'
    && m.windowDays === 90 && m.runTrials === 15 && m.studyTrials === 105 && m.reason === undefined, 'one game: the scores with their SE and days, the bar, the gains, the trials; no reason text');
  ok(tl.rows.map((r) => r.tuner).join() === 'statistical,statistical,RL ticket,quantum,unattributed,unattributed,unattributed,unattributed', 'one game: rows in chain order');
  ok(tl.unit === 'draws', 'one game: the unit its gate window counts in');
  ok(JSON.stringify(tun.statistical.windowByUnit) === '{"draws":{"min":90,"max":90}}' && tun.statistical.aboveFair === 0 && tun.statistical.fairPerBet === null,
    'tuning: the gate window per unit (draws, days, trading days, weeks), never pooled under one word');
  // keno: a capped-profit score is measured above a fair ticket - the side carries that ticket's value and the raw profit
  const kenoGate = { date: '2026-10-17T10:29:23+00:00', window_days: 90, margin: 0, decision: 'replaced', challenger: 0.0566, incumbent: -0.0616, default: -0.2016,
    challenger_tuning: { score: 0.0566, kind: 'capped_profit', days: 90, profit_per_bet: -0.1444, fair_capped_per_bet: -0.6435, score_se: 0.1869 },
    incumbent_tuning: { score: -0.0616, kind: 'capped_profit', days: 90, profit_per_bet: -0.6074, fair_capped_per_bet: -0.6435, score_se: 0.0977 }, default_tuning: null };
  const kh = { weekends: ['2026-10-17'], newest: '2026-10-17', decisions: {}, carriers: {}, commits: 0, error: null };
  const krecs = admin.tuningRecords({ keno: { 'Markov Model': kenoGate } }, kh, [{ key: 'hyperoptStatistics', at: Z('2026-10-17T09:00:00Z'), finishedAt: Z('2026-10-17T14:00:00Z') }]);
  const kt = admin.tuningGame(krecs, 'keno', kh).rows[0];
  const ks = admin.tuningSummary(krecs, kh).byTuner.find((x) => x.tuner === 'statistical');
  ok(kt.challenger.fairPerBet === -0.6435 && kt.challenger.profitPerBet === -0.1444 && kt.incumbent.profitPerBet === -0.6074 && kt.default.fairPerBet === undefined
    && m.challenger.fairPerBet === undefined && ks.aboveFair === 1 && ks.fairPerBet.min === -0.6435 && JSON.stringify(ks.windowByUnit) === '{"days":{"min":90,"max":90}}',
    'keno: a capped-profit side carries the fair ticket it is measured above and its raw profit per bet; a hits side neither');
  // mid-chain on Saturday: the tuners that have not written yet count no gate, with no margin to speak of
  ok(t.byTuner.find((x) => x.tuner === 'boosting').n === 0 && t.byTuner.find((x) => x.tuner === 'boosting').marginZero === 0, 'a tuner with no decision this weekend: n 0');

  // a commit whose file is not JSON (Python's json.dump writes NaN) is a fact about that commit, kept: no decision read
  // there; a file not in a commit the same - neither is a git failure. A git that fails otherwise is not kept.
  const nanRepo = fs.mkdtempSync(path.join(os.tmpdir(), 'admin-nan-'));
  const nanGit = (args) => String(execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', ...args],
    { cwd: nanRepo, stdio: 'pipe', env: { PATH: process.env.PATH, HOME: process.env.HOME || '/root', GIT_AUTHOR_DATE: '2026-10-10T12:00:00Z', GIT_COMMITTER_DATE: '2026-10-10T12:00:00Z' } }));
  try {
    nanGit(['init', '-q']);
    fs.writeFileSync(path.join(nanRepo, 'bestParams_lotto.json'), '{"tuningGate": {"Markov Model": {"date": "2026-10-10T10:00:00+00:00", "decision": "kept", "x": NaN}}}');
    nanGit(['add', 'bestParams_lotto.json']);
    nanGit(['commit', '-q', '-m', 'Saving latest statistical hyperopt']);
    const nanSha = nanGit(['rev-parse', 'HEAD']).trim();
    ok(await gates.gatesAt(nanRepo, nanSha, 'lotto') === null && await gates.gatesAt(nanRepo, nanSha, 'lotto') === null && await gates.gatesAt(nanRepo, nanSha, 'keno') === null,
      'a file that is not JSON in a commit, or not in it at all: null, read again from the cache - never a rejection');
    const nh = await gates.history(nanRepo, ['lotto'], { lotto: { 'Markov Model': markov17 } });
    ok(nh.error === null && nh.weekends.join() === '2026-10-17,2026-10-10' && JSON.stringify(nh.decisions.lotto['Markov Model']) === '["kept",null]',
      `the history reads on: that weekend did not decide the row, git did not fail: ${JSON.stringify(nh)}`);
    const failed = [];
    for (let i = 0; i < 2; i += 1) { try { await gates.gatesAt(bare, nanSha, 'lotto'); } catch (e) { failed.push(e.code); } }
    ok(failed.length === 2 && failed.every((c) => typeof c === 'number'), 'a git that fails for another reason (here: no repository) rejects each time: never kept as "not in that commit"');
  } finally {
    fs.rmSync(nanRepo, { recursive: true, force: true });
  }

  // git missing or broken: the working tree's weekend still reads, the history says why it stops there
  const app3 = fakeApp();
  admin.install(app3, { dataDir, root: bare });
  const tb = (await requestAsync(app3, '/api/admin/tuning', ADMIN)).body;
  ok(tb.history.weekends.join() === '2026-10-17' && /^(git exited with code \d+|git failed|git is not installed here)$/.test(tb.history.error) && tb.tiles.n === 7 && tb.tiles.unattributed === 6,
    `no git: this weekend only, said why in a fixed phrase (never git's own words or a path); without a commit or a run nothing is attributed by family alone: ${tb.history.error}`);

  // --- the week
  const live = { lock: { pid: 4242, script: 'Predictor.py', key: 'predictor' }, current: null, queue: [], council: { status: 'running' }, events };
  const view = (nowIso, extra = {}) => {
    const inputs = admin.readAll(admin.resolveDirs({ dataDir }));
    const dirs = admin.resolveDirs({ dataDir, root: tmp });
    return admin.weekView({ jobs: TEST_JOBS, sched: admin.readScheduler(dirs), slices: Object.fromEntries(TEST_JOBS.map((j) => [j.key, admin.readSlices(dirs, j)])),
      controlsView: admin.withHistory(admin.controlsOverview(inputs, report), admin.weekReading(inputs, report), admin.loadReadings(readingsDir)), tuning: t, live: { ...live, ...extra }, now: new Date(nowIso) });
  };
  const w = view('2026-10-21T18:00:00Z');
  const we = w.tiles.weekend;
  ok(w.weekend.saturday === '2026-10-17' && we.runs === 4 && we.ok === 3 && we.expected === 4 && we.missing.length === 0 && we.notOk[0].key === 'nullControls' && we.notOk[0].result === 'exit 1'
    && we.from === '2026-10-17T07:00:00Z' && we.to === '2026-10-18T22:40:00Z' && we.hours === 39.67, `week: the weekend's runs n/n ok and its window: ${JSON.stringify(we)}`);
  const stat = w.runs.find((r) => r.key === 'hyperoptStatistics');
  ok(JSON.stringify(stat.usual) === '{"median":110,"min":100,"max":120,"n":2,"from":"2026-10-03T08:38:29Z","to":"2026-10-10T08:52:22Z","basis":"earlier runs in the job log"}'
    && stat.minutes === 300 && stat.overUsual === true && stat.vsMedian === 2.73 && we.slowest.key === 'hyperoptStatistics',
    `week: a weekly job against the median of its earlier weekends, from its log: ${JSON.stringify(stat.usual)}`);
  ok(JSON.stringify(stat.warnings) === '[{"sig":"LOCKBOX","n":1,"new":false},{"sig":"RuntimeWarning (MarketModels.py)","n":1,"new":true},{"sig":"Traceback","n":1,"new":false}]'
    && stat.noiseFiltered === 2 && stat.comparedWith.runs === 2,
    `week: warning signatures, new only when absent from every earlier weekend the logs hold (Traceback was there on 3 Oct, in the rotated log): ${JSON.stringify(stat.warnings)}`);
  const q0run = w.runs.find((r) => r.key === 'nullControls');
  ok(q0run.usual === null && q0run.warnings[0].new === null && q0run.comparedWith === null && we.newWarnings.map((x) => `${x.key}:${x.sig}`).join() === 'hyperoptStatistics:RuntimeWarning (MarketModels.py)',
    'week: with nothing earlier to compare, neither usual nor new is claimed');
  const sat = w.runs.find((r) => r.key === 'predictor');
  ok(sat.usual.median === 120 && sat.usual.n === 5 && sat.usual.basis === 'earlier runs in the scheduler history' && sat.warnings[0].new === false && w.runs.length === 4,
    'week: the predictor against its earlier runs; dry and blocked entries are not runs');
  ok(w.tiles.today.day === '2026-10-21' && w.tiles.today.run.minutes === 108 && w.tiles.today.run.warnings.find((x) => x.sig === 'Traceback').new === true && w.tiles.today.running === null,
    'week: today\'s predictor, a new Traceback in it flagged');
  ok(w.since.from === '2026-10-18T22:00:00Z' && w.since.predictorRuns.length === 3 && JSON.stringify(w.since.lock) === '{"held":true,"pid":4242,"script":"Predictor.py","key":"predictor"}'
    && w.since.council.status === 'running' && w.since.queue.length === 0, 'week: since Monday (Brussels): the predictor runs, the lock holder, the Council API');
  ok(JSON.stringify(w.since.signIns) === '{"failed":1,"locked":0,"ok":1,"last7Days":{"failed":1,"locked":1},"scanned":3,"oldestScanned":"2026-10-16T10:00:00Z"}'
    && !/mallory|203\.0\.113/.test(JSON.stringify(w)), 'week: sign-ins counted, with no name or address sent');
  ok(w.tiles.tuners.replaced === 3 && w.tiles.tuners.replacedUnder1Se === 1 && w.tiles.tuners.byTuner.length === t.byTuner.length && w.tiles.q0.rowsAbove === ch.tiles.q0.rowsAbove
    && w.tiles.q2.above === ch.tiles.q2.above, 'week: the tuners\' and controls\' tiles are the Tuning and Controls pages\' own');
  ok(w.tiles.tuners.byTuner.every((x, i) => x.job === t.byTuner[i].job && x.games === t.byTuner[i].games && x.marginOneSe === t.byTuner[i].marginOneSe)
    && w.rankingAt === ch.rankingAt && typeof w.rankingAt === 'string',
    'week: each tuner\'s job, games and 1-SE margins as on the Tuning page; the ranking time the Q0 tile reads against');
  const lm = w.matrix.games.find((g) => g.game === 'lotto');
  ok(w.matrix.games.length === 11 && lm.q0.above === 2 && lm.q0.history.weeks === 2 && lm.q2.history.changed === true && lm.gates.replaced === 3 && lm.gates.notThisWeekend === 1
    && w.matrix.games.find((g) => g.game === 'euromillions').q0 === null && w.readings.stored === 2, 'week: the controls matrix, each cell with its weeks; colour only where changed');
  // before Saturday's first run the page stays on last weekend; once a job of it runs, on this one
  ok(view('2026-10-24T05:00:00Z').weekend.saturday === '2026-10-17', 'week: on Saturday morning, still last weekend');
  const sat24 = view('2026-10-24T08:00:00Z', { current: { key: 'predictor', name: 'Daily predictor', startedAt: Z('2026-10-24T07:00:00Z') } });
  ok(sat24.weekend.saturday === '2026-10-24' && sat24.tiles.weekend.runs === 0 && sat24.tiles.weekend.running.key === 'predictor'
    && sat24.tiles.weekend.missing.map((e) => `${e.key}${e.day ? `@${e.day}` : ''}`).join() === 'predictor@2026-10-25,hyperoptStatistics,nullControls',
    'week: once Saturday\'s predictor runs, the new weekend - the running job is not missing, Sunday\'s still is');
  const weekRoute = (await requestAsync(app2, '/api/admin/week', ADMIN)).body;
  ok(weekRoute.since.council.healthy === true && weekRoute.since.council.since === '2026-10-20T20:16:00Z' && weekRoute.since.lock.held === true && weekRoute.matrix.games.length === 11,
    'week route: the live parts from the modules the server passes');
  const app4 = fakeApp();
  admin.install(app4, { dataDir, root: tmp, services: { status: () => Promise.reject(new Error('probe failed')) } });
  const w4 = (await requestAsync(app4, '/api/admin/week', ADMIN)).body;
  ok(w4.runs.length === 0 && w4.since.lock === null && w4.since.council.error === 'probe failed' && w4.since.signIns === null,
    'week route: without the job list there are no runs; what is not known here reads null, a failed probe says so');
  const app5 = fakeApp();
  admin.install(app5, { dataDir, root: tmp, services: { status: async () => [{ key: 'councilApi', name: 'Council API', status: 'stopped', healthy: false, summary: '', since: null, restarts: 2,
    lastExit: null, lastError: 'cannot open /root/sequencePredictor/log/council-api.log: ENOENT: no such file or directory, open \'/root/sequencePredictor/log/council-api.log\'' }] } });
  const w5 = (await requestAsync(app5, '/api/admin/week', ADMIN)).body;
  ok(w5.since.council.lastError === 'cannot open council-api.log: ENOENT: no such file or directory, open \'council-api.log\'',
    `week route: the Council's last error without the server's paths: ${w5.since.council.lastError}`);
  ok(w5.matrix.fewWindows === 250 && w5.matrix.games.find((g) => g.game === 'lotto').q0.modes.map((x) => x.mode).sort().join() === 'shuffled,synthetic',
    'week route: Q2\'s power rule, and each Q0 control record\'s time for what a run wrote');
}

// production's own files for phase P2, read-only: the week and the tuning gates with production's root (git read-only)
async function phaseTwoOnProduction() {
  const PROD_ROOT = '/root/sequencePredictor';
  if (!fs.existsSync(path.join(PROD_ROOT, 'config', 'scheduler.json'))) return;
  const jobs = require('../jobs');
  const sizes = [];
  for (const readingsAt of [path.join(PROD_ROOT, 'data', 'readings'), path.join(__dirname, '..', 'data', 'readings')]) {
    const prod = fakeApp();
    admin.install(prod, { root: PROD_ROOT, readingsDir: readingsAt, jobs: { JOBS: jobs.JOBS } });
    for (const url of ['/api/admin/week', '/api/admin/tuning', ...admin.GAMES.map((g) => `/api/admin/tuning/${g}`), '/api/admin/controls']) {
      const r = await requestAsync(prod, url, ADMIN);
      const text = JSON.stringify(r.body);
      sizes.push([url, text.length]);
      ok(r.statusCode === 200 && !/NaN|undefined/.test(text), `production ${url} (${fs.existsSync(readingsAt) ? 'with' : 'without'} readings): ${r.statusCode}`);
      ok(text.length < 40 * 1024, `production ${url}: ${text.length} bytes, under 40 KB`);
    }
    const t = (await requestAsync(prod, '/api/admin/tuning', ADMIN)).body;
    ok(t.byTuner.reduce((s, x) => s + x.n, 0) === t.tiles.n && t.byGame.reduce((s, x) => s + x.n, 0) === t.tiles.n
      && t.byGame.reduce((s, x) => s + x.replaced, 0) === t.tiles.replaced, 'production: the tuning tiles count the rows of the cards');
    const w = (await requestAsync(prod, '/api/admin/week', ADMIN)).body;
    ok(w.tiles.weekend.runs === w.runs.length && w.tiles.weekend.ok === w.runs.filter((r) => r.result === 'ok').length && w.matrix.games.length === 11
      && w.tiles.tuners.n === t.tiles.n, 'production: the week\'s tiles count its runs and match the Tuning page');
  }
  const biggest = sizes.sort((a, b) => b[1] - a[1])[0];
  console.log(`admin.js phase P2 on production: ${sizes.length} views, the largest ${biggest[0]} at ${biggest[1]} bytes`);
}

// --- production's own files, read-only, when this checkout runs next to them --------------------------------
const PROD = '/root/sequencePredictor/data';
if (fs.existsSync(path.join(PROD, 'database', 'modelPerformance.json'))) {
  sameRanking(path.join(PROD, 'database'), admin.GAMES, 'production');
  const prod = fakeApp();
  admin.install(prod, { dataDir: PROD });
  const sizes = [];
  const urls = ['/api/admin/controls', '/api/admin/models', '/api/admin/phase-shift', '/api/admin/randomness']
    .concat(admin.GAMES.flatMap((g) => [`/api/admin/controls/${g}`, `/api/admin/models/${g}`, `/api/admin/models/${g}?scope=top`, `/api/admin/phase-shift/${g}`, `/api/admin/randomness/${g}`]));
  urls.forEach((url) => {
    const r = request(prod, url, ADMIN);
    if (r.statusCode === 404 && /randomness\/|phase-shift\//.test(url)) return;   // a game without that record
    const text = JSON.stringify(r.body);
    sizes.push([url, text.length]);
    ok(r.statusCode === 200 && !/NaN|undefined/.test(text), `production ${url}: ${r.statusCode}`);
    // the plan's target is ~20 KB a page; the limit leaves room for the records to grow (more rows, more weeks)
    // so a normal week's growth does not fail the suite - a jump past it means raw records leaked into a response
    ok(text.length < 40 * 1024, `production ${url}: ${text.length} bytes, under 40 KB`);
  });
  const pc = request(prod, '/api/admin/controls', ADMIN).body;
  const games = pc.groups.flatMap((g) => g.games);
  ok(pc.tiles.q0.rowsAbove === games.filter((g) => g.q0.ranking === 'hits').reduce((s, g) => s + g.q0.above.n, 0)
    && pc.tiles.fc.variants === games.reduce((s, g) => s + (g.fc ? g.fc.variants.length : 0), 0), 'production: the tiles count the rows of the cards');
  const pm = request(prod, '/api/admin/models', ADMIN).body;
  const report0 = rankings.loadReport(path.join(PROD, 'database'));
  // the users' five as /lottery picks them: server.js dayFiles' newest file, generateTable's old rule on its names
  const newestNames = (game) => {
    const dir = path.join(PROD, 'database', game);
    const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json')).sort((a, b) => new Date(b.replace('.json', '')) - new Date(a.replace('.json', '')));
    return [...new Set((JSON.parse(fs.readFileSync(path.join(dir, files[0]), 'utf-8')).newPrediction || []).map((x) => x.name || 'not known'))];
  };
  ok(pm.games.every((g) => JSON.stringify(g.top) === JSON.stringify(oldTop(oldRankingFor(path.join(PROD, 'database'), g.game), newestNames(g.game))))
    && pm.games.length === Object.keys(report0.games).length, 'production: every game\'s top five is the five its newest day\'s table shows a user');
  const markets = require('../markets');
  const pickedPlan = markets.pickPlan(markets.describeMarket(markets.loadMarket(path.join(PROD, 'markets'), 'crypto')), 'crypto').model.name;
  const pickedWeek = markets.pickWeek(markets.describeMarket(markets.loadMarket(path.join(PROD, 'markets'), 'sharesweek')), 'shares').model.name;
  ok(pm.games.find((g) => g.game === 'crypto').follows.row === pickedPlan && pm.games.find((g) => g.game === 'sharesweek').follows.row === pickedWeek
    && pm.games.find((g) => g.game === 'lotto').follows === undefined, 'production: a market game names the row its users\' page follows');
  const biggest = sizes.sort((a, b) => b[1] - a[1])[0];
  console.log(`admin.js on production: ${sizes.length} views, the largest ${biggest[0]} at ${biggest[1]} bytes`);
}

// --- the same routes through express itself, when it can be loaded (NODE_PATH) --------------------------------
let express = null;
try { express = require('express'); } catch (e) { express = null; }
const cleanUp = () => { fs.rmSync(tmp, { recursive: true, force: true }); fs.rmSync(bare, { recursive: true, force: true }); };
function throughExpress() {
  if (!express) {
    cleanUp();
    console.log(`admin.js: ${checks} checks passed (express not installed here: the route checks ran on the stand-in)`);
    return Promise.resolve();
  }
  const real = express();
  real.use((req, res, next) => { const who = req.get('x-test-user'); req.user = who === 'admin' ? ADMIN : (who === 'user' ? USER : undefined); next(); });
  admin.install(real, { dataDir, root: tmp, jobs: { JOBS: TEST_JOBS } });
  return new Promise((resolve) => {
    const server = real.listen(0, '127.0.0.1', async () => {
      const base = `http://127.0.0.1:${server.address().port}`;
      try {
        const get = async (url, who) => { const r = await fetch(base + url, { headers: who ? { 'x-test-user': who } : {} }); return { status: r.status, body: await r.json() }; };
        ok((await get('/api/admin/models/lotto?scope=top', 'user')).status === 403, 'express: 403 for a user');
        ok((await get('/api/admin/models/lotto?scope=top')).status === 401, 'express: 401 without a session');
        const r = await get('/api/admin/models/lotto?scope=top', 'admin');
        ok(r.status === 200 && r.body.shown === 5, 'express: 200 for the administrator, the scope read from the query');
        ok((await get('/api/admin/controls/bingo', 'admin')).status === 404 && (await get('/api/admin/else', 'admin')).body.error === 'not found', 'express: the JSON 404s');
        const wk = await get('/api/admin/week', 'admin');
        const tg = await get('/api/admin/tuning/lotto', 'admin');
        ok(wk.status === 200 && wk.body.matrix.games.length === 11 && tg.status === 200 && tg.body.rows.length === 8 && (await get('/api/admin/tuning', 'user')).status === 403
          && (await get('/api/admin/tuning/bingo', 'admin')).status === 404, 'express: the routes that answer later, the guard and the 404 in front of them');
        console.log(`admin.js: ${checks} checks passed (express included)`);
      } catch (e) {
        console.error(e); process.exitCode = 1;
      } finally {
        server.close();
        cleanUp();
        resolve();
      }
    });
  });
}
phaseTwo().then(phaseTwoOnProduction).then(throughExpress).catch((e) => {
  console.error(e);
  process.exitCode = 1;
  cleanUp();
});
