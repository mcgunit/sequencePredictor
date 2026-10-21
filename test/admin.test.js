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
function request(app, url, user) {
  const [pathname, qs] = url.split('?');
  const res = { statusCode: 200, headers: {}, body: undefined,
    status(c) { this.statusCode = c; return this; }, set(k, v) { this.headers[k] = v; return this; }, json(b) { this.body = b; return this; } };
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
    return undefined;
  };
  next();
  return res;
}
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
const cleanUp = () => fs.rmSync(tmp, { recursive: true, force: true });
if (!express) {
  cleanUp();
  console.log(`admin.js: ${checks} checks passed (express not installed here: the route checks ran on the stand-in)`);
} else {
  const real = express();
  real.use((req, res, next) => { const who = req.get('x-test-user'); req.user = who === 'admin' ? ADMIN : (who === 'user' ? USER : undefined); next(); });
  admin.install(real, { dataDir, root: tmp });
  const server = real.listen(0, '127.0.0.1', async () => {
    const base = `http://127.0.0.1:${server.address().port}`;
    try {
      const get = async (url, who) => { const r = await fetch(base + url, { headers: who ? { 'x-test-user': who } : {} }); return { status: r.status, body: await r.json() }; };
      ok((await get('/api/admin/models/lotto?scope=top', 'user')).status === 403, 'express: 403 for a user');
      ok((await get('/api/admin/models/lotto?scope=top')).status === 401, 'express: 401 without a session');
      const r = await get('/api/admin/models/lotto?scope=top', 'admin');
      ok(r.status === 200 && r.body.shown === 5, 'express: 200 for the administrator, the scope read from the query');
      ok((await get('/api/admin/controls/bingo', 'admin')).status === 404 && (await get('/api/admin/else', 'admin')).body.error === 'not found', 'express: the JSON 404s');
      console.log(`admin.js: ${checks} checks passed (express included)`);
    } catch (e) {
      console.error(e); process.exitCode = 1;
    } finally {
      server.close();
      cleanUp();
    }
  });
}
