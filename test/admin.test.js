'use strict';
// The admin section's JSON (admin.js) and the shared History ranking (ranking.js): the guard (401 signed out, 403 for
// a user, through for the administrator), every route's shape on hand-made records written to a temp dir, the rules
// the builders apply (Q0's row tests with the band as context, the live split, the keno bound against a fair ticket, the combination's p
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
// a deep-learning row's learning_check as Helpers.py writes it (src/LearningCheck.py plus the day, 25 Oct 2026)
const learningCheck = (verdict, valLoss, baselineLoss, margin, sameTicket, date = '2026-10-25') => ({ learning_check: { valLoss, baselineLoss, margin, verdict, sameTicket,
  inputBlind: sameTicket >= 0.95, windows: 699, date } });
const lagHistory = (lags) => lags.map((lag, i) => ({ run: `2026-10-${String(i + 1).padStart(2, '0')}`, lag, avg_hits: 1 + i / 100, n: 80 + i, z: 2 + i / 10, profile_mean: 0.8, metric: 'main_hits' }));
const lag = (lags) => { const h = lagHistory(lags); const counts = {}; h.forEach((r) => { counts[r.lag] = (counts[r.lag] || 0) + 1; }); const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return { peak: h[h.length - 1], history: h, runs: h.length, lag_counts: counts, consensus_lag: Number(best[0]), consensus_runs: best[1], best_lag: h[h.length - 1].lag }; };
// Q0's row tests as the daily report writes them (src/RowChance.py, 27 Oct 2026): a game's summary and each row's block.
// lotto: B confirmed on new draws and the one row that holds, A followed (running), C under 5% only, E judged not confirmed,
// Young too short to be compared; keno (a payout game: information only) K2 followed; crypto L under 5% on its live days.
const fresh = (state, since, length, draws, extra = {}) => ({ since, flagDraws: length, flagP: 0.03, flagMean: 1.4, flagNull: 0.8, verdictLength: length, draws, last: '2026-10-17',
  state, verdictAt: null, decidedOn: null, p: null, mean: draws ? 1.5 : null, null: draws ? 0.8 : null, ...extra });
const tested = (draws, notLive, mean, nul, p, pHolm, extra = {}) => ({ chance: { draws, notLive, mean, null: nul, nullSd: 0.1, p, inFamily: pHolm !== null, pHolm, last: '2026-10-17', ...extra } });
const summary = (method, over) => ({ method, alpha: 0.05, verdictAlpha: 0.01, minDraws: 10, permutations: method === 'permutation' ? 20000 : null, block: method === 'permutation' ? 7 : null,
  family: 0, liveMax: 0, under: 0, expected: 0, holds: 0, followed: 0, running: 0, stopped: 0, confirmed: 0, notConfirmed: 0, checkedThrough: '2026-10-18', stored: 'written', ...over });
const report = {
  generatedAt: '2026-10-21T08:48:01', since: { date: '2026-09-27', label: 'September 2026 design freeze' },
  games: {
    lotto: {
      metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'A Model',
      models: [
        // A collapsed onto the frequencies (input-blind), C learns, D carries a verdict LearningCheck.py never writes, Young
        // memorises (worse than the frequencies, its ticket follows its input) and is outside the top five
        lottoRow('A Model', 14, 1.3, 3, { multi_pick: { draws: 4, avg_hits: 1.5, base_avg_hits: 1.25, win_rate: 0.25, base_win_rate: 0, extra_hit_rates: [0.25, 0, 0] },
          ...learningCheck('counts-only', 2.30331, 2.3034, 0.004, 1.0), ...tested(14, 0, 1.3, 0.8, 0.021, 0.126, { fresh: fresh('running', '2026-10-11', 10, 4) }) }),
        lottoRow('B Model', 30, 1.2, 4, tested(30, 0, 1.2, 0.8, 0.004, 0.024, { fresh: fresh('confirmed', '2026-09-13', 12, 20, { verdictAt: '2026-10-10', decidedOn: '2026-10-12', p: 0.006 }) })),
        lottoRow('C Model', 90, 1.0, 3, { ...learningCheck('learns', 2.2501, 2.3034, 0.004, 0.31), ...tested(80, 10, 1.0, 0.85, 0.04, 0.16) }),
        lottoRow('D Model', 90, 0.9, 3, { ...learningCheck('maybe', 2.3, 2.3, 0, 0.5), ...tested(80, 10, 0.9, 0.85, 0.3, 0.9) }),
        lottoRow('E Model', 90, 0.85, 2, { learning_check: null, ...tested(80, 10, 0.85, 0.85, 0.5, 1,
          { fresh: fresh('not confirmed', '2026-06-07', 30, 60, { verdictAt: '2026-08-29', decidedOn: '2026-10-21', p: 0.5, mean: 0.8 }) }) }),
        lottoRow('F Model', 90, 0.8, 2, { multi_pick: { draws: 4, avg_hits: 1, base_avg_hits: 1, win_rate: 0, base_win_rate: 0, extra_hit_rates: [0, 0, 0] }, ...tested(80, 10, 0.8, 0.85, 0.7, 1) }),
        lottoRow('Young Model', 3, 2, 3, { ...learningCheck('counts-only', 3.5212, 2.3034, 0.004, 0.0472, '2026-10-24'), ...tested(3, 0, 2, 0.8, 0.02, null) }),
      ],
      chance: summary('exact', { family: 6, liveMax: 80, under: 3, expected: 0.27, holds: 1, followed: 1, running: 1, confirmed: 1, notConfirmed: 1 }),
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
          profit_by_size: { 5: { bets: 13, profit_total: -11, profit_per_bet: -0.846 }, 10: { bets: 48, profit_total: -16, profit_per_bet: -0.333 } }, ...tested(48, 0, 6.02, 5.71, 0.2, 0.4) },
        { name: 'K2 Model', draws: 300, avg_hits: 6.3, profit_bound: 0.04, best_hits: 12, profit_total: 80, profit_per_bet: 0.15, bets: 560, profit_by_size: null,
          ...tested(250, 50, 6.3, 5.71, 0.03, 0.06, { fresh: fresh('running', '2026-10-04', 52, 14) }) },
      ],
      chance: summary('exact', { family: 2, liveMax: 250, under: 1, expected: 0.09, followed: 1, running: 1 }),
      combinations: { metric: 'profit_per_draw', minDrawsForRanking: 10, candidateRows: ['K1 Model', 'K2 Model'], evaluated: 1,
        best: { members: ['K1 Model', 'K2 Model'], size: 2, how: 'pair', draws: 48, profit_per_bet: 1.2, profit_per_draw: 44.4, win_day_rate: 0.24, avg_hits: 6, best_hits: 11, value: 44.4 },
        ranking: [{ members: ['K1 Model', 'K2 Model'], size: 2, how: 'pair', draws: 48, profit_per_bet: 1.2, profit_per_draw: 44.4, win_day_rate: 0.24, avg_hits: 6, best_hits: 11, value: 44.4 }],
        bestSingle: { name: 'K2 Model', value: 21.3, draws: 48 }, beatsBestSingle: true, control: { shuffles: 100, best_mean: 33.8, best_p95: 99.7, p_value: 0.297 } },
    },
    crypto: {
      metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'L Model',
      models: [lottoRow('L Model', 20, 1.25, 3, tested(20, 0, 1.25, 0.9, 0.03, 0.06)), lottoRow('M Model', 50, 0.94, 3, tested(20, 30, 0.9, 0.85, 0.3, 0.3)),
        lottoRow('N Model', 50, 0.5, 2, { chance: { draws: 0, notLive: 50, inFamily: false } })],
      chance: summary('permutation', { family: 2, liveMax: 20, under: 1, expected: 0.1 }),
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

// controls: grouped, every game, the Q0 row tests (27 Oct 2026) with the band as context
const c = request(app, '/api/admin/controls', ADMIN).body;
ok(c.groups.map((g) => g.name).join() === 'Lottery,Markets' && c.groups[0].games.length === 7 && c.groups[1].games.length === 4, 'controls: 11 games, grouped Lottery / Markets');
const cg = Object.fromEntries(c.groups.flatMap((g) => g.games).map((g) => [g.game, g]));
ok(cg.lotto.q0.band.ceiling === undefined && cg.lotto.q0.band.from === 'synthetic' && cg.lotto.q0.band.seeds === 3 && cg.lotto.q0.band.history === 120 && cg.lotto.q0.band.at === '2026-10-18T10:11:55Z'
  && cg.lotto.q0.band.modes.length === 2, `Q0: the band as context - its control, seeds, history and its records' times; its ceiling only in the drawer: ${JSON.stringify(cg.lotto.q0.band)}`);
const lq = cg.lotto.q0;
ok(lq.test.family === 6 && lq.test.under === 3 && lq.test.expected === 0.27 && lq.test.holds === 1 && lq.lowest.row === 'B Model' && lq.lowest.p === 0.004
  && lq.rows.map((r) => r.row).join() === 'B Model,A Model,C Model' && lq.confirmed.join() === 'B Model' && lq.followed.join() === 'A Model' && lq.error === null,
  `Q0: the game's test; the rows named are those under 5%, followed or confirmed, lowest p first (E, judged not confirmed, only counted): ${JSON.stringify(lq.rows.map((r) => r.row))}`);
ok(lq.rows[1].fresh.flagP === undefined && lq.rows[1].fresh.decidedOn === undefined && lq.rows[1].fresh.since === '2026-10-11',
  'the overview leaves a fresh record\'s flag figures to the drawer');
const A = lq.rows.find((r) => r.row === 'A Model');
ok(A.mean === 1.3 && A.live === 14 && A.null === 0.8 && A.p === 0.021 && A.pHolm === 0.126 && A.fresh.state === 'running' && A.fresh.verdictLength === 10 && A.fresh.draws === 4
  && lq.rows.every((r) => r.mean !== undefined && r.live !== undefined && r.null !== undefined) && !lq.rows.some((r) => r.row === 'Young Model'),
  'Q0: each row line carries its own mean, live draws and null beside its p (the trust rule); a row outside the family is not named under 5%');
ok(lq.above === undefined && lq.best === undefined && lq.rankedRows === undefined, 'Q0: no "above the band" anywhere');
ok(cg.lotto.unit === 'draws' && cg.crypto.unit === 'days' && cg.cryptoweek.unit === 'weeks' && cg.shares.unit === 'trading days', 'every game says what one of its draws is');
ok(cg.keno.q0.ranking === 'profit' && cg.keno.q0.inTile === false && cg.keno.q0.leader.row === 'K1 Model' && cg.keno.q0.followed.join() === 'K2 Model' && cg.keno.q0.test.under === 1,
  'Q0: a payout game is tested on hits, names its profit-bound leader without a euro figure and stays out of the headline');
const L = cg.crypto.q0.rows.find((r) => r.row === 'L Model');
ok(L && L.live === 20 && L.notLive === 0 && cg.crypto.q0.test.method === 'permutation' && !cg.crypto.q0.rows.some((r) => r.row === 'M Model' || r.row === 'N Model'),
  `Q0: a market row tested on its live days; rows over 5% are left out of the card: ${JSON.stringify(L)}`);
const headline = c.groups.flatMap((g) => g.games).filter((g) => g.q0.test && g.q0.inTile);
ok(c.tiles.q0.confirmed === 1 && JSON.stringify(c.tiles.q0.confirmedRows) === '[{"game":"lotto","row":"B Model"}]'
  && c.tiles.q0.confirmed === headline.reduce((s, g) => s + g.q0.confirmed.length, 0) && c.tiles.q0.under === 4 && c.tiles.q0.expected === 0.37 && c.tiles.q0.holds === 1
  && c.tiles.q0.followed === 1 && c.tiles.q0.notConfirmed === 1 && c.tiles.q0.family === 8 && c.tiles.q0.games === 2 && c.tiles.q0.hitRankedGames === 2,
  `Q0 tile: the confirmed rows of the hit-ranked games, this week's figures beside them, counted from the cards: ${JSON.stringify(c.tiles.q0)}`);
ok(JSON.stringify(c.tiles.q0.payout) === '{"games":["keno"],"under":1,"expected":0.09,"holds":0,"followed":1,"confirmed":0,"notConfirmed":0}',
  'Q0 tile: the payout games apart, information only');
// a payout game's confirmed row is counted apart and never in the tile's headline (27 Oct 2026 review)
const kenoConfirmed = JSON.parse(JSON.stringify(report));
kenoConfirmed.games.keno.models[1].chance.fresh = fresh('confirmed', '2026-08-02', 52, 52, { verdictAt: '2026-10-10', decidedOn: '2026-10-12', p: 0.004 });
const kc = admin.controlsOverview({}, kenoConfirmed).tiles.q0;
ok(kc.confirmed === 1 && kc.confirmedRows.every((x) => x.game !== 'keno') && kc.payout.confirmed === 1,
  `Q0 tile: a payout game's confirmed row stays out of the headline: ${JSON.stringify({ confirmed: kc.confirmed, payout: kc.payout })}`);
// Q2's power rule (controlRules.json, 27 Oct 2026) applies to the older records too: crypto's version-1 "above" on 106
// test windows reads "too few draws", so no tile names an above on too few windows any more; lotto's 3-seed "above" on
// 496 windows keeps what its flags said (rule 1) until a version-2 record replaces it
const FEW = require('../controlRules.json').q2.few_windows;
ok(cg.lotto.q2.status === 'above' && cg.lotto.q2.seeds === 3 && cg.lotto.q2.windowsTest === 496 && cg.lotto.q2.rule === 1 && cg.lotto.q2.thresholdSe === null
  && cg.crypto.q2.status === 'too few draws' && cg.crypto.q2.fewWindows === true && cg.crypto.q2.minWindows === FEW
  && c.tiles.q2.tooFewGames.join() === 'crypto' && c.tiles.q2.failedGames.length === 0 && c.tiles.q2.aboveFewWindows === undefined && c.tiles.q2.above === 1,
  `Q2: status, seeds and test windows; a version-1 above on fewer than ${FEW} windows reads too few draws`);
ok(c.meta.q2.fewWindows === FEW && !/0\.8/.test(c.meta.q2.rule) && c.meta.q2.rule.includes(`under ${FEW} test windows`), `Q2 meta: the power rule from controlRules.json: ${c.meta.q2.rule}`);
ok(cg.lotto.q2.drawsUsed === 1000 && cg.crypto.q2.drawsUsed === 212 && cg.crypto.q2.drawsCap === 1000,
  `Q2: the draws the windows came from, (train + test) / 2 + window - 1, apart from the 1000 asked: ${cg.crypto.q2.drawsUsed}`);
ok(c.tiles.q0.noTest.some((x) => x.game === 'euromillions' && x.why === 'no ranking for this game yet') && !c.tiles.q0.noTest.some((x) => x.game === 'lotto')
  && cg.euromillions.errors.includes('q0') && !cg.lotto.errors.includes('q0') && c.tiles.rows.missing.join() === 'shares,cryptoweek,sharesweek',
  'the tiles name the games with no row test (never "none under") and the market games with no rows report');
ok(cg.lotto.fc.fits === 1 && cg.lotto.fc.ignored === 1 && cg.lotto.fc.chanceShare === 0.25 && cg.lotto.fc.tableRows === 275 && cg.lotto.fc.lockboxRows === 25 && cg.lotto.fc.variants[0].key === 'lr' && cg.lotto.fc.variants[0].share === 0.3117,
  'FC: the verdict counts, the chance share 3 / (9 + 3), the table and lockbox rows in draws');
ok(cg.crypto.rows.window === 231 && cg.crypto.rows.rowsScored === 1 && cg.crypto.rows.betterThanUniform.join() === 'Chronos Model' && cg.crypto.rows.betterThanReference.length === 0
  && cg.crypto.rows.referenceVsUniform.mean === 0.0923 && cg.lotto.rows === null, 'market rows: the window, the rows against uniform and GARCH; a lottery game has none');
const drawer = request(app, '/api/admin/controls/crypto', ADMIN).body;
ok(drawer.q0.modes.shuffled.perRow['M Model'] === 0.642 && drawer.q2.perSeed.real.length === 0 && drawer.rows.table.length === 3 && drawer.regimes[0].label === 'calm', 'the drawer: the modes, the rows table and the regime reading');
ok(drawer.q0.table.map((r) => r.row).join() === 'L Model,M Model' && drawer.q0.table[1].notLive === 30 && drawer.q0.band.from === 'shuffled' && drawer.q0.band.ceiling === 0.8,
  'the drawer: every row of the family, lowest p first, with its not-live draws; the band as context');
const lottoTable = request(app, '/api/admin/controls/lotto', ADMIN).body.q0.table;
ok(lottoTable.map((r) => r.row).join() === 'B Model,A Model,C Model,D Model,E Model,F Model' && lottoTable.find((r) => r.row === 'E Model').fresh.state === 'not confirmed',
  'the lottery drawer: the whole family, the judged row with its verdict; the short row is not compared');
// the Market rows help names the probability floor, each row's kind and the Regime HMM feature set (the old markets page's C16 / C17)
ok(drawer.rows.floor === 0.001 && drawer.rows.table.map((x) => x.kind).join() === 'market,foundation,baseline' && drawer.rows.hmmFeatures === 'returns' && drawer.q2.quantum === true,
  `the drawer: the probability floor, each row's kind, the HMM feature set and the Q2 quantum flag: ${JSON.stringify([drawer.rows.floor, drawer.rows.table.map((x) => x.kind)])}`);
const lottoDrawer = request(app, '/api/admin/controls/lotto', ADMIN).body;
ok(lottoDrawer.q2.perSeed.real.length === 3 && lottoDrawer.fc.variants[1].above[0] === 'C Model' && lottoDrawer.rows === null && lottoDrawer.regimes === null, 'the lottery drawer: per seed, the variants in full, no market parts');

// The 26 Oct 2026 report (MarketRows.py): the baselines carry a log-score and every row is held against the column
// frequencies (vs_frequency). The fixture above is a report from before: its new parts read null or empty.
ok(cg.crypto.rows.frequency === null && cg.crypto.rows.betterThanFrequency.length === 0 && cg.crypto.rows.referenceVsFrequency === null
  && cg.crypto.rows.frequencyVsUniform === null && cg.crypto.rows.baselines === 1 && drawer.rows.smoothing === null && drawer.rows.table.every((x) => x.vsFrequency === null)
  && c.tiles.rows.frequencyGames === 0 && c.tiles.rows.referenceAboveFrequency === 0, 'market rows: a report from before 26 Oct 2026 has no frequency parts');
const iv = (mean, lo, hi, verdict) => ({ mean, lo, hi, days: 231, verdict });
const freqRecord = { market: 'crypto', generated_at: '2026-11-01T09:54:09+00:00', days_requested: 250, days_scored: 231, first_day: '2026-03-01', last_day: '2026-10-31',
  uniform_log_score: -2.3026, probability_floor: 0.0001, reference: 'GARCH Model', frequency_reference: 'column_frequency', frequency_smoothing: 1, lockbox_days_withheld: 19, errors: {},
  rows: [
    { name: 'GARCH Model', kind: 'market', days: 231, log_score: -2.208, vs_uniform: iv(0.095, 0.07, 0.12, 'better'), vs_reference: null, vs_frequency: iv(0.027, 0.01, 0.045, 'better') },
    { name: 'Chronos Model', kind: 'base', days: 231, log_score: -2.228, vs_uniform: iv(0.074, 0.05, 0.1, 'better'), vs_reference: iv(-0.02, -0.03, -0.01, 'worse'), vs_frequency: iv(0.007, -0.01, 0.02, 'no difference') },
    { name: 'column_frequency', kind: 'baseline', days: 231, log_score: -2.235, vs_uniform: iv(0.0676, 0.0476, 0.087, 'better'), vs_reference: iv(-0.027, -0.045, -0.01, 'worse'), vs_frequency: null },
    { name: 'global_frequency', kind: 'baseline', days: 231, log_score: -2.236, vs_uniform: iv(0.066, 0.045, 0.086, 'better'), vs_reference: iv(-0.028, -0.05, -0.01, 'worse'), vs_frequency: iv(0.011, 0.004, 0.019, 'better') },
    { name: 'Regime HMM Model', kind: 'market', days: 231, log_score: -2.246, vs_uniform: iv(0.057, 0.03, 0.08, 'better'), vs_reference: iv(-0.038, -0.05, -0.02, 'worse'), vs_frequency: iv(-0.011, -0.03, 0.01, 'no difference') },
    { name: 'random', kind: 'baseline', days: 231, log_score: -2.3026, vs_uniform: iv(0, 0, 0, 'no difference'), vs_reference: iv(-0.095, -0.12, -0.07, 'worse'), vs_frequency: iv(-0.068, -0.087, -0.048, 'worse') },
  ] };
const fr = admin.rowsOf(freqRecord);
ok(fr.rowsScored === 2 && fr.baselines === 3 && fr.betterThanUniform.join() === 'Chronos Model,Regime HMM Model' && fr.betterThanFrequency.length === 0 && fr.betterThanReference.length === 0
  && fr.frequency === 'column_frequency' && fr.referenceVsFrequency.verdict === 'better' && fr.frequencyVsUniform.mean === 0.0676,
  `market rows: two rows beat uniform, neither beats the frequencies, and a baseline beating another (global over column) is no row that beats them: ${JSON.stringify(fr)}`);
const fo = admin.controlsOverview({ crypto: { rows: freqRecord } }, report).tiles.rows;
ok(fo.frequencyGames === 1 && fo.referenceAboveFrequency === 1 && fo.beatReference === 0, 'the tile: GARCH beats the frequencies in 1 of the 1 game with the new report');
const fd = admin.controlsDrawer('crypto', { rows: freqRecord }, report).rows;
const fdRow = (name) => fd.table.find((x) => x.row === name);
ok(fd.smoothing === 1 && fd.table.length === 6 && fdRow('column_frequency').logScore === -2.235 && fdRow('column_frequency').frequency === true && fdRow('column_frequency').vsFrequency === null
  && fdRow('Chronos Model').vsFrequency.verdict === 'no difference' && fdRow('random').vsFrequency.mean === -0.068 && fd.table[2].row === 'column_frequency',
  'the drawer: every row with its interval against the frequencies, the baselines scored and sorted among the rows by log-score');
const fread = admin.weekReading({ crypto: { rows: freqRecord } }, report).games.crypto.rows;
ok(fread.rows.map((r) => r.row).join() === 'GARCH Model,Chronos Model,Regime HMM Model' && fread.rows[0].vsFrequency.verdict === 'better' && fread.rows[0].vsReference === undefined
  && fread.baselines.map((r) => r.row).join() === 'column_frequency,global_frequency,random' && fread.baselines[0].vsFrequency === undefined && fread.baselines[2].vsUniform.verdict === 'no difference'
  && admin.stateOf('rows', { rows: fread }) === 'beat GARCH: none · GARCH vs uniform: better',
  `the reading: the model rows as before (now with vs the frequencies), the scored baselines apart, the week-over-week state over the model rows: ${JSON.stringify(fread)}`);
const oldRead = admin.weekReading({ crypto: { rows: { ...freqRecord, frequency_reference: undefined, rows: freqRecord.rows.map(({ vs_frequency, ...r }) => (r.kind === 'baseline' ? { ...r, log_score: null } : r)) } } }, report).games.crypto.rows;
ok(oldRead.rows.length === 3 && oldRead.rows.every((r) => r.vsFrequency === undefined) && oldRead.baselines.length === 0, 'the reading of a report from before: no vsFrequency, no scored baselines');

// models: the summary and one game
const m = request(app, '/api/admin/models', ADMIN).body;
const mk = m.games.find((g) => g.game === 'keno');
ok(mk.best.value === 0.117 && mk.best.profitPerBet === -0.195 && /fair ticket/.test(mk.measure.against) && mk.chance.inTile === false && mk.band === undefined,
  'keno: the bound comes with the raw profit per bet and what it is measured against; its row test is on hits, information only');
const ml = m.games.find((g) => g.game === 'lotto');
ok(ml.top.join() === 'A Model,B Model,C Model,D Model,E Model' && ml.since.best.row === 'B Model' && ml.best.inBand === undefined && m.topN === 5, 'the summary: the five users see, the best since the freeze');
ok(ml.best.chance.row === 'A Model' && ml.best.chance.mean === 1.3 && ml.best.chance.live === 14 && ml.best.chance.null === 0.8 && ml.best.chance.p === 0.021
  && ml.best.chance.pHolm === 0.126 && ml.best.chance.fresh.state === 'running' && ml.chance.test.under === 3 && ml.chance.error === null && ml.chance.inTile === true,
  'the summary: the best row\'s test on its live draws, its own mean, draws and null beside its p (the all-draws value never stands alone next to it)');
ok(m.games.find((g) => g.game === 'crypto').top.join() === 'L Model,N Model' && request(app, '/api/admin/models/crypto?scope=top', ADMIN).body.rows.map((r) => r.row).join() === 'L Model,N Model',
  'the top rows are those in the newest day\'s table: a ranked row missing from it is passed over, as on the users\' pages');
const mc = m.games.find((g) => g.game === 'crypto');
const cm = request(app, '/api/admin/models/crypto', ADMIN).body;
ok(mc.best.live === 20 && cm.rows.find((r) => r.row === 'M Model').live === 20 && cm.rows.find((r) => r.row === 'N Model').live === 0 && ml.best.live === undefined,
  'a market row\'s live draws next to all its draws; none for a lottery row');
const all = request(app, '/api/admin/models/lotto', ADMIN).body;
const top = request(app, '/api/admin/models/lotto?scope=top', ADMIN).body;
ok(all.scope === 'all' && all.shown === 7 && all.of === 7 && top.scope === 'top' && top.shown === 5 && top.rows.every((r) => r.top) && top.of === 7, 'scope: every row, or the top five only');
ok(all.rows.find((r) => r.row === 'Young Model').young === true && all.rows.every((r) => r.inBand === undefined) && all.band === undefined
  && all.rows.find((r) => r.row === 'D Model').chance.p === 0.3 && all.rows.find((r) => r.row === 'Young Model').chance.inFamily === false
  && all.rows.find((r) => r.row === 'Young Model').chance.pHolm === null && all.chance.test.family === 6,
  'the young flag; no greying any more - each row its test line, a short row outside the family without a corrected p');
ok(all.multiPick.rows.length === 2 && top.multiPick.rows.length === 1 && all.multiPick.chance[9] === 0.08395, 'multi-pick: the rows with extras, filtered by the scope; the exact chance');
ok(all.combination.control.pValue === 0.188 && all.combination.control.consistentWithLuck === true && all.combination.beatsBestSingle === null, 'combination: the p value travels with the value; "beats" only for profit');
const kg = request(app, '/api/admin/models/keno', ADMIN).body;
ok(kg.rows[0].bySize.length === 2 && kg.rows[0].bySize[0].size === 5 && kg.combination.measure.capped === false && kg.combination.beatsBestSingle === true, 'keno: per ticket size, the uncapped combination');

// the learning check of the deep-learning rows (src/LearningCheck.py, 25 Oct 2026): every drawer row carries its
// newest one with the losses, the input-blind fact and its day, or null; the summary names those of the best row
// and the top five; a label only - no row moves, leaves the top five or changes its value on it
const lr = (name) => all.rows.find((r) => r.row === name).learning;
ok(JSON.stringify(lr('A Model')) === JSON.stringify({ verdict: 'counts-only', valLoss: 2.3033, baselineLoss: 2.3034, margin: 0.004, sameTicket: 1, inputBlind: true, windows: 699, date: '2026-10-25' }),
  `the drawer: a collapsed row's check with its losses, input-blind and its day: ${JSON.stringify(lr('A Model'))}`);
ok(lr('Young Model').verdict === 'counts-only' && lr('Young Model').inputBlind === false && lr('Young Model').valLoss > lr('Young Model').baselineLoss && lr('Young Model').date === '2026-10-24'
  && lr('C Model').verdict === 'learns', 'the drawer: a memorising row reads counts-only without being input-blind; a row that learns says so');
ok(['B Model', 'D Model', 'E Model', 'F Model'].every((n) => lr(n) === null) && kg.rows.every((r) => r.learning === null),
  'the drawer: no check, a null one or a verdict LearningCheck.py never writes is no learning check');
ok(JSON.stringify(Object.keys(ml.learning)) === '["A Model","C Model"]' && ml.learning['A Model'].verdict === 'counts-only' && Object.keys(mk.learning).length === 0,
  `the summary: the checks of the best row and the top five only (Young Model is outside them): ${JSON.stringify(Object.keys(ml.learning))}`);
ok(top.rows.find((r) => r.row === 'A Model').learning.inputBlind === true && top.rows.every((r) => r.learning === null || ['A Model', 'C Model'].includes(r.row)),
  'the top scope keeps the checks of its rows');
const unlabelled = JSON.parse(JSON.stringify(report));
unlabelled.games.lotto.models.forEach((x) => { delete x.learning_check; });
const plain = admin.modelsGame(unlabelled, 'lotto');
const labelled = admin.modelsGame(report, 'lotto');
const plainLotto = admin.modelsSummary(unlabelled).games.find((g) => g.game === 'lotto');
ok(JSON.stringify(plain.rows.map((r) => [r.rank, r.row, r.value, r.top, r.chance])) === JSON.stringify(labelled.rows.map((r) => [r.rank, r.row, r.value, r.top, r.chance]))
  && plain.top.join() === labelled.top.join() && plainLotto.best.row === ml.best.row && plainLotto.top.join() === ml.top.join() && Object.keys(plainLotto.learning).length === 0
  && plain.rows.every((r) => r.learning === null), 'the label moves no row: the same ranking, values, best row and top five without it');

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
ok(rl.status === 'watch' && rl.reasons.join() === 'entropy,q2' && rl.q2.real === 0.59 && rc.reasons.join() === 'ae' && rc.q2.status === 'too few draws' && rc.window === null && rw.games.length === 2,
  'randomness: which tripwire is up, the Q2 verdict next to it (too few draws is no Q2 reason); a market game with the autoencoder only');
ok(rw.meta.fewWindows === FEW && rc.q2.minWindows === FEW && rl.q2.label === 'ABOVE both controls', 'randomness: the power rule from controlRules.json, the label of an above');
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

// a lottery row's live record (src/Liveness.py, 27 Oct 2026): live_draws (live as stored + recovered), live.recovered,
// and - when the report ranks on it (rankedOn 'live', the owner's decision 1) - the live value as the ranking value
// with every stored draw's beside it, the 10-draw guard on live draws. A report without it shows no split.
const lotLive = {
  generatedAt: '2026-10-27T09:00:00', games: { vikinglotto: {
    metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'Meta Model', rankedOn: 'live',
    live: { metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'Meta Model', models: [] },
    liveness: { checked: 'stamp+git', error: null, draws: 48, allLive: 32, allCounted: 40, rowsLive: 418, rowsRecovered: 67, rowsNotLive: 57, newestNotLive: '2026-06-10',
      rewriteAudit: { rows: 67, hitsAtClose: 42, hitsStored: 57 }, unscheduled: [] },
    // the row tests on the live record (the 26 Oct 2026 production figures: Bayes 0.878 on 41 live draws, p 0.166)
    chance: summary('exact', { family: 2, liveMax: 41, under: 0, expected: 0.36 }),
    models: [
      { name: 'Meta Model', draws: 10, avg_hits: 1.0, best_hits: 2, live_draws: 10, live: { draws: 10, recovered: 0, not_live: 0, avg_hits: 1.0, best_hits: 2 },
        ...tested(10, 0, 1.0, 0.75, 0.2, 0.33) },
      { name: 'Bayes Model', draws: 48, avg_hits: 1.021, best_hits: 3, live_draws: 41, live: { draws: 41, recovered: 8, not_live: 7, avg_hits: 0.878, best_hits: 3 },
        ...tested(41, 7, 0.878, 0.75, 0.166, 0.33) },
      { name: 'New Model', draws: 12, avg_hits: 1.5, best_hits: 3, live_draws: 4, live: { draws: 4, recovered: 0, not_live: 8, avg_hits: 1.25, best_hits: 2 },
        ...tested(4, 8, 1.25, 0.75, 0.3, null) },
    ] } },
};
const vq = admin.controlsDrawer('vikinglotto', { q0: { synthetic: { best_row_mean: 0.86, best_row_sd_across_seeds: 0.07, generated_at: '2026-10-25T10:00:00', rows_compared: 8,
  seeds: [1, 2, 3], days: 120 } } }, lotLive).q0;
const vBayes = vq.table.find((r) => r.row === 'Bayes Model');
ok(vBayes && vBayes.live === 41 && vBayes.notLive === 7 && vBayes.recovered === 8 && vBayes.mean === 0.878 && vq.table.length === 2 && vq.rows.length === 0 && vq.band.ceiling === 1,
  `Q0: a lottery row tested on its live draws (the recovered ones as they stood at the close), its back-built or late ones apart; over 5%, not named on the card: ${JSON.stringify(vBayes)}`);
const vm = admin.modelsGame(lotLive, 'vikinglotto');
const vrow = (name) => vm.rows.find((r) => r.row === name);
ok(vm.rankedOn === 'live' && vm.minDraws === 10 && vrow('Bayes Model').value === 0.878 && vrow('Bayes Model').allValue === 1.021 && vrow('Bayes Model').liveDraws === 41
  && vrow('Bayes Model').draws === 48 && vrow('Bayes Model').live === 41 && vrow('Bayes Model').recovered === 8 && vrow('Bayes Model').young === false
  && vrow('New Model').young === true && vrow('New Model').value === 1.25 && vrow('New Model').allValue === 1.5 && /saved before the sales close/.test(vm.window),
  'models: ranked on the live record - the live value, every stored draw\'s beside it, the guard on live draws (12 stored, 4 live: short)');
ok(vm.liveness.checked === 'stamp+git' && vm.liveness.rowsRecovered === 67 && vm.liveness.rewriteAudit.hitsAtClose === 42 && vm.liveness.newestNotLive === '2026-06-10',
  'models: the game\'s liveness line - how it was judged, the rows recovered and their audit, the newest not live');
const vs = admin.modelsSummary(lotLive).games.find((g) => g.game === 'vikinglotto');
ok(vs.rankedOn === 'live' && vs.best.row === 'Meta Model' && vs.best.value === 1.0 && vs.best.allValue === 1.0 && vs.best.live === 10 && vs.rankedRows === 2,
  'the overview: the best row by the live record, its all-draws value beside it, ranked rows counted on live draws');
const noLive = JSON.parse(JSON.stringify(lotLive));
delete noLive.games.vikinglotto.rankedOn;
noLive.games.vikinglotto.models.forEach((m) => { delete m.live_draws; delete m.live; });
const nm = admin.modelsGame(noLive, 'vikinglotto');
ok(nm.rankedOn === 'all' && nm.rows.every((r) => r.live === undefined && r.recovered === undefined && r.allValue === null) && vrow('Bayes Model').value !== nm.rows.find((r) => r.row === 'Bayes Model').value,
  'a lottery report without a live record (before 27 Oct, or not judged): every stored draw, no split');
// a live record that reads 'unchecked' (no git, no earlier check: every unchecked draw counted not live) is no evidence
// of back-built or late tickets: no split, in the models nor in Q0
const uncheckedLive = JSON.parse(JSON.stringify(lotLive));
delete uncheckedLive.games.vikinglotto.rankedOn;
uncheckedLive.games.vikinglotto.liveness.checked = 'unchecked';
uncheckedLive.games.vikinglotto.since = { date: '2026-09-27', label: 'September 2026 design freeze', metric: 'avg_hits', minDrawsForRanking: 10, bestModel: 'Bayes Model',
  models: [{ name: 'Bayes Model', draws: 12, avg_hits: 1.1, best_hits: 3 }] };
// the report tests nothing on an unchecked record (Helpers.py: a chance block with its reason, no row block)
uncheckedLive.games.vikinglotto.chance = { method: 'exact', alpha: 0.05, verdictAlpha: 0.01, error: 'liveness not checked: git failed', family: 0 };
uncheckedLive.games.vikinglotto.models.forEach((m) => { delete m.chance; });
const um = admin.modelsGame(uncheckedLive, 'vikinglotto');
const uq = admin.q0Of('vikinglotto', { synthetic: { best_row_mean: 0.86, best_row_sd_across_seeds: 0.07, generated_at: '2026-10-25T10:00:00', rows_compared: 8, seeds: [1, 2, 3], days: 120 } },
  uncheckedLive.games.vikinglotto);
ok(um.rankedOn === 'all' && um.rows.every((r) => r.live === undefined && r.recovered === undefined && r.chance === null) && um.since.live === null && um.liveness.checked === 'unchecked'
  && um.chance.test === null && um.chance.error === 'liveness not checked: git failed' && uq.test === null && uq.error === 'liveness not checked: git failed' && uq.rows.length === 0,
  'an unchecked live record: no live split ("back-built or late" would be a claim without evidence) and no row test, the line says why');

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
  && cut.tiles.q0.band.oneControl.map((x) => x.game).join() === 'lotto,keno,crypto' && cutGames.lotto.q0.test.family === 6 && !cutGames.lotto.errors.includes('q0'),
  'a band from one control (context) names the other and why it is not there; the tile names the games; the row test is untouched by it');
const cutDrawer = admin.controlsDrawer('lotto', admin.readControls(cutDirs, 'lotto'), report);
ok(cutDrawer.errors.join() === 'q2,fc' && cutDrawer.q0.band.missing[0].unreadable === true, 'the drawer: the same parts unreadable, the band from one control');
// no band record read at all: only the context goes (bandUnreadable), never Q0 itself - it is the report's row test
const noBandDir = path.join(tmp, 'noband');
fs.cpSync(cutDir, noBandDir, { recursive: true });
fs.writeFileSync(path.join(noBandDir, 'results/lotto-shuffled.json'), '{"report": {"best_row_mean": ');
const noBand = admin.controlsOverview(admin.readAll(admin.resolveDirs({ dataDir, controlsDir: noBandDir })), report).groups[0].games.find((g) => g.game === 'lotto');
ok(noBand.q0.band === null && noBand.q0.bandUnreadable === true && !noBand.errors.includes('q0') && noBand.q0.test.under === 3,
  'both band records unreadable: the band (context) is missing and said so; Q0 is no error');
ok(admin.modelsGame(report, 'lotto', 'all', admin.readControls(cutDirs, 'lotto')).band === undefined, 'the Models page has no band any more');
const brokenQ2 = { ...broken.lotto, q2: { ...broken.lotto.q2, verdict: { ...broken.lotto.q2.verdict, per_seed_real: 5 } } };
ok(admin.controlsDrawer('lotto', brokenQ2, report).q2.perSeed.real.length === 0, 'a per-seed field that is not a list reads as none');
// a version-2 Q2 record (the 1 Nov 2026 run on, RandomnessDiscrimination.compute_verdict): its own status, the threshold
// with its standard error, k of both bounds, the seeds each comparison scored; a failed run reads failed - in the tile
// too - and never within
const q2v2 = (over) => ({ ...broken.lotto.q2, repetitions: 20, seeds: Array.from({ length: 20 }, (_, i) => 2026440001 + i), generated_at: '2026-11-01T10:17:37',
  verdict: { version: 2, status: 'within', metric: 'best AUC of the classifier suite, per repetition', real_max_of_suite_mean: 0.5671, real_max_of_suite_sd: 0.0502,
    null_max_of_suite_mean: 0.5762, null_max_of_suite_sd: 0.0441, shuffled_max_of_suite_mean: 0.5881, shuffled_max_of_suite_sd: 0.0589,
    seeds_scored: { real: 20, null: 20, shuffled: 20 }, band_coverage: 0.97725, band_k: 2.193, shuffled_k: 2.193, null_threshold: 0.6729, null_threshold_se: 0.0186,
    shuffled_threshold: 0.7173, per_seed_real: [0.58, 0.55], per_seed_null: [0.57, 0.6], per_seed_shuffled: [0.59, 0.6],
    above_null_band: false, survives_marginals_control: false, too_few_draws: false, few_windows: FEW, windows_test: 496, ...over } });
const d2 = admin.controlsDrawer('lotto', { ...broken.lotto, q2: q2v2({}) }, report).q2;
ok(d2.status === 'within' && d2.rule === 2 && d2.threshold === 0.673 && d2.thresholdExact === 0.6729 && d2.thresholdSe === 0.019 && d2.bandK === 2.19 && d2.shuffledK === 2.19
  && d2.shuffledThreshold === 0.717 && JSON.stringify(d2.seedsScored) === '{"real":20,"null":20,"shuffled":20}' && d2.seeds === 20 && d2.minWindows === FEW && d2.label === 'within the null band',
  `the drawer: a version-2 record's threshold, its error, k and the shuffled bound: ${JSON.stringify(d2)}`);
ok(admin.controlsDrawer('lotto', brokenQ2, report).q2.seedsScored === null && admin.controlsDrawer('lotto', brokenQ2, report).q2.bandK === null, 'a version-1 record has none of them');
const failedQ2 = { ...broken, lotto: { ...broken.lotto, q2: q2v2({ status: 'failed', real_max_of_suite_mean: null, real_max_of_suite_sd: null, null_threshold: null, null_threshold_se: null,
  band_k: null, shuffled_threshold: null, seeds_scored: { real: 20, null: 1, shuffled: 20 } }) } };
const failedView = admin.controlsOverview(failedQ2, report);
const fl = failedView.groups[0].games.find((g) => g.game === 'lotto').q2;
ok(fl.status === 'failed' && fl.real === null && fl.threshold === null && failedView.tiles.q2.failedGames.join() === 'lotto' && failedView.tiles.q2.above === 0
  && admin.weekReading(failedQ2, report).games.lotto.q2.status === 'failed', 'a failed Q2 run reads failed on the card, in the tile and in the reading');
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
// 11 Oct: lotto's A not followed yet (B confirmed already); 4 Oct: a reading written under the band rule (no q0.rule), as
// the stored readings of 18 and 25 Oct 2026 are - Q0's weeks start again at the first reading of the row tests
storedWeek('2026-10-11', (g) => { g.lotto.q0.rows = g.lotto.q0.rows.filter((r) => r.row !== 'A Model'); g.lotto.q2.status = 'within'; g.crypto.rows = null; });
storedWeek('2026-10-04', (g) => { Object.values(g).forEach((x) => { if (x.q0) { delete x.q0.rule; x.q0.above = []; } }); });
fs.writeFileSync(path.join(readingsDir, '2026-10-25.json'), '{"version": 1, "sunday": "2026-10-');
fs.writeFileSync(path.join(readingsDir, '2026-10-20.json'), JSON.stringify({ version: 1, sunday: '2026-10-20', games: {} }));
const stored = admin.loadReadings(readingsDir);
ok(stored.sundays.join() === '2026-10-04,2026-10-11' && stored.unreadable.join() === '2026-10-20.json,2026-10-25.json', 'the stored readings, and the files that are not one');

const ch = request(app, '/api/admin/controls', ADMIN).body;
const hg = Object.fromEntries(ch.groups.flatMap((g) => g.games).map((g) => [g.game, g]));
ok(JSON.stringify(hg.lotto.history.q0) === '{"state":"confirmed: B · followed: A","sunday":"2026-10-18","weeks":1,"changed":true,"last":"confirmed: B · followed: none","live":true}',
  `Q0: the rows confirmed and followed - A followed since last week is the change; no reading of 18 Oct yet, so the report on disk: ${JSON.stringify(hg.lotto.history.q0)}`);
ok(hg.lotto.q0.rows.every((r) => r.weeks === undefined), 'no row carries a "kth week" any more: one record re-read week after week is no streak');
ok(JSON.stringify(hg.lotto.history.q2) === '{"state":"above","sunday":"2026-10-18","weeks":1,"changed":true,"last":"within"}',
  'a change from last week carries what it read then - the only thing the page colours');
ok(hg.lotto.history.fc.state === 'lr fit · qk ignored' && hg.lotto.history.fc.weeks === 3 && hg.lotto.history.fc.changed === false && hg.lotto.history.fc.last === undefined,
  'a state that holds every week: its count of weeks, no change, nothing to colour');
ok(hg.crypto.history.rows.state === 'beat GARCH: none · GARCH vs uniform: better' && hg.crypto.history.rows.changed === true && hg.crypto.history.rows.last === null
  && hg.keno.history.q0.weeks === 2 && hg.keno.history.q0.state === 'confirmed: none · followed: K2' && hg.euromillions.history.q0 === null && hg.lotto.history.rows === undefined,
  'a record where last week had none is a change (last: null); Q0 counts back to the reading under the band rule and no further; a game without a ranking has no history; a lottery game no rows');
ok(admin.stateOf('q0', stored.bySunday.get('2026-10-04').games.keno) === 'band rule' && admin.stateOf('q0', { q0: { rule: 'row test', test: null, rows: [] } }) === 'no row test',
  'a stored reading without a Q0 rule reads "band rule" (the streak restarts on Sun 1 Nov 2026); a reading without a test says so');
const fromBand = admin.streakOf({ bySunday: new Map([['2026-10-11', JSON.parse(JSON.stringify(stored.bySunday.get('2026-10-04')))]]) }, 'keno', 'q0',
  admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report).games.keno);
ok(fromBand.weeks === 1 && fromBand.changed === null && fromBand.noRunBefore === true,
  `a band-rule reading dated by its band records of another week is no reading of Q0 that week: ${JSON.stringify(fromBand)}`);
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
// Q0's time is the report that ran the row tests (27 Oct 2026); a reading under the band rule is still dated by its older
// band record - NullControls rewrote a game's synthetic record hours before its shuffled one, so half rewritten, it was last week's
ok(midRun.games.lotto.q0.at === '2026-10-21T08:48:01Z' && midRun.games.lotto.q0.rule === 'row test', 'a reading\'s Q0 time is the report\'s');
const halfQ0 = JSON.parse(JSON.stringify(midRun));
delete halfQ0.games.lotto.q0.rule;
halfQ0.games.lotto.q0.modes.synthetic.at = '2026-10-25T10:11:55Z';
const hh = admin.historyOf(halfQ0, stored);
ok(hh.games.lotto.q0.sunday === '2026-10-18' && hh.games.lotto.q0.behind === true && hh.games.lotto.q0.state === 'band rule',
  `a band-rule reading half rewritten is behind, not this week's: ${JSON.stringify(hh.games.lotto.q0)}`);

// a stored week whose reading holds an older record of a control (it failed or skipped the game that Sunday, so last
// week's record was still on disk) is no reading of that control: it ends the run, it is not one more week
const staleStored = { ...stored, bySunday: new Map([...stored.bySunday].map(([day, r]) => [day, JSON.parse(JSON.stringify(r))])) };
const r11 = staleStored.bySunday.get('2026-10-11');
r11.games.lotto.q2 = { ...staleStored.bySunday.get('2026-10-04').games.lotto.q2 };
r11.games.lotto.fc = { ...staleStored.bySunday.get('2026-10-04').games.lotto.fc };
r11.games.lotto.q0 = JSON.parse(JSON.stringify(staleStored.bySunday.get('2026-10-04').games.lotto.q0));
const cur18 = admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report);
const sf = admin.streakOf(staleStored, 'lotto', 'fc', cur18.games.lotto);
ok(JSON.stringify(sf) === '{"state":"lr fit · qk ignored","sunday":"2026-10-18","weeks":1,"changed":null,"noRunBefore":true}',
  `a stale copy is not counted twice: weeks 1, nothing to compare, said: ${JSON.stringify(sf)}`);
ok(admin.streakOf(staleStored, 'lotto', 'q2', cur18.games.lotto).changed === null && admin.streakOf(staleStored, 'lotto', 'q0', cur18.games.lotto).weeks === 1,
  'the same for Q2 and Q0');
const staleView = admin.withHistory(admin.controlsOverview(admin.readAll(admin.resolveDirs({ dataDir })), report), cur18, staleStored);
ok(staleView.groups[0].games[0].history.q0.weeks === 1 && staleView.groups[0].games[0].history.q0.noRunBefore === true, 'a stale Q0 copy gets no extra week');

// Q2's rule (27 Oct 2026): a reading stores the verdict version its status was read under (q2.rule; none = 1). The week
// after a change of rule is another measurement - 3 seeds against 20 - so the run of weeks stops there, and the change
// is said (ruleChanged, with the rule and the seeds of the week before) whether the status changed with it or not
ok(stored.bySunday.get('2026-10-11').games.lotto.q2.rule === 1 && stored.bySunday.get('2026-10-11').games.lotto.q2.thresholdSe === null
  && cur18.games.lotto.q2.rule === 1 && cur18.games.lotto.q2.shuffledThreshold === null, 'a reading of a version-1 record holds rule 1 and none of the version-2 numbers');
const v2Now = (status) => ({ ...cur18.games.lotto, q2: { ...cur18.games.lotto.q2, status, rule: 2, seeds: 20, thresholdSe: 0.019 } });
const ruled = admin.streakOf(stored, 'lotto', 'q2', v2Now('above'));
ok(JSON.stringify(ruled) === '{"state":"above","sunday":"2026-10-18","weeks":1,"changed":true,"last":"within","ruleChanged":true,"lastRule":1,"lastSeeds":3}',
  `the first week under rule 2: what changed, and that the week before was read under rule 1 on 3 seeds: ${JSON.stringify(ruled)}`);
const steady = admin.streakOf(stored, 'lotto', 'q2', v2Now('within'));
ok(steady.weeks === 1 && steady.changed === false && steady.ruleChanged === true
  && admin.streakOf(stored, 'lotto', 'q2', { ...cur18.games.lotto, q2: { ...cur18.games.lotto.q2, status: 'within' } }).weeks === 2,
  'the same status under a new rule starts a new run (weeks 1, unchanged); under the same rule it counts on (2nd week)');
const unruled = { ...stored, bySunday: new Map([...stored.bySunday].map(([day, r]) => { const x = JSON.parse(JSON.stringify(r)); delete x.games.lotto.q2.rule; return [day, x]; })) };
ok(admin.streakOf(unruled, 'lotto', 'q2', cur18.games.lotto).ruleChanged === undefined && admin.streakOf(unruled, 'lotto', 'q2', v2Now('above')).ruleChanged === true,
  'a reading written before 27 Oct 2026 has no rule: it was read under rule 1');
ok(admin.streakOf(stored, 'lotto', 'fc', cur18.games.lotto).ruleChanged === undefined, 'only Q2 has a rule');

// Q0 once the reading of its own Sunday is stored: the state is that reading's (that Sunday's report), never today's;
// where today's report follows or confirms other rows is said apart (sinceReading)
const r18 = admin.weekReading(admin.readAll(admin.resolveDirs({ dataDir })), report, { sunday: '2026-10-18' });
r18.games.lotto.q0.rows = r18.games.lotto.q0.rows.filter((r) => r.row !== 'A Model');
r18.rankingAt = '2026-10-18T08:47:31Z';
const withOwn = { ...stored, bySunday: new Map([...stored.bySunday, ['2026-10-18', r18]]) };
const own = admin.streakOf(withOwn, 'lotto', 'q0', cur18.games.lotto);
ok(JSON.stringify(own) === '{"state":"confirmed: B · followed: none","sunday":"2026-10-18","weeks":2,"changed":false,"sinceReading":{"now":"confirmed: B · followed: A","added":["A Model"],"dropped":[],"rankingAt":"2026-10-18T08:47:31Z"}}',
  `Q0 reads its own Sunday's reading, the report's newly followed row said apart: ${JSON.stringify(own)}`);
r18.games.lotto.q0.rows = [];
const flipped = admin.streakOf(withOwn, 'lotto', 'q0', cur18.games.lotto);
ok(flipped.state === 'confirmed: none · followed: none' && flipped.changed === true && flipped.last === 'confirmed: B · followed: none'
  && flipped.sinceReading.now === 'confirmed: B · followed: A' && flipped.sinceReading.added.join() === 'B Model,A Model',
  'a later report never rewrites a stored reading: the change is the readings\' own');
const ownView = admin.withHistory(admin.controlsOverview(admin.readAll(admin.resolveDirs({ dataDir })), report), cur18, withOwn);
ok(ownView.groups[0].games[0].history.q0.sinceReading.added.join() === 'B Model,A Model', 'the page carries the same history');

// the band (context) names each control record's time
const lives = admin.q0Of('crypto', { synthetic: { best_row_mean: 1, best_row_sd_across_seeds: 0.1, generated_at: '2026-10-18T10:00:00', rows_compared: 5, seeds: [1, 2, 3], days: 100 } },
  { metric: 'avg_hits', minDrawsForRanking: 10, models: [{ name: 'A Model', avg_hits: 2, draws: 50, live_draws: 20 }] });
ok(lives.band.modes.length === 1 && lives.band.modes[0].at === '2026-10-18T10:00:00Z' && lives.test === null && lives.error === 'not in this report - written before 27 Oct 2026, or its test failed',
  'Q0: the band names each control record\'s time; a report without row tests has no test, and says so');

// the runs: three jobs, the scheduler's history and their logs
const TEST_JOBS = [
  { key: 'predictor', name: 'Daily predictor', log: 'predictor.log', daily: { hour: 9, minute: 0 } },
  { key: 'hyperoptStatistics', name: 'Weekly tuning: statistical models', log: 'hyperoptStatistics.log', plan: 'weeklyTuning' },
  { key: 'nullControls', name: 'Weekly controls: null histories (Q0 band, context)', log: 'nullControls.log', plan: 'weeklyControls' },
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
writeLog('nullControls.log', [slice('Weekly controls: null histories (Q0 band, context)', '2026-10-18 10:00:01', ['/src/MarketModels.py:448: RuntimeWarning: invalid value'], 1, 45600)]);
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
  // the keno sub-selection step, its own job since 26 Oct 2026: its KenoSubsetTuning record says HyperoptStatistics like the
  // statistical job's and commits as "statistical", so the row tells them apart - the keno step's from the day it joined
  // the chain, the statistical job's before (that job tuned the row until then)
  const ms31 = Z('2026-10-31T21:40:00Z');
  const k31 = { ...kenoGate, date: '2026-10-31T21:40:00+00:00', tuner: 'HyperoptStatistics' };
  const k24 = { ...kenoGate, date: '2026-10-24T12:17:00+00:00', tuner: 'HyperoptStatistics' };
  const m31 = { ...kenoGate, date: '2026-10-31T10:16:00+00:00', tuner: 'HyperoptStatistics' };
  ok(gates.familyOf('KenoSubsetTuning', ms31) === 'keno sub-selections' && gates.familyOf('KenoSubsetTuning', Z('2026-10-24T12:17:00Z')) === 'statistical'
    && gates.tunerOfScript('HyperoptStatistics', 'KenoSubsetTuning', ms31) === 'keno sub-selections' && gates.tunerOfScript('HyperoptStatistics', 'Markov Model', ms31) === 'statistical'
    && gates.tunerOfScript('/x/HyperoptStatistics.py') === 'statistical' && gates.tunerOfJob('kenoSubsetTuning') === 'keno sub-selections',
    'the keno step by its row from 26 Oct 2026 on, the statistical job by its script; the keno job is the keno step');
  ok(gates.attribute('KenoSubsetTuning', k31).tuner === 'keno sub-selections' && gates.attribute('KenoSubsetTuning', k24).tuner === 'statistical'
    && gates.attribute('KenoSubsetTuning', { ...k31, tuner: undefined }, { carrier: { tuner: 'statistical' } }).by === 'family + commit',
    'a KenoSubsetTuning record: the keno step\'s since it runs on its own, the statistical job\'s before; a "statistical" commit confirms either');
  // the cut-over day is written twice (gates.js TUNERS, jobs.js JOBS): every tuner's job is a scheduled job, and a tuner
  // that joined later says the same day as its job, or the Tuning page and This week would disagree about a weekend
  const jobsByKey = Object.fromEntries(require('../jobs').JOBS.map((j) => [j.key, j]));
  ok(gates.TUNERS.every((x) => jobsByKey[x.job] && (x.since || null) === (jobsByKey[x.job].since || null)),
    'every tuner of gates.js is a job of jobs.js, with the same since');
  const kh31 ={ weekends: ['2026-10-31'], newest: '2026-10-31', decisions: {}, carriers: {}, commits: 0, error: null };
  const runs31 = [{ key: 'hyperoptStatistics', at: Z('2026-10-31T09:00:00Z'), finishedAt: Z('2026-10-31T14:00:00Z'), exit: 0, seconds: 18000 },
    { key: 'kenoSubsetTuning', at: Z('2026-10-31T21:30:00Z'), finishedAt: Z('2026-10-31T21:45:00Z'), exit: 0, seconds: 900 }];
  const s31 = admin.tuningSummary(admin.tuningRecords({ keno: { 'Markov Model': m31, KenoSubsetTuning: k31 } }, kh31, runs31), kh31, runs31);
  const kenoStep = s31.byTuner.find((x) => x.tuner === 'keno sub-selections');
  const stat31 = s31.byTuner.find((x) => x.tuner === 'statistical');
  ok(kenoStep && kenoStep.job === 'kenoSubsetTuning' && kenoStep.n === 1 && kenoStep.replacedRows.keno.join() === 'KenoSubsetTuning' && kenoStep.run.minutes === 15
    && stat31.n === 1 && stat31.replacedRows.keno.join() === 'Markov Model' && stat31.run.minutes === 300 && s31.byTuner.map((x) => x.tuner).pop() === 'keno sub-selections',
    `each run's drawer counts its own decisions: the keno step its KenoSubsetTuning record, the statistical job the rest (${s31.byTuner.map((x) => `${x.tuner} ${x.n}`).join(', ')})`);
  const kh24 = { weekends: ['2026-10-24'], newest: '2026-10-24', decisions: {}, carriers: {}, commits: 0, error: null };
  const s24 = admin.tuningSummary(admin.tuningRecords({ keno: { KenoSubsetTuning: k24 } }, kh24, []), kh24);
  ok(!s24.byTuner.some((x) => x.tuner === 'keno sub-selections') && s24.byTuner.find((x) => x.tuner === 'statistical').n === 1,
    'the weekend before the keno step existed: no card of its own, its row the statistical job\'s decision');
  // the lockbox days a decision did not score (26 Oct 2026): out of the window for a row tuner, out of the quantum
  // tuner's score table before its holdout was cut; an older record has no count
  const lockRow = gates.describe('Markov Model', { ...markov17, lockbox_days_withheld: 22 });
  const lockQ = gates.describe('QuantumVQC Model', { ...markov17, lockbox_days_withheld: 60, lockbox_withheld_from: 'table' });
  const lockGame = admin.tuningGame(admin.tuningRecords({ keno: { KenoSubsetTuning: { ...k31, lockbox_days_withheld: 22 } } }, kh31, runs31), 'keno', kh31).rows[0];
  ok(lockRow.lockboxDays === 22 && lockRow.lockboxFrom === 'window' && lockQ.lockboxDays === 60 && lockQ.lockboxFrom === 'table'
    && gates.describe('Mystery Model', mystery).lockboxDays === null && gates.describe('Mystery Model', mystery).lockboxFrom === null
    && lockGame.lockboxDays === 22 && lockGame.lockboxFrom === 'window' && m.lockboxDays === null && m.lockboxFrom === null,
    'the lockbox days a decision withheld and out of what, on the drawer\'s row; an older record has neither');

  // the margins of 27 Oct 2026: every tuner asks for k standard errors (margin_se: 1, quantum 2.5), a revert takes its
  // own margin (revert_margin: 0 for statistical, RL and ensemble), a challenger without an SE is kept (fail closed),
  // and the RL Position row says whether its sizing sits out. An older record has none of the keys.
  const at31 = (h) => `2026-10-31T${h}:00+00:00`;
  const q31 = { ...gate(at31('23:10'), 'kept', 0.9, 0.75, 0.7, { margin: 0.2, tuner: 'HyperoptQuantum' }), margin_se: 2.5, revert_margin: 0.2 };
  const vqc31 = { ...gate(at31('23:20'), 'replaced', 1.1, 0.75, 0.7, { margin: 0.2, tuner: 'HyperoptQuantum' }), margin_se: 2.5, revert_margin: 0.2 };
  const rev31 = { ...gate(at31('10:00'), 'reverted', 0.7, 0.6, 0.65, { margin: 0.08, tuner: 'HyperoptStatistics' }), margin_se: 1, revert_margin: 0 };
  const noSe31 = { ...gate(at31('10:05'), 'kept', 0.9, 0.6, 0.5, { margin: 0, se: null, tuner: 'HyperoptStatistics' }), margin_se: 1, revert_margin: 0,
    reason: 'no standard error - kept: challenger +0.9000 has none, so its margin of 1 standard error cannot be measured' };
  const pos31 = { ...gate(at31('22:00'), 'kept', 0.0084, 0.0083, -0.17, { margin: 0.08, kind: 'capped_profit', days: 40, tuner: 'HyperoptRLTicket' }),
    margin_se: 1, revert_margin: 0, sit_out: true, sit_out_reason: 'served sizing +0.0083 (SE 0.0800) does not beat the 0 of never betting by one standard error of its own - the row sits out' };
  const dq = gates.describe('QuantumMetaLearner Model', q31);
  const dr = gates.describe('Markov Model', rev31);
  const dn = gates.describe('MarkovBayesian Model', noSe31);
  const dp = gates.describe('RL Position Model', pos31);
  const dOld = gates.describe('Mystery Model', mystery);
  ok(Math.abs(dq.marginSe - 2.5) < 1e-9 && dq.marginK === 2.5 && dq.revertMargin === 0.2 && dr.marginK === 1 && Math.abs(dr.marginSe - 1) < 1e-9 && dr.revertMargin === 0
    && dn.marginSe === null && dn.marginK === 1 && dp.sitOut === true && /sits out$/.test(dp.sitOutReason) && dq.sitOut === null && dq.sitOutReason === null
    && gates.describe('RL Position Model', { ...pos31, sit_out: false }).sitOut === false,
    `a record's k, its revert margin and the RL Position sit-out: ${JSON.stringify({ dq, dn, dp })}`);
  ok(dOld.marginK === null && dOld.revertMargin === null && dOld.sitOut === null && dOld.sitOutReason === null && dOld.marginSe === 0,
    'an older record without the new keys: no k, no revert margin, no sit-out - and its marginSe as before');
  const h31 = { weekends: ['2026-10-31'], newest: '2026-10-31', decisions: {}, carriers: {}, commits: 0, error: null };
  const recs31 = admin.tuningRecords({ lotto: { 'QuantumMetaLearner Model': q31, 'QuantumVQC Model': vqc31, 'Markov Model': rev31, 'MarkovBayesian Model': noSe31 },
    crypto: { 'RL Position Model': pos31 } }, h31, []);
  const sum31 = Object.fromEntries(admin.tuningSummary(recs31, h31).byTuner.map((x) => [x.tuner, x]));
  ok(JSON.stringify(sum31.quantum.marginSeCounts) === '{"2.5":2}' && sum31.quantum.marginOneSe === 0 && sum31.quantum.marginZero === 0
    && JSON.stringify(sum31.statistical.marginSeCounts) === '{"1":2}' && sum31.statistical.marginOneSe === 1 && sum31.statistical.marginZero === 0
    && JSON.stringify(sum31['RL ticket'].marginSeCounts) === '{"1":1}' && JSON.stringify(sum31.boosting.marginSeCounts) === '{}',
    `the margin chips by k: quantum at 2.5 SE, a row kept for want of a challenger SE at its k, never at margin 0: ${JSON.stringify(Object.values(sum31).map((x) => [x.tuner, x.marginSeCounts, x.marginZero]))}`);
  const game31 = Object.fromEntries(admin.tuningGame(recs31, 'lotto', h31).rows.map((r) => [r.row, r]));
  const pos31row = admin.tuningGame(recs31, 'crypto', h31).rows[0];
  ok(game31['QuantumMetaLearner Model'].marginSe === 2.5 && game31['QuantumMetaLearner Model'].marginK === 2.5 && game31['QuantumMetaLearner Model'].revertMargin === 0.2
    && game31['Markov Model'].revertMargin === 0 && game31['MarkovBayesian Model'].marginK === 1 && game31['MarkovBayesian Model'].marginSe === null
    && pos31row.sitOut === true && /never betting/.test(pos31row.sitOutReason) && game31['Markov Model'].sitOut === null
    && m.marginK === null && m.revertMargin === null && m.sitOut === null && m.sitOutReason === null,
    'one game: each row\'s k, revert margin and sit-out for the drawer; an older record\'s all null');
  ok(JSON.stringify(tun.statistical.marginSeCounts) === '{}' && tun.statistical.marginZero === 2 && JSON.stringify(tun.quantum.marginSeCounts) === '{"1":1}',
    'older records: margin 0 counted as before, a margin of one challenger SE as 1');

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
  const view = (nowIso, extra = {}, jobs = TEST_JOBS) => {
    const inputs = admin.readAll(admin.resolveDirs({ dataDir }));
    const dirs = admin.resolveDirs({ dataDir, root: tmp });
    return admin.weekView({ jobs, sched: admin.readScheduler(dirs), slices: Object.fromEntries(jobs.map((j) => [j.key, admin.readSlices(dirs, j)])),
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
  ok(w.tiles.tuners.replaced === 3 && w.tiles.tuners.replacedUnder1Se === 1 && w.tiles.tuners.byTuner.length === t.byTuner.length && w.tiles.q0.confirmed === ch.tiles.q0.confirmed
    && w.tiles.q2.above === ch.tiles.q2.above, 'week: the tuners\' and controls\' tiles are the Tuning and Controls pages\' own');
  ok(w.tiles.tuners.byTuner.every((x, i) => x.job === t.byTuner[i].job && x.games === t.byTuner[i].games && x.marginOneSe === t.byTuner[i].marginOneSe
      && JSON.stringify(x.marginSeCounts) === JSON.stringify(t.byTuner[i].marginSeCounts))
    && w.rankingAt === ch.rankingAt && typeof w.rankingAt === 'string',
    'week: each tuner\'s job, games and 1-SE margins as on the Tuning page; the ranking time the Q0 tile reads against');
  const lm = w.matrix.games.find((g) => g.game === 'lotto');
  ok(w.matrix.games.length === 11 && lm.q0.test.confirmed === 1 && lm.q0.test.under === 3 && lm.q0.confirmedRows.join() === 'B Model' && lm.q0.followedRows.join() === 'A Model'
    && lm.q0.history.changed === true && lm.q0.at === w.rankingAt && lm.q0.band.at === '2026-10-18T10:11:55Z' && lm.q2.history.changed === true && lm.gates.replaced === 3
    && lm.gates.notThisWeekend === 1 && w.readings.stored === 2, 'week: the controls matrix - Q0 the row test\'s cell at the report\'s time, its band times apart; colour only where changed');
  const em = w.matrix.games.find((g) => g.game === 'euromillions').q0;
  ok(em.test === null && em.error === 'no ranking for this game yet' && em.band === null && em.at === null, 'week: a game without a row test says why, never a count of none');
  // before Saturday's first run the page stays on last weekend; once a job of it runs, on this one
  ok(view('2026-10-24T05:00:00Z').weekend.saturday === '2026-10-17', 'week: on Saturday morning, still last weekend');
  const sat24 = view('2026-10-24T08:00:00Z', { current: { key: 'predictor', name: 'Daily predictor', startedAt: Z('2026-10-24T07:00:00Z') } });
  ok(sat24.weekend.saturday === '2026-10-24' && sat24.tiles.weekend.runs === 0 && sat24.tiles.weekend.running.key === 'predictor'
    && sat24.tiles.weekend.missing.map((e) => `${e.key}${e.day ? `@${e.day}` : ''}`).join() === 'predictor@2026-10-25,hyperoptStatistics,nullControls',
    'week: once Saturday\'s predictor runs, the new weekend - the running job is not missing, Sunday\'s still is');
  // a job that joined its plan after a weekend (jobs.js since: the keno step on 26 Oct 2026) is no run missing from that
  // weekend - only from the weekends whose Saturday is that day or later
  const late = [...TEST_JOBS, { key: 'kenoSubsetTuning', name: 'Weekly tuning: keno sub-selections', log: 'kenoSubsetTuning.log', plan: 'weeklyTuning', since: '2026-10-19' }];
  const before = view('2026-10-21T18:00:00Z', {}, late).tiles.weekend;
  const after = view('2026-10-24T08:00:00Z', { current: { key: 'predictor', name: 'Daily predictor', startedAt: Z('2026-10-24T07:00:00Z') } }, late).tiles.weekend;
  ok(before.expected === 4 && before.missing.length === 0 && after.expected === 5
    && after.missing.map((e) => `${e.key}${e.day ? `@${e.day}` : ''}`).join() === 'predictor@2026-10-25,hyperoptStatistics,nullControls,kenoSubsetTuning',
    'week: a job that joined its plan on Mon 19 Oct is not expected of the weekend of 17 Oct, and is of the next one');
  // a since inside a weekend (its Sunday) is after that weekend's Saturday: not expected there; on the Saturday itself it is
  const joined = (since) => view('2026-10-24T08:00:00Z', { current: { key: 'predictor', name: 'Daily predictor', startedAt: Z('2026-10-24T07:00:00Z') } },
    [...TEST_JOBS, { ...late[late.length - 1], since }]).tiles.weekend;
  ok(joined('2026-10-25').expected === 4 && !joined('2026-10-25').missing.some((e) => e.key === 'kenoSubsetTuning')
    && joined('2026-10-24').expected === 5 && joined('2026-10-24').missing.some((e) => e.key === 'kenoSubsetTuning'),
    'week: a job that joined on the Sunday of a weekend is not expected of it, one that joined on its Saturday is');
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
  ok(w5.matrix.fewWindows === FEW && w5.matrix.games.find((g) => g.game === 'lotto').q0.band.modes.map((x) => x.mode).sort().join() === 'shuffled,synthetic',
    'week route: Q2\'s power rule (controlRules.json), and each Q0 band record\'s time for what a NullControls run wrote');
  const cm = w5.matrix.games.find((g) => g.game === 'crypto').q2;
  ok(cm.status === 'too few draws' && cm.fewWindows === true && cm.minWindows === FEW && w5.tiles.q2.aboveFewWindows === undefined,
    'week route: a Q2 cell on too few test windows reads too few draws, with the windows it was judged by');
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
    const kenoSince = jobs.JOBS.find((j) => j.key === 'kenoSubsetTuning').since;
    ok(w.weekend.saturday >= kenoSince || !w.tiles.weekend.missing.some((e) => e.key === 'kenoSubsetTuning'),
      `production: the keno step is no run missing from a weekend before it joined the chain (weekend of ${w.weekend.saturday})`);
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
  const prodHeadline = games.filter((g) => g.q0.test && g.q0.inTile);
  ok(pc.tiles.q0.confirmed === prodHeadline.reduce((s, g) => s + g.q0.confirmed.length, 0) && pc.tiles.q0.under === prodHeadline.reduce((s, g) => s + g.q0.test.under, 0)
    && pc.tiles.q0.noTest.length + prodHeadline.length + pc.tiles.q0.payout.games.length === games.length
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
