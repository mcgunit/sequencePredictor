'use strict';
// Admin › Parameters (params.js): which card every bestParams key lands on, the view's shape on stand-in files, and -
// read-only - every key of the checkout's own bestParams_<game>.json files accounted for exactly once (a card, the
// shared card, the tuners' records, "stored but not read" or "not assigned"), with nothing left unassigned.
//   node test/params.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const params = require('../params');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const rowOf = (key) => { const f = params.familyOf(key); return f ? f.rows[0] : null; };

// --- the longest prefix wins, exact keys and switches first ---------------------------------------------------------
ok(rowOf('markovAlpha') === 'Markov Model', 'markovAlpha: Markov');
ok(rowOf('markovTransitionMode') === 'Markov Model', 'markovTransitionMode: Markov (HyperoptStatistics tunes it with the Markov row)');
ok(rowOf('markovMcAlpha') === 'MarkovMonteCarlo Model', 'markovMc*: MarkovMonteCarlo, not Markov');
ok(rowOf('markovBayesianAlpha') === 'MarkovBayesian Model', 'markovBayesian*');
ok(rowOf('markovBayesianEnhancedAlpha') === 'MarkovBayesianEnhanched Model', 'markovBayesianEnhanced*: the Enhanced row');
ok(rowOf('usevMarkovBayesianEnhanced') === 'MarkovBayesianEnhanched Model', 'the Enhanced row\'s switch');
ok(rowOf('xgBoostEstimators') === 'XGBoost Model' && rowOf('xgBoostMlEstimators') === 'XGBoostMultiLabel Model', 'xgBoost vs xgBoostMl');
ok(rowOf('useBoost') === 'XGBoost Model', 'useBoost is the XGBoost row\'s switch');
ok(rowOf('catBoostMlBorderCount') === 'CatBoostMultiLabel Model' && rowOf('memoriseCatBoost') === 'CatBoost Model', 'CatBoost keys');
ok(rowOf('metaLearnerSubsetMode') === 'MetaLearner Model' && rowOf('metaLearnerV2SubsetMode') === 'MetaLearnerV2 Model', 'metaLearner vs V2');
ok(rowOf('quantumKernel_C') === 'QuantumMetaLearner Model' && rowOf('quantumVqc_epochs') === 'QuantumVQC Model', 'quantum keys');
ok(rowOf('batchSize') === 'LSTM Base Model' && rowOf('labelSmoothing') === 'LSTM Base Model', 'bare DL keys: the LSTM card');
ok(rowOf('transformer_batchSize') === 'Transformer Model' && rowOf('unifiedLstmGruTcn_gruUnits') === 'UnifiedLstmGruTcn Model', 'prefixed DL keys');
ok(rowOf('unifiedLstmTcn_tcnUnits') === 'UnifiedLstmTcn Model', 'unifiedLstmTcn_ is not unifiedLstmGruTcn_');
ok(rowOf('regimeHmmFeatures') === 'Regime HMM Model' && rowOf('garchDist') === 'GARCH Model', 'market rows');
ok(rowOf('rlTicketEpochs') === 'RL Ticket Model' && rowOf('rlPositionSitOut') === 'RL Position Model', 'the two RL rows');
ok(rowOf('use_7') === 'KenoSubsetTuning', 'keno sub-selection switches');
ok(rowOf('somethingNew') === null, 'an unknown key has no family');

// --- the view on a stand-in file ---------------------------------------------------------------------------------
const stand = {
  yearsOfHistory: 10, markovAlpha: 0.5, markovMcAlpha: 0.2, useMarkov: false, transformer_windowSize: 30, transformer_yearsOfHistory: 10,
  lstmMarkovAlpha: 0.05, somethingNew: 1, garchDist: 't', sampleTemperatures: { 'Markov Model': 0.5 },
  modelScores: { 'Markov Model': 0.84 },
  tuningGate: { 'Markov Model': { date: '2026-11-07T10:36:04+00:00', tuner: 'HyperoptStatistics', decision: 'kept', reason: 'r', window_days: 90, run_trials: 15, study_trials: 150 } },
};
const v = params.parametersView('lotto', stand, null, new Date('2026-11-08T12:00:00Z'));
const card = (id) => v.groups.flatMap((g) => g.cards).find((c) => c.id === id);
const markov = card('markov-model');
ok(markov && markov.params.length === 1 && markov.params[0].key === 'markovAlpha', 'Markov card: its own key only');
ok(markov.switches[0].key === 'useMarkov' && markov.switches[0].value === false && markov.switches[0].set, 'Markov switch: off, set');
ok(markov.gates.length === 1 && markov.gates[0].tuner === 'HyperoptStatistics' && markov.gates[0].windowDays === 90, 'Markov gate trimmed');
ok(markov.scores[0].score === 0.84, 'Markov score from modelScores');
ok(markov.variants.memorise.state === 'on' && markov.variants.sampled[0].temperature === 0.5 && markov.variants.sampled[0].set, 'Markov variants: memorise on, sampled at 0.5 (set)');
ok(card('markovmontecarlo-model').variants.sampled[0].temperature === 1 && !card('markovmontecarlo-model').variants.sampled[0].set, 'an unlisted row samples at 1, not set');
const tr = card('transformer-model');
ok(tr.params.find((p) => p.key === 'transformer_yearsOfHistory').read === true && /shared yearsOfHistory/.test(tr.params.find((p) => p.key === 'transformer_yearsOfHistory').note), 'per-model yearsOfHistory: read, falls back to the shared one');
ok(tr.params.find((p) => p.key === 'transformer_windowSize').read === true, 'transformer_windowSize is read');
ok(v.unread.length === 1 && v.unread[0].key === 'lstmMarkovAlpha', 'old LSTM key: stored but not read');
ok(v.unassigned.some((u) => u.key === 'somethingNew' && u.note === null), 'an unknown key is shown, not hidden');
ok(v.unassigned.some((u) => u.key === 'garchDist' && /does not run/.test(u.note)), 'a market key in a lottery file: shown with why');
ok(v.shared.find((s) => s.key === 'yearsOfHistory').value === 10 && v.shared.find((s) => s.key === 'memoriseWindowDraws').set === false, 'shared card');
ok(!v.groups.some((g) => g.label === 'Market rows'), 'no market cards on a lottery game');
ok(v.empty.includes('TCN Base Model'), 'a model with nothing stored is named');

const m = params.parametersView('crypto', { garchDist: 't', markovAlpha: 0.3 });
ok(m.groups.some((g) => g.label === 'Market rows') && !m.variants.sampled && !m.variants.memorise, 'market: market rows, no variants');
ok(m.groups.flatMap((g) => g.cards).every((c) => c.variants.memorise.state === 'none' && !c.variants.sampled.length), 'market cards: no variant twins');
const p3 = params.parametersView('pick3', { markovAlpha: 0.3 });
ok(!p3.variants.sampled && p3.variants.memorise, 'pick3: memorise yes, sampled no (positional)');
const off = params.parametersView('lotto', { useMemoriseRows: false, markovAlpha: 1, catBoostEstimators: 9 });
ok(off.groups.flatMap((g) => g.cards).every((c) => c.variants.memorise.state !== 'on'), 'useMemoriseRows false: no memorise twin anywhere');
const cb = params.parametersView('lotto', { catBoostEstimators: 9 });
ok(cb.groups.flatMap((g) => g.cards).find((c) => c.id === 'catboost-model').variants.memorise.state === 'off', 'CatBoost memorise off unless memoriseCatBoost');
const none = params.parametersView('shares', null);
ok(!none.file.exists && none.groups.length === 0, 'no file: no cards, says so');

// --- readParams on a temp dir; a missing file is null, not an error ------------------------------------------------
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'params-test-'));
fs.writeFileSync(path.join(tmp, 'bestParams_lotto.json'), JSON.stringify({ markovAlpha: 1 }));
ok(params.readParams(tmp, 'lotto').params.markovAlpha === 1 && params.readParams(tmp, 'lotto').stat, 'readParams reads the file');
ok(params.readParams(tmp, 'keno').params === null, 'readParams: no file is null');
fs.rmSync(tmp, { recursive: true, force: true });

// --- the route: unknown game 404, a known one the view ------------------------------------------------------------
const routes = {};
params.install({ get: (p, h) => { routes[p] = h; } }, { root: path.join(__dirname, '..'), answer: (res, build) => res.send(build()) });
const call = (game) => { let out; routes['/api/admin/parameters/:game']({ params: { game } }, { send: (b) => { out = b; } }); return out; };
ok(call('nope').status === 404, 'unknown game: 404');

// --- every key of the checkout's files accounted for exactly once (read-only) -------------------------------------
const root = path.join(__dirname, '..');
params.GAMES.forEach((game) => {
  const { params: p } = params.readParams(root, game);
  if (!p) return;
  const view = params.parametersView(game, p);
  const counted = view.groups.flatMap((g) => g.cards).reduce((n, c) => n + c.params.length + c.switches.filter((s) => s.set).length, 0)
    + view.shared.filter((s) => s.set).length + view.unread.length + view.unassigned.length + params.RECORDS.filter((k) => k in p).length;
  ok(counted === Object.keys(p).length, `${game}: ${counted} of ${Object.keys(p).length} keys accounted for`);
  ok(view.unassigned.length === 0, `${game}: nothing unassigned (${view.unassigned.map((u) => u.key).join(', ')})`);
  ok(call(game).game === game, `${game}: the route answers its view`);
});

// --- manual overrides on a temp root: set, change, release, the refusals, the fresh-retrain requests ------------------
{
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'params-ov-'));
  fs.writeFileSync(path.join(root, 'bestParams_lotto.json'), JSON.stringify({
    yearsOfHistory: 10, markovAlpha: 0.65, markovOrder: 2, markovBlendMode: 'log', useMarkov: true, batchSize: 4, labelSmoothing: 0.08,
    gnn_windowSize: 14, gnn_yearsOfHistory: 10, subsetEnsembleModels: ['A'], sampleTemperatures: { 'Markov Model': 1.5 },
  }));
  const now = new Date('2026-11-08T21:00:00Z');
  const file = () => JSON.parse(fs.readFileSync(path.join(root, params.OVERRIDES_FILE), 'utf-8'));
  const refused = (r, re, msg) => ok(!r.ok && re.test(r.message), `${msg}: ${r.message}`);

  let r = params.setOverride(root, 'lotto', 'markovAlpha', 0.5, ' try lower ', 'alice', now);
  ok(r.ok && /0\.5/.test(r.message) && /next lottery run/.test(r.message) && !/new weights/.test(r.message), `statistical override: no retrain (${r.message})`);
  let o = file().games.lotto;
  ok(o.overrides.markovAlpha.value === 0.5 && o.overrides.markovAlpha.tuned === 0.65 && o.overrides.markovAlpha.by === 'alice'
    && o.overrides.markovAlpha.note === 'try lower' && o.overrides.markovAlpha.rows[0] === 'Markov Model' && !o.fresh, 'the override as stored');
  ok(JSON.parse(fs.readFileSync(path.join(root, 'bestParams_lotto.json'), 'utf-8')).markovAlpha === 0.65, 'the tuners\' file is never written');

  r = params.setOverride(root, 'lotto', 'batchSize', 8, '', 'alice', now);
  ok(r.ok && /LSTM Base, LSTM Smoothed train from new weights/.test(r.message) && /takes longer/.test(r.message), `DL override: both LSTM rows retrain (${r.message})`);
  o = file().games.lotto;
  ok(o.fresh['LSTM Base Model'] === now.toISOString() && o.fresh['LSTM Smoothed Model'] === now.toISOString() && !o.overrides.batchSize.note, 'fresh requests for both LSTM rows');
  r = params.setOverride(root, 'lotto', 'labelSmoothing', 0.0, null, 'alice', now);
  ok(r.ok && /LSTM Smoothed trains/.test(r.message) && file().games.lotto.overrides.labelSmoothing.rows.join() === 'LSTM Smoothed Model', 'labelSmoothing: the Smoothed row only');
  r = params.setOverride(root, 'lotto', 'yearsOfHistory', 5, null, 'alice', now);
  ok(r.ok && /LSTM Base/.test(r.message) && !/GNN/.test(r.message), `shared yearsOfHistory: the rows without their own key retrain, not GNN (${r.message})`);
  ok(params.setOverride(root, 'lotto', 'useMarkov', false, null, 'bob', now).ok, 'a switch can be set');
  ok(params.setOverride(root, 'lotto', 'useMarkovMonteCarlo', false, null, 'bob', now).ok, 'a switch the file does not have can be set too');
  ok(params.setOverride(root, 'lotto', 'sampleTemperatures/Markov Model', 0.7, null, 'bob', now).ok, 'a sample temperature');
  ok(params.setOverride(root, 'lotto', 'markovBlendMode', 'linear', null, 'bob', now).ok, 'a word setting');

  refused(params.setOverride(root, 'lotto', 'markovOrder', 1.5, null, 'x', now), /whole number/, 'an int stays an int');
  refused(params.setOverride(root, 'lotto', 'markovAlpha', -1, null, 'x', now), /at least 0/, 'no negative number');
  refused(params.setOverride(root, 'lotto', 'markovAlpha', '0.5', null, 'x', now), /is a number/, 'a number, not a string');
  refused(params.setOverride(root, 'lotto', 'useMarkov', 'false', null, 'x', now), /true or false/, 'a switch is boolean');
  refused(params.setOverride(root, 'lotto', 'sampleTemperatures/Markov Model', 0, null, 'x', now), /above 0/, 'temperature above 0');
  refused(params.setOverride(root, 'lotto', 'subsetEnsembleModels', ['B'], null, 'x', now), /list or an object/, 'lists are not editable');
  refused(params.setOverride(root, 'lotto', 'markovRecencyWeight', 1, null, 'x', now), /not in bestParams_lotto/, 'only keys in the file');
  refused(params.setOverride(root, 'lotto', 'garchDist', 't', null, 'x', now), /not a setting of a model lotto runs/, 'a market key on a lottery game');
  refused(params.setOverride(root, 'lotto', 'tuningGate', 1, null, 'x', now), /cannot be overridden/, 'the tuners\' records');
  refused(params.setOverride(root, 'lotto', 'lstmMarkovAlpha', 1, null, 'x', now), /cannot be overridden/, 'an unread key');
  refused(params.setOverride(root, 'pick3', 'sampleTemperatures/Markov Model', 1, null, 'x', now), /no sampled rows/, 'no temperature on pick3');
  ok(params.setOverride(root, 'nope', 'markovAlpha', 1, null, 'x', now).status === 404, 'unknown game: 404');

  // the view: tuned values shown, manual ones on top, the served merge and the pending retrains
  const view = params.gameView(root, 'lotto');
  const cards = view.groups.flatMap((g) => g.cards);
  const markov = cards.find((c) => c.id === 'markov-model');
  const alpha = markov.params.find((p) => p.key === 'markovAlpha');
  ok(alpha.value === 0.65 && alpha.override.value === 0.5 && alpha.override.by === 'alice' && alpha.editable && alpha.type === 'float', 'view: tuned value with the override');
  ok(markov.switches[0].override.value === false && markov.manual === 4, `view: switch override, 4 manual on the Markov card (${markov.manual})`);
  ok(markov.variants.sampled[0].temperature === 1.5 && markov.variants.sampled[0].override.value === 0.7, 'view: temperature tuned 1.5, manual 0.7');
  const lstm = cards.find((c) => c.id === 'lstm-base-model');
  ok(lstm.retrain.length === 2 && lstm.retrain.every((x) => x.pending) && lstm.deepLearning, 'view: both LSTM rows pending a fresh retrain');
  ok(view.overrides.length === 8 && view.overrides.find((x) => x.key === 'batchSize').retrains.length === 2, `view: every override listed (${view.overrides.length})`);
  ok(view.shared.find((x) => x.key === 'yearsOfHistory').override.value === 5, 'view: shared override');
  ok(params.served({ a: 1, sampleTemperatures: { x: 2 } }, { a: { value: 3 }, 'sampleTemperatures/y': { value: 4 }, b: 'bad' }).a === 3, 'served: override wins');

  // the predictor honours a request once: data/models/fresh-lotto.json
  fs.mkdirSync(path.join(root, 'data', 'models'), { recursive: true });
  fs.writeFileSync(path.join(root, 'data', 'models', 'fresh-lotto.json'), JSON.stringify({ 'LSTM Base Model': now.toISOString() }));
  const after = params.gameView(root, 'lotto').groups.flatMap((g) => g.cards).find((c) => c.id === 'lstm-base-model');
  ok(!after.retrain.find((x) => x.row === 'LSTM Base Model').pending && after.retrain.find((x) => x.row === 'LSTM Smoothed Model').pending, 'view: one done, one pending');

  const later = new Date('2026-11-09T08:00:00Z');
  r = params.releaseOverride(root, 'lotto', 'batchSize', 'carol', later);
  ok(r.ok && /tuned value 4/.test(r.message) && /new weights/.test(r.message), `release: the tuned value is back, the LSTM rows retrain (${r.message})`);
  o = file().games.lotto;
  ok(!o.overrides.batchSize && o.fresh['LSTM Base Model'] === later.toISOString(), 'release: removed, a new fresh request');
  r = params.releaseOverride(root, 'lotto', 'markovAlpha', 'carol', later);
  ok(r.ok && !/new weights/.test(r.message) && !file().games.lotto.overrides.markovAlpha, 'release a statistical override');
  ok(params.releaseOverride(root, 'lotto', 'markovAlpha', 'carol', later).status === 409, 'release twice: 409');
  ok(/code default/.test(params.releaseOverride(root, 'lotto', 'useMarkovMonteCarlo', 'c', later).message), 'release of a key not in the file: the code default');
  fs.rmSync(root, { recursive: true, force: true });
}

console.log(`params.test.js: ${checks} checks passed`);
