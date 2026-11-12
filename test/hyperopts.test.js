'use strict';
// The deep-learning hyperopts started from the admin pages (hyperopts.js, owner 11 Nov 2026): the form's checks, the
// tuner's arguments, queueing on the GPU worker's lane, the record with the tuner's result file, Apply (the gate's
// parameters, this run's best against the gate, the changed-since warning, the process.lock refusal), and a real launch
// of the lane through setsid with a stand-in for worker/remote-hyperopt.sh (no ssh, no worker) - in a temporary folder.
//   node test/hyperopts.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const hyperopts = require('../hyperopts');
const workers = require('../workers');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hyperopts-'));
const root = path.join(tmp, 'root');
fs.mkdirSync(root, { recursive: true });
const served0 = { yearsOfHistory: 10, gnn_gcnUnits: 48, gnn_dropout: 0.2, markovAlpha: 1, tuningGate: { 'Markov Model': { decision: 'kept' } } };
const writeServed = (v) => fs.writeFileSync(path.join(root, 'bestParams_lotto.json'), JSON.stringify(v, null, 4));
const readServed = () => JSON.parse(fs.readFileSync(path.join(root, 'bestParams_lotto.json'), 'utf-8'));
writeServed(served0);

(async () => {
  // --- the model table is the tuner's ---
  const py = fs.readFileSync(path.join(__dirname, '..', 'HyperoptDeepLearning.py'), 'utf-8');
  const block = py.slice(py.indexOf('MODEL_DISPLAY_NAMES = {'), py.indexOf('}', py.indexOf('MODEL_DISPLAY_NAMES = {')));
  const pyModels = Object.fromEntries([...block.matchAll(/"(\w+)": "([^"]+)"/g)].map((m) => [m[1], m[2]]));
  ok(JSON.stringify(pyModels) === JSON.stringify(hyperopts.MODELS), `MODELS matches HyperoptDeepLearning.MODEL_DISPLAY_NAMES: ${JSON.stringify(pyModels)}`);
  ok(/STUDY_SUFFIX = "skill"/.test(fs.readFileSync(path.join(__dirname, '..', 'src', 'DeepLearningTuning.py'), 'utf-8'))
    && hyperopts.studyKey('lotto', 'gnn_model') === 'lotto-gnn_model-skill', 'the study key is the tuner\'s');

  // --- the other tuners' rows are theirs (STRATEGY_DISPLAY_NAMES), and remote-hyperopt.sh runs every script ---
  const pyMap = (file, name) => {
    const src = fs.readFileSync(path.join(__dirname, '..', file), 'utf-8');
    const at = src.indexOf(`${name} = {`);
    const blk = src.slice(at, src.indexOf('}', at));
    return Object.fromEntries([...blk.matchAll(/"(\w+)": "([^"]+)"/g)].map((m) => [m[1], m[2]]));
  };
  const stat = pyMap('HyperoptStatistics.py', 'STRATEGY_DISPLAY_NAMES');
  ok(Object.entries(stat).every(([k, name]) => hyperopts.TUNERS.statistics.rows[k] === name) && Object.keys(stat).length === 8, `statistics rows match: ${JSON.stringify(stat)}`);
  const mm = fs.readFileSync(path.join(__dirname, '..', 'src', 'MarketModels.py'), 'utf-8');
  ok(mm.includes(`GARCH_NAME = "${hyperopts.TUNERS.statistics.rows.Garch}"`) && mm.includes(`HMM_NAME = "${hyperopts.TUNERS.statistics.rows.RegimeHmm}"`), 'the market rows\' names');
  ok(JSON.stringify(pyMap('HyperoptBoost.py', 'STRATEGY_DISPLAY_NAMES')) === JSON.stringify(hyperopts.TUNERS.boost.rows), 'boosting rows match');
  const rh = fs.readFileSync(path.join(__dirname, '..', 'worker', 'remote-hyperopt.sh'), 'utf-8');
  ok(Object.values(hyperopts.TUNERS).every((t) => rh.includes(t.script) && fs.existsSync(path.join(__dirname, '..', t.script))), 'every tuner script exists and remote-hyperopt.sh accepts it');

  // --- the form ---
  const p = hyperopts.parse({ game: 'lotto', model: 'gnn_model' });
  ok(p.ok && p.params.trials === 15 && p.params.days === 3 && p.params.trainSeconds === 240 && p.params.pruneAfterDays === 2
    && p.params.trialTimeout === 7200 && p.params.gateSeMargin === 1 && p.params.gate === true && p.params.fresh === false, `the tuner's defaults: ${JSON.stringify(p)}`);
  ok(!hyperopts.parse({ game: 'crypto', model: 'gnn_model' }).ok, 'a market is refused');
  ok(!hyperopts.parse({ game: 'lotto', model: 'GNN Model' }).ok, 'the display name is not a model type');
  ok(!hyperopts.parse({ game: 'lotto', model: 'gnn_model', trials: 2.5 }).ok, 'trials are whole');
  ok(!hyperopts.parse({ game: 'lotto', model: 'gnn_model', trials: 101 }).ok, 'at most 100 trials');
  ok(!hyperopts.parse({ game: 'lotto', model: 'gnn_model', days: 3, pruneAfterDays: 4 }).ok, 'pruning after more days than a trial has is refused');
  ok(!hyperopts.parse({ game: 'lotto', model: 'gnn_model', trainSeconds: '240; rm -rf /' }).ok, 'not a number: refused');
  const q = hyperopts.parse({ game: 'pick3', model: 'transformer_model', trials: '4', days: '5', pruneAfterDays: 0, gate: false, fresh: true, gateSeMargin: '0.5' });
  ok(q.ok && q.params.trials === 4 && q.params.gate === false && q.params.fresh === true && q.params.gateSeMargin === 0.5, 'strings from the form are read');
  ok(JSON.stringify(hyperopts.argsOf(q.params)) === JSON.stringify(['-g', 'pick3', '-m', 'transformer_model', '-t', '4', '-d', '5', '--train-seconds', '240',
    '--prune-after-days', '0', '--trial-timeout', '7200', '--gate', 'false', '--gate-se-margin', '0.5', '--variety-line', '0.95', '--fresh']), `the tuner's arguments: ${hyperopts.argsOf(q.params).join(' ')}`);
  ok(hyperopts.argsOf(q.params).every((a) => /^[A-Za-z0-9,._-]+$/.test(a)), 'every argument passes remote-hyperopt.sh\'s check');
  ok(hyperopts.budgetOf(p.params) === 17 * 7200 + 3600, 'the budget: every trial and both references at the trial timeout, plus an hour');
  ok(hyperopts.budgetOf({ ...p.params, trials: 100, trialTimeout: 21600 }) === 14 * 24 * 3600, 'at most two weeks');

  // --- start on a stub lane ---
  let queued = [];
  let laneCfg = { host: 'root@gpu.test' };
  let state = { current: null, queued: [], configured: true, host: 'gpu.test' };
  const lane = {
    config: () => laneCfg,
    queue: (id, by, days, deps) => { queued.push({ id, by, days, deps }); state.queued.push(id); return { ok: true, status: 200, message: 'Hyperopt started on the GPU worker' }; },
    cancel: (id) => { state.queued = state.queued.filter((x) => x !== id); return { ok: true, status: 200, message: 'Removed from the queue' }; },
    stopRun: (runId) => ({ ok: true, status: 200, message: `stopping ${runId}` }),
    backtestState: () => state,
  };
  const W = { lane: (k) => (k === 'gpu' ? lane : null) };
  const t0 = new Date('2026-11-11T12:00:00Z');
  const r = hyperopts.start(root, { game: 'lotto', model: 'gnn_model', trials: 5 }, 'tester', W, t0);
  ok(r.ok && r.id === 'lotto-gnn-model-20261111120000', `started: ${JSON.stringify(r)}`);
  ok(queued.length === 1 && queued[0].deps.hyperopt.budget === 7 * 7200 + 3600 && queued[0].deps.hyperopt.args.includes('gnn_model'), 'queued on the GPU lane as a hyperopt with its budget and arguments');
  const rec = hyperopts.readRecord(root, r.id);
  ok(rec.status === 'queued' && rec.row === 'GNN Model' && rec.worker === 'gpu' && rec.by === 'tester', 'the record');
  ok(fs.existsSync(path.join(root, 'config', 'hyperopts', `${r.id}.served.json`)), 'the served file is snapshotted');
  ok(hyperopts.start(root, { game: 'lotto', model: 'gnn_model' }, 'tester', W, t0).status === 409, 'the same second twice: refused');
  laneCfg = null;
  ok(hyperopts.start(root, { game: 'lotto', model: 'gnn_model' }, 'tester', W).status === 503, 'no GPU worker: refused');
  laneCfg = { host: 'root@gpu.test' };

  // --- the views ---
  let v = hyperopts.list(root, state).hyperopts[0];
  ok(v.status === 'queued' && v.place === 1 && v.study === null, 'queued, first in line, no result yet');
  state.current = { id: r.id, runId: 'gpu-hyperopt-x', type: 'hyperopt' };
  state.queued = [];
  ok(hyperopts.list(root, state).hyperopts[0].status === 'running', 'started on the lane: running before the record says so');
  const resultFile = path.join(root, 'config', 'hyperopts', `${r.id}.result.json`);
  const study = { game: 'lotto', model: 'gnn_model', status: 'running', wanted: 5, progress: { done: 2, states: { COMPLETE: 2 }, best: { trial: 7, score: 0.01, params: { gnn_gcnUnits: 64 } } } };
  fs.writeFileSync(resultFile, JSON.stringify({ studies: { 'lotto-gnn_model-skill': study } }));
  v = hyperopts.viewOf(root, hyperopts.readRecord(root, r.id), state, { full: true });
  ok(v.study.progress.done === 2 && Object.keys(v.apply).length === 0, 'the progress shows; nothing to apply while it runs');
  ok(hyperopts.apply(root, r.id, {}, 'tester', t0, { lockHolder: () => null }).status === 409, 'apply refused before the study is done');

  // the gate kept what is served: nothing to write unless this run's best is applied against it
  Object.assign(study, { status: 'done', decision: 'kept', reason: 'inside one SE', params: null, challengerParams: { gnn_gcnUnits: 64, gnn_dropout: 0.3 },
    record: { decision: 'kept', reason: 'inside one SE', challenger: 0.01, incumbent: 0.008 }, summary: 'x' });
  fs.writeFileSync(resultFile, JSON.stringify({ studies: { 'lotto-gnn_model-skill': study } }));
  state.current = null;
  const recRunning = hyperopts.readRecord(root, r.id);
  fs.writeFileSync(path.join(root, 'config', 'hyperopts', `${r.id}.json`), JSON.stringify({ ...recRunning, status: 'done' }));
  v = hyperopts.viewOf(root, hyperopts.readRecord(root, r.id), state, { full: true });
  const va = v.apply['lotto-gnn_model-skill'];
  ok(v.status === 'done' && va.gate === null && va.challenger.changes.length === 2 && va.challenger.changedSince.length === 0, `the view: no gate write, the best trial would change two values: ${JSON.stringify(v.apply)}`);
  ok(!('summary' in hyperopts.list(root, state).hyperopts[0].study), 'the list leaves the long parts out');
  let a = hyperopts.apply(root, r.id, { what: 'gate' }, 'tester', t0, { lockHolder: () => null });
  ok(!a.ok && /kept/.test(a.message), `the gate's "kept" has nothing to write: ${a.message}`);
  a = hyperopts.apply(root, r.id, { what: 'challenger' }, 'tester', t0, { lockHolder: () => 4242 });
  ok(!a.ok && /process\.lock, PID 4242/.test(a.message) && readServed().gnn_gcnUnits === 48, 'refused while a job holds process.lock, nothing written');
  // a value changed on this server since the run started: the gate compared against the old one
  writeServed({ ...served0, gnn_dropout: 0.25 });
  a = hyperopts.apply(root, r.id, { what: 'challenger' }, 'tester', t0, { lockHolder: () => null });
  ok(!a.ok && /changed since the run started \(gnn_dropout\)/.test(a.message) && readServed().gnn_gcnUnits === 48, `changed since: asks first: ${a.message}`);
  a = hyperopts.apply(root, r.id, { what: 'challenger', anyway: true }, 'tester', t0, { lockHolder: () => null });
  const after = readServed();
  ok(a.ok && after.gnn_gcnUnits === 64 && after.gnn_dropout === 0.3 && after.markovAlpha === 1 && after.yearsOfHistory === 10, `applied: only the model's values change: ${a.message}`);
  ok(after.tuningGate['Markov Model'].decision === 'kept' && after.tuningGate['GNN Model'].decision === 'replaced'
    && /applied by hand by tester against the gate's verdict \(kept/.test(after.tuningGate['GNN Model'].reason)
    && after.tuningGate['GNN Model'].applied.hyperopt === r.id && after.tuningGate['GNN Model'].applied.changedSince[0] === 'gnn_dropout', 'the gate record says it was applied by hand, against the gate');
  ok(hyperopts.readRecord(root, r.id).applied.by === 'tester', 'the hyperopt record says so');
  ok(hyperopts.apply(root, r.id, { what: 'challenger' }, 'tester', t0, { lockHolder: () => null }).status === 409, 'applied once only');
  ok(Object.keys(hyperopts.viewOf(root, hyperopts.readRecord(root, r.id), state, { full: true }).apply).length === 0, 'and nothing more to apply');

  // a gate that replaced: its parameters are applied as the tuner would have written them
  writeServed(served0);
  const r2 = hyperopts.start(root, { game: 'lotto', model: 'gnn_model' }, 'tester', W, new Date('2026-11-11T12:00:01Z'));
  fs.writeFileSync(path.join(root, 'config', 'hyperopts', `${r2.id}.result.json`), JSON.stringify({ studies: { 'lotto-gnn_model-skill': {
    status: 'done', decision: 'replaced', reason: 'beats both', params: { gnn_gcnUnits: 96 }, challengerParams: { gnn_gcnUnits: 96 }, record: { decision: 'replaced', reason: 'beats both' } } } }));
  ok(hyperopts.viewOf(root, hyperopts.readRecord(root, r2.id), state, { full: true }).apply['lotto-gnn_model-skill'].challenger === null, 'replaced: one choice only');
  ok(hyperopts.apply(root, r2.id, { what: 'challenger' }, 'tester', t0, { lockHolder: () => null }).status === 409, 'and not as "against the gate"');
  a = hyperopts.apply(root, r2.id, { what: 'gate' }, 'tester', t0, { lockHolder: () => null });
  ok(a.ok && readServed().gnn_gcnUnits === 96 && readServed().tuningGate['GNN Model'].decision === 'replaced' && readServed().tuningGate['GNN Model'].reason === 'beats both', 'the gate\'s result is written with its own record');

  // cancel and stop
  const r3 = hyperopts.start(root, { game: 'keno', model: 'lstm_model' }, 'tester', W, new Date('2026-11-11T12:00:02Z'));
  ok(hyperopts.cancel(root, r3.id, 'tester', W).ok && hyperopts.readRecord(root, r3.id).status === 'cancelled' && !state.queued.includes(r3.id), 'a queued one is cancelled');
  state.current = { id: r2.id, runId: 'gpu-hyperopt-y', type: 'hyperopt' };
  ok(/^Hyperopt stopping: its finished trials stay/.test(hyperopts.cancel(root, r2.id, 'tester', W).message), 'the running one is stopped through the lane');
  state.current = null;
  ok(hyperopts.cancel(root, 'nope', 'tester', W).status === 404, 'unknown id');

  // --- the other five tuners (owner, 12 Nov 2026): the CPU lane, the DL360's fallback, the variety line ---
  ok(hyperopts.settings(root).varietyLine === 95, 'no setting: the 95% line');
  ok(!hyperopts.saveSettings(root, { varietyLine: 40 }, 'tester').ok && !hyperopts.saveSettings(root, { varietyLine: 'x' }, 'tester').ok, 'a line under 50% or not a number: refused');
  ok(hyperopts.saveSettings(root, { varietyLine: 80 }, 'tester').ok && hyperopts.settings(root).varietyLine === 80
    && JSON.parse(fs.readFileSync(path.join(root, 'config', 'tuning.json'), 'utf-8')).varietyLine === 0.8, 'the setting is stored as a share for the tuners (config/tuning.json)');
  const sp = hyperopts.parse({ tuner: 'statistics', game: 'lotto', model: 'Markov' }, root);
  ok(sp.ok && sp.params.trials === 15 && sp.params.days === 90 && sp.params.trialTimeout === 1200 && sp.params.varietyLine === 80 && !('trainSeconds' in sp.params),
    `the statistical tuner's defaults, the line from the setting: ${JSON.stringify(sp)}`);
  ok(JSON.stringify(hyperopts.argsOf(sp.params)) === JSON.stringify(['-g', 'lotto', '-s', 'Markov', '-t', '15', '-d', '90', '--trial-timeout', '1200', '--gate-se-margin', '1', '--variety-line', '0.8']),
    `its arguments: ${hyperopts.argsOf(sp.params).join(' ')}`);
  ok(hyperopts.argsOf({ ...sp.params, gate: false }).includes('--no-gate'), 'gate off: --no-gate');
  ok(!hyperopts.parse({ tuner: 'statistics', game: 'pick3', model: 'PoissonMarkov' }).ok, 'a strategy that does not apply to a positional game: refused');
  ok(!hyperopts.parse({ tuner: 'statistics', game: 'lotto', model: 'Garch' }).ok && hyperopts.parse({ tuner: 'statistics', game: 'crypto', model: 'Garch' }).ok, 'GARCH on the markets only');
  ok(!hyperopts.parse({ tuner: 'boost', game: 'lotto', model: 'Markov' }).ok, 'a statistical row is not a boosting one');
  ok(!hyperopts.parse({ tuner: 'nope', game: 'lotto' }).ok, 'an unknown tuner: refused');
  ok(!hyperopts.parse({ tuner: 'quantum', game: 'lotto', varietyLine: 101 }).ok, 'a line over 100%: refused');
  const qp = hyperopts.parse({ tuner: 'quantum', game: 'lotto' });
  ok(qp.ok && qp.params.model === '' && qp.params.days === 300 && qp.params.gateSeMargin === 2.5, 'quantum: every row of the game, its own defaults');
  ok(hyperopts.argsOf(qp.params).every((x) => /^[A-Za-z0-9,._-]+$/.test(x)) && hyperopts.argsOf(sp.params).every((x) => /^[A-Za-z0-9,._-]+$/.test(x)), 'every argument passes remote-hyperopt.sh\'s check');
  ok(hyperopts.budgetOf(sp.params) === 17 * 1200 + 3600 + 1800, 'a CPU budget: the trials and references at their limit, an hour, half an hour to switch the DL360 on');

  let cpuCfg = { host: 'root@dl360.test' };
  const cpuQueued = [];
  const cpuState = { current: null, queued: [], configured: true, host: 'dl360.test' };
  const cpuLane = {
    config: () => cpuCfg,
    queue: (id, by, days, deps) => { cpuQueued.push({ id, deps }); cpuState.queued.push(id); return { ok: true, status: 200, message: 'Hyperopt queued for the CPU worker' }; },
    cancel: (id) => { cpuState.queued = cpuState.queued.filter((x) => x !== id); return { ok: true, status: 200, message: 'Removed' }; },
    stopRun: () => ({ ok: true, status: 200, message: 'stopping' }),
    backtestState: () => cpuState,
  };
  const W2 = { lane: (k) => (k === 'gpu' ? lane : k === 'cpu' ? cpuLane : null) };
  writeServed({ ...served0, markovAlpha: 1, modelScores: { 'Markov Model': 0.5, 'XGBoost Model': 0.7 } });
  const s1 = hyperopts.start(root, { tuner: 'statistics', game: 'lotto', model: 'Markov', trials: 3 }, 'tester', W2, new Date('2026-11-12T10:00:00Z'));
  ok(s1.ok && s1.id === 'lotto-statistics-markov-20261112100000' && cpuQueued[0].deps.hyperopt.args[0] === 'HyperoptStatistics.py' && !cpuQueued[0].deps.hyperopt.local,
    `queued on the CPU lane with the tuner's script first: ${JSON.stringify(cpuQueued[0])}`);
  const srec = hyperopts.readRecord(root, s1.id);
  ok(srec.tuner === 'statistics' && srec.row === 'Markov Model' && srec.worker === 'cpu', 'the record names the tuner, the row and the lane');
  cpuCfg = null;
  const s2 = hyperopts.start(root, { tuner: 'ensemble', game: 'keno' }, 'tester', W2, new Date('2026-11-12T10:00:01Z'));
  ok(s2.ok && cpuQueued[1].deps.hyperopt.local === true && /on this server/.test(s2.message), `no DL360 set up: it runs on this server: ${s2.message}`);
  cpuCfg = { host: 'root@dl360.test' };
  let lst = hyperopts.list(root, { gpu: state, cpu: cpuState });
  ok(lst.tuners.statistics.rows.Markov === 'Markov Model' && lst.variety.varietyLine === 80 && lst.cpu.configured && lst.hyperopts.find((h) => h.id === s1.id).place === 1,
    'the list: the tuners, the line, the CPU lane, the place in its queue');
  cpuState.current = { id: s1.id, runId: 'cpu-hyperopt-1', type: 'hyperopt', local: true };
  cpuState.queued = cpuState.queued.filter((x) => x !== s1.id);
  lst = hyperopts.list(root, { gpu: state, cpu: cpuState });
  ok(lst.hyperopts.find((h) => h.id === s1.id).status === 'running' && lst.hyperopts.find((h) => h.id === s1.id).where === 'this server', 'running on this server (the fallback): the view says where');
  ok(/stopping/.test(hyperopts.cancel(root, s1.id, 'tester', W2).message), 'stopped through the CPU lane');
  cpuState.current = null;

  // a finished statistical study: the gate replaced; Apply writes its params, the derived keys, the vote weight and the record
  fs.writeFileSync(path.join(root, 'config', 'hyperopts', `${s1.id}.result.json`), JSON.stringify({ studies: { lotto_Markov: {
    tuner: 'statistics', game: 'lotto', row: 'Markov Model', status: 'done', decision: 'replaced', reason: 'beats both by 1 SE',
    params: { markovAlpha: 3, markovSortedPrediction: true }, remove: ['gnn_dropout'], modelScores: { 'Markov Model': 0.9 },
    rows: { 'Markov Model': { decision: 'replaced', reason: 'beats both by 1 SE', variety: { rule: 'x', blind_trials: [2] } } },
    challengerParams: null, progress: { done: 3, states: { COMPLETE: 3 }, best: null } } } }));
  v = hyperopts.viewOf(root, hyperopts.readRecord(root, s1.id), cpuState, { full: true });
  const sa = v.apply.lotto_Markov;
  ok(v.studies.length === 1 && sa.gate.changes.length === 4 && sa.gate.changes.some((c) => c.key === 'gnn_dropout' && c.removed) && sa.gate.changes.some((c) => c.key === 'modelScores.Markov Model' && c.value === 0.9)
    && sa.challenger === null, `the view: values, a removed key, the vote weight: ${JSON.stringify(sa)}`);
  a = hyperopts.apply(root, s1.id, { what: 'gate', study: 'lotto_Markov' }, 'tester', t0, { lockHolder: () => null });
  const sv = readServed();
  ok(a.ok && sv.markovAlpha === 3 && sv.markovSortedPrediction === true && !('gnn_dropout' in sv) && sv.modelScores['Markov Model'] === 0.9 && sv.modelScores['XGBoost Model'] === 0.7
    && sv.tuningGate['Markov Model'].decision === 'replaced' && sv.tuningGate['Markov Model'].applied.hyperopt === s1.id && sv.tuningGate['Markov Model'].reason === 'beats both by 1 SE',
    `applied: only this row's values, weight and record change: ${a.message}`);
  ok(hyperopts.apply(root, s1.id, { what: 'gate', study: 'lotto_Markov' }, 'tester', t0, { lockHolder: () => null }).status === 409, 'applied once');

  // a quantum run: several studies, each applied on its own
  const s3 = hyperopts.start(root, { tuner: 'quantum', game: 'lotto' }, 'tester', W2, new Date('2026-11-12T10:00:02Z'));
  fs.writeFileSync(path.join(root, 'config', 'hyperopts', `${s3.id}.result.json`), JSON.stringify({ studies: {
    lotto_quantum_kernel: { status: 'done', row: 'QuantumMetaLearner Model', decision: 'kept', reason: 'inside 2.5 SE', params: {}, rows: { 'QuantumMetaLearner Model': { decision: 'kept' } } },
    lotto_quantum_vqc: { status: 'done', row: 'QuantumVQC Model', decision: 'replaced', reason: 'beats both', params: { quantumVqcLayers: 3 }, rows: { 'QuantumVQC Model': { decision: 'replaced' } } },
    lotto_classical_svm: { status: 'running', row: 'ClassicalSVM Model', progress: { done: 1 } } } }));
  v = hyperopts.viewOf(root, hyperopts.readRecord(root, s3.id), cpuState, { full: true });
  ok(v.studies.length === 3 && v.study.key === 'lotto_classical_svm' && v.apply.lotto_quantum_kernel.gate === null && v.apply.lotto_quantum_vqc.gate.changes[0].key === 'quantumVqcLayers' && !v.apply.lotto_classical_svm,
    'three studies: the running one on the table line, the kept one has nothing to write, the replaced one its value');
  ok(!hyperopts.apply(root, s3.id, { what: 'gate', study: 'lotto_quantum_kernel' }, 'tester', t0, { lockHolder: () => null }).ok, 'kept: nothing to write');
  a = hyperopts.apply(root, s3.id, { what: 'gate', study: 'lotto_quantum_vqc' }, 'tester', t0, { lockHolder: () => null });
  ok(a.ok && readServed().quantumVqcLayers === 3 && readServed().tuningGate['QuantumVQC Model'].decision === 'replaced', `one study applied: ${a.message}`);
  v = hyperopts.viewOf(root, hyperopts.readRecord(root, s3.id), cpuState, { full: true });
  ok(!v.apply.lotto_quantum_vqc && v.appliedStudies.lotto_quantum_vqc.by === 'tester' && 'lotto_quantum_kernel' in v.apply, 'that study is marked applied, the others stay open');

  // the lock check, for real: this process holds it
  fs.writeFileSync(path.join(root, 'process.lock'), String(process.pid));
  ok(hyperopts.lockHolder(root) === process.pid, 'a live PID in process.lock is a holder');
  fs.writeFileSync(path.join(root, 'process.lock'), '999999999');
  ok(hyperopts.lockHolder(root) === null, 'a dead one is not');
  fs.unlinkSync(path.join(root, 'process.lock'));

  // the log, with progress bars folded to their last state
  fs.mkdirSync(path.join(root, 'log', 'hyperopts'), { recursive: true });
  fs.writeFileSync(path.join(root, 'log', 'hyperopts', `${r.id}.log`), 'start\n[#--] 1/3\r[##-] 2/3\r[###] 3/3\ndone\n');
  ok(hyperopts.logTail(root, r.id).text === 'start\n[###] 3/3\ndone\n', 'the log tail');
  fs.writeFileSync(path.join(root, 'log', 'hyperopts', `${r.id}.log`),
    'Epoch 1/9\n\r\x1b[1m  1/717\x1b[0m \x1b[37m━━\x1b[0m loss: 2.30\x08\x08\x08\r\x1b[1m717/717\x1b[0m \x1b[32m━━\x1b[0m loss: 2.29\x08\x08\nEpoch 2/9\n');
  ok(hyperopts.logTail(root, r.id).text === 'Epoch 1/9\n717/717 ━━ loss: 2.29\nEpoch 2/9\n', 'Keras bars: no colour codes, no backspaces');
  ok(hyperopts.logTail(root, '../etc') === null, 'no path out of the folder');

  // --- the lane for real: a hyperopt entry launches remote-hyperopt.sh's stand-in with its budget and arguments ---
  const cfgDir = path.join(tmp, 'config');
  const logDir = path.join(tmp, 'log');
  fs.mkdirSync(cfgDir, { recursive: true });
  fs.writeFileSync(path.join(cfgDir, 'workers.json'), JSON.stringify({ gpu: { host: 'root@gpu.test' } }));
  const fakeBt = path.join(tmp, 'fake-backtest.sh');
  const fakeHo = path.join(tmp, 'fake-hyperopt.sh');
  fs.writeFileSync(fakeBt, `#!/usr/bin/env bash\necho "backtest $*" >> ${tmp}/calls\n`);
  fs.writeFileSync(fakeHo, `#!/usr/bin/env bash\necho "hyperopt $*" >> ${tmp}/calls\necho "tuner output"\nsleep 0.3\nexit 0\n`);
  workers._internals.useDirs(cfgDir, logDir, fakeBt, path.join(tmp, 'no-power.sh'), fakeHo);
  const G = workers.lane('gpu');
  const res = G.queue('lotto-gnn-model-1', 'tester', 3, { hyperopt: { budget: 1234, args: ['-g', 'lotto', '-m', 'gnn_model', '--fresh'] } });
  ok(res.ok && /^Hyperopt started on the GPU worker/.test(res.message), `the lane says hyperopt: ${res.message}`);
  ok(G.backtestState().current.type === 'hyperopt', 'its state says what runs');
  const end = Date.now() + 8000;
  while (Date.now() < end && G.backtestState().current) { G.poll(); await sleep(100); }
  ok(!G.backtestState().current && G.state.history[0].type === 'hyperopt' && G.state.history[0].exit === 0, 'it ended, recorded as a hyperopt');
  const calls = fs.readFileSync(path.join(tmp, 'calls'), 'utf-8');
  ok(/^hyperopt lotto-gnn-model-1 1234 gpu-hyperopt-\S+ -g lotto -m gnn_model --fresh$/m.test(calls) && !/backtest/.test(calls), `the hyperopt script got id, budget, run id and the tuner's arguments: ${calls}`);
  const hlog = fs.readFileSync(path.join(logDir, 'hyperopts', 'lotto-gnn-model-1.log'), 'utf-8');
  ok(/=== gpu worker: hyperopt lotto-gnn-model-1 started/.test(hlog) && /tuner output/.test(hlog) && /finished status=0/.test(hlog), 'its own log, with markers');
  ok(!fs.existsSync(path.join(logDir, 'backtest-gpu.log')), 'the backtests\' log is left alone');

  // --- the CPU lane: the DL360 does not come up (75) - the hyperopt runs again, on this server ---
  fs.writeFileSync(path.join(cfgDir, 'workers.json'), JSON.stringify({ gpu: { host: 'root@gpu.test' }, cpu: { host: 'root@dl360.test', localDir: '/opt/test-local' } }));
  fs.writeFileSync(fakeHo, `#!/usr/bin/env bash\necho "hyperopt $WORKER_HOST $WORKER_DIR $WORKER_PYTHON \${WORKER_ILO_ENV:-noilo} $*" >> ${tmp}/calls-cpu\nsleep 0.2\n[ "$WORKER_HOST" = local ] && exit 0\nexit 75\n`);
  const C = workers.lane('cpu');
  const cres = C.queue('lotto-statistics-markov-1', 'tester', 90, { hyperopt: { budget: 999, args: ['HyperoptStatistics.py', '-g', 'lotto', '-s', 'Markov'] } });
  ok(cres.ok && C.backtestState().current.type === 'hyperopt' && !C.backtestState().current.local, `started on the DL360 first: ${cres.message}`);
  const cend = Date.now() + 15000;
  while (Date.now() < cend && (C.backtestState().current || C.backtestState().queued.length)) { C.poll(); C.startNext(); await sleep(100); }
  const ccalls = fs.readFileSync(path.join(tmp, 'calls-cpu'), 'utf-8').trim().split('\n');
  ok(ccalls.length === 2 && /^hyperopt root@dl360\.test \/opt\/sequencePredictor-worker \S+ noilo lotto-statistics-markov-1 999 cpu-hyperopt-\S+ HyperoptStatistics\.py -g lotto -s Markov$/.test(ccalls[0])
    && /^hyperopt local \/opt\/test-local python3 noilo lotto-statistics-markov-1 999 cpu-hyperopt-\S+ HyperoptStatistics\.py -g lotto -s Markov$/.test(ccalls[1]), `the DL360 first, then this server in its own folder: ${ccalls.join(' | ')}`);
  ok(C.state.history[1].exit === 75 && /moved to this server/.test(C.state.history[1].note) && C.state.history[0].exit === 0, 'the history says it moved, and that the second run ended well');

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`hyperopts.test.js: ${checks} checks passed`);
})().catch((e) => { console.error(e); fs.rmSync(tmp, { recursive: true, force: true }); process.exit(1); });
