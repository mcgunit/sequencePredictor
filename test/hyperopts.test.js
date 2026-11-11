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
    '--prune-after-days', '0', '--trial-timeout', '7200', '--gate', 'false', '--gate-se-margin', '0.5', '--fresh']), `the tuner's arguments: ${hyperopts.argsOf(q.params).join(' ')}`);
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
  ok(v.study.progress.done === 2 && v.apply === null, 'the progress shows; nothing to apply while it runs');
  ok(hyperopts.apply(root, r.id, {}, 'tester', t0, { lockHolder: () => null }).status === 409, 'apply refused before the study is done');

  // the gate kept what is served: nothing to write unless this run's best is applied against it
  Object.assign(study, { status: 'done', decision: 'kept', reason: 'inside one SE', params: null, challengerParams: { gnn_gcnUnits: 64, gnn_dropout: 0.3 },
    record: { decision: 'kept', reason: 'inside one SE', challenger: 0.01, incumbent: 0.008 }, summary: 'x' });
  fs.writeFileSync(resultFile, JSON.stringify({ studies: { 'lotto-gnn_model-skill': study } }));
  state.current = null;
  const recRunning = hyperopts.readRecord(root, r.id);
  fs.writeFileSync(path.join(root, 'config', 'hyperopts', `${r.id}.json`), JSON.stringify({ ...recRunning, status: 'done' }));
  v = hyperopts.viewOf(root, hyperopts.readRecord(root, r.id), state, { full: true });
  ok(v.status === 'done' && v.apply.gate === null && v.apply.challenger.changes.length === 2 && v.apply.challenger.changedSince.length === 0, `the view: no gate write, the best trial would change two values: ${JSON.stringify(v.apply)}`);
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
  ok(hyperopts.viewOf(root, hyperopts.readRecord(root, r.id), state, { full: true }).apply === null, 'and nothing more to apply');

  // a gate that replaced: its parameters are applied as the tuner would have written them
  writeServed(served0);
  const r2 = hyperopts.start(root, { game: 'lotto', model: 'gnn_model' }, 'tester', W, new Date('2026-11-11T12:00:01Z'));
  fs.writeFileSync(path.join(root, 'config', 'hyperopts', `${r2.id}.result.json`), JSON.stringify({ studies: { 'lotto-gnn_model-skill': {
    status: 'done', decision: 'replaced', reason: 'beats both', params: { gnn_gcnUnits: 96 }, challengerParams: { gnn_gcnUnits: 96 }, record: { decision: 'replaced', reason: 'beats both' } } } }));
  ok(hyperopts.viewOf(root, hyperopts.readRecord(root, r2.id), state, { full: true }).apply.challenger === null, 'replaced: one choice only');
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

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`hyperopts.test.js: ${checks} checks passed`);
})().catch((e) => { console.error(e); fs.rmSync(tmp, { recursive: true, force: true }); process.exit(1); });
