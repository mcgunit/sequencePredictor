'use strict';
// The backtests started from the admin pages (backtests.js): the table of rows matches RunBacktest.py's own (read by
// importing it - the script is never run), a start writes its record and queues it, every refusal says why, the list
// reconciles a record with the scheduler, cancel and stop - all on a temp root with a stand-in scheduler; the
// deep-learning rows go to a stand-in GPU worker (workers.js) with their own limits.
//   node test/backtests.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const bt = require('../backtests');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- the rows, as RunBacktest.py covers them (imported, not run) ----------------------------------------------------
let py = null;
try {
  py = JSON.parse(execFileSync('python3', ['-c', 'import json, RunBacktest; print(json.dumps(RunBacktest.covered(), sort_keys=True))'],
    { cwd: path.join(__dirname, '..'), encoding: 'utf-8', env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1' } }));
} catch (e) { py = null; }
if (py) {
  const sorted = (o) => JSON.stringify(Object.fromEntries(Object.keys(o).sort().map((k) => [k, [...o[k]].sort()])));
  ok(sorted(py) === sorted(bt.ROWS), `backtests.js ROWS = RunBacktest.covered()`);
  const gpu = JSON.parse(execFileSync('python3', ['-c', 'import json, RunBacktest as r; print(json.dumps([sorted(r.GPU_ROWS), r.GPU_DAYS_DEFAULT, r.GPU_DAYS_MAX]))'],
    { cwd: path.join(__dirname, '..'), encoding: 'utf-8', env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1' } }));
  ok(JSON.stringify(gpu) === JSON.stringify([[...bt.GPU_ROWS].sort(), bt.GPU_DAYS_DEFAULT, bt.GPU_DAYS_MAX]), 'backtests.js GPU rows and limits = RunBacktest\'s');
}
ok(bt.canBacktest('lotto', 'Markov Model') && bt.canBacktest('crypto', 'GARCH Model') && bt.canBacktest('crypto', 'Markov Model'), 'statistical and market rows');
ok(bt.canBacktest('keno', 'CatBoost Model') && !bt.canBacktest('pick3', 'XGBoostMultiLabel Model') && !bt.canBacktest('crypto', 'XGBoost Model'), 'boosting: not multi-label on pick3, none on the markets');
ok(!bt.canBacktest('lotto', 'GARCH Model') && !bt.canBacktest('lotto', 'QuantumVQC Model'), 'no market row on lotto, no quantum row yet');
ok(bt.canBacktest('lotto', 'LSTM Base Model') && bt.canBacktest('pick3', 'GNN Model') && !bt.canBacktest('crypto', 'LSTM Base Model'), 'the DL rows on the lottery games only');
ok(bt.limitsFor('Transformer Model').gpu && bt.limitsFor('Transformer Model').daysDefault === 3 && bt.limitsFor('Transformer Model').daysMax === 30
  && !bt.limitsFor('Markov Model').gpu && bt.limitsFor('Markov Model').daysMax === 1000, 'the DL rows: GPU, 3 draws by default, 30 at most');

// --- a stand-in scheduler -----------------------------------------------------------------------------------------
function scheduler() {
  const s = { queued: [], current: null, stopped: [], cancelled: [], refuse: null };
  s.queueBacktest = (id) => { if (s.refuse) return s.refuse; s.queued.push(id); return { ok: true, status: 200, message: `Backtest queued (number ${s.queued.length} in the queue)` }; };
  s.backtestState = () => ({ current: s.current, queued: [...s.queued], budget: 3600, nextSlotSeconds: 7200 });
  s.cancelBacktest = (id) => { s.queued = s.queued.filter((x) => x !== id); s.cancelled.push(id); return { ok: true, status: 200, message: 'Backtest removed from the queue' }; };
  s.stopRun = (runId) => { s.stopped.push(runId); return { ok: true, status: 200, message: 'stopping' }; };
  return s;
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'backtests-'));
const jobs = scheduler();
const now = new Date('2026-11-08T21:00:00Z');
const refused = (r, re, msg) => ok(!r.ok && re.test(r.message), `${msg}: ${r.message}`);

let r = bt.start(root, { game: 'lotto', row: 'Markov Model' }, 'alice', jobs, now);
ok(r.ok && r.id === 'lotto-markov-model-20261108210000' && jobs.queued[0] === r.id && /queued/.test(r.message), `start: ${r.message}`);
let rec = bt.readRecord(root, r.id);
ok(rec.days === 90 && rec.values === 'served' && rec.by === 'alice' && rec.status === 'queued' && rec.requestedAt === now.toISOString(), 'the record: the defaults, queued');
refused(bt.start(root, { game: 'lotto', row: 'Markov Model' }, 'alice', jobs, now), /this very second/, 'the same id twice');
refused(bt.start(root, { game: 'lotto', row: 'QuantumVQC Model' }, 'a', jobs, now), /cannot be backtested here yet/, 'a quantum row');
refused(bt.start(root, { game: 'lotto', row: 'LSTM Base Model' }, 'a', jobs, now), /no GPU worker/, 'a DL row without a GPU worker');
refused(bt.start(root, { game: 'pick3', row: 'PoissonMarkov Model' }, 'a', jobs, now), /does not run/, 'a row the game does not run');
refused(bt.start(root, { game: 'lotto', row: 'Markov Model', days: 0 }, 'a', jobs, now), /1 to 1000/, 'days too few');
refused(bt.start(root, { game: 'lotto', row: 'Markov Model', days: 2.5 }, 'a', jobs, now), /whole number/, 'days not whole');
refused(bt.start(root, { game: 'lotto', row: 'Markov Model', values: 'manual' }, 'a', jobs, now), /served .* or tuned/, 'values');
refused(bt.start(root, { game: 'lotto', row: 'Markov Model' }, 'a', null, now), /not available/, 'no scheduler');
jobs.refuse = { ok: false, status: 409, message: 'the schedule is disabled' };
const later = new Date('2026-11-08T21:00:05Z');
refused(bt.start(root, { game: 'keno', row: 'XGBoost Model', days: 30, values: 'tuned' }, 'b', jobs, later), /disabled/, 'refused by the scheduler');
ok(!bt.readRecord(root, 'keno-xgboost-model-20261108210005'), 'a refused start leaves no record');
jobs.refuse = null;
const r2 = bt.start(root, { game: 'keno', row: 'XGBoost Model', days: 30, values: 'tuned' }, 'b', jobs, later);
ok(r2.ok && bt.readRecord(root, r2.id).values === 'tuned', 'a second backtest, tuned values');
ok(bt.readRecord(root, '../etc/passwd') === null && bt.readRecord(root, 'a b') === null, 'ids are names, never paths');

// --- the list: newest first, queue places, a result summarised, a lost run shown as ended --------------------------
let list = bt.list(root, jobs.backtestState());
ok(list.backtests.length === 2 && list.backtests[0].id === r2.id && list.backtests[0].place === 2 && list.backtests[1].place === 1, 'newest first, with queue places');
ok(list.rows['Markov Model'].includes('lotto') && list.daysDefault === 90 && list.scheduler.budget === 3600, 'the rows and the scheduler for the page');
bt.writeRecord(root, { ...bt.readRecord(root, r.id), status: 'done', result: { score: 0.84, tuning: { kind: 'hits' }, days: [{ date: '2026-11-04', hits: 1 }], values: { markovAlpha: 0.5 }, lockboxWithheld: 19 } });
jobs.queued = jobs.queued.filter((x) => x !== r.id);
list = bt.list(root, jobs.backtestState());
const done = list.backtests.find((b) => b.id === r.id);
ok(done.result.dayCount === 1 && done.result.days === undefined && done.result.values === undefined && done.result.score === 0.84, 'the list summarises a result (no day lines)');
const full = bt.viewOf(bt.readRecord(root, r.id), jobs.backtestState(), { full: true });
ok(full.result.days.length === 1 && full.result.values.markovAlpha === 0.5, 'the detail keeps every day');
bt.writeRecord(root, { ...bt.readRecord(root, r2.id), status: 'running' });
jobs.queued = [];
const lost = bt.list(root, jobs.backtestState()).backtests.find((b) => b.id === r2.id);
ok(lost.status === 'ended' && /without writing a result/.test(lost.error), 'running in the record, not in the scheduler: ended');
jobs.current = { id: r2.id, runId: 'backtest-2026' };
ok(bt.list(root, jobs.backtestState()).backtests.find((b) => b.id === r2.id).runId === 'backtest-2026', 'the running one carries its run id (for Stop)');

// --- cancel and stop ------------------------------------------------------------------------------------------------
r = bt.cancel(root, r2.id, 'carol', jobs);
ok(r.ok && jobs.stopped[0] === 'backtest-2026' && /stopping/i.test(r.message), 'stop the running backtest');
jobs.current = null;
const r3 = bt.start(root, { game: 'eurodreams', row: 'LaplaceMonteCarlo Model', days: 10 }, 'c', jobs, new Date('2026-11-08T22:00:00Z'));
r = bt.cancel(root, r3.id, 'carol', jobs);
ok(r.ok && jobs.cancelled.includes(r3.id) && bt.readRecord(root, r3.id).status === 'cancelled' && /carol/.test(bt.readRecord(root, r3.id).error), 'cancel a queued one');
refused(bt.cancel(root, r3.id, 'carol', jobs), /cancelled, not queued/, 'cancel twice');
ok(bt.cancel(root, 'nope', 'x', jobs).status === 404, 'cancel an unknown one: 404');

// --- the deep-learning rows: the GPU worker's queue -----------------------------------------------------------------
function gpuWorker() {
  const w = { queued: [], current: null, stopped: [], cancelled: [], days: {} };
  w.queue = (id, by, days) => { w.queued.push(id); w.days[id] = days; return { ok: true, status: 200, message: `Backtest queued for the GPU worker (number ${w.queued.length})` }; };
  w.backtestState = () => ({ current: w.current, queued: [...w.queued], configured: true, host: '10.0.0.9' });
  w.cancel = (id) => { w.queued = w.queued.filter((x) => x !== id); w.cancelled.push(id); return { ok: true, status: 200, message: 'Backtest removed from the queue' }; };
  w.stopRun = (runId) => { w.stopped.push(runId); return { ok: true, status: 200, message: 'stopping' }; };
  return w;
}
const gw = gpuWorker();
const t9 = new Date('2026-11-10T09:00:00Z');
const g1 = bt.start(root, { game: 'keno', row: 'GNN Model' }, 'dora', jobs, t9, gw);
ok(g1.ok && gw.queued[0] === g1.id && gw.days[g1.id] === 3 && !jobs.queued.includes(g1.id) && /GPU worker/.test(g1.message), `a DL row goes to the GPU worker with 3 draws: ${g1.message}`);
ok(bt.readRecord(root, g1.id).days === 3, 'its record: 3 draws');
refused(bt.start(root, { game: 'keno', row: 'GNN Model', days: 31 }, 'dora', jobs, new Date('2026-11-10T09:00:01Z'), gw), /1 to 30/, 'a DL row: 30 draws at most');
refused(bt.start(root, { game: 'crypto', row: 'LSTM Base Model' }, 'dora', jobs, new Date('2026-11-10T09:00:02Z'), gw), /lottery games/, 'a DL row on a market');
const g2 = bt.start(root, { game: 'lotto', row: 'LSTM Base Model', days: 30, values: 'tuned' }, 'dora', jobs, new Date('2026-11-10T09:00:03Z'), gw);
ok(g2.ok && gw.days[g2.id] === 30, '30 draws allowed');
list = bt.list(root, jobs.backtestState(), gw.backtestState());
const lg1 = list.backtests.find((b) => b.id === g1.id);
ok(lg1.worker === 'gpu' && lg1.place === 1 && list.backtests.find((b) => b.id === g2.id).place === 2, 'the GPU queue places, worker gpu');
ok(list.gpu.configured && list.gpu.host === '10.0.0.9' && list.gpu.queued === 2 && list.limits['GNN Model'].daysMax === 30, 'the list carries the GPU worker and the limits');
ok(bt.list(root, jobs.backtestState(), null).backtests.find((b) => b.id === g1.id).status === 'queued' && !bt.list(root, null, null).gpu.configured, 'no GPU worker state: shown as recorded, not configured');
gw.current = { id: g1.id, runId: 'gpu-backtest-1' };
gw.queued = gw.queued.filter((x) => x !== g1.id);
ok(bt.list(root, jobs.backtestState(), gw.backtestState()).backtests.find((b) => b.id === g1.id).runId === 'gpu-backtest-1', 'the running GPU one carries its run id');
r = bt.cancel(root, g1.id, 'erin', jobs, gw);
ok(r.ok && gw.stopped[0] === 'gpu-backtest-1' && jobs.stopped.length === 1, 'Stop goes to the GPU worker, not the scheduler');
r = bt.cancel(root, g2.id, 'erin', jobs, gw);
ok(r.ok && gw.cancelled.includes(g2.id) && bt.readRecord(root, g2.id).status === 'cancelled', 'cancel a queued GPU one');
ok(bt.cancel(root, g2.id, 'erin', jobs, null).status === 503 || !bt.cancel(root, g2.id, 'erin', jobs, null).ok, 'no GPU worker: cannot cancel there');

fs.rmSync(root, { recursive: true, force: true });
console.log(`backtests.test.js: ${checks} checks passed${py ? '' : ' (python3 not available: the row table was not compared)'}`);
