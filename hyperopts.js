'use strict';
// Hyperopts started from the admin pages (owner, 11 Nov 2026; docs/operations.md, The workers): one deep-learning
// model on one game, HyperoptDeepLearning.py on the GPU worker (workers.js gpu lane, worker/remote-hyperopt.sh) - the
// same queue as the deep-learning backtests, one run at a time, beside the schedule (no process.lock here).
//
// Record only (the owner's decision): the run writes nothing that is served. The worker's copy of
// bestParams_<game>.json is thrown away; the tuner's --result-file comes back as config/hyperopts/<id>.result.json
// (per study: trials done, this run's best, the gate's verdict and the parameters it would write), and the page shows
// it. Apply writes those parameters - the gate's, or this run's best trial against the gate - with the gate's record
// into bestParams_<game>.json on this server; the next pipeline run commits it (its git add -A), as every tuner write.
// Apply is refused while a job holds process.lock (a tuner or the predictor may be writing the same file), and warns
// when a value it would replace was changed since the run started (the served file is snapshotted then).
//
// Files (all gitignored, under config/hyperopts/): <id>.json the request and its status (the server writes it,
// remote-hyperopt.sh marks running/done/failed/stopped), <id>.result.json the tuner's result, <id>.served.json the
// served file when the run was queued. The run's output: log/hyperopts/<id>.log.
const fs = require('fs');
const path = require('path');

const DIR = path.join('config', 'hyperopts');
const LOG_DIR = path.join('log', 'hyperopts');
const KEPT = 100;
const GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
// HyperoptDeepLearning.MODEL_DISPLAY_NAMES (test/hyperopts.test.js checks it against the file)
const MODELS = {
  lstm_model: 'LSTM Base Model',
  tcn_model: 'TCN Base Model',
  unified_lstm_tcn_model: 'UnifiedLstmTcn Model',
  unified_lstm_gru_tcn_model: 'UnifiedLstmGruTcn Model',
  transformer_model: 'Transformer Model',
  gnn_model: 'GNN Model',
  autoencoder_model: 'Autoencoder Model',
};
// the form's fields: {default, min, max} - the defaults are the tuner's own (HyperoptDeepLearning.py, src/DeepLearningTuning.py)
const FIELDS = {
  trials: { default: 15, min: 1, max: 100, integer: true },
  days: { default: 3, min: 1, max: 30, integer: true },
  trainSeconds: { default: 240, min: 30, max: 3600, integer: true },
  pruneAfterDays: { default: 2, min: 0, max: 30, integer: true },
  trialTimeout: { default: 7200, min: 600, max: 6 * 3600, integer: true },
  gateSeMargin: { default: 1, min: 0, max: 5, integer: false },
};
const BUDGET_MAX = 14 * 24 * 3600;
// the run's wall-clock budget: every trial at its timeout, the gate's two references (full trials), an hour to start
const budgetOf = (p) => Math.min(BUDGET_MAX, (p.trials + (p.gate ? 2 : 0)) * p.trialTimeout + 3600);
const STATUSES = ['queued', 'running', 'done', 'failed', 'out of time', 'stopped', 'lost', 'cancelled', 'ended'];

const ID = /^[A-Za-z0-9_-]{1,80}$/;
const fileOf = (root, id, kind = '') => path.join(root, DIR, `${id}${kind ? `.${kind}` : ''}.json`);
const studyKey = (game, model) => `${game}-${model}-skill`;   // src/DeepLearningTuning.study_name
function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return null; }
}
function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.tmp`, `${JSON.stringify(value, null, 1)}\n`);
  fs.renameSync(`${file}.tmp`, file);
}
const readRecord = (root, id) => (ID.test(String(id || '')) ? readJson(fileOf(root, id)) : null);
function allRecords(root) {
  let names = [];
  try { names = fs.readdirSync(path.join(root, DIR)).filter((n) => /^[A-Za-z0-9_-]+\.json$/.test(n)); } catch (e) { return []; }
  return names.map((n) => readRecord(root, n.slice(0, -5))).filter((r) => r && r.id && r.game)
    .sort((a, b) => String(b.requestedAt || '').localeCompare(String(a.requestedAt || '')));
}

// The tuner's arguments for a record (remote-hyperopt.sh adds -s false and --result-file)
function argsOf(rec) {
  return ['-g', rec.game, '-m', rec.model, '-t', String(rec.trials), '-d', String(rec.days), '--train-seconds', String(rec.trainSeconds),
    '--prune-after-days', String(rec.pruneAfterDays), '--trial-timeout', String(rec.trialTimeout), '--gate', rec.gate ? 'true' : 'false',
    '--gate-se-margin', String(rec.gateSeMargin), ...(rec.fresh ? ['--fresh'] : [])];
}

// Apply's view of a finished study: what it would write, and which served values changed since the run started
function applyView(root, rec, study) {
  if (!study || study.status !== 'done') return null;
  const served = readJson(path.join(root, `bestParams_${rec.game}.json`)) || {};
  const before = readJson(fileOf(root, rec.id, 'served')) || {};
  const side = (params) => {
    if (!params || typeof params !== 'object' || !Object.keys(params).length) return null;
    const keys = Object.keys(params);
    return {
      changes: keys.filter((k) => JSON.stringify(served[k]) !== JSON.stringify(params[k])).map((k) => ({ key: k, served: served[k] ?? null, value: params[k] })),
      changedSince: keys.filter((k) => JSON.stringify(served[k]) !== JSON.stringify(before[k])),
    };
  };
  return { gate: side(study.params), challenger: study.decision === 'replaced' ? null : side(study.challengerParams) };
}

// The record as the pages read it, with the lane's state (`ws`: workers.lane('gpu').backtestState()) and the result
function viewOf(root, rec, ws, { full = false } = {}) {
  const out = { ...rec };
  const result = readJson(fileOf(root, rec.id, 'result'));
  const study = result && result.studies ? result.studies[studyKey(rec.game, rec.model)] || null : null;
  out.study = study ? { ...study } : null;
  if (out.study && !full) { delete out.study.summary; delete out.study.record; }
  if (ws) {
    const place = ws.queued.indexOf(rec.id);
    const running = ws.current && ws.current.id === rec.id;
    if (rec.status === 'queued' && place >= 0) out.place = place + 1;
    if (running) {
      out.runId = ws.current.runId;
      if (rec.status === 'queued') out.status = 'running';
    }
    if ((rec.status === 'queued' && place < 0 && !running) || (rec.status === 'running' && !running)) {
      out.status = 'ended';
      out.error = rec.status === 'running' ? `the run ended without marking its record (killed, or the server lost it) - see log/hyperopts/${rec.id}.log`
        : "no longer in the GPU worker's queue";
    }
  }
  if (full) out.apply = rec.applied ? null : applyView(root, rec, study);
  return out;
}

function list(root, ws) {
  return {
    hyperopts: allRecords(root).slice(0, KEPT).map((r) => viewOf(root, r, ws)),
    games: GAMES, models: MODELS, fields: FIELDS, budgetMaxSeconds: BUDGET_MAX,
    gpu: ws ? { configured: Boolean(ws.configured), host: ws.host || null, running: ws.current ? ws.current.id : null, queued: ws.queued.length }
      : { configured: false, host: null, running: null, queued: 0 },
  };
}

// The form's values, checked: {ok, params} or {ok:false, status, message}
function parse(body) {
  const b = body || {};
  const game = String(b.game || '');
  const model = String(b.model || '');
  if (!MODELS[model]) return { ok: false, status: 400, message: `${model || 'that model'} is not a deep-learning model the tuner knows` };
  if (!GAMES.includes(game)) return { ok: false, status: 400, message: `${game || 'that game'} is not a lottery game the deep-learning tuner runs` };
  const p = { game, model };
  for (const [k, f] of Object.entries(FIELDS)) {
    const raw = b[k];
    const v = raw === undefined || raw === null || raw === '' ? f.default : Number(raw);
    if (!Number.isFinite(v) || (f.integer && !Number.isInteger(v)) || v < f.min || v > f.max) {
      return { ok: false, status: 400, message: `${k} is ${f.integer ? 'a whole number' : 'a number'} from ${f.min} to ${f.max}` };
    }
    p[k] = v;
  }
  if (p.pruneAfterDays > p.days) return { ok: false, status: 400, message: 'prune after days is at most the days per trial (0 = no pruning)' };
  p.gate = b.gate === undefined ? true : b.gate === true || b.gate === 'true';
  p.fresh = b.fresh === true || b.fresh === 'true';
  return { ok: true, params: p };
}

function start(root, body, user, workers, now = new Date()) {
  const parsed = parse(body);
  if (!parsed.ok) return parsed;
  const p = parsed.params;
  const lane = workers && typeof workers.lane === 'function' ? workers.lane('gpu') : null;
  if (!lane || !lane.config()) return { ok: false, status: 503, message: 'no GPU worker is set up (config/workers.json): the deep-learning hyperopts run there' };
  const id = `${p.game}-${p.model.replace(/_/g, '-')}-${now.toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`;
  if (readRecord(root, id)) return { ok: false, status: 409, message: 'a hyperopt of this model was started this very second - try again' };
  const rec = { id, tuner: 'deeplearning', ...p, row: MODELS[p.model], budget: budgetOf(p), by: String(user || 'admin').slice(0, 64),
    requestedAt: now.toISOString(), status: 'queued', worker: 'gpu' };
  writeJson(fileOf(root, id), rec);
  writeJson(fileOf(root, id, 'served'), readJson(path.join(root, `bestParams_${p.game}.json`)) || {});
  const queued = lane.queue(id, rec.by, p.days, { hyperopt: { budget: rec.budget, args: argsOf(rec) } });
  if (!queued.ok) {
    ['', 'served'].forEach((k) => { try { fs.unlinkSync(fileOf(root, id, k)); } catch (e) { /* gone */ } });
    return queued;
  }
  return { ok: true, status: 200, id,
    message: `${MODELS[p.model].replace(/ Model$/, '')} on ${p.game}, ${p.trials} trials of ${p.days} draws${p.fresh ? ', fresh' : ''}: ${queued.message}`,
    detail: `${p.game} ${p.model} t${p.trials} d${p.days} s${p.trainSeconds} gate ${p.gate ? p.gateSeMargin : 'off'}${p.fresh ? ' fresh' : ''}` };
}

function cancel(root, id, user, workers) {
  const rec = readRecord(root, id);
  if (!rec) return { ok: false, status: 404, message: 'no such hyperopt' };
  const lane = workers && typeof workers.lane === 'function' ? workers.lane('gpu') : null;
  if (!lane) return { ok: false, status: 503, message: 'no GPU worker is available here' };
  const ws = lane.backtestState();
  if (ws.current && ws.current.id === id) {
    const r = lane.stopRun(ws.current.runId, user);
    return r.ok ? { ...r, message: 'Hyperopt stopping: its finished trials stay in the study' } : r;
  }
  if (rec.status !== 'queued') return { ok: false, status: 409, message: `that hyperopt is ${rec.status}, not queued` };
  const r = lane.cancel(id, user);
  writeJson(fileOf(root, id), { ...rec, status: 'cancelled', finishedAt: new Date().toISOString(), error: `cancelled by ${String(user || 'admin').slice(0, 64)}` });
  return r.ok ? { ...r, message: 'Hyperopt cancelled' } : { ok: true, status: 200, message: 'Hyperopt cancelled' };
}

// a job holds process.lock (a live PID in it): a tuner or the predictor may be writing bestParams_<game>.json
function lockHolder(root) {
  let pid;
  try { pid = Number(String(fs.readFileSync(path.join(root, 'process.lock'), 'utf-8')).trim()); } catch (e) { return null; }
  if (!Number.isInteger(pid) || pid <= 0) return null;
  try { process.kill(pid, 0); return pid; } catch (e) { return e.code === 'EPERM' ? pid : null; }
}

// Apply: body {what: 'gate' | 'challenger', anyway: bool}. {ok, status, message}
function apply(root, id, body, user, now = new Date(), deps = {}) {
  const rec = readRecord(root, id);
  if (!rec) return { ok: false, status: 404, message: 'no such hyperopt' };
  if (rec.applied) return { ok: false, status: 409, message: `already applied by ${rec.applied.by} on ${rec.applied.at}` };
  const result = readJson(fileOf(root, id, 'result'));
  const study = result && result.studies ? result.studies[studyKey(rec.game, rec.model)] : null;
  if (!study || study.status !== 'done') return { ok: false, status: 409, message: 'the study has not finished: there is nothing to apply yet' };
  const what = (body || {}).what === 'challenger' ? 'challenger' : 'gate';
  const params = what === 'gate' ? study.params : study.challengerParams;
  if (!params || typeof params !== 'object' || !Object.keys(params).length) {
    return { ok: false, status: 409, message: what === 'gate' ? `the gate kept what is served (${study.reason || 'no reason given'}): nothing to write - "Apply this run's best" writes the best trial against it`
      : 'this run has no scored trial to apply' };
  }
  if (what === 'challenger' && study.decision === 'replaced') return { ok: false, status: 409, message: 'the gate already chose this run\'s best: apply the gate\'s result' };
  const holder = (deps.lockHolder || lockHolder)(root);
  if (holder) return { ok: false, status: 409, message: `a job is running (process.lock, PID ${holder}) and may be writing bestParams_${rec.game}.json - apply when it has finished` };
  const file = path.join(root, `bestParams_${rec.game}.json`);
  const served = readJson(file);
  if (!served || typeof served !== 'object') return { ok: false, status: 500, message: `bestParams_${rec.game}.json does not read` };
  const before = readJson(fileOf(root, id, 'served')) || {};
  const changed = Object.keys(params).filter((k) => JSON.stringify(served[k]) !== JSON.stringify(before[k]));
  if (changed.length && !(body || {}).anyway) {
    return { ok: false, status: 409, changed, message: `${changed.length} of these values changed since the run started (${changed.slice(0, 4).join(', ')}${changed.length > 4 ? ', ...' : ''}): the gate compared against the old ones - apply anyway?` };
  }
  const by = String(user || 'admin').slice(0, 64);
  const at = now.toISOString();
  const record = { ...(study.record || {}) };
  if (what === 'challenger') {
    record.reason = `applied by hand by ${by} against the gate's verdict (${study.decision}: ${study.reason || record.reason || ''})`;
    record.decision = 'replaced';
  }
  record.applied = { by, at, what, hyperopt: id, ...(changed.length ? { changedSince: changed } : {}) };
  const next = { ...served, ...params, tuningGate: { ...(served.tuningGate || {}), [rec.row]: record } };
  writeJson(file, next);
  writeJson(fileOf(root, id), { ...rec, applied: { by, at, what, keys: Object.keys(params).length } });
  return { ok: true, status: 200,
    message: `${Object.keys(params).length} values of ${rec.row} written to bestParams_${rec.game}.json (${what === 'gate' ? `the gate's ${study.decision}` : "this run's best trial"}) - the next run serves them and its commit includes the file`,
    detail: `${id} ${what}${changed.length ? ` anyway (${changed.length} changed)` : ''}` };
}

// The run's log, its last `bytes` (64 KB by default)
function logTail(root, id, bytes = 64 * 1024) {
  if (!ID.test(String(id || ''))) return null;
  const file = path.join(root, LOG_DIR, `${id}.log`);
  let fd;
  try {
    fd = fs.openSync(file, 'r');
    const size = fs.fstatSync(fd).size;
    const n = Math.min(size, bytes);
    const buf = Buffer.alloc(n);
    fs.readSync(fd, buf, 0, n, size - n);
    // progress bars rewrite one line with carriage returns (Keras also with backspaces and colour codes): drop the
    // codes and the backspaces, keep the last state of each line
    const text = buf.toString('utf-8').replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '').replace(/\x08/g, '').split('\n').map((l) => l.split('\r').filter(Boolean).pop() || '').join('\n');
    return { size, text, truncated: size > n };
  } catch (e) { return { size: 0, text: '', truncated: false }; } finally { if (fd !== undefined) fs.closeSync(fd); }
}

function install(app, { root, workers = null, guard, jsonBody, csrf, safe, json, reply, audit }) {
  const laneState = () => (workers && typeof workers.lane === 'function' ? workers.lane('gpu').backtestState() : null);
  app.get('/api/admin/hyperopts', guard, safe((req, res) => json(res, list(root, laneState()))));
  app.get('/api/admin/hyperopts/:id', guard, safe((req, res) => {
    const rec = readRecord(root, req.params.id);
    if (!rec) return json(res, { error: 'no such hyperopt' }, 404);
    return json(res, viewOf(root, rec, laneState(), { full: true }));
  }));
  app.get('/api/admin/hyperopts/:id/log', guard, safe((req, res) => {
    const rec = readRecord(root, req.params.id);
    if (!rec) return json(res, { error: 'no such hyperopt' }, 404);
    return json(res, logTail(root, rec.id));
  }));
  app.post('/api/admin/hyperopts/start', guard, jsonBody, csrf, safe((req, res) => {
    const r = start(root, req.body, req.user.name, workers);
    if (r.ok) audit.record('hyperopt.started', req, req.user.name, r.detail);
    reply(res, r, r.ok ? { id: r.id } : {});
  }));
  app.post('/api/admin/hyperopts/:id/cancel', guard, jsonBody, csrf, safe((req, res) => {
    const r = cancel(root, String(req.params.id || ''), req.user.name, workers);
    if (r.ok) audit.record('hyperopt.cancelled', req, req.user.name, String(req.params.id));
    reply(res, r);
  }));
  app.post('/api/admin/hyperopts/:id/apply', guard, jsonBody, csrf, safe((req, res) => {
    const r = apply(root, String(req.params.id || ''), req.body, req.user.name);
    if (r.ok) audit.record('hyperopt.applied', req, req.user.name, r.detail);
    reply(res, r);
  }));
}

module.exports = { DIR, GAMES, MODELS, FIELDS, STATUSES, budgetOf, argsOf, parse, studyKey, readRecord, list, viewOf, start, cancel, apply, logTail, lockHolder, install };
