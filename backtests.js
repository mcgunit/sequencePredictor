'use strict';
// Backtests started from the admin pages (owner's roadmap step 4, 8 Nov 2026): one model on one game, walked forward
// over the newest draws with the weekend tuner's own score (RunBacktest.py), run by the scheduler as its "backtest"
// job (jobs.js: behind every scheduled job, and only when it can end before the next scheduled run).
//
// One record per backtest in config/backtests/<id>.json (gitignored; data/ is committed by the pipeline): the server
// writes the request (status "queued"), RunBacktest.py rewrites it with its progress and its result. Admins start
// them freely (owner: researchers are admins).
//
// The deep-learning rows (GPU_ROWS, since 10 Nov 2026) run on the GPU worker instead (workers.js: its own queue, no
// process.lock, not tied to the schedule), 3 draws by default and 30 at most (owner): each draw is a retrain.
const fs = require('fs');
const path = require('path');

const DIR = path.join('config', 'backtests');
const KEPT = 100;                // the newest records the list shows
const DAYS_MAX = 1000;
const DAYS_DEFAULT = 90;         // the weekend tuners' window (HyperoptStatistics/HyperoptBoost DEFAULT_DAYS)
const VALUES = ['served', 'tuned'];

// What RunBacktest.py can backtest: {row: [games]} - the same table as its covered(), checked by test/backtests.test.js.
const LOTTERY = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
const MARKETS = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];
const POSITIONAL = ['pick3', 'jokerplus'];
const NOT_POSITIONAL = ['MarkovBayesian Model', 'MarkovBayesianEnhanched Model', 'PoissonMarkov Model', 'HybridStatisticalModel',
  'XGBoostMultiLabel Model', 'LightGBMMultiLabel Model', 'CatBoostMultiLabel Model'];
const MARKET_ONLY = ['GARCH Model', 'Regime HMM Model'];
const MARKET_ROWS = ['Markov Model', 'MarkovMonteCarlo Model', 'PoissonMonteCarlo Model', 'LaplaceMonteCarlo Model', ...MARKET_ONLY];
const STATISTICS = ['Markov Model', 'MarkovMonteCarlo Model', 'MarkovBayesian Model', 'MarkovBayesianEnhanched Model', 'PoissonMonteCarlo Model',
  'PoissonMarkov Model', 'LaplaceMonteCarlo Model', 'HybridStatisticalModel', 'GARCH Model', 'Regime HMM Model'];
// the deep-learning rows (RunBacktest.GPU_ROWS), lottery games only, on the GPU worker
const GPU_ROWS = ['LSTM Base Model', 'TCN Base Model', 'UnifiedLstmTcn Model', 'UnifiedLstmGruTcn Model', 'Transformer Model', 'GNN Model', 'Autoencoder Model'];
const GPU_DAYS_DEFAULT = 3;      // the deep-learning gate's window (src/DeepLearningTuning.DAYS)
const GPU_DAYS_MAX = 30;
const BOOSTING = ['XGBoost Model', 'XGBoostMultiLabel Model', 'LightGBM Model', 'LightGBMMultiLabel Model', 'CatBoost Model', 'CatBoostMultiLabel Model'];
const gamesFor = (row) => (GPU_ROWS.includes(row) ? [...LOTTERY] : MARKET_ONLY.includes(row) ? [...MARKETS]
  : [...LOTTERY.filter((g) => !(POSITIONAL.includes(g) && NOT_POSITIONAL.includes(row))), ...(MARKET_ROWS.includes(row) ? MARKETS : [])]);
const ROWS = Object.fromEntries([...STATISTICS, ...BOOSTING, ...GPU_ROWS].map((row) => [row, gamesFor(row)]));
const canBacktest = (game, row) => Boolean(ROWS[row] && ROWS[row].includes(game));
const onGpu = (row) => GPU_ROWS.includes(row);
// how many draws a backtest of the row walks: {daysDefault, daysMax, gpu}
const limitsFor = (row) => (onGpu(row) ? { daysDefault: GPU_DAYS_DEFAULT, daysMax: GPU_DAYS_MAX, gpu: true } : { daysDefault: DAYS_DEFAULT, daysMax: DAYS_MAX, gpu: false });
const LIMITS = Object.fromEntries(Object.keys(ROWS).map((row) => [row, limitsFor(row)]));

const ID = /^[A-Za-z0-9_-]{1,80}$/;
const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const fileOf = (root, id) => path.join(root, DIR, `${id}.json`);

function readRecord(root, id) {
  if (!ID.test(String(id || ''))) return null;
  try { return JSON.parse(fs.readFileSync(fileOf(root, id), 'utf-8')); } catch (e) { return null; }
}
function writeRecord(root, rec) {
  fs.mkdirSync(path.join(root, DIR), { recursive: true });
  const file = fileOf(root, rec.id);
  fs.writeFileSync(`${file}.tmp`, `${JSON.stringify(rec, null, 1)}\n`);
  fs.renameSync(`${file}.tmp`, file);
}
function allRecords(root) {
  let names = [];
  try { names = fs.readdirSync(path.join(root, DIR)).filter((n) => n.endsWith('.json')); } catch (e) { return []; }
  return names.map((n) => readRecord(root, n.slice(0, -5))).filter((r) => r && r.id)
    .sort((a, b) => String(b.requestedAt || '').localeCompare(String(a.requestedAt || '')));
}

// The record as the pages read it, with what the scheduler knows: a queued record's place, and a record that says
// "running" or "queued" while the scheduler no longer has it (the server was restarted mid-run, the process was killed)
// shown as ended without a result rather than running forever.
// `js` is the state of the queue the record's row runs in (the scheduler's, or the GPU worker's for GPU_ROWS).
function viewOf(rec, js, { full = false } = {}) {
  const out = { ...rec, worker: onGpu(rec.row) ? 'gpu' : 'server' };
  if (!full && out.result) {
    const { days, values, ...summary } = out.result;
    out.result = { ...summary, dayCount: Array.isArray(days) ? days.length : 0 };
  }
  if (js) {
    const place = js.queued.indexOf(rec.id);
    const running = js.current && js.current.id === rec.id;
    if (rec.status === 'queued' && place >= 0) out.place = place + 1;
    if (running) {
      out.runId = js.current.runId;
      if (rec.status === 'queued') out.status = 'running';   // started, still importing: no progress written yet
    }
    if ((rec.status === 'queued' && place < 0 && !running) || (rec.status === 'running' && !running)) {
      out.status = 'ended';
      out.error = rec.status === 'running'
        ? `the backtest ended without writing a result (killed, or the connection was lost) - see log/${onGpu(rec.row) ? 'backtest-gpu.log' : 'backtest.log'}`
        : onGpu(rec.row) ? 'no longer in the GPU worker\'s queue' : 'no longer in the queue (removed by hand on the Jobs page?)';
    }
  }
  return out;
}

// ws: the GPU worker's state (workers.backtestState), null without one
function list(root, js, ws = null) {
  const recs = allRecords(root).slice(0, KEPT).map((r) => viewOf(r, onGpu(r.row) ? ws : js));
  return {
    backtests: recs, rows: ROWS, limits: LIMITS, daysDefault: DAYS_DEFAULT, daysMax: DAYS_MAX,
    // when a backtest queued now could start: budget 0 = after the next scheduled run
    scheduler: js ? { budget: js.budget, nextSlotSeconds: js.nextSlotSeconds, running: js.current ? js.current.id : null, queued: js.queued.length } : null,
    gpu: ws ? { configured: Boolean(ws.configured), host: ws.host || null, running: ws.current ? ws.current.id : null, queued: ws.queued.length } : { configured: false, host: null, running: null, queued: 0 },
  };
}

// Starts one: validates, writes the record, queues the scheduler's job (or the GPU worker's). {ok, status, message, id}
function start(root, body, user, jobs, now = new Date(), workers = null) {
  const b = body || {};
  const game = String(b.game || '');
  const row = String(b.row || '');
  if (!ROWS[row]) return { ok: false, status: 400, message: `${row || 'that row'} cannot be backtested here yet: the statistical, market, boosting and deep-learning rows can; the quantum, ensemble, meta-learner and RL rows come later` };
  if (!canBacktest(game, row)) return { ok: false, status: 400, message: `${game || 'that game'} does not run ${row}${onGpu(row) ? ' (the deep-learning rows are backtested on the lottery games)' : ''}` };
  const lim = limitsFor(row);
  const days = b.days === undefined || b.days === null || b.days === '' ? lim.daysDefault : Number(b.days);
  if (!Number.isInteger(days) || days < 1 || days > lim.daysMax) return { ok: false, status: 400, message: `days is a whole number from 1 to ${lim.daysMax}${lim.gpu ? ' for a deep-learning row (each draw is a retrain)' : ''}` };
  const values = b.values === undefined ? 'served' : String(b.values);
  if (!VALUES.includes(values)) return { ok: false, status: 400, message: 'values is served (with the manual values) or tuned' };
  const lane = lim.gpu ? workers : jobs;
  if (lim.gpu && (!workers || typeof workers.queue !== 'function')) return { ok: false, status: 503, message: 'no GPU worker is available here' };
  if (!lim.gpu && (!jobs || typeof jobs.queueBacktest !== 'function')) return { ok: false, status: 503, message: 'the schedule is not available here' };
  const id = `${game}-${slug(row)}-${now.toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`;
  if (readRecord(root, id)) return { ok: false, status: 409, message: 'a backtest of this row was started this very second - try again' };
  const rec = { id, game, row, days, values, by: String(user || 'admin').slice(0, 64), requestedAt: now.toISOString(), status: 'queued' };
  writeRecord(root, rec);
  const queued = lim.gpu ? lane.queue(id, rec.by, days) : lane.queueBacktest(id, rec.by);
  if (!queued.ok) {
    try { fs.unlinkSync(fileOf(root, id)); } catch (e) { /* gone */ }
    return queued;
  }
  return { ok: true, status: 200, id, message: `${row.replace(/ Model$/, '')} on ${game}, ${days} draws (${values} values): ${queued.message}`,
    detail: `${game} ${row} ${days} ${values}${lim.gpu ? ' gpu' : ''}` };
}

// Cancels a queued backtest or stops the running one. {ok, status, message}
function cancel(root, id, user, jobs, workers = null) {
  const rec = readRecord(root, id);
  if (!rec) return { ok: false, status: 404, message: 'no such backtest' };
  const gpu = onGpu(rec.row);
  const lane = gpu ? workers : jobs;
  if (!lane) return { ok: false, status: 503, message: gpu ? 'no GPU worker is available here' : 'the schedule is not available here' };
  const js = lane.backtestState();
  if (js.current && js.current.id === id) {
    const r = lane.stopRun(js.current.runId, user);
    return r.ok ? { ...r, message: 'Backtest stopping' } : r;
  }
  if (rec.status !== 'queued') return { ok: false, status: 409, message: `that backtest is ${rec.status}, not queued` };
  const r = gpu ? lane.cancel(id, user) : lane.cancelBacktest(id, user);
  writeRecord(root, { ...rec, status: 'cancelled', finishedAt: new Date().toISOString(), error: `cancelled by ${String(user || 'admin').slice(0, 64)}` });
  return r.ok ? r : { ok: true, status: 200, message: 'Backtest cancelled' };
}

// The routes (administrator only; system.js passes its guard, form token, JSON body and audit log).
function install(app, { root, jobs, workers = null, guard, jsonBody, csrf, safe, json, reply, audit }) {
  const state = () => (jobs && typeof jobs.backtestState === 'function' ? jobs.backtestState() : null);
  const gpuState = () => (workers && typeof workers.backtestState === 'function' ? workers.backtestState() : null);
  app.get('/api/admin/backtests', guard, safe((req, res) => json(res, list(root, state(), gpuState()))));
  app.get('/api/admin/backtests/:id', guard, safe((req, res) => {
    const rec = readRecord(root, req.params.id);
    if (!rec) return json(res, { error: 'no such backtest' }, 404);
    return json(res, viewOf(rec, onGpu(rec.row) ? gpuState() : state(), { full: true }));
  }));
  app.post('/api/admin/backtests/start', guard, jsonBody, csrf, safe((req, res) => {
    const r = start(root, req.body, req.user.name, jobs, new Date(), workers);
    if (r.ok) audit.record('backtest.started', req, req.user.name, r.detail);
    reply(res, r, r.ok ? { id: r.id } : {});
  }));
  app.post('/api/admin/backtests/:id/cancel', guard, jsonBody, csrf, safe((req, res) => {
    const r = cancel(root, String(req.params.id || ''), req.user.name, jobs && typeof jobs.backtestState === 'function' ? jobs : null, workers);
    if (r.ok) audit.record('backtest.cancelled', req, req.user.name, String(req.params.id));
    return reply(res, r);
  }));
}

module.exports = { DIR, ROWS, GPU_ROWS, LIMITS, DAYS_DEFAULT, DAYS_MAX, GPU_DAYS_DEFAULT, GPU_DAYS_MAX, canBacktest, onGpu, limitsFor, readRecord, writeRecord, list, start, cancel, viewOf, install };
