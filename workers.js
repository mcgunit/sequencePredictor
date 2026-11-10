'use strict';
// The GPU worker (owner, 8-10 Nov 2026; docs/workers.md): the backtests of the deep-learning rows run on the RTX 2080
// Ti machine, an LXC that is always on and shared with other services. This server stays the only writer of data/:
// for each backtest it runs worker/gpu-backtest.sh, which copies the code and the training data there, runs
// RunBacktest.py over ssh and copies the backtest's record back (progress every 15 s, then the result).
//
// Its own queue, beside the scheduler's (jobs.js): a worker run takes no process.lock and leaves the production GPU
// alone, so it neither waits for the scheduled jobs nor has to end before the next one. One run at a time; a run
// survives a server restart (setsid, adopted on start, as the scheduler's jobs). Stop sends SIGTERM to the script
// alone, which stops the remote run so it writes "stopped".
//
// The worker's address lives in config/workers.json (gitignored; never in the repository), for example
//   {"gpu": {"host": "root@<address>", "dir": "/opt/sequencePredictor-worker", "gpuMemoryMb": 6144}}
// Without the file (or with "enabled": false) the deep-learning rows answer that no GPU worker is set up.
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const config = require('./config');

const ROOT = __dirname;
// the folders, set once (test/workers.test.js points them at a temporary folder)
let CONFIG_DIR, CONFIG_FILE, STATE_FILE, DISABLE_FILE, RUN_DIR, LOG_DIR, SCRIPT;
function useDirs(configDir = config.CONFIG_DIR, logDir = path.join(ROOT, 'log'), script = path.join(ROOT, 'worker', 'gpu-backtest.sh')) {
  SCRIPT = script;
  CONFIG_DIR = configDir;
  CONFIG_FILE = path.join(configDir, 'workers.json');
  STATE_FILE = path.join(configDir, 'workers-state.json');
  DISABLE_FILE = path.join(configDir, 'scheduler.disabled');   // the scheduler's kill switch stops this too
  RUN_DIR = path.join(configDir, 'runs');
  LOG_DIR = logDir;
}
useDirs();
const LOG_NAME = 'backtest-gpu.log';
const TICK_MS = 10 * 1000;
const PID_WAIT_MS = 30 * 1000;
const GONE_GRACE_MS = 5 * 1000;
const HISTORY_KEPT = 40;
const LOG_MAX_BYTES = 10 * 1024 * 1024;
// A deep-learning backtest's budget: a quarter of an hour to start (copy, TensorFlow, CUDA) and a quarter of an hour
// a draw (one retrain capped at 240 s, its learning check, the fresh child) - 3 draws 1 h, 30 draws 7 h 45 min.
const BASE_SECONDS = 15 * 60;
const PER_DRAW_SECONDS = 15 * 60;
const DEFAULTS = { dir: '/opt/sequencePredictor-worker', gpuMemoryMb: 6144 };

const budgetFor = (days) => BASE_SECONDS + Math.max(1, Number(days) || 1) * PER_DRAW_SECONDS;

// The GPU worker's settings, or null when none is set up (no file, unreadable, no host, "enabled": false).
function gpuConfig(file) {
  file = file || CONFIG_FILE;
  let raw;
  try { raw = JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return null; }
  const g = raw && typeof raw.gpu === 'object' && raw.gpu ? raw.gpu : null;
  if (!g || g.enabled === false || typeof g.host !== 'string' || !/^[A-Za-z0-9_.@-]{1,120}$/.test(g.host)) return null;
  const dir = typeof g.dir === 'string' && /^\/[A-Za-z0-9_./-]{1,200}$/.test(g.dir) ? g.dir : DEFAULTS.dir;
  const python = typeof g.python === 'string' && /^\/[A-Za-z0-9_./-]{1,200}$/.test(g.python) ? g.python : `${dir}/venv/bin/python`;
  const mb = Number.isInteger(g.gpuMemoryMb) && g.gpuMemoryMb >= 512 && g.gpuMemoryMb <= 65536 ? g.gpuMemoryMb : DEFAULTS.gpuMemoryMb;
  return { host: g.host, dir, python, gpuMemoryMb: mb };
}

// --- state ----------------------------------------------------------------------------------------------------------
const EMPTY = { version: 1, queue: [], current: null, history: [] };
let state = null;
let timer = null;

function log(text) {
  const line = `${new Date().toISOString()} [gpu worker] ${text}`;
  try { fs.mkdirSync(LOG_DIR, { recursive: true }); fs.appendFileSync(path.join(LOG_DIR, 'scheduler.log'), `${line}\n`); } catch (e) { /* a convenience */ }
  console.log(line);
}
function loadState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    return { ...EMPTY, ...(parsed && typeof parsed === 'object' ? parsed : {}) };
  } catch (e) {
    return JSON.parse(JSON.stringify(EMPTY));
  }
}
function saveState() {
  try {
    fs.mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
    fs.writeFileSync(`${STATE_FILE}.tmp`, JSON.stringify(state, null, 2), { mode: 0o600 });
    fs.renameSync(`${STATE_FILE}.tmp`, STATE_FILE);
  } catch (e) {
    log(`could not write workers-state.json: ${e.message}`);
  }
}
const ensure = () => { if (!state) state = loadState(); return state; };

function alive(pid, runId) {
  if (!pid) return false;
  try { return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf-8').includes(runId); } catch (e) { return false; }
}
const exitCodeFrom = (text) => (/^-?\d+$/.test(String(text).trim()) ? Number(String(text).trim()) : null);
const shellQuote = (v) => `'${String(v).replace(/'/g, "'\\''")}'`;

// --- launching ------------------------------------------------------------------------------------------------------
function launch(entry, cfg, { spawnFn = spawn } = {}) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  fs.mkdirSync(RUN_DIR, { recursive: true, mode: 0o700 });
  const logFile = path.join(LOG_DIR, LOG_NAME);
  try { if (fs.statSync(logFile).size > LOG_MAX_BYTES) fs.renameSync(logFile, `${logFile}.1`); } catch (e) { /* none yet */ }
  const runId = `gpu-backtest-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  const pidFile = path.join(RUN_DIR, `${runId}.pid`);
  const exitFile = path.join(RUN_DIR, `${runId}.exit`);
  const budget = budgetFor(entry.days);
  // the wrapper publishes the script's PID (not its own: Stop signals the script alone, see worker/gpu-backtest.sh)
  const script = [
    `cd ${shellQuote(ROOT)} || exit 97`,
    `printf '=== gpu worker: backtest %s started %s UTC ===\n' ${shellQuote(entry.id)} "$(date -u '+%F %T')" >> ${shellQuote(logFile)}`,
    `bash ${shellQuote(SCRIPT)} ${shellQuote(entry.id)} ${shellQuote(budget)} ${shellQuote(runId)} >> ${shellQuote(logFile)} 2>&1 &`,
    `echo $! > ${shellQuote(pidFile)}`,
    'wait $!',
    'status=$?',
    `printf '=== gpu worker: backtest %s finished status=%s ===\n' ${shellQuote(entry.id)} "$status" >> ${shellQuote(logFile)}`,
    `echo "$status" > ${shellQuote(exitFile)}`,
    `rm -f ${shellQuote(pidFile)}`,
  ].join('\n');
  const env = {
    PATH: process.env.PATH || '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    HOME: process.env.HOME || '/root', LANG: process.env.LANG || 'C.UTF-8', TZ: process.env.TZ || 'UTC',
    WORKER_HOST: cfg.host, WORKER_DIR: cfg.dir, WORKER_PYTHON: cfg.python, WORKER_GPU_MB: String(cfg.gpuMemoryMb),
  };
  try {
    const child = spawnFn('setsid', ['--fork', 'bash', '-c', script], { cwd: ROOT, env, stdio: 'ignore', detached: true });
    if (child && child.unref) child.unref();
    if (child && child.on) child.on('error', (e) => log(`could not launch the backtest ${entry.id}: ${e.message}`));
  } catch (e) {
    log(`could not launch the backtest ${entry.id}: ${e.message}`);
    record({ id: entry.id, at: Date.now(), finishedAt: Date.now(), exit: null, note: `launch failed: ${e.message}` });
    return;
  }
  state.current = { id: entry.id, runId, by: entry.by, days: entry.days, budget, host: cfg.host, startedAt: Date.now(), pid: null, pidFile, exitFile, logFile };
  saveState();
  log(`started the backtest ${entry.id} on ${cfg.host} (budget ${Math.round(budget / 60)} min)`);
}

function record(entry) {
  state.history.unshift(entry);
  state.history = state.history.slice(0, HISTORY_KEPT);
  saveState();
}

function finish(exit, note, endedAt) {
  const run = state.current;
  if (!run) return;
  const ended = endedAt || Date.now();
  record({ id: run.id, runId: run.runId, host: run.host, at: run.startedAt, finishedAt: ended, seconds: Math.round((ended - run.startedAt) / 1000), exit, note: note || null });
  log(`the backtest ${run.id} on ${run.host} finished${exit === null ? ` (${note || 'outcome unknown'})` : ` with exit code ${exit}`}`);
  [run.pidFile, run.exitFile].forEach((f) => { try { fs.unlinkSync(f); } catch (e) { /* gone */ } });
  state.current = null;
  saveState();
}

function poll() {
  const run = state.current;
  if (!run) return;
  let exit = null;
  let endedAt = null;
  try {
    exit = exitCodeFrom(fs.readFileSync(run.exitFile, 'utf-8'));
    if (exit !== null) endedAt = fs.statSync(run.exitFile).mtimeMs;
  } catch (e) { /* running */ }
  if (exit !== null) return finish(exit, null, endedAt);
  if (!run.pid) {
    try { run.pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null; if (run.pid) saveState(); } catch (e) { /* not yet */ }
    if (!run.pid && Date.now() - run.startedAt > PID_WAIT_MS) return finish(null, 'the run never reported a PID - launch failed');
    return;
  }
  // the script's command line carries the run id as its third argument. The wrapper writes the exit code a moment
  // after the script ends, so a gone script gets a few seconds before it counts as killed.
  if (alive(run.pid, run.runId)) { run.goneAt = null; return; }
  if (!run.goneAt) { run.goneAt = Date.now(); return; }
  if (Date.now() - run.goneAt >= GONE_GRACE_MS) finish(null, 'the run disappeared without writing an exit code (killed?)');
}

function startNext(deps = {}) {
  if (state.current || !state.queue.length) return;
  if (fs.existsSync(DISABLE_FILE)) return;
  const cfg = (deps.gpuConfig || gpuConfig)();
  if (!cfg) return;
  const entry = state.queue.shift();
  saveState();
  launch(entry, cfg, deps);
}

function tick() {
  try { poll(); startNext(); } catch (e) { log(`tick failed: ${e.stack || e.message}`); }
}

// --- the actions (backtests.js) -------------------------------------------------------------------------------------
function queue(id, by, days, deps = {}) {
  ensure();
  if (!(deps.gpuConfig || gpuConfig)()) return { ok: false, status: 503, message: 'no GPU worker is set up (config/workers.json): the deep-learning rows cannot be backtested on this server' };
  if (fs.existsSync(DISABLE_FILE)) return { ok: false, status: 409, message: 'the schedule is disabled - remove config/scheduler.disabled first' };
  if (state.queue.some((q) => q.id === id) || (state.current && state.current.id === id)) return { ok: false, status: 409, message: 'that backtest is already queued' };
  state.queue.push({ id, by: String(by || 'admin').slice(0, 64), days, queuedAt: Date.now() });
  saveState();
  log(`queued the backtest ${id} (started by hand (${by}))`);
  startNext(deps);
  const started = Boolean(state.current && state.current.id === id);
  const place = state.queue.findIndex((q) => q.id === id) + 1;
  return { ok: true, status: 200, started, place,
    message: started ? 'Backtest started on the GPU worker' : `Backtest queued for the GPU worker${place ? ` (number ${place})` : ''}: it starts when the one before it ends` };
}

function cancel(id, by) {
  ensure();
  const before = state.queue.length;
  state.queue = state.queue.filter((q) => q.id !== id);
  if (before === state.queue.length) return { ok: false, status: 409, message: 'that backtest is not in the queue' };
  saveState();
  log(`the backtest ${id} removed from the queue by ${by}`);
  return { ok: true, status: 200, message: 'Backtest removed from the queue' };
}

function stopRun(runId, by, { kill = (pid, signal) => process.kill(pid, signal), isAlive = alive } = {}) {
  ensure();
  const run = state.current;
  if (!run || run.runId !== String(runId || '')) return { ok: false, status: 409, message: 'that backtest is no longer running' };
  if (!run.pid) return { ok: false, status: 409, message: 'that backtest has not reported its PID yet' };
  if (!isAlive(run.pid, run.runId)) return { ok: false, status: 409, message: 'that backtest is already gone - the page will catch up in a moment' };
  try {
    kill(run.pid, 'SIGTERM');   // the script alone: it stops the remote run itself (worker/gpu-backtest.sh)
    log(`the backtest ${run.id} stopped by ${by}`);
    return { ok: true, status: 200, message: 'Backtest stopping' };
  } catch (e) {
    return { ok: false, status: 500, message: `could not stop it: ${e.message}` };
  }
}

// What backtests.js reads, in the scheduler's shape (backtestState): the running one, the queued ids, the worker.
function backtestState() {
  ensure();
  const cfg = gpuConfig();
  return {
    current: state.current ? { id: state.current.id, runId: state.current.runId } : null,
    queued: state.queue.map((q) => q.id),
    configured: Boolean(cfg), host: cfg ? cfg.host.replace(/^.*@/, '') : null,
    disabled: fs.existsSync(DISABLE_FILE),
  };
}

function start() {
  state = loadState();
  const run = state.current;
  if (run) {
    let pid = run.pid;
    if (!pid) { try { pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null; } catch (e) { /* none */ } run.pid = pid; }
    if (alive(pid, run.runId)) log(`adopted the backtest ${run.id} (pid ${pid}) - it survived the restart`);
    else {
      run.goneAt = null;   // poll reads an exit code written while the server was down; else it was lost with it
      poll();
      if (state.current) finish(null, 'the server was restarted and the run was gone when it came back');
    }
  }
  tick();
  timer = setInterval(tick, TICK_MS);
  if (timer.unref) timer.unref();
}
function stop() {
  if (timer) { clearInterval(timer); timer = null; }
}

module.exports = {
  start, stop, queue, cancel, stopRun, backtestState, gpuConfig, budgetFor,
  _internals: { state: () => ensure(), reset: () => { state = JSON.parse(JSON.stringify(EMPTY)); }, poll, startNext, launch, useDirs, tick },
};
