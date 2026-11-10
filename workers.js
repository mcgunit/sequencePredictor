'use strict';
// The worker machines (owner, 8-10 Nov 2026; docs/operations.md, The workers). This server stays the only writer of
// data/: for each backtest it runs worker/remote-backtest.sh, which copies the code and the training data to the
// worker, runs RunBacktest.py there over ssh and copies the backtest's record back (progress every 15 s, then the
// result). Two lanes, each its own queue beside the scheduler's (jobs.js): a worker run takes no process.lock and
// leaves this server's CPU and GPU alone, so it neither waits for the scheduled jobs nor has to end before the next one.
//
//   gpu  the RTX 2080 Ti machine, an LXC that is always on and shared with other services: the deep-learning rows.
//   cpu  the HP DL360 (24 threads, 148 GB; about half this server's speed, measured 10 Nov 2026): the statistical and
//        boosting rows. Power hungry, so off unless there is work: a run switches it on through its iLO when it does
//        not answer (worker/power.sh), and the machine switches itself off after its idle minutes without a backtest,
//        a council request or a login (worker/idle-shutdown.sh, on the machine; only a boot the worker marked). The
//        Jobs & services page shows it and switches it on or off by hand (install below).
//
// One run at a time per lane; a run survives a server restart (setsid, adopted on start, as the scheduler's jobs).
// Stop sends SIGTERM to the script alone, which stops the remote run so it writes "stopped".
//
// The addresses live in config/workers.json (gitignored; never in the repository), for example
//   {"gpu": {"host": "root@<address>", "gpuMemoryMb": 6144},
//    "cpu": {"host": "root@<address>", "iloEnv": "/root/.config/sequencePredictor/ilo.env", "idleMinutes": 10}}
// "dir" (/opt/sequencePredictor-worker) and "python" (<dir>/venv/bin/python) are optional; "enabled": false switches
// a lane off. Without a gpu entry the deep-learning rows cannot be backtested; without a cpu entry the other rows run
// in the scheduler's queue on this server, as before.
const fs = require('fs');
const path = require('path');
const { spawn, execFile } = require('child_process');
const config = require('./config');

const ROOT = __dirname;
// the folders, set once (test/workers.test.js points them at a temporary folder)
let CONFIG_DIR, CONFIG_FILE, DISABLE_FILE, RUN_DIR, LOG_DIR, SCRIPT, POWER_SCRIPT;
function useDirs(configDir = config.CONFIG_DIR, logDir = path.join(ROOT, 'log'), script = path.join(ROOT, 'worker', 'remote-backtest.sh'),
  powerScript = path.join(ROOT, 'worker', 'power.sh')) {
  CONFIG_DIR = configDir;
  CONFIG_FILE = path.join(configDir, 'workers.json');
  DISABLE_FILE = path.join(configDir, 'scheduler.disabled');   // the scheduler's kill switch stops the workers too
  RUN_DIR = path.join(configDir, 'runs');
  LOG_DIR = logDir;
  SCRIPT = script;
  POWER_SCRIPT = powerScript;
  Object.values(LANES || {}).forEach((l) => { l.state = null; });
}
let LANES = null;
const TICK_MS = 10 * 1000;
const PID_WAIT_MS = 30 * 1000;
const GONE_GRACE_MS = 5 * 1000;
const HISTORY_KEPT = 40;
const LOG_MAX_BYTES = 10 * 1024 * 1024;
const POWER_FRESH_MS = 30 * 1000;
const DEFAULT_DIR = '/opt/sequencePredictor-worker';
const safePath = (v) => typeof v === 'string' && /^\/[A-Za-z0-9_./-]{1,200}$/.test(v);

const KINDS = {
  gpu: {
    word: 'GPU worker', stateFile: 'workers-state.json', log: 'backtest-gpu.log',
    // a quarter of an hour to start (copy, TensorFlow, CUDA) and a quarter of an hour a draw (one retrain capped at
    // 240 s, its learning check, the fresh child) - 3 draws 1 h, 30 draws 7 h 45 min
    budget: (days) => 15 * 60 + Math.max(1, Number(days) || 1) * 15 * 60,
    settings: (g) => ({ gpuMemoryMb: Number.isInteger(g.gpuMemoryMb) && g.gpuMemoryMb >= 512 && g.gpuMemoryMb <= 65536 ? g.gpuMemoryMb : 6144 }),
    remoteEnv: (c) => `CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=false SP_GPU_MEMORY_MB=${c.gpuMemoryMb}`,
  },
  cpu: {
    word: 'CPU worker', stateFile: 'workers-cpu-state.json', log: 'backtest-cpu.log',
    // the statistical and boosting rows: no schedule to end before; 12 h at most (the DL360 is about half this
    // server's speed: a 90-draw boosting backtest takes about an hour here)
    budget: () => 12 * 3600,
    settings: (g) => ({
      iloEnv: safePath(g.iloEnv) ? g.iloEnv : null,
      idleMinutes: Number.isInteger(g.idleMinutes) && g.idleMinutes >= 5 && g.idleMinutes <= 240 ? g.idleMinutes : 10,
    }),
    remoteEnv: () => 'CUDA_VISIBLE_DEVICES=-1',
  },
};

// A lane's settings, or null when it is not set up (no file, unreadable, no host, "enabled": false).
function configOf(kind, file) {
  let raw;
  try { raw = JSON.parse(fs.readFileSync(file || CONFIG_FILE, 'utf-8')); } catch (e) { return null; }
  const g = raw && typeof raw[kind] === 'object' && raw[kind] ? raw[kind] : null;
  if (!g || !KINDS[kind] || g.enabled === false || typeof g.host !== 'string' || !/^[A-Za-z0-9_.@-]{1,120}$/.test(g.host)) return null;
  const dir = safePath(g.dir) ? g.dir : DEFAULT_DIR;
  return { kind, host: g.host, dir, python: safePath(g.python) ? g.python : `${dir}/venv/bin/python`, ...KINDS[kind].settings(g) };
}

function log(text) {
  const line = `${new Date().toISOString()} [workers] ${text}`;
  try { fs.mkdirSync(LOG_DIR, { recursive: true }); fs.appendFileSync(path.join(LOG_DIR, 'scheduler.log'), `${line}\n`); } catch (e) { /* a convenience */ }
  console.log(line);
}
function alive(pid, runId) {
  if (!pid) return false;
  try { return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf-8').includes(runId); } catch (e) { return false; }
}
const exitCodeFrom = (text) => (/^-?\d+$/.test(String(text).trim()) ? Number(String(text).trim()) : null);
const shellQuote = (v) => `'${String(v).replace(/'/g, "'\\''")}'`;
const EMPTY = { version: 1, queue: [], current: null, history: [] };
const baseEnv = () => ({
  PATH: process.env.PATH || '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
  HOME: process.env.HOME || '/root', LANG: process.env.LANG || 'C.UTF-8', TZ: process.env.TZ || 'UTC',
});
const scriptEnv = (c) => ({
  ...baseEnv(), WORKER_HOST: c.host, WORKER_DIR: c.dir, WORKER_PYTHON: c.python, WORKER_REMOTE_ENV: KINDS[c.kind].remoteEnv(c),
  ...(c.iloEnv ? { WORKER_ILO_ENV: c.iloEnv, WORKER_IDLE_MINUTES: String(c.idleMinutes) } : {}),
});

// --- one lane: its queue and its run ----------------------------------------------------------------------------------
function makeLane(kind) {
  const K = KINDS[kind];
  const lane = { kind, state: null };
  const file = () => path.join(CONFIG_DIR, K.stateFile);
  const ensure = () => {
    if (!lane.state) {
      try {
        const parsed = JSON.parse(fs.readFileSync(file(), 'utf-8'));
        lane.state = { ...EMPTY, ...(parsed && typeof parsed === 'object' ? parsed : {}) };
      } catch (e) { lane.state = JSON.parse(JSON.stringify(EMPTY)); }
    }
    return lane.state;
  };
  const save = () => {
    try {
      fs.mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
      fs.writeFileSync(`${file()}.tmp`, JSON.stringify(lane.state, null, 2), { mode: 0o600 });
      fs.renameSync(`${file()}.tmp`, file());
    } catch (e) { log(`could not write ${K.stateFile}: ${e.message}`); }
  };
  const record = (entry) => {
    const s = ensure();
    s.history.unshift(entry);
    s.history = s.history.slice(0, HISTORY_KEPT);
    save();
  };
  lane.config = () => configOf(kind);

  lane.launch = (entry, c, { spawnFn = spawn } = {}) => {
    const s = ensure();
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.mkdirSync(RUN_DIR, { recursive: true, mode: 0o700 });
    const logFile = path.join(LOG_DIR, K.log);
    try { if (fs.statSync(logFile).size > LOG_MAX_BYTES) fs.renameSync(logFile, `${logFile}.1`); } catch (e) { /* none yet */ }
    const runId = `${kind}-backtest-${new Date().toISOString().replace(/[:.]/g, '-')}`;
    const pidFile = path.join(RUN_DIR, `${runId}.pid`);
    const exitFile = path.join(RUN_DIR, `${runId}.exit`);
    const budget = K.budget(entry.days);
    // the wrapper publishes the script's PID (not its own: Stop signals the script alone, see worker/remote-backtest.sh)
    const script = [
      `cd ${shellQuote(ROOT)} || exit 97`,
      `printf '=== ${kind} worker: backtest %s started %s UTC ===\n' ${shellQuote(entry.id)} "$(date -u '+%F %T')" >> ${shellQuote(logFile)}`,
      `bash ${shellQuote(SCRIPT)} ${shellQuote(entry.id)} ${shellQuote(budget)} ${shellQuote(runId)} >> ${shellQuote(logFile)} 2>&1 &`,
      `echo $! > ${shellQuote(pidFile)}`,
      'wait $!',
      'status=$?',
      `printf '=== ${kind} worker: backtest %s finished status=%s ===\n' ${shellQuote(entry.id)} "$status" >> ${shellQuote(logFile)}`,
      `echo "$status" > ${shellQuote(exitFile)}`,
      `rm -f ${shellQuote(pidFile)}`,
    ].join('\n');
    try {
      const child = spawnFn('setsid', ['--fork', 'bash', '-c', script], { cwd: ROOT, env: scriptEnv(c), stdio: 'ignore', detached: true });
      if (child && child.unref) child.unref();
      if (child && child.on) child.on('error', (e) => log(`could not launch the backtest ${entry.id}: ${e.message}`));
    } catch (e) {
      log(`could not launch the backtest ${entry.id}: ${e.message}`);
      record({ id: entry.id, at: Date.now(), finishedAt: Date.now(), exit: null, note: `launch failed: ${e.message}` });
      return;
    }
    s.current = { id: entry.id, runId, by: entry.by, days: entry.days, budget, host: c.host, startedAt: Date.now(), pid: null, pidFile, exitFile, logFile };
    save();
    log(`started the backtest ${entry.id} on the ${K.word} ${c.host} (budget ${Math.round(budget / 60)} min)`);
  };

  const finish = (exit, note, endedAt) => {
    const s = ensure();
    const run = s.current;
    if (!run) return;
    const ended = endedAt || Date.now();
    s.current = null;
    record({ id: run.id, runId: run.runId, host: run.host, at: run.startedAt, finishedAt: ended, seconds: Math.round((ended - run.startedAt) / 1000), exit, note: note || null });
    log(`the backtest ${run.id} on the ${K.word} finished${exit === null ? ` (${note || 'outcome unknown'})` : ` with exit code ${exit}`}`);
    [run.pidFile, run.exitFile].forEach((f) => { try { fs.unlinkSync(f); } catch (e) { /* gone */ } });
  };

  lane.poll = () => {
    const s = ensure();
    const run = s.current;
    if (!run) return;
    let exit = null;
    let endedAt = null;
    try {
      exit = exitCodeFrom(fs.readFileSync(run.exitFile, 'utf-8'));
      if (exit !== null) endedAt = fs.statSync(run.exitFile).mtimeMs;
    } catch (e) { /* running */ }
    if (exit !== null) return finish(exit, null, endedAt);
    if (!run.pid) {
      try { run.pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null; if (run.pid) save(); } catch (e) { /* not yet */ }
      if (!run.pid && Date.now() - run.startedAt > PID_WAIT_MS) return finish(null, 'the run never reported a PID - launch failed');
      return;
    }
    // the script's command line carries the run id as its third argument. The wrapper writes the exit code a moment
    // after the script ends, so a gone script gets a few seconds before it counts as killed.
    if (alive(run.pid, run.runId)) { run.goneAt = null; return; }
    if (!run.goneAt) { run.goneAt = Date.now(); return; }
    if (Date.now() - run.goneAt >= GONE_GRACE_MS) finish(null, 'the run disappeared without writing an exit code (killed?)');
  };

  lane.startNext = (deps = {}) => {
    const s = ensure();
    if (s.current || !s.queue.length || fs.existsSync(DISABLE_FILE)) return;
    const c = (deps.config || lane.config)();
    if (!c) return;
    const entry = s.queue.shift();
    save();
    lane.launch(entry, c, deps);
  };

  lane.queue = (id, by, days, deps = {}) => {
    const s = ensure();
    if (!(deps.config || lane.config)()) return { ok: false, status: 503, message: `no ${K.word} is set up (config/workers.json)` };
    if (fs.existsSync(DISABLE_FILE)) return { ok: false, status: 409, message: 'the schedule is disabled - remove config/scheduler.disabled first' };
    if (s.queue.some((q) => q.id === id) || (s.current && s.current.id === id)) return { ok: false, status: 409, message: 'that backtest is already queued' };
    s.queue.push({ id, by: String(by || 'admin').slice(0, 64), days, queuedAt: Date.now() });
    save();
    log(`queued the backtest ${id} for the ${K.word} (started by hand (${by}))`);
    lane.startNext(deps);
    const started = Boolean(s.current && s.current.id === id);
    const place = s.queue.findIndex((q) => q.id === id) + 1;
    return { ok: true, status: 200, started, place,
      message: started ? `Backtest started on the ${K.word}` : `Backtest queued for the ${K.word}${place ? ` (number ${place})` : ''}: it starts when the one before it ends` };
  };

  lane.cancel = (id, by) => {
    const s = ensure();
    const before = s.queue.length;
    s.queue = s.queue.filter((q) => q.id !== id);
    if (before === s.queue.length) return { ok: false, status: 409, message: 'that backtest is not in the queue' };
    save();
    log(`the backtest ${id} removed from the ${K.word}'s queue by ${by}`);
    return { ok: true, status: 200, message: 'Backtest removed from the queue' };
  };

  lane.stopRun = (runId, by, { kill = (pid, signal) => process.kill(pid, signal), isAlive = alive } = {}) => {
    const run = ensure().current;
    if (!run || run.runId !== String(runId || '')) return { ok: false, status: 409, message: 'that backtest is no longer running' };
    if (!run.pid) return { ok: false, status: 409, message: 'that backtest has not reported its PID yet' };
    if (!isAlive(run.pid, run.runId)) return { ok: false, status: 409, message: 'that backtest is already gone - the page will catch up in a moment' };
    try {
      kill(run.pid, 'SIGTERM');   // the script alone: it stops the remote run itself (worker/remote-backtest.sh)
      log(`the backtest ${run.id} stopped by ${by}`);
      return { ok: true, status: 200, message: 'Backtest stopping' };
    } catch (e) {
      return { ok: false, status: 500, message: `could not stop it: ${e.message}` };
    }
  };

  // What backtests.js reads, in the scheduler's shape (backtestState): the running one, the queued ids, the worker.
  lane.backtestState = () => {
    const s = ensure();
    const c = lane.config();
    return {
      current: s.current ? { id: s.current.id, runId: s.current.runId } : null,
      queued: s.queue.map((q) => q.id),
      configured: Boolean(c), host: c ? c.host.replace(/^.*@/, '') : null,
      disabled: fs.existsSync(DISABLE_FILE),
      lastFinishedAt: s.history.length ? s.history[0].finishedAt || null : null,
    };
  };

  lane.adopt = () => {
    lane.state = null;
    const s = ensure();
    const run = s.current;
    if (!run) return;
    let pid = run.pid;
    if (!pid) { try { pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null; } catch (e) { /* none */ } run.pid = pid; }
    if (alive(pid, run.runId)) { log(`adopted the backtest ${run.id} on the ${K.word} (pid ${pid}) - it survived the restart`); return; }
    run.goneAt = null;   // poll reads an exit code written while the server was down; else it was lost with it
    lane.poll();
    if (s.current) finish(null, 'the server was restarted and the run was gone when it came back');
  };
  return lane;
}
LANES = { gpu: makeLane('gpu'), cpu: makeLane('cpu') };
useDirs();
const lane = (kind) => LANES[kind] || null;

// --- the CPU worker's power (worker/power.sh) ------------------------------------------------------------------------
let power = { at: 0, value: null, error: null, pending: null };
let powerAction = null;   // {action, by, at} while a switch-on waits for the machine
function runPower(args, c, timeoutMs, { execFn = execFile } = {}) {
  return new Promise((resolve) => {
    execFn('bash', [POWER_SCRIPT, ...args], { cwd: ROOT, env: scriptEnv(c), timeout: timeoutMs, encoding: 'utf-8' },
      (error, stdout, stderr) => resolve({ error, stdout: String(stdout || ''), stderr: String(stderr || '') }));
  });
}
// The DL360's state for the page, read at most every 30 s (an IPMI call and an ssh call): null when no cpu lane.
async function powerStatus(deps = {}) {
  const c = configOf('cpu');
  if (!c) return null;
  if (!c.iloEnv) return { configured: true, host: c.host.replace(/^.*@/, ''), power: 'unknown', reachable: null, note: 'no iLO settings (iloEnv): it is never switched on or off by this server' };
  if (Date.now() - power.at > POWER_FRESH_MS && !power.pending) {
    power.pending = runPower(['status'], c, 45 * 1000, deps).then((r) => {
      let value = null;
      try { value = JSON.parse(r.stdout.trim().split('\n').pop()); } catch (e) { value = null; }
      power = { at: Date.now(), value, error: value ? null : (r.stderr || (r.error && r.error.message) || 'no answer').slice(0, 200), pending: null };
    });
  }
  if (!power.value && power.pending) await Promise.race([power.pending, new Promise((r) => setTimeout(r, deps.waitMs ?? 8000))]);
  const v = power.value || {};
  const cpu = LANES.cpu.backtestState();
  return {
    configured: true, host: c.host.replace(/^.*@/, ''), idleMinutes: c.idleMinutes,
    power: v.power || 'unknown', reachable: v.reachable ?? null, woken: Boolean(v.woken),
    wokenAt: v.wokenAt ? v.wokenAt * 1000 : null, lastBusy: v.lastBusy ? v.lastBusy * 1000 : null,
    readAt: power.at || null, error: power.error, switching: powerAction,
    running: cpu.current ? cpu.current.id : null, queued: cpu.queued.length,
  };
}

// Switch on by hand: the same wake as a backtest's (the boot is marked, so it switches itself off when idle).
function powerOn(by, deps = {}) {
  const c = configOf('cpu');
  if (!c || !c.iloEnv) return { ok: false, status: 503, message: 'no CPU worker with iLO settings is set up (config/workers.json)' };
  if (powerAction) return { ok: false, status: 409, message: `already ${powerAction.action === 'on' ? 'switching on' : 'switching off'} (since ${new Date(powerAction.at).toISOString().slice(11, 16)} UTC)` };
  powerAction = { action: 'on', by, at: Date.now() };
  log(`the CPU worker switched on by ${by}`);
  runPower(['wake'], c, 20 * 60 * 1000, deps).then((r) => {
    const failed = r.error ? (r.stderr.trim().split('\n').pop() || r.error.message) : null;
    log(failed ? `switching the CPU worker on failed: ${failed}` : 'the CPU worker is up');
    powerAction = null;
    power.at = 0;
  });
  return { ok: true, status: 200, message: `Switching the DL360 on: it takes 5 to 10 minutes, and it switches itself off after ${c.idleMinutes} idle minutes` };
}

// Switch off by hand: refused while a backtest runs or waits for it (the council goes down with it).
function powerOff(by, deps = {}) {
  const c = configOf('cpu');
  if (!c || !c.iloEnv) return { ok: false, status: 503, message: 'no CPU worker with iLO settings is set up (config/workers.json)' };
  const s = LANES.cpu.backtestState();
  if (s.current || s.queued.length) return { ok: false, status: 409, message: 'a backtest is running or queued on the DL360 - stop or cancel it first' };
  if (powerAction) return { ok: false, status: 409, message: 'it is being switched on - wait until it is up' };
  powerAction = { action: 'off', by, at: Date.now() };
  log(`the CPU worker switched off by ${by}`);
  runPower(['off'], c, 60 * 1000, deps).then(() => { powerAction = null; power.at = 0; });
  return { ok: true, status: 200, message: 'Shutting the DL360 down: the council goes down with it' };
}

// --- start-up ---------------------------------------------------------------------------------------------------------
let timer = null;
function tick() {
  Object.values(LANES).forEach((l) => {
    try { l.poll(); l.startNext(); } catch (e) { log(`${l.kind} tick failed: ${e.stack || e.message}`); }
  });
}
function start() {
  Object.values(LANES).forEach((l) => { try { l.adopt(); } catch (e) { log(`${l.kind} adoption failed: ${e.message}`); } });
  tick();
  timer = setInterval(tick, TICK_MS);
  if (timer.unref) timer.unref();
}
function stop() {
  if (timer) { clearInterval(timer); timer = null; }
}

// The routes (administrator only; system.js passes its guard, form token, JSON body and audit log).
function install(app, { guard, jsonBody, csrf, safe, json, reply, audit }) {
  app.get('/api/admin/workers', guard, safe(async (req, res) => {
    const gpu = LANES.gpu.backtestState();
    json(res, {
      gpu: { configured: gpu.configured, host: gpu.host, running: gpu.current ? gpu.current.id : null, queued: gpu.queued.length },
      cpu: await powerStatus(),
    });
  }));
  app.post('/api/admin/workers/cpu/on', guard, jsonBody, csrf, safe((req, res) => {
    const r = powerOn(req.user.name);
    if (r.ok && audit) audit.record('worker.on', req, req.user.name, 'cpu');
    reply(res, r);
  }));
  app.post('/api/admin/workers/cpu/off', guard, jsonBody, csrf, safe((req, res) => {
    const r = powerOff(req.user.name);
    if (r.ok && audit) audit.record('worker.off', req, req.user.name, 'cpu');
    reply(res, r);
  }));
}

module.exports = {
  start, stop, lane, configOf, install, powerStatus, powerOn, powerOff, KINDS,
  _internals: { useDirs, tick, lanes: () => LANES, resetPower: () => { power = { at: 0, value: null, error: null, pending: null }; powerAction = null; } },
};
