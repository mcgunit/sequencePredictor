// Supervises the long-running side processes the UI depends on, so they are
// not hand-started any more. Today that is the Council API (src/llmCouncil/
// api.py): the page was a dead link whenever nobody remembered to start it.
//
// This is deliberately the small half of README roadmap item 8. A *service*
// is cheap to restart and must simply always be up, so it is an ordinary
// child of this process: pm2 stops it with the server and it comes back when
// the server does. The scheduled pipeline jobs of item 8 are the opposite
// case - a 12-hour tuning run must survive a deploy - and will be launched
// through `setsid --fork` instead, by a module that can reuse the status,
// logging and admin-page shape established here.
//
// Nothing here is reachable without a login: install() mounts its routes
// after auth.middleware, and every action is admin-only, POST and CSRF
// checked.

const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

const auth = require('./auth');

const LOG_DIR = path.join(__dirname, 'log');
const LOG_MAX_BYTES = 10 * 1024 * 1024;   // rotate at 10MB, keep one old file
// Generous on purpose: the Council API's own /health probes the llama.cpp
// endpoints, which costs ~2 s while they are off (it caches for 15 s, so only
// the first call after that window is slow). A probe shorter than the service's
// answer would report a healthy service as unreachable.
const PROBE_TIMEOUT_MS = 5000;
const BACKOFF_START_MS = 2000;
const BACKOFF_MAX_MS = 5 * 60 * 1000;
// More than this many restarts inside the window means the service is broken
// rather than unlucky (a missing dependency, a bad config): stop restarting
// and say so on the page, instead of hammering it forever.
const CRASH_LIMIT = 5;
const CRASH_WINDOW_MS = 10 * 60 * 1000;

// The child gets an explicit, minimal environment. Inheriting this process's
// env would hand it whatever pm2 froze at boot - on this box that includes a
// VS Code session's GIT_ASKPASS and SSH_CONNECTION - which is exactly the
// kind of difference that makes a service behave one way under pm2 and
// another from a terminal.
function childEnv(extra) {
  const base = {
    PATH: process.env.PATH || '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    HOME: process.env.HOME || '/root',
    LANG: process.env.LANG || 'C.UTF-8',
    TZ: process.env.TZ || 'UTC',
    PYTHONUNBUFFERED: '1',                 // so the log file fills as it runs
  };
  return Object.assign(base, extra || {});
}

const COUNCIL_PORT = Number(process.env.COUNCIL_API_PORT || 8099);

const SERVICES = [
  {
    key: 'councilApi',
    name: 'Council API',
    description: 'Serves the Council page (src/llmCouncil/api.py). Stays up while the llama.cpp '
               + 'boxes are off and reports their state, so the page can say the council has to be summoned.',
    command: 'python3',
    args: ['api.py', '--config', 'config.json', '--port', String(COUNCIL_PORT)],
    cwd: path.join(__dirname, 'src', 'llmCouncil'),
    logFile: path.join(LOG_DIR, 'councilApi.log'),
    port: COUNCIL_PORT,
    healthPath: '/health',
    // Default on; COUNCIL_API_AUTOSTART=off leaves it to be started by hand.
    autostart: String(process.env.COUNCIL_API_AUTOSTART || 'on').toLowerCase() !== 'off',
    // What the admin page shows from that service's /health body.
    summarize: (body) => {
      const status = body && body.status;
      if (!status) return null;
      const down = (status.members || []).filter((m) => !m.ok).map((m) => m.name);
      if (status.head && !status.head.ok) down.push(status.head.name);
      return status.ready
        ? `models ready (${status.reachable}/${status.total} members)`
        : `models not answering (${down.join(', ') || 'none configured'}) - the page shows "summon the council"`;
    },
  },
];

const state = new Map();   // key -> runtime state

function runtime(service) {
  if (!state.has(service.key)) {
    state.set(service.key, {
      child: null, pid: null, status: 'stopped', since: null, restarts: 0,
      lastExit: null, lastError: null, crashes: [], backoff: BACKOFF_START_MS,
      stopping: false, timer: null, health: null, healthAt: 0,
    });
  }
  return state.get(service.key);
}

function serviceByKey(key) {
  return SERVICES.find((s) => s.key === key) || null;
}

// --- logging ---------------------------------------------------------------
function openLog(service) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  try {
    const stat = fs.statSync(service.logFile);
    if (stat.size > LOG_MAX_BYTES) fs.renameSync(service.logFile, `${service.logFile}.1`);
  } catch (e) { /* no log yet */ }
  return fs.openSync(service.logFile, 'a');
}

function logLine(service, text) {
  try {
    fs.appendFileSync(service.logFile, `${new Date().toISOString()} [supervisor] ${text}\n`);
  } catch (e) { /* the log is a convenience, never a failure path */ }
  console.log(`[${service.key}] ${text}`);
}

function tailLog(service, lines = 40, bytes = 16384) {
  try {
    const stat = fs.statSync(service.logFile);
    const start = Math.max(0, stat.size - bytes);
    const fd = fs.openSync(service.logFile, 'r');
    const buffer = Buffer.alloc(stat.size - start);
    fs.readSync(fd, buffer, 0, buffer.length, start);
    fs.closeSync(fd);
    const text = buffer.toString('utf-8');
    return text.split('\n').filter(Boolean).slice(-lines);
  } catch (e) {
    return [];
  }
}

// --- health ----------------------------------------------------------------
// Is something already answering on the port? Used before spawning, so a
// second checkout (or an API someone started by hand) is adopted and shown as
// "external" instead of being duplicated or crash-looping on EADDRINUSE.
function probe(service) {
  return new Promise((resolve) => {
    if (!service.port || !service.healthPath) return resolve(null);
    const req = http.get({ host: '127.0.0.1', port: service.port, path: service.healthPath, timeout: PROBE_TIMEOUT_MS },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          if (res.statusCode !== 200) return resolve(null);
          try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8'))); } catch (e) { resolve({}); }
        });
      });
    req.on('timeout', () => req.destroy());
    req.on('error', () => resolve(null));
  });
}

async function refreshHealth(service) {
  const info = runtime(service);
  info.health = await probe(service);
  info.healthAt = Date.now();
  return info.health;
}

// --- lifecycle -------------------------------------------------------------
async function start(service, { manual = false } = {}) {
  const info = runtime(service);
  if (info.child) return { ok: true, message: 'already running' };

  const existing = await probe(service);
  if (existing) {
    info.status = 'external';
    info.health = existing;
    info.healthAt = Date.now();
    logLine(service, `port ${service.port} already answers - adopting that instance, not starting a second one`);
    return { ok: true, message: `something already answers on port ${service.port} - adopted` };
  }

  let fd;
  try {
    fd = openLog(service);
  } catch (e) {
    info.status = 'failed';
    info.lastError = `cannot open ${service.logFile}: ${e.message}`;
    return { ok: false, message: info.lastError };
  }

  logLine(service, `starting: ${service.command} ${service.args.join(' ')} (cwd ${service.cwd})`);
  let child;
  try {
    child = spawn(service.command, service.args, {
      cwd: service.cwd,
      env: childEnv(service.env),
      stdio: ['ignore', fd, fd],
    });
  } catch (e) {
    fs.closeSync(fd);
    info.status = 'failed';
    info.lastError = e.message;
    logLine(service, `could not start: ${e.message}`);
    return { ok: false, message: e.message };
  }

  info.child = child;
  info.pid = child.pid;
  info.status = 'running';
  info.since = Date.now();
  info.stopping = false;
  info.lastError = null;
  if (manual) { info.crashes = []; info.backoff = BACKOFF_START_MS; }

  child.on('error', (error) => {
    info.lastError = error.message;
    logLine(service, `process error: ${error.message}`);
  });

  child.on('exit', (code, signal) => {
    try { fs.closeSync(fd); } catch (e) { /* already closed */ }
    info.child = null;
    info.pid = null;
    info.lastExit = { code, signal, at: Date.now() };
    // Any cached health belongs to the process that just died. Keeping it
    // would make a stopped service look like one started outside the server
    // (status() reads a positive health without a child as "external").
    info.health = null;
    info.healthAt = 0;
    const ranFor = info.since ? Math.round((Date.now() - info.since) / 1000) : 0;

    if (info.stopping) {
      info.status = 'stopped';
      logLine(service, `stopped after ${ranFor}s`);
      return;
    }

    const now = Date.now();
    info.crashes = info.crashes.filter((t) => now - t < CRASH_WINDOW_MS).concat(now);
    info.restarts += 1;
    logLine(service, `exited with ${signal ? `signal ${signal}` : `code ${code}`} after ${ranFor}s`);

    if (info.crashes.length >= CRASH_LIMIT) {
      info.status = 'failed';
      info.lastError = `${info.crashes.length} exits within ${Math.round(CRASH_WINDOW_MS / 60000)} minutes - not restarting again`;
      logLine(service, info.lastError);
      return;
    }

    // A service that ran for a while is unlucky, one that dies at once is
    // broken - so the wait only grows for the second case, and the first
    // retry is always the short one.
    const delay = ranFor > 60 ? BACKOFF_START_MS : info.backoff;
    info.backoff = ranFor > 60 ? BACKOFF_START_MS : Math.min(info.backoff * 2, BACKOFF_MAX_MS);
    info.status = 'restarting';
    logLine(service, `restarting in ${Math.round(delay / 1000)}s`);
    info.timer = setTimeout(() => { info.timer = null; start(service).catch(() => {}); }, delay);
    if (info.timer.unref) info.timer.unref();
  });

  return { ok: true, message: `started (pid ${child.pid})` };
}

function stop(service, { manual = true } = {}) {
  const info = runtime(service);
  if (info.timer) { clearTimeout(info.timer); info.timer = null; }
  if (!info.child) {
    info.status = info.status === 'external' ? 'external' : 'stopped';
    return { ok: true, message: info.status === 'external' ? 'not ours to stop (started outside the server)' : 'already stopped' };
  }
  info.stopping = manual;
  info.health = null;
  info.healthAt = 0;
  info.child.kill('SIGTERM');
  const child = info.child;
  const killer = setTimeout(() => { try { child.kill('SIGKILL'); } catch (e) { /* gone */ } }, 5000);
  if (killer.unref) killer.unref();
  return { ok: true, message: 'stopping' };
}

async function restart(service) {
  stop(service, { manual: true });
  await new Promise((resolve) => setTimeout(resolve, 600));
  const info = runtime(service);
  info.crashes = [];
  info.backoff = BACKOFF_START_MS;
  return start(service, { manual: true });
}

function startAll() {
  SERVICES.forEach((service) => {
    if (!service.autostart) {
      runtime(service).status = 'disabled';
      console.log(`[${service.key}] autostart is off`);
      return;
    }
    start(service).then((result) => console.log(`[${service.key}] ${result.message}`)).catch(() => {});
  });
}

function stopAll() {
  SERVICES.forEach((service) => {
    const info = runtime(service);
    if (info.timer) { clearTimeout(info.timer); info.timer = null; }
    if (info.child) {
      info.stopping = true;
      try { info.child.kill('SIGTERM'); } catch (e) { /* gone */ }
    }
  });
}

// --- status ----------------------------------------------------------------
async function status() {
  return Promise.all(SERVICES.map(async (service) => {
    const info = runtime(service);
    // A failed probe is never cached: the service may just have finished
    // starting, and probing a closed local port fails instantly anyway.
    if (!info.health || Date.now() - info.healthAt > 5000) await refreshHealth(service);
    // Something answers but we have no child: it was started outside.
    if (!info.child && info.health && info.status !== 'external' && info.status !== 'restarting') info.status = 'external';
    return {
      key: service.key,
      name: service.name,
      description: service.description,
      command: `${service.command} ${service.args.join(' ')}`,
      cwd: service.cwd,
      port: service.port,
      logFile: service.logFile,
      status: info.status,
      pid: info.pid,
      since: info.since,
      restarts: info.restarts,
      lastExit: info.lastExit,
      lastError: info.lastError,
      healthy: Boolean(info.health),
      summary: info.health && service.summarize ? service.summarize(info.health) : null,
      log: tailLog(service),
    };
  }));
}

// --- page ------------------------------------------------------------------
const BADGE = {
  running: ['#27ae60', 'running'],
  external: ['#2980b9', 'running (started outside the server)'],
  restarting: ['#f39c12', 'restarting'],
  stopped: ['#7f8c8d', 'stopped'],
  disabled: ['#7f8c8d', 'autostart off'],
  failed: ['#c0392b', 'failed'],
};

function since(at) {
  if (!at) return '-';
  const seconds = Math.round((Date.now() - at) / 1000);
  if (seconds < 90) return `${seconds}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

function page(req, render, services, message) {
  const esc = auth.escapeHtml;
  const csrf = `<input type="hidden" name="_csrf" value="${esc(req.user.csrf)}">`;
  const cards = services.map((s) => {
    const [colour, label] = BADGE[s.status] || ['#7f8c8d', s.status];
    const managed = s.status !== 'external';
    return `
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">${esc(s.name)}</span>
          <span class="card-meta" style="margin-left: 10px; color: ${colour}; font-weight: bold;">${esc(label)}</span>
          ${s.pid ? `<span class="card-meta" style="margin-left: 10px;">pid ${s.pid} · up ${since(s.since)}</span>` : ''}
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <p style="color: #7f8c8d; margin-top: 0;">${esc(s.description)}</p>
        <table style="min-width: 0; margin-bottom: 15px;">
          <tr><th style="text-align:left;">Command</th><td style="text-align:left;"><code>${esc(s.command)}</code></td></tr>
          <tr><th style="text-align:left;">Working directory</th><td style="text-align:left;"><code>${esc(s.cwd)}</code></td></tr>
          <tr><th style="text-align:left;">Answers on</th><td style="text-align:left;">127.0.0.1:${s.port} ${s.healthy ? '<span style="color:#27ae60;">yes</span>' : '<span style="color:#c0392b;">no</span>'}</td></tr>
          ${s.summary ? `<tr><th style="text-align:left;">Reported state</th><td style="text-align:left;">${esc(s.summary)}</td></tr>` : ''}
          <tr><th style="text-align:left;">Restarts</th><td style="text-align:left;">${s.restarts}${s.lastExit ? ` · last exit ${s.lastExit.signal ? `signal ${esc(s.lastExit.signal)}` : `code ${s.lastExit.code}`} ${since(s.lastExit.at)} ago` : ''}</td></tr>
          ${s.lastError ? `<tr><th style="text-align:left;">Last error</th><td style="text-align:left; color:#c0392b;">${esc(s.lastError)}</td></tr>` : ''}
          <tr><th style="text-align:left;">Log</th><td style="text-align:left;"><code>${esc(s.logFile)}</code></td></tr>
        </table>
        <form method="post" action="/admin/jobs/action" class="inline-form">${csrf}
          <input type="hidden" name="key" value="${esc(s.key)}">
          <button type="submit" name="action" value="start" class="nav-btn"${s.status === 'running' || s.status === 'external' ? ' disabled' : ''}>Start</button>
          <button type="submit" name="action" value="restart" class="nav-btn"${managed ? '' : ' disabled'}>Restart</button>
          <button type="submit" name="action" value="stop" class="nav-btn" style="background:#c0392b; border-color:#a93226;"${managed && s.pid ? '' : ' disabled'}>Stop</button>
        </form>
        ${s.status === 'external' ? '<p style="color:#7f8c8d;">This instance was started outside the server (by hand, or by the other checkout), so the buttons above cannot manage it.</p>' : ''}
        <pre style="background:#2c3e50; color:#ecf0f1; padding:12px; border-radius:6px; overflow-x:auto; max-height:260px; font-size:0.85em;">${esc(s.log.join('\n') || 'no log yet')}</pre>
      </div>
    </div>`;
  }).join('');

  return render.header('Jobs', req.user) + `
    <h1>Jobs</h1>
    ${message ? `<p style="color:#27ae60; font-weight:bold;">${esc(message)}</p>` : ''}
    <p style="color:#7f8c8d;">Services the web server keeps running. This page refreshes every 15 seconds.
       The scheduled pipeline jobs - the daily predictor and the weekly tuning chain - join this page with roadmap item 8; they are still started by cron today.</p>
    ${cards}
    <script>setTimeout(function () { location.reload(); }, 15000);</script>` + render.footer();
}

function install(app, render) {
  const adminOnly = auth.requireAdmin(render);

  app.get('/admin/jobs', adminOnly, async (req, res) => {
    res.send(page(req, render, await status(), req.query.msg ? String(req.query.msg).slice(0, 200) : null));
  });

  app.post('/admin/jobs/action', adminOnly, async (req, res) => {
    if (!auth.csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    const service = serviceByKey(String((req.body || {}).key || ''));
    const action = String((req.body || {}).action || '');
    if (!service) return res.redirect('/admin/jobs?msg=' + encodeURIComponent('unknown service'));
    let result;
    if (action === 'start') result = await start(service, { manual: true });
    else if (action === 'stop') result = stop(service);
    else if (action === 'restart') result = await restart(service);
    else result = { message: 'unknown action' };
    logLine(service, `${action} requested by ${req.user.name}: ${result.message}`);
    res.redirect('/admin/jobs?msg=' + encodeURIComponent(`${service.name}: ${result.message}`));
  });
}

module.exports = { install, startAll, stopAll, status, SERVICES };
