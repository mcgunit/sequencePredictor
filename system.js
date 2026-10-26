// system.js - the JSON behind the app's System pages (21 Oct 2026: the owner's decision to replace the old
// server-rendered pages with the React app under /app; admin plan 3.4 and 3.6). Jobs & services and People are for
// the administrator; Account and What's new for every signed-in reader.
//
//   GET  /api/admin/jobs                      the schedule: mode, lock, running job (elapsed vs usual), queue, every
//                                             job with its usual duration and whether it may run now, every kept
//                                             run, today's predictor, the last failure, the Council API
//   GET  /api/admin/jobs/run/:id/log          one run's log slice: its warning signatures first, then its last lines
//   POST /api/admin/jobs/run|cancel|stop      Run now, take out of the queue, stop the running job
//   POST /api/admin/services/:name/:action    start | stop | restart a supervised service
//   GET  /api/admin/people                    the accounts, the sign-in exceptions of the last 7 days, the activity
//   POST /api/admin/people/add|password|delete
//   GET  /api/account                         the reader's own account and recent activity
//   POST /api/account/password                the reader's own password, the current one required
//   GET  /api/whats-new                       the notes the reader's audience may see, seen or not, and the tour
//   POST /api/whats-new/seen|replay           mark notes seen (and the tour done); show the tour again
//
// The rules stay where they were (the old pages' forms called the same functions until cleanup batch 6b, 26 Oct
// 2026): which job may run, what cancel and stop do (jobs.js runRefusal, runNow, cancelQueued, stopRun);
// what a service allows (services.js act, actionsFor); accounts and passwords, with their audit entries (auth.js
// addUser, setUserPassword, deleteUser, changeOwnPassword); what a reader sees and has seen (whatsnew.js). The run
// lines and usual durations are This week's (admin.js runLine, usualOf), read from the same log slices.
//
// Every POST: the admin guard where the old route had one (admin.guard: a JSON 401 signed out, 403 for a user), then
// the session's form token (auth.csrfOk: _csrf in the JSON body - the token GET /api/me serves), and JSON errors only:
// 400 a request that breaks a rule or is not JSON, 403 a wrong or missing token, 404 nothing by that name, 409
// refused by the state it found, 500 the system refused. Times are ISO UTC instants; the pages show them in Brussels
// time (tz). No password hash or salt is ever sent, and a user never gets an administrator's view.
'use strict';

const fs = require('fs');
const path = require('path');
const admin = require('./admin');
const audit = require('./audit');
const auth = require('./auth');
const brussels = require('./brussels');
const whatsnew = require('./whatsnew');

const SERVICES_WAIT_MS = 6000;           // services.js probes a service's health for up to 5 s
const LOG_LINES = 200;                   // a run's log drawer: at most this many lines, warnings included
const LOG_TAIL_BYTES = 512 * 1024;       // read from the end of a run's slice, never the whole log (up to 10 MB)
const LINE_CHARS = 500;                  // one log line, cut
const LIVE_LOG_LINES = 20;               // the running job's tail (jobs.status)
const SERVICE_LOG_BYTES = 64 * 1024;     // the Council API's log, read from its end ...
const SERVICE_LOG_LINES = 20;            // ... without its /health probe lines
const HEALTH_LINE = /"GET \/health\b/;
const ACTIVITY_SHOWN = 20;               // People: the newest events, the rest behind ?limit=
const ACTIVITY_MAX = 1000;               // audit.recent's own ceiling
const EXCEPTIONS_SHOWN = 50;
const EXCEPTION_DAYS = 7;
const ACCOUNT_EVENTS = 15;               // the old account page's
const RUN_ID = /^([A-Za-z]+)-(\d{10,16})$/;

const list = (v) => (Array.isArray(v) ? v : []);
const num = (x) => { const v = Number(x); return x === null || x === undefined || x === '' || !Number.isFinite(v) ? null : v; };
const r1 = (x) => (num(x) === null ? null : Math.round(num(x) * 10) / 10);
const isoMs = admin.isoMs;
const isoAt = (text) => { const ms = Date.parse(text); return Number.isFinite(ms) ? isoMs(ms) : null; };
// One log line as a page may show it: the last state of a progress bar (\r-rewritten), no absolute paths, cut.
const cutLine = (line) => admin.noPaths(String(line).split('\r').filter(Boolean).pop() || '').slice(0, LINE_CHARS);
// A run's id in these routes: its job and its start in epoch ms (the kept runs carry no run id of their own).
const runIdOf = (key, at) => `${key}-${Math.round(at)}`;
// A directory as a page may show it: relative to the checkout (src/llmCouncil), or its own name when it lies outside.
function dirOf(dir, root) {
  if (!dir) return null;
  const rel = root ? path.relative(root, String(dir)) : null;
  if (rel === '') return '.';
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel) ? rel.split(path.sep).join('/') : path.basename(String(dir));
}

// A browser and system in two words, from a user agent; the device a sign-in is compared on is the agent itself with
// its version numbers taken out (deviceKey), so a browser update is not a new device.
function deviceOf(agent) {
  const a = String(agent || '');
  if (!a) return 'unknown';
  const browser = /Edg(?:e|A|iOS)?\//.test(a) ? 'Edge' : (/OPR\/|Opera/.test(a) ? 'Opera' : (/Firefox\/|FxiOS\//.test(a) ? 'Firefox'
    : (/Chrome\/|CriOS\//.test(a) ? 'Chrome' : (/Safari\//.test(a) ? 'Safari' : null))));
  const system = /iPhone|iPad|iPod/.test(a) ? 'iOS' : (/Android/.test(a) ? 'Android' : (/Windows/.test(a) ? 'Windows'
    : (/Macintosh|Mac OS X/.test(a) ? 'macOS' : (/CrOS/.test(a) ? 'ChromeOS' : (/Linux/.test(a) ? 'Linux' : null)))));
  if (!browser && !system) return a.split(/[\s/;(]/)[0].slice(0, 30) || 'unknown';
  return browser && system ? `${browser} on ${system}` : (browser || system);
}
const deviceKey = (agent) => String(agent || '').replace(/\d+(?:[._]\d+)*/g, '#');
// The address a sign-in is compared on: an IPv6 address by its /64 network (a phone rotates the interface half of its
// address inside the same home network), an IPv4 address whole.
function networkOf(address) {
  const a = String(address || '').replace(/%.*$/, '').toLowerCase();
  if (!a.includes(':')) return a || 'unknown';
  const parts = a.split('::');
  const left = parts[0] ? parts[0].split(':') : [];
  const right = parts.length > 1 && parts[1] ? parts[1].split(':') : [];
  const groups = parts.length > 1 ? [...left, ...Array(Math.max(0, 8 - left.length - right.length)).fill('0'), ...right] : left;
  return `${groups.slice(0, 4).map((g) => g.replace(/^0+(?=.)/, '')).join(':')}::/64`;
}

// --- reading a file's end ---------------------------------------------------------------------------------------
// The bytes [from, to) of a file, at most `limit` of them counted back from `to`: {text, whole} - whole when the read
// reached `from`. null when the file cannot be read.
function readEnd(file, from, to, limit) {
  let fd;
  try {
    const size = fs.statSync(file).size;
    const end = Math.min(to === null || to === undefined ? size : to, size);
    const start = Math.max(from, end - limit, 0);
    const buffer = Buffer.alloc(Math.max(0, end - start));
    fd = fs.openSync(file, 'r');
    fs.readSync(fd, buffer, 0, buffer.length, start);
    return { text: buffer.toString('utf-8'), whole: start <= from };
  } catch (e) {
    return null;
  } finally {
    if (fd !== undefined) try { fs.closeSync(fd); } catch (e) { /* closed */ }
  }
}
// Lines of a read: the first one dropped when the read began mid-line, the empty end of a final newline dropped.
function linesOf(read) {
  const lines = read.text.split('\n');
  if (!read.whole) lines.shift();
  if (lines.length && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

// The newest lines of a service's log without its /health probe lines (the Council API logs one per probe - about
// 2,000 of them by Oct 2026 - which would push everything else out of a short tail).
function serviceLog(file) {
  const read = file ? readEnd(file, 0, null, SERVICE_LOG_BYTES) : null;
  if (!read) return { lines: [], healthDropped: 0, file: file ? path.basename(file) : null };
  const all = linesOf(read);
  const kept = all.filter((line) => !HEALTH_LINE.test(line));
  return { lines: kept.slice(-SERVICE_LOG_LINES).map(cutLine), healthDropped: all.length - kept.length, file: path.basename(file),
    scanned: { lines: all.length, kb: Math.round(read.text.length / 1024), whole: read.whole } };
}

// --- GET /api/admin/jobs -------------------------------------------------------------------------------------------
const resultOf = (h) => {
  if (h.dry) return 'dry run';
  if (h.blocked) return 'waited on the lock';   // it stayed queued; its run, if any, is an entry of its own
  if (h.exit === 0) return 'ok';
  return num(h.exit) === null ? 'unknown' : `exit ${h.exit}`;
};
// status: jobs.status({full: true}); jobs: jobs.JOBS; slices: {job key: admin.readSlices}; services: services.status()
// (or {error}); actionsFor: services.actionsFor; serviceLogs: {service key: serviceLog}; root: the checkout, which a
// service's working directory is named relative to. Pure, for the test.
function jobsView({ status, jobs = [], slices = {}, services = null, actionsFor = null, serviceLogs = {}, root = null, now = new Date() }) {
  const nowMs = now.getTime();
  const errors = [];
  const byKey = Object.fromEntries(list(jobs).map((j) => [j.key, j]));
  const s = status || null;
  if (!s) errors.push('jobs');
  const history = list(s && s.history);
  const ran = history.filter(admin.ran);
  const usual = (key, at) => (byKey[key] ? admin.usualOf({ key, at }, byKey[key], slices[key], ran) : null);
  const short = (h) => ({ id: runIdOf(h.key, h.at), key: h.key, start: isoMs(h.at), minutes: r1(num(h.seconds) === null ? null : h.seconds / 60), result: resultOf(h) });

  const runs = history.map((h) => {
    const job = byKey[h.key];
    const out = { id: runIdOf(h.key, h.at), key: h.key, name: h.name || (job ? job.name : h.key), kind: h.dry ? 'dry' : (h.blocked ? 'blocked' : 'run'),
      start: isoMs(h.at), end: isoMs(h.finishedAt), minutes: h.dry ? null : r1(num(h.seconds) === null ? null : h.seconds / 60),
      result: resultOf(h), note: admin.noPaths(h.note || h.error || null), reason: h.reason || null, sha: h.sha || null };
    if (out.kind === 'run' && job) {
      const line = admin.runLine(h, job, slices[h.key], ran);
      out.usual = line.usual ? { median: r1(line.usual.median), min: r1(line.usual.min), max: r1(line.usual.max), n: line.usual.n } : null;
      out.overUsual = line.overUsual;
      out.vsMedian = line.vsMedian;
      // the run's own log slice: its warning signatures, how many none of the job's earlier runs had; null when the
      // slice is no longer in the log (rotated away)
      out.warnings = line.warnings ? { signatures: line.warnings.length, new: line.warnings.filter((w) => w.new).length } : null;
    }
    return out;
  });

  const cur = s && s.current ? s.current : null;
  let running = null;
  if (cur) {
    const minutes = (nowMs - cur.startedAt) / 60000;
    const u = usual(cur.key, cur.startedAt);
    running = { id: runIdOf(cur.key, cur.startedAt), runId: cur.runId, key: cur.key, name: cur.name, reason: cur.reason || null,
      start: isoMs(cur.startedAt), minutes: r1(minutes), pid: cur.pid || null, sha: cur.sha || null,
      usual: u ? { median: r1(u.median), min: r1(u.min), max: r1(u.max), n: u.n, basis: u.basis } : null,
      vsMedian: u && u.median ? Math.round((minutes / u.median) * 100) / 100 : null, overUsual: u ? minutes > u.max : null,
      log: list(cur.log).slice(-LIVE_LOG_LINES).map(cutLine) };
  }
  // script: a file name (jobs.js lockHolder falls back to the holder's last argument, which can be an absolute path)
  const lock = s ? (s.lock ? { held: true, pid: s.lock.pid, script: s.lock.script ? path.basename(String(s.lock.script)) : null, key: s.lock.key || null,
    ours: Boolean(cur && s.lock.key === cur.key) }
    : { held: false }) : null;

  const daily = list(jobs).find((j) => j.daily) || null;
  const today = brussels.dayOf(nowMs);
  const todays = daily ? ran.filter((h) => h.key === daily.key && brussels.dayOf(h.at) === today) : [];
  const failed = ran.filter((h) => h.exit !== 0);

  const svc = services && !services.error ? list(services).map((x) => ({
    key: x.key, name: x.name, description: x.description, status: x.status, healthy: Boolean(x.healthy), summary: x.summary || null,
    // what it runs and where (services.js: "python3 api.py ..." in src/llmCouncil), without an absolute path
    command: admin.noPaths(x.command || null), cwd: dirOf(x.cwd, root),
    pid: x.pid || null, port: x.port || null, since: isoMs(x.since),
    upMinutes: x.status === 'running' && num(x.since) !== null ? r1((nowMs - x.since) / 60000) : null,
    restarts: num(x.restarts) || 0, lastExit: x.lastExit ? { code: x.lastExit.code, signal: x.lastExit.signal || null, at: isoMs(x.lastExit.at) } : null,
    lastError: admin.noPaths(x.lastError || null), checkedAt: isoMs(x.checkedAt),
    actions: actionsFor ? actionsFor(x) : null, log: serviceLogs[x.key] || null,
  })) : null;
  if (services && services.error) errors.push('services');

  return {
    builtAt: isoMs(nowMs), tz: brussels.TZ,
    mode: s ? { mode: s.mode, reason: s.modeReason } : null,
    lock, running,
    queue: list(s && s.queue).map((q) => ({ key: q.key, name: q.name || (byKey[q.key] || {}).name || q.key, reason: q.reason || null,
      queuedAt: isoMs(q.queuedAt), waitedMinutes: num(q.queuedAt) === null ? null : r1((nowMs - q.queuedAt) / 60000), manual: Boolean(q.manual) })),
    jobs: list(s && s.jobs).map((j) => {
      const job = byKey[j.key] || {};
      const u = usual(j.key, nowMs);
      const last = ran.find((h) => h.key === j.key) || null;
      return { key: j.key, name: j.name, description: j.description, command: j.command, log: j.logFile ? path.basename(j.logFile) : null,
        plan: j.plan || null, daily: Boolean(job.daily), nextAt: isoMs(j.nextAt), trigger: j.nextAt ? null : j.next,
        // the median of the job's earlier successful runs (admin.js usualOf: a weekly job's from its log, reaching
        // further back than the kept runs; the predictor's from the kept runs), in minutes
        usual: u ? { median: r1(u.median), min: r1(u.min), max: r1(u.max), n: u.n, from: u.from, to: u.to, basis: u.basis } : null,
        last: last ? short(last) : null,
        canRun: !j.refusal, refusal: j.refusal || null };
    }),
    runs: { kept: s ? s.kept || null : null, shown: runs.length, ran: ran.length, items: runs },
    today: daily ? { day: today, key: daily.key, run: todays.length ? short(todays[0]) : null, running: Boolean(cur && cur.key === daily.key) } : null,
    // the newest run that did not end with exit 0 (a failure, or killed: exit unknown), over the runs kept
    lastFailure: failed.length ? { ...short(failed[0]), name: failed[0].name || null, note: admin.noPaths(failed[0].note || failed[0].error || null) } : null,
    services: svc,
    servicesError: services && services.error ? services.error : null,
    errors,
  };
}

// --- GET /api/admin/jobs/run/:id/log ---------------------------------------------------------------------------------
// One run's own lines: the slice of its job's log between the scheduler's markers (admin.js slicesOf), read from its
// end. Its warning signatures come first - each with how often, whether new (none of the job's earlier runs in the
// log had it) and its newest line - then the slice's last lines, at most LOG_LINES together. history: the kept runs
// (jobs.status), current: the running job.
function runLogView({ id, jobs = [], slices = {}, history = [], current = null }) {
  const m = RUN_ID.exec(String(id || ''));
  if (!m) return { error: 'a run id is <job>-<start in epoch ms>', status: 400 };
  const job = list(jobs).find((j) => j.key === m[1]);
  if (!job) return { error: 'unknown job', status: 404 };
  const at = Number(m[2]);
  const mine = list(slices[job.key]);
  // the slice and the run nearest the id's start: a job run again a minute after it failed has both runs, and both
  // slices, within SLICE_MATCH_MS of either start
  const own = admin.nearestAt(mine, at);
  if (!own) return { error: 'no log slice for this run: rotated out of the log, or it never started (a dry run)', status: 404 };
  const ran = list(history).filter(admin.ran);
  const live = current && current.key === job.key ? [{ at: current.startedAt, live: true }] : [];
  const named = admin.nearestAt([...ran.filter((x) => x.key === job.key), ...live], at, (x) => x.at);
  const isCurrent = Boolean(named && named.live);
  const kept = named && !named.live ? named : null;
  // a run older than the kept history is read from its markers alone
  const h = kept || (isCurrent ? { key: job.key, at: current.startedAt, exit: null, seconds: null }
    : { key: job.key, at: own.start, exit: own.status, seconds: own.seconds, finishedAt: own.seconds === null ? null : own.start + own.seconds * 1000 });
  const line = admin.runLine(h, job, mine, ran);
  const warnings = list(line.warnings).slice(0, LOG_LINES).map((w) => ({ ...w, line: own.examples && own.examples[w.sig] !== undefined ? cutLine(own.examples[w.sig]) : null }));
  const read = readEnd(own.file, own.from, own.to, LOG_TAIL_BYTES);
  const lines = read ? linesOf(read) : [];
  const room = Math.max(0, LOG_LINES - warnings.length);
  const tail = room ? lines.slice(-room).map(cutLine) : [];
  return {
    id: runIdOf(job.key, h.at), key: job.key, job: job.name, log: own.file ? path.basename(own.file) : null,
    start: isoMs(h.at), end: isoMs(h.finishedAt), minutes: line.minutes, result: isCurrent && !kept ? 'running' : line.result,
    note: admin.noPaths(line.note), sha: line.sha || own.sha || null, inKeptRuns: Boolean(kept), running: isCurrent,
    warnings, noiseFiltered: line.noiseFiltered, comparedWith: line.comparedWith,
    // of: the slice's lines when the log was last parsed (a running job's grows); read: the bytes read from the end
    lines: { shown: tail.length, of: own.lines, whole: Boolean(read && read.whole), readKb: read ? Math.round(read.text.length / 1024) : 0 },
    tail,
    error: read ? undefined : 'the log file could not be read',
  };
}

// --- GET /api/admin/people ---------------------------------------------------------------------------------------------
const entryOf = (e) => ({ at: isoAt(e.at), event: e.event, label: audit.label(e.event), user: e.user, address: e.ip || null,
  device: deviceOf(e.agent), detail: e.detail || null });
const LOCK_MINUTES = Math.round(auth.LOGIN_WINDOW_MS / 60000);
// The sign-in exceptions of the last EXCEPTION_DAYS days, newest first: failed sign-ins; lockouts - one per address
// and name whose failures reach auth.js's limit (LOGIN_MAX_FAILURES within LOGIN_WINDOW_MS, counted as auth.js counts
// them: a sign-in clears them), whether or not anyone tries again; the attempts refused while locked (login.locked);
// and sign-ins from a device or an address the same name had not signed in from before (in the events scanned; a
// name's first sign-in on record counts as new, and a name deleted or added again starts afresh). events:
// audit.recent, newest first.
function exceptionsOf(events, nowMs) {
  const fromMs = nowMs - EXCEPTION_DAYS * 86400000;
  const known = {};
  const failing = {};   // `${address}|${lower-cased name}` -> {first, count}
  const failKey = (e) => `${e.ip || 'unknown'}|${String(e.user || '').toLowerCase()}`;
  const out = [];
  list(events).slice().reverse().forEach((e) => {
    if (!e || typeof e.at !== 'string') return;
    const ms = Date.parse(e.at);
    const inWindow = Number.isFinite(ms) && ms >= fromMs && ms <= nowMs;
    if (e.event === 'login.failed') {
      const key = failKey(e);
      if (!failing[key] || ms - failing[key].first > auth.LOGIN_WINDOW_MS) failing[key] = { first: ms, count: 0 };
      failing[key].count += 1;
      if (inWindow) out.push({ ...entryOf(e), kind: 'failed sign-in' });
      if (inWindow && failing[key].count === auth.LOGIN_MAX_FAILURES) {
        out.push({ ...entryOf(e), kind: 'locked out', detail: `${auth.LOGIN_MAX_FAILURES} failed within ${LOCK_MINUTES} min` });
      }
      return;
    }
    if (e.event === 'login.locked') {
      if (inWindow) out.push({ ...entryOf(e), kind: 'refused while locked' });
      return;
    }
    if (e.event === 'user.added' || e.event === 'user.deleted') { delete known[e.user]; return; }
    if (e.event !== 'login.ok') return;
    delete failing[failKey(e)];
    const k = known[e.user] || (known[e.user] = { addresses: new Set(), devices: new Set(), n: 0 });
    const address = networkOf(e.ip);
    const device = deviceKey(e.agent);
    if (inWindow) {
      const newAddress = !k.addresses.has(address);
      const newDevice = !k.devices.has(device);
      const kind = !k.n ? 'first sign-in on record' : (newAddress && newDevice ? 'new device and address' : (newAddress ? 'new address' : (newDevice ? 'new device' : null)));
      if (kind) out.push({ ...entryOf(e), kind });
    }
    k.addresses.add(address);
    k.devices.add(device);
    k.n += 1;
  });
  return { from: isoMs(fromMs), items: out.reverse() };
}

// A name's last sign-in: the newer of the audit log's (logged: audit.lastLogins' entry, with the browser) and the
// account's own record (lastLoginAt); never one from before the account was created - a name deleted and added again
// starts with none.
function lastSignInOf(record, logged) {
  const created = record ? Date.parse(record.createdAt) : NaN;
  const a = logged && !(Number.isFinite(created) && Date.parse(logged.at) < created) ? logged : null;
  const fromAudit = a ? { at: isoAt(a.at), address: a.ip || null, device: deviceOf(a.agent), from: 'activity' } : null;
  const fromAccount = record && record.lastLoginAt ? { at: isoAt(record.lastLoginAt), address: record.lastLoginIp || null, device: null, from: 'account' } : null;
  if (fromAudit && fromAccount) return fromAccount.at > fromAudit.at ? fromAccount : fromAudit;
  return fromAudit || fromAccount;
}

// users: auth.listUsers (never a hash); adminName: the environment's administrator, or null (open access); events:
// audit.recent (newest first); lastLogins: audit.lastLogins (each name's newest sign-in); total: audit.count; limit:
// how many events of the activity to send.
function peopleView({ users = [], adminName = null, writable = false, events = [], lastLogins = {}, total = null, limit = ACTIVITY_SHOWN, now = new Date() }) {
  const nowMs = now.getTime();
  const lastSignIn = (name, record) => lastSignInOf(record, (lastLogins || {})[name]);
  const accounts = [
    ...(adminName ? [{ name: adminName, role: 'admin', source: 'environment', created: null, passwordChangedAt: null, signIns: null, lastSignIn: lastSignIn(adminName, null) }] : []),
    ...list(users).map((u) => ({ name: u.name, role: u.role, source: 'users.json', created: isoAt(u.createdAt), passwordChangedAt: isoAt(u.passwordChangedAt),
      signIns: u.logins, lastSignIn: lastSignIn(u.name, u) })),
  ];
  const ex = exceptionsOf(events, nowMs);
  const count = (kind) => ex.items.filter((x) => kind.includes(x.kind)).length;
  const oldest = list(events).length ? isoAt(events[events.length - 1].at) : null;
  const scannedAll = total !== null && total.events === list(events).length;
  return {
    builtAt: isoMs(nowMs), tz: brussels.TZ,
    auth: { enabled: Boolean(adminName), writable, minPasswordLength: auth.MIN_PASSWORD_LENGTH, userName: '3-32 letters, digits, dots, underscores or dashes' },
    tiles: {
      users: accounts.length,
      failed: count(['failed sign-in']), lockouts: count(['locked out']), refusedWhileLocked: count(['refused while locked']),
      newDeviceOrAddress: count(['first sign-in on record', 'new device and address', 'new address', 'new device']),
      days: EXCEPTION_DAYS, from: ex.from, to: isoMs(nowMs),
      // what was read: every event in the log, or the newest ones back to `oldest` (then complete says whether that
      // reaches the window's start)
      scanned: { events: list(events).length, oldest, all: scannedAll, complete: scannedAll || (oldest !== null && oldest <= ex.from) },
    },
    users: accounts,
    exceptions: { shown: Math.min(ex.items.length, EXCEPTIONS_SHOWN), of: ex.items.length, items: ex.items.slice(0, EXCEPTIONS_SHOWN),
      compared: 'each sign-in against the same name\'s earlier sign-ins in the events scanned; a device is the browser\'s user agent without its version numbers, an IPv6 address counts by its /64 network' },
    activity: { shown: Math.min(limit, list(events).length), of: total ? total.events : null, oldest: total ? isoAt(total.oldest) : null,
      items: list(events).slice(0, limit).map(entryOf) },
  };
}

// --- GET /api/account, GET /api/whats-new ---------------------------------------------------------------------------
// lastLogin: audit.lastLogins' entry for the reader's name (read further back than the activity rows, so failed
// attempts on the name cannot push it out)
function accountView(user, { record = null, events = [], lastLogin = null, refusal = null } = {}) {
  const lastIn = lastSignInOf(record, lastLogin);
  return {
    name: user.name, role: user.role === 'admin' ? 'admin' : 'user', open: user.open === true,
    source: user.open ? 'open access' : (user.role === 'admin' ? 'environment' : 'users.json'),
    created: record ? isoAt(record.createdAt) : null, passwordChangedAt: record ? isoAt(record.passwordChangedAt) : null,
    password: { canChange: !refusal, why: refusal ? refusal.message : null, minLength: auth.MIN_PASSWORD_LENGTH },
    lastSignIn: lastIn ? { at: lastIn.at, address: lastIn.address, device: lastIn.device } : null,
    // sign-ins and changes recorded for this name, newest first: a session the reader did not start shows here
    activity: { shown: list(events).length, max: ACCOUNT_EVENTS, items: list(events).map((e) => { const x = entryOf(e); delete x.user; return x; }) },
  };
}

function whatsNewView(user) {
  const open = user.open === true;
  const state = open ? {} : whatsnew.stateOf(user);
  const seen = new Set(state.seen || []);
  // a note for everyone may still point at an administrator's page: a user gets it without that link
  const linkOf = (l) => (l && (user.role === 'admin' || !/^\/admin(\/|$)/.test(String(l.href))) ? { href: l.href, label: l.label } : null);
  const items = whatsnew.visible(user).map((a) => ({ id: a.id, date: a.date, level: a.level, audience: a.audience, title: a.title,
    body: list(a.body), link: linkOf(a.link), seen: open ? null : seen.has(a.id) }));
  const pending = whatsnew.pending(user);
  return {
    open, audience: user.role === 'admin' ? 'admin' : 'all',
    unseen: open ? null : items.filter((a) => !a.seen).length, of: items.length,
    // what the app pops up now: the tour for an account that never finished it, else the unseen major notes
    pending: pending ? { kind: pending.kind, ids: pending.items.map((a) => a.id) } : null,
    tour: { done: open ? null : Boolean(state.tourDone), title: whatsnew.TOUR.title, intro: whatsnew.TOUR.intro,
      steps: whatsnew.tourSteps(user).map((x) => ({ title: x.title, body: x.body })) },
    items,
  };
}

// --- routes ---------------------------------------------------------------------------------------------------------
// options: root (the checkout: log/ for the slices), logDir; jobs, services (the modules; their status and actions);
// jsonBody (the JSON body parser - express.json by default, a stand-in in the test). Installed before admin.install,
// whose catch-all answers anything else under /api/admin with a JSON 404.
function install(app, options = {}) {
  const dirs = admin.resolveDirs({ root: options.root, logDir: options.logDir });
  const jobs = options.jobs || null;
  const services = options.services || null;
  const json = (res, body, status = 200) => { res.set('Cache-Control', 'no-store'); res.status(status).json(body); };
  const fail = (res, e, error = 'could not build this view') => {
    console.error(`[${new Date().toISOString()}] system.js:`, e);
    return json(res, { error }, 500);
  };
  // an action that throws (a config/ that cannot be written, a launch that fails) is a JSON 500 too, never express's
  // HTML error page with its stack and the server's paths
  const safe = (fn) => (req, res, next) => {
    const failed = (e) => fail(res, e, 'the server failed while doing this - see the server log');
    try {
      const r = fn(req, res, next);
      if (r && typeof r.catch === 'function') r.catch(failed);
    } catch (e) { failed(e); }
  };
  // a build's body, or the JSON error it names
  const answer = (res, body) => (body && body.error && body.status ? json(res, { error: body.error }, body.status) : json(res, body));
  // a shared rule's {ok, status, message}: 200 with the message, or the rule's status with it as the error - without
  // the server's absolute paths (services.js names its log file when it cannot open it)
  const reply = (res, result, extra = {}) => (result.ok ? json(res, { ok: true, message: admin.noPaths(result.message), ...extra })
    : json(res, { error: admin.noPaths(result.message) }, result.status || 500));

  // JSON bodies for these POSTs only (server.js parses the sign-in form with express.urlencoded); a body that is not JSON is
  // a JSON 400, never express's HTML error page.
  const parser = options.jsonBody || require('express').json({ limit: '16kb' });
  const jsonBody = (req, res, next) => parser(req, res, (err) => {
    if (!err) return next();
    return json(res, { error: err.type === 'entity.too.large' ? 'the request body is too large' : 'the request body is not valid JSON' }, err.type === 'entity.too.large' ? 413 : 400);
  });
  const csrf = (req, res, next) => (auth.csrfOk(req) ? next() : json(res, { error: 'Invalid form token - reload the page and try again.' }, 403));
  const signedIn = (req, res, next) => (req.user ? next() : json(res, { error: 'login required' }, 401));
  const body = (req) => req.body || {};

  const jobList = () => list(jobs && jobs.JOBS);
  const slicesAll = () => Object.fromEntries(jobList().map((j) => [j.key, admin.readSlices(dirs, j)]));
  const jobsStatus = () => {
    try { return jobs && typeof jobs.status === 'function' ? jobs.status({ full: true }) : null; } catch (e) { return null; }   // the schedule has not started
  };
  const needs = (fn) => (req, res, next) => (jobs && typeof jobs[fn] === 'function' ? next() : json(res, { error: 'the schedule is not available here' }, 503));

  // --- Jobs & services (administrator) ---
  app.get('/api/admin/jobs', admin.guard, async (req, res) => {
    try {
      let all = null;
      if (services && typeof services.status === 'function') {
        try { all = await admin.within(services.status(), SERVICES_WAIT_MS); } catch (e) { all = { error: admin.noPaths(e.message) }; }
      }
      const serviceLogs = Object.fromEntries(list(all).map((x) => [x.key, serviceLog(x.logFile)]));
      json(res, jobsView({ status: jobsStatus(), jobs: jobList(), slices: slicesAll(), services: all, actionsFor: services && services.actionsFor, serviceLogs,
        root: dirs.root }));
    } catch (e) { fail(res, e); }
  });
  app.get('/api/admin/jobs/run/:id/log', admin.guard, (req, res) => {
    try {
      const s = jobsStatus();
      const m = RUN_ID.exec(String(req.params.id || ''));
      const job = m ? jobList().find((j) => j.key === m[1]) : null;
      answer(res, runLogView({ id: req.params.id, jobs: jobList(), slices: job ? { [job.key]: admin.readSlices(dirs, job) } : {},
        history: s ? s.history : [], current: s ? s.current : null }));
    } catch (e) { fail(res, e); }
  });
  app.post('/api/admin/jobs/run', admin.guard, jsonBody, csrf, needs('runNow'), safe((req, res) => {
    const result = jobs.runNow(String(body(req).key || ''), req.user.name);
    reply(res, result, { started: Boolean(result.started) });
  }));
  app.post('/api/admin/jobs/cancel', admin.guard, jsonBody, csrf, needs('cancelQueued'), safe((req, res) => reply(res, jobs.cancelQueued(String(body(req).key || ''), req.user.name))));
  app.post('/api/admin/jobs/stop', admin.guard, jsonBody, csrf, needs('stopRun'), safe((req, res) => reply(res, jobs.stopRun(String(body(req).runId || ''), req.user.name))));
  app.post('/api/admin/services/:name/:action', admin.guard, jsonBody, csrf, safe(async (req, res) => {
    if (!services || typeof services.act !== 'function') return json(res, { error: 'no supervised services here' }, 503);
    if (!list(services.SERVICES).some((x) => x.key === req.params.name)) return json(res, { error: 'unknown service' }, 404);
    if (!list(services.ACTIONS).includes(req.params.action)) return json(res, { error: 'unknown action: start, stop or restart' }, 404);
    return reply(res, await services.act(req.params.name, req.params.action, req.user.name));
  }));

  // --- People (administrator) ---
  app.get('/api/admin/people', admin.guard, (req, res) => {
    const raw = req.query.limit;
    let limit = ACTIVITY_SHOWN;
    if (raw === 'all') limit = ACTIVITY_MAX;
    else if (raw !== undefined && raw !== '') {
      limit = Number(raw);
      if (!Number.isInteger(limit) || limit < 1 || limit > ACTIVITY_MAX) return json(res, { error: `limit is 1-${ACTIVITY_MAX} or all` }, 400);
    }
    try {
      json(res, peopleView({ users: auth.listUsers(), adminName: auth.adminName(), writable: !auth.managementRefusal(),
        events: audit.recent(ACTIVITY_MAX), lastLogins: audit.lastLogins(ACTIVITY_MAX), total: audit.count(), limit }));
    } catch (e) { fail(res, e); }
  });
  app.post('/api/admin/people/add', admin.guard, jsonBody, csrf, safe((req, res) => reply(res, auth.addUser(req, body(req).username, body(req).password))));
  app.post('/api/admin/people/password', admin.guard, jsonBody, csrf, safe((req, res) => reply(res, auth.setUserPassword(req, body(req).username, body(req).password))));
  app.post('/api/admin/people/delete', admin.guard, jsonBody, csrf, safe((req, res) => reply(res, auth.deleteUser(req, body(req).username))));

  // --- Account and What's new (every signed-in reader; always the reader's own) ---
  app.get('/api/account', signedIn, (req, res) => {
    try {
      const u = req.user;
      const record = !u.open && u.role !== 'admin' ? auth.listUsers().find((x) => x.name === u.name) || null : null;
      json(res, accountView(u, { record, events: u.open ? [] : audit.recent(ACCOUNT_EVENTS, u.name),
        lastLogin: u.open ? null : audit.lastLogins(ACTIVITY_MAX)[u.name] || null, refusal: auth.ownPasswordRefusal(u) }));
    } catch (e) { fail(res, e); }
  });
  app.post('/api/account/password', signedIn, jsonBody, csrf, safe((req, res) => {
    const result = auth.changeOwnPassword(req, res, body(req));
    // the change ends every older session, this one included: the fresh one carries a fresh form token
    reply(res, result, result.ok && result.session ? { csrf: result.session.csrf } : {});
  }));
  app.get('/api/whats-new', signedIn, (req, res) => {
    try { json(res, whatsNewView(req.user)); } catch (e) { fail(res, e); }
  });
  app.post('/api/whats-new/seen', signedIn, jsonBody, csrf, safe((req, res) => {
    if (req.user.open) return json(res, { error: 'Authentication is off: there is no account to record it against.' }, 409);
    const mine = new Set(whatsnew.visible(req.user).map((a) => a.id));
    // all: every note the reader may see ("Mark all as seen"; idsFrom keeps at most 50 named ids)
    const ids = body(req).all === true ? [...mine] : whatsnew.idsFrom(body(req).ids);
    const seen = ids.filter((id) => mine.has(id));
    const tour = body(req).tour === true;
    if (!seen.length && !tour) return json(res, { error: 'no note of yours was named' }, 400);
    whatsnew.markSeen(req.user, seen, tour);
    return json(res, { ok: true, message: `${seen.length} marked seen${tour ? ', the introduction done' : ''}`, seen, ignored: ids.filter((id) => !mine.has(id)) });
  }));
  app.post('/api/whats-new/replay', signedIn, jsonBody, csrf, safe((req, res) => {
    if (req.user.open) return json(res, { error: 'Authentication is off: there is no account to record it against.' }, 409);
    whatsnew.replay(req.user);
    return json(res, { ok: true, message: 'the introduction shows again on the next page' });
  }));
}

module.exports = { install, jobsView, runLogView, peopleView, exceptionsOf, accountView, whatsNewView, serviceLog, deviceOf, deviceKey, networkOf, readEnd };
