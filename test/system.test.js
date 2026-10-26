'use strict';
// The System pages' JSON (system.js) and the rules it calls (jobs.js runNow / cancelQueued / stopRun, services.js act,
// auth.js addUser / setUserPassword / deleteUser / changeOwnPassword, whatsnew.js): the guard (401 signed out, 403
// for a user), the form token on every POST, each action's happy path and refusal, and no password hash or salt in
// any answer - on stand-ins only. Nothing here reads or writes the checkout's config/ or log/, starts a process or
// probes a port:
//   - config.js is a stand-in for every module that asks for it: config/ is a temp dir (users.json, audit.jsonl,
//     user-state.json, scheduler.json), with a stand-in administrator;
//   - jobs.js and services.js run as copies in temp dirs (their log/ and process.lock are the copies' own), with
//     child_process and http stubbed: a "started" service is a fake child, and no job can launch - the copy's
//     process.lock names this test's own process, so a queued job waits behind it, and a launch would only reach the
//     stub, which the test checks was never asked;
//   - only when asked (SYSTEM_TEST_PROD=1, which npm test never sets), production's scheduler.json and job logs are
//     read (never written) to check the views' size and shape; its sign-in audit log is never read.
//   node test/system.test.js        (NODE_PATH=<a node_modules with express> adds the same routes through express)
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { EventEmitter } = require('events');

const REPO = path.join(__dirname, '..');
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- stand-ins ---------------------------------------------------------------------------------------------------
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'system-'));
const CONFIG_DIR = path.join(tmp, 'config');
const JOBS_ROOT = path.join(tmp, 'jobsroot');
const SVC_ROOT = path.join(tmp, 'svcroot');
[CONFIG_DIR, path.join(JOBS_ROOT, 'log'), path.join(SVC_ROOT, 'log')].forEach((d) => fs.mkdirSync(d, { recursive: true }));
const ADMIN_NAME = 'boss';
const STUB_CONFIG = { CONFIG_DIR, ADMIN_USER: ADMIN_NAME, ADMIN_PASSWORD: 'correct horse battery staple', SESSION_SECRET: 'ab'.repeat(32),
  INTERFACE: '127.0.0.1', PORT: 0, TRUST_PROXY: false };
const spawned = [];
const killed = [];
const STUB_CP = {
  spawn(command, args) {
    spawned.push([command, args]);
    const child = new EventEmitter();
    child.pid = 4242;
    child.unref = () => {};
    child.kill = (signal) => { killed.push(signal); setImmediate(() => child.emit('exit', null, signal)); return true; };
    return child;
  },
  execFileSync() { throw new Error('stub: no git here'); },
  execFile(cmd, args, opts, cb) { (typeof opts === 'function' ? opts : cb)(new Error('stub: no git here')); },
};
const STUB_HTTP = { get() { const req = new EventEmitter(); req.destroy = () => {}; setImmediate(() => req.emit('error', new Error('stub: nothing listens'))); return req; } };
fs.copyFileSync(path.join(REPO, 'jobs.js'), path.join(JOBS_ROOT, 'jobs.js'));
fs.copyFileSync(path.join(REPO, 'services.js'), path.join(SVC_ROOT, 'services.js'));
const original = Module.prototype.require;
const inside = (file, dir) => Boolean(file) && file.startsWith(dir + path.sep);
Module.prototype.require = function stubbed(id) {
  const from = this.filename || '';
  if (id === './config' && (inside(from, REPO) || inside(from, tmp))) return STUB_CONFIG;
  if (inside(from, tmp)) {
    if (id === 'child_process') return STUB_CP;
    if (id === 'http') return STUB_HTTP;
    if (id.startsWith('./')) return original.call(this, path.join(REPO, id));   // the copies' other modules are the checkout's
  }
  return original.apply(this, arguments);
};
delete process.env.SCHEDULER;   // dry: the schedule starts only what an administrator asks for by hand

const auth = require('../auth');
const audit = require('../audit');
const whatsnew = require('../whatsnew');
const admin = require('../admin');
const system = require('../system');
const api = require('../api');
const { ANNOUNCEMENTS } = require('../announcements');
ok(auth.USERS_FILE === path.join(CONFIG_DIR, 'users.json') && audit.FILE === path.join(CONFIG_DIR, 'audit.jsonl') && whatsnew.STATE_FILE.startsWith(CONFIG_DIR),
  'the accounts, the audit log and the reader state are the temp dir\'s, never the checkout\'s config/');

// --- a stand-in express: use / get / post with middleware chains, run in order; every answer kept ------------------
const answers = [];
function standIn() {
  const stack = [];
  const add = (method) => (pattern, ...fns) => { stack.push({ method, pattern, fns }); };
  return { stack, use(prefix, fn) { stack.push(typeof prefix === 'function' ? { method: '*', prefix: '', fns: [prefix] } : { method: '*', prefix, fns: [fn] }); },
    get: add('GET'), post: add('POST'), set() {} };
}
function matchRoute(pattern, pathname) {
  const p = pattern.split('/'); const a = pathname.split('/');
  if (p.length !== a.length) return null;
  const params = {};
  for (let k = 0; k < p.length; k += 1) {
    if (p[k].startsWith(':')) params[p[k].slice(1)] = decodeURIComponent(a[k]);
    else if (p[k] !== a[k]) return null;
  }
  return params;
}
function call(app, method, url, { user, body, headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const [pathname, qs] = url.split('?');
    const out = { status: 200, headers: {}, body: undefined, text: null, redirect: null };
    const done = () => { answers.push(out.raw || out.text || ''); resolve(out); };
    const res = {
      status(c) { out.status = c; return this; }, set(k, v) { out.headers[k] = v; return this; },
      setHeader(k, v) { out.headers[k] = v; }, getHeader(k) { return out.headers[k]; },
      json(b) { out.raw = JSON.stringify(b); out.body = JSON.parse(out.raw); done(); return this; },
      send(t) { out.text = String(t); done(); return this; },
      redirect(a, b) { out.redirect = b === undefined ? a : b; out.status = b === undefined ? 302 : a; done(); },
    };
    const req = { method, user, body: body === undefined ? {} : body, query: Object.fromEntries(new URLSearchParams(qs || '')), params: {}, path: pathname,
      originalUrl: url, headers: { 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) Chrome/141.0.0.0 Safari/537.36', ...headers }, ip: '203.0.113.9' };
    req.get = (h) => req.headers[String(h).toLowerCase()];
    const chain = [];
    app.stack.forEach((l) => {
      if (l.method !== '*' && l.method !== method) return;
      if (l.prefix !== undefined) {
        if (l.prefix === '' || pathname === l.prefix || pathname.startsWith(`${l.prefix}/`)) l.fns.forEach((fn) => chain.push({ fn, params: {} }));
        return;
      }
      const params = (Array.isArray(l.pattern) ? l.pattern : [l.pattern]).map((p) => matchRoute(p, pathname)).find(Boolean);
      if (params) l.fns.forEach((fn) => chain.push({ fn, params }));
    });
    let i = 0;
    const next = (err) => {
      if (err) return reject(err);
      if (i >= chain.length) { out.status = 404; out.text = 'fell through'; return done(); }
      const { fn, params } = chain[i++];
      req.params = params;
      try { const r = fn(req, res, next); if (r && typeof r.catch === 'function') r.catch(reject); } catch (e) { reject(e); }
      return undefined;
    };
    next();
  });
}
const BAD_JSON = Symbol('not json');
// the JSON parser stand-in: the body arrives parsed, or as BAD_JSON for a body express.json would refuse
const jsonBody = (req, res, next) => (req.body === BAD_JSON ? next({ type: 'entity.parse.failed' }) : next());

const ADMIN = { name: ADMIN_NAME, role: 'admin', open: false, csrf: 'tok-admin' };
const USER = { name: 'reader', role: 'user', open: false, csrf: 'tok-user' };
const OPEN = { name: 'open access', role: 'admin', open: true, csrf: 'tok-open' };
const withToken = (who, body = {}) => ({ ...body, _csrf: who.csrf });
const auditEvents = () => fs.readFileSync(audit.FILE, 'utf-8').trim().split('\n').map((l) => JSON.parse(l));
const lastEvent = () => auditEvents().pop();
const usersOnDisk = () => JSON.parse(fs.readFileSync(auth.USERS_FILE, 'utf-8')).users;

// --- the pure parts ----------------------------------------------------------------------------------------------
const UA_WIN = (v) => `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${v}.0.0.0 Safari/537.36`;
const UA_IPHONE = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1';
ok(system.deviceOf(UA_WIN(141)) === 'Chrome on Windows' && system.deviceOf(UA_IPHONE) === 'Safari on iOS' && system.deviceOf('curl/8.5.0') === 'curl' && system.deviceOf('') === 'unknown',
  'a user agent reads as a browser on a system');
ok(system.deviceKey(UA_WIN(141)) === system.deviceKey(UA_WIN(142)) && system.deviceKey(UA_WIN(141)) !== system.deviceKey(UA_IPHONE),
  'a browser update is not a new device; another browser is');
ok(admin.noPaths(' create mode 100644 data/database/pick3/2026-10-20.json') === ' create mode 100644 data/database/pick3/2026-10-20.json'
  && admin.noPaths('saved 2026/10/21 at 09:00') === 'saved 2026/10/21 at 09:00'
  && admin.noPaths("cannot open /root/x/log/a.log: ENOENT, open '/root/x/log/a.log' (cwd /srv/app/src)") === "cannot open a.log: ENOENT, open 'a.log' (cwd src)",
  'paths: an absolute path is cut to its file name; a relative path and a date stay as they are');
ok(system.networkOf('2a02:a03f:8722:b301:322d:e75:225b:eb45') === system.networkOf('2a02:a03f:8722:b301:9a43:46f8:edc5:a22e')
  && system.networkOf('2a02:a03f:8722:b301::1') === '2a02:a03f:8722:b301::/64' && system.networkOf('2a02:a03f:8722:b302::1') !== system.networkOf('2a02:a03f:8722:b301::1')
  && system.networkOf('198.51.100.1') === '198.51.100.1' && system.networkOf('198.51.100.1') !== system.networkOf('198.51.100.2'),
  'addresses: an IPv6 address by its /64 network, an IPv4 address whole');
{
  // a job run again a minute after it failed: each run gets its own slice, its own lines and its own warnings
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'system-rerun-'));
  fs.writeFileSync(path.join(dir, 'predictor.log'), [
    '=== scheduler: Daily predictor started 2026-10-21 10:00:05 aaa1111 UTC ===', 'ImportError: the first run broke', '=== scheduler: Daily predictor finished status=1 after 4s ===',
    '=== scheduler: Daily predictor started 2026-10-21 10:01:05 aaa1111 UTC ===', 'the second run', '=== scheduler: Daily predictor finished status=0 after 30s ===', ''].join('\n'));
  const job = { key: 'predictor', name: 'Daily predictor', log: 'predictor.log', daily: { hour: 9, minute: 0 } };
  const slices = { predictor: admin.readSlices({ logDir: dir }, job) };
  const t1 = Date.parse('2026-10-21T10:00:04Z');
  const t2 = Date.parse('2026-10-21T10:01:04Z');
  const runs = [{ at: t2, finishedAt: t2 + 30000, key: 'predictor', seconds: 30, exit: 0 }, { at: t1, finishedAt: t1 + 4000, key: 'predictor', seconds: 4, exit: 1 }];
  const second = system.runLogView({ id: `predictor-${t2}`, jobs: [job], slices, history: runs });
  const first = system.runLogView({ id: `predictor-${t1}`, jobs: [job], slices, history: runs });
  const view = system.jobsView({ status: { mode: 'on', history: runs, queue: [], jobs: [], kept: 40 }, jobs: [job], slices });
  ok(second.result === 'ok' && second.tail.join() === 'the second run' && !second.warnings.length && first.result === 'exit 1' && first.tail.join() === 'ImportError: the first run broke'
    && view.runs.items.map((r) => `${r.result}:${r.warnings.signatures}`).join() === 'ok:0,exit 1:1', 'log: two runs a minute apart each read their own slice');
  fs.rmSync(dir, { recursive: true, force: true });
}
{
  // a service's command and working directory: no absolute path, the directory relative to the checkout
  const root = path.join(os.tmpdir(), 'checkout');
  const svc = (cwd, command = 'python3 api.py --config config.json --port 8099') => system.jobsView({ status: null, root,
    services: [{ key: 'councilApi', name: 'Council API', status: 'stopped', command, cwd }] }).services[0];
  const here = svc(path.join(root, 'src', 'llmCouncil'));
  const away = svc('/srv/other/llmCouncil', '/usr/bin/python3 /srv/other/llmCouncil/api.py --port 8099');
  ok(here.command === 'python3 api.py --config config.json --port 8099' && here.cwd === 'src/llmCouncil' && svc(root).cwd === '.'
    && away.cwd === 'llmCouncil' && away.command === 'python3 api.py --port 8099' && svc(null).cwd === null,
    `services: the command without paths, the working directory relative to the checkout or by name: ${here.cwd}, ${away.cwd}, ${away.command}`);
}
{
  // sign-in exceptions on hand-made events (newest first, as audit.recent gives them)
  const at0 = Date.parse('2026-10-21T20:00:00Z');
  const at = (h) => new Date(at0 - h * 3600000).toISOString();
  const e = (h, event, user, ip, agent = UA_IPHONE) => ({ at: at(h), event, user, ip, agent });
  const events = [
    e(400, 'login.ok', 'jos', '2a02:a03f:8722:b301:1111:2222:3333:4444'),
    e(96, 'login.ok', 'jos', '2a02:a03f:8722:b301:322d:e75:225b:eb45'),                 // the same home /64, a rotated address
    ...[10, 9.99, 9.98, 9.97, 9.96].map((h) => e(h, 'login.failed', 'Jos', '192.0.2.66', 'curl/8')),   // five: locked
    e(9.9, 'login.locked', 'jos', '192.0.2.66', 'curl/8'), e(9.8, 'login.locked', 'jos', '192.0.2.66', 'curl/8'),
    ...[8, 7.99, 7.98, 7.97].map((h) => e(h, 'login.failed', 'eve', '192.0.2.7', 'curl/8')),             // four, then a sign-in clears them
    e(7.9, 'login.ok', 'eve', '192.0.2.7', 'curl/8'), e(7.8, 'login.failed', 'eve', '192.0.2.7', 'curl/8'),
    e(72, 'login.ok', 'test', '198.51.100.5'), e(48, 'user.deleted', 'test', '198.51.100.1'), e(2, 'user.added', 'test', '198.51.100.1'),
    e(1, 'login.ok', 'test', '198.51.100.5'),                                                           // the new account's first
  ].sort((a, b) => (a.at < b.at ? 1 : -1));
  const ex = system.exceptionsOf(events, at0).items;
  const kinds = (user) => ex.filter((x) => x.user.toLowerCase() === user).map((x) => x.kind).join();
  ok(ex.filter((x) => x.kind === 'locked out').length === 1 && kinds('jos') === 'refused while locked,refused while locked,locked out,failed sign-in,failed sign-in,failed sign-in,failed sign-in,failed sign-in',
    `exceptions: one lockout when the failures reach the limit, each refused attempt apart: ${kinds('jos')}`);
  ok(!ex.some((x) => x.user === 'eve' && x.kind === 'locked out'), 'exceptions: a sign-in clears the failures, as auth.js does: no lockout');
  ok(!ex.some((x) => x.user === 'jos' && /new|first/.test(x.kind)), 'exceptions: a rotated IPv6 address in a known /64 is not a new address');
  ok(kinds('test') === 'first sign-in on record,first sign-in on record', 'exceptions: a name deleted and added again starts afresh');
  const testUser = { name: 'test', role: 'user', createdAt: at(2), lastLoginAt: null, lastLoginIp: null, logins: null };
  const pv = system.peopleView({ users: [testUser], adminName: 'boss', events: events.filter((x) => x.at < at(1.5)), lastLogins: { test: events.find((x) => x.user === 'test' && x.event === 'login.ok' && x.at < at(2)) },
    total: { events: events.length, oldest: events[events.length - 1].at }, now: new Date(at0) });
  ok(pv.users[1].lastSignIn === null && pv.tiles.lockouts === 1 && pv.tiles.refusedWhileLocked === 2,
    'people: a re-added name does not inherit the deleted account\'s sign-in; lockouts and refused attempts counted apart');
  const fifteen = Array.from({ length: 15 }, (_, k) => e(0.5 - k / 100, 'login.failed', 'test', '192.0.2.9', 'curl/8'));
  const acc = system.accountView({ name: 'test', role: 'user', open: false }, { record: testUser, events: fifteen, lastLogin: events.find((x) => x.user === 'test' && x.event === 'login.ok') });
  ok(acc.lastSignIn && acc.lastSignIn.at === at(1).replace(/\.\d{3}Z$/, 'Z') && acc.lastSignIn.address === '198.51.100.5',
    'account: the last sign-in survives 15 failed attempts on the name');
}

// --- users and the audit trail on disk -----------------------------------------------------------------------------
const NOW = Date.now();
const ago = (hours) => new Date(NOW - hours * 3600000).toISOString();
const seed = (at, event, user, ip, agent, detail) => JSON.stringify({ at, event, user, ip, agent, ...(detail ? { detail } : {}) });
fs.writeFileSync(audit.FILE, [
  seed(ago(240), 'login.ok', 'reader', '198.51.100.1', UA_WIN(141)),         // 10 days ago: the device and address it uses
  seed(ago(200), 'login.failed', 'reader', '192.0.2.66', 'curl/8.5.0'),       // failed, but outside the 7 days
  seed(ago(72), 'login.ok', ADMIN_NAME, '198.51.100.2', UA_IPHONE),           // the administrator's first on record
  seed(ago(48), 'login.ok', 'reader', '198.51.100.1', UA_WIN(142)),           // same device after a browser update, same address
  seed(ago(30), 'login.failed', 'reader', '192.0.2.66', 'curl/8.5.0'),
  seed(ago(29), 'login.failed', 'reader', '192.0.2.66', 'curl/8.5.0'),
  seed(ago(28), 'login.locked', 'reader', '192.0.2.66', 'curl/8.5.0'),
  seed(ago(24), 'login.ok', 'reader', '198.51.100.77', UA_WIN(142)),          // a new address
  seed(ago(20), 'login.ok', 'reader', '198.51.100.77', UA_IPHONE),            // a new device
].join('\n') + '\n', { mode: 0o600 });
const seeded = auditEvents().length;
ok(auth.addUser({ user: ADMIN, ip: '203.0.113.9', headers: {} }, 'reader', 'reader-password-1').ok, 'a stand-in account for the reader');
{
  // the account is older than the sign-ins above (a sign-in from before an account's creation is not its own)
  const file = JSON.parse(fs.readFileSync(auth.USERS_FILE, 'utf-8'));
  file.users.find((u) => u.name === 'reader').createdAt = ago(300);
  fs.writeFileSync(auth.USERS_FILE, JSON.stringify(file), { mode: 0o600 });
}

// --- the jobs copy: a stand-in schedule ------------------------------------------------------------------------------
const MARK = (name, at, sha) => `=== scheduler: ${name} started ${at} ${sha} UTC ===`;
const DONE = (name, status, seconds) => `=== scheduler: ${name} finished status=${status} after ${seconds}s ===`;
const BOOST = 'Weekly tuning: boosting models';
const filler = Array.from({ length: 300 }, (_, k) => `trial ${k} scored 0.${k}`);
fs.writeFileSync(path.join(JOBS_ROOT, 'log', 'hyperoptBoost.log'), [
  'a line before any marker',
  MARK(BOOST, '2026-10-10 21:00:00', 'aaa1111'),
  '/root/sequencePredictor/src/Helpers.py:10: ConvergenceWarning: lbfgs failed to converge',
  DONE(BOOST, 0, 3600),
  MARK(BOOST, '2026-10-17 21:00:00', 'bbb2222'),
  ...Array(3).fill('/root/sequencePredictor/src/Helpers.py:10: ConvergenceWarning: lbfgs failed to converge'),
  '/root/sequencePredictor/src/Boost.py:20: UserWarning: a new thing in /root/sequencePredictor/data/x.json',
  'LOCKBOX: 30 draws withheld',
  'tuning  10%\r tuning  50%\r tuning 100% done',
  ...filler,
  'the last line of run two',
  DONE(BOOST, 0, 4200),
  MARK(BOOST, '2026-10-18 21:00:00', 'ccc3333'),
  'run three was killed here',
  MARK(BOOST, '2026-10-19 21:00:00', 'ddd4444'),
  'run four is still writing',
  ' create mode 100644 data/database/pick3/2026-10-20.json',
  'saved 2026/10/21',
].join('\n') + '\n');
const T = (iso) => Date.parse(iso);
const RUN1 = T('2026-10-10T21:00:00Z') + 2000;
const RUN2 = T('2026-10-17T21:00:00Z') + 2000;
const RUN3 = T('2026-10-18T21:00:00Z') + 2000;
const RUN4 = T('2026-10-19T21:00:00Z') + 2000;
const history = [   // newest first, as jobs.js keeps it
  { at: NOW - 2 * 3600000, finishedAt: NOW - 3600000, key: 'predictor', name: 'Daily predictor', reason: 'schedule 09:00', seconds: 3600, exit: 0, sha: 'eee5555', pid: 11, note: null },
  { at: RUN3, finishedAt: RUN3 + 600000, key: 'hyperoptBoost', name: BOOST, reason: 'triggered', seconds: 600, exit: null, sha: 'ccc3333', pid: 12, note: 'the job disappeared without writing an exit code (killed?)' },
  { at: T('2026-10-18T10:00:00Z'), finishedAt: T('2026-10-18T12:00:00Z'), key: 'nullControls', name: 'Weekly controls: null histories (Q0)', reason: 'x', seconds: 7200, exit: 1, sha: null, pid: 13, note: null },
  { at: RUN2, finishedAt: RUN2 + 4200000, key: 'hyperoptBoost', name: BOOST, reason: 'triggered', seconds: 4200, exit: 0, sha: 'bbb2222', pid: 14, note: null },
  { at: T('2026-10-17T09:00:00Z'), key: 'weeklyTuning', name: 'Weekly tuning chain', reason: 'x', dry: true, occurrence: 'weeklyTuning@2026-W42' },
  { at: T('2026-10-16T09:00:00Z'), finishedAt: T('2026-10-16T15:00:00Z'), key: 'predictor', name: 'Daily predictor', reason: 'x', seconds: 21600, exit: null, blocked: true,
    note: 'blocked: process.lock held by PID 9 (Predictor.py) for over 6.0 h - the job is still queued, nothing has been skipped' },
];
fs.writeFileSync(path.join(CONFIG_DIR, 'scheduler.json'), JSON.stringify({ version: 1, occurrences: {}, queue: [], current: null, history,
  baselinedAt: 'x', baselinedPlans: { weeklyTuning: 'x', weeklyControls: 'x' } }));
fs.writeFileSync(path.join(JOBS_ROOT, 'process.lock'), String(process.pid));   // held by this test: nothing queued can start
const logged = [];
const quiet = console.log;
console.log = (...a) => { logged.push(a.join(' ')); };
// the expected 500s log their error to the console: kept out of the test's output while they are asked for
const loud = console.error;
const quietErrors = (on) => { console.error = on ? (...a) => { logged.push(a.join(' ')); } : loud; };
fs.writeFileSync(path.join(CONFIG_DIR, 'scheduler.disabled'), '');   // off while it starts: nothing is due
const jobs = require(path.join(JOBS_ROOT, 'jobs.js'));
jobs.start();
jobs.stop();   // only its timer: the state stays loaded, and no tick runs while the test does
fs.unlinkSync(path.join(CONFIG_DIR, 'scheduler.disabled'));
const services = require(path.join(SVC_ROOT, 'services.js'));
const councilLog = path.join(SVC_ROOT, 'log', 'councilApi.log');
fs.writeFileSync(councilLog, [
  ...Array.from({ length: 30 }, (_, k) => `20:0${k % 10}:00 INFO    127.0.0.1 - "GET /health HTTP/1.1" 200 -`),
  '2026-10-21T20:03:07.703Z [supervisor] starting: python3 api.py --config config.json --port 8099 (cwd /root/sequencePredictor/src/llmCouncil)',
  '20:03:07 INFO    council api on http://127.0.0.1:8099',
  '20:04:16 INFO    127.0.0.1 - "GET /endpoints HTTP/1.1" 200 -',
].join('\n') + '\n');
ok(services.SERVICES[0].logFile === councilLog && jobs.JOBS.length === 11, 'the copies log into their own temp dirs');

// --- the app under test ------------------------------------------------------------------------------------------
const app = standIn();
api.install(app, { dataDir: path.join(tmp, 'markets') });
system.install(app, { root: JOBS_ROOT, jobs, services, jsonBody });
admin.install(app, { root: JOBS_ROOT, jobs, services, audit });

const ADMIN_GETS = ['/api/admin/jobs', `/api/admin/jobs/run/hyperoptBoost-${RUN2}/log`, '/api/admin/people'];
const ADMIN_POSTS = ['/api/admin/jobs/run', '/api/admin/jobs/cancel', '/api/admin/jobs/stop', '/api/admin/services/councilApi/start',
  '/api/admin/people/add', '/api/admin/people/password', '/api/admin/people/delete'];
const READER_POSTS = ['/api/account/password', '/api/whats-new/seen', '/api/whats-new/replay'];

async function main() {
  // --- /api/me serves the form token ---
  const me = await call(app, 'GET', '/api/me', { user: USER });
  ok(me.body.csrf === 'tok-user' && me.body.user.name === 'reader' && me.body.user.admin === false, '/api/me: the reader and the session\'s form token');

  // --- the guard: signed out 401, a user 403, nothing done ---
  const queueBefore = JSON.stringify(jobs._internals.state().queue);
  const usersBefore = fs.readFileSync(auth.USERS_FILE, 'utf-8');
  for (const url of ADMIN_GETS) {
    const out = await call(app, 'GET', url, {});
    const forUser = await call(app, 'GET', url, { user: USER });
    ok(out.status === 401 && out.body.error === 'login required' && forUser.status === 403 && forUser.body.error === 'admin only'
      && forUser.headers['Cache-Control'] === 'no-store', `GET ${url}: 401 signed out, a JSON 403 for a user`);
  }
  for (const url of ADMIN_POSTS) {
    const body = withToken(USER, { key: 'predictor', username: 'mallory', password: 'long-enough-1', runId: 'x' });
    const out = await call(app, 'POST', url, { body });
    const forUser = await call(app, 'POST', url, { user: USER, body });
    ok(out.status === 401 && forUser.status === 403 && forUser.body.error === 'admin only', `POST ${url}: 401 signed out, 403 for a user even with their own token`);
  }
  for (const url of ['/api/account', '/api/whats-new']) ok((await call(app, 'GET', url, {})).status === 401, `GET ${url}: 401 signed out`);
  for (const url of READER_POSTS) ok((await call(app, 'POST', url, { body: { _csrf: 'x' } })).status === 401, `POST ${url}: 401 signed out`);
  ok(JSON.stringify(jobs._internals.state().queue) === queueBefore && fs.readFileSync(auth.USERS_FILE, 'utf-8') === usersBefore && !spawned.length,
    'a refused request changed nothing: no queue entry, no account, no process');

  // --- the form token: missing or wrong is a JSON 403, and changes nothing ---
  for (const url of ADMIN_POSTS) {
    const none = await call(app, 'POST', url, { user: ADMIN, body: { key: 'predictor', username: 'mallory', password: 'long-enough-1', runId: 'x' } });
    const wrong = await call(app, 'POST', url, { user: ADMIN, body: { _csrf: 'tok-user', key: 'predictor', username: 'mallory', password: 'long-enough-1' } });
    ok(none.status === 403 && wrong.status === 403 && /form token/.test(none.body.error) && /form token/.test(wrong.body.error), `POST ${url}: 403 without the token or with another session's`);
  }
  for (const url of READER_POSTS) {
    const wrong = await call(app, 'POST', url, { user: USER, body: { _csrf: 'tok-admin', current: 'reader-password-1', password: 'x'.repeat(9), confirm: 'x'.repeat(9), ids: ['a'] } });
    ok(wrong.status === 403 && /form token/.test(wrong.body.error), `POST ${url}: 403 with a token that is not the reader's`);
  }
  ok(JSON.stringify(jobs._internals.state().queue) === queueBefore && fs.readFileSync(auth.USERS_FILE, 'utf-8') === usersBefore && !spawned.length
    && auditEvents().length === seeded + 1, 'no token, no change: the queue, the accounts, the processes and the audit log as they were');
  const bad = await call(app, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: BAD_JSON });
  ok(bad.status === 400 && bad.body.error === 'the request body is not valid JSON', 'a body that is not JSON: a JSON 400, never an HTML page');

  // --- GET /api/admin/jobs ---
  const j = (await call(app, 'GET', '/api/admin/jobs', { user: ADMIN })).body;
  ok(j.mode.mode === 'dry' && j.lock.held && j.lock.pid === process.pid && j.lock.key === null && j.lock.ours === false && j.running === null
    && j.lock.script && !j.lock.script.includes('/'), `jobs: the mode, the lock and who holds it (not a job of the schedule, its script by file name), nothing running: ${j.lock.script}`);
  ok(j.jobs.length === 11 && j.jobs.every((x) => x.canRun && x.refusal === null && x.log && !x.log.includes('/')), 'jobs: every job may run now, its log named by file name only');
  const boost = j.jobs.find((x) => x.key === 'hyperoptBoost');
  ok(boost.usual.median === 65 && boost.usual.n === 2 && boost.usual.basis === 'earlier runs in the job log' && boost.last.result === 'unknown' && boost.trigger.includes('Saturday'),
    `jobs: the usual duration is the median of the earlier successful runs in the log (60 and 70 min): ${JSON.stringify(boost.usual)}`);
  const pred = j.jobs.find((x) => x.key === 'predictor');
  ok(pred.daily && pred.nextAt && pred.trigger === null && pred.usual.basis === 'earlier runs in the scheduler history' && pred.usual.n === 1 && pred.last.result === 'ok',
    'jobs: the predictor\'s next slot as an instant, its usual duration from the kept runs');
  ok(j.runs.kept === 40 && j.runs.shown === history.length && j.runs.ran === 4 && j.runs.items.map((r) => r.kind).join() === 'run,run,run,run,dry,blocked'
    && j.runs.items[5].result === 'waited on the lock', 'jobs: every kept entry, the dry and the blocked ones named as such, with the count of real runs');
  const r2 = j.runs.items.find((r) => r.id === `hyperoptBoost-${RUN2}`);
  ok(r2.minutes === 70 && r2.usual.median === 60 && r2.overUsual === true && r2.warnings.signatures === 3 && r2.warnings.new === 2,
    `jobs: a run against the runs before it, with its warning signatures and how many are new: ${JSON.stringify(r2)}`);
  ok(j.lastFailure.id === `hyperoptBoost-${RUN3}` && j.lastFailure.result === 'unknown' && /disappeared/.test(j.lastFailure.note),
    'jobs: the last failure is the newest run that did not exit 0 (a killed one counts)');
  const council = j.services[0];
  ok(council.key === 'councilApi' && council.status === 'stopped' && council.healthy === false && council.actions.start && council.actions.restart && !council.actions.stop
    && council.checkedAt && council.upMinutes === null, `services: the Council API, stopped, its last health check, the buttons its state allows: ${JSON.stringify(council.actions)}`);
  ok(council.log.healthDropped === 30 && council.log.lines.length === 3 && council.log.lines[0].includes('(cwd llmCouncil)') && council.log.file === 'councilApi.log',
    'services: the log tail without the /health probes, paths cut to file names');
  ok(council.command === 'python3 api.py --config config.json --port 8099' && council.cwd === 'llmCouncil',
    `services: what it runs, and where by name alone (this copy lies outside the checkout the view is built for): ${council.command} in ${council.cwd}`);
  ok(j.runs.items.find((r) => r.id === `hyperoptBoost-${RUN2}`).sha === 'bbb2222' && j.runs.items.find((r) => r.key === 'nullControls').sha === null,
    'jobs: each run\'s commit, null when none was recorded');
  const jText = JSON.stringify(j);
  ok(!jText.includes(tmp) && !jText.includes('/root/') && !/NaN|undefined/.test(jText), 'jobs: no absolute path, no NaN');

  // --- GET /api/admin/jobs/run/:id/log ---
  const log2 = (await call(app, 'GET', `/api/admin/jobs/run/hyperoptBoost-${RUN2}/log`, { user: ADMIN })).body;
  ok(log2.warnings.map((w) => `${w.sig}:${w.n}:${w.new}`).join(' | ') === 'ConvergenceWarning (Helpers.py):3:false | LOCKBOX:1:true | UserWarning (Boost.py):1:true',
    `log: the warning signatures first, with how often and whether new: ${log2.warnings.map((w) => w.sig).join(', ')}`);
  ok(log2.warnings.every((w) => w.line && !w.line.includes('/root/')) && log2.warnings[2].line === 'Boost.py:20: UserWarning: a new thing in x.json',
    'log: each signature with its newest line, paths cut to file names');
  ok(log2.tail.length + log2.warnings.length === 200 && log2.tail[log2.tail.length - 1] === 'the last line of run two' && log2.lines.of === 307 && log2.lines.whole,
    `log: at most 200 lines, read from the end of the run's own slice (not the next run's): ${log2.tail.length} + ${log2.warnings.length}, of ${log2.lines.of}`);
  ok(log2.result === 'ok' && log2.minutes === 70 && log2.inKeptRuns && log2.log === 'hyperoptBoost.log' && log2.sha === 'bbb2222', 'log: the run\'s own line');
  const progressed = (await call(app, 'GET', `/api/admin/jobs/run/hyperoptBoost-${RUN2}/log`, { user: ADMIN })).body;
  ok(!progressed.tail.some((l) => l.includes('\r')), 'log: a progress bar shows its last state');
  const log1 = (await call(app, 'GET', `/api/admin/jobs/run/hyperoptBoost-${RUN1}/log`, { user: ADMIN })).body;
  ok(log1.inKeptRuns === false && log1.result === 'ok' && log1.minutes === 60 && log1.tail.length === 1 && log1.warnings.length === 1 && log1.warnings[0].new === null,
    'log: a run older than the kept runs is read from its markers alone');
  const log3 = (await call(app, 'GET', `/api/admin/jobs/run/hyperoptBoost-${RUN3}/log`, { user: ADMIN })).body;
  ok(log3.tail.join('|') === 'run three was killed here' && log3.result === 'unknown', 'log: a killed run ends where the next run starts');
  ok((await call(app, 'GET', '/api/admin/jobs/run/nonsense/log', { user: ADMIN })).status === 400
    && (await call(app, 'GET', `/api/admin/jobs/run/bingo-${RUN2}/log`, { user: ADMIN })).status === 404
    && (await call(app, 'GET', `/api/admin/jobs/run/weeklyTuning-${T('2026-10-17T09:00:00Z')}/log`, { user: ADMIN })).status === 404
    && (await call(app, 'GET', `/api/admin/jobs/run/predictor-${NOW}/log`, { user: ADMIN })).body.error.startsWith('no log slice'),
    'log: a bad id 400, an unknown job or a dry entry 404, a run without a slice says why');

  // --- POST /api/admin/jobs/run | cancel ---
  const run = await call(app, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: withToken(ADMIN, { key: 'predictor' }) });
  const q = jobs._internals.state().queue;
  ok(run.status === 200 && run.body.message === 'Daily predictor: queued, first in the queue' && run.body.started === false && q.length === 1 && q[0].manual && q[0].reason === 'started by hand (boss)',
    'run: queued by hand, waiting behind the lock - nothing started, and the message says where it waits');
  ok(!spawned.length && logged.some((l) => /waiting for process.lock \(held by PID \d+/.test(l)), 'run: the queue waits for the lock and says so; no process was started');
  const again = await call(app, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: withToken(ADMIN, { key: 'predictor' }) });
  ok(again.status === 409 && again.body.error === 'Daily predictor: already running or queued', 'run: a job is never queued twice (409)');
  const jq = (await call(app, 'GET', '/api/admin/jobs', { user: ADMIN })).body;
  ok(jq.queue.length === 1 && jq.queue[0].manual && jq.jobs.find((x) => x.key === 'predictor').refusal === 'Daily predictor: already running or queued'
    && jq.jobs.find((x) => x.key === 'nullControls').canRun, 'jobs: the queue, and the Run now refusal on the queued job only');
  ok((await call(app, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: withToken(ADMIN, { key: 'rm -rf' }) })).status === 400, 'run: an unknown job is a 400');
  fs.writeFileSync(path.join(CONFIG_DIR, 'scheduler.disabled'), '');
  const off = await call(app, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: withToken(ADMIN, { key: 'nullControls' }) });
  const offView = (await call(app, 'GET', '/api/admin/jobs', { user: ADMIN })).body;
  fs.unlinkSync(path.join(CONFIG_DIR, 'scheduler.disabled'));
  ok(off.status === 409 && /schedule is disabled/.test(off.body.error) && offView.mode.mode === 'off' && offView.jobs.every((x) => !x.canRun),
    'run: refused while the schedule is off (409), and every job says so');
  const cancel = await call(app, 'POST', '/api/admin/jobs/cancel', { user: ADMIN, body: withToken(ADMIN, { key: 'predictor' }) });
  ok(cancel.status === 200 && cancel.body.message === 'Daily predictor: removed from the queue' && jobs._internals.state().queue.length === 0, 'cancel: out of the queue, the job named');
  ok((await call(app, 'POST', '/api/admin/jobs/cancel', { user: ADMIN, body: withToken(ADMIN, { key: 'predictor' }) })).status === 409
    && (await call(app, 'POST', '/api/admin/jobs/cancel', { user: ADMIN, body: withToken(ADMIN, { key: 'bingo' }) })).status === 400, 'cancel: not queued 409, unknown 400');

  // --- POST /api/admin/jobs/stop: the refusals through the route, the signal on the copy with a stand-in kill ---
  ok((await call(app, 'POST', '/api/admin/jobs/stop', { user: ADMIN, body: withToken(ADMIN, { runId: 'nothing-runs' }) })).body.error === 'that job is no longer running',
    'stop: nothing running is a 409');
  const state = jobs._internals.state();
  state.current = { runId: 'hyperoptBoost-2026-10-19T21-00-00-000Z', key: 'hyperoptBoost', name: BOOST, reason: 'test', startedAt: RUN4, pid: null,
    pidFile: path.join(CONFIG_DIR, 'runs', 'x.pid'), exitFile: path.join(CONFIG_DIR, 'runs', 'x.exit'), logFile: path.join(JOBS_ROOT, 'log', 'hyperoptBoost.log'), sha: 'ddd4444', mode: 'dry' };
  const noPid = await call(app, 'POST', '/api/admin/jobs/stop', { user: ADMIN, body: withToken(ADMIN, { runId: state.current.runId }) });
  state.current.pid = 2147483646;   // no such process: alive() reads /proc and finds nothing
  const gone = await call(app, 'POST', '/api/admin/jobs/stop', { user: ADMIN, body: withToken(ADMIN, { runId: state.current.runId }) });
  const behind = await call(app, 'POST', '/api/admin/jobs/stop', { user: ADMIN, body: withToken(ADMIN, { runId: 'hyperoptBoost-an-older-run' }) });
  ok(noPid.status === 409 && /not reported its PID/.test(noPid.body.error) && gone.status === 409 && /already gone/.test(gone.body.error)
    && behind.status === 409 && /no longer running/.test(behind.body.error), 'stop: no PID yet, already gone, or another run than the page shows: 409, nothing signalled');
  const signals = [];
  const stopped = jobs.stopRun(state.current.runId, 'boss', { kill: (pid, sig) => signals.push([pid, sig]), isAlive: () => true });
  ok(stopped.ok && stopped.message === `${BOOST}: stopping` && signals.length === 1 && signals[0][0] === -2147483646 && signals[0][1] === 'SIGTERM'
    && fs.readFileSync(path.join(JOBS_ROOT, 'log', 'scheduler.log'), 'utf-8').includes('stopped by boss'), 'stop: SIGTERM to the run\'s whole process group, said in the scheduler log');
  const refusedKill = jobs.stopRun(state.current.runId, 'boss', { kill: () => { throw new Error('EPERM'); }, isAlive: () => true });
  ok(!refusedKill.ok && refusedKill.status === 500 && refusedKill.message === 'could not stop it: EPERM', 'stop: a signal the system refuses is a 500 with its reason');
  const jr = (await call(app, 'GET', '/api/admin/jobs', { user: ADMIN })).body;
  ok(jr.running.id === `hyperoptBoost-${RUN4}` && jr.running.runId === state.current.runId && jr.running.sha === 'ddd4444' && jr.running.usual.median === 65 && jr.running.log.length === 20
    && jr.running.log.every((l) => !l.includes('/root/')) && jr.jobs.find((x) => x.key === 'hyperoptBoost').refusal === `${BOOST}: already running or queued`
    && jr.running.log.slice(-2).join('|') === ' create mode 100644 data/database/pick3/2026-10-20.json|saved 2026/10/21',
    'jobs: the running job with its run id for Stop, elapsed against the usual, its last 20 log lines without absolute paths (relative ones and dates kept)');
  const log4 = (await call(app, 'GET', `/api/admin/jobs/run/hyperoptBoost-${RUN4}/log`, { user: ADMIN })).body;
  ok(log4.running && log4.result === 'running' && log4.tail.join('|') === 'run four is still writing| create mode 100644 data/database/pick3/2026-10-20.json|saved 2026/10/21',
    'log: the running job\'s slice, read to the end of the file, relative paths and dates as written');
  state.current = null;
  // the route's happy path, on a stand-in schedule that records what it is asked
  const asked = [];
  const fakeApp = standIn();
  system.install(fakeApp, { root: JOBS_ROOT, jsonBody, jobs: { JOBS: [], stopRun: (id, by) => { asked.push([id, by]); return { ok: true, status: 200, message: 'X: stopping' }; } } });
  const fakeStop = await call(fakeApp, 'POST', '/api/admin/jobs/stop', { user: ADMIN, body: withToken(ADMIN, { runId: 'r-1' }) });
  ok(fakeStop.status === 200 && fakeStop.body.ok && fakeStop.body.message === 'X: stopping' && asked.join() === 'r-1,boss', 'stop: the route passes the run id and who asked');
  ok((await call(fakeApp, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: withToken(ADMIN, { key: 'predictor' }) })).status === 503, 'a server without the schedule says so (503)');
  const brokenApp = standIn();
  system.install(brokenApp, { root: JOBS_ROOT, jsonBody, jobs: { JOBS: [], runNow: () => { throw new Error(`EACCES: permission denied, mkdir '${JOBS_ROOT}/log'`); } },
    services: { SERVICES: [{ key: 'councilApi' }], ACTIONS: ['start'], act: async () => ({ ok: false, status: 500, message: `Council API: cannot open ${JOBS_ROOT}/log/councilApi.log: EACCES: permission denied, open '${JOBS_ROOT}/log/councilApi.log'` }) } });
  quietErrors(true);
  const thrown = await call(brokenApp, 'POST', '/api/admin/jobs/run', { user: ADMIN, body: withToken(ADMIN, { key: 'predictor' }) });
  quietErrors(false);
  const svcFail = await call(brokenApp, 'POST', '/api/admin/services/councilApi/start', { user: ADMIN, body: withToken(ADMIN) });
  ok(thrown.status === 500 && thrown.body.error && !thrown.raw.includes('/') && svcFail.status === 500 && svcFail.body.error === "Council API: cannot open councilApi.log: EACCES: permission denied, open 'councilApi.log'",
    `an action that throws is a JSON 500, and a refusal names no absolute path: ${thrown.raw} ${svcFail.raw}`);

  // --- POST /api/admin/services/:name/:action ---
  const svc = (action, name = 'councilApi') => call(app, 'POST', `/api/admin/services/${name}/${action}`, { user: ADMIN, body: withToken(ADMIN) });
  const stopIdle = await svc('stop');
  ok(stopIdle.status === 409 && stopIdle.body.error === 'Council API: already stopped' && !killed.length, 'services: stopping a stopped service changes nothing (409)');
  const started = await svc('start');
  ok(started.status === 200 && started.body.message === 'Council API: started (pid 4242)' && spawned.length === 1 && spawned[0][0] === 'python3' && spawned[0][1][0] === 'api.py',
    'services: start (the stand-in child)');
  const up = (await call(app, 'GET', '/api/admin/jobs', { user: ADMIN })).body.services[0];
  ok(up.status === 'running' && up.pid === 4242 && up.upMinutes === 0 && !up.actions.start && up.actions.stop && up.actions.restart, 'services: running, its uptime, Start greyed');
  ok((await svc('start')).status === 409, 'services: starting a running service is a 409');
  const restarted = await svc('restart');
  ok(restarted.status === 200 && killed.length === 1 && spawned.length === 2, 'services: restart stops the child and starts a new one');
  const down = await svc('stop');
  await new Promise((resolve) => setImmediate(resolve));
  ok(down.status === 200 && down.body.message === 'Council API: stopping' && killed.length === 2 && killed[1] === 'SIGTERM', 'services: stop sends SIGTERM');
  ok((await svc('reboot')).status === 404 && (await svc('start', 'bingo')).status === 404, 'services: an unknown action or service is a 404');
  ok(fs.readFileSync(councilLog, 'utf-8').includes('stop requested by boss: stopping'), 'services: every action is said in the service\'s own log');
  ok(!spawned.some(([command]) => command === 'setsid'), 'no job was ever launched');

  // --- GET /api/admin/people ---
  const p = (await call(app, 'GET', '/api/admin/people', { user: ADMIN })).body;
  ok(p.users.map((u) => `${u.name}:${u.role}:${u.source}`).join() === 'boss:admin:environment,reader:user:users.json' && p.auth.writable,
    'people: the administrator from the environment and the accounts on file');
  ok(p.users[1].lastSignIn.from === 'activity' && p.users[1].lastSignIn.device === 'Safari on iOS' && p.users[0].lastSignIn.at === new Date(NOW - 72 * 3600000).toISOString().replace(/\.\d{3}Z$/, 'Z'),
    'people: the last sign-in per name, the administrator included');
  ok(p.tiles.users === 2 && p.tiles.failed === 2 && p.tiles.lockouts === 0 && p.tiles.refusedWhileLocked === 1 && p.tiles.newDeviceOrAddress === 3 && p.tiles.days === 7
    && p.tiles.scanned.all && p.tiles.scanned.complete, `people: 7 days of failed sign-ins, lockouts, refused attempts and new devices or addresses, over every event: ${JSON.stringify(p.tiles)}`);
  ok(p.exceptions.items.map((x) => x.kind).join() === 'new device,new address,refused while locked,failed sign-in,failed sign-in,first sign-in on record',
    `people: the exceptions newest first; a browser update is not a new device: ${p.exceptions.items.map((x) => x.kind).join()}`);
  ok(p.activity.shown === seeded + 1 && p.activity.of === seeded + 1 && p.activity.items[0].event === 'user.added' && p.activity.items[0].detail === 'by boss',
    'people: the activity with its true count');
  const limited = (await call(app, 'GET', '/api/admin/people?limit=3', { user: ADMIN })).body.activity;
  ok(limited.shown === 3 && limited.items.length === 3 && limited.of === seeded + 1, 'people: ?limit= shows fewer, the count stays the true one');
  ok((await call(app, 'GET', '/api/admin/people?limit=all', { user: ADMIN })).body.activity.items.length === seeded + 1
    && (await call(app, 'GET', '/api/admin/people?limit=0', { user: ADMIN })).status === 400 && (await call(app, 'GET', '/api/admin/people?limit=x', { user: ADMIN })).status === 400,
    'people: ?limit=all, and a bad limit is a 400');

  // --- POST /api/admin/people/add | password | delete ---
  const people = (action, body) => call(app, 'POST', `/api/admin/people/${action}`, { user: ADMIN, body: withToken(ADMIN, body) });
  const refusals = [
    ['add', { username: 'x', password: 'long-enough-1' }, 400, /User name must be/],
    ['add', { username: 'carol', password: 'short' }, 400, /at least 8/],
    ['add', { username: 'BOSS', password: 'long-enough-1' }, 409, /administrator account/],
    ['add', { username: 'Reader', password: 'long-enough-1' }, 409, /already exists/],
    ['password', { username: 'reader', password: 'short' }, 400, /at least 8/],
    ['password', { username: 'nobody', password: 'long-enough-1' }, 404, /not found/],
    ['password', { username: ADMIN_NAME, password: 'long-enough-1' }, 409, /set in \.env/],
    ['delete', { username: 'nobody' }, 404, /not found/],
    ['delete', { username: ADMIN_NAME }, 409, /cannot be deleted/],
  ];
  for (const [action, body, status, text] of refusals) {
    const r = await people(action, body);
    ok(r.status === status && text.test(r.body.error), `people/${action} ${JSON.stringify(body.username)}: ${status} ${r.body.error}`);
  }
  ok(usersOnDisk().length === 1 && auditEvents().length === seeded + 1, 'people: a refused change writes neither an account nor an audit entry');
  const blocker = path.join(tmp, 'not-a-dir');
  fs.writeFileSync(blocker, '');
  STUB_CONFIG.CONFIG_DIR = path.join(blocker, 'config');   // config/ cannot be made: saveUsers throws ENOTDIR
  quietErrors(true);
  const unwritable = await people('add', { username: 'carol', password: 'carol-password-1' });
  quietErrors(false);
  STUB_CONFIG.CONFIG_DIR = CONFIG_DIR;
  ok(unwritable.status === 500 && unwritable.body.error && !unwritable.raw.includes('/') && !usersOnDisk().some((u) => u.name === 'carol'),
    `people/add with config/ unwritable: a JSON 500 without the server's paths, nothing added: ${unwritable.raw}`);
  const added = await people('add', { username: 'carol', password: 'carol-password-1' });
  ok(added.status === 200 && added.body.message === 'User carol added.' && usersOnDisk().some((u) => u.name === 'carol')
    && lastEvent().event === 'user.added' && lastEvent().user === 'carol' && lastEvent().detail === 'by boss', 'people/add: the account, and its audit entry');
  const before = usersOnDisk().find((u) => u.name === 'carol').hash;
  const reset = await people('password', { username: 'carol', password: 'carol-password-2' });
  const carol = usersOnDisk().find((u) => u.name === 'carol');
  ok(reset.status === 200 && carol.hash !== before && carol.passwordChangedAt && lastEvent().event === 'password.admin', 'people/password: a new hash, the change time, the audit entry');
  const removed = await people('delete', { username: 'carol' });
  ok(removed.status === 200 && !usersOnDisk().some((u) => u.name === 'carol') && lastEvent().event === 'user.deleted', 'people/delete: gone, and said');
  STUB_CONFIG.ADMIN_USER = ''; STUB_CONFIG.ADMIN_PASSWORD = '';   // open access: accounts are read-only
  const openAdd = await call(app, 'POST', '/api/admin/people/add', { user: OPEN, body: withToken(OPEN, { username: 'eve', password: 'long-enough-1' }) });
  const openView = (await call(app, 'GET', '/api/admin/people', { user: OPEN })).body;
  STUB_CONFIG.ADMIN_USER = ADMIN_NAME; STUB_CONFIG.ADMIN_PASSWORD = 'correct horse battery staple';
  ok(openAdd.status === 403 && /WEB_USER/.test(openAdd.body.error) && !openView.auth.writable && openView.users.length === 1 && !usersOnDisk().some((u) => u.name === 'eve'),
    'people: with authentication off no account is created (403), and the page says it is read-only');

  // --- GET /api/account, POST /api/account/password ---
  const acc = (await call(app, 'GET', '/api/account', { user: USER })).body;
  ok(acc.name === 'reader' && acc.role === 'user' && acc.source === 'users.json' && acc.created && acc.password.canChange && acc.password.minLength === 8
    && acc.activity.items.length === 9 && acc.activity.items.every((x) => x.user === undefined) && acc.lastSignIn.device === 'Safari on iOS',
    'account: the reader\'s own record and the events on their name only');
  const accAdmin = (await call(app, 'GET', '/api/account', { user: ADMIN })).body;
  ok(accAdmin.role === 'admin' && accAdmin.source === 'environment' && !accAdmin.password.canChange && /\.env/.test(accAdmin.password.why) && accAdmin.activity.items.length === 1,
    'account: the administrator\'s password is the environment\'s');
  const own = (body, who = USER) => call(app, 'POST', '/api/account/password', { user: who, body: withToken(who, body) });
  const wrongCurrent = await own({ current: 'guess', password: 'reader-password-2', confirm: 'reader-password-2' });
  ok(wrongCurrent.status === 400 && /not your current password/.test(wrongCurrent.body.error) && lastEvent().event === 'password.rejected', 'account: a wrong current password is a 400, recorded');
  ok((await own({ current: 'reader-password-1', password: 'short', confirm: 'short' })).status === 400
    && (await own({ current: 'reader-password-1', password: 'reader-password-2', confirm: 'reader-password-3' })).status === 400
    && (await own({ current: 'reader-password-1', password: 'reader-password-1', confirm: 'reader-password-1' })).status === 400, 'account: too short, not confirmed, unchanged: 400');
  ok((await own({ current: 'x', password: 'long-enough-1', confirm: 'long-enough-1' }, ADMIN)).status === 409
    && (await own({ current: 'x', password: 'long-enough-1', confirm: 'long-enough-1' }, OPEN)).status === 409, 'account: the administrator and open access cannot change one here (409)');
  ok(auth.addUser({ user: ADMIN, ip: '203.0.113.9', headers: {} }, 'guesser', 'guesser-password-1').ok, 'a stand-in account whose password is guessed at');
  const GUESSER = { name: 'guesser', role: 'user', open: false, csrf: 'tok-guesser' };
  const guessed = [];
  for (let k = 0; k < 5; k += 1) guessed.push((await own({ current: `guess-${k}`, password: 'x'.repeat(9), confirm: 'x'.repeat(9) }, GUESSER)).status);
  const sixth = await call(app, 'POST', '/api/account/password', { user: GUESSER, body: withToken(GUESSER, { current: 'guess-5', password: 'x'.repeat(9), confirm: 'x'.repeat(9) }), headers: {} });
  const guesserHash = usersOnDisk().find((u) => u.name === 'guesser').hash;
  const right = await own({ current: 'guesser-password-1', password: 'x'.repeat(9), confirm: 'x'.repeat(9) }, GUESSER);
  ok(guessed.join() === '400,400,400,400,400' && sixth.status === 429 && /Too many/.test(sixth.body.error) && right.status === 429
    && usersOnDisk().find((u) => u.name === 'guesser').hash === guesserHash, `account: the 6th wrong current password within 15 minutes is a 429, and then nothing changes: ${guessed.join()},${sixth.status},${right.status}`);
  const hashBefore = usersOnDisk().find((u) => u.name === 'reader').hash;
  const changed = await own({ current: 'reader-password-1', password: 'reader-password-2', confirm: 'reader-password-2' });
  ok(changed.status === 200 && /changed/.test(changed.body.message) && /^[0-9a-f]{32}$/.test(changed.body.csrf) && changed.body.csrf !== USER.csrf
    && /^sp_session=.+HttpOnly; SameSite=Strict/.test(changed.headers['Set-Cookie']) && usersOnDisk().find((u) => u.name === 'reader').hash !== hashBefore
    && lastEvent().event === 'password.self', 'account: changed, a fresh session and form token for this browser, the audit entry');

  // --- GET /api/whats-new, POST /api/whats-new/seen | replay ---
  const forAll = ANNOUNCEMENTS.filter((a) => a.audience !== 'admin');
  const wn = (await call(app, 'GET', '/api/whats-new', { user: USER })).body;
  ok(wn.items.length === forAll.length && wn.items.every((a) => a.audience === 'all' && a.seen === false) && wn.unseen === forAll.length
    && wn.tour.done === false && wn.pending.kind === 'tour' && wn.tour.steps.every((s) => s.title !== 'Users, Jobs and Activity'),
    'what\'s new: a user gets the notes for everyone and the tour without the administrator\'s step');
  const wnAdmin = (await call(app, 'GET', '/api/whats-new', { user: ADMIN })).body;
  ok(wnAdmin.items.length === ANNOUNCEMENTS.length && wnAdmin.tour.steps.length === whatsnew.TOUR.steps.length, 'what\'s new: the administrator gets every note and step');
  const adminLinks = (v) => v.items.filter((a) => a.link && /^\/admin(\/|$)/.test(a.link.href)).length;
  ok(adminLinks(wn) === 0 && adminLinks(wnAdmin) > 0 && wn.items.some((a) => a.link), 'what\'s new: a user never gets a link to an administrator\'s page; their other links stay');
  const adminOnly = ANNOUNCEMENTS.find((a) => a.audience === 'admin').id;
  const seen = await call(app, 'POST', '/api/whats-new/seen', { user: USER, body: withToken(USER, { ids: [forAll[0].id, adminOnly, 'nonsense'] }) });
  ok(seen.status === 200 && seen.body.seen.join() === forAll[0].id && seen.body.ignored.join() === `${adminOnly},nonsense`, 'seen: only the reader\'s own notes are recorded, the rest named');
  const after = (await call(app, 'GET', '/api/whats-new', { user: USER })).body;
  ok(after.items[0].seen === true && after.unseen === forAll.length - 1, 'seen: the note reads as seen');
  ok((await call(app, 'POST', '/api/whats-new/seen', { user: USER, body: withToken(USER, { ids: [] }) })).status === 400, 'seen: naming nothing is a 400');
  const ALLER = { name: 'aller', role: 'user', open: false, csrf: 'tok-aller' };
  const markAll = await call(app, 'POST', '/api/whats-new/seen', { user: ALLER, body: withToken(ALLER, { all: true }) });
  ok(markAll.status === 200 && markAll.body.seen.length === forAll.length && (await call(app, 'GET', '/api/whats-new', { user: ALLER })).body.unseen === 0,
    `seen: all marks every note the reader may see, past the 50 a list of ids may name: ${markAll.body.message}`);
  await call(app, 'POST', '/api/whats-new/seen', { user: USER, body: withToken(USER, { tour: true }) });
  ok((await call(app, 'GET', '/api/whats-new', { user: USER })).body.tour.done === true, 'seen: the tour done');
  ok((await call(app, 'POST', '/api/whats-new/replay', { user: USER, body: withToken(USER) })).status === 200
    && (await call(app, 'GET', '/api/whats-new', { user: USER })).body.tour.done === false, 'replay: the tour shows again');
  ok((await call(app, 'POST', '/api/whats-new/seen', { user: OPEN, body: withToken(OPEN, { ids: [forAll[0].id] }) })).status === 409
    && (await call(app, 'GET', '/api/whats-new', { user: OPEN })).body.unseen === null, 'what\'s new: open access has no account to record against (409)');
  ok(!spawned.some(([command]) => command === 'setsid'), 'still no job launched');

  // --- no hash or salt in any answer ---
  const secrets = usersOnDisk().flatMap((u) => [u.hash, u.salt]);
  ok(answers.length > 100 && answers.every((text) => !secrets.some((s) => text.includes(s)) && !/"(hash|salt)"/.test(text)),
    `no password hash or salt in any of the ${answers.length} answers`);
}

// --- production's own files, read-only, only when asked (SYSTEM_TEST_PROD=1): the views' size and shape ---------------
// The schedule and the job logs only - never the sign-in audit log (names, addresses, browsers). npm test never sets it.
let prodSummary = null;
function onProduction() {
  if (process.env.SYSTEM_TEST_PROD !== '1') return;
  const PROD = '/root/sequencePredictor';
  const schedFile = path.join(PROD, 'config', 'scheduler.json');
  if (!fs.existsSync(schedFile)) return;
  const sched = JSON.parse(fs.readFileSync(schedFile, 'utf-8'));
  const dirs = admin.resolveDirs({ root: PROD });
  const slices = Object.fromEntries(jobs.JOBS.map((j) => [j.key, admin.readSlices(dirs, j)]));
  // the status jobs.status({full: true}) would give for production's state, from its file (the lock and mode not read)
  const status = { mode: 'on', modeReason: 'production (read from its file)', lock: null, current: null, queue: sched.queue || [], kept: 40, history: sched.history || [],
    jobs: jobs.JOBS.map((job) => ({ key: job.key, name: job.name, description: job.description, command: `python3 ${job.script} ${job.args.join(' ')}`,
      logFile: path.join(PROD, 'log', job.log), next: 'x', nextAt: job.daily ? Date.now() : null, plan: job.plan || null, refusal: null })) };
  const view = system.jobsView({ status, jobs: jobs.JOBS, slices, services: [], serviceLogs: {} });
  const text = JSON.stringify(view);
  ok(!/NaN|undefined/.test(text) && !text.includes('/root/') && text.length < 40 * 1024, `production jobs view: ${text.length} bytes, no path, no NaN`);
  ok(view.runs.shown === (sched.history || []).length && view.jobs.filter((x) => x.usual).length >= 8, `production: every kept run, ${view.jobs.filter((x) => x.usual).length} jobs with a usual duration`);
  let largest = 0;
  view.runs.items.filter((r) => r.kind === 'run' && r.warnings).forEach((r) => {
    const l = system.runLogView({ id: r.id, jobs: jobs.JOBS, slices, history: status.history });
    const t = JSON.stringify(l);
    largest = Math.max(largest, t.length);
    ok(!l.error && l.tail.length + l.warnings.length <= 200 && !t.includes('/root/') && !/NaN/.test(t), `production log of ${r.id}: ${l.tail.length} + ${l.warnings.length} lines, ${t.length} bytes`);
  });
  const council = system.serviceLog(path.join(PROD, 'log', 'councilApi.log'));
  ok(council.lines.length <= 20 && council.lines.every((l) => !/GET \/health/.test(l) && !l.includes('/root/')), `production Council API log: ${council.lines.length} lines, ${council.healthDropped} health probes dropped`);
  prodSummary = `system.js on production: jobs ${text.length} bytes, ${view.runs.items.length} runs, the largest run log ${largest} bytes`;
}

// --- the same routes through express itself, when it can be loaded (NODE_PATH) --------------------------------------
async function throughExpress() {
  let express;
  try { express = require('express'); } catch (e) { return false; }
  const real = express();
  real.use(express.urlencoded({ extended: false, limit: '16kb' }));
  real.use((req, res, next) => { const who = req.get('x-test-user'); req.user = who === 'admin' ? ADMIN : (who === 'user' ? USER : undefined); next(); });
  system.install(real, { root: JOBS_ROOT, jobs, services });
  admin.install(real, { root: JOBS_ROOT, jobs, services, audit });
  const server = await new Promise((resolve) => { const s = real.listen(0, '127.0.0.1', () => resolve(s)); });
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const send = async (method, url, who, body, type = 'application/json') => {
      const r = await fetch(base + url, { method, headers: { ...(who ? { 'x-test-user': who } : {}), ...(body !== undefined ? { 'content-type': type } : {}) }, body });
      return { status: r.status, type: r.headers.get('content-type') || '', body: await r.text() };
    };
    ok((await send('GET', '/api/admin/jobs', 'user')).status === 403 && (await send('GET', '/api/admin/jobs')).status === 401
      && (await send('GET', '/api/admin/jobs', 'admin')).status === 200, 'express: the guard in front of the jobs view');
    const malformed = await send('POST', '/api/admin/jobs/cancel', 'admin', '{"key": ');
    ok(malformed.status === 400 && malformed.type.includes('application/json') && JSON.parse(malformed.body).error === 'the request body is not valid JSON', 'express: malformed JSON is a JSON 400');
    const simple = await send('POST', '/api/admin/jobs/cancel', 'admin', JSON.stringify({ key: 'predictor', _csrf: ADMIN.csrf }), 'text/plain');
    ok(simple.status === 403 && simple.type.includes('application/json'), 'express: a cross-site "simple" text/plain post carries no readable token: 403');
    const real409 = await send('POST', '/api/admin/jobs/cancel', 'admin', JSON.stringify({ key: 'predictor', _csrf: ADMIN.csrf }));
    ok(real409.status === 409 && JSON.parse(real409.body).error === 'not in the queue', 'express: a JSON post with the token reaches the rule');
    ok((await send('GET', '/api/admin/else', 'admin')).status === 404 && (await send('GET', `/api/admin/jobs/run/hyperoptBoost-${RUN2}/log`, 'admin')).status === 200,
      'express: system.js before admin.js - its routes answer, the rest of /api/admin is admin.js\'s JSON 404');
  } finally {
    server.close();
  }
  return true;
}

const cleanUp = () => { Module.prototype.require = original; console.log = quiet; console.error = loud; fs.rmSync(tmp, { recursive: true, force: true }); };
main().then(() => { onProduction(); return throughExpress(); }).then((viaExpress) => {
  cleanUp();
  if (prodSummary) console.log(prodSummary);
  console.log(`system.js: ${checks} checks passed${viaExpress ? ' (express included)' : ' (express not installed here: the route checks ran on the stand-in)'}`);
}).catch((e) => {
  cleanUp();
  console.error(e);
  process.exitCode = 1;
});
