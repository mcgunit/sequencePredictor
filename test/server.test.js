'use strict';
// server.js, the entry point, after the old pages were removed (cleanup batch 6b, 26 Oct 2026): it wires the sign-in,
// the switch to the app, the JSON APIs, the React app and the last handlers, and starts the services and the schedule
// once it listens. Loaded here with a recording stand-in for express and its config's CONFIG_DIR in a temp dir, so
// nothing listens, nothing is written into the checkout and nothing starts. Checked: the exact list of routes, none of
// the retired pages' routes, the order that matters (the body parser and the sign-in before the redirects, the System
// JSON before the admin catch-all, the last handlers last), that requiring server.js starts nothing, and what the
// listen callback, the shutdown and the process handlers do (with the services and the schedule stood in for).
//   node test/server.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const REPO = path.join(__dirname, '..');
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- the stand-ins: express records every registration, config points at a temp dir ----------------------------
const tmpConfig = fs.mkdtempSync(path.join(os.tmpdir(), 'server-cfg-'));
process.on('exit', () => fs.rmSync(tmpConfig, { recursive: true, force: true }));   // a failed check too
const regs = [];   // { method, paths: [...] or null for a pathless use(), fns }
const listens = [];
const serverEvents = {};
const closes = [];
function fakeApp() {
  const rec = (method) => function (first, ...rest) {
    if (typeof first === 'string' || Array.isArray(first)) regs.push({ method, paths: [].concat(first), fns: rest });
    else regs.push({ method, paths: null, fns: [first, ...rest] });
    return this;
  };
  return {
    use: rec('use'), get: rec('get'), post: rec('post'), put: rec('put'), delete: rec('delete'), all: rec('all'),
    set() { return this; },
    listen(...args) {
      listens.push(args);
      return { on(event, fn) { serverEvents[event] = fn; return this; }, close(fn) { closes.push(fn); } };
    },
  };
}
const tagged = (name, options) => Object.assign(function standIn(req, res, next) { if (next) next(); }, { standIn: name, options });
const expressStub = Object.assign(fakeApp, {
  static: (dir, options) => tagged('static', options), urlencoded: (options) => tagged('urlencoded', options),
  json: (options) => tagged('json', options), Router: () => fakeApp(),
});
const CREDENTIALS = { ADMIN_USER: 'admin', ADMIN_PASSWORD: 'a-test-password-only' };   // as production: a login required, whatever .env says
const realLoad = Module._load;
Module._load = function standIns(request, parent) {
  if (request === 'express') return expressStub;
  const loaded = realLoad.apply(this, arguments);
  if (request === './config' && parent && parent.filename && parent.filename.startsWith(REPO + path.sep)) {
    return new Proxy(loaded, { get: (t, k) => (k === 'CONFIG_DIR' ? tmpConfig : (k in CREDENTIALS ? CREDENTIALS[k] : t[k])) });
  }
  return loaded;
};

const auth = require('../auth');
const services = require('../services');
const jobs = require('../jobs');
const switchover = require('../switchover');
const config = require('../config');

// the services and the schedule as recorders: whatever server.js calls, nothing real starts or stops
const calls = [];
const original = { startAll: services.startAll, stopAll: services.stopAll, start: jobs.start, stop: jobs.stop, finish: switchover.finish };
services.startAll = () => { calls.push('services.startAll'); };
services.stopAll = () => { calls.push('services.stopAll'); };
jobs.start = () => { calls.push('jobs.start'); };
jobs.stop = () => { calls.push('jobs.stop'); };
let finishSpan = null;
let finishCalls = 0;
switchover.finish = function recordedFinish(app) {
  finishCalls += 1;
  const from = regs.length;
  const out = original.finish.call(this, app);
  finishSpan = [from, regs.length];
  return out;
};

const EVENTS = ['SIGTERM', 'SIGINT', 'exit', 'uncaughtException', 'unhandledRejection'];
const before = Object.fromEntries(EVENTS.map((e) => [e, process.listeners(e).slice()]));
const added = {};
try {
  require(path.join(REPO, 'server.js'));
} finally {
  Module._load = realLoad;
  // the handlers the server registers for itself: kept aside and taken off this process, which is a test
  EVENTS.forEach((e) => {
    added[e] = process.listeners(e).filter((l) => !before[e].includes(l));
    added[e].forEach((l) => process.removeListener(e, l));
  });
}

// --- the routes: exactly the live ones (the harness's format: METHOD path, a pathless middleware once) -----------
const LIVE = [
  'GET /api/account', 'GET /api/admin/controls', 'GET /api/admin/controls/:game', 'GET /api/admin/draw/:game/:date', 'GET /api/admin/history/:game',
  'GET /api/admin/jobs', 'GET /api/admin/jobs/run/:id/log', 'GET /api/admin/market/:market/workbench',
  'GET /api/admin/market/:market/workbench/books/:rule', 'GET /api/admin/market/:market/workbench/orders/:game', 'GET /api/admin/market/:market/workbench/instrument/:symbol',
  'GET /api/admin/models', 'GET /api/admin/models/:game', 'GET /api/admin/people', 'GET /api/admin/phase-shift', 'GET /api/admin/phase-shift/:game',
  'GET /api/admin/randomness', 'GET /api/admin/randomness/:game', 'GET /api/admin/tuning', 'GET /api/admin/tuning/:game', 'GET /api/admin/week',
  'GET /api/home', 'GET /api/lottery', 'GET /api/lottery/:game', 'GET /api/lottery/:game/:date', 'GET /api/markets/:market', 'GET /api/me',
  'GET /api/quotes/:market', 'GET /api/whats-new', 'GET /app', 'GET /app/*',
  'GET /council/api/endpoints', 'GET /council/api/job/:id', 'GET /council/api/sessions', 'GET /council/api/sessions/:id', 'GET /council/api/status',
  'GET /login', 'GET /logout',
  'POST /api/account/password', 'POST /api/admin/jobs/cancel', 'POST /api/admin/jobs/run', 'POST /api/admin/jobs/stop',
  'POST /api/admin/people/add', 'POST /api/admin/people/delete', 'POST /api/admin/people/password', 'POST /api/admin/services/:name/:action',
  'POST /api/whats-new/replay', 'POST /api/whats-new/seen',
  'POST /council/api/ask', 'POST /council/api/sessions/:id/delete', 'POST /login', 'POST /logout',
  'USE (middleware)', 'USE /api/admin', 'USE /api/home', 'USE /api/lottery', 'USE /app',
].sort();
const lines = regs.flatMap((r) => (r.paths ? r.paths.map((p) => `${r.method.toUpperCase()} ${p}`) : ['USE (middleware)']));
const found = [...new Set(lines)].sort();
ok(LIVE.length === 57, `the live list has 57 entries (${LIVE.length})`);
ok(JSON.stringify(found) === JSON.stringify(LIVE),
  `exactly the live routes: missing ${JSON.stringify(LIVE.filter((l) => !found.includes(l)))}, extra ${JSON.stringify(found.filter((l) => !LIVE.includes(l)))}`);

// --- none of the retired pages' routes, under any method -----------------------------------------------------
const DEAD = [
  '/', '/lottery', '/database', '/database/:folder', '/database/:folder/:file', '/markets/crypto', '/markets/shares', '/council',
  '/account', '/admin/users', '/admin/activity', '/admin/jobs', '/whats-new',
  '/account/password', '/admin/users/add', '/admin/users/password', '/admin/users/delete', '/admin/jobs/action',
  '/admin/jobs/run', '/admin/jobs/cancel', '/admin/jobs/stop', '/whats-new/seen', '/whats-new/replay',
];
ok(DEAD.length === 23, 'the 23 retired routes (13 pages, 10 form posts)');
for (const p of DEAD) ok(!regs.some((r) => r.paths && r.paths.includes(p)), `no route for the retired ${p}`);

// --- the order that matters ---------------------------------------------------------------------------------
const pathless = regs.map((r, i) => [r, i]).filter(([r]) => r.method === 'use' && !r.paths);
ok(regs[0].method === 'use' && !regs[0].paths && regs[0].fns[0].standIn === 'urlencoded' && regs[0].fns[0].options.extended === false
  && regs[0].fns[0].options.limit === '16kb', 'first: the form body parser (the sign-in form posts), 16 kB at most');
ok(regs[1].method === 'use' && !regs[1].paths && regs[1].fns.length === 1 && regs[1].fns[0] === auth.middleware, 'second: the sign-in middleware');
ok(regs[2].method === 'use' && !regs[2].paths && regs[2].fns.length === 1 && regs[2].fns[0] === switchover.redirects,
  'the redirects right after the sign-in middleware (they need the reader), before every route');
ok(finishCalls === 1 && finishSpan && finishSpan[1] > finishSpan[0] && finishSpan[1] === regs.length,
  `the last handlers (switchover.finish) once, and last: nothing registered after them (${finishSpan} of ${regs.length})`);
ok(pathless.every(([, i]) => i < 3 || i >= finishSpan[0]), 'no other pathless middleware: only the parser, the sign-in, the redirects and the last handlers');
// where a route is registered (its first registration), by METHOD path
const regAt = (method, p) => regs.findIndex((r) => r.method === method && r.paths && r.paths.includes(p));
ok(regs[3].paths && regs[3].paths.every((p) => p.startsWith('/council/api/')) && regAt('post', '/council/api/ask') < regAt('get', '/login'),
  'the Council API right after the redirects, then the sign-in routes');
const adminUses = regs.map((r, i) => (r.method === 'use' && r.paths && r.paths.includes('/api/admin') ? i : -1)).filter((i) => i >= 0);
const adminRoutes = regs.map((r, i) => [r, i]).filter(([r]) => r.method !== 'use' && r.paths && r.paths.some((p) => p.startsWith('/api/admin/')));
const adminLive = LIVE.filter((l) => /^(GET|POST) \/api\/admin\//.test(l)).length;
ok(adminUses.length === 2 && adminRoutes.length === adminLive && adminRoutes.every(([, i]) => i < adminUses[1]),
  `every /api/admin route (System's jobs, services and people included) before the admin catch-all (${adminRoutes.length} of ${adminLive})`);
ok(regAt('post', '/api/admin/jobs/run') < adminUses[0] && regAt('get', '/api/admin/people') < adminUses[0],
  'the System JSON before admin.js (its guard and its catch-all)');
ok(regAt('get', '/app/*') > adminUses[1] && regAt('get', '/app/*') < finishSpan[0], 'the app after the APIs, before the last handlers');

// --- requiring server.js starts nothing ---------------------------------------------------------------------
ok(listens.length === 1 && listens[0][0] === config.PORT && listens[0][1] === config.INTERFACE && typeof listens[0][2] === 'function',
  `one listen, on the configured port and interface (${listens.length ? listens[0].slice(0, 2).join(' ') : 'none'})`);
ok(calls.length === 0, `loading started nothing: no service, no schedule, the listen callback not run (${calls.join(', ')})`);
ok(typeof serverEvents.error === 'function', 'a listen error is handled (it stops the process with a message)');
ok(auth.CONFIG_DIR === tmpConfig && auth.USERS_FILE.startsWith(tmpConfig + path.sep) && auth.SECRET_FILE.startsWith(tmpConfig + path.sep),
  'config/ is the temp dir: nothing is written into the checkout');

// --- once it listens: the supervised services, then the schedule --------------------------------------------
const log = console.log;
const logged = [];
console.log = (...a) => logged.push(a.join(' '));
try { listens[0][2](); } finally { console.log = log; }
ok(JSON.stringify(calls) === JSON.stringify(['services.startAll', 'jobs.start']), `the listen callback starts the services, then the schedule (${calls.join(', ')})`);
ok(logged.length === 1 && logged[0].includes(`http://${config.INTERFACE}:${config.PORT}`) && logged[0].includes('login required (admin: admin)'),
  `it says where it listens and that a login is required (${logged[0]})`);

// --- the shutdown and process handlers -----------------------------------------------------------------------
EVENTS.forEach((e) => ok(added[e].length === 1, `one ${e} handler`));
for (const signal of ['SIGTERM', 'SIGINT']) {
  calls.length = 0; closes.length = 0;
  const timers = [];
  const realTimeout = global.setTimeout;
  global.setTimeout = (fn, ms) => { const t = { fn, ms, unrefd: false, unref() { t.unrefd = true; return t; } }; timers.push(t); return t; };
  console.log = () => {};
  try { added[signal][0](); } finally { global.setTimeout = realTimeout; console.log = log; }
  ok(JSON.stringify(calls) === JSON.stringify(['jobs.stop', 'services.stopAll']) && closes.length === 1 && typeof closes[0] === 'function',
    `${signal}: the schedule's timer stops, the services stop, the server closes (${calls.join(', ')})`);
  ok(timers.length === 1 && timers[0].ms === 6000 && timers[0].unrefd, `${signal}: a 6 s fallback exit that does not hold the process`);
}
calls.length = 0;
added.exit[0]();
ok(JSON.stringify(calls) === JSON.stringify(['services.stopAll']), 'on exit: the services stop');
const err = console.error;
const errors = [];
console.error = (...a) => errors.push(a.map(String).join(' '));
try {
  added.uncaughtException[0](new Error('boom'));
  added.unhandledRejection[0]('nope');
} finally { console.error = err; }
ok(errors.length === 2 && /uncaught exception:.*boom/.test(errors[0]) && /unhandled rejection:.*nope/.test(errors[1]),
  'an uncaught exception and an unhandled rejection are logged, the process stays up');

Object.assign(services, { startAll: original.startAll, stopAll: original.stopAll });
Object.assign(jobs, { start: original.start, stop: original.stop });
switchover.finish = original.finish;
console.log(`server: ${checks} checks passed`);
