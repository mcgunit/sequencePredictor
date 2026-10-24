'use strict';
// The switch to the app (24 Oct 2026): every old page address answers with its app page, for each reader as the old
// page was; an unknown address and a thrown error answer in the sign-in look (JSON under the APIs) and never carry the
// stack; the sign-in page is in the app's look and escapes what it echoes. A throwaway Express app on a free local
// port serves the last handlers (when express can be loaded); no part of server.js is started.
//   NODE_PATH=/root/sequencePredictor/node_modules node test/switchover.test.js
const assert = require('assert');
const fs = require('fs');
const http = require('http');
const path = require('path');
let express = null;   // the last handlers run through express itself when it can be loaded (NODE_PATH), as test/admin.test.js
try { express = require('express'); } catch (e) { express = null; }
const switchover = require('../switchover');
const auth = require('../auth');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const ADMIN = { name: 'admin', role: 'admin', open: false };
const USER = { name: 'reader', role: 'user', open: false };
const OPEN = { name: '', role: 'admin', open: true };

// --- where each old page sends each reader ------------------------------------------------------------------
const CASES = [
  // [path, query, admin's target, user's target]
  ['/', {}, '/app/', '/app/'],
  ['/lottery', {}, '/app/lottery', '/app/lottery'],
  ['/lottery/', {}, '/app/lottery', '/app/lottery'],
  ['/database', {}, '/app/admin/games', '/app/lottery'],
  ['/database/lotto', {}, '/app/admin/games/lotto', '/app/lottery/lotto'],
  ['/database/lotto', { month: '2026-09' }, '/app/admin/games/lotto?month=2026-09', '/app/lottery/lotto?month=2026-09'],
  ['/database/lotto', { month: '2026-09"><x' }, '/app/admin/games/lotto', '/app/lottery/lotto'],
  ['/database/vikinglotto/2026-10-21.json', {}, '/app/admin/games/vikinglotto/2026-10-21', '/app/lottery/vikinglotto/2026-10-21'],
  ['/database/keno/2026-2-3.json', {}, '/app/admin/games/keno/2026-02-03', '/app/lottery/keno/2026-02-03'],
  ['/database/lotto/2026-02-30.json', {}, '/app/admin/games/lotto', '/app/lottery/lotto'],
  ['/database/crypto/2026-10-20.json', {}, '/app/admin/games/crypto/2026-10-20', '/app/crypto'],
  ['/database/sharesweek', {}, '/app/admin/games/sharesweek', '/app/shares'],
  ['/database/nothing-here', {}, '/app/admin/games', '/app/lottery'],
  ['/database/%2e%2e', {}, '/app/admin/games', '/app/lottery'],
  ['/markets/crypto', {}, '/app/admin/markets/crypto', '/app/crypto'],
  ['/markets/shares/', {}, '/app/admin/markets/shares', '/app/shares'],
  ['/council', {}, '/app/council', '/app/council'],
  ['/council', { session: '0123456789ab' }, '/app/council?session=0123456789ab', '/app/council?session=0123456789ab'],
  ['/council', { session: '//evil.example' }, '/app/council', '/app/council'],
  ['/account', {}, '/app/account', '/app/account'],
  ['/whats-new', {}, '/app/whats-new', '/app/whats-new'],
  ['/admin/users', {}, '/app/admin/people', '/app/'],
  ['/admin/activity', {}, '/app/admin/people', '/app/'],
  ['/admin/jobs', {}, '/app/admin/jobs', '/app/'],
];
for (const [p, q, a, u] of CASES) {
  ok(switchover.targetOf(p, q, ADMIN) === a, `admin ${p} ${JSON.stringify(q)} -> ${a} (got ${switchover.targetOf(p, q, ADMIN)})`);
  ok(switchover.targetOf(p, q, USER) === u, `user ${p} ${JSON.stringify(q)} -> ${u} (got ${switchover.targetOf(p, q, USER)})`);
  ok(switchover.targetOf(p, q, OPEN) === a, `open access reads as the administrator: ${p}`);
}
// not an old page: left to its own route
for (const p of ['/app', '/app/', '/app/lottery', '/api/me', '/api/admin/models', '/council/api/status', '/login', '/logout', '/markets/gold', '/favicon.ico']) {
  ok(switchover.targetOf(p, {}, ADMIN) === null && switchover.targetOf(p, {}, USER) === null, `${p} is not an old page`);
}
// every target is a local app address: never another host, never what the address carried
for (const [p, q] of CASES) for (const who of [ADMIN, USER]) {
  const t = switchover.targetOf(p, q, who);
  ok(/^\/app\/[A-Za-z0-9/_?=-]*$/.test(t), `a plain local app address: ${t}`);
}

// --- the middleware: GET and HEAD only, 302, no-store --------------------------------------------------------
function callRedirects(method, url, user) {
  const [p, qs] = url.split('?');
  const out = { next: false, status: null, location: null, headers: {} };
  const req = { method, path: p, query: Object.fromEntries(new URLSearchParams(qs || '')), user };
  const res = { set(k, v) { out.headers[k.toLowerCase()] = v; return this; }, redirect(s, l) { out.status = s; out.location = l; } };
  switchover.redirects(req, res, () => { out.next = true; });
  return out;
}
let r = callRedirects('GET', '/database/lotto?month=2026-10', USER);
ok(r.status === 302 && r.location === '/app/lottery/lotto?month=2026-10' && r.headers['cache-control'] === 'no-store' && !r.next, 'GET an old page: 302 to the app, not kept by the browser');
r = callRedirects('HEAD', '/', ADMIN);
ok(r.status === 302 && r.location === '/app/', 'HEAD as GET');
r = callRedirects('POST', '/admin/jobs/action', ADMIN);
ok(r.next && r.status === null, 'a POST passes on to its own route');
r = callRedirects('GET', '/api/lottery', USER);
ok(r.next && r.status === null, 'the APIs pass on');

// --- server.js: the redirects right after the sign-in middleware, the last handlers last ----------------------
const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf-8');
const at = (needle) => server.indexOf(needle);
ok(at('app.use(auth.middleware);') > 0 && at('app.use(switchover.redirects);') > at('app.use(auth.middleware);'), 'redirects after the sign-in middleware (they need the reader)');
ok(at('app.use(switchover.redirects);') < at('council.install(app'), 'redirects before the first old page route (/council)');
const routeCalls = [...server.matchAll(/^(?:app\.(?:get|post|use)\(|\w+\.install\(app)/gm)].map((m) => m.index);
ok(at('switchover.finish(app);') > Math.max(...routeCalls), 'the 404 and error handlers after every route');

// --- the sign-in page --------------------------------------------------------------------------------------------
function signIn() {
  const page = auth.loginPage('Wrong <b>password</b>', '/app/lottery?x="><script>');
  ok(page.includes('name="viewport"') && page.includes("localStorage.getItem('predictor.theme')") && page.includes('IBM+Plex+Sans'), "sign-in: phone width, the reader's colour choice, the app's face");
  ok(page.includes('Wrong &lt;b&gt;password&lt;/b&gt;') && !page.includes('<b>password'), 'sign-in: the message escaped');
  ok(page.includes('value="/app/lottery?x=&quot;&gt;&lt;script&gt;"') && !page.includes('"><script>'), 'sign-in: next escaped');
  ok(page.includes('name="username"') && page.includes('name="password"') && page.includes('action="/login"'), 'sign-in: the same form fields and action');
  ok(!auth.loginPage(null, '/').includes('class="err"'), 'sign-in: no error box without an error');
}

// --- the last handlers on a throwaway app -----------------------------------------------------------------------
if (!express) {
  signIn();
  console.log(`switchover: ${checks} checks passed (express not installed here: the 404 and error answers were not run)`);
  return;
}
const app = express();
app.use(express.json({ limit: '1kb' }));
app.get('/boom', () => { throw new Error('secret detail at /root/sequencePredictor/server.js:123'); });
app.get('/api/boom', () => { throw new Error('secret detail'); });
app.post('/api/echo', (req, res) => res.json(req.body));
app.get('/async', (req, res, next) => { setImmediate(() => next(new Error('later'))); });
switchover.finish(app);

function get(port, method, p, body, type) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, method, path: p, headers: body ? { 'Content-Type': type || 'application/json' } : {} }, (res) => {
      let data = '';
      res.on('data', (c) => { data += c; });
      res.on('end', () => resolve({ status: res.statusCode, type: res.headers['content-type'] || '', body: data }));
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

const logged = [];
const realError = console.error;
console.error = (...a) => logged.push(a.map(String).join(' '));
const listener = app.listen(0, '127.0.0.1', async () => {
  const { port } = listener.address();
  try {
    let a = await get(port, 'GET', '/no/such/page');
    ok(a.status === 404 && /text\/html/.test(a.type) && a.body.includes('Page not found') && a.body.includes('href="/app/"'), 'an unknown page: 404 in the sign-in look, with the way home');
    a = await get(port, 'GET', '/api/no-such');
    ok(a.status === 404 && /json/.test(a.type) && JSON.parse(a.body).error === 'not found', 'an unknown API address: JSON 404');
    a = await get(port, 'GET', '/council/api/nothing');
    ok(a.status === 404 && /json/.test(a.type), "the Council API's unknown address: JSON 404");
    a = await get(port, 'GET', '/boom');
    ok(a.status === 500 && a.body.includes('Something went wrong') && !a.body.includes('secret detail') && !a.body.includes('/root/') && !/\bat \w/.test(a.body),
      'a thrown error: 500 page without the message, a path or the stack');
    ok(logged.some((l) => l.includes('GET /boom') && l.includes('secret detail')), 'the error and its stack go to the log');
    a = await get(port, 'GET', '/api/boom');
    ok(a.status === 500 && JSON.parse(a.body).error === 'server error' && !a.body.includes('secret'), 'a thrown error under the API: JSON 500, no detail');
    a = await get(port, 'GET', '/async');
    ok(a.status === 500 && a.body.includes('Something went wrong'), 'an error passed on later: the same 500');
    const before = logged.length;
    a = await get(port, 'POST', '/api/echo', '{"broken": ');
    ok(a.status === 400 && /json/.test(a.type) && !a.body.includes(' at '), "a malformed body: the parser's 400 as JSON, no stack");
    ok(logged.length === before, "a reader's malformed body is not logged as a server error");
    a = await get(port, 'POST', '/api/echo', JSON.stringify({ x: 'y'.repeat(2000) }));
    ok(a.status === 413 && /json/.test(a.type), 'a too large body: 413 as JSON');
  } finally {
    console.error = realError;
    listener.close();
  }

  signIn();
  console.log(`switchover: ${checks} checks passed`);
});
