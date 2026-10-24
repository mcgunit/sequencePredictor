// The council session store: what it must keep true (see the header of
// councilSessions.js). Runs against a temporary directory, never config/.
// And council.js's /council/api/status when the Council API is down: the
// administrator is told why, a user only that the council cannot sit (on a
// stand-in for the Council API on a free port - the real one is never called).
// Run with: npm test
const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const Module = require('module');

const { createStore } = require('../councilSessions');

let checks = 0;
const ok = (cond, what) => { assert(cond, what); checks += 1; };

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'council-sessions-'));
const store = createStore(root);

try {
  // --- nothing exists until a question is asked --------------------------------
  ok(store.list('alice').length === 0, 'a new account has no sessions');
  ok(store.load('alice', 'abcdefabcdef') === null, 'an unknown id loads as null');
  ok(store.load('alice', '../../etc/passwd') === null, 'an id that is not twelve hex characters is refused, not used as a path');

  // --- the first question creates the session, and titles it -----------------
  const first = store.appendTurn('alice', null, { jobId: 'job1', question: '  What is   the capital of Belgium?  ', context: null });
  ok(store.ID.test(first.session.id) && store.ID.test(first.turn.id), 'session and turn ids are twelve hex characters');
  ok(first.session.title === 'What is the capital of Belgium?', 'the first question, whitespace-normalised, is the title');
  ok(first.turn.state === 'pending' && first.turn.jobId === 'job1', 'a submitted question is recorded as pending with its job id');
  ok(store.list('alice').length === 1 && store.list('alice')[0].turns === 1, 'the session is listed with one turn');
  ok(store.list('alice')[0].pending === true, 'the listing says an answer is still outstanding');

  // --- the answer completes the turn --------------------------------------------
  const done = store.updateTurn('alice', first.session.id, first.turn.id, { state: 'done', result: { head: { answer: 'Brussels' } } });
  ok(done.state === 'done' && done.finished, 'completing a turn stamps when it finished');
  ok(store.load('alice', first.session.id).turns[0].result.head.answer === 'Brussels', 'the result is persisted');
  ok(store.list('alice')[0].pending === false, 'nothing outstanding any more');

  // --- follow-up questions go to the same session -------------------------------
  const second = store.appendTurn('alice', first.session.id, { jobId: 'job2', question: 'And of France?', context: 'prior: Brussels' });
  ok(second.session.id === first.session.id && second.session.turns.length === 2, 'a follow-up lands in the same session');
  ok(second.session.title === 'What is the capital of Belgium?', 'the title stays the first question');
  ok(second.turn.context === 'prior: Brussels', 'the context travels with the turn');

  // --- sessions with zero messages are never shown -------------------------------
  const dir = store.userDir('alice');
  fs.writeFileSync(path.join(dir, 'aaaaaaaaaaaa.json'), JSON.stringify({ id: 'aaaaaaaaaaaa', owner: 'alice', title: 'ghost', created: 'x', updated: 'x', turns: [] }));
  ok(store.list('alice').length === 1, 'a hand-made session file with no turns is not listed');
  ok(store.load('alice', 'aaaaaaaaaaaa') === null, 'nor can it be loaded');

  // --- one account cannot see another's --------------------------------------
  store.appendTurn('bob', null, { jobId: 'job3', question: 'Bob asks', context: null });
  ok(store.list('bob').length === 1 && store.list('alice').length === 1, 'each account lists only its own sessions');
  ok(store.load('bob', first.session.id) === null, "an id from another account's session is not found");
  ok(store.userDir('alice') !== store.userDir('Alice'), 'directories are per exact account name');
  ok(store.userDir('a/b') === store.userDir('a/b') && !store.userDir('a/b').includes('a/b'), 'unsafe characters never reach the path');
  ok(store.userDir('a b') !== store.userDir('a_b'), 'two names that sanitise alike still get different directories');

  // --- what a restarted server has to pick up -------------------------------------
  const pending = store.pendingTurns();
  ok(pending.length === 2, 'two turns are still pending across all accounts (alice job2, bob job3)');
  ok(pending.every((p) => p.owner && p.sessionId && p.turnId && p.jobId), 'each carries enough to resume polling');

  // --- deleting -------------------------------------------------------------------
  ok(store.remove('bob', first.session.id) === false, "an account cannot delete another's session");
  ok(store.remove('alice', first.session.id) === true && store.list('alice').length === 0, 'the owner can');
  ok(store.updateTurn('alice', first.session.id, first.turn.id, { state: 'done' }) === null, 'updating a deleted session is a no-op, not a crash');

  console.log(`councilSessions.js: ${checks} checks passed`);
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}

// --- council.js /council/api/status: up, then the Council API not answering -------
async function statusRoute() {
  checks = 0;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'council-status-'));
  const standIn = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    if (req.url === '/health') return res.end(JSON.stringify({ ok: true, busy: false, status: { ready: true } }));
    return res.end(JSON.stringify({ members: [], head: null, status: { ready: true, members: [] } }));
  });
  await new Promise((resolve) => standIn.listen(0, '127.0.0.1', resolve));
  const port = standIn.address().port;
  process.env.COUNCIL_API_HOST = '127.0.0.1';
  process.env.COUNCIL_API_PORT = String(port);
  process.env.COUNCIL_ACCESS = 'user';
  // the session store in the temp dir; the form check and express only where council.js asks for them
  const stubs = { './auth': { csrfOk: () => false }, './councilSessions': createStore(tmp) };
  try { require.resolve('express'); } catch (e) { stubs.express = { json: () => (req, res, next) => next() }; }
  const councilFile = path.join(__dirname, '..', 'council.js');
  const original = Module.prototype.require;
  Module.prototype.require = function stubbed(id) {
    if (this.filename === councilFile && Object.prototype.hasOwnProperty.call(stubs, id)) return stubs[id];
    return original.apply(this, arguments);
  };
  const routes = {};
  try {
    const council = require('../council');
    council.install({ get: (p, ...fns) => { routes[`GET ${p}`] = fns; }, post: (p, ...fns) => { routes[`POST ${p}`] = fns; } },
      { header: () => '', footer: () => '', escapeHtml: (t) => t });
  } finally { Module.prototype.require = original; }
  const get = (url, user) => new Promise((resolve) => {
    const res = { statusCode: 200, headers: {}, body: undefined,
      status(c) { this.statusCode = c; return this; }, set(k, v) { this.headers[k] = v; return this; },
      json(b) { this.body = b; resolve(this); return this; } };
    routes[`GET ${url}`][0]({ method: 'GET', user, params: {}, query: {}, headers: {} }, res, () => resolve(res));
  });
  const alice = { name: 'alice', role: 'user', open: false };
  const admin = { name: 'admin', role: 'admin', open: false };
  const keys = (body) => JSON.stringify(Object.keys(body).sort());

  try {
    const upUser = await get('/council/api/status', alice);
    const upAdmin = await get('/council/api/status', admin);
    ok(upUser.body.ready === true && keys(upUser.body) === '["busy","ready"]' && upAdmin.body.ready === true && keys(upAdmin.body) === '["busy","ready"]',
      `the Council API answers: ready, and no apiError for anyone (${JSON.stringify(upAdmin.body)})`);

    await new Promise((resolve) => standIn.close(resolve));
    const logged = console.error;
    console.error = () => {};
    let downUser;
    try { downUser = await get('/council/api/status', alice); } finally { console.error = logged; }
    const downAdmin = await get('/council/api/status', admin);
    const downOpen = await get('/council/api/status', { name: 'open access', role: 'admin', open: true });
    ok(downAdmin.statusCode === 200 && downAdmin.body.ready === false && /ECONNREFUSED/.test(downAdmin.body.apiError) && downAdmin.headers['Cache-Control'] === 'no-store'
      && /ECONNREFUSED/.test(downOpen.body.apiError),
      `the Council API down: the administrator (and open access) are told why (${downAdmin.body.apiError})`);
    ok(downUser.statusCode === 200 && downUser.body.ready === false && keys(downUser.body) === '["busy","ready"]'
      && !JSON.stringify(downUser.body).includes('127.0.0.1') && !JSON.stringify(downUser.body).includes(String(port)),
      `a user only that the council cannot sit, no address (${JSON.stringify(downUser.body)})`);
    const endpoints = await get('/council/api/endpoints', admin);
    ok(endpoints.statusCode === 503 && /summoned/.test(endpoints.body.error), 'and the administrator\'s /endpoints is the 503 the page reads as the Council API not answering');
    console.log(`council.js status: ${checks} checks passed`);
  } finally {
    if (standIn.listening) standIn.close();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}
statusRoute().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
