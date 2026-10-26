// The Council's JSON under /council/api (the app's Council page): a thin proxy
// to the Python API (api.py), and the asker's own sessions.
//
// The Python API binds to 127.0.0.1 and has no authentication of its own, so
// every request to it must come through here, behind auth.middleware. Nothing
// in this file may be mounted before that middleware.
//
// Requests are proxied rather than called from the browser directly: that
// keeps the API off the network, avoids CORS entirely, and leaves login and
// CSRF in one place.

const http = require('http');

const auth = require('./auth');
const sessions = require('./councilSessions');

// Where api.py listens. Override with COUNCIL_API_HOST / COUNCIL_API_PORT.
const API_HOST = process.env.COUNCIL_API_HOST || '127.0.0.1';
const API_PORT = Number(process.env.COUNCIL_API_PORT || 8099);

// A council run costs minutes of CPU on the inference box, so decide
// deliberately who may start one. 'admin' restricts the Council to admins;
// 'user' allows any logged-in user.
const ACCESS = process.env.COUNCIL_ACCESS === 'admin' ? 'admin' : 'user';

// Generous: a run with real models takes minutes. This is only the proxy hop
// to a local API that answers immediately (it returns a job id, it does not
// wait for the council), so it should never be reached.
const PROXY_TIMEOUT_MS = 30000;

// One call to api.py. Returns { status, body } with body parsed as JSON, or
// rejects when the API is unreachable - which is the normal case when the
// Python process is not running, so it must produce a readable message
// rather than a stack trace on the page.
function callApi(method, apiPath, payload) {
  return new Promise((resolve, reject) => {
    const data = payload === undefined ? null : Buffer.from(JSON.stringify(payload), 'utf8');
    const req = http.request({
      host: API_HOST,
      port: API_PORT,
      path: apiPath,
      method,
      headers: data
        ? { 'Content-Type': 'application/json', 'Content-Length': data.length }
        : {},
      timeout: PROXY_TIMEOUT_MS
    }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        try {
          resolve({ status: res.statusCode, body: JSON.parse(text) });
        } catch (e) {
          resolve({ status: 502, body: { error: 'council api returned non-JSON' } });
        }
      });
    });
    req.on('timeout', () => req.destroy(new Error('council api timed out')));
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

function denied(req) {
  return ACCESS === 'admin' && req.user && !req.user.open && req.user.role !== 'admin';
}

// Who sits on the council and how to reach them (GET /council/api/endpoints: the members, their labs and
// temperatures, the head, the preset, each endpoint's reachability) is the administrator's (owner, 23 Oct 2026).
// Open access (no login configured) is the administrator, as everywhere else.
function isAdmin(req) {
  return Boolean(req.user && (req.user.open === true || req.user.role === 'admin'));
}

// What a user is sent of a run (owner, 23 Oct 2026: the members' and endpoints' details are the administrator's).
// api.py's job and the turn filed from it carry both - a lab on every member and on the head, and a failed run's
// "EndpointsUnavailable: <member>: http://<host>:<port> unreachable: ..." (its preflight, when the model boxes are
// off) - so a user's copy is cleaned here, on the server, not only on the page. The administrator gets them whole.
const NOT_SITTING = 'the council was not sitting (a member could not be reached)';
const NOT_ANSWERING = 'The council cannot sit right now: its service is not answering.';
const ADDRESSES = [
  /\bhttps?:\/\/[^\s)'"<>]+/gi,                        // a URL
  /\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b/g,             // an IPv4 address, with its port
  /\b[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*:\d{2,5}\b/gi,    // host:port (localhost:8090, modelbox:8081)
];
function plainError(text) {
  if (typeof text !== 'string') return text;
  if (/^EndpointsUnavailable\b/.test(text)) return NOT_SITTING;
  return ADDRESSES.reduce((t, re) => t.replace(re, 'an endpoint'), text);
}
// a member, the head or a seat without its lab, its error without an address
function plainSeat(seat) {
  if (!seat || typeof seat !== 'object' || Array.isArray(seat)) return seat;
  const { lab, ...rest } = seat;
  return rest.error === undefined ? rest : { ...rest, error: plainError(rest.error) };
}
function plainResult(result) {
  if (!result || typeof result !== 'object' || Array.isArray(result)) return result;
  return { ...result, members: Array.isArray(result.members) ? result.members.map(plainSeat) : result.members, head: plainSeat(result.head) };
}
function plainProgress(progress) {
  if (!progress || typeof progress !== 'object' || Array.isArray(progress)) return progress;
  const seats = progress.seats && typeof progress.seats === 'object'
    ? Object.fromEntries(Object.entries(progress.seats).map(([name, s]) => [name, plainSeat(s)])) : progress.seats;
  return { ...progress, seats, head: plainSeat(progress.head) };
}
// a job (api.py's job.public()) or a filed turn, as the reader gets it
function forReader(req, run) {
  if (isAdmin(req) || !run || typeof run !== 'object') return run;
  const out = { ...run };
  if (out.error !== undefined) out.error = plainError(out.error);
  if (out.result !== undefined) out.result = plainResult(out.result);
  if (out.progress !== undefined) out.progress = plainProgress(out.progress);
  return out;
}
// the Council API did not answer the proxy: its reason (connect ECONNREFUSED 127.0.0.1:8099) is the administrator's and
// the server log's
function apiDown(req, res, e) {
  if (isAdmin(req)) return res.status(503).json({ error: `The members of the council have to be summoned (${e.message})` });
  console.error(`[${new Date().toISOString()}] council.js: the council api did not answer: ${e.message}`);
  return res.status(503).json({ error: NOT_ANSWERING });
}

// Whose run a job is: every job is filed in its asker's session when api.py accepts it (and re-tracked from the
// pending turns after a restart), so a job id is answered only to the account whose session holds it. Kept in memory
// for the jobs this process filed or tracks; any other id is looked up in the caller's own sessions.
const JOB_OWNERS_KEPT = 500;
const jobOwners = new Map();   // job id -> account name
function noteJobOwner(jobId, owner) {
  if (!jobId) return;
  jobOwners.delete(String(jobId));
  jobOwners.set(String(jobId), String(owner));
  if (jobOwners.size > JOB_OWNERS_KEPT) jobOwners.delete(jobOwners.keys().next().value);
}
function ownsJob(req, jobId) {
  const name = String(req.user.name);
  if (jobOwners.has(jobId)) return jobOwners.get(jobId) === name;
  return sessions.list(name).some((s) => {
    const session = sessions.load(name, s.id);
    return Boolean(session && session.turns.some((t) => t.jobId === jobId));
  });
}

// --- recording answers server-side ---------------------------------------------
// A turn is filed as pending the moment api.py accepts it, and completed by
// this poller when the job ends - not by the browser, which may be closed by
// then. One timer per outstanding job; api.py keeps its last twenty jobs, so
// a job that has vanished from it (the API restarted) is closed as failed
// rather than polled forever.
const TRACK_INTERVAL_MS = 3000;
const TRACK_GIVE_UP_MS = 3 * 60 * 60 * 1000;   // longer than any run (request_timeout_s x retries)
const tracked = new Set();

function track(owner, sessionId, turnId, jobId, since) {
  const key = `${sessionId}/${turnId}`;
  noteJobOwner(jobId, owner);
  if (tracked.has(key)) return;
  tracked.add(key);
  const startedAt = since ? Date.parse(since) || Date.now() : Date.now();

  const finish = (patch) => {
    tracked.delete(key);
    sessions.updateTurn(owner, sessionId, turnId, patch);
  };

  const step = async () => {
    if (Date.now() - startedAt > TRACK_GIVE_UP_MS) {
      return finish({ state: 'failed', error: 'no answer arrived within three hours - the run was given up' });
    }
    let reply;
    try {
      reply = await callApi('GET', `/job/${jobId}`);
    } catch (e) {
      // The API is down. Keep waiting: it is supervised and will come back,
      // and its in-memory job may still be there if only the network hiccuped.
      const timer = setTimeout(step, TRACK_INTERVAL_MS * 5);
      if (timer.unref) timer.unref();
      return;
    }
    if (reply.status === 404) {
      return finish({ state: 'failed', error: 'the council api no longer knows this run (it was restarted) - ask again' });
    }
    const job = reply.body || {};
    if (job.state === 'done') return finish({ state: 'done', result: job.result || null });
    if (job.state === 'failed') return finish({ state: 'failed', error: job.error || 'run failed' });
    const timer = setTimeout(step, TRACK_INTERVAL_MS);
    if (timer.unref) timer.unref();
  };
  step();
}

// Whatever was still pending when the server last stopped.
function resumeTracking() {
  sessions.pendingTurns().forEach((p) => {
    if (p.jobId) track(p.owner, p.sessionId, p.turnId, p.jobId, p.asked);
    else sessions.updateTurn(p.owner, p.sessionId, p.turnId, { state: 'failed', error: 'the run was never started' });
  });
}

function install(app) {
  // JSON bodies for the proxied POSTs only. server.js installs
  // express.urlencoded for the sign-in form; this adds JSON without touching that.
  const jsonBody = require('express').json({ limit: '64kb' });
  const rejectCsrf = (res) => res.status(403).json({ error: 'Invalid form token - reload the page and try again.' });

  // Ask. The question goes to api.py; the turn is filed in the caller's
  // session (a new one when none is given) and tracked to completion here.
  // CSRF-checked like every other POST: a council run costs minutes of CPU
  // on the model box, which is exactly what a cross-site page must not be
  // able to trigger.
  app.post('/council/api/ask', jsonBody, async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    if (!auth.csrfOk(req)) return rejectCsrf(res);
    const body = req.body || {};
    const sessionId = body.session && sessions.ID.test(String(body.session)) ? String(body.session) : null;
    if (sessionId && !sessions.load(req.user.name, sessionId)) {
      return res.status(404).json({ error: 'that session is not yours or no longer exists' });
    }
    try {
      const { status, body: reply } = await callApi('POST', '/ask', { question: body.question, context: body.context });
      if (status !== 202 || !reply || !reply.id) return res.status(status).json(reply);
      const { session, turn } = sessions.appendTurn(req.user.name, sessionId, {
        jobId: reply.id, question: String(body.question || ''), context: body.context ? String(body.context) : null,
      });
      track(req.user.name, session.id, turn.id, reply.id, turn.asked);
      res.status(202).json(Object.assign({}, forReader(req, reply), { session: session.id, turn: turn.id }));
    } catch (e) {
      apiDown(req, res, e);
    }
  });

  app.get('/council/api/job/:id', async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    // The id comes from our own API; still constrain it rather than
    // interpolating whatever arrives into a request path.
    if (!/^[a-f0-9]{1,32}$/.test(req.params.id)) {
      return res.status(400).json({ error: 'bad job id' });
    }
    // Only the asker's own run: another account's question and answer are
    // not this caller's to read (a job id is not a secret).
    if (!ownsJob(req, req.params.id)) return res.status(404).json({ error: 'no such job' });
    try {
      const { status, body } = await callApi('GET', `/job/${req.params.id}`);
      res.status(status).json(forReader(req, body));
    } catch (e) {
      apiDown(req, res, e);
    }
  });

  // Whether the council can sit now (api.py /health: at least one member and the head answer), for every reader: the
  // page keeps Ask off while it cannot, rather than file a run bound to fail. Only {ready, busy}: who does not answer,
  // and where, is the administrator's (/endpoints). The Council API itself not answering is not ready either; the
  // administrator is also told why (apiError: the proxy's reason, an address in it), since /endpoints is then a 503.
  app.get('/council/api/status', async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    res.set('Cache-Control', 'no-store');
    try {
      const { status, body } = await callApi('GET', '/health');
      const s = body && body.status;
      res.json({ ready: status === 200 && Boolean(s && s.ready === true), busy: Boolean(body && body.busy === true) });
    } catch (e) {
      if (isAdmin(req)) return res.json({ ready: false, busy: false, apiError: e.message });
      console.error(`[${new Date().toISOString()}] council.js: the council api did not answer: ${e.message}`);
      res.json({ ready: false, busy: false });
    }
  });

  app.get('/council/api/endpoints', async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    // the members and their endpoints are the administrator's (isAdmin above)
    if (!isAdmin(req)) return res.status(403).set('Cache-Control', 'no-store').json({ error: 'admin only' });
    try {
      const { status, body } = await callApi('GET', '/endpoints');
      res.status(status).json(body);
    } catch (e) {
      apiDown(req, res, e);
    }
  });

  // Sessions: always the caller's own. There is no admin view of other
  // people's conversations, on purpose.
  app.get('/council/api/sessions', (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    res.json({ sessions: sessions.list(req.user.name) });
  });

  app.get('/council/api/sessions/:id', (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    const session = sessions.load(req.user.name, req.params.id);
    if (!session) return res.status(404).json({ error: 'no such session' });
    res.json({ session: Array.isArray(session.turns) ? { ...session, turns: session.turns.map((t) => forReader(req, t)) } : session });
  });

  app.post('/council/api/sessions/:id/delete', jsonBody, (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    if (!auth.csrfOk(req)) return rejectCsrf(res);
    res.json({ removed: sessions.remove(req.user.name, req.params.id) });
  });

  resumeTracking();
}

module.exports = { install };
