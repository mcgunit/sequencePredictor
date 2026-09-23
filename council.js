// Council chat page and a thin proxy to the Python API (api.py).
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
// deliberately who may start one. 'admin' restricts the page to admins;
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

function install(app, { header, footer, escapeHtml }) {
  // JSON bodies for the proxied POSTs only. server.js installs
  // express.urlencoded for its forms; this adds JSON without touching that.
  const jsonBody = require('express').json({ limit: '64kb' });
  const rejectCsrf = (res) => res.status(403).json({ error: 'Invalid form token - reload the page and try again.' });

  app.get('/council', (req, res) => {
    if (denied(req)) return res.status(403).send('Not available for your account.');
    res.send(page(req, header, footer, escapeHtml));
  });

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
      res.status(202).json(Object.assign({}, reply, { session: session.id, turn: turn.id }));
    } catch (e) {
      res.status(503).json({ error: `The members of the council have to be summoned (${e.message})` });
    }
  });

  app.get('/council/api/job/:id', async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    // The id comes from our own API; still constrain it rather than
    // interpolating whatever arrives into a request path.
    if (!/^[a-f0-9]{1,32}$/.test(req.params.id)) {
      return res.status(400).json({ error: 'bad job id' });
    }
    try {
      const { status, body } = await callApi('GET', `/job/${req.params.id}`);
      res.status(status).json(body);
    } catch (e) {
      res.status(503).json({ error: `The members of the council have to be summoned (${e.message})` });
    }
  });

  app.get('/council/api/endpoints', async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    try {
      const { status, body } = await callApi('GET', '/endpoints');
      res.status(status).json(body);
    } catch (e) {
      res.status(503).json({ error: `The members of the council have to be summoned (${e.message})` });
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
    res.json({ session });
  });

  app.post('/council/api/sessions/:id/delete', jsonBody, (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    if (!auth.csrfOk(req)) return rejectCsrf(res);
    res.json({ removed: sessions.remove(req.user.name, req.params.id) });
  });

  resumeTracking();
}

function page(req, header, footer, escapeHtml) {
  const esc = escapeHtml || ((s) => String(s));
  return header('LLM Council', req.user) + `
  <style>
    /* The site's .container is 1000px; a two-column layout wants more. */
    .container:has(.council-layout) { max-width: 1300px; }

    .council-layout { display: flex; gap: 24px; align-items: flex-start; }
    .council-main { flex: 2 1 0; min-width: 0; }   /* min-width:0 lets long
                                                      answers wrap instead of
                                                      stretching the column */
    .council-side { flex: 1 1 0; min-width: 260px; position: sticky; top: 100px; }

    /* One column below the site's content width, table first so it stays
       visible without scrolling past the whole conversation. */
    @media (max-width: 900px) {
      .council-layout { flex-direction: column; }
      .council-side { position: static; width: 100%; min-width: 0; order: -1; }
    }

    .council-log { min-height: 200px; }
    .turn { margin-bottom: 25px; }
    .turn-q {
      background: #2c3e50; color: white; padding: 12px 16px; border-radius: 8px;
      margin-bottom: 12px; white-space: pre-wrap; word-break: break-word;
    }
    .turn-when { color: #95a5a6; font-size: 0.8em; margin: -8px 0 10px 4px; }
    .turn-status { color: #7f8c8d; font-style: italic; padding: 8px 0; }
    .member {
      background: white; border: 1px solid #e1e4e8; border-radius: 6px;
      padding: 12px 16px; margin-bottom: 10px;
    }
    .member-name { font-weight: bold; color: #2c3e50; font-size: 0.9em; }
    .member-name span { color: #7f8c8d; font-weight: normal; }
    .member-body, .head-body { white-space: pre-wrap; word-break: break-word; margin-top: 6px; }
    .member-failed { color: #c0392b; }
    .head {
      background: #f1f8ff; border: 1px solid #b6d4f5; border-left: 4px solid #3498db;
      border-radius: 6px; padding: 14px 18px; margin-top: 14px;
    }
    .head-value {
      margin-top: 10px; padding: 8px 12px; background: #2ecc71; color: white;
      border-radius: 4px; font-weight: bold; display: inline-block;
    }
    .head-novalue { margin-top: 10px; color: #c0392b; font-weight: bold; }
    .council-error { color: #c0392b; font-weight: bold; padding: 10px 0; }
    .table-stage { display: flex; justify-content: center; padding: 4px 0; }
    .table-stage svg { max-width: 460px; width: 100%; height: auto; }
    .table-phase {
      text-align: center; color: #7f8c8d; font-size: 0.9em; font-style: italic;
      min-height: 1.2em; margin-top: 4px;
    }
    .table-legend {
      display: flex; flex-wrap: wrap; gap: 4px 12px; justify-content: center;
      margin-top: 12px; font-size: 0.78em; color: #7f8c8d;
    }
    .table-legend span { display: flex; align-items: center; gap: 5px; }
    .table-legend i {
      width: 10px; height: 10px; border-radius: 50%; display: inline-block;
    }
    .seat-ring { animation: seat-pulse 1.4s ease-in-out infinite; transform-origin: center; }
    @keyframes seat-pulse { 0%,100% { opacity: 0.25; r: 26px; } 50% { opacity: 0.7; r: 32px; } }
    .report-line { stroke-dasharray: 4 4; animation: report-flow 0.8s linear infinite; }
    @keyframes report-flow { to { stroke-dashoffset: -8; } }
    @media (prefers-reduced-motion: reduce) {
      .seat-ring, .report-line { animation: none; }
    }
    /* what the members are saying, under the table */
    .voices { list-style: none; padding: 0; margin: 12px 0 0 0; font-size: 0.85em; }
    .voices li {
      display: flex; gap: 8px; align-items: baseline; padding: 5px 6px;
      border-top: 1px solid #f0f0f0; cursor: pointer;
    }
    .voices li:hover { background: #f8f9fa; }
    .voices i { width: 9px; height: 9px; border-radius: 50%; flex: none; display: inline-block; position: relative; top: 1px; }
    .voices b { color: #2c3e50; flex: none; }
    .voices span { color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
    .voices .voice-head b { color: #2980b9; }
    /* sessions */
    .sessions { list-style: none; padding: 0; margin: 8px 0 0 0; max-height: 320px; overflow-y: auto; }
    .sessions li {
      display: flex; gap: 8px; align-items: center; padding: 7px 8px; border-radius: 6px;
      cursor: pointer; border: 1px solid transparent;
    }
    .sessions li:hover { background: #f8f9fa; }
    .sessions li.active { background: #eaf4fd; border-color: #aed6f1; }
    .sessions .s-title { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #2c3e50; }
    .sessions .s-meta { color: #95a5a6; font-size: 0.78em; flex: none; }
    .sessions .s-del { color: #bdc3c7; background: none; border: none; cursor: pointer; font-size: 1em; padding: 0 2px; flex: none; }
    .sessions .s-del:hover { color: #c0392b; }
    .sessions .s-pending { width: 8px; height: 8px; border-radius: 50%; background: #f39c12; flex: none; }
    #council-input { width: 100%; padding: 12px; font-size: 1em; font-family: inherit;
      border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
    #council-context { width: 100%; padding: 10px; font-family: monospace; font-size: 0.85em;
      border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
    .council-bar { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
    .council-bar .nav-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
  <h1 style="margin-bottom: 5px;">🏛️ LLM Council</h1>
  <p style="color: #7f8c8d; margin-top: 0;" id="council-panel">Loading panel…</p>
  <div class="council-layout">
  <div class="council-main">
  <div class="card expanded">
    <div class="card-header" onclick="toggleCard(this)">
      <span class="card-title">Ask the council</span><div class="card-icon">▼</div>
    </div>
    <div class="card-body">
      <textarea id="council-input" rows="3" placeholder="Ask a question…"></textarea>
      <div class="council-bar">
        <button id="council-ask" class="nav-btn">Ask</button>
        <button id="council-new" class="nav-btn" title="Start a fresh conversation">New session</button>
        <span id="council-hint" style="color: #7f8c8d; font-size: 0.9em;"></span>
      </div>
      <details style="margin-top: 14px;">
        <summary style="cursor: pointer; color: #7f8c8d;">Context (optional)</summary>
        <p style="color: #7f8c8d; font-size: 0.85em; margin: 8px 0;">
          Sent to every member with the question. Every member sees the same text,
          so a wrong premise here misleads all of them at once.
        </p>
        <textarea id="council-context" rows="4" placeholder="Prior answers, constraints…"></textarea>
      </details>
    </div>
  </div>
  <div class="council-log" id="council-log"></div>
  </div><!-- /council-main -->
  <div class="council-side">
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)">
        <span class="card-title" style="font-size: 1.05em;">The table</span>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body" style="padding: 12px;">
        <div class="table-stage" id="council-table"></div>
        <div class="table-phase" id="council-phase">Waiting for a question.</div>
        <div class="table-legend">
          <span><i style="background:#dfe4ea"></i>waiting</span>
          <span><i style="background:#f39c12"></i>answering</span>
          <span><i style="background:#2ecc71"></i>answered</span>
          <span><i style="background:#3498db"></i>cached</span>
          <span><i style="background:#e74c3c"></i>failed</span>
        </div>
        <ul class="voices" id="council-voices"></ul>
      </div>
    </div>
    <div class="card expanded" style="margin-top: 16px;">
      <div class="card-header" onclick="toggleCard(this)">
        <span class="card-title" style="font-size: 1.05em;">Your sessions</span>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body" style="padding: 12px;">
        <p style="color:#7f8c8d; font-size:0.85em; margin:0;">Kept for your account. A session starts with its first question.</p>
        <ul class="sessions" id="council-sessions"><li style="color:#95a5a6; cursor:default;">Loading…</li></ul>
      </div>
    </div>
  </div><!-- /council-side -->
  </div><!-- /council-layout -->
  <script>
  (function () {
    var CSRF = ${JSON.stringify(req.user && req.user.csrf ? String(req.user.csrf) : '')};
    var log = document.getElementById('council-log');
    var input = document.getElementById('council-input');
    var contextBox = document.getElementById('council-context');
    var askBtn = document.getElementById('council-ask');
    var newBtn = document.getElementById('council-new');
    var hint = document.getElementById('council-hint');
    var sessionList = document.getElementById('council-sessions');
    var voices = document.getElementById('council-voices');
    var busy = false;
    var councilOut = false;     // the models are off: asking is disabled, reading is not
    var members = [];
    // The session shown right now, from the URL so a reload comes back to
    // it; null until the first question of a new conversation is accepted -
    // there is no empty session to create.
    var SESSION_ID = /^[a-f0-9]{12}$/;
    var currentSession = (function () {
      var id = new URLSearchParams(location.search).get('session');
      return id && SESSION_ID.test(id) ? id : null;
    })();
    function el(tag, cls, text) {
      var n = document.createElement(tag);
      if (cls) n.className = cls;
      if (text !== undefined) n.textContent = text;   // textContent, never innerHTML:
      return n;                                        // model output is untrusted.
    }
    function excerpt(text, max) {
      var one = String(text || '').replace(/\\s+/g, ' ').trim();
      return one.length > max ? one.slice(0, max - 1) + '…' : one;
    }
    function setSessionInUrl(id) {
      var url = new URL(location.href);
      if (id) url.searchParams.set('session', id); else url.searchParams.delete('session');
      history.replaceState(null, '', url.toString());
    }
    // Two different "not available" cases, and the page must tell them apart:
    // the API itself is down (fetch rejects, handled below), or the API is up
    // but the llama.cpp boxes behind it are off - api.py reports that in
    // d.status, and the models are deliberately not powered 24/7. Asking in
    // either state would only produce a failed run, so the button stays
    // disabled until the council can actually sit. Past sessions stay
    // readable either way: they are on this server, not on the model box.
    function summon(detail) {
      councilOut = true;
      document.getElementById('council-panel').textContent =
        'The members of the council have to be summoned.' + (detail ? '  (' + detail + ')' : '');
      askBtn.disabled = true;
      hint.textContent = 'the model endpoints are not answering - earlier sessions can still be read';
      tablePhase.textContent = 'No session - the council is not in.';
    }
    fetch('/council/api/endpoints').then(function (r) { return r.json(); }).then(function (d) {
      if (!d.members) return;
      members = d.members;
      var status = d.status;
      var mode = d.ask_members === 'parallel' ? 'asked all at once' : 'asked one after another';
      if (status && !status.ready) {
        var down = (status.members || []).filter(function (m) { return !m.ok; }).map(function (m) { return m.name; });
        if (status.head && !status.head.ok) down.push(status.head.name);
        return summon(down.length ? 'not answering: ' + down.join(', ') : 'no members reachable');
      }
      var names = d.members.map(function (m) { return m.name; }).join(', ');
      var downSome = status ? (status.total - status.reachable) : 0;
      document.getElementById('council-panel').textContent =
        'Members: ' + names + (d.head ? '  |  Head: ' + d.head.name : '') +
        '  |  Preset: ' + (d.preset || 'default') + '  |  ' + mode +
        (downSome > 0 ? '  |  ' + downSome + ' member(s) not answering' : '');
      // Show the empty table straight away, so the seats are visible before
      // the first question rather than appearing from nowhere.
      updateTable({
        seats: idleTable(d.members),
        phase: null,
        head: d.head ? { name: d.head.name, state: 'waiting' } : null
      });
      tablePhase.textContent = 'Waiting for a question.';
    }).catch(function () {
      summon('the council api is not running');
    });
    // Round table: one seat per member around the head. Seat colours come
    // from the run's actual progress (api.py reports which member is being
    // asked), never from a timer - an animation that does not track the run
    // would be decoration pretending to be information. A seat that has
    // answered gets a speech bubble with the opening of what it said; the
    // full text is the bubble's tooltip and the voices list under the table.
    var SEAT_COLOURS = {
      waiting:  { fill: '#dfe4ea', stroke: '#b2bec3', text: '#636e72' },
      asking:   { fill: '#f39c12', stroke: '#e67e22', text: '#ffffff' },
      answered: { fill: '#2ecc71', stroke: '#27ae60', text: '#ffffff' },
      cached:   { fill: '#3498db', stroke: '#2980b9', text: '#ffffff' },
      failed:   { fill: '#e74c3c', stroke: '#c0392b', text: '#ffffff' }
    };
    var SVGNS = 'http://www.w3.org/2000/svg';
    function svg(tag, attrs) {
      var n = document.createElementNS(SVGNS, tag);
      Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
      return n;
    }
    // Model names are mostly name + version ("qwen2.5-1.5b"), so splitting on
    // punctuation and taking one letter per part gives "q5". Take the letters
    // of the name itself instead: qwen -> qw, llama -> ll, phi -> ph.
    function initials(name) {
      var letters = String(name).replace(/^head-/, '').match(/^[a-zA-Z]+/);
      return letters ? letters[0].slice(0, 2).toLowerCase() : '?';
    }
    function seatSpeech(seat) {
      if (!seat) return null;
      if (seat.state === 'failed') return { text: 'failed: ' + (seat.error || 'no answer'), full: seat.error || 'failed' };
      if (seat.answer && (seat.state === 'answered' || seat.state === 'cached')) {
        return { text: excerpt(seat.answer, 26), full: seat.answer };
      }
      return null;
    }
    function drawTable(progress) {
      var W = 460, H = 280, cx = W / 2, cy = H / 2, R = 92;
      var seatNames = Object.keys(progress.seats || {});
      var root = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
      root.setAttribute('aria-label', 'Council progress');
      // the table itself
      root.appendChild(svg('ellipse', {
        cx: cx, cy: cy, rx: R + 6, ry: (R + 6) * 0.72,
        fill: '#f5f3ee', stroke: '#d8d2c4', 'stroke-width': 2
      }));
      var headState = (progress.head && progress.head.state) || 'waiting';
      // members reporting to the head: only drawn while the head is working,
      // which is when that is actually happening.
      var positions = seatNames.map(function (name, i) {
        var angle = (-90 + (360 / seatNames.length) * i) * Math.PI / 180;
        return { name: name, x: cx + R * Math.cos(angle), y: cy + R * 0.72 * Math.sin(angle),
                 ux: Math.cos(angle), uy: Math.sin(angle) };
      });
      if (headState === 'asking') {
        positions.forEach(function (p) {
          var line = svg('line', {
            x1: p.x, y1: p.y, x2: cx, y2: cy,
            stroke: '#3498db', 'stroke-width': 2, opacity: 0.55
          });
          line.setAttribute('class', 'report-line');
          root.appendChild(line);
        });
      }
      // the head, at the centre
      var headColour = SEAT_COLOURS[headState] || SEAT_COLOURS.waiting;
      if (headState === 'asking') {
        var halo = svg('circle', { cx: cx, cy: cy, r: 26, fill: headColour.fill, opacity: 0.3 });
        halo.setAttribute('class', 'seat-ring');
        root.appendChild(halo);
      }
      var headCircle = svg('circle', {
        cx: cx, cy: cy, r: 26, fill: headColour.fill,
        stroke: headColour.stroke, 'stroke-width': 3
      });
      if (progress.head && progress.head.answer) {
        var headTitle = svg('title', {});
        headTitle.textContent = progress.head.answer;
        headCircle.appendChild(headTitle);
      }
      root.appendChild(headCircle);
      var headLabel = svg('text', {
        x: cx, y: cy + 5, 'text-anchor': 'middle',
        fill: headColour.text, 'font-size': 15, 'font-weight': 'bold'
      });
      headLabel.textContent = 'H';
      root.appendChild(headLabel);
      // the members
      var bubbles = seatNames.length <= 6;   // past six seats the bubbles would overlap
      positions.forEach(function (p) {
        var seat = progress.seats[p.name] || { state: 'waiting' };
        var colour = SEAT_COLOURS[seat.state] || SEAT_COLOURS.waiting;
        if (seat.state === 'asking') {
          var ring = svg('circle', { cx: p.x, cy: p.y, r: 26, fill: colour.fill, opacity: 0.3 });
          ring.setAttribute('class', 'seat-ring');
          root.appendChild(ring);
        }
        var circle = svg('circle', {
          cx: p.x, cy: p.y, r: 21, fill: colour.fill,
          stroke: colour.stroke, 'stroke-width': 2.5
        });
        var speech = seatSpeech(seat);
        if (speech) {
          var title = svg('title', {});
          title.textContent = speech.full;
          circle.appendChild(title);
        }
        root.appendChild(circle);
        var mark = svg('text', {
          x: p.x, y: p.y + 4, 'text-anchor': 'middle',
          fill: colour.text, 'font-size': 12, 'font-weight': 'bold'
        });
        mark.textContent = seat.state === 'failed' ? '!' : initials(p.name);
        root.appendChild(mark);
        // Seats near the left and right rim sit close to the table edge, so
        // their labels are anchored outward instead of centred under the seat,
        // which otherwise overlaps the ellipse stroke.
        var side = (p.x - cx) / R;
        var anchor = side < -0.6 ? 'end' : (side > 0.6 ? 'start' : 'middle');
        var dx = anchor === 'end' ? 24 : (anchor === 'start' ? -24 : 0);
        var label = svg('text', {
          x: p.x - dx, y: p.y + (p.y > cy ? 36 : -28), 'text-anchor': anchor,
          fill: '#2c3e50', 'font-size': 10
        });
        label.textContent = p.name + (seat.seconds !== undefined ? ' (' + seat.seconds + 's)' : '');
        root.appendChild(label);
        // the speech bubble, pushed outward from the seat and kept inside the
        // drawing; the full answer is the tooltip of both bubble and seat.
        if (bubbles && speech) {
          var bw = 128, bh = 18;
          var bx = p.x + p.ux * 96 - bw / 2, by = p.y + p.uy * 74 - bh / 2;
          if (p.uy < -0.5) by = p.y - 28 - 14 - bh;          // above its label
          if (p.uy > 0.5) by = p.y + 36 + 6;                  // below its label
          bx = Math.max(2, Math.min(W - bw - 2, bx));
          by = Math.max(2, Math.min(H - bh - 2, by));
          var g = svg('g', {});
          var bubbleTitle = svg('title', {});
          bubbleTitle.textContent = speech.full;
          g.appendChild(bubbleTitle);
          g.appendChild(svg('rect', {
            x: bx, y: by, width: bw, height: bh, rx: 6, ry: 6,
            fill: seat.state === 'failed' ? '#fdecea' : '#ffffff',
            stroke: seat.state === 'failed' ? '#e74c3c' : '#b2bec3', 'stroke-width': 1
          }));
          var quote = svg('text', {
            x: bx + 7, y: by + 12.5, fill: seat.state === 'failed' ? '#c0392b' : '#2c3e50',
            'font-size': 9.5, 'font-style': 'italic'
          });
          quote.textContent = '“' + speech.text + '”';
          g.appendChild(quote);
          root.appendChild(g);
        }
      });
      return root;
    }
    // One table for the page, in the side column, showing the current run -
    // or the last one once it has finished. Per-turn tables would push the
    // conversation down and leave a row of stale diagrams behind.
    var tableStage = document.getElementById('council-table');
    var tablePhase = document.getElementById('council-phase');
    var PHASE_TEXT = {
      members: 'the council is deliberating…',
      head: 'the head is weighing the answers…',
      done: 'the council has reported.',
      starting: 'summoning the council…'
    };
    function updateTable(progress) {
      tableStage.innerHTML = '';
      tableStage.appendChild(drawTable(progress));
      tablePhase.textContent = PHASE_TEXT[progress.phase] || '';
      updateVoices(progress);
    }
    // What the members are saying, one line each, under the table - built
    // from the same progress the seats are coloured from, so it fills in as
    // the members finish. Clicking a line jumps to that member's full answer.
    function updateVoices(progress) {
      voices.innerHTML = '';
      var names = Object.keys(progress.seats || {});
      var any = false;
      names.forEach(function (name) {
        var seat = progress.seats[name] || {};
        var speech = seatSpeech(seat);
        var li = el('li');
        var dot = el('i');
        dot.style.background = (SEAT_COLOURS[seat.state] || SEAT_COLOURS.waiting).fill;
        li.appendChild(dot);
        li.appendChild(el('b', null, name));
        li.appendChild(el('span', null, speech ? excerpt(speech.full, 140)
          : (seat.state === 'asking' ? 'thinking…' : (seat.state === 'waiting' ? 'waiting' : ''))));
        if (speech) { any = true; li.title = speech.full; }
        li.addEventListener('click', function () {
          var box = log.querySelector('[data-member="' + name.replace(/"/g, '') + '"]:last-of-type');
          if (box) box.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
        voices.appendChild(li);
      });
      if (progress.head && (progress.head.answer || progress.head.state === 'asking' || progress.head.error)) {
        var hl = el('li', 'voice-head');
        var hd = el('i');
        hd.style.background = (SEAT_COLOURS[progress.head.state] || SEAT_COLOURS.waiting).fill;
        hl.appendChild(hd);
        hl.appendChild(el('b', null, progress.head.name || 'head'));
        hl.appendChild(el('span', null, progress.head.answer ? excerpt(progress.head.value ? 'ANSWER: ' + progress.head.value + ' — ' + progress.head.answer : progress.head.answer, 140)
          : (progress.head.error ? 'failed: ' + progress.head.error : 'weighing the answers…')));
        if (progress.head.answer) hl.title = progress.head.answer;
        voices.appendChild(hl);
      }
      voices.hidden = !names.length;
      if (!any && !names.some(function (n) { return progress.seats[n].state !== 'waiting'; })) voices.hidden = true;
    }
    function idleTable(members) {
      var seats = {};
      members.forEach(function (m) { seats[m.name] = { state: 'waiting' }; });
      return seats;
    }
    // A member's answer, in the conversation. Rendered as soon as the
    // progress carries it, and not again when the final result arrives.
    function renderMember(turn, m) {
      if (turn.querySelector('[data-member="' + String(m.name).replace(/"/g, '') + '"]')) return;
      var box = el('div', 'member');
      box.setAttribute('data-member', m.name);
      var name = el('div', 'member-name');
      name.textContent = m.name;
      var meta = el('span');
      meta.textContent = '  ' + (m.lab || '') + '  ' + (m.seconds !== undefined ? m.seconds + 's' : '') + (m.cached ? '  (cached)' : '');
      name.appendChild(meta);
      box.appendChild(name);
      box.appendChild(el('div', m.ok ? 'member-body' : 'member-body member-failed',
                         m.ok ? m.answer : 'failed: ' + m.error));
      var status = turn.querySelector('.turn-status');
      if (status) turn.insertBefore(box, status); else turn.appendChild(box);
    }
    function renderFromProgress(turn, progress) {
      Object.keys(progress.seats || {}).forEach(function (name) {
        var seat = progress.seats[name];
        if (seat.state === 'answered' || seat.state === 'cached') {
          renderMember(turn, { name: name, ok: true, answer: seat.answer, seconds: seat.seconds, cached: seat.state === 'cached' });
        } else if (seat.state === 'failed') {
          renderMember(turn, { name: name, ok: false, error: seat.error || 'no answer', seconds: seat.seconds });
        }
      });
    }
    function renderResult(turn, result) {
      (result.members || []).forEach(function (m) { renderMember(turn, m); });
      if (result.head && !turn.querySelector('.head')) {
        var head = el('div', 'head');
        head.appendChild(el('div', 'member-name', result.head.name));
        if (result.head.ok) {
          head.appendChild(el('div', 'head-body', result.head.answer));
          if (result.head.value) head.appendChild(el('div', 'head-value', result.head.value));
          else head.appendChild(el('div', 'head-novalue', 'no ANSWER: line found'));
        } else {
          head.appendChild(el('div', 'head-body member-failed', 'failed: ' + result.head.error));
        }
        turn.appendChild(head);
      }
    }
    // The last progress snapshot is dropped when a job finishes, so the
    // finished table is rebuilt from the result instead.
    function finalProgress(result) {
      var seats = {};
      (result.members || []).forEach(function (m) {
        seats[m.name] = {
          state: m.cached ? 'cached' : (m.ok ? 'answered' : 'failed'),
          seconds: m.seconds, answer: m.answer, error: m.error
        };
      });
      return {
        seats: seats,
        phase: 'done',
        head: result.head
          ? { name: result.head.name,
              state: result.head.ok ? 'answered' : 'failed',
              seconds: result.head.seconds, answer: result.head.answer,
              value: result.head.value, error: result.head.error }
          : null
      };
    }
    function poll(id, turn, status, turnId) {
      fetch('/council/api/job/' + encodeURIComponent(id))
        .then(function (r) { return r.json().then(function (b) { return { status: r.status, body: b }; }); })
        .then(function (res) {
          var job = res.body || {};
          if (res.status === 404) {
            // The API restarted and forgot the run. The server's own tracker
            // records that on the turn; read it back rather than guessing.
            return settleFromSession(turn, status, turnId);
          }
          if (job.state === 'done') {
            status.remove();
            if (job.result) updateTable(finalProgress(job.result));
            renderResult(turn, job.result);
            finish();
          } else if (job.state === 'failed') {
            status.remove();
            turn.appendChild(el('div', 'council-error', job.error || 'run failed'));
            tablePhase.textContent = 'the council could not sit.';
            finish();
          } else {
            if (job.progress && job.progress.seats) {
              updateTable(job.progress);
              renderFromProgress(turn, job.progress);
              status.textContent = PHASE_TEXT[job.progress.phase] || 'working…';
            } else {
              status.textContent = (job.state || 'working') + '…';
            }
            setTimeout(function () { poll(id, turn, status, turnId); }, 2000);
          }
        })
        .catch(function (e) {
          // A single failed poll is not a failed run - the job keeps going.
          status.textContent = 'lost contact, retrying…';
          setTimeout(function () { poll(id, turn, status, turnId); }, 5000);
        });
    }
    function settleFromSession(turn, status, turnId) {
      if (!currentSession || !turnId) {
        status.remove();
        turn.appendChild(el('div', 'council-error', 'the council api no longer knows this run - ask again'));
        return finish();
      }
      fetch('/council/api/sessions/' + currentSession).then(function (r) { return r.json(); }).then(function (d) {
        var t = ((d.session || {}).turns || []).filter(function (x) { return x.id === turnId; })[0];
        status.remove();
        if (t && t.state === 'done' && t.result) { renderResult(turn, t.result); updateTable(finalProgress(t.result)); }
        else turn.appendChild(el('div', 'council-error', (t && t.error) || 'the council api no longer knows this run - ask again'));
        finish();
      }).catch(function () { status.remove(); finish(); });
    }
    function finish() {
      busy = false;
      askBtn.disabled = councilOut;
      hint.textContent = councilOut ? hint.textContent : '';
      input.focus();
      loadSessions();
    }
    function newTurn(question, when) {
      var turn = el('div', 'turn');
      turn.appendChild(el('div', 'turn-q', question));
      if (when) turn.appendChild(el('div', 'turn-when', new Date(when).toLocaleString()));
      log.appendChild(turn);
      return turn;
    }
    function ask() {
      var question = input.value.trim();
      if (!question || busy || councilOut) return;
      busy = true;
      askBtn.disabled = true;
      hint.textContent = 'running - this can take several minutes';
      var turn = newTurn(question, null);
      var status = el('div', 'turn-status', 'starting…');
      turn.appendChild(status);
      turn.scrollIntoView({ behavior: 'smooth', block: 'start' });
      input.value = '';
      fetch('/council/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question, context: contextBox.value.trim() || null,
                               session: currentSession, _csrf: CSRF })
      })
        .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
        .then(function (res) {
          if (!res.ok) {
            status.remove();
            turn.appendChild(el('div', 'council-error',
              res.body.busy ? 'The council is already answering another question. Try again shortly.'
                            : (res.body.error || 'could not start the run')));
            finish();
            return;
          }
          if (res.body.session && res.body.session !== currentSession) {
            currentSession = res.body.session;
            setSessionInUrl(currentSession);
          }
          loadSessions();
          poll(res.body.id, turn, status, res.body.turn);
        })
        .catch(function (e) {
          status.remove();
          turn.appendChild(el('div', 'council-error', 'could not reach the server: ' + e.message));
          finish();
        });
    }
    // --- sessions ------------------------------------------------------------
    function loadSessions() {
      fetch('/council/api/sessions').then(function (r) { return r.json(); }).then(function (d) {
        sessionList.innerHTML = '';
        var list = d.sessions || [];
        if (!list.length) {
          var empty = el('li', null, 'No sessions yet - your first question starts one.');
          empty.style.color = '#95a5a6'; empty.style.cursor = 'default';
          sessionList.appendChild(empty);
          return;
        }
        list.forEach(function (s) {
          var li = el('li', s.id === currentSession ? 'active' : '');
          if (s.pending) li.appendChild(el('span', 's-pending'));
          var title = el('span', 's-title', s.title || '(untitled)');
          title.title = s.title || '';
          li.appendChild(title);
          li.appendChild(el('span', 's-meta', s.turns + (s.turns === 1 ? ' turn · ' : ' turns · ') + new Date(s.updated).toLocaleDateString()));
          var del = el('button', 's-del', '×');
          del.title = 'Delete this session';
          del.addEventListener('click', function (e) {
            e.stopPropagation();
            if (!confirm('Delete this session and its ' + s.turns + ' turn(s)?')) return;
            fetch('/council/api/sessions/' + s.id + '/delete', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ _csrf: CSRF })
            }).then(function () {
              if (s.id === currentSession) newSession();
              loadSessions();
            });
          });
          li.appendChild(del);
          li.addEventListener('click', function () { openSession(s.id); });
          sessionList.appendChild(li);
        });
      }).catch(function () {
        sessionList.innerHTML = '';
        sessionList.appendChild(el('li', null, 'could not load your sessions'));
      });
    }
    function openSession(id) {
      if (busy) return;
      fetch('/council/api/sessions/' + id).then(function (r) { return r.json(); }).then(function (d) {
        if (!d.session) return;
        currentSession = id;
        setSessionInUrl(id);
        log.innerHTML = '';
        var lastResult = null;
        d.session.turns.forEach(function (t) {
          var turn = newTurn(t.question, t.asked);
          if (t.state === 'done' && t.result) { renderResult(turn, t.result); lastResult = t.result; }
          else if (t.state === 'failed') turn.appendChild(el('div', 'council-error', t.error || 'run failed'));
          else if (t.state === 'pending') {
            var status = el('div', 'turn-status', 'still running…');
            turn.appendChild(status);
            if (t.jobId) { busy = true; askBtn.disabled = true; poll(t.jobId, turn, status, t.id); }
          }
        });
        if (lastResult) updateTable(finalProgress(lastResult));
        loadSessions();
        var last = log.lastElementChild;
        if (last) last.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
    function newSession() {
      if (busy) return;
      currentSession = null;
      setSessionInUrl(null);
      log.innerHTML = '';
      updateTable({ seats: idleTable(members), phase: null, head: null });
      tablePhase.textContent = 'Waiting for a question.';
      loadSessions();
      input.focus();
    }
    askBtn.addEventListener('click', ask);
    newBtn.addEventListener('click', newSession);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
    });
    if (currentSession) openSession(currentSession); else loadSessions();
    input.focus();
  })();
  </script>
  ` + footer();
}

module.exports = { install };
