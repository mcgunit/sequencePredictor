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

function install(app, { header, footer, escapeHtml }) {
  // JSON bodies for the two proxied POSTs only. server.js installs
  // express.urlencoded for its forms; this adds JSON without touching that.
  const jsonBody = require('express').json({ limit: '64kb' });

  app.get('/council', (req, res) => {
    if (denied(req)) return res.status(403).send('Not available for your account.');
    res.send(page(req, header, footer, escapeHtml));
  });

  app.post('/council/api/ask', jsonBody, async (req, res) => {
    if (denied(req)) return res.status(403).json({ error: 'not permitted' });
    try {
      const { status, body } = await callApi('POST', '/ask', {
        question: req.body && req.body.question,
        context: req.body && req.body.context
      });
      res.status(status).json(body);
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
    .table-stage svg { max-width: 420px; width: 100%; height: auto; }
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
    #council-input { width: 100%; padding: 12px; font-size: 1em; font-family: inherit;
      border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
    #council-context { width: 100%; padding: 10px; font-family: monospace; font-size: 0.85em;
      border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
    .council-bar { display: flex; gap: 10px; align-items: center; margin-top: 10px; }
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
    var hint = document.getElementById('council-hint');
    var busy = false;

    function el(tag, cls, text) {
      var n = document.createElement(tag);
      if (cls) n.className = cls;
      if (text !== undefined) n.textContent = text;   // textContent, never innerHTML:
      return n;                                        // model output is untrusted.
    }

    // Two different "not available" cases, and the page must tell them apart:
    // the API itself is down (fetch rejects, handled below), or the API is up
    // but the llama.cpp boxes behind it are off - api.py reports that in
    // d.status, and the models are deliberately not powered 24/7. Asking in
    // either state would only produce a failed run, so the button stays
    // disabled until the council can actually sit.
    function summon(detail) {
      document.getElementById('council-panel').textContent =
        'The members of the council have to be summoned.' + (detail ? '  (' + detail + ')' : '');
      askBtn.disabled = true;
      hint.textContent = 'the model endpoints are not answering';
      tablePhase.textContent = 'No session - the council is not in.';
    }

    fetch('/council/api/endpoints').then(function (r) { return r.json(); }).then(function (d) {
      if (!d.members) return;
      var status = d.status;
      if (status && !status.ready) {
        var down = (status.members || []).filter(function (m) { return !m.ok; }).map(function (m) { return m.name; });
        if (status.head && !status.head.ok) down.push(status.head.name);
        return summon(down.length ? 'not answering: ' + down.join(', ') : 'no members reachable');
      }
      var names = d.members.map(function (m) { return m.name; }).join(', ');
      var downSome = status ? (status.total - status.reachable) : 0;
      document.getElementById('council-panel').textContent =
        'Members: ' + names + (d.head ? '  |  Head: ' + d.head.name : '') +
        '  |  Preset: ' + (d.preset || 'default') +
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
    // would be decoration pretending to be information.
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

    function drawTable(progress) {
      var W = 440, H = 250, cx = W / 2, cy = H / 2, R = 92;
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
        return { name: name, x: cx + R * Math.cos(angle), y: cy + R * 0.72 * Math.sin(angle) };
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
      root.appendChild(svg('circle', {
        cx: cx, cy: cy, r: 26, fill: headColour.fill,
        stroke: headColour.stroke, 'stroke-width': 3
      }));
      var headLabel = svg('text', {
        x: cx, y: cy + 5, 'text-anchor': 'middle',
        fill: headColour.text, 'font-size': 15, 'font-weight': 'bold'
      });
      headLabel.textContent = 'H';
      root.appendChild(headLabel);

      // the members
      positions.forEach(function (p) {
        var seat = progress.seats[p.name] || { state: 'waiting' };
        var colour = SEAT_COLOURS[seat.state] || SEAT_COLOURS.waiting;

        if (seat.state === 'asking') {
          var ring = svg('circle', { cx: p.x, cy: p.y, r: 26, fill: colour.fill, opacity: 0.3 });
          ring.setAttribute('class', 'seat-ring');
          root.appendChild(ring);
        }
        root.appendChild(svg('circle', {
          cx: p.x, cy: p.y, r: 21, fill: colour.fill,
          stroke: colour.stroke, 'stroke-width': 2.5
        }));

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
    }

    function idleTable(members) {
      var seats = {};
      members.forEach(function (m) { seats[m.name] = { state: 'waiting' }; });
      return seats;
    }

    function renderResult(turn, result) {
      (result.members || []).forEach(function (m) {
        var box = el('div', 'member');
        var name = el('div', 'member-name');
        name.textContent = m.name;
        var meta = el('span');
        meta.textContent = '  ' + (m.lab || '') + '  ' + m.seconds + 's' + (m.cached ? '  (cached)' : '');
        name.appendChild(meta);
        box.appendChild(name);
        box.appendChild(el('div', m.ok ? 'member-body' : 'member-body member-failed',
                           m.ok ? m.answer : 'failed: ' + m.error));
        turn.appendChild(box);
      });

      if (result.head) {
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

    function poll(id, turn, status) {
      fetch('/council/api/job/' + encodeURIComponent(id))
        .then(function (r) { return r.json(); })
        .then(function (job) {
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
              status.textContent = PHASE_TEXT[job.progress.phase] || 'working…';
            } else {
              status.textContent = (job.state || 'working') + '…';
            }
            setTimeout(function () { poll(id, turn, status); }, 2000);
          }
        })
        .catch(function (e) {
          // A single failed poll is not a failed run - the job keeps going.
          status.textContent = 'lost contact, retrying…';
          setTimeout(function () { poll(id, turn, status); }, 5000);
        });
    }

    // The last progress snapshot is dropped when a job finishes, so the
    // finished table is rebuilt from the result instead.
    function finalProgress(result) {
      var seats = {};
      (result.members || []).forEach(function (m) {
        seats[m.name] = {
          state: m.cached ? 'cached' : (m.ok ? 'answered' : 'failed'),
          seconds: m.seconds
        };
      });
      return {
        seats: seats,
        phase: 'done',
        head: result.head
          ? { name: result.head.name,
              state: result.head.ok ? 'answered' : 'failed',
              seconds: result.head.seconds }
          : null
      };
    }

    function finish() {
      busy = false;
      askBtn.disabled = false;
      hint.textContent = '';
      input.focus();
    }

    function ask() {
      var question = input.value.trim();
      if (!question || busy) return;

      busy = true;
      askBtn.disabled = true;
      hint.textContent = 'running - this can take several minutes';

      var turn = el('div', 'turn');
      turn.appendChild(el('div', 'turn-q', question));
      var status = el('div', 'turn-status', 'starting…');
      turn.appendChild(status);
      log.appendChild(turn);
      turn.scrollIntoView({ behavior: 'smooth', block: 'start' });
      input.value = '';

      fetch('/council/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': CSRF },
        body: JSON.stringify({ question: question, context: contextBox.value.trim() || null })
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
          poll(res.body.id, turn, status);
        })
        .catch(function (e) {
          status.remove();
          turn.appendChild(el('div', 'council-error', 'could not reach the server: ' + e.message));
          finish();
        });
    }

    askBtn.addEventListener('click', ask);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
    });
    input.focus();
  })();
  </script>
  ` + footer();
}

module.exports = { install };