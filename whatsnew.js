// First-login introduction and the "what's new" note, for a server-rendered
// UI with no framework and no build step.
//
// What an account has already seen lives in config/user-state.json - not in
// users.json, because the administrator has no record there (the credential
// is the environment's), and password hashes should not sit next to churny
// interface state. The content lives in announcements.js, which is the only
// file to edit when a feature ships.
//
// The dialog is injected by generateHeader, so it can appear on whichever
// page the reader lands on, and it is plain HTML with a form: dismissing it
// works without JavaScript. "Later" only hides it for the session.

const fs = require('fs');
const path = require('path');

const config = require('./config');
const { ANNOUNCEMENTS, TOUR } = require('./announcements');

const STATE_FILE = path.join(config.CONFIG_DIR, 'user-state.json');

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// --- per-account state ------------------------------------------------------
function loadState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) {
    if (e.code !== 'ENOENT') console.log(`user-state.json could not be read (${e.message}) - treating it as empty`);
    return {};
  }
}

function saveState(state) {
  try {
    fs.mkdirSync(config.CONFIG_DIR, { recursive: true, mode: 0o700 });
    const tmp = `${STATE_FILE}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(state, null, 2), { mode: 0o600 });
    fs.renameSync(tmp, STATE_FILE);
  } catch (e) {
    // Losing this only means someone is shown the note twice.
    console.log(`Could not write user-state.json: ${e.message}`);
  }
}

function stateOf(user) {
  return loadState()[user.name] || {};
}

function markSeen(user, ids, tourDone) {
  const state = loadState();
  const mine = state[user.name] || {};
  const seen = new Set(mine.seen || []);
  ids.forEach((id) => seen.add(id));
  state[user.name] = {
    ...mine,
    seen: [...seen].slice(-200),        // ids are permanent; keep the recent ones
    tourDone: tourDone || mine.tourDone || false,
    updatedAt: new Date().toISOString(),
  };
  saveState(state);
}

// --- what this account should be shown --------------------------------------
function visible(user) {
  return ANNOUNCEMENTS.filter((a) => a.audience !== 'admin' || user.role === 'admin');
}

function unseen(user) {
  const seen = new Set(stateOf(user).seen || []);
  return visible(user).filter((a) => !seen.has(a.id));
}

// What to show right now: the introduction for an account that has never seen
// anything, otherwise the unseen 'major' notes. 'minor' ones only light the
// navbar dot. Never for the open-access pseudo user - there is no account to
// record it against.
function pending(user) {
  if (!user || user.open) return null;
  const state = stateOf(user);
  const items = unseen(user);
  if (!state.tourDone) return { kind: 'tour', items };
  const major = items.filter((a) => a.level === 'major');
  return major.length ? { kind: 'news', items: major } : null;
}

function hasUnseen(user) {
  return Boolean(user && !user.open && (!stateOf(user).tourDone || unseen(user).length));
}

// --- rendering ---------------------------------------------------------------
const DIALOG_CSS = `
  .sp-dialog-backdrop {
    position: fixed; inset: 0; background: rgba(44, 62, 80, 0.55); z-index: 3000;
    display: flex; align-items: center; justify-content: center; padding: 20px;
  }
  .sp-dialog {
    background: #fff; border-radius: 10px; max-width: 640px; width: 100%;
    max-height: 85vh; overflow-y: auto; box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    padding: 26px 30px; color: #2c3e50;
  }
  .sp-dialog h2 { margin: 0 0 6px 0; color: #2c3e50; }
  .sp-dialog h3 { margin: 18px 0 4px 0; font-size: 1.05em; color: #2c3e50; }
  .sp-dialog p { margin: 6px 0; line-height: 1.5; }
  .sp-dialog .sp-date { color: #7f8c8d; font-size: 0.85em; }
  .sp-dialog .sp-actions { margin-top: 22px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .sp-dialog .sp-later { color: #7f8c8d; background: none; border: none; cursor: pointer; text-decoration: underline; font-size: 0.95em; }
  .sp-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #e67e22; margin-left: 4px; vertical-align: super; }
  @media (max-width: 600px) { .sp-dialog { padding: 20px 18px; } }
`;

function body(item) {
  return item.body.map((line) => `<p>${escapeHtml(line)}</p>`).join('');
}

function dialog(user) {
  const show = pending(user);
  if (!show) return '';
  const ids = show.items.map((a) => a.id);
  const csrf = escapeHtml(user.csrf || '');

  const content = show.kind === 'tour'
    ? `<h2>${escapeHtml(TOUR.title)}, ${escapeHtml(user.name)}</h2>
       <p>${escapeHtml(TOUR.intro)}</p>
       ${TOUR.steps.filter((s) => s.audience !== 'admin' || user.role === 'admin')
         .map((s) => `<h3>${escapeHtml(s.title)}</h3><p>${escapeHtml(s.body)}</p>`).join('')}`
    : `<h2>What's new</h2>
       ${show.items.map((a) => `
          <h3>${escapeHtml(a.title)} <span class="sp-date">${escapeHtml(a.date)}</span></h3>
          ${body(a)}
          ${a.link ? `<p><a href="${escapeHtml(a.link.href)}">${escapeHtml(a.link.label)}</a></p>` : ''}`).join('')}`;

  return `
    <div class="sp-dialog-backdrop" id="sp-dialog" role="dialog" aria-modal="true" aria-labelledby="sp-dialog-title">
      <div class="sp-dialog" id="sp-dialog-title">
        ${content}
        <div class="sp-actions">
          <form method="post" action="/whats-new/seen">
            <input type="hidden" name="_csrf" value="${csrf}">
            ${ids.map((id) => `<input type="hidden" name="id" value="${escapeHtml(id)}">`).join('')}
            ${show.kind === 'tour' ? '<input type="hidden" name="tour" value="1">' : ''}
            <button type="submit" class="nav-btn" style="background:#27ae60; border-color:#1e8449;">${show.kind === 'tour' ? 'Start using it' : 'Got it'}</button>
          </form>
          <button type="button" class="sp-later" onclick="document.getElementById('sp-dialog').remove();">Later</button>
          <span style="color:#7f8c8d; font-size:0.85em;">Everything here stays on the <a href="/whats-new">What's new</a> page.</span>
        </div>
      </div>
    </div>
    <script>
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') { var d = document.getElementById('sp-dialog'); if (d) d.remove(); }
      });
    </script>`;
}

// The navbar entry: a question mark that carries a dot while something is
// unseen, so a minor note is noticeable without interrupting anyone.
function navLink(user) {
  if (!user || user.open) return '';
  return `<a href="/whats-new" title="What's new">?${hasUnseen(user) ? '<span class="sp-dot"></span>' : ''}</a>`;
}

function page(req, render) {
  const user = req.user;
  const items = visible(user);
  const seen = new Set(stateOf(user).seen || []);
  return render.header("What's new", user) + `
    <h1>What's new</h1>
    <p style="color:#7f8c8d;">Everything added to this site, newest first. ${user.open ? '' : 'The introduction can be opened again below.'}</p>
    ${items.map((a) => `
      <div class="card expanded">
        <div class="card-header" onclick="toggleCard(this)">
          <div>
            <span class="card-title">${escapeHtml(a.title)}</span>
            <span class="card-meta" style="margin-left:10px;">${escapeHtml(a.date)}${seen.has(a.id) ? '' : ' · new'}</span>
          </div>
          <div class="card-icon">▼</div>
        </div>
        <div class="card-body">
          ${body(a)}
          ${a.link ? `<p><a href="${escapeHtml(a.link.href)}" class="nav-btn" style="text-decoration:none;">${escapeHtml(a.link.label)}</a></p>` : ''}
        </div>
      </div>`).join('')}
    ${user.open ? '' : `
    <form method="post" action="/whats-new/replay" style="margin-top:20px;">
      <input type="hidden" name="_csrf" value="${escapeHtml(user.csrf || '')}">
      <button type="submit" class="nav-btn">Show me the introduction again</button>
    </form>`}` + render.footer();
}

function install(app, render, auth) {
  app.get('/whats-new', (req, res) => res.send(page(req, render)));

  app.post('/whats-new/seen', (req, res) => {
    if (!auth.csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    const body = req.body || {};
    // A browser sends one `id` field per note, which Express turns into an
    // array; a single note arrives as a plain string. Comma-separated values
    // are accepted too, so a hand-made request cannot silently store one
    // nonsense id instead of four real ones.
    const ids = [].concat(body.id || [])
      .flatMap((value) => String(value).split(','))
      .map((value) => value.trim())
      .filter(Boolean)
      .slice(0, 50);
    markSeen(req.user, ids, body.tour === '1');
    res.redirect(auth.safeNext(req.get('referer') ? new URL(req.get('referer'), 'http://x').pathname : '/'));
  });

  app.post('/whats-new/replay', (req, res) => {
    if (!auth.csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    const state = loadState();
    if (state[req.user.name]) { state[req.user.name].tourDone = false; saveState(state); }
    res.redirect('/');
  });
}

module.exports = { install, dialog, navLink, page, pending, hasUnseen, DIALOG_CSS, STATE_FILE };
