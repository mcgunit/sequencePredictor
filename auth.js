// Login and basic user management for the web UI (README roadmap item 3).
//
// One admin comes from the environment (WEB_USER / WEB_PASSWORD, read in
// config.js) and never touches the disk; further users live in
// config/users.json (gitignored, salted scrypt hashes) and are managed by the
// admin on /admin/users. Two roles: "admin" sees everything, "user" sees the
// prediction and History pages. Sessions are a signed HttpOnly SameSite cookie
// (HMAC-SHA256 with a secret from WEB_SESSION_SECRET or generated once into
// config/session.secret), so no session store and no npm dependency is needed.
// With WEB_USER / WEB_PASSWORD both unset, authentication is off (local
// development): every page is open, the user page is read-only and shows a
// notice. Exactly one of the two set is a configuration error (see
// misconfigured()) and the server refuses to start.
//
// The password travels in clear text in the login form, so the app should sit
// behind HTTPS; the cookie gets the Secure flag whenever the request arrived
// over HTTPS (directly or via X-Forwarded-Proto). server.js sets
// app.set('trust proxy', config.TRUST_PROXY) so req.ip - the login lockout
// key - is the client behind a local reverse proxy, not the proxy itself.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const config = require('./config');
const audit = require('./audit');

const COOKIE_NAME = 'sp_session';
const SESSION_HOURS = 24;                 // renewed while in use ...
const SESSION_MAX_DAYS = 7;               // ... but never beyond this from the sign-in
const LOGIN_MAX_FAILURES = 5;             // per client address + user name ...
const LOGIN_WINDOW_MS = 15 * 60 * 1000;   // ... within this window, then locked for the rest of it
const FAILURE_MAP_LIMIT = 5000;           // tracked address|user keys before pruning
const MIN_SECRET_BYTES = 32;
const USERNAME_RE = /^[A-Za-z0-9._-]{3,32}$/;
const MIN_PASSWORD_LENGTH = 8;

const USERS_FILE = path.join(config.CONFIG_DIR, 'users.json');
const SECRET_FILE = path.join(config.CONFIG_DIR, 'session.secret');

function enabled() {
  return Boolean(config.ADMIN_USER && config.ADMIN_PASSWORD);
}

function misconfigured() {
  return Boolean(config.ADMIN_USER) !== Boolean(config.ADMIN_PASSWORD);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// --- users file -------------------------------------------------------------
function ensureConfigDir() {
  fs.mkdirSync(config.CONFIG_DIR, { recursive: true, mode: 0o700 });
}

function loadUsers() {
  try {
    const parsed = JSON.parse(fs.readFileSync(USERS_FILE, 'utf-8'));
    return Array.isArray(parsed.users) ? parsed.users.filter((u) => u && typeof u.name === 'string') : [];
  } catch (e) {
    if (e.code !== 'ENOENT') console.log(`users.json could not be read (${e.message}) - treating it as empty`);
    return [];
  }
}

function saveUsers(users) {
  ensureConfigDir();
  const tmp = `${USERS_FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify({ users }, null, 2), { mode: 0o600 });
  fs.renameSync(tmp, USERS_FILE);
}

function findUser(users, name) {
  return users.find((u) => u.name === name);
}

// --- passwords --------------------------------------------------------------
function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  return { salt, hash: crypto.scryptSync(password, salt, 64).toString('hex') };
}

function verifyPassword(password, salt, hash) {
  const candidate = crypto.scryptSync(String(password), String(salt), 64);
  let stored;
  try { stored = Buffer.from(String(hash), 'hex'); } catch (e) { return false; }
  return stored.length === candidate.length && crypto.timingSafeEqual(candidate, stored);
}

// The admin password is hashed once per process with a random salt so the
// check runs through the same constant-time comparison as user passwords, and
// a dummy credential keeps the cost of a sign-in with an unknown user name the
// same as with a known one (no user-name enumeration through response time).
const adminCredential = enabled() ? hashPassword(config.ADMIN_PASSWORD) : null;
const dummyCredential = hashPassword(crypto.randomBytes(16).toString('hex'));

function safeEqual(a, b) {
  const left = Buffer.from(String(a)); const right = Buffer.from(String(b));
  return left.length === right.length && crypto.timingSafeEqual(left, right);
}

// --- session cookie ---------------------------------------------------------
let sessionSecret = null;
function secret() {
  if (sessionSecret) return sessionSecret;
  if (config.SESSION_SECRET) {
    const value = String(config.SESSION_SECRET).trim();
    const buffer = /^[0-9a-fA-F]+$/.test(value) && value.length % 2 === 0 ? Buffer.from(value, 'hex') : Buffer.from(value, 'utf-8');
    if (buffer.length >= MIN_SECRET_BYTES) { sessionSecret = buffer; return sessionSecret; }
    console.log(`WEB_SESSION_SECRET has ${buffer.length} bytes, fewer than ${MIN_SECRET_BYTES} - ignoring it and using ${SECRET_FILE}`);
  }
  try {
    const stored = Buffer.from(fs.readFileSync(SECRET_FILE, 'utf-8').trim(), 'hex');
    if (stored.length >= MIN_SECRET_BYTES) { sessionSecret = stored; return sessionSecret; }
  } catch (e) { /* generated below */ }
  ensureConfigDir();
  sessionSecret = crypto.randomBytes(32);
  fs.writeFileSync(SECRET_FILE, sessionSecret.toString('hex'), { mode: 0o600 });
  return sessionSecret;
}

function sign(data) {
  return crypto.createHmac('sha256', secret()).update(data).digest('base64url');
}

// Ties an admin session to the current WEB_USER / WEB_PASSWORD: rotating the
// password ends every admin session. Keyed with the session secret, so the
// value in a cookie cannot be used to test password guesses offline.
function adminFingerprint() {
  return crypto.createHmac('sha256', secret()).update(`admin\n${config.ADMIN_USER}\n${config.ADMIN_PASSWORD}`).digest('base64url').slice(0, 22);
}

function isHttps(req) {
  return req.secure || String(req.headers['x-forwarded-proto'] || '').split(',')[0].trim() === 'https';
}

// A fresh sign-in gets a new token, issue time and absolute limit; a renewal
// (existing given) keeps all three so forms rendered before the renewal keep
// working and a leaked cookie still dies at the absolute limit.
function issueSession(req, res, name, role, existing = null) {
  const now = Date.now();
  const payload = {
    u: name,
    r: role,
    iat: existing ? existing.iat : now,
    max: existing ? existing.max : now + SESSION_MAX_DAYS * 86400 * 1000,
    csrf: existing ? existing.csrf : crypto.randomBytes(16).toString('hex'),
  };
  payload.exp = Math.min(now + SESSION_HOURS * 3600 * 1000, payload.max);
  if (role === 'admin') payload.a = adminFingerprint();
  const data = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const flags = [`${COOKIE_NAME}=${data}.${sign(data)}`, 'Path=/', 'HttpOnly', 'SameSite=Strict',
    `Max-Age=${Math.max(1, Math.floor((payload.exp - now) / 1000))}`];
  if (isHttps(req)) flags.push('Secure');
  res.setHeader('Set-Cookie', flags.join('; '));
  return payload;
}

function clearSession(res) {
  res.setHeader('Set-Cookie', `${COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0`);
}

function readSession(req) {
  const header = String(req.headers.cookie || '');
  if (header.length > 4096) return null;
  const raw = header.split(';').map((c) => c.trim()).find((c) => c.startsWith(`${COOKIE_NAME}=`));
  if (!raw) return null;
  const value = raw.slice(COOKIE_NAME.length + 1);
  const dot = value.lastIndexOf('.');
  if (dot < 0) return null;
  const data = value.slice(0, dot); const signature = value.slice(dot + 1);
  if (!safeEqual(signature, sign(data))) return null;
  try {
    const payload = JSON.parse(Buffer.from(data, 'base64url').toString('utf-8'));
    if (!payload || typeof payload.u !== 'string' || typeof payload.csrf !== 'string') return null;
    const now = Date.now();
    if (!(payload.exp > now) || !(payload.max > now) || !(payload.iat <= now)) return null;
    return payload;
  } catch (e) { return null; }
}

// Is the signed session still backed by a live account? Admin sessions must
// carry the fingerprint of the current WEB_USER / WEB_PASSWORD; a user's
// session must be younger than the user's last password change and the user
// must still exist. The role comes from here, never from the cookie alone.
function resolveSession(session) {
  if (session.r === 'admin') {
    if (!enabled() || session.u !== config.ADMIN_USER || !safeEqual(session.a || '', adminFingerprint())) return null;
    return { name: session.u, role: 'admin' };
  }
  const user = findUser(loadUsers(), session.u);
  if (!user) return null;
  const changed = Date.parse(user.passwordChangedAt || user.createdAt || '') || 0;
  if (session.iat < changed) return null;
  return { name: user.name, role: 'user' };
}

// --- login rate limit ---------------------------------------------------------
const failures = new Map(); // "address|user" -> { count, first }
function failureKey(req, name) { return `${req.ip}|${String(name).toLowerCase()}`; }
function pruneFailures(now) {
  if (failures.size < FAILURE_MAP_LIMIT) return;
  for (const [key, entry] of failures) if (now - entry.first > LOGIN_WINDOW_MS) failures.delete(key);
  while (failures.size >= FAILURE_MAP_LIMIT) failures.delete(failures.keys().next().value); // oldest first
}
function isLocked(req, name) {
  const key = failureKey(req, name); const entry = failures.get(key);
  if (!entry) return false;
  if (Date.now() - entry.first > LOGIN_WINDOW_MS) { failures.delete(key); return false; }
  return entry.count >= LOGIN_MAX_FAILURES;
}
function recordFailure(req, name) {
  const key = failureKey(req, name); const now = Date.now();
  pruneFailures(now);
  const entry = failures.get(key);
  if (!entry || now - entry.first > LOGIN_WINDOW_MS) failures.set(key, { count: 1, first: now });
  else entry.count += 1;
}
function clearFailures(req, name) { failures.delete(failureKey(req, name)); }

// --- middleware ---------------------------------------------------------------
// Only a plain same-origin path survives: no scheme, no host, no backslash
// (browsers read "/\evil.com" as "//evil.com"), no whitespace.
function safeNext(target) {
  if (typeof target !== 'string' || target.length > 512 || !target.startsWith('/') || target.startsWith('//') || /[\\\s]/.test(target)) return '/';
  try {
    const url = new URL(target, 'http://sp.local');
    if (url.origin !== 'http://sp.local' || url.username || url.password) return '/';
    return url.pathname + url.search;
  } catch (e) { return '/'; }
}

// Open access has no session to bind a form token to, and an empty token
// makes csrfOk() refuse every POST - which silently disabled every button on
// the Jobs page (start a service, run a job) in local development. A token
// minted once per process keeps csrfOk's shape and its protection (a
// cross-origin page still cannot read it) while letting the forms work. It
// grants nothing: the open-access guards on user management are separate and
// stay.
const OPEN_ACCESS_CSRF = crypto.randomBytes(16).toString('hex');

function middleware(req, res, next) {
  if (!enabled()) {
    req.user = { name: 'open access', role: 'admin', open: true, csrf: OPEN_ACCESS_CSRF };
    return next();
  }
  if (req.path === '/login' || req.path === '/logout') return next();
  const session = readSession(req);
  const account = session ? resolveSession(session) : null;
  if (!account) {
    // A correctly signed session whose account is gone (deleted user, changed
    // admin password, password reset): worth recording, and rare by nature.
    if (session) audit.record('session.rejected', req, session.u);
    clearSession(res);
    if (req.method === 'GET') return res.redirect(`/login?next=${encodeURIComponent(req.originalUrl)}`);
    return res.status(401).send('Login required');
  }
  req.user = { name: account.name, role: account.role, open: false, csrf: session.csrf };
  // Sliding renewal: re-issue when less than half the lifetime is left, with
  // the same token and absolute limit.
  if (session.exp - Date.now() < SESSION_HOURS * 1800 * 1000 && session.exp < session.max) {
    issueSession(req, res, account.name, account.role, session);
  }
  next();
}

function requireAdmin(render) {
  return (req, res, next) => {
    if (req.user && req.user.role === 'admin') return next();
    res.status(403).send(render.header('Forbidden', req.user) +
      '<h1>Not allowed</h1><p>This page is for the administrator. <a href="/">Back to the predictions</a>.</p>' + render.footer());
  };
}

function csrfOk(req) {
  const body = req.body || {};
  return Boolean(req.user && req.user.csrf && typeof body._csrf === 'string' && safeEqual(body._csrf, req.user.csrf));
}

// --- pages ---------------------------------------------------------------------
function loginPage(error, next) {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Sign in - Sequence Predictor</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
    .box { background: white; padding: 30px 36px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 320px; }
    h1 { margin: 0 0 20px 0; font-size: 1.3em; color: #2c3e50; }
    label { display: block; font-weight: 600; margin-top: 12px; color: #2c3e50; }
    input { width: 100%; box-sizing: border-box; padding: 10px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; }
    button { width: 100%; margin-top: 18px; padding: 10px; background: #27ae60; color: white; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
    .error { color: #c0392b; margin-top: 12px; }
  </style></head><body><div class="box">
  <h1>📊 Sequence Predictor</h1>
  <form method="post" action="/login" autocomplete="on">
    <input type="hidden" name="next" value="${escapeHtml(next || '/')}">
    <label>User name<input name="username" autocomplete="username" required autofocus></label>
    <label>Password<input type="password" name="password" autocomplete="current-password" required></label>
    <button type="submit">Sign in</button>
    ${error ? `<div class="error">${escapeHtml(error)}</div>` : ''}
  </form></div></body></html>`;
}

function usersPage(render, req, message, isError) {
  const users = loadUsers();
  const writable = enabled();
  const csrf = `<input type="hidden" name="_csrf" value="${escapeHtml(req.user.csrf)}">`;
  const rows = users.map((u) => `
      <tr>
        <td style="text-align: left; font-weight: bold;">${escapeHtml(u.name)}</td>
        <td>user</td>
        <td>${escapeHtml((u.createdAt || '').slice(0, 10))}</td>
        <td>${u.lastLoginAt ? `${escapeHtml(shortTime(u.lastLoginAt))}<br><span style="color:#7f8c8d; font-size:0.85em;">${escapeHtml(u.lastLoginIp || '')} · ${u.logins || 1} sign-in(s)</span>` : '<span style="color:#888;">never</span>'}</td>
        <td>${writable ? `
          <form method="post" action="/admin/users/password" class="inline-form">${csrf}<input type="hidden" name="username" value="${escapeHtml(u.name)}">
            <input type="password" name="password" placeholder="new password" minlength="${MIN_PASSWORD_LENGTH}" required><button type="submit" class="nav-btn">Set password</button></form>
          <form method="post" action="/admin/users/delete" class="inline-form" onsubmit="return confirm('Delete this user?');">${csrf}<input type="hidden" name="username" value="${escapeHtml(u.name)}">
            <button type="submit" class="nav-btn" style="background: #c0392b; border-color: #a93226;">Delete</button></form>` : '<span style="color: #888;">read-only while authentication is off</span>'}
        </td>
      </tr>`).join('');
  const notice = !writable
    ? '<p style="background: #fdf2e9; border: 1px solid #f5cba7; padding: 10px; border-radius: 6px;">Authentication is <b>off</b>: set <code>WEB_USER</code> and <code>WEB_PASSWORD</code> in the server\'s environment to require a login. Accounts can only be managed while it is on.</p>'
    : `<p style="color: #7f8c8d;">Administrator: <b>${escapeHtml(config.ADMIN_USER)}</b> (from the environment). Users below sign in with their own password and see the predictions and the History pages only. Changing a user's password ends that user's open sessions.</p>`;
  return render.header('Users', req.user) + `
    <h1>Users</h1>
    <p><a href="/admin/activity" class="nav-btn" style="text-decoration:none;">Recent activity →</a></p>
    ${notice}
    ${message ? `<p style="color: ${isError ? '#c0392b' : '#27ae60'}; font-weight: bold;">${escapeHtml(message)}</p>` : ''}
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)"><span class="card-title">Accounts</span><div class="card-icon">▼</div></div>
      <div class="card-body">
        <div class="table-wrapper"><table style="min-width: 0;">
          <tr><th style="text-align: left;">User</th><th>Role</th><th>Created</th><th>Last seen</th><th>Actions</th></tr>
          ${rows || '<tr><td colspan="5" style="color: #888;">No users yet.</td></tr>'}
        </table></div>
      </div>
    </div>
    ${writable ? `
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)"><span class="card-title">Add a user</span><div class="card-icon">▼</div></div>
      <div class="card-body">
        <form method="post" action="/admin/users/add" class="auth-form">${csrf}
          <label>User name <span style="color: #7f8c8d; font-weight: normal;">(3-32 letters, digits, . _ -)</span><input name="username" pattern="[A-Za-z0-9._-]{3,32}" required></label>
          <label>Password <span style="color: #7f8c8d; font-weight: normal;">(at least ${MIN_PASSWORD_LENGTH} characters)</span><input type="password" name="password" minlength="${MIN_PASSWORD_LENGTH}" required></label>
          <button type="submit" class="nav-btn" style="background: #27ae60; border-color: #1e8449;">Add user</button>
        </form>
      </div>
    </div>` : ''}` + render.footer();
}


function shortTime(iso) {
  if (!iso) return '-';
  return String(iso).replace('T', ' ').slice(0, 16) + ' UTC';
}

function activityTable(entries, showUser) {
  if (!entries.length) return '<p style="color:#888;">Nothing recorded yet.</p>';
  const rows = entries.map((entry) => `
      <tr>
        <td style="text-align:left; white-space:nowrap;">${escapeHtml(shortTime(entry.at))}</td>
        ${showUser ? `<td style="text-align:left; font-weight:bold;">${escapeHtml(entry.user)}</td>` : ''}
        <td style="text-align:left;">${escapeHtml(audit.label(entry.event))}${entry.detail ? ` <span style="color:#7f8c8d;">(${escapeHtml(entry.detail)})</span>` : ''}</td>
        <td style="text-align:left;">${escapeHtml(entry.ip || '-')}</td>
        <td style="text-align:left; color:#7f8c8d; font-size:0.85em;">${escapeHtml((entry.agent || '').slice(0, 60))}</td>
      </tr>`).join('');
  return `<div class="table-wrapper"><table style="min-width:0;">
      <tr><th style="text-align:left;">When</th>${showUser ? '<th style="text-align:left;">User</th>' : ''}<th style="text-align:left;">What</th><th style="text-align:left;">From</th><th style="text-align:left;">Browser</th></tr>
      ${rows}
    </table></div>`;
}

// Every signed-in user gets this page: their own password, and the sign-ins
// recorded for their name so they can see a session they did not start.
function accountPage(render, req, message, isError) {
  const user = req.user;
  const csrf = `<input type="hidden" name="_csrf" value="${escapeHtml(user.csrf)}">`;
  const isAdmin = user.role === 'admin';
  const form = user.open
    ? '<p style="background:#fdf2e9; border:1px solid #f5cba7; padding:10px; border-radius:6px;">Authentication is <b>off</b> on this server, so there is no password to change.</p>'
    : isAdmin
      ? `<p>The administrator signs in with <code>WEB_USER</code> and <code>WEB_PASSWORD</code> from the server's <code>.env</code> file, which is deliberately not editable from a web page: it is the credential that could change everyone else's. To change it, edit <code>.env</code> and restart the server (<code>pm2 restart sequencePredictor</code>). Every administrator session ends the moment the password changes.</p>`
      : `<form method="post" action="/account/password" class="auth-form">${csrf}
          <label>Current password<input type="password" name="current" autocomplete="current-password" required></label>
          <label>New password <span style="color:#7f8c8d; font-weight:normal;">(at least ${MIN_PASSWORD_LENGTH} characters)</span><input type="password" name="password" minlength="${MIN_PASSWORD_LENGTH}" autocomplete="new-password" required></label>
          <label>New password again<input type="password" name="confirm" minlength="${MIN_PASSWORD_LENGTH}" autocomplete="new-password" required></label>
          <button type="submit" class="nav-btn" style="background:#27ae60; border-color:#1e8449;">Change my password</button>
        </form>
        <p style="color:#7f8c8d;">Changing it signs out your other browsers and keeps this one signed in.</p>`;

  const history = user.open ? [] : audit.recent(15, user.name);
  return render.header('Your account', user) + `
    <h1>Your account</h1>
    <p style="color:#7f8c8d;">Signed in as <b>${escapeHtml(user.name)}</b> - ${isAdmin ? 'administrator' : 'user'}.</p>
    ${message ? `<p style="color:${isError ? '#c0392b' : '#27ae60'}; font-weight:bold;">${escapeHtml(message)}</p>` : ''}
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)"><span class="card-title">Password</span><div class="card-icon">▼</div></div>
      <div class="card-body">${form}</div>
    </div>
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)">
        <div><span class="card-title">Your recent activity</span><span class="card-meta" style="margin-left:10px;">sign-ins and changes recorded for your name</span></div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">${activityTable(history, false)}</div>
    </div>` + render.footer();
}

function activityPage(render, req) {
  return render.header('Activity', req.user) + `
    <h1>Activity</h1>
    <p style="color:#7f8c8d;">Sign-ins, failed attempts and account changes, newest first. Recorded in
      <code>${escapeHtml(audit.FILE)}</code>, which is rotated at 2 MB with one older file kept, so it cannot grow without bound.
      Addresses are the caller's as seen through the reverse proxy.</p>
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)"><span class="card-title">Last 200 events</span><div class="card-icon">▼</div></div>
      <div class="card-body">${activityTable(audit.recent(200), true)}</div>
    </div>` + render.footer();
}

// --- routes ----------------------------------------------------------------------
function install(app, render) {
  app.get('/login', (req, res) => {
    if (!enabled()) return res.redirect('/');
    const session = readSession(req);
    if (session && resolveSession(session)) return res.redirect(safeNext(req.query.next));
    res.send(loginPage(null, safeNext(req.query.next)));
  });

  app.post('/login', (req, res) => {
    if (!enabled()) return res.redirect('/');
    const body = req.body || {};
    const username = String(body.username || '').trim();
    const password = String(body.password || '');
    const next = safeNext(body.next);
    if (!username || !password || username.length > 64 || password.length > 1024) {
      return res.status(400).send(loginPage('User name and password are required.', next));
    }
    if (isLocked(req, username)) {
      audit.record('login.locked', req, username);
      return res.status(429).send(loginPage('Too many failed attempts - try again in 15 minutes.', next));
    }

    let role = null;
    if (safeEqual(username, config.ADMIN_USER)) {
      if (verifyPassword(password, adminCredential.salt, adminCredential.hash)) role = 'admin';
    } else {
      const user = findUser(loadUsers(), username);
      if (user) {
        if (verifyPassword(password, user.salt, user.hash)) role = 'user';
      } else {
        verifyPassword(password, dummyCredential.salt, dummyCredential.hash); // same cost as a known name
      }
    }
    if (!role) {
      recordFailure(req, username);
      audit.record('login.failed', req, username);
      return res.status(401).send(loginPage('Unknown user name or wrong password.', next));
    }
    clearFailures(req, username);
    if (role === 'user') {
      // Last seen lives on the account itself, so the Users page can show it
      // without reading the audit trail for every row.
      const users = loadUsers();
      const user = findUser(users, username);
      if (user) {
        user.lastLoginAt = new Date().toISOString();
        user.lastLoginIp = String(req.ip || '').replace(/^::ffff:/, '');
        user.logins = (user.logins || 0) + 1;
        saveUsers(users);
      }
    }
    audit.record('login.ok', req, username, role === 'admin' ? 'administrator' : null);
    issueSession(req, res, username, role);
    res.redirect(next);
  });

  const logout = (req, res) => {
    // /logout deliberately bypasses the middleware (an expired session must
    // still get a clean redirect instead of a bare 401), so req.user does not
    // exist here - the session is read directly to name who signed out.
    const session = readSession(req);
    if (session && resolveSession(session)) audit.record('logout', req, session.u);
    clearSession(res);
    res.redirect('/login');
  };
  app.post('/logout', logout);
  app.get('/logout', logout);

  app.get('/account', (req, res) => {
    res.send(accountPage(render, req, req.query.msg ? String(req.query.msg).slice(0, 200) : null, req.query.err === '1'));
  });

  app.post('/account/password', (req, res) => {
    const backToAccount = (message, isError) => res.redirect(`/account?msg=${encodeURIComponent(message)}${isError ? '&err=1' : ''}`);
    if (!csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    if (!enabled() || req.user.open) return backToAccount('Authentication is off on this server.', true);
    if (req.user.role === 'admin') return backToAccount('The administrator password is set in .env, not here.', true);

    const body = req.body || {};
    const current = String(body.current || '');
    const password = String(body.password || '');
    const confirm = String(body.confirm || '');
    const users = loadUsers();
    const user = findUser(users, req.user.name);
    if (!user) return backToAccount('Your account no longer exists.', true);
    if (!verifyPassword(current, user.salt, user.hash)) {
      audit.record('password.rejected', req, req.user.name);
      return backToAccount('That is not your current password.', true);
    }
    if (password.length < MIN_PASSWORD_LENGTH) return backToAccount(`The new password must be at least ${MIN_PASSWORD_LENGTH} characters.`, true);
    if (password !== confirm) return backToAccount('The two new passwords do not match.', true);
    if (password === current) return backToAccount('The new password is the same as the current one.', true);

    Object.assign(user, hashPassword(password), { passwordChangedAt: new Date().toISOString() });
    saveUsers(users);
    audit.record('password.self', req, req.user.name);
    // The change revokes every session older than it - including this one, so
    // the browser that made the change is given a fresh session rather than
    // being thrown back to the login page.
    issueSession(req, res, user.name, 'user');
    backToAccount('Your password has been changed. Your other browsers have been signed out.', false);
  });

  const adminOnly = requireAdmin(render);
  // In open mode the page is read-only: without an administrator identity
  // nobody may create accounts that would become valid the moment
  // authentication is switched on.
  const writable = (req, res, next) => (enabled() ? next() : res.status(403).send('User management needs WEB_USER and WEB_PASSWORD to be set.'));

  app.get('/admin/activity', adminOnly, (req, res) => {
    res.send(activityPage(render, req));
  });

  app.get('/admin/users', adminOnly, (req, res) => {
    res.send(usersPage(render, req, req.query.msg ? String(req.query.msg).slice(0, 200) : null, req.query.err === '1'));
  });

  const back = (res, message, isError) => res.redirect(`/admin/users?msg=${encodeURIComponent(message)}${isError ? '&err=1' : ''}`);
  const rejectCsrf = (res) => res.status(403).send('Invalid form token - reload the page and try again.');

  app.post('/admin/users/add', adminOnly, writable, (req, res) => {
    if (!csrfOk(req)) return rejectCsrf(res);
    const body = req.body || {};
    const username = String(body.username || '').trim();
    const password = String(body.password || '');
    if (!USERNAME_RE.test(username)) return back(res, 'User name must be 3-32 letters, digits, dots, underscores or dashes.', true);
    if (password.length < MIN_PASSWORD_LENGTH) return back(res, `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`, true);
    if (username.toLowerCase() === String(config.ADMIN_USER).toLowerCase()) return back(res, 'That name is the administrator account.', true);
    const users = loadUsers();
    if (users.some((u) => u.name.toLowerCase() === username.toLowerCase())) return back(res, `User ${username} already exists.`, true);
    users.push({ name: username, role: 'user', createdAt: new Date().toISOString(), ...hashPassword(password) });
    saveUsers(users);
    audit.record('user.added', req, username, `by ${req.user.name}`);
    back(res, `User ${username} added.`, false);
  });

  app.post('/admin/users/password', adminOnly, writable, (req, res) => {
    if (!csrfOk(req)) return rejectCsrf(res);
    const body = req.body || {};
    const username = String(body.username || '');
    const password = String(body.password || '');
    if (password.length < MIN_PASSWORD_LENGTH) return back(res, `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`, true);
    const users = loadUsers();
    const user = findUser(users, username);
    if (!user) return back(res, `User ${username} not found.`, true);
    Object.assign(user, hashPassword(password), { passwordChangedAt: new Date().toISOString() });
    saveUsers(users);
    audit.record('password.admin', req, username, `by ${req.user.name}`);
    back(res, `Password of ${username} changed - the user's open sessions are ended.`, false);
  });

  app.post('/admin/users/delete', adminOnly, writable, (req, res) => {
    if (!csrfOk(req)) return rejectCsrf(res);
    const username = String((req.body || {}).username || '');
    const users = loadUsers();
    if (!findUser(users, username)) return back(res, `User ${username} not found.`, true);
    saveUsers(users.filter((u) => u.name !== username));
    audit.record('user.deleted', req, username, `by ${req.user.name}`);
    back(res, `User ${username} deleted.`, false);
  });
}

// requireAdmin / csrfOk / ensureConfigDir are exported for the other page
// modules (council.js and the account, audit and job pages to come) so every
// one of them gates and validates exactly like the user pages here.
module.exports = {
  enabled, misconfigured, middleware, install, escapeHtml, safeNext,
  requireAdmin, csrfOk, ensureConfigDir, USERS_FILE, SECRET_FILE, CONFIG_DIR: config.CONFIG_DIR,
  MIN_PASSWORD_LENGTH,
};
