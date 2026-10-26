// Login and basic user management for the web UI (README roadmap item 3).
//
// One admin comes from the environment (WEB_USER / WEB_PASSWORD, read in
// config.js) and never touches the disk; further users live in
// config/users.json (gitignored, salted scrypt hashes) and are managed by the
// admin in the app (system.js, /api/admin/people). Two roles: "admin" sees
// everything, "user" the user view. Sessions are a signed HttpOnly SameSite cookie
// (HMAC-SHA256 with a secret from WEB_SESSION_SECRET or generated once into
// config/session.secret), so no session store and no npm dependency is needed.
// With WEB_USER / WEB_PASSWORD both unset, authentication is off (local
// development): everything is open and accounts are read-only
// (managementRefusal). Exactly one of the two set is a configuration error (see
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
const shell = require('./shell');   // the server-side pages' look: sign-in, not found, server error

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
// (existing given) keeps all three so the form token the app already holds keeps
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
const failures = new Map(); // "address|user" (a sign-in) or "password|user" (a password change) -> { count, first }
function failureKey(req, name) { return `${req.ip}|${String(name).toLowerCase()}`; }
// A password change is guessed at from a session that is already open: its attempts count per name alone, so a
// stolen cookie gains nothing by changing address.
const OWN_PASSWORD = { ip: 'password' };
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
// cross-origin page still cannot read it) while letting the app's actions work. It
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
    // the JSON API (api.js, the React app) gets a 401 it can act on, never the login page as a 200
    if (req.path.startsWith('/api/')) return res.status(401).set('Cache-Control', 'no-store').json({ error: 'login required', login: `/login?next=${encodeURIComponent('/app')}` });
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

function csrfOk(req) {
  const body = req.body || {};
  return Boolean(req.user && req.user.csrf && typeof body._csrf === 'string' && safeEqual(body._csrf, req.user.csrf));
}

// --- account changes, for the app's JSON (system.js) ---------------------------
// Each answers {ok, status, message}: status as the JSON routes give it - 400 a
// value that breaks a rule, 403 user management while authentication is off,
// 404 no such user, 409 refused by what it found (the name is taken, it is the
// administrator). The audit entries are written here.
const refuse = (status, message) => ({ ok: false, status, message });
const isAdminName = (name) => String(name).toLowerCase() === String(config.ADMIN_USER).toLowerCase();

// In open mode accounts are read-only: without an administrator identity nobody
// may create accounts that would become valid the moment authentication is
// switched on.
function managementRefusal() {
  return enabled() ? null : refuse(403, 'User management needs WEB_USER and WEB_PASSWORD to be set.');
}

function addUser(req, rawName, rawPassword) {
  const off = managementRefusal();
  if (off) return off;
  const username = String(rawName || '').trim();
  const password = String(rawPassword || '');
  if (!USERNAME_RE.test(username)) return refuse(400, 'User name must be 3-32 letters, digits, dots, underscores or dashes.');
  if (password.length < MIN_PASSWORD_LENGTH) return refuse(400, `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
  if (isAdminName(username)) return refuse(409, 'That name is the administrator account.');
  const users = loadUsers();
  if (users.some((u) => u.name.toLowerCase() === username.toLowerCase())) return refuse(409, `User ${username} already exists.`);
  users.push({ name: username, role: 'user', createdAt: new Date().toISOString(), ...hashPassword(password) });
  saveUsers(users);
  audit.record('user.added', req, username, `by ${req.user.name}`);
  return { ok: true, status: 200, message: `User ${username} added.` };
}

// The administrator's own credential is the environment's (WEB_USER /
// WEB_PASSWORD): it has no record here, so it can be neither given a password
// nor deleted - the one administrator always remains.
function setUserPassword(req, rawName, rawPassword) {
  const off = managementRefusal();
  if (off) return off;
  const username = String(rawName || '');
  const password = String(rawPassword || '');
  if (password.length < MIN_PASSWORD_LENGTH) return refuse(400, `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
  if (isAdminName(username)) return refuse(409, 'The administrator password is set in .env, not here.');
  const users = loadUsers();
  const user = findUser(users, username);
  if (!user) return refuse(404, `User ${username} not found.`);
  Object.assign(user, hashPassword(password), { passwordChangedAt: new Date().toISOString() });
  saveUsers(users);
  audit.record('password.admin', req, username, `by ${req.user.name}`);
  return { ok: true, status: 200, message: `Password of ${username} changed - the user's open sessions are ended.` };
}

function deleteUser(req, rawName) {
  const off = managementRefusal();
  if (off) return off;
  const username = String(rawName || '');
  if (isAdminName(username)) return refuse(409, 'The administrator account comes from the environment and cannot be deleted.');
  const users = loadUsers();
  if (!findUser(users, username)) return refuse(404, `User ${username} not found.`);
  saveUsers(users.filter((u) => u.name !== username));
  audit.record('user.deleted', req, username, `by ${req.user.name}`);
  return { ok: true, status: 200, message: `User ${username} deleted.` };
}

// Why the signed-in reader cannot change their own password here, or null.
function ownPasswordRefusal(user) {
  if (!enabled() || !user || user.open) return refuse(409, 'Authentication is off on this server.');
  if (user.role === 'admin') return refuse(409, 'The administrator password is set in .env, not here.');
  return null;
}

// The reader's own password, the current one required. The change revokes every
// session older than it - including this one, so the browser that made it is
// given a fresh session (and with it a fresh form token: session.csrf) rather
// than being thrown back to the login page.
function changeOwnPassword(req, res, body) {
  const refused = ownPasswordRefusal(req.user);
  if (refused) return refused;
  const current = String((body || {}).current || '');
  const password = String((body || {}).password || '');
  const confirm = String((body || {}).confirm || '');
  const users = loadUsers();
  const user = findUser(users, req.user.name);
  if (!user) return refuse(404, 'Your account no longer exists.');
  // the sign-in's limit on guessing the current password too: LOGIN_MAX_FAILURES wrong ones within LOGIN_WINDOW_MS
  if (isLocked(OWN_PASSWORD, user.name)) return refuse(429, 'Too many wrong passwords - try again in 15 minutes.');
  if (!verifyPassword(current, user.salt, user.hash)) {
    recordFailure(OWN_PASSWORD, user.name);
    audit.record('password.rejected', req, req.user.name);
    return refuse(400, 'That is not your current password.');
  }
  if (password.length < MIN_PASSWORD_LENGTH) return refuse(400, `The new password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
  if (password !== confirm) return refuse(400, 'The two new passwords do not match.');
  if (password === current) return refuse(400, 'The new password is the same as the current one.');

  Object.assign(user, hashPassword(password), { passwordChangedAt: new Date().toISOString() });
  saveUsers(users);
  clearFailures(OWN_PASSWORD, user.name);
  audit.record('password.self', req, req.user.name);
  const session = issueSession(req, res, user.name, 'user');
  return { ok: true, status: 200, message: 'Your password has been changed. Your other browsers have been signed out.', session };
}

// An account as the app may show it: never the salt or the hash.
function publicUser(u) {
  return { name: u.name, role: u.role === 'admin' ? 'admin' : 'user', createdAt: u.createdAt || null, passwordChangedAt: u.passwordChangedAt || null,
    lastLoginAt: u.lastLoginAt || null, lastLoginIp: u.lastLoginIp || null, logins: Number.isFinite(u.logins) ? u.logins : null };
}
function listUsers() {
  return loadUsers().map(publicUser);
}

// --- the sign-in page --------------------------------------------------------------
// In the app's look (shell.js). It stays server-side: the app itself needs a session to load.
function loginPage(error, next) {
  return shell.page('Sign in', `
  <h1>Predictor</h1>
  <p class="sub">Sign in to see the predictions and their record.</p>
  <form method="post" action="/login" autocomplete="on">
    <input type="hidden" name="next" value="${escapeHtml(next || '/')}">
    <label>User name<input name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required autofocus></label>
    <label>Password<input type="password" name="password" autocomplete="current-password" required></label>
    ${error ? `<p class="err" role="alert">${escapeHtml(error)}</p>` : ''}
    <button type="submit">Sign in</button>
  </form>
  <p class="foot">Not advice: the models are not expected to beat the draw or the market, and this site measures whether they do.</p>`);
}

// --- routes ----------------------------------------------------------------------
function install(app) {
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
      // Last seen lives on the account itself, so the People page (system.js)
      // can show it without reading the audit trail for every row.
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
}

// csrfOk is exported for the JSON routes (council.js, system.js), so every POST
// checks the session's form token the same way; the account changes for
// system.js, the app's JSON.
module.exports = {
  enabled, misconfigured, middleware, install, loginPage,
  csrfOk, ensureConfigDir, USERS_FILE, SECRET_FILE, CONFIG_DIR: config.CONFIG_DIR,
  MIN_PASSWORD_LENGTH, USERNAME_RE, LOGIN_MAX_FAILURES, LOGIN_WINDOW_MS, adminName: () => (enabled() ? config.ADMIN_USER : null),
  managementRefusal, addUser, setUserPassword, deleteUser, ownPasswordRefusal, changeOwnPassword, listUsers,
};
