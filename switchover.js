// The switch to the app (batch 6, 24 Oct 2026): the server-rendered pages retire at once, for every reader.
//
// redirects, mounted right after the sign-in middleware (so it knows the reader): every old page address answers with
// its app page - the administrator's view or the user's, as the old page was - and the old handlers behind it are no
// longer reached (removed in a later cleanup). 302 and no-store: the target depends on who asks, and a browser must not
// keep it. Targets are built from known game names and checked dates only, never from what the address carried.
//
// finish(app), installed after every route: an unknown address answers 404 (JSON under the APIs, a page in the
// sign-in look otherwise), and a thrown error answers 500 the same way. The stack goes to the error log, never to the
// reader: Express's own handler prints it in the page whenever NODE_ENV is not "production", as here.

const shell = require('./shell');
const draws = require('./draws');
const { GAMES, LOTTERY_GAMES } = require('./admin');

// the administrator's view (every model, the workbench); open access (no sign-in configured) reads as the administrator
const seesAll = (user) => !user || user.open === true || user.role === 'admin';
const MONTH = /^\d{4}-\d{2}$/;
const SESSION = /^[a-f0-9]{12}$/;   // a Council session (council.js)
const PLAIN = /^[A-Za-z0-9_.-]{1,128}$/;
// a user's page for a market's game view: the market's own page (the week games are on it too)
const marketPage = (game) => `/app/${String(game).replace(/week$/, '')}`;

// The app page for an old page address, or null when the address is not an old page.
function targetOf(path, query, user) {
  const p = path.length > 1 ? path.replace(/\/+$/, '') : path;
  const all = seesAll(user);
  const q = query || {};
  if (p === '/') return '/app/';
  if (p === '/lottery') return '/app/lottery';
  if (p === '/database') return all ? '/app/admin/games' : '/app/lottery';
  const db = /^\/database\/([^/]+)(?:\/([^/]+))?$/.exec(p);
  if (db) {
    const folder = PLAIN.test(db[1]) ? db[1] : '';
    const game = folder ? draws.gameFromFolder(folder) : null;
    const known = GAMES.includes(game);
    const iso = db[2] && PLAIN.test(db[2]) ? draws.isoOfFile(db[2]) : null;
    const month = typeof q.month === 'string' && MONTH.test(q.month) ? q.month : null;
    if (all) {
      if (!known) return '/app/admin/games';
      if (iso) return `/app/admin/games/${game}/${iso}`;
      return `/app/admin/games/${game}${month ? `?month=${month}` : ''}`;
    }
    if (!known) return '/app/lottery';
    if (!LOTTERY_GAMES.includes(game)) return marketPage(game);
    if (iso) return `/app/lottery/${game}/${iso}`;
    return `/app/lottery/${game}${month ? `?month=${month}` : ''}`;
  }
  const mk = /^\/markets\/(crypto|shares)$/.exec(p);
  if (mk) return all ? `/app/admin/markets/${mk[1]}` : `/app/${mk[1]}`;
  if (p === '/council') return typeof q.session === 'string' && SESSION.test(q.session) ? `/app/council?session=${q.session}` : '/app/council';
  if (p === '/account') return '/app/account';
  if (p === '/whats-new') return '/app/whats-new';
  if (p === '/admin/users' || p === '/admin/activity') return all ? '/app/admin/people' : '/app/';
  if (p === '/admin/jobs') return all ? '/app/admin/jobs' : '/app/';
  return null;
}

function redirects(req, res, next) {
  if (req.method !== 'GET' && req.method !== 'HEAD') return next();
  const to = targetOf(req.path, req.query, req.user);
  if (!to) return next();
  res.set('Cache-Control', 'no-store');
  return res.redirect(302, to);
}

// the JSON APIs: their errors stay JSON, as their own answers are
const isApi = (req) => req.path.startsWith('/api/') || req.path === '/api' || req.path.startsWith('/council/api/');

// one short notice with the way home (heading and line are trusted text)
function notice(title, heading, line) {
  return shell.page(title, `
  <h1>${heading}</h1>
  <p class="sub">${line}</p>
  <p><a href="/app/">Go to the home page</a></p>`);
}
const notFoundPage = () => notice('Not found', 'Page not found', 'This address is not part of the site, or no longer is.');
const badRequestPage = () => notice('Bad request', 'Bad request', 'The server could not read this request.');
const errorPage = () => notice('Server error', 'Something went wrong', 'The server could not answer this request. It has been logged; try again in a moment.');

function finish(app) {
  app.use((req, res) => {
    if (isApi(req)) return res.status(404).json({ error: 'not found' });
    res.status(404).send(notFoundPage());
  });
  // Express tells an error handler by its four parameters
  app.use((error, req, res, next) => {
    // a client's own mistake the parser names (a malformed or too large body): its status and message, no stack
    const status = Number(error && (error.status || error.statusCode));
    const client = status >= 400 && status < 500;
    if (!client) console.error(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}:`, error);
    if (res.headersSent) return next(error);   // half an answer is out: Express's own handler closes the connection
    const code = client ? status : 500;
    if (isApi(req)) return res.status(code).json({ error: client && error.expose ? String(error.message) : client ? 'bad request' : 'server error' });
    res.status(code).send(client ? badRequestPage() : errorPage());
  });
}

module.exports = { redirects, targetOf, finish, notFoundPage, errorPage, isApi };
