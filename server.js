const express = require('express');
const path = require('path');

const config = require("./config");
const auth = require("./auth");
const council = require("./council");
const services = require("./services");
const jobs = require("./jobs");
const switchover = require('./switchover');   // the switch to the app: the old pages' redirects, the last 404 and error answers (24 Oct 2026)
const api = require('./api');   // the JSON API and the React app under /app (15 Oct 2026)
const lottery = require('./lottery');   // the users' Lottery and Home JSON under /api/lottery, /api/home (23 Oct 2026)
const admin = require('./admin');   // the admin section's JSON under /api/admin (21 Oct 2026)
const system = require('./system');   // the System pages' JSON: Jobs & services, People, Account, What's new (21 Oct 2026)
const audit = require('./audit');   // the sign-in events the admin This week page counts

const app = express();

// The sign-in (auth.js, README roadmap item 3): its form post needs the body
// parser; the middleware redirects anonymous requests to /login when
// WEB_USER / WEB_PASSWORD are set and is a no-op otherwise. Half-configured
// credentials would silently mean "open", so they stop the server instead.
// trust proxy makes req.ip - the login lockout key - the client's address
// when a reverse proxy on this machine forwards to us.
if (auth.misconfigured()) {
  console.error('Set both WEB_USER and WEB_PASSWORD to require a login, or neither for open access - refusing to start with only one of them.');
  process.exit(1);
}
app.set('trust proxy', config.TRUST_PROXY);
app.use(express.urlencoded({ extended: false, limit: '16kb' }));
app.use(auth.middleware);
// The switch to the app (24 Oct 2026): every old page address answers with its app page, for each reader as the old
// page was (switchover.js). The old pages themselves were removed in cleanup batch 6b (26 Oct 2026).
app.use(switchover.redirects);

// The LLM Council's JSON under /council/api (council.js); its page is the app's.
council.install(app);

// Paths
const dataPath = path.join(__dirname, 'data', 'database');
// The weekly control experiments (NullControls.py, RandomnessDiscrimination.py)
// write here; controls.js reads them for the admin's controls (admin.js).
// Optional: the admin JSON answers without them.
const controlsPath = path.join(__dirname, 'data', 'controls');
// The Crypto and Shares records (README roadmap item 4): the settlement's
// exports under data/markets/, read by markets.js.
const marketsPath = path.join(__dirname, 'data', 'markets');

// Login and logout (auth.js). The Optuna dashboard is not started or linked
// from here - start it by hand when needed:
// optuna-dashboard sqlite:///db.sqlite3
auth.install(app);
api.install(app, { dataDir: marketsPath });
// The users' Lottery and Home (lottery.js): every signed-in reader gets the user view - the top five rows of the
// coming draw by today's ranking, each past draw's five as ranked before it, nothing before the first saved ranking.
// The point-in-time ranking reads this checkout's git history, its cache in the gitignored config/ (as admin.js).
lottery.install(app, { root: __dirname, databaseDir: dataPath, marketsDir: marketsPath });
// The System pages' JSON (system.js): Jobs & services and People for the
// administrator, Account and What's new for every signed-in reader, with the
// JSON POST actions (form token from /api/me). It calls the rules of jobs.js,
// services.js, auth.js and whatsnew.js. Installed before admin.js, whose
// catch-all answers the rest of /api/admin with a 404.
system.install(app, { root: __dirname, jobs, services });
// The admin section's JSON (/api/admin/*, administrator only): the game,
// control and market records from the folders above; This week also asks
// the schedule (its jobs, the lock, the queue), the Council API's supervisor
// and the sign-in log what they know now.
admin.install(app, { databaseDir: dataPath, controlsDir: controlsPath, marketsDir: marketsPath, root: __dirname, jobs, services, audit });
// The React app itself: /app, /app/* and its built files (api.js).
api.installApp(app);
// Last: an unknown address answers 404 and a thrown error 500, as a page or as JSON, never with the stack (switchover.js).
switchover.finish(app);

// The UI is reachable from outside (a Tailscale funnel proxies to this
// port), pm2 restarts the process on exit, and this process now owns the
// daily pipeline schedule - so an unexpected throw must be logged with its
// stack instead of dying silently or being swallowed.
process.on('uncaughtException', (error) => {
  console.error(`[${new Date().toISOString()}] uncaught exception:`, error);
});
process.on('unhandledRejection', (reason) => {
  console.error(`[${new Date().toISOString()}] unhandled rejection:`, reason);
});

const server = app.listen(config.PORT, config.INTERFACE, () => {
  console.log(`Server running at http://${config.INTERFACE}:${config.PORT}` +
              (auth.enabled() ? ` - login required (admin: ${config.ADMIN_USER})` : ' - open access: set WEB_USER and WEB_PASSWORD to require a login'));
  services.startAll();
  // The schedule starts last and, by default, in dry mode: it records what
  // it would have run so it can be watched for a weekend next to the still
  // existing crontab entries (see jobs.js and README roadmap item 8).
  jobs.start();
});

// Services are ordinary children, so they must go down with the server
// rather than being left behind on a restart or a deploy.
['SIGTERM', 'SIGINT'].forEach((signal) => process.on(signal, () => {
  console.log(`${signal} received - stopping supervised services`);
  // Only the schedule's timer stops. A pipeline job launched with
  // setsid --fork keeps running on purpose and is adopted again at the next
  // start - that is what makes a deploy safe during a 12-hour tuning run.
  jobs.stop();
  services.stopAll();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 6000).unref();
}));
process.on('exit', () => services.stopAll());
server.on('error', (error) => {
  console.error(`Could not listen on ${config.INTERFACE}:${config.PORT}: ${error.message}`);
  process.exit(1);
});
