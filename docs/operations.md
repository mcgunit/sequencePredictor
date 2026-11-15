# Operations: install, run, server, pages and schedule

This page covers how to install and run the project: the Python pipeline and how to start it by hand, the Node server and its settings, the web app (structure, pages, menu and the in-app wiki), login, users and the audit log, the supervised services, the schedule that starts the daily and weekly jobs, the worker machines, testing, and where the lottery data comes from.

## Installation

### Python (the pipeline)

Create and activate a virtual environment, then install the pinned packages:

```
python3 -m venv ~/sequencePredictor
source ~/sequencePredictor/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` is the reference. It pins, among others, `tensorflow[and-cuda]==2.16.2`, `keras==3.7.0`, `keras-tcn==3.1.2`, `numpy`, `pandas`, `scikit_learn`, `statsmodels`, `pmdarima`, `optuna`, `optuna_dashboard`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, `art` and `asciichartpy`.

### Docker

Two Dockerfiles are in the repository root:

- `Dockerfile` (CPU, `python:3.11`): it installs its own list of packages, whose pins differ from `requirements.txt` (TensorFlow 2.18 and optuna 3.4.1 there, against 2.16.2 and 4.3.0 in `requirements.txt`).
- `Dockerfile.gpu` (`nvidia/cuda`): it installs `requirements.txt`.

Build and run:

```
docker build -t sequence_predictor .
docker run --rm -it -u $(id -u) -v {absolute path to the repository}:/opt/sequencePredictor sequence_predictor /bin/bash
```

The second command opens a shell inside the container, with the repository mounted.

### Node (the server and the web app)

In the repository root:

```
npm i
npm run build:web
```

`npm run build:web` installs the dependencies of `web/` and writes `web/dist` (both git-ignored). Run it again after every pull that changes `web/`, then restart the server. For pm2:

```
npm i pm2 -g
```

## How to run prediction

The server's scheduler starts the pipeline (see *The schedule*). To run it by hand:

```
python3 Predictor.py -a true -g lotto
```

| Flag | Default | Meaning |
|---|---|---|
| `-a`, `--ai` | `false` | `true` enables the heavy deep-learning rows (LSTM, TCN, unified). Without it only the lightweight rows run, and a `SubsetEnsemble Model` selection that contains a missing row is skipped for that day. |
| `-g`, `--games` | all eleven games | Comma-separated list: `euromillions,lotto,eurodreams,jokerplus,keno,pick3,vikinglotto,crypto,shares,cryptoweek,sharesweek`. |
| `--dl-model-seconds` | 240 | Training budget per deep-learning model per day; see [models](models.md). |
| `--dl-timeout` | 0 (auto) | Deadline of the whole per-day deep-learning child; see [models](models.md). |
| `-b`, `--boost` | `true` | Enables the gradient boosting rows. |
| `-d`, `--days` | 31 | Number of days handled by the run. |
| `-r`, `--rebuild_history` | `false` | History rebuild; see [models](models.md). |
| `-s`, `--save` | `true` | `false` skips the commit and push at the end. |

Every Python entry point takes the same `process.lock` in the repository root, so only one pipeline job runs at a time on the server. A second one started by hand while the lock is held exits.

Two shell scripts are the hand-run path of the scheduled jobs. Both `cd` into a fixed checkout path, written in the script.

- `runPredictor.sh` runs `python3 Predictor.py -a true` for all games in one go, into `log/predictor.log`.
- `runHyperopt.sh` waits up to six hours for `process.lock`, then runs the weekly tuning chain and the four control experiments in the scheduler's order, with the same arguments and log files. It runs the null band every time; the schedule runs it monthly.

`test/jobs.test.js` fails when the commands, logs or order of `runHyperopt.sh` and the schedule drift apart.

A single model file can be run directly for a quick test, for example `python3 src/LSTM.py`; its `__main__` section points to the data and sets the parameters.

## Run server

The server is a Node.js (Express) server. It serves JSON APIs and the React app from `web/` under `/app`. The only pages it renders itself are the sign-in page, "page not found", "bad request" and "something went wrong" (`shell.js`).

It listens on the interface and port set in `config.js` (`127.0.0.1:3001`).

```
npm start
```

`npm start` runs `nodemon server.js`; nodemon ignores `data/`, `hyperOptCache/`, `trainingData/` and `config/`.

### What `server.js` wires

`server.js` only wires modules, in this order:

1. the sign-in middleware (`auth.js`); the server refuses to start with only one of `WEB_USER` and `WEB_PASSWORD` set;
2. the redirects of the old page addresses (`switchover.js`);
3. the Council's JSON (`council.js`), the login routes, the market and user JSON (`api.js`), Home and Lottery (`lottery.js`), the System pages' JSON (`system.js`, which also installs the parameters, backtests, hyperopts, workers and wiki routes), the admin JSON (`admin.js`, `workbench.js`);
4. the app under `/app` (`api.installApp`);
5. the last handlers: 404 and 500 (`switchover.finish`).

Once it listens, it starts the supervised services (`services.js`), the schedule (`jobs.js`) and the worker queues (`workers.js`). On SIGTERM or SIGINT it stops the timers and the services; running jobs and worker runs keep running and are adopted at the next start. Loading `server.js` without listening starts nothing.

### Settings

Settings come from the environment or from a `.env` file in the repository root (git-ignored; copy `.env.example`). `config.js` loads the file at start; real environment variables take precedence. A change to `.env` needs a server restart.

| Variable | Default | Meaning |
|---|---|---|
| `WEB_USER`, `WEB_PASSWORD` | unset | The administrator. Both set: login required. Both unset: open access. Only one: the server does not start. |
| `WEB_SESSION_SECRET` (or `SESSION_SECRET`) | generated | Secret that signs the session cookie, at least 32 bytes; a shorter value is ignored with a warning. Unset: generated once into `config/session.secret`. |
| `WEB_TRUST_PROXY` | `loopback` | Which reverse proxy's `X-Forwarded-For` and `X-Forwarded-Proto` Express believes: an address or CIDR, `true` (any) or `false` (none). |
| `SCHEDULER` | `dry` | `on`, `dry` or `off`; see *The schedule*. |
| `COUNCIL_API_AUTOSTART` | `on` | `off` leaves the Council API to be started by hand. |
| `COUNCIL_API_PORT` | 8099 | Port of the Council API, for the service and for the server's proxy. |
| `COUNCIL_API_HOST` | `127.0.0.1` | Host the server's Council proxy calls. |
| `COUNCIL_ACCESS` | `user` | `admin` restricts the Council to the administrator. |

`.env.example` lists the login, session, proxy and the first two Council variables; `SCHEDULER`, `COUNCIL_API_HOST` and `COUNCIL_ACCESS` are not in it.

### Running under pm2

```
pm2 start server.js --name sequencePredictor --time
pm2 save
pm2 startup
```

`pm2 save` stores the run list and `pm2 startup` starts it at boot. Leave `--watch` off: the pipeline writes into the repository on every run (predictions, tuned parameters, model artifacts), and a watching pm2 would restart the server several times a day. Restart deliberately after a deploy:

```
pm2 restart sequencePredictor
```

A restart is needed to pick up: a new `web/dist`, a change to `.env`, a change to the job list in `jobs.js`, and a change of the machine's time zone (Node fixes its zone at start).

### Optuna dashboard

The server does not start or link the Optuna dashboard. Start it by hand when needed:

```
optuna-dashboard sqlite:///db.sqlite3
```

## The web app (`web/`, served under `/app`)

The interface is a React application in `web/` (Vite, TypeScript, ECharts), built to `web/dist` and served by the same server under `/app`, behind the same session. It is the only interface.

- A missing build answers `/app` with status 503 and the instruction to run `npm run build:web`.
- A signed-out client gets a 401 with a JSON body on `/api/*`, never the login page; the app then sends the reader to `/login` with the page to come back to.
- Every JSON POST carries the session's form token (`_csrf`, read from `GET /api/me`).

### The JSON behind the pages

| Module | Routes | Serves |
|---|---|---|
| `api.js` | `GET /api/me`, `/api/markets/:market`, `/api/quotes/:market` | who is signed in; the user view of Crypto and Stocks and the live quotes ([markets](markets.md)) |
| `lottery.js` | `GET /api/home`, `/api/lottery`, `/api/lottery/:game`, `/api/lottery/:game/:date` | Home and the users' Lottery pages ([scoring](scoring.md) for the top five) |
| `admin.js` | `GET /api/admin/week`, `/controls`, `/models`, `/phase-shift`, `/randomness`, `/tuning` (each also per game) | This week and the admin record pages |
| `workbench.js` | `GET /api/admin/history/:game`, `/draw/:game/:date`, `/market/:market/workbench...` | Games in detail and the market workbench ([markets](markets.md)) |
| `system.js` | `/api/admin/jobs...`, `/api/admin/services/:name/:action`, `/api/admin/people...`, `/api/account...`, `/api/whats-new...` | Jobs & services, People, Account, What's new |
| `params.js`, `backtests.js`, `hyperopts.js` | `/api/admin/parameters/:game...`, `/api/admin/backtests...`, `/api/admin/hyperopts...` | Parameters, Backtests, Hyperopts ([tuning](tuning.md)) |
| `docs.js` | `GET /api/admin/docs`, `/api/admin/docs/:page` | the wiki |
| `workers.js` | `GET /api/admin/workers`, `POST /api/admin/workers/cpu/on`, `/off` | the Workers card |
| `council.js` | `/council/api/*` | the Council |

Everything under `/api/admin` is for the administrator only. An unknown address under `/api/` or `/council/api/` answers a JSON 404.

### Pages and menu

Every signed-in reader sees the user pages:

| Menu item | Address | Shows |
|---|---|---|
| Home | `/app/` | the choice between Lottery, Crypto and Stocks, each with its headline |
| Lottery | `/app/lottery` | per game the coming draw's top five rows and each past draw's top five as ranked before it |
| Crypto | `/app/crypto` | the crypto plan and its record ([markets](markets.md)) |
| Stocks | `/app/shares` | the same for the shares |
| Council | `/app/council` | the ask box and the reader's own sessions |
| Account | `/app/account` | own password and own sign-ins (from the name at the foot of the menu) |
| What's new | `/app/whats-new` | the announcements; can replay the introduction |

The administrator also sees the Admin menu:

| Group | Menu item | Address |
|---|---|---|
| | This week (with a status dot) | `/app/admin` |
| Controls | All controls, with anchors for Q0 row tests, Q2 randomness, Feature control, Market rows | `/app/admin/controls` |
| Models | Tuning gates | `/app/admin/tuning` |
| Models | Parameters | `/app/admin/parameters` |
| Models | Backtests | `/app/admin/backtests` |
| Models | Hyperopts | `/app/admin/hyperopts` |
| Models | Best model & rankings | `/app/admin/models` |
| Models | Phase-shift | `/app/admin/phase-shift` |
| Models | Randomness watch | `/app/admin/randomness` |
| Models | Games in detail | `/app/admin/games`, `/app/admin/games/<game>`, `/app/admin/games/<game>/<date>` |
| Models | Crypto workbench, Stocks workbench | `/app/admin/markets/crypto`, `/app/admin/markets/shares` |
| System | Jobs & services | `/app/admin/jobs` |
| System | People | `/app/admin/people` |
| System | Wiki | `/app/admin/wiki` |
| System | What's new | `/app/whats-new` |

The dot on This week is red when a run failed or an output is missing and amber when a run has a new warning signature. Which rows a table shows is display only: nothing is dropped from the day files.

The page topics are documented where they belong: the controls in [research](research.md), the tuning gates, parameters, backtests and hyperopts in [tuning](tuning.md), rankings and phase-shift in [scoring](scoring.md), the market pages and workbench in [markets](markets.md).

### The wiki (Admin › Wiki)

The wiki (`/app/admin/wiki`, `/app/admin/wiki/<page>`) shows this documentation inside the app, to administrators only. `docs.js` lists `README.md` (as *Overview*) and the pages of `docs/*.md` in a fixed order, the changelog last, and serves each as markdown text; the app renders it (`web/src/pages/admin/WikiPage.tsx`). Nothing is written, and only a listed page can be asked for.

- `GET /api/admin/docs` lists the pages: id, title (the page's first `#` line), file, size and last change.
- `GET /api/admin/docs/:page` returns one page's text; an unknown page answers 404.

### Explanations behind a "?"

Every page shows its data and keeps the prose that explains it (how to read a table or chart, definitions, method, caveats) behind a small round **?** that opens in place: a native `<details>`, no hover (`Help` in `web/src/components/admin/Card.tsx`). Titles, summary lines, the data, the headline results, a plan's orders and a one-line version of any warning a reader must see before acting stay visible.

### Old page addresses

`switchover.js` (`redirects`, mounted right after the sign-in middleware) answers every address of the former server-rendered pages with a 302 to its app page, with `Cache-Control: no-store` because the target depends on who asks. Open access reads as the administrator.

| Old address | Administrator | User |
|---|---|---|
| `/` | `/app/` | `/app/` |
| `/lottery` | `/app/lottery` | `/app/lottery` |
| `/database` | `/app/admin/games` | `/app/lottery` |
| `/database/<folder>[?month=]` | `/app/admin/games/<game>` | `/app/lottery/<game>`, or the market's own page for a market folder |
| `/database/<folder>/<day file>` | `/app/admin/games/<game>/<date>` | `/app/lottery/<game>/<date>` |
| `/markets/<m>` | `/app/admin/markets/<m>` | `/app/<m>` |
| `/council[?session=]` | `/app/council` | `/app/council` |
| `/account`, `/whats-new` | the same under `/app` | the same under `/app` |
| `/admin/users`, `/admin/activity` | `/app/admin/people` | `/app/` |
| `/admin/jobs` | `/app/admin/jobs` | `/app/` |

Targets are built from known game names and checked dates only. Only GET and HEAD are redirected; a POST to an old form address answers 404 like any unknown address.

### The last handlers

`switchover.finish`, installed after every route: an unknown address answers 404 and a thrown error 500, as JSON under `/api/` and `/council/api/` and as a page otherwise. A malformed or too large request body answers its own 4xx status. The error and its stack go to the error log and never into the answer.

### Tests of the app's server side

`node test/api.test.js`, `test/lottery.test.js`, `test/admin.test.js`, `test/workbench.test.js`, `test/system.test.js`, `test/server.test.js` (the routes and their order, and that loading the server starts nothing) and `test/switchover.test.js` (the redirects, the 404 and error answers). All are part of `npm test`.

## Login and users

Set `WEB_USER` and `WEB_PASSWORD` to require a login (`auth.js`):

```
cp .env.example .env      # then edit WEB_USER and WEB_PASSWORD
npm start
```

or, without a file:

```
WEB_USER=admin WEB_PASSWORD='a long passphrase' npm start
```

### The administrator and the users

- `WEB_USER` / `WEB_PASSWORD` are the **administrator**. They live only in the environment, never on disk, and cannot be changed from the web; the Account page says so and how to change them.
- With both unset the app is open (local development): every visitor reads as the administrator and the People page is read-only with a notice.
- The administrator sees everything, including the **People** page (`/app/admin/people`), where accounts are added, deleted and given a new password.
- A user signs in with their own password and sees Home, Lottery, Crypto, Stocks, the Council, their account and What's new.
- Accounts are stored in `config/users.json`: user name and salted `scrypt` hash, never the password. Every file account is a plain user.
- The `config/` folder is git-ignored: the daily predictor commits everything under `data/`, so accounts, secrets and state do not live there.

### The account page

Every signed-in user has an account page (`/app/account`). A user changes their own password there; the current password is required, and the change signs their other browsers out while keeping the one that made it. The page also lists the sign-ins recorded for their name, so a session they did not start is visible to them.

### Sessions

- The session is a signed cookie, `HttpOnly` and `SameSite=Strict`, with the `Secure` flag when the request arrived over HTTPS (directly or via `X-Forwarded-Proto`). There is no session store.
- Sessions last 24 hours, renewed while in use, never beyond 7 days from the sign-in.
- Changing a user's password ends that user's sessions; rotating `WEB_PASSWORD` ends the administrator's; a deleted user is out on the next request.
- Five failed sign-ins for a user name from one client address lock that combination for 15 minutes. A sign-in with an unknown name costs the same time as with a known one, so names cannot be enumerated.
- The form token (`_csrf`) is checked on every JSON POST of the app and the Council (`auth.csrfOk`) and survives a session renewal.
- `WEB_TRUST_PROXY` makes the lockout key on the real client behind a reverse proxy.

The login form sends the password in clear text: put the server behind HTTPS (a reverse proxy or TLS terminator) when it is reachable beyond the local machine.

## Audit log

`audit.js` appends one JSON object per line to `config/audit.jsonl`: the time, the event, the user name, the caller's address and a truncated browser string.

Recorded events:

- sign-ins: successful, failed, locked out, sign-outs, a session that is no longer valid;
- passwords: changed by the user, wrong current password, set by the administrator;
- accounts: created, deleted;
- parameters: a manual override set or released;
- backtests: started, cancelled or stopped, deleted;
- hyperopts: settings saved, started, cancelled, applied, deleted;
- the DL360: switched on or off by hand.

The file is rotated at 2 MB with one older file kept, so it is bounded. A failure to write it is logged and swallowed, so the audit trail can never break a sign-in.

The administrator reads the events on the People page's Activity card (`/app/admin/people#activity`), which also shows last seen and the number of sign-ins per account.

## First login, and what's new

A new account meets a short introduction the first time it opens any page: what the project is, the three sections and what each page is for; the Admin menu step is shown only to an administrator. After that, a feature that ships with a note pops that note once, and only for the accounts that may see it.

`announcements.js` is the single file to edit:

- one entry per feature, newest first;
- `level: 'major'` opens the dialog, `'minor'` only counts on the menu's What's new link;
- `audience: 'all'` or `'admin'`;
- ids are permanent, unique and match their date. An edited id would show an old note to everyone again.

Every string is escaped when rendered, so the file holds text, not markup. `node test/announcements.test.js` checks these rules.

What an account has seen lives in `config/user-state.json`, apart from the password hashes; the administrator has no record in `users.json`. The dialog is the app's `WhatsNewDialog` (from `/api/whats-new`, `system.js` on `whatsnew.js`), so it finds the reader on whichever page they land on. "Later" hides it for the session. What's new (`/app/whats-new`) lists everything and can replay the introduction.

## Is a run in progress?

Because every Python entry point takes `process.lock`, the server can tell whether the pipeline is busy without asking the scheduler (`pipeline.js`), whether the job was started by the schedule or by hand.

- `/api/home` and `/api/lottery` carry a `running` field: what runs and since when, never the pid.
- Home and the Lottery page show a line that names the job and how long it has been running, and read their data again every minute while it lasts. Every reader sees it.
- The command line behind the locked pid names the job: for example *The next draws' predictions are being computed*, *The crypto day's predictions are being computed*, *Tomorrow's stock predictions are being computed*, *Weekly tuning: boosting models*, *Weekly tuning: keno sub-selections*. A step that shares its script with another is told apart by its arguments (`jobs.jobOfCmdline`).
- A lock whose process is gone shows nothing; the scripts clean stale locks up themselves.

## Jobs & services and the supervised services

The administrator's **Jobs & services** page (`/app/admin/jobs`, from `GET /api/admin/jobs`, read every 15 seconds) shows the services, the schedule's jobs, the queue, the run history and the workers.

`services.js` keeps the processes the interface depends on running. It supervises one service, the **Council API** (`src/llmCouncil/api.py`):

- it starts with the server and stops with it;
- when it dies it is restarted with a growing delay (2 seconds, doubling up to 5 minutes; reset after a run of more than a minute);
- more than five exits within ten minutes mark it failed instead of restarting it further;
- something already answering on its port (another checkout, an instance started by hand) is adopted and shown as such, not duplicated;
- `COUNCIL_API_AUTOSTART=off` disables the autostart;
- its log is `log/councilApi.log`, rotated at 10 MB.

The page shows state, pid, uptime, restart count, last exit, the tail of the service log and start, stop and restart buttons (`POST /api/admin/services/:name/:action`).

A service is an ordinary child of the server: it is cheap to restart and comes and goes with the server. The scheduled pipeline jobs are the opposite case and are launched detached; see the next section.

## The schedule (`jobs.js`)

The server starts the daily predictors, the weekly tuning chain and the weekly control experiments itself. There is no crontab entry.

### The three daily runs

| Key | Name | Command | Slot | Caught up for | Log |
|---|---|---|---|---|---|
| `cryptoPredictor` | Night predictor: crypto | `Predictor.py -a true -g crypto,cryptoweek` | 00:05 UTC, every day | 20 h | `log/cryptoPredictor.log` |
| `predictor` | Morning predictor: lottery games | `Predictor.py -a true -g euromillions,lotto,eurodreams,jokerplus,keno,pick3,vikinglotto` | 06:00 local time, every day | 20 h | `log/predictor.log` |
| `sharesPredictor` | Evening predictor: stocks | `Predictor.py -a true -g shares,sharesweek` | 18:05 local time, Monday to Friday | until 08:30 the next morning | `log/sharesPredictor.log` |

- The crypto slot is in UTC, five minutes after the crypto day closes (01:05 Brussels in winter, 02:05 in summer). A crypto week closes on Monday 00:00 UTC, so Monday's run brings the new week.
- The other slots are in the server process's local time, Europe/Brussels on the production machine, so they follow summer and winter time.
- The lottery run is usually done by about 08:00, before the first sales close.
- The stocks run starts as soon as Euronext's day is complete (18:00), so the card is up before the 09:00 open. After 08:30 a missed card is no longer worth starting.
- Only the lottery predictor (`predictor`) finishing starts the weekly plans. A bare `Predictor.py` started by hand is read as the lottery predictor.
- Job children get a minimal environment: `PATH`, `HOME`, `LANG`, `PYTHONUNBUFFERED=1` and `TZ` from the server's environment, or `UTC` when it has none.

### The weekly tuning chain (Saturday)

Started when Saturday's lottery predictor finishes, in this order ([tuning](tuning.md) documents each step):

| Key | Command | Log |
|---|---|---|
| `hyperoptStatistics` | `HyperoptStatistics.py` | `hyperoptStatistics.log` |
| `hyperoptBoost` | `HyperoptBoost.py` | `hyperoptBoost.log` |
| `hyperoptRLTicket` | `HyperoptRLTicket.py` | `hyperoptRLTicket.log` |
| `hyperoptEnsemble` | `HyperoptEnsemble.py` | `hyperoptEnsemble.log` |
| `hyperoptQuantum` | `HyperoptQuantum.py` | `hyperoptQuantum.log` |
| `trainMetaLearner` | `TrainMetaLearner.py` | `TrainMetaLearner.log` |
| `kenoSubsetTuning` | `HyperoptStatistics.py -g keno -s KenoSubsetTuning` | `kenoSubsetTuning.log` |

The order is load-bearing: the quantum tuner runs before the retrain so the retrain sees fresh parameters, and the keno sub-selection step runs last because it scores the artifacts the retrain has just written.

### The weekly control experiments (Sunday)

Started when Sunday's lottery predictor finishes, in this order ([research](research.md) documents each control):

| Key | Command | When |
|---|---|---|
| `nullControls` | `NullControls.py -g <all games> -m both -n 3 --market-repetitions 5 --seed weekly -d 120` | first Sunday of the month only |
| `randomnessDiscrimination` | `RandomnessDiscrimination.py -g <all games> -w 10 -n 20 --seed weekly` | every Sunday |
| `featureControl` | `IrrelevantFeatureControl.py -g <all games> -r 3 -k 3 --seed weekly` | every Sunday |
| `marketRows` | `MarketRows.py -g crypto,shares,cryptoweek,sharesweek -d 250` | every Sunday |

- The controls are measurements of the rows, never inputs to them, and take hours; that is why they have their own day instead of the tail of the chain.
- The null band is monthly: it is queued only by the plan occurrence whose Sunday is day 1 to 7 of its month (`monthly` in `jobs.js`). A catch-up on Monday or Tuesday counts as the Sunday before. The scheduler log says when it is left out, dry run included.
- The feature control reads the meta-learner table the Saturday chain caches and the market rows read the tuned market values, so the controls come after the chain.
- When the plan's last job (`marketRows`) finishes, whatever its exit code, the week's reading is written to `data/readings/<sunday>.json`. On a week without a fresh band the reading carries the latest band with its own date.
- On a first Sunday the three weekly controls start only after the band, so they and the reading land later.
- This week (`/app/admin`) expects the band only of a first Sunday's weekend; on the others it lists it as not due, with the Sunday it runs next. A step that carries `since` in `jobs.js` is expected only of the weekends from that day on.
- With a CPU worker set up, the controls run on that machine instead; see *The Sunday controls on the DL360*.

### The backtest job

A fifteenth job, `backtest` (`RunBacktest.py`), is never scheduled and has no *Run now*: backtests are started from the Backtests page ([tuning](tuning.md)). When a backtest runs in this server's queue:

- every scheduled job in the queue goes first, wherever it waits;
- a backtest starts only when at least 15 minutes remain before the next daily slot;
- it gets a budget that ends 10 minutes before that slot, three hours at most.

### How the queue behaves

- **A plan starts when its predictor finishes**, not at a fixed hour. The trigger is seen either from the scheduler's own run or from `process.lock` being released by a lottery predictor started outside the scheduler. Only a predictor finishing after that day's 06:00 anchor counts.
- **A job that finds the pipeline busy waits** in a first-in-first-out queue (one job at a time) instead of exiting. A chain or a control running into a daily slot queues that day's predictor behind it; nothing is pre-empted.
- **A missed slot is caught up** within the window in the table above. The plans have a safety net: when a plan has not started, it is started from 23:00 of its day until 09:00 two days later (Monday 09:00 for the chain, Tuesday 09:00 for the controls).
- **A first start fires nothing it never saw.** The slots inside their catch-up window at the first ever start are recorded as already handled, and so is the pending occurrence of a plan newly added to a running installation.
- **A running job survives a deploy.** Every job is launched with `setsid --fork`, so its parent is PID 1 and pm2's tree kill walks past it. The restarted server finds the run in `config/scheduler.json`, adopts it and reports its real duration and exit code when it ends. On SIGTERM only the timer stops.
- **A blocked queue stays noisy.** Waiting on the lock is logged again every half hour; after six hours the wait is recorded as a blocked entry on the Jobs & services page. The job stays queued.
- **Every run is recorded**: job, the reason it started, duration, exit code and the git SHA it ran at, on the Jobs & services page and in `log/scheduler.log`. A job that ended without writing an exit code is recorded as that, never as a success. Each job's own log gets a start and a finish marker.

### Modes

| Mode | How | Effect |
|---|---|---|
| `dry` | the default (no `SCHEDULER` set) | records what it would have started and starts nothing |
| `on` | `SCHEDULER=on` | runs the schedule |
| `off` | `SCHEDULER=off`, or `touch config/scheduler.disabled` | nothing is scheduled and nothing can be started by hand; the file takes effect without a restart and also stops the worker queues |

Leaving `on` is a real rollback: the queue survives a restart in `config/scheduler.json`, and anything the schedule had queued is discarded when the mode is no longer `on`. A job started by hand is kept.

### Actions on the Jobs & services page

All are for the administrator only, POST, and checked for the form token.

| Action | Route | Notes |
|---|---|---|
| Run now | `POST /api/admin/jobs/run` | queues the job as started by hand; works in `dry` too; refused when the job already runs or waits |
| Cancel | `POST /api/admin/jobs/cancel` | takes a waiting job out of the queue |
| Stop | `POST /api/admin/jobs/stop` | SIGTERM to the run's whole process group, so the Python process goes with its wrapper; named by run id |
| Log | `GET /api/admin/jobs/run/:id/log` | the run's log |

### State, logs and limits

- `config/scheduler.json`: handled occurrences, the queue, the running job, the history (the newest 40 runs).
- `config/runs/`: per run a pid file while it runs and an exit file when it ends; swept after 7 days.
- `log/scheduler.log` and the job logs: rotated at 10 MB, one older file kept.
- The scheduler ticks every 10 seconds.
- `test/jobs.test.js` pins `TZ=Europe/Brussels` and walks the slots, the plan triggers, the monthly rule and the mirror of `runHyperopt.sh`.

## The workers (`workers.js`)

Two machines run work beside this server. The main server stays the only writer of `data/` and of `bestParams_*.json`: a worker only returns a record file, a result file or a control's records.

### The machines

- **The GPU worker:** an RTX 2080 Ti machine, an LXC container that is always on and shared with other services. It runs the deep-learning rows' backtests and the deep-learning hyperopts. TensorFlow gets one logical GPU of `gpuMemoryMb` (`SP_GPU_MEMORY_MB`, `HyperoptDeepLearning.cap_gpu_memory`), leaving the rest of the card to the other services.
- **The CPU worker:** an HP DL360 G7 (24 threads, 148 GB; no AVX, so no TensorFlow). It also hosts the LLM council. It runs the statistical, market and boosting rows' backtests, the other five tuners' hyperopts and the Sunday controls. It is about half this server's speed, so the weekend tuning chain stays on this server. It is power hungry, so it is off unless there is work.

Without a `gpu` entry the deep-learning rows cannot be backtested. Without a `cpu` entry the other backtests run in the scheduler's queue on this server, the hyperopts run on this server directly and the controls run in the scheduler's queue.

### Power of the CPU worker

- **Switched on when needed.** A run for it, when the machine does not answer, switches it on through its iLO: `worker/power.sh wake` uses IPMI with the password from a root-only file (`iloEnv`), passed through `IPMI_PASSWORD` and `-E`, never on a command line. It waits for ssh (15 minutes at most) and marks the boot as the worker's (`/run/sequencePredictor-worker/woken`). A machine that does not come up often has a USB disk plugged in, which stops its start-up.
- **Switches itself off.** On the machine, `sp-worker-idle.timer` runs `/usr/local/sbin/sp-worker-idle` (`worker/idle-shutdown.sh`; units `worker/sp-worker-idle.*`) every minute. Only on a marked boot (`/run` is emptied at every boot, so a start by hand is never switched off) it shuts the machine down cleanly after `IDLE_MINUTES` (10, `/etc/default/sp-worker-idle`) without any of: a running backtest, a running control, a running hyperopt, a council request (a `slot` line of a `llama@` service in the journal), or an active login (`who -u` idle below the window).
- **By hand.** Jobs & services › Workers (`GET /api/admin/workers`, read every 15 seconds) shows it: on or off, whether the worker switched it on, idle minutes. *Switch on* (`POST /api/admin/workers/cpu/on`) is the same marked wake. *Shut down* (`POST /api/admin/workers/cpu/off`) is refused while a backtest or hyperopt runs or waits for it and while a control runs or waits; the council goes down with the machine. Both go to the audit log.

### Settings

`config/workers.json` (git-ignored, so the addresses are never in the repository):

```
{"gpu": {"host": "root@<address>", "gpuMemoryMb": 6144},
 "cpu": {"host": "root@<address>", "iloEnv": "/root/.config/sequencePredictor/ilo.env", "idleMinutes": 10}}
```

| Key | Default | Meaning |
|---|---|---|
| `host` | required | ssh target; the server reaches the workers over ssh with root's key |
| `dir` | `/opt/sequencePredictor-worker` | the worker's folder |
| `python` | `<dir>/venv/bin/python` | the worker's Python |
| `enabled` | `true` | `false` switches the lane off |
| `gpuMemoryMb` (gpu) | 6144 | GPU memory TensorFlow may take |
| `iloEnv` (cpu) | none | root-only file with the iLO settings; without it the machine is never switched on or off by this server |
| `idleMinutes` (cpu) | 10 | what the page says; the machine's own window is `/etc/default/sp-worker-idle` |
| `localDir` (cpu) | `/opt/sequencePredictor-worker-local` | where a hyperopt runs on this server as fallback |

On each worker, `<dir>/venv` holds the pipeline's Python packages (without TensorFlow on the CPU worker), `<dir>/app` the copy the backtests use, `<dir>/hyperopt` the copy and the Optuna studies of the hyperopts, and `<dir>/controls` the copy of the controls.

The CPU worker also needs the foundation libraries in `/root/.foundation-libs` and the Chronos weights ([models](models.md), *Foundation models*: the same install commands, run with the worker's Python). Which models a cached meta-learner table is expected to hold depends on whether those libraries are installed, so a worker without them refuses every table the main server cached.

### A backtest on a worker

`worker/remote-backtest.sh`:

1. wakes the CPU worker when needed; while the machine starts, the record carries a `phase` the page shows;
2. copies the code (no `.git`, `config/`, `data/`, `web/`, `log/` or `test/`), `data/trainingData`, `config/paramOverrides.json` and the backtest's record to `<dir>/app`;
3. runs `RunBacktest.py --id <id> --budget <s>` there over ssh;
4. copies the record back every 15 seconds (the page's progress) and at the end.

`config/` otherwise never leaves the server: it holds the accounts and the session secret.

### The queues

- One queue per worker, beside the scheduler's. A worker run takes no `process.lock` and uses neither this server's CPU nor its GPU, so it neither waits for the scheduled jobs nor has to end before the next one.
- One run at a time per worker, first come first served.
- A backtest's budget is 15 minutes plus 15 minutes a draw on the GPU worker and 12 hours on the CPU worker. A hyperopt's budget follows from its form ([tuning](tuning.md)), two weeks at most.
- `config/scheduler.disabled` stops the worker queues too.
- State: `config/workers-state.json` (GPU) and `config/workers-cpu-state.json` (CPU). Logs: `log/backtest-gpu.log`, `log/backtest-cpu.log`, and the queues' lines in `log/scheduler.log` as `[workers]`.
- A record keeps the lane it was queued in (`worker`: gpu, cpu or server).

### Stop and restarts

- *Stop* sends SIGTERM to the script alone, not its process group, which would cut the ssh session first. The script signals the remote run, which stops its training child and writes *stopped*; while the machine is still starting, the script writes *stopped* itself.
- A run survives a server restart (`setsid`) and is adopted at the next start, as the scheduler's jobs are. A run lost with the server shows as *ended*.

### Hyperopts on the workers

The Hyperopts page ([tuning](tuning.md)) starts tuner runs through the same queues: the deep-learning tuner on the GPU worker, the other five on the CPU worker. `worker/remote-hyperopt.sh`:

- copies the code with the served `bestParams_<game>.json` and `data/trainingData` to `<dir>/hyperopt`, and `data/database` for the tuners that score the stored predictions;
- keeps the Optuna studies (`db.sqlite3`) and the run marker there between runs, so each machine has its own study history;
- runs the tuner detached from the ssh session, with no commit and no push;
- copies its output to `log/hyperopts/<id>.log` and its result to `config/hyperopts/<id>.result.json` every 15 seconds.

When the CPU worker does not come up, or cannot be copied to, before the tuner ran there, the script ends with 75 and the run moves to this server: first in the queue, in `localDir`, with this server's `python3`, niced. Its record says so.

### The Sunday controls on the DL360

With a `cpu` entry in `config/workers.json`, the control plan's four jobs run on the DL360 instead of in the scheduler's queue.

- Same trigger, same order, same log files and markers, same week's reading, written when the market rows finish.
- The scheduler's history carries them with `worker: 'cpu'`; the Jobs page's runs say *on the DL360*.
- They have their own first-in-first-out queue in `jobs.js` (`state.remote` in `config/scheduler.json`): one at a time, beside this server's run, taking no `process.lock` here. The daily runs do not wait behind them.
- The band takes about twice as long there as on this server, so on a first Sunday the weekly controls and the reading come later, inside the plan's window (until Tuesday 09:00).
- The Jobs page's usual times mix the runs here and there.

`worker/remote-control.sh` does one control:

1. the wake, as for a backtest;
2. the code to `<dir>/controls`, so a backtest and a control never share a copy;
3. once per plan occurrence `data/trainingData`, `data/hyperOptCache` and `config/paramOverrides.json`, so the controls of one Sunday read the same draws even when they run into Monday; `data/controls` every time;
4. the control runs there detached from the ssh session (a network interruption does not end it), 48 hours at most; its output is copied into the job's log every 15 seconds;
5. at the end the records it writes are copied back into `data/controls/`: the band's `results/` and game folders, `discrimination/`, `features/`, `markets/`.

A feature control that wrote no game's record ends with status 1 and the line `NOTHING WRITTEN`: a run in which every cached table was refused is a failed run on the Jobs page, not a short successful one.

**Fallback.** Anything failing before the control started there (the machine does not come up, cannot be copied to) ends with 75. That control and every one waiting behind it are then queued on this server, in their order, and the scheduler's history keeps the attempt with the reason. A control waiting for the DL360 when its entry is taken out of `config/workers.json` runs here too.

*Stop* signals the wrapper alone, which stops the control there and copies back what it wrote. *Cancel* takes a waiting one out. A run survives a server restart and is adopted.

### Tests

- `node test/workers.test.js`: both queues, a real launch of a stand-in script, Stop, adoption, the power status and switches with a stand-in `power.sh`.
- `node test/backtests.test.js`: which rows go to which worker and the draw limits.
- `node test/jobs-remote.test.js`: the controls on the DL360 with a stand-in wrapper (routing, the history entry, the fallback in order, Stop, Cancel, adoption, a plan occurrence's shared snapshot).
- `node test/hyperopts.test.js`: the Hyperopts page's server side, including the CPU lane's move to this server.
- `node test/docs.test.js`: the wiki's list and pages, the markdown reader, and the documentation itself (every page present, every link leading somewhere, no person named).

## Testing

```
npm test
```

`npm test` is the test suite. It runs, one after the other and stopping at the first failure:

- the twenty-one Node tests in `test/*.test.js`: announcements, jobs, jobs-remote, readings, council, controls, prizes, params, backtests, workers, hyperopts, docs, markets, api, admin, workbench, lottery, pipeline, system, server, switchover;
- the Python self-checks: `python3 -m src.<Module>` for the modules that carry one (for example `src.TuningScore`, `src.TuningResult`, `src.TuningGate`, `src.Lockbox`, `src.PrizeTables`, `src.MarketSettle`, `src.MarketOrders`, `src.Markov`, `src.Helpers`) and the `_self_check()` of the entry points `NullControls`, `RandomnessDiscrimination`, `HyperoptEnsemble`, `HyperoptStatistics`, `HyperoptBoost`, `HyperoptRLTicket`, `MarketRows` and `SwitchSharesEuronext`.

The full list is the `test` script in `package.json`. Each test can be run on its own, for example `node test/jobs.test.js` or `python3 -m src.TuningResult`.

The `test/` folder also holds `trainingData`, `database`, `models` and `backtestResults` folders, to try a model without touching the real data.

## Fetching data

The lottery draws come from the Nationale Loterij API (`src/DataFetcher.py`):

```
https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&date-from=<ms>&size=62&date-to=<ms>&game-names=<Game>
```

For the latest draws of every game: `https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&previous-draws=5`. The dates are in milliseconds since 1970.

- A failed answer is tried three times in all, after waits of 30 and 120 seconds; each failure is logged with its HTTP status and the start of the reply.
- When the API still fails, the game falls back to its CSV on disk.
- The CSV files with the real draws can also be downloaded from the lottery's website.

The market data (crypto and shares) is documented in [markets](markets.md).
