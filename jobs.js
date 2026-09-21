// The schedule that used to live in crontab, moved into the web server
// (README roadmap item 8).
//
// WHY. cron can only start a job at a fixed hour. On 2026-09-19 the daily
// predictor finished at 10:30 and the weekly tuning chain sat idle until its
// 14:00 slot, then ran 16.5 h to 06:29 - while process.lock already
// guarantees that only one Python job runs at a time, so the waiting was
// pure loss. Worse, a tuner that found the lock held simply exited, which is
// how a whole week of tuning can be skipped in silence. Here the chain
// starts when Saturday's predictor *finishes*, a job that finds the lock
// held waits in a queue instead of exiting, and a run missed while the box
// was down is caught up.
//
// WHY THE JOBS ARE NOT ORDINARY CHILDREN. pm2 runs this server with
// treekill, so an ordinary child of this process is killed 1.6 s into any
// `pm2 restart` - measured on this box. Every job is therefore launched
// through `setsid --fork`, which reparents it to PID 1: a deploy, a crash or
// a restart of the web server cannot touch a running 12-hour tuning job.
// The price is that there is no child handle to wait on, so a run reports
// itself through two files in config/runs/ (its PID while it runs, its exit
// code when it ends) and this module polls them. That same indirection is
// what lets a restarted server ADOPT a run that is still in progress instead
// of losing track of it.
//
// SAFETY. The default mode is 'dry': the scheduler works out what it would
// run, records it, and starts nothing. Deploying this file therefore cannot
// double-run the pipeline next to the crontab entries, which still exist
// until cutover. At cutover the three crontab entries are removed and
// SCHEDULER=on goes into .env. config/scheduler.disabled switches the
// schedule off again without a restart.

const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const config = require('./config');
const auth = require('./auth');

const ROOT = __dirname;
const LOG_DIR = path.join(ROOT, 'log');
const RUN_DIR = path.join(config.CONFIG_DIR, 'runs');
const STATE_FILE = path.join(config.CONFIG_DIR, 'scheduler.json');
const DISABLE_FILE = path.join(config.CONFIG_DIR, 'scheduler.disabled');
const LOCK_FILE = path.join(ROOT, 'process.lock');
const SCHEDULER_LOG = path.join(LOG_DIR, 'scheduler.log');

// Ten seconds: cheap (a handful of stat calls) and it bounds both how long
// a finished job keeps showing as running and the gap between two steps
// of the weekly chain - minutes of slack over a run of hours, against the
// 3.5 idle hours the fixed cron slot cost.
const TICK_MS = 10 * 1000;
const LOG_MAX_BYTES = 10 * 1024 * 1024;    // rotate at 10MB, keep one old file
const HISTORY_KEPT = 40;
const OCCURRENCES_KEPT = 120;
const RUN_FILE_DAYS = 7;
// How long after a run is launched we keep looking for its PID file before
// calling the launch failed. A Python start-up is seconds; this is generous.
const PID_WAIT_MS = 30 * 1000;
// A blocked queue must stay noisy. runHyperopt.sh gave up after 6 h and said
// so; waiting is better than giving up, but only if the waiting is visible -
// otherwise the site can stop producing predictions with one log line as the
// only evidence.
const WAIT_RELOG_MS = 30 * 60 * 1000;
const WAIT_ALARM_MS = 6 * 3600 * 1000;

// --- what runs, and when ----------------------------------------------------
// One entry per Python entry point, mirroring runPredictor.sh and
// runHyperopt.sh exactly - same commands, same log files, same order. The
// chain order is load-bearing and documented there: every tuner takes the
// shared process.lock, HyperoptQuantum.py must run BEFORE TrainMetaLearner.py
// so the weekly retrain sees freshly tuned quantum parameters, and
// TrainMetaLearner.py stays last.
const CHAIN_PLAN = 'weeklyTuning';
const CHAIN_WEEKDAY = 6;                   // Saturday
const CHAIN_ANCHOR_HOUR = 9;               // ...its predictor slot
const CHAIN_CATCHUP_AFTER_MS = 14 * 3600 * 1000;   // Saturday 23:00
const CHAIN_CATCHUP_UNTIL_MS = 48 * 3600 * 1000;   // Monday 09:00

const JOBS = [
  {
    key: 'predictor',
    name: 'Daily predictor',
    description: "Today's predictions for every game, deep learning rows included and time-boxed "
               + '(Predictor.py -a true). Replaces the two 09:00 crontab entries.',
    script: 'Predictor.py',
    args: ['-a', 'true'],
    log: 'predictor.log',
    daily: { hour: 9, minute: 0 },
    // How long after its slot a missed run is still worth starting. The box
    // being down over a night is the case this exists for; past 20 h the
    // next morning's run is close enough to supersede it.
    catchUpHours: 20,
  },
  {
    key: 'hyperoptStatistics',
    name: 'Weekly tuning: statistical models',
    description: 'Tunes the statistical rows into bestParams_<game>.json (HyperoptStatistics.py).',
    script: 'HyperoptStatistics.py', args: [], log: 'hyperoptStatistics.log', chain: true,
  },
  {
    key: 'hyperoptBoost',
    name: 'Weekly tuning: boosting models',
    description: 'Tunes XGBoost Model into the same files (HyperoptBoost.py).',
    script: 'HyperoptBoost.py', args: [], log: 'hyperoptBoost.log', chain: true,
  },
  {
    key: 'hyperoptRLTicket',
    name: 'Weekly tuning: RL ticket model',
    description: 'Tunes RL Ticket Model - pure numpy, minutes not hours (HyperoptRLTicket.py).',
    script: 'HyperoptRLTicket.py', args: [], log: 'hyperoptRLTicket.log', chain: true,
  },
  {
    key: 'hyperoptEnsemble',
    name: 'Weekly tuning: ensemble subsets',
    description: 'Selects the rows SubsetEnsemble Model votes over, from the stored day JSONs (HyperoptEnsemble.py).',
    script: 'HyperoptEnsemble.py', args: [], log: 'hyperoptEnsemble.log', chain: true,
  },
  {
    key: 'hyperoptQuantum',
    name: 'Weekly tuning: quantum meta-learners',
    description: 'Tunes the quantum-kernel SVC and the VQC. Must run before the retrain below, '
               + 'which is the whole point: fresh parameters, not week-old ones (HyperoptQuantum.py).',
    script: 'HyperoptQuantum.py', args: [], log: 'hyperoptQuantum.log', chain: true,
  },
  {
    key: 'trainMetaLearner',
    name: 'Weekly retrain: meta-learner',
    description: 'Retrains the stacking meta-learner on the freshly tuned parameters. Always last (TrainMetaLearner.py).',
    script: 'TrainMetaLearner.py', args: [], log: 'TrainMetaLearner.log', chain: true,
  },
];

const CHAIN_KEYS = JOBS.filter((j) => j.chain).map((j) => j.key);

function jobByKey(key) {
  return JOBS.find((j) => j.key === key) || null;
}

// --- mode -------------------------------------------------------------------
// 'on'  - the schedule starts jobs (cutover state)
// 'dry' - the schedule only records what it would have started (default, and
//         what makes deploying next to a live crontab safe)
// 'off' - nothing scheduled and no manual runs either (kill switch)
function mode() {
  if (fs.existsSync(DISABLE_FILE)) return 'off';
  const raw = String(process.env.SCHEDULER || 'dry').trim().toLowerCase();
  return ['on', 'dry', 'off'].includes(raw) ? raw : 'dry';
}

function modeReason() {
  if (fs.existsSync(DISABLE_FILE)) return `${path.relative(ROOT, DISABLE_FILE)} exists`;
  return process.env.SCHEDULER ? 'SCHEDULER in the environment' : 'default (no SCHEDULER set)';
}

// --- small helpers ----------------------------------------------------------
function pad(n) { return String(n).padStart(2, '0'); }

function dayId(date) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

// ISO week, so "this week's tuning chain" survives new year's week.
function weekId(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const week = Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
  return `${d.getUTCFullYear()}-W${pad(week)}`;
}

function dailySlot(now, at) {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate(), at.hour, at.minute, 0, 0);
}

function nextDaily(now, at) {
  const slot = dailySlot(now, at);
  if (slot > now) return slot;
  const next = new Date(slot);
  next.setDate(next.getDate() + 1);
  return next;
}

// The most recent <weekday> at <hour>:<minute> that is not in the future.
function lastWeekdaySlot(now, weekday, hour, minute) {
  const slot = new Date(now.getFullYear(), now.getMonth(), now.getDate(), hour, minute, 0, 0);
  let back = (slot.getDay() - weekday + 7) % 7;
  if (back === 0 && slot > now) back = 7;
  slot.setDate(slot.getDate() - back);
  return slot;
}

function humanDuration(seconds) {
  if (seconds === null || seconds === undefined) return '-';
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

function since(at) {
  return at ? humanDuration((Date.now() - at) / 1000) : '-';
}

function stamp(at) {
  if (!at) return '-';
  const d = new Date(at);
  return `${dayId(d)} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// PIDs are recycled, and a recycled one would make a dead job look alive
// forever - or, worse, point the Stop button at a stranger's process group
// after a reboot. The wrapper's command line contains the run id (its PID and
// exit files are named after it), so this is identity, not just liveness.
function exitCodeFrom(text) {
  const raw = String(text).trim();
  return /^-?\d+$/.test(raw) ? Number(raw) : null;
}

function alive(pid, runId) {
  if (!pid) return false;
  let cmdline;
  try {
    cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf-8');
  } catch (e) {
    return false;
  }
  return runId ? cmdline.includes(runId) : true;
}

function gitSha() {
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'],
      { cwd: ROOT, encoding: 'utf-8', timeout: 5000 }).trim();
  } catch (e) {
    return null;
  }
}

function log(text) {
  const line = `${new Date().toISOString()} [scheduler] ${text}`;
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.appendFileSync(SCHEDULER_LOG, `${line}\n`);
  } catch (e) { /* the log is a convenience, never a failure path */ }
  console.log(line);
}

// --- log rotation -----------------------------------------------------------
// README item 8 asks for it: predictor.log had grown to 20 MB. Rotating a
// file that a running job holds open is safe on Linux but confusing (the
// writer keeps filling the renamed file), so it only happens while no
// pipeline job holds the lock, and always right before we start one.
function rotate(file) {
  try {
    if (fs.statSync(file).size > LOG_MAX_BYTES) {
      fs.renameSync(file, `${file}.1`);
      log(`rotated ${path.basename(file)} (over ${Math.round(LOG_MAX_BYTES / 1048576)} MB)`);
    }
  } catch (e) { /* no such log yet */ }
}

function rotateAll() {
  if (lockHolder()) return;
  JOBS.forEach((job) => rotate(path.join(LOG_DIR, job.log)));
  rotate(SCHEDULER_LOG);
}

// --- the shared lock --------------------------------------------------------
// Mirrors what server.js shows in its banner: every Python entry point takes
// this one PID lock, so it is the honest answer to "is the pipeline busy",
// and it stays true for a job someone starts by hand in a terminal.
function lockHolder() {
  let pid;
  try {
    pid = Number(fs.readFileSync(LOCK_FILE, 'utf-8').trim());
  } catch (e) {
    return null;
  }
  if (!pid || Number.isNaN(pid)) return null;
  let cmdline;
  try {
    cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf-8').replace(/\0/g, ' ').trim();
  } catch (e) {
    return null;                     // stale lock of a dead run; the scripts clean it up
  }
  const job = JOBS.find((j) => cmdline.includes(j.script));
  return { pid, script: job ? job.script : cmdline.split(' ').slice(-1)[0], key: job ? job.key : null };
}

// --- state ------------------------------------------------------------------
// Everything that has to survive a restart: which occurrences have been
// handled, what is queued, what is running, and the run history the Jobs
// page shows. In config/ because it is gitignored - data/ is committed by
// the pipeline itself.
const EMPTY = { version: 1, occurrences: {}, queue: [], current: null, history: [], baselinedAt: null };

let state = null;
let timer = null;
let lastLock = null;          // for spotting "the predictor just finished"
let waiting = null;           // {since, lastLogAt, reported} while the lock blocks the queue

function loadState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    return Object.assign({}, EMPTY, parsed && typeof parsed === 'object' ? parsed : {});
  } catch (e) {
    if (e.code !== 'ENOENT') log(`scheduler.json unreadable (${e.message}) - starting from empty state`);
    return JSON.parse(JSON.stringify(EMPTY));
  }
}

function saveState() {
  try {
    fs.mkdirSync(config.CONFIG_DIR, { recursive: true, mode: 0o700 });
    const tmp = `${STATE_FILE}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(state, null, 2), { mode: 0o600 });
    fs.renameSync(tmp, STATE_FILE);
  } catch (e) {
    log(`could not write scheduler.json: ${e.message}`);
  }
}

function remember(id, note) {
  state.occurrences[id] = { at: new Date().toISOString(), note };
  const ids = Object.keys(state.occurrences);
  if (ids.length > OCCURRENCES_KEPT) {
    ids.sort((a, b) => String(state.occurrences[a].at).localeCompare(String(state.occurrences[b].at)))
      .slice(0, ids.length - OCCURRENCES_KEPT)
      .forEach((old) => delete state.occurrences[old]);
  }
}

function record(entry) {
  state.history.unshift(entry);
  state.history = state.history.slice(0, HISTORY_KEPT);
  saveState();
}

// --- the queue --------------------------------------------------------------
// FIFO and single-worker, which is what makes the decisions taken with the
// owner true: a Saturday chain that overruns into Sunday 09:00 queues that
// day's predictor behind it rather than pre-empting the tuner. A job key is
// never queued twice - a second copy of today's predictor would compute the
// same day again.
function enqueue(key, reason, manual) {
  const job = jobByKey(key);
  if (!job) return false;
  if (state.current && state.current.key === key) return false;
  if (state.queue.some((q) => q.key === key)) return false;
  // `manual` marks the one thing dry mode is still allowed to start: a job an
  // administrator asked for by hand. Without the mark, a queue written while
  // the schedule was on would drain the next time the process came up in dry
  // mode - the opposite of what dry mode promises.
  state.queue.push({ key, reason, queuedAt: Date.now(), manual: Boolean(manual) });
  log(`queued ${job.name} (${reason})`);
  return true;
}

// A plan is one occurrence that may expand into several queued jobs: the
// weekly chain is six of them, kept in order by the FIFO queue.
function enqueuePlan(plan, occurrenceId, reason) {
  const keys = plan === CHAIN_PLAN ? CHAIN_KEYS : [plan];
  const dry = mode() === 'dry';
  remember(occurrenceId, reason);
  if (dry) {
    log(`DRY RUN: would start ${keys.length === 1 ? jobByKey(keys[0]).name : `the weekly chain (${keys.length} jobs)`} - ${reason}`);
    record({ at: Date.now(), key: plan, name: keys.length === 1 ? jobByKey(keys[0]).name : 'Weekly tuning chain',
             reason, dry: true, occurrence: occurrenceId });
    return;
  }
  const queued = keys.filter((key) => enqueue(key, reason));
  if (queued.length !== keys.length) {
    // Said out loud rather than swallowed: an occurrence is marked handled
    // even when its job was already waiting, so the log has to show that
    // today's slot is covered by that waiting run and not by a new one.
    log(`${keys.length - queued.length} of ${keys.length} job(s) for ${occurrenceId} were already queued or running - `
      + 'this occurrence is covered by them, not started again');
  }
  saveState();
}

// --- launching --------------------------------------------------------------
// The child gets an explicit, minimal environment for the same reason
// services.js does: inheriting this process's env would hand a 12-hour job
// whatever pm2 froze at boot. cron gave these scripts less than this.
function childEnv() {
  return {
    PATH: process.env.PATH || '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    HOME: process.env.HOME || '/root',
    LANG: process.env.LANG || 'C.UTF-8',
    TZ: process.env.TZ || 'UTC',
    PYTHONUNBUFFERED: '1',
  };
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

function launch(entry) {
  const job = jobByKey(entry.key);
  const logFile = path.join(LOG_DIR, job.log);
  fs.mkdirSync(LOG_DIR, { recursive: true });
  fs.mkdirSync(RUN_DIR, { recursive: true, mode: 0o700 });
  rotate(logFile);

  const runId = `${job.key}-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  const pidFile = path.join(RUN_DIR, `${runId}.pid`);
  const exitFile = path.join(RUN_DIR, `${runId}.exit`);
  const sha = gitSha();

  // The wrapper is what makes a reparented job observable: it publishes its
  // own PID, runs the script exactly as the shell scripts did, and writes
  // the exit code where the poller can find it. The markers in the job's own
  // log make a run traceable there too, which cron never gave us.
  const script = [
    `echo $$ > ${shellQuote(pidFile)}`,
    `cd ${shellQuote(ROOT)} || exit 97`,
    `printf '=== scheduler: %s started %s UTC ===\n' ${shellQuote(job.name)} "$(date -u '+%F %T')${sha ? ` ${sha}` : ''}" >> ${shellQuote(logFile)}`,
    'started=$(date +%s)',
    `python3 ${shellQuote(job.script)}${job.args.length ? ` ${job.args.map(shellQuote).join(' ')}` : ''} >> ${shellQuote(logFile)} 2>&1`,
    'status=$?',
    `printf '=== scheduler: %s finished status=%s after %ss ===\n' ${shellQuote(job.name)} "$status" "$(( $(date +%s) - started ))" >> ${shellQuote(logFile)}`,
    `echo "$status" > ${shellQuote(exitFile)}`,
    `rm -f ${shellQuote(pidFile)}`,
  ].join('\n');

  try {
    // setsid --fork: the job's parent becomes PID 1, so pm2's treekill walks
    // right past it on the next deploy. Verified on this box.
    const starter = spawn('setsid', ['--fork', 'bash', '-c', script], {
      cwd: ROOT, env: childEnv(), stdio: 'ignore', detached: true,
    });
    starter.unref();
    starter.on('error', (error) => log(`could not launch ${job.name}: ${error.message}`));
  } catch (e) {
    log(`could not launch ${job.name}: ${e.message}`);
    record({ at: Date.now(), key: job.key, name: job.name, reason: entry.reason, exit: null,
             error: `launch failed: ${e.message}` });
    return;
  }

  state.current = {
    runId, key: job.key, name: job.name, reason: entry.reason, startedAt: Date.now(),
    pid: null, pidFile, exitFile, logFile, sha, mode: mode(),
  };
  saveState();
  log(`started ${job.name}${sha ? ` at ${sha}` : ''} (${entry.reason})`);
  // The PID file appears a moment after the wrapper does; picking it up now
  // rather than on the next tick is what makes the Jobs page show a pid and
  // a Stop button straight away instead of "starting" for 20 seconds.
  const soon = setTimeout(() => { try { pollCurrent(); } catch (e) { /* the tick reports it */ } }, 1500);
  if (soon.unref) soon.unref();
}

// --- watching a run ---------------------------------------------------------
function finish(exit, note, finishedAt) {
  const run = state.current;
  if (!run) return;
  // A run that ended while the server was down is finalized on the next
  // start, so "now" would put its whole downtime into its duration. The exit
  // file's timestamp is when it actually ended.
  const ended = finishedAt || Date.now();
  const seconds = Math.round((ended - run.startedAt) / 1000);
  record({
    at: run.startedAt, finishedAt: ended, key: run.key, name: run.name, reason: run.reason,
    seconds, exit, sha: run.sha, pid: run.pid, note: note || null,
  });
  log(`${run.name} finished after ${humanDuration(seconds)}${exit === null ? ` (${note || 'outcome unknown'})` : ` with exit code ${exit}`}`);
  [run.pidFile, run.exitFile].forEach((file) => { try { fs.unlinkSync(file); } catch (e) { /* already gone */ } });
  state.current = null;
  saveState();

  // Saturday's predictor finishing is the weekly chain's trigger - the whole
  // point of the item: no fixed 14:00 slot, no idle hours.
  if (run.key === 'predictor') maybeTriggerChain(new Date(), 'the predictor finishing');
}

function pollCurrent() {
  const run = state.current;
  if (!run) return;

  let exit = null;
  let endedAt = null;
  try {
    // The wrapper's `echo "$status" > file` creates the file and writes to it
    // as two steps, so a poll can land on an empty one - and Number('') is 0,
    // which would record a killed or half-written run as a success. Only a
    // complete integer counts; anything else falls through to the liveness
    // check below and is reported honestly.
    exit = exitCodeFrom(fs.readFileSync(run.exitFile, 'utf-8'));
    if (exit !== null) endedAt = fs.statSync(run.exitFile).mtimeMs;
  } catch (e) { /* still running, or died without writing */ }
  if (exit !== null) return finish(exit, null, endedAt);

  if (!run.pid) {
    try {
      run.pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null;
      if (run.pid) saveState();
    } catch (e) { /* not written yet */ }
    // No PID and no exit code after the grace period means the launch itself
    // failed - setsid missing, bash missing, the log not writable.
    if (!run.pid && Date.now() - run.startedAt > PID_WAIT_MS) {
      return finish(null, 'the job never reported a PID - launch failed');
    }
    return;
  }

  if (!alive(run.pid, run.runId)) {
    // The process is gone and no exit file: killed (a reboot, an OOM, a
    // manual kill -9). Recorded as such rather than as a success.
    return finish(null, 'the job disappeared without writing an exit code (killed?)');
  }
}

// --- what is due ------------------------------------------------------------
// Pure, so test/jobs.test.js can walk a week through it without touching the
// filesystem: given the recorded occurrences and a moment in time, what
// should be started?
function dueAt(occurrences, now) {
  const due = [];

  const predictor = jobByKey('predictor');
  // The most recent slot, which before 09:00 is yesterday's - otherwise a
  // box that comes back at 04:00 would silently skip the day it missed
  // instead of catching it up.
  let slot = dailySlot(now, predictor.daily);
  if (slot > now) slot = new Date(slot.getTime() - 86400000);
  const late = now - slot;
  if (late >= 0 && late <= predictor.catchUpHours * 3600 * 1000) {
    const id = `predictor@${dayId(slot)}`;
    if (!occurrences[id]) {
      due.push({ plan: 'predictor', occurrence: id,
                 reason: late > 30 * 60 * 1000
                   ? `catch-up: the ${pad(predictor.daily.hour)}:${pad(predictor.daily.minute)} run was missed by ${humanDuration(late / 1000)}`
                   : `schedule ${pad(predictor.daily.hour)}:${pad(predictor.daily.minute)}` });
    }
  }

  // The chain is normally started by Saturday's predictor finishing. This is
  // only the safety net: the box was down, or that predictor never ran, and
  // the week would otherwise be skipped in silence - the exact failure this
  // item exists to remove.
  const anchor = lastWeekdaySlot(now, CHAIN_WEEKDAY, CHAIN_ANCHOR_HOUR, 0);
  const sinceAnchor = now - anchor;
  const chainId = `${CHAIN_PLAN}@${weekId(anchor)}`;
  if (!occurrences[chainId] && sinceAnchor >= CHAIN_CATCHUP_AFTER_MS && sinceAnchor <= CHAIN_CATCHUP_UNTIL_MS) {
    due.push({ plan: CHAIN_PLAN, occurrence: chainId,
               reason: "catch-up: this week's tuning chain has not run" });
  }
  return due;
}

// Which weekly occurrence a predictor finishing at `now` should start, or
// null for "not this one". Pure, so test/jobs.test.js can walk the Saturday
// edges - the ones that could otherwise run the chain twice in a weekend.
function chainTriggerAt(occurrences, now) {
  if (now.getDay() !== CHAIN_WEEKDAY) return null;
  const anchor = lastWeekdaySlot(now, CHAIN_WEEKDAY, CHAIN_ANCHOR_HOUR, 0);
  // Only a predictor finishing AFTER Saturday's own 09:00 slot is the
  // trigger. Before it - a Friday catch-up run spilling past midnight, a
  // hand-started run at 04:00, a restart finalizing a Friday-evening run -
  // the anchor is LAST Saturday, and firing then would both start a ten-hour
  // chain at 04:00 ahead of the day's predictions and consume the previous
  // week's occurrence id, leaving this week's free to fire the whole chain a
  // second time at 23:00. Recording nothing is right: the 09:00 predictor,
  // or the 23:00 safety net, still starts it exactly once.
  if (dayId(anchor) !== dayId(now)) return null;
  const id = `${CHAIN_PLAN}@${weekId(anchor)}`;
  return occurrences[id] ? null : id;
}

function maybeTriggerChain(now, why) {
  const id = chainTriggerAt(state.occurrences, now);
  if (id) enqueuePlan(CHAIN_PLAN, id, `triggered by ${why}`);
}

// A first start must not fire every slot it never saw: the state file is new,
// not the schedule. Everything currently inside its catch-up window is marked
// as already handled, and said so in the log.
function baseline(now) {
  const skipped = dueAt(state.occurrences, now);
  skipped.forEach((entry) => remember(entry.occurrence, 'baseline: first start of the scheduler'));
  state.baselinedAt = new Date().toISOString();
  saveState();
  if (skipped.length) {
    log(`first start: ${skipped.map((s) => s.occurrence).join(', ')} treated as already handled `
      + '(they belong to the crontab era, not to this schedule)');
  }
}

// --- the worker -------------------------------------------------------------
// Which queued entries the current mode may start. Dry mode starts only what
// an administrator asked for by hand; anything the schedule queued in an
// earlier 'on' session is dropped, so going back to dry is a real rollback
// and not a delayed start.
function allowedQueue(queue, current) {
  return current === 'on' ? queue.slice() : queue.filter((q) => q.manual);
}

function startNext() {
  if (state.current) return;
  const current = mode();
  if (current === 'off') return;
  // Dry mode starts nothing the schedule queued - including a queue written
  // in an earlier 'on' session and restored from scheduler.json. Those
  // entries are dropped rather than kept, so switching back to dry is a real
  // rollback and not a delayed start.
  const kept = allowedQueue(state.queue, current);
  if (kept.length !== state.queue.length) {
    state.queue.filter((q) => !kept.includes(q)).forEach((q) => log(
      `dropped ${jobByKey(q.key).name} from the queue: the schedule is ${current} and that entry was not started by hand`));
    state.queue = kept;
    saveState();
  }
  if (!state.queue.length) return;

  const holder = lockHolder();
  if (holder) {
    // Someone else's job - cron until cutover, or a run started by hand -
    // holds process.lock. Waiting is the whole improvement over the tuners,
    // which exited here and skipped the week. But a lock can also be held by
    // a run that hangs alive (one did for two days in August 2026), and then
    // waiting quietly would be its own silent failure: so the wait is
    // re-logged every half hour and, past the 6 h runHyperopt.sh used to give
    // up at, recorded as a blocked entry that shows on the Jobs page.
    const head = state.queue[0];
    const name = jobByKey(head.key).name;
    const now = Date.now();
    if (!waiting) waiting = { since: now, lastLogAt: 0, reported: false };
    if (now - waiting.lastLogAt >= WAIT_RELOG_MS) {
      waiting.lastLogAt = now;
      log(`waiting for process.lock (held by PID ${holder.pid}, ${holder.script}`
        + `${holder.key ? '' : ' - not a job this server started'}) before starting ${name}`
        + `${waiting.since !== now ? `; waiting ${humanDuration((now - waiting.since) / 1000)} so far` : ''}`);
    }
    if (!waiting.reported && now - waiting.since >= WAIT_ALARM_MS) {
      waiting.reported = true;
      const note = `blocked: process.lock held by PID ${holder.pid} (${holder.script}) for over `
        + `${humanDuration(WAIT_ALARM_MS / 1000)} - the job is still queued, nothing has been skipped`;
      record({ at: waiting.since, finishedAt: now, key: head.key, name, reason: head.reason,
               seconds: Math.round((now - waiting.since) / 1000), exit: null, blocked: true, note });
      log(note);
    }
    return;
  }
  waiting = null;
  const entry = state.queue.shift();
  saveState();
  launch(entry);
}

function tick() {
  try {
    pollCurrent();

    const now = new Date();
    if (mode() !== 'off') {
      dueAt(state.occurrences, now).forEach((entry) => enqueuePlan(entry.plan, entry.occurrence, entry.reason));
    }

    // "The predictor just finished" also has to be seen when the predictor
    // was not ours: during the dry-run weekend cron still starts it, and the
    // trigger has to be observed to be trusted. The lock is how we see it.
    const holder = lockHolder();
    if (lastLock && lastLock.key === 'predictor' && !holder && mode() !== 'off') {
      maybeTriggerChain(now, 'the predictor finishing (run started outside the scheduler)');
    }
    lastLock = holder;

    startNext();
  } catch (e) {
    log(`tick failed: ${e.stack || e.message}`);
  }
}

// --- start-up ---------------------------------------------------------------
function adoptOrClose() {
  const run = state.current;
  if (!run) return;
  let pid = run.pid;
  if (!pid) {
    try { pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null; } catch (e) { /* none */ }
    run.pid = pid;
  }
  if (alive(pid, run.runId)) {
    log(`adopted ${run.name} (pid ${pid}, started ${stamp(run.startedAt)}) - it survived the restart, as intended`);
    saveState();
    return;
  }
  pollCurrent();                       // reads the exit file if the run ended while we were down
  if (state.current) finish(null, 'the server was restarted and the job was gone when it came back');
}

function sweepRunFiles() {
  try {
    const cutoff = Date.now() - RUN_FILE_DAYS * 86400 * 1000;
    const keep = state.current ? state.current.runId : null;
    fs.readdirSync(RUN_DIR).forEach((name) => {
      if (keep && name.startsWith(keep)) return;      // the run we are about to adopt
      const file = path.join(RUN_DIR, name);
      try { if (fs.statSync(file).mtimeMs < cutoff) fs.unlinkSync(file); } catch (e) { /* gone */ }
    });
  } catch (e) { /* no run directory yet */ }
}

function start() {
  state = loadState();
  const current = mode();
  log(`schedule is ${current} (${modeReason()}) - ${current === 'on' ? 'jobs will be started'
    : current === 'dry' ? 'nothing will be started, only recorded' : 'disabled'}`);
  sweepRunFiles();
  adoptOrClose();
  if (!state.baselinedAt) baseline(new Date());
  rotateAll();
  tick();
  timer = setInterval(tick, TICK_MS);
  if (timer.unref) timer.unref();
}

// Only the timer stops. A running job is deliberately NOT killed: surviving a
// deploy is the reason it was launched through setsid in the first place, and
// the next start adopts it.
function stop() {
  if (timer) { clearInterval(timer); timer = null; }
}

// --- status and page --------------------------------------------------------
function nextDueText(job, now) {
  if (job.daily) return `${stamp(nextDaily(now, job.daily))} (daily ${pad(job.daily.hour)}:${pad(job.daily.minute)})`;
  return "when Saturday's predictor finishes";
}

function lastRunOf(key) {
  return state.history.find((h) => h.key === key && !h.dry && !h.blocked) || null;
}

function status() {
  const now = new Date();
  const holder = lockHolder();
  return {
    mode: mode(),
    modeReason: modeReason(),
    lock: holder,
    current: state.current ? Object.assign({}, state.current, {
      log: tail(state.current.logFile, 20),
    }) : null,
    queue: state.queue.map((q) => Object.assign({}, q, { name: jobByKey(q.key).name })),
    jobs: JOBS.map((job) => ({
      key: job.key, name: job.name, description: job.description,
      command: `python3 ${job.script}${job.args.length ? ` ${job.args.join(' ')}` : ''}`,
      logFile: path.join(LOG_DIR, job.log),
      next: nextDueText(job, now),
      last: lastRunOf(job.key),
    })),
    history: state.history.slice(0, 12),
  };
}

function tail(file, lines) {
  try {
    const stat = fs.statSync(file);
    const start = Math.max(0, stat.size - 8192);
    const fd = fs.openSync(file, 'r');
    const buffer = Buffer.alloc(stat.size - start);
    fs.readSync(fd, buffer, 0, buffer.length, start);
    fs.closeSync(fd);
    return buffer.toString('utf-8').split('\n').filter(Boolean).slice(-lines);
  } catch (e) {
    return [];
  }
}

const MODE_BADGE = {
  on: ['#27ae60', 'running the schedule'],
  dry: ['#f39c12', 'dry run - recording only'],
  off: ['#c0392b', 'disabled'],
};

// Rendered into the admin Jobs page next to the supervised services, which is
// where an operator already looks.
function section(req) {
  const esc = auth.escapeHtml;
  const s = status();
  const csrf = `<input type="hidden" name="_csrf" value="${esc(req.user.csrf)}">`;
  const [colour, label] = MODE_BADGE[s.mode];

  const running = s.current ? `
    <p><b>${esc(s.current.name)}</b> is running - pid ${s.current.pid || 'starting'}, ${since(s.current.startedAt)} so far,
       started because of ${esc(s.current.reason)}${s.current.sha ? ` at <code>${esc(s.current.sha)}</code>` : ''}.
       <form method="post" action="/admin/jobs/stop" class="inline-form" style="display:inline;">${csrf}
         <input type="hidden" name="runId" value="${esc(s.current.runId)}">
         <button type="submit" class="nav-btn" style="background:#c0392b; border-color:#a93226;">Stop it</button>
       </form></p>
    <pre style="background:#2c3e50; color:#ecf0f1; padding:12px; border-radius:6px; overflow-x:auto; max-height:200px; font-size:0.85em;">${esc(s.current.log.join('\n') || 'no output yet')}</pre>`
    : `<p style="color:#7f8c8d;">No scheduled job is running.${s.lock ? ` Something else holds <code>process.lock</code>: pid ${s.lock.pid}, ${esc(s.lock.script)}.` : ''}</p>`;

  const queue = s.queue.length ? `
    <h3 style="margin-bottom:6px;">Waiting</h3>
    <table style="min-width:0;"><tr><th style="text-align:left;">Job</th><th style="text-align:left;">Queued</th><th style="text-align:left;">Why</th><th></th></tr>
    ${s.queue.map((q) => `<tr>
      <td style="text-align:left;">${esc(q.name)}</td>
      <td style="text-align:left;">${since(q.queuedAt)} ago</td>
      <td style="text-align:left;">${esc(q.reason)}</td>
      <td><form method="post" action="/admin/jobs/cancel" class="inline-form">${csrf}
        <input type="hidden" name="key" value="${esc(q.key)}">
        <button type="submit" class="nav-btn">Cancel</button></form></td></tr>`).join('')}
    </table>` : '';

  const rows = s.jobs.map((job) => `
    <tr>
      <td style="text-align:left;"><b>${esc(job.name)}</b><br><span style="color:#7f8c8d; font-size:0.9em;">${esc(job.description)}</span></td>
      <td style="text-align:left;"><code>${esc(job.command)}</code></td>
      <td style="text-align:left;">${esc(job.next)}</td>
      <td style="text-align:left;">${job.last
        ? `${stamp(job.last.at)} · ${humanDuration(job.last.seconds)} · ${job.last.exit === 0
            ? '<span style="color:#27ae60;">ok</span>'
            : `<span style="color:#c0392b;">${job.last.exit === null ? esc(job.last.note || 'unknown') : `exit ${job.last.exit}`}</span>`}`
        : '<span style="color:#7f8c8d;">not yet under this schedule</span>'}</td>
      <td><form method="post" action="/admin/jobs/run" class="inline-form">${csrf}
        <input type="hidden" name="key" value="${esc(job.key)}">
        <button type="submit" class="nav-btn"${s.mode === 'off' ? ' disabled' : ''}>Run now</button></form></td>
    </tr>`).join('');

  const history = s.history.length ? `
    <h3 style="margin-bottom:6px;">Recent runs</h3>
    <table style="min-width:0;"><tr><th style="text-align:left;">When</th><th style="text-align:left;">Job</th><th style="text-align:left;">Took</th><th style="text-align:left;">Result</th><th style="text-align:left;">Why</th></tr>
    ${s.history.map((h) => `<tr>
      <td style="text-align:left;">${stamp(h.at)}</td>
      <td style="text-align:left;">${esc(h.name)}${h.sha ? ` <code style="color:#7f8c8d;">${esc(h.sha)}</code>` : ''}</td>
      <td style="text-align:left;">${h.dry ? '-' : humanDuration(h.seconds)}</td>
      <td style="text-align:left;">${h.dry ? '<span style="color:#f39c12;">dry run - not started</span>'
        : h.blocked ? `<span style="color:#f39c12;">waiting: ${esc(h.note || 'the lock is held')}</span>`
        : h.exit === 0 ? '<span style="color:#27ae60;">ok</span>'
        : `<span style="color:#c0392b;">${h.exit === null ? esc(h.note || h.error || 'unknown') : `exit ${h.exit}`}</span>`}</td>
      <td style="text-align:left;">${esc(h.reason || '')}</td></tr>`).join('')}
    </table>` : '';

  return `
    <div class="card expanded">
      <div class="card-header" onclick="toggleCard(this)">
        <div>
          <span class="card-title">Scheduled pipeline jobs</span>
          <span class="card-meta" style="margin-left:10px; color:${colour}; font-weight:bold;">${esc(label)}</span>
          ${s.current ? `<span class="card-meta" style="margin-left:10px;">${esc(s.current.name)} · ${since(s.current.startedAt)}</span>` : ''}
        </div>
        <div class="card-icon">▼</div>
      </div>
      <div class="card-body">
        <p style="color:#7f8c8d; margin-top:0;">The daily predictor and the weekly tuning chain, owned by this server instead of crontab
           (README roadmap item 8). One job runs at a time, behind the same <code>process.lock</code> every Python entry point takes;
           the chain starts when Saturday's predictor finishes; a job launched here survives a deploy because it is reparented with
           <code>setsid --fork</code>. Mode comes from ${esc(s.modeReason)} - create <code>config/scheduler.disabled</code> to stop the
           schedule without a restart.</p>
        ${s.mode === 'dry' ? '<p style="color:#f39c12;"><b>Dry run.</b> The schedule only records what it would have started, so it is safe next to the crontab entries. Set <code>SCHEDULER=on</code> in .env and remove them to cut over. <i>Run now</i> still really runs a job.</p>' : ''}
        ${running}
        ${queue}
        <h3 style="margin-bottom:6px;">The schedule</h3>
        <table style="min-width:0;"><tr><th style="text-align:left;">Job</th><th style="text-align:left;">Command</th><th style="text-align:left;">Next</th><th style="text-align:left;">Last run</th><th></th></tr>${rows}</table>
        ${history}
      </div>
    </div>`;
}

// --- routes -----------------------------------------------------------------
function install(app, render) {
  const adminOnly = auth.requireAdmin(render);
  const back = (res, message) => res.redirect('/admin/jobs?msg=' + encodeURIComponent(message));

  app.post('/admin/jobs/run', adminOnly, (req, res) => {
    if (!auth.csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    const job = jobByKey(String((req.body || {}).key || ''));
    if (!job) return back(res, 'unknown job');
    if (mode() === 'off') return back(res, 'the schedule is disabled - remove config/scheduler.disabled first');
    const queued = enqueue(job.key, `started by hand (${req.user.name})`, true);
    saveState();
    startNext();
    return back(res, queued ? `${job.name}: queued` : `${job.name}: already running or queued`);
  });

  app.post('/admin/jobs/cancel', adminOnly, (req, res) => {
    if (!auth.csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    const key = String((req.body || {}).key || '');
    const before = state.queue.length;
    state.queue = state.queue.filter((q) => q.key !== key);
    saveState();
    if (before !== state.queue.length) log(`${jobByKey(key).name} removed from the queue by ${req.user.name}`);
    return back(res, before !== state.queue.length ? 'removed from the queue' : 'not in the queue');
  });

  app.post('/admin/jobs/stop', adminOnly, (req, res) => {
    if (!auth.csrfOk(req)) return res.status(403).send('Invalid form token - reload the page and try again.');
    const run = state.current;
    if (!run || run.runId !== String((req.body || {}).runId || '')) return back(res, 'that job is no longer running');
    if (!run.pid) return back(res, 'that job has not reported its PID yet');
    if (!alive(run.pid, run.runId)) return back(res, 'that job is already gone - the page will catch up in a moment');
    try {
      // Negative PID: the whole process group setsid created, so the Python
      // process goes with the wrapper instead of being left behind.
      process.kill(-run.pid, 'SIGTERM');
      log(`${run.name} (pid ${run.pid}) stopped by ${req.user.name}`);
      return back(res, `${run.name}: stopping`);
    } catch (e) {
      return back(res, `could not stop it: ${e.message}`);
    }
  });
}

module.exports = {
  install, start, stop, status, section, JOBS,
  // Pure schedule logic for test/jobs.test.js, plus the few entry points the
  // launch/adoption test drives directly (there is no other way to exercise
  // a reparented job without a browser and a real 12-hour tuner).
  _internals: {
    dueAt, weekId, dayId, dailySlot, nextDaily, lastWeekdaySlot, CHAIN_KEYS, CHAIN_PLAN,
    enqueue, startNext, tick, lockHolder, maybeTriggerChain, state: () => state,
    chainTriggerAt, exitCodeFrom, allowedQueue,
  },
};
