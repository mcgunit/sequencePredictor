// pipeline.js - is the pipeline busy right now? Moved out of server.js on 24 Oct 2026 (the old pages' banner); the
// app's Home and Lottery read it (lottery.js: `running` on /api/home and /api/lottery).
//
// Every Python entry point (Predictor.py, the five tuners, TrainMetaLearner) takes this one PID lock file in the repo
// root, so it is the cheapest honest answer to "is a run in progress" - no scheduler needed, and it stays true when a
// job is started by hand or by cron. /proc gives the command line, so the banner can say which job it is rather than
// just "busy".
'use strict';

const fs = require('fs');
const path = require('path');
const jobs = require('./jobs');

const PIPELINE_LOCK = path.join(__dirname, 'process.lock');
const JOB_NAMES = {
  'Predictor.py': 'Today\'s predictions are being computed',
  'HyperoptStatistics.py': 'Weekly tuning: statistical models',
  'HyperoptBoost.py': 'Weekly tuning: boosting models',
  'HyperoptRLTicket.py': 'Weekly tuning: RL ticket model',
  'HyperoptEnsemble.py': 'Weekly tuning: ensemble subsets',
  'HyperoptQuantum.py': 'Weekly tuning: quantum meta-learners',
  'HyperoptDeepLearning.py': 'Tuning: deep learning models',
  'TrainMetaLearner.py': 'Retraining the meta-learner',
  'RandomnessDiscrimination.py': 'Weekly controls: randomness discrimination',
  'NullControls.py': 'Weekly controls: null histories',
  'IrrelevantFeatureControl.py': 'Weekly controls: irrelevant-feature control',
  'MarketsDaily.py': 'Refreshing the market games (bars and bins)',
};
// A step that runs another step's script, named by its job: the script and its arguments, as jobs.js reads the lock
// (jobOfCmdline). Since 26 Oct 2026 the keno sub-selection step runs HyperoptStatistics.py -g keno -s KenoSubsetTuning,
// which the script alone named "Weekly tuning: statistical models".
const STEP_NAMES = {
  kenoSubsetTuning: 'Weekly tuning: keno sub-selections',
};
const readProcCmdline = (pid) => fs.readFileSync(`/proc/${pid}/cmdline`, 'utf-8');

// {pid, since (ms, the lock's modification time), what} of the job holding the lock, or null when nothing runs.
// lockFile and readCmdline are the tests' (a lock in a temp dir, a command line without /proc); the server passes none.
function pipelineStatus(lockFile = PIPELINE_LOCK, readCmdline = readProcCmdline) {
  let pid;
  let since = null;
  try {
    pid = Number(fs.readFileSync(lockFile, 'utf-8').trim());
    since = fs.statSync(lockFile).mtimeMs;
  } catch (e) {
    return null;                       // no lock: nothing running
  }
  if (!pid || Number.isNaN(pid)) return null;
  let cmdline;
  try {
    cmdline = readCmdline(pid).replace(/\0/g, ' ').trim();
  } catch (e) {
    return null;                       // stale lock of a dead run - the scripts clean it up themselves
  }
  const step = jobs.jobOfCmdline(cmdline);
  const script = Object.keys(JOB_NAMES).find((name) => cmdline.includes(name));
  const what = (step && STEP_NAMES[step.key]) || (script ? JOB_NAMES[script] : 'A pipeline job is running');
  return { pid, since, what };
}

// What every reader of the app gets: the job in the banner's words and since when (ISO, to the second) - never its pid
// or the lock's path. null when nothing runs.
function runningNow(lockFile = PIPELINE_LOCK, readCmdline = readProcCmdline) {
  const status = pipelineStatus(lockFile, readCmdline);
  if (!status) return null;
  return { what: status.what, since: new Date(status.since).toISOString().replace(/\.\d{3}Z$/, 'Z') };
}

module.exports = { PIPELINE_LOCK, JOB_NAMES, STEP_NAMES, pipelineStatus, runningNow };
