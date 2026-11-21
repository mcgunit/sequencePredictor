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
// 160 since 16 Nov 2026: with a job per lottery game nine daily jobs run, and 40 runs held under five days of them
const HISTORY_KEPT = 160;
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
// One entry per step - a Python entry point and its arguments - mirroring
// runPredictor.sh and runHyperopt.sh exactly: same commands, same log files,
// same order. The chain order is load-bearing and documented there: every
// tuner takes the shared process.lock, HyperoptQuantum.py must run BEFORE
// TrainMetaLearner.py so the weekly retrain sees freshly tuned quantum
// parameters, and the keno sub-selection tuning (HyperoptStatistics.py -g
// keno -s KenoSubsetTuning) runs AFTER it, last: its day table scores the
// artifacts the retrain writes (26 Oct 2026 - inside the statistical job it
// tuned on last weekend's, which the retrain then replaced). Of the steps it
// used to precede, only HyperoptEnsemble.py reads what it writes (the
// weighted vote's mode and temperature, its fallback for the SubsetEnsemble
// row's keno slices), and reads it a weekend older now - writing what it
// scored with as the row's own keys, so the row is served as it was gated
// rather than with the values this step writes after it. The control
// experiments (README "Null controls") are measurements of the rows, never
// inputs to them: their own plan, started only by Sunday's predictor
// finishing (or the Sunday 23:00 catch-up net), so they are never in the
// chain and never hold up Sunday's predictions. Same FIFO rule as the chain:
// a control run that overruns into Monday 09:00 queues that day's predictor
// behind it.
const CHAIN_PLAN = 'weeklyTuning';
const CONTROLS_PLAN = 'weeklyControls';
// Two weekly plans, each a run of jobs started by the predictor finishing on
// its weekday (after that day's 09:00 slot), and caught up by the safety net
// in dueAt when that never happens. The tuning chain owns Saturday; the four
// control experiments (README "Null controls") own Sunday - the null band only
// the month's first since 1 Nov 2026 (JOBS monthly) - so a chain that
// overruns into Sunday morning still lets Sunday's predictions go first - the
// queue stays the agreed FIFO, and the controls only ever start behind a
// finished predictor.
const PLANS = {
  [CHAIN_PLAN]: {
    name: 'Weekly tuning chain', label: "this week's tuning chain",
    weekday: 6, anchorHour: 6,                       // Saturday, its lottery predictor's slot (06:00; 03:00 from 8 to 9 Nov 2026)
    catchUpAfterMs: 17 * 3600 * 1000,                // Saturday 23:00
    catchUpUntilMs: 51 * 3600 * 1000,                // Monday 09:00
  },
  [CONTROLS_PLAN]: {
    name: 'Weekly control experiments', label: "this week's control experiments",
    weekday: 0, anchorHour: 6,                       // Sunday, its lottery predictor's slot (06:00; 03:00 from 8 to 9 Nov 2026)
    catchUpAfterMs: 17 * 3600 * 1000,                // Sunday 23:00
    catchUpUntilMs: 51 * 3600 * 1000,                // Tuesday 09:00
  },
};

// The games of the three daily runs (Predictor.py -g), the schedule of 8 Nov 2026: crypto right after its UTC
// day closes, the lottery games in the early morning (06:00 Brussels since 9 Nov 2026: at 00:30 UTC the Nationale
// Loterij API answered every game with an empty reply), the shares in the evening once Euronext's day is complete.
const CRYPTO_GAMES = 'crypto,cryptoweek,cryptomonth';   // the month game since 18 Nov 2026 (roadmap item 11)
const NIGHT_GAMES = 'euromillions,lotto,eurodreams,jokerplus,keno,pick3,vikinglotto';
const EVENING_GAMES = 'shares,sharesweek,sharesmonth';

// A predictor job per lottery game (16 Nov 2026; until then one run made all seven, key 'predictor'): a game that
// crashes, hangs or gets no data costs only its own run - the others have theirs, each with its own log, its own
// retries and its own push. All seven are due at 06:00 Brussels and run one after the other in this order (the
// queue is FIFO); the last of them finishing starts the Saturday tuning chain and the Sunday controls (see
// lotteryPending). The crypto and the stocks run stay one job each: a base game and its week game share their bars.
const LOTTERY = 'lottery';
const LOTTERY_GAMES = [
  ['euromillions', 'EuroMillions'], ['lotto', 'Lotto'], ['eurodreams', 'EuroDreams'], ['jokerplus', 'Joker+'],
  ['keno', 'Keno'], ['pick3', 'Pick3'], ['vikinglotto', 'Vikinglotto'],
];
// The key the one run of all seven had: still in the kept history, and in a state file written before the split.
const LEGACY_LOTTERY_KEY = 'predictor';
// The one job of all seven games as it was until 16 Nov 2026, for the pages only (never queued, not in JOBS): its
// kept runs keep their name and usual times, and its log stays readable from the Jobs page.
const LEGACY_JOBS = [{
  key: LEGACY_LOTTERY_KEY, group: LOTTERY, legacy: true, name: 'Morning predictor: lottery games',
  description: 'The one run of all seven lottery games, until 16 Nov 2026. Since then each game has its own job.',
  script: 'Predictor.py', args: ['-a', 'true', '-g', LOTTERY_GAMES.map(([game]) => game).join(',')], log: 'predictor.log',
  daily: { hour: 6, minute: 0 },
}];
function lotteryJob([game, label]) {
  return {
    key: `${game}Predictor`,
    group: LOTTERY,
    game,
    name: `Morning predictor: ${label}`,
    description: `The next draw's predictions for ${label}, deep learning rows included and time-boxed `
               + `(Predictor.py -a true -g ${game}). The seven lottery games each have their own job, all due at 06:00 Brussels `
               + 'and run one after the other, so a game that fails or hangs costs only its own run; the morning is usually '
               + 'done by about 10:30, before the first sales close (13:00 on Sundays for Keno, Pick3 and Joker+). A run the '
               + 'Nationale Loterij API gives no draws ends with status 74 and is started again an hour later, up to six times. '
               + 'A run far longer than usual is stopped and the next game starts. The last of the seven finishing starts the '
               + 'Saturday tuning chain and the Sunday controls. One run made all seven games until 16 Nov 2026.',
    script: 'Predictor.py',
    args: ['-a', 'true', '-g', game],
    log: `${game}Predictor.log`,
    // 06:00 Brussels since 9 Nov 2026: at 00:30 UTC the Nationale Loterij API answered every game with an empty reply
    daily: { hour: 6, minute: 0 },
    // A run without data is a failed run (16 Nov 2026; on 9 and 16 Nov the API answered every game with a web page
    // and the run ended with status 0, the day's predictions not made): Predictor.py ends with status 74
    // (src/DataFetcher.py NO_DATA_EXIT) when the fetch failed, and the run is started again an hour after it ended,
    // up to six times (see "a run without data" below).
    noData: { exit: 74, retries: 6, everyMs: 3600 * 1000 },
    // A hung game must not hold the six behind it (the decision of 16 Nov 2026): see "far longer than usual".
    stopWhenSlow: true,
    // How long after its slot a missed run is still worth starting. The box
    // being down over a night is the case this exists for; past 20 h (02:00)
    // the next morning's run is close enough to supersede it.
    catchUpHours: 20,
  };
}

const JOBS = [
  {
    // First in this list: a catch-up after a night the box was down queues crypto ahead of the lottery games.
    key: 'cryptoPredictor',
    name: 'Night predictor: crypto',
    description: "The crypto day's predictions (Predictor.py -a true -g crypto,cryptoweek,cryptomonth) at 00:05 UTC, five minutes after the "
               + 'UTC day closes: the daily book buys at 00:00 UTC, so the card is up minutes after it instead of hours (01:05 Brussels '
               + 'in winter, 02:05 in summer; until 8 Nov 2026 it came with the 09:00 run). A crypto week closes on Monday 00:00 UTC, '
               + 'so Monday\'s run brings the new week too.',
    script: 'Predictor.py',
    args: ['-a', 'true', '-g', CRYPTO_GAMES],
    log: 'cryptoPredictor.log',
    daily: { hour: 0, minute: 5, utc: true },
    catchUpHours: 20,
  },
  ...LOTTERY_GAMES.map(lotteryJob),
  {
    // The Council rows (docs/roadmap.md item 12, 17 Nov 2026): tickets from the Council's local LLMs, added to each
    // game's day file as rows. A job of its own behind the lottery predictors (`after`: queued when the last of
    // them ends, ahead of the weekly plans), so a slow or silent LLM costs no game its predictions. The models run
    // on the DL360, which is switched on first (`wake`) and switches itself off when idle again, and since
    // 20 Nov 2026 on the GPU worker too (always on; the row models' own config, src/llmCouncil/rows.json).
    key: 'councilRows',
    name: 'Council rows',
    description: 'Asks the Council\'s local LLMs for a ticket for every lottery game\'s next draw and adds the answers to the '
               + 'day files as rows, scored like every other row (CouncilRows.py). The row models are a set of their own '
               + '(src/llmCouncil/rows.json), on the GPU worker and the DL360. Two kinds: from the draws alone, one row per '
               + 'member and one that combines them; from the other rows\' recent record, one row. It starts '
               + 'when the morning\'s lottery predictors have ended, switches the DL360 on first, and skips a game whose '
               + 'day file already holds the rows. A member on the GPU worker is left out while a hyperopt or backtest runs there. An answer that is no valid ticket is asked again twice; after that the '
               + 'row is absent that day. Status 75: no Council model answered.',
    script: 'CouncilRows.py', args: [], log: 'councilRows.log',
    after: LOTTERY, wake: 'cpu', unreachableExit: 75,
  },
  {
    // The decision of 31 Oct 2026: a shares card must be up before Euronext opens at 09:00, or its
    // open-to-close book is one nobody could have followed. Euronext's day is complete after 18:00 Brussels
    // (src/MarketData.py EURONEXT_READY), so the run is at 18:05 (18:30 until 8 Nov 2026) and the card is up the
    // evening before.
    key: 'sharesPredictor',
    name: 'Evening predictor: stocks',
    description: "Tomorrow's stock predictions, from the Euronext day that just closed (Predictor.py -a true -g shares,sharesweek,sharesmonth), "
               + 'Monday to Friday at 18:05 Brussels, as soon as the day is complete (18:00), so the card is up before the 09:00 open. '
               + 'Waits for process.lock like any job: a tuner or control holding it delays the card.',
    script: 'Predictor.py',
    args: ['-a', 'true', '-g', EVENING_GAMES],
    log: 'sharesPredictor.log',
    daily: { hour: 18, minute: 5, weekdays: [1, 2, 3, 4, 5] },
    // until 08:30 the next morning: a card started later is no longer up before the open (14 h 25 min after 18:05)
    catchUpHours: 14 + 25 / 60,
  },
  {
    key: 'hyperoptStatistics',
    name: 'Weekly tuning: statistical models',
    description: 'Tunes the statistical rows into bestParams_<game>.json (HyperoptStatistics.py).',
    script: 'HyperoptStatistics.py', args: [], log: 'hyperoptStatistics.log', plan: CHAIN_PLAN,
  },
  {
    key: 'hyperoptBoost',
    name: 'Weekly tuning: boosting models',
    description: 'Tunes XGBoost Model into the same files (HyperoptBoost.py).',
    script: 'HyperoptBoost.py', args: [], log: 'hyperoptBoost.log', plan: CHAIN_PLAN,
  },
  {
    key: 'hyperoptRLTicket',
    name: 'Weekly tuning: RL ticket model',
    description: 'Tunes, through the champion/challenger gate, the RL Ticket Model of the lottery games and the RL Position Model of the four market games '
               + '(how much to put on the vote\'s call, scored as paper money) - pure numpy, minutes not hours (HyperoptRLTicket.py).',
    script: 'HyperoptRLTicket.py', args: [], log: 'hyperoptRLTicket.log', plan: CHAIN_PLAN,
  },
  {
    key: 'hyperoptEnsemble',
    name: 'Weekly tuning: ensemble subsets',
    description: 'Selects the rows SubsetEnsemble Model votes over, from the stored day JSONs, through the champion/challenger gate since 15 Oct 2026 (HyperoptEnsemble.py).',
    script: 'HyperoptEnsemble.py', args: [], log: 'hyperoptEnsemble.log', plan: CHAIN_PLAN,
  },
  {
    key: 'hyperoptQuantum',
    name: 'Weekly tuning: quantum meta-learners',
    description: 'Tunes the quantum-kernel SVC, the VQC and their classical RBF-SVM control, three Optuna '
               + 'studies per game, each through the champion/challenger gate since 14 Oct 2026. Must run before the retrain below, '
               + 'which is the whole point: fresh parameters, not week-old ones (HyperoptQuantum.py).',
    script: 'HyperoptQuantum.py', args: [], log: 'hyperoptQuantum.log', plan: CHAIN_PLAN,
  },
  {
    key: 'trainMetaLearner',
    name: 'Weekly retrain: meta-learner',
    description: 'Retrains the stacking meta-learner on the freshly tuned parameters (TrainMetaLearner.py). Only the keno '
               + 'sub-selection tuning comes after it, because it scores the artifacts written here.',
    script: 'TrainMetaLearner.py', args: [], log: 'TrainMetaLearner.log', plan: CHAIN_PLAN,
  },
  {
    key: 'kenoSubsetTuning',
    name: 'Weekly tuning: keno sub-selections',
    description: 'Tunes how the weighted vote and the five meta-learner rows cut their keno ticket into the 5-10-number '
               + 'sub-selections, through the champion/challenger gate. Last since 26 Oct 2026: it scores the artifacts the '
               + 'retrain has just written and the rows this chain tuned - inside the statistical job it tuned on last '
               + 'weekend\'s, which the retrain then replaced (HyperoptStatistics.py -g keno -s KenoSubsetTuning, about a quarter of an hour).',
    script: 'HyperoptStatistics.py', args: ['-g', 'keno', '-s', 'KenoSubsetTuning'], log: 'kenoSubsetTuning.log', plan: CHAIN_PLAN,
    // since: the day the step joined its plan (Brussels). This week (admin.js weekView) expects it only of the
    // weekends whose Saturday is that day or later - a weekend it falls inside is not expected to have run it - and
    // gates.js TUNERS carries the same day. Without it the weekend of 24 Oct 2026 would list the step as not run
    // until the next chain.
    since: '2026-10-26',
  },
  {
    key: 'nullControls',
    name: 'Monthly controls: null histories (Q0 band, context)',
    description: 'Scores every tracked row on fair synthetic and shuffled histories, three of each (five for the market games, a fresh block of seeds every run): '
               + 'what the best of the rows scores on nothing, the null band (NullControls.py, hours: a full backtest per history, which is '
               + 'why the controls have their own Sunday plan). Context only since 27 Oct 2026, in the Controls drawer: Q0 is the daily '
               + "report's row test, and nothing is greyed or counted on the band. Monthly since 1 Nov 2026 (the decision of 30 Oct 2026): "
               + 'on the first Sunday of the month only, as the first job of that Sunday\'s controls - the three weekly controls follow it that day; '
               + 'on the other Sundays the band of the last first Sunday stays on the page, with its date.',
    script: 'NullControls.py', args: ['-g', 'lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek', '-m', 'both', '-n', '3', '--market-repetitions', '5', '--seed', 'weekly', '-d', '120'],
    log: 'nullControls.log', plan: CONTROLS_PLAN,
    // monthly: queued only by the occurrence of its plan whose day (the plan's weekday, Brussels) is the first of that
    // weekday in the month - day 1 to 7 - from `since` on; before it, every week. enqueuePlan honours it (planKeysOn),
    // and This week (admin.js weekView) expects the job only of such a weekend.
    monthly: { since: '2026-11-01' },
  },
  {
    key: 'randomnessDiscrimination',
    name: 'Weekly controls: randomness discrimination (Q2)',
    description: 'Can a classifier suite tell windows of real draws from fair simulated ones better than it tells two fair '
               + 'histories apart? Held to a null band and a shuffled control; the verdict on the Controls page and the Randomness watch '
               + '(RandomnessDiscrimination.py; for the market games (crypto, shares and their week games) the fair history is a matched '
               + 'geometric random walk; twenty repetitions for every game on a fresh block of seeds every week, about 80 minutes, since '
               + '1 Nov 2026 - three, five for the markets, left the band to the luck of the seeds; "too few draws" under the held-out '
               + 'windows of controlRules.json, fixed by the history before the run).',
    script: 'RandomnessDiscrimination.py', args: ['-g', 'lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek', '-w', '10', '-n', '20', '--seed', 'weekly'],
    log: 'randomnessDiscrimination.log', plan: CONTROLS_PLAN,
  },
  {
    key: 'featureControl',
    name: 'Weekly controls: irrelevant-feature control (meta-learners)',
    description: 'Refits every served meta-learner variant on the table the Saturday chain cached, with shuffled-copy noise '
               + 'columns appended: does any meta-learner give a provably irrelevant column stable importance, and which base '
               + 'models matter more than noise does? Verdicts per game and variant on the Controls page '
               + '(IrrelevantFeatureControl.py, refits only - no backtest - about an hour for the seven lottery games together, '
               + 'almost all of it the VQC refits: one keno-sized VQC fit takes two to three minutes and each game gets four; the four market games (crypto, shares and their week games) '
               + 'join once the Saturday chain has cached their table - the week games from the chain of 10 Oct 2026; the noise columns are drawn from a fresh seed every week).',
    script: 'IrrelevantFeatureControl.py', args: ['-g', 'lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek', '-r', '3', '-k', '3', '--seed', 'weekly'],
    log: 'featureControl.log', plan: CONTROLS_PLAN,
  },
  {
    key: 'marketRows',
    name: 'Weekly controls: market rows under a proper score',
    description: 'Backtests the market rows - GARCH, the Regime HMM and its two ablations - with the positional base rows over the '
               + 'newest 250 game days of crypto, shares and their week games, and scores every row by the mean log-score of the probability it gave '
               + 'the bin that then happened, with a paired bootstrap interval against GARCH: the reading that decides whether regimes '
               + 'add anything beyond volatility, before any hit rate or paper P&L is read (MarketRows.py, about ten minutes per '
               + 'market; the verdicts on the Controls page).',
    script: 'MarketRows.py', args: ['-g', 'crypto,shares,cryptoweek,sharesweek', '-d', '250'], log: 'marketRows.log', plan: CONTROLS_PLAN,
  },
  {
    // The roadmap step 4 (8 Nov 2026): a backtest of one model on one game, started by an administrator from
    // Admin › Parameters (backtests.js). Never scheduled and never "Run now": each queue entry carries its own id
    // (RunBacktest.py --id), several can wait, and the scheduled jobs always go first (startNext). A backtest only
    // starts when it has at least BACKTEST_MIN_SECONDS before the next scheduled run, and gets --budget to end
    // BACKTEST_MARGIN_SECONDS before it (at most BACKTEST_MAX_SECONDS).
    key: 'backtest',
    name: 'Backtest (started from Admin › Parameters)',
    description: 'One model on one game, walked forward over the newest draws with the weekend tuner\'s own score, the lockbox days '
               + 'withheld (RunBacktest.py). Started from Admin › Parameters; queued behind every scheduled job, and only started '
               + 'when it can finish well before the next scheduled run.',
    script: 'RunBacktest.py', args: [], log: 'backtest.log', manualOnly: true,
  },
];
const BACKTEST_MAX_SECONDS = 3 * 3600;
const BACKTEST_MARGIN_SECONDS = 10 * 60;
const BACKTEST_MIN_SECONDS = 15 * 60;

const planKeys = (plan) => JOBS.filter((j) => j.plan === plan).map((j) => j.key);
const CHAIN_KEYS = planKeys(CHAIN_PLAN);
const CONTROL_KEYS = planKeys(CONTROLS_PLAN);

// A monthly job (JOBS monthly: {since}) runs on its plan's day only when that day is the first of its weekday in the
// month (day 1 to 7), from `since` on - every week before it. day: 'YYYY-MM-DD', the plan's weekday.
function runsOnDay(job, day) {
  if (!job || !job.monthly) return true;
  if (job.monthly.since && day < job.monthly.since) return true;
  return Number(String(day).slice(8, 10)) <= 7;
}
// The plan's day of an occurrence handled at `now` (its trigger on the day itself, or the catch-up net up to two
// days later): the most recent plan weekday at its anchor hour, as dueAt names the occurrence.
function planDayOf(plan, now) {
  const spec = PLANS[plan];
  return dayId(lastWeekdaySlot(now, spec.weekday, spec.anchorHour, 0));
}
// The plan's jobs an occurrence on `day` queues, and the monthly ones it leaves out (with the day each runs next).
function planKeysOn(plan, day) {
  const jobs = JOBS.filter((j) => j.plan === plan);
  return {
    keys: jobs.filter((j) => runsOnDay(j, day)).map((j) => j.key),
    skipped: jobs.filter((j) => !runsOnDay(j, day)).map((j) => ({ key: j.key, next: nextMonthlyDay(j, day) })),
  };
}
// The next day after `day` (same weekday, a week on at a time) a monthly job runs on.
function nextMonthlyDay(job, day) {
  const d = new Date(`${day}T12:00:00`);
  for (let i = 0; i < 6; i += 1) {
    d.setDate(d.getDate() + 7);
    if (runsOnDay(job, dayId(d))) return dayId(d);
  }
  return null;
}

function jobByKey(key) {
  return JOBS.find((j) => j.key === key) || null;
}
// A queue entry's or a run's name, also for a key this schedule no longer has (a state file from before a change).
const nameOf = (key) => (jobByKey(key) || {}).name || String(key);

// A lottery game's predictor run: one of the seven jobs, or the one run of all of them a state file from before
// 16 Nov 2026 may still hold as running.
function isLottery(key) {
  return key === LEGACY_LOTTERY_KEY || (jobByKey(key) || {}).group === LOTTERY;
}
// The lottery games' runs this morning still has to make, as job names: queued, running (but for `except`, the run
// that is finishing) or waiting for the next try of a run without data. The weekly plans start behind the last of
// them, not behind the first: the tuners read every game's history, and a chain queued early would hold the games
// still waiting behind ten hours of tuning.
function lotteryPending(except = null) {
  const keys = new Set();
  if (state.current && state.current !== except && isLottery(state.current.key)) keys.add(state.current.key);
  state.queue.forEach((q) => { if (isLottery(q.key)) keys.add(q.key); });
  Object.keys(state.retry).forEach((key) => { if (isLottery(key)) keys.add(key); });
  return [...keys].map(nameOf);
}
// The last lottery run of the morning can also go without finishing: removed from the queue, a retry dropped as hours
// late, a launch that fails. The weekly plans wait for "no lottery run left" and only a run's finishing looked, so
// they then waited for the 23:00 net (found in review, 16 Nov 2026). Each of those three asks here.
function lotteryLeft(key, why, now = new Date()) {
  if (!isLottery(key) || mode() === 'off' || lotteryPending().length) return;
  maybeTriggerChain(now, why);
}
// The jobs that follow the morning's lottery predictors (`after`; the Council rows, 17 Nov 2026): queued when no
// lottery game is running or queued any more. A game that waits for a retry is not waited for - its try may be
// hours away, and the job is queued again behind it, where it adds only that game's rows. Called before the weekly
// plans are, so it sits ahead of them in the queue. Only a run that ended queues it: a game taken out of the queue
// made nothing, and Run now is there for the games made before it.
function queueAfterLottery(except, why) {
  if (mode() === 'off') return [];
  if (state.current && state.current !== except && isLottery(state.current.key)) return [];
  if (state.queue.some((q) => isLottery(q.key))) return [];
  const queued = JOBS.filter((j) => j.after === LOTTERY).filter((j) => enqueue(j.key, `after ${why}`, false));
  if (queued.length) saveState();
  return queued.map((j) => j.key);
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

// A daily slot is Brussels (the box's local) time, or UTC with daily.utc (the night runs since 8 Nov 2026: crypto's
// day closes at 00:00 UTC whatever the season, and the lottery run must stay behind it in summer too).
function dailySlot(now, at) {
  if (at.utc) return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), at.hour, at.minute, 0, 0));
  return new Date(now.getFullYear(), now.getMonth(), now.getDate(), at.hour, at.minute, 0, 0);
}

// A day on: in UTC for a UTC slot (a local setDate would move it an hour across a clock change).
function shiftDays(slot, at, days) {
  if (at.utc) slot.setUTCDate(slot.getUTCDate() + days);
  else slot.setDate(slot.getDate() + days);
}

// A daily job with weekdays (JOBS daily.weekdays, the evening stocks run: Monday to Friday) runs on those days only.
const onWeekday = (slot, at) => !at.weekdays || at.weekdays.includes(at.utc ? slot.getUTCDay() : slot.getDay());

// How a daily slot reads: "18:05", or "00:05 UTC".
const slotText = (at) => `${pad(at.hour)}:${pad(at.minute)}${at.utc ? ' UTC' : ''}`;

function nextDaily(now, at) {
  const next = dailySlot(now, at);
  if (next <= now) shiftDays(next, at, 1);
  while (!onWeekday(next, at)) shiftDays(next, at, 1);
  return next;
}

// The most recent slot of a daily job that is not in the future (before its hour: yesterday's, or the last weekday's).
function lastDaily(now, at) {
  const slot = dailySlot(now, at);
  if (slot > now) shiftDays(slot, at, -1);
  while (!onWeekday(slot, at)) shiftDays(slot, at, -1);
  return slot;
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
      { cwd: ROOT, encoding: 'utf-8', timeout: 5000, stdio: ['ignore', 'pipe', 'ignore'] }).trim();
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
  // never the log a control on the DL360 is writing into now
  const busy = state && state.remote && state.remote.current ? state.remote.current.logFile : null;
  JOBS.forEach((job) => { if (path.join(LOG_DIR, job.log) !== busy) rotate(path.join(LOG_DIR, job.log)); });
  rotate(SCHEDULER_LOG);
}

// --- the shared lock --------------------------------------------------------
// The same lock pipeline.js reads for the app's run line: every Python entry point takes
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
  const job = jobOfCmdline(cmdline);
  return { pid, script: job ? job.script : cmdline.split(' ').slice(-1)[0], key: job ? job.key : null };
}

// The job a command line runs: the one whose script and arguments it carries in full, else the one that runs its
// script without arguments, else the first that names its script (a run started by hand with its own arguments).
// Two steps run HyperoptStatistics.py since 26 Oct 2026; matching on the script alone named the statistical job
// while the keno sub-selection step held the lock, and the Jobs page (system.js compares the keys) would have called
// the schedule's own run a lock "not started by the schedule".
function jobOfCmdline(cmdline) {
  const named = JOBS.filter((j) => cmdline.includes(j.script));
  // in full: its last argument ends there, so "-g lotto" is not read into a longer game list that starts with it
  const carries = (j) => { const full = [j.script, ...j.args].join(' '); return cmdline.endsWith(full) || cmdline.includes(`${full} `); };
  // a bare "Predictor.py" or one with its own game list (a hand run) is a lottery predictor, the kind whose finishing
  // starts the weekly plans
  return named.find((j) => j.args.length && carries(j))
    || named.find((j) => !j.args.length) || named.find((j) => j.group === LOTTERY) || named[0] || null;
}

// --- state ------------------------------------------------------------------
// Everything that has to survive a restart: which occurrences have been
// handled, what is queued, what is running, and the run history the Jobs
// page shows. In config/ because it is gitignored - data/ is committed by
// the pipeline itself.
// remote: the Sunday controls' own queue and run on the CPU worker (the DL360, since 11 Nov 2026; see "The Sunday
// controls on the DL360" below).
// retry: a run without data waiting for its next try, per job key ({attempt, of, at, since}); durations: the seconds
// of each job's last successful runs, for "far longer than usual" (both since 16 Nov 2026).
// stopped: the run the schedule stopped last, until its process group is gone ({pid, key, name, at, members,
// killedAt, reportedAt}; reapStopped).
const EMPTY = { version: 1, occurrences: {}, queue: [], current: null, history: [], baselinedAt: null, baselinedPlans: {}, lotterySplitAt: null,
  remote: { current: null, queue: [] }, retry: {}, durations: {}, stopped: null };

let state = null;
let timer = null;
let lastLock = null;          // for spotting "the predictor just finished"
let waiting = null;           // {since, lastLogAt, reported} while the lock blocks the queue

function loadState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    const out = Object.assign({}, EMPTY, parsed && typeof parsed === 'object' ? parsed : {});
    // a scheduler.json from before 11 Nov 2026 has no remote part; never share EMPTY's object
    const r = out.remote && typeof out.remote === 'object' ? out.remote : {};
    out.remote = { current: r.current || null, queue: Array.isArray(r.queue) ? r.queue : [] };
    const plain = (v) => (v && typeof v === 'object' && !Array.isArray(v) ? v : null);
    out.retry = plain(out.retry) || {};
    out.stopped = plain(out.stopped) && Number.isInteger(out.stopped.pid) && out.stopped.pid > 1 ? out.stopped : null;
    // a scheduler.json from before 16 Nov 2026 has no durations: the kept history gives the first ones
    out.durations = plain(parsed && parsed.durations) || durationsFrom(out.history);
    return out;
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
// FIFO and single-worker, which is what makes the scheduling decisions
// hold: a Saturday chain that overruns into Sunday 09:00 queues that
// day's predictor behind it rather than pre-empting the tuner. A job key is
// never queued twice - a second copy of today's predictor would compute the
// same day again.
// opts.local: queue it on this server even when it is a control and the DL360 is set up (the fallback);
// opts.occurrence: the plan occurrence that queued it (the controls on the DL360 share one copy of the data).
function enqueue(key, reason, manual, opts = {}) {
  const job = jobByKey(key);
  if (!job) return false;
  if (state.current && state.current.key === key) return false;
  if (state.queue.some((q) => q.key === key)) return false;
  if (remoteHas(key)) return false;
  if (!opts.local && toWorker(key)) {
    state.remote.queue.push({ key, reason, queuedAt: Date.now(), manual: Boolean(manual), occurrence: opts.occurrence || null });
    log(`queued ${job.name} for the DL360 (${reason})`);
    return true;
  }
  // `manual` marks the one thing dry mode is still allowed to start: a job an
  // administrator asked for by hand. Without the mark, a queue written while
  // the schedule was on would drain the next time the process came up in dry
  // mode - the opposite of what dry mode promises.
  // opts.retry: which retry of a run without data this is (launch carries it into the run)
  state.queue.push({ key, reason, queuedAt: Date.now(), manual: Boolean(manual), ...(opts.retry ? { retry: opts.retry } : {}) });
  log(`queued ${job.name} (${reason})`);
  return true;
}

// A backtest's queue entry: its own id and arguments, so several can wait (enqueue keeps one entry per job key).
function enqueueBacktest(id, reason) {
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(String(id || ''))) return false;
  if (state.queue.some((q) => q.backtestId === id) || (state.current && state.current.backtestId === id)) return false;
  state.queue.push({ key: 'backtest', backtestId: id, args: ['--id', id], reason, queuedAt: Date.now(), manual: true });
  log(`queued backtest ${id} (${reason})`);
  return true;
}

// Seconds from now to the next scheduled daily run (any daily job), the slot a backtest must end before.
function secondsToNextSlot(now = new Date()) {
  const next = JOBS.filter((j) => j.daily).map((j) => nextDaily(now, j.daily).getTime());
  return next.length ? Math.max(0, Math.round((Math.min(...next) - now.getTime()) / 1000)) : BACKTEST_MAX_SECONDS + BACKTEST_MARGIN_SECONDS;
}
// The budget a backtest starting now gets, or 0 when it should not start (too close to the next scheduled run).
function backtestBudget(now = new Date()) {
  const room = secondsToNextSlot(now) - BACKTEST_MARGIN_SECONDS;
  return room >= BACKTEST_MIN_SECONDS ? Math.min(BACKTEST_MAX_SECONDS, room) : 0;
}

// A plan is one occurrence that may expand into several queued jobs: the
// weekly tuning chain is seven of them and the control plan four (three on a
// Sunday that is not the month's first), kept in order by the FIFO queue.
// A monthly job (the null band since 1 Nov 2026) is left out of every occurrence but its month's first, and said so.
function enqueuePlan(plan, occurrenceId, reason, now = new Date()) {
  const on = PLANS[plan] ? planKeysOn(plan, planDayOf(plan, now)) : { keys: [plan], skipped: [] };
  const keys = on.keys;
  const left = on.skipped.map((x) => `${jobByKey(x.key).name}: monthly, not this one${x.next ? ` - next on ${x.next}` : ''}`).join('; ');
  const dry = mode() === 'dry';
  remember(occurrenceId, reason);
  if (left) log(`${occurrenceId} leaves out ${left}`);
  if (dry) {
    log(`DRY RUN: would start ${PLANS[plan] ? `${PLANS[plan].name} (${keys.length} jobs${left ? `; left out: ${left}` : ''})` : jobByKey(keys[0]).name} - ${reason}`);
    record({ at: Date.now(), key: plan, name: PLANS[plan] ? PLANS[plan].name : jobByKey(keys[0]).name,
             reason, dry: true, occurrence: occurrenceId });
    return;
  }
  const queued = keys.filter((key) => enqueue(key, reason, false, { occurrence: occurrenceId }));
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

// A job whose work is done by a worker machine's services, not on it (`wake`: the Council rows ask the LLMs on the
// DL360): the machine is switched on first, as worker/remote-control.sh does for a control - the same marked wake,
// so it switches itself off when idle again. A wake that fails is written in the job's log and the job still runs:
// it finds out itself whether the services answer. Without that worker set up, nothing is woken.
function wakeLines(job, logFile) {
  if (!job.wake) return [];
  let c = null;
  let env = {};
  try { c = workerConfig(); env = c ? workerEnv(c) : {}; } catch (e) { c = null; }
  if (!c || !env.WORKER_ILO_ENV) return [];
  const vars = ['WORKER_HOST', 'WORKER_ILO_ENV', 'WORKER_IDLE_MINUTES'].filter((k) => env[k]).map((k) => `${k}=${shellQuote(env[k])}`).join(' ');
  return [`${vars} bash ${shellQuote(path.join(ROOT, 'worker', 'power.sh'))} wake >> ${shellQuote(logFile)} 2>&1 || echo 'the wake failed: going on' >> ${shellQuote(logFile)}`];
}

function launch(entry) {
  const job = jobByKey(entry.key);
  const logFile = path.join(LOG_DIR, job.log);
  fs.mkdirSync(LOG_DIR, { recursive: true });
  fs.mkdirSync(RUN_DIR, { recursive: true, mode: 0o700 });
  rotate(logFile);

  const runId = `${job.key}-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  // a backtest's entry carries its own arguments (its id) and gets the budget that ends it before the next scheduled run
  const args = entry.args ? [...entry.args, ...(job.key === 'backtest' ? ['--budget', String(entry.budget || backtestBudget())] : [])] : job.args;
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
    ...wakeLines(job, logFile),
    `python3 ${shellQuote(job.script)}${args.length ? ` ${args.map(shellQuote).join(' ')}` : ''} >> ${shellQuote(logFile)} 2>&1`,
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
    notify({ kind: 'job-failed', level: 'error', title: `${job.name} could not be started`, body: `The launch failed: ${e.message}`, link: JOBS_LINK });
    lotteryLeft(job.key, `${job.name} failing to start`);
    return;
  }

  // this run is the next try of whatever waited for one
  if (state.retry[job.key]) delete state.retry[job.key];

  state.current = {
    runId, key: job.key, name: job.name, reason: entry.reason, startedAt: Date.now(),
    pid: null, pidFile, exitFile, logFile, sha, mode: mode(), ...(entry.backtestId ? { backtestId: entry.backtestId } : {}),
    ...(entry.retry ? { retry: entry.retry } : {}),
  };
  saveState();
  log(`started ${job.name}${sha ? ` at ${sha}` : ''} (${entry.reason})`);
  // The PID file appears a moment after the wrapper does; picking it up now
  // rather than on the next tick is what makes the Jobs page show a pid and
  // a Stop button straight away instead of "starting" for 20 seconds.
  const soon = setTimeout(() => { try { pollCurrent(); } catch (e) { /* the tick reports it */ } }, 1500);
  if (soon.unref) soon.unref();
}

// --- notifications ----------------------------------------------------------
// What the administrator is told in the app (notifications.js, 16 Nov 2026).
// Loaded only when a note is raised and inside its own try, as readings.js
// is: a broken notifications.js cannot stop a job from being recorded or the
// next one from starting. setNotifier replaces it in the tests.
const JOBS_LINK = { href: '/admin/jobs', label: 'Jobs & services' };
let notifier = (note) => require('./notifications').raise(note);
function notify(note) {
  try { return notifier(note); } catch (e) {
    try { log(`could not raise a notification (${note && note.title}): ${e.message}`); } catch (x) { /* the log is a convenience */ }
    return null;
  }
}

const clock = (at) => { const d = new Date(at); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; };

// --- a run without data -------------------------------------------------------
// The decision of 16 Nov 2026. A job with a noData spec (the lottery
// predictor) that ends with that status got no draws for any game: it is
// started again everyMs after it ended, at most `retries` times, each try a
// queue entry like any other (it waits for the lock and for whatever runs).
// The pending try lives in state.retry, so it survives a restart; any start
// of the job - the next morning's slot, a run by hand - is that try and
// clears it. Answers null for any other run, else {note, waiting, gaveUp}:
// waiting while another try is planned.
function noDataOutcome(run, exit, ended) {
  const job = jobByKey(run.key);
  const spec = job && job.noData;
  if (!spec || exit !== spec.exit) return null;
  const done = run.retry || 0;
  if (done >= spec.retries) {
    delete state.retry[run.key];
    return { note: `no data from the lottery API on retry ${done} of ${spec.retries}: no further retry, the next scheduled run tries again`,
      waiting: false, gaveUp: true, done };
  }
  const at = ended + spec.everyMs;
  state.retry[run.key] = { attempt: done + 1, of: spec.retries, at, since: (state.retry[run.key] || {}).since || run.startedAt };
  return { note: `no data from the lottery API: tries again at ${clock(at)} (retry ${done + 1} of ${spec.retries})`, waiting: true, gaveUp: false, done, at };
}

// Queues the tries whose hour has come (tick). One that is hours late - the
// schedule was off, the server down - is dropped: the next slot is closer.
const RETRY_STALE_MS = 3 * 3600 * 1000;
function startRetries(now = Date.now()) {
  Object.keys(state.retry).forEach((key) => {
    const r = state.retry[key];
    const job = jobByKey(key);
    if (!job || !r || !Number.isFinite(r.at)) { delete state.retry[key]; saveState(); return; }
    if (now < r.at) return;
    delete state.retry[key];
    if (now - r.at > RETRY_STALE_MS) {
      log(`retry ${r.attempt} of ${r.of} of ${job.name} was due ${humanDuration((now - r.at) / 1000)} ago: dropped, the next scheduled run tries again`);
      saveState();
      lotteryLeft(key, `a late retry of ${job.name} being dropped`, new Date(now));
    } else if (!enqueue(key, `retry ${r.attempt} of ${r.of}: the run before got no data`, false, { retry: r.attempt })) {
      log(`retry ${r.attempt} of ${r.of} of ${job.name}: it is already queued or running - that run is the retry`);
    }
    saveState();
  });
}

// What a finished run of this server's queue tells the administrator.
// The seven lottery jobs share their notes about the API (dedupe per group, NO_DATA_NOTE_MS): an API that answers
// nobody is one note for the morning, not seven.
const NO_DATA_NOTE_MS = 6 * 3600 * 1000;
function notifyFinished(run, exit, note, again, seconds) {
  const job = jobByKey(run.key) || {};
  const once = (what) => (job.group ? { dedupe: `${what}:${job.group}`, dedupeMs: NO_DATA_NOTE_MS } : {});
  const shared = job.group ? ' The other lottery games that get no data are tried again the same way; this note is not repeated for them this morning.' : '';
  if (again) {
    // one note when the data first fails to come, one when the tries end; not one per try
    if (!again.done) {
      notify({ kind: 'no-data', level: 'warning', title: `${run.name}: no data from the lottery API`,
        body: [`The API gave no draws, so no prediction was made (the log has what it sent). The run is started again at ${clock(again.at)}, `
          + `and then every hour, up to ${job.noData.retries} times.${shared}`], link: JOBS_LINK, ...once('no-data') });
    } else if (again.gaveUp) {
      notify({ kind: 'no-data-ended', level: 'error', title: `${run.name}: still no data after ${again.done} retries`,
        body: [`Today's predictions for it were not made. Nothing further is tried until the next scheduled run; Run now on Jobs & services starts one by hand.${
          job.group ? ' This note is given once for the morning: Jobs & services shows each game\'s last run.' : ''}`],
        link: JOBS_LINK, ...once('no-data-ended') });
    }
    return;
  }
  if (exit === 0) {
    if (run.retry) {
      notify({ kind: 'no-data-ended', level: 'info', title: `${run.name}: the data arrived on retry ${run.retry}`,
        body: [`The run that got no data was started again and finished normally.${job.group ? ' Jobs & services shows the other games\' runs.' : ''}`],
        link: JOBS_LINK, ...once('no-data-arrived') });
    }
    return;
  }
  if (run.hungAt) {
    // the schedule stopped it (stopHung): the administrator did not, and has to know the game's day was not made
    notify({ kind: 'job-failed', level: 'error', title: `${run.name} was stopped: far longer than usual`,
      body: [`${hungText(run)}. Its predictions for today were not made; the queue moves on once its processes are gone. Run now on Jobs & services starts it again.`],
      link: JOBS_LINK });
    return;
  }
  if (run.stoppedBy) return;      // an administrator stopped it: they know
  if (job.unreachableExit !== undefined && exit === job.unreachableExit) {
    notify({ kind: 'job-failed', level: 'warning', title: `${run.name}: no Council model answered`,
      body: ['The LLMs on the DL360 did not answer (is the machine on, are its llama services up?). No Council rows were made for today\'s draws; '
        + 'the other rows are not affected. Run now on Jobs & services tries again; the job log says what it found.'],
      link: JOBS_LINK, dedupe: `unreachable:${run.key}`, dedupeMs: NO_DATA_NOTE_MS });
    return;
  }
  notify({ kind: 'job-failed', level: 'error',
    title: exit === null ? `${run.name}: ${note || 'the outcome is unknown'}` : `${run.name} ended with exit code ${exit}`,
    body: [`Started ${stamp(run.startedAt)} (${run.reason}), ran ${humanDuration(seconds)}.`], link: JOBS_LINK });
}

// --- far longer than usual ----------------------------------------------------
// state.durations keeps the seconds of each job's last DURATIONS_KEPT
// successful runs (a control on the DL360 under its own key: it is slower
// there). A run is "far longer than usual" once it has run 1.5 times the
// longest of them and at least half an hour more - the longest, not the
// median, because a predictor with nothing to compute ends in minutes and a
// full one takes hours. Said once per run, and only with SLOW_MIN_RUNS to
// compare with; a backtest has its own budget and is left out.
const DURATIONS_KEPT = 8;
const SLOW_MIN_RUNS = 3;
const SLOW_FACTOR = 1.5;
const SLOW_MARGIN_S = 30 * 60;
function keepDuration(key, seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return;
  state.durations[key] = [...(state.durations[key] || []), seconds].slice(-DURATIONS_KEPT);
}
function durationsFrom(history) {
  const out = {};
  (Array.isArray(history) ? history.slice().reverse() : []).forEach((h) => {
    if (!h || h.exit !== 0 || h.dry || h.blocked || !Number.isFinite(h.seconds)) return;
    const key = h.worker ? `${h.key}@${h.worker}` : h.key;
    out[key] = [...(out[key] || []), h.seconds].slice(-DURATIONS_KEPT);
  });
  return out;
}
function slowLimit(key) {
  const runs = state.durations[key] || [];
  if (runs.length < SLOW_MIN_RUNS) return null;
  const longest = Math.max(...runs);
  return { longest, limit: Math.max(longest * SLOW_FACTOR, longest + SLOW_MARGIN_S) };
}
function checkSlow(run, key, where, now = Date.now(), deps = {}) {
  if (!run || run.key === 'backtest') return false;
  const seconds = (now - run.startedAt) / 1000;
  const usual = run.slowAt ? null : slowLimit(key);
  const said = Boolean(usual) && seconds > usual.limit;
  if (said) {
    run.slowAt = now;
    saveState();
    const job = jobByKey(run.key) || {};
    log(`${run.name}${where} has run ${humanDuration(seconds)}: far longer than usual (the longest of its last runs took ${humanDuration(usual.longest)})`);
    notify({ kind: 'slow', level: 'warning', title: `${run.name}${where} runs far longer than usual`,
      body: [`It has run ${humanDuration(seconds)} since ${stamp(run.startedAt)}; the longest of its last ${state.durations[key].length} successful runs took ${humanDuration(usual.longest)}. `
        + `It is still running: the Jobs & services page shows its log and can stop it.${
          job.stopWhenSlow ? ` The schedule stops it itself once it has run ${humanDuration(stopLimit(key).limit)}.` : ''}`], link: JOBS_LINK });
  }
  stopHung(run, key, now, deps);
  return said;
}

// A job with stopWhenSlow (the seven lottery games, the decision of 16 Nov 2026) is not only reported: a hung game
// would hold the games behind it, so it is stopped and the queue moves on. The limit is wider than the notice's,
// because stopping a run that was only slow costs that game its day: twice the longest of its last successful runs
// and at least STOP_FLOOR_S, and only with a week of runs to compare with (a game's draw days take longer than its
// other days) - with fewer, a fixed STOP_FALLBACK_S, about what all seven games take together. The whole process
// group gets SIGTERM as from the Stop button, and SIGKILL when that has not ended it a minute later. The run is
// recorded as failed, with a notification; nothing starts it again but its next slot or Run now.
const STOP_MIN_RUNS = 7;
const STOP_FACTOR = 2;
const STOP_FLOOR_S = 2 * 3600;
const STOP_FALLBACK_S = 4 * 3600;
const STOP_KILL_AFTER_MS = 60 * 1000;
const STOPPED_BY_SCHEDULE = 'the schedule';
function stopLimit(key) {
  const runs = state.durations[key] || [];
  const longest = runs.length ? Math.max(...runs) : null;
  if (runs.length < STOP_MIN_RUNS) return { longest, limit: STOP_FALLBACK_S, runs: runs.length, fallback: true };
  return { longest, limit: Math.max(longest * STOP_FACTOR, STOP_FLOOR_S), runs: runs.length, fallback: false };
}
function hungText(run) {
  const h = run.hung || {};
  return `It had run ${humanDuration(h.seconds)} since ${stamp(run.startedAt)}; ${h.fallback
    ? `with fewer than ${STOP_MIN_RUNS} successful runs to compare with, a lottery game's run is stopped after ${humanDuration(h.limit)}`
    : `the longest of its last ${h.runs} successful runs took ${humanDuration(h.longest)}, and a run is stopped after ${humanDuration(h.limit)}`}`;
}
function stopHung(run, key, now = Date.now(), deps = {}) {
  const job = jobByKey(run.key);
  if (!job || !job.stopWhenSlow || run !== state.current || !run.pid || mode() === 'off') return false;
  if (run.hungAt) {
    if (!run.killedAt && now - run.hungAt >= STOP_KILL_AFTER_MS) {
      run.killedAt = now;
      saveState();
      try {
        (deps.kill || ((pid, signal) => process.kill(pid, signal)))(-run.pid, 'SIGKILL');
        log(`${run.name} (pid ${run.pid}) did not end on SIGTERM: SIGKILL`);
      } catch (e) {
        log(`${run.name} (pid ${run.pid}): SIGKILL failed (${e.message})`);
      }
    }
    return false;
  }
  const usual = stopLimit(key);
  const seconds = (now - run.startedAt) / 1000;
  if (seconds <= usual.limit) return false;
  run.hungAt = now;
  run.hung = { seconds: Math.round(seconds), limit: usual.limit, longest: usual.longest, runs: usual.runs, fallback: usual.fallback };
  // the group's processes as they are now: the ones looked for after the stop (reapStopped)
  run.hungMembers = (deps.list || groupList)(run.pid);
  saveState();
  const answer = stopRun(run.runId, STOPPED_BY_SCHEDULE, deps);
  log(`${run.name} has run ${humanDuration(seconds)}, past its limit of ${humanDuration(usual.limit)}: ${answer.ok ? 'stopped, the queue moves on' : `could not be stopped (${answer.message})`}`);
  return answer.ok;
}

// After the stop. SIGTERM to the group ends the wrapper at once, so the run is finished (pollCurrent: gone without an
// exit code) long before the minute is over and the SIGKILL above never came - while a process of the group that
// ignores SIGTERM keeps process.lock and holds the games behind it for hours (found in review, 16 Nov 2026). So the
// stopped run's process group is remembered (state.stopped, kept across a restart) and looked at on every tick:
// the queue waits while a process of it is left, SIGKILL goes to the group a minute after the stop, and when a
// process is still there a minute after that, a notification says so and the queue stops waiting (the lock rule
// still holds it when that process has the lock). A group id cannot be given out again while a process of the
// group lives, and only the processes that were in it at the stop count, so the kill cannot reach another job.
function groupList(pgid) {
  const out = [];
  let names;
  try { names = fs.readdirSync('/proc'); } catch (e) { return out; }
  names.forEach((name) => {
    if (!/^\d+$/.test(name)) return;
    try {
      const stat = fs.readFileSync(`/proc/${name}/stat`, 'utf-8');
      // pid (comm) state ppid pgrp session ... starttime is the 22nd field; comm may hold spaces and brackets
      const f = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
      if (Number(f[2]) === pgid && f[0] !== 'Z') out.push({ pid: Number(name), start: f[19] });
    } catch (e) { /* ended meanwhile */ }
  });
  return out;
}
// Those of the group's processes that were there at the stop (`known`, from groupList then): the same PID with the
// same start time - the kernel's own tick count, compared as written, since this container's clock of "since boot"
// is not the one those ticks count from.
function groupMembers(pgid, known) {
  const was = new Set([].concat(known || []).map((m) => `${m.pid}:${m.start}`));
  return groupList(pgid).filter((m) => was.has(`${m.pid}:${m.start}`)).map((m) => m.pid);
}

// What a stop in the middle of a write can leave in the checkout: git's index lock (a later game's commit fails on
// it, and its day files stay unpushed) and tracked files changed but not committed. Null when there is neither.
function checkoutAfterStop() {
  const lines = [];
  const lock = path.join(ROOT, '.git', 'index.lock');
  if (fs.existsSync(lock)) {
    lines.push('git\'s index lock (.git/index.lock) was left behind: the commits of the runs after it fail until it is removed. '
      + 'With no job running: rm .git/index.lock in the checkout.');
  }
  try {
    const changed = execFileSync('git', ['status', '--porcelain', '--untracked-files=no'],
      { cwd: ROOT, encoding: 'utf-8', timeout: 20000, stdio: ['ignore', 'pipe', 'ignore'], env: { ...childEnv(), GIT_OPTIONAL_LOCKS: '0' } })
      .split('\n').filter(Boolean).map((l) => l.slice(3));
    if (changed.length) {
      lines.push(`${changed.length} tracked file${changed.length === 1 ? ' has' : 's have'} changes that are not committed (${changed.slice(0, 4).join(', ')}${changed.length > 4 ? ', …' : ''}): `
        + 'a file written halfway is among them when the stop came during a write. The next run of that game writes its files again; git diff shows what is there now.');
    }
  } catch (e) { /* no git here, or it did not answer: nothing to say */ }
  return lines.length ? lines : null;
}

function reapStopped(now = Date.now(), deps = {}) {
  const s = state.stopped;
  if (!s) return null;
  const left = (deps.members || groupMembers)(s.pid, s.members);
  if (!left.length) {
    state.stopped = null;
    saveState();
    log(`${s.name} (pid ${s.pid}): its processes are gone${s.killedAt ? ' after SIGKILL' : ''}`);
    const found = (deps.checkout || checkoutAfterStop)();
    if (found) {
      log(`${s.name} was stopped during a write: ${found.join(' ')}`);
      notify({ kind: 'job-failed', level: 'warning', title: `${s.name} was stopped during a write: the checkout needs a look`, body: found, link: JOBS_LINK });
    }
    return 'gone';
  }
  if (mode() === 'off') return 'left';
  if (!s.killedAt) {
    if (now - s.at < STOP_KILL_AFTER_MS) return 'waiting';
    s.killedAt = now;
    saveState();
    try {
      (deps.kill || ((pid, signal) => process.kill(pid, signal)))(-s.pid, 'SIGKILL');
      log(`${s.name} (pid ${s.pid}): ${left.length} process${left.length === 1 ? '' : 'es'} did not end on SIGTERM (${left.join(', ')}): SIGKILL`);
    } catch (e) {
      log(`${s.name} (pid ${s.pid}): SIGKILL failed (${e.message})`);
    }
    return 'killed';
  }
  if (!s.reportedAt && now - s.killedAt >= STOP_KILL_AFTER_MS) {
    s.reportedAt = now;
    saveState();
    log(`${s.name} (pid ${s.pid}): ${left.join(', ')} still there after SIGKILL - the queue no longer waits for it`);
    notify({ kind: 'job-failed', level: 'error', title: `${s.name} was stopped but a process of it is still there`,
      body: [`Process${left.length === 1 ? '' : 'es'} ${left.join(', ')} did not end on SIGTERM or SIGKILL. While it holds process.lock the next jobs wait; end it by hand (kill -9).`],
      link: JOBS_LINK });
  }
  return 'left';
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
  // a run without data: its next try is planned before the run is recorded, so the record says when it is
  const again = noDataOutcome(run, exit, ended);
  record({
    at: run.startedAt, finishedAt: ended, key: run.key, name: run.name, reason: run.reason,
    seconds, exit, sha: run.sha, pid: run.pid,
    note: (run.hungAt && exit !== 0 ? `stopped by the schedule: far longer than usual. ${hungText(run)}` : note) || (again && again.note) || null,
    ...(run.retry ? { retry: run.retry } : {}), ...(run.stoppedBy ? { stoppedBy: run.stoppedBy } : {}),
  });
  if (exit === 0) keepDuration(run.key, seconds);
  log(`${run.name} finished after ${humanDuration(seconds)}${exit === null ? ` (${note || 'outcome unknown'})` : ` with exit code ${exit}`}`
    + `${again ? ` - ${again.note}` : ''}`);
  [run.pidFile, run.exitFile].forEach((file) => { try { fs.unlinkSync(file); } catch (e) { /* already gone */ } });
  state.current = null;
  // the schedule stopped it: its process group is watched until it is gone (reapStopped)
  if (run.hungAt && run.pid && exit !== 0) {
    state.stopped = { pid: run.pid, key: run.key, name: run.name, at: run.hungAt, members: run.hungMembers || [], killedAt: run.killedAt || null, reportedAt: null };
  }
  saveState();
  notifyFinished(run, exit, note, again, seconds);

  // Saturday's lottery predictions finishing is the weekly chain's trigger -
  // the whole point of the item: no fixed 14:00 slot, no idle hours. With a
  // job per game (16 Nov 2026) that is the last of them: not while another
  // game's run is queued or running, and not while a run without data waits
  // for its next try - the chain would tune on a history a draw short and
  // hold that try behind it for hours, so the last run's finishing is the
  // trigger (or, the retries used up, the last of them).
  if (isLottery(run.key)) {
    // this run's lock going is this finishing: tick() must not read it as a run started outside the scheduler
    lastLock = null;
    // not after a run without data: it made nothing to read
    if (exit !== ((jobByKey(run.key) || {}).noData || {}).exit) queueAfterLottery(run, 'the lottery predictions');
    const left = lotteryPending(run);
    if (left.length) log(`the weekly plans wait for ${left.length === 1 ? left[0] : `${left.length} lottery games' runs`}${again && again.waiting ? ' (a run with data)' : ''}`);
    else maybeTriggerChain(new Date(), 'the lottery predictions finishing');
  }
  // The Sunday controls' last job finishing writes the week's reading.
  scheduleReading(run);
}

// --- the week's reading -----------------------------------------------------
// When the Sunday plan's last job (the market rows report) finishes - whatever its exit code: a
// failed control leaves last week's record in place, and the reading names
// every part that is not from this week - the four controls are summarised
// into data/readings/<sunday>.json (readings.js), the admin pages' week-over-
// week history (21 Oct 2026, the decision). It runs after the run is
// recorded, on a later turn of the event loop, inside its own try: a reading
// that cannot be built or written is logged and changes nothing else - not
// this run's record, not the schedule, not the next job, which has already
// been started by then. readings.js is loaded only here, so even a broken
// readings.js or admin.js cannot stop this module from loading.
// The Q0 band is monthly since 1 Nov 2026 and runs first on its Sunday, so the
// plan's last job is one every Sunday runs: the reading is written every week,
// and on a first Sunday only after the band (hours) and the three weekly
// controls behind it. A week without a fresh band reads the latest, dated.
const READING_JOB = CONTROL_KEYS[CONTROL_KEYS.length - 1];

// The Sunday whose reading a finished run writes, as YYYY-MM-DD, or null: the
// controls plan's last job, started inside that plan's own window - Sunday
// 00:00 until its catch-up net closes on Tuesday 09:00 (local time, like the
// schedule). A band run started by hand later in the week writes nothing: it is
// not that weekend's control (the CLI can still write it, said as such).
function readingSundayOf(run) {
  if (!run || run.key !== READING_JOB) return null;
  const week = controlsWeekOf(run);
  return week && week.inWindow ? week.sunday : null;
}
// The Sunday a control run belongs to, and whether it started inside that
// plan's window; null without a start.
function controlsWeekOf(run) {
  if (!run || !Number.isFinite(run.startedAt)) return null;
  const spec = PLANS[CONTROLS_PLAN];
  const sunday = lastWeekdaySlot(new Date(run.startedAt), spec.weekday, 0, 0);
  const anchor = new Date(sunday);
  anchor.setHours(spec.anchorHour, 0, 0, 0);
  return { sunday: dayId(sunday), inWindow: run.startedAt < anchor.getTime() + spec.catchUpUntilMs };
}
// Why a control run that finished wrote no reading, with the command that
// rebuilds the week's - so nobody takes the stored reading for one that holds
// this run's record. root: where data/readings is.
function noReadingText(run, root = ROOT) {
  const week = controlsWeekOf(run);
  if (!week) return null;
  const name = run.name || (jobByKey(run.key) || {}).name || run.key;
  const rebuild = `to rebuild that week's reading: node readings.js --controls data/controls --sunday ${week.sunday} `
    + '--database <a folder holding that Sunday\'s modelPerformance.json> (never today\'s ranking)';
  if (!week.inWindow) return `${name} started after the controls' window (Tuesday 09:00): no reading includes it; ${rebuild}`;
  let stored = false;
  try { stored = fs.existsSync(path.join(root, 'data', 'readings', `${week.sunday}.json`)); } catch (e) { stored = false; }
  if (stored) return `${name} finished after the reading of Sunday ${week.sunday} was written: that reading keeps the record it found then; ${rebuild}`;
  return `${name} finished: the reading of Sunday ${week.sunday} is written when ${jobByKey(READING_JOB).name} finishes`;
}

// Builds and writes the reading for a finished run; answers what was written
// or null. Never throws. deps (for the test): readings, log, root.
function writeWeekReading(run, deps = {}) {
  const say = deps.log || log;
  let sunday = null;
  try {
    sunday = readingSundayOf(run);
    if (!sunday) return null;
    const readings = deps.readings || require('./readings');
    const root = deps.root || ROOT;
    const result = readings.writeReading(readings.buildReading({ root, sunday, by: 'scheduler' }), { root });
    say(`the week's reading for Sunday ${sunday}: ${result.written ? 'written' : 'unchanged'} (${path.relative(root, result.file)}, ${result.bytes} bytes)`);
    return result;
  } catch (e) {
    try { say(`could not write the week's reading${sunday ? ` for Sunday ${sunday}` : ''}: ${e.message} - nothing else is affected`); } catch (x) { /* the log is a convenience */ }
    (deps.notify || notify)({ kind: 'reading', level: 'error', title: `The week's reading${sunday ? ` for Sunday ${sunday}` : ''} was not written`,
      body: [`${e.message}. The controls' records are not affected${sunday ? `; to write it by hand: node readings.js --controls data/controls --sunday ${sunday}` : ''}.`],
      link: { href: '/admin/controls', label: 'Controls' }, dedupe: `reading:${sunday || 'unknown'}` });
    return null;
  }
}

// Called by finish() for every run; only the controls' last job, inside its
// window, schedules anything. Any other control run says in one log line why
// it wrote no reading (outside the window, or not the last control - then a
// reading already written for its Sunday keeps the record it found). deps.defer
// replaces setImmediate in the test.
function scheduleReading(run, deps = {}) {
  try {
    if (!readingSundayOf(run)) {
      if (run && CONTROL_KEYS.includes(run.key)) {
        const text = noReadingText(run, deps.root || ROOT);
        if (text) (deps.log || log)(text);
      }
      return false;
    }
    (deps.defer || setImmediate)(() => { writeWeekReading(run, deps); });
    return true;
  } catch (e) {
    return false;
  }
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

// --- the Sunday controls on the DL360 ---------------------------------------------------------------------------------
// The decision of 11 Nov 2026: when config/workers.json has a cpu entry (workers.js), the control plan's jobs
// run on the DL360 instead of in this queue - same trigger (Sunday's predictor finishing), same order, same records
// and the same week's reading, but this server's queue no longer waits behind them (the monthly Q0 band held the
// daily runs up to twelve hours). They have their own FIFO here (state.remote): one at a time, beside the scheduler's
// own run, taking no process.lock on this server. worker/remote-control.sh switches the DL360 on through its iLO
// when it does not answer, copies the code and the data a control reads to <dir>/controls (one copy per occurrence,
// so the four controls of a Sunday read the same draws), runs the control there over ssh and copies its records back
// into data/controls/ - this server stays the only writer of data/. When the DL360 does not come up (or cannot be
// copied to), the wrapper ends with REMOTE_NOT_STARTED before anything ran there, and this control and the ones
// behind it are queued here instead, as before 11 Nov 2026 (the fallback): no week's reading is lost.
const REMOTE_NOT_STARTED = 75;
let workerConfig = () => { try { return require('./workers').configOf('cpu'); } catch (e) { return null; } };
let workerEnv = (c) => require('./workers').scriptEnv(c);
const REMOTE_SCRIPT = path.join(ROOT, 'worker', 'remote-control.sh');
let remoteScript = REMOTE_SCRIPT;

// A control goes to the DL360 when it is set up; anything else stays here.
function toWorker(key) {
  return CONTROL_KEYS.includes(key) && Boolean(workerConfig());
}
function remoteHas(key) {
  const r = state.remote;
  return Boolean((r.current && r.current.key === key) || r.queue.some((q) => q.key === key));
}

function launchRemote(entry, c) {
  const job = jobByKey(entry.key);
  const logFile = path.join(LOG_DIR, job.log);
  fs.mkdirSync(LOG_DIR, { recursive: true });
  fs.mkdirSync(RUN_DIR, { recursive: true, mode: 0o700 });
  rotate(logFile);
  const runId = `${job.key}-dl360-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  const pidFile = path.join(RUN_DIR, `${runId}.pid`);
  const exitFile = path.join(RUN_DIR, `${runId}.exit`);
  const sha = gitSha();
  // one copy of the data per plan occurrence; a run by hand gets its own
  const snapshot = String(entry.occurrence || runId).replace(/[^A-Za-z0-9_.-]/g, '-').slice(0, 100);
  // the same markers as launch(): admin.js slices the job's log by them; the PID file holds the wrapper's PID (Stop
  // signals it alone - it stops the run on the DL360 and copies back what that wrote)
  const script = [
    `cd ${shellQuote(ROOT)} || exit 97`,
    `printf '=== scheduler: %s started %s UTC ===\n' ${shellQuote(job.name)} "$(date -u '+%F %T')${sha ? ` ${sha}` : ''}" >> ${shellQuote(logFile)}`,
    'started=$(date +%s)',
    `bash ${shellQuote(remoteScript)} ${shellQuote(runId)} ${shellQuote(snapshot)} ${shellQuote(job.script)} ${job.args.map(shellQuote).join(' ')} >> ${shellQuote(logFile)} 2>&1 &`,
    `echo $! > ${shellQuote(pidFile)}`,
    'wait $!',
    'status=$?',
    `printf '=== scheduler: %s finished status=%s after %ss ===\n' ${shellQuote(job.name)} "$status" "$(( $(date +%s) - started ))" >> ${shellQuote(logFile)}`,
    `echo "$status" > ${shellQuote(exitFile)}`,
    `rm -f ${shellQuote(pidFile)}`,
  ].join('\n');
  try {
    const starter = spawn('setsid', ['--fork', 'bash', '-c', script], { cwd: ROOT, env: { ...childEnv(), ...workerEnv(c) }, stdio: 'ignore', detached: true });
    starter.unref();
    starter.on('error', (error) => log(`could not launch ${job.name} for the DL360: ${error.message}`));
  } catch (e) {
    log(`could not launch ${job.name} for the DL360: ${e.message} - queued here instead`);
    enqueue(entry.key, `${entry.reason}; here: the DL360 run could not be launched`, entry.manual, { local: true });
    return;
  }
  state.remote.current = {
    runId, key: job.key, name: job.name, reason: entry.reason, startedAt: Date.now(), pid: null, pidFile, exitFile, logFile,
    sha, mode: mode(), host: c.host.replace(/^.*@/, ''), occurrence: entry.occurrence || null, manual: Boolean(entry.manual),
  };
  saveState();
  log(`started ${job.name} on the DL360 ${c.host.replace(/^.*@/, '')}${sha ? ` at ${sha}` : ''} (${entry.reason})`);
}

function startNextRemote() {
  const r = state.remote;
  if (r.current) return;
  const current = mode();
  if (current === 'off' || !r.queue.length) return;
  const kept = allowedQueue(r.queue, current);
  if (kept.length !== r.queue.length) {
    r.queue.filter((q) => !kept.includes(q)).forEach((q) => log(
      `dropped ${jobByKey(q.key).name} from the DL360's queue: the schedule is ${current} and that entry was not started by hand`));
    r.queue = kept;
    saveState();
  }
  if (!r.queue.length) return;
  const c = workerConfig();
  if (!c) {
    // the cpu entry was taken out of config/workers.json while controls waited for it: they run here
    const moved = r.queue.splice(0);
    saveState();
    moved.forEach((q) => enqueue(q.key, `${q.reason}; here: the DL360 is no longer set up`, q.manual, { local: true }));
    saveState();
    return;
  }
  const entry = r.queue.shift();
  saveState();
  launchRemote(entry, c);
}

function finishRemote(exit, note, finishedAt) {
  const r = state.remote;
  const run = r.current;
  if (!run) return;
  const ended = finishedAt || Date.now();
  const seconds = Math.round((ended - run.startedAt) / 1000);
  const notStarted = exit === REMOTE_NOT_STARTED;
  const text = notStarted ? 'did not start on the DL360 (it did not come up, or could not be copied to): run on this server instead' : note;
  record({
    at: run.startedAt, finishedAt: ended, key: run.key, name: run.name, reason: run.reason,
    seconds, exit, sha: run.sha, pid: run.pid, note: text || null, worker: 'cpu', host: run.host,
    ...(run.stoppedBy ? { stoppedBy: run.stoppedBy } : {}),
  });
  if (exit === 0) keepDuration(`${run.key}@cpu`, seconds);
  log(`${run.name} on the DL360 finished after ${humanDuration(seconds)}${exit === null ? ` (${note || 'outcome unknown'})` : ` with exit code ${exit}`}`);
  [run.pidFile, run.exitFile].forEach((file) => { try { fs.unlinkSync(file); } catch (e) { /* already gone */ } });
  r.current = null;
  saveState();
  if (notStarted) {
    // the fallback: this control and every one waiting behind it run here, in their order
    const moved = [run, ...r.queue.splice(0)];
    saveState();
    moved.forEach((q) => enqueue(q.key, `${q.reason}; here: the DL360 did not come up`, q.manual, { local: true }));
    log(`the DL360 did not come up: ${moved.map((q) => jobByKey(q.key).name).join(', ')} queued on this server`);
    saveState();
    notify({ kind: 'worker-fallback', level: 'warning', title: 'The DL360 did not come up: the controls run on this server',
      body: [`${moved.map((q) => jobByKey(q.key).name).join(', ')} ${moved.length === 1 ? 'is' : 'are'} queued here instead, behind whatever this server runs - `
        + 'the predictors wait for them. The DL360\'s log (log/remote-control.log) says why it did not start.'], link: JOBS_LINK });
    return;
  }
  // the DL360 has to switch itself off after its idle minutes: checked later (workers.js)
  try { require('./workers').watchOff(); } catch (e) { /* the watch is a convenience */ }
  if (exit !== 0 && !run.stoppedBy) {
    notify({ kind: 'job-failed', level: 'error',
      title: exit === null ? `${run.name} on the DL360: ${note || 'the outcome is unknown'}` : `${run.name} on the DL360 ended with exit code ${exit}`,
      body: [`Started ${stamp(run.startedAt)} (${run.reason}), ran ${humanDuration(seconds)}.`], link: JOBS_LINK });
  }
  // the reading, as for a run here: written when the plan's last job finishes, from the records copied back
  scheduleReading(run);
}

function pollRemote() {
  const run = state.remote.current;
  if (!run) return;
  let exit = null;
  let endedAt = null;
  try {
    exit = exitCodeFrom(fs.readFileSync(run.exitFile, 'utf-8'));
    if (exit !== null) endedAt = fs.statSync(run.exitFile).mtimeMs;
  } catch (e) { /* still running, or died without writing */ }
  if (exit !== null) return finishRemote(exit, null, endedAt);
  if (!run.pid) {
    try {
      run.pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null;
      if (run.pid) saveState();
    } catch (e) { /* not written yet */ }
    if (!run.pid && Date.now() - run.startedAt > PID_WAIT_MS) return finishRemote(null, 'the job never reported a PID - launch failed');
    return;
  }
  // the wrapper writes the exit code a moment after the script ends: a gone script gets a few seconds
  if (alive(run.pid, run.runId)) { run.goneAt = null; return; }
  if (!run.goneAt) { run.goneAt = Date.now(); return; }
  if (Date.now() - run.goneAt >= REMOTE_GONE_GRACE_MS) finishRemote(null, 'the job disappeared without writing an exit code (killed?)');
}
const REMOTE_GONE_GRACE_MS = 5 * 1000;

function adoptRemote() {
  const run = state.remote.current;
  if (!run) return;
  if (!run.pid) {
    try { run.pid = Number(String(fs.readFileSync(run.pidFile, 'utf-8')).trim()) || null; } catch (e) { /* none */ }
  }
  if (alive(run.pid, run.runId)) {
    log(`adopted ${run.name} on the DL360 (pid ${run.pid}, started ${stamp(run.startedAt)}) - it survived the restart, as intended`);
    saveState();
    return;
  }
  run.goneAt = null;
  pollRemote();
  if (state.remote.current) finishRemote(null, 'the server was restarted and the job was gone when it came back');
}

// For status(): what runs and waits on the DL360 (the running one with its log's tail), or null.
function remoteStatus() {
  const r = state.remote;
  const c = workerConfig();
  if (!c && !r.current && !r.queue.length) return null;
  return {
    configured: Boolean(c), host: c ? c.host.replace(/^.*@/, '') : (r.current ? r.current.host : null),
    current: r.current ? Object.assign({}, r.current, { log: tail(r.current.logFile, 20) }) : null,
    queue: r.queue.map((q) => Object.assign({}, q, { name: jobByKey(q.key).name })),
  };
}

// --- what is due ------------------------------------------------------------
// Pure, so test/jobs.test.js can walk a week through it without touching the
// filesystem: given the recorded occurrences and a moment in time, what
// should be started?
function dueAt(occurrences, now) {
  const due = [];

  // Each daily job's most recent slot, which before its hour is yesterday's (or, for one with weekdays, the last
  // weekday's) - otherwise a box that comes back at 04:00 would silently skip the day it missed instead of catching
  // it up.
  JOBS.filter((j) => j.daily).forEach((job) => {
    const slot = lastDaily(now, job.daily);
    const late = now - slot;
    const when = slotText(job.daily);
    if (late >= 0 && late <= job.catchUpHours * 3600 * 1000) {
      const id = `${job.key}@${dayId(slot)}`;
      if (!occurrences[id]) {
        due.push({ plan: job.key, occurrence: id,
                   reason: late > 30 * 60 * 1000 ? `catch-up: the ${when} run was missed by ${humanDuration(late / 1000)}` : `schedule ${when}` });
      }
    }
  });

  // A plan is normally started by its weekday's predictor finishing. This is
  // only the safety net: the box was down, or that predictor never ran, and
  // the week would otherwise be skipped in silence - the exact failure this
  // item exists to remove.
  Object.keys(PLANS).forEach((plan) => {
    const spec = PLANS[plan];
    const anchor = lastWeekdaySlot(now, spec.weekday, spec.anchorHour, 0);
    const sinceAnchor = now - anchor;
    const id = `${plan}@${weekId(anchor)}`;
    if (!occurrences[id] && sinceAnchor >= spec.catchUpAfterMs && sinceAnchor <= spec.catchUpUntilMs) {
      due.push({ plan, occurrence: id, reason: `catch-up: ${spec.label} has not run` });
    }
  });
  return due;
}

// Which weekly occurrence a predictor finishing at `now` should start, or
// null for "not this one". Pure, so test/jobs.test.js can walk the Saturday
// edges - the ones that could otherwise run the chain twice in a weekend.
function planTriggerAt(plan, occurrences, now) {
  const spec = PLANS[plan];
  if (now.getDay() !== spec.weekday) return null;
  const anchor = lastWeekdaySlot(now, spec.weekday, spec.anchorHour, 0);
  // Only a predictor finishing AFTER Saturday's anchor (06:00, its slot; 09:00 until 8 Nov 2026, 03:00 until 9 Nov) is the
  // trigger. Before it - a Friday catch-up run spilling past midnight, a
  // hand-started run at 04:00, a restart finalizing a Friday-evening run -
  // the anchor is LAST Saturday, and firing then would both start a ten-hour
  // chain at 04:00 ahead of the day's predictions and consume the previous
  // week's occurrence id, leaving this week's free to fire the whole chain a
  // second time at 23:00. Recording nothing is right: the night predictor,
  // or the 23:00 safety net, still starts it exactly once.
  if (dayId(anchor) !== dayId(now)) return null;
  const id = `${plan}@${weekId(anchor)}`;
  return occurrences[id] ? null : id;
}

function chainTriggerAt(occurrences, now) {
  return planTriggerAt(CHAIN_PLAN, occurrences, now);
}

// Every plan gets its chance: a predictor finishing on a Saturday starts the
// tuning chain, on a Sunday the control experiments.
function maybeTriggerChain(now, why) {
  Object.keys(PLANS).forEach((plan) => {
    const id = planTriggerAt(plan, state.occurrences, now);
    if (id) enqueuePlan(plan, id, `triggered by ${why}`, now);
  });
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

// A plan added to a running installation must not fire its catch-up net on
// the deploy: the plan is new, not missed. Its occurrence currently inside
// the net is marked handled once, per plan, and said so in the log.
// Pure, for the test: which plans are new to this installation, and which of
// their occurrences sit inside the catch-up net right now.
function planBaseline(baselinedPlans, occurrences, now) {
  const fresh = Object.keys(PLANS).filter((plan) => !(baselinedPlans || {})[plan]);
  const skipped = fresh.length ? dueAt(occurrences, now).filter((entry) => fresh.includes(entry.plan)) : [];
  return { fresh, skipped };
}

function baselinePlans(now) {
  state.baselinedPlans = state.baselinedPlans || {};
  const { fresh, skipped } = planBaseline(state.baselinedPlans, state.occurrences, now);
  if (!fresh.length) return;
  skipped.forEach((entry) => remember(entry.occurrence, `baseline: ${entry.plan} was added to the schedule`));
  fresh.forEach((plan) => { state.baselinedPlans[plan] = new Date().toISOString(); });
  saveState();
  if (skipped.length) {
    log(`plan added: ${skipped.map((s) => s.occurrence).join(', ')} treated as already handled (the plan is new, not missed)`);
  }
}

// A job per lottery game (16 Nov 2026) on an installation that ran them as one job: once, on the first start with
// this schedule. The seven new jobs' slots of a day whose one run was handled are handled too (deployed after
// 06:00, they would otherwise all start as catch-ups of a morning already made); a retry or a queue entry of the
// one run becomes one per game; a queue entry of a job this schedule does not have is dropped, and said.
function splitLottery(now) {
  if (state.lotterySplitAt) return;
  const games = JOBS.filter((j) => j.group === LOTTERY);
  const covered = dueAt(state.occurrences, now).filter((entry) => {
    const job = jobByKey(entry.plan);
    return job && job.group === LOTTERY && state.occurrences[`${LEGACY_LOTTERY_KEY}@${entry.occurrence.split('@')[1]}`];
  });
  covered.forEach((entry) => remember(entry.occurrence, 'covered by that day\'s run of all seven lottery games'));
  const waits = state.retry[LEGACY_LOTTERY_KEY];
  if (waits) {
    delete state.retry[LEGACY_LOTTERY_KEY];
    games.forEach((j) => { state.retry[j.key] = Object.assign({}, waits); });
  }
  const queue = [];
  state.queue.forEach((q) => {
    if (q.key === LEGACY_LOTTERY_KEY) games.forEach((j) => queue.push(Object.assign({}, q, { key: j.key })));
    else if (jobByKey(q.key)) queue.push(q);
    else log(`dropped ${q.key} from the queue: this schedule has no such job`);
  });
  // a job key is never queued twice
  state.queue = queue.filter((q, i) => q.key === 'backtest' || queue.findIndex((x) => x.key === q.key) === i);
  state.lotterySplitAt = new Date().toISOString();
  saveState();
  if (covered.length || waits) {
    log(`a job per lottery game: ${covered.length ? `${covered.map((c) => c.occurrence).join(', ')} treated as already handled (that day's one run made them)` : 'no slot to mark'}`
      + `${waits ? `; the retry of the one run (${waits.attempt} of ${waits.of}) is now one per game` : ''}`);
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
  // the run the schedule stopped still has processes: they go first (reapStopped; two minutes at most)
  if (state.stopped && !state.stopped.reportedAt) return;

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
      notify({ kind: 'blocked', level: 'warning', title: `${name} has waited ${humanDuration(WAIT_ALARM_MS / 1000)} for process.lock`,
        body: [`The lock is held by PID ${holder.pid} (${holder.script}). The job is still queued; nothing has been skipped.`], link: JOBS_LINK });
    }
    return;
  }
  waiting = null;
  const pick = pickNext(state.queue, backtestBudget());
  if (!pick) {
    if (!backtestWaitLogged) log(`backtest ${state.queue[0].backtestId || ''} waits: less than ${BACKTEST_MIN_SECONDS / 60} min before the next scheduled run`);
    backtestWaitLogged = true;
    return;
  }
  backtestWaitLogged = false;
  const [entry] = state.queue.splice(pick.index, 1);
  saveState();
  launch(pick.budget ? { ...entry, budget: pick.budget } : entry);
}
let backtestWaitLogged = false;

// Which queue entry starts next: the first scheduled job (anything but a backtest), wherever it waits; with only
// backtests queued, the first of them when `budget` (backtestBudget) allows one now. {index, budget} or null.
function pickNext(queue, budget) {
  const at = queue.findIndex((q) => q.key !== 'backtest');
  if (at >= 0) return { index: at, budget: 0 };
  return queue.length && budget > 0 ? { index: 0, budget } : null;
}

function tick() {
  try {
    pollCurrent();
    pollRemote();

    const now = new Date();
    reapStopped(now.getTime());
    if (mode() !== 'off') {
      dueAt(state.occurrences, now).forEach((entry) => enqueuePlan(entry.plan, entry.occurrence, entry.reason, now));
      startRetries(now.getTime());
    }
    checkSlow(state.current, state.current && state.current.key, '', now.getTime());
    checkSlow(state.remote.current, state.remote.current && `${state.remote.current.key}@cpu`, ' on the DL360', now.getTime());

    // "The predictor just finished" also has to be seen when the predictor
    // was not ours: during the dry-run weekend cron still starts it, and the
    // trigger has to be observed to be trusted. The lock is how we see it.
    const holder = lockHolder();
    // Not while the scheduler's own predictor is still ending: it lets go of the lock a moment before its exit code
    // is written, and what its finishing starts depends on that code (finish).
    // Nor while this schedule still has a lottery game to run: the last of those is the trigger then.
    if (lastLock && isLottery(lastLock.key) && !holder && mode() !== 'off' && !lotteryPending().length) {
      maybeTriggerChain(now, 'the predictor finishing (run started outside the scheduler)');
    }
    lastLock = holder;

    startNext();
    startNextRemote();
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
    const keep = [state.current, state.remote && state.remote.current].filter(Boolean).map((r) => r.runId);
    fs.readdirSync(RUN_DIR).forEach((name) => {
      if (keep.some((k) => name.startsWith(k))) return;      // a run we are about to adopt
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
  adoptRemote();
  if (!state.baselinedAt) baseline(new Date());
  baselinePlans(new Date());
  splitLottery(new Date());
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

// --- status -----------------------------------------------------------------
const WEEKDAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
function nextDueText(job, now) {
  if (job.manualOnly) return 'when an administrator starts one (Admin › Parameters)';
  if (job.after === LOTTERY) return 'when the morning\'s lottery predictors have ended';
  if (job.daily) return `${stamp(nextDaily(now, job.daily))} (${job.daily.weekdays ? 'Monday to Friday' : 'daily'} ${slotText(job.daily)})`;
  const spec = job.plan && PLANS[job.plan];
  const day = spec ? WEEKDAY_NAMES[spec.weekday] : 'Saturday';
  if (job.monthly && spec) {
    // the next plan day it runs on: this one when today is it and its predictor may not have finished yet, else later
    const today = planDayOf(job.plan, now);
    const next = now.getDay() === spec.weekday && runsOnDay(job, dayId(now)) ? dayId(now) : nextMonthlyDay(job, today);
    return `when the first ${day}'s predictor of the month finishes (next: ${next})`;
  }
  return `when ${day}'s predictor finishes`;
}

function lastRunOf(key) {
  return state.history.find((h) => h.key === key && !h.dry && !h.blocked) || null;
}

// --- the actions, for the app's JSON (system.js) --------------------------------
// Each answers {ok, status, message}: status as the JSON routes give it - 400 a
// request that names no job, 409 refused by the state it found (the schedule is
// off, already queued, no longer running), 500 the system refused.

// Why a job may not be started by hand now, or null: unknown, the schedule is
// switched off (the Run now button is disabled then), or it is already running
// or queued (enqueue never takes a job twice). Also on every job of status().
function runRefusal(key, current = mode()) {
  const job = jobByKey(key);
  if (!job) return { status: 400, message: 'unknown job' };
  if (job.manualOnly) return { status: 400, message: `${job.name}: start it from Admin › Parameters` };
  if (current === 'off') return { status: 409, message: 'the schedule is disabled - remove config/scheduler.disabled first' };
  if ((state.current && state.current.key === key) || state.queue.some((q) => q.key === key) || remoteHas(key)) {
    return { status: 409, message: `${job.name}: already running or queued${remoteHas(key) ? ' on the DL360' : ''}` };
  }
  return null;
}

// Run now: queued as a manual entry (the one thing dry mode still starts), and
// started at once when the lock is free. started: it is the running job now.
function runNow(key, by) {
  const refused = runRefusal(String(key || ''));
  if (refused) return Object.assign({ ok: false }, refused);
  const job = jobByKey(String(key));
  const queued = enqueue(job.key, `started by hand (${by})`, true);
  saveState();
  startNext();
  startNextRemote();
  if (!queued) return { ok: false, status: 409, message: `${job.name}: already running or queued` };
  const remote = state.remote.current && state.remote.current.key === job.key ? 'current'
    : state.remote.queue.some((q) => q.key === job.key) ? 'queued' : null;
  if (remote) {
    const at = state.remote.queue.findIndex((q) => q.key === job.key) + 1;
    return { ok: true, status: 200, started: remote === 'current',
      message: `${job.name}: ${remote === 'current' ? 'started on the DL360' : `queued for the DL360${at ? `, ${at === 1 ? 'first' : `number ${at}`} in its queue` : ''}`}` };
  }
  const started = Boolean(state.current && state.current.key === job.key);
  const place = state.queue.findIndex((q) => q.key === job.key) + 1;
  return { ok: true, status: 200, message: `${job.name}: ${started ? 'started' : `queued${place ? `, ${place === 1 ? 'first' : `number ${place}`} in the queue` : ''}`}`, started };
}

// Takes one backtest out of the queue (a running one is stopped with stopRun).
function cancelBacktest(id, by) {
  const before = state.queue.length;
  state.queue = state.queue.filter((q) => q.backtestId !== id);
  if (before === state.queue.length) return { ok: false, status: 409, message: 'that backtest is not in the queue' };
  saveState();
  log(`backtest ${id} removed from the queue by ${by}`);
  return { ok: true, status: 200, message: 'Backtest removed from the queue' };
}
// Queue a backtest and start it when allowed (backtests.js has written its record).
function queueBacktest(id, by) {
  if (mode() === 'off') return { ok: false, status: 409, message: 'the schedule is disabled - remove config/scheduler.disabled first' };
  if (!enqueueBacktest(id, `started by hand (${by})`)) return { ok: false, status: 409, message: 'that backtest is already queued' };
  saveState();
  startNext();
  const started = Boolean(state.current && state.current.backtestId === id);
  const place = state.queue.findIndex((q) => q.backtestId === id) + 1;
  return { ok: true, status: 200, started, place,
    message: started ? 'Backtest started' : `Backtest queued${place ? ` (number ${place} in the queue)` : ''}: it starts after the scheduled jobs, when it can finish before the next scheduled run` };
}
// What the Backtests page reads: the backtest running now (its id and run id, for Stop) and the queued ids in order.
function backtestState() {
  return { current: state.current && state.current.backtestId ? { id: state.current.backtestId, runId: state.current.runId } : null,
    queued: state.queue.filter((q) => q.backtestId).map((q) => q.backtestId), budget: backtestBudget(), nextSlotSeconds: secondsToNextSlot() };
}

// Cancel: takes a job out of the queue; a running job is stopped, not cancelled.
function cancelQueued(key, by, now = new Date()) {
  const before = state.queue.length + state.remote.queue.length;
  state.queue = state.queue.filter((q) => q.key !== key);
  state.remote.queue = state.remote.queue.filter((q) => q.key !== key);
  if (before === state.queue.length + state.remote.queue.length) {
    return jobByKey(key) ? { ok: false, status: 409, message: 'not in the queue' } : { ok: false, status: 400, message: 'unknown job' };
  }
  saveState();
  const name = (jobByKey(key) || { name: key }).name;
  log(`${name} removed from the queue by ${by}`);
  lotteryLeft(key, `${name} being removed from the queue`, now);
  return { ok: true, status: 200, message: `${name}: removed from the queue` };
}

// Stop: SIGTERM to the running job's whole process group, named by its run id
// so a page that is behind can never stop the next job. deps (for the test):
// kill, isAlive.
function stopRun(runId, by, { kill = (pid, signal) => process.kill(pid, signal), isAlive = alive } = {}) {
  const far = state.remote.current;
  if (far && far.runId === String(runId || '')) {
    if (!far.pid) return { ok: false, status: 409, message: 'that job has not reported its PID yet' };
    if (!isAlive(far.pid, far.runId)) return { ok: false, status: 409, message: 'that job is already gone - the page will catch up in a moment' };
    try {
      // the wrapper alone, not its group: it stops the run on the DL360 and copies back what it wrote (remote-control.sh)
      kill(far.pid, 'SIGTERM');
      far.stoppedBy = by;
      saveState();
      log(`${far.name} on the DL360 (pid ${far.pid}) stopped by ${by}`);
      return { ok: true, status: 200, message: `${far.name}: stopping on the DL360` };
    } catch (e) {
      return { ok: false, status: 500, message: `could not stop it: ${e.message}` };
    }
  }
  const run = state.current;
  if (!run || run.runId !== String(runId || '')) return { ok: false, status: 409, message: 'that job is no longer running' };
  if (!run.pid) return { ok: false, status: 409, message: 'that job has not reported its PID yet' };
  if (!isAlive(run.pid, run.runId)) return { ok: false, status: 409, message: 'that job is already gone - the page will catch up in a moment' };
  try {
    // Negative PID: the whole process group setsid created, so the Python
    // process goes with the wrapper instead of being left behind.
    kill(-run.pid, 'SIGTERM');
    run.stoppedBy = by;
    saveState();
    log(`${run.name} (pid ${run.pid}) stopped by ${by}`);
    return { ok: true, status: 200, message: `${run.name}: stopping` };
  } catch (e) {
    return { ok: false, status: 500, message: `could not stop it: ${e.message}` };
  }
}

// full: every kept run (HISTORY_KEPT) instead of the newest 12, for the app's
// Jobs page.
function status({ full = false } = {}) {
  const now = new Date();
  const holder = lockHolder();
  const current = mode();
  return {
    mode: current,
    modeReason: modeReason(),
    lock: holder,
    current: state.current ? Object.assign({}, state.current, {
      log: tail(state.current.logFile, 20),
    }) : null,
    queue: state.queue.map((q) => Object.assign({}, q, { name: nameOf(q.key) })),
    // the Sunday controls on the DL360: null when it is not set up and nothing runs or waits there
    remote: remoteStatus(),
    // a run without data waiting for its next try: [{key, name, attempt, of, at}]
    retries: Object.keys(state.retry).filter((key) => jobByKey(key)).map((key) => ({ key, name: jobByKey(key).name,
      attempt: state.retry[key].attempt, of: state.retry[key].of, at: state.retry[key].at })),
    jobs: JOBS.map((job) => ({
      key: job.key, name: job.name, description: job.description,
      // the seven lottery games' jobs: group 'lottery' and the game each makes
      group: job.group || null, game: job.game || null,
      command: `python3 ${job.script}${job.args.length ? ` ${job.args.join(' ')}` : ''}`,
      logFile: path.join(LOG_DIR, job.log),
      next: nextDueText(job, now),
      nextAt: job.daily ? nextDaily(now, job.daily).getTime() : null,
      plan: job.plan || null,
      last: lastRunOf(job.key),
      refusal: (runRefusal(job.key, current) || {}).message || null,
    })),
    history: full ? state.history.slice() : state.history.slice(0, 12),
    kept: HISTORY_KEPT,
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

// What runs or waits on the DL360 for the schedule (workers.js: no switching it off by hand meanwhile), or null.
function remoteBusy() {
  if (!state || !state.remote) return null;
  const r = state.remote;
  return r.current || r.queue.length ? { running: r.current ? r.current.name : null, queued: r.queue.length } : null;
}

module.exports = {
  start, stop, status, JOBS, LEGACY_JOBS, HISTORY_KEPT, LOTTERY, runRefusal, runNow, cancelQueued, stopRun, remoteBusy,
  // the backtests started from Admin › Parameters (backtests.js)
  queueBacktest, cancelBacktest, backtestState,
  // pipeline.js names the lock holder's step by it (the keno step runs the statistical tuner's script)
  jobOfCmdline,
  // Pure schedule logic for test/jobs.test.js, plus the few entry points the
  // launch/adoption test drives directly (there is no other way to exercise
  // a reparented job without a browser and a real 12-hour tuner).
  _internals: {
    dueAt, weekId, dayId, dailySlot, nextDaily, lastDaily, lastWeekdaySlot, CHAIN_KEYS, CHAIN_PLAN,
    CONTROL_KEYS, CONTROLS_PLAN, PLANS, planKeys, planTriggerAt, nextDueText, planBaseline,
    enqueue, startNext, tick, lockHolder, jobOfCmdline, maybeTriggerChain, state: () => state,
    chainTriggerAt, exitCodeFrom, allowedQueue, READING_JOB, readingSundayOf, writeWeekReading, scheduleReading,
    runsOnDay, planDayOf, planKeysOn, nextMonthlyDay, enqueuePlan, enqueueBacktest, backtestBudget, secondsToNextSlot, pickNext,
    // the Sunday controls on the DL360
    REMOTE_NOT_STARTED, toWorker, startNextRemote, pollRemote, finishRemote, adoptRemote, remoteStatus,
    // a run without data, far longer than usual, the notifications (16 Nov 2026)
    finish, noDataOutcome, startRetries, checkSlow, keepDuration, durationsFrom, slowLimit, stopRun,
    // a job per lottery game (16 Nov 2026)
    LOTTERY, LOTTERY_GAMES, isLottery, lotteryPending, splitLottery, stopLimit, stopHung, reapStopped, groupList, groupMembers, checkoutAfterStop, lotteryLeft, queueAfterLottery, wakeLines, cancelQueued, STOP_FALLBACK_S, STOP_FLOOR_S, STOP_KILL_AFTER_MS,
    setNotifier: (fn) => { notifier = fn; }, setLastLock: (v) => { lastLock = v; },
    useWorker: (configFn, envFn, script) => { workerConfig = configFn; workerEnv = envFn || (() => ({})); remoteScript = script || REMOTE_SCRIPT; },
  },
};
