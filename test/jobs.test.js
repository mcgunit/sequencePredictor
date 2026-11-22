// What the schedule in jobs.js must keep doing (see the header there).
// Pure logic only: no job is started, nothing is written, so this is safe to
// run anywhere. Run with: npm test
// The schedule is Brussels time (production's server runs in it); the night runs are UTC slots, so the tests that
// compare the two need the zone pinned, whatever the machine running them is set to.
process.env.TZ = 'Europe/Brussels';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const jobs = require('../jobs');
const { dueAt, weekId, dayId, lastWeekdaySlot, nextDaily, CHAIN_KEYS, CHAIN_PLAN, CONTROL_KEYS, CONTROLS_PLAN, PLANS, planTriggerAt,
        chainTriggerAt, exitCodeFrom, allowedQueue } = jobs._internals;

let checks = 0;
const ok = (cond, what) => { assert(cond, what); checks += 1; };

const at = (iso) => new Date(iso);                      // local time, like the schedule
const plans = (occurrences, now) => dueAt(occurrences, now).map((d) => d.plan);
const reasonFor = (occurrences, now, plan) => (dueAt(occurrences, now).find((d) => d.plan === plan) || {}).reason || '';

// --- the schedule still mirrors runPredictor.sh and runHyperopt.sh ----------
// This is the point of the whole item: the same commands, in the same order,
// with the same logs - only the trigger changed. A renamed script must break
// here rather than silently stop being scheduled.
const byKey = Object.fromEntries(jobs.JOBS.map((j) => [j.key, j]));
ok(jobs.JOBS.length === 23, 'twenty-three jobs: the night crypto run, a morning run per lottery game (seven), the Council rows and the backtest record behind them, the evening stocks run, the seven steps of the weekly tuning chain, the four control experiments and the backtest an administrator starts');
// A job per lottery game (16 Nov 2026): one game's failure, or its hanging, cannot cost the other six their run.
const LOTTERY_JOBS = jobs.JOBS.filter((j) => j.group === jobs.LOTTERY);
const LOTTERY_KEYS = LOTTERY_JOBS.map((j) => j.key);
ok(LOTTERY_KEYS.join() === 'euromillionsPredictor,lottoPredictor,eurodreamsPredictor,jokerplusPredictor,kenoPredictor,pick3Predictor,vikinglottoPredictor'
  && !byKey.predictor, 'seven lottery jobs, one per game, in the order the one run made them; the one job for all seven is gone');
ok(LOTTERY_JOBS.every((j) => j.script === 'Predictor.py' && j.args.join(' ') === `-a true -g ${j.game}` && j.key === `${j.game}Predictor` && j.log === `${j.key}.log`),
  'each is Predictor.py -a true as runPredictor.sh has it, for its one game, with its own log');
ok(LOTTERY_JOBS.every((j) => j.daily.hour === 6 && j.daily.minute === 0 && !j.daily.utc && !j.daily.weekdays),
  'the lottery games run from 06:00 Brussels, every day (since 9 Nov 2026; 00:30 UTC on the night before, 09:00 until 8 Nov)');
ok(LOTTERY_JOBS.every((j) => j.stopWhenSlow === true && j.noData && j.noData.exit === 74) && jobs.JOBS.filter((j) => j.stopWhenSlow).length === 7
  && jobs.JOBS.filter((j) => j.group).length === 7,
  'each retries a run without data and is stopped when it hangs; no other job is in a group or is stopped by the schedule');
ok(jobs._internals.isLottery('kenoPredictor') && jobs._internals.isLottery('predictor') && !jobs._internals.isLottery('cryptoPredictor') && !jobs._internals.isLottery('nope'),
  'a lottery run is one of the seven, or a run of the one job from before the split');
ok(byKey.cryptoPredictor.script === 'Predictor.py' && byKey.cryptoPredictor.args.join(' ') === '-a true -g crypto,cryptoweek,cryptomonth'
  && byKey.cryptoPredictor.daily.hour === 0 && byKey.cryptoPredictor.daily.minute === 5 && byKey.cryptoPredictor.daily.utc === true && !byKey.cryptoPredictor.daily.weekdays,
  'crypto runs every day at 00:05 UTC, five minutes after its day closes, ahead of the lottery run');
ok(byKey.sharesPredictor.script === 'Predictor.py' && byKey.sharesPredictor.args.join(' ') === '-a true -g shares,sharesweek,sharesmonth'
  && byKey.sharesPredictor.daily.hour === 18 && byKey.sharesPredictor.daily.minute === 5 && !byKey.sharesPredictor.daily.utc && byKey.sharesPredictor.daily.weekdays.join() === '1,2,3,4,5',
  'the stocks run in the evening, Monday to Friday 18:05 Brussels, as soon as Euronext\'s day is complete (18:00) and before the next 09:00 open');
{
  // every game Predictor.py knows runs in exactly one of the two daily runs
  const source = fs.readFileSync(path.join(__dirname, '..', 'Predictor.py'), 'utf-8');
  const all = /default="([a-z0-9,]+)"/.exec(source.slice(source.indexOf("'--games'")))[1].split(',').sort();
  const ran = [byKey.cryptoPredictor, ...LOTTERY_JOBS, byKey.sharesPredictor].flatMap((j) => j.args[j.args.indexOf('-g') + 1].split(',')).sort();
  ok(ran.join() === all.join(), 'the daily runs together cover every game, each once');
}
ok(CHAIN_KEYS.join(',') === 'hyperoptStatistics,hyperoptBoost,hyperoptRLTicket,hyperoptEnsemble,hyperoptQuantum,trainMetaLearner,kenoSubsetTuning',
  'the weekly chain keeps the order runHyperopt.sh documented');
ok(CONTROL_KEYS.join(',') === 'nullControls,randomnessDiscrimination,featureControl,marketRows',
  'the control plan runs the Q0 band first (hours, context since 27 Oct 2026, monthly since 1 Nov 2026), then Q2 (about 80 minutes), the feature control (refits, about an hour) and the market rows report (minutes) - the last job every Sunday runs, which writes the reading');
ok(byKey.nullControls.name === 'Monthly controls: null histories (Q0 band, context)' && /context only/i.test(byKey.nullControls.description) && !/greyed under/.test(byKey.nullControls.description)
  && /first Sunday of the month/.test(byKey.nullControls.description) && /1 Nov 2026/.test(byKey.nullControls.description),
  'the null-histories job is named for what it writes since 27 Oct 2026 (the Q0 band, context - Q0 itself is the daily report\'s row test) and says it is monthly since 1 Nov 2026');
ok(byKey.nullControls.monthly && byKey.nullControls.monthly.since === '2026-11-01' && jobs.JOBS.filter((j) => j.monthly).map((j) => j.key).join() === 'nullControls',
  'only the null band is monthly, from 1 Nov 2026 (the decision of 30 Oct 2026)');
ok(jobs.JOBS.every((j) => !j.plan || PLANS[j.plan]), 'every planned job belongs to a known plan');
ok(CHAIN_KEYS.indexOf('hyperoptQuantum') < CHAIN_KEYS.indexOf('trainMetaLearner'),
  'the quantum tuner must run before the retrain - that is why the retrain is weekly at all');
// The keno sub-selection tuning scores the artifacts the retrain writes (26 Oct 2026): after it, and last - the
// statistical tuner's plain run leaves that strategy out (HyperoptStatistics DEFAULT_STRATEGIES, its self-check)
ok(CHAIN_KEYS.indexOf('trainMetaLearner') === CHAIN_KEYS.length - 2 && CHAIN_KEYS[CHAIN_KEYS.length - 1] === 'kenoSubsetTuning',
  'the meta-learner retrain is followed only by the keno sub-selection tuning, which scores its artifacts');
ok(byKey.kenoSubsetTuning.script === 'HyperoptStatistics.py' && byKey.kenoSubsetTuning.args.join(' ') === '-g keno -s KenoSubsetTuning'
  && byKey.hyperoptStatistics.args.length === 0, 'the keno step runs one strategy of the statistical tuner on keno; the statistical job runs its default list');
// A job that joined its plan later says since when (a day): This week (admin.js weekView) does not expect it of an
// earlier weekend, so the weekend of 24 Oct 2026 does not list the keno step as not run
ok(byKey.kenoSubsetTuning.since === '2026-10-26' && jobs.JOBS.filter((j) => j.since).every((j) => j.plan && /^\d{4}-\d{2}-\d{2}$/.test(j.since)),
  'the keno step joined the chain on 26 Oct 2026; only a planned job carries a since, and it is a day');
ok(byKey.nullControls.args.includes('-d') && byKey.nullControls.args.includes('both') && byKey.randomnessDiscrimination.args.includes('-w'),
  'the controls run with an explicit window, both control kinds and every game');
ok(['nullControls', 'randomnessDiscrimination', 'featureControl'].every((k) => byKey[k].args[1].split(',').length === 11
  && byKey[k].args[1].endsWith('crypto,shares,cryptoweek,sharesweek')), 'every control covers the seven lottery games and the four market games');
ok(byKey.marketRows.args.join(' ') === '-g crypto,shares,cryptoweek,sharesweek -d 250', 'the market rows report scores the four market games over an explicit window');
// Q2 on 20 seeds for every game from 1 Nov 2026 (B2 of the 26 Oct reading): three, five for the markets, left its null
// band to the luck of the seeds; the market games get the same 20, so no --market-repetitions (NullControls keeps its own)
ok(byKey.randomnessDiscrimination.args.join(' ') === '-g lotto,euromillions,eurodreams,vikinglotto,keno,pick3,jokerplus,crypto,shares,cryptoweek,sharesweek -w 10 -n 20 --seed weekly'
  && !byKey.randomnessDiscrimination.args.includes('--market-repetitions') && byKey.nullControls.args.includes('--market-repetitions'),
  `Q2 runs 20 repetitions for every game, the market games included: ${byKey.randomnessDiscrimination.args.join(' ')}`);
ok(/80 minutes/.test(byKey.randomnessDiscrimination.description), 'the Jobs page says how long Q2 takes now');
ok(byKey.featureControl.args.includes('-r') && byKey.featureControl.args.includes('-k'), 'the feature control runs with explicit repeats and noise columns');

// The shell scripts stay as the hand-run path, so they must not drift from
// the schedule: same commands, same logs, same order, or one of the two is
// wrong. Matched on the whole line - script, arguments and log - since two
// steps run HyperoptStatistics.py (26 Oct 2026): by script alone the keno
// step would be found at the statistical job's line.
const shell = (name) => fs.readFileSync(path.join(__dirname, '..', name), 'utf-8');
const runHyperopt = shell('runHyperopt.sh');
ok(shell('runPredictor.sh').includes('python3 Predictor.py -a true'),
  'runPredictor.sh still runs what the predictor jobs run (by hand: every game in one run)');
const lineOf = (job) => `python3 ${[job.script, ...job.args].join(' ')} >> /root/sequencePredictor/log/${job.log} 2>&1`;
const orderInShell = CHAIN_KEYS.map((key) => runHyperopt.indexOf(lineOf(byKey[key])));
ok(orderInShell.every((i) => i > 0), 'every chain job appears in runHyperopt.sh with its arguments and its log');
ok(orderInShell.every((pos, i) => i === 0 || pos > orderInShell[i - 1]),
  'the chain runs in runHyperopt.sh order - if one moves, both must move');
const controlsInShell = CONTROL_KEYS.map((key) => runHyperopt.indexOf(lineOf(byKey[key])));
ok(controlsInShell.every((i) => i > 0), 'every control job appears in runHyperopt.sh with its arguments and its log');
ok(controlsInShell.every((i) => i > orderInShell[orderInShell.length - 1]) && controlsInShell.every((pos, i) => i === 0 || pos > controlsInShell[i - 1]),
  'the hand-run path runs the controls after the chain\'s last step (the keno sub-selection tuning), in the plan order');

jobs.JOBS.forEach((job) => {
  ok(fs.existsSync(path.join(__dirname, '..', job.script)), `${job.key}: ${job.script} exists in the repo`);
  ok(typeof job.log === 'string' && job.log.endsWith('.log'), `${job.key}: has a log file`);
  ok(typeof job.description === 'string' && job.description.length > 20, `${job.key}: says what it does on the Jobs page`);
});
ok(new Set(jobs.JOBS.map((j) => j.log)).size === jobs.JOBS.length, 'every job writes to its own log');
ok(jobs.JOBS.filter((j) => j.daily).map((j) => j.key).join() === ['cryptoPredictor', ...LOTTERY_KEYS, 'sharesPredictor'].join(),
  'only the predictor runs are on a clock, crypto first (a catch-up queues it ahead of the lottery games; admin.js and system.js pick the lottery games by their group)');

// Who holds process.lock, from its command line: the step whose script and arguments it carries, so the keno step
// is not read as the statistical job (system.js would call the schedule's own run a stranger's lock)
{
  const { jobOfCmdline } = jobs._internals;
  ok(jobs.jobOfCmdline === jobOfCmdline, 'exported for pipeline.js too, which names the running step by it');
  const keyOf = (cmdline) => (jobOfCmdline(cmdline) || { key: null }).key;
  ok(keyOf('python3 HyperoptStatistics.py -g keno -s KenoSubsetTuning') === 'kenoSubsetTuning', 'the lock of the keno step names the keno step');
  ok(keyOf('python3 HyperoptStatistics.py') === 'hyperoptStatistics', 'the statistical job is still the statistical job');
  ok(keyOf('/usr/bin/python3 /root/sequencePredictor/HyperoptStatistics.py -g lotto') === 'hyperoptStatistics',
    'a hand run with its own arguments is the job that runs the script without any');
  ok(keyOf('python3 Predictor.py -a true') === 'euromillionsPredictor' && keyOf('python3 Predictor.py') === 'euromillionsPredictor'
    && jobs._internals.isLottery(keyOf('python3 Predictor.py -a true')),
    'a hand run of every game (runPredictor.sh) is read as a lottery run: its finishing can start the weekly plans');
  ok(keyOf('python3 Predictor.py -a true -g keno') === 'kenoPredictor' && keyOf('python3 Predictor.py -a true -g lotto') === 'lottoPredictor'
    && keyOf('/usr/bin/python3 /root/sequencePredictor/Predictor.py -a true -g pick3 ') === 'pick3Predictor',
    'the lock of a game\'s run names that game');
  ok(keyOf('python3 Predictor.py -a true -g shares,sharesweek,sharesmonth') === 'sharesPredictor'
    && keyOf('python3 Predictor.py -a true -g crypto,cryptoweek,cryptomonth') === 'cryptoPredictor'
    && keyOf('python3 Predictor.py -a true -g euromillions,lotto,eurodreams,jokerplus,keno,pick3,vikinglotto') === 'euromillionsPredictor',
    'the locks of the crypto and stocks runs name those runs, so neither starts the weekly plans; the one run of seven games from before the split is a lottery run, not EuroMillions\' own (its arguments go on after the game)');
  ok(keyOf('python3 HyperoptDeepLearning.py') === null, 'a script no job runs is no job');
}

// --- the weekly control experiments: their own Sunday plan ------------------
// Measurements of the rows, never inputs to them, and hours long: started by
// Sunday's predictor finishing, so a tuning chain that overruns into Sunday
// still lets that day's predictions go first, on the agreed FIFO.
const SUN = '2026-09-27';
ok(PLANS[CONTROLS_PLAN].weekday === 0 && PLANS[CHAIN_PLAN].weekday === 6, 'tuning on Saturday, controls on Sunday');
ok(planTriggerAt(CONTROLS_PLAN, {}, at(`${SUN}T10:30:00`)) === `${CONTROLS_PLAN}@2026-W39`,
  "a predictor finishing after Sunday's slot starts this week's controls");
ok(planTriggerAt(CONTROLS_PLAN, {}, at('2026-09-26T10:30:00')) === null, 'Saturday starts no controls');
ok(planTriggerAt(CONTROLS_PLAN, {}, at(`${SUN}T08:00:00`)) === `${CONTROLS_PLAN}@2026-W39`, "Sunday's 06:00 predictor finishing in the morning starts the controls");
ok(planTriggerAt(CONTROLS_PLAN, {}, at(`${SUN}T05:30:00`)) === null, "a chain-delayed Saturday run finishing before Sunday's 06:00 anchor starts nothing");
ok(planTriggerAt(CHAIN_PLAN, {}, at(`${SUN}T10:30:00`)) === null, 'Sunday starts no tuning chain');
ok(planTriggerAt(CONTROLS_PLAN, { [`${CONTROLS_PLAN}@2026-W39`]: { at: 'x' } }, at(`${SUN}T10:30:00`)) === null,
  'controls that already ran this week are not started again');
ok(!plans({}, at(`${SUN}T22:00:00`)).includes(CONTROLS_PLAN), "no catch-up while Sunday's run could still be going");
ok(plans({}, at(`${SUN}T23:30:00`)).includes(CONTROLS_PLAN), 'by Sunday night controls that never started are caught up');
ok(plans({}, at('2026-09-29T08:00:00')).includes(CONTROLS_PLAN) && !plans({}, at('2026-09-29T10:00:00')).includes(CONTROLS_PLAN),
  'and until Tuesday morning, then the week is written off');
ok(dueAt({}, at('2026-09-28T12:00:00')).find((d) => d.plan === CONTROLS_PLAN).occurrence === `${CONTROLS_PLAN}@2026-W39`,
  'Monday still belongs to the ISO week of the Sunday that anchors it');
ok(dueAt({}, at('2026-09-28T08:00:00')).map((d) => d.plan).filter((k) => PLANS[k]).join(',') === `${CHAIN_PLAN},${CONTROLS_PLAN}`,
  'when both weekly plans are owed at once the tuning chain is queued first');

// --- the morning lottery predictor (06:00 Brussels since 9 Nov 2026) -----------
// at() is local (Brussels) time; the slot is local too, so it is 06:00 in summer and winter alike
ok(plans({}, at('2026-11-11T06:00:00')).includes('kenoPredictor'), 'due at its slot: 06:00 Brussels');
ok(reasonFor({}, at('2026-11-11T06:00:00'), 'kenoPredictor') === 'schedule 06:00', 'on time, so not reported as a catch-up');
ok(!plans({}, at('2026-11-11T05:59:00')).includes('kenoPredictor'), 'nothing is due before the slot');
ok(plans({}, at('2026-07-15T06:00:00')).includes('kenoPredictor') && !plans({}, at('2026-07-15T05:59:00')).includes('kenoPredictor'),
  'in summer the same 06:00 Brussels');
ok(reasonFor({}, at('2026-11-11T09:30:00'), 'kenoPredictor').startsWith('catch-up'), 'three and a half hours late is a catch-up');
ok(!plans({ 'kenoPredictor@2026-11-11': { at: 'x' } }, at('2026-11-11T09:30:00')).includes('kenoPredictor'),
  'an occurrence that has been handled is never started twice');
ok(dueAt({}, at('2026-11-11T20:00:00')).find((d) => d.plan === 'kenoPredictor').occurrence === 'kenoPredictor@2026-11-11',
  'a run missed in the morning is caught up the same evening, as that day\'s occurrence');
ok(dueAt({}, at('2026-11-12T01:30:00')).find((d) => d.plan === 'kenoPredictor').occurrence === 'kenoPredictor@2026-11-11'
  && !plans({}, at('2026-11-12T02:30:00')).includes('kenoPredictor'),
  'past the 20 h window (02:00 the next night) it is dropped: that morning\'s run supersedes it');
ok(nextDaily(at('2026-10-24T07:00:00'), byKey.kenoPredictor.daily).getTime() === at('2026-10-25T06:00:00').getTime()
  && nextDaily(at('2026-10-25T12:00:00'), byKey.kenoPredictor.daily).getTime() === at('2026-10-26T06:00:00').getTime(),
  'across the end of summer time (25 Oct 2026) the slot stays 06:00 Brussels');
ok(jobs._internals.nextDueText(byKey.kenoPredictor, at('2026-11-11T12:00:00')).endsWith('(daily 06:00)'), 'and the Jobs page says when it runs');

// --- the night crypto run (8 Nov 2026) ------------------------------------------
ok(reasonFor({}, at('2026-11-11T01:05:00'), 'cryptoPredictor') === 'schedule 00:05 UTC', 'crypto at 00:05 UTC, 01:05 Brussels in winter');
ok(!plans({}, at('2026-11-11T01:04:00')).includes('cryptoPredictor'), 'not before its day has closed');
ok(plans({}, at('2026-07-15T02:05:00')).includes('cryptoPredictor') && !plans({}, at('2026-07-15T01:30:00')).includes('cryptoPredictor'),
  'in summer at 02:05 Brussels - never before 00:00 UTC, when the day closes');
ok(plans({}, at('2026-11-14T01:05:00')).includes('cryptoPredictor') && plans({}, at('2026-11-15T01:05:00')).includes('cryptoPredictor'),
  'every day, the weekend included');
ok(dueAt({}, at('2026-11-11T07:00:00')).map((d) => d.plan).filter((k) => k === 'cryptoPredictor' || LOTTERY_KEYS.includes(k)).join() === ['cryptoPredictor', ...LOTTERY_KEYS].join(),
  'after a night and morning the box was down all are owed, crypto queued first, then the seven games in their order');

// --- the evening stocks run (31 Oct 2026; 18:05 since 8 Nov 2026) ---------------
ok(reasonFor({}, at('2026-09-23T18:05:00'), 'sharesPredictor') === 'schedule 18:05', 'a Wednesday evening at 18:05');
ok(!plans({}, at('2026-09-23T18:04:00')).includes('sharesPredictor'), 'not before it');
ok(dueAt({}, at('2026-09-24T04:00:00')).find((d) => d.plan === 'sharesPredictor').occurrence === 'sharesPredictor@2026-09-23',
  'a missed evening is caught up overnight');
ok(plans({}, at('2026-09-24T08:29:00')).includes('sharesPredictor') && !plans({}, at('2026-09-24T08:31:00')).includes('sharesPredictor'),
  'until 08:30: a card started later would not be up before the open');
ok(!plans({}, at('2026-09-26T18:45:00')).includes('sharesPredictor') && !plans({}, at('2026-09-27T18:45:00')).includes('sharesPredictor'),
  'no run on Saturday or Sunday evening');
ok(dueAt({}, at('2026-09-26T07:00:00')).find((d) => d.plan === 'sharesPredictor').occurrence === 'sharesPredictor@2026-09-25',
  "Friday evening's run is still caught up early on Saturday");
ok(nextDaily(at('2026-09-25T19:00:00'), byKey.sharesPredictor.daily).getTime() === at('2026-09-28T18:05:00').getTime(),
  'after Friday evening the next one is Monday evening');
ok(jobs._internals.nextDueText(byKey.sharesPredictor, at('2026-09-25T19:00:00')).endsWith('(Monday to Friday 18:05)'), 'and the Jobs page says when it runs');

// --- the weekly tuning chain ------------------------------------------------
// Normally started by Saturday's predictor finishing (maybeTriggerChain);
// dueAt is only the safety net for a week that would otherwise be skipped.
const SAT = '2026-09-26';                                  // a Saturday
ok(PLANS[CHAIN_PLAN].anchorHour === 6 && PLANS[CONTROLS_PLAN].anchorHour === 6
  && lastWeekdaySlot(at(`${SAT}T20:00:00`), 6, 6, 0).getTime() === at(`${SAT}T06:00:00`).getTime(),
  'the weekly plans are anchored to 06:00, the lottery predictor\'s slot that triggers them (09:00 until 8 Nov 2026, 03:00 until 9 Nov)');
ok(!plans({}, at(`${SAT}T22:00:00`)).includes(CHAIN_PLAN), 'no catch-up while Saturday\'s run could still be going');
ok(plans({}, at(`${SAT}T23:30:00`)).includes(CHAIN_PLAN), 'by Saturday night a chain that never started is caught up');
ok(plans({}, at('2026-09-27T12:00:00')).includes(CHAIN_PLAN), 'still caught up on Sunday');
ok(plans({}, at('2026-09-28T08:00:00')).includes(CHAIN_PLAN), 'and until Monday morning');
ok(!plans({}, at('2026-09-28T10:00:00')).includes(CHAIN_PLAN), 'after Monday 09:00 the week is written off, not started late');
ok(!plans({ [`${CHAIN_PLAN}@2026-W39`]: { at: 'x' } }, at('2026-09-27T12:00:00')).includes(CHAIN_PLAN),
  'a chain that already ran this week is not started again');
ok(dueAt({}, at('2026-09-27T12:00:00')).find((d) => d.plan === CHAIN_PLAN).occurrence === `${CHAIN_PLAN}@2026-W39`,
  'Sunday still belongs to the week of the Saturday that anchors it');

// --- the trigger, and why it must not fire before Saturday's 06:00 anchor ------
// A review of this module found the chain could run twice in one weekend: a
// Friday catch-up predictor finishing at 04:30 on Saturday anchors to LAST
// Saturday, so it fired under the previous week's id and left this week's id
// free for the 23:00 safety net to fire the whole chain again.
ok(chainTriggerAt({}, at(`${SAT}T10:30:00`)) === `${CHAIN_PLAN}@2026-W39`,
  "a predictor finishing after Saturday's slot starts this week's chain");
ok(chainTriggerAt({}, at(`${SAT}T08:00:00`)) === `${CHAIN_PLAN}@2026-W39`,
  "Saturday's 06:00 predictor finishing in the morning starts this week's chain");
ok(chainTriggerAt({}, at(`${SAT}T02:30:00`)) === null && chainTriggerAt({}, at(`${SAT}T05:30:00`)) === null,
  'a run finishing before Saturday 06:00 belongs to no chain - it is Friday\'s run spilling over');
ok(chainTriggerAt({}, at(`${SAT}T23:59:00`)) === `${CHAIN_PLAN}@2026-W39`,
  'late on Saturday it is still this week');
ok(chainTriggerAt({ [`${CHAIN_PLAN}@2026-W39`]: { at: 'x' } }, at(`${SAT}T10:30:00`)) === null,
  'a chain that already ran this week is never triggered again');
ok(chainTriggerAt({}, at('2026-09-27T10:30:00')) === null, 'only Saturday triggers the chain');
// The whole weekend, in order: the spillover records nothing, so the safety
// net fires exactly one chain and the trigger then has nothing left to do.
const weekend = {};
ok(chainTriggerAt(weekend, at(`${SAT}T02:30:00`)) === null, 'weekend: the 02:30 spillover starts nothing');
const net = dueAt(weekend, at(`${SAT}T23:30:00`)).find((d) => d.plan === CHAIN_PLAN);
ok(net && net.occurrence === `${CHAIN_PLAN}@2026-W39`, 'weekend: the safety net queues it once');
weekend[net.occurrence] = { at: 'x' };
ok(chainTriggerAt(weekend, at('2026-09-27T01:00:00')) === null && !dueAt(weekend, at('2026-09-27T12:00:00')).some((d) => d.plan === CHAIN_PLAN),
  'weekend: and nothing starts a second chain afterwards');

// --- an exit file is an outcome only when it holds a whole number -----------
// `echo "$status" > file` creates the file and fills it as two steps, and
// Number('') is 0 - which would record a killed run as a success.
ok(exitCodeFrom('0') === 0 && exitCodeFrom('7') === 7 && exitCodeFrom('137\n') === 137, 'real exit codes parse');
ok(exitCodeFrom('') === null && exitCodeFrom('   ') === null, 'an empty or half-written file is not exit code 0');
ok(exitCodeFrom('ok') === null && exitCodeFrom('0 1') === null, 'garbage is not an exit code either');

// --- what a mode is allowed to start ----------------------------------------
// Dry mode promises to start nothing, and a queue survives in
// config/scheduler.json across a restart - so a chain queued while the
// schedule was on must not drain after a rollback to dry.
const mixed = [{ key: 'kenoPredictor', manual: false }, { key: 'hyperoptBoost', manual: true }];
ok(allowedQueue(mixed, 'on').length === 2, 'on mode starts everything queued');
ok(allowedQueue(mixed, 'dry').length === 1 && allowedQueue(mixed, 'dry')[0].manual === true,
  'dry mode keeps only what an administrator started by hand');
ok(allowedQueue(mixed, 'dry') !== mixed && allowedQueue(mixed, 'on') !== mixed, 'the queue is never aliased back');

// --- the monthly null band (30 Oct 2026, the decision) ------------
// The Q0 band runs only on the first Sunday of the month (day 1 to 7) from 1 Nov 2026 on; the other three controls
// every Sunday. The occurrence's Sunday decides, also for a catch-up on Monday or Tuesday.
{
  const { planKeysOn, planDayOf, runsOnDay, nextMonthlyDay } = jobs._internals;
  const on = (day) => planKeysOn(CONTROLS_PLAN, day);
  ok(on('2026-11-01').keys.join() === CONTROL_KEYS.join() && on('2026-11-01').skipped.length === 0,
    'a first Sunday (1 Nov 2026) queues all four controls, the band first');
  ok(on('2026-11-08').keys.join() === 'randomnessDiscrimination,featureControl,marketRows'
    && on('2026-11-08').skipped.map((x) => `${x.key}:${x.next}`).join() === 'nullControls:2026-12-06',
    'a second Sunday leaves the band out, and names the Sunday it runs next');
  ok(['2026-11-15', '2026-11-22', '2026-11-29'].every((d) => !on(d).keys.includes('nullControls')) && on('2026-12-06').keys.includes('nullControls')
    && on('2027-01-03').keys.includes('nullControls') && !on('2027-01-10').keys.includes('nullControls'),
    'only the Sunday on day 1 to 7 of its month runs the band');
  ok(on('2026-10-25').keys.join() === CONTROL_KEYS.join() && runsOnDay(byKey.nullControls, '2026-10-18'),
    'before 1 Nov 2026 the band was weekly - an earlier Sunday is read as it ran');
  ok(planKeysOn(CHAIN_PLAN, '2026-11-07').keys.join() === CHAIN_KEYS.join(), 'the tuning chain is untouched');
  ok(planDayOf(CONTROLS_PLAN, at('2026-11-01T10:30:00')) === '2026-11-01' && planDayOf(CONTROLS_PLAN, at('2026-11-02T08:00:00')) === '2026-11-01'
    && planDayOf(CONTROLS_PLAN, at('2026-11-10T08:30:00')) === '2026-11-08',
    'a catch-up on Monday or Tuesday morning is the Sunday before\'s occurrence: first or not by that Sunday');
  ok(nextMonthlyDay(byKey.nullControls, '2026-11-01') === '2026-12-06' && nextMonthlyDay(byKey.nullControls, '2026-10-25') === '2026-11-01',
    'the next band: the next first Sunday');
}

// --- occurrence identity ----------------------------------------------------
// Ids are the only thing standing between "ran once" and "ran twice", so they
// must not drift with the year boundary.
ok(weekId(at('2026-01-01T12:00:00')) === '2026-W01', 'ISO week of a Thursday 1 January');
ok(weekId(at('2027-01-03T12:00:00')) === '2026-W53', 'the Sunday of a week that started in the old year');
ok(dayId(at('2026-09-05T23:00:00')) === '2026-09-05', 'day ids are zero padded');
ok(nextDaily(at('2026-09-23T10:00:00'), { hour: 9, minute: 0 }).getTime() === at('2026-09-24T09:00:00').getTime(),
  'after today\'s slot the next one is tomorrow');


// --- what the control jobs say and do on the Jobs page -----------------
{
  const { nextDueText, planBaseline } = jobs._internals;
  const anyNow = at('2026-09-26T20:00:00');
  ok(CONTROL_KEYS.filter((k) => k !== 'nullControls').every((k) => nextDueText(byKey[k], anyNow) === "when Sunday's predictor finishes"),
    'the weekly control jobs say Sunday on the Jobs page, from the plan table');
  ok(nextDueText(byKey.nullControls, at('2026-10-31T20:00:00')) === "when the first Sunday's predictor of the month finishes (next: 2026-11-01)"
    && nextDueText(byKey.nullControls, at('2026-11-01T08:00:00')).endsWith('(next: 2026-11-01)')
    && nextDueText(byKey.nullControls, at('2026-11-03T12:00:00')).endsWith('(next: 2026-12-06)')
    && nextDueText(byKey.nullControls, at('2026-11-08T12:00:00')).endsWith('(next: 2026-12-06)'),
    'the monthly band says which Sunday it runs next');
  ok(CHAIN_KEYS.every((k) => nextDueText(byKey[k], anyNow) === "when Saturday's predictor finishes"), 'the chain jobs still say Saturday');
  // Every scheduled entry point takes process.lock, or the scheduler's "one
  // job at a time behind the same lock" promise is void for it.
  jobs.JOBS.forEach((job) => {
    const text = fs.readFileSync(path.join(__dirname, '..', job.script), 'utf-8');
    ok(text.includes('create_lock') && text.includes('remove_lock'), `${job.key}: ${job.script} takes and releases process.lock`);
  });
  // A plan added to a running installation is baselined on its own: the
  // occurrence inside its catch-up net at the deploy is marked handled, once,
  // instead of starting hours of controls on the deploy.
  const deploy = planBaseline({}, {}, at('2026-09-28T21:00:00'));   // a Monday evening deploy
  ok(deploy.fresh.length === 2 && deploy.skipped.map((e) => e.occurrence).join(',') === `${CONTROLS_PLAN}@2026-W39`,
    'at a Monday deploy the controls occurrence inside the net is skipped; the chain is past its net');
  const sunday = planBaseline({ [CHAIN_PLAN]: 'x' }, {}, at('2026-09-27T12:00:00'));
  ok(sunday.fresh.join(',') === CONTROLS_PLAN && sunday.skipped.length === 0,
    'a Sunday-noon deploy skips nothing: the net opens at 23:00 and this week\'s controls still run tonight');
  const done = planBaseline({ [CHAIN_PLAN]: 'x', [CONTROLS_PLAN]: 'x' }, {}, at('2026-09-28T21:00:00'));
  ok(done.fresh.length === 0 && done.skipped.length === 0, 'a baselined plan is never baselined again - its next miss is a real miss');
}

// --- enqueuePlan itself, on a copy of jobs.js in a temp dir --------------------------------------------------------
// The copy's config is a stand-in next to it and the schedule is off (scheduler.disabled): start() starts nothing, and
// enqueuePlan only fills the queue the worker would drain - nothing is launched.
{
  const os = require('os');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'jobs-monthly-'));
  fs.copyFileSync(path.join(__dirname, '..', 'jobs.js'), path.join(dir, 'jobs.js'));
  fs.writeFileSync(path.join(dir, 'config.js'), `module.exports = { CONFIG_DIR: ${JSON.stringify(path.join(dir, 'config'))} };`);
  fs.mkdirSync(path.join(dir, 'config'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
  fs.writeFileSync(path.join(dir, 'config', 'scheduler.json'), JSON.stringify({ version: 1, occurrences: {}, queue: [], history: [], current: null,
    baselinedAt: 'x', baselinedPlans: { [CHAIN_PLAN]: 'x', [CONTROLS_PLAN]: 'x' } }));
  const quiet = console.log;
  console.log = () => {};
  let copy;
  try {
    copy = require(path.join(dir, 'jobs.js'));
    copy.start();
    const c = copy._internals;
    const queued = () => c.state().queue.map((q) => q.key).join();
    c.enqueuePlan(CONTROLS_PLAN, `${CONTROLS_PLAN}@2026-W44`, 'test', at('2026-11-01T10:30:00'));
    const first = queued();
    c.state().queue.length = 0;
    c.enqueuePlan(CONTROLS_PLAN, `${CONTROLS_PLAN}@2026-W45`, 'test', at('2026-11-08T10:30:00'));
    const second = queued();
    c.state().queue.length = 0;
    c.enqueuePlan(CONTROLS_PLAN, `${CONTROLS_PLAN}@2026-W45b`, 'test', at('2026-11-02T08:00:00'));
    const catchUp = queued();
    c.state().queue.length = 0;
    console.log = quiet;
    const logText = fs.readFileSync(path.join(dir, 'log', 'scheduler.log'), 'utf-8');
    ok(first === 'nullControls,randomnessDiscrimination,featureControl,marketRows', `a first Sunday queues the band, first: ${first}`);
    ok(second === 'randomnessDiscrimination,featureControl,marketRows', `a second Sunday does not queue it: ${second}`);
    ok(catchUp === first, `a Monday-morning catch-up of a first Sunday queues it too: ${catchUp}`);
    ok(/weeklyControls@2026-W45 leaves out Monthly controls: null histories \(Q0 band, context\): monthly, not this one - next on 2026-12-06/.test(logText),
      'and the log says the band was left out, and when it runs next');
  } finally {
    console.log = quiet;
    if (copy) copy.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// --- backtests (Admin › Parameters, 8 Nov 2026): never Run now, behind every scheduled job, a budget before the next slot
{
  const { backtestBudget, secondsToNextSlot, pickNext, nextDueText } = jobs._internals;
  ok(jobs.runRefusal('backtest').status === 400 && /Admin › Parameters/.test(jobs.runRefusal('backtest').message), 'a backtest is never "Run now"');
  ok(nextDueText(byKey.backtest, at('2026-11-09T12:00:00')) === 'when an administrator starts one (Admin › Parameters)', 'the Jobs page says how one starts');
  // Monday 9 Nov, winter: crypto 01:05, lottery 01:30, stocks 18:05 (Brussels)
  ok(secondsToNextSlot(at('2026-11-09T12:00:00')) === (6 * 60 + 5) * 60, 'noon on a weekday: the stocks run at 18:05 is next');
  ok(backtestBudget(at('2026-11-09T12:00:00')) === 3 * 3600, 'six hours of room: the 3 h maximum');
  ok(backtestBudget(at('2026-11-09T17:00:00')) === 55 * 60, 'an hour and five before 18:05: ends 10 min before it');
  ok(backtestBudget(at('2026-11-09T17:45:00')) === 0, '20 min before a slot: too little room, waits');
  ok(secondsToNextSlot(at('2026-11-07T19:00:00')) === (6 * 60 + 5) * 60, 'Saturday 19:00: the 01:05 crypto run is next (no stocks run on a Saturday)');
  ok(backtestBudget(at('2026-11-10T00:50:00')) === 0, 'between the crypto and lottery runs: no room');
  const bt = (id) => ({ key: 'backtest', backtestId: id });
  ok(JSON.stringify(pickNext([bt('a'), { key: 'kenoPredictor' }, bt('b')], 0)) === '{"index":1,"budget":0}', 'a scheduled job jumps every queued backtest');
  ok(JSON.stringify(pickNext([bt('a'), bt('b')], 1800)) === '{"index":0,"budget":1800}', 'only backtests: the first, with its budget');
  ok(pickNext([bt('a')], 0) === null && pickNext([], 3600) === null, 'no room, or nothing queued: nothing starts');
}

// --- a run without data, far longer than usual, the notifications (16 Nov 2026) --------------------------------------
// On a copy of jobs.js in a temp dir, the schedule off (nothing is launched): the runs are put in state.current by hand
// and finished through finish(), the notes collected through setNotifier.
{
  const os = require('os');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'jobs-nodata-'));
  fs.copyFileSync(path.join(__dirname, '..', 'jobs.js'), path.join(dir, 'jobs.js'));
  fs.writeFileSync(path.join(dir, 'config.js'), `module.exports = { CONFIG_DIR: ${JSON.stringify(path.join(dir, 'config'))} };`);
  fs.mkdirSync(path.join(dir, 'config'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
  const HOUR = 3600 * 1000;
  // an old state file: no retry, no durations - the durations come from its history (newest first)
  const hist = [1000, 1200, 900].map((seconds, i) => ({ at: i, finishedAt: i, key: 'kenoPredictor', name: 'x', seconds, exit: 0 }))
    .concat([{ at: 9, key: 'kenoPredictor', seconds: 50000, exit: 1 }, { at: 9, key: 'nullControls', seconds: 700, exit: 0, worker: 'cpu' }]);
  fs.writeFileSync(path.join(dir, 'config', 'scheduler.json'), JSON.stringify({ version: 1, occurrences: {}, queue: [], history: hist, current: null,
    baselinedAt: 'x', baselinedPlans: { [CHAIN_PLAN]: 'x', [CONTROLS_PLAN]: 'x' } }));
  const quiet = console.log;
  console.log = () => {};
  let copy;
  try {
    copy = require(path.join(dir, 'jobs.js'));
    copy.start();
    copy.stop();
    const c = copy._internals;
    const st = () => c.state();
    const notes = [];
    c.setNotifier((n) => { notes.push(n); return n; });
    const spec = copy.JOBS.find((j) => j.key === 'kenoPredictor').noData;
    const t0 = Date.now() - 10 * 60 * 1000;
    const begin = (extra = {}) => { st().current = Object.assign({ key: 'kenoPredictor', name: 'Morning predictor: Keno', reason: 'test', startedAt: t0, runId: 'r', pid: 1, pidFile: path.join(dir, 'p'), exitFile: path.join(dir, 'e') }, extra); };
    const results = [];
    results.push(spec && spec.exit === 74 && spec.retries === 6 && spec.everyMs === HOUR, 'a lottery game: status 74, six retries, an hour apart');
    results.push(JSON.stringify(st().durations.kenoPredictor) === '[900,1200,1000]' && st().durations['nullControls@cpu'][0] === 700 && JSON.stringify(st().retry) === '{}',
      `an old state file gets its durations from the history, successful runs only: ${JSON.stringify(st().durations)}`);

    // the first run without data: a retry an hour after it ended, one note, the weekly plans wait
    begin();
    c.finish(74, null, t0 + 60000);
    let r = st().retry.kenoPredictor;
    results.push(r && r.attempt === 1 && r.of === 6 && r.at === t0 + 60000 + HOUR && st().current === null, `a run without data plans retry 1 an hour after it ended: ${JSON.stringify(r)}`);
    results.push(/tries again at \d\d:\d\d \(retry 1 of 6\)/.test(st().history[0].note) && st().history[0].exit === 74, `the record says when: ${st().history[0].note}`);
    results.push(notes.length === 1 && notes[0].kind === 'no-data' && notes[0].level === 'warning' && /up to 6 times/.test(notes[0].body[0]), 'one note: no data, and what happens next');
    results.push(copy.status().retries.length === 1 && copy.status().retries[0].attempt === 1 && copy.status().retries[0].name === 'Morning predictor: Keno', 'the Jobs page gets the pending retry');
    // survives a restart
    const saved = JSON.parse(fs.readFileSync(path.join(dir, 'config', 'scheduler.json'), 'utf-8'));
    results.push(saved.retry.kenoPredictor.attempt === 1, 'the pending retry is in the state file (a restart keeps it)');

    // not before its hour; then queued as a retry
    c.startRetries(r.at - 1000);
    results.push(st().queue.length === 0 && st().retry.kenoPredictor, 'not queued before its hour');
    c.startRetries(r.at + 1000);
    results.push(st().queue.length === 1 && st().queue[0].key === 'kenoPredictor' && st().queue[0].retry === 1 && !st().queue[0].manual && !st().retry.kenoPredictor
      && /retry 1 of 6/.test(st().queue[0].reason), `queued at its hour, marked as retry 1: ${JSON.stringify(st().queue[0])}`);
    st().queue.length = 0;

    // retries 1..5 without data: each plans the next, no further note; the sixth gives up with one
    for (let n = 1; n <= 5; n += 1) { begin({ retry: n }); c.finish(74, null, t0 + n * HOUR); }
    results.push(st().retry.kenoPredictor.attempt === 6 && notes.length === 1, 'five retries without data: retry 6 planned, still one note');
    begin({ retry: 6 });
    c.finish(74, null, t0 + 7 * HOUR);
    results.push(!st().retry.kenoPredictor && /no further retry/.test(st().history[0].note) && st().history[0].retry === 6, `the sixth without data ends it: ${st().history[0].note}`);
    results.push(notes.length === 2 && notes[1].kind === 'no-data-ended' && notes[1].level === 'error' && /after 6 retries/.test(notes[1].title), 'and says so, as an error');

    // data arrives on a retry: a note that it recovered, no retry left
    notes.length = 0;
    begin();
    c.finish(74, null, t0 + 60000);
    delete st().retry.kenoPredictor;   // as launch() does when the retry starts
    begin({ retry: 1 });
    c.finish(0, null, t0 + HOUR + 120000);
    results.push(!st().retry.kenoPredictor && notes.length === 2 && notes[1].kind === 'no-data-ended' && notes[1].level === 'info' && /retry 1/.test(notes[1].title), 'data on retry 1: the retries end, a note says it arrived');

    // a pending retry hours late is dropped; a run by hand in between is that try
    begin();
    c.finish(74, null, t0);
    c.startRetries(st().retry.kenoPredictor.at + 4 * HOUR);
    // the game's own run is what must not be queued: on a weekly-plan day the drop queues that day's plans
    // (lotteryLeft, 16 Nov 2026), and this test runs on the real calendar (found on a Saturday, 21 Nov 2026)
    results.push(!st().retry.kenoPredictor && !st().queue.some((q) => q.key === 'kenoPredictor'), 'a retry due more than three hours ago is dropped');
    st().queue = [];
    begin();
    c.finish(74, null, t0);
    c.enqueue('kenoPredictor', 'run now', true);
    c.startRetries(st().retry.kenoPredictor.at + 1000);
    results.push(st().queue.length === 1 && st().queue[0].manual === true && !st().queue[0].retry && !st().retry.kenoPredictor, 'already queued by hand: that run is the retry');
    st().queue.length = 0;

    // other jobs and other exits: the plain failure note; stopped by an administrator: none
    notes.length = 0;
    begin({ key: 'cryptoDaily', name: 'Crypto daily' });
    c.finish(74, null, t0 + 1000);
    results.push(!st().retry.cryptoDaily && notes.length === 1 && notes[0].kind === 'job-failed' && /exit code 74/.test(notes[0].title), 'status 74 means "no data" only for a job that says so');
    begin();
    c.finish(1, null, t0 + 1000);
    results.push(notes.length === 2 && notes[1].kind === 'job-failed' && /Morning predictor: Keno ended with exit code 1/.test(notes[1].title) && !st().retry.kenoPredictor, 'a failed run: a note, no retry');
    begin({ stoppedBy: 'hana' });
    c.finish(null, 'killed', t0 + 1000);
    results.push(notes.length === 2 && st().history[0].stoppedBy === 'hana', 'stopped by an administrator: recorded, no note');
    begin();
    c.finish(0, null, t0 + 1100 * 1000);
    results.push(notes.length === 2 && st().durations.kenoPredictor.slice(-1)[0] === 1100, 'a normal run: no note, its duration kept');

    // far longer than usual: 1.5 x the longest of the last runs and at least half an hour more, said once
    notes.length = 0;
    st().durations.kenoPredictor = [900, 1200, 1000];
    const lim = c.slowLimit('kenoPredictor');
    results.push(lim && lim.longest === 1200 && lim.limit === 3000, `the limit: the longest (1200 s) + 30 min: ${JSON.stringify(lim)}`);
    st().durations.long = [7200, 8000, 7000];
    results.push(c.slowLimit('long').limit === 12000 && c.slowLimit('cryptoDaily') === null, '1.5 x for a long job; no limit with fewer than three runs');
    begin();
    results.push(c.checkSlow(st().current, 'kenoPredictor', '', t0 + 2900 * 1000) === false && !notes.length, 'under the limit: nothing');
    results.push(c.checkSlow(st().current, 'kenoPredictor', '', t0 + 3100 * 1000) === true && notes.length === 1 && notes[0].kind === 'slow' && /far longer than usual/.test(notes[0].title), 'over it: a note');
    results.push(c.checkSlow(st().current, 'kenoPredictor', '', t0 + 9000 * 1000) === false && notes.length === 1, 'once per run');
    st().current = null;
    const logText = fs.readFileSync(path.join(dir, 'log', 'scheduler.log'), 'utf-8');
    results.push(/the weekly plans wait for Morning predictor: Keno \(a run with data\)/.test(logText), 'a run without data is not the weekly plans\' trigger');

    // --- a job per lottery game (16 Nov 2026) ---
    const nameOfJob = (key) => copy.JOBS.find((j) => j.key === key).name;
    const game = (key, extra = {}) => begin(Object.assign({ key, name: nameOfJob(key) }, extra));
    // the games share their notes about the API: the same key for six hours, so the store keeps one for the morning
    notes.length = 0;
    st().retry = {};
    game('kenoPredictor');
    c.finish(74, null, t0 + 1000);
    game('lottoPredictor');
    c.finish(74, null, t0 + 2000);
    results.push(notes.length === 2 && notes.every((n) => n.kind === 'no-data' && n.dedupe === 'no-data:lottery' && n.dedupeMs === 6 * HOUR)
      && /not repeated for them this morning/.test(notes[0].body[0]), 'two games without data: both raise the note under one key for the lottery group');
    results.push(st().retry.kenoPredictor.attempt === 1 && st().retry.lottoPredictor.attempt === 1, 'each game has its own retry');
    results.push(c.lotteryPending().join() === 'Morning predictor: Keno,Morning predictor: Lotto', `both are still to be made: ${c.lotteryPending().join()}`);
    {
      const notifications = require(path.join(__dirname, '..', 'notifications.js'));
      notifications._internals.useFile(path.join(dir, 'config', 'notifications.json'));
      c.setNotifier(notifications.raise);
      st().retry = {};
      game('kenoPredictor');
      c.finish(74, null, t0 + 1000);
      game('lottoPredictor');
      c.finish(74, null, t0 + 2000);
      game('cryptoPredictor');
      c.finish(1, null, t0 + 3000);
      const kept = notifications.view({ name: 'x' }).items;
      results.push(kept.filter((n) => n.kind === 'no-data').length === 1 && /Keno/.test(kept.find((n) => n.kind === 'no-data').title)
        && kept.filter((n) => n.kind === 'job-failed').length === 1, `the store keeps one no-data note for the morning (the first game's), other notes as before: ${kept.map((n) => n.kind).join()}`);
      c.setNotifier((n) => { notes.push(n); return n; });
    }
    for (let n = 1; n <= 6; n += 1) { game('pick3Predictor', { retry: n }); c.finish(74, null, t0 + n * HOUR); }
    results.push(notes.slice(-1)[0].kind === 'no-data-ended' && notes.slice(-1)[0].dedupe === 'no-data-ended:lottery', 'the retries used up: one key for the group too');
    game('lottoPredictor', { retry: 2 });
    c.finish(0, null, t0 + 2 * HOUR);
    results.push(notes.slice(-1)[0].level === 'info' && notes.slice(-1)[0].dedupe === 'no-data-arrived:lottery', 'the data arriving on a retry: one key for the group');

    // the weekly plans wait for the last game: not while another is queued, running or waiting for a retry
    st().retry = {};
    st().queue.length = 0;
    results.push(c.lotteryPending().length === 0, 'nothing queued, running or waiting: no lottery run is left');
    st().queue.push({ key: 'lottoPredictor', reason: 'schedule 06:00', manual: false }, { key: 'cryptoPredictor', reason: 'x', manual: false });
    game('kenoPredictor');
    results.push(c.lotteryPending().join() === 'Morning predictor: Keno,Morning predictor: Lotto' && c.lotteryPending(st().current).join() === 'Morning predictor: Lotto',
      'a running and a queued game are left; the run that is finishing does not count itself, the crypto run never counts');
    c.finish(0, null, t0 + 600 * 1000);
    results.push(/the weekly plans wait for Morning predictor: Lotto\n/.test(fs.readFileSync(path.join(dir, 'log', 'scheduler.log'), 'utf-8')), 'a game finishing with another queued does not start the weekly plans');
    st().queue.length = 0;
    st().retry.predictor = { attempt: 1, of: 6, at: t0 };
    results.push(c.lotteryPending().join() === 'predictor', 'a retry of the one run from before the split still counts');
    st().retry = {};
    // a run of the one job from before the split that was still running at the deploy is finished like any other
    begin({ key: 'predictor', name: 'Morning predictor: lottery games' });
    c.finish(0, null, t0 + 1000);
    results.push(st().current === null && st().history[0].key === 'predictor' && copy.status().history[0].key === 'predictor', 'a run of the one job from before the split ends and is shown');

    // --- a hung game is stopped (16 Nov 2026): twice the longest of a week of runs, at least two hours; four hours without ---
    {
      const kills = [];
      const deps = { kill: (pid, signal) => { kills.push(`${pid} ${signal}`); }, isAlive: () => true };
      st().durations.kenoPredictor = [900, 1200, 1000];
      let lim = c.stopLimit('kenoPredictor');
      results.push(lim.fallback === true && lim.limit === c.STOP_FALLBACK_S && lim.limit === 4 * 3600 && lim.runs === 3, `fewer than seven runs to compare with: four hours: ${JSON.stringify(lim)}`);
      st().durations.kenoPredictor = [600, 700, 900, 1200, 1000, 800, 650];
      lim = c.stopLimit('kenoPredictor');
      results.push(lim.fallback === false && lim.limit === c.STOP_FLOOR_S && lim.limit === 7200 && lim.longest === 1200, `a week of short runs: the two-hour floor: ${JSON.stringify(lim)}`);
      st().durations.long = [5000, 4000, 4100, 4200, 4300, 4400, 4500];
      results.push(c.stopLimit('long').limit === 10000, 'a week of long runs: twice the longest');
      notes.length = 0;
      game('kenoPredictor', { pid: 4242 });
      const run = st().current;
      // the schedule is off in this copy: a run is reported, never stopped
      results.push(c.checkSlow(run, 'kenoPredictor', '', t0 + 7300 * 1000, deps) === true && kills.length === 0 && !run.hungAt
        && /The schedule stops it itself once it has run 2\.0 h/.test(notes[0].body[0]), `with the schedule off nothing is stopped; the notice names the limit: ${notes[0] && notes[0].body[0]}`);
      fs.unlinkSync(path.join(dir, 'config', 'scheduler.disabled'));
      delete run.slowAt;
      notes.length = 0;
      results.push(c.checkSlow(run, 'kenoPredictor', '', t0 + 7100 * 1000, deps) === true && notes.length === 1 && kills.length === 0 && !run.hungAt,
        'past the notice\'s limit and under the stop limit: the notice, the run goes on');
      c.checkSlow(run, 'kenoPredictor', '', t0 + 7201 * 1000, deps);
      results.push(kills.join() === '-4242 SIGTERM' && run.hungAt === t0 + 7201 * 1000 && run.stoppedBy === 'the schedule' && run.hung.limit === 7200 && run.hung.seconds === 7201,
        `past the stop limit: SIGTERM to its process group, recorded as the schedule's stop: ${kills.join()} ${JSON.stringify(run.hung)}`);
      c.checkSlow(run, 'kenoPredictor', '', t0 + 7231 * 1000, deps);
      results.push(kills.length === 1, 'not signalled again within the minute');
      c.checkSlow(run, 'kenoPredictor', '', t0 + 7262 * 1000, deps);
      c.checkSlow(run, 'kenoPredictor', '', t0 + 7300 * 1000, deps);
      results.push(kills.join() === '-4242 SIGTERM,-4242 SIGKILL' && run.killedAt === t0 + 7262 * 1000, `still there a minute later: SIGKILL, once: ${kills.join()}`);
      fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
      st().queue.push({ key: 'pick3Predictor', reason: 'schedule 06:00', manual: false });
      c.finish(null, 'killed', t0 + 7263 * 1000);
      const rec = st().history[0];
      results.push(rec.key === 'kenoPredictor' && rec.exit === null && rec.stoppedBy === 'the schedule' && /^stopped by the schedule: far longer than usual\. It had run 2\.0 h/.test(rec.note),
        `the record says who stopped it and why: ${rec.note}`);
      results.push(notes.length === 2 && notes[1].kind === 'job-failed' && notes[1].level === 'error' && /Morning predictor: Keno was stopped: far longer than usual/.test(notes[1].title)
        && /were not made/.test(notes[1].body[0]), 'and a note says the game\'s day was not made (an administrator\'s own stop raises none)');
      results.push(st().durations.kenoPredictor.length === 7 && st().queue.length === 1 && st().queue[0].key === 'pick3Predictor' && !st().retry.kenoPredictor,
        'its duration is not kept as a usual one, the next game is still queued, nothing starts the stopped game again');
      st().queue.length = 0;
      // only the lottery games are stopped: the crypto run is reported and left to an administrator
      fs.unlinkSync(path.join(dir, 'config', 'scheduler.disabled'));
      st().durations.cryptoPredictor = [600, 700, 900, 1200, 1000, 800, 650];
      game('cryptoPredictor', { pid: 4343 });
      c.checkSlow(st().current, 'cryptoPredictor', '', t0 + 90000 * 1000, deps);
      results.push(kills.length === 2 && !st().current.hungAt && !/stops it itself/.test(notes.slice(-1)[0].body[0]), 'a job without stopWhenSlow is never stopped by the schedule');
      // a run whose PID is not known yet cannot be stopped: nothing is recorded as hung
      game('lottoPredictor', { pid: null });
      c.checkSlow(st().current, 'lottoPredictor', '', t0 + 90000 * 1000, deps);
      results.push(kills.length === 2 && !st().current.hungAt, 'no PID yet: not marked as stopped');
      fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
      st().current = null;
    }

    // --- after the stop (review, 16 Nov 2026): the stopped run's process group is watched until it is gone ---
    {
      const { execFileSync: run } = require('child_process');
      const kills = [];
      const found = [];
      let members = [5001, 5002];
      const deps = { kill: (pid, signal) => { kills.push(`${pid} ${signal}`); }, members: () => members, checkout: () => found.shift() || null };
      // the stop block above ended a run that had its SIGKILL already: remembered as such
      results.push(st().stopped && st().stopped.pid === 4242 && st().stopped.key === 'kenoPredictor' && st().stopped.killedAt === t0 + 7262 * 1000,
        `a run the schedule stopped is remembered when it is finished: ${JSON.stringify(st().stopped)}`);
      // the usual course: SIGTERM ends the wrapper, the run is finished at once, a child of it lives on
      fs.unlinkSync(path.join(dir, 'config', 'scheduler.disabled'));
      game('kenoPredictor', { pid: 4444 });
      c.checkSlow(st().current, 'kenoPredictor', '', t0 + 7201 * 1000, { kill: deps.kill, isAlive: () => true, list: () => ['m'] });
      c.finish(null, 'the job disappeared without writing an exit code (killed?)', t0 + 7202 * 1000);
      const s0 = st().stopped;
      results.push(kills.join() === '-4444 SIGTERM' && s0.pid === 4444 && s0.at === t0 + 7201 * 1000 && s0.killedAt === null && st().current === null && s0.members.join() === 'm',
        `finished before the minute is over: the group is remembered, not yet killed: ${JSON.stringify(s0)}`);
      notes.length = 0;
      results.push(c.reapStopped(t0 + 7230 * 1000, deps) === 'waiting' && kills.length === 1, 'inside the minute: nothing more is sent');
      // the queue waits for it
      st().queue.push({ key: 'pick3Predictor', reason: 'run by hand', manual: true });
      c.startNext();
      results.push(st().current === null && st().queue.length === 1, 'the next game does not start while the stopped one has processes left');
      results.push(c.reapStopped(t0 + 7262 * 1000, deps) === 'killed' && kills.join() === '-4444 SIGTERM,-4444 SIGKILL' && st().stopped.killedAt === t0 + 7262 * 1000,
        `a minute after the stop with a process left: SIGKILL to the group: ${kills.join()}`);
      results.push(c.reapStopped(t0 + 7280 * 1000, deps) === 'left' && kills.length === 2 && notes.length === 0, 'killed once, and no note inside the next minute');
      // a restart in between keeps it
      results.push(JSON.parse(fs.readFileSync(path.join(dir, 'config', 'scheduler.json'), 'utf-8')).stopped.pid === 4444, 'the stopped group is in the state file: a restart goes on watching it');
      members = [];
      results.push(c.reapStopped(t0 + 7290 * 1000, deps) === 'gone' && st().stopped === null && notes.length === 0, 'its processes gone and the checkout clean: forgotten, no note');
      results.push(c.reapStopped(t0 + 7300 * 1000, deps) === null, 'nothing stopped: nothing to do');
      // it never goes: said once, and the queue stops waiting (the lock rule is what still holds it)
      members = [5003];
      st().stopped = { pid: 4445, key: 'kenoPredictor', name: 'Morning predictor: Keno', at: t0, killedAt: t0 + 60000, reportedAt: null };
      c.reapStopped(t0 + 121000, deps);
      c.reapStopped(t0 + 131000, deps);
      results.push(notes.length === 1 && notes[0].level === 'error' && /a process of it is still there/.test(notes[0].title) && /5003/.test(notes[0].body[0]) && st().stopped.reportedAt === t0 + 121000,
        `still there a minute after SIGKILL: one error note naming the process: ${notes[0] && notes[0].body[0]}`);
      // with the schedule off nothing is killed
      st().stopped = { pid: 4446, key: 'kenoPredictor', name: 'Morning predictor: Keno', at: t0, killedAt: null, reportedAt: null };
      fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
      results.push(c.reapStopped(t0 + 600000, deps) === 'left' && kills.length === 2, 'with the schedule off the stopped group is not killed');
      fs.unlinkSync(path.join(dir, 'config', 'scheduler.disabled'));
      // a stop during a write: what it left in the checkout is named
      members = [];
      notes.length = 0;
      found.push(['git\'s index lock (.git/index.lock) was left behind', '2 tracked files have changes that are not committed (a, b)']);
      c.reapStopped(t0 + 600000, deps);
      results.push(notes.length === 1 && notes[0].kind === 'job-failed' && notes[0].level === 'warning' && /stopped during a write/.test(notes[0].title) && notes[0].body.length === 2,
        'an index lock or uncommitted tracked files after the stop: one warning that names them');
      results.push(c.checkoutAfterStop() === null, 'a folder without git and without an index lock: nothing to say');
      fs.mkdirSync(path.join(dir, '.git'), { recursive: true });
      fs.writeFileSync(path.join(dir, '.git', 'index.lock'), '');
      const lockLines = c.checkoutAfterStop();
      results.push(Array.isArray(lockLines) && /index\.lock/.test(lockLines[0]), 'an index lock left behind is found');
      fs.rmSync(path.join(dir, '.git'), { recursive: true, force: true });
      st().queue.length = 0;

      // the real thing: a process group that ignores SIGTERM is found by its group id, and only what started before the stop
      const pidFile = path.join(dir, 'deaf.pid');
      run('setsid', ['--fork', 'bash', '-c', `trap '' TERM; echo $$ > ${pidFile}; sleep 60 & wait`], { stdio: 'ignore' });
      let pgid = null;
      for (let i = 0; i < 40 && !pgid; i += 1) {
        try { pgid = Number(fs.readFileSync(pidFile, 'utf-8').trim()) || null; } catch (e) { /* not yet */ }
        if (!pgid) run('sleep', ['0.1']);
      }
      run('sleep', ['0.3']);
      const known = c.groupList(pgid);
      const live = c.groupMembers(pgid, known);
      results.push(live.length === 2 && live.includes(pgid), `the wrapper and its child are the group's members: ${live.join()} of ${pgid}`);
      results.push(c.groupMembers(pgid, []).length === 0 && c.groupMembers(pgid, known.map((m) => ({ pid: m.pid, start: `${m.start}1` }))).length === 0,
        'only the processes that were there at the stop count: the same PID with another start time is another process, never killed');
      process.kill(-pgid, 'SIGTERM');
      run('sleep', ['0.3']);
      st().stopped = { pid: pgid, key: 'kenoPredictor', name: 'Morning predictor: Keno', at: Date.now(), members: known, killedAt: null, reportedAt: null };
      const first = c.reapStopped(Date.now() + 61000, { checkout: () => null });
      run('sleep', ['0.5']);
      const second = c.reapStopped(Date.now() + 62000, { checkout: () => null });
      results.push(first === 'killed' && second === 'gone' && st().stopped === null && c.groupList(pgid).length === 0,
        `a group that ignores SIGTERM is ended by the SIGKILL a minute later: ${first}, ${second}`);
      fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
    }

    // --- the last lottery run going without finishing starts the weekly plans too (review, 16 Nov 2026) ---
    {
      const saturday = at('2026-11-21T07:30:00');
      const id = c.planTriggerAt(CHAIN_PLAN, {}, saturday);
      const fresh = () => { st().occurrences = {}; st().queue.length = 0; st().retry = {}; st().current = null; };
      fs.unlinkSync(path.join(dir, 'config', 'scheduler.disabled'));
      // removed from the queue
      fresh();
      st().queue.push({ key: 'pick3Predictor', reason: 'schedule 06:00', manual: false }, { key: 'vikinglottoPredictor', reason: 'schedule 06:00', manual: false });
      copy.cancelQueued('pick3Predictor', 'admin', saturday);
      results.push(!st().occurrences[id], 'a game removed from the queue with another still queued: the weekly plans wait');
      copy.cancelQueued('vikinglottoPredictor', 'admin', saturday);
      results.push(Boolean(id) && Boolean(st().occurrences[id]) && /removed from the queue/.test(st().occurrences[id].reason || JSON.stringify(st().occurrences[id])),
        `the last game removed from the queue: the weekly plans start, not at the 23:00 net: ${JSON.stringify(st().occurrences[id])}`);
      // a job that is no lottery game changes nothing
      fresh();
      st().queue.push({ key: 'cryptoPredictor', reason: 'x', manual: false });
      copy.cancelQueued('cryptoPredictor', 'admin', saturday);
      results.push(!st().occurrences[id], 'the crypto run removed from the queue starts no plan');
      // another weekday: nothing
      fresh();
      st().queue.push({ key: 'pick3Predictor', reason: 'schedule 06:00', manual: false });
      copy.cancelQueued('pick3Predictor', 'admin', at('2026-11-18T07:30:00'));
      results.push(Object.keys(st().occurrences).length === 0, 'on a weekday without a plan nothing starts');
      // a retry dropped as hours late
      fresh();
      st().retry.kenoPredictor = { attempt: 2, of: 6, at: saturday.getTime() - 4 * HOUR, since: saturday.getTime() - 6 * HOUR };
      c.startRetries(saturday.getTime());
      results.push(!st().retry.kenoPredictor && Boolean(st().occurrences[id]), 'the last waiting retry dropped as hours late: the weekly plans start');
      fresh();
      st().retry.kenoPredictor = { attempt: 2, of: 6, at: saturday.getTime() - 4 * HOUR, since: saturday.getTime() - 6 * HOUR };
      st().retry.lottoPredictor = { attempt: 2, of: 6, at: saturday.getTime() + HOUR, since: saturday.getTime() - 6 * HOUR };
      c.startRetries(saturday.getTime());
      results.push(!st().occurrences[id] && Boolean(st().retry.lottoPredictor), 'a late retry dropped while another game still waits for its try: the plans wait');
      // with the schedule off nothing starts
      fresh();
      fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
      st().queue.push({ key: 'pick3Predictor', reason: 'schedule 06:00', manual: false });
      copy.cancelQueued('pick3Predictor', 'admin', saturday);
      results.push(!st().occurrences[id], 'with the schedule off the plans are not started');
      fresh();
      st().history = st().history.filter((h) => !h.dry);
    }

    // --- the Council rows follow the morning's lottery predictors (17 Nov 2026) ---
    {
      const job = copy.JOBS.find((j) => j.key === 'councilRows');
      results.push(job && job.script === 'CouncilRows.py' && job.after === copy.LOTTERY && !job.daily && !job.plan && job.wake === 'cpu' && job.unreachableExit === 75,
        'the Council rows are a job of their own: no slot and no plan, started after the lottery games');
      results.push(/lottery predictors have ended/.test(copy.status().jobs.find((j) => j.key === 'councilRows').next), 'its trigger is said in words on the Jobs page');
      {
        const record = copy.JOBS.find((j) => j.key === 'backtestRecord');
        results.push(record && record.script === 'BacktestRecord.py' && record.after === copy.LOTTERY && !record.daily && !record.plan && !record.wake
          && record.args.join(' ') === '--budget 7200' && copy.JOBS.indexOf(record) === copy.JOBS.indexOf(job) + 1,
          'the backtest record (22 Nov 2026) follows the lottery games too, behind the Council rows, with its two hours');
      }
      const fresh = () => { st().occurrences = {}; st().queue.length = 0; st().retry = {}; st().current = null; };
      const queuedKeys = () => st().queue.map((q) => q.key).join();
      // with the schedule off nothing follows
      fresh();
      game('kenoPredictor');
      c.finish(0, null, t0 + 1000);
      results.push(queuedKeys() === '', 'with the schedule off nothing is queued behind the games');
      fs.unlinkSync(path.join(dir, 'config', 'scheduler.disabled'));
      // a game ending with another still queued: not yet
      fresh();
      st().queue.push({ key: 'pick3Predictor', reason: 'schedule 06:00', manual: false });
      game('kenoPredictor');
      c.finish(0, null, t0 + 1000);
      results.push(queuedKeys() === 'pick3Predictor', 'a game ending with another still queued: the Council rows wait');
      // the last game ending: queued, and ahead of the weekly plan it also starts
      fresh();
      game('vikinglottoPredictor');
      c.finish(0, null, t0 + 1000);
      results.push(queuedKeys() === 'councilRows,backtestRecord' && /^after the lottery predictions/.test(st().queue[0].reason) && st().queue[0].manual === false,
        `the last game ending queues the Council rows: ${JSON.stringify(st().queue)}`);
      // a failed last game still lets the others' rows be made; a run without data made nothing to read
      fresh();
      game('vikinglottoPredictor');
      c.finish(1, null, t0 + 1000);
      results.push(queuedKeys() === 'councilRows,backtestRecord', 'the last game failing: the other games\' rows are still asked');
      fresh();
      game('vikinglottoPredictor');
      c.finish(74, null, t0 + 1000);
      results.push(queuedKeys() === '' && Boolean(st().retry.vikinglottoPredictor), 'a run without data queues nothing: its retry does when it ends');
      // a game waiting for its retry is not waited for
      fresh();
      st().retry.lottoPredictor = { attempt: 1, of: 6, at: Date.now() + HOUR, since: t0 };
      game('vikinglottoPredictor');
      c.finish(0, null, t0 + 1000);
      results.push(queuedKeys() === 'councilRows,backtestRecord', 'another game waiting for its retry does not hold the Council rows');
      // already queued: not twice; the crypto run never queues it
      game('pick3Predictor');
      c.finish(0, null, t0 + 1000);
      results.push(queuedKeys() === 'councilRows,backtestRecord', 'not queued twice');
      fresh();
      game('cryptoPredictor');
      c.finish(0, null, t0 + 1000);
      results.push(queuedKeys() === '', 'the crypto run queues no Council rows');
      // the last game leaving the queue without running
      fresh();
      st().queue.push({ key: 'pick3Predictor', reason: 'schedule 06:00', manual: false });
      copy.cancelQueued('pick3Predictor', 'admin', at('2026-11-18T07:30:00'));
      results.push(queuedKeys() === '', 'a game removed from the queue made nothing: no Council rows are queued for it');
      // no model answered: one warning, not the generic failure
      fresh();
      notes.length = 0;
      begin({ key: 'councilRows', name: 'Council rows' });
      c.finish(75, null, t0 + 1000);
      results.push(notes.length === 1 && notes[0].level === 'warning' && /no Council model answered/.test(notes[0].title) && notes[0].dedupe === 'unreachable:councilRows',
        `status 75 is said as what it is: ${notes[0] && notes[0].title}`);
      notes.length = 0;
      begin({ key: 'councilRows', name: 'Council rows' });
      c.finish(1, null, t0 + 1000);
      results.push(notes.length === 1 && notes[0].level === 'error' && /ended with exit code 1/.test(notes[0].title), 'any other failure is the usual note');
      // the wake: only with the DL360 set up, and with its own settings
      results.push(c.wakeLines(job, '/tmp/x.log').length === 0 && c.wakeLines(copy.JOBS.find((j) => j.key === 'kenoPredictor'), '/tmp/x.log').length === 0,
        'without the DL360 set up nothing is woken; a job without wake never wakes it');
      c.useWorker(() => ({ kind: 'cpu', host: 'root@h' }), () => ({ WORKER_HOST: 'root@h', WORKER_ILO_ENV: '/e/ilo.env', WORKER_IDLE_MINUTES: '10' }));
      const wake = c.wakeLines(job, '/tmp/x.log');
      results.push(wake.length === 1 && /^WORKER_HOST='root@h' WORKER_ILO_ENV='\/e\/ilo\.env' WORKER_IDLE_MINUTES='10' bash '.*worker\/power\.sh' wake >> '\/tmp\/x\.log' 2>&1 \|\| echo/.test(wake[0]),
        `the DL360 is switched on first, a failed wake does not stop the job: ${wake[0]}`);
      c.useWorker(() => null);
      fresh();
      notes.length = 0;
      st().history = st().history.filter((h) => h.key !== 'councilRows');
      fs.writeFileSync(path.join(dir, 'config', 'scheduler.disabled'), '');
    }

    // --- the one-time move from one job to seven (splitLottery) ---
    {
      results.push(typeof st().lotterySplitAt === 'string', 'the first start with this schedule records the move');
      const lotteryKeys = copy.JOBS.filter((j) => j.group === copy.LOTTERY).map((j) => j.key);
      const morning = at('2026-11-17T09:00:00');
      st().lotterySplitAt = null;
      st().occurrences = { 'predictor@2026-11-17': { at: 'x' } };
      st().retry = { predictor: { attempt: 2, of: 6, at: t0, since: t0 - HOUR } };
      st().queue = [{ key: 'predictor', reason: 'retry 1 of 6', manual: false, retry: 1 }, { key: 'gone', reason: 'x' }, { key: 'hyperoptBoost', reason: 'x', manual: true }];
      c.splitLottery(morning);
      results.push(lotteryKeys.every((k) => st().occurrences[`${k}@2026-11-17`]) && !st().occurrences['cryptoPredictor@2026-11-17'],
        'deployed after the morning\'s one run: the seven games\' slots of that day are handled, the other jobs\' are not touched');
      results.push(!c.dueAt(st().occurrences, morning).some((d) => lotteryKeys.includes(d.plan)), 'so none of them starts as a catch-up');
      results.push(!st().retry.predictor && lotteryKeys.every((k) => st().retry[k].attempt === 2 && st().retry[k].at === t0), 'a waiting retry of the one run becomes one per game');
      results.push(st().queue.map((q) => q.key).join() === [...lotteryKeys, 'hyperoptBoost'].join() && st().queue[0].retry === 1,
        `a queued one run becomes seven, a job this schedule does not have leaves the queue: ${st().queue.map((q) => q.key).join()}`);
      st().occurrences = { 'predictor@2026-11-18': { at: 'x' } };
      c.splitLottery(at('2026-11-18T09:00:00'));
      results.push(Object.keys(st().occurrences).length === 1, 'once: a later start marks nothing');
      // deployed before the morning's run: nothing to mark, the seven run at 06:00
      st().lotterySplitAt = null;
      st().occurrences = { 'predictor@2026-11-16': { at: 'x' } };
      st().retry = {};
      st().queue = [];
      c.splitLottery(at('2026-11-17T05:00:00'));
      results.push(Object.keys(st().occurrences).length === 1 && c.dueAt(st().occurrences, at('2026-11-17T06:00:00')).filter((d) => lotteryKeys.includes(d.plan)).length === 7,
        'deployed before 06:00: nothing is marked and the seven games are due at their slot');
      st().occurrences = {};
    }

    // a notifier that throws costs nothing
    c.setNotifier(() => { throw new Error('broken'); });
    begin();
    c.finish(1, null, t0 + 1000);
    results.push(st().current === null && st().history[0].exit === 1, 'a broken notifier does not stop a run from being recorded');
    console.log = quiet;
    for (let i = 0; i < results.length; i += 2) ok(results[i], results[i + 1]);
  } finally {
    console.log = quiet;
    if (copy) copy.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

console.log(`jobs.js: ${checks} checks passed (${jobs.JOBS.length} jobs, ${CHAIN_KEYS.length} in the weekly chain)`);
