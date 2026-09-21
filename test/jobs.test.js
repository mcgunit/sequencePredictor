// What the schedule in jobs.js must keep doing (see the header there).
// Pure logic only: no job is started, nothing is written, so this is safe to
// run anywhere. Run with: npm test
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const jobs = require('../jobs');
const { dueAt, weekId, dayId, lastWeekdaySlot, nextDaily, CHAIN_KEYS, CHAIN_PLAN,
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
ok(jobs.JOBS.length === 7, 'seven jobs: the daily predictor and the six of the weekly chain');
ok(byKey.predictor.script === 'Predictor.py' && byKey.predictor.args.join(' ') === '-a true',
  'the daily job is Predictor.py -a true, exactly as runPredictor.sh had it');
ok(byKey.predictor.daily.hour === 9 && byKey.predictor.daily.minute === 0, 'the predictor keeps its 09:00 slot');
ok(CHAIN_KEYS.join(',') === 'hyperoptStatistics,hyperoptBoost,hyperoptRLTicket,hyperoptEnsemble,hyperoptQuantum,trainMetaLearner',
  'the weekly chain keeps the order runHyperopt.sh documented');
ok(CHAIN_KEYS.indexOf('hyperoptQuantum') < CHAIN_KEYS.indexOf('trainMetaLearner'),
  'the quantum tuner must run before the retrain - that is why the retrain is weekly at all');
ok(CHAIN_KEYS[CHAIN_KEYS.length - 1] === 'trainMetaLearner', 'the meta-learner retrain stays last');

// The shell scripts stay as the hand-run path, so they must not drift from
// the schedule: same scripts, same order, or one of the two is wrong.
const shell = (name) => fs.readFileSync(path.join(__dirname, '..', name), 'utf-8');
const runHyperopt = shell('runHyperopt.sh');
ok(shell('runPredictor.sh').includes('python3 Predictor.py -a true'),
  'runPredictor.sh still runs what the predictor job runs');
const orderInShell = CHAIN_KEYS.map((key) => runHyperopt.indexOf(`python3 ${byKey[key].script}`));
ok(orderInShell.every((i) => i > 0), 'every chain job appears in runHyperopt.sh');
ok(orderInShell.every((pos, i) => i === 0 || pos > orderInShell[i - 1]),
  'the chain runs in runHyperopt.sh order - if one moves, both must move');

jobs.JOBS.forEach((job) => {
  ok(fs.existsSync(path.join(__dirname, '..', job.script)), `${job.key}: ${job.script} exists in the repo`);
  ok(typeof job.log === 'string' && job.log.endsWith('.log'), `${job.key}: has a log file`);
  ok(typeof job.description === 'string' && job.description.length > 20, `${job.key}: says what it does on the Jobs page`);
});
ok(new Set(jobs.JOBS.map((j) => j.log)).size === jobs.JOBS.length, 'every job writes to its own log');
ok(jobs.JOBS.filter((j) => j.daily).length === 1, 'only the predictor runs on a clock');

// --- the daily predictor ----------------------------------------------------
ok(plans({}, at('2026-09-23T09:00:00')).includes('predictor'), 'due at its slot');
ok(reasonFor({}, at('2026-09-23T09:00:00'), 'predictor').startsWith('schedule'), 'on time, so not reported as a catch-up');
ok(reasonFor({}, at('2026-09-23T12:00:00'), 'predictor').startsWith('catch-up'), 'three hours late is a catch-up');
ok(!plans({ 'predictor@2026-09-23': { at: 'x' } }, at('2026-09-23T12:00:00')).includes('predictor'),
  'an occurrence that has been handled is never started twice');
// The case the catch-up exists for: the box was down over the night and comes
// back before the next slot. Yesterday's run must still be recognised.
ok(plans({}, at('2026-09-24T04:00:00')).includes('predictor'), 'a run missed yesterday is caught up after midnight');
ok(dueAt({}, at('2026-09-24T04:00:00')).find((d) => d.plan === 'predictor').occurrence === 'predictor@2026-09-23',
  'and it is yesterday\'s occurrence that is caught up, not today\'s');
ok(!plans({}, at('2026-09-24T08:00:00')).includes('predictor'),
  'past the 20 h window the morning run supersedes it, so it is dropped');
ok(!plans({}, at('2026-09-23T08:59:00')).includes('predictor'), 'nothing is due before the slot');

// --- the weekly tuning chain ------------------------------------------------
// Normally started by Saturday's predictor finishing (maybeTriggerChain);
// dueAt is only the safety net for a week that would otherwise be skipped.
const SAT = '2026-09-26';                                  // a Saturday
ok(lastWeekdaySlot(at(`${SAT}T20:00:00`), 6, 9, 0).getTime() === at(`${SAT}T09:00:00`).getTime(),
  'the chain is anchored to Saturday 09:00, the predictor slot that triggers it');
ok(!plans({}, at(`${SAT}T22:00:00`)).includes(CHAIN_PLAN), 'no catch-up while Saturday\'s run could still be going');
ok(plans({}, at(`${SAT}T23:30:00`)).includes(CHAIN_PLAN), 'by Saturday night a chain that never started is caught up');
ok(plans({}, at('2026-09-27T12:00:00')).includes(CHAIN_PLAN), 'still caught up on Sunday');
ok(plans({}, at('2026-09-28T08:00:00')).includes(CHAIN_PLAN), 'and until Monday morning');
ok(!plans({}, at('2026-09-28T10:00:00')).includes(CHAIN_PLAN), 'after Monday 09:00 the week is written off, not started late');
ok(!plans({ [`${CHAIN_PLAN}@2026-W39`]: { at: 'x' } }, at('2026-09-27T12:00:00')).includes(CHAIN_PLAN),
  'a chain that already ran this week is not started again');
ok(dueAt({}, at('2026-09-27T12:00:00')).find((d) => d.plan === CHAIN_PLAN).occurrence === `${CHAIN_PLAN}@2026-W39`,
  'Sunday still belongs to the week of the Saturday that anchors it');

// --- the trigger, and why it must not fire before Saturday 09:00 ------------
// A review of this module found the chain could run twice in one weekend: a
// Friday catch-up predictor finishing at 04:30 on Saturday anchors to LAST
// Saturday, so it fired under the previous week's id and left this week's id
// free for the 23:00 safety net to fire the whole six-job chain again.
ok(chainTriggerAt({}, at(`${SAT}T10:30:00`)) === `${CHAIN_PLAN}@2026-W39`,
  "a predictor finishing after Saturday's slot starts this week's chain");
ok(chainTriggerAt({}, at(`${SAT}T04:30:00`)) === null,
  'a run finishing before Saturday 09:00 belongs to no chain - it is Friday\'s run spilling over');
ok(chainTriggerAt({}, at(`${SAT}T23:59:00`)) === `${CHAIN_PLAN}@2026-W39`,
  'late on Saturday it is still this week');
ok(chainTriggerAt({ [`${CHAIN_PLAN}@2026-W39`]: { at: 'x' } }, at(`${SAT}T10:30:00`)) === null,
  'a chain that already ran this week is never triggered again');
ok(chainTriggerAt({}, at('2026-09-27T10:30:00')) === null, 'only Saturday triggers the chain');
// The whole weekend, in order: the spillover records nothing, so the safety
// net fires exactly one chain and the trigger then has nothing left to do.
const weekend = {};
ok(chainTriggerAt(weekend, at(`${SAT}T04:30:00`)) === null, 'weekend: the 04:30 spillover starts nothing');
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
const mixed = [{ key: 'predictor', manual: false }, { key: 'hyperoptBoost', manual: true }];
ok(allowedQueue(mixed, 'on').length === 2, 'on mode starts everything queued');
ok(allowedQueue(mixed, 'dry').length === 1 && allowedQueue(mixed, 'dry')[0].manual === true,
  'dry mode keeps only what an administrator started by hand');
ok(allowedQueue(mixed, 'dry') !== mixed && allowedQueue(mixed, 'on') !== mixed, 'the queue is never aliased back');

// --- occurrence identity ----------------------------------------------------
// Ids are the only thing standing between "ran once" and "ran twice", so they
// must not drift with the year boundary.
ok(weekId(at('2026-01-01T12:00:00')) === '2026-W01', 'ISO week of a Thursday 1 January');
ok(weekId(at('2027-01-03T12:00:00')) === '2026-W53', 'the Sunday of a week that started in the old year');
ok(dayId(at('2026-09-05T23:00:00')) === '2026-09-05', 'day ids are zero padded');
ok(nextDaily(at('2026-09-23T10:00:00'), { hour: 9, minute: 0 }).getTime() === at('2026-09-24T09:00:00').getTime(),
  'after today\'s slot the next one is tomorrow');

console.log(`jobs.js: ${checks} checks passed (${jobs.JOBS.length} jobs, ${CHAIN_KEYS.length} in the weekly chain)`);
