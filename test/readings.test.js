'use strict';
// The week's reading (readings.js) and the jobs.js hook that writes it: what a reading holds on hand-made records
// (Q0's row tests as the ranking given ran them, with the band as context; Q2, the feature-control variants, the market
// rows' intervals, every part from another week named), the atomic, idempotent write, the CLI, and the hook - which
// Sunday a finished run writes, that a failing reading is logged and changes nothing else, and, on a copy of jobs.js
// in a temp dir with stand-in modules, that finish() writes it after the run is recorded whatever its exit code. No
// job is started and nothing is written outside the temp dirs.
//   node test/readings.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const admin = require('../admin');
const readings = require('../readings');
const jobs = require('../jobs');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };
const REPO = path.join(__dirname, '..');

// --- hand-made records: the week of Sunday 18 Oct 2026 -----------------------------------------------------------
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'readings-'));
const controlsDir = path.join(tmp, 'controls');
const databaseDir = path.join(tmp, 'database');
['results', 'discrimination', 'features', 'markets'].forEach((d) => fs.mkdirSync(path.join(controlsDir, d), { recursive: true }));
fs.mkdirSync(databaseDir);
const write = (dir, rel, obj) => fs.writeFileSync(path.join(dir, rel), JSON.stringify(obj));
const row = (name, draws, hits) => ({ name, draws, avg_hits: hits, profit_bound: null, best_hits: 3, profit_per_bet: null, bets: 0 });
// Q0's row tests as the daily report writes them (src/RowChance.py, 27 Oct 2026)
const tested = (draws, notLive, mean, p, pHolm, fresh) => ({ chance: { draws, notLive, mean, null: 0.8, nullSd: 0.1, p, pHolm, inFamily: pHolm !== null, last: '2026-10-17',
  ...(fresh ? { fresh: { since: fresh[1], flagDraws: 10, flagP: 0.03, flagMean: 1.4, flagNull: 0.8, verdictLength: 10, draws: fresh[2], last: '2026-10-17', state: fresh[0],
    verdictAt: fresh[0] === 'running' ? null : '2026-10-10', decidedOn: fresh[0] === 'running' ? null : '2026-10-12', p: fresh[0] === 'running' ? null : 0.004, mean: 1.5, null: 0.8 } } : {}) } });
const testOf = (method, over) => ({ method, alpha: 0.05, verdictAlpha: 0.01, minDraws: 10, family: 0, liveMax: 0, under: 0, expected: 0, holds: 0, followed: 0, running: 0, stopped: 0,
  confirmed: 0, notConfirmed: 0, checkedThrough: '2026-10-18', stored: 'written', ...over });
write(databaseDir, 'modelPerformance.json', { generatedAt: '2026-10-18T08:47:31', games: {
  // A's live record (src/Liveness.py, 27 Oct 2026): 11 of its 14 draws saved before the sales close, 2 of them rewritten later
  lotto: { metric: 'avg_hits', minDrawsForRanking: 10, chance: testOf('exact', { family: 3, liveMax: 80, under: 2, expected: 0.14, holds: 1, followed: 1, running: 1, confirmed: 1 }),
    models: [{ ...row('A Model', 14, 1.3), live_draws: 11, live: { draws: 11, recovered: 2, not_live: 3, avg_hits: 1.1 }, ...tested(11, 3, 1.1, 0.03, 0.06, ['running', '2026-10-11', 3]) },
      { ...row('B Model', 30, 1.2), live_draws: 30, live: { draws: 30, recovered: 0, not_live: 0, avg_hits: 1.2 }, ...tested(30, 0, 1.2, 0.004, 0.012, ['confirmed', '2026-09-13', 14]) },
      { ...row('C Model', 90, 1.0), ...tested(80, 10, 0.9, 0.2, 0.2) }, { ...row('Young Model', 3, 2), ...tested(3, 0, 2, 0.02, null) }] },
  crypto: { metric: 'avg_hits', minDrawsForRanking: 10, chance: testOf('permutation', { family: 1, liveMax: 12, under: 1, expected: 0.05 }),
    models: [{ ...row('L Model', 20, 1.25), ...tested(12, 8, 1.25, 0.02, 0.02) }, { ...row('M Model', 50, 0.5), ...tested(9, 41, 0.5, 0.9, null) }],
    since: { date: '2026-09-27', minDrawsForRanking: 10, models: [{ name: 'L Model', draws: 12 }] } },
} });
const q0 = (game, mode, best, sd, at) => ({ report: { rows_compared: 8, row_mean: 0.78, best_row_mean: best, best_row_sd_across_seeds: sd, selection_gain: 0.1, random_baseline: 0.79,
  per_row_mean: { 'C Model': 0.7 }, game, mode, seeds: [1, 2, 3], days: 120, metric: 'hits per draw', winner_per_seed: [], generated_at: at }, per_seed: [] });
write(controlsDir, 'results/lotto-synthetic.json', q0('lotto', 'synthetic', 0.95, 0.0878, '2026-10-18T10:11:55'));
write(controlsDir, 'results/lotto-shuffled.json', q0('lotto', 'shuffled', 0.8778, 0.0293, '2026-10-18T22:45:39'));   // Mon 00:45 Brussels: still the 18 Oct week
write(controlsDir, 'results/crypto-shuffled.json', q0('crypto', 'shuffled', 0.7, 0.05, '2026-10-18T22:09:59'));
const q2 = (game, real, threshold, at, windows = 496) => ({ game, window: 10, draws: 1000, repetitions: 3, seeds: [1, 2, 3], generated_at: at,
  shape: { windows_train: 1486, windows_test: windows, features: 45 },
  verdict: { real_max_of_suite_mean: real, null_max_of_suite_mean: 0.509, null_max_of_suite_sd: 0.0306, shuffled_max_of_suite_mean: 0.5419, null_threshold_2sd: threshold,
    above_null_band: real > threshold, survives_marginals_control: real > 0.5419, too_few_draws: false, windows_test: windows } });
write(controlsDir, 'discrimination/lotto-w10.json', q2('lotto', 0.5902, 0.5703, '2026-10-18T08:48:16'));
write(controlsDir, 'discrimination/keno-w10.json', q2('keno', 0.58, 0.61, '2026-10-11T08:52:00'));   // last week's: this week's run did not replace it
// a version-2 record (RandomnessDiscrimination.compute_verdict from 1 Nov 2026 on): its own status, the threshold's error,
// the shuffled control's own bound
write(controlsDir, 'discrimination/shares-w10.json', { game: 'shares', window: 10, draws: 1000, repetitions: 20, generated_at: '2026-10-18T09:30:00',
  shape: { windows_train: 1486, windows_test: 496, features: 50 },
  verdict: { version: 2, status: 'above band, not shuffled', real_max_of_suite_mean: 0.6912, real_max_of_suite_sd: 0.048, null_max_of_suite_mean: 0.5791,
    null_max_of_suite_sd: 0.038, shuffled_max_of_suite_mean: 0.6252, shuffled_max_of_suite_sd: 0.063, band_k: 2.193, shuffled_k: 2.193, null_threshold: 0.6624,
    null_threshold_se: 0.0161, shuffled_threshold: 0.7634, above_null_band: true, survives_marginals_control: false, too_few_draws: false, few_windows: 250, windows_test: 496 } });
const variant = (fits, ratio, above) => ({ noise: { share_of_train_importance: 0.3117, noise_to_real_ratio: ratio, train_t: 2.5, heldout_band: 0.008 },
  columns: { a: 0, b: 0, c: 0 }, real_above_noise_band: above, heldout_auc_without_noise: 0.5012, heldout_auc_with_noise_mean: 0.4987, fits_noise: fits, inconclusive: !fits && ratio > 0.1 });
write(controlsDir, 'features/lotto.json', { game: 'lotto', generated_at: '2026-10-18T09:10:00+00:00', repeats: 3, noise_columns: 3, table: { days: 275, lockbox_days_withheld: 25 },
  variants: { logistic: variant(false, 1.031, []), quantum_kernel: variant(true, 1.2, ['C Model']) } });
write(controlsDir, 'markets/crypto-rows.json', { market: 'crypto', generated_at: '2026-10-18T09:54:09+00:00', days_requested: 250, days_scored: 231, reference: 'GARCH Model',
  lockbox_days_withheld: 19, errors: {}, rows: [
    { name: 'GARCH Model', kind: 'market', days: 231, log_score: -2.21, vs_uniform: { mean: 0.0923, lo: 0.0647, hi: 0.1183, verdict: 'better' }, vs_reference: null },
    { name: 'Chronos Model', kind: 'base', days: 200, log_score: -2.25, vs_uniform: { mean: 0.05, lo: 0.01, hi: 0.09, verdict: 'better' }, vs_reference: { mean: -0.04, lo: -0.08, hi: 0, verdict: 'worse' } },
    { name: 'Uniform', kind: 'baseline', days: 231, log_score: null },
  ] });

// --- buildReading -------------------------------------------------------------------------------------------------
const reading = readings.buildReading({ controlsDir, databaseDir, root: tmp, sunday: '2026-10-18', now: new Date('2026-10-19T00:50:00Z') });
ok(reading.version === admin.READING_VERSION && reading.sunday === '2026-10-18' && reading.by === 'scheduler' && reading.rankingAt === '2026-10-18T08:47:31Z'
  && reading.builtAt === '2026-10-19T00:50:00Z' && reading.controls === 'controls', `the head: version, week, who built it, the ranking read against, the records' folder: ${JSON.stringify({ ...reading, games: undefined })}`);
ok(Object.keys(reading.games).join() === admin.GAMES.join(), 'every game has an entry, with or without records');
const lotto = reading.games.lotto;
ok(lotto.unit === 'draws' && lotto.q0.rule === 'row test' && lotto.q0.at === '2026-10-18T08:47:31Z' && lotto.q0.band.ceiling === 1.126 && lotto.q0.band.from === 'synthetic'
  && lotto.q0.band.seeds === 3 && lotto.q0.band.history === 120 && lotto.q0.modes.shuffled.ceiling === 0.936 && lotto.q0.modes.synthetic.at === '2026-10-18T10:11:55Z',
  `Q0: the row tests' rule and the report's time; the band as context, both controls with their own times: ${JSON.stringify(lotto.q0.band)}`);
ok(JSON.stringify(lotto.q0.test) === '{"method":"exact","family":3,"minDraws":10,"liveMax":80,"under":2,"expected":0.14,"holds":1,"followed":1,"stopped":0,"confirmed":1,"notConfirmed":0,"checkedThrough":"2026-10-18"}',
  `Q0: the game's test as the report ran it: ${JSON.stringify(lotto.q0.test)}`);
ok(lotto.q0.rows.map((r) => `${r.row}:${r.state || '-'}`).join() === 'B Model:confirmed,A Model:running' && lotto.q0.rows[1].live === 11 && lotto.q0.rows[1].mean === 1.1
  && lotto.q0.rows[1].null === 0.8 && lotto.q0.rows[1].p === 0.03 && lotto.q0.rows[1].since === '2026-10-11' && lotto.q0.rows[1].fresh === 3 && lotto.q0.rows[1].of === 10,
  `Q0: the rows the card named (under 5%, followed or judged), each with its own figures beside its p and its new draws: ${JSON.stringify(lotto.q0.rows)}`);
ok(reading.games.crypto.q0.rows.length === 1 && reading.games.crypto.q0.rows[0].row === 'L Model' && reading.games.crypto.q0.rows[0].live === 12 && reading.games.crypto.q0.test.method === 'permutation',
  'Q0: a market row tested on its live days');
ok(admin.stateOf('q0', lotto) === 'confirmed: B · followed: A' && admin.stateOf('q0', reading.games.crypto) === 'confirmed: none · followed: none',
  'Q0\'s week-over-week state: the rows confirmed and followed - never "holds" or "under 5%"');
ok(reading.games.euromillions.q0 === null && reading.games.euromillions.q2 === null && reading.games.euromillions.rows === null, 'a game without records reads null, not "none under"');
ok(lotto.q2.status === 'above' && lotto.q2.real === 0.59 && lotto.q2.shuffled === 0.542 && lotto.q2.threshold === 0.57 && lotto.q2.seeds === 3 && lotto.q2.windowsTest === 496,
  'Q2: real, shuffled and threshold, the verdict, the seeds and the test windows');
ok(lotto.q2.rule === 1 && lotto.q2.thresholdSe === null && lotto.q2.shuffledThreshold === null, 'Q2: a version-1 record is read under rule 1, without the version-2 numbers');
const shares = reading.games.shares.q2;
ok(shares.status === 'above band, not shuffled' && shares.rule === 2 && shares.threshold === 0.662 && shares.thresholdSe === 0.016 && shares.shuffledThreshold === 0.763 && shares.seeds === 20,
  `Q2: a version-2 record keeps its status, its rule, the threshold's error and the shuffled bound: ${JSON.stringify(shares)}`);
ok(lotto.fc.variants.length === 2 && lotto.fc.variants[0].key === 'lr' && lotto.fc.variants[0].verdict === 'inconclusive' && lotto.fc.variants[0].ratio === 1.031
  && lotto.fc.variants[0].auc.join() === '0.501,0.499' && lotto.fc.variants[0].above === undefined && lotto.fc.variants[1].above.join() === 'C Model' && lotto.fc.tableRows === 275,
  `FC: per variant the verdict, the noise ratio, held-out AUC without and with noise, the columns above the band: ${JSON.stringify(lotto.fc.variants)}`);
const cr = reading.games.crypto.rows;
ok(cr.window === 231 && cr.reference === 'GARCH Model' && cr.rows.length === 2 && cr.rows[0].vsReference === undefined && cr.rows[1].vsReference.verdict === 'worse'
  && cr.rows[1].vsUniform.lo === 0.01 && cr.rows[1].days === 200 && cr.rows[0].days === undefined && reading.games.lotto.rows === null,
  'market rows: every scored row against uniform and against GARCH with its interval; days only where they differ from the window');
ok(reading.notThisWeek.length === 1 && reading.notThisWeek[0].game === 'keno' && reading.notThisWeek[0].part === 'q2',
  `a record from another week is named, so the reading never passes it off as this week's: ${JSON.stringify(reading.notThisWeek)}`);
ok(reading.from === '2026-10-11T08:52:00Z' && reading.to === '2026-10-18T22:45:39Z', 'the span of the records read');
assert.throws(() => readings.buildReading({ controlsDir, databaseDir, root: tmp, sunday: '2026-10-19' }), /not a Sunday/); checks += 1;
const noRanking = readings.buildReading({ controlsDir, databaseDir: path.join(tmp, 'none'), root: tmp, sunday: '2026-10-18' });
ok(noRanking.games.lotto.q0 === null && noRanking.rankingAt === null && admin.stateOf('q0', noRanking.games.lotto) === null,
  'without a ranking nothing was tested: no Q0 part, never "none under"');
const before = JSON.parse(JSON.stringify(reading.games.lotto));
delete before.q0.rule;
before.q0.above = [{ row: 'A Model' }];
ok(admin.stateOf('q0', before) === 'band rule', 'a reading written before 27 Oct 2026 (no Q0 rule) reads "band rule": the Q0 weeks start again on Sun 1 Nov 2026');

// --- writeReading ------------------------------------------------------------------------------------------------
const out = path.join(tmp, 'readings');
const first = readings.writeReading(reading, { outDir: out });
const file = path.join(out, '2026-10-18.json');
ok(first.written && first.file === file && fs.existsSync(file) && fs.readdirSync(out).join() === '2026-10-18.json', 'written to <sunday>.json, no temporary file left');
const text = fs.readFileSync(file, 'utf-8');
ok(JSON.stringify(JSON.parse(text)) === JSON.stringify(reading) && text.split('\n').filter((l) => /^ {4}"[a-z0-9]+": \{/.test(l)).length === 11 && first.bytes === Buffer.byteLength(text),
  'the file is the reading, one line per game');
const mtime = fs.statSync(file).mtimeMs;
const again = readings.writeReading({ ...reading, builtAt: '2026-10-20T10:00:00Z' }, { outDir: out });
ok(!again.written && fs.statSync(file).mtimeMs === mtime && fs.readFileSync(file, 'utf-8') === text, 'the same week rebuilt with nothing new leaves the file alone');
const changed = JSON.parse(JSON.stringify(reading));
changed.games.lotto.q2.status = 'within';
ok(readings.writeReading(changed, { outDir: out }).written && JSON.parse(fs.readFileSync(file, 'utf-8')).games.lotto.q2.status === 'within', 'the same week with new content rewrites the same file');
assert.throws(() => readings.writeReading({ ...reading, sunday: '2026-10-17' }, { outDir: out }), /Sunday/); checks += 1;
ok(fs.readdirSync(out).join() === '2026-10-18.json', 'nothing else in the folder');
const loaded = admin.loadReadings(out);
ok(loaded.sundays.join() === '2026-10-18' && loaded.bySunday.get('2026-10-18').games.lotto.q2.status === 'within' && loaded.unreadable.length === 0, 'admin.js reads it back');

// --- the CLI ----------------------------------------------------------------------------------------------------
const said = [];
const complained = [];
const cliOut = path.join(tmp, 'cli');
const code = readings.main(['--controls', controlsDir, '--sunday', '2026-10-18', '--database', databaseDir, '--out', cliOut, '--note', 'backfilled from a copy'], (l) => said.push(l), (l) => complained.push(l));
const cli = JSON.parse(fs.readFileSync(path.join(cliOut, '2026-10-18.json'), 'utf-8'));
ok(code === 0 && cli.by === 'cli' && cli.note === 'backfilled from a copy' && cli.controls === controlsDir && said[0].startsWith('wrote ') && said.some((l) => l.startsWith('  lotto: Q0 2 of 3 under 5% (≈0.14 by chance), 1 hold, 1 followed, 1 confirmed (band 1.126 of 2026-10-18, context)'))
  && said.some((l) => /not from the week of 2026-10-18: keno q2/.test(l)), `the CLI writes the week, says what it read: ${said.slice(0, 3).join(' / ')}`);
ok(said.some((l) => l.includes('Q2 above band, not shuffled 0.691 vs 0.662±0.016 (shuffled 0.625, bound 0.763, 20 seeds, 496 test windows, rule 2)'))
  && said.some((l) => l.includes('Q2 above 0.590 vs 0.570 (shuffled 0.542, 3 seeds, 496 test windows, rule 1)')),
  `the CLI line: the threshold with its error and the shuffled bound under rule 2, the plain figures under rule 1: ${said.filter((l) => /lotto|shares/.test(l)).join(' / ')}`);
ok(readings.main(['--controls', controlsDir, '--sunday', '2026-10-18', '--database', databaseDir, '--out', cliOut, '--note', 'backfilled from a copy'], (l) => said.push(l), (l) => complained.push(l)) === 0
  && said[said.length - 13].startsWith('unchanged '), 'run twice: unchanged');
ok(readings.main(['--controls', controlsDir, '--sunday', '2026-10-17'], () => {}, (l) => complained.push(l)) === 2 && /not a Sunday/.test(complained[complained.length - 1]), 'a day that is not a Sunday is refused');
ok(readings.main(['--sunday', '2026-10-18'], () => {}, (l) => complained.push(l)) === 2 && readings.main(['--controls', controlsDir, '--sunday', '2026-10-18', '--force'], () => {}, (l) => complained.push(l)) === 2
  && readings.main(['--controls', path.join(tmp, 'nowhere'), '--sunday', '2026-10-18'], () => {}, (l) => complained.push(l)) === 2, 'missing, unknown or wrong arguments are refused');

// --- the hook in jobs.js: which Sunday, and that a failure stays a log line ---------------------------------------
const { READING_JOB, readingSundayOf, writeWeekReading, scheduleReading, CONTROL_KEYS } = jobs._internals;
const local = (y, m, d, h, min = 0) => new Date(y, m - 1, d, h, min).getTime();   // the schedule's own clock
const q0Run = (startedAt) => ({ key: READING_JOB, name: 'Rows', startedAt });
// the Q0 band is monthly since 1 Nov 2026 and runs first on its Sunday: the plan's last job, which every Sunday runs, is
// the market rows report, so the reading is written every week
ok(READING_JOB === 'marketRows' && READING_JOB === CONTROL_KEYS[CONTROL_KEYS.length - 1] && CONTROL_KEYS[0] === 'nullControls',
  'the reading follows the controls plan\'s last job, the market rows report; the monthly band is the plan\'s first');
ok(readingSundayOf(q0Run(local(2026, 11, 8, 12))) === '2026-11-08' && readingSundayOf(q0Run(local(2026, 11, 15, 11))) === '2026-11-15',
  'a Sunday that is not the month\'s first (no band run) still writes its reading');
ok(readingSundayOf(q0Run(local(2026, 11, 2, 2))) === '2026-11-01' && readingSundayOf({ key: 'nullControls', startedAt: local(2026, 11, 1, 11) }) === null,
  'on a first Sunday the reading waits for the market rows behind the band (into Monday): the band itself writes none');
ok(readingSundayOf(q0Run(local(2026, 10, 18, 12))) === '2026-10-18' && readingSundayOf(q0Run(local(2026, 10, 19, 1))) === '2026-10-18'
  && readingSundayOf(q0Run(local(2026, 10, 20, 8, 59))) === '2026-10-18', 'Sunday\'s run, and its catch-up until Tuesday 09:00, write that Sunday\'s reading');
ok(readingSundayOf(q0Run(local(2026, 10, 20, 9, 30))) === null && readingSundayOf(q0Run(local(2026, 10, 22, 12))) === null,
  'a reading job started later in the week is not that weekend\'s control: no reading');
ok(readingSundayOf({ key: 'predictor', startedAt: local(2026, 10, 18, 9) }) === null && readingSundayOf({ key: 'randomnessDiscrimination', startedAt: local(2026, 10, 18, 11) }) === null
  && readingSundayOf(null) === null && readingSundayOf({ key: READING_JOB }) === null, 'no other job writes one');
const lines = [];
const calls = [];
const standIn = { buildReading: (o) => { calls.push(['build', o]); return { sunday: o.sunday }; }, writeReading: (r, o) => { calls.push(['write', r, o]); return { file: path.join(o.root, 'data/readings/x.json'), written: true, bytes: 9 }; } };
const result = writeWeekReading(q0Run(local(2026, 10, 18, 12)), { readings: standIn, log: (l) => lines.push(l), root: tmp });
ok(result.written && calls.length === 2 && calls[0][1].sunday === '2026-10-18' && calls[0][1].root === tmp && calls[0][1].by === 'scheduler' && calls[1][2].root === tmp
  && /reading for Sunday 2026-10-18: written/.test(lines[0]), `the stand-in is asked for that Sunday's reading under the checkout: ${lines[0]}`);
const broken = { buildReading: () => { throw new Error('boom'); }, writeReading: () => { throw new Error('never'); } };
ok(writeWeekReading(q0Run(local(2026, 10, 18, 12)), { readings: broken, log: (l) => lines.push(l) }) === null && /could not write the week's reading for Sunday 2026-10-18: boom - nothing else is affected/.test(lines[1]),
  'a reading that fails is one log line, never a throw');
ok(writeWeekReading(q0Run(local(2026, 10, 18, 12)), { readings: broken, log: () => { throw new Error('log down'); } }) === null, 'not even when the log fails too');
const deferred = [];
ok(scheduleReading(q0Run(local(2026, 10, 18, 12)), { defer: (fn) => deferred.push(fn), readings: standIn, log: () => {}, root: tmp }) === true && deferred.length === 1 && calls.length === 2,
  'finish() only schedules it: nothing is built before the run is recorded and the schedule moves on');
deferred[0]();
ok(calls.length === 4, 'and it is built on the later turn');
ok(scheduleReading({ key: 'predictor', startedAt: local(2026, 10, 18, 9) }, { defer: (fn) => deferred.push(fn), log: (l) => lines.push(l) }) === false && deferred.length === 1 && lines.length === 2,
  'other runs schedule nothing, and a job that is not a control says nothing');
// a control that writes no reading says why in one line, with the command that rebuilds the week's
fs.mkdirSync(path.join(tmp, 'data', 'readings'), { recursive: true });
fs.writeFileSync(path.join(tmp, 'data', 'readings', '2026-10-18.json'), '{}');
const told = [];
const tell = { defer: (fn) => deferred.push(fn), log: (l) => told.push(l), root: tmp };
ok(scheduleReading({ key: 'randomnessDiscrimination', name: 'Q2', startedAt: local(2026, 10, 19, 8) }, tell) === false
  && scheduleReading({ key: READING_JOB, name: 'Rows', startedAt: local(2026, 10, 20, 10) }, tell) === false
  && scheduleReading({ key: 'featureControl', name: 'FC', startedAt: local(2026, 10, 25, 10) }, tell) === false && deferred.length === 1 && told.length === 3
  && /^Q2 finished after the reading of Sunday 2026-10-18 was written: .*node readings\.js --controls data\/controls --sunday 2026-10-18 --database/.test(told[0])
  && /^Rows started after the controls' window \(Tuesday 09:00\): no reading includes it; .*--sunday 2026-10-18/.test(told[1])
  && /^FC finished: the reading of Sunday 2026-10-25 is written when Weekly controls: market rows under a proper score finishes$/.test(told[2]),
  `a control re-run after the reading, or the reading's job past Tuesday 09:00, says the reading does not hold it and how to rebuild it: ${told.join(' / ')}`);

ok(scheduleReading({ key: 'nullControls', name: 'Q0 band', startedAt: local(2026, 11, 1, 11) }, tell) === false && deferred.length === 1
  && /^Q0 band finished: the reading of Sunday 2026-11-01 is written when Weekly controls: market rows under a proper score finishes$/.test(told[3]),
  `the monthly band, first on its Sunday, says the reading comes after the market rows: ${told[3]}`);

// --- finish() itself, on a copy of jobs.js in a temp dir: a stand-in market rows run (the reading job) that ended, adopted at start-up -------
// The copy's own config and readings are stand-in files next to it: its scheduler state, log and run files
// all live in the temp dir, the schedule is switched off (nothing is due or started), and the readings stand-in
// records what it is asked - or throws.
function copyOfJobs(behaviour) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'jobs-reading-'));
  fs.copyFileSync(path.join(REPO, 'jobs.js'), path.join(dir, 'jobs.js'));
  fs.writeFileSync(path.join(dir, 'config.js'), `module.exports = { CONFIG_DIR: ${JSON.stringify(path.join(dir, 'config'))} };`);
  fs.writeFileSync(path.join(dir, 'readings.js'), `const calls = global.__readingCalls;
module.exports = {
  buildReading: (o) => { calls.push(o); if (${JSON.stringify(behaviour)} === 'throw') throw new Error('records unreadable'); return { sunday: o.sunday }; },
  writeReading: (r, o) => ({ file: require('path').join(o.root, 'data', 'readings', r.sunday + '.json'), written: true, bytes: 1 }),
};`);
  const config = path.join(dir, 'config');
  fs.mkdirSync(path.join(config, 'runs'), { recursive: true });
  fs.writeFileSync(path.join(config, 'scheduler.disabled'), '');   // mode off: nothing is due, nothing starts
  const runId = 'marketRows-2026-10-18T10-00-15-000Z';
  const exitFile = path.join(config, 'runs', `${runId}.exit`);
  fs.writeFileSync(exitFile, '1\n');   // the run failed: the reading is written all the same
  const startedAt = local(2026, 10, 18, 12);
  fs.writeFileSync(path.join(config, 'scheduler.json'), JSON.stringify({ version: 1, occurrences: {}, queue: [], history: [], baselinedAt: 'x',
    baselinedPlans: { weeklyTuning: 'x', weeklyControls: 'x' },
    current: { runId, key: 'marketRows', name: 'Weekly controls: market rows under a proper score', reason: 'test', startedAt, pid: null,
      pidFile: path.join(config, 'runs', `${runId}.pid`), exitFile, logFile: path.join(dir, 'log', 'marketRows.log'), sha: null, mode: 'on' } }));
  return dir;
}
async function finishOnCopy(behaviour) {
  const dir = copyOfJobs(behaviour);
  global.__readingCalls = [];
  const quiet = console.log;
  console.log = () => {};
  let copy;
  try {
    copy = require(path.join(dir, 'jobs.js'));
    copy.start();   // adopts the run: its exit file is there, so finish() records it
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    console.log = quiet;
    if (copy) copy.stop();
  }
  const saved = JSON.parse(fs.readFileSync(path.join(dir, 'config', 'scheduler.json'), 'utf-8'));
  const log = fs.readFileSync(path.join(dir, 'log', 'scheduler.log'), 'utf-8');
  const calls2 = global.__readingCalls;
  fs.rmSync(dir, { recursive: true, force: true });
  return { saved, log, calls: calls2, dir };
}

(async () => {
  const good = await finishOnCopy('ok');
  const recorded = good.saved.history.find((h) => h.key === 'marketRows');
  ok(recorded && recorded.exit === 1 && good.saved.current === null, 'finish() recorded the failed run and cleared it');
  ok(good.calls.length === 1 && good.calls[0].sunday === '2026-10-18' && good.calls[0].root === good.dir && good.calls[0].by === 'scheduler',
    'and then asked for the reading of the Sunday the run started on, under the checkout - whatever the exit code');
  ok(/the week's reading for Sunday 2026-10-18: written \(data\/readings\/2026-10-18\.json, 1 bytes\)/.test(good.log), 'the scheduler log says so');
  const bad = await finishOnCopy('throw');
  const kept = bad.saved.history.find((h) => h.key === 'marketRows');
  ok(kept && kept.exit === 1 && bad.saved.current === null && bad.calls.length === 1, 'a reading that throws: the run is recorded all the same');
  ok(/could not write the week's reading for Sunday 2026-10-18: records unreadable - nothing else is affected/.test(bad.log), 'and the failure is one line in the scheduler log, not a crash');
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`readings.js: ${checks} checks passed`);
})().catch((e) => {
  console.error(e);
  fs.rmSync(tmp, { recursive: true, force: true });
  process.exitCode = 1;
});
