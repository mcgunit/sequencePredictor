// The rules the admin pages apply to the control experiments (controls.js):
// Q0's row tests as the daily report writes them (27 Oct 2026, src/RowChance.py)
// with the old band read as context, the Q2 verdict on the "Randomness watch"
// card, the feature control. Fixtures are production-shaped records, written to
// a temp dir so the readers run too.
//   node test/controls.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const controls = require('../controls');

let passed = 0;
function ok(condition, message) { assert.ok(condition, message); passed += 1; }

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'controls-'));
fs.mkdirSync(path.join(dir, 'results'));
fs.mkdirSync(path.join(dir, 'discrimination'));
const write = (rel, obj) => fs.writeFileSync(path.join(dir, rel), JSON.stringify(obj));

// --- Q0 ---------------------------------------------------------------------
const synthetic = { rows_compared: 8, row_mean: 0.7809, best_row_mean: 0.8833, selection_gain: 0.1024,
  random_baseline: 0.7944, best_row_sd_across_seeds: 0.0507, game: 'lotto', mode: 'synthetic',
  seeds: [1, 2, 3], days: 120, generated_at: '2026-09-21T22:12:00' };
const shuffled = { rows_compared: 8, row_mean: 0.8208, best_row_mean: 0.9639, selection_gain: 0.1431,
  random_baseline: 0.7944, best_row_sd_across_seeds: 0.0192, game: 'lotto', mode: 'shuffled',
  seeds: [1, 2, 3], days: 120, generated_at: '2026-09-21T22:29:32' };
write('results/lotto-synthetic.json', { report: synthetic, per_seed: [] });
write('results/lotto-shuffled.json', { report: shuffled, per_seed: [] });
write('results/keno-synthetic.json', { report: { best_row_mean: null, rows_compared: 0 }, per_seed: [] });
fs.writeFileSync(path.join(dir, 'results', 'pick3-synthetic.json'), '{not json');

const lotto = controls.loadNullControls(dir, 'lotto');
ok(Object.keys(lotto).sort().join(',') === 'shuffled,synthetic', 'both controls of a game are read');
ok(Object.keys(controls.loadNullControls(dir, 'keno')).length === 0, 'a control that produced no score is not a control');
ok(Object.keys(controls.loadNullControls(dir, 'pick3')).length === 0, 'a broken file is ignored');
ok(Object.keys(controls.loadNullControls(dir, 'vikinglotto')).length === 0, 'a game never run has no controls');

const band = controls.nullBand(lotto);
ok(band.from === 'shuffled', 'the band comes from the higher control');
ok(Math.abs(band.ceiling - (0.9639 + 2 * 0.0192)) < 1e-9, 'ceiling = best row on nothing + 2 sd across seeds');
ok(Math.abs(band.modes.synthetic.ceiling - (0.8833 + 2 * 0.0507)) < 1e-9, 'each control keeps its own ceiling');
ok(band.generatedAt === '2026-09-21T22:29:32' && band.modes.shuffled.rows === 8 && band.modes.shuffled.seeds === 3, 'the band carries its provenance');
ok(band.metric === 'hits per draw', 'metric defaults to hits per draw for records written before the field existed');
ok(controls.nullBand({}) === null && controls.nullBand(null) === null, 'no controls, no band');
ok(controls.nullBand({ synthetic: { best_row_mean: 'n/a' } }) === null, 'a non-numeric best row is no band, not a crash');
write('results/eurodreams-synthetic.json', { report: { best_row_mean: 'n/a', rows_compared: 8 }, per_seed: [] });
ok(Object.keys(controls.loadNullControls(dir, 'eurodreams')).length === 0, 'a record with a non-numeric best row is not a control');
ok(controls.nullBand({ synthetic: { best_row_mean: 0.5, seeds: [1] } }).ceiling === 0.5, 'a single seed has no spread: the ceiling is the best row itself');

ok(controls.withinBand === undefined, 'the band labels nothing since 27 Oct 2026: no "within the band" rule is exported');

// The row tests (src/RowChance.py): the blocks of the 27 Oct 2026 report on a scratch clone of production, as written
const lottoChance = { method: 'exact', alpha: 0.05, verdictAlpha: 0.01, minDraws: 10, permutations: null, block: null, family: 27, liveMax: 75, under: 1,
  expected: 1.1, holds: 0, followed: 3, running: 3, stopped: 0, confirmed: 0, notConfirmed: 0, checkedThrough: '2026-10-25', stored: 'written' };
const autoencoder = { draws: 16, notLive: 0, mean: 1.1875, null: 0.8, nullSd: 0.196, p: 0.03892, inFamily: true, pHolm: 1.0, last: '2026-10-24',
  fresh: { since: '2026-10-18', flagDraws: 14, flagP: 0.01972, flagMean: 1.2857, flagNull: 0.8, verdictLength: 14, draws: 2, last: '2026-10-24', state: 'running',
    verdictAt: null, decidedOn: null, p: null, mean: 0.5, null: 0.8 } };
const markov = { draws: 74, notLive: 25, mean: 0.8919, null: 0.8541, nullSd: 0.0936, p: 0.3655, inFamily: true, pHolm: 1.0, last: '2026-10-24' };
const info = { metric: 'avg_hits', minDrawsForRanking: 10, chance: lottoChance,
  models: [{ name: 'Autoencoder Model', draws: 16, chance: autoencoder }, { name: 'Markov Model', draws: 99, chance: markov }, { name: 'New Model', draws: 2 },
    { name: 'Odd Model', draws: 30, chance: { ...markov, p: 'n/a', fresh: { ...autoencoder.fresh, state: 'maybe' } } }] };
const dc = controls.describeChance(info);
ok(dc.error === null && dc.test.family === 27 && dc.test.under === 1 && dc.test.expected === 1.1 && dc.test.method === 'exact' && dc.test.checkedThrough === '2026-10-25',
  `the game's summary as written: ${JSON.stringify(dc.test)}`);
ok(dc.rows['Autoencoder Model'].p === 0.03892 && dc.rows['Autoencoder Model'].fresh.state === 'running' && dc.rows['Autoencoder Model'].fresh.verdictLength === 14
  && dc.rows['Markov Model'].notLive === 25 && dc.rows['Markov Model'].fresh === null && dc.rows['New Model'] === undefined,
  'each row\'s block by name: its p, its live and not-live draws, a followed row\'s fresh record; a row without a block has none');
ok(dc.rows['Odd Model'].p === null && dc.rows['Odd Model'].fresh === null, 'a p that is not a number reads null, a fresh state this reader does not know drops the fresh record');
const before = controls.describeChance({ metric: 'avg_hits', models: [{ name: 'A Model', draws: 40 }] });
ok(before.test === null && before.error === 'not in this report - written before 27 Oct 2026, or its test failed' && Object.keys(before.rows).length === 0, 'a report from before 27 Oct 2026: no test, said why');
const unchecked = controls.describeChance({ metric: 'avg_hits', chance: { method: 'exact', error: 'liveness not checked: git failed', family: 0 },
  models: [{ name: 'A Model', draws: 40, chance: markov }] });
ok(unchecked.test === null && unchecked.error === 'liveness not checked: git failed' && Object.keys(unchecked.rows).length === 0,
  'a lottery game whose liveness could not be judged: no test, its reason, no row read as tested');
ok(controls.describeChance(null) === null && controls.FRESH_STATES.join() === 'running,stopped,confirmed,not confirmed', 'no ranking, nothing; the fresh states');

// --- Q2 ---------------------------------------------------------------------
const verdict = (over) => Object.assign({
  metric: 'best AUC of the classifier suite, per repetition',
  real_max_of_suite_mean: 0.588, real_max_of_suite_sd: 0.0491,
  null_max_of_suite_mean: 0.5648, null_max_of_suite_sd: 0.0761,
  shuffled_max_of_suite_mean: 0.5817, shuffled_max_of_suite_sd: 0.0469,
  null_threshold_2sd: 0.7169, above_null_band: false, survives_marginals_control: true,
}, over);
write('discrimination/lotto-w10.json', { game: 'lotto', window: 10, draws: 1000, repetitions: 5, quantum: true,
  generated_at: '2026-09-21T22:12:00', verdict: verdict({}) });
write('discrimination/lotto-w20.json', { game: 'lotto', window: 20, draws: 1000, repetitions: 3, quantum: false,
  generated_at: '2026-09-28T04:00:00', verdict: verdict({ above_null_band: true, real_max_of_suite_mean: 0.74 }) });
write('discrimination/pick3-w10.json', { game: 'pick3', window: 10, generated_at: '2026-09-21T22:12:00',
  verdict: verdict({ real_max_of_suite_mean: 0.4955, above_null_band: false, survives_marginals_control: false }) });
fs.writeFileSync(path.join(dir, 'discrimination', 'keno-w10.json'), '{');

const newest = controls.loadDiscrimination(dir, 'lotto');
ok(newest && newest.window === 20, 'the newest record of a game wins, whatever its window');
ok(controls.loadDiscrimination(dir, 'keno') === null && controls.loadDiscrimination(dir, 'jokerplus') === null,
  'a broken or missing record is no record');
ok(controls.loadDiscrimination(path.join(dir, 'nowhere'), 'lotto') === null, 'a missing folder is no record');

const d20 = controls.describeDiscrimination(newest);
ok(d20.evidence && d20.aboveBand && d20.survivesMarginals && d20.status === 'above' && d20.label === 'ABOVE both controls', 'above band + beats shuffled = evidence');
ok(d20.real === 0.74 && d20.threshold === 0.7169 && d20.window === 20 && d20.quantum === false, 'the description carries the numbers');
const d10 = controls.describeDiscrimination(JSON.parse(fs.readFileSync(path.join(dir, 'discrimination', 'lotto-w10.json'))));
ok(!d10.evidence && d10.label === 'within the null band', 'inside the band is no evidence, whatever the shuffled control says');
const dpick3 = controls.describeDiscrimination(controls.loadDiscrimination(dir, 'pick3'));
ok(!dpick3.evidence && !dpick3.survivesMarginals && dpick3.label === 'within the null band', 'pick3: below everything');
const dMarginals = controls.describeDiscrimination({ game: 'x', verdict: verdict({ above_null_band: true, survives_marginals_control: false }) });
ok(!dMarginals.evidence && dMarginals.aboveBand && dMarginals.label.includes('shuffled control'),
  'above the band but not the shuffled history is marginals, not time - said so');
// too few draws (14 Oct 2026): the flag from RandomnessDiscrimination.compute_verdict, else the same rule on the band
const dTooFew = controls.describeDiscrimination({ game: 'x', verdict: verdict({ above_null_band: true, survives_marginals_control: true, real_max_of_suite_mean: 0.93, null_threshold_2sd: 0.9724, too_few_draws: true, windows_test: 106 }) });
ok(!dTooFew.evidence && dTooFew.tooFew && dTooFew.windowsTest === 106 && dTooFew.label === 'too few draws',
  'an underpowered record is no evidence even above both controls, and says why');
const dOldTooFew = controls.describeDiscrimination({ game: 'x', shape: { windows_test: 148 }, verdict: verdict({ above_null_band: false, null_threshold_2sd: 0.8934 }) });
ok(dOldTooFew.tooFew && dOldTooFew.label === 'too few draws' && dOldTooFew.windowsTest === 148, 'an older record without the flag is read by the windows rule, its window count from its shape');
ok(controls.describeDiscrimination({ game: 'x', verdict: verdict({ null_threshold_2sd: 0.8934 }) }).windowsTest === null, 'no shape, no window count');
ok(!controls.describeDiscrimination({ game: 'x', verdict: verdict({ null_threshold_2sd: 0.7515 }) }).tooFew, 'a band under 0.8 is powered');
ok(controls.describeDiscrimination({ game: 'x', verdict: verdict({ null_threshold_2sd: 0.7515, too_few_draws: true }) }).tooFew
  && !controls.describeDiscrimination({ game: 'x', verdict: verdict({ null_threshold_2sd: 0.8934, too_few_draws: false }) }).tooFew,
  'an older record without a window count keeps its own flag');

// Q2's power rule from 27 Oct 2026: one number, shared with RandomnessDiscrimination.py through controlRules.json, and
// applied to the older (version 1) records too where their window count is known - a property of the history
const RULES = require('../controlRules.json');
ok(controls.FEW_WINDOWS === RULES.q2.few_windows && Number.isInteger(controls.FEW_WINDOWS) && controls.FEW_WINDOWS > 0, 'the power rule is controlRules.json\'s');
// the 25 Oct 2026 eurodreams record: version 1 read it "above" (0.704 against 0.687, 3 seeds) on 150 test windows
const eurodreams25 = { game: 'eurodreams', window: 10, draws: 1000, repetitions: 3, shape: { windows_train: 452, windows_test: 150, features: 40 },
  verdict: verdict({ real_max_of_suite_mean: 0.7042, real_max_of_suite_sd: 0.147, null_max_of_suite_mean: 0.5535, null_max_of_suite_sd: 0.0668,
    shuffled_max_of_suite_mean: 0.6948, null_threshold_2sd: 0.687, above_null_band: true, survives_marginals_control: true, too_few_draws: false, windows_test: 150 }) };
const de = controls.describeDiscrimination(eurodreams25);
ok(de.status === 'too few draws' && de.tooFew && !de.evidence && de.version === 1 && de.fewWindows === controls.FEW_WINDOWS && de.threshold === 0.687 && de.thresholdSe === null,
  `a version-1 "above" on 150 test windows reads too few draws: ${JSON.stringify([de.status, de.fewWindows])}`);
const v1 = (over) => controls.describeDiscrimination({ game: 'x', shape: { windows_test: 496 }, verdict: verdict(over) });
ok(v1({ above_null_band: true, survives_marginals_control: true, too_few_draws: false }).status === 'above' && v1({}).status === 'within'
  && v1({ above_null_band: true, survives_marginals_control: false }).status === 'above band, not shuffled'
  && v1({ above_null_band: true, survives_marginals_control: true, too_few_draws: true }).status === 'above',
  'a version-1 record on enough windows keeps what its flags said; the windows decide too few, not its old flag');
ok(v1({}).version === 1 && v1({}).bandK === null && v1({}).shuffledThreshold === null, 'a version-1 record has none of the version-2 numbers');
// the edge, as compute_verdict draws it: FEW_WINDOWS test windows are enough, one fewer is too few
const atEdge = (windows) => controls.describeDiscrimination({ game: 'x', shape: { windows_test: windows }, verdict: verdict({ above_null_band: true, survives_marginals_control: true }) }).status;
ok(atEdge(controls.FEW_WINDOWS) === 'above' && atEdge(controls.FEW_WINDOWS - 1) === 'too few draws', `${controls.FEW_WINDOWS} test windows are enough, ${controls.FEW_WINDOWS - 1} are too few`);
// a version-2 record (1 Nov 2026 on) carries its own status and the numbers it was read from: read as they are
const v2 = (over) => ({ version: 2, status: 'within', metric: 'best AUC of the classifier suite, per repetition',
  real_max_of_suite_mean: 0.567, real_max_of_suite_sd: 0.05, null_max_of_suite_mean: 0.576, null_max_of_suite_sd: 0.044,
  shuffled_max_of_suite_mean: 0.588, shuffled_max_of_suite_sd: 0.059, seeds_scored: { real: 20, null: 20, shuffled: 20 }, band_coverage: 0.97725,
  band_k: 2.193, shuffled_k: 2.193, null_threshold: 0.6725, null_threshold_se: 0.0185, shuffled_threshold: 0.7174,
  above_null_band: false, survives_marginals_control: false, too_few_draws: false, few_windows: 250, windows_test: 496, ...over });
const dv2 = controls.describeDiscrimination({ game: 'lotto', window: 10, repetitions: 20, shape: { windows_test: 496 }, verdict: v2({}) });
ok(dv2.status === 'within' && dv2.version === 2 && dv2.threshold === 0.6725 && dv2.thresholdSe === 0.0185 && dv2.bandK === 2.193 && dv2.shuffledK === 2.193
  && dv2.shuffledThreshold === 0.7174 && dv2.fewWindows === 250 && dv2.label === 'within the null band',
  `version 2: the threshold from null_threshold, its error, k and the shuffled bound pass through: ${JSON.stringify(dv2)}`);
const crypto2 = controls.describeDiscrimination({ game: 'crypto', verdict: v2({ status: 'above', real_max_of_suite_mean: 0.938, above_null_band: true, survives_marginals_control: true }) });
ok(crypto2.status === 'above' && crypto2.evidence && crypto2.label === 'ABOVE both controls', 'version 2: above both controls');
ok(controls.describeDiscrimination({ game: 'x', verdict: v2({ status: 'within', above_null_band: true, survives_marginals_control: true }) }).status === 'within',
  'version 2: the record\'s status is the one source, the flags are not read again');
const failed = controls.describeDiscrimination({ game: 'x', verdict: v2({ status: 'failed', real_max_of_suite_mean: null, null_threshold: null, null_threshold_se: null, band_k: null }) });
ok(failed.status === 'failed' && failed.failed && !failed.tooFew && !failed.evidence && failed.threshold === null && failed.real === null && /^failed/.test(failed.label),
  'version 2: a run where a comparison scored fewer than 2 seeds reads failed, never within');
ok(controls.describeDiscrimination({ game: 'x', verdict: v2({ status: 'something new' }) }).status === 'failed', 'a status this reader does not know is no reading');
ok(controls.describeDiscrimination({ game: 'x', verdict: v2({ status: 'too few draws', few_windows: 500, too_few_draws: true }) }).fewWindows === 500
  && controls.describeDiscrimination({ game: 'x', verdict: v2({ status: 'too few draws', few_windows: 500 }) }).tooFew,
  'version 2: the windows the record was judged by are its own, not today\'s');
ok(controls.Q2_STATUSES.every((s) => typeof s === 'string') && controls.Q2_STATUSES.length === 5 && controls.Q2_STATUSES.includes('failed'), 'five statuses, failed among them');

// --- irrelevant-feature control ---------------------------------------------
fs.mkdirSync(path.join(dir, 'features'));
const variant = (over) => Object.assign({
  repeats: 3, noise_columns: 3,
  noise: { train_importance_mean: 0.001, train_importance_sd: 0.0005, train_t: 3.2, share_of_train_importance: 0.03,
    noise_to_real_ratio: 0.04, heldout_importance_mean: 0.0002, heldout_importance_sd: 0.001, heldout_band: 0.0022 },
  columns: {}, real_above_noise_band: ['Markov Model', 'XGBoost Model'],
  heldout_auc_without_noise: 0.6123, heldout_auc_with_noise_mean: 0.6101, heldout_cost_of_noise: 0.0022,
  fits_noise: false, rule: 'the rule', seconds: 12.3,
}, over);
write('features/lotto.json', { game: 'lotto', generated_at: '2026-10-04T12:00:00', repeats: 3, noise_columns: 3,
  table: { days: 275, positional: false, behind_file_by: 1, lockbox_days_withheld: 25 },
  variants: {
    logistic: variant({}),
    gradient_boosting: variant({ fits_noise: true, real_above_noise_band: [],
      noise: { share_of_train_importance: 0.31, train_t: 5.5, heldout_band: 0.01 } }),
    quantum_vqc: { error: 'boom', seconds: 1 },
  } });
write('features/keno.json', { game: 'lotto', variants: {} });
fs.writeFileSync(path.join(dir, 'features', 'pick3.json'), '{');

const fc = controls.loadFeatureControl(dir, 'lotto');
ok(fc && fc.game === 'lotto', 'the feature-control record loads');
ok(controls.loadFeatureControl(dir, 'keno') === null && controls.loadFeatureControl(dir, 'pick3') === null
  && controls.loadFeatureControl(dir, 'jokerplus') === null && controls.loadFeatureControl(path.join(dir, 'nowhere'), 'lotto') === null,
  'a record naming another game, a broken file, no file or no folder is no record');
const fd = controls.describeFeatureControl(fc);
const fdv = (over) => controls.describeFeatureControl({ game: 'x', variants: { logistic: variant(over) } }).variants[0];
ok(fdv({ inconclusive: true, verdict: 'inconclusive', noise: { noise_to_real_ratio: 0.6, train_t: 1.1 } }).verdict === 'inconclusive'
   && fdv({ inconclusive: true, verdict: 'inconclusive' }).inconclusive && !fdv({ inconclusive: true }).fitsNoise,
  'a variant the script marks inconclusive reads inconclusive');
ok(fdv({ noise: { noise_to_real_ratio: 0.6, train_t: 1.1 }, rule: 'old rule' }).rule.includes('Inconclusive: the noise ratio is exceeded') && !fdv({ rule: 'old rule' }).rule.includes('Inconclusive')
   && fdv({ noise: { noise_to_real_ratio: 0.6, train_t: 1.1 } }).verdict === 'inconclusive' && fdv({}).verdict === 'noise ignored'
   && fdv({ fits_noise: true, inconclusive: false }).verdict === 'fits noise' && !fdv({ fits_noise: true }).inconclusive,
  'an older record without the field is read by the ratio rule; fits noise wins over inconclusive');
ok(fd.variants.length === 3 && fd.tableDays === 275 && fd.behind === 1 && fd.lockboxDays === 25 && fd.generatedAt === '2026-10-04T12:00:00' && fd.repeats === 3,
  'the description carries the table, the lockbox count and every variant');
const lg = fd.variants.find((v) => v.key === 'logistic');
const gb = fd.variants.find((v) => v.key === 'gradient_boosting');
const vq = fd.variants.find((v) => v.key === 'quantum_vqc');
ok(lg.label === 'MetaLearner (logistic)' && !lg.fitsNoise && lg.verdict === 'noise ignored' && lg.above.length === 2
  && lg.heldoutWithout === 0.6123 && lg.heldoutWith === 0.6101 && lg.noiseShare === 0.03 && lg.noiseRatio === 0.04 && lg.rule === 'the rule',
  'a variant that ignores noise reads as such, with its base models above the band and its numbers');
ok(gb.fitsNoise && gb.verdict === 'fits noise' && gb.noiseShare === 0.31 && gb.noiseT === 5.5 && gb.above.length === 0 && fd.anyFitsNoise,
  'a variant that fits noise is flagged, and the game with it');
ok(vq.error === 'boom' && vq.verdict === 'failed' && !vq.fitsNoise && vq.noiseShare === null && vq.above.length === 0,
  'a failed variant reads as failed, with no numbers');
const partial = controls.describeFeatureControl({ game: 'x', variants: { classical_svm: { fits_noise: false } } });
ok(partial.variants[0].label === 'ClassicalSVM (RBF control)' && partial.variants[0].noiseShare === null
  && partial.variants[0].heldoutWithout === null && partial.variants[0].above.length === 0 && !partial.anyFitsNoise
  && partial.tableDays === null, 'a record missing its numbers describes without throwing, with nulls');
ok(Object.keys(controls.VARIANT_LABELS).join(',') === 'logistic,gradient_boosting,quantum_kernel,quantum_vqc,classical_svm',
  'every served variant has a label');

fs.rmSync(dir, { recursive: true, force: true });
console.log(`controls.js: ${passed} checks passed`);
