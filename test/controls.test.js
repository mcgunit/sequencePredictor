// The rules the History page applies to the weekly control experiments
// (controls.js): the Q0 band under the "Best model per game" card and the Q2
// verdict on the "Randomness watch" card. Fixtures are the 21 Sept 2026
// production-shaped records, written to a temp dir so the readers run too.
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

ok(controls.withinBand(1.0, band) && controls.withinBand(0.5, band), 'a row at or under the ceiling is within the band');
ok(!controls.withinBand(1.01, band), 'a row above the ceiling is not');
ok(!controls.withinBand(null, band) && !controls.withinBand(undefined, band) && !controls.withinBand('x', band), 'no value, not within');
ok(!controls.withinBand(0.1, null), 'no band, nothing is within it');

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
ok(d20.evidence && d20.aboveBand && d20.survivesMarginals && d20.label === 'ABOVE the null band', 'above band + beats shuffled = evidence');
ok(d20.real === 0.74 && d20.threshold === 0.7169 && d20.window === 20 && d20.quantum === false, 'the description carries the numbers');
const d10 = controls.describeDiscrimination(JSON.parse(fs.readFileSync(path.join(dir, 'discrimination', 'lotto-w10.json'))));
ok(!d10.evidence && d10.label === 'within the null band', 'inside the band is no evidence, whatever the shuffled control says');
const dpick3 = controls.describeDiscrimination(controls.loadDiscrimination(dir, 'pick3'));
ok(!dpick3.evidence && !dpick3.survivesMarginals && dpick3.label === 'within the null band', 'pick3: below everything');
const dMarginals = controls.describeDiscrimination({ game: 'x', verdict: verdict({ above_null_band: true, survives_marginals_control: false }) });
ok(!dMarginals.evidence && dMarginals.aboveBand && dMarginals.label.includes('shuffled control'),
  'above the band but not the shuffled history is marginals, not time - said so');

fs.rmSync(dir, { recursive: true, force: true });
console.log(`controls.js: ${passed} checks passed`);
