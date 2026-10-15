// controls.js - reads the outputs of the two weekly control experiments for the
// History page and turns them into the two numbers the page shows:
//
//   Q0  NullControls.py            -> data/controls/results/<game>-<mode>.json
//       What the BEST of the tracked rows scores on a history with nothing in
//       it. The band under which a row on the "Best model per game" card has
//       not shown anything a leaderboard over noise would not show.
//   Q2  RandomnessDiscrimination.py -> data/controls/discrimination/<game>-w<N>.json
//       Whether a classifier suite separates real draw windows from fair
//       simulated ones better than the same suite separates two fair
//       histories. The controlled verdict next to the daily entropy tripwire
//       on the "Randomness watch" card.
//   FC  IrrelevantFeatureControl.py -> data/controls/features/<game>.json
//       Whether any meta-learner variant gives noise columns appended to its
//       table stable importance, and which base models matter more than
//       noise does. The "Meta-learner feature control" card.
//
// Pure functions over parsed records, so the rules are testable
// (test/controls.test.js); the readers swallow missing or broken files - the
// experiments are optional and a page must never fail because of them.
'use strict';

const fs = require('fs');
const path = require('path');

const MODES = ['synthetic', 'shuffled'];

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return null; }
}

// --- Q0 ---------------------------------------------------------------------
function loadNullControls(dir, game) {
  const reports = {};
  MODES.forEach((mode) => {
    const record = readJson(path.join(dir, 'results', `${game}-${mode}.json`));
    const report = record && record.report;
    if (report && typeof report.best_row_mean === 'number' && Number.isFinite(report.best_row_mean)) reports[mode] = report;
  });
  return reports;
}

// The band: the mean over control histories of the best row's score, plus two
// standard deviations across them - the rule RandomnessDiscrimination's
// verdict applies to its null comparison - taken over whichever control is
// higher (shuffled usually is: the marginals survive there). A row at or below
// it is "within the band": the best of N worthless rows scores that much.
function nullBand(reports) {
  const modes = Object.keys(reports || {});
  if (!modes.length) return null;
  const per = {};
  let ceiling = -Infinity;
  let from = null;
  let generatedAt = '';
  modes.forEach((mode) => {
    const r = reports[mode];
    const best = Number(r.best_row_mean);
    if (!Number.isFinite(best)) return;
    const sd = Number(r.best_row_sd_across_seeds) || 0;
    const top = best + 2 * sd;
    per[mode] = {
      best, sd, ceiling: top,
      rows: r.rows_compared, seeds: Array.isArray(r.seeds) ? r.seeds.length : 0, days: r.days,
      random: r.random_baseline, generatedAt: r.generated_at || null,
      metric: r.metric || 'hits per draw',
    };
    if (top > ceiling) { ceiling = top; from = mode; }
    if (String(r.generated_at || '') > generatedAt) generatedAt = String(r.generated_at || '');
  });
  if (from === null) return null;
  return { ceiling, from, modes: per, metric: per[from].metric, generatedAt: generatedAt || null };
}

function withinBand(value, band) {
  if (!band || value === null || value === undefined) return false;
  const v = Number(value);
  return Number.isFinite(v) && v <= band.ceiling;
}

// --- Q2 ---------------------------------------------------------------------
// The newest record for the game, whatever window it was run with.
function loadDiscrimination(dir, game) {
  const folder = path.join(dir, 'discrimination');
  let files;
  try { files = fs.readdirSync(folder); } catch (e) { return null; }
  const records = files
    .filter((f) => f.startsWith(`${game}-w`) && f.endsWith('.json'))
    .map((f) => readJson(path.join(folder, f)))
    .filter((r) => r && r.verdict && r.game === game);
  if (!records.length) return null;
  records.sort((a, b) => String(b.generated_at || '').localeCompare(String(a.generated_at || '')));
  return records[0];
}

function describeDiscrimination(record) {
  const v = record.verdict;
  const aboveBand = Boolean(v.above_null_band);
  const survivesMarginals = Boolean(v.survives_marginals_control);
  // Evidence needs both: clearing the null band AND beating the shuffled
  // history, otherwise the classifier is reading marginals a fair process
  // reproduces anyway.
  // An underpowered record (RandomnessDiscrimination.POWER_CEILING: the null band's ceiling at 0.8 or more)
  // can show nothing either way; older records without the flag are read by the same rule.
  const threshold = num(v.null_threshold_2sd);
  const tooFew = v.too_few_draws === undefined ? (threshold !== null && threshold >= 0.8) : Boolean(v.too_few_draws);
  const evidence = aboveBand && survivesMarginals && !tooFew;
  return {
    evidence, aboveBand, survivesMarginals, tooFew,
    windowsTest: num(v.windows_test !== undefined ? v.windows_test : (record.shape || {}).windows_test),   // older records carry it in shape only
    real: v.real_max_of_suite_mean, realSd: v.real_max_of_suite_sd,
    nullMean: v.null_max_of_suite_mean, nullSd: v.null_max_of_suite_sd,
    threshold: v.null_threshold_2sd, shuffled: v.shuffled_max_of_suite_mean,
    window: record.window, repetitions: record.repetitions, draws: record.draws,
    quantum: Boolean(record.quantum), generatedAt: record.generated_at || null,
    label: tooFew ? 'too few draws' : (evidence ? 'ABOVE the null band'
      : (aboveBand ? 'above the band, but no better than the shuffled control' : 'within the null band')),
  };
}

// --- irrelevant-feature control ---------------------------------------------
// The served row each variant key stands for, in the words the History page
// uses for the rows themselves.
const VARIANT_LABELS = {
  logistic: 'MetaLearner (logistic)',
  gradient_boosting: 'MetaLearnerV2 (gradient boosting)',
  quantum_kernel: 'QuantumMetaLearner (quantum kernel)',
  quantum_vqc: 'QuantumVQC',
  classical_svm: 'ClassicalSVM (RBF control)',
};

function loadFeatureControl(dir, game) {
  const record = readJson(path.join(dir, 'features', `${game}.json`));
  return record && record.game === game && record.variants && typeof record.variants === 'object' ? record : null;
}

const num = (x) => {
  const v = Number(x);
  return x === null || x === undefined || !Number.isFinite(v) ? null : v;
};

// One row per variant; a variant that failed carries its error and nothing
// else. Missing numbers read as null, never as NaN in the page.
function describeFeatureControl(record) {
  const variants = Object.keys(record.variants || {}).map((key) => {
    const v = record.variants[key] || {};
    const noise = v.noise || {};
    const fitsNoise = Boolean(v.fits_noise);
    // "inconclusive" (FeatureControl.summarise, 14 Oct 2026): the noise ratio is exceeded but not the t limit;
    // an older record without the field is read by the same rule
    const ratio = num(noise.noise_to_real_ratio);
    const inconclusive = !fitsNoise && (v.inconclusive === undefined ? (ratio !== null && ratio > 0.1) : Boolean(v.inconclusive));
    return {
      key, label: VARIANT_LABELS[key] || key,
      error: v.error || null,
      fitsNoise, inconclusive, verdict: v.error ? 'failed' : (fitsNoise ? 'fits noise' : (inconclusive ? 'inconclusive' : 'noise ignored')),
      noiseShare: num(noise.share_of_train_importance), noiseRatio: num(noise.noise_to_real_ratio),
      noiseT: num(noise.train_t), band: num(noise.heldout_band),
      above: Array.isArray(v.real_above_noise_band) ? v.real_above_noise_band.map(String) : [],
      heldoutWithout: num(v.heldout_auc_without_noise), heldoutWith: num(v.heldout_auc_with_noise_mean),
      cost: num(v.heldout_cost_of_noise), seconds: num(v.seconds),
      realColumns: v.columns && typeof v.columns === 'object' ? Object.keys(v.columns).length : null,   // the base-model columns tested
      // a record written before the third verdict existed carries a rule without it; the tooltip gets the clause anyway
      rule: (v.rule || '') + (inconclusive && !/inconclusive/.test(v.rule || '') ? ' Inconclusive: the noise ratio is exceeded but the pooled noise importance is not two standard errors above zero - the repeats disagree, too few to say.' : ''),
    };
  });
  const table = record.table || {};
  return {
    game: record.game, generatedAt: record.generated_at || null,
    repeats: num(record.repeats), noiseColumns: num(record.noise_columns),
    tableDays: num(table.days), positional: Boolean(table.positional), behind: num(table.behind_file_by),
    lockboxDays: num(table.lockbox_days_withheld),
    variants, anyFitsNoise: variants.some((v) => v.fitsNoise),
  };
}

module.exports = { MODES, loadNullControls, nullBand, withinBand, loadDiscrimination, describeDiscrimination,
  VARIANT_LABELS, loadFeatureControl, describeFeatureControl };
