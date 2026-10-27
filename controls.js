// controls.js - reads the outputs of the control experiments for the admin
// pages and turns them into the numbers the pages show:
//
//   Q0  src/RowChance.py           -> modelPerformance.json, per game and per model "chance"
//       Since 27 Oct 2026 the daily report's row test: each ranked row on its
//       live draws against the fair draw (lottery, exact) or its own tickets
//       shuffled within blocks of 7 live days (markets), Holm over the card,
//       and a fixed-length verdict on new draws for a row that read p < 0.05.
//       NullControls.py            -> data/controls/results/<game>-<mode>.json
//       What the BEST of 8 untuned rows scores on 120 fair draws: the old band,
//       context only since 27 Oct 2026 (the Controls drawer) - it knows neither
//       a row's record length nor its own fair level, so nothing is labelled,
//       greyed or counted on it any more.
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
// Q2's power rule: a record on fewer held-out windows than this reads "too few draws". One number, shared with
// RandomnessDiscrimination.py through controlRules.json (27 Oct 2026); admin.js takes it from here.
const FEW_WINDOWS = require('./controlRules.json').q2.few_windows;

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
// standard deviations across them, taken over whichever control is higher
// (shuffled usually is: the marginals survive there) - what the best of N
// worthless rows scores: the selection effect, shown as context next to the
// row tests (until 27 Oct 2026 a row at or below it was "within the band").
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

// The row tests of one game, as the daily report wrote them (src/RowChance.py, 27 Oct 2026): test, the game's summary
// (or, without one, error: why there is no test - a report from before, a game whose liveness could not be judged, a
// test that failed); rows, each model's block by name. Numbers that are not numbers read null, never NaN; a fresh state
// this reader does not know drops the fresh block rather than claim one. null without a ranking for the game.
const ALPHA = 0.05;
const VERDICT_ALPHA = 0.01;
const FRESH_STATES = ['running', 'stopped', 'confirmed', 'not confirmed'];
const date = (x) => (typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x) ? x : null);
function freshOf(f) {
  if (!f || typeof f !== 'object' || !FRESH_STATES.includes(f.state)) return null;
  return { state: f.state, since: date(f.since), flagDraws: num(f.flagDraws), flagP: num(f.flagP), flagMean: num(f.flagMean), flagNull: num(f.flagNull),
    verdictLength: num(f.verdictLength), draws: num(f.draws) || 0, last: date(f.last), verdictAt: date(f.verdictAt), decidedOn: date(f.decidedOn),
    p: num(f.p), mean: num(f.mean), null: num(f.null) };
}
function describeChance(info) {
  if (!info || !Array.isArray(info.models)) return null;
  const c = info.chance && typeof info.chance === 'object' ? info.chance : null;
  const error = !c ? 'not in this report - written before 27 Oct 2026, or its test failed' : (c.error ? String(c.error) : null);
  const test = c && !error ? {
    method: c.method === 'permutation' ? 'permutation' : 'exact', alpha: num(c.alpha) ?? ALPHA, verdictAlpha: num(c.verdictAlpha) ?? VERDICT_ALPHA,
    minDraws: num(c.minDraws), permutations: num(c.permutations), block: num(c.block), family: num(c.family) || 0, liveMax: num(c.liveMax) || 0,
    under: num(c.under) || 0, expected: num(c.expected) || 0, holds: num(c.holds) || 0, followed: num(c.followed) || 0, running: num(c.running) || 0,
    stopped: num(c.stopped) || 0, confirmed: num(c.confirmed) || 0, notConfirmed: num(c.notConfirmed) || 0,
    checkedThrough: date(c.checkedThrough), stored: typeof c.stored === 'string' ? c.stored : null,
  } : null;
  const rows = {};
  if (test) {
    info.models.forEach((m) => {
      const b = m && m.chance;
      if (!b || typeof b !== 'object') return;
      rows[m.name] = { draws: num(b.draws) || 0, notLive: num(b.notLive) || 0, mean: num(b.mean), null: num(b.null), nullSd: num(b.nullSd), p: num(b.p),
        pHolm: num(b.pHolm), inFamily: b.inFamily === true, last: date(b.last), fresh: freshOf(b.fresh) };
    });
  }
  return { test, error, rows };
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

// The statuses a record reads (admin.js q2Of, the readings' Q2 state): RandomnessDiscrimination.STATUSES.
const Q2_STATUSES = ['above', 'above band, not shuffled', 'within', 'too few draws', 'failed'];
const Q2_LABELS = {
  above: 'ABOVE both controls',
  'above band, not shuffled': 'above the band, but no better than the shuffled control',
  within: 'within the null band',
  'too few draws': 'too few draws',
  failed: 'failed: a comparison scored fewer than 2 seeds',
};

// Evidence needs both: clearing the null band AND beating the shuffled history, otherwise the classifier is reading
// marginals a fair process reproduces anyway. A record written from 1 Nov 2026 on (verdict version 2) carries its
// status, computed by RandomnessDiscrimination.compute_verdict - read as it is, one source of truth. An older record
// (version 1: mean + 2 sd of 3 seeds, 5 for the markets, and the bare shuffled mean) has its status derived here from
// its flags, with one rule of version 2 applied to it too: too few draws under FEW_WINDOWS held-out windows, a property
// of the history and not of the run - so the 25 Oct 2026 eurodreams "above" on 150 windows reads "too few draws". A
// record without a window count falls back to its own flag, or to the band rule it was written under (0.8 or more).
function describeDiscrimination(record) {
  const v = record.verdict;
  const version = num(v.version) || 1;
  const aboveBand = Boolean(v.above_null_band);
  const survivesMarginals = Boolean(v.survives_marginals_control);
  const windowsTest = num(v.windows_test !== undefined ? v.windows_test : (record.shape || {}).windows_test);   // older records carry it in shape only
  const threshold = num(v.null_threshold !== undefined ? v.null_threshold : v.null_threshold_2sd);
  // the windows the record was judged by: its own (version 2), else today's rule
  const fewWindows = num(v.few_windows) ?? FEW_WINDOWS;
  let status;
  if (version >= 2) {
    // a status this reader does not know is no reading: never a clean "within"
    status = Q2_STATUSES.includes(v.status) ? v.status : 'failed';
  } else {
    const tooFew = windowsTest !== null ? windowsTest < fewWindows
      : (v.too_few_draws === undefined ? (threshold !== null && threshold >= 0.8) : Boolean(v.too_few_draws));
    status = tooFew ? 'too few draws' : (aboveBand && survivesMarginals ? 'above' : (aboveBand ? 'above band, not shuffled' : 'within'));
  }
  return {
    status, version, evidence: status === 'above', aboveBand, survivesMarginals, tooFew: status === 'too few draws', failed: status === 'failed',
    windowsTest, fewWindows,
    real: num(v.real_max_of_suite_mean), realSd: num(v.real_max_of_suite_sd),
    nullMean: num(v.null_max_of_suite_mean), nullSd: num(v.null_max_of_suite_sd),
    shuffled: num(v.shuffled_max_of_suite_mean), shuffledSd: num(v.shuffled_max_of_suite_sd),
    // version 2: the threshold's standard error, k of each bound and the shuffled control's own bound (null before)
    threshold, thresholdSe: num(v.null_threshold_se), bandK: num(v.band_k), shuffledK: num(v.shuffled_k), shuffledThreshold: num(v.shuffled_threshold),
    window: record.window, repetitions: record.repetitions, draws: record.draws,
    quantum: Boolean(record.quantum), generatedAt: record.generated_at || null,
    label: Q2_LABELS[status],
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

module.exports = { MODES, FEW_WINDOWS, Q2_STATUSES, FRESH_STATES, loadNullControls, nullBand, describeChance, loadDiscrimination, describeDiscrimination,
  VARIANT_LABELS, loadFeatureControl, describeFeatureControl };
