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
  const evidence = aboveBand && survivesMarginals;
  return {
    evidence, aboveBand, survivesMarginals,
    real: v.real_max_of_suite_mean, realSd: v.real_max_of_suite_sd,
    nullMean: v.null_max_of_suite_mean, nullSd: v.null_max_of_suite_sd,
    threshold: v.null_threshold_2sd, shuffled: v.shuffled_max_of_suite_mean,
    window: record.window, repetitions: record.repetitions, draws: record.draws,
    quantum: Boolean(record.quantum), generatedAt: record.generated_at || null,
    label: evidence ? 'ABOVE the null band'
      : (aboveBand ? 'above the band, but no better than the shuffled control' : 'within the null band'),
  };
}

module.exports = { MODES, loadNullControls, nullBand, withinBand, loadDiscrimination, describeDiscrimination };
