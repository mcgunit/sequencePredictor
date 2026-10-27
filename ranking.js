// ranking.js - which model rows a reader sees, and in which order (3 Oct 2026, the owner's decision; 21 Oct 2026:
// a user sees the top five rows only, the administrator every row with an All / Top switch).
//
// One ranking for every page that says "the top models": the History ranking in
// data/database/modelPerformance.json (Helpers.generate_model_performance_report), in its own order - the
// established rows by the game's metric, the rows with fewer scored draws than the ranking minimum after them. A
// lottery game ranked on the live record (games[g].rankedOn 'live', 27 Oct 2026: only the tickets saved before each
// draw's sales close count) is in that order already; its metric's words say so (metricTextOf).
// admin.js (the admin JSON), workbench.js and lottery.js (the users' Lottery and Home) all read it from here, so they
// never name a different top five.
'use strict';

const fs = require('fs');
const path = require('path');

const TOP_MODELS_SHOWN = 5;
const REPORT_FILE = 'modelPerformance.json';
// the metric a game's ranking reads, in words
const METRIC_TEXT = { profit_bound: 'capped profit per bet (lower bound)', profit_per_bet: 'profit per bet', avg_hits: 'average hits' };

// What a game's ranking reads, in words - with the live record's rule when it ranks on it
const LIVE_TEXT = ' over the tickets saved before each draw\'s sales close';
const metricTextOf = (metric, rankedOn) => (METRIC_TEXT[metric] || METRIC_TEXT.avg_hits) + (rankedOn === 'live' ? LIVE_TEXT : '');

// The ranking of one game in a parsed report: {names (in order), metric (in words)}, or null when the game has none.
function rankingOf(report, game) {
  const info = report && report.games && report.games[game];
  if (!info || !Array.isArray(info.models) || !info.models.length) return null;
  return { names: info.models.map((m) => String(m.name)), metric: metricTextOf(info.metric, info.rankedOn) };
}

// The first n names of a ranking - the users' top five. With `present` (the names a day's table has), the first n
// of those: a ranked row can be missing from a day.
function topNames(ranking, n = TOP_MODELS_SHOWN, present = null) {
  if (!ranking || !Array.isArray(ranking.names)) return [];
  const names = Array.isArray(present) ? ranking.names.filter((name) => present.includes(name)) : ranking.names;
  return names.slice(0, n);
}

// The report, parsed once per change of the file (it is about 1.5 MB, and several cards and routes read it per
// request). Keyed by the file's modification time and size; null when it is missing. Helpers.py rewrites it in
// place, so a read can catch it half written: a file that does not parse keeps the last good report (and is read
// again on its next change), so a user's top-five filter never drops to "no ranking" mid-write. Callers must not
// change the object they get: the next caller gets the same one.
const reportCache = new Map();   // file -> {stamp, report}
function loadReport(databaseDir) {
  const file = path.join(databaseDir, REPORT_FILE);
  let stamp;
  try { const s = fs.statSync(file); stamp = `${s.mtimeMs}:${s.size}`; } catch (e) { reportCache.delete(file); return null; }
  const hit = reportCache.get(file);
  if (hit && hit.stamp === stamp) return hit.report;
  let report = null;
  try { report = JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { report = hit ? hit.report : null; }
  reportCache.set(file, { stamp, report });
  return report;
}

// The ranking of one game as stored under databaseDir (data/database).
function rankingFor(databaseDir, game) {
  return rankingOf(loadReport(databaseDir), game);
}

// The names in a game's newest day file (data/database/<game>/YYYY-M-D.json, the newest date first): the
// next-prediction table the Lottery shows today. Read once per change of that file; null
// when the game has no day file (or it cannot be read).
const todayCache = new Map();   // file -> {stamp, names}
const dayNumber = (file) => {
  const m = /^(\d{4})-(\d{1,2})-(\d{1,2})\.json$/.exec(file);
  return m ? Number(m[1]) * 10000 + Number(m[2]) * 100 + Number(m[3]) : null;
};
function namesToday(databaseDir, game) {
  let files = [];
  try { files = fs.readdirSync(path.join(databaseDir, game)).filter((f) => dayNumber(f) !== null); } catch (e) { return null; }
  if (!files.length) return null;
  const file = path.join(databaseDir, game, files.sort((a, b) => dayNumber(b) - dayNumber(a))[0]);
  let stamp;
  try { const s = fs.statSync(file); stamp = `${s.mtimeMs}:${s.size}`; } catch (e) { return null; }
  const hit = todayCache.get(file);
  if (hit && hit.stamp === stamp) return hit.names;
  let names = null;
  try {
    const day = JSON.parse(fs.readFileSync(file, 'utf-8'));
    names = [...new Set((Array.isArray(day.newPrediction) ? day.newPrediction : []).map((m) => (m && m.name) || 'not known'))];
  } catch (e) { names = null; }
  todayCache.set(file, { stamp, names });
  return names;
}

// The users' top five of a game today: the first five ranked names that are in the newest day's table - the rows
// the Lottery shows a user (the topNames rule). A ranked row missing from today's table
// is passed over and the sixth comes in; without a day file it is the first five of the ranking.
function usersTop(report, game, present) {
  return topNames(rankingOf(report, game), TOP_MODELS_SHOWN, Array.isArray(present) ? present : null);
}

module.exports = { TOP_MODELS_SHOWN, METRIC_TEXT, metricTextOf, rankingOf, topNames, loadReport, rankingFor, namesToday, usersTop };
