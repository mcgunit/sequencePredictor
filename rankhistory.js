// rankhistory.js - the History ranking as it stood at any moment (22 Oct 2026, admin batch 4; the owner's rule: no
// hindsight). The admin's All / Top switch on a past draw shows the five rows ranked first BEFORE that draw, never
// today's five; the users' Lottery (the next batch) counts its history over the same five.
//
// The source is git: every daily run commits data/database/modelPerformance.json (Helpers.git_push, "saving last
// predictions"), one version per run since the first on 25 Aug 2026 05:21 UTC. Before it no ranking was recorded, and
// nothing here invents one. Each version is read once with `git show <sha>:<file>` and reduced to what a ranking is -
// per game the ranked names in order, the metric, the minimum of scored draws, each name's value and draws - kept in
// memory and in a gitignored cache file keyed by sha (a sha names its content and its commit time, so an entry never
// goes stale); a new commit is read when HEAD moves. git runs through gates.js's pooled execFile: fixed arguments, a
// timeout, at most four at once; when git is missing or fails the history says so and the callers show no Top. One
// version git could not read (a timeout, an overflow) leaves the draws it would decide without a Top - never an older
// one in its place - and is asked again after 30 s, not on every request.
//
// A version that does not parse is skipped, as ranking.js keeps the last good report when the file is caught half
// written: the pages showed the version before it.
//
// THE CUT-OFF: the ranking users saw before a draw is the newest version committed before the draw's day ended its
// morning run - 15:00 Brussels time on the draw's date. Every game here draws, or closes, later that day: lotto
// (Wed, Sat), euromillions (Tue, Fri), eurodreams (Mon, Thu) and vikinglotto (Wed) draw in the evening, keno, pick3 and
// Joker+ every evening; a crypto day closes at 00:00 UTC (02:00 Brussels in summer, 01:00 in winter), a share
// session at 16:00 New York (22:00 Brussels in most weeks, never before 19:00 on an early close). The daily run
// starts at 09:00 Brussels and its commit had landed by 12:36 Brussels on every day of the history (3 Sep). A hand run
// in the afternoon or evening (25 Aug, 2, 7, 8 and 10 Sep) counts from the next draw on: it may hold that day's own
// result, so 15:00 keeps it out even where it came before the draw. A run later than 15:00 on a normal day means that
// day's draw is read on the day before's version - an older ranking, never a later one.
// The week games (cryptoweek, dated by the week's Sunday; sharesweek, by its last session) are one draw per week: the
// cut-off is 15:00 Brussels on the week's Monday - after the run that made the week's ticket (Monday for crypto, the
// Saturday before for shares), before the share week's first open (15:30 Brussels) - the ranking the week was
// followed on.
'use strict';

const fs = require('fs');
const path = require('path');
const brussels = require('./brussels');
const gates = require('./gates');
const rankings = require('./ranking');

const REPORT_PATH = 'data/database/modelPerformance.json';
const CACHE_VERSION = 1;
const CUTOFF_HOUR = 15;                                  // Brussels wall time on the draw's (the week's first) day
const WEEK_GAMES = ['cryptoweek', 'sharesweek'];
const TOP_N = rankings.TOP_MODELS_SHOWN;
const GAME = /^[a-z0-9]+$/;
const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

const num = (x) => { const v = Number(x); return x === null || x === undefined || x === '' || !Number.isFinite(v) ? null : v; };
const r4 = (x) => (num(x) === null ? null : Math.round(num(x) * 10000) / 10000);
// modelPerformance's generatedAt is UTC written without a zone
function isoUtc(value) {
  if (value === null || value === undefined || value === '') return null;
  const text = String(value);
  const date = new Date(/[zZ]$|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isNaN(date.getTime()) ? null : date.toISOString().replace(/\.\d{3}Z$/, 'Z');
}
const isoMs = (ms) => (Number.isFinite(ms) ? new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null);

// One version of the report, reduced: {generatedAt, games: {game: {names, metric, minDraws, values, draws}}}; null
// when the text is not a report.
function reduce(text) {
  let report;
  try { report = JSON.parse(text); } catch (e) { return null; }
  if (!report || typeof report !== 'object' || !report.games || typeof report.games !== 'object') return null;
  const games = {};
  Object.entries(report.games).forEach(([game, info]) => {
    if (!GAME.test(game) || !info || !Array.isArray(info.models) || !info.models.length) return;
    const models = info.models.filter((m) => m && m.name !== undefined && m.name !== null);
    games[game] = {
      names: models.map((m) => String(m.name)), metric: String(info.metric || 'avg_hits'), minDraws: num(info.minDrawsForRanking),
      values: models.map((m) => r4(m[info.metric || 'avg_hits'])), draws: models.map((m) => num(m.draws)),
    };
  });
  return { generatedAt: isoUtc(report.generatedAt), games };
}

// --- the cache: memory, and a file -----------------------------------------------------------------------------
const memory = new Map();        // sha -> {generatedAt, games} | {skip: 'not in commit' | 'not a report'}
const cacheRead = new Set();     // cache files already read into memory
const lists = new Map();         // root -> {head, history, until (a history with an unread version: kept until then)}
const failed = new Map();        // sha -> {until, error}: a version git could not read, not asked again before `until`
const RETRY_MS = 30 * 1000;      // as often as gates.js reads HEAD again
// A cache entry as readVersion makes it: a skip, or a reduced report whose every game has its names (strings) with a
// value and a draw count each. Anything else (a hand edit, a reduce() changed without a CACHE_VERSION bump) is not
// taken, and that version is read again from git.
function validEntry(v) {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return false;
  if (v.skip !== undefined) return v.skip === 'not in commit' || v.skip === 'not a report';
  if ((v.generatedAt !== null && typeof v.generatedAt !== 'string') || !v.games || typeof v.games !== 'object' || Array.isArray(v.games)) return false;
  return Object.entries(v.games).every(([game, g]) => GAME.test(game) && g && typeof g === 'object' && typeof g.metric === 'string'
    && Array.isArray(g.names) && g.names.every((n) => typeof n === 'string')
    && Array.isArray(g.values) && g.values.length === g.names.length && Array.isArray(g.draws) && g.draws.length === g.names.length);
}
function readCache(file) {
  if (!file || cacheRead.has(file)) return;
  cacheRead.add(file);
  let parsed;
  try { parsed = JSON.parse(fs.readFileSync(file, 'utf-8')); } catch (e) { return; }   // none yet, or cut short: read again from git
  if (!parsed || parsed.version !== CACHE_VERSION || parsed.file !== REPORT_PATH || !parsed.versions || typeof parsed.versions !== 'object') return;
  Object.entries(parsed.versions).forEach(([sha, v]) => { if (gates.SHA.test(sha) && validEntry(v) && !memory.has(sha)) memory.set(sha, v); });
}
// Written whole to a temporary file and renamed, so a reader never sees half of it; a failure is logged and the
// history carries on from memory.
function writeCache(file) {
  if (!file) return;
  const out = { version: CACHE_VERSION, file: REPORT_PATH, note: 'rankhistory.js: the History ranking per git version, reduced; safe to delete', versions: Object.fromEntries(memory) };
  const tmp = `${file}.${process.pid}.tmp`;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(tmp, JSON.stringify(out));
    fs.renameSync(tmp, file);
  } catch (e) {
    console.error(`[${new Date().toISOString()}] rankhistory.js: could not write the cache file:`, e.message);
    try { fs.unlinkSync(tmp); } catch (e2) { /* never written */ }
  }
}

// One version from git, reduced. A file not in that commit, or not a report there, is a fact about the commit (kept);
// any other failure (a timeout, no git) is not, and rejects.
async function readVersion(root, sha) {
  try {
    return reduce(await gates.git(root, ['show', `${sha}:${REPORT_PATH}`])) || { skip: 'not a report' };
  } catch (error) {
    if (typeof error.code === 'number' && !error.killed && gates.NOT_IN_COMMIT.test(error.stderr || '')) return { skip: 'not in commit' };
    throw error;
  }
}

// --- the cut-off -----------------------------------------------------------------------------------------------
// The instant a draw's ranking is read at (the header explains the hour): 15:00 Brussels on the draw's date, or on
// the week's Monday for a week game. null for a date that is not a day.
function cutoffOf(game, date) {
  if (!ISO_DAY.test(String(date)) || !brussels.isDay(date)) return null;
  const day = WEEK_GAMES.includes(game) ? brussels.addDays(date, -((brussels.weekdayOfDay(date) + 6) % 7)) : date;
  return brussels.instant(day, CUTOFF_HOUR, 0);
}

// --- the history -------------------------------------------------------------------------------------------------
// versions: the readable versions, oldest first: {sha, at (commit time, ms), generatedAt, games}. unread: the commit
// times of the versions git could not read (a timeout, an overflow): a draw whose cut-off comes after one of them,
// with no readable version in between, has no ranking - never the older version before it.
function makeHistory(versions, { head = null, error = null, skipped = 0, unread = [] } = {}) {
  const first = versions.length ? versions[0] : null;
  const firstText = first ? `${brussels.dayOf(first.at)}` : null;
  const none = (code, extra = {}) => {
    const text = {
      unavailable: versions.length
        ? `the ranking saved before this draw could not be read (${error || 'git history not available'})`
        : `the ranking history could not be read (${error || 'git history not available'})`,
      'before-first': `no ranking recorded before ${firstText || 'the first saved ranking'}`,
      'not-ranked': 'the game was not in the History ranking yet',
      'no-ranked-row': 'no row of this draw was in the ranking then',
    }[code];
    return { none: code, text, ...extra };
  };
  // The newest version committed before `instant` (ms): the ranking of `game` in it, or why there is none.
  function rankingAt(game, instant) {
    if (error && !versions.length) return none('unavailable');
    if (!Number.isFinite(instant)) return none('before-first');
    let v = null;
    for (let i = versions.length - 1; i >= 0; i -= 1) { if (versions[i].at < instant) { v = versions[i]; break; } }
    if (unread.some((at) => at < instant && (!v || at > v.at))) return none('unavailable');
    if (!v) return none('before-first', { first: first ? { committedAt: isoMs(first.at), generatedAt: first.generatedAt } : null });
    const g = v.games ? v.games[game] : null;
    const version = { sha: v.sha, committedAt: isoMs(v.at), generatedAt: v.generatedAt };
    if (!g) return none('not-ranked', { version });
    return { version, names: g.names, metric: g.metric, metricText: rankings.METRIC_TEXT[g.metric] || rankings.METRIC_TEXT.avg_hits,
      minDraws: g.minDraws, values: g.values, draws: g.draws };
  }
  // The top n of a draw: the first n names of the ranking at its cut-off that are in the draw's table (present: the
  // names of its rows; a ranked row can be missing from a day). {top, ranking, cutoff} or {none, text, cutoff}.
  function topAt(game, date, present, n = TOP_N) {
    const cutoff = cutoffOf(game, date);
    const at = rankingAt(game, cutoff === null ? NaN : cutoff);
    const cut = isoMs(cutoff);
    if (at.none) return { ...at, cutoff: cut };
    const top = rankings.topNames({ names: at.names }, n, Array.isArray(present) ? present : null);
    if (!top.length) return { ...none('no-ranked-row', { version: at.version }), cutoff: cut };
    return { top, ranking: at, cutoff: cut };
  }
  // the first readable version that ranks `game` (a game entered the ranking after 25 Aug: Joker+, the markets)
  function firstOf(game) {
    const v = versions.find((x) => x.games && x.games[game]);
    return v ? { sha: v.sha, committedAt: isoMs(v.at), generatedAt: v.generatedAt } : null;
  }
  return { head, error, skipped, unread: unread.length, versions, first: first ? { sha: first.sha, committedAt: isoMs(first.at), generatedAt: first.generatedAt } : null,
    rankingAt, topAt, cutoffOf, firstOf };
}

// The history of the checkout at `root`, newest HEAD (re-read at most every 30 s, as gates.js): the versions read
// before come from memory or the cache file, a new commit's from git. Never rejects: a git failure is the history's
// `error` (a fixed phrase; the full message goes to the server log), with whatever versions are known. Requests that
// arrive while a load runs share it, so a new commit is read once.
const pending = new Map();       // root -> the load in progress
function load(root, options = {}) {
  if (!pending.has(root)) pending.set(root, loadNow(root, options).finally(() => pending.delete(root)));
  return pending.get(root);
}
async function loadNow(root, { cacheFile = null } = {}) {
  readCache(cacheFile);
  let head;
  try { head = await gates.headOf(root); } catch (e) { return makeHistory([], { error: gates.gitError(e, 'rankhistory.js') }); }
  const hit = lists.get(root);
  if (hit && hit.head === head && (!hit.until || Date.now() < hit.until)) return hit.history;
  let commits;
  try {
    const out = await gates.git(root, ['log', '--format=%H%x09%ct', head, '--', REPORT_PATH]);
    commits = String(out).split('\n').map((line) => line.split('\t')).filter((p) => p.length === 2 && gates.SHA.test(p[0]))
      .map(([sha, ct]) => ({ sha, at: Number(ct) * 1000 })).filter((c) => Number.isFinite(c.at));
  } catch (e) { return makeHistory([], { error: gates.gitError(e, 'rankhistory.js') }); }
  // git log lists newest first; oldest first by commit time, git's order kept on a tie
  commits = commits.reverse().map((c, i) => ({ ...c, i })).sort((a, b) => a.at - b.at || a.i - b.i);
  // a version git could not read is asked again after RETRY_MS, not on every request
  const missing = commits.filter((c) => !memory.has(c.sha) && !(failed.has(c.sha) && Date.now() < failed.get(c.sha).until));
  let read = 0;
  if (missing.length) {
    await Promise.all(missing.map(async (c) => {
      try {
        memory.set(c.sha, await readVersion(root, c.sha));
        failed.delete(c.sha);
        read += 1;
      } catch (e) { failed.set(c.sha, { until: Date.now() + RETRY_MS, error: gates.gitError(e, 'rankhistory.js') }); }
    }));
    if (read) writeCache(cacheFile);
  }
  const unread = commits.filter((c) => !memory.has(c.sha));
  const error = unread.length ? (failed.get(unread[0].sha) || {}).error || 'a version could not be read' : null;
  const versions = commits.filter((c) => memory.has(c.sha) && !memory.get(c.sha).skip).map((c) => ({ sha: c.sha, at: c.at, ...memory.get(c.sha) }));
  const history = makeHistory(versions, { head, error, skipped: commits.filter((c) => memory.has(c.sha) && memory.get(c.sha).skip).length, unread: unread.map((c) => c.at) });
  // a history with a version that could not be read is kept until that version is asked again
  lists.set(root, error ? { head, history, until: Date.now() + RETRY_MS } : { head, history });
  return history;
}

// For the tests: forget what was read (memory, the cache files read, the per-root lists).
function reset() { memory.clear(); cacheRead.clear(); lists.clear(); failed.clear(); }

module.exports = { REPORT_PATH, CUTOFF_HOUR, WEEK_GAMES, reduce, cutoffOf, makeHistory, load, reset };
