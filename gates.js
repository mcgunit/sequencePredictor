// gates.js - the champion/challenger gate's decisions (src/TuningGate.py) for the admin section's Tuning gates and
// This week pages (21 Oct 2026, admin plan 3.3). This weekend's decisions are read from
// bestParams_<game>.json["tuningGate"] at the checkout's root; earlier weekends' from git: the tuners' own
// "Saving latest <tuner> hyperopt" commits (Helpers.git_push), read with `git show <sha>:bestParams_<game>.json`.
//
// Which tuner decided a row. A record written since 21 Oct 2026 names the script that wrote it ("tuner",
// TuningGate.challenge). An older one is attributed by its row's family - the rows each tuner gates, named exactly
// as the tuners write them - and only when an independent fact agrees: the tuner commit that first carries the
// record, or else the scheduler run whose window holds its date. When the two disagree, or nothing confirms the
// family, the record is "unattributed" - never a guess from times alone (admin plan decision 5: three prototypes
// disagreed per tuner, one of them by pulling records decided a minute after a run into it).
//
// git runs through execFile with fixed arguments (no shell): a sha must be 40 hex digits and a game a plain name,
// every call has a timeout, at most four run at once, and nothing waits on git when it is missing or slow - the
// pages then say the history is not available. What a commit holds never changes, so a file is read once per sha;
// the list of commits once per HEAD.
'use strict';

const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const brussels = require('./brussels');

const GIT_TIMEOUT_MS = 5000;
const GIT_PARALLEL = 4;
const HEAD_TTL_MS = 30 * 1000;      // how long a HEAD read is trusted before the commit list is checked again
const COMMITS_SCANNED = 200;        // tuner commits (five a weekend): about 40 weekends
const WEEKENDS_SHOWN = 4;
const DECISIONS = ['replaced', 'kept', 'reverted'];

// The tuners in chain order: the label the pages use, the scheduler job, the script (a record's "tuner"), the
// commit subject's middle part, and the rows each one gates (HyperoptStatistics STRATEGY_DISPLAY_NAMES and
// KenoSubsetTuning, HyperoptBoost STRATEGY_DISPLAY_NAMES, HyperoptRLTicket TICKET_ROW / POSITION_ROW,
// HyperoptEnsemble ROW_NAME, HyperoptQuantum VARIANT_ROWS). A row not named here has no family.
const TUNERS = [
  { tuner: 'statistical', job: 'hyperoptStatistics', script: 'HyperoptStatistics', commit: 'statistical',
    rows: ['Markov Model', 'MarkovMonteCarlo Model', 'MarkovBayesian Model', 'MarkovBayesianEnhanched Model', 'PoissonMonteCarlo Model',
      'PoissonMarkov Model', 'LaplaceMonteCarlo Model', 'HybridStatisticalModel', 'GARCH Model', 'Regime HMM Model', 'KenoSubsetTuning'] },
  { tuner: 'boosting', job: 'hyperoptBoost', script: 'HyperoptBoost', commit: 'boosting',
    rows: ['XGBoost Model', 'XGBoostMultiLabel Model', 'LightGBM Model', 'LightGBMMultiLabel Model', 'CatBoost Model', 'CatBoostMultiLabel Model'] },
  { tuner: 'RL ticket', job: 'hyperoptRLTicket', script: 'HyperoptRLTicket', commit: 'RL ticket', rows: ['RL Ticket Model', 'RL Position Model'] },
  { tuner: 'ensemble', job: 'hyperoptEnsemble', script: 'HyperoptEnsemble', commit: 'subset ensemble', rows: ['SubsetEnsemble Model'] },
  { tuner: 'quantum', job: 'hyperoptQuantum', script: 'HyperoptQuantum', commit: 'quantum', rows: ['QuantumMetaLearner Model', 'QuantumVQC Model', 'ClassicalSVM Model'] },
];
const SAVE_SUBJECT = /^Saving latest (.+) hyperopt$/;
const SHA = /^[0-9a-f]{40}$/;
const GAME = /^[a-z0-9]+$/;

const familyOf = (row) => (TUNERS.find((t) => t.rows.includes(row)) || {}).tuner || null;
const tunerOfScript = (script) => {
  const name = path.basename(String(script || '')).replace(/\.py$/, '');
  return (TUNERS.find((t) => t.script === name) || {}).tuner || null;
};
const tunerOfSubject = (subject) => {
  const m = SAVE_SUBJECT.exec(String(subject || '').trim());
  return m ? (TUNERS.find((t) => t.commit === m[1]) || {}).tuner || null : null;
};
const tunerOfJob = (job) => (TUNERS.find((t) => t.job === job) || {}).tuner || null;
// The weekend a decision or a commit belongs to, named by its Saturday (Brussels): the chain starts on Saturday and
// can run into Monday; a hand run during the week counts with the weekend before it.
const weekendOf = (ms) => brussels.lastWeekday(ms, 6);
const dateOf = (record) => { const ms = Date.parse(record && record.date); return Number.isFinite(ms) ? ms : null; };

// --- this weekend: the working tree -------------------------------------------------
const gatesOf = (served) => (served && served.tuningGate && typeof served.tuningGate === 'object' && !Array.isArray(served.tuningGate) ? served.tuningGate : {});
// One game's tuningGate records, parsed once per change of its file; null when the file is missing. A tuner writes
// the file in place, so a read can catch it half written: a file that does not parse keeps the last good records (and
// is read again on its next change), as ranking.js does with modelPerformance.json.
const fileCache = new Map();   // file -> {stamp, gates}
function readGates(root, game) {
  const file = path.join(root, `bestParams_${game}.json`);
  let stamp;
  try { const s = fs.statSync(file); stamp = `${s.mtimeMs}:${s.size}`; } catch (e) { fileCache.delete(file); return null; }
  const hit = fileCache.get(file);
  if (hit && hit.stamp === stamp) return hit.gates;
  let gates;
  try { gates = gatesOf(JSON.parse(fs.readFileSync(file, 'utf-8'))); } catch (e) { gates = hit ? hit.gates : null; }
  fileCache.set(file, { stamp, gates });
  return gates;
}
// The mtime/size signature of the files readGates reads, for a caller's own cache.
const stampOf = (root, games) => games.map((g) => { try { const s = fs.statSync(path.join(root, `bestParams_${g}.json`)); return `${s.mtimeMs}:${s.size}`; } catch (e) { return 'none'; } }).join('|');

// --- the numbers of one decision ---------------------------------------------------------
const num = (x) => { const v = Number(x); return x === null || x === undefined || x === '' || typeof x === 'boolean' || !Number.isFinite(v) ? null : v; };
// What a record says, as numbers: the three scores with their standard errors and the days each was scored on, the
// margin and the bar the challenger had to clear (decide(): the better scored reference plus the margin), and the
// challenger's gain in its own standard errors - over the incumbent, over the better reference, and over the bar
// (the one that decided). null wherever a side was not scored. A keno capped-profit score (TuningScore, since 7 Oct
// 2026) is the capped profit above a fair ticket of the same size, not money: such a side carries the fair ticket's
// own capped value (fairPerBet) and the raw, uncapped profit per bet in EUR (profitPerBet) next to its score.
function describe(row, record) {
  const r = record || {};
  const side = (score, tuning) => {
    const t = tuning || {};
    const out = { score: num(score), se: num(t.score_se), days: num(t.days) };
    if (num(t.fair_capped_per_bet) !== null) { out.fairPerBet = num(t.fair_capped_per_bet); out.profitPerBet = num(t.profit_per_bet); }
    return out;
  };
  const ch = side(r.challenger, r.challenger_tuning);
  const inc = side(r.incumbent, r.incumbent_tuning);
  const def = side(r.default, r.default_tuning);
  const margin = num(r.margin) === null ? 0 : num(r.margin);
  const refs = [inc.score, def.score].filter((x) => x !== null);
  const best = refs.length ? Math.max(...refs) : null;
  const bar = best === null ? null : best + margin;
  const per = (gain) => (ch.score !== null && gain !== null && ch.se ? (ch.score - gain) / ch.se : null);
  const at = dateOf(r);
  return {
    row, decision: DECISIONS.includes(r.decision) ? r.decision : null, date: at === null ? null : new Date(at).toISOString().replace(/\.\d{3}Z$/, 'Z'),
    weekend: at === null ? null : weekendOf(at), script: r.tuner ? String(r.tuner) : null,
    kind: ((r.challenger_tuning || {}).kind) || null, windowDays: num(r.window_days),
    runTrials: num(r.run_trials), studyTrials: num(r.study_trials),
    challenger: ch, incumbent: inc, default: def, margin,
    // the margin in the challenger's own standard errors: 1 for the tuners that ask for one SE, 0 for "any gain"
    marginSe: ch.se ? margin / ch.se : null,
    bar, gainSe: per(inc.score), gainBestSe: per(best), gainBarSe: per(bar),
  };
}

// --- who decided it ---------------------------------------------------------------------
// carrier: the tuner commit that first carries this record ({sha, tuner}); run: the scheduler run whose window holds
// its date ({tuner, job}). Both optional.
function attribute(row, record, { carrier = null, run = null } = {}) {
  if (record && record.tuner) {
    const tuner = tunerOfScript(record.tuner);
    return tuner ? { tuner, by: 'record' } : { tuner: 'unattributed', by: 'record', why: `written by ${record.tuner}, not a known tuner` };
  }
  const family = familyOf(row);
  if (!family) return { tuner: 'unattributed', why: 'no tuner gates a row of this name' };
  const evidence = carrier ? { tuner: carrier.tuner, by: 'commit' } : (run ? { tuner: run.tuner, by: 'run' } : null);
  if (!evidence) return { tuner: 'unattributed', family, why: 'no tuner commit or scheduler run confirms the row\'s family' };
  if (evidence.tuner !== family) return { tuner: 'unattributed', family, why: `the row is a ${family} row but the ${evidence.by} is the ${evidence.tuner} tuner's` };
  return { tuner: family, by: `family + ${evidence.by}` };
}
// The scheduler run whose own window holds a decision; a minute of slack only when no run's window does (the gate's
// date is taken before the tuner writes and commits). runs: [{key, at, finishedAt}] of config/scheduler.json.
function runOf(ms, runs) {
  const tuned = (runs || []).filter((h) => tunerOfJob(h.key) && Number.isFinite(h.at) && Number.isFinite(h.finishedAt));
  const within = (pad) => tuned.find((h) => h.at - pad <= ms && ms <= h.finishedAt + pad) || null;
  const hit = within(0) || within(60 * 1000);
  return hit ? { tuner: tunerOfJob(hit.key), job: hit.key } : null;
}

// --- git -------------------------------------------------------------------------------------
let active = 0;
const queued = [];
function pooled(task) {
  return new Promise((resolve, reject) => {
    const run = () => {
      active += 1;
      Promise.resolve().then(task).then(resolve, reject).finally(() => {
        active -= 1;
        const next = queued.shift();
        if (next) next();
      });
    };
    if (active < GIT_PARALLEL) run(); else queued.push(run);
  });
}
function git(root, args) {
  return pooled(() => new Promise((resolve, reject) => {
    execFile('git', args, {
      cwd: root, timeout: GIT_TIMEOUT_MS, maxBuffer: 32 * 1024 * 1024, encoding: 'utf8', windowsHide: true,
      env: { PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin', HOME: process.env.HOME || '/root', LANG: 'C', GIT_TERMINAL_PROMPT: '0' },
    }, (error, stdout, stderr) => {
      if (error) { error.stderr = String(stderr || ''); reject(error); } else resolve(stdout);
    });
  }));
}
const heads = new Map();   // root -> {at, sha}
async function headOf(root) {
  const hit = heads.get(root);
  if (hit && Date.now() - hit.at < HEAD_TTL_MS) return hit.sha;
  const sha = String(await git(root, ['rev-parse', 'HEAD'])).trim();
  if (!SHA.test(sha)) throw new Error('git rev-parse HEAD did not name a commit');
  heads.set(root, { at: Date.now(), sha });
  return sha;
}
// The tuners' save commits reachable from HEAD, oldest first: {sha, at (committer time, ms), tuner, weekend}.
const commitLists = new Map();   // root -> {head, commits}
async function saveCommits(root) {
  const head = await headOf(root);
  const hit = commitLists.get(root);
  if (hit && hit.head === head) return hit.commits;
  const out = await git(root, ['log', `-n${COMMITS_SCANNED}`, '--format=%H%x09%ct%x09%s', '--grep=^Saving latest ', head]);
  const commits = String(out).split('\n').map((line) => line.split('\t')).filter((p) => p.length >= 3 && SHA.test(p[0]))
    .map(([sha, ct, ...subject]) => ({ sha, at: Number(ct) * 1000, tuner: tunerOfSubject(subject.join('\t')) }))
    .filter((c) => c.tuner && Number.isFinite(c.at))
    .map((c) => ({ ...c, weekend: weekendOf(c.at) }))
    .sort((a, b) => a.at - b.at);
  commitLists.set(root, { head, commits });
  return commits;
}
// One game's records at a commit, kept small ({row: {date, decision}}); null when the file is not in that commit, or
// is there but not JSON (Python's json.dump writes NaN): facts about the commit, kept - read once per sha and game, a
// commit never changes. Any other failure (a timeout, no git, a git that exits for another reason - a read racing a
// repack) is not cached, so the next request tries again.
const NOT_IN_COMMIT = /does not exist in|exists on disk, but not in/;
const blobs = new Map();   // root@sha:game -> Promise
function gatesAt(root, sha, game) {
  if (!SHA.test(String(sha)) || !GAME.test(String(game))) return Promise.resolve(null);
  const key = `${root}@${sha}:${game}`;
  if (!blobs.has(key)) {
    const read = git(root, ['show', `${sha}:bestParams_${game}.json`]).then((text) => {
      let parsed;
      try { parsed = JSON.parse(text); } catch (e) { return null; }
      const out = {};
      Object.entries(gatesOf(parsed)).forEach(([row, r]) => { out[row] = { date: dateOf(r), decision: r && DECISIONS.includes(r.decision) ? r.decision : null }; });
      return out;
    }, (error) => {
      if (typeof error.code === 'number' && !error.killed && NOT_IN_COMMIT.test(error.stderr || '')) return null;
      blobs.delete(key);
      throw error;
    });
    blobs.set(key, read);
  }
  return blobs.get(key);
}

// --- the weekends --------------------------------------------------------------------------
// current: {game: tuningGate records} of the working tree. Answers
//   weekends   the last WEEKENDS_SHOWN weekends with decisions, newest first (their Saturdays)
//   decisions  {game: {row: [decision or null per weekend]}} - null: that weekend did not decide the row (its record
//              there is from an earlier weekend, or it has none)
//   carriers   {game: {row: {sha, tuner}}}: the tuner commit that first carries each current record
//   commits    the tuner commits read; error: why git could not be read (then only the working tree's weekend)
// This weekend - the working tree's newest - is read from the working tree; each earlier one from its last tuner commit,
// which carries every tuner's records of that weekend (each commit adds everything the working tree changed).
async function history(root, games, current, { weekends: shown = WEEKENDS_SHOWN } = {}) {
  const safeGames = games.filter((g) => GAME.test(g));
  const ownWeekends = new Set();
  safeGames.forEach((g) => Object.values(current[g] || {}).forEach((r) => { const ms = dateOf(r); if (ms !== null) ownWeekends.add(weekendOf(ms)); }));
  const newest = [...ownWeekends].sort().pop() || null;
  let commits = [];
  let error = null;
  try { commits = await saveCommits(root); } catch (e) { error = gitError(e); }
  const weekends = [...new Set([...commits.map((c) => c.weekend), ...(newest ? [newest] : [])])]
    .filter((w) => !newest || w <= newest).sort().reverse().slice(0, shown);
  const lastOf = (w) => commits.filter((c) => c.weekend === w).pop() || null;
  const decisions = {};
  const carriers = {};
  await Promise.all(safeGames.map(async (game) => {
    const rows = {};
    const own = current[game] || {};
    // each earlier weekend's records, from its last tuner commit
    const earlier = await Promise.all(weekends.map(async (w) => {
      if (w === newest) return null;
      const c = lastOf(w);
      if (!c || error) return undefined;
      try { return await gatesAt(root, c.sha, game); } catch (e) { error = error || gitError(e); return undefined; }
    }));
    const names = new Set(Object.keys(own));
    earlier.forEach((g) => Object.keys(g || {}).forEach((row) => names.add(row)));
    names.forEach((row) => {
      rows[row] = weekends.map((w, i) => {
        if (w === newest) { const ms = dateOf(own[row]); return own[row] && ms !== null && weekendOf(ms) === w ? own[row].decision || null : null; }
        const at = earlier[i];
        if (at === undefined) return undefined;   // that weekend could not be read
        const r = at && at[row];
        return r && r.date !== null && weekendOf(r.date) === w ? r.decision : null;
      });
    });
    decisions[game] = rows;
    // the first tuner commit of a record's own weekend that carries it, date for date
    const found = {};
    const byWeekend = {};
    Object.entries(own).forEach(([row, r]) => { const ms = dateOf(r); if (ms !== null) (byWeekend[weekendOf(ms)] = byWeekend[weekendOf(ms)] || []).push([row, ms]); });
    await Promise.all(Object.entries(byWeekend).map(async ([w, recs]) => {
      const inWeekend = commits.filter((c) => c.weekend === w);
      let files;
      try { files = await Promise.all(inWeekend.map((c) => gatesAt(root, c.sha, game))); } catch (e) { error = error || gitError(e); return; }
      recs.forEach(([row, ms]) => {
        const k = files.findIndex((f) => f && f[row] && f[row].date === ms);
        if (k >= 0) found[row] = { sha: inWeekend[k].sha, tuner: inWeekend[k].tuner };
      });
    }));
    carriers[game] = found;
  }));
  return { weekends, newest, decisions, carriers, commits: commits.length, error };
}
// Why git could not be read, in a fixed phrase for the page (git's own words can name the checkout's path, a parse
// error quotes the file); the full message goes to the server log.
const gitError = (e) => {
  console.error(`[${new Date().toISOString()}] gates.js: git:`, String((e && (e.stderr || e.message)) || e).trim());
  if (e && e.killed) return `git took over ${GIT_TIMEOUT_MS / 1000} s`;
  if (e && e.code === 'ENOENT') return 'git is not installed here';
  return typeof (e && e.code) === 'number' ? `git exited with code ${e.code}` : 'git failed';
};

module.exports = { TUNERS, DECISIONS, WEEKENDS_SHOWN, familyOf, tunerOfScript, tunerOfSubject, tunerOfJob, weekendOf, readGates, stampOf,
  describe, attribute, runOf, saveCommits, gatesAt, history };
