// readings.js - the week's reading (21 Oct 2026, the owner's decision): data/readings/<sunday>.json, one summary per
// weekend of the four Sunday controls for every game - Q0's row tests as the report on disk ran them (27 Oct 2026: under
// 5% against what chance puts there, holds, the rows followed and confirmed on new draws; the band as context), Q2's real,
// shuffled and threshold AUC with its seeds and test windows; each feature-control variant's verdict, noise ratio
// and held-out AUC; every market row against uniform and against GARCH - each part with its own record's time.
// data/controls/ is gitignored and overwritten every Sunday; a reading is not: it is tracked in git (the daily
// commit's `git add -A` sweeps it in), so "week after week" becomes something the admin pages can count ("2nd week
// above") instead of a memory note.
//
// jobs.js writes it when the Sunday plan's last job (nullControls) finishes, whatever its exit code. The CLI rebuilds
// a week or backfills one from a copy of that week's records:
//   node readings.js --controls <dir> --sunday YYYY-MM-DD [--database <dir>] [--out <dir>] [--note <text>]
// --database: the data/database folder whose modelPerformance.json holds the Q0 row tests, by default this
// checkout's. For a backfill that is the ranking as it stood that Sunday (git show <commit>:data/database/
// modelPerformance.json into a folder), never today's: a later ranking read against an old band is hindsight.
// The Q2 records are read by today's controls.js: rebuilt from a week before 1 Nov 2026 (verdict rule 1), a record is
// re-judged by the windows rule of 27 Oct 2026 - cryptoweek on 18 Oct reads "too few draws" where the stored reading
// says "above" - so rebuild such a week only to replace a reading that is missing or broken.
//
// What a reading holds is built by admin.js weekReading - the Controls page's own builders - so it is what the page
// showed that week. Nothing here colours or judges a number.
'use strict';

const fs = require('fs');
const path = require('path');
const admin = require('./admin');
const brussels = require('./brussels');
const rankings = require('./ranking');

const ROOT = __dirname;
const USAGE = 'usage: node readings.js --controls <dir> --sunday YYYY-MM-DD [--database <dir>] [--out <dir>] [--note <text>]';
const isSunday = (day) => brussels.isDay(day) && brussels.weekdayOfDay(day) === 0;

// The reading of one week from the records in controlsDir (default <root>/data/controls) and the ranking in
// databaseDir (default <root>/data/database). Reads only; throws on a day that is not a Sunday.
function buildReading({ controlsDir, databaseDir, root = ROOT, sunday, now = new Date(), by = 'scheduler', note = null } = {}) {
  if (!isSunday(sunday)) throw new Error(`not a Sunday: ${sunday}`);
  const dirs = admin.resolveDirs({ root, controlsDir, databaseDir });
  const inside = path.relative(dirs.root, dirs.controlsDir);
  return admin.weekReading(admin.readAll(dirs), rankings.loadReport(dirs.databaseDir), {
    sunday, now, by, note, controls: inside && !inside.startsWith('..') && !path.isAbsolute(inside) ? inside : dirs.controlsDir,
  });
}

// One line per game, so a week's diff in git reads game by game; the other keys one per line.
function serialize(reading) {
  const { games, ...head } = reading;
  const top = Object.entries(head).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`);
  const body = Object.entries(games || {}).map(([g, v]) => `    ${JSON.stringify(g)}: ${JSON.stringify(v)}`);
  return `{\n${top.join(',\n')},\n  "games": {\n${body.join(',\n')}\n  }\n}\n`;
}

// Writes <outDir or root/data/readings>/<sunday>.json through a temporary file in the same folder and a rename, so a
// reader never sees half a file. The same week rewrites the same file; a reading that differs from the stored one
// only in when it was built leaves the file as it is (nothing new for git). Answers {file, written, bytes}.
function writeReading(reading, { root = ROOT, outDir } = {}) {
  if (!reading || !isSunday(reading.sunday) || !reading.games) throw new Error('not a reading of a Sunday');
  const dir = outDir || path.join(root, 'data', 'readings');
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${reading.sunday}.json`);
  const text = serialize(reading);
  const bytes = Buffer.byteLength(text);
  let same = false;
  try { same = JSON.stringify({ ...JSON.parse(fs.readFileSync(file, 'utf-8')), builtAt: null }) === JSON.stringify({ ...reading, builtAt: null }); } catch (e) { same = false; }
  if (same) return { file, written: false, bytes };
  const tmp = path.join(dir, `.${reading.sunday}.json.${process.pid}.tmp`);
  try {
    fs.writeFileSync(tmp, text);
    fs.renameSync(tmp, file);
  } catch (e) {
    try { fs.unlinkSync(tmp); } catch (x) { /* never written */ }
    throw e;
  }
  return { file, written: true, bytes };
}

// --- the CLI ------------------------------------------------------------------------------
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const m = /^--(controls|sunday|database|out|note)(?:=(.*))?$/.exec(argv[i]);
    if (!m) throw new Error(`unknown argument: ${argv[i]}`);
    const value = m[2] !== undefined ? m[2] : argv[(i += 1)];
    if (value === undefined || value === '') throw new Error(`--${m[1]} needs a value`);
    out[m[1]] = value;
  }
  return out;
}
const f3 = (x) => (x === null || x === undefined ? '-' : Number(x).toFixed(3));
// "±0.019" after a figure that has an error, nothing after one that does not (a Q2 record read under rule 1)
const pm3 = (x) => (x === null || x === undefined ? '' : `±${f3(x)}`);
// one line per game, to check a reading against its records by eye. Q0 (27 Oct 2026): the row tests - under 5% next to
// what chance puts there, holds, followed, confirmed - with the band as context; a reading from before reads "band rule"
const f2 = (x) => (x === null || x === undefined ? '-' : Number(x).toFixed(2));
function q0Line(q, unit) {
  if (!q) return 'Q0 none';
  const band = q.band ? ` (band ${f3(q.band.ceiling)}, context)` : '';
  if (!q.rule) return `Q0 band rule: ${Array.isArray(q.above) ? `${q.above.length} above` : 'no ranking'}${band}`;
  const t = q.test;
  if (!t) return `Q0 no row test (${q.error || 'not in the report'})${band}`;
  if (!t.family) return `Q0 too few live ${unit}: ${t.liveMax} of ${t.minDraws}${band}`;
  return `Q0 ${t.under} of ${t.family} under 5% (≈${f2(t.expected)} by chance), ${t.holds} hold, ${t.followed} followed, ${t.confirmed} confirmed${band}`;
}
function describeGame(game, g) {
  const q0 = q0Line(g.q0, g.unit);
  // Q2: the threshold with its standard error and the shuffled control against its own bound from rule 2 (1 Nov 2026) on
  // (", bound": the real mean is what is held against it, as on the Controls page - not the shuffled mean)
  const shuffledBound = g.q2 && g.q2.shuffledThreshold !== null && g.q2.shuffledThreshold !== undefined ? `, bound ${f3(g.q2.shuffledThreshold)}` : '';
  const q2 = !g.q2 ? 'Q2 none' : `Q2 ${g.q2.status} ${f3(g.q2.real)} vs ${f3(g.q2.threshold)}${pm3(g.q2.thresholdSe)} (shuffled ${f3(g.q2.shuffled)}${shuffledBound}, `
    + `${g.q2.seeds} seeds, ${g.q2.windowsTest} test windows, rule ${g.q2.rule || 1})`;
  const fc = !g.fc ? 'FC none' : `FC ${g.fc.variants.map((v) => `${v.key} ${v.verdict}`).join(', ')}`;
  const rows = !g.rows ? '' : ` | rows ${g.rows.rows.filter((r) => r.vsReference && r.vsReference.verdict === 'better').length} of ${g.rows.rows.length - 1} beat ${g.rows.reference} over ${g.rows.window} ${g.unit}`;
  return `  ${game}: ${q0} | ${q2} | ${fc}${rows}${g.errors.length ? ` | unreadable: ${g.errors.join(', ')}` : ''}`;
}
function main(argv, say = console.log, complain = console.error) {
  let args;
  try { args = parseArgs(argv); } catch (e) { complain(`${e.message}\n${USAGE}`); return 2; }
  if (!args.controls || !args.sunday) { complain(USAGE); return 2; }
  if (!isSunday(args.sunday)) { complain(`--sunday ${args.sunday} is not a Sunday`); return 2; }
  const controlsDir = path.resolve(args.controls);
  if (!fs.existsSync(controlsDir)) { complain(`no such folder: ${controlsDir}`); return 2; }
  const reading = buildReading({ controlsDir, databaseDir: args.database ? path.resolve(args.database) : undefined, sunday: args.sunday, by: 'cli', note: args.note || null });
  const result = writeReading(reading, { outDir: args.out ? path.resolve(args.out) : undefined });
  say(`${result.written ? 'wrote' : 'unchanged'} ${result.file} (${result.bytes} bytes): records ${reading.from} to ${reading.to}, ranking ${reading.rankingAt || 'none'}`);
  if (reading.notThisWeek.length) say(`  not from the week of ${reading.sunday}: ${reading.notThisWeek.map((t) => `${t.game} ${t.part} (${t.at})`).join(', ')}`);
  Object.entries(reading.games).forEach(([game, g]) => say(describeGame(game, g)));
  return 0;
}

if (require.main === module) process.exitCode = main(process.argv.slice(2));

module.exports = { buildReading, writeReading, serialize, isSunday, main };
