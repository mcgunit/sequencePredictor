// admin.js - the JSON behind the admin section of the React app (21 Oct 2026: the owner's decision to replace the
// old pages with the app; the users get a slim Lottery / Crypto / Shares / Council, the administrator everything).
// Read-only routes under /api/admin, all behind the session (auth.middleware answers a signed-out request with a
// JSON 401 first) and behind the admin guard below (a signed-in user who is not the administrator gets a JSON 403):
//
//   GET /api/admin/controls               the four Sunday controls for all 11 games, grouped Lottery / Markets
//   GET /api/admin/controls/:game         one game's four controls in full (the drawer)
//   GET /api/admin/models                 Best model per game, with each game's top five (the users' rows)
//   GET /api/admin/models/:game           one game's ranking, its lotto multi-pick table and its best combination
//   GET /api/admin/phase-shift[/:game]    the tracked lag peaks (the old "Phase-shift check" card)
//   GET /api/admin/randomness[/:game]     the Randomness watch (entropy, KL, the autoencoder, the Q2 verdict)
//   GET /api/admin/week                   This week: the weekend's runs, today's predictor, the tuners, the controls
//                                         matrix with each cell's run of weeks, since Monday (phase P2)
//   GET /api/admin/tuning[/:game]         Tuning gates: this weekend's decisions by tuner and by game, one game's
//                                         rows with their scores and their last four weekends (phase P2)
//
// Week over week (21 Oct 2026, the owner's decision): data/readings/<sunday>.json, one summary of the four Sunday
// controls per weekend (weekReading below; readings.js writes it after the Sunday plan, tracked in git). A control's
// cell carries its own Sunday, the number of weeks in a row it has read the same, and - only when last week read
// differently - what it read then: the page colours a change, never a state that holds every week.
//
// The per-game routes take ?scope=all (the default, every row) or ?scope=top (the users' top five only, from
// ranking.js usersTop: the first five ranked rows that are in the game's newest day file - the five rows /lottery
// shows a user; for a market game the five of its History game view).
//
// The figures are the ones the old History cards print (server.js generatePerformanceSummary,
// generateMultiPickSummary, generateCombinationSummary, generateLagAnalysis, generateRandomnessWatch and the
// feature-control card), built by pure functions below so the rules are testable (test/admin.test.js), with what
// the admin plan lists as broken fixed at the source: a keno profit bound comes with the raw profit per bet and
// what the bound is measured against (a fair ticket), the Q0 band is compared hits with hits in every game, a
// combination's p value travels with its value. Nothing here colours a number: each figure carries its window and
// unit, and the page decides how to show it. No raw record is ever sent.
//
// Every source file is read once per change (cache keyed by modification time and size): modelPerformance.json is
// about 1.5 MB, the control records are many small files.
'use strict';

const fs = require('fs');
const path = require('path');
const brussels = require('./brussels');
const controls = require('./controls');
const gates = require('./gates');
const markets = require('./markets');
const rankings = require('./ranking');

const LOTTERY_GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];
const MARKET_GAMES = ['crypto', 'shares', 'cryptoweek', 'sharesweek'];
const GAMES = LOTTERY_GAMES.concat(MARKET_GAMES);
const GROUPS = [{ name: 'Lottery', games: LOTTERY_GAMES }, { name: 'Markets', games: MARKET_GAMES }];
const groupOf = (game) => (MARKET_GAMES.includes(game) ? 'Markets' : 'Lottery');
const isMarketGame = (game) => MARKET_GAMES.includes(game);
// What one scored row - one draw of the game - is: every count of draws, table rows, windows and histories in a
// game's JSON is in this unit (the week games' "draws" are weeks, shares' are trading days).
const UNIT = { lotto: 'draws', euromillions: 'draws', eurodreams: 'draws', vikinglotto: 'draws', jokerplus: 'days', pick3: 'days',
  keno: 'days', crypto: 'days', shares: 'trading days', cryptoweek: 'weeks', sharesweek: 'weeks' };
// RandomnessDiscrimination's power_rule: a record is underpowered "below about 250 held-out windows"
const FEW_WINDOWS = 250;
// the feature-control variants by their short code, as on the mockup's dots
const VARIANT_CODE = { logistic: 'lr', gradient_boosting: 'gb', quantum_kernel: 'qk', quantum_vqc: 'qv', classical_svm: 'svm' };
// the Randomness watch tripwires (server.js generateRandomnessWatch): deliberately loose, an alarm and not a verdict
const ENTROPY_LOW = 0.95;
const KL_HIGH = 0.1;
const AE_WINDOW = 30;   // Predictor.py: the autoencoder's minimum z over the last 30 scored draws
const AE_ALERT_Z = -3;
// the phase-shift card (Helpers.py): draws +1 .. +30 per run, the last 60 runs kept, a lag "tracked" from 3 runs
// with at least half of them agreeing; the trail shows the newest 8 steps
const LAG_MAX = 30;
const LAG_RUNS_KEPT = 60;
const TRAIL_SHOWN = 8;
const SCOPES = ['all', 'top'];
// the day the markets went live (crypto 1 Oct 2026, shares 5 Oct; every stored day before is back-built): a since
// record from on or before it covers every live day
const MARKETS_STARTED = '2026-10-01';

// --- small helpers -----------------------------------------------------------
const num = (x) => { const v = Number(x); return x === null || x === undefined || x === '' || !Number.isFinite(v) ? null : v; };
const round = (digits) => (x) => (num(x) === null ? null : Math.round(num(x) * 10 ** digits) / 10 ** digits);
const r2 = round(2);
const r3 = round(3);
const r4 = round(4);
// Instants as ISO UTC with a Z. The records mix forms: the Q0 and Q2 generated_at and modelPerformance's
// generatedAt are UTC written without a zone, the feature-control and market-rows times carry +00:00.
function isoUtc(value) {
  if (value === null || value === undefined || value === '') return null;
  const text = String(value);
  const date = new Date(/[zZ]$|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isNaN(date.getTime()) ? null : date.toISOString().replace(/\.\d{3}Z$/, 'Z');
}
const latest = (values) => values.filter(Boolean).sort().pop() || null;
const earliest = (values) => values.filter(Boolean).sort()[0] || null;
// an array field of a record, or [] when the record has something else there
const list = (v) => (Array.isArray(v) ? v : []);
const range = (values) => {
  const v = values.filter((x) => x !== null && x !== undefined);
  return v.length ? { min: Math.min(...v), max: Math.max(...v) } : null;
};
// A message for the page without the server's absolute paths: each one cut to its file name (services.js writes
// "cannot open /root/.../council.log: ...").
const noPaths = (text) => (text === null || text === undefined ? null : String(text).replace(/(?:\/[\w.@+-]+)+\/([\w.@+-]+)/g, '$1'));

// --- reading, once per change --------------------------------------------------
const cache = new Map();   // key -> {sig, value}
function stampOf(file) {
  try { const s = fs.statSync(file); return `${file}@${s.mtimeMs}:${s.size}`; } catch (e) { return `${file}@none`; }
}
// The value of build(), rebuilt only when one of the files it reads changed (or appeared, or went).
function memo(key, files, build) {
  const sig = files.map(stampOf).join('|');
  const hit = cache.get(key);
  if (hit && hit.sig === sig) return hit.value;
  const value = build();
  cache.set(key, { sig, value });
  return value;
}

// The folders read: data/database (modelPerformance.json), data/controls (the Sunday records), data/markets (the
// regime logs), data/readings (the weekly readings), each overridable; root is the checkout (bestParams_<game>.json
// and its git history for the tuning gates, config/scheduler.json for the runs, log/ for their log slices).
function resolveDirs({ dataDir, root, databaseDir, controlsDir, marketsDir, readingsDir, schedulerFile, logDir } = {}) {
  const top = root || __dirname;
  const data = dataDir || path.join(top, 'data');
  return {
    databaseDir: databaseDir || path.join(data, 'database'),
    controlsDir: controlsDir || path.join(data, 'controls'),
    marketsDir: marketsDir || path.join(data, 'markets'),
    readingsDir: readingsDir || path.join(data, 'readings'),
    schedulerFile: schedulerFile || path.join(top, 'config', 'scheduler.json'),
    logDir: logDir || path.join(top, 'log'),
    root: top,
  };
}

const isFile = (file) => { try { return fs.statSync(file).isFile(); } catch (e) { return false; } };
// One game's control records, each read through controls.js / markets.js and kept until its file changes.
// unreadable: the parts whose file is there but did not load (cut short, not JSON, not this game's record) -
// 'q2', 'fc', 'rows', and 'q0:<mode>' per control - so the page says so instead of "no record yet". The writers
// (NullControls.py, RandomnessDiscrimination.py, IrrelevantFeatureControl.py, MarketRows.py) write in place, so a
// Sunday reader can catch a file half written.
function readControls(dirs, game) {
  const cd = dirs.controlsDir;
  const q0Files = controls.MODES.map((mode) => path.join(cd, 'results', `${game}-${mode}.json`));
  const q2Folder = path.join(cd, 'discrimination');
  let q2Files = [];
  try { q2Files = fs.readdirSync(q2Folder).filter((f) => f.startsWith(`${game}-w`) && f.endsWith('.json')).sort().map((f) => path.join(q2Folder, f)); } catch (e) { q2Files = []; }
  const fcFile = path.join(cd, 'features', `${game}.json`);
  const rowsFile = path.join(cd, 'markets', `${game}-rows.json`);
  const q0 = memo(`q0:${cd}:${game}`, q0Files, () => controls.loadNullControls(cd, game));
  const q2 = memo(`q2:${cd}:${game}`, [q2Folder, ...q2Files], () => controls.loadDiscrimination(cd, game));
  const fc = memo(`fc:${cd}:${game}`, [fcFile], () => controls.loadFeatureControl(cd, game));
  const rows = isMarketGame(game) ? memo(`rows:${cd}:${game}`, [rowsFile], () => markets.loadRows(cd, game)) : null;
  const unreadable = [
    ...controls.MODES.filter((mode, i) => !q0[mode] && isFile(q0Files[i])).map((mode) => `q0:${mode}`),
    ...(!q2 && q2Files.length ? ['q2'] : []),
    ...(!fc && isFile(fcFile) ? ['fc'] : []),
    ...(isMarketGame(game) && !rows && isFile(rowsFile) ? ['rows'] : []),
  ];
  return { q0, q2, fc, rows, unreadable };
}
// The parts of a game's errors that come from files that did not load: a part with nothing to show (q2, fc,
// rows), and Q0 when no control gave a band (a band from one control says so itself: band.missing).
function unreadableErrors(unreadable, view) {
  const parts = unreadable || [];
  const out = ['q2', 'fc', 'rows'].filter((part) => parts.includes(part) && !view[part]);
  if (!view.q0Band && parts.some((u) => u.startsWith('q0:'))) out.unshift('q0');
  return out;
}
// The controls a band is not built from, each with whether its file is there but did not load: a band from one
// control (the other missing, or caught half written) is not the higher of the two the rule names.
const bandMissing = (band, unreadable) => controls.MODES.filter((mode) => !band.modes[mode])
  .map((mode) => ({ mode, unreadable: (unreadable || []).includes(`q0:${mode}`) }));
function readRegimes(dirs, game) {
  if (!isMarketGame(game)) return null;
  return memo(`regimes:${dirs.marketsDir}:${game}`, [path.join(dirs.marketsDir, `${game}-regimes.json`)], () => markets.loadRegimes(dirs.marketsDir, game));
}
const readReport = (dirs) => rankings.loadReport(dirs.databaseDir);
// Market games: the row the users' page (/app/crypto, /app/shares, api.js) follows - the daily plan for crypto and
// shares, the week card for cryptoweek and sharesweek, by best book (markets.js pickPlan / pickWeek) - which is not
// the History ranking's top five. null for a lottery game or a market without its record.
function followsOf(game, record) {
  if (!isMarketGame(game) || !record) return null;
  const market = game.replace(/week$/, '');
  const week = game !== market;
  const pick = (week ? markets.pickWeek : markets.pickPlan)(markets.describeMarket(record), market);
  return pick.model ? { row: pick.model.name, page: market === 'crypto' ? 'Crypto' : 'Shares', book: week ? 'week' : 'daily', byBook: pick.byBook, rule: pick.ruleName } : null;
}
function readFollows(dirs, game) {
  if (!isMarketGame(game)) return null;
  return memo(`follows:${dirs.marketsDir}:${game}`, [path.join(dirs.marketsDir, `${game}.json`)], () => followsOf(game, markets.loadMarket(dirs.marketsDir, game)));
}
const readAll = (dirs) => Object.fromEntries(GAMES.map((game) => [game, readControls(dirs, game)]));

// --- the ranking a game's rows are read against ---------------------------------
const gameInfo = (report, game) => (report && report.games && report.games[game] && Array.isArray(report.games[game].models) ? report.games[game] : null);
// today: the names in the game's newest day file (ranking.namesToday), or null - then the first five of the ranking
const topOf = (report, game, today = null) => rankings.usersTop(report, game, today);
const isMoney = (metric) => metric === 'profit_bound' || metric === 'profit_per_bet';
// What a game's ranking value measures, in words: its unit, and for a profit bound its cap, its bound and what it
// is measured against.
function measureOf(game, metric) {
  if (metric === 'profit_bound') {
    return { unit: 'EUR per bet', text: rankings.METRIC_TEXT.profit_bound, capEur: 20, bound: 'mean minus 1 SE over the draws',
      against: game === 'keno' ? 'a fair ticket of its size (0 = a fair ticket; not money won)' : 'the payout table (money)' };
  }
  if (metric === 'profit_per_bet') return { unit: 'EUR per bet', text: rankings.METRIC_TEXT.profit_per_bet, against: 'the payout table (money)' };
  return { unit: 'hits per draw', text: rankings.METRIC_TEXT.avg_hits };
}
// Market games: a row's live draws are the days whose ticket was out before the close, over its whole record - the
// report's live_draws (Helpers.py, from 21 Oct 2026), so draws minus live = back-built or late days. A report written
// before that has only the since record, which counts the same days but from the freeze date on: it stands in only
// while the freeze is on or before the day the markets went live (27 Sep today); a later freeze would turn earlier
// live days into back-built ones, so then the split is not shown.
function liveOf(game, info) {
  if (!isMarketGame(game) || !info) return () => undefined;
  const byName = {};
  if (info.models.some((m) => m && num(m.live_draws) !== null)) {
    info.models.forEach((m) => { if (m) byName[m.name] = num(m.live_draws) || 0; });
    return (name) => (byName[name] === undefined ? 0 : byName[name]);
  }
  if (!info.since || !Array.isArray(info.since.models) || !(String(info.since.date) <= MARKETS_STARTED)) return () => undefined;
  info.since.models.forEach((m) => { if (m) byName[m.name] = num(m.draws) || 0; });
  return (name) => (byName[name] === undefined ? 0 : byName[name]);
}

// --- Q0: the null band ---------------------------------------------------------
// Every game is compared hits with hits: the band is hits per draw on fair histories, so a payout game's rows are
// read on their average hits too (never a euro figure next to a hits band), and stay out of the headline count
// (inTile false) because their ranking reads a profit bound.
// unreadable: readControls' list, so a band from one control names the other and why it is not there
function q0Of(game, reports, info, unreadable = []) {
  const band = controls.nullBand(reports || {});
  const mode = band ? band.modes[band.from] : null;
  const models = info ? info.models : [];
  const minDraws = info ? num(info.minDrawsForRanking) || 0 : 0;
  const ranked = models.filter((m) => num(m.draws) !== null && m.draws >= minDraws);
  const payout = info ? info.metric !== 'avg_hits' : false;
  const live = liveOf(game, info);
  const perRow = band && reports[band.from] && reports[band.from].per_row_mean ? reports[band.from].per_row_mean : {};
  const describe = (m) => {
    const out = { row: m.name, hits: r3(m.avg_hits), draws: m.draws };
    const l = live(m.name);
    if (l !== undefined) { out.live = l; out.notLive = m.draws - l; }
    // the row's own level in the control the band comes from: one record for both numbers
    if (typeof perRow[m.name] === 'number') out.own = r3(perRow[m.name]);
    return out;
  };
  const byHits = ranked.filter((m) => num(m.avg_hits) !== null).sort((a, b) => b.avg_hits - a.avg_hits);
  const above = band ? byHits.filter((m) => m.avg_hits > band.ceiling) : [];
  const lengths = above.map((m) => m.draws);
  const lives = isMarketGame(game) ? above.map((m) => live(m.name) || 0) : [];
  return {
    ranking: payout ? 'profit' : 'hits',
    // at: the record the band comes from; modes: every control record's own time (a band half rewritten on a Sunday
    // mixes this week's and last week's)
    band: band ? { ceiling: r3(band.ceiling), from: band.from, rows: mode.rows, seeds: mode.seeds, history: mode.days, metric: band.metric,
      at: isoUtc(mode.generatedAt), modes: Object.keys(band.modes).map((m) => ({ mode: m, at: isoUtc(band.modes[m].generatedAt) })),
      missing: bandMissing(band, unreadable) } : null,
    rankedRows: ranked.length, allRows: models.length, minDraws,
    best: byHits.length ? { ...describe(byHits[0]), clears: band ? byHits[0].avg_hits > band.ceiling : null } : null,
    leader: payout && models.length ? { row: models[0].name, draws: models[0].draws } : null,
    above: {
      n: above.length,
      longest: lengths.length ? Math.max(...lengths) : null,
      longestLive: lives.length ? Math.max(...lives) : null,
      sameLength: above.length ? new Set(lengths).size === 1 : null,
      // the live counts on their own: rows of one record length can differ in how many of their days were live
      sameLive: lives.length ? new Set(lives).size === 1 : null,
      inTile: !payout,
      rows: above.map(describe),
    },
  };
}

// --- Q2: the randomness discrimination -----------------------------------------
// The draws a record actually used: RandomnessDiscrimination cuts one window per newest draw of each class (size
// windows from size + window - 1 draws) and splits both classes' windows into train and test, so train + test is
// twice the windows per class. drawsCap is only the --draws it asked for (at most the newest that many).
function q2DrawsUsed(record, window) {
  const shape = (record && record.shape) || {};
  const train = num(shape.windows_train);
  const test = num(shape.windows_test);
  return train === null || test === null || num(window) === null ? null : Math.round((train + test) / 2) + window - 1;
}
function q2Of(record) {
  if (!record) return null;
  const d = controls.describeDiscrimination(record);
  return {
    status: d.tooFew ? 'too few draws' : (d.evidence ? 'above' : (d.aboveBand ? 'above band, not shuffled' : 'within')),
    real: r3(d.real), threshold: r3(d.threshold), shuffled: r3(d.shuffled),
    seeds: d.repetitions, windowsTest: d.windowsTest, fewWindows: d.windowsTest !== null && d.windowsTest < FEW_WINDOWS,
    window: d.window, drawsCap: d.draws, drawsUsed: q2DrawsUsed(record, d.window), quantum: d.quantum, at: isoUtc(d.generatedAt),
  };
}

// --- the meta-learner feature control ----------------------------------------------
function fcOf(record) {
  if (!record) return null;
  const d = controls.describeFeatureControl(record);
  const columns = d.variants.map((v) => v.realColumns).filter((x) => x);
  const count = (verdict) => d.variants.filter((v) => v.verdict === verdict).length;
  const maxColumns = columns.length ? Math.max(...columns) : null;
  return {
    fits: count('fits noise'), inconclusive: count('inconclusive'), ignored: count('noise ignored'), failed: count('failed'),
    // the chance share: what noise columns get from a model that cannot tell them from the base models
    columns: maxColumns, chanceShare: maxColumns !== null && d.noiseColumns !== null ? r3(d.noiseColumns / (maxColumns + d.noiseColumns)) : null,
    // table.days and lockbox_days_withheld count table rows: draws, in the game's unit; the lockbox rows are withheld from the table
    tableRows: d.tableDays, lockboxRows: d.lockboxDays, behind: d.behind, positional: d.positional,
    repeats: d.repeats, noiseColumns: d.noiseColumns,
    columnsAbove: d.variants.reduce((s, v) => s + v.above.length, 0), at: isoUtc(d.generatedAt),
    variants: d.variants.map((v) => {
      const out = { key: VARIANT_CODE[v.key] || v.key, verdict: v.verdict, share: r4(v.noiseShare), auc: [r2(v.heldoutWithout), r2(v.heldoutWith)] };
      if (v.above.length) out.above = v.above;
      return out;
    }),
  };
}

// --- market rows: the market rows against GARCH and uniform ---------------------------
function rowsOf(record) {
  if (!record) return null;
  const d = markets.describeRows(record);
  const scored = d.rows.filter((r) => r.logScore !== null && !r.isReference);
  const ref = d.referenceRow;
  return {
    window: d.days, requested: d.requested, firstDay: d.firstDay, lastDay: d.lastDay, lockboxWithheld: d.lockboxWithheld,
    reference: d.reference, rowsTotal: d.rows.length, rowsScored: scored.length, baselines: d.rows.filter((r) => r.logScore === null).length,
    betterThanUniform: scored.filter((r) => r.vsUniform && r.vsUniform.verdict === 'better').map((r) => r.name),
    betterThanReference: scored.filter((r) => r.vsReference && r.vsReference.verdict === 'better').map((r) => r.name),
    noDifferenceVsReference: scored.filter((r) => r.vsReference && r.vsReference.verdict === 'no difference').length,
    referenceVsUniform: ref && ref.vsUniform ? { mean: r4(ref.vsUniform.mean), lo: r4(ref.vsUniform.lo), hi: r4(ref.vsUniform.hi), verdict: ref.vsUniform.verdict } : null,
    errors: Object.keys(d.errors).length, hmmFeatures: d.hmmFeatures, at: isoUtc(d.generatedAt),
  };
}

// --- GET /api/admin/controls -----------------------------------------------------
// inputs: {game: {q0, q2, fc, rows}} as readControls returns them; report: the parsed modelPerformance.json.
// One part of a game's view, or null with the part named in errors when its record cannot be read as built: a
// record that parses but has a broken shape costs its own cell, never the whole page (controls.js: a page must
// never fail because of one record).
function guarded(errors, part, build) {
  try { return build(); } catch (e) {
    console.error(`[${new Date().toISOString()}] admin.js: ${part}:`, e.message);
    errors.push(part);
    return null;
  }
}
// a game's Q0 with no band and no rows, for a game whose Q0 could not be built
const NO_Q0 = { ranking: 'hits', band: null, rankedRows: 0, allRows: 0, minDraws: 0, best: null, leader: null,
  above: { n: 0, longest: null, longestLive: null, sameLength: null, sameLive: null, inTile: false, rows: [] } };
function controlsOverview(inputs, report, now = new Date()) {
  const games = {};
  GAMES.forEach((game) => {
    const input = inputs[game] || {};
    const errors = [];
    games[game] = {
      game, unit: UNIT[game],
      q0: guarded(errors, 'q0', () => q0Of(game, input.q0, gameInfo(report, game), input.unreadable)) || NO_Q0,
      q2: guarded(errors, 'q2', () => q2Of(input.q2)),
      fc: guarded(errors, 'fc', () => fcOf(input.fc)),
      rows: isMarketGame(game) ? guarded(errors, 'rows', () => rowsOf(input.rows)) : null,
      errors,
    };
    const g = games[game];
    unreadableErrors(input.unreadable, { q0Band: g.q0.band, q2: g.q2, fc: g.fc, rows: g.rows }).forEach((part) => { if (!errors.includes(part)) errors.push(part); });
  });
  const all = GAMES.map((g) => games[g]);
  const q0s = all.filter((g) => g.q0.band);
  const hitRanked = q0s.filter((g) => g.q0.ranking === 'hits');
  const q2s = all.filter((g) => g.q2);
  const inGroup = (name) => (g) => groupOf(g.game) === name;
  const fcs = all.filter((g) => g.fc);
  const rows = all.filter((g) => g.rows);
  const q0Times = GAMES.flatMap((g) => Object.values((inputs[g] || {}).q0 || {}).map((r) => isoUtc(r && r.generated_at)));
  const seedsOf = (list, pick) => [...new Set(list.map(pick).filter((x) => x !== null && x !== undefined))].sort((a, b) => a - b);
  const tiles = {
    q0: {
      // the headline rule (plan decision 2 still open): every ranked row of the hit-ranked games against its game's band
      rowsAbove: hitRanked.reduce((s, g) => s + g.q0.above.n, 0), games: hitRanked.filter((g) => g.q0.above.n).length, hitRankedGames: hitRanked.length,
      longest: range(hitRanked.flatMap((g) => g.q0.above.rows.map((r) => r.draws))),
      longestLiveMarkets: range(hitRanked.filter((g) => isMarketGame(g.game)).flatMap((g) => g.q0.above.rows.map((r) => r.live || 0))),
      withOwnLevel: hitRanked.flatMap((g) => g.q0.above.rows).filter((r) => r.own !== undefined).length,
      alternatives: {
        allGamesOnHits: { rowsAbove: q0s.reduce((s, g) => s + g.q0.above.n, 0), games: q0s.filter((g) => g.q0.above.n).length },
        bestRowOnly: { games: hitRanked.filter((g) => g.q0.best && g.q0.best.clears).map((g) => g.game) },
      },
      // the games compared on no band (no record, or one that could not be read): named, never counted as "none above"
      noBand: all.filter((g) => !g.q0.band).map((g) => g.game),
      // the games whose band comes from one control only (the other missing or unreadable): counted, and named
      oneControl: all.filter((g) => g.q0.band && g.q0.band.missing.length).map((g) => ({ game: g.game, from: g.q0.band.from, missing: g.q0.band.missing })),
      // the run: every Q0 record, both controls (a band's own time is the record it comes from)
      from: earliest(q0Times), at: latest(q0Times),
    },
    q2: {
      above: q2s.filter((g) => g.q2.status === 'above').length, of: q2s.length,
      aboveGames: q2s.filter((g) => g.q2.status === 'above').map((g) => g.game),
      tooFewGames: q2s.filter((g) => g.q2.status === 'too few draws').map((g) => g.game),
      aboveFewWindows: q2s.filter((g) => g.q2.status === 'above' && g.q2.fewWindows).map((g) => ({ game: g.game, windowsTest: g.q2.windowsTest })),
      seeds: Object.fromEntries(GROUPS.map((grp) => [grp.name, seedsOf(q2s.filter(inGroup(grp.name)), (g) => g.q2.seeds)])),
      // the games out of the denominator, named: no record yet, or one that could not be read
      missing: all.filter((g) => !g.q2 && !g.errors.includes('q2')).map((g) => g.game), unreadable: all.filter((g) => !g.q2 && g.errors.includes('q2')).map((g) => g.game),
      from: earliest(q2s.map((g) => g.q2.at)), at: latest(q2s.map((g) => g.q2.at)),
    },
    fc: {
      fits: fcs.reduce((s, g) => s + g.fc.fits, 0), inconclusive: fcs.reduce((s, g) => s + g.fc.inconclusive, 0),
      ignored: fcs.reduce((s, g) => s + g.fc.ignored, 0), failed: fcs.reduce((s, g) => s + g.fc.failed, 0),
      variants: fcs.reduce((s, g) => s + g.fc.variants.length, 0), games: fcs.length,
      missing: all.filter((g) => !g.fc && !g.errors.includes('fc')).map((g) => g.game), unreadable: all.filter((g) => !g.fc && g.errors.includes('fc')).map((g) => g.game),
      from: earliest(fcs.map((g) => g.fc.at)), at: latest(fcs.map((g) => g.fc.at)),
    },
    rows: {
      beatReference: rows.reduce((s, g) => s + g.rows.betterThanReference.length, 0), games: rows.length,
      referenceAboveUniform: rows.filter((g) => g.rows.referenceVsUniform && g.rows.referenceVsUniform.verdict === 'better').length,
      windows: rows.map((g) => ({ game: g.game, window: g.rows.window, unit: g.unit })),
      missing: MARKET_GAMES.filter((g) => !games[g].rows),
      unreadable: MARKET_GAMES.filter((g) => !games[g].rows && games[g].errors.includes('rows')),
      from: earliest(rows.map((g) => g.rows.at)), at: latest(rows.map((g) => g.rows.at)),
    },
  };
  return {
    builtAt: isoUtc(now.toISOString()),
    rankingAt: isoUtc(report && report.generatedAt),
    meta: {
      q0: { rule: 'best of the tracked rows on fair histories + 2 sd across seeds, the higher of synthetic and shuffled', metric: 'hits per draw',
        rows: range(q0s.map((g) => g.q0.band.rows)), history: range(q0s.map((g) => g.q0.band.history)),
        seeds: Object.fromEntries(GROUPS.map((grp) => [grp.name, seedsOf(q0s.filter(inGroup(grp.name)), (g) => g.q0.band.seeds)])),
        ranked: 'rows with at least minDraws scored draws; payout games compared on hits and left out of the headline count',
        live: 'market games: live = days whose ticket was out before the close, over the whole record (live_draws); notLive = back-built or late days' },
      q2: { metric: 'best AUC of the classifier suite, mean over seeds', axis: [0.5, 1.0], fewWindows: FEW_WINDOWS,
        rule: 'above = real above the null mean + 2 sd and above the shuffled control; too few draws when the threshold is 0.8 or more' },
      fc: { share: "the noise columns' share of training importance", chanceShare: 'noise columns / (base-model columns + noise columns)',
        auc: '[held-out AUC without noise, with noise]', variants: Object.fromEntries(Object.entries(VARIANT_CODE).map(([k, c]) => [c, controls.VARIANT_LABELS[k] || k])) },
      rows: { score: 'mean log-score difference per draw against the reference (GARCH) and uniform, 95% interval', pnl: 'left out: price units' },
    },
    tiles,
    groups: GROUPS.map((grp) => ({ name: grp.name, games: grp.games.map((g) => games[g]) })),
  };
}

// --- GET /api/admin/controls/:game -------------------------------------------------
function controlsDrawer(game, input, report, regimesRecord = null, now = new Date()) {
  const info = gameInfo(report, game);
  const reports = (input && input.q0) || {};
  const unreadable = (input && input.unreadable) || [];
  const errors = [];
  const q0 = guarded(errors, 'q0', () => q0Of(game, reports, info, unreadable)) || NO_Q0;
  const modes = guarded(errors, 'q0', () => buildModes(reports)) || {};
  const q2 = guarded(errors, 'q2', () => q2Full(input && input.q2));
  const fc = guarded(errors, 'fc', () => fcFull(input && input.fc));
  const rows = isMarketGame(game) ? guarded(errors, 'rows', () => rowsFull(input && input.rows)) : null;
  const regimes = isMarketGame(game) ? guarded(errors, 'regimes', () => markets.describeRegimes(regimesRecord, null).map((r) => ({
    row: r.row, date: r.date, label: r.label, regimes: r.regimes, probability: r3(r.probability), volatilityRank: r.volatilityRank,
    template: r.template, features: r.features, history: r.history }))) : null;
  unreadableErrors(unreadable, { q0Band: q0.band, q2, fc, rows }).forEach((part) => errors.push(part));
  return {
    game, group: groupOf(game), unit: UNIT[game], builtAt: isoUtc(now.toISOString()), rankingAt: isoUtc(report && report.generatedAt),
    q0: { ...q0, modes }, q2, fc, rows, regimes, errors: [...new Set(errors)],
  };
}
// the drawer's parts, each built on its own (controlsDrawer guards them one by one)
function buildModes(reports) {
  const modes = {};
  Object.keys(reports).forEach((mode) => {
    const r = reports[mode];
    modes[mode] = {
      best: r4(r.best_row_mean), sd: r4(r.best_row_sd_across_seeds), ceiling: r3(Number(r.best_row_mean) + 2 * (Number(r.best_row_sd_across_seeds) || 0)),
      rowMean: r4(r.row_mean), random: r4(r.random_baseline), selectionGain: r4(r.selection_gain), rows: r.rows_compared,
      seeds: Array.isArray(r.seeds) ? r.seeds.length : null, history: r.days, metric: r.metric || 'hits per draw',
      winners: Array.isArray(r.winner_per_seed) ? r.winner_per_seed : [], at: isoUtc(r.generated_at),
      perRow: Object.fromEntries(Object.entries(r.per_row_mean || {}).map(([k, v]) => [k, r4(v)])),
    };
  });
  return modes;
}
function q2Full(rec) {
  let q2 = null;
  if (rec) {
    const v = rec.verdict || {};
    const shape = rec.shape || {};
    q2 = { ...q2Of(rec), label: controls.describeDiscrimination(rec).label, metric: v.metric || null,
      windowsTrain: num(shape.windows_train), features: num(shape.features),
      realMean: r4(v.real_max_of_suite_mean), realSd: r4(v.real_max_of_suite_sd), nullMean: r4(v.null_max_of_suite_mean), nullSd: r4(v.null_max_of_suite_sd),
      shuffledMean: r4(v.shuffled_max_of_suite_mean), shuffledSd: r4(v.shuffled_max_of_suite_sd), thresholdExact: r4(v.null_threshold_2sd),
      perSeed: { real: list(v.per_seed_real).map(r4), null: list(v.per_seed_null).map(r4), shuffled: list(v.per_seed_shuffled).map(r4) },
      aboveBand: Boolean(v.above_null_band), survivesShuffled: Boolean(v.survives_marginals_control) };
  }
  return q2;
}
function fcFull(record) {
  let fc = null;
  if (record) {
    const d = controls.describeFeatureControl(record);
    // the verdict rule is the record's, the same for every variant: sent once, and on a variant only where it differs
    const rule = (d.variants.find((x) => x.rule) || {}).rule || null;
    fc = { ...fcOf(record), rule, variants: d.variants.map((x) => {
      const out = { key: VARIANT_CODE[x.key] || x.key, label: x.label, verdict: x.verdict, error: x.error, columns: x.realColumns,
        noiseShare: r4(x.noiseShare), chanceShare: x.realColumns && d.noiseColumns !== null ? r3(d.noiseColumns / (x.realColumns + d.noiseColumns)) : null,
        ratio: r3(x.noiseRatio), t: r2(x.noiseT), band: r4(x.band), above: x.above,
        aucWithout: r4(x.heldoutWithout), aucWith: r4(x.heldoutWith), cost: r4(x.cost), seconds: r2(x.seconds) };
      if (x.rule && x.rule !== rule) out.rule = x.rule;
      return out;
    }) };
  }
  return fc;
}
function rowsFull(record) {
  if (!record) return null;
  const d = markets.describeRows(record);
  const interval = (x) => (x ? { mean: r4(x.mean), lo: r4(x.lo), hi: r4(x.hi), verdict: x.verdict } : null);
  return { ...rowsOf(record), uniform: r4(d.uniform), floor: d.floor,
    table: d.rows.map((r) => ({ row: r.name, kind: r.kind, reference: r.isReference, days: r.days, logScore: r4(r.logScore), se: r4(r.se),
      vsReference: interval(r.vsReference), vsUniform: interval(r.vsUniform), exact: r3(r.exact), adjacent: r3(r.adjacent), direction: r3(r.direction) })) };
}

// --- GET /api/admin/models -----------------------------------------------------------
// The Best model per game card: the best row, its value with unit, its scored draws, the best since the freeze, the
// Q0 band read on hits, and the five rows the users see (for a market game also the row its users' page follows).
function sinceBestOf(report, info) {
  const declared = report && report.since ? report.since : null;
  const since = info && info.since ? info.since : null;
  if (since) {
    const best = list(since.models).find((m) => m && m.name === since.bestModel) || null;
    return { date: since.date, label: since.label, minDraws: since.minDrawsForRanking, rows: list(since.models).length,
      best: best ? { row: best.name, value: num(best[info.metric]), draws: best.draws, tooFew: best.draws < since.minDrawsForRanking } : null };
  }
  return declared ? { date: declared.date, label: declared.label, none: true } : null;
}
// The Q0 band a game's ranking is read against: what the page shows, and whether a row's average hits are within
// it (on the unrounded ceiling, as the old card). It greys rows only in the hit-ranked games; for the payout games it
// is read on hits, information only.
function bandForModels(reports, metric, unreadable = []) {
  const band = controls.nullBand(reports || {});
  if (!band) return { band: null, modes: null, within: () => false };
  const mode = band.modes[band.from];
  const greys = metric === 'avg_hits';
  // both controls as the old card's band title gave them: the best row's mean +- sd across the fair histories
  const modes = Object.fromEntries(Object.entries(band.modes).map(([name, m]) => [name, { best: r3(m.best), sd: r3(m.sd), ceiling: r3(m.ceiling),
    rows: m.rows, seeds: m.seeds, history: m.days, random: r3(m.random), at: isoUtc(m.generatedAt) }]));
  return { band: { ceiling: r3(band.ceiling), from: band.from, rows: mode.rows, history: mode.days, metric: band.metric, at: isoUtc(mode.generatedAt), greys,
    missing: bandMissing(band, unreadable) },
    modes, within: (hits) => greys && controls.withinBand(hits, band) };
}
// users: {today: {game: the names in its newest day file} (ranking.namesToday), so the top five are the users' five;
// follows: {market game: the row its users' page follows} (followsOf)}
function modelsSummary(report, inputs = {}, { today = {}, follows = {} } = {}, now = new Date()) {
  const games = GAMES.filter((g) => gameInfo(report, g)).map((game) => {
    const info = gameInfo(report, game);
    const best = info.models[0];
    const minDraws = num(info.minDrawsForRanking) || 0;
    const { band, within } = bandForModels((inputs[game] || {}).q0, info.metric, (inputs[game] || {}).unreadable);
    const live = liveOf(game, info);
    return {
      game, group: groupOf(game), unit: UNIT[game], metric: info.metric, measure: measureOf(game, info.metric), minDraws,
      rows: info.models.length, rankedRows: info.models.filter((m) => m.draws >= minDraws).length,
      best: { row: best.name, value: num(best[info.metric]), draws: best.draws, young: best.draws < minDraws,
        // a payout game's raw profit per bet next to its bound: keno's bound is against a fair ticket, not money won
        profitPerBet: isMoney(info.metric) ? num(best.profit_per_bet) : null, bets: isMoney(info.metric) ? num(best.bets) : null,
        avgHits: num(best.avg_hits), inBand: within(best.avg_hits), live: live(best.name) },
      band,
      since: sinceBestOf(report, info),
      top: topOf(report, game, today[game]),
      // a market game's users' page follows its best book, not the top five
      follows: isMarketGame(game) ? follows[game] || null : undefined,
    };
  });
  return { builtAt: isoUtc(now.toISOString()), generatedAt: isoUtc(report && report.generatedAt), window: 'every scored draw',
    since: report && report.since ? report.since : null, topN: rankings.TOP_MODELS_SHOWN,
    groups: GROUPS.map((grp) => ({ name: grp.name, games: grp.games.filter((g) => games.some((x) => x.game === g)) })), games };
}

// --- GET /api/admin/models/:game ------------------------------------------------------
function multiPickOf(info, keep) {
  const cfg = info.multi_pick;
  if (!cfg) return null;
  const all = info.models.filter((m) => m.multi_pick && m.multi_pick.draws > 0);
  if (!all.length) return null;
  const six = (cfg.chance || {})[String(cfg.draw)] || {};
  const nine = (cfg.chance || {})[String(cfg.draw + cfg.extra)] || {};
  const rows = all.filter((m) => keep(m.name));
  return {
    draw: cfg.draw, extra: cfg.extra, minHits: cfg.min_hits, perExtraChance: r4(cfg.per_extra_chance),
    chance: { [cfg.draw]: num(six.win), [cfg.draw + cfg.extra]: num(nine.win) }, grids: nine.grids, price: nine.price,
    rowsWithExtras: all.length, draws: range(rows.map((m) => m.multi_pick.draws)),
    rows: rows.map((m) => {
      const mp = m.multi_pick;
      return { row: m.name, draws: mp.draws, avgHits6: num(mp.base_avg_hits), avgHits9: num(mp.avg_hits), win6: num(mp.base_win_rate), win9: num(mp.win_rate),
        extraHits: (mp.extra_hit_rates || []).map(num) };
    }),
  };
}
function combinationOf(info) {
  const combo = info.combinations;
  if (!combo) return null;
  const isProfit = combo.metric === 'profit_per_draw';
  const head = { metric: combo.metric, measure: isProfit ? { unit: 'EUR per draw', text: 'net profit per draw of playing every member', capped: false }
    : { unit: 'hits per draw', text: 'best line held per draw (average of the best member\'s hits)' },
    candidateRows: Array.isArray(combo.candidateRows) ? combo.candidateRows.length : 0, evaluated: num(combo.evaluated) };
  if (!combo.best) return { ...head, best: null, note: combo.note || 'no combination could be scored' };
  const line = (r) => ({ members: r.members, how: r.how, size: r.size, value: num(r.value), draws: r.draws,
    profitPerBet: isProfit ? num(r.profit_per_bet) : null, winDayRate: isProfit ? num(r.win_day_rate) : null,
    avgHits: num(r.avg_hits), bestDay: num(r.best_hits) });
  const c = combo.control || null;
  return {
    ...head,
    best: line(combo.best),
    bestSingle: combo.bestSingle ? { row: combo.bestSingle.name, value: num(combo.bestSingle.value), draws: combo.bestSingle.draws } : null,
    // "beats the best single row" only means something where the metric adds up (profit)
    beatsBestSingle: isProfit ? Boolean(combo.beatsBestSingle) : null,
    // the control travels with the value it qualifies
    control: !c ? null : (c.p_value !== undefined
      ? { shuffles: c.shuffles, bestMean: num(c.best_mean), bestP95: num(c.best_p95), pValue: num(c.p_value), consistentWithLuck: num(c.p_value) > 0.05 }
      : { error: c.error || 'not available' }),
    ranking: (combo.ranking || []).map(line),
  };
}
function modelsGame(report, game, scope = 'all', input = {}, { today = null, follows = null } = {}, now = new Date()) {
  const info = gameInfo(report, game);
  if (!info) return null;
  const top = topOf(report, game, today);
  const keep = (name) => scope !== 'top' || top.includes(name);
  const minDraws = num(info.minDrawsForRanking) || 0;
  const { band, modes, within } = bandForModels(input.q0, info.metric, input.unreadable);
  const live = liveOf(game, info);
  const since = info.since || null;
  const sinceByName = {};
  list(since && since.models).forEach((m) => { if (m) sinceByName[m.name] = m; });
  const rows = info.models.map((m, i) => {
    const sm = sinceByName[m.name];
    const out = {
      rank: i + 1, row: m.name, top: top.includes(m.name), value: num(m[info.metric]),
      profitPerBet: info.metric === 'profit_bound' ? num(m.profit_per_bet) : null, bets: info.metric === 'profit_bound' ? num(m.bets) : null,
      avgHits: num(m.avg_hits), bestDay: num(m.best_hits), draws: m.draws, live: live(m.name), young: m.draws < minDraws,
      inBand: within(m.avg_hits),
      since: since ? (sm ? { value: num(sm[info.metric]), draws: sm.draws, young: sm.draws < since.minDrawsForRanking } : null) : undefined,
    };
    if (game === 'keno') {
      out.bySize = m.profit_by_size ? Object.keys(m.profit_by_size).map((k) => ({ size: Number(k), profitPerBet: num(m.profit_by_size[k].profit_per_bet), bets: num(m.profit_by_size[k].bets) })) : null;
    }
    if (out.since === undefined) delete out.since;
    return out;
  }).filter((r) => keep(r.row));
  return {
    game, group: groupOf(game), unit: UNIT[game], builtAt: isoUtc(now.toISOString()), generatedAt: isoUtc(report.generatedAt), window: 'every scored draw',
    metric: info.metric, measure: measureOf(game, info.metric), minDraws, scope, shown: rows.length, of: info.models.length, top,
    follows: isMarketGame(game) ? follows || null : undefined,
    band: band ? { ...band, modes } : null,
    since: since ? { date: since.date, label: since.label, minDraws: since.minDrawsForRanking, bestModel: since.bestModel,
      live: isMarketGame(game) ? 'only days whose ticket was out before the close' : null } : sinceBestOf(report, info),
    rows,
    multiPick: multiPickOf(info, keep),
    // a set result over every established row: the scope does not filter it
    combination: combinationOf(info),
  };
}

// --- GET /api/admin/phase-shift ------------------------------------------------------
// One row of the old card: this run's peak, how often the most frequent peak won across the kept runs, and the
// peak trail run-length encoded (a stable peak reads +30 x5, a drift +26 -> +28 -> +30), the newest TRAIL_SHOWN steps.
function lagRow(name, row) {
  const peak = row.peak || {};
  const runs = num(row.runs) || 0;
  const share = runs ? (num(row.consensus_runs) || 0) / runs : 0;
  // a step is [lag, runs, first run, last run, avg hits, z] - the last two of its newest run (LAG_META.trailFields)
  const segments = [];
  (row.history || []).forEach((r) => {
    const last = segments[segments.length - 1];
    if (last && last[0] === r.lag) { last[1] += 1; last[3] = r.run; last[4] = num(r.avg_hits); last[5] = num(r.z); } else segments.push([r.lag, 1, r.run, r.run, num(r.avg_hits), num(r.z)]);
  });
  // Helpers.py records a run only when the peak's n changed: a scan that added no draw is not a run. Its peak is then
  // the newest recorded run's, dated by that run; the report date is only when it was scanned.
  const newest = (row.history || [])[(row.history || []).length - 1] || null;
  const unchanged = Boolean(newest && newest.run !== peak.run && num(newest.n) === num(peak.n));
  return {
    row: name,
    peak: { lag: peak.lag, avgHits: num(peak.avg_hits), z: num(peak.z), n: num(peak.n), profileMean: num(peak.profile_mean),
      run: (unchanged ? newest.run : peak.run) || null, scanned: peak.run || null, unchanged },
    runs, mostFrequent: { lag: num(row.consensus_lag), runs: num(row.consensus_runs) || 0, share: r3(share) },
    // several runs agreeing on one lag; no control measures what this share or z reach by chance
    tracked: runs >= 3 && share >= 0.5,
    trail: segments.slice(-TRAIL_SHOWN), earlierSteps: Math.max(0, segments.length - TRAIL_SHOWN),
  };
}
const byRanking = (report, game) => {
  const order = (rankings.rankingOf(report, game) || { names: [] }).names;
  return (a, b) => {
    const ia = order.indexOf(a); const ib = order.indexOf(b);
    if (ia !== ib) return (ia < 0 ? Infinity : ia) - (ib < 0 ? Infinity : ib);
    return a.localeCompare(b);
  };
};
const LAG_META = { lags: `+1 .. +${LAG_MAX} (+1 is the draw the prediction was made for)`, runsKept: LAG_RUNS_KEPT, trailShown: TRAIL_SHOWN,
  trailFields: ['lag', 'runs', 'from', 'to', 'avgHits', 'z'],
  tracked: 'at least 3 runs, and the most frequent peak in at least half of them',
  z: "how far the run's peak sticks out of the row's own lag profile", control: 'none: no control measures what z or the tracked share reach by chance' };
function phaseShiftSummary(report, today = {}, now = new Date()) {
  const games = GAMES.filter((g) => gameInfo(report, g) && gameInfo(report, g).lagAnalysis && Object.keys(gameInfo(report, g).lagAnalysis).length).map((game) => {
    const la = gameInfo(report, game).lagAnalysis;
    const top = topOf(report, game, today[game]);
    const rows = Object.keys(la).map((name) => lagRow(name, la[name]));
    const tracked = rows.filter((r) => r.tracked);
    // the lags most rows' most frequent peak agrees on, over every row (the table's Most frequent column), with how
    // many of those rows are tracked; most rows first
    const byLag = {};
    rows.forEach((r) => {
      const k = r.mostFrequent.lag;
      if (k === null) return;
      byLag[k] = byLag[k] || { lag: k, rows: 0, tracked: 0 };
      byLag[k].rows += 1;
      if (r.tracked) byLag[k].tracked += 1;
    });
    return { game, group: groupOf(game), unit: UNIT[game], rows: rows.length, tracked: tracked.length,
      topTracked: rows.filter((r) => top.includes(r.row) && r.tracked).length, newestRun: latest(rows.map((r) => r.peak.run)),
      scanned: latest(rows.map((r) => r.peak.scanned)), runs: range(rows.map((r) => r.runs)),
      commonLags: Object.values(byLag).sort((a, b) => b.rows - a.rows || b.tracked - a.tracked || a.lag - b.lag).slice(0, 3) };
  });
  return { builtAt: isoUtc(now.toISOString()), generatedAt: isoUtc(report && report.generatedAt), meta: LAG_META, games };
}
function phaseShiftGame(report, game, scope = 'all', today = null, now = new Date()) {
  const info = gameInfo(report, game);
  if (!info || !info.lagAnalysis) return null;
  const top = topOf(report, game, today);
  const names = Object.keys(info.lagAnalysis).sort(byRanking(report, game));
  const rows = names.filter((n) => scope !== 'top' || top.includes(n)).map((n) => ({ ...lagRow(n, info.lagAnalysis[n]), top: top.includes(n) }));
  // the top rows without a lag record, named so a short Top 5 table says why
  return { game, group: groupOf(game), unit: UNIT[game], builtAt: isoUtc(now.toISOString()), generatedAt: isoUtc(report.generatedAt), meta: LAG_META,
    scope, shown: rows.length, of: names.length, top, topMissing: top.filter((n) => !names.includes(n)), rows };
}

// --- GET /api/admin/randomness ------------------------------------------------------
// Per game the tripwires over the newest draws, and the weekly Q2 verdict next to them (the controlled test of the
// same question). The status says which tripwire is up, never just "watch".
function randomnessOf(game, info, q2Record) {
  const rw = info.randomnessWatch || null;
  const aw = info.anomalyWatch || null;
  const q2 = q2Of(q2Record);
  const reasons = [];
  if (rw && num(rw.entropy_norm) !== null && rw.entropy_norm < ENTROPY_LOW) reasons.push('entropy');
  if (rw && num(rw.kl_vs_history) !== null && rw.kl_vs_history > KL_HIGH) reasons.push('kl');
  if (aw && aw.alert) reasons.push('ae');
  if (q2 && q2.status === 'above') reasons.push('q2');
  return {
    game, group: groupOf(game), unit: UNIT[game], status: reasons.length ? 'watch' : 'normal', reasons,
    window: rw ? num(rw.window) : null, drawsTotal: rw ? num(rw.draws_total) : null,
    entropy: rw ? num(rw.entropy_norm) : null, klHistory: rw ? num(rw.kl_vs_history) : null, klUniform: rw ? num(rw.kl_vs_uniform) : null,
    ae: aw ? { minZ: num(aw.min_z_recent), latestZ: r3(aw.latest_z), date: aw.date || null, alert: Boolean(aw.alert), window: AE_WINDOW } : null,
    q2: q2 ? (() => {
      const d = controls.describeDiscrimination(q2Record);
      return { ...q2, label: d.label, realSd: r3(d.realSd), nullMean: r3(d.nullMean), nullSd: r3(d.nullSd) };
    })() : null,
    trend: rw && Array.isArray(rw.trend) ? rw.trend.map((t) => ({ end: t.end_date, entropy: num(t.entropy_norm), klHistory: num(t.kl_vs_history), klUniform: num(t.kl_vs_uniform) })) : [],
    modelRows: rw && rw.model_kl_vs_real ? Object.keys(rw.model_kl_vs_real).length : 0,
  };
}
const RANDOMNESS_META = { tripwires: { entropyBelow: ENTROPY_LOW, klHistoryAbove: KL_HIGH, aeBelow: AE_ALERT_Z },
  window: 'the newest scored draws (window); pick3 and Joker+ per digit position, averaged', ae: `the most negative rolling z over the last ${AE_WINDOW} draws`,
  modelKl: 'KL(predicted || real) over the window; no reference value', q2: 'the weekly controlled test (Controls page)', fewWindows: FEW_WINDOWS };
const randomnessGames = (report) => GAMES.filter((g) => { const info = gameInfo(report, g); return info && (info.randomnessWatch || info.anomalyWatch); });
function randomnessSummary(report, q2Records = {}, now = new Date()) {
  return { builtAt: isoUtc(now.toISOString()), generatedAt: isoUtc(report && report.generatedAt), meta: RANDOMNESS_META,
    games: randomnessGames(report).map((g) => randomnessOf(g, gameInfo(report, g), q2Records[g] || null)) };
}
function randomnessGame(report, game, scope = 'all', q2Record = null, today = null, now = new Date()) {
  const info = gameInfo(report, game);
  if (!info || !(info.randomnessWatch || info.anomalyWatch)) return null;
  const top = topOf(report, game, today);
  const kl = (info.randomnessWatch && info.randomnessWatch.model_kl_vs_real) || {};
  const names = Object.keys(kl).sort((a, b) => kl[a] - kl[b]);
  const models = names.filter((n) => scope !== 'top' || top.includes(n)).map((n) => ({ row: n, kl: num(kl[n]), top: top.includes(n) }));
  // the top rows without a KL row over this window, named so a short Top 5 table says why
  return { ...randomnessOf(game, info, q2Record), builtAt: isoUtc(now.toISOString()), generatedAt: isoUtc(report.generatedAt), meta: RANDOMNESS_META,
    scope, shown: models.length, of: names.length, top, topMissing: top.filter((n) => !names.includes(n)), models };
}

// --- the week's reading: one weekend's four Sunday controls ---------------------------------------------------
// What data/readings/<sunday>.json holds (readings.js writes it when the Sunday plan's last job finishes; its CLI
// rebuilds or backfills one): per game the four controls as they read that week, each part with its own record's
// time. Built by the Controls page's own builders (q0Of, q2Of) and readers (controls.js, markets.js), so a reading
// is what the page showed that week, kept: the Q0 rows above the band are read against the ranking on disk when the
// reading is built (rankingAt), never a later one.
const READING_VERSION = 1;
const READING_FILE = /^(\d{4}-\d{2}-\d{2})\.json$/;
const PARTS = ['q0', 'q2', 'fc', 'rows'];
const VERDICT_SHORT = { 'fits noise': 'fit', inconclusive: 'inconcl.', 'noise ignored': 'ignored', failed: 'failed' };
const shortRow = (name) => String(name).replace(/ Model$/, '');
const isoMs = (ms) => (Number.isFinite(ms) ? new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null);
// The Sunday (Brussels) a control record belongs to: the controls run from Sunday morning into Monday, a catch-up
// until Tuesday morning.
const sundayOf = (at) => { const ms = Date.parse(at); return Number.isFinite(ms) ? brussels.lastWeekday(ms, 0) : null; };
const interval = (x) => (x ? { mean: r4(x.mean), lo: r4(x.lo), hi: r4(x.hi), verdict: x.verdict } : null);
// The time a part's record is from. Q0's is its older control record: on a Sunday Q0 rewrites a game's synthetic
// record hours before its shuffled one, and a band built from one new record and one of last week's is not yet this
// week's (a stored reading written before 21 Oct 2026 kept the newer time, so the modes are read, not q0.at).
const partAt = (part, entry) => {
  const x = entry && entry[part];
  if (!x) return null;
  if (part === 'q0' && x.modes && Object.keys(x.modes).length) return earliest(Object.values(x.modes).map((m) => m && m.at));
  return x.at || null;
};

function readingOf(game, input, report) {
  const errors = [];
  const info = gameInfo(report, game);
  const q0 = guarded(errors, 'q0', () => {
    const v = q0Of(game, input.q0, info, input.unreadable);
    if (!v.band) return null;
    const modes = {};
    Object.keys(input.q0 || {}).forEach((mode) => {
      const r = input.q0[mode];
      modes[mode] = { best: r4(r.best_row_mean), sd: r4(r.best_row_sd_across_seeds), ceiling: r3(Number(r.best_row_mean) + 2 * (Number(r.best_row_sd_across_seeds) || 0)),
        rows: r.rows_compared, seeds: Array.isArray(r.seeds) ? r.seeds.length : null, at: isoUtc(r.generated_at) };
    });
    const { ceiling, from, rows, seeds, history, metric } = v.band;
    return { band: { ceiling, from, rows, seeds, history, metric }, modes, ranking: v.ranking,
      // above null: no ranking to read the band against (no modelPerformance.json, or no rows for this game) - never "none above"
      rankedRows: info ? v.rankedRows : null, minDraws: info ? v.minDraws : null, above: info ? v.above.rows : null,
      // the older of the two records: the band is this week's only once both are (partAt)
      at: earliest(Object.values(modes).map((m) => m.at)) };
  });
  const q2 = guarded(errors, 'q2', () => {
    const v = q2Of(input.q2);
    return v && { status: v.status, real: v.real, shuffled: v.shuffled, threshold: v.threshold, seeds: v.seeds, windowsTest: v.windowsTest,
      window: v.window, drawsUsed: v.drawsUsed, at: v.at };
  });
  const fc = guarded(errors, 'fc', () => {
    if (!input.fc) return null;
    const d = controls.describeFeatureControl(input.fc);
    return { tableRows: d.tableDays, lockboxRows: d.lockboxDays, repeats: d.repeats, noiseColumns: d.noiseColumns, at: isoUtc(d.generatedAt),
      // per variant: the verdict, the noise ratio, the noise columns' share, [held-out AUC without, with noise], the base columns above the band
      variants: d.variants.map((v) => {
        const out = { key: VARIANT_CODE[v.key] || v.key, verdict: v.verdict, ratio: r3(v.noiseRatio), share: r4(v.noiseShare), auc: [r3(v.heldoutWithout), r3(v.heldoutWith)] };
        if (v.above.length) out.above = v.above;   // left out when none is above
        return out;
      }) };
  });
  const rows = isMarketGame(game) ? guarded(errors, 'rows', () => {
    if (!input.rows) return null;
    const d = markets.describeRows(input.rows);
    return { window: d.days, requested: d.requested, lockboxWithheld: d.lockboxWithheld, firstDay: d.firstDay, lastDay: d.lastDay, reference: d.reference,
      at: isoUtc(d.generatedAt),
      // the scored rows (the baselines carry no probability), each against uniform and against the reference: mean log-score
      // difference per draw of the game, 95% interval, verdict; a row's days only where they differ from the window
      rows: d.rows.filter((r) => r.logScore !== null).map((r) => {
        const out = { row: r.name, kind: r.kind, vsUniform: interval(r.vsUniform) };
        if (!r.isReference) out.vsReference = interval(r.vsReference);
        if (r.days !== d.days) out.days = r.days;
        return out;
      }) };
  }) : null;
  unreadableErrors(input.unreadable, { q0Band: q0 && q0.band, q2, fc, rows }).forEach((part) => { if (!errors.includes(part)) errors.push(part); });
  return { unit: UNIT[game], q0, q2, fc, rows, errors };
}
// sunday: the week the reading is written for ('YYYY-MM-DD', a Sunday) - then every part whose record is from another
// week is named in notThisWeek (a control that failed leaves last week's record in place); null for the reading of the
// records on disk now, where each part carries its own week. by / controls / note say who built it from what.
function weekReading(inputs, report, { sunday = null, now = new Date(), by = null, controls: source = null, note = null } = {}) {
  const games = {};
  GAMES.forEach((game) => { games[game] = readingOf(game, inputs[game] || {}, report); });
  // every record's time: Q0 per control, the other parts one record each
  const times = GAMES.flatMap((game) => PARTS.flatMap((part) => {
    const x = games[game][part];
    if (!x) return [];
    return part === 'q0' ? Object.keys(x.modes).map((mode) => ({ game, part: `q0 ${mode}`, at: x.modes[mode].at })) : [{ game, part, at: x.at }];
  })).filter((t) => t.at);
  const reading = { version: READING_VERSION, sunday, builtAt: isoUtc(now.toISOString()), by, controls: source, note,
    rankingAt: isoUtc(report && report.generatedAt), from: earliest(times.map((t) => t.at)), to: latest(times.map((t) => t.at)) };
  if (sunday) reading.notThisWeek = times.filter((t) => sundayOf(t.at) !== sunday);
  reading.games = games;
  return reading;
}

// --- week over week: the stored readings -------------------------------------------------------------------------
// Every data/readings/<sunday>.json, read once per change of its file: {bySunday, sundays (oldest first), unreadable}.
// A file that does not parse, is of another version, or names another day than its file name, is listed unreadable.
function loadReadings(dir) {
  let files = [];
  try { files = fs.readdirSync(dir).filter((f) => READING_FILE.test(f)).sort(); } catch (e) { files = []; }
  const bySunday = new Map();
  const unreadable = [];
  files.forEach((f) => {
    const day = f.slice(0, 10);
    const file = path.join(dir, f);
    const reading = memo(`reading:${file}`, [file], () => {
      try {
        const r = JSON.parse(fs.readFileSync(file, 'utf-8'));
        return r && r.version === READING_VERSION && r.sunday === day && r.games && typeof r.games === 'object' ? r : null;
      } catch (e) { return null; }
    });
    if (reading && brussels.isDay(day) && brussels.weekdayOfDay(day) === 0) bySunday.set(day, reading); else unreadable.push(f);
  });
  return { bySunday, sundays: [...bySunday.keys()], unreadable };
}
// What one control read in one game one week, as one comparable string; null when there is no record. Two weeks read
// the same when the strings are equal: Q0 rows above the band or none; Q2's status; every feature-control variant's
// verdict; which market rows beat the reference, and the reference against uniform.
function stateOf(part, entry) {
  const x = entry && entry[part];
  if (!x) return null;
  if (part === 'q0') return x.above === null || x.above === undefined ? 'no ranking' : (x.above.length ? 'above' : 'none above');
  if (part === 'q2') return x.status || null;
  if (part === 'fc') return list(x.variants).map((v) => `${v.key} ${VERDICT_SHORT[v.verdict] || v.verdict}`).join(' · ');
  if (part === 'rows') {
    const ref = shortRow(x.reference);
    const beat = list(x.rows).filter((r) => r.vsReference && r.vsReference.verdict === 'better').map((r) => shortRow(r.row)).sort();
    const own = list(x.rows).find((r) => r.row === x.reference);
    return `beat ${ref}: ${beat.length ? beat.join(', ') : 'none'} · ${ref} vs uniform: ${(own && own.vsUniform && own.vsUniform.verdict) || 'not scored'}`;
  }
  return null;
}
// One game's part as a stored reading holds it for `day`: undefined when there is no reading of that week, or when the
// part in it is not that week's own measurement - a control that failed or skipped the game leaves last week's record
// on disk, and the reading carries it again (notThisWeek), so counting it would count one run twice; null when the
// reading has no record of it.
function storedPart(stored, day, game, part) {
  const reading = stored.bySunday.get(day);
  if (!reading) return undefined;
  const entry = reading.games && reading.games[game];
  if (!entry || !entry[part]) return null;
  return sundayOf(partAt(part, entry)) === day ? entry : undefined;
}
// A control's run of weeks: the week its record is from (its own Sunday), how many weeks in a row ending there it read
// the same - counting back through stored readings exactly a week apart, so a week without a reading, or whose reading
// holds an older record of this part (that control did not run that week: noRunBefore for the week before), ends the
// run - and whether the week before read differently: null when there is no reading of that week to compare with, and
// `last` (what it read then; null = no record) only when it changed. weeks 1 with changed null is the first week on
// record. Q0 is read against a ranking, and every stored reading is fixed at its own Sunday's: so once the reading of
// the record's own Sunday is written, its state is that reading's (sinceReading: the rows today's ranking puts above
// or below the band that it did not), and before that the records on disk against the live ranking stand in (live).
function streakOf(stored, game, part, entry) {
  const now = stateOf(part, entry);
  const sunday = sundayOf(partAt(part, entry));
  if (now === null || !sunday) return null;
  const then = (day) => { const x = storedPart(stored, day, game, part); return x ? stateOf(part, x) : x; };
  const own = part === 'q0' ? storedPart(stored, sunday, game, part) : undefined;
  const state = own ? stateOf(part, own) : now;
  const before = brussels.addDays(sunday, -7);
  const last = then(before);
  let weeks = 1;
  for (let day = before; then(day) === state; day = brussels.addDays(day, -7)) weeks += 1;
  const out = { state, sunday, weeks, changed: last === undefined ? null : last !== state };
  if (out.changed) out.last = last;
  if (last === undefined && stored.bySunday.has(before)) out.noRunBefore = true;
  if (part === 'q0' && !own) out.live = true;
  if (own) {
    const was = list(own.q0.above).map((r) => r.row);
    const is = list(entry.q0.above).map((r) => r.row);
    const added = is.filter((r) => !was.includes(r));
    const dropped = was.filter((r) => !is.includes(r));
    if (added.length || dropped.length || now !== state) out.sinceReading = { now, added, dropped, rankingAt: stored.bySunday.get(sunday).rankingAt || null };
  }
  return out;
}
// How many weeks in a row, ending at `sunday`, a row has been above its game's Q0 band. read: the reading of `sunday`
// itself is stored (streakOf) - then the count starts there, and a row above only against the live ranking, not in
// that reading, has no weeks (null); otherwise this week is the live one and the stored weeks before it count on.
function rowWeeks(stored, game, row, sunday, read = false) {
  const above = (day) => {
    const x = storedPart(stored, day, game, 'q0');
    return Boolean(x && Array.isArray(x.q0.above) && x.q0.above.some((r) => r.row === row));
  };
  if (read && !above(sunday)) return null;
  let weeks = 1;
  for (let day = brussels.addDays(sunday, -7); above(day); day = brussels.addDays(day, -7)) weeks += 1;
  return weeks;
}
// current: weekReading of the records on disk (no sunday); stored: loadReadings. Per game, each part's streak, with
// behind: true for a part whose record is older than the newest part's (the controls are overwritten while they run).
function historyOf(current, stored) {
  const out = {};
  GAMES.forEach((game) => {
    out[game] = {};
    PARTS.filter((part) => part !== 'rows' || isMarketGame(game)).forEach((part) => { out[game][part] = streakOf(stored, game, part, current.games[game]); });
  });
  const newest = latest(GAMES.flatMap((game) => Object.values(out[game]).map((s) => s && s.sunday)));
  GAMES.forEach((game) => Object.values(out[game]).forEach((s) => { if (s && newest && s.sunday < newest) s.behind = true; }));
  return { games: out, newest };
}
const readingsMeta = (stored, newest) => ({ stored: stored.sundays.length, sundays: stored.sundays.slice(-8), newestRecords: newest || null,
  unreadable: stored.unreadable, rule: 'weeks: the weeks in a row, ending at the record\'s own Sunday, the control read the same in the stored readings (a missing week, or one whose reading holds an older record of the control, ends the run); changed: whether the week before read differently (null: no reading of that week); Q0: the state of the reading of its own Sunday once written (that Sunday\'s ranking), live before that' });
// The Controls page with its history: each game's history per part, each Q0 row above the band with its weeks above.
function withHistory(view, current, stored) {
  const hist = historyOf(current, stored);
  view.groups.forEach((grp) => grp.games.forEach((g) => {
    g.history = hist.games[g.game];
    const q0 = g.history.q0;
    if (q0 && q0.sunday && g.q0 && g.q0.above && g.q0.above.rows.length) {
      g.q0 = { ...g.q0, above: { ...g.q0.above, rows: g.q0.above.rows.map((r) => ({ ...r, weeks: rowWeeks(stored, g.game, r.row, q0.sunday, !q0.live) })) } };
    }
  }));
  view.readings = readingsMeta(stored, hist.newest);
  return view;
}

// --- the runs: config/scheduler.json and the jobs' own logs ------------------------------------------------------
// jobs.js wraps every run in two markers in the job's own log, the start in UTC with the code version:
//   === scheduler: <job name> started 2026-10-18 10:00:15 9e070619 UTC ===
//   === scheduler: <job name> finished status=0 after 45928s ===
// A run's log slice is what lies between them. Its warning signatures: a Python warning, error or exception class
// (with the file that raised it, when the line names one), LOCKBOX, Traceback. Every other line a naive /warn|error/
// grep would catch is counted as filtered noise (TF/XLA start-up chatter, the gate's lines on "standard error").
const MARKER_START = /^=== scheduler: (.+?) started (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: (\S+))? UTC ===$/;
const MARKER_END = /^=== scheduler: (.+?) finished status=(\S+) after (\d+)s ===$/;
const WARNING_CLASS = /\b([A-Z][A-Za-z]*(?:Warning|Error|Exception))\b/;
const WARNING_SOURCE = /([A-Za-z0-9_]+\.py):\d+/;
const NOISE = /warn|error/i;
const SLICE_MATCH_MS = 2 * 60 * 1000;   // a run and its log slice start within seconds of each other
function slicesOf(text) {
  const out = [];
  let cur = null;
  String(text).split('\n').forEach((line) => {
    if (line.startsWith('=== scheduler: ')) {
      let m = MARKER_START.exec(line);
      if (m) {
        if (cur) out.push(cur);   // a run that never wrote its finished marker (killed)
        cur = { start: Date.parse(`${m[2].replace(' ', 'T')}Z`), sha: m[3] || null, status: null, seconds: null, sig: {}, noise: 0 };
        return;
      }
      m = MARKER_END.exec(line);
      if (m && cur) {
        cur.status = /^-?\d+$/.test(m[2]) ? Number(m[2]) : null;
        cur.seconds = Number(m[3]);
        out.push(cur);
        cur = null;
        return;
      }
    }
    if (!cur) return;
    let hit = false;
    const cls = line.includes('Warning') || line.includes('Error') || line.includes('Exception') ? WARNING_CLASS.exec(line) : null;
    if (cls) {
      const src = WARNING_SOURCE.exec(line);
      const key = src ? `${cls[1]} (${src[1]})` : cls[1];
      cur.sig[key] = (cur.sig[key] || 0) + 1;
      hit = true;
    }
    if (line.includes('LOCKBOX')) { cur.sig.LOCKBOX = (cur.sig.LOCKBOX || 0) + 1; hit = true; }
    if (line.includes('Traceback')) { cur.sig.Traceback = (cur.sig.Traceback || 0) + 1; hit = true; }
    if (!hit && NOISE.test(line)) cur.noise += 1;
  });
  if (cur) out.push(cur);   // a run still writing
  return out;
}
// One job's slices, oldest first, from its log and the rotated one (jobs.js keeps one); each file parsed once per change.
function readSlices(dirs, job) {
  if (!job || !job.log) return [];
  return [`${job.log}.1`, job.log].map((f) => path.join(dirs.logDir, f))
    .flatMap((file) => memo(`log:${file}`, [file], () => { try { return slicesOf(fs.readFileSync(file, 'utf-8')); } catch (e) { return []; } }))
    .sort((a, b) => a.start - b.start);
}
function readScheduler(dirs) {
  return memo(`scheduler:${dirs.schedulerFile}`, [dirs.schedulerFile], () => {
    try {
      const s = JSON.parse(fs.readFileSync(dirs.schedulerFile, 'utf-8'));
      return { history: list(s.history), current: s.current || null, queue: list(s.queue) };
    } catch (e) { return null; }
  });
}
const median = (values) => {
  const s = values.slice().sort((a, b) => a - b);
  const n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null;
};
const ran = (h) => Boolean(h) && !h.dry && !h.blocked && Number.isFinite(h.at);
// A run's usual duration in minutes: median, range and count of the earlier successful runs, and their basis - a
// weekly job's earlier runs from its own log's markers (they reach further back than the 40 runs the scheduler
// keeps), the predictor's from the scheduler's history (or when a log holds none).
function usualOf(h, job, slices, history) {
  let minutes = [];
  let at = [];
  let basis = 'earlier runs in the job log';
  if (!job.daily) {
    const prev = list(slices).filter((s) => s.status === 0 && s.seconds !== null && s.start < h.at - SLICE_MATCH_MS);
    minutes = prev.map((s) => s.seconds / 60);
    at = prev.map((s) => s.start);
  }
  if (!minutes.length) {
    const prev = history.filter((x) => x.key === h.key && x.exit === 0 && num(x.seconds) !== null && x.at < h.at - SLICE_MATCH_MS);
    minutes = prev.map((x) => x.seconds / 60);
    at = prev.map((x) => x.at);
    basis = 'earlier runs in the scheduler history';
  }
  if (!minutes.length) return null;
  return { median: r3(median(minutes)), min: r3(Math.min(...minutes)), max: r3(Math.max(...minutes)), n: minutes.length,
    from: isoMs(Math.min(...at)), to: isoMs(Math.max(...at)), basis };
}
// One run: when, how long against the usual, the result, and its warning signatures - each new only when none of the
// same job's earlier slices the logs still hold had it (a signature that skipped a weekend is recurring, not new);
// null when there is nothing earlier to compare with. warnings null: the run's log slice is no longer in its log.
function runLine(h, job, slices, history) {
  const own = list(slices).find((s) => Math.abs(s.start - h.at) < SLICE_MATCH_MS) || null;
  const earlier = list(slices).filter((s) => s.start < h.at - SLICE_MATCH_MS);
  const seen = new Set(earlier.flatMap((s) => Object.keys(s.sig)));
  const minutes = num(h.seconds) === null ? null : h.seconds / 60;
  const usual = usualOf(h, job, slices, history);
  return {
    key: h.key, job: job.name, start: isoMs(h.at), end: isoMs(h.finishedAt), minutes: r3(minutes),
    result: h.exit === 0 ? 'ok' : (num(h.exit) === null ? 'unknown' : `exit ${h.exit}`), note: h.note || h.error || null, sha: h.sha || null,
    usual, overUsual: usual && minutes !== null ? minutes > usual.max : null, vsMedian: usual && minutes !== null && usual.median ? r2(minutes / usual.median) : null,
    warnings: own ? Object.entries(own.sig).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([sig, n]) => ({ sig, n, new: earlier.length ? !seen.has(sig) : null })) : null,
    noiseFiltered: own ? own.noise : null,
    comparedWith: earlier.length ? { runs: earlier.length, from: isoMs(earlier[0].start), to: isoMs(earlier[earlier.length - 1].start) } : null,
  };
}
// Whether a run belongs to the weekend of `saturday`: the daily job on Saturday or Sunday; a weekly plan's job started
// from Saturday 00:00 until Tuesday 09:00, where the controls' catch-up net ends (Brussels time).
function inWeekend(h, job, saturday) {
  if (!job || !Number.isFinite(h.at)) return false;
  if (job.daily) { const day = brussels.dayOf(h.at); return day === saturday || day === brussels.addDays(saturday, 1); }
  return Boolean(job.plan) && h.at >= brussels.instant(saturday) && h.at < brussels.instant(brussels.addDays(saturday, 3), 9);
}

// --- GET /api/admin/tuning ---------------------------------------------------------------------------------------
const TUNER_ORDER = gates.TUNERS.map((t) => t.tuner).concat('unattributed');
// current: {game: tuningGate records} of the working tree; hist: gates.history; runs: the scheduler's runs. One entry
// per record: its numbers (gates.describe), who decided it (gates.attribute) and its last weekends' decisions, aligned
// with hist.weekends (null: that weekend did not decide the row; 'not read': git could not be read for that weekend).
function tuningRecords(current, hist, runs) {
  const out = [];
  GAMES.forEach((game) => Object.entries(current[game] || {}).forEach(([row, rec]) => {
    const d = gates.describe(row, rec);
    const ms = Date.parse(d.date);
    const attribution = gates.attribute(row, rec, { carrier: ((hist.carriers || {})[game] || {})[row] || null, run: Number.isFinite(ms) ? gates.runOf(ms, runs) : null });
    const past = ((hist.decisions || {})[game] || {})[row] || [];
    out.push({ game, ...d, tuner: attribution.tuner, attribution, history: list(hist.weekends).map((w, i) => (past[i] === undefined ? 'not read' : past[i])) });
  }));
  return out;
}
const countOf = (recs) => ({ n: recs.length, replaced: recs.filter((r) => r.decision === 'replaced').length,
  kept: recs.filter((r) => r.decision === 'kept').length, reverted: recs.filter((r) => r.decision === 'reverted').length });
// replacements whose challenger gained less than one of its own standard errors over the incumbent (gainSe), or over
// the better of incumbent and default - the reference the margin is added to (gainBestSe)
const under1Se = (recs, key) => recs.filter((r) => r.decision === 'replaced' && r[key] !== null && r[key] < 1).length;
const atMarginZero = (recs) => recs.filter((r) => !(r.margin > 0)).length;
const rowsBy = (recs, decision) => {
  const out = {};
  recs.filter((r) => r.decision === decision).forEach((r) => { (out[r.game] = out[r.game] || []).push(r.row); });
  return out;
};
const TUNING_META = {
  decide: 'replaced when the challenger beats the better scored reference (incumbent, default) by the margin; reverted when the untuned default is served instead; kept otherwise (src/TuningGate.py decide)',
  scores: 'each the gate\'s own score on the gate window: hits, positional, capped_profit, log_score and subset_bound are lower bounds (mean - 1 SE); holdout_hits and holdout_profit are means on the quantum tuner\'s holdout; a keno capped_profit score is the capped profit above a fair ticket (fairPerBet, its capped value per bet), not money won - profitPerBet is the raw, uncapped profit per bet in EUR',
  margin: 'margin in score units; marginSe = margin / challenger SE (1 for the tuners that ask one SE, 0 for any gain)',
  gains: 'gainSe = (challenger - incumbent) / challenger SE; gainBestSe over the better reference; gainBarSe over the bar (better reference + margin), the one that decided',
  trials: 'runTrials: this run\'s scored trials; studyTrials: the study\'s completed trials the sampler was guided by',
  weekend: 'a decision belongs to the weekend of the Saturday on or before it (Brussels)',
  attribution: 'record: the record names its tuner (since 21 Oct 2026); family + commit / family + run: the row\'s tuner confirmed by the tuner commit that first carries the record, or the scheduler run holding its date; unattributed otherwise',
};
function tuningSummary(records, hist, runs = [], missing = [], now = new Date()) {
  const weekend = hist.newest || latest(records.map((r) => r.weekend));
  const mine = records.filter((r) => r.weekend === weekend);
  const older = records.filter((r) => r.weekend !== weekend);
  const dates = mine.map((r) => r.date).filter(Boolean);
  // what this weekend did with the rows the weekend before replaced (held: kept against a new challenger)
  const at = list(hist.weekends).indexOf(weekend);
  const previous = at >= 0 ? list(hist.weekends)[at + 1] || null : null;
  const replacedThen = previous ? records.filter((r) => r.history[at + 1] === 'replaced') : [];
  const thisTime = (decision) => replacedThen.filter((r) => r.weekend === weekend && r.decision === decision).length;
  const held = previous ? { weekend: previous, replaced: replacedThen.length, kept: thisTime('kept'), replacedAgain: thisTime('replaced'),
    reverted: thisTime('reverted'), notDecided: replacedThen.filter((r) => r.weekend !== weekend).length } : null;
  const byTuner = TUNER_ORDER.map((tuner) => {
    const recs = mine.filter((r) => r.tuner === tuner);
    if (tuner === 'unattributed' && !recs.length) return null;
    const spec = gates.TUNERS.find((t) => t.tuner === tuner);
    const run = spec ? runs.filter((h) => h.key === spec.job && ran(h) && gates.weekendOf(h.at) === weekend).sort((x, y) => x.at - y.at).pop() || null : null;   // the latest, if run twice
    // the gate window per unit: one backtest "day" is a draw, a day, a trading day or a week, by game
    const windows = {};
    recs.filter((r) => r.windowDays !== null).forEach((r) => { (windows[UNIT[r.game]] = windows[UNIT[r.game]] || []).push(r.windowDays); });
    return {
      tuner, job: spec ? spec.job : null, ...countOf(recs), marginZero: atMarginZero(recs),
      marginOneSe: recs.filter((r) => r.marginSe !== null && Math.abs(r.marginSe - 1) < 1e-6).length,
      replacedUnder1Se: under1Se(recs, 'gainSe'), replacedUnder1SeVsBetterReference: under1Se(recs, 'gainBestSe'),
      games: new Set(recs.map((r) => r.game)).size, kinds: [...new Set(recs.map((r) => r.kind).filter(Boolean))],
      // the records whose capped-profit score is measured above a fair ticket (keno): its fair value per bet
      aboveFair: recs.filter((r) => r.challenger.fairPerBet !== undefined).length, fairPerBet: range(recs.map((r) => r4(r.challenger.fairPerBet))),
      windowDays: range(recs.map((r) => r.windowDays)),
      windowByUnit: Object.fromEntries(Object.values(UNIT).filter((u, i, all) => all.indexOf(u) === i && windows[u]).map((u) => [u, range(windows[u])])),
      runTrials: range(recs.map((r) => r.runTrials)), studyTrials: range(recs.map((r) => r.studyTrials)),
      run: run ? { start: isoMs(run.at), end: isoMs(run.finishedAt), minutes: r3(num(run.seconds) === null ? null : run.seconds / 60), result: run.exit === 0 ? 'ok' : (num(run.exit) === null ? 'unknown' : `exit ${run.exit}`) } : null,
      replacedRows: rowsBy(recs, 'replaced'), revertedRows: rowsBy(recs, 'reverted'),
    };
  }).filter(Boolean);
  const byGame = GAMES.map((game) => {
    const recs = mine.filter((r) => r.game === game);
    return { game, group: groupOf(game), ...countOf(recs), marginZero: atMarginZero(recs), notThisWeekend: older.filter((r) => r.game === game).length,
      noFile: missing.includes(game), replacedRows: recs.filter((r) => r.decision === 'replaced').map((r) => r.row), revertedRows: recs.filter((r) => r.decision === 'reverted').map((r) => r.row) };
  });
  return {
    builtAt: isoUtc(now.toISOString()), weekend,
    tiles: {
      ...countOf(mine), marginZero: atMarginZero(mine), replacedUnder1Se: under1Se(mine, 'gainSe'), replacedUnder1SeVsBetterReference: under1Se(mine, 'gainBestSe'),
      under1SeTuners: [...new Set(mine.filter((r) => r.decision === 'replaced' && r.gainSe !== null && r.gainSe < 1).map((r) => r.tuner))],
      unattributed: mine.filter((r) => r.tuner === 'unattributed').length, notThisWeekend: older.length, games: new Set(mine.map((r) => r.game)).size,
      from: earliest(dates), to: latest(dates), heldFromLastWeekend: held,
    },
    byTuner, byGame,
    unattributed: records.filter((r) => r.tuner === 'unattributed').map((r) => ({ game: r.game, row: r.row, weekend: r.weekend, family: r.attribution.family || null, why: r.attribution.why })),
    history: { weekends: list(hist.weekends), commits: hist.commits || 0, error: hist.error || null,
      source: 'the newest weekend from bestParams_<game>.json, each earlier one from its last "Saving latest ... hyperopt" commit' },
    meta: TUNING_META,
  };
}
// --- GET /api/admin/tuning/:game -------------------------------------------------------------------------------------
// a keno side also carries its fair ticket's capped value and its raw profit per bet (gates.describe)
const sideOf = (s) => ({ score: r4(s.score), se: r4(s.se), days: s.days,
  ...(s.fairPerBet !== undefined ? { fairPerBet: r4(s.fairPerBet), profitPerBet: r4(s.profitPerBet) } : {}) });
function tuningGame(records, game, hist, now = new Date()) {
  const weekend = hist.newest || latest(records.map((r) => r.weekend));
  const order = (r) => TUNER_ORDER.indexOf(r.tuner);
  const recs = records.filter((r) => r.game === game).sort((a, b) => order(a) - order(b) || a.row.localeCompare(b.row));
  const mine = recs.filter((r) => r.weekend === weekend);
  return {
    // unit: what one backtest day of the gate window is in this game (a draw, a day, a trading day, a week)
    game, group: groupOf(game), unit: UNIT[game], builtAt: isoUtc(now.toISOString()), weekend, ...countOf(mine), notThisWeekend: recs.length - mine.length,
    weekends: list(hist.weekends), historyError: hist.error || null,
    rows: recs.map((r) => {
      const out = {
        row: r.row, tuner: r.tuner, by: r.attribution.by || null, decision: r.decision, thisWeekend: r.weekend === weekend, weekend: r.weekend, date: r.date,
        kind: r.kind, windowDays: r.windowDays, runTrials: r.runTrials, studyTrials: r.studyTrials,
        default: sideOf(r.default), incumbent: sideOf(r.incumbent), challenger: sideOf(r.challenger),
        margin: r4(r.margin), marginSe: r2(r.marginSe), bar: r4(r.bar), gainSe: r2(r.gainSe), gainBestSe: r2(r.gainBestSe), gainBarSe: r2(r.gainBarSe),
        history: r.history,
      };
      if (r.script) out.script = r.script;
      if (r.tuner === 'unattributed') { out.why = r.attribution.why; out.family = r.attribution.family || null; }
      return out;
    }),
    meta: TUNING_META,
  };
}

// --- GET /api/admin/week ---------------------------------------------------------------------------------------------
// The sign-in events since a moment, counted - no names, no addresses. events: audit.recent's newest-first entries.
function signInsOf(events, fromMs, weekMs) {
  if (!Array.isArray(events)) return null;
  const count = (since, event) => events.filter((e) => e && e.event === event && Date.parse(e.at) >= since).length;
  const oldest = events.length ? isoUtc(events[events.length - 1].at) : null;
  return { failed: count(fromMs, 'login.failed'), locked: count(fromMs, 'login.locked'), ok: count(fromMs, 'login.ok'),
    last7Days: { failed: count(weekMs, 'login.failed'), locked: count(weekMs, 'login.locked') }, scanned: events.length, oldestScanned: oldest };
}
// jobs: [{key, name, log, plan, daily}] (jobs.js JOBS); sched: config/scheduler.json; slices: {job key: log slices};
// controlsView: controlsOverview with its history (withHistory); tuning: tuningSummary; live: what the running server
// knows now - {lock, current, queue, council, events} - each missing when not known (null lock = free).
function weekView({ jobs = [], sched = null, slices = {}, controlsView, tuning, live = {}, now = new Date() }) {
  const nowMs = now.getTime();
  const history = list(sched && sched.history).filter(ran);
  const byKey = Object.fromEntries(list(jobs).map((j) => [j.key, j]));
  const daily = list(jobs).find((j) => j.daily) || null;
  const current = live.current !== undefined ? live.current : (sched && sched.current) || null;
  const queue = live.queue !== undefined ? live.queue : list(sched && sched.queue);
  const currentRun = current && Number.isFinite(current.startedAt) ? { key: current.key, at: current.startedAt } : null;
  // the weekend: this Saturday's once anything of it has started, else the one before
  const weekendRuns = (sat) => history.filter((h) => inWeekend(h, byKey[h.key], sat)).sort((a, b) => a.at - b.at);
  let saturday = brussels.lastWeekday(nowMs, 6);
  let runs = weekendRuns(saturday);
  if (!runs.length && !(currentRun && inWeekend(currentRun, byKey[currentRun.key], saturday))) {
    saturday = brussels.addDays(saturday, -7);
    runs = weekendRuns(saturday);
  }
  const sunday = brussels.addDays(saturday, 1);
  const lines = runs.map((h) => runLine(h, byKey[h.key], slices[h.key], history));
  // what a weekend runs: the daily job on Saturday and on Sunday, every job of the two weekly plans once
  const expected = [...(daily ? [{ key: daily.key, day: saturday }, { key: daily.key, day: sunday }] : []),
    ...list(jobs).filter((j) => j.plan).map((j) => ({ key: j.key }))];
  const missing = expected.filter((e) => !runs.some((h) => h.key === e.key && (!e.day || brussels.dayOf(h.at) === e.day)));
  const runningNow = currentRun && inWeekend(currentRun, byKey[currentRun.key], saturday) ? { key: currentRun.key, start: isoMs(currentRun.at) } : null;
  const isRunning = (e) => Boolean(runningNow) && runningNow.key === e.key && (!e.day || e.day === brussels.dayOf(currentRun.at));
  const over = lines.filter((l) => l.overUsual).sort((a, b) => (b.minutes - b.usual.max) - (a.minutes - a.usual.max))[0] || null;
  const starts = runs.map((h) => h.at);
  const ends = runs.map((h) => h.finishedAt).filter(Number.isFinite);
  const today = brussels.dayOf(nowMs);
  const todays = daily ? history.filter((h) => h.key === daily.key && brussels.dayOf(h.at) === today).sort((a, b) => a.at - b.at) : [];
  const mondayMs = brussels.instant(brussels.lastWeekday(nowMs, 1));
  const games = controlsView.groups.flatMap((grp) => grp.games);
  const gateOf = Object.fromEntries(list(tuning && tuning.byGame).map((g) => [g.game, g]));
  const ct = controlsView.tiles;
  return {
    builtAt: isoUtc(now.toISOString()), tz: brussels.TZ,
    // the ranking the Q0 rows above are read against (the Controls page's rankingAt)
    rankingAt: controlsView.rankingAt || null,
    weekend: { saturday, from: isoMs(brussels.instant(saturday)), until: isoMs(brussels.instant(brussels.addDays(saturday, 3), 9)) },
    tiles: {
      weekend: {
        runs: lines.length, ok: lines.filter((l) => l.result === 'ok').length, expected: expected.length,
        missing: missing.filter((e) => !isRunning(e)), running: runningNow,
        notOk: lines.filter((l) => l.result !== 'ok').map((l) => ({ key: l.key, result: l.result, note: l.note })),
        from: isoMs(starts.length ? Math.min(...starts) : NaN), to: isoMs(ends.length ? Math.max(...ends) : NaN),
        hours: starts.length && ends.length ? r2((Math.max(...ends) - Math.min(...starts)) / 3600000) : null,
        slowest: over ? { key: over.key, minutes: over.minutes, usual: over.usual } : null,
        newWarnings: lines.flatMap((l) => list(l.warnings).filter((w) => w.new).map((w) => ({ key: l.key, sig: w.sig, n: w.n }))),
      },
      today: {
        day: today, run: todays.length ? runLine(todays[todays.length - 1], daily, slices[daily.key], history) : null,
        running: current && daily && current.key === daily.key ? { start: isoMs(current.startedAt) } : null,
      },
      tuners: tuning ? { weekend: tuning.weekend, ...tuning.tiles,
        // job: the scheduler job (a run's drawer finds its tuner's decisions); games and the margins for the Tuning card
        byTuner: tuning.byTuner.map((t) => ({ tuner: t.tuner, job: t.job, n: t.n, replaced: t.replaced, kept: t.kept, reverted: t.reverted, games: t.games,
          marginZero: t.marginZero, marginOneSe: t.marginOneSe, replacedUnder1Se: t.replacedUnder1Se })) } : null,
      q0: ct.q0, q2: ct.q2, fc: ct.fc, rows: ct.rows,
    },
    runs: lines,
    matrix: {
      columns: ['q0', 'q2', 'fc', 'rows', 'gates'],
      // RandomnessDiscrimination's power rule: a Q2 record on fewer test windows than this is underpowered
      fewWindows: FEW_WINDOWS,
      games: games.map((g) => {
        const h = g.history || {};
        const gate = gateOf[g.game];
        return {
          game: g.game, group: groupOf(g.game), unit: g.unit, errors: g.errors,
          // at: the record the band comes from; modes: each control record's own time (what a run wrote is judged on both)
          q0: g.q0.band ? { above: g.q0.above.n, ceiling: g.q0.band.ceiling, longest: g.q0.above.longest, longestLive: g.q0.above.longestLive,
            sameLength: g.q0.above.sameLength, sameLive: g.q0.above.sameLive, inTile: g.q0.above.inTile, ranking: g.q0.ranking, seeds: g.q0.band.seeds,
            at: g.q0.band.at, modes: g.q0.band.modes, history: h.q0 || null } : null,
          q2: g.q2 ? { status: g.q2.status, real: g.q2.real, threshold: g.q2.threshold, seeds: g.q2.seeds, windowsTest: g.q2.windowsTest, fewWindows: g.q2.fewWindows,
            at: g.q2.at, history: h.q2 || null } : null,
          fc: g.fc ? { fits: g.fc.fits, inconclusive: g.fc.inconclusive, ignored: g.fc.ignored, failed: g.fc.failed, variants: g.fc.variants.length, at: g.fc.at, history: h.fc || null } : null,
          rows: g.rows ? { beat: g.rows.betterThanReference.length, scored: g.rows.rowsScored, reference: g.rows.reference, window: g.rows.window,
            referenceVsUniform: g.rows.referenceVsUniform ? g.rows.referenceVsUniform.verdict : null, at: g.rows.at, history: h.rows || null } : null,
          gates: gate ? { replaced: gate.replaced, kept: gate.kept, reverted: gate.reverted, notThisWeekend: gate.notThisWeekend } : null,
        };
      }),
    },
    readings: controlsView.readings || null,
    since: {
      from: isoMs(mondayMs),
      predictorRuns: daily ? history.filter((h) => h.key === daily.key && h.at >= mondayMs).sort((a, b) => a.at - b.at).map((h) => runLine(h, daily, slices[daily.key], history)) : [],
      queue: list(queue).map((q) => ({ key: q.key, name: (byKey[q.key] || {}).name || q.key, queuedAt: isoMs(q.queuedAt), manual: Boolean(q.manual) })),
      running: current ? { key: current.key, name: (byKey[current.key] || {}).name || current.name || current.key, start: isoMs(current.startedAt) } : null,
      // null: not known here; {held: false}: free
      lock: live.lock === undefined ? null : (live.lock ? { held: true, pid: live.lock.pid, script: live.lock.script, key: live.lock.key || null } : { held: false }),
      council: live.council === undefined ? null : live.council,
      signIns: signInsOf(live.events, mondayMs, nowMs - 7 * 86400000),
    },
  };
}

// --- routes ------------------------------------------------------------------------
// The guard: auth.middleware has already answered a signed-out /api request with its 401; this one turns away a
// signed-in reader who is not the administrator. Open access (no login configured) is the administrator.
function guard(req, res, next) {
  if (!req.user) return res.status(401).set('Cache-Control', 'no-store').json({ error: 'login required' });
  if (req.user.role !== 'admin') return res.status(403).set('Cache-Control', 'no-store').json({ error: 'admin only' });
  return next();
}

function install(app, options = {}) {
  const dirs = resolveDirs(options);
  const json = (res, body, status = 200) => { res.set('Cache-Control', 'no-store'); res.status(status).json(body); };
  // a route's body, or the JSON error that says why there is none; a build that throws is logged and answered 500
  const answer = (res, build) => {
    let body;
    try { body = build(); } catch (e) {
      console.error(`[${new Date().toISOString()}] admin.js:`, e);
      return json(res, { error: 'could not build this view' }, 500);
    }
    return body && body.error ? json(res, { error: body.error }, body.status || 404) : json(res, body);
  };
  const gameOf = (req) => (GAMES.includes(req.params.game) ? req.params.game : null);
  const scopeOf = (req) => (req.query.scope === undefined || req.query.scope === '' ? 'all' : String(req.query.scope));
  const report = () => readReport(dirs);
  // the names in each game's newest day file: the users' top five are the first five ranked rows among them
  const todayOf = (game) => rankings.namesToday(dirs.databaseDir, game);
  const todayAll = () => Object.fromEntries(GAMES.map((g) => [g, todayOf(g)]));
  const followsAll = () => Object.fromEntries(MARKET_GAMES.map((g) => [g, readFollows(dirs, g)]));
  const noReport = { error: 'no modelPerformance.json yet', status: 404 };
  const badRequest = (req) => {
    if (!gameOf(req)) return { error: 'unknown game', status: 404 };
    if (!SCOPES.includes(scopeOf(req))) return { error: 'scope is all or top', status: 400 };
    return null;
  };

  // the routes that wait on git or on the Council API's health probe; a failure is a 500, never an unhandled rejection
  const answerLater = (res, build) => Promise.resolve().then(build).then(
    (body) => (body && body.error ? json(res, { error: body.error }, body.status || 404) : json(res, body)),
    (e) => {
      console.error(`[${new Date().toISOString()}] admin.js:`, e);
      return json(res, { error: 'could not build this view' }, 500);
    });
  // the Controls page and its history: the records on disk read as a reading, against the stored readings
  const controlsWithHistory = () => {
    const inputs = readAll(dirs);
    return withHistory(controlsOverview(inputs, report()), weekReading(inputs, report()), loadReadings(dirs.readingsDir));
  };
  // this weekend's gate records with their attribution and their last weekends (git: cached per sha)
  const readTuning = async () => {
    const files = Object.fromEntries(GAMES.map((g) => [g, gates.readGates(dirs.root, g)]));
    const current = Object.fromEntries(GAMES.map((g) => [g, files[g] || {}]));
    let hist;
    try { hist = await gates.history(dirs.root, GAMES, current); } catch (e) {
      // the full message to the server log; the page gets a fixed phrase, never a path or a file excerpt
      console.error(`[${new Date().toISOString()}] admin.js: tuning history:`, e);
      hist = { weekends: [], newest: null, decisions: {}, carriers: {}, commits: 0, error: 'git history not available' };
    }
    const sched = readScheduler(dirs);
    const runs = list(sched && sched.history).filter(ran);
    return { records: tuningRecords(current, hist, runs), hist, runs, missing: GAMES.filter((g) => files[g] === null) };
  };
  // the job list, from jobs.js when the server passes it (options.jobs); the week has no runs without it
  const jobList = () => list(options.jobs && options.jobs.JOBS).map((j) => ({ key: j.key, name: j.name, log: j.log, plan: j.plan || null, daily: Boolean(j.daily) }));
  const COUNCIL_WAIT_MS = 6000;   // services.js probes for up to 5 s
  const within = (promise, ms) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`no answer within ${ms / 1000} s`)), ms);
    Promise.resolve(promise).then((v) => { clearTimeout(timer); resolve(v); }, (e) => { clearTimeout(timer); reject(e); });
  });
  // what the running server knows now: the lock, the running job and the queue (jobs.status), the Council API
  // (services.status, which probes its health), the audit events; each left out when its module was not passed
  const liveStatus = async () => {
    const out = {};
    if (options.jobs && typeof options.jobs.status === 'function') {
      try {
        const s = options.jobs.status();
        out.lock = s.lock || null;
        out.current = s.current ? { key: s.current.key, name: s.current.name, startedAt: s.current.startedAt } : null;
        out.queue = list(s.queue);
      } catch (e) { /* the schedule has not started yet: the scheduler file stands in */ }
    }
    if (options.services && typeof options.services.status === 'function') {
      try {
        const all = await within(options.services.status(), COUNCIL_WAIT_MS);
        const c = list(all).find((s) => s.key === 'councilApi') || null;
        out.council = c ? { name: c.name, status: c.status, healthy: c.healthy, summary: c.summary, since: isoMs(c.since), restarts: c.restarts,
          lastExit: c.lastExit ? { code: c.lastExit.code, signal: c.lastExit.signal, at: isoMs(c.lastExit.at) } : null, lastError: noPaths(c.lastError), checkedAt: isoUtc(new Date().toISOString()) } : null;
      } catch (e) { out.council = { error: noPaths(e.message), checkedAt: isoUtc(new Date().toISOString()) }; }
    }
    if (options.audit && typeof options.audit.recent === 'function') {
      try { out.events = options.audit.recent(1000); } catch (e) { /* no audit log here */ }
    }
    return out;
  };

  app.use('/api/admin', guard);
  app.get('/api/admin/controls', (req, res) => answer(res, controlsWithHistory));
  app.get('/api/admin/controls/:game', (req, res) => answer(res, () => {
    const game = gameOf(req);
    if (!game) return { error: 'unknown game', status: 404 };
    return controlsDrawer(game, readControls(dirs, game), report(), readRegimes(dirs, game));
  }));
  app.get('/api/admin/models', (req, res) => answer(res, () => (report() ? modelsSummary(report(), readAll(dirs), { today: todayAll(), follows: followsAll() }) : noReport)));
  app.get('/api/admin/models/:game', (req, res) => answer(res, () => {
    const bad = badRequest(req);
    if (bad) return bad;
    if (!report()) return noReport;
    return modelsGame(report(), gameOf(req), scopeOf(req), readControls(dirs, gameOf(req)), { today: todayOf(gameOf(req)), follows: readFollows(dirs, gameOf(req)) }) || { error: 'no ranking for this game yet', status: 404 };
  }));
  app.get('/api/admin/phase-shift', (req, res) => answer(res, () => (report() ? phaseShiftSummary(report(), todayAll()) : noReport)));
  app.get('/api/admin/phase-shift/:game', (req, res) => answer(res, () => {
    const bad = badRequest(req);
    if (bad) return bad;
    if (!report()) return noReport;
    return phaseShiftGame(report(), gameOf(req), scopeOf(req), todayOf(gameOf(req))) || { error: 'no phase-shift record for this game', status: 404 };
  }));
  app.get('/api/admin/randomness', (req, res) => answer(res, () => {
    if (!report()) return noReport;
    return randomnessSummary(report(), Object.fromEntries(GAMES.map((g) => [g, readControls(dirs, g).q2])));
  }));
  app.get('/api/admin/randomness/:game', (req, res) => answer(res, () => {
    const bad = badRequest(req);
    if (bad) return bad;
    if (!report()) return noReport;
    return randomnessGame(report(), gameOf(req), scopeOf(req), readControls(dirs, gameOf(req)).q2, todayOf(gameOf(req))) || { error: 'no randomness record for this game', status: 404 };
  }));
  app.get('/api/admin/tuning', (req, res) => answerLater(res, async () => {
    const t = await readTuning();
    return tuningSummary(t.records, t.hist, t.runs, t.missing);
  }));
  app.get('/api/admin/tuning/:game', (req, res) => answerLater(res, async () => {
    const game = gameOf(req);
    if (!game) return { error: 'unknown game', status: 404 };
    const t = await readTuning();
    if (t.missing.includes(game)) return { error: 'no bestParams file for this game', status: 404 };
    return tuningGame(t.records, game, t.hist);
  }));
  app.get('/api/admin/week', (req, res) => answerLater(res, async () => {
    const [t, live] = await Promise.all([readTuning(), liveStatus()]);
    const jobs = jobList();
    return weekView({ jobs, sched: readScheduler(dirs), slices: Object.fromEntries(jobs.map((j) => [j.key, readSlices(dirs, j)])),
      controlsView: controlsWithHistory(), tuning: tuningSummary(t.records, t.hist, t.runs, t.missing), live });
  }));
  // anything else under /api/admin is a JSON 404, never the old pages' HTML
  app.use('/api/admin', (req, res) => json(res, { error: 'not found' }, 404));
}

module.exports = {
  GAMES, LOTTERY_GAMES, MARKET_GAMES, UNIT, isoUtc, resolveDirs, readControls, readRegimes, readAll, readReport,
  q0Of, q2Of, fcOf, rowsOf, controlsOverview, controlsDrawer, modelsSummary, modelsGame, phaseShiftSummary, phaseShiftGame,
  randomnessSummary, randomnessGame, guard, install,
  READING_VERSION, PARTS, sundayOf, weekReading, loadReadings, stateOf, streakOf, historyOf, withHistory,
  slicesOf, readSlices, readScheduler, runLine, weekView, tuningRecords, tuningSummary, tuningGame,
};
