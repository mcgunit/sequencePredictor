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
const controls = require('./controls');
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
// regime logs), each overridable; root is the checkout (bestParams_<game>.json, for the tuning gates of phase P2).
function resolveDirs({ dataDir, root, databaseDir, controlsDir, marketsDir } = {}) {
  const data = dataDir || path.join(root || __dirname, 'data');
  return {
    databaseDir: databaseDir || path.join(data, 'database'),
    controlsDir: controlsDir || path.join(data, 'controls'),
    marketsDir: marketsDir || path.join(data, 'markets'),
    root: root || __dirname,
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
  return {
    ranking: payout ? 'profit' : 'hits',
    band: band ? { ceiling: r3(band.ceiling), from: band.from, rows: mode.rows, seeds: mode.seeds, history: mode.days, metric: band.metric,
      at: isoUtc(mode.generatedAt), missing: bandMissing(band, unreadable) } : null,
    rankedRows: ranked.length, allRows: models.length, minDraws,
    best: byHits.length ? { ...describe(byHits[0]), clears: band ? byHits[0].avg_hits > band.ceiling : null } : null,
    leader: payout && models.length ? { row: models[0].name, draws: models[0].draws } : null,
    above: {
      n: above.length,
      longest: lengths.length ? Math.max(...lengths) : null,
      longestLive: above.length && isMarketGame(game) ? Math.max(...above.map((m) => live(m.name) || 0)) : null,
      sameLength: above.length ? new Set(lengths).size === 1 : null,
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
  above: { n: 0, longest: null, longestLive: null, sameLength: null, inTile: false, rows: [] } };
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

  app.use('/api/admin', guard);
  app.get('/api/admin/controls', (req, res) => answer(res, () => controlsOverview(readAll(dirs), report())));
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
  // anything else under /api/admin is a JSON 404, never the old pages' HTML
  app.use('/api/admin', (req, res) => json(res, { error: 'not found' }, 404));
}

module.exports = {
  GAMES, LOTTERY_GAMES, MARKET_GAMES, UNIT, isoUtc, resolveDirs, readControls, readRegimes, readAll, readReport,
  q0Of, q2Of, fcOf, rowsOf, controlsOverview, controlsDrawer, modelsSummary, modelsGame, phaseShiftSummary, phaseShiftGame,
  randomnessSummary, randomnessGame, guard, install,
};
