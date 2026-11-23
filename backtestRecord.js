'use strict';
// The backtest record on the lottery pages (roadmap item 13, decided 21 and 22 Nov 2026): beside the best model, the
// best combination and each of the top 5, in how many of ten past calendar months the row's ticket ended in profit,
// with the same count for random tickets. Reads data/backtestRecord/<game>.json, written by BacktestRecord.py (the
// months, a draw's money and the random players: src/BacktestMonths.py). Nothing is computed about a model here.
//
//   a row            tested: its months in profit, of the months it has a played draw in, and the random reference;
//                    `changed` when the settings it was backtested with are not the ones it is served with now (the
//                    weekend job backtests it anew). Without an entry: "not tested yet" - shown, never hidden.
//   a combination    every member played on every draw: the members' months added up. One member without a record
//                    and the combination is not tested yet.
//   random tickets   the average number of months in profit over the file's random players; for a combination the
//                    sums of as many players as it has members (the same offsets as BacktestMonths.random_count).
const fs = require('fs');
const path = require('path');
const params = require('./params');

const VERSION = 1;
const MONEY = { vikinglotto: 2 };   // BacktestRecord.MONEY: a record booked the old way is not shown
const MEMBER_STEP = 53;
const PLAY_STEP = 17;
const NOT_TESTED = 'not tested yet';
const cache = new Map();   // file -> {mtimeMs, size, record}

const dirOf = (dirs) => path.join(path.dirname(dirs.databaseDir), 'backtestRecord');
function load(dirs, game) {
  const file = path.join(dirOf(dirs), `${game}.json`);
  let stat;
  try { stat = fs.statSync(file); } catch (e) { cache.delete(file); return null; }
  const kept = cache.get(file);
  if (kept && kept.mtimeMs === stat.mtimeMs && kept.size === stat.size) return kept.record;
  let record = null;
  try {
    const data = JSON.parse(fs.readFileSync(file, 'utf-8'));
    if (data && data.version === VERSION && (data.money || 1) === (MONEY[game] || 1) && data.rows && typeof data.rows === 'object' && Array.isArray(data.months)) record = data;
  } catch (e) { record = null; }
  cache.set(file, { mtimeMs: stat.mtimeMs, size: stat.size, record });
  return record;
}
const stampOf = (dirs, game) => { try { const s = fs.statSync(path.join(dirOf(dirs), `${game}.json`)); return `${s.mtimeMs}:${s.size}`; } catch (e) { return '-'; } };

// the settings a row was backtested with are the ones it is served with now (BacktestRecord.served_now)
function servedNow(entry, served) {
  if (!served) return true;   // the served file could not be read: nothing is claimed
  const fromDefault = new Set(Array.isArray(entry.fromDefault) ? entry.fromDefault : []);
  return Object.entries(entry.values || {}).every(([key, value]) => (
    Object.prototype.hasOwnProperty.call(served, key) ? JSON.stringify(served[key]) === JSON.stringify(value) : fromDefault.has(key)));
}
// the average number of months in profit over the random players, for members = [[plays], ...]; null without a simulation
function randomCount(record, members, months) {
  const sims = record.random && record.random.months;
  if (!sims || !months.length || !members.length) return null;
  let total = 0;
  let n = null;
  for (const month of months) {
    const plays = sims[month];
    if (!plays) return null;
    const series = members.map((member) => member.map((play) => plays[play]));
    if (series.some((member) => !member.length || member.some((s) => !Array.isArray(s) || !s.length))) return null;
    n = Math.min(...series.flat().map((s) => s.length));
    for (let sim = 0; sim < n; sim += 1) {
      let value = 0;
      series.forEach((member, m) => member.forEach((s, p) => { value += s[(sim + m * MEMBER_STEP + p * PLAY_STEP) % n]; }));
      if (value > 0) total += 1;
    }
  }
  return n ? Math.round((total / n) * 10) / 10 : null;
}
const playedMonths = (record, entry) => record.months.filter((m) => entry.months && entry.months[m] && entry.months[m].draws > 0);
const round2 = (v) => Math.round(v * 100) / 100;
const monthWord = (n) => (n === 1 ? 'month' : 'months');
function textOf(b) {
  if (b.state !== 'tested') return b.why || NOT_TESTED;
  const random = b.random === null ? '' : `; random tickets: ${b.random.toFixed(1)} on average`;
  return `profit in ${b.inProfit} of ${b.months} backtested ${monthWord(b.months)}${random}${b.changed ? '; settings changed since' : ''}`;
}
function detailOf(b, what) {
  if (b.state !== 'tested') return `${what} has no backtest record yet. The record covers the rows a backtest can replay: the statistical and the boosting rows.`;
  return [
    `Backtest over ${b.months} calendar ${monthWord(b.months)} between ${b.from} and ${b.to}: ${what} played on every draw, each draw's ticket made from the history before it.`,
    `Staked ${b.stake.toFixed(2)} EUR, result ${b.profit.toFixed(2)} EUR.`,
    b.unscored ? `${b.unscored} ${b.unscored === 1 ? 'draw' : 'draws'} in those months could not be played (no prize table for the draw, or no playable ticket).` : null,
    b.random === null ? null : `Random tickets played the same way end ${b.random.toFixed(1)} of those ${monthWord(b.months)} in profit on average.`,
    b.changed ? 'Its settings have changed since this backtest; it is backtested again at the weekend.' : null,
    'The settings were tuned with these months in view, so the count flatters a tuned row. A count, not proof.',
  ].filter(Boolean).join(' ');
}
// brief: the few words beside a ticket on Home ("backtest 3 of 10"); text: the line; detail: the tooltip
const briefOf = (b) => (b.state === 'tested' ? `backtest ${b.inProfit} of ${b.months}` : NOT_TESTED);
const finish = (b, what) => ({ ...b, brief: briefOf(b), text: textOf(b), detail: detailOf(b, what) });
const none = (why, what) => finish({ state: 'none', why }, what);

function rowOf(record, name, served) {
  const entry = record && record.rows[name];
  if (!entry || !entry.months) return none(NOT_TESTED, 'This row');
  const months = playedMonths(record, entry);
  if (!months.length) return none(NOT_TESTED, 'This row');
  const sum = (key) => round2(months.reduce((a, m) => a + (Number(entry.months[m][key]) || 0), 0));
  return finish({
    state: 'tested', inProfit: months.filter((m) => entry.months[m].profit > 0).length, months: months.length, of: record.months.length,
    random: randomCount(record, [Array.isArray(entry.plays) && entry.plays.length ? entry.plays : ['ticket']], months),
    from: months[0], to: months[months.length - 1], stake: sum('stake'), profit: sum('profit'), unscored: sum('unscored'),
    changed: !servedNow(entry, served), testedAt: entry.testedAt || null,
  }, 'this row\'s ticket');
}
function combinationOf(record, members, served) {
  const names = (members || []).map(String);
  if (!record || !names.length) return none(NOT_TESTED, 'This combination');
  const entries = names.map((name) => record.rows[name]);
  const missing = names.filter((name, i) => !entries[i] || !entries[i].months);
  if (missing.length) {
    return none(`${NOT_TESTED} (${missing.length} of ${names.length} ${names.length === 1 ? 'member has' : 'members have'} no backtest)`, 'This combination');
  }
  // the months every member has a played draw in
  const months = record.months.filter((m) => entries.every((e) => e.months[m] && e.months[m].draws > 0));
  if (!months.length) return none(NOT_TESTED, 'This combination');
  const total = (m, key) => entries.reduce((a, e) => a + (Number(e.months[m][key]) || 0), 0);
  return finish({
    state: 'tested', inProfit: months.filter((m) => total(m, 'profit') > 0).length, months: months.length, of: record.months.length,
    random: randomCount(record, entries.map((e) => (Array.isArray(e.plays) && e.plays.length ? e.plays : ['ticket'])), months),
    from: months[0], to: months[months.length - 1],
    stake: round2(months.reduce((a, m) => a + total(m, 'stake'), 0)), profit: round2(months.reduce((a, m) => a + total(m, 'profit'), 0)),
    unscored: Math.max(...entries.map((e) => months.reduce((a, m) => a + (Number(e.months[m].unscored) || 0), 0))),
    changed: entries.some((e) => !servedNow(e, served)), testedAt: entries.map((e) => e.testedAt || '').sort()[0] || null,
  }, 'every member\'s ticket');
}

// One game's reader: row(name), combination(members). The served settings are read once per call of this function.
function forGame(dirs, game) {
  const record = load(dirs, game);
  let served = null;
  if (record) {
    try {
      const file = params.readParams(dirs.root, game);
      served = params.served(file.params || {}, params.gameOverrides(dirs.root, game).overrides);
    } catch (e) { served = null; }
  }
  return {
    months: record ? record.months : [],
    row: (name) => rowOf(record, String(name), served),
    combination: (members) => combinationOf(record, members, served),
  };
}

module.exports = { VERSION, NOT_TESTED, load, stampOf, forGame, rowOf, combinationOf, randomCount, servedNow, textOf };
