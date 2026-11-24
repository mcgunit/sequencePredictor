'use strict';
// The phase-shift page as a "what if" (roadmap item 17, 24 Nov 2026): what a row's tickets would have paid had each
// been played on the draw at the row's peak lag instead of on the draw it was made for. Shown on the administrators'
// Phase-shift page only: no ranking, ticket or plan reads it.
//
//   the lag used      for every draw the peak lag the row had in the newest run recorded BEFORE that draw's day
//                     (data/database/lagPeakHistory.json, one entry per run): never a lag chosen with the draw in
//                     view. A draw before the row's first recorded run is not counted.
//   what if           the ticket the row made lag - 1 draws earlier, scored on this draw with the money rules of the
//                     History pages (draws.dayFigures: the payout tables, or the draw's own prize table).
//   as played         the row's own ticket for the draw (lag +1), over the same draws.
//   any lag           the average over every lag from +1 to +30 the day files can give for the draw: what a lag
//                     picked at random pays. A peak lag has this to beat, not only the row as played.
//   counted           a draw where the row has a ticket at lag +1 and at the lag used, and both could be scored
//                     in euro (a prize-table game before its first table cannot).
//
// The peak is found on hits, the what-if is in euro, and nothing here says what the difference reaches by chance.
const fs = require('fs');
const path = require('path');
const draws = require('./draws');
const workbench = require('./workbench');

const LAG_MAX = 30;
const GAMES = ['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'];   // the games counted in euro
const HISTORY_FILE = 'lagPeakHistory.json';
const round2 = (v) => Math.round(v * 100) / 100;
const cache = new Map();   // databaseDir|game -> {stamp, value}
const stampOf = (file) => { try { const s = fs.statSync(file); return `${s.mtimeMs}:${s.size}`; } catch (e) { return '-'; } };

function historyOf(databaseDir, game) {
  try {
    const all = JSON.parse(fs.readFileSync(path.join(databaseDir, HISTORY_FILE), 'utf-8'));
    return all && all[game] && typeof all[game] === 'object' ? all[game] : {};
  } catch (e) { return {}; }
}
// the peak lag of the newest run recorded before `date` (runs: oldest first), or null
function lagBefore(runs, date) {
  let lag = null;
  for (const r of runs) {
    if (!r || typeof r.run !== 'string' || r.run >= date) break;
    if (Number.isInteger(r.lag) && r.lag >= 1 && r.lag <= LAG_MAX) lag = r.lag;
  }
  return lag;
}
// One game: {game, lagMax, rows: [...], total}. days: oldest first [{date, realResult, rows: Map(name -> row)}]
function compute(game, days, history, prizeOf) {
  const rows = [];
  Object.keys(history).forEach((name) => {
    const runs = (Array.isArray(history[name]) ? history[name] : []).slice().sort((a, b) => String(a.run).localeCompare(String(b.run)));
    const sum = { draws: 0, stake: 0, played: 0, whatIf: 0, anyLag: 0, better: 0, from: null, to: null };
    const lags = new Map();
    days.forEach((day, i) => {
      if (!day.realResult.length) return;
      const lag = lagBefore(runs, day.date);
      if (lag === null) return;
      const prize = prizeOf(day.date);
      // the money of the ticket the row made `k` draws earlier, on this draw: {profit, stake} or null
      const at = (k) => {
        const src = days[i - k];
        const row = src ? src.rows.get(name) : null;
        if (!row) return null;
        const f = draws.dayFigures(game, { realResult: day.realResult, currentPrediction: [row] }, undefined, prize);
        return typeof f.profit === 'number' && typeof f.stake === 'number' && f.stake > 0 ? f : null;
      };
      const played = at(0);
      const shifted = lag === 1 ? played : at(lag - 1);
      if (!played || !shifted) return;
      let any = 0; let n = 0;
      for (let k = 0; k < LAG_MAX; k += 1) {
        const f = k === 0 ? played : (k === lag - 1 ? shifted : at(k));
        if (f) { any += f.profit; n += 1; }
      }
      sum.draws += 1; sum.stake += shifted.stake; sum.played += played.profit; sum.whatIf += shifted.profit; sum.anyLag += any / n;
      if (shifted.profit > played.profit) sum.better += 1;
      sum.from = sum.from || day.date; sum.to = day.date;
      lags.set(lag, (lags.get(lag) || 0) + 1);
    });
    if (!sum.draws) return;
    const newest = runs.length ? runs[runs.length - 1] : null;
    rows.push({
      row: name, draws: sum.draws, from: sum.from, to: sum.to, stake: round2(sum.stake), played: round2(sum.played), whatIf: round2(sum.whatIf), anyLag: round2(sum.anyLag),
      gain: round2(sum.whatIf - sum.played), overAnyLag: round2(sum.whatIf - sum.anyLag), betterDraws: sum.better,
      lagNow: newest && Number.isInteger(newest.lag) ? newest.lag : null,
      lags: [...lags.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 4).map(([lag, n]) => ({ lag, draws: n })), lagsUsed: lags.size,
    });
  });
  const add = (key) => round2(rows.reduce((a, r) => a + r[key], 0));
  const total = rows.length ? { rows: rows.length, draws: rows.reduce((a, r) => a + r.draws, 0), stake: add('stake'), played: add('played'), whatIf: add('whatIf'), anyLag: add('anyLag'),
    gain: add('gain'), overAnyLag: add('overAnyLag'), aheadOfPlayed: rows.filter((r) => r.whatIf > r.played).length, aheadOfAnyLag: rows.filter((r) => r.whatIf > r.anyLag).length } : null;
  return { game, lagMax: LAG_MAX, rows, total };
}
// A game's what-if from its day files, remembered until a day file, the peak history or the prize tables change
function forGame(dirs, game) {
  if (!GAMES.includes(game)) return null;
  const dir = path.join(dirs.databaseDir, game);
  const files = workbench.dayFilesOf(dir).slice().reverse();
  if (!files.length) return null;
  const newest = files[files.length - 1];
  const stamp = [files.length, newest.file, stampOf(path.join(dir, newest.file)), stampOf(path.join(dirs.databaseDir, HISTORY_FILE)), draws.prizeStamp(dirs.databaseDir, game)].join('|');
  const key = `${dirs.databaseDir}|${game}`;
  const hit = cache.get(key);
  if (hit && hit.stamp === stamp) return hit.value;
  const history = historyOf(dirs.databaseDir, game);
  const days = files.map((f) => {
    const d = workbench.readDay(path.join(dir, f.file)) || { realResult: [], currentPrediction: [] };
    return { date: f.date, realResult: d.realResult, rows: new Map(d.currentPrediction.map((m) => [String(m.name), m])) };
  });
  const prizes = draws.PRIZE_GAMES.includes(game) ? draws.prizesOf(dirs.databaseDir, game) : null;
  const value = compute(game, days, history, (date) => (prizes ? draws.prizeOfDraw(prizes, date) : null));
  cache.set(key, { stamp, value });
  return value;
}

module.exports = { GAMES, LAG_MAX, lagBefore, compute, forGame };
