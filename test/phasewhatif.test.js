'use strict';
// phasewhatif.js (roadmap item 17, 24 Nov 2026): a row's tickets played at its peak lag, in euro, on hand-made Pick3
// days. Pick3 pays a straight hit 500 for 1 EUR among its bets; the amounts are draws.js's own, compared between lags.
const whatif = require('../phasewhatif');
const draws = require('../draws');

let checks = 0;
function ok(cond, what) { checks += 1; if (!cond) { console.error(`FAILED: ${what}`); process.exit(1); } }

ok(whatif.lagBefore([{ run: '2026-11-01', lag: 3 }, { run: '2026-11-05', lag: 2 }], '2026-11-05') === 3
  && whatif.lagBefore([{ run: '2026-11-01', lag: 3 }, { run: '2026-11-05', lag: 2 }], '2026-11-06') === 2
  && whatif.lagBefore([{ run: '2026-11-01', lag: 3 }], '2026-11-01') === null && whatif.lagBefore([{ run: '2026-11-01', lag: 99 }], '2026-11-09') === null,
  'the lag of the newest run recorded before the draw\'s day, never of that day or later');

// six draws; row A's ticket of each day is the result two draws later: at lag +3 it wins every time, as played never
const results = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]];
const dates = ['2026-11-01', '2026-11-02', '2026-11-03', '2026-11-04', '2026-11-05', '2026-11-06'];
const row = (name, ticket) => ({ name, predictions: [ticket] });
const days = dates.map((date, i) => ({ date, realResult: results[i],
  rows: new Map([['A', row('A', results[i + 2] || [9, 9, 9])], ['B', row('B', [0, 0, 0])]]) }));
const one = (ticket, real) => draws.dayFigures('pick3', { realResult: real, currentPrediction: [row('A', ticket)] });
const win = one([3, 3, 3], [3, 3, 3]);
const lose = one([5, 5, 5], [3, 3, 3]);
ok(win.profit > 0 && lose.profit < 0 && win.stake === lose.stake, `a straight hit pays, a miss costs the stake: ${JSON.stringify([win, lose])}`);

const history = { A: [{ run: '2026-11-03', lag: 3 }, { run: '2026-11-05', lag: 1 }], B: [{ run: '2026-11-03', lag: 2 }], C: [{ run: '2026-11-03', lag: 2 }] };
const v = whatif.compute('pick3', days, history, () => null);
const a = v.rows.find((r) => r.row === 'A');
// A: draws of 4 and 5 Nov use lag +3 (the run of 3 Nov), 6 Nov lag +1 (the run of 5 Nov); 1-3 Nov have no run before them
ok(a && a.draws === 3 && a.from === '2026-11-04' && a.to === '2026-11-06' && a.lagsUsed === 2 && a.lags[0].lag === 3 && a.lags[0].draws === 2 && a.lagNow === 1, `the draws counted and the lags used: ${JSON.stringify(a)}`);
ok(a.played === Math.round(3 * lose.profit * 100) / 100 && a.whatIf === Math.round((2 * win.profit + lose.profit) * 100) / 100 && a.stake === 3 * win.stake
  && a.gain === Math.round((a.whatIf - a.played) * 100) / 100 && a.betterDraws === 2, `as played against the what-if: ${JSON.stringify(a)}`);
// any lag: on 4 Nov four lags can be read (+1..+4), one of them wins; on 5 Nov five, one wins; on 6 Nov six, one wins
const any = (win.profit + 3 * lose.profit) / 4 + (win.profit + 4 * lose.profit) / 5 + (win.profit + 5 * lose.profit) / 6;
ok(Math.abs(a.anyLag - any) < 0.011 && a.overAnyLag === Math.round((a.whatIf - a.anyLag) * 100) / 100 && a.whatIf > a.anyLag, `a lag picked at random: ${a.anyLag} against ${any}`);
const b = v.rows.find((r) => r.row === 'B');
ok(b && b.draws === 3 && b.played === b.whatIf && b.gain === 0 && b.betterDraws === 0 && !v.rows.some((r) => r.row === 'C'), 'a row that never hits gains nothing; a row without tickets is left out');
ok(v.total.rows === 2 && v.total.draws === 6 && v.total.aheadOfPlayed === 1 && v.total.aheadOfAnyLag === 1 && v.total.whatIf === Math.round((a.whatIf + b.whatIf) * 100) / 100, `the totals: ${JSON.stringify(v.total)}`);
ok(whatif.compute('pick3', days, {}, () => null).total === null && whatif.forGame({ databaseDir: '/nonexistent' }, 'crypto') === null && whatif.forGame({ databaseDir: '/nonexistent' }, 'pick3') === null,
  'no recorded peak, a market game, no day files: nothing');
console.log(`phasewhatif: ${checks} checks passed`);
