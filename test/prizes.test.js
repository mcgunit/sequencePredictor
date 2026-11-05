'use strict';
// The History pages' profit for lotto, EuroMillions, EuroDreams (draws.js PRIZE_GAMES, 4 Nov 2026) and Vikinglotto (5 Nov): each draw's own
// prize table as the pipeline keeps it (src/PrizeTables.py), one grid of a row's ticket. Here, on a temp data folder:
// the table read and re-read when it changes, the tier a ticket reaches (lotto's bonus among the six, a tier nobody won
// valued at the jackpot or the tier's median, EuroDreams' annuity), no profit for a draw without a table or a ticket that
// is not a full grid, the day and month figures keeping the best ticket with the profit next to it, and the shape flag.
// The pipeline's own scorer agrees ticket for ticket on production's files (checked by hand on 4 Nov 2026: 3,639
// tickets, no difference).
//   node test/prizes.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const draws = require('../draws');

let checks = 0;
const eq = (a, b, msg) => { assert.deepStrictEqual(a, b, msg); checks += 1; };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'prizes-'));
const databaseDir = path.join(tmp, 'data', 'database');
const write = (game, table) => {
  const dir = path.join(tmp, 'data', 'trainingData', game);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, `${game}-prizes.json`), JSON.stringify(table));
};
const tier = (name, mains, special, amount) => ({ name, mains, special, amount, winners: 1, type: 'PARAMUTUEL' });
try {
  fs.mkdirSync(databaseDir, { recursive: true });
  eq(draws.prizesOf(databaseDir, 'lotto'), null, 'no file yet: no tables');
  eq(draws.prizesOf(databaseDir, 'keno'), null, 'keno has a payout table, not prize tables');
  write('lotto', {
    '2026-11-04': { price: 1.5, jackpot: 1500000, tiers: [tier('  6', 6, null, 0), tier('5+1', 5, 1, 0), tier('  3', 3, null, 7.5), tier('1+1', 1, 1, 1.5)] },
    '2026-10-31': { price: 1.5, jackpot: 1000000, tiers: [tier('  6', 6, null, 0), tier('5+1', 5, 1, 60000), tier('  3', 3, null, 8.5), tier('1+1', 1, 1, 1.5)] },
    '2026-10-28': { price: null, jackpot: null, tiers: [], none: 'no draw in the API' },
  });
  const lotto = draws.prizesOf(databaseDir, 'lotto');
  const real = [1, 2, 3, 4, 5, 6, 7];
  const split = draws.splitRealResult(real, 'lotto');
  const prize = draws.prizeOfDraw(lotto, '2026-11-04');
  eq(draws.prizeTicketProfit('lotto', [1, 2, 3, 40, 41, 42], split, prize), 6, '3 hits: 7.50 - 1.50');
  eq(draws.prizeTicketProfit('lotto', [1, 40, 41, 42, 43, 7], split, prize), 0, '1+1: the bonus among the six, the stake back');
  eq(draws.prizeTicketProfit('lotto', [10, 40, 41, 42, 43, 44], split, prize), -1.5, 'no tier');
  eq(draws.prizeTicketProfit('lotto', [1, 2, 3, 4, 5, 6], split, prize), 1500000 - 1.5, 'nobody won the top tier: the jackpot');
  eq(draws.prizeTicketProfit('lotto', [1, 2, 3, 4, 5, 7], split, prize), 60000 - 1.5, 'nobody won 5+1: its median where it paid');
  eq(draws.prizeTicketProfit('lotto', [1, 2, 3], split, prize), null, 'not a full grid');
  eq(draws.prizeOfDraw(lotto, '2026-10-28'), null, 'a draw the API has no table for');
  eq(draws.prizeOfDraw(lotto, '2026-11-01'), null, 'a draw not in the file');
  eq(draws.prizeTicketProfit('lotto', [1, 2, 3, 40, 41, 42], split, null), null, 'no table: no profit');

  write('euromillions', { '2026-11-03': { price: 2.5, jackpot: 17000000, tiers: [tier('5+2', 5, 2, 0), tier('2+1', 2, 1, 8.1), tier('2+0', 2, 0, 4)] } });
  const em = draws.prizeOfDraw(draws.prizesOf(databaseDir, 'euromillions'), '2026-11-03');
  const emSplit = draws.splitRealResult([1, 2, 3, 4, 5, 6, 7], 'euromillions');
  eq(draws.prizeTicketProfit('euromillions', [1, 2, 30, 31, 32, 6, 9], emSplit, em), 5.6, '2 + 1 star');
  eq(draws.prizeTicketProfit('euromillions', [1, 2, 30, 31, 32], emSplit, em), null, 'an RL ticket without its stars: hits only');

  write('eurodreams', { '2026-11-02': { price: 2.5, jackpot: null, tiers: [tier('6+1', 6, 1, 0), tier('6+0', 6, 0, 0), tier('2/', 2, null, 2.5)] } });
  const ed = draws.prizeOfDraw(draws.prizesOf(databaseDir, 'eurodreams'), '2026-11-02');
  const edSplit = draws.splitRealResult([1, 2, 3, 4, 5, 6, 3], 'eurodreams');
  eq(draws.prizeTicketProfit('eurodreams', [1, 2, 3, 4, 5, 6, 3], edSplit, ed), 7200000 - 2.5, 'the annuity at its rule value');
  eq(draws.prizeTicketProfit('eurodreams', [1, 2, 30, 31, 32, 33, 5], edSplit, ed), 0, '2 of any dream: the stake back');

  // the day: the best ticket stays, the profit over the grids it could score is added; the month sums them
  const day = { realResult: real, currentPrediction: [
    { name: 'A', predictions: [[1, 2, 3, 40, 41, 42]] }, { name: 'B', predictions: [[10, 40, 41, 42, 43, 44]] }, { name: 'C', predictions: [[1, 2]] }] };
  const f = draws.dayFigures('lotto', day, () => true, prize);
  eq([f.profit, f.stake, f.grids, f.best.mains], [4.5, 3, 2, 3], 'day: two grids scored, the short row has none, best stays');
  const none = draws.dayFigures('lotto', day, () => true, null);
  eq([none.profit, none.stake, none.grids, none.best.mains], [null, null, 0, 3], 'day without a table: hits only');
  const months = draws.historyMonths('lotto', [
    { date: '2026-11-04', day, keep: () => true, prize },
    { date: '2026-10-28', day, keep: () => true, prize: null },
    { date: '2026-10-25', day, keep: () => true, prize: null }]);
  eq([months[0].profit, months[0].grids, months[1].profit, months[1].stake, months[1].grids], [4.5, 2, null, null, 0], 'months: summed where a table was');
  const row = draws.scoreRow('lotto', day.currentPrediction[0], real, split, split.mains, prize);
  eq([row.tickets[0].profit, row.tickets[0].stake], [6, 1.5], 'a ticket of the draw table');
  eq([draws.scoreRow('lotto', day.currentPrediction[2], real, split, split.mains, prize).tickets[0].stake], [null], 'not a grid: no stake');
  eq([draws.shapeOf('lotto', { prizes: true }).prizes, draws.shapeOf('lotto', { prizes: true }).money, draws.shapeOf('lotto').prizes], [true, false, undefined], 'shape');
  eq(draws.shapeOf('keno', { prizes: true }).prizes, undefined, 'keno has its payout table');

  // Vikinglotto (5 Nov 2026): 'k' without the viking number, 'k+Fav' with it, '6/' whatever it is; one number pays 2.00
  write('vikinglotto', { '2026-11-04': { price: 2, jackpot: 3000000, tiers: [tier('6/', 6, null, 0), tier('5+Fav', 5, 1, 0),
    tier('3+Fav', 3, 1, 60), tier('3', 3, 0, 30), tier('1+Fav', 1, 1, 4), tier('1', 1, 0, 2)] } });
  const vk = draws.prizeOfDraw(draws.prizesOf(databaseDir, 'vikinglotto'), '2026-11-04');
  const vkSplit = draws.splitRealResult([11, 18, 23, 36, 42, 44, 5], 'vikinglotto');
  eq(draws.prizeTicketProfit('vikinglotto', [11, 1, 2, 3, 4, 6, 1], vkSplit, vk), 0, 'one number: the price back');
  eq(draws.prizeTicketProfit('vikinglotto', [11, 18, 23, 3, 4, 6, 5], vkSplit, vk), 58, '3+Fav');
  eq(draws.prizeTicketProfit('vikinglotto', [11, 18, 23, 36, 42, 44, 1], vkSplit, vk), 3000000 - 2, '6/: the jackpot, any viking');
  eq(draws.prizeTicketProfit('vikinglotto', [1, 9, 35, 34, 36, 48], vkSplit, vk), null, 'an RL ticket without its viking: hits only');
  eq(draws.shapeOf('vikinglotto', { prizes: true }).prizes, true, 'vikinglotto in euro');

  // a table that changes is read again
  const before = draws.prizeStamp(databaseDir, 'lotto');
  write('lotto', { '2026-11-04': { price: 1.5, jackpot: 1, tiers: [tier('  6', 6, null, 0), tier('  3', 3, null, 9.5)] } });
  fs.utimesSync(path.join(tmp, 'data', 'trainingData', 'lotto', 'lotto-prizes.json'), new Date(), new Date(Date.now() + 5000));
  eq(draws.prizeStamp(databaseDir, 'lotto') !== before, true, 'the stamp follows the file');
  eq(draws.prizeTicketProfit('lotto', [1, 2, 3, 40, 41, 42], split, draws.prizeOfDraw(draws.prizesOf(databaseDir, 'lotto'), '2026-11-04')), 8, 're-read');
  console.log(`prizes: ${checks} checks passed`);
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
