'use strict';
// The backtest record on the pages (backtestRecord.js, roadmap item 13): a row's and a combination's count from a
// record file, "not tested yet" without one, the random reference (the same arithmetic as
// src/BacktestMonths.random_count), "settings changed since", and the summary of Home carrying it.
// Run: node test/backtestRecord.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const bt = require('../backtestRecord');
const lottery = require('../lottery');

let checks = 0;
const ok = (cond, what) => { assert.ok(cond, what); checks += 1; };

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'backtest-record-'));
try {
  const dirs = { root, databaseDir: path.join(root, 'data', 'database') };
  fs.mkdirSync(path.join(root, 'data', 'backtestRecord'), { recursive: true });
  const month = (draws, stake, profit) => ({ draws, bets: draws, stake, profit, unscored: 0 });
  const record = {
    version: 1, game: 'pick3', months: ['2026-08', '2026-09', '2026-10'], lockbox: null,
    random: { sims: 4, months: {
      '2026-08': { ticket: [1, -1, -1, -1] }, '2026-09': { ticket: [1, -1, -1, -1] }, '2026-10': { ticket: [1, 1, -1, -1] } } },
    rows: {
      'A Model': { values: { alpha: 0.5, order: 2 }, fromDefault: ['order'], plays: ['ticket'], testedAt: '2026-11-22T10:00:00Z',
        months: { '2026-08': month(31, 124, -60), '2026-09': month(30, 120, 80), '2026-10': month(31, 124, 12.5) } },
      'B Model': { values: { beta: 1 }, fromDefault: [], plays: ['ticket'], testedAt: '2026-11-21T10:00:00Z',
        months: { '2026-08': month(31, 124, 100), '2026-09': month(30, 120, -100), '2026-10': month(0, 0, 0) } },
      'E Model': { error: 'ValueError: no data', errorAt: '2026-11-22T10:00:00Z' },
      'S Model': { error: 'TrialPruned: ', errorAt: '2026-11-27T12:00:00Z', slow: { at: '2026-11-27T12:00:00Z', seconds: 7198, allowed: 7198 }, values: { alpha: 0.5 }, fromDefault: [] },
      'T Model': { error: 'TrialPruned: ', errorAt: '2026-11-27T12:00:00Z', slow: { at: '2026-11-27T12:00:00Z', seconds: 7198, allowed: 7198 }, values: { alpha: 0.9 }, fromDefault: [] },
    },
  };
  const file = path.join(root, 'data', 'backtestRecord', 'pick3.json');
  fs.writeFileSync(file, JSON.stringify(record));
  fs.writeFileSync(path.join(root, 'bestParams_pick3.json'), JSON.stringify({ alpha: 0.5, beta: 1 }));

  const g = bt.forGame(dirs, 'pick3');
  const a = g.row('A Model');
  ok(a.state === 'tested' && a.inProfit === 2 && a.months === 3 && a.of === 3 && a.from === '2026-08' && a.to === '2026-10', `a row's count: ${JSON.stringify(a)}`);
  ok(a.stake === 368 && a.profit === 32.5 && a.changed === false, 'its stake and result over the months, settings unchanged');
  // months in profit over the 4 players: 08 -> 1, 09 -> 1, 10 -> 2 = 4 of 4 players = 1.0
  ok(a.random === 1, `the random reference: ${a.random}`);
  ok(a.text === 'profit in 2 of 3 backtested months; random tickets: 1.0 on average', a.text);
  ok(/A count, not proof\.$/.test(a.detail) && /flatters a tuned row/.test(a.detail) && /Staked 368\.00 EUR, result 32\.50 EUR/.test(a.detail), a.detail);
  const b = g.row('B Model');
  ok(b.months === 2 && b.inProfit === 1 && b.to === '2026-09' && b.random === 0.5, `a month without a played draw is not counted: ${JSON.stringify(b)}`);
  ok(b.text === 'profit in 1 of 2 backtested months; random tickets: 0.5 on average', b.text);
  const gaps = JSON.parse(JSON.stringify(record));
  gaps.rows['A Model'].months['2026-09'].unscored = 2;
  ok(/2 draws in those months could not be played/.test(bt.rowOf(gaps, 'A Model', null).detail) && !/could not be played/.test(a.detail), 'draws that could not be played are said');
  const none = g.row('Z Model');
  ok(none.state === 'none' && none.text === 'not tested yet' && /no backtest record yet/.test(none.detail), 'a row without a record is said, not hidden');
  ok(g.row('E Model').state === 'none' && g.row('A Model · Sampled').state === 'none', 'a failed row and a variant row are not tested yet');
  const slow = g.row('S Model');
  ok(slow.state === 'none' && slow.slow === true && slow.text === 'too slow to backtest' && /longer than the daily job/.test(slow.detail), `a row out of time on its served settings: ${JSON.stringify(slow)}`);
  ok(g.row('T Model').slow === undefined && g.row('T Model').text === 'not tested yet', 'out of time on other settings than the served ones: not tested yet');
  ok(/1 of 2 members have no backtest/.test(g.combination(['A Model', 'S Model']).text), 'a combination with a slow member is not tested yet');

  // a combination: the members' months added up, over the months all of them played
  const c = g.combination(['A Model', 'B Model']);
  ok(c.state === 'tested' && c.months === 2 && c.inProfit === 1 && c.stake === 488 && c.profit === 20, `the combination: ${JSON.stringify(c)}`);
  // players (member B shifted by 53 % 4 = 1): 08: [1-1, -1-1, -1-1, -1+1] -> 0 in profit; 09 the same -> 0
  ok(c.random === 0, `its random reference, two players a draw: ${c.random}`);
  const half = g.combination(['A Model', 'Z Model', 'Y Model']);
  ok(half.state === 'none' && half.text === 'not tested yet (2 of 3 members have no backtest)', half.text);
  ok(g.combination([]).state === 'none', 'no members, no record');

  // settings changed since: a tuned value differs, or a value that was tuned then has none now
  fs.writeFileSync(path.join(root, 'bestParams_pick3.json'), JSON.stringify({ alpha: 0.7, beta: 1 }));
  const changed = bt.forGame(dirs, 'pick3');
  ok(changed.row('A Model').changed === true && changed.row('B Model').changed === false, 'only the row whose setting changed');
  ok(/settings changed since$/.test(changed.row('A Model').text) && /backtested again at the weekend/.test(changed.row('A Model').detail), changed.row('A Model').text);
  ok(changed.combination(['A Model', 'B Model']).changed === true, 'a combination with one changed member');
  ok(bt.servedNow({ values: { a: 1 }, fromDefault: ['a'] }, {}) && !bt.servedNow({ values: { a: 1 }, fromDefault: [] }, {}), 'from the default then and now is no change; tuned then and gone now is');
  fs.mkdirSync(path.join(root, 'config'), { recursive: true });
  fs.writeFileSync(path.join(root, 'config', 'paramOverrides.json'), JSON.stringify({ version: 1, games: { pick3: { overrides: { alpha: { value: 0.5 } } } } }));
  ok(bt.forGame(dirs, 'pick3').row('A Model').changed === false, 'a manual override that restores the backtested value: unchanged');

  // no file, a broken file, another version: not tested yet everywhere
  ok(bt.forGame(dirs, 'lotto').row('A Model').text === 'not tested yet' && bt.forGame(dirs, 'lotto').months.length === 0, 'no file');
  fs.writeFileSync(path.join(root, 'data', 'backtestRecord', 'lotto.json'), '{ broken');
  ok(bt.forGame(dirs, 'lotto').row('A Model').state === 'none', 'a broken file');
  fs.writeFileSync(path.join(root, 'data', 'backtestRecord', 'lotto.json'), JSON.stringify({ ...record, version: 99 }));
  ok(bt.forGame(dirs, 'lotto').row('A Model').state === 'none', 'another version');
  // a Vikinglotto record booked at the old grid price (no money mark) is not shown; one booked the new way is
  fs.writeFileSync(path.join(root, 'data', 'backtestRecord', 'vikinglotto.json'), JSON.stringify(record));
  ok(bt.forGame(dirs, 'vikinglotto').row('A Model').state === 'none', 'vikinglotto booked the old way');
  fs.writeFileSync(path.join(root, 'data', 'backtestRecord', 'vikinglotto.json'), JSON.stringify({ ...record, money: 2, updatedAt: 'x' }));
  ok(bt.forGame(dirs, 'vikinglotto').row('A Model').state === 'tested', 'vikinglotto booked at 10.00 a grid');

  // keno: a row's plays are its ticket sizes, each on players of its own
  const keno = { ...record, game: 'keno', random: { sims: 4, months: { '2026-10': { 5: [3, -1, -1, -1], 6: [-1, -1, 2, -1] } } }, months: ['2026-10'],
    rows: { 'K Model': { values: {}, fromDefault: [], plays: ['5', '6'], months: { '2026-10': { draws: 31, bets: 62, stake: 62, profit: 5, unscored: 0 } } } } };
  // players: size 5 at s, size 6 at s+17 (%4 = s+1): [3-1, -1+2, -1-1, -1-1] -> 2 of 4 in profit
  ok(bt.rowOf(keno, 'K Model', null).random === 0.5 && bt.rowOf(keno, 'K Model', null).inProfit === 1, 'keno: the sizes a row plays, added per player');
  ok(bt.randomCount(keno, [['7']], ['2026-10']) === null && bt.randomCount(keno, [['5']], ['2026-09']) === null, 'no simulation, no reference');

  // Home's summary carries it beside the best model and the best combination
  const live = (name, draws, profit) => ({ name, live: { draws, profit_total: profit, avg_hits: 0.3 } });
  const report = { games: { pick3: { minDrawsForRanking: 10, models: [live('A Model', 50, 120), live('B Model', 40, -20)],
    combinations: { metric: 'profit_per_draw', best: { members: ['A Model', 'B Model'], draws: 36, profit_total: 99 }, control: { p_value: 0.3 } } } } };
  const s = lottery.summaryOf(report, 'pick3', bt.forGame(dirs, 'pick3'));
  ok(s.best.backtest.inProfit === 2 && s.combination.backtest.months === 2 && /^profit in 1 of 2/.test(s.combination.backtest.text), 'Home summary: the count beside both');
  ok(lottery.summaryOf(report, 'pick3').best.backtest === undefined, 'and without a record reader nothing is added (as before)');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
console.log(`backtestRecord: ${checks} checks passed`);
