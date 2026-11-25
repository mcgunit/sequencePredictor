'use strict';
// standing.js (roadmap item 16, 23 Nov 2026): how long the best model, the top 5 and the best combination have held
// their place, counted over the rankings saved before each draw and today's report for the coming draw.
const fs = require('fs');
const os = require('os');
const path = require('path');
const rankhistory = require('../rankhistory');
const rankings = require('../ranking');
const standing = require('../standing');

let checks = 0;
function ok(cond, what) { checks += 1; if (!cond) { console.error(`FAILED: ${what}`); process.exit(1); } }

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'standing-'));
try {
  const dirs = { root, databaseDir: path.join(root, 'data', 'database') };
  const dir = path.join(dirs.databaseDir, 'lotto');
  fs.mkdirSync(dir, { recursive: true });
  const days = ['2026-11-04', '2026-11-07', '2026-11-11', '2026-11-14', '2026-11-18'];
  days.forEach((d) => fs.writeFileSync(path.join(dir, `${d}.json`), '{}'));
  // a report entry: models in ranking order, each with its live profit; the best combination's members
  const info = (order, profits, members) => ({ metric: 'profit_bound', rankedOn: 'live', live: { minDrawsForRanking: 2 },
    models: order.map((name) => ({ name, live: { draws: 5, profit_total: profits[name] ?? -5, profit_bound: 0 } })),
    combinations: members ? { best: { members } } : undefined });
  const six = ['A', 'B', 'C', 'D', 'E', 'F'];
  // the version saved on the morning of each draw; none before the first draw (it is not counted)
  const version = (day, order, profits, members) => {
    const reduced = rankhistory.reduce(JSON.stringify({ generatedAt: `${day}T06:00:00`, games: { lotto: info(order, profits, members) } }));
    return { sha: day, at: Date.parse(`${day}T06:30:00Z`), generatedAt: reduced.generatedAt, games: reduced.games };
  };
  const hist = rankhistory.makeHistory([
    version('2026-11-07', six, { A: 3 }, ['A', 'B']),
    version('2026-11-11', six, { B: 4 }, ['B', 'A']),                       // the same combination, another order
    version('2026-11-14', ['F', 'A', 'B', 'C', 'D', 'E'], { A: 3 }, ['A', 'C']),
    version('2026-11-18', ['F', 'A', 'B', 'C', 'D', 'E'], { A: 6 }, ['A', 'C']),
  ]);
  const reduced = hist.rankingAt('lotto', Date.parse('2026-11-11T13:00:00Z'));
  ok(reduced.best === 'B' && reduced.combo.join() === 'A,B', `a saved ranking keeps its best model and its combination, sorted: ${JSON.stringify([reduced.best, reduced.combo])}`);
  const report = { games: { lotto: info(six, { A: 7 }, ['C', 'A']) } };
  const st = standing.forGame(dirs, 'lotto', hist, report);
  // counted: 7, 11, 14, 18 Nov and the coming draw; 4 Nov has no ranking saved before it
  ok(st.draws === 5 && st.from === '2026-11-07', `the draws counted: ${st.draws} from ${st.from}`);
  const a = st.best('A');
  ok(a.state === 'tracked' && a.streak === 3 && a.total === 4 && a.of === 5 && a.since === '2026-11-14' && a.score === 80, `the best model: ${JSON.stringify(a)}`);
  ok(a.text === 'best model for 3 draws in a row (since the draw of 2026-11-14); in that place on 4 of the 5 draws checked (80%)' && a.brief === '80% of 5 draws, 3 in a row', a.text);
  ok(st.best('B').state === 'none' && st.best('B').text === standing.NOT_TRACKED, 'a row that is not the best today has no count');
  const e = st.top('E');
  ok(e.streak === 1 && e.total === 3 && e.since === null && e.score === 60 && /^in the top 5 for 1 draw in a row; in that place on 3 of the 5 /.test(e.text), `back in the top 5 today: ${JSON.stringify(e)}`);
  ok(st.top('F').state === 'none' && st.top('A').streak === 5 && st.top('A').score === 100, 'the top 5: the first five of each saved ranking');
  const c = st.combination(['C', 'A']);
  ok(c.streak === 3 && c.total === 3 && c.since === '2026-11-14' && st.combination(['A', 'B']).state === 'none', `a combination is its members in any order: ${JSON.stringify(c)}`);
  const ch = st.changes();
  ok(ch.best.length === 3 && ch.best[0].name === 'A' && ch.best[0].from === '2026-11-14' && ch.best[0].to === null && ch.best[0].draws === 3
    && ch.best[1].name === 'B' && ch.best[1].draws === 1 && ch.best[2].name === 'A' && ch.best[2].from === '2026-11-07', `the best model's runs, newest first: ${JSON.stringify(ch.best)}`);
  ok(ch.combination.length === 2 && ch.combination[1].draws === 2 && ch.combination[1].members.join() === 'A,B' && ch.combination[1].to === '2026-11-11', `the combination's runs: ${JSON.stringify(ch.combination)}`);
  // the five the page shows today count for the coming draw, when given
  ok(standing.forGame(dirs, 'lotto', hist, report, ['F', 'A', 'B', 'C', 'D']).top('F').streak === 3, 'the rows shown today decide the coming draw');
  // no saved ranking at all: today's report alone; no report: nothing
  const bare = standing.forGame(dirs, 'lotto', rankhistory.makeHistory([]), report);
  ok(bare.draws === 1 && bare.best('A').streak === 1 && bare.best('A').since === null, 'without a saved ranking the coming draw is the only one');
  ok(standing.forGame(dirs, 'lotto', hist, null).draws === 4 && standing.forGame(dirs, 'keno', hist, null).best('A').state === 'none', 'no report, no game: no count');
  // a report saved before the live record: the first ranked row with enough draws is its best model
  const old = rankings.bestOf({ minDrawsForRanking: 3, models: [{ name: 'X', draws: 2 }, { name: 'Y · Sampled', draws: 9 }, { name: 'Z', draws: 9 }] });
  ok(old && old.name === 'Z' && rankings.bestOf({ rankedOn: 'live', models: [{ name: 'X', draws: 9 }] }) === null && rankings.comboOf({}) === null, 'the best model of an old report');
  const variant = rankings.bestOf({ rankedOn: 'live', live: { minDrawsForRanking: 1 }, models: [{ name: 'X · Memorise', live: { draws: 9, profit_total: 50 } }, { name: 'Y', live: { draws: 9, profit_total: 2 } }] });
  ok(variant && variant.name === 'Y', 'a variant row is never the best model');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
console.log(`standing: ${checks} checks passed`);
