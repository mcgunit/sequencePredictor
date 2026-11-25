'use strict';
// How long the best model, the top 5 and the best combination have held their place (roadmap item 16, 23 Nov 2026).
// Nothing is computed about a model here: the saved rankings are read (rankhistory.js: one version per run, in git
// since 25 Aug 2026) and counted.
//
//   a draw's standing   the ranking saved before the draw (rankhistory's cut-off: no hindsight): its best model as
//                       Home names it (rankings.bestOf), the first five of its ranking, its best combination. A
//                       draw without a saved ranking before it is not counted.
//   the coming draw     today's report: the best model, the five rows the pages show, the best combination. It is
//                       the last draw of every count.
//   in a row            the draws up to and including the coming one over which the place was held without a
//                       break; `since` is the first of them.
//   the score           the share of the counted draws on which the place was held, in percent.
//   a combination       the same one when its members are the same, in any order.
//   the changes         the best model's and the best combination's runs, newest first: who held the place over
//                       which draws.
const path = require('path');
const rankings = require('./ranking');
const workbench = require('./workbench');

const CHANGES_SHOWN = 8;
const NOT_TRACKED = 'not tracked yet';
const key = (members) => (Array.isArray(members) && members.length ? members.map(String).slice().sort().join('\u0001') : null);
const drawWord = (n) => (n === 1 ? 'draw' : 'draws');

// The standings of a game, oldest first, the coming draw last: [{date, best, top: Set, combo}]
function statesOf(dirs, game, hist, report, tops = null) {
  const files = workbench.dayFilesOf(path.join(dirs.databaseDir, game)).slice().reverse();
  const states = [];
  files.forEach((f) => {
    const cutoff = hist.cutoffOf(game, f.date);
    const at = hist.rankingAt(game, cutoff === null ? NaN : cutoff);
    if (at.none) return;
    states.push({ date: f.date, best: at.best || null, top: new Set(rankings.topNames({ names: at.names })), combo: key(at.combo), members: at.combo || [] });
  });
  const info = report && report.games ? report.games[game] : null;
  if (info) {
    const best = rankings.bestOf(info);
    const members = rankings.comboOf(info) || [];
    states.push({ date: null, best: best ? best.name : null, top: new Set(Array.isArray(tops) ? tops.map(String) : rankings.usersTop(report, game, null)), combo: key(members), members });
  }
  return states;
}
function measure(states, holds, place) {
  if (!states.length || !holds(states[states.length - 1])) return { state: 'none', brief: NOT_TRACKED, text: NOT_TRACKED };
  let streak = 0;
  for (let i = states.length - 1; i >= 0 && holds(states[i]); i -= 1) streak += 1;
  const total = states.filter(holds).length;
  const first = states[states.length - streak];
  const since = first.date;   // null: the coming draw is the first
  const score = Math.round((total / states.length) * 100);
  const sinceText = since ? ` (since the draw of ${since})` : '';
  return {
    state: 'tracked', streak, total, of: states.length, since, score, from: states[0].date,
    brief: `${score}% of ${states.length} draws, ${streak} in a row`,
    text: `${place} for ${streak} ${drawWord(streak)} in a row${sinceText}; in that place on ${total} of the ${states.length} draws checked (${score}%)`,
  };
}
// the runs of one place, newest first: [{name | members, from, to, draws}]; to null = up to the coming draw
function changesOf(states, of, label) {
  const runs = [];
  states.forEach((s) => {
    const k = of(s);
    const last = runs[runs.length - 1];
    if (last && last.k === k) { last.to = s.date; last.draws += 1; return; }
    runs.push({ k, ...label(s), from: s.date, to: s.date, draws: 1 });
  });
  return runs.filter((r) => r.k !== null).reverse().slice(0, CHANGES_SHOWN).map(({ k, ...r }) => r);
}

// One game's reader. tops: the names of the five rows the pages show for the coming draw (the ranking's first five without).
function forGame(dirs, game, hist, report, tops = null) {
  const states = statesOf(dirs, game, hist, report, tops);
  return {
    draws: states.length, from: states.length ? states[0].date : null,
    // what the counts are, once per game (the tooltip of every line)
    note: `Counted over the rankings saved before each draw${states.length && states[0].date ? ` since ${states[0].date}` : ''}, the coming draw included. Score: the share of those draws in this place. `
      + 'It says nothing about the next draw.',
    best: (name) => measure(states, (s) => s.best === String(name), 'best model'),
    top: (name) => measure(states, (s) => s.top.has(String(name)), `in the top ${rankings.TOP_MODELS_SHOWN}`),
    combination: (members) => { const k = key(members); return measure(states, (s) => k !== null && s.combo === k, 'best combination'); },
    changes: () => ({
      best: changesOf(states, (s) => s.best, (s) => ({ name: s.best })),
      combination: changesOf(states, (s) => s.combo, (s) => ({ members: s.members })),
    }),
  };
}

module.exports = { NOT_TRACKED, CHANGES_SHOWN, statesOf, measure, changesOf, forGame };
