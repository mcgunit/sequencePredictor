import type { ReactNode } from 'react';
import { MARKET_GAMES, type ControlTiles, type Follows, type Measure, type MissingControl, type Q0Fresh, type Q0Line } from '../../api';
import { DASH, count, day, fixed, isNum, perUnit, short, signedFixed, stamp, weekday } from '../../format';

// Small pieces the admin pages share: the game button that opens a drawer, a group row in a table, and the
// words that put a figure's unit and window on its own line.

export const isMarketGame = (game: string) => MARKET_GAMES.includes(game);

// "draws", "days", "trading days", "weeks" for a count other than one
export const units = (unit: string) => `${perUnit(unit)}s`;

// A row's record: its scored draws in the game's unit and how many of them were live - the ticket saved before the
// draw's sales close (lottery since 27 Oct 2026; a ticket rewritten after the draw counts as it stood at the close,
// recovered) or out before the close (markets). Back-built or late draws never count as record length. Without a
// live record (a report from before, a game not judged) the draws alone.
//   "48 draws · 41 live (8 as saved before the close, rewritten later), 7 back-built or late"
export function recordText(draws: number | null, unit: string, live: number | null | undefined, recovered?: number | null): string {
  const n = count(draws, unit);
  if (!isNum(live) || !isNum(draws)) return n;
  const rec = isNum(recovered) && recovered > 0 ? ` (${recovered} as saved before the close, rewritten later)` : '';
  if (live >= draws) return `${n} · all live${rec}`;
  if (live <= 0) return `${n} · none live (back-built or late)`;
  return `${n} · ${live} live${rec}, ${draws - live} back-built or late`;
}

export function GameButton({ game, onOpen, label }: { game: string; onOpen: (game: string) => void; label?: string }) {
  return (
    <button type="button" className="gbtn" onClick={() => onOpen(game)} aria-haspopup="dialog">
      {label ?? game}
    </button>
  );
}

export function GroupRow({ cols, children }: { cols: number; children: ReactNode }) {
  return (
    <tr className="grp">
      <th colSpan={cols}>{children}</th>
    </tr>
  );
}

// "3 seeds", "3-5 seeds"
export function seedsText(seeds: number[] | undefined): string {
  if (!seeds || !seeds.length) return '';
  const lo = Math.min(...seeds);
  const hi = Math.max(...seeds);
  return `${lo === hi ? lo : `${lo}-${hi}`} seed${hi === 1 ? '' : 's'}`;
}

// The earliest and latest of a set of instants (ISO strings sort as time).
export function earliest(xs: (string | null | undefined)[]): string | null {
  const v = xs.filter((x): x is string => Boolean(x)).sort();
  return v.length ? v[0] : null;
}
export function latest(xs: (string | null | undefined)[]): string | null {
  const v = xs.filter((x): x is string => Boolean(x)).sort();
  return v.length ? v[v.length - 1] : null;
}

// A since-the-freeze value with its unit on the same line: hits as they are, a profit bound with what it is
// measured against (keno's is against a fair ticket, so it never reads as money won).
export const boundWord = (g: { measure: Measure }) => (g.measure.against?.startsWith('a fair ticket') ? 'bound vs a fair ticket' : 'bound EUR/bet');
export function sinceValue(g: { metric: string; measure: Measure }, value: number | null): string {
  return g.metric === 'avg_hits' ? fixed(value) : `${signedFixed(value)} ${boundWord(g)}`;
}

// Whose five rows a game's top five are, in a few words: a lottery game's are the rows its users see today (the
// first five ranked rows in the newest day's table); a market game's users' page follows its best book instead.
export function topWhose(g: { game: string; follows: Follows | null }): string {
  if (!isMarketGame(g.game)) return "the rows users see today";
  const f = g.follows;
  const follows = f ? `; users' ${f.page} page follows ${short(f.row)} (${f.byBook ? `best ${f.book} book: ${f.rule}` : `no ${f.book} book long enough yet`})` : '';
  return `History top 5, not the users' rows${follows}`;
}

// The value of a game's best row with what it measures on the line under it. A profit bound is never shown
// as a win: keno's is measured against a fair ticket, and the raw profit per bet stands next to it.
export function valueCell(g: { metric: string; unit: string; measure: Measure }, value: number | null, profitPerBet: number | null, bets: number | null) {
  if (g.metric === 'avg_hits') return { main: fixed(value), sub: [`hits per ${perUnit(g.unit)}`] };
  const against = g.measure.against?.startsWith('a fair ticket') ? 'bound vs a fair ticket, not money won' : 'bound in EUR per bet (payout table)';
  return { main: signedFixed(value), sub: [against, `raw profit ${signedFixed(profitPerBet)} EUR per bet${bets !== null ? ` · ${bets} bets` : ''}`] };
}

// The Q0 band (context since 27 Oct 2026: the best of 8 untuned rows on 120 fair draws, NullControls.py) built from one
// control only - the rule takes the higher of synthetic and shuffled, so a band without one of them can be lower than
// the rule's: "shuffled only (synthetic missing)". '' for a band from both.
export function oneControlText(b: { from: string; missing: MissingControl[] }): string {
  if (!b.missing.length) return '';
  return `${b.from} only (${b.missing.map((m) => `${m.mode} ${m.unreadable ? 'unreadable' : 'missing'}`).join(', ')})`;
}

// "a · b · c" without the empty parts
export const joinParts = (parts: (string | null | undefined | false)[], sep = ' · ') => parts.filter(Boolean).join(sep);

// What a feature-control table is besides its length: built per position, and how far it lags the game's data
// file of draws ("1 draw behind the data file"; the old card's "behind the file").
export function tableNote(f: { behind: number | null; positional: boolean }, unit: string): string {
  return joinParts([f.positional ? 'positional' : '', f.behind ? `${count(f.behind, unit)} behind the data file` : '']);
}

// --- Q0, the row tests (27 Oct 2026, src/RowChance.py) -----------------------------------------------------------
// A p value as the cards print it: three decimals, "< 0.001" below; a corrected p two decimals (three under 0.01).
export const pText = (p: number | null) => (!isNum(p) ? DASH : p < 0.001 ? '< 0.001' : p.toFixed(3));
export const correctedText = (p: number | null) => (!isNum(p) ? DASH : p < 0.001 ? '< 0.001' : p < 0.01 ? p.toFixed(3) : p.toFixed(2));
// what a row's hits are held against: a fair draw (lottery) or its own tickets shuffled (markets)
export const nullWord = (method: string | null | undefined) => (method === 'permutation' ? 'shuffled' : 'fair');
// "22 live draws", "1 live week"
export const liveCount = (n: number | null, unit: string) => (isNum(n) ? `${n} live ${n === 1 ? perUnit(unit) : units(unit)}` : DASH);

// One row's test on one line, its own figures next to its p (the plan-card trust rule: a p never stands next to an
// all-draws average): "WeightedEnsemble 1.091 on 22 live draws vs fair 0.800 · p 0.058 · corrected 1.00". name false
// leaves the row's name out (a table cell that has it); a row outside the family has no corrected p.
export function chanceLine(r: Q0Line, unit: string, method: string | null | undefined, name = true): string {
  return joinParts([
    `${name ? `${short(r.row)} ` : ''}${fixed(r.mean)} on ${liveCount(r.live, unit)} vs ${nullWord(method)} ${fixed(r.null)}`,
    `p ${pText(r.p)}`,
    r.inFamily && isNum(r.pHolm) ? `corrected ${correctedText(r.pHolm)}` : '',
  ]);
}

// A followed row's fresh record in one line: running ("followed since Sun 11 Oct · 4 of 28 new draws · so far 1.500 vs
// fair 0.800, not a verdict"), stopped (no new live draw for four weeks), or its verdict ("confirmed on 28 new draws ·
// 1.214 vs fair 0.800 · p 0.006 (needs < 0.01) · draw of 13 Feb"). A running row shows no p: read week after week, a
// p would be a verdict taken whenever it dips.
export function freshText(f: Q0Fresh, unit: string, method: string | null | undefined, verdictAlpha = 0.01): string {
  const vs = `vs ${nullWord(method)}`;
  const length = f.verdictLength ?? 0;
  if (f.state === 'confirmed' || f.state === 'not confirmed') {
    return joinParts([
      `${f.state} on ${count(length, unit).replace(/^(\d+) /, '$1 new ')}`,
      `${fixed(f.mean)} ${vs} ${fixed(f.null)}`,
      `p ${pText(f.p)} (needs < ${verdictAlpha})`,
      f.verdictAt ? `${perUnit(unit)} of ${day(f.verdictAt)}` : '',
    ]);
  }
  const so = Math.min(f.draws, length);
  return joinParts([
    `followed since ${weekday(f.since)}`,
    f.state === 'stopped' ? `stopped: no new live ${perUnit(unit)} for four weeks${f.last ? ` (last ${day(f.last)})` : ''}` : '',
    `${so} of ${length} new ${length === 1 ? perUnit(unit) : units(unit)}`,
    so ? `so far ${fixed(f.mean)} ${vs} ${fixed(f.null)}, not a verdict` : 'no new one yet',
  ]);
}

// The Q0 tile's foot (the Controls and Week pages): under the rows confirmed on new draws in the hit-ranked games - the one
// figure that counts - this week's figures, each with what chance gives next to it.
export function q0TileFoot(t: ControlTiles['q0'], unitOf: (game: string) => string): string[] {
  const p = t.payout;
  return [
    t.confirmedRows.length ? `confirmed: ${t.confirmedRows.map((x) => `${short(x.row)} (${x.game})`).join(', ')}` : '',
    `${t.holds} hold this week (corrected for the rows compared) · ${t.under} under 5%, ≈${fixed(t.expected, 1)} by chance · ${t.followed} followed${t.stopped ? ` (${t.stopped} stopped)` : ''}`,
    `${t.family} rows tested on live draws in ${t.games} of ${t.hitRankedGames} hit-ranked games`,
    t.tooFew.length ? `too few live draws yet: ${t.tooFew.map((x) => `${x.game} ${x.liveMax} of ${count(x.minDraws, unitOf(x.game))}`).join(', ')}` : '',
    t.noTest.length ? `no row test: ${t.noTest.map((x) => `${x.game}${x.why ? ` (${x.why})` : ''}`).join(', ')}` : '',
    p.games.length
      ? `payout games, on hits, not counted: ${p.under} under 5% (≈${fixed(p.expected, 1)}) · ${p.followed} followed · ${p.confirmed} confirmed`
      : '',
    `report ${stamp(t.at)}`,
  ];
}
