import type { ReactNode } from 'react';
import { MARKET_GAMES, type Follows, type Measure, type MissingControl } from '../../api';
import { count, fixed, isNum, perUnit, short, signedFixed } from '../../format';

// Small pieces the admin pages share: the game button that opens a drawer, a group row in a table, and the
// words that put a figure's unit and window on its own line.

export const isMarketGame = (game: string) => MARKET_GAMES.includes(game);

// "draws", "days", "trading days", "weeks" for a count other than one
export const units = (unit: string) => `${perUnit(unit)}s`;

// A row's record: its scored draws in the game's unit and, for a market game, how many of them were live
// (the ticket out before the close). Back-built or late days never count as record length.
export function recordText(draws: number | null, unit: string, live: number | null | undefined): string {
  const n = count(draws, unit);
  if (!isNum(live) || !isNum(draws)) return n;
  if (live >= draws) return `${n} · all live`;
  if (live <= 0) return `${n} · none live (back-built or late)`;
  return `${n} · ${live} live, ${draws - live} back-built or late`;
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

// A Q0 band built from one control only - the rule takes the higher of synthetic and shuffled, so a band without
// one of them can be lower than the rule's: "shuffled only (synthetic missing)". '' for a band from both.
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
