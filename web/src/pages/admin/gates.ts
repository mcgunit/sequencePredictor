// The tuning gates' words, shared by the Tuning gates page, its drawer and This week: what each kind of gate score
// is (never hits per draw: a bound is the mean minus one standard error; the quantum tuner's are means on its own
// holdout; keno's capped profit is measured above a fair ticket, never money won), and the margin a set of decisions
// was made under.
import type { Range } from '../../api';
import { range, signedFixed } from '../../format';
import { joinParts, units } from './bits';

const KIND: Record<string, { short: string; long: string }> = {
  hits: { short: 'hits bound', long: 'hits bound, mean − 1 SE' },
  positional: { short: 'positional bound', long: 'positional hits bound, mean − 1 SE' },
  capped_profit: { short: 'capped profit bound', long: 'capped profit bound per bet, mean − 1 SE' },
  log_score: { short: 'log-score bound', long: 'log-score bound, mean − 1 SE' },
  subset_bound: { short: 'subset bound', long: 'subset bound, mean − 1 SE' },
  holdout_hits: { short: 'holdout hits', long: 'holdout hits, a mean on the quantum holdout (no SE taken off)' },
  holdout_profit: { short: 'holdout profit', long: 'holdout profit, a mean on the quantum holdout (no SE taken off)' },
};
// A keno capped-profit score is the capped profit above a fair ticket of the same size (src/TuningScore.py since 7 Oct
// 2026): a positive one is not a win. fair: the fair ticket's own capped value per bet.
const fairText = (fair: number | Range) =>
  `above a fair ticket (fair ${typeof fair === 'number' ? signedFixed(fair, 2) : fair.min === fair.max ? signedFixed(fair.min, 2) : `${signedFixed(fair.min, 2)} to ${signedFixed(fair.max, 2)}`}/bet)`;
export const kindShort = (kind: string | null) => (kind ? KIND[kind]?.short ?? `${kind} (gate score)` : 'gate score');
export const kindLong = (kind: string | null, fair: number | null = null) =>
  kind === 'capped_profit' && fair !== null ? `capped profit ${fairText(fair)}, per bet, mean − 1 SE: not money won` : kind ? KIND[kind]?.long ?? `${kind} (gate score)` : 'gate score';
// A tuner's kinds in a few words; its capped-profit records measured above a fair ticket said as such, with how many
export function kindsText(t: { kinds: string[]; aboveFair: number; fairPerBet: Range | null }): string {
  return t.kinds
    .map((k) => (k === 'capped_profit' && t.aboveFair && t.fairPerBet ? `${kindShort(k)} (${t.aboveFair} of them ${fairText(t.fairPerBet)}: not money won)` : kindShort(k)))
    .join(', ');
}
// "90 draws, 90 days, 90 trading days, 90 weeks": a gate window in each game's own unit, never one pooled word
export function windowText(byUnit: Record<string, Range>): string {
  return Object.entries(byUnit)
    .map(([unit, r]) => `${range(r)} ${units(unit)}`)
    .join(', ');
}

// "margin 0" (any gain replaces), "margin 1 SE", or the mix; '' for no decision at all (a tuner of the chain that has
// not written this weekend's yet) - never "margin mixed" over zero gates
export function marginText(t: { n: number; marginZero: number; marginOneSe: number | null }): string {
  if (!t.n) return '';
  if (t.marginZero === t.n) return 'margin 0';
  if (t.marginOneSe === t.n) return 'margin 1 SE';
  return joinParts([t.marginZero ? `${t.marginZero} at margin 0` : '', t.marginOneSe ? `${t.marginOneSe} at 1 SE` : '']) || 'margin mixed';
}

// A weekend's decision as a letter: R replaced, K kept, V reverted; – the weekend did not decide the row; ? git could not be read
export function letterOf(decision: string | null): string {
  if (decision === 'replaced') return 'R';
  if (decision === 'kept') return 'K';
  if (decision === 'reverted') return 'V';
  if (decision === 'not read') return '?';
  return '–';
}
