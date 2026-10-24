import { DASH, dayYear, isNum, money, short, unsigned } from '../../format';
import type { Best, DayFigures, Shape } from '../../workbench';

// The draws' components serve two readers: the admin's Workbench (every row, the All / Top 5 switch, the ranking's
// version on each line) and the users' Lottery (next batch: the top five only, as known that day, in the user's words).
export type Mode = 'admin' | 'user';

export const GAME_TITLES: Record<string, string> = {
  lotto: 'Lotto',
  euromillions: 'EuroMillions',
  eurodreams: 'EuroDreams',
  vikinglotto: 'Vikinglotto',
  jokerplus: 'Joker+',
  pick3: 'Pick 3',
  keno: 'Keno',
  crypto: 'Crypto day',
  cryptoweek: 'Crypto week',
  shares: 'Shares day',
  sharesweek: 'Shares week',
};
export const gameTitle = (game: string) => GAME_TITLES[game] ?? game;

// The market a market game belongs to, and the currency its paper money is in (markets.js MARKETS: USDT, USD)
export const MARKET_OF: Record<string, string> = { crypto: 'crypto', cryptoweek: 'crypto', shares: 'shares', sharesweek: 'shares' };
export const CURRENCY_OF: Record<string, string> = { crypto: 'USDT', cryptoweek: 'USDT', shares: 'USD', sharesweek: 'USD' };

// The highest number a game draws, so a vote chart has a bar for every number (a gap is a number no row carried)
export const NUMBER_MAX: Record<string, number> = { lotto: 45, euromillions: 50, eurodreams: 40, vikinglotto: 48, keno: 70 };

// What one draw of a market game is, in a few words (the old game view's market note)
export const MARKET_DRAW: Record<string, string> = {
  crypto: 'one UTC day: per coin the bin (0-9) its close-to-close move fell in',
  cryptoweek: 'one UTC week, Monday to Monday: per coin the bin (0-9) its week move fell in',
  shares: 'one New York trading day: per share the bin (0-9) its close-to-close move fell in',
  sharesweek: 'one trading week to its last session: per share the bin (0-9) its week move fell in',
};

// "October 2026" for "2026-10"
export function monthTitle(m: string): string {
  const d = new Date(`${m}-15T12:00:00Z`);
  if (Number.isNaN(d.getTime())) return m;
  return new Intl.DateTimeFormat('en-GB', { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d);
}

// An ISO day inside a server sentence ("no ranking recorded before 2026-08-25") read as "25 Aug 2026"
export const prettyDates = (text: string) => text.replace(/\b\d{4}-\d{2}-\d{2}\b/g, (iso) => dayYear(iso));

// The specials' group name over the drawn row and in the table head
export function specialsLabel(shape: Shape): string {
  const s = shape.specials;
  if (!s.length) return shape.bonus ? 'Bonus' : '';
  if (s[0].startsWith('Star')) return 'Stars';
  return s[0];
}

// The best ticket as the old page wrote it: lotto "N (M)" (M the bonus ball), the special games "N (M)" (M the
// specials), Joker+ "L/R (Z)", a market game "N of 5" (slots in the right bin); keno and pick3 have none.
export function bestText(shape: Shape, best: Best | null): string {
  if (!best) return DASH;
  const n = (v: number | null) => (isNum(v) ? String(v) : DASH);
  if (shape.family === 'jokerplus') return `${n(best.left)}/${n(best.right)} (${n(best.sign)})`;
  if (shape.family === 'market') return `${n(best.mains)} of ${shape.slots.length || DASH}`;
  if (shape.family === 'lotto' || shape.family === 'specials') return `${n(best.mains)} (${n(best.specials)})`;
  return DASH;
}

// What "N (M)" means for the game, said once next to the figure
export function bestMeaning(shape: Shape): string {
  switch (shape.family) {
    case 'lotto':
      return 'drawn numbers (bonus ball)';
    case 'specials':
      return `numbers (${specialsLabel(shape).toLowerCase()})`;
    case 'jokerplus':
      return 'leading/trailing digits in place (sign)';
    case 'market':
      return 'slots in the right bin';
    default:
      return '';
  }
}

export const eur = (v: number | null | undefined, decimals = 2) => `${money(v, decimals)} EUR`;
// an amount without a sign: a stake, a price
export const eurU = (v: number | null | undefined, decimals = 2) => `${unsigned(v, decimals)} EUR`;

// One draw's figures, or one month's, on two lines: the figure in bold, then over how many rows and tickets it is.
// The counts are the ones the table under it shows (draws.dayFigures over the same rows).
export function figureLines(shape: Shape, f: DayFigures | { rows: number; tickets: number; profit: number | null; stake: number | null; best: Best | null; multi: number | null }, mode: Mode): { main: string; sub: string[] } {
  const rows = `${f.rows} ${f.rows === 1 ? 'row' : 'rows'}`;
  const whose = mode === 'user' ? `the ${rows} shown` : rows;
  if (shape.money) {
    const tickets = `${f.tickets} ${f.tickets === 1 ? 'ticket' : 'tickets'}`;
    const sub = [`net over ${tickets} of ${whose} · stake ${eurU(f.stake)}`];
    if (shape.family === 'jokerplus' && f.best) sub.push(`best ${bestText(shape, f.best)} ${bestMeaning(shape)} by ${short(f.best.by)}`);
    return { main: eur(f.profit), sub };
  }
  if (!f.best) return { main: DASH, sub: [`no scored row among ${whose}`] };
  const sub = [`${bestMeaning(shape)} · best of ${whose}, by ${short(f.best.by)}`];
  if (shape.family === 'lotto' && isNum(f.multi)) sub.push(`best of 9 numbers (multi-pick): ${f.multi}`);
  return { main: bestText(shape, f.best), sub };
}

// Lotto's multi-pick, the two facts the old page's legend gave beside the sizes and chances: the order of a row's six,
// and what the chances leave out (the chance of 3 or more counts the six drawn numbers, never the bonus ball)
export const MULTI_PICK_ORDER =
  'A row with shaded numbers lists its six in the model’s own probability order, highest first, not small to large; a row without them (a vote, the RL ticket, a row whose ranking failed that run) keeps its six small to large.';

// How hits are counted, per family (the old page's help), in the reader's words
export function howHitsCounted(shape: Shape, mode: Mode): string {
  const user = mode === 'user';
  switch (shape.family) {
    case 'lotto':
      return 'Hits read N (M): N of the 6 drawn numbers, and M = 1 (amber) when a played number is the bonus ball; 5 (1) is a high tier, 6 (0) the jackpot. A row with three shaded numbers lists its six in the model’s own order, most probable first, and the shaded 7th to 9th are its next three, for a system play; “of 9” counts hits among all nine.';
    case 'specials':
      return `Hits read N (M): N of the drawn numbers and M of the drawn ${specialsLabel(shape).toLowerCase()} (green); a special number counts only among the specials. A row without special cells plays the numbers only.`;
    case 'jokerplus':
      return 'Hits read L/R (Z): L digits in place from the left (green), R from the right (blue), Z = 1 when the sign is right (amber). Each run pays by its length and the two add up; all six in place is its own tier (with the right sign, the jackpot); a right sign alone refunds the 1.50 EUR ticket. The profit is net of the ticket. A player chooses only the sign: the six digits are generated by the system, so the digits shown cannot be played.';
    case 'keno':
      return 'A hit is a played number among the 20 drawn. Each row plays its tickets of 5 to 10 numbers at 1 EUR each; the sizes played changed over time (one ticket a row on many older draws, one per size on the newest), as the row’s lines show. Its 20-number ranking line is not a ticket. The profit is net: a win is the prize minus the stake, a loss −1 EUR.';
    case 'pick3':
      return 'A hit is the right digit in the right place (green). A ticket plays straight, box, front pair and back pair at 1 EUR each (a triple cannot play box, so 3 EUR); the profit is net, with the bets won next to it.';
    case 'market':
      return user
        ? 'A hit is the right bin (0-9) for the right instrument (green); one bin off is pale green. A blind guess is right 1 time in 10 per slot.'
        : 'A hit is the right bin (0-9) in the right slot (green); one bin off is pale green. Chance is 1 in 10 per slot, 28% within one bin, 50% for the direction. The arrow is the position the paper books take: ▲ long (a bin in the upper half), ▼ not long (a paper short with shorts on); the RL Position Model reads ×size, ×0 sits out.';
    default:
      return '';
  }
}

// What the Regime HMM rows' shared feature set means for the Single ablation (markets.js hmmFeaturesNote, 14 Oct
// 2026: the 10 Oct studies chose returns alone in all four markets, which changed what Single is).
export function hmmFeaturesText(features: string | null): string {
  if (!features) return '';
  const set = features.split(',').map((f) => f.trim()).filter(Boolean);
  const head = `The three Regime HMM rows run on one tuned feature set, ${set.join(', ')}, shared on purpose so the ablations stay ablations.`;
  return set.length === 1 && set[0] === 'returns'
    ? `${head} On returns alone, Single is a Gaussian with a constant mean and variance per instrument, a random walk with drift, so "not better than Single" says the returns hold no regimes in mean or variance; the volatility and momentum columns are not in play.`
    : `${head} With volatility and momentum in the set, Single is a linear regression of the returns on those columns, so "not better than Single" says the regimes add nothing to that regression.`;
}
