import type { ReactNode } from 'react';
import { DASH, isNum, money, short, unsigned } from '../../format';
import type { DrawResult, MarketDay, Row, Shape, Ticket } from '../../workbench';
import { Legend, Scroll } from '../admin/Card';
import Chip from '../admin/Chip';
import { specialsLabel, type Mode } from './text';

// Every row's tickets in the old History table's grid (owner, 24 Oct 2026: "the table with the hits like it was"):
// one column per number - Num 1..n, then the special columns, Joker+'s Sign, a market's instruments -, lotto's
// 7th-9th multi-pick numbers in their shaded columns, and a cell coloured the way the game scores it (draws.js
// marks): green a hit, amber the lotto bonus and Joker+'s sign, blue Joker+'s trailing run, pale green a market's
// bin one off or a pick3 digit drawn in another place. The Hits and Profit columns say in words what the colours
// show. Wide tables scroll inside their card, as the old ones did.

interface Props {
  shape: Shape;
  rows: Row[];
  result: DrawResult | null; // null: the coming draw (no hits yet)
  market: MarketDay | null; // a market day's figures per row (the newest 30 settled days)
  mode: Mode;
  currency?: string; // a market game's paper money
  ranked: boolean; // the rows carry a rank (a ranking existed for this draw)
  showTop: boolean; // mark the rows in the draw's top five (All)
  rankWhose?: string; // whose ranking the rank is in: the one before a past draw, today's for the coming draw
}

const plural = (n: number | null | undefined, one: string, many = `${one}s`) => `${isNum(n) ? n : DASH} ${n === 1 ? one : many}`;
// a draws.js mark letter -> the cell's colour class: hit, bonus/sign, trailing run, one off
const CELL: Record<string, string> = { m: 'hm', l: 'hm', s: 'hm', b: 'hb', z: 'hb', r: 'hr', a: 'ha' };
const ordinal = (n: number) => `${n}${n === 1 ? 'st' : n === 2 ? 'nd' : n === 3 ? 'rd' : 'th'}`;

function NameCell({ row, mode, ranked, showTop, rankWhose, first }: { row: Row; mode: Mode; ranked: boolean; showTop: boolean; rankWhose: string; first: boolean }) {
  if (!first) return <td className="mdl mut">{short(row.name)}</td>;
  return (
    <td className="mdl">
      <span className="strong">{short(row.name)}</span>
      {showTop && row.top ? (
        <>
          {' '}
          <Chip kind="model">top 5</Chip>
        </>
      ) : null}
      {mode === 'admin' && ranked ? <span className="sub">{row.rank ? `#${row.rank} ${rankWhose}` : `not ${rankWhose}`}</span> : null}
    </td>
  );
}

// The grid's columns for this table: how many main cells, which special/sign columns, whether lotto's extras show
function columnsOf(shape: Shape, rows: Row[]) {
  const tickets = rows.flatMap((r) => r.tickets);
  const mains = Math.max(0, ...tickets.map((t) => t.mains.length));
  const specials = shape.family === 'specials' ? Math.max(0, ...tickets.map((t) => (t.specials ? t.specials.length : 0))) : 0;
  const sign = shape.family === 'jokerplus';
  const extras = shape.family === 'lotto' && rows.some((r) => r.multiPick) ? Math.max(3, ...rows.map((r) => (r.multiPick ? r.multiPick.numbers.length : 0))) : 0;
  return { mains, specials, sign, extras };
}

function headCells(shape: Shape, cols: ReturnType<typeof columnsOf>): ReactNode[] {
  const out: ReactNode[] = [];
  for (let i = 0; i < cols.mains; i += 1) {
    const label = shape.family === 'market' ? (shape.slots[i] ?? `Slot ${i + 1}`) : `Num ${i + 1}`;
    out.push(<th key={`m${i}`} className="c">{label}</th>);
  }
  const special = specialsLabel(shape).replace(/s$/, '');
  for (let i = 0; i < cols.specials; i += 1) out.push(<th key={`s${i}`} className="c sp">{cols.specials > 1 ? `${special} ${i + 1}` : special}</th>);
  if (cols.sign) out.push(<th key="sign" className="c sp">Sign</th>);
  const base = cols.mains;
  for (let i = 0; i < cols.extras; i += 1) out.push(<th key={`x${i}`} className="c x">{ordinal(base + i + 1)}</th>);
  return out;
}

// One ticket's number cells, coloured by its marks; a shorter ticket's missing cells stay empty
function bodyCells(shape: Shape, row: Row, t: Ticket, ti: number, cols: ReturnType<typeof columnsOf>, result: DrawResult | null): ReactNode[] {
  const out: ReactNode[] = [];
  for (let i = 0; i < cols.mains; i += 1) {
    const v = t.mains[i];
    if (v === undefined) {
      out.push(<td key={`m${i}`} className="c none" />);
      continue;
    }
    let cls = CELL[t.mark[i] ?? ''] ?? '';
    // a market's bin one off the drawn one (pale green); a pick3 digit drawn in another place (pale green)
    if (!cls && shape.family === 'market' && result && isNum(result.mains[i]) && Math.abs(v - (result.mains[i] as number)) === 1) cls = 'ha';
    if (!cls && shape.family === 'pick3' && (t.markMember[i] ?? '') === 'm') cls = 'ha';
    out.push(<td key={`m${i}`} className={`c ${cls}`}>{v}</td>);
  }
  for (let i = 0; i < cols.specials; i += 1) {
    const v = t.specials ? t.specials[i] : undefined;
    out.push(
      <td key={`s${i}`} className={`c sp ${v === undefined ? 'none' : (CELL[t.markSpecials[i] ?? ''] ?? '')}`}>
        {v === undefined ? (i === 0 && t.specials && !t.specials.length ? '–' : '') : v}
      </td>,
    );
  }
  if (cols.sign) {
    const sign = t.signName ?? (t.specials && t.specials.length ? String(t.specials[0]) : null);
    out.push(<td key="sign" className={`c sp word ${CELL[t.markSpecials[0] ?? ''] ?? ''}`}>{sign ?? '–'}</td>);
  }
  // lotto's 7th-9th: the main ticket's (the first) extras only, as the old table had them
  const mp = ti === 0 ? row.multiPick : null;
  for (let i = 0; i < cols.extras; i += 1) {
    const v = mp ? mp.numbers[i] : undefined;
    out.push(
      <td key={`x${i}`} className={`c x ${v !== undefined && (mp?.mark[i] ?? '') === 'm' ? 'hm' : ''}${v === undefined ? ' none' : ''}`}>{v === undefined ? '–' : v}</td>,
    );
  }
  return out;
}

// The Hits cell in words: "2 (1) · of 9: 3", "1/2 (0)", "2 in place", "2 of 5"
function hitsOf(shape: Shape, row: Row, t: Ticket, market: MarketDay | null, mode: Mode): { main: string; sub: string | null } {
  const h = t.hits;
  const n = (v: number | null | undefined) => (isNum(v) ? String(v) : DASH);
  if (!h) return { main: DASH, sub: null };
  switch (shape.family) {
    case 'lotto': {
      const nine = row.multiPick && isNum(row.multiPick.nine) ? `of 9: ${row.multiPick.nine}` : null;
      return { main: `${n(h.mains)} (${n(h.specials)})`, sub: nine };
    }
    case 'specials':
      return t.specials && !t.specials.length ? { main: n(h.mains), sub: 'numbers only' } : { main: `${n(h.mains)} (${n(h.specials)})`, sub: null };
    case 'jokerplus':
      return { main: `${n(h.left)}/${n(h.right)} (${n(h.sign)})`, sub: null };
    case 'pick3':
      // the admin also sees the old page's membership count; a user the digits in place only
      return { main: `${n(h.positional)} in place`, sub: mode === 'admin' ? `membership ${n(h.membership)} (the old Hits)` : null };
    case 'market': {
      const m = market?.models[row.name];
      const of = shape.slots.length || t.mains.length;
      return { main: `${n(m?.exact ?? h.mains)} of ${of}`, sub: m ? `within one bin ${n(m.adjacent)} · direction ${n(m.direction)} of ${n(m.positions)}` : null };
    }
    default:
      return { main: n(h.mains), sub: null };
  }
}

const BET_WORDS: [keyof NonNullable<Ticket['bets']>, string][] = [
  ['straight', 'straight'],
  ['box', 'box'],
  ['frontPair', 'front pair'],
  ['backPair', 'back pair'],
  ['consolation', 'consolation'],
];
function betsWon(t: Ticket): string | null {
  if (!t.bets) return null;
  const won = BET_WORDS.filter(([k]) => (t.bets?.[k] ?? 0) > 0).map(([k, w]) => `${w} ${unsigned(t.bets?.[k], 0)}`);
  return won.length ? `won: ${won.join(', ')} EUR` : 'no bet won';
}

function legendOf(shape: Shape, extras: boolean) {
  switch (shape.family) {
    case 'lotto':
      return [
        { mark: 'nbm', label: 'a drawn number' },
        { mark: 'nbb', label: 'the bonus ball' },
        ...(extras ? [{ mark: 'nbx', label: '7th-9th: the multi-pick extras' }] : []),
      ];
    case 'specials':
      return [{ mark: 'nbm', label: `a hit, among the numbers or the ${specialsLabel(shape).toLowerCase()}` }];
    case 'jokerplus':
      return [
        { mark: 'nbm', label: 'leading run' },
        { mark: 'nbr', label: 'trailing run' },
        { mark: 'nbb', label: 'the sign' },
      ];
    case 'pick3':
      return [
        { mark: 'nbm', label: 'the right digit in place' },
        { mark: 'nba', label: 'drawn, in another place' },
      ];
    case 'market':
      return [
        { mark: 'nbm', label: 'the right bin' },
        { mark: 'nba', label: 'one bin off' },
      ];
    default:
      return [{ mark: 'nbm', label: 'a drawn number' }];
  }
}

export default function TicketTable({ shape, rows, result, market, mode, currency, ranked, showTop, rankWhose = 'in that ranking' }: Props) {
  const scored = result !== null;
  if (!rows.length) return <div className="empty">No rows in this scope.</div>;
  const keno = shape.family === 'keno';
  const cols = columnsOf(shape, rows);
  const money_ = shape.money && scored;
  const pnl = shape.family === 'market' && scored && market !== null;
  const hitsHead =
    shape.family === 'lotto' || shape.family === 'specials'
      ? `Hits${cols.extras ? ' · of 9' : ''}`
      : shape.family === 'jokerplus'
        ? 'Hits L/R (Z)'
        : shape.family === 'market'
          ? 'Exact'
          : 'Hits';
  return (
    <>
      <Scroll>
        <table className="tbl tix">
          <thead>
            <tr>
              <th>Model</th>
              <th className="rk" title={keno ? 'the ticket: its size, or the row\'s ranking (not played)' : 'the row\'s ticket'}>
                {keno ? 'Ticket' : '#'}
              </th>
              {headCells(shape, cols)}
              {scored ? <th>{hitsHead}</th> : null}
              {money_ ? <th className="num">Profit EUR</th> : null}
              {pnl ? <th className="num">P&amp;L {currency ?? ''}, long</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) =>
              row.tickets.map((t, i) => {
                const h = hitsOf(shape, row, t, market, mode);
                const m = market?.models[row.name];
                const ranking = keno && t.kind === 'ranking';
                return (
                  <tr key={`${row.name}-${i}`} className={i === 0 ? 'first' : undefined}>
                    <NameCell row={row} mode={mode} ranked={ranked} showTop={showTop} rankWhose={rankWhose} first={i === 0} />
                    <td className="rk">{keno ? (ranking ? 'ranking' : t.mains.length) : i + 1}</td>
                    {bodyCells(shape, row, t, i, cols, result)}
                    {scored ? (
                      <td className="hits">
                        <span className="strong">{keno ? plural(t.hits?.mains, 'hit') : h.main}</span>
                        {!keno && h.sub ? <span className="sub">{h.sub}</span> : null}
                      </td>
                    ) : null}
                    {money_ ? (
                      <td className="num hits">
                        {ranking ? (
                          <span className="mut">not played</span>
                        ) : (
                          <>
                            <span className={isNum(t.profit) && t.profit > 0 ? 'strong' : undefined}>{money(t.profit)}</span>
                            <span className="sub">
                              stake {unsigned(t.stake)}
                              {shape.family === 'pick3' ? ` · ${betsWon(t)}` : ''}
                            </span>
                          </>
                        )}
                      </td>
                    ) : null}
                    {pnl ? (
                      <td className="num hits">
                        {m ? money(m.pnlCash) : DASH}
                        {m ? <span className="sub">{plural(m.trades, 'long position')}</span> : null}
                      </td>
                    ) : null}
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      </Scroll>
      <div className="card-b">
        <Legend items={keno ? [{ mark: 'nbm', label: 'a drawn number' }] : legendOf(shape, cols.extras > 0)} />
      </div>
    </>
  );
}
