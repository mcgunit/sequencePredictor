import { DASH, isNum, money, short, unsigned } from '../../format';
import type { DrawResult, MarketDay, Row, Shape, Ticket } from '../../workbench';
import { Legend, Scroll } from '../admin/Card';
import Chip from '../admin/Chip';
import { Nb, Slot } from './ResultStrip';
import { eur, eurU, specialsLabel, type Mode } from './text';

// Every row's tickets with the hits marked the way the game scores them (draws.js marks): lotto's bonus and the
// multi-pick 7th-9th numbers, the special games' own group, Joker+'s runs and sign, pick3's digits in place and its
// bets, keno's ticket per size, a market's bin per slot with the position the books take. The Hits and Profit
// columns say in words what the colours show. On a phone the table folds into one block per row.

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

function NameCell({ row, mode, ranked, showTop, rankWhose }: { row: Row; mode: Mode; ranked: boolean; showTop: boolean; rankWhose: string }) {
  return (
    <td className="lead nm">
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

// the number cells of one ticket
function Cells({ shape, row, t, result, market, extrasInTable }: { shape: Shape; row: Row; t: Ticket; result: DrawResult | null; market: MarketDay | null; extrasInTable: boolean }) {
  if (shape.family === 'market') {
    const k = market?.k ?? 10;
    const sizes = market?.models[row.name]?.sizes ?? null;
    return (
      <span className="nums">
        {t.mains.map((v, i) => {
          const drawn = result?.mains[i];
          const mark = t.mark[i] === 'm' ? 'm' : isNum(drawn) && Math.abs(v - drawn) === 1 ? 'a' : '';
          // the position the paper books take (only on a settled day with its market figures)
          const size = sizes ? sizes[i] : null;
          const under = market ? (sizes ? `×${isNum(size) ? size : 0}` : v >= k / 2 ? '▲' : '▼') : undefined;
          return (
            <Slot key={i} label={shape.slots[i] ?? String(i + 1)} under={under}>
              <Nb v={v} mark={mark} />
            </Slot>
          );
        })}
      </span>
    );
  }
  const mains = t.mains.map((v, i) => <Nb key={`m${i}`} v={v} mark={t.mark[i] ?? ''} />);
  let second = null;
  if (shape.family === 'jokerplus') {
    const sign = t.signName ?? (t.specials && t.specials.length ? String(t.specials[0]) : null);
    second = sign ? <Nb v={sign} word mark={t.markSpecials[0] === 'z' ? 'z' : ''} /> : null;
  } else if (shape.family === 'specials' && t.specials) {
    second = t.specials.length ? (
      t.specials.map((v, i) => <Nb key={`s${i}`} v={v} mark={t.markSpecials[i] === 's' ? 's' : ''} />)
    ) : (
      <span className="nb none word">no {specialsLabel(shape).toLowerCase()}</span>
    );
  } else if (shape.family === 'lotto' && extrasInTable) {
    const mp = row.multiPick;
    second = mp
      ? mp.numbers.map((v, i) => <Nb key={`x${i}`} v={v} extra mark={mp.mark[i] === 'm' ? 'm' : ''} />)
      : [0, 1, 2].map((i) => (
          <span key={`x${i}`} className="nb none">
            –
          </span>
        ));
  }
  return (
    <span className="nums">
      {mains}
      {second ? (
        <>
          <span className="gap" />
          {second}
        </>
      ) : null}
    </span>
  );
}

// The Hits cell in words: "2 (1) · of 9: 3", "1/2 (0)", "2 in place", "2 of 5"
function hitsOf(shape: Shape, row: Row, t: Ticket, market: MarketDay | null): { main: string; sub: string | null } {
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
      return { main: `${n(h.positional)} in place`, sub: `membership ${n(h.membership)} (the old Hits)` };
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
      return [{ mark: 'nbm', label: 'the right digit in place' }];
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
  const extras = shape.family === 'lotto' && rows.some((r) => r.multiPick);
  if (!rows.length) return <div className="empty">No rows in this scope.</div>;
  if (shape.family === 'keno') return <KenoBlocks rows={rows} mode={mode} ranked={ranked} showTop={showTop} scored={scored} rankWhose={rankWhose} />;
  const money_ = shape.money && scored;
  const pnl = shape.family === 'market' && scored && market !== null;
  const numbersHead =
    shape.family === 'market'
      ? `Bin per slot${market ? ' · position' : ''}`
      : shape.family === 'lotto'
        ? extras
          ? 'Numbers · 7th-9th shaded'
          : 'Numbers'
        : shape.family === 'specials'
          ? `Numbers · ${specialsLabel(shape)}`
          : shape.family === 'jokerplus'
            ? 'Digits · sign'
            : 'Digits';
  const hitsHead =
    shape.family === 'lotto' || shape.family === 'specials' ? 'Hits N (M)' : shape.family === 'jokerplus' ? 'Hits L/R (Z)' : shape.family === 'market' ? 'Exact' : 'Hits';
  return (
    <>
      <Scroll>
        <table className="tbl stk">
          <thead>
            <tr>
              <th>Model</th>
              <th>{numbersHead}</th>
              {scored ? <th>{hitsHead}</th> : null}
              {money_ ? <th className="num">Profit EUR</th> : null}
              {pnl ? <th className="num">P&amp;L {currency ?? ''}, long</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) =>
              row.tickets.map((t, i) => {
                const h = hitsOf(shape, row, t, market);
                const m = market?.models[row.name];
                return (
                  <tr key={`${row.name}-${i}`}>
                    {i === 0 ? <NameCell row={row} mode={mode} ranked={ranked} showTop={showTop} rankWhose={rankWhose} /> : <td className="lead nm mut">{short(row.name)}</td>}
                    <td data-l="Numbers">
                      <Cells shape={shape} row={row} t={t} result={result} market={market} extrasInTable={extras} />
                    </td>
                    {scored ? (
                      <td data-l="Hits">
                        <span className="strong">{h.main}</span>
                        {h.sub ? <span className="sub">{h.sub}</span> : null}
                      </td>
                    ) : null}
                    {money_ ? (
                      <td className="num" data-l="Profit">
                        <span className={isNum(t.profit) && t.profit > 0 ? 'strong' : undefined}>{money(t.profit)}</span>
                        <span className="sub">
                          stake {unsigned(t.stake)}
                          {shape.family === 'pick3' ? ` · ${betsWon(t)}` : ''}
                        </span>
                      </td>
                    ) : null}
                    {pnl ? (
                      <td className="num" data-l="P&L">
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
        <Legend items={legendOf(shape, extras)} />
      </div>
    </>
  );
}

// Keno: per row its 20-number ranking (not played) and one ticket per size, each with its hits and net profit
function KenoBlocks({ rows, mode, ranked, showTop, scored, rankWhose }: { rows: Row[]; mode: Mode; ranked: boolean; showTop: boolean; scored: boolean; rankWhose: string }) {
  return (
    <>
      <div className="kgrid">
        {rows.map((row) => {
          const played = row.tickets.filter((t) => t.kind !== 'ranking');
          const profit = played.reduce((s, t) => s + (t.profit ?? 0), 0);
          const stake = played.reduce((s, t) => s + (t.stake ?? 0), 0);
          return (
            <div className="kb" key={row.name}>
              <div className="kb-h">
                <b>{short(row.name)}</b>
                {showTop && row.top ? <Chip kind="model">top 5</Chip> : null}
                {mode === 'admin' && ranked ? <span className="mut">{row.rank ? `#${row.rank} ${rankWhose}` : `not ${rankWhose}`}</span> : null}
                <span className="fig">
                  {scored ? <b>{eur(profit)}</b> : null}
                  <small>
                    {plural(played.length, 'ticket')} · stake {eurU(stake)}
                  </small>
                </span>
              </div>
              {row.tickets.map((t, i) => (
                <div className="kl" key={i}>
                  <span>{t.kind === 'ranking' ? 'ranking' : `${t.mains.length} numbers`}</span>
                  <span className="nums sm">
                    {t.mains.map((v, j) => (
                      <Nb key={j} v={v} mark={t.mark[j] ?? ''} />
                    ))}
                  </span>
                  <span className="num">
                    {scored ? plural(t.hits?.mains, 'hit') : ''}
                    {t.kind === 'ranking' ? `${scored ? ' · ' : ''}not played` : scored ? ` · ${money(t.profit)}` : ''}
                  </span>
                </div>
              ))}
            </div>
          );
        })}
      </div>
      <div className="card-b">
        <Legend items={[{ mark: 'nbm', label: 'a drawn number' }]} />
      </div>
    </>
  );
}
