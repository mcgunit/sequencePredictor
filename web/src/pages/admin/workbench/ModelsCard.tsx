import type { ReactNode } from 'react';
import { Card, Scroll } from '../../../components/admin/Card';
import { DASH, day, money, share, short } from '../../../format';
import type { BookFigures, Workbench } from '../../../workbench';

// Models over the scored days (C12): each row's accuracy against chance and its paper books under every rule. The old
// table's hover-only figures (trades, win rate, money per position) are on the lines under each total.

function Rate({ v, above }: { v: number | null; above: boolean }) {
  return above ? <b>{share(v)}</b> : <span>{share(v)}</span>;
}

function Book({ b, cur, sub }: { b: BookFigures | null; cur: string; sub?: ReactNode }) {
  if (!b) return <span className="mut">{DASH}</span>;
  return (
    <>
      <span className={b.total !== null && b.total > 0 ? 'strong' : undefined}>{money(b.total)}</span>
      <span className="sub">{b.trades ? `${b.trades} trades · ${share(b.winRate)} won · ${money(b.perTrade)} ${cur} a trade` : 'no trade'}</span>
      {sub}
    </>
  );
}

export default function ModelsCard({ w }: { w: Workbench }) {
  const cur = w.currency;
  const open = w.models.some((m) => m.open);
  const c = w.chance;
  return (
    <Card
      id="models"
      title="Models"
      help={`Exact: the right bin; within one bin: the bin or a neighbour; direction: the predicted bin's middle on the right side of zero. Each row over its own scored days. The books trade ${w.stake ?? 100} ${cur} a position with a fee on each leg; a book's total is in ${cur}. The RL Position Model sizes its positions (0 to 2 times the stake) and is long only, so its long-and-short figures are its long-only ones.`}
      meta={[`${w.models.length} rows`, `${w.scoredDays ?? DASH} scored days`, `bold: above chance (exact ${share(c.exact)}, within one bin ${share(c.adjacent)}, direction ${share(c.direction)})`]}
      flush
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Exact</th>
              <th className="num hide-sm">Within one</th>
              <th className="num hide-sm">Direction</th>
              <th className="num">Daily {cur}</th>
              <th className="num hide-sm">Hold while up</th>
              {open ? <th className="num hide-sm">Open to close</th> : null}
              <th className="num hide-sm">Long &amp; short</th>
            </tr>
          </thead>
          <tbody>
            {w.models.map((m) => (
              <tr key={m.name}>
                <td>
                  {short(m.name)}
                  {m.name === w.best ? <span className="tag">best exact</span> : null}
                  <span className="sub">
                    {m.days ?? DASH} days, {day(m.firstDay)} – {day(m.lastDay)}
                    {m.sized ? ' · sized, long only' : ''}
                  </span>
                  <span className="sub show-sm">
                    within one {share(m.adjacent)} · direction {share(m.direction)} · hold {money(m.hold?.total)}
                    {m.open ? ` · open ${money(m.open.total)}` : ''} · long &amp; short {money(m.longShort?.total)}
                  </span>
                </td>
                <td className="num">
                  <Rate v={m.exact} above={m.aboveExact} />
                </td>
                <td className="num hide-sm">
                  <Rate v={m.adjacent} above={m.aboveAdjacent} />
                </td>
                <td className="num hide-sm">
                  <Rate v={m.direction} above={m.aboveDirection} />
                </td>
                <td className="num">
                  <Book b={m.daily} cur={cur} />
                </td>
                <td className="num hide-sm">
                  <Book b={m.hold} cur={cur} />
                </td>
                {open ? (
                  <td className="num hide-sm">
                    <Book b={m.open} cur={cur} />
                  </td>
                ) : null}
                <td className="num hide-sm">
                  <Book
                    b={m.longShort}
                    cur={cur}
                    sub={m.longShort ? <span className="sub">hold {money(m.longShort.hold)}{m.longShort.open !== null ? ` · open ${money(m.longShort.open)}` : ''}{m.sized ? ' · long only' : ''}</span> : null}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}
