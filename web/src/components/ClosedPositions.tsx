import { useState } from 'react';
import type { TraderClosed } from '../api';
import { DASH, cap, day, int, isNum, money, pct, rate, signClass } from '../format';

const NEWEST = 10;
const REASON_CHIP: Record<string, string> = { calls: 'down', stop: 'down', take: 'up', longest: 'flat' };

// The trader's sold long positions. Text columns first, the two numbers at the right edge, as on the card.
export default function ClosedPositions({ closed, noun }: { closed: TraderClosed[]; noun: string }) {
  const [all, setAll] = useState(false);
  const rows = all ? closed : closed.slice(0, NEWEST);
  const late = closed.some((r) => !r.live);
  return (
    <div className="card span-3">
      <div className="card-h">
        <h2>Closed positions</h2>
        <div className="meta">{closed.length ? `${closed.length} positions · newest first` : 'newest first'}</div>
      </div>
      {rows.length === 0 ? (
        <div className="card-b empty">No sold position yet.</div>
      ) : (
        <div className="card-b flush">
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Sold</th>
                  <th>{cap(noun || 'coin')}</th>
                  <th className="hide-sm">Bought</th>
                  <th className="hide-sm">Why sold</th>
                  <th className="num hide-sm" title="the days the position was held">
                    Days
                  </th>
                  <th className="num" title="the price change from the buy to the sale, before costs">
                    Move
                  </th>
                  <th className="num" title="paper money, costs included">
                    Profit
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={`${r.sold}-${r.symbol}-${i}`} title={[isNum(r.entryPrice) ? `bought at ${rate(r.entryPrice)}` : '', isNum(r.exitPrice) ? `sold at ${rate(r.exitPrice)}` : ''].filter(Boolean).join(' · ') || undefined}>
                    <td className="mut">{day(r.sold)}</td>
                    <td className="sym">
                      {r.symbol || DASH}
                      {r.live ? null : (
                        <span className="tag" title="not decided before the booking hour of its days: filled in afterwards from the record">
                          filled in
                        </span>
                      )}
                    </td>
                    <td className="hide-sm mut">{day(r.bought)}</td>
                    <td className="hide-sm">
                      <span className={`chip ${(r.reason && REASON_CHIP[r.reason]) || 'flat'}`}>{r.reasonText}</span>
                    </td>
                    <td className="num hide-sm">{isNum(r.days) ? int(r.days) : DASH}</td>
                    <td className={`num ${signClass(r.movePct)}`}>{pct(r.movePct)}</td>
                    <td className={`num strong ${signClass(r.pnl)}`}>{money(r.pnl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {closed.length > NEWEST || late ? (
        <div className="card-f">
          {late ? <span className="mut">&quot;filled in&quot;: a position the record holds without a decision stored before its booking hour.</span> : null}
          {closed.length > NEWEST ? (
            <button type="button" className="linkbtn more" onClick={() => setAll((v) => !v)}>
              {all ? `Show the ${NEWEST} newest` : `Show all ${closed.length}`}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
