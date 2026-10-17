import { useState } from 'react';
import type { ClosedPosition } from '../api';
import { DASH, cap, day, money, pct, signClass } from '../format';

const NEWEST = 10;

// Text columns first, the two numbers at the right edge, as in the open positions.
export default function ClosedPositions({ closed, noun, pending }: { closed: ClosedPosition[]; noun: string; unit?: string; pending?: boolean }) {
  const [all, setAll] = useState(false);
  const rows = all ? closed : closed.slice(0, NEWEST);
  const short = (m: string | null) => (m ? m.replace(/ Model$/, '') : '');
  return (
    <div className="card span-3">
      <div className="card-h">
        <h2>Closed positions</h2>
        <div className="meta">
          {closed.length ? `${closed.length} positions · newest first` : 'newest first'}
        </div>
      </div>
      {rows.length === 0 ? (
        <div className="card-b empty">{pending ? 'Positions are written by the next run.' : 'No closed position yet.'}</div>
      ) : (
        <div className="card-b flush">
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th title="the day the position was played (a week position: its week's close); Sold says when it was sold">Day</th>
                  <th>{cap(noun || 'coin')}</th>
                  <th className="hide-sm">Model</th>
                  <th className="hide-sm">Sold</th>
                  <th className="num">Move</th>
                  <th className="num">Profit</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const cls = signClass(r.pnl);
                  return (
                    <tr key={`${r.closed}-${r.symbol}-${r.game}-${i}`}>
                      <td className="mut">{day(r.closed)}</td>
                      <td className="sym">
                        {r.symbol || DASH}
                        <span className="tag">{r.game || DASH}</span>
                      </td>
                      <td className="hide-sm">{r.model ? <span className="chip model">{short(r.model)}</span> : <span className="mut">{DASH}</span>}</td>
                      <td className="hide-sm mut">{r.exit || DASH}</td>
                      <td className={`num ${signClass(r.movePct)}`}>{pct(r.movePct)}</td>
                      <td className={`num strong ${cls}`}>{money(r.pnl)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {closed.length > NEWEST ? (
        <div className="card-f">
          <button type="button" className="linkbtn more" onClick={() => setAll((v) => !v)}>
            {all ? `Show the ${NEWEST} newest` : `Show all ${closed.length}`}
          </button>
        </div>
      ) : null}
    </div>
  );
}
