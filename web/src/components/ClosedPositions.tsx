import { useState } from 'react';
import type { ClosedPosition } from '../api';
import { DASH, cap, day, money, pct, signClass } from '../format';

const NEWEST = 10;

export default function ClosedPositions({ closed, noun }: { closed: ClosedPosition[]; noun: string }) {
  const [all, setAll] = useState(false);
  const rows = all ? closed : closed.slice(0, NEWEST);
  return (
    <div className="card">
      <h2>
        Closed positions <small>newest first</small>
      </h2>
      <div className="scroll" style={{ marginTop: 8 }}>
        <table className="tbl">
          <thead>
            <tr>
              <th title="the day the position was played; the exit column says when it closed">Day</th>
              <th>{cap(noun || 'coin')}</th>
              <th>Game</th>
              <th className="num">Move</th>
              <th className="num">Profit</th>
              <th className="hide-sm">Exit</th>
              <th className="hide-sm">Model</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr className="none">
                <td colSpan={7}>No closed position yet.</td>
              </tr>
            ) : (
              rows.map((r, i) => {
                const cls = signClass(r.pnl);
                return (
                  <tr key={`${r.closed}-${r.symbol}-${r.game}-${i}`}>
                    <td>{day(r.closed)}</td>
                    <td className="sym">{r.symbol || DASH}</td>
                    <td>
                      <span className="chip flat">{r.game || DASH}</span>
                    </td>
                    <td className={`num ${cls}`}>{pct(r.movePct)}</td>
                    <td className={`num ${cls}`}>{money(r.pnl)}</td>
                    <td className="hide-sm">{r.exit ? <span className="chip flat">{r.exit}</span> : <span className="mut">{DASH}</span>}</td>
                    <td className="hide-sm">{r.model ? <span className="chip model">{r.model}</span> : <span className="mut">{DASH}</span>}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      {closed.length > NEWEST ? (
        <button type="button" className="linkbtn more" onClick={() => setAll((v) => !v)}>
          {all ? `show the ${NEWEST} newest` : `show all ${closed.length} closed positions`}
        </button>
      ) : null}
    </div>
  );
}
