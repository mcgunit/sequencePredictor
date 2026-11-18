import type { TraderClosed } from '../api';
import { DASH, day, money, signClass } from '../format';

// The trader's newest sold positions as a short list beside the daily chart; hidden in the one-column
// layout, where Closed positions follows directly with the same rows.
export default function TradesLog({ closed, unit }: { closed: TraderClosed[]; unit?: string }) {
  return (
    <div className="card hide-1col">
      <div className="card-h">
        <h2>Latest positions</h2>
        <div className="meta">{unit || ''}</div>
      </div>
      {closed.length === 0 ? (
        <div className="card-b empty">No sold position yet.</div>
      ) : (
        <div className="card-b flush compact">
          <table className="tbl">
            <tbody>
              {closed.slice(0, 6).map((e, i) => (
                <tr key={`${e.sold}-${e.symbol}-${i}`}>
                  <td className="mut">{day(e.sold)}</td>
                  <td className="sym">{e.symbol || DASH}</td>
                  <td className={`num strong ${signClass(e.pnl)}`}>{money(e.pnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
