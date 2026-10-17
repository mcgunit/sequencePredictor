import type { LogEntry } from '../api';
import { DASH, day, money, signClass } from '../format';

// The newest closed positions as a short list beside the daily chart; hidden in the one-column
// layout, where Closed positions follows directly with the same rows.
export default function TradesLog({ log, pending, unit }: { log: LogEntry[]; pending?: boolean; unit?: string }) {
  return (
    <div className="card hide-1col">
      <div className="card-h">
        <h2>Latest positions</h2>
        <div className="meta">{unit || ''}</div>
      </div>
      {log.length === 0 ? (
        <div className="card-b empty">{pending ? 'Positions are written by the next run.' : 'No settled trade yet.'}</div>
      ) : (
        <div className="card-b flush compact">
          <table className="tbl">
            <tbody>
              {log.slice(0, 6).map((e, i) => (
                <tr key={`${e.date}-${e.symbol}-${e.game}-${i}`}>
                  <td className="mut">{day(e.date)}</td>
                  <td className="sym">
                    {e.symbol || DASH}
                    <span className="tag">{e.game}</span>
                  </td>
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
