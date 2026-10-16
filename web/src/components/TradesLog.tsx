import type { LogEntry } from '../api';
import { DASH, day, money, signClass } from '../format';

export default function TradesLog({ log, pending }: { log: LogEntry[]; pending?: boolean }) {
  return (
    <div className="card">
      <h2>
        Trades log <small>newest first</small>
      </h2>
      {log.length === 0 ? (
        <p className="note">{pending ? 'Positions are written by the next run.' : 'No settled trade yet.'}</p>
      ) : (
        <div className="log" style={{ marginTop: 8 }}>
          {log.map((e, i) => (
            <span key={`${e.date}-${e.symbol}-${e.game}-${i}`} className={`t ${signClass(e.pnl)}`}>
              {e.symbol || DASH} {money(e.pnl)}{' '}
              <small>
                {day(e.date)}
                {e.game === 'week' ? ' · week' : ''}
              </small>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
