import { useState } from 'react';
import type { ClosedPosition } from '../api';
import { DASH, cap, day, money, pct, rate, signClass } from '../format';

const NEWEST = 10;
// an order position's exit (api.js orderExitText): its kind as a chip, then when
const EXIT_CHIP: Record<string, { label: string; chip: string }> = {
  limit: { label: 'limit filled', chip: 'up' }, 'next card': { label: 'next card', chip: 'flat' }, early: { label: 'daily call not up', chip: 'down' },
  fallback: { label: 'fallback time', chip: 'flat' }, close: { label: '00:00 UTC', chip: 'flat' },
};
const EXIT_PREFIX = /^(limit filled|daily call not up, at market|next card, at market|no next card, at market \(fallback\)|00:00 UTC close) ?/;
function ExitCell({ r }: { r: ClosedPosition }) {
  const k = r.exitKind ? EXIT_CHIP[r.exitKind] : null;
  if (!k) return <>{r.exit || DASH}</>;
  const title = [r.entryPrice ? `bought at ${rate(r.entryPrice)}` : '', r.exitPrice ? `sold at ${rate(r.exitPrice)}` : '', r.estimated ? 'the card time is estimated (15 min after the ticket)' : '']
    .filter(Boolean)
    .join(' · ');
  return (
    <span className="ord-exit" title={title || undefined}>
      <span className={`chip ${k.chip}`}>{k.label}</span> {r.exit.replace(EXIT_PREFIX, '')}
    </span>
  );
}

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
                      <td className="hide-sm mut">
                        <ExitCell r={r} />
                      </td>
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
