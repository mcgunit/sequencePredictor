import type { Quotes, TraderCall, TraderCell, TraderRow, TraderView } from '../api';
import { DASH, cap, day, isNum, pct, rate, time } from '../format';

// The paper trader's card (roadmap item 11): one row per instrument for a reader who looks once a morning. The two
// instruction columns say what to do without it and what to do with it; the server writes the words (api.js traderView).

const CALL: Record<string, { glyph: string; chip: string }> = {
  up: { glyph: '▲', chip: 'up' },
  down: { glyph: '▼', chip: 'down' },
  flat: { glyph: '■', chip: 'flat' },
};

function Call({ label, call }: { label: string; call: TraderCall }) {
  const c = call ? CALL[call] : null;
  return (
    <span className={`chip ${c ? c.chip : 'flat'}`} title={`${label} prediction: ${call ?? 'none'}`}>
      {label[0].toUpperCase()} {c ? c.glyph : DASH}
    </span>
  );
}

function Cell({ c }: { c: TraderCell }) {
  return (
    <>
      {c.chip ? <span className={`chip ${c.tone || 'flat'}`}>{c.chip}</span> : c.subs.length ? null : <span className="mut">{DASH}</span>}
      {c.subs.map((t, i) => (
        <span key={i} className="sub">
          {t}
        </span>
      ))}
    </>
  );
}

function Row({ r, quote }: { r: TraderRow; quote: number | undefined }) {
  const entry = r.entry && isNum(r.entry.price) && r.entry.price > 0 ? r.entry.price : null;
  const live = entry !== null && isNum(quote) ? (quote / entry - 1) * 100 : null;
  const move = live ?? r.movePctClose;
  const cls = isNum(move) ? (move > 0 ? 'pos' : move < 0 ? 'neg' : 'mut') : 'mut';
  return (
    <tr title={r.text}>
      <td className="sym">{r.symbol}</td>
      <td>
        <Cell c={r.without} />
      </td>
      <td>
        <Cell c={r.held} />
      </td>
      <td className="hide-sm">
        <span className="calls">
          <Call label="day" call={r.calls.day} /> <Call label="week" call={r.calls.week} /> <Call label="month" call={r.calls.month} />
        </span>
      </td>
      <td className="num hide-sm">
        {r.entry ? (
          <>
            {rate(r.entry.price)}
            <span className="sub">{r.entry.dayText}</span>
          </>
        ) : (
          <span className="mut">{DASH}</span>
        )}
      </td>
      <td className={`num hide-sm${isNum(quote) ? '' : ' mut'}`}>{isNum(quote) ? rate(quote) : DASH}</td>
      <td className={`num ${cls}`} title={live !== null && isNum(r.movePctClose) ? `Live; ${pct(r.movePctClose)} at the last close.` : undefined}>
        {r.entry ? <span className="strong">{pct(move)}</span> : <span className="mut">{DASH}</span>}
      </td>
    </tr>
  );
}

export default function TraderCard({ trader, noun, unit, quotes }: { trader: TraderView; noun: string; unit: string; quotes: Quotes | null }) {
  const liveAt = quotes?.at ? time(quotes.at) : '';
  const toDo = trader.rows.filter((r) => r.state === 'buy' || r.state === 'sell').length;
  return (
    <div className="card span-3">
      <div className="card-h">
        <h2>Trader</h2>
        <div className="meta dots">
          <span>{trader.stateText}</span>
          {trader.past ? <span>the next instructions come with the next run</span> : <span>{toDo ? `${toDo} to do` : 'nothing to do'}</span>}
        </div>
      </div>
      <div className="card-b flush">
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>{cap(noun || 'coin')}</th>
                <th>If you do not have it</th>
                <th>If you have it</th>
                <th className="hide-sm" title="the day, week and month predictions the decision reads">
                  Predictions
                </th>
                <th className="num hide-sm" title="the price and day the trader bought at">
                  Bought
                </th>
                <th className="num hide-sm">
                  Now{liveAt ? <span className="live">live {liveAt}</span> : null}
                </th>
                <th className="num" title="the price change since the trader bought, live when a quote is shown">
                  Since bought
                </th>
              </tr>
            </thead>
            <tbody>
              {trader.rows.map((r) => (
                <Row key={r.symbol} r={r} quote={quotes?.quotes[r.symbol]} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card-f stack">
        <div className="plines">
          <p>
            <span className="lbl">The trader&apos;s record so far (paper money, long positions).</span> {trader.lines.record}
          </p>
          {trader.lines.hold ? (
            <p>
              <span className="lbl">Against holding.</span> {trader.lines.hold}
            </p>
          ) : null}
          {trader.lines.random ? (
            <p>
              <span className="lbl">Against chance.</span> {trader.lines.random}
            </p>
          ) : null}
        </div>
        <details className="help">
          <summary>How to read this</summary>
          {trader.help.map((t, i) => (
            <p key={i}>{t}</p>
          ))}
          <p>
            {unit ? `Amounts are in ${unit}. ` : ''}&quot;Now&quot; is a live quote and &quot;Since bought&quot; is live while a quote is shown: information only, the record uses its own prices.
            {trader.generatedAt ? ` Decided ${day(trader.generatedAt)} ${time(trader.generatedAt)} (Belgian time).` : ''}
          </p>
        </details>
      </div>
    </div>
  );
}
