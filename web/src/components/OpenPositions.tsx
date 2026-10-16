import type { Band, MarketView, OpenPosition, OpenStatus, Quotes } from '../api';
import { DASH, cap, day, int, isNum, pct, rate, time } from '../format';
import NoneChips from './NoneChips';

const STATUS_CHIP: Record<OpenStatus, string> = { 'buy now': 'act', 'buy at open': 'act', held: 'up', 'too late': 'flat' };

function bandTitle(b: Band): string {
  const parts = [isNum(b.bin) ? `bin ${b.bin}` : '', isNum(b.lowPct) && isNum(b.highPct) ? `${pct(b.lowPct)} to ${pct(b.highPct)} from the open rate` : ''].filter(Boolean);
  return `The call counts as right inside this band${parts.length ? ` (${parts.join(', ')})` : ''}.`;
}

function Row({ p, quote, liveAt }: { p: OpenPosition; quote: number | undefined; liveAt: string }) {
  // Only a held position has a live profit: a "buy now" row has nothing to measure yet, and a "too late"
  // row was sold at the close - its result comes with the morning run.
  const held = p.status === 'held';
  const live = held && isNum(quote) && isNum(p.openRate) && p.openRate > 0 ? (quote / p.openRate - 1) * 100 : null;
  const profit = p.status === 'too late' ? null : (live ?? p.profitPct ?? p.profitPctClose);
  const cls = isNum(profit) ? (profit > 0 ? 'pos' : profit < 0 ? 'neg' : 'mut') : 'mut';
  const statusText = p.statusText || '';
  const band = p.band && (isNum(p.band.low) || isNum(p.band.high)) ? p.band : null;
  return (
    <tr>
      <td className="sym">
        {p.symbol || DASH}
        {p.game === 'week' && p.since ? <span className="sub">{p.since}</span> : null}
        {p.game !== 'week' && p.modelsUp ? (
          <span className="sub">
            {int(p.modelsUp.up)}/{int(p.modelsUp.of)} models up
          </span>
        ) : null}
      </td>
      <td>
        <span className="chip flat">{p.game || DASH}</span>
      </td>
      <td>
        <span className={`chip ${STATUS_CHIP[p.status] ?? 'flat'}`} title={statusText || undefined}>
          {p.status === 'held' ? '▲ held' : p.status || DASH}
        </span>
      </td>
      <td className="num">
        <span title={p.openRateNote || undefined}>{rate(p.openRate)}</span>
        {band ? (
          <span className="sub" title={bandTitle(band)}>
            band {rate(band.low)} – {rate(band.high)}
          </span>
        ) : null}
      </td>
      <td className={`num${isNum(quote) ? '' : ' mut'}`}>
        {isNum(quote) ? (
          <>
            {rate(quote)} <span className="live">live{liveAt ? ` ${liveAt}` : ''}</span>
          </>
        ) : (
          DASH
        )}
      </td>
      <td className={`num ${cls}`}>
        {pct(profit)}
        {live !== null && isNum(p.profitPctClose) ? <span className="sub">{pct(p.profitPctClose)} at the close</span> : null}
        {p.status === 'too late' ? <span className="sub">settled after the morning run</span> : null}
      </td>
      <td>
        {p.sellText ? <span className="order">{p.sellText}</span> : <span className="mut">{DASH}</span>}
        {p.guard && p.guard.state === 'sell' ? (
          <span className="sub show-sm" title={p.guard.reason || undefined}>
            {p.guard.text}
          </span>
        ) : null}
      </td>
      <td className="hide-sm">
        {p.guard ? (
          <span className={`chip ${p.guard.state === 'sell' ? 'down' : 'up'}`} title={p.guard.reason || undefined}>
            {p.guard.text || p.guard.state}
          </span>
        ) : (
          <span className="mut">{DASH}</span>
        )}
      </td>
    </tr>
  );
}

export default function OpenPositions({ data, quotes }: { data: MarketView; quotes: Quotes | null }) {
  const { day: d, week: w, open, none, notes, instruments, noun, generatedAt } = data;
  const liveAt = quotes?.at ? time(quotes.at) : '';
  const state = [d?.stateText, w?.stateText].filter(Boolean).join(' · ');
  const stateTitle = [d?.stateTitle, w?.stateTitle].filter(Boolean).join(' ');
  const short = (m: string | null | undefined) => (m ? m.replace(/ Model$/, '') : '');
  const models = [short(d?.model), short(w?.model)].filter(Boolean).join(' · ');
  const modelTitle = [
    d ? `Day game: ${d.model ?? 'no model'}${isNum(d.modelCount) ? ` of ${d.modelCount}` : ''}${d.ruleName ? ` (${d.ruleName})` : ''}.` : '',
    w ? `Week game: ${w.model ?? 'no model'}${isNum(w.modelCount) ? ` of ${w.modelCount}` : ''}${w.ruleName ? ` (${w.ruleName})` : ''}.` : '',
    'The pick can change.',
  ]
    .filter(Boolean)
    .join(' ');
  const closeLines = [d?.closeText ? `Day game: ${d.closeText}` : '', w?.closeText ? `Week game: ${w.closeText}` : ''].filter(Boolean);

  return (
    <div className="card">
      <h2>
        Open positions
        <small>
          <span title={stateTitle || undefined}>{state}</span>
          {models ? (
            <>
              {state ? ' · ' : ''}calls by{' '}
              <span className="chip model" title={modelTitle}>
                {models}
              </span>
            </>
          ) : null}
        </small>
      </h2>
      <div className="scroll" style={{ marginTop: 8 }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>{cap(noun || 'coin')}</th>
              <th>Game</th>
              <th>Status</th>
              <th className="num">Open rate</th>
              <th className="num">Now</th>
              <th className="num">Profit</th>
              <th>Sell</th>
              <th className="hide-sm" title={notes.guardText || notes.guard || undefined}>
                Daily call
              </th>
            </tr>
          </thead>
          <tbody>
            {open.length === 0 ? (
              <tr className="none">
                <td colSpan={8}>No open position.</td>
              </tr>
            ) : (
              open.map((p, i) => <Row key={`${p.symbol}-${p.game}-${i}`} p={p} quote={quotes?.quotes[p.symbol]} liveAt={liveAt} />)
            )}
          </tbody>
        </table>
      </div>
      <NoneChips label="No day position today" calls={none.day} total={instruments.length} />
      <NoneChips label="No week position this week" calls={none.week} total={instruments.length} />
      {notes.entry || notes.sell || notes.guard ? (
        <div className="pill-row">
          {notes.entry ? (
            <span className={`chip ${data.market === 'crypto' ? 'warn' : 'flat'}`} title={notes.entryText || notes.entry}>
              {notes.entry}
            </span>
          ) : null}
          {notes.sell ? (
            <span className="chip flat" title={notes.sellText || notes.sell}>
              {notes.sell}
            </span>
          ) : null}
          {notes.guard ? (
            <span className="chip flat" title={notes.guardText || notes.guard}>
              {notes.guard}
            </span>
          ) : null}
        </div>
      ) : null}
      <details className="help">
        <summary>how to read this</summary>
        <p>
          A day position is a buy when you read this and a sell at the close shown in the Sell column. {notes.closeDay || ''} A week position is
          bought at the week's start and sold at its end. "Now" is a live quote, information only; every booked result uses the close.
        </p>
        {notes.entryText ? <p>{notes.entryText}</p> : d?.entryNote ? <p>{d.entryNote}</p> : null}
        {notes.sellText ? <p>{notes.sellText}</p> : null}
        {notes.guardText ? <p>{notes.guardText}</p> : null}
        {closeLines.length ? <p>{closeLines.join(' · ')}</p> : null}
        {generatedAt ? (
          <p>
            Plan generated {day(generatedAt)} {time(generatedAt)} (Belgian time).
          </p>
        ) : null}
      </details>
    </div>
  );
}
