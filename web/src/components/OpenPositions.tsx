import type { Band, Guard, MarketView, OpenPosition, OpenStatus, Quotes } from '../api';
import { DASH, cap, day, int, isNum, pct, rate, time } from '../format';
import NoneChips from './NoneChips';

const STATUS_CHIP: Record<OpenStatus, string> = { 'buy now': 'act', 'buy at open': 'act', held: 'up', 'too late': 'flat', sold: 'flat', out: 'flat' };
// The Buy column: the chip says what to do, the line under it when.
function buyCell(p: OpenPosition): { chip: string; text: string; sub: string | null } {
  const when = p.buyText && p.buyText !== DASH ? p.buyText : null;
  if (p.status === 'held') return { chip: 'Held', text: '', sub: when ? `bought ${when}` : null };
  if (p.status === 'too late') return { chip: 'Too late', text: '', sub: null };
  if (p.status === 'sold') return { chip: 'Sold', text: '', sub: when ? `bought ${when}` : null };
  if (p.status === 'out') return { chip: 'Out', text: '', sub: p.since || null };
  if (when === 'now' || !when) return { chip: 'Buy now', text: '', sub: 'market order' };
  return { chip: 'Buy', text: when, sub: p.status === 'buy at open' ? 'market order at the open' : 'market order' };
}

function bandTitle(b: Band): string {
  const parts = [isNum(b.bin) ? `bin ${b.bin}` : '', isNum(b.lowPct) && isNum(b.highPct) ? `${pct(b.lowPct)} to ${pct(b.highPct)} from the entry` : ''].filter(Boolean);
  return `The call counts as right inside this band${parts.length ? ` (${parts.join(', ')})` : ''}.`;
}

// The daily call's word, glyph and colour, the same as the no-position chips.
function callOf(g: Guard): { word: string; glyph: string; chip: string } {
  const word = g.text.replace(/^daily call /, '');
  if (word === 'up') return { word: 'Up', glyph: '▲', chip: 'up' };
  if (word === 'down') return { word: 'Down', glyph: '▼', chip: 'down' };
  return { word: cap(word), glyph: '■', chip: word === 'flat' ? 'flat' : 'flat' };
}

// The Guard cell: once the guard is booked it says what it does (Buy, Hold, Sell, Out, or Week end on the week's
// last day) with the daily call under it; before that, the daily call alone, as information.
function guardOf(p: OpenPosition): { label: string; chip: string; sub: string | null } | null {
  if (!p.guard) return null;
  const g = p.guard;
  const call = callOf(g);
  if (g.booked === false) return { label: `${call.glyph} ${call.word}`, chip: call.chip, sub: null };
  const up = g.text === 'daily call up';
  switch (g.state) {
    case 'buy':
      return { label: '▲ Buy', chip: 'up', sub: g.text };
    case 'sell':
      return { label: '▼ Sell', chip: 'down', sub: g.text };
    case 'weekend':
      return { label: 'Week end', chip: 'flat', sub: g.text };
    case 'out':
      return { label: 'Out', chip: 'flat', sub: g.text };
    default:
      return { label: up ? '▲ Hold' : 'Hold', chip: up ? 'up' : 'flat', sub: g.text };
  }
}

// A sized row's position (the RL Position Model): a multiple of the paper stake
function sizeTitle(size: number, stake: number | null | undefined, unit: string): string {
  return `Position size ×${size}${isNum(stake) ? `: ${Math.round(size * stake)} ${unit} at the paper stake of ${stake} ${unit} a position` : ''}.`;
}

function Row({
  p,
  quote,
  withCall,
  models,
  stake,
  unit,
}: {
  p: OpenPosition;
  quote: number | undefined;
  withCall: boolean;
  models: { day: string; week: string };
  stake: number | null | undefined;
  unit: string;
}) {
  // Only a held position has a live move: a "buy now" row has nothing to measure yet, a "too late" or "sold"
  // row was sold at the close - its result comes with the morning run - and an "out" row holds nothing.
  const held = p.status === 'held';
  const late = p.status === 'too late' || p.status === 'sold';
  const live = held && isNum(quote) && isNum(p.openRate) && p.openRate > 0 ? (quote / p.openRate - 1) * 100 : null;
  const move = late ? null : (live ?? p.profitPct ?? p.profitPctClose);
  const cls = isNum(move) ? (move > 0 ? 'pos' : move < 0 ? 'neg' : 'mut') : 'mut';
  const band = p.band && (isNum(p.band.low) || isNum(p.band.high)) ? p.band : null;
  const coinTitle = p.game === 'week' ? p.since ?? undefined : p.modelsUp ? `${int(p.modelsUp.up)} of ${int(p.modelsUp.of)} models call it up` : undefined;
  const moveTitle = late
    ? p.status === 'sold'
      ? 'Sold at the close by the guard; its move shows after the morning run, its money when the week closes.'
      : 'Sold at the close; the result comes with the morning run.'
    : p.status === 'out'
      ? p.since?.startsWith('sold') ? 'No position held; its sale this week is under Buy.' : 'No position held this week.'
      : live !== null && isNum(p.profitPctClose)
        ? `Live; ${pct(p.profitPctClose)} at the last close.`
        : undefined;
  const call = p.guard ? callOf(p.guard) : null;
  const guard = guardOf(p);
  const buy = buyCell(p);
  const model = p.game === 'week' ? models.week : models.day;
  return (
    <tr>
      <td className="sym" title={[coinTitle, model ? `${p.game === 'week' ? 'Weekly' : 'Daily'} model: ${model}` : ''].filter(Boolean).join(' · ') || undefined}>
        {p.symbol || DASH}
        <span className="tag">{p.game || DASH}</span>
      </td>
      <td title={p.statusText || undefined}>
        <span className={`chip ${STATUS_CHIP[p.status] ?? 'flat'}`}>{buy.chip}</span>
        {isNum(p.size) ? (
          <span className="strong" title={sizeTitle(p.size, stake, unit)}>
            {' '}
            ×{p.size}
          </span>
        ) : null}
        {buy.text ? <span className="strong"> {buy.text}</span> : null}
        {buy.sub ? <span className="sub">{buy.sub}</span> : null}
        {p.earlier ? <span className="sub">{p.earlier}</span> : null}
      </td>
      <td>
        {late ? <span className="mut">sold at the close</span> : p.sellText && p.sellText !== DASH ? <span className="strong">{p.sellText}</span> : <span className="mut">{DASH}</span>}
        {p.sellNote && !late ? <span className="sub">{p.sellNote}</span> : null}
        {call && p.guard && !late && p.status !== 'out' && (p.guard.state === 'sell' || p.guard.state === 'weekend') ? (
          <span className="sub show-sm" title={p.guard.reason || undefined}>
            {p.guard.state === 'weekend' ? 'week end' : `${p.guard.booked === false ? 'info' : 'guard'} · ${p.guard.text}`}
          </span>
        ) : null}
      </td>
      {withCall ? (
        <td className="hide-sm">
          {guard && p.guard ? (
            <>
              <span className={`chip ${guard.chip}`} title={p.guard.reason || undefined}>
                {guard.label}
              </span>
              {guard.sub ? <span className="sub">{guard.sub}</span> : null}
            </>
          ) : (
            <span className="mut" title={p.game === 'week' ? "today's daily call for this coin arrives with the morning run" : 'a day position is the daily call itself'}>
              {DASH}
            </span>
          )}
        </td>
      ) : null}
      <td className="num hide-sm">
        <span title={p.openRateNote || undefined}>{rate(p.openRate)}</span>
        {band ? (
          <span className="sub" title={bandTitle(band)}>
            {rate(band.low)} – {rate(band.high)}
          </span>
        ) : null}
      </td>
      <td className={`num hide-sm${isNum(quote) ? '' : ' mut'}`}>{isNum(quote) ? rate(quote) : DASH}</td>
      <td className={`num ${cls}`} title={moveTitle}>
        <span className="strong">{pct(move)}</span>
        {isNum(quote) ? <span className="sub show-sm">now {rate(quote)}</span> : null}
      </td>
    </tr>
  );
}

export default function OpenPositions({ data, quotes }: { data: MarketView; quotes: Quotes | null }) {
  const { day: d, week: w, open, none, notes, instruments, noun, generatedAt, market } = data;
  const liveAt = quotes?.at ? time(quotes.at) : '';
  const stateTitle = [d?.stateTitle, w?.stateTitle].filter(Boolean).join(' ');
  const short = (m: string | null | undefined) => (m ? m.replace(/ Model$/, '') : '');
  // why the model is followed: its paper book, or the best exact rate while no book has the positions it needs (10 a
  // day, 5 a week: markets.js pickPlan, pickWeek)
  const pickNote = (g: { byBook?: boolean; minTrades?: number }) =>
    g.byBook === false
      ? `the best exact rate, as no book has ${g.minTrades ?? 'enough'} positions yet; the pick can change`
      : 'the best paper book of its horizon so far; the pick can change';
  const dayTitle = d ? `${d.model ?? 'no model'}${isNum(d.modelCount) ? `, one of ${d.modelCount} models` : ''}${d.ruleName ? `, ${d.ruleName}` : ''}: ${pickNote(d)}.` : '';
  const weekTitle = w ? `${w.model ?? 'no model'}${isNum(w.modelCount) ? `, one of ${w.modelCount} models` : ''}${w.ruleName ? `, ${w.ruleName}` : ''}: ${pickNote(w)}.` : '';
  const models = { day: short(d?.model), week: short(w?.model) };
  // the daily-call column whenever there is a week position: it is what the week position is watched by
  const withCall = open.some((p) => p.guard || p.game === 'week');
  const unbooked = open.some((p) => p.guard && p.guard.booked === false);

  return (
    <div className="card span-3">
      <div className="card-h">
        <h2>Open positions</h2>
        <div className="meta dots" title={stateTitle || undefined}>
          {d?.stateText ? <span>{d.stateText}</span> : null}
          {w?.stateText ? <span>{w.stateText}</span> : null}
        </div>
      </div>
      <div className="models">
        {d?.model ? (
          <span className="model-pill" title={dayTitle}>
            <span className="tag">day</span> model <b>{models.day}</b>
          </span>
        ) : null}
        {w?.model ? (
          <span className="model-pill" title={weekTitle}>
            <span className="tag">week</span> model <b>{models.week}</b>
          </span>
        ) : null}
      </div>
      <div className="card-b flush">
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>{cap(noun || 'coin')}</th>
                <th>Buy</th>
                <th>Sell</th>
                {withCall ? (
                  <th className="hide-sm" title={notes.guardText || notes.guard || undefined}>
                    Guard{unbooked ? ' (info)' : ''}
                  </th>
                ) : null}
                <th className="num hide-sm" title="the rate the position is measured from, and under it the band where the call counts as right">
                  Entry · band
                </th>
                <th className="num hide-sm">
                  Now{liveAt ? <span className="live">live {liveAt}</span> : null}
                </th>
                <th className="num" title="the price change since the entry, live for a held position">
                  Since entry
                </th>
              </tr>
            </thead>
            <tbody>
              {open.length === 0 ? (
                <tr className="none">
                  <td colSpan={withCall ? 7 : 6}>No open position: nothing to buy and nothing held.</td>
                </tr>
              ) : (
                open.map((p, i) => (
                  <Row key={`${p.symbol}-${p.game}-${i}`} p={p} quote={quotes?.quotes[p.symbol]} withCall={withCall} models={models} stake={d?.stake} unit={data.unit} />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card-f">
        <div className="nones">
          <NoneChips label="Not traded today" calls={none.day} total={instruments.length} />
          <NoneChips
            label={w?.guarded ? 'Week call not up' : 'Not held this week'}
            title={w?.guarded ? 'the weekly model does not call these up: no week position this week' : undefined}
            calls={none.week}
            total={instruments.length}
          />
          {market === 'crypto' && notes.entry ? (
            <span className="chip warn" title={notes.entryText || notes.entry}>
              {notes.entry}
            </span>
          ) : null}
          {withCall && unbooked && notes.guard ? (
            <span className="chip flat" title={notes.guardText || notes.guard}>
              {notes.guard}
            </span>
          ) : null}
        </div>
        <details className="help">
          <summary>How to read this</summary>
          <p>
            Buy and Sell are the orders: a day position is bought when you read this and sold at the close shown. {notes.closeDay || ''} A week
            position is sold at the week's end{w?.guardOrders ? ' at the latest; the guard can sell it sooner' : ''}. Every sell is a market order at
            that time, whatever the price; the band only says when the call counts as right. "Now" is a live quote and "Since entry" is live for
            a held position: information only, every booked result uses the close.{' '}
            {w?.guardOrders
              ? 'The Guard column watches a week position with the daily call: Buy on an up call, Hold while it stays up, Sell at today\'s close when it is flat or down, Out until the next up call brings it back.'
              : 'The Guard column shows the daily call next to a week position, as information.'}
          </p>
          {notes.entryText ? <p>{notes.entryText}</p> : d?.entryNote ? <p>{d.entryNote}</p> : null}
          {open.some((p) => isNum(p.size)) ? (
            <p>
              A ×N in the Buy column is the daily model&apos;s position size: N times the paper stake{isNum(d?.stake) ? ` of ${d.stake} ${data.unit}` : ''}.
            </p>
          ) : null}
          {notes.guardText ? <p>{notes.guardText}</p> : null}
          {generatedAt ? (
            <p>
              Plan generated {day(generatedAt)} {time(generatedAt)} (Belgian time).
            </p>
          ) : null}
        </details>
      </div>
    </div>
  );
}
