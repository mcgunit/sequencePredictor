import { Card, Scroll } from '../../../components/admin/Card';
import { DASH, day, dayYear, isNum, money, pct, rate, share, short, stamp, time, unsigned } from '../../../format';
import type { BookLine, Call, OrderShadow, Plan, RecordMix, SoFar, WeekCard, Workbench } from '../../../workbench';

// Today's plan and the week game in their admin detail (C2, C3). Each card opens with its labelled lines (the 13 Oct
// plan-card rule): one model's calls - whose they are and why that model; following this card so far - what doing
// what the card said made, live days only; the sell. The counts on those lines are the ones in the tables beside
// them, and each caveat sits on the line whose number it qualifies.

export const ahead = (v: number | null) => (!isNum(v) ? '' : v >= 0 ? `${unsigned(v)} ahead of it` : `${unsigned(v)} behind it`);
const amount = (v: number | null, cur: string) => `${money(v)} ${cur}`;

export function bandText(c: Call | null): { price: string; move: string } {
  if (!c) return { price: DASH, move: '' };
  const { low, high } = c.band;
  const price = low === null && high === null ? DASH : low === null ? `below ${rate(high)}` : high === null ? `above ${rate(low)}` : `${rate(low)} – ${rate(high)}`;
  const ml = c.move.low;
  const mh = c.move.high;
  const move = ml === null && mh === null ? '' : ml === null ? `below ${pct((mh ?? 0) * 100)}` : mh === null ? `above ${pct(ml * 100)}` : `${pct(ml * 100)} to ${pct(mh * 100)}`;
  return { price, move };
}

export function callText(c: Call | null): string {
  if (!c) return 'no call';
  const glyph = c.word === 'up' ? '▲' : c.word === 'down' ? '▼' : '■';
  return `${glyph} ${c.word} · bin ${c.bin ?? DASH}${isNum(c.size) ? ` · ×${c.size}` : ''}`;
}

// a window of days, or of weeks by their ends: "4 – 21 Oct", the years shown when they differ
export function spanText(first: string | null, last: string | null): string {
  if (!first || !last) return '';
  const fmt = first.slice(0, 4) === last.slice(0, 4) ? day : dayYear;
  return first === last ? fmt(first) : `${fmt(first)} – ${fmt(last)}`;
}

// what the model's own settled days are made of in the followed record: back-built before the first live one, live,
// late. A week is dated by its end, so the weeks say so.
function mixText(r: RecordMix | null, weekly: boolean): string {
  if (!r) return '';
  const before = r.firstLive ? `${weekly ? ', ending' : ''} before ${day(r.firstLive)}` : '';
  const parts = [r.backBuilt ? `${r.backBuilt} back-built${before}` : '', r.live ? `${r.live} live` : '', r.late ? `${r.late} late` : ''].filter(Boolean);
  return parts.length ? ` (${parts.join('; ')})` : '';
}

function WhyLine({ w, book, record, models, model, minTrades, rule, unit, weekly }: { w: Workbench; book: BookLine; record: RecordMix | null; models: number; model: string; minTrades: number; rule: string; unit: string; weekly: boolean }) {
  const cur = w.currency;
  return (
    <p>
      <span className="lbl">One model&apos;s calls.</span> <b>{short(model)}</b>, one of the {models} models on {weekly ? "this week's" : "today's"} ticket, not a vote.{' '}
      {book.exact === null ? (
        <>
          Picked by its {weekly ? 'weekly ' : ''}paper book, the best under the {rule} rule among the rows with at least {minTrades} positions:{' '}
          <b>{amount(book.total, cur)}</b> from {book.positions ?? DASH} positions of {book.stake ?? DASH} {cur} over its {book.settled ?? DASH} settled {unit}
          {book.firstDay ? `, ${weekly ? 'the weeks ending ' : ''}${spanText(book.firstDay, book.lastDay)}` : ''}
          {mixText(record, weekly)}; buying every {w.noun} under the same rule over the same {unit} {amount(book.market, cur)} ({ahead(book.ahead)}).
        </>
      ) : (
        <>
          Picked by its exact rate, <b>{share(book.exact)}</b> over its {book.settled ?? DASH} scored {unit}
          {book.firstDay ? `, ${weekly ? 'the weeks ending ' : ''}${spanText(book.firstDay, book.lastDay)}` : ''}: no row has a book of {minTrades} positions yet.
        </>
      )}{' '}
      The model&apos;s record, not a reader&apos;s result; {weekly ? 'the model is picked again when the week settles' : 'the pick can change tomorrow'}.
    </p>
  );
}

function SoFarLine({ s, w, weekly, caveat }: { s: SoFar | null; w: Workbench; weekly: boolean; caveat: string }) {
  const unit = weekly ? 'weeks' : 'days';
  return (
    <p>
      <span className="lbl">Following this card so far.</span>{' '}
      {s && s.liveDays ? (
        <>
          {s.liveDays} live {s.liveDays === 1 ? unit.slice(0, -1) : unit}
          {weekly
            ? ` (${s.liveDays === 1 ? 'ending' : 'the first ending'} ${day(s.firstLive)}${s.lastLive && s.lastLive !== s.firstLive ? `, the last ending ${day(s.lastLive)}` : ''})`
            : ` since ${day(s.firstLive)}`}
          , {s.positions ?? DASH} positions: <b>{amount(s.total, w.currency)}</b> on paper; buying everything{' '}
          {amount(s.market, w.currency)} ({ahead(s.ahead)}). {caveat}
          {weekly ? ' A weekly book needs months of live weeks before it says anything about skill.' : ''} The {weekly ? 'week by week' : 'day by day'} is under Following the plan.
        </>
      ) : (
        <>not followed live yet.</>
      )}
    </p>
  );
}

// The order card's lines as the user card would say them now (api.js, from the reader's side), next to the current card:
// a shadow while ORDERS_SHOWN_FROM is not set. "Following this card so far" counts from the order book's own start.
// Folded, as every collapsible; the book's own code is shown after each coin in grey.
export function OrderShadowBlock({ sh, w, weekly }: { sh: OrderShadow | null | undefined; w: Workbench; weekly: boolean }) {
  if (!sh) return null;
  const from = w.orders?.[weekly ? 'week' : 'day']?.since ?? null;
  const shown = w.orders?.shownFrom ?? null;
  return (
    <details className="fold">
      <summary>
        The order card: what the user card would say now <span className="mut">· {shown ? `shown to users from ${day(shown)}` : 'a shadow, not shown to users yet'} · variant {sh.variant}</span>
      </summary>
      <div className="plines">
        <p>
          <span className="lbl">Following this card so far (order rule{from ? `, since ${day(from)}` : ''}).</span> {sh.soFar.text}
        </p>
        {sh.rows.map((r) => (
          <p key={r.symbol}>
            <span className="lbl">
              {r.symbol} · {r.chip}
            </span>{' '}
            <span className="mut">({r.status})</span>{' '}
            {r.branches.map((b, i) => (
              <span key={i} className={b.belowEntry ? 'warn' : undefined}>
                {b.who ? <b>{b.who}:</b> : null} {b.text}{' '}
              </span>
            ))}
          </p>
        ))}
        {sh.none.length ? <p className="mut">Call not up, no order: {sh.none.map((n) => `${n.symbol} (${n.call})`).join(', ')}.</p> : null}
        {sh.newcomer ? (
          <p>
            <span className="lbl">Newcomer entries so far.</span> {sh.newcomer.text.replace(/^Newcomer entries so far \(a separate record\): /, 'A separate record: ')}
          </p>
        ) : null}
      </div>
    </details>
  );
}

const PLAN_ORDER: Record<string, (p: Plan, c: Call | null) => string> = {
  'no-call': (p) => `no call from ${short(p.model)}`,
  'no-buy': (_p, c) => `no buy: the call is ${c?.word ?? 'not up'}`,
  'too-late': () => 'too late: the day has closed',
  'buy-now': () => 'buy now, a market order',
  'buy-at-open': (p) => `buy at the open, ${time(p.openAt)}, a market order`,
};

export function PlanCard({ w }: { w: Workbench }) {
  const p = w.plan;
  if (!p) {
    return (
      <Card id="plan" title="Today's plan">
        <div className="empty">No plan: no model has a call for the next {w.market === 'crypto' ? 'day' : 'trading day'} yet.</div>
      </Card>
    );
  }
  const crypto = w.market === 'crypto';
  const caveat = crypto
    ? 'The book buys at 00:00 UTC, before the ticket is up: a real entry is later and at another price.'
    : 'Bought at the open and sold at the close, as a reader can.';
  return (
    <Card
      id="plan"
      title={
        <>
          Today&apos;s plan<span className="q2">for {stamp(p.day)}</span>
        </>
      }
      help={`The calls of one model, the one the users' ${crypto ? 'Crypto' : 'Shares'} page follows: the best paper book under the ${p.rule} rule among the rows with at least ${p.minTrades} positions, else the best exact rate. Up is a bin in the upper half; the band is the predicted bin as prices, the range the close must land in for the call to be right. The last column is context, not a vote.`}
      meta={[p.status, p.madeAt ? `ticket ${stamp(p.madeAt)}` : '', p.calendarKnown ? '' : 'a trading day as far as the calendar is published']}
      flush
    >
      <div className="card-b">
        <div className="plines">
          <WhyLine w={w} book={p.book} record={p.record} models={p.models} model={p.model} minTrades={p.minTrades} rule={p.rule} unit="days" weekly={false} />
          <SoFarLine s={p.soFar} w={w} weekly={false} caveat={caveat} />
          <p>
            <span className="lbl">Sell.</span> At the close, whatever the price: <b>{stamp(p.closeAt)}</b> Brussels time{crypto ? ' (00:00 UTC)' : ''}. It is not a limit order at the
            band; the band only says when the call counts as right.
          </p>
        </div>
        <OrderShadowBlock sh={p.orders} w={w} weekly={false} />
      </div>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>{w.noun === 'coin' ? 'Coin' : 'Share'}</th>
              <th>Call</th>
              <th className="hide-sm">Band</th>
              <th>Order now</th>
              <th className="num hide-sm">Models up</th>
            </tr>
          </thead>
          <tbody>
            {p.rows.map((r) => {
              const b = bandText(r.call);
              return (
                <tr key={r.symbol}>
                  <td className="sym">
                    {r.symbol}
                    <span className="sub">last close {rate(r.base)}</span>
                  </td>
                  <td>
                    {callText(r.call)}
                    <span className="sub show-sm">
                      band {b.price} · {r.up} of {r.of} models up
                    </span>
                  </td>
                  <td className="hide-sm">
                    {b.price}
                    {b.move ? <span className="sub">{b.move}</span> : null}
                  </td>
                  <td>{(PLAN_ORDER[r.order] ?? (() => r.order))(p, r.call)}</td>
                  <td className="num hide-sm">
                    {r.up} of {r.of}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

const WEEK_ORDER: Record<string, (wk: WeekCard, c: Call | null) => string> = {
  'no-call': (wk) => `no call from ${short(wk.model)}`,
  'no-position': (_wk, c) => `no position: the call is ${c?.word ?? 'not up'}`,
  closed: () => 'the week has closed',
  'before-start': (wk) => `buy at the week's start, ${stamp(wk.openAt)}`,
  'first-day': () => "buy now: the week's first day",
  running: () => "held since the week's start; sold at the week's close",
};
const WHERE: Record<string, string> = { inside: 'inside the band', below: 'below the band', above: 'above the band' };

export function WeekGameCard({ w }: { w: Workbench }) {
  const wk = w.week;
  if (!wk) {
    return (
      <Card id="week" title="Week game">
        <div className="empty">No week ticket yet.</div>
      </Card>
    );
  }
  const crypto = w.market === 'crypto';
  const caveat = crypto ? 'The book buys at Monday 00:00 UTC, before the Monday ticket is up: a real entry is later.' : "Bought at the week's first open.";
  return (
    <Card
      id="week"
      title={
        <>
          Week game<span className="q2">week ending {stamp(wk.weekEnd)}</span>
        </>
      }
      help={`The second game: one call per ${w.noun} for the whole week, judged once at the week's close. The card follows the best weekly book under the ${wk.rule} rule among the rows with at least ${wk.minTrades} positions. Not the admin page This week.`}
      meta={[wk.status, isNum(wk.settledWeeks) ? `${wk.settledWeeks} settled weeks` : '', wk.madeAt ? `ticket ${stamp(wk.madeAt)}` : '']}
      flush
    >
      <div className="card-b">
        <div className="plines">
          <WhyLine w={w} book={wk.book} record={wk.record} models={wk.models} model={wk.model} minTrades={wk.minTrades} rule={wk.rule} unit="weeks" weekly />
          <SoFarLine s={wk.soFar} w={w} weekly caveat={caveat} />
          <p>
            <span className="lbl">Sell.</span> At the week&apos;s close, whatever the price: <b>{stamp(wk.closeAt)}</b> Brussels time. The call is judged then, once.{' '}
            {wk.guard.applies
              ? 'The user pages follow this week with the guard: sold at the close of the first flat or down daily call, at the week’s close at the latest.'
              : `The user pages count this week as the plain week; they follow the guard from the week of ${day(wk.guard.from)} (its record is under The guard).`}
          </p>
        </div>
        <OrderShadowBlock sh={wk.orders} w={w} weekly />
      </div>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>{w.noun === 'coin' ? 'Coin' : 'Share'}</th>
              <th>Call</th>
              <th className="hide-sm">Band at the week&apos;s close</th>
              <th>So far</th>
              <th className="hide-sm">Order now</th>
              <th className="num hide-sm">Models up</th>
            </tr>
          </thead>
          <tbody>
            {wk.rows.map((r) => {
              const b = bandText(r.call);
              const sf = r.soFar;
              const order = (WEEK_ORDER[r.order] ?? (() => r.order))(wk, r.call);
              return (
                <tr key={r.symbol}>
                  <td className="sym">
                    {r.symbol}
                    <span className="sub">week base {rate(r.base)}</span>
                  </td>
                  <td>
                    {callText(r.call)}
                    <span className="sub show-sm">
                      band {b.price} · {order} · {r.up} of {r.of} up
                    </span>
                  </td>
                  <td className="hide-sm">
                    {b.price}
                    {b.move ? <span className="sub">{b.move} over the week</span> : null}
                  </td>
                  <td>
                    {sf ? (
                      <>
                        {pct((sf.move ?? NaN) * 100)} after {sf.closes} {sf.closes === 1 ? 'close' : 'closes'}
                        <span className="sub">
                          {rate(sf.last)} on {day(sf.lastDate)} · {WHERE[sf.where] ?? sf.where}, judged only at the close
                        </span>
                      </>
                    ) : (
                      <span className="mut">no close yet</span>
                    )}
                  </td>
                  <td className="hide-sm">{order}</td>
                  <td className="num hide-sm">
                    {r.up} of {r.of}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}
