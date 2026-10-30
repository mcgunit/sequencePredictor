import { useState } from 'react';
import { Card, Scroll } from '../../../components/admin/Card';
import { Failed, Loading } from '../../../components/admin/States';
import { DASH, day, isNum, money, pct, rate, short, stamp } from '../../../format';
import { useLoad } from '../../../hooks/useLoad';
import { getOrderDetail, type OrderRow, type OrderStats, type OrdersGame, type Workbench } from '../../../workbench';

// The crypto order book (src/MarketOrders.py, 30 Oct 2026; workbench.js ordersOf): what the day and week cards tell a
// reader to do, booked at the card's publication with a sell limit. One card per game: the limit variants side by
// side (the main book, the literal 00:00 UTC reference, the week's late ledger with its refusals and the day's newcomer
// ledger - 31 Oct 2026: a roll card's buy for the reader who did not buy before, its own record) against
// buying every coin at the card, the variant each card showed with the log and the review rule, the intraday coverage,
// and folded the per-order tables (their own route). The user pages show it from shownFrom on; until then it is the
// shadow next to the current card (Today's plan and Week game carry the lines it would show).

const skipsText = (s: Record<string, number>) => {
  const parts = Object.entries(s).map(([k, n]) => `${n} ${k}`);
  return parts.length ? parts.join(', ') : 'none';
};
const VARIANT_NAME: Record<string, string> = { edge: 'edge (the bin’s lower edge)', mid: 'mid (the bin’s middle)', fixed: 'fixed (+1 % a day, +4 % a week)' };

function StatsCells({ s, cur }: { s: OrderStats | null; cur: string }) {
  if (!s) {
    return (
      <>
        <td className="num mut">{DASH}</td>
        <td className="num mut hide-sm">{DASH}</td>
        <td className="num mut hide-sm">{DASH}</td>
        <td className="num mut hide-sm">{DASH}</td>
        <td className="num mut">{DASH}</td>
      </>
    );
  }
  return (
    <>
      <td className="num" title={`${s.closed} closed, ${s.open} open, ${s.pending} pending; ${s.estimated} with an estimated card time`}>
        {s.orders}
      </td>
      <td className="num hide-sm">{s.filled}</td>
      <td className="num hide-sm">{s.timeStop}</td>
      <td className="num hide-sm">{s.early}</td>
      <td className={`num strong ${isNum(s.total) ? (s.total > 0 ? 'pos' : s.total < 0 ? 'neg' : '') : 'mut'}`} title={`${money(s.total)} ${cur} over the closed orders; ${s.rolls} rolls`}>
        {money(s.total)}
      </td>
    </>
  );
}

function GameCard({ g, w }: { g: OrdersGame; w: Workbench }) {
  const cur = w.currency;
  const week = g.game === 'week';
  const newcomer = !week && g.variants.some((v) => v.newcomer);
  const bench = g.benchmark.stats;
  const runs = g.shown.map((r) => `${r.variant} on ${r.cards} ${week ? (r.cards === 1 ? 'week card' : 'week cards') : r.cards === 1 ? 'card' : 'cards'} (${r.from === r.to ? day(r.from) : `${day(r.from)} – ${day(r.to)}`})`);
  const rec = g.record;
  return (
    <Card
      id={`orders-${g.game}`}
      title={
        <>
          Order book<span className="q2">{week ? 'the week card' : 'the day card'}</span>
        </>
      }
      help={`${g.rule}.${g.lateRule ? ` Late entries: ${g.lateRule}.` : ''}${g.newcomerRule ? ` Newcomer entries: ${g.newcomerRule}.` : ''} Every order is bought at the open of the first minute bar at or after the card's publication (estimated at ${g.constants.publishLagFallbackMin ?? 15} min after the ticket before 30 Oct); a limit fills at the first bar whose high reaches it plus ${isNum(g.constants.fillThrough) ? (g.constants.fillThrough * 100).toFixed(2) : '0.05'} %; no order when the limit is under ${isNum(g.constants.minGap) ? (g.constants.minGap * 100).toFixed(2) : '0.30'} % above the price.`}
      meta={[
        g.since ? `since ${day(g.since)}` : '',
        `primary ${g.primary}`,
        g.shownFrom ? `users from ${day(g.shownFrom)}` : 'shadow: not on the user page yet',
        g.generatedAt ? `record ${stamp(g.generatedAt)}` : '',
      ]}
      flush
    >
      <div className="card-b">
        <div className="plines">
          <p>
            <span className="lbl">The user line.</span> Per card the variant it showed - {runs.length ? runs.join('; ') : 'no card yet'}:{' '}
            {rec ? (
              <>
                <b>{rec.orders}</b> {rec.orders === 1 ? 'order' : 'orders'}, {rec.filled} filled at the limit, {rec.timeStop} at the {week ? "week's end" : 'next card'}
                {week ? `, ${rec.early} on a flat or down daily card` : ''}: <b>{money(rec.total)} {cur}</b> on paper
              </>
            ) : (
              'no order yet'
            )}
            ; buying every coin at the {week ? 'week card' : 'card'} {money(bench?.total ?? null)} {cur} over {bench?.orders ?? 0} positions. A mechanics record while it is this short, not evidence.
          </p>
          <p>
            <span className="lbl">Variant log.</span>{' '}
            {g.variantLog.length ? g.variantLog.map((l) => `${day(l.date)}: ${l.variant} (${l.reason})`).join('; ') : 'empty'}. {g.variants.length} variants tracked.{' '}
            <span className="mut">Review: {g.review}.</span>
          </p>
          {g.missing.length ? (
            <p className="warn">
              <span className="lbl">Coverage.</span> {g.missing.length} {g.missing.length === 1 ? 'card was' : 'cards were'} published outside the stored hour bars ({g.missing.map((d) => day(d)).join(', ')}): their orders stay pending until the bars are fetched.
            </p>
          ) : null}
        </div>
      </div>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Variant</th>
              <th className="num">Orders</th>
              <th className="num hide-sm">Filled</th>
              <th className="num hide-sm">{week ? "Week's end" : 'Next card'}</th>
              <th className="num hide-sm">Early</th>
              <th className="num">{cur}</th>
              <th className="num" title="the literal end the owner first named: 00:00 UTC, the bin's own horizon - no roll; Admin only">
                00:00 UTC end
              </th>
              {week ? <th className="num">Late {cur}</th> : null}
              {newcomer ? (
                <th className="num" title="the newcomer ledger: a roll card's buy for the reader who did not buy before; its own record, never in the user line">
                  Newcomer {cur}
                </th>
              ) : null}
              <th className="hide-sm">No order</th>
            </tr>
          </thead>
          <tbody>
            {g.variants.map((v) => (
              <tr key={v.variant}>
                <td title={v.text}>
                  <b>{VARIANT_NAME[v.variant] ?? v.variant}</b>
                  {v.primary ? <span className="chip act">primary</span> : null}
                </td>
                <StatsCells s={v.main} cur={cur} />
                <td className="num" title={v.reference ? `${v.reference.orders} orders, ${v.reference.filled} filled, ${v.reference.timeStop} sold at 00:00 UTC` : undefined}>
                  {money(v.reference?.total ?? null)}
                </td>
                {week ? (
                  <td className="num" title={v.late ? `${v.late.orders} shown (${v.late.open} open); refused: ${skipsText(v.late.refused)}` : undefined}>
                    {money(v.late?.total ?? null)}
                  </td>
                ) : null}
                {newcomer ? (
                  <td className="num" title={v.newcomer ? `${v.newcomer.orders} entries (${v.newcomer.open} open, ${v.newcomer.filled} filled, ${v.newcomer.timeStop} at the next card, ${v.newcomer.rolls} rolls); not placed: ${skipsText(v.newcomer.skips)}` : undefined}>
                    {money(v.newcomer?.total ?? null)}
                  </td>
                ) : null}
                <td className="hide-sm mut">{v.main ? skipsText(v.main.skips) : DASH}</td>
              </tr>
            ))}
            <tr className="grey">
              <td>buying every coin at the card</td>
              <StatsCells s={bench} cur={cur} />
              <td className="num">{DASH}</td>
              {week ? <td className="num">{DASH}</td> : null}
              {newcomer ? <td className="num">{DASH}</td> : null}
              <td className="hide-sm">no limit, sold at the same deadline</td>
            </tr>
          </tbody>
        </table>
      </Scroll>
      {week && g.variants.some((v) => v.late) ? (
        <div className="card-b">
          <p className="line mut">
            Late entries (a separate ledger, never in the user line): refused {skipsText(g.variants[0].late?.refused ?? {})} in the {g.primary} variant.
          </p>
        </div>
      ) : null}
      {newcomer ? (
        <div className="card-b">
          <p className="line mut">
            Newcomer entries (a separate ledger, never in the user line): {g.variants.map((v) => `${v.variant} ${v.newcomer?.orders ?? 0}`).join(', ')}. One is booked on every card that rolls a held
            position - bought at that card&apos;s price with the same new limit and deadline - and it rolls with the position after that.
          </p>
        </div>
      ) : null}
      <div className="card-b">
        <details className="fold">
          <summary>
            Intraday coverage <span className="mut">· Binance 1h bars and 1m windows around each card</span>
          </summary>
          <Scroll>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Coin</th>
                  <th>First hour</th>
                  <th>Last hour</th>
                  <th className="num">1h bars</th>
                  <th className="num">1m bars</th>
                </tr>
              </thead>
              <tbody>
                {g.coverage.map((c) => (
                  <tr key={c.symbol}>
                    <td className="sym">{c.symbol}</td>
                    <td>{stamp(c.first1h)}</td>
                    <td>{stamp(c.last1h)}</td>
                    <td className="num">{c.bars1h ?? DASH}</td>
                    <td className="num">{c.bars1m ?? DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Scroll>
        </details>
        <OrderTables market={w.market} game={g.game} primary={g.primary} cur={cur} />
      </div>
    </Card>
  );
}

// the per-order tables, fetched when the fold is first opened
function OrderTables({ market, game, primary, cur }: { market: string; game: 'day' | 'week'; primary: string; cur: string }) {
  const [opened, setOpened] = useState(false);
  return (
    <details className="fold" onToggle={(e) => e.currentTarget.open && setOpened(true)}>
      <summary>
        Every order <span className="mut">· every variant, the 00:00 UTC reference{game === 'week' ? ', the late ledger' : ', the newcomer ledger'}</span>
      </summary>
      {opened ? <OrderTablesBody market={market} game={game} primary={primary} cur={cur} /> : null}
    </details>
  );
}

const s = (v: unknown) => (typeof v === 'string' ? v : null);
const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
function OrderTablesBody({ market, game, primary, cur }: { market: string; game: 'day' | 'week'; primary: string; cur: string }) {
  const load = useLoad(() => getOrderDetail(market, game), [market, game]);
  const [variant, setVariant] = useState(primary);
  const [ledger, setLedger] = useState<'main' | 'reference' | 'late' | 'newcomer'>('main');
  if (load.kind === 'loading') return <Loading />;
  if (load.kind === 'error') return <Failed error={load.error} retry={load.retry} />;
  const d = load.data;
  const variants = Object.keys(d.orders);
  const src = ledger === 'main' ? d.orders : ledger === 'reference' ? d.reference : ledger === 'late' ? d.late ?? {} : d.newcomer ?? {};
  const rows: OrderRow[] = src[variant] ?? [];
  return (
    <>
      <div className="segbar">
        <div className="seg" role="group" aria-label="variant">
          {variants.map((v) => (
            <button key={v} type="button" aria-pressed={v === variant} onClick={() => setVariant(v)}>
              {v}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="ledger">
          {(['main', 'reference', ...(d.late ? ['late'] : []), ...(d.newcomer ? ['newcomer'] : [])] as ('main' | 'reference' | 'late' | 'newcomer')[]).map((l) => (
            <button key={l} type="button" aria-pressed={l === ledger} onClick={() => setLedger(l)}>
              {l === 'main' ? 'next-card end' : l === 'reference' ? '00:00 UTC end' : l === 'late' ? 'late entries' : 'newcomer entries'}
            </button>
          ))}
        </div>
        <span className="note mut">{d.detailFrom ? `from ${day(d.detailFrom)}, newest first` : ''}</span>
      </div>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>{game === 'week' ? 'Week ending' : 'Day'}</th>
              <th>Coin</th>
              <th>Card</th>
              <th className="num">P</th>
              <th className="num">X</th>
              <th className="num hide-sm">X vs P</th>
              <th>Status</th>
              <th className="hide-sm">Exit</th>
              <th className="num hide-sm">Exit price</th>
              <th className="num">{cur}</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr className="none">
                <td colSpan={10}>No order in this ledger.</td>
              </tr>
            ) : (
              rows.map((r, i) => {
                const status = s(r.status) ?? DASH;
                const money4 = n(r.money);
                return (
                  <tr key={i} className={r.live === false ? 'grey' : undefined}>
                    <td>
                      {day(r.interval)}
                      {n(r.days) && (n(r.days) ?? 0) > 1 && game === 'day' ? <span className="sub">{n(r.days)} days (rolled)</span> : null}
                    </td>
                    <td className="sym">
                      {s(r.symbol)}
                      <span className="sub">{short(s(r.model) ?? '')} · bin {n(r.bin) ?? DASH}</span>
                    </td>
                    <td>
                      {stamp(s(r.publishedAt))}
                      {r.estimated ? <span className="sub">estimated</span> : null}
                    </td>
                    <td className="num">{rate(n(r.entryPrice))}</td>
                    <td className="num">{rate(n(r.limit))}</td>
                    <td className="num hide-sm">{pct(n(r.limitPct), 2)}</td>
                    <td>
                      {status}
                      {s(r.reason) ? <span className="sub">{s(r.reason)}</span> : null}
                      {r.provisional ? <span className="sub">provisional</span> : null}
                    </td>
                    <td className="hide-sm">
                      {s(r.exitKind) ?? DASH}
                      {s(r.exitAt) ? <span className="sub">{stamp(s(r.exitAt))}</span> : null}
                    </td>
                    <td className="num hide-sm">{rate(n(r.exitPrice))}</td>
                    <td className={`num strong ${isNum(money4) ? (money4 > 0 ? 'pos' : money4 < 0 ? 'neg' : '') : 'mut'}`}>{money(money4)}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </Scroll>
      {ledger === 'main' && d.benchmark.length ? (
        <p className="line mut">
          Buying every coin at the card, per {game === 'week' ? 'week' : 'day'}: {d.benchmark.map((b) => `${day(b.interval)} ${money(b.money)}${b.closed ? '' : ' (open)'}`).join(' · ')}.
        </p>
      ) : null}
    </>
  );
}

export default function OrdersCard({ w }: { w: Workbench }) {
  const o = w.orders;
  if (!o || (!o.day && !o.week)) return null;
  return (
    <>
      {o.day ? <GameCard g={o.day} w={w} /> : null}
      {o.week ? <GameCard g={o.week} w={w} /> : null}
    </>
  );
}
