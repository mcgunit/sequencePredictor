import { Link } from 'react-router-dom';
import { Card, Scroll } from '../../../components/admin/Card';
import { DASH, day, isNum, money, pct, rate, share, short, stamp } from '../../../format';
import type { Workbench } from '../../../workbench';
import { drawHref } from '../GameHistoryPage';

// The rest of the old market page, each as its own card: day by day (C14), the guard on the week positions (C15),
// the regime reading (C17) and the background, how a day becomes a draw (C1, C10, C11).

const movePct = (v: number | null) => pct(isNum(v) ? v * 100 : null, 1);

// Day by day: the newest settled days as moves and bins; a day opens the draw view with every row's bins
export function DaysCard({ w }: { w: Workbench }) {
  const syms = w.days[0]?.instruments.map((i) => i.symbol) ?? w.instruments.map((i) => i.symbol);
  return (
    <Card
      id="days"
      title="Day by day"
      help="Per instrument the close-to-close move and the bin it fell in. A day opens its draw: every row's bins with the position the books take, exact, direction and P&L."
      meta={[`newest ${w.days.length} settled days`, 'chance 1 in 10 per slot']}
      flush
    >
      {w.days.length ? (
        <Scroll>
          <table className="tbl">
            <thead>
              <tr>
                <th>Day</th>
                {syms.map((s) => (
                  <th key={s} className="num hide-sm">
                    {s}
                  </th>
                ))}
                <th className="hide-sm">Most exact</th>
                <th className="num">Mean exact</th>
              </tr>
            </thead>
            <tbody>
              {w.days.map((d) => (
                <tr key={d.date}>
                  <td>
                    <Link to={drawHref(w.market, d.date)}>{day(d.date)} ›</Link>
                    <span className="sub show-sm">
                      {d.instruments.map((i) => `${i.symbol} ${movePct(i.ret)} bin ${i.bin ?? DASH}`).join(' · ')}
                      {d.best ? ` · most exact ${short(d.best)}` : ''}
                    </span>
                  </td>
                  {syms.map((s) => {
                    const i = d.instruments.find((x) => x.symbol === s);
                    return (
                      <td key={s} className="num hide-sm">
                        {i ? movePct(i.ret) : DASH}
                        <span className="sub">bin {i?.bin ?? DASH}</span>
                      </td>
                    );
                  })}
                  <td className="hide-sm">{d.best ? short(d.best) : DASH}</td>
                  <td className="num">{share(d.exactMean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Scroll>
      ) : (
        <div className="card-b">
          <div className="empty">No settled day yet.</div>
        </div>
      )}
    </Card>
  );
}

// The guard on the week positions: week by week against the plain week, over the live weeks
export function GuardCard({ w }: { w: Workbench }) {
  const g = w.guard;
  const cur = w.currency;
  if (!g) {
    return (
      <Card id="guard" title="The guard on the week positions">
        <div className="empty">No guard record yet.</div>
      </Card>
    );
  }
  return (
    <Card
      id="guard"
      title="The guard on the week positions"
      help={`The owner's rule: ${g.rule}.`}
      meta={[`runs forward from ${day(g.forwardFrom)}`, `the user pages count it from the week of ${day(g.shownFrom)}`]}
      flush
    >
      <div className="card-b">
        <p className="line">
          Over the {g.live.weeks} live weeks: the guard <b>{money(g.live.guard)}</b>, the plain week {money(g.live.plain)}, buying everything {money(g.live.market)} {cur}
          {g.live.afterTheFact ? ` · ${g.live.afterTheFact} of them booked after the fact (before the guard ran forward on ${day(g.forwardFrom)})` : ''}. The user pages
          counted the plain week before the week of {day(g.shownFrom)}: those weeks are here only.
        </p>
      </div>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Week ending</th>
              <th>Week row</th>
              <th className="num hide-sm">Positions</th>
              <th className="num">Guard {cur}</th>
              <th className="num">Plain week</th>
              <th className="num hide-sm">Buying everything</th>
              <th className="hide-sm">Note</th>
            </tr>
          </thead>
          <tbody>
            {g.weeks.map((x) => {
              const note = [`user pages: ${x.userPages}${x.live ? '' : ' (not live)'}`, x.afterTheFact ? 'booked after the fact' : ''].filter(Boolean).join(' · ');
              return (
                <tr key={x.date} className={x.live ? undefined : 'grey'}>
                  <td>{day(x.date)}</td>
                  <td>
                    {x.model ? short(x.model) : DASH}
                    <span className="sub show-sm">
                      {x.positions ?? DASH} positions · buying everything {money(x.market)} · {note}
                    </span>
                  </td>
                  <td className="num hide-sm">{x.positions ?? DASH}</td>
                  <td className="num strong">{money(x.guard)}</td>
                  <td className="num">{money(x.plain)}</td>
                  <td className="num hide-sm">{money(x.market)}</td>
                  <td className="hide-sm">{note}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

// The regime reading: which regime the Regime HMM rows believe the market is in after the newest game day
export function RegimesCard({ w }: { w: Workbench }) {
  return (
    <Card
      id="regimes"
      title="Regime reading"
      help="A regime is a Gaussian over the instruments' returns; the label is the dominant regime's volatility against the market's, the template number keeps its identity across refits. The expected return is the mixture's mean per instrument: a reading, not a recommendation. The weekly readings and their history are on Controls › Regimes."
      meta={[`${w.regimes.length} rows`]}
      flush
    >
      {w.regimes.length ? (
        <Scroll>
          <table className="tbl">
            <thead>
              <tr>
                <th>Row</th>
                <th>Regime</th>
                <th className="num">Probability</th>
                <th className="hide-sm">Expected next-day return</th>
              </tr>
            </thead>
            <tbody>
              {w.regimes.map((r) => {
                const expected = r.expected.map((e) => `${e.symbol} ${pct(isNum(e.value) ? e.value * 100 : null, 2)}`).join(', ');
                return (
                  <tr key={r.row}>
                    <td>
                      {short(r.row)}
                      <span className="sub">
                        after {day(r.date)} · {r.regimes ?? DASH} {r.regimes === 1 ? 'regime' : 'regimes'} · features {r.features.join(', ') || DASH} · {r.readings ?? DASH} readings kept
                      </span>
                    </td>
                    <td>
                      T{r.template ?? DASH} · {r.label || DASH}
                      <span className="sub">
                        {isNum(r.volatilityRank) && isNum(r.regimes) && r.regimes > 1 ? `${r.volatilityRank} of ${r.regimes} by volatility` : 'one regime'}
                      </span>
                      <span className="sub show-sm">{expected}</span>
                    </td>
                    <td className="num">{share(r.probability)}</td>
                    <td className="hide-sm">{expected}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Scroll>
      ) : (
        <div className="card-b">
          <div className="empty">No regime reading yet.</div>
        </div>
      )}
    </Card>
  );
}

// Background: how a day becomes a draw, with the newest settled day as the worked example (each instrument's ten bins,
// the cut points under them, where the day landed and what the day's most exact row had called)
export function BackgroundCard({ w }: { w: Workbench }) {
  const ex = w.example;
  const best = ex?.best ?? null;
  const crypto = w.market === 'crypto';
  return (
    <Card id="background" title="Background: how a day becomes a draw" meta={[w.calendar, `chance: exact ${share(w.chance.exact)}, within one bin ${share(w.chance.adjacent)}, direction ${share(w.chance.direction)}`]}>
      <details className="fold">
        <summary>
          Two closes, one number, one of {w.k} bins <span className="mut">· the newest settled day as the example</span>
        </summary>
        <div className="plines">
          <p>
            <span className="lbl">1. Two closes, one number.</span> A day&apos;s move is the change from the previous close to its close; nothing within the day is used.{' '}
            {crypto ? 'A crypto day is the UTC day: its close is 00:00 UTC, 02:00 Belgian time in summer, 01:00 in winter.' : 'A share day is the New York session: its close is 16:00 New York time, 22:00 Belgian time in most weeks (21:00 in the weeks the US and Europe change clocks on different dates; three hours earlier on an early close).'}
          </p>
          <p>
            <span className="lbl">2. The move goes into one of {w.k} bins.</span> Each {w.noun}&apos;s past daily moves, sorted and cut into {w.k} equal piles, from the days before
            only: bin 0 its worst tenth, bin {w.k - 1} its best. A blind pick is right 1 time in {w.k}, the bar every model has to beat.
          </p>
          <p>
            <span className="lbl">3. The bins side by side are the draw.</span> One slot per {w.noun}; a model&apos;s ticket is one bin per slot, made by the morning run after the
            previous close and judged at the day&apos;s close. A hit is the right bin in the right slot.
          </p>
        </div>
        {ex ? (
          <>
            <p className="line">
              <b>{stamp(ex.date)}</b>: the draw reads {ex.instruments.map((i) => i.bin ?? DASH).join(' ')}.
              {best
                ? ` ${short(best.name)}, the most exact row that day, had played ${best.bins.map((b) => b ?? DASH).join(' ')}: ${best.exact ?? DASH} of ${best.positions ?? DASH} in the right bin, ${best.adjacent ?? DASH} within one, ${best.direction ?? DASH} on the right side of zero.`
                : ''}
            </p>
            <div className="bstrips">
              {ex.instruments.map((i, n) => {
                const called = best?.bins[n] ?? null;
                return (
                  <div key={i.symbol} className="bst">
                    <div className="lab">
                      <b>{i.symbol}</b> {movePct(i.ret)} → bin {i.bin ?? DASH}
                      <small>
                        close {rate(i.close.before)} → {rate(i.close.on)}
                        {isNum(called) ? ` · called ${called}` : ''}
                      </small>
                    </div>
                    <div>
                      <div className="bbox">
                        {Array.from({ length: w.k }, (_, b) => {
                          // the day's bin dark; the call outlined green, filled pale green when it is one bin off
                          const cls = [b === i.bin ? 'on' : '', b === called ? 'pk' : '', b === called && isNum(i.bin) && Math.abs(b - i.bin) === 1 ? 'adj' : ''].filter(Boolean).join(' ');
                          return (
                            <i key={b} className={cls || undefined}>
                              {b}
                            </i>
                          );
                        })}
                      </div>
                      <div className="bedges" aria-hidden="true">
                        {i.edges.map((e, k) => (
                          <span key={k} style={{ left: `${((k + 1) / w.k) * 100}%` }}>
                            {movePct(e)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="legend">
              <span className="swatch">
                <i className="nbd" />
                where the day landed
              </span>
              {best ? (
                <span className="swatch">
                  <i className="sq up" />
                  {short(best.name)}&apos;s call (green outline; pale green when one bin off)
                </span>
              ) : null}
              <span className="swatch">the cut points between the bins, as moves</span>
            </div>
          </>
        ) : null}
      </details>
      <details className="fold">
        <summary>How to read the charts and the books</summary>
        <div className="plines">
          <p>
            A bin is a move interval, so a call is drawn as the price it stands for on the previous close (the bin&apos;s middle) with its band, the price range the close must
            land in. The two end bins have no outer edge: their band runs to the chart&apos;s edge.
          </p>
          <p>
            The paper books buy every up call (a bin in the upper half) at {w.stake ?? 100} {w.currency} a position with a {isNum(w.feePerLeg) ? `${(w.feePerLeg * 100).toFixed(2)}%` : DASH} fee on
            each leg{crypto ? ', at the previous close (00:00 UTC), hours before the ticket is up: they score the call, not what a reader could have made' : ''}. With shorts on, a bin
            in the lower half is a paper short, which is never an order.
          </p>
          <p>
            The week game is the second game: the same {w.noun}s, one call for the whole week, scored once at the week&apos;s close. Its draws are under{' '}
            {w.weekGame ? <Link to={`/admin/games/${w.weekGame}`}>Games › {w.weekGame}</Link> : 'Games'}.
          </p>
        </div>
      </details>
    </Card>
  );
}
