import { useCallback, useMemo, useState } from 'react';
import { Card, Legend, Scroll } from '../../../components/admin/Card';
import { Failed } from '../../../components/admin/States';
import { DASH, day, isNum, money, short } from '../../../format';
import { useEChart, type ChartOption, type Tokens } from '../../../hooks/useEChart';
import { useLoad } from '../../../hooks/useLoad';
import { getBooks, type BookRule, type BookSeries, type Books, type Workbench } from '../../../workbench';
import { frame, lineSeries, tooltip } from './chart';

// Paper trading (C8): every row's calls traded on paper under each rule, long only or long and short, against the
// market under the same rule. The headline names each book's best row among the rows with enough settled days; the
// chart draws one book's cumulative money per model on one scale, the highlighted row thick with its total at its end,
// the market dashed, every other row thin and faint (their totals are in the Models table).

type Base = 'daily' | 'hold' | 'open';
const BASE_TEXT: Record<Base, string> = { daily: 'Daily round trip', hold: 'Hold while up', open: 'Open to close' };
const MARKET_TEXT: Record<Base, string> = { daily: 'buying everything', hold: 'buy and hold', open: 'buying everything' };

type Range = '1M' | '3M' | '6M' | '1Y' | 'all';
const RANGES: Range[] = ['1M', '3M', '6M', '1Y', 'all'];
const RANGE_DAYS: Record<Exclude<Range, 'all'>, number> = { '1M': 31, '3M': 92, '6M': 183, '1Y': 366 };
// the newest n dates of a book (the totals stay cumulative since the book's first day)
function sliceOf(s: BookSeries, range: Range): BookSeries {
  if (range === 'all' || s.dates.length <= RANGE_DAYS[range]) return s;
  const from = s.dates.length - RANGE_DAYS[range];
  const cut = (xs: (number | null)[]) => xs.slice(from);
  return { ...s, dates: s.dates.slice(from), market: cut(s.market), sold: s.sold ? cut(s.sold) : null, series: Object.fromEntries(Object.entries(s.series).map(([k, v]) => [k, cut(v)])) };
}

const lastOf = (xs: (number | null)[] | undefined) => {
  if (!xs) return null;
  for (let i = xs.length - 1; i >= 0; i -= 1) if (isNum(xs[i])) return xs[i];
  return null;
};

// the row a book's headline names: the best total among the rows with at least minDays settled days (a sized row
// never heads a long-and-short book), as markets.js picks it
function headlineOf(w: Workbench, b: Books, base: Base, shorts: boolean, s: BookSeries | null): string | null {
  if (!shorts && b.headline[base]?.best) return b.headline[base]?.best?.model ?? null;
  if (shorts && base === 'daily' && b.headline.longShort?.best) return b.headline.longShort.best.model;
  if (!s) return null;
  const ok = new Map(w.models.map((m) => [m.name, (m.days ?? 0) >= b.minDays && !(shorts && m.sized)]));
  let best: string | null = null;
  let top = -Infinity;
  s.order.forEach((name) => {
    const v = lastOf(s.series[name]);
    if (ok.get(name) && isNum(v) && v > top) {
      top = v;
      best = name;
    }
  });
  return best;
}

function BooksChart({ s, pick, all, cur }: { s: BookSeries; pick: string | null; all: boolean; cur: string }) {
  const build = useCallback(
    (t: Tokens): ChartOption => {
      const context = all ? s.order.filter((n) => n !== pick).map((n) => lineSeries(short(n), s.series[n] ?? [], t.faint, 'context')) : [];
      return {
        ...frame(t, s.dates, { right: 72, zoom: true }),
        tooltip: tooltip(t, s.dates, (i) =>
          [pick ? `${short(pick)}: ${money(s.series[pick]?.[i])}` : '', `Market: ${money(s.market[i])}`, s.sold ? `Selling everything: ${money(s.sold[i])}` : '']
            .filter(Boolean)
            .join('<br/>'),
        ),
        // the highlighted row first, else the market: the zoom slider's shadow draws the first series (z sets the layers)
        series: [
          ...(pick ? [lineSeries(short(pick), s.series[pick] ?? [], t.accent, 'main', (v) => money(v))] : []),
          lineSeries('Market', s.market, t.muted, 'dashed', (v) => money(v)),
          ...(s.sold ? [lineSeries('Selling everything', s.sold, t.down, 'dashed', (v) => money(v))] : []),
          ...context,
        ],
      };
    },
    [s, pick, all, cur],
  );
  const ref = useEChart(s.dates.length ? build : null);
  return <div ref={ref} className="chart tall" role="img" aria-label={`Cumulative ${cur} per model, ${s.dates.length} days`} />;
}

export default function BooksCard({ w }: { w: Workbench }) {
  const b = w.books;
  const [base, setBase] = useState<Base>('daily');
  const [shorts, setShorts] = useState(false);
  const [all, setAll] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [range, setRange] = useState<Range>('all');
  const has = (r: string) => Boolean(b?.rules.includes(r as BookRule));
  const shortsOn = shorts && has(`${base}_ls`);
  const rule = (shortsOn ? `${base}_ls` : has(base) ? base : 'daily') as BookRule;
  const load = useLoad(() => (b ? getBooks(w.market, rule) : Promise.reject(new Error('no paper books yet'))), [w.market, rule]);
  const loaded = load.kind === 'ok' ? load.data : null;
  const shown = useMemo(() => (loaded && loaded.rule === rule ? sliceOf(loaded, range) : null), [loaded, rule, range]);
  if (!b) {
    return (
      <Card id="books" title="Paper books">
        <div className="empty">No paper books yet.</div>
      </Card>
    );
  }
  const cur = b.currency || w.currency;
  const s = load.kind === 'ok' && load.data.rule === rule ? load.data : null;
  const best = headlineOf(w, b, base, shortsOn, s);
  const pick = picked && s?.series[picked] ? picked : best;
  const bases = (['daily', 'hold', 'open'] as Base[]).filter((x) => has(x));
  const heads: { label: string; best: { model: string; total: number | null } | null; market: string }[] = [];
  bases.forEach((x) => {
    const h = b.headline[x];
    if (h) heads.push({ label: BASE_TEXT[x], best: h.best, market: `${MARKET_TEXT[x]} ${money(h.market)}` });
  });
  if (b.headline.longShort) {
    const l = b.headline.longShort;
    heads.push({ label: 'Long & short, daily', best: l.best, market: `buying everything ${money(l.bought)} · selling everything ${money(l.sold)}` });
  }
  return (
    <Card
      id="books"
      title="Paper books"
      help={
        <>
          {BASE_TEXT.daily}: {b.ruleText.daily}. {BASE_TEXT.hold}: {b.ruleText.hold}.{b.ruleText.open ? ` ${BASE_TEXT.open}: ${b.ruleText.open}.` : ''} Long and short: {b.ruleText.short}. The
          dashed line is the market under the same rule, which a book has to beat before it means anything. Paper money: no slippage, fills at the open or the close.
        </>
      }
      meta={[`${b.stake ?? DASH} ${cur} a position`, isNum(b.feePerLeg) ? `${(b.feePerLeg * 100).toFixed(2)}% a leg` : '', `${b.days} settled days, ${day(b.first)} – ${day(b.last)}`]}
      flush
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Book</th>
              <th>Best book, rows with ≥ {b.minDays} settled days</th>
              <th>The market, same rule</th>
            </tr>
          </thead>
          <tbody>
            {heads.map((h) => (
              <tr key={h.label}>
                <td>{h.label}</td>
                <td>
                  {h.best ? (
                    <>
                      <b>
                        {money(h.best.total)} {cur}
                      </b>{' '}
                      {short(h.best.model)}
                    </>
                  ) : (
                    <span className="mut">no row with {b.minDays} days</span>
                  )}
                </td>
                <td>
                  {h.market} {cur}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Scroll>
      <div className="card-b">
        <p className="line">
          {b.seasonedRows ?? DASH} rows have at least {b.minDays} settled days and can head a book; a sized row (the RL Position Model) is long only and never heads a long-and-short book.
        </p>
        <div className="ctl">
          <div className="seg" role="group" aria-label="Rule">
            {bases.map((x) => (
              <button key={x} type="button" aria-pressed={base === x} onClick={() => setBase(x)}>
                {x === 'daily' ? 'Daily' : BASE_TEXT[x]}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Shorts">
            <button type="button" aria-pressed={!shortsOn} onClick={() => setShorts(false)}>
              Long only
            </button>
            <button type="button" aria-pressed={shortsOn} onClick={() => setShorts(true)} disabled={!has(`${base}_ls`)}>
              Long &amp; short
            </button>
          </div>
          <div className="seg" role="group" aria-label="Rows drawn">
            <button type="button" aria-pressed={!all} onClick={() => setAll(false)}>
              One row
            </button>
            <button type="button" aria-pressed={all} onClick={() => setAll(true)}>
              Every row
            </button>
          </div>
          {s && s.dates.length > RANGE_DAYS['1M'] ? (
            <div className="seg" role="group" aria-label="Range">
              {RANGES.filter((r) => r === 'all' || s.dates.length > RANGE_DAYS[r]).map((r) => (
                <button key={r} type="button" aria-pressed={range === r} onClick={() => setRange(r)}>
                  {r === 'all' ? 'All' : r}
                </button>
              ))}
            </div>
          ) : null}
          {s ? (
            <label className="pick">
              <span className="vh">Row highlighted</span>
              <select value={pick ?? ''} onChange={(e) => setPicked(e.target.value || null)}>
                {s.order.map((n) => (
                  <option key={n} value={n}>
                    {short(n)}
                    {n === best ? ' (best book)' : ''}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
        {load.kind === 'error' ? (
          <Failed error={load.error} retry={load.retry} bare />
        ) : s ? (
          <>
            <BooksChart s={shown ?? s} pick={pick} all={all} cur={cur} />
            <Legend
              items={[
                { mark: '', label: pick ? `${short(pick)}${pick === best ? ', the best book' : ''}` : 'no row' },
                { mark: 'dash', label: `the market, ${MARKET_TEXT[base]}` },
                ...(s.sold ? [{ mark: 'dash down', label: 'selling everything' }] : []),
                ...(all ? [{ mark: 'se', label: 'every other row' }] : []),
              ]}
            />
            <p className="line">
              On {day(s.dates[s.dates.length - 1])}, cumulative {cur}: {pick ? `${short(pick)} ${money(lastOf(s.series[pick]))}` : ''} · the market {money(lastOf(s.market))}
              {s.sold ? ` · selling everything ${money(lastOf(s.sold))}` : ''}, {BASE_TEXT[base].toLowerCase()}
              {shortsOn ? ', long and short' : ', long only'}.
            </p>
          </>
        ) : (
          <div className="chart tall empty">Loading…</div>
        )}
      </div>
    </Card>
  );
}
