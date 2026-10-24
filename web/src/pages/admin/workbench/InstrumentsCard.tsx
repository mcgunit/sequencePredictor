import { useCallback, useMemo, useState } from 'react';
import { Card, Legend, Scroll } from '../../../components/admin/Card';
import { Failed } from '../../../components/admin/States';
import { DASH, day, dayYear, isNum, pct, rate, short, stamp } from '../../../format';
import { useEChart, type ChartOption, type Tokens } from '../../../hooks/useEChart';
import { useLoad } from '../../../hooks/useLoad';
import { getInstrument, type InstrumentSeries, type Workbench } from '../../../workbench';
import { frame, tooltip } from './chart';

// The instruments (C9): one instrument at a time - its closes over a year, and one model's call per tracked day drawn
// as the price it stands for (the bin's middle on the previous close) with its band, the right days green. Moves
// shows the same on the move scale. "Judged at / Right now" is said once, above the picker; every model's call for the
// next day is under the chart.

type View = 'price' | 'moves';
type Range = 'tracked' | '1M' | '3M' | '6M' | '1Y' | 'all';
const RANGES: Range[] = ['tracked', '1M', '3M', '6M', '1Y', 'all'];
const RANGE_DAYS: Record<Exclude<Range, 'tracked' | 'all'>, number> = { '1M': 31, '3M': 92, '6M': 183, '1Y': 366 };

// a bin as a move interval and its representative (markets.js marketRepresentative): the middle, or for an end bin
// its edge moved outward by half the neighbouring bin's width
function binOf(b: number, e: number[], k: number) {
  const lo = b > 0 ? (e[b - 1] ?? null) : null;
  const hi = b < k - 1 ? (e[b] ?? null) : null;
  let rep: number | null = null;
  if (lo !== null && hi !== null) rep = (lo + hi) / 2;
  else if (b === 0 && e.length > 1) rep = e[0] - (e[1] - e[0]) / 2;
  else if (b === k - 1 && e.length > 1) rep = e[e.length - 1] + (e[e.length - 1] - e[e.length - 2]) / 2;
  return { lo, hi, rep };
}

interface Course {
  cats: string[];
  close: (number | null)[];
  move: (number | null)[];
  rep: (number | null)[]; // price (price view) or move (moves view) the call stands for
  lo: (number | null)[];
  hi: (number | null)[];
  hit: (boolean | null)[];
  calls: number;
  hits: number;
}

function courseOf(s: InstrumentSeries, model: string | null, view: View, start: number): Course {
  const cats: string[] = [];
  const out: Course = { cats, close: [], move: [], rep: [], lo: [], hi: [], hit: [], calls: 0, hits: 0 };
  for (let i = start; i < s.dates.length; i += 1) {
    cats.push(s.dates[i]);
    const c = s.closes[i];
    const m = s.moves[i];
    out.close.push(c);
    out.move.push(m);
    const j = i - s.from;
    const b = model && j >= 0 ? (s.course[model]?.[j] ?? null) : null;
    const e = j >= 0 ? s.edges[j] : null;
    if (b === null || !e || !isNum(c) || !isNum(m)) {
      out.rep.push(null);
      out.lo.push(null);
      out.hi.push(null);
      out.hit.push(null);
      continue;
    }
    const iv = binOf(b, e, s.k);
    const base = c * Math.exp(-m);
    const at = (x: number | null) => (x === null ? null : view === 'price' ? base * Math.exp(x) : x);
    out.rep.push(at(iv.rep));
    out.lo.push(at(iv.lo));
    out.hi.push(at(iv.hi));
    const actual = e.filter((x) => m > x).length;
    out.hit.push(actual === b);
    out.calls += 1;
    if (actual === b) out.hits += 1;
  }
  // the next day's call, on the newest close
  const n = model ? s.next.find((x) => x.model === model) : null;
  if (s.nextDay && n) {
    cats.push(s.nextDay);
    out.close.push(null);
    out.move.push(null);
    const base = s.nextBase;
    const mv = (x: number | null) => (x === null || !isNum(base) || base <= 0 ? null : Math.log(x / base));
    out.rep.push(view === 'price' ? n.price : mv(n.price));
    out.lo.push(view === 'price' ? n.low : mv(n.low));
    out.hi.push(view === 'price' ? n.high : mv(n.high));
    out.hit.push(null);
  }
  return out;
}

function PriceChart({ s, c, view, model }: { s: InstrumentSeries; c: Course; view: View; model: string | null }) {
  const build = useCallback(
    (t: Tokens): ChartOption => {
      const main = view === 'price' ? c.close : c.move;
      const finite = [...main, ...c.rep, ...c.lo, ...c.hi].filter(isNum);
      const lo0 = finite.length ? Math.min(...finite) : 0;
      const hi0 = finite.length ? Math.max(...finite) : 1;
      const pad = (hi0 - lo0) * 0.06 || Math.abs(hi0) * 0.02 || 1;
      const yMin = view === 'moves' ? Math.min(lo0 - pad, 0) : lo0 - pad;
      const yMax = view === 'moves' ? Math.max(hi0 + pad, 0) : hi0 + pad;
      // the band as a ribbon: an invisible base at its low end and its height stacked on it; an open end runs to the edge.
      // stackStrategy 'all': ECharts' default stacks a value only on one of the same sign, so a positive height on a
      // negative low end (a down bin on the moves scale) would be drawn up from 0
      const has = c.rep.map((r) => r !== null);
      const base = c.lo.map((l, i) => (has[i] ? (l ?? yMin) : null));
      const height = c.hi.map((h, i) => (has[i] && base[i] !== null ? (h ?? yMax) - (base[i] as number) : null));
      const fmt = (v: unknown) => (view === 'price' ? rate(v) : pct(isNum(v) ? v * 100 : null, 2));
      const o = frame(t, c.cats, { left: view === 'price' ? 64 : 48, right: 12, min: yMin, max: yMax, zoom: true });
      return {
        ...o,
        yAxis: { ...(o.yAxis as object), axisLabel: { color: t.muted, fontSize: 11, width: view === 'price' ? 58 : 42, align: 'right', overflow: 'truncate', formatter: (v: number) => fmt(v) } },
        tooltip: tooltip(t, c.cats, (i) =>
          [
            view === 'price' ? `Close: ${rate(c.close[i])}` : `Move: ${fmt(c.move[i])}`,
            c.rep[i] !== null ? `${short(model)}: ${fmt(c.rep[i])} (band ${c.lo[i] === null ? 'open' : fmt(c.lo[i])} – ${c.hi[i] === null ? 'open' : fmt(c.hi[i])})${c.hit[i] ? ', right bin' : ''}` : '',
          ]
            .filter(Boolean)
            .join('<br/>'),
        ),
        // the close (or the move) first: the zoom slider's shadow draws the first series; z sets the layers
        series: [
          view === 'price'
            ? { name: 'Close', type: 'line', data: c.close, symbol: 'none', lineStyle: { width: 1.6, color: t.fg }, itemStyle: { color: t.fg }, z: 4 }
            : { name: 'Move', type: 'bar', data: c.move.map((v) => ({ value: v, itemStyle: { color: isNum(v) && v < 0 ? t.down : t.up } })), barMaxWidth: 8, z: 3 },
          { name: 'band base', type: 'line', data: base, stack: 'band', stackStrategy: 'all', symbol: 'none', lineStyle: { opacity: 0 }, silent: true, connectNulls: false },
          { name: 'band', type: 'line', data: height, stack: 'band', stackStrategy: 'all', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: t.accent, opacity: 0.14 }, silent: true, connectNulls: false },
          {
            name: short(model),
            type: 'line',
            data: c.rep.map((v, i) => ({ value: v, itemStyle: { color: c.hit[i] ? t.up : t.accent } })),
            symbol: 'circle',
            symbolSize: 5,
            lineStyle: { width: 0 },
            z: 5,
            connectNulls: false,
          },
        ],
      } as ChartOption;
    },
    [s, c, view, model],
  );
  const ref = useEChart(c.cats.length ? build : null);
  return <div ref={ref} className="chart tall" role="img" aria-label={`${s.symbol}: ${view === 'price' ? 'closes' : 'moves'} with ${short(model)}'s calls`} />;
}

function Instrument({ w, symbol }: { w: Workbench; symbol: string }) {
  const load = useLoad(() => getInstrument(w.market, symbol), [w.market, symbol]);
  const [view, setView] = useState<View>('price');
  const [range, setRange] = useState<Range>('tracked');
  const [picked, setPicked] = useState<string | null>(null);
  const s = load.kind === 'ok' && load.data.symbol === symbol ? load.data : null;
  const model = s ? (picked && s.course[picked] ? picked : (s.best ?? s.order[0] ?? null)) : null;
  const start = !s ? 0 : range === 'tracked' ? s.from : range === 'all' ? 0 : Math.max(0, s.dates.length - RANGE_DAYS[range]);
  const c = useMemo(() => (s ? courseOf(s, model, view, start) : null), [s, model, view, start]);
  if (load.kind === 'error') return <Failed error={load.error} retry={load.retry} bare />;
  if (!s || !c) return <div className="chart tall empty">Loading…</div>;
  const up = s.next.filter((n) => n.up).length;
  // a day in another year than the last close shows its year ("22 Oct 2025")
  const sameYear = (d: string | undefined) => !d || !s.lastDate || d.slice(0, 4) === s.lastDate.slice(0, 4);
  return (
    <>
      <p className="line">
        <b>
          {s.symbol} · {s.name}
        </b>{' '}
        · last close {rate(s.lastClose)} {s.quote} on {day(s.lastDate)} · {s.dates.length} closes from {sameYear(s.dates[0]) ? day(s.dates[0]) : dayYear(s.dates[0])}, calls from{' '}
        {sameYear(s.dates[s.from]) ? day(s.dates[s.from]) : dayYear(s.dates[s.from])}
        {s.active ? '' : ' · not traded any more'}
      </p>
      <div className="ctl">
        <div className="seg" role="group" aria-label="Scale">
          <button type="button" aria-pressed={view === 'price'} onClick={() => setView('price')}>
            Prices
          </button>
          <button type="button" aria-pressed={view === 'moves'} onClick={() => setView('moves')}>
            Moves
          </button>
        </div>
        <div className="seg" role="group" aria-label="Range">
          {RANGES.filter((r) => r === 'tracked' || r === 'all' || s.dates.length > RANGE_DAYS[r]).map((r) => (
            <button key={r} type="button" aria-pressed={range === r} onClick={() => setRange(r)}>
              {r === 'tracked' ? 'Tracked' : r === 'all' ? 'All' : r}
            </button>
          ))}
        </div>
        <label className="pick">
          <span className="vh">Model drawn</span>
          <select value={model ?? ''} onChange={(e) => setPicked(e.target.value || null)}>
            {s.order.map((n) => (
              <option key={n} value={n}>
                {short(n)}
                {n === s.best ? ' (best exact rate)' : ''}
              </option>
            ))}
          </select>
        </label>
      </div>
      <PriceChart s={s} c={c} view={view} model={model} />
      <Legend
        items={[
          view === 'price' ? { mark: 'ln', label: 'close' } : { mark: 'bar', label: 'the day’s move' },
          { mark: 'dot acc', label: `${short(model)}'s call, the bin's middle` },
          { mark: 'dot up', label: 'a call in the right bin' },
          { mark: 'acc14', label: 'its band (an open end runs to the edge)' },
        ]}
      />
      <p className="line">
        {short(model)} in this range: {c.hits} of {c.calls} called days in the right bin{c.calls ? ` (${Math.round((c.hits / c.calls) * 100)}%)` : ''}, chance 1 in {s.k}
        {s.nextDay ? `; its call for ${day(s.nextDay)} is the last point` : ''}.
      </p>
      <details className="fold">
        <summary>
          Every model&apos;s call for {day(s.nextDay)}: {s.next.length} calls, {up} long <span className="mut">· price and band on the close of {day(s.nextDate)}</span>
        </summary>
        <Scroll>
          <table className="tbl">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Bin</th>
                <th>Position</th>
                <th className="num">Price</th>
                <th className="num hide-sm">Band</th>
              </tr>
            </thead>
            <tbody>
              {s.order
                .map((name) => s.next.find((n) => n.model === name))
                .concat(s.next.filter((n) => !s.order.includes(n.model)))
                .filter((n): n is InstrumentSeries['next'][number] => Boolean(n))
                .map((n) => (
                  <tr key={n.model}>
                    <td>{short(n.model)}</td>
                    <td className="num">{n.bin ?? DASH}</td>
                    <td>{isNum(n.size) ? (n.size > 0 ? `▲ long ×${n.size}` : '× 0, sits out') : n.up ? '▲ long' : '▼ not long (a paper short with shorts on)'}</td>
                    <td className="num">
                      {rate(n.price)}
                      <span className="sub show-sm">{n.low === null ? `below ${rate(n.high)}` : n.high === null ? `above ${rate(n.low)}` : `${rate(n.low)} – ${rate(n.high)}`}</span>
                    </td>
                    <td className="num hide-sm">{n.low === null ? `below ${rate(n.high)}` : n.high === null ? `above ${rate(n.low)}` : `${rate(n.low)} – ${rate(n.high)}`}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </Scroll>
      </details>
    </>
  );
}

export default function InstrumentsCard({ w }: { w: Workbench }) {
  const list = [...w.instruments].sort((a, b) => a.position - b.position);
  const [symbol, setSymbol] = useState(list.find((i) => i.active)?.symbol ?? list[0]?.symbol ?? '');
  const j = w.judged;
  return (
    <Card
      id="instruments"
      title={w.noun === 'coin' ? 'Coins' : 'Shares'}
      help="A call is scored close to close: right when the close lands in the band, whatever happened in between. The model drawn by default has the best exact rate over the scored days; Today's plan follows the best book instead, so the two can differ."
      meta={[`${list.length} ${w.noun}s`, w.calendar]}
    >
      {j ? (
        <p className="line">
          Judged at the close of {stamp(j.day)}: <b>{stamp(j.at)}</b> Brussels time. Right now {j.status}.
        </p>
      ) : null}
      <div className="ctl">
        <div className="seg" role="group" aria-label={w.noun}>
          {list.map((i) => (
            <button key={i.symbol} type="button" aria-pressed={symbol === i.symbol} onClick={() => setSymbol(i.symbol)}>
              {i.symbol}
            </button>
          ))}
        </div>
      </div>
      {symbol ? <Instrument key={symbol} w={w} symbol={symbol} /> : <div className="empty">No instrument.</div>}
    </Card>
  );
}
