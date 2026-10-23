import { useMemo, useState } from 'react';
import { isNum } from '../../format';
import { useEChart, type ChartOption, type Tokens } from '../../hooks/useEChart';
import type { Shape, Votes } from '../../workbench';
import { Legend } from '../admin/Card';
import { NUMBER_MAX, type Mode } from './text';

// The vote under a table (B8, B10): per number how many of the shown rows' first tickets carry it, the drawn numbers
// green. The positional games (pick3, Joker+, the markets) are counted per slot, so a bar lights only for the value
// drawn in its own slot. With All the day file's own weighted vote of every row is the second view, as the old chart
// drew it (the positional games' pooled). Bars start at zero, one bar per value, so the heights compare; the per-slot
// charts share one top, every shown row, so a full bar means every row in any slot.

type Bars = { cats: string[]; values: number[]; hot: boolean[] };

// max: a common top for charts that compare (the per-slot charts: every shown row), else the chart's own
function barsBuild(b: Bars, share: boolean, mini: boolean, max?: number) {
  return (t: Tokens): ChartOption => ({
    animation: false,
    textStyle: { fontFamily: t.font, color: t.fg },
    grid: { left: share ? 40 : 30, right: 6, top: 10, bottom: 22, containLabel: false },
    xAxis: {
      type: 'category',
      data: b.cats,
      axisLine: { lineStyle: { color: t.line } },
      axisTick: { show: false },
      axisLabel: { color: t.muted, fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: max && max > 0 ? max : undefined,
      minInterval: share ? undefined : 1,
      splitNumber: mini ? 2 : 4,
      splitLine: { lineStyle: { color: t.line, opacity: 0.7 } },
      axisLabel: { color: t.muted, fontSize: 10, formatter: (v: number) => (share ? `${(v * 100).toFixed(0)}%` : String(v)) },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: t.card,
      borderColor: t.line,
      textStyle: { color: t.fg, fontSize: 12 },
      formatter: (params: unknown) => {
        const p = (Array.isArray(params) ? params[0] : params) as { dataIndex?: number } | undefined;
        const i = p?.dataIndex ?? 0;
        const v = b.values[i];
        return `<b>${b.cats[i]}</b>${b.hot[i] ? ' (drawn)' : ''}: ${share ? `${(v * 100).toFixed(1)}%` : `${v} rows`}`;
      },
    },
    series: [
      {
        type: 'bar',
        barCategoryGap: '18%',
        data: b.values.map((v, i) => ({ value: v, itemStyle: { color: b.hot[i] ? t.up : t.barB, borderRadius: [2, 2, 0, 0] } })),
      },
    ],
  });
}

function BarChart({ bars, share, mini, label, max }: { bars: Bars; share: boolean; mini?: boolean; label: string; max?: number }) {
  // the bars are rebuilt by the parent on every render: key the option on their content
  const key = JSON.stringify(bars);
  const build = useMemo(() => barsBuild(bars, share, Boolean(mini), max), [key, share, mini, max]);
  const ref = useEChart(bars.cats.length ? build : null);
  return <div ref={ref} className={mini ? 'chart mini' : 'chart'} role="img" aria-label={label} />;
}

// value -> count as a full axis (every number of the game, every digit 0-9), drawn ones marked
function axisOf(pairs: [number, number][], drawn: Set<number>, from: number, to: number): Bars {
  const m = new Map(pairs);
  const cats: string[] = [];
  const values: number[] = [];
  const hot: boolean[] = [];
  for (let v = from; v <= to; v += 1) {
    cats.push(String(v));
    values.push(m.get(v) ?? 0);
    hot.push(drawn.has(v));
  }
  return { cats, values, hot };
}

export default function VotesChart({ votes, shape, game, slotLabels, mode }: { votes: Votes; shape: Shape; game: string; slotLabels?: string[]; mode: Mode }) {
  const [view, setView] = useState<'rows' | 'stored'>('rows');
  const stored = votes.stored && mode === 'admin' ? votes.stored : null;
  const showing = stored && view === 'stored' ? 'stored' : 'rows';
  const rowsWord = `${votes.rows} ${votes.rows === 1 ? 'row' : 'rows'}`;
  if (!votes.rows && !stored) return <div className="empty">{mode === 'user' ? 'No rows shown, so no vote.' : 'No rows in this scope, so no vote.'}</div>;

  const switcher = stored ? (
    <div className="ctl">
      <div className="seg" role="group" aria-label="Vote shown">
        <button type="button" aria-pressed={showing === 'rows'} onClick={() => setView('rows')}>
          Rows shown ({votes.rows})
        </button>
        <button type="button" aria-pressed={showing === 'stored'} onClick={() => setView('stored')}>
          Stored vote, every row
        </button>
      </div>
      <span className="note">
        {showing === 'rows'
          ? `how many of the ${rowsWord} carry each value`
          : `the day file's weighted vote of every row${shape.positional ? ', pooled over the slots' : ', mains only'}${stored.from === 'previous file' ? ' (this file has none: the previous file’s, the same vote)' : ''}`}
      </span>
    </div>
  ) : null;

  const legend = <Legend items={[{ mark: 'nbm', label: votes.drawn.length || votes.slots.some((s) => isNum(s.drawn)) ? 'drawn' : 'drawn (none yet: the coming draw)' }, { mark: 'barb', label: 'not drawn' }]} />;

  if (showing === 'stored' && stored) {
    const max = shape.positional ? 9 : (NUMBER_MAX[game] ?? Math.max(0, ...stored.values.map((p) => p[0])));
    const drawn = shape.positional ? new Set<number>() : new Set(votes.drawn);
    const bars = axisOf(stored.values, drawn, shape.positional ? 0 : 1, max);
    return (
      <>
        {switcher}
        <BarChart bars={bars} share label={`Stored vote of every row, ${bars.cats.length} values`} />
        {legend}
      </>
    );
  }

  if (votes.kind === 'slots') {
    return (
      <>
        {switcher}
        <div className="vgrid">
          {votes.slots.map((s, i) => {
            const label = slotLabels?.[i] ?? `Slot ${i + 1}`;
            const bars = axisOf(s.counts, new Set(isNum(s.drawn) ? [s.drawn] : []), 0, 9);
            const onDrawn = isNum(s.drawn) ? (new Map(s.counts).get(s.drawn) ?? 0) : null;
            return (
              <div key={i} className="vslot">
                <h3>
                  {label}
                  {isNum(s.drawn) ? ` · drawn ${s.drawn}: ${onDrawn} of ${votes.rows}` : ''}
                </h3>
                <BarChart bars={bars} share={false} mini max={votes.rows} label={`${label}: rows per digit, out of ${votes.rows}`} />
              </div>
            );
          })}
        </div>
        {legend}
      </>
    );
  }

  const max = NUMBER_MAX[game] ?? Math.max(0, ...votes.counts.map((p) => p[0]), ...votes.drawn);
  const drawn = new Set(votes.drawn);
  const bars = axisOf(votes.counts, drawn, 1, max);
  const counts = new Map(votes.counts);
  const top = Math.max(0, ...votes.counts.map((p) => p[1]));
  const most = votes.counts.filter((p) => p[1] === top && top > 0).map((p) => p[0]);
  return (
    <>
      {switcher}
      <BarChart bars={bars} share={false} label={`Rows per number, ${votes.rows} rows`} />
      {legend}
      <p className="note">
        {votes.drawn.length ? (
          <>
            Drawn, with the rows carrying each: {votes.drawn.map((v) => `${v} (${counts.get(v) ?? 0})`).join(' · ')} of {rowsWord}.{' '}
          </>
        ) : null}
        {most.length ? `Carried most: ${most.join(', ')} (${top} of ${rowsWord}).` : ''}
      </p>
    </>
  );
}

export const slotNames = (shape: Shape) => (shape.slots.length ? shape.slots : shape.family === 'jokerplus' ? ['1', '2', '3', '4', '5', '6'].map((x) => `Digit ${x}`) : ['Digit 1', 'Digit 2', 'Digit 3']);
