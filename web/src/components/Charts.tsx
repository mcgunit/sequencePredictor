import { useCallback } from 'react';
import type { MarketView } from '../api';
import { axisDates, day, esc, isNum, money, plural } from '../format';
import { useEChart, type ChartOption, type Tokens } from '../hooks/useEChart';

const num = (v: unknown): number | null => (isNum(v) ? v : null);

function firstIndex(params: unknown): number {
  const p = (Array.isArray(params) ? params[0] : params) as { dataIndex?: number } | undefined;
  return p?.dataIndex ?? 0;
}

// One plot area for both wide charts (fixed margins, no containLabel), so the cumulative and the daily
// chart stacked in the same column put their dates at the same x; the y labels right-aligned in 40px.
const WIDE_GRID = { left: 44, right: 56, top: 14, bottom: 22, containLabel: false };

function frame(t: Tokens, categories: string[]): ChartOption {
  return {
    animation: false,
    textStyle: { fontFamily: t.font, color: t.fg },
    grid: WIDE_GRID,
    xAxis: {
      type: 'category',
      data: categories,
      axisLine: { lineStyle: { color: t.line } },
      axisTick: { show: false },
      axisLabel: { color: t.muted, fontSize: 11, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      splitNumber: 4,
      splitLine: { lineStyle: { color: t.line, opacity: 0.7 } },
      axisLabel: { color: t.muted, fontSize: 11, width: 40, align: 'right', overflow: 'truncate' },
    },
  };
}

function tip(t: Tokens) {
  return { backgroundColor: t.card, borderColor: t.line, textStyle: { color: t.fg, fontSize: 12 }, extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.12);' };
}

function bar(v: number | null, t: Tokens) {
  const x = v ?? 0;
  return { value: x, itemStyle: { color: x >= 0 ? t.up : t.down, borderRadius: x >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3] } };
}

function market(data: (number | null)[], t: Tokens, endLabel = false) {
  return {
    name: 'Market',
    type: 'line' as const,
    data,
    symbol: 'none',
    lineStyle: { width: 1.4, color: t.muted, type: 'dashed' as const },
    itemStyle: { color: t.muted },
    ...(endLabel ? { endLabel: { show: true, color: t.muted, fontSize: 11, formatter: (p: { value?: unknown }) => money(p.value) } } : {}),
  };
}

function Legend({ bars }: { bars?: boolean }) {
  return (
    <div className="meta">
      <span className="swatch">
        <i className={bars ? 'bar' : ''} />
        Plan
      </span>
      <span className="swatch">
        <i className="dash" />
        Market
      </span>
    </div>
  );
}

// The record as three panels (MarketPage places them on the grid): cumulative and per instrument on top,
// daily profit beside the latest positions below. "Plan" is what following the day plan made, "Market"
// buying every instrument every day under the same rule.
export function useChartPanels(data: MarketView, pending = false) {
  const { daily, cumulative, perCoin, unit, noun, status } = data;
  const amount = (v: unknown) => `${money(v)}${unit ? ` ${unit}` : ''}`;

  const buildDaily = useCallback(
    (t: Tokens): ChartOption => ({
      ...frame(t, axisDates(daily.map((d) => d.date))),
      tooltip: {
        ...tip(t),
        trigger: 'axis',
        formatter: (params: unknown) => {
          const d = daily[firstIndex(params)];
          return d ? `<b>${day(d.date)}</b><br/>Plan: ${amount(d.pnl)}<br/>Market: ${amount(d.market)}` : '';
        },
      },
      series: [
        { name: 'Plan', type: 'bar', barMaxWidth: 18, data: daily.map((d) => bar(num(d.pnl), t)) },
        market(
          daily.map((d) => num(d.market)),
          t,
        ),
      ],
    }),
    [daily, unit],
  );

  const buildCumulative = useCallback(
    (t: Tokens): ChartOption => ({
      ...frame(t, axisDates(cumulative.map((d) => d.date))),
      tooltip: {
        ...tip(t),
        trigger: 'axis',
        formatter: (params: unknown) => {
          const d = cumulative[firstIndex(params)];
          return d ? `<b>${day(d.date)}</b><br/>Plan: ${amount(d.plan)}<br/>Market: ${amount(d.market)}` : '';
        },
      },
      series: [
        {
          name: 'Plan',
          type: 'line',
          data: cumulative.map((d) => num(d.plan)),
          symbol: 'circle',
          symbolSize: 5,
          showSymbol: cumulative.length <= 40,
          lineStyle: { width: 2.4, color: t.accent },
          itemStyle: { color: t.accent },
          areaStyle: { color: t.accent, opacity: 0.1 },
          endLabel: { show: true, color: t.fg, fontSize: 11, fontWeight: 600, formatter: (p: { value?: unknown }) => money(p.value) },
        },
        market(
          cumulative.map((d) => num(d.market)),
          t,
          true,
        ),
      ],
    }),
    [cumulative, unit],
  );

  // per instrument: horizontal bars, best on top, the amount at the end of each bar; the names stay at the
  // left edge (onZero off) and the value axis leaves room on both sides for the labels
  const buildPerCoin = useCallback(
    (t: Tokens): ChartOption => {
      const sorted = [...perCoin].sort((a, b) => (num(a.pnl) ?? 0) - (num(b.pnl) ?? 0));
      const values = sorted.map((r) => num(r.pnl) ?? 0);
      const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
      const span = hi - lo || 1;
      return {
        animation: false,
        textStyle: { fontFamily: t.font, color: t.fg },
        grid: { left: 4, right: 12, top: 14, bottom: 22, containLabel: true },
        xAxis: {
          type: 'value',
          min: lo < 0 ? lo - span * 0.3 : 0,
          max: hi > 0 ? hi + span * 0.3 : 0,
          splitNumber: 3,
          splitLine: { lineStyle: { color: t.line, opacity: 0.7 } },
          axisLabel: { show: false },
        },
        yAxis: {
          type: 'category',
          data: sorted.map((r) => r.symbol || ''),
          axisLine: { onZero: false, lineStyle: { color: t.line } },
          axisTick: { show: false },
          axisLabel: { color: t.fg, fontSize: 12, fontWeight: 600 },
        },
        tooltip: {
          ...tip(t),
          trigger: 'item',
          formatter: (params: unknown) => {
            const r = sorted[firstIndex(params)];
            return r ? `<b>${esc(r.symbol)}</b><br/>${amount(r.pnl)} over ${plural(r.positions, 'position')}, ${isNum(r.won) ? r.won : 0} won` : '';
          },
        },
        series: [
          {
            type: 'bar',
            barMaxWidth: 16,
            data: values.map((x) => ({
              value: x,
              itemStyle: { color: x >= 0 ? t.up : t.down, borderRadius: x >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3] },
              label: { show: true, position: x >= 0 ? 'right' : 'left', color: t.muted, fontSize: 11, formatter: () => money(x) },
            })),
          },
        ],
      };
    },
    [perCoin, unit],
  );

  const dailyRef = useEChart(daily.length ? buildDaily : null);
  const cumulativeRef = useEChart(cumulative.length ? buildCumulative : null);
  const perCoinRef = useEChart(perCoin.length ? buildPerCoin : null);
  const noPositions = pending ? 'Positions are written by the next run.' : 'No closed position yet.';

  const cumulativePanel = (
    <div className="card span-2">
      <div className="card-h">
        <h2>
          Cumulative profit{status.since ? <span className="q2">since {day(status.since)}</span> : null}
        </h2>
        <Legend />
      </div>
      <div className="card-b fill">{cumulative.length ? <div ref={cumulativeRef} className="chart" /> : <div className="chart empty">No settled day yet.</div>}</div>
    </div>
  );
  const perCoinPanel = (
    <div className="card">
      <div className="card-h">
        <h2>Profit per {noun || 'coin'}</h2>
        <div className="meta">{pending ? '' : `${plural(status.positions, 'position')}, daily and weekly`}</div>
      </div>
      <div className="card-b fill">{perCoin.length ? <div ref={perCoinRef} className="chart" /> : <div className="chart empty">{noPositions}</div>}</div>
    </div>
  );
  const dailyPanel = (
    <div className="card span-2">
      <div className="card-h">
        <h2>Daily profit</h2>
        <Legend bars />
      </div>
      <div className="card-b fill">{daily.length ? <div ref={dailyRef} className="chart short" /> : <div className="chart short empty">No settled day yet.</div>}</div>
    </div>
  );
  return { cumulativePanel, perCoinPanel, dailyPanel };
}
