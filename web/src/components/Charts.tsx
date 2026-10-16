import { useCallback } from 'react';
import type { MarketView } from '../api';
import { axisDates, day, esc, isNum, money, plural } from '../format';
import { useEChart, type ChartOption, type Tokens } from '../hooks/useEChart';

const num = (v: unknown): number | null => (isNum(v) ? v : null);

function firstIndex(params: unknown): number {
  const p = (Array.isArray(params) ? params[0] : params) as { dataIndex?: number } | undefined;
  return p?.dataIndex ?? 0;
}

// Axes and text coloured from the tokens, no animation.
function frame(t: Tokens, categories: string[]): ChartOption {
  return {
    animation: false,
    textStyle: { fontFamily: t.font, color: t.fg },
    grid: { left: 8, right: 10, top: 12, bottom: 4, containLabel: true },
    xAxis: {
      type: 'category',
      data: categories,
      axisLine: { lineStyle: { color: t.line } },
      axisTick: { show: false },
      axisLabel: { color: t.muted, fontSize: 11 },
    },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: t.line } }, axisLabel: { color: t.muted, fontSize: 11 } },
  };
}

function tip(t: Tokens) {
  return { backgroundColor: t.card, borderColor: t.line, textStyle: { color: t.fg, fontSize: 12 } };
}

function bar(v: number | null, t: Tokens) {
  const x = v ?? 0;
  return { value: x, itemStyle: { color: x >= 0 ? t.up : t.down, borderRadius: x >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3] } };
}

function dashed(data: (number | null)[], t: Tokens) {
  return {
    name: 'Buying everything',
    type: 'line' as const,
    data,
    symbol: 'none',
    lineStyle: { width: 1.2, color: t.muted, type: 'dashed' as const },
    itemStyle: { color: t.muted },
  };
}

export default function Charts({ data }: { data: MarketView }) {
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
          return d ? `${day(d.date)}<br/>Followed plan: ${amount(d.pnl)}<br/>Buying everything: ${amount(d.market)}` : '';
        },
      },
      series: [
        { name: 'Followed plan', type: 'bar', barWidth: '60%', data: daily.map((d) => bar(num(d.pnl), t)) },
        dashed(
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
          return d ? `${day(d.date)}<br/>Followed plan: ${amount(d.plan)}<br/>Buying everything: ${amount(d.market)}` : '';
        },
      },
      series: [
        {
          name: 'Followed plan',
          type: 'line',
          data: cumulative.map((d) => num(d.plan)),
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { width: 2.2, color: t.accent },
          itemStyle: { color: t.accent },
          areaStyle: { color: t.accent, opacity: 0.08 },
        },
        dashed(
          cumulative.map((d) => num(d.market)),
          t,
        ),
      ],
    }),
    [cumulative, unit],
  );

  const buildPerCoin = useCallback(
    (t: Tokens): ChartOption => ({
      ...frame(
        t,
        perCoin.map((r) => r.symbol || ''),
      ),
      tooltip: {
        ...tip(t),
        trigger: 'item',
        formatter: (params: unknown) => {
          const r = perCoin[firstIndex(params)];
          return r ? `${esc(r.symbol)}: ${amount(r.pnl)} over ${plural(r.positions, 'position')}, ${isNum(r.won) ? r.won : 0} won` : '';
        },
      },
      series: [{ type: 'bar', barWidth: '55%', data: perCoin.map((r) => bar(num(r.pnl), t)) }],
    }),
    [perCoin, unit],
  );

  const dailyRef = useEChart(daily.length ? buildDaily : null);
  const cumulativeRef = useEChart(cumulative.length ? buildCumulative : null);
  const perCoinRef = useEChart(perCoin.length ? buildPerCoin : null);

  return (
    <div className="charts">
      <div className="card">
        <h2>
          Daily profit <small>{unit ? `${unit}, ` : ''}day game, live days</small>
        </h2>
        {daily.length ? <div ref={dailyRef} className="chart" /> : <div className="chart empty">No settled day yet.</div>}
        <div className="legend">Bars: the followed plan per day. Dashed: buying every {noun || 'coin'} every day.</div>
      </div>
      <div className="card">
        <h2>Cumulative profit</h2>
        {cumulative.length ? <div ref={cumulativeRef} className="chart" /> : <div className="chart empty">No settled day yet.</div>}
        <div className="legend">Running total{status.since ? ` since ${day(status.since)}` : ''}, against the market.</div>
      </div>
      <div className="card">
        <h2>
          Per {noun || 'coin'} <small>day positions · {plural(status.positions, 'position')}</small>
        </h2>
        {perCoin.length ? <div ref={perCoinRef} className="chart" /> : <div className="chart empty">No closed position yet.</div>}
        <div className="legend">Net result per {noun || 'coin'}; positions on hover.</div>
      </div>
    </div>
  );
}
