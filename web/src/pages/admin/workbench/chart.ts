import { axisDates, day } from '../../../format';
import type { ChartOption, Tokens } from '../../../hooks/useEChart';

// The market workbench's charts share one frame with the market pages' (Charts.tsx): dates on a category axis, a value
// axis with its labels right-aligned in a fixed margin, the tokens for every colour. Every chart starts its value axis
// where the data says (money can be negative) and draws one series per line, so the lines compare on one scale.

// zoom: a thin slider under the x axis (C8, C9) - drag a handle to zoom, the window between them to pan. Nothing zooms
// inside the plot: a wheel or a pinch there would take the page's scroll on a phone. The range buttons pick what the
// slider runs over; a new range or row starts it full again.
export function frame(t: Tokens, dates: string[], opts: { left?: number; right?: number; percent?: boolean; min?: number; max?: number; zoom?: boolean } = {}): ChartOption {
  return {
    animation: false,
    textStyle: { fontFamily: t.font, color: t.fg },
    grid: { left: opts.left ?? 48, right: opts.right ?? 64, top: 14, bottom: opts.zoom ? 50 : 22, containLabel: false },
    ...(opts.zoom ? { dataZoom: [slider(t, opts.left ?? 48, opts.right ?? 64)] } : {}),
    xAxis: {
      type: 'category',
      data: axisDates(dates),
      axisLine: { lineStyle: { color: t.line } },
      axisTick: { show: false },
      axisLabel: { color: t.muted, fontSize: 11, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      scale: opts.min === undefined,
      min: opts.min,
      max: opts.max,
      splitNumber: 4,
      splitLine: { lineStyle: { color: t.line, opacity: 0.7 } },
      axisLabel: {
        color: t.muted,
        fontSize: 11,
        width: (opts.left ?? 48) - 6,
        align: 'right',
        overflow: 'truncate',
        formatter: opts.percent ? (v: number) => `${(v * 100).toFixed(0)}%` : undefined,
      },
    },
  };
}

// the slider, as wide as the plot, in the tokens' colours (a hex token with an alpha suffix for the window). Without
// brushSelect the window itself is what pans (ECharts draws no move handle then)
function slider(t: Tokens, left: number, right: number) {
  return {
    type: 'slider' as const,
    xAxisIndex: 0,
    left,
    right,
    bottom: 6,
    height: 14,
    showDetail: false,
    brushSelect: false,
    borderColor: t.line,
    backgroundColor: 'transparent',
    fillerColor: `${t.accent}1f`,
    dataBackground: { lineStyle: { color: t.faint, width: 1 }, areaStyle: { color: t.faint, opacity: 0.12 } },
    selectedDataBackground: { lineStyle: { color: t.accent, width: 1 }, areaStyle: { color: t.accent, opacity: 0.12 } },
    handleStyle: { color: t.card, borderColor: t.muted },
    emphasis: { handleStyle: { borderColor: t.accent } },
  };
}

export function tooltip(t: Tokens, dates: string[], line: (i: number) => string) {
  return {
    trigger: 'axis' as const,
    backgroundColor: t.card,
    borderColor: t.line,
    textStyle: { color: t.fg, fontSize: 12 },
    extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.12);',
    formatter: (params: unknown) => {
      const p = (Array.isArray(params) ? params[0] : params) as { dataIndex?: number } | undefined;
      const i = p?.dataIndex ?? 0;
      return dates[i] ? `<b>${day(dates[i])}</b><br/>${line(i)}` : '';
    },
  };
}

// A line series: the highlighted one thick in the accent with its last value at its end, context lines thin and faint,
// the market dashed
export function lineSeries(name: string, data: (number | null)[], color: string, kind: 'main' | 'context' | 'dashed', endLabel?: (v: unknown) => string) {
  return {
    name,
    type: 'line' as const,
    data,
    symbol: 'none',
    connectNulls: false,
    z: kind === 'main' ? 5 : kind === 'dashed' ? 4 : 2,
    lineStyle: { width: kind === 'main' ? 2.4 : kind === 'dashed' ? 1.4 : 1, color, type: kind === 'dashed' ? ('dashed' as const) : ('solid' as const), opacity: kind === 'context' ? 0.55 : 1 },
    itemStyle: { color },
    emphasis: { disabled: true },
    ...(endLabel ? { endLabel: { show: true, color, fontSize: 11, fontWeight: kind === 'main' ? 600 : 400, formatter: (p: { value?: unknown }) => endLabel(p.value) } } : {}),
  };
}

// The running total of a series of day results. workbench.js sends them to four decimals, the record's own precision:
// the sum is taken back to four decimals (float noise) and then to the cent the way the server rounds its totals, so
// the newest running value is the total the summary line prints.
export function running(xs: (number | null)[]): number[] {
  let s = 0;
  return xs.map((x) => {
    s += x ?? 0;
    return Math.round((Math.round(s * 1e4) / 1e4) * 100) / 100;
  });
}
