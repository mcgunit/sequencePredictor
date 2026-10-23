import { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { ComposeOption, ECharts } from 'echarts/core';
import type { BarSeriesOption, LineSeriesOption } from 'echarts/charts';
import type { GridComponentOption, TooltipComponentOption } from 'echarts/components';

// Only the pieces the panels and the Workbench's charts use (bars and lines), so the bundle stays small.
echarts.use([BarChart, LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

export type ChartOption = ComposeOption<BarSeriesOption | LineSeriesOption | GridComponentOption | TooltipComponentOption>;

// The CSS tokens, read at render time so the charts follow light and dark mode.
export interface Tokens {
  fg: string;
  muted: string;
  line: string;
  accent: string;
  up: string;
  down: string;
  card: string;
  font: string;
  faint: string; // the Workbench's context lines and the chance line
  warn: string;
  upBg: string;
  barB: string;
}

export function readTokens(): Tokens {
  const css = getComputedStyle(document.documentElement);
  const t = (name: string) => css.getPropertyValue(name).trim();
  return {
    fg: t('--fg'),
    muted: t('--muted'),
    line: t('--line'),
    accent: t('--accent'),
    up: t('--up'),
    down: t('--down'),
    card: t('--card'),
    font: 'IBM Plex Sans, system-ui, sans-serif',
    faint: t('--faint'),
    warn: t('--warn'),
    upBg: t('--up-bg'),
    barB: t('--bar-b'),
  };
}

// Binds an ECharts instance to the returned ref. `build` turns the tokens into
// an option and must be memoised by the caller; it runs again when it changes
// and when the colour scheme flips. Pass null while there is nothing to draw.
export function useEChart(build: ((t: Tokens) => ChartOption) | null) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<ECharts | null>(null);
  const [theme, setTheme] = useState(0);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const bump = () => setTheme((n) => n + 1);
    mq.addEventListener('change', bump);
    return () => mq.removeEventListener('change', bump);
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || !build) return;
    if (chart.current && chart.current.getDom() !== el) {
      chart.current.dispose();
      chart.current = null;
    }
    if (!chart.current) chart.current = echarts.init(el, undefined, { renderer: 'canvas' });
    chart.current.setOption(build(readTokens()), { notMerge: true });
  }, [build, theme]);

  // Resize with the element, not only the window: a chart that fills its card grows when the card
  // beside it in the same grid row is taller.
  const drawn = build !== null;
  useEffect(() => {
    const onResize = () => chart.current?.resize();
    window.addEventListener('resize', onResize);
    const el = ref.current;
    const ro = typeof ResizeObserver !== 'undefined' && el ? new ResizeObserver(onResize) : null;
    if (ro && el) ro.observe(el);
    return () => {
      window.removeEventListener('resize', onResize);
      ro?.disconnect();
    };
  }, [drawn]);

  // the instance goes with the component
  useEffect(
    () => () => {
      chart.current?.dispose();
      chart.current = null;
    },
    [],
  );

  return ref;
}
