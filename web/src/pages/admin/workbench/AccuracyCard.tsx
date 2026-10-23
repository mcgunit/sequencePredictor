import { useCallback } from 'react';
import { Card, Legend } from '../../../components/admin/Card';
import { day, isNum, share, short } from '../../../format';
import { useEChart, type ChartOption, type Tokens } from '../../../hooks/useEChart';
import type { Workbench } from '../../../workbench';
import { frame, lineSeries, tooltip } from './chart';

// Accuracy per day (C13): the mean exact rate over every row each scored day, the best row's that day as a bar, and
// chance as a dashed line, on one 0-100% scale.
export default function AccuracyCard({ w }: { w: Workbench }) {
  const a = w.accuracy;
  const dates = a.days.map((d) => d.date);
  const chance = a.chance ?? w.chance.exact ?? 0.1;
  const build = useCallback(
    (t: Tokens): ChartOption => ({
      ...frame(t, dates, { left: 40, right: 16, percent: true, min: 0, max: 1 }),
      tooltip: tooltip(t, dates, (i) => {
        const d = a.days[i];
        return `Mean over every row: ${share(d.mean)}<br/>Best row: ${share(d.best)}${d.bestModel ? ` (${short(d.bestModel)})` : ''}<br/>Chance: ${share(chance)}`;
      }),
      series: [
        { name: 'Best row', type: 'bar', barMaxWidth: 10, data: a.days.map((d) => d.best), itemStyle: { color: t.barB, borderRadius: [2, 2, 0, 0] } },
        lineSeries('Chance', dates.map(() => chance), t.muted, 'dashed'),
        { ...lineSeries('Mean', a.days.map((d) => d.mean), t.accent, 'main'), symbol: 'circle', symbolSize: 4, showSymbol: dates.length <= 60 },
      ],
    }),
    [a, chance],
  );
  const ref = useEChart(dates.length ? build : null);
  const means = a.days.map((d) => d.mean).filter(isNum);
  const avg = means.length ? means.reduce((s, x) => s + x, 0) / means.length : null;
  const above = means.filter((x) => x > chance).length;
  return (
    <Card id="accuracy" title="Accuracy per day" meta={[`${a.days.length} scored days`, a.days.length ? `${day(dates[0])} – ${day(dates[dates.length - 1])}` : '']}>
      {a.days.length ? (
        <>
          <p className="line">
            Mean exact rate over every row: <b>{share(avg)}</b> on average over the {a.days.length} days, above chance ({share(chance)}) on {above} of them. A day is {w.instruments.filter((i) => i.active).length} calls per row: the best row of a
            day is luck as often as not.
          </p>
          <div ref={ref} className="chart" role="img" aria-label={`Mean exact rate per day over ${a.days.length} days, average ${share(avg)}, chance ${share(chance)}`} />
          <Legend
            items={[
              { mark: '', label: 'mean exact rate, every row' },
              { mark: 'ci', label: "the day's best row" },
              { mark: 'dash', label: 'chance' },
            ]}
          />
        </>
      ) : (
        <div className="empty">No scored day yet.</div>
      )}
    </Card>
  );
}
