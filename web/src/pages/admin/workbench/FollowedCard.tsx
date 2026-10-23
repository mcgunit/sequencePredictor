import { useCallback, useState } from 'react';
import { Card, Legend, Scroll } from '../../../components/admin/Card';
import { DASH, day, money, short } from '../../../format';
import { useEChart, type ChartOption, type Tokens } from '../../../hooks/useEChart';
import type { Followed, FollowedPoint, Workbench } from '../../../workbench';
import { frame, lineSeries, running, tooltip } from './chart';
import { ahead, spanText } from './PlanCards';

// Following the plan (C5-C7): what doing exactly what the plan cards said made, live days only, against buying
// everything under the same rule; each day's row picked from the record as it stood before that day. The late days
// (tickets too late to follow) and the back-built history sit apart, closed, with their totals on their summary line.

const SHOWN = 31;

function RunningChart({ f, cur }: { f: Followed; cur: string }) {
  const dates = f.live.map((p) => p[0]);
  const plan = running(f.live.map((p) => p[4]));
  const market = running(f.live.map((p) => p[5]));
  const build = useCallback(
    (t: Tokens): ChartOption => ({
      ...frame(t, dates),
      tooltip: tooltip(t, dates, (i) => `Plan: ${money(plan[i])} ${cur}<br/>Buying everything: ${money(market[i])} ${cur}`),
      series: [lineSeries('Buying everything', market, t.muted, 'dashed', (v) => money(v)), lineSeries('Plan', plan, t.accent, 'main', (v) => money(v))],
    }),
    [f, cur],
  );
  const ref = useEChart(dates.length > 1 ? build : null);
  if (dates.length < 2) return null;
  return (
    <>
      <div ref={ref} className="chart short" role="img" aria-label={`Running totals over ${dates.length} live days: plan ${money(plan[plan.length - 1])}, buying everything ${money(market[market.length - 1])} ${cur}`} />
      <Legend
        items={[
          { mark: '', label: 'following the plan, running total' },
          { mark: 'dash', label: 'buying everything, running total' },
        ]}
      />
    </>
  );
}

function pickedBy(p: FollowedPoint) {
  if (!p[1]) return 'no plan yet';
  return `${short(p[1])}${p[2] === 'exact' ? ' (by exact rate)' : ''}`;
}

function PointsTable({ points, week, withRunning, cur }: { points: FollowedPoint[]; week: boolean; withRunning: boolean; cur: string }) {
  const [all, setAll] = useState(false);
  const run = running(points.map((p) => p[4]));
  const runMarket = running(points.map((p) => p[5]));
  const rows = points.map((p, i) => ({ p, run: run[i], runMarket: runMarket[i] })).reverse();
  const shown = all ? rows : rows.slice(0, SHOWN);
  return (
    <>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>{week ? 'Week ending' : 'Day'}</th>
              <th>Row followed</th>
              <th className="num hide-sm">Positions</th>
              <th className="num">Result {cur}</th>
              {withRunning ? <th className="num">Running</th> : null}
              <th className="num hide-sm">Buying everything</th>
              {withRunning ? <th className="num hide-sm">Its running</th> : null}
            </tr>
          </thead>
          <tbody>
            {shown.map(({ p, run: r, runMarket: rm }) => (
              <tr key={p[0]}>
                <td>{day(p[0])}</td>
                <td>
                  {pickedBy(p)}
                  <span className="sub show-sm">
                    {p[3]} {p[3] === 1 ? 'position' : 'positions'} · buying everything {money(p[5])}
                    {withRunning ? `, running ${money(rm)}` : ''}
                  </span>
                </td>
                <td className="num hide-sm">{p[3]}</td>
                <td className="num">{money(p[4])}</td>
                {withRunning ? <td className="num strong">{money(r)}</td> : null}
                <td className="num hide-sm">{money(p[5])}</td>
                {withRunning ? <td className="num hide-sm">{money(rm)}</td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      </Scroll>
      {rows.length > SHOWN ? (
        <p className="note pad">
          <button type="button" className="linkbtn" onClick={() => setAll((x) => !x)}>
            {all ? `Show the newest ${SHOWN}` : `Show all ${rows.length}`}
          </button>
        </p>
      ) : null}
    </>
  );
}

function Part({ f, w, title, chart }: { f: Followed | null; w: Workbench; title: string; chart: boolean }) {
  const cur = w.currency;
  if (!f) {
    return (
      <div className="card-b">
        <h3 className="subh">{title}</h3>
        <p className="line">No followed record yet.</p>
      </div>
    );
  }
  const unit = f.week ? 'weeks' : 'days';
  // a week is dated by its end, so the weeks' dates say so
  const span = f.live.length ? spanText(f.live[0][0], f.live[f.live.length - 1][0]) : '';
  return (
    <>
      <div className="card-b">
        <h3 className="subh">
          {title}, {f.rule}
        </h3>
        <p className="line">
          {f.liveDays ? (
            <>
              {f.week ? (
                <>
                  Followed for {f.liveDays} live {f.liveDays === 1 ? 'week' : unit} ({f.liveDays === 1 ? 'the week ending' : 'the weeks ending'} {span}, {f.positions ?? DASH} positions)
                </>
              ) : (
                <>
                  Followed since {day(f.firstLive)} ({f.liveDays} live {unit}
                  {span ? `, ${span}` : ''}, {f.positions ?? DASH} positions)
                </>
              )}
              : <b>{money(f.total)} {cur}</b> · buying every {w.noun} under the same rule {money(f.market)} (
              {ahead(f.ahead)}). The row each {f.week ? 'week' : 'day'} is the plan&apos;s pick from the record before it: the best book with at least {f.minTrades} positions,
              else the best exact rate.
            </>
          ) : (
            <>
              No live {unit} yet
              {f.firstLive ? (f.week ? ` (the first that could be followed ends ${day(f.firstLive)})` : ` (the first could be followed from ${day(f.firstLive)})`) : ''}.
            </>
          )}
        </p>
        {chart && f.live.length ? <RunningChart f={f} cur={cur} /> : null}
      </div>
      {f.live.length ? <PointsTable points={f.live} week={f.week} withRunning cur={cur} /> : null}
      <div className="card-b">
        {f.late.n ? (
          <details className="fold">
            <summary>
              Late {unit} ({f.late.n}): {money(f.late.total)} {cur}, buying everything {money(f.late.market)}{' '}
              <span className="mut">· tickets made too late to follow, not in the live total</span>
            </summary>
            <PointsTable points={f.late.points} week={f.week} withRunning={false} cur={cur} />
          </details>
        ) : null}
        {f.backBuilt.n ? (
          <details className="fold">
            <summary>
              Back-built {unit} {f.week ? 'ending ' : ''}before {day(f.firstLive)} ({f.backBuilt.n}): the rule would have made {money(f.backBuilt.total)} {cur}, buying everything{' '}
              {money(f.backBuilt.market)} <span className="mut">· tickets made after the {unit} closed, not a record anyone could have followed</span>
            </summary>
            <PointsTable points={f.backBuilt.points} week={f.week} withRunning={false} cur={cur} />
          </details>
        ) : null}
      </div>
    </>
  );
}

export default function FollowedCard({ w }: { w: Workbench }) {
  return (
    <Card
      id="followed"
      title="Following the plan"
      help={`The result a reader would have after doing exactly what the plan cards said: every ${w.market === 'crypto' ? 'UTC day' : 'trading day'} Today's plan, every week the week game. A day counts as live when its ticket was made before the plan's entry. ${w.stake ?? 100} ${w.currency} per ${w.noun} bought and a fee on each leg; a sized row scaled by its size; down or flat calls buy nothing and shorts are not included. Paper money, no slippage; a handful of days says nothing about skill.`}
      meta={[w.followed.day ? `${w.followed.day.liveDays} live days` : '', w.followed.week ? `${w.followed.week.liveDays} live weeks` : '']}
      flush
    >
      <Part f={w.followed.day} w={w} title="Day to day" chart />
      <Part f={w.followed.week} w={w} title="Week to week" chart={false} />
    </Card>
  );
}
