import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getAdminBacktest, getAdminBacktests, type Backtest, type BacktestList } from '../../api';
import { DASH, fixed, short, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { usePoll } from '../../hooks/usePoll';
import { useSend } from '../../hooks/useSend';
import { useParamDrawer } from '../../hooks/useUrlState';
import { Card } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Confirm from '../../components/admin/Confirm';
import Drawer, { Section } from '../../components/admin/Drawer';
import Toast from '../../components/admin/Toast';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import BacktestStart from './BacktestStart';

// /admin/backtests (owner's roadmap step 4, 8 Nov 2026): the backtests administrators started - queued, running with
// their progress, and finished with the weekend tuner's own score next to the gate's - read again every 5 s while
// one is queued or running (30 s otherwise). A backtest opens its drawer (?bt=): every scored draw, the values it ran
// with. Started here or from a model's card on Admin › Parameters; run by the scheduler (Jobs & services), or for a
// deep-learning row by the GPU worker (workers.js: its own queue, beside the schedule).
const LIVE_MS = 5000;
const IDLE_MS = 30000;

const kindText = (kind: unknown) =>
  kind === 'skill_vs_counts' ? 'skill over the counts' : kind === 'hits' ? 'hits bound' : kind === 'positional' ? 'slot-hit bound' : kind === 'log_score' ? 'log-score bound' : kind === 'capped_profit' ? 'capped profit/bet bound' : typeof kind === 'string' ? kind.replace(/_/g, ' ') : 'score';
const minutes = (s: number | null) => (s === null ? DASH : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);

export default function BacktestsPage() {
  const [every, setEvery] = useState(LIVE_MS);
  const poll = usePoll(getAdminBacktests, every);
  const drawer = useParamDrawer('bt');
  const sender = useSend(poll.refresh);
  const [starting, setStarting] = useState(false);
  const [stop, setStop] = useState<Backtest | null>(null);

  useEffect(() => {
    document.title = 'Backtests · Predictor';
  }, []);
  const d = poll.data;
  const live = Boolean(d && d.backtests.some((b) => b.status === 'queued' || b.status === 'running'));
  useEffect(() => setEvery(live ? LIVE_MS : IDLE_MS), [live]);

  const head = (
    <PageHead
      title="Backtests"
      sub={d ? `${d.backtests.filter((b) => b.status === 'running').length} running · ${d.backtests.filter((b) => b.status === 'queued').length} queued · ${d.backtests.length} kept` : undefined}
      meta={['Brussels time']}
      aside={
        <button type="button" className="btn" onClick={() => { sender.reset(); setStarting(true); }} disabled={!d}>
          New backtest
        </button>
      }
    />
  );
  if (!d) {
    return (
      <div className="admin">
        {head}
        {poll.error ? <Failed error={poll.error} retry={() => void poll.refresh()} /> : <Loading />}
      </div>
    );
  }
  return (
    <div className="admin">
      {head}
      <p className="note">
        One model on one game, walked forward over the newest draws with the weekend tuner&apos;s own score; the lockbox draws are never scored or shown. The scheduler
        runs them one at a time behind every scheduled job, and only when one can end before the next scheduled run
        {d.scheduler ? (d.scheduler.budget ? ` (one starting now may take up to ${minutes(d.scheduler.budget)})` : ' (too close to the next scheduled run now: a new one waits until after it)') : ''}.
        Start one from a model&apos;s card on <Link to="/admin/parameters">Parameters</Link> or with New backtest; it shows under Jobs &amp; services while it runs.
      </p>
      <p className="note">
        The deep-learning rows run on the GPU worker instead
        {d.gpu.configured ? ` (${d.gpu.host ?? 'set up'}${d.gpu.running ? ', one running' : ''}${d.gpu.queued ? `, ${d.gpu.queued} queued` : ''})` : ' - none is set up on this server (config/workers.json), so they cannot start here'}: one at a
        time, beside the schedule, each draw a retrain with the daily run&apos;s 240 s of training.
      </p>
      {d.backtests.length ? <BacktestTable d={d} onOpen={drawer.open} onStop={(b) => { sender.reset(); setStop(b); }} /> : <p className="note">No backtest yet.</p>}
      {drawer.value ? <BacktestDrawer key={drawer.value} id={drawer.value} onClose={drawer.close} /> : null}
      {starting ? (
        <BacktestStart
          sender={sender}
          table={d.rows}
          daysDefault={d.daysDefault}
          daysMax={d.daysMax}
          limits={d.limits}
          onStarted={() => setStarting(false)}
          onCancel={() => setStarting(false)}
        />
      ) : null}
      {stop ? (
        <Confirm
          title={stop.status === 'running' ? `Stop the backtest of ${short(stop.row)} on ${stop.game}` : `Cancel the backtest of ${short(stop.row)} on ${stop.game}`}
          action={stop.status === 'running' ? 'Stop' : 'Cancel backtest'}
          danger
          busy={sender.busy}
          error={sender.error}
          onConfirm={() => void sender.send(`/api/admin/backtests/${encodeURIComponent(stop.id)}/cancel`).then((done) => { if (done) setStop(null); })}
          onCancel={() => setStop(null)}
          lines={[stop.status === 'running' ? 'The days done so far are lost: a stopped backtest keeps no score.' : 'It is taken out of the queue.']}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

function StatusCell({ b }: { b: Backtest }) {
  if (b.status === 'running') {
    const p = b.progress;
    return (
      <div className="progress">
        <progress max={p && p.total ? p.total : 1} value={p ? p.done : 0} aria-label={`${p ? p.done : 0} of ${p ? p.total : b.days} draws`} />
        <span className="sub">
          {p ? `${p.done} of ${p.total} draws · ${minutes(p.seconds)}${p.eta !== null && p.done ? ` · about ${minutes(p.eta)} left` : ''}` : 'starting'}
        </span>
      </div>
    );
  }
  if (b.status === 'queued') return <span>queued{b.worker === 'gpu' ? ' for the GPU worker' : ''}{b.place ? `, number ${b.place}` : ''}</span>;
  return (
    <span>
      <Chip kind={b.status === 'done' ? 'flat' : b.status === 'failed' || b.status === 'ended' ? 'down' : 'dim'}>{b.status}</Chip>
      {b.error ? <span className="sub">{b.error}</span> : null}
    </span>
  );
}

function ResultCell({ b }: { b: Backtest }) {
  const r = b.result;
  if (!r) return <span className="mut">{DASH}</span>;
  const g = r.gate;
  const same = g && g.windowDays === b.days;
  return (
    <span>
      <b>{r.score === null ? 'no score' : fixed(r.score, 4)}</b> <span className="mut">{kindText(r.tuning.kind)} · {r.dayCount} draws scored{r.lockboxWithheld ? `, ${r.lockboxWithheld} lockbox withheld` : ''}</span>
      {g && g.incumbent !== null ? (
        <span className="sub">
          gate {stamp(g.date)}: served {fixed(g.incumbent, 4)}
          {g.default !== null ? `, default ${fixed(g.default, 4)}` : ''}
          {same ? ' (same window size)' : ` on its ${g.windowDays ?? '?'}-draw window: not comparable`}
        </span>
      ) : null}
    </span>
  );
}

function BacktestTable({ d, onOpen, onStop }: { d: BacktestList; onOpen: (id: string) => void; onStop: (b: Backtest) => void }) {
  return (
    <Card title="Backtests" meta={[`${d.backtests.length} kept`, 'newest first']} flush>
      <table className="tbl stk">
        <thead>
          <tr>
            <th>Model</th>
            <th>Game</th>
            <th>Draws</th>
            <th>Status</th>
            <th>Result</th>
            <th>Started by</th>
            <th aria-label="Action" />
          </tr>
        </thead>
        <tbody>
          {d.backtests.map((b) => (
            <tr key={b.id}>
              <td data-l="Model">
                <button type="button" className="linkbtn" onClick={() => onOpen(b.id)}>
                  {short(b.row)}
                </button>
                {b.values === 'tuned' ? <span className="sub">tuned values only</span> : b.manual.length ? <span className="sub">with {b.manual.length} manual</span> : null}
                {b.worker === 'gpu' ? <span className="sub">GPU worker</span> : null}
              </td>
              <td data-l="Game">{b.game}</td>
              <td data-l="Draws" className="num">
                {b.days}
              </td>
              <td data-l="Status">
                <StatusCell b={b} />
              </td>
              <td data-l="Result">
                <ResultCell b={b} />
              </td>
              <td data-l="Started by" className="mut">
                {b.by} · {stamp(b.requestedAt)}
              </td>
              <td>
                {b.status === 'running' || b.status === 'queued' ? (
                  <button type="button" className="linkbtn" onClick={() => onStop(b)}>
                    {b.status === 'running' ? 'Stop' : 'Cancel'}
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

// Every scored draw as a bar (its hits), the values the backtest ran with, and the manual ones among them.
function BacktestDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const load = useLoad(() => getAdminBacktest(id), [id]);
  const b = load.kind === 'ok' ? load.data : null;
  return (
    <Drawer title={b ? `${short(b.row)} · ${b.game}` : 'Backtest'} sub={b ? `${b.days} draws · ${b.values} values · ${b.status}` : undefined} onClose={onClose}>
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {b ? <DrawerBody b={b} /> : null}
    </Drawer>
  );
}

function DrawerBody({ b }: { b: Backtest }) {
  const r = b.result;
  const days = r ? r.days : [];
  const max = Math.max(1, ...days.map((x) => x.hits ?? 0));
  const profit = days.some((x) => x.profit !== null) ? days.reduce((s, x) => s + (x.profit ?? 0), 0) : null;
  const t = r ? r.tuning : {};
  const num = (k: string) => (typeof t[k] === 'number' ? (t[k] as number) : null);
  return (
    <>
      <Section title="Run">
        <p className="line">
          Started by {b.by} {stamp(b.requestedAt)}
          {b.startedAt ? ` · ran ${stamp(b.startedAt)}` : ''}
          {b.seconds !== null ? ` for ${minutes(b.seconds)}` : ''}
          {b.error ? <span className="mut"> · {b.error}</span> : null}
        </p>
      </Section>
      {r && t.kind === 'skill_vs_counts' ? (
        <Section title="Score">
          <p className="line">
            <b>{r.score === null ? 'no score' : fixed(r.score, 4)}</b> (the deep-learning tuner&apos;s score: minus the mean excess loss over the counts; 0 = as good as
            serving the counts, above 0 = better)
            {num('score_se') !== null ? ` · SE ${fixed(num('score_se'), 4)} over the draws (retrains on almost the same history: a floor, not a test)` : ''}
            {` · learns on ${num('learns_days') ?? 0} of ${r.dayCount} draws`}
            {num('hits') !== null && (num('hits_draws') ?? 0) > 0
              ? ` · ${fixed(num('hits'), 0)} hits on ${num('hits_draws')} tickets (each retrain's ticket against the next draw; on the record, not in the score)`
              : ''}
          </p>
          <p className="line mut">
            {r.dayCount} draws scored{r.lockboxWithheld ? `; ${r.lockboxWithheld} lockbox draws retrained but not scored or shown` : ''}. Each retrain&apos;s own validation split
            still contains the lockbox period, as in the tuner.{r.worker ? ` Ran on ${r.worker}.` : ''}
            {r.gate && r.gate.incumbent !== null
              ? ` The deep-learning gate (${stamp(r.gate.date)}, ${r.gate.windowDays ?? '?'} draws) scored the served values ${fixed(r.gate.incumbent, 4)}${r.gate.default !== null ? ` and the defaults ${fixed(r.gate.default, 4)}` : ''}${r.gate.windowDays === b.days ? '; same window size, so comparable (the window moves with each new draw)' : ': another window size, not comparable'}.`
              : ''}
          </p>
        </Section>
      ) : r ? (
        <Section title="Score">
          <p className="line">
            <b>{r.score === null ? 'no score' : fixed(r.score, 4)}</b> ({kindText(t.kind)}: the mean over the scored draws minus one standard error, as the gate
            scores) · mean {num('hits_mean') !== null ? `${fixed(num('hits_mean'), 3)} hits a draw` : num('log_score_mean') !== null ? `log score ${fixed(num('log_score_mean'), 3)}` : DASH}
            {num('score_se') !== null ? ` · SE ${fixed(num('score_se'), 3)}` : ''}
            {num('profit_per_bet') !== null ? ` · profit per bet ${fixed(num('profit_per_bet'), 2)} EUR` : ''}
            {profit !== null ? ` · profit ${fixed(profit, 2)} EUR over the draws` : ''}
          </p>
          <p className="line mut">
            {r.dayCount} draws scored{r.lockboxWithheld ? `; ${r.lockboxWithheld} lockbox draws backtested but not scored or shown` : ''}.
            {r.gate && r.gate.incumbent !== null
              ? ` The weekend gate (${stamp(r.gate.date)}, ${r.gate.windowDays ?? '?'} draws) scored the served values ${fixed(r.gate.incumbent, 4)}${r.gate.default !== null ? ` and the defaults ${fixed(r.gate.default, 4)}` : ''}${r.gate.windowDays === b.days ? '; same window size, so comparable (the window moves with each new draw)' : ': another window size, not comparable'}.`
              : ''}
          </p>
        </Section>
      ) : null}
      {days.length && t.kind === 'skill_vs_counts' ? (
        <Section title={`Per draw (${days.length})`}>
          <table className="tbl">
            <thead>
              <tr>
                <th>Draw</th>
                <th className="num">Excess loss over the counts</th>
                <th>Learning check</th>
              </tr>
            </thead>
            <tbody>
              {days.map((x, i) => (
                <tr key={`${x.date ?? ''}-${i}`}>
                  <td>{x.date ?? DASH}</td>
                  <td className="num">{x.excess === null ? DASH : fixed(x.excess, 4)}</td>
                  <td className="mut">{x.verdict ?? DASH}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="line mut">Below 0 = better than the counts on that retrain&apos;s validation windows.</p>
        </Section>
      ) : days.length ? (
        <Section title={`Hits per draw (${days.length})`}>
          <svg viewBox={`0 0 ${Math.max(days.length * 6, 60)} 40`} width="100%" height="64" preserveAspectRatio="none" role="img" aria-label={`Hits per scored draw, at most ${max}`}>
            {days.map((x, i) => (
              <rect key={i} x={i * 6} y={40 - ((x.hits ?? 0) / max) * 38} width={4} height={((x.hits ?? 0) / max) * 38} fill="currentColor" opacity={0.6}>
                <title>{`${x.date ?? ''}: ${x.hits ?? DASH} hits${x.profit !== null ? `, ${x.profit} EUR` : ''}`}</title>
              </rect>
            ))}
          </svg>
          <p className="line mut">
            {days[0].date} → {days[days.length - 1].date} · at most {max} {max === 1 ? 'hit' : 'hits'} in a draw
          </p>
        </Section>
      ) : null}
      {b.manual.length ? (
        <Section title="Manual values in this backtest">
          {b.manual.map((m) => (
            <p key={m.key} className="line">
              <code>{m.key}</code> {String(m.value)} <span className="mut">(tuned {m.tuned === null || m.tuned === undefined ? 'not set' : String(m.tuned)})</span>
            </p>
          ))}
        </Section>
      ) : null}
      {r && Object.keys(r.values).length ? (
        <Section title="Values it ran with">
          <table className="tbl">
            <tbody>
              {Object.entries(r.values).map(([k, v]) => (
                <tr key={k}>
                  <td>
                    <code>{k}</code>
                  </td>
                  <td>{typeof v === 'number' && !Number.isInteger(v) ? Number(v.toPrecision(4)) : String(v)}</td>
                  <td className="mut">{r.fromDefault.includes(k) ? 'code default (not in the file)' : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      ) : null}
    </>
  );
}
