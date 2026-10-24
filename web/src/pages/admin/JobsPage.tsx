import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { getAdminJobs, type JobLine, type JobsView, type RunItem, type ServiceLine } from '../../api';
import { DASH, cap, isNum, span, stamp, time, weekday } from '../../format';
import { usePoll } from '../../hooks/usePoll';
import { useSend } from '../../hooks/useSend';
import { useFilterParam, useParamDrawer } from '../../hooks/useUrlState';
import Tile from '../../components/Tile';
import { Card, Legend } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Confirm from '../../components/admin/Confirm';
import Toast from '../../components/admin/Toast';
import { DurationBar } from '../../components/admin/Bars';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import { GroupRow } from './bits';
import { JobDrawer, RunLogDrawer } from './JobsDrawers';
import { MODE_NOTE, MODE_WORD, PLAN_GROUPS, POLL_MS, SERVICE_SHORT, SERVICE_WORD, afterText, basisText, cancelAsk, lockText, runsWindow, serviceAsk, stopAsk, type Ask } from './jobs';
import { durationText, jobLabel, usualRange, usualText } from './week';

// /admin/jobs: the schedule's health at a glance - the mode, what runs and for how long against its usual, what
// waits, today's predictor, the Council API, the last failure - and the actions one step away behind a confirmation
// that names the cost. A job opens its drawer (?job=: its command, its usual duration, Run now), a run its log
// (?run=: warnings first, then its last lines). The JSON is read again every 15 s while the page is visible. Every
// time is Brussels time.
export default function JobsPage() {
  const poll = usePoll(getAdminJobs, POLL_MS);
  const run = useParamDrawer('run', ['job']);
  const job = useParamDrawer('job', ['run']);
  const [filter, setFilter] = useFilterParam('runs');
  const sender = useSend(poll.refresh);
  const [ask, setAsk] = useState<Ask | null>(null);
  const { hash } = useLocation();

  useEffect(() => {
    document.title = 'Jobs & services · Predictor';
  }, []);
  const loaded = poll.data !== null;
  useEffect(() => {
    if (loaded && hash) document.getElementById(hash.slice(1))?.scrollIntoView({ block: 'start' });
  }, [hash, loaded]);

  const d = poll.data;
  if (!d) {
    return (
      <>
        <PageHead title="Jobs & services" />
        {poll.error ? <Failed error={poll.error} retry={() => void poll.refresh()} /> : <Loading />}
      </>
    );
  }

  const onAsk = (a: Ask) => {
    sender.reset();
    setAsk(a);
  };
  const confirm = async () => {
    if (!ask) return;
    if (await sender.send(ask.url, ask.body)) setAsk(null);
  };
  const openJob = d.jobs.find((j) => j.key === job.value) ?? null;

  return (
    <div className="admin">
      <PageHead title="Jobs & services" sub={d.mode ? `Schedule ${MODE_WORD[d.mode.mode].toLowerCase()}` : 'Schedule not known'} meta={[`as of ${stamp(d.builtAt)}`, 'read every 15 s', 'Brussels time']} />
      {poll.error ? (
        <p className="statusline" role="status">
          <Chip kind="warn">not refreshed</Chip> {poll.error.message} · showing the answer of {time(d.builtAt)}
        </p>
      ) : null}
      {d.errors.includes('jobs') ? (
        <p className="statusline" role="status">
          <Chip kind="down">schedule not available</Chip> This server has no schedule running: nothing below can be started or stopped.
        </p>
      ) : null}
      <Tiles d={d} />
      <div className="grid">
        <RunningCard d={d} onAsk={onAsk} onOpenRun={run.open} />
        <CouncilCard d={d} onAsk={onAsk} />
        <ScheduleCard d={d} onOpenJob={job.open} onOpenRun={run.open} />
        <RunsCard d={d} filter={filter} setFilter={setFilter} onOpenRun={run.open} />
      </div>
      {run.value ? (
        <RunLogDrawer key={run.value} id={run.value} d={d} onClose={run.close} />
      ) : openJob ? (
        <JobDrawer job={openJob} d={d} onClose={job.close} onAsk={onAsk} onOpenRun={run.open} onRuns={(key) => setFilter(key, '#runs', ['job'])} />
      ) : null}
      {ask ? (
        <Confirm
          title={ask.title}
          lines={ask.lines}
          action={ask.action}
          danger={ask.danger}
          tz={ask.tz}
          busy={sender.busy}
          error={sender.error}
          onConfirm={() => void confirm()}
          onCancel={() => setAsk(null)}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

// --- tiles ---------------------------------------------------------------------------------------------------------
function Tiles({ d }: { d: JobsView }) {
  const r = d.running;
  const predictor = d.jobs.find((j) => j.daily) ?? null;
  const today = d.today;
  const sv = d.services?.[0] ?? null;
  const lf = d.lastFailure;
  const win = runsWindow(d);
  return (
    <div className="kpis">
      <Tile
        href="#schedule"
        label="Mode"
        value={d.mode ? MODE_WORD[d.mode.mode] : DASH}
        unit={d.mode ? 'schedule' : 'not known'}
        foot={[
          d.mode ? { text: MODE_NOTE[d.mode.mode], warn: d.mode.mode === 'off' } : '',
          d.mode?.reason ? `source: ${d.mode.reason}` : '',
        ]}
      />
      <Tile
        href="#running"
        label="Running"
        value={r ? durationText(r.minutes, r.usual) : DASH}
        unit={r ? jobLabel(r.key, r.name) : 'nothing running'}
        foot={
          r
            ? [
                `since ${stamp(r.start)}`,
                r.usual ? `${usualText(r.usual, r.minutes)} · median ${durationText(r.usual.median, r.usual)} · ${basisText(r.usual)}` : 'no earlier run to compare',
                r.overUsual ? { text: 'longer than its usual maximum', warn: true } : '',
              ]
            : [lockText(d.lock)]
        }
      />
      <Tile
        href="#running"
        label="Queue"
        value={String(d.queue.length)}
        unit={d.queue.length === 1 ? 'job waiting' : 'jobs waiting'}
        foot={d.queue.length ? d.queue.map((q) => `${jobLabel(q.key, q.name)}${q.manual ? ' (by hand)' : ''} · ${durationText(q.waitedMinutes)}`) : ['nothing waits']}
      />
      <Tile
        href="#schedule"
        label="Today"
        value={today?.run ? durationText(today.run.minutes, predictor?.usual) : today?.running ? 'running' : DASH}
        unit={today?.run ? `predictor ${today.run.result}` : today?.running ? `predictor since ${time(r?.start)}` : 'no predictor run yet'}
        foot={[
          today ? `${weekday(today.day)}, Brussels day` : 'no daily job here',
          today?.run ? `started ${time(today.run.start)} · ${usualText(predictor?.usual ?? null, today.run.minutes)}` : '',
          today?.run && today.run.result !== 'ok' ? { text: `${today.run.result}: see Runs`, warn: true } : '',
          predictor?.nextAt ? `next ${stamp(predictor.nextAt)}` : '',
        ]}
      />
      <Tile
        href="#council"
        label={sv?.name ?? 'Council API'}
        value={sv ? SERVICE_SHORT[sv.status] ?? cap(sv.status) : DASH}
        unit={sv ? (sv.healthy ? 'answers' : 'no answer') : 'not known'}
        foot={
          sv
            ? [
                sv.status === 'external' || sv.status === 'disabled' ? SERVICE_WORD[sv.status] : '',
                isNum(sv.upMinutes) ? `up ${durationText(sv.upMinutes)} since ${stamp(sv.since)}` : '',
                sv.summary ?? '',
                `${sv.restarts} restart${sv.restarts === 1 ? '' : 's'} · checked ${time(sv.checkedAt)}`,
                !sv.healthy ? { text: 'its health check does not answer', warn: true } : '',
              ]
            : [d.servicesError ? { text: `the health probe did not answer: ${d.servicesError}`, warn: true } : 'no supervised service here']
        }
      />
      <Tile
        href="#runs"
        label="Last failure"
        value={lf ? lf.result : d.runs.ran ? 'None' : DASH}
        unit={lf ? jobLabel(lf.key, lf.name ?? undefined) : d.runs.ran ? `in ${d.runs.ran} run${d.runs.ran === 1 ? '' : 's'}` : 'no run kept yet'}
        foot={[
          lf ? stamp(lf.start) : '',
          lf?.note ?? '',
          lf ? `the newest of the ${d.runs.ran} runs kept that did not end ok` : '',
          win.from ? `runs kept: ${span(win.from, win.to)}` : '',
        ]}
      />
    </div>
  );
}

// --- running and queue -----------------------------------------------------------------------------------------------
function RunningCard({ d, onAsk, onOpenRun }: { d: JobsView; onAsk: (a: Ask) => void; onOpenRun: (id: string) => void }) {
  const r = d.running;
  const label = r ? jobLabel(r.key, r.name) : '';
  return (
    <Card
      id="running"
      className="span-2"
      title="Running & queue"
      help={
        <>
          One job runs at a time, behind the process.lock every pipeline script takes; the others wait in the queue in the order they came. The bar is the
          running job&apos;s time on the scale of its usual median (the tick), from 0 to 2.5×; the shaded band is the usual range of its earlier successful
          runs. Next to its start, its process id and the commit it runs at. Stop sends SIGTERM to the job&apos;s whole process group; Cancel takes a waiting
          job out of the queue. Both ask first.
        </>
      }
      meta={[`as of ${time(d.builtAt)}`, 'Brussels time']}
    >
      {r ? (
        <div className="now">
          <div className="now-h">
            <button type="button" className="gbtn row" onClick={() => onOpenRun(r.id)} aria-haspopup="dialog">
              {label}
            </button>
            <span className="mut">
              since {stamp(r.start)}
              {r.pid !== null ? ` · pid ${r.pid}` : ''}
              {r.sha ? (
                <>
                  {' · at '}
                  <code>{r.sha}</code>
                </>
              ) : null}
            </span>
            <button type="button" className="btn danger" disabled={r.pid === null} onClick={() => onAsk(stopAsk(r, d))}>
              Stop
            </button>
          </div>
          {r.pid === null ? <p className="note">It has not reported its process id yet: Stop works once it has.</p> : null}
          {r.reason ? <p className="note">{cap(r.reason)}.</p> : null}
          <DurationBar minutes={r.minutes} usual={r.usual} label={`${label}: ${durationText(r.minutes, r.usual)} so far, ${usualText(r.usual, r.minutes)}`} />
          <p className={r.overUsual ? 'line warn' : 'line'}>
            <b>{durationText(r.minutes, r.usual)}</b> so far · {usualText(r.usual, r.minutes)}
            {r.usual ? ` · median ${durationText(r.usual.median, r.usual)}, ${basisText(r.usual)}` : ''}
            {isNum(r.vsMedian) ? ` · ${r.vsMedian.toFixed(2)}× the median` : ''}
            {r.overUsual ? ' · longer than its usual maximum' : ''}
          </p>
          <Legend
            items={[
              { mark: 'fill', label: 'so far' },
              { mark: 'band', label: 'usual range' },
              { mark: 'tick', label: 'usual median' },
            ]}
          />
          <h3 className="subh">Its log · the last {r.log.length} lines</h3>
          <pre className="logtail">{r.log.length ? r.log.join('\n') : 'no output yet'}</pre>
        </div>
      ) : (
        <p className="note">Nothing running · {lockText(d.lock)}.</p>
      )}
      <h3 className="subh">Queue · {d.queue.length ? `${d.queue.length} waiting` : 'empty'}</h3>
      {d.queue.length ? (
        <ul className="qlist">
          {d.queue.map((q, i) => (
            <li key={q.key}>
              <span className="qn">{i + 1}</span>
              <span className="qt">
                <b>{jobLabel(q.key, q.name)}</b>
                <small>
                  queued {stamp(q.queuedAt)} · waiting {durationText(q.waitedMinutes)}
                  {q.reason ? ` · ${q.reason}` : ''}
                </small>
              </span>
              {q.manual ? <Chip kind="flat">by hand</Chip> : null}
              <button type="button" className="btn" onClick={() => onAsk(cancelAsk(q, d))}>
                Cancel
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="note">Nothing waits{r ? ` behind ${label}` : ''}.</p>
      )}
    </Card>
  );
}

// --- the Council API (the supervised services) -------------------------------------------------------------------------
function CouncilCard({ d, onAsk }: { d: JobsView; onAsk: (a: Ask) => void }) {
  if (!d.services) {
    return (
      <Card id="council" title="Council API" meta={['Brussels time']}>
        <p className="note">
          {d.servicesError ? `The services' health probe did not answer within 6 s: ${d.servicesError}. The page asks again in 15 s.` : 'No supervised service on this server.'}
        </p>
      </Card>
    );
  }
  return (
    <div className="stack">
      {d.services.map((sv) => (
        <ServiceCard key={sv.key} sv={sv} onAsk={onAsk} />
      ))}
    </div>
  );
}

function ServiceCard({ sv, onAsk }: { sv: ServiceLine; onAsk: (a: Ask) => void }) {
  const log = sv.log;
  return (
    <Card
      id={sv.key === 'councilApi' ? 'council' : sv.key}
      title={sv.name}
      help={sv.description}
      meta={[`checked ${time(sv.checkedAt)}`, 'Brussels time']}
      foot={
        <>
          <div className="acts">
            <button type="button" className="btn" disabled={!sv.actions.start} onClick={() => onAsk(serviceAsk(sv, 'start'))}>
              Start
            </button>
            <button type="button" className="btn" disabled={!sv.actions.restart} onClick={() => onAsk(serviceAsk(sv, 'restart'))}>
              Restart
            </button>
            <button type="button" className="btn danger" disabled={!sv.actions.stop} onClick={() => onAsk(serviceAsk(sv, 'stop'))}>
              Stop
            </button>
          </div>
          {sv.status === 'external' ? <span className="fnote">Started outside the server: these buttons cannot manage it.</span> : null}
        </>
      }
    >
      <dl className="kv">
        <dt>State</dt>
        <dd>
          <span className="ln">
            <Chip kind={sv.status === 'failed' ? 'down' : 'flat'}>{SERVICE_WORD[sv.status] ?? sv.status}</Chip>
            {sv.healthy ? 'answers its health check' : <Chip kind="down">no answer to its health check</Chip>}
          </span>
          {sv.summary ? <span className="sub">{sv.summary}</span> : null}
        </dd>
        <dt>Up</dt>
        <dd>
          {isNum(sv.upMinutes) ? `${durationText(sv.upMinutes)} since ${stamp(sv.since)}` : DASH}
          <span className="sub">
            {[sv.pid !== null ? `pid ${sv.pid}` : '', sv.port !== null ? `port ${sv.port}` : ''].filter(Boolean).join(' · ') || DASH}
          </span>
        </dd>
        <dt>Restarts</dt>
        <dd>
          {sv.restarts}
          {sv.lastExit ? ` · last exit ${sv.lastExit.signal ? `signal ${sv.lastExit.signal}` : `code ${sv.lastExit.code ?? DASH}`} ${stamp(sv.lastExit.at)}` : ''}
          {sv.lastError ? <span className="sub">last error: {sv.lastError}</span> : null}
        </dd>
        <dt>Runs</dt>
        <dd>
          {sv.command ? <code>{sv.command}</code> : DASH}
          {sv.cwd ? <span className="sub">in {sv.cwd}</span> : null}
        </dd>
      </dl>
      <h3 className="subh">
        Its log{log?.file ? ` · ${log.file}` : ''} · the last {log?.lines.length ?? 0} lines
      </h3>
      <pre className="logtail">{log && log.lines.length ? log.lines.join('\n') : 'no log yet'}</pre>
      {log && log.healthDropped ? (
        <p className="note">
          {log.healthDropped} /health probe line{log.healthDropped === 1 ? '' : 's'} left out
          {log.scanned ? ` of the last ${log.scanned.lines} lines (${log.scanned.kb} KB${log.scanned.whole ? ', the whole log' : ''})` : ''}.
        </p>
      ) : null}
    </Card>
  );
}

// --- the schedule ----------------------------------------------------------------------------------------------------
function ScheduleCard({ d, onOpenJob, onOpenRun }: { d: JobsView; onOpenJob: (key: string) => void; onOpenRun: (id: string) => void }) {
  const queued = new Set(d.queue.map((q) => q.key));
  const groups = PLAN_GROUPS.map((g) => ({ ...g, jobs: d.jobs.filter((j) => (j.plan ?? null) === g.plan) })).filter((g) => g.jobs.length);
  const other = d.jobs.filter((j) => !PLAN_GROUPS.some((g) => g.plan === (j.plan ?? null)));
  if (other.length) groups.push({ plan: 'other', label: 'Other', name: 'other jobs', jobs: other });
  return (
    <Card
      id="schedule"
      className="span-3"
      title="Schedule"
      flush
      help={
        <>
          The daily predictor runs at its slot; the tuning chain starts when Saturday&apos;s predictor finishes, the controls when Sunday&apos;s does, each job
          after the one before. Took: the last run against the usual range of the runs of that job before it. A job&apos;s name opens what it runs, its usual
          duration and Run now. To stop the schedule without a restart, create config/scheduler.disabled on the server: nothing new starts while it is there,
          Run now included. Remove it to resume.
        </>
      }
      meta={[`${d.jobs.length} jobs`, 'Brussels time']}
    >
      <table className="tbl stk">
        <thead>
          <tr>
            <th>Job</th>
            <th>Next run</th>
            <th>Last run</th>
            <th>Took</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => [
            <GroupRow key={`g-${g.label}`} cols={5}>
              {g.label}
            </GroupRow>,
            ...g.jobs.map((j) => <ScheduleRow key={j.key} j={j} d={d} queued={queued.has(j.key)} onOpenJob={onOpenJob} onOpenRun={onOpenRun} />),
          ])}
        </tbody>
      </table>
    </Card>
  );
}

function ScheduleRow({ j, d, queued, onOpenJob, onOpenRun }: { j: JobLine; d: JobsView; queued: boolean; onOpenJob: (key: string) => void; onOpenRun: (id: string) => void }) {
  const running = d.running?.key === j.key;
  // the last run as the Runs card reads it: against the runs of this job before it
  const item = j.last ? d.runs.items.find((x) => x.id === j.last?.id) ?? null : null;
  const usual = item?.usual ?? j.usual;
  const over = item ? item.overUsual === true : Boolean(j.last && isNum(j.last.minutes) && usual && isNum(usual.max) && j.last.minutes > usual.max);
  const range = usualRange(usual, j.last?.minutes ?? null);
  const after = afterText(j, d);
  return (
    <tr>
      <td className="lead">
        <button type="button" className="gbtn row" onClick={() => onOpenJob(j.key)} aria-haspopup="dialog">
          {jobLabel(j.key, j.name)}
        </button>
      </td>
      <td data-l="Next">
        {running ? <Chip kind="flat">running now</Chip> : queued ? <Chip kind="flat">queued</Chip> : j.nextAt ? stamp(j.nextAt) : <span className="mut">{after?.when ?? j.trigger ?? DASH}</span>}
      </td>
      <td data-l="Last run">
        {j.last ? (
          <button type="button" className="linkbtn" onClick={() => onOpenRun(j.last?.id ?? '')} aria-haspopup="dialog">
            {stamp(j.last.start)}
          </button>
        ) : (
          <span className="mut">none kept</span>
        )}
      </td>
      <td data-l="Took" className={over ? 'warn' : undefined}>
        {j.last ? (
          <>
            {durationText(j.last.minutes, usual)}
            <span className="sub">{range && usual ? `usual ${range} · ${usual.n} earlier run${usual.n === 1 ? '' : 's'}` : 'no earlier run'}</span>
          </>
        ) : (
          DASH
        )}
      </td>
      <td data-l="Result">{j.last ? <Chip kind={j.last.result === 'ok' ? 'flat' : 'down'}>{j.last.result}</Chip> : <span className="mut">{DASH}</span>}</td>
    </tr>
  );
}

// --- every kept run --------------------------------------------------------------------------------------------------
function RunsCard({ d, filter, setFilter, onOpenRun }: { d: JobsView; filter: string | null; setFilter: (v: string | null) => void; onOpenRun: (id: string) => void }) {
  const counts = new Map<string, number>();
  d.runs.items.forEach((x) => counts.set(x.key, (counts.get(x.key) ?? 0) + 1));
  const known = filter && counts.has(filter) ? filter : null;
  const items = known ? d.runs.items.filter((x) => x.key === known) : d.runs.items;
  // the count and the window of the rows below (a job's, when one is chosen); kept is the schedule's cap, not a count
  const win = runsWindow(d, items);
  const jobs = d.jobs.filter((j) => counts.has(j.key));
  const knownJob = known ? d.jobs.find((j) => j.key === known) : null;
  const all = `${d.runs.shown} run${d.runs.shown === 1 ? '' : 's'}`;
  const shown = known ? `${jobLabel(known, knownJob?.name)}: ${items.length} of the ${all}` : all;
  return (
    <Card
      id="runs"
      className="span-3"
      title="Runs"
      flush
      help={
        <>
          Every run the schedule keeps, newest first: dry runs (the schedule only recorded them) and waits on the lock included. Every bar is on one scale: the
          run&apos;s time as a multiple of the usual median of that job&apos;s runs before it (the tick), from 0 to 2.5×; the shaded band is their usual range.
          Amber: longer than the usual maximum. Next to a run&apos;s start, the commit it ran at. A run opens its log, warning signatures first.
        </>
      }
      meta={[shown, d.runs.kept !== null ? `the newest ${d.runs.kept} are kept` : '', win.from ? span(win.from, win.to) : '', 'Brussels time']}
      head={
        <label className="pick">
          <span className="vh">Show the runs of</span>
          <select value={known ?? ''} onChange={(e) => setFilter(e.target.value || null)}>
            <option value="">All jobs ({d.runs.items.length})</option>
            {jobs.map((j) => (
              <option key={j.key} value={j.key}>
                {jobLabel(j.key, j.name)} ({counts.get(j.key)})
              </option>
            ))}
          </select>
        </label>
      }
      foot={
        <>
          <Legend
            items={[
              { mark: 'fill', label: 'this run' },
              { mark: 'fill warn', label: 'over the usual maximum' },
              { mark: 'band', label: 'usual range' },
              { mark: 'tick', label: 'usual median' },
            ]}
          />
          <span className="fnote">Scale 0 – 2.5 × the usual median</span>
        </>
      }
    >
      <div className="runs">
        <div className="run head">
          <span>Job · start</span>
          <span className="ax" aria-hidden="true">
            <span className="l" style={{ left: 0 }}>
              0
            </span>
            <span style={{ left: '40%' }}>usual</span>
            <span style={{ left: '80%' }}>2×</span>
          </span>
          <span>Time taken</span>
          <span>Result · warnings</span>
        </div>
        {items.map((x) => (
          <RunRow key={x.id} x={x} onOpen={onOpenRun} />
        ))}
        {!items.length ? <div className="empty">No run kept{known ? ' for this job' : ''}.</div> : null}
      </div>
    </Card>
  );
}

function RunRow({ x, onOpen }: { x: RunItem; onOpen: (id: string) => void }) {
  const label = jobLabel(x.key, x.name);
  const range = usualRange(x.usual, x.minutes);
  const w = x.warnings;
  return (
    <div className="run">
      <span className="job">
        <button type="button" className="gbtn row" onClick={() => onOpen(x.id)} aria-haspopup="dialog">
          {label}
        </button>
        <small>
          {stamp(x.start)}
          {x.sha ? (
            <>
              {' · at '}
              <code>{x.sha}</code>
            </>
          ) : null}
        </small>
      </span>
      {x.kind === 'run' ? (
        <DurationBar minutes={x.minutes} usual={x.usual} label={`${label}: ${durationText(x.minutes, x.usual)}${range ? `, usual ${range}` : ''}`} />
      ) : (
        <DurationBar minutes={null} usual={null} label={`${label}: ${x.kind === 'dry' ? 'recorded only' : 'waited on the lock'}`} />
      )}
      <span className={x.overUsual ? 'dur warn' : 'dur'}>
        {x.kind === 'run' ? (
          <>
            <b>{isNum(x.minutes) ? durationText(x.minutes, x.usual) : DASH}</b>
            <small>
              {range ? `usual ${range} · ${x.usual?.n ?? 0} earlier run${x.usual?.n === 1 ? '' : 's'}` : 'no earlier run'}
            </small>
          </>
        ) : (
          <span className="mut">{x.kind === 'dry' ? 'not started' : `waited${isNum(x.minutes) ? ` ${durationText(x.minutes)}+` : ''} on process.lock, stayed queued`}</span>
        )}
      </span>
      <span className="chips">
        <Chip kind={x.result === 'ok' ? 'flat' : x.kind === 'run' ? 'down' : 'dim'}>{x.kind === 'dry' ? 'dry run: recorded only' : x.result}</Chip>
        {w && w.signatures ? (
          <Chip kind={w.new ? 'warn' : 'flat'}>
            {w.signatures} warning signature{w.signatures === 1 ? '' : 's'}
            {w.new ? ` · ${w.new} new` : ''}
          </Chip>
        ) : null}
        {x.sliceGone ? <Chip kind="dim">log slice gone</Chip> : null}
        {x.note && x.result !== 'ok' ? <span className="was">{x.note}</span> : null}
      </span>
    </div>
  );
}
