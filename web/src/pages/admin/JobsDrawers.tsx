import { ApiError, getAdminRunLog, type JobLine, type JobsView, type RunLog } from '../../api';
import { DASH, cap, isNum, span, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import Drawer, { Section } from '../../components/admin/Drawer';
import Chip from '../../components/admin/Chip';
import { Legend } from '../../components/admin/Card';
import { DurationBar } from '../../components/admin/Bars';
import { afterText, basisText, fromRunId, runAsk, takesText, type Ask } from './jobs';
import { durationText, jobLabel, usualRange, usualText } from './week';

// The Jobs page's two drawers: a job (?job=) - what it does, Run now with its cost, when it runs, how long it usually
// takes, its last run and its command - and a run (?run=) - when, how long against its usual and at which commit,
// then its log slice: the warning signatures first, then its last lines.

export function JobDrawer({
  job,
  d,
  onClose,
  onAsk,
  onOpenRun,
  onRuns,
}: {
  job: JobLine;
  d: JobsView;
  onClose: () => void;
  onAsk: (a: Ask) => void;
  onOpenRun: (id: string) => void;
  onRuns: (key: string) => void;
}) {
  const label = jobLabel(job.key, job.name);
  const u = job.usual;
  const kept = d.runs.items.filter((x) => x.key === job.key).length;
  const after = afterText(job, d);
  const running = d.running?.key === job.key;
  return (
    <Drawer title={label} sub={`${job.name} · Brussels time`} onClose={onClose}>
      <Section title="What it does">
        <p className="line">{job.description || DASH}</p>
      </Section>
      <Section title="Run now" aside={running ? <Chip kind="flat">running now</Chip> : null}>
        <p className="line">
          {label} {takesText(u)}.
        </p>
        {job.canRun ? null : <p className="line">Not now: {job.refusal ?? 'refused by the schedule'}.</p>}
        <button type="button" className="btn primary" disabled={!job.canRun} onClick={() => onAsk(runAsk(job, d))}>
          Run now…
        </button>
      </Section>
      <Section title="When">
        <p className="line">
          {job.nextAt ? (
            <>
              Next run <b>{stamp(job.nextAt)}</b> · daily
            </>
          ) : after ? (
            <>
              {cap(after.when)} · {after.place}
            </>
          ) : (
            job.trigger ?? DASH
          )}
        </p>
      </Section>
      <Section title="How long">
        {u ? (
          <>
            <p className="line">
              Usual {usualRange(u)}, median <b>{durationText(u.median, u)}</b>: {basisText(u)}
              {u.from ? `, ${span(u.from, u.to)}` : ''}.
            </p>
          </>
        ) : (
          <p className="line">No earlier successful run to say how long it takes.</p>
        )}
      </Section>
      <Section title="Last run" aside={job.last ? <Chip kind={job.last.result === 'ok' ? 'flat' : 'down'}>{job.last.result}</Chip> : null}>
        {job.last ? (
          <p className="line">
            <button type="button" className="linkbtn" onClick={() => onOpenRun(job.last?.id ?? '')}>
              {stamp(job.last.start)} ›
            </button>{' '}
            · {durationText(job.last.minutes, u)}
          </p>
        ) : (
          <p className="line">None among the runs kept.</p>
        )}
        {kept ? (
          <button type="button" className="linkbtn more" onClick={() => onRuns(job.key)}>
            Its {kept} kept run{kept === 1 ? '' : 's'} ›
          </button>
        ) : null}
      </Section>
      <Section title="Command">
        <pre className="logtail cmd">{job.command || DASH}</pre>
        {job.log ? <p className="line">Writes its log to {job.log}.</p> : null}
      </Section>
    </Drawer>
  );
}

// A failed read's own words, without the request it was ("... (GET /api/...: 404)") or a closing full stop
const plain = (e: Error) => e.message.replace(/ \(GET [^)]*\)$/, '').replace(/\.$/, '');

export function RunLogDrawer({ id, d, onClose }: { id: string; d: JobsView; onClose: () => void }) {
  const item = d.runs.items.find((x) => x.id === id) ?? null;
  const live = d.running?.id === id ? d.running : null;
  // a dry entry (a plan's key) or a wait on the lock started nothing: there is no log slice to ask for
  const noLog = item !== null && item.kind !== 'run';
  // the running job's log grows: read it again with every refresh of the page
  const load = useLoad(() => (noLog ? Promise.resolve(null) : getAdminRunLog(id)), [id, noLog, live ? d.builtAt : '']);
  const log = load.kind === 'ok' ? load.data : null;
  const guess = fromRunId(id);
  const key = item?.key ?? live?.key ?? log?.key ?? guess.key;
  const start = item?.start ?? live?.start ?? log?.start ?? guess.start;
  const name = item?.name ?? live?.name ?? log?.job ?? undefined;
  const usual = item?.usual ?? live?.usual ?? null;
  const minutes = live ? live.minutes : item?.minutes ?? log?.minutes ?? null;
  const result = live ? 'running' : item?.result ?? log?.result ?? DASH;
  const end = item?.end ?? log?.end ?? null;
  const sha = item?.sha ?? live?.sha ?? log?.sha ?? null;
  const range = usualRange(usual, minutes);
  return (
    <Drawer title={key ? jobLabel(key, name) : 'Run'} sub={`${stamp(start)} · Brussels time`} onClose={onClose}>
      <Section title="Run" aside={<Chip kind={result === 'ok' || result === 'running' ? 'flat' : item && item.kind !== 'run' ? 'dim' : 'down'}>{result}</Chip>}>
        <p className="line">
          {span(start, end)} · <b>{live ? `${durationText(minutes, usual)} so far` : durationText(minutes, usual)}</b>
          {range ? ` · usual ${range}` : ''}
          {usual ? ` · ${basisText(usual)}` : ''}
          {isNum(live?.vsMedian ?? item?.vsMedian) ? ` · ${(live?.vsMedian ?? item?.vsMedian ?? 0).toFixed(2)}× the median` : ''}
          {sha ? (
            <>
              {' · at commit '}
              <code>{sha}</code>
            </>
          ) : null}
        </p>
        {item?.reason || live?.reason ? <p className="line">{cap(item?.reason ?? live?.reason ?? '')}.</p> : null}
        {item?.note || log?.note ? <p className="line">{item?.note ?? log?.note}</p> : null}
        {item?.kind === 'dry' ? <p className="line">A dry run: the schedule only recorded it, nothing was started.</p> : null}
        {item?.kind === 'blocked' ? (
          <p className="line">{`It waited${isNum(item.minutes) ? ` over ${durationText(item.minutes)}` : ''} on process.lock and stayed in the queue; its run is listed separately.`}</p>
        ) : null}
        {item?.kind !== 'dry' && item?.kind !== 'blocked' ? (
          <>
            <DurationBar minutes={minutes} usual={usual} label={`${durationText(minutes, usual)} against ${usualText(usual, minutes)}`} />
            <Legend
              items={[
                { mark: 'fill', label: 'this run' },
                { mark: 'band', label: 'usual range' },
                { mark: 'tick', label: 'usual median' },
              ]}
            />
          </>
        ) : null}
      </Section>
      {noLog ? null : load.kind === 'loading' ? (
        <Section title="Its log">
          <p className="line">Reading its log…</p>
        </Section>
      ) : load.kind === 'error' ? (
        <Section title="Its log" aside={<Chip kind="dim">not read</Chip>}>
          <p className="line">{cap(plain(load.error))}.</p>
          {!(load.error instanceof ApiError && load.error.status === 404) ? (
            <button type="button" className="linkbtn more" onClick={load.retry}>
              Try again
            </button>
          ) : null}
        </Section>
      ) : log ? (
        <LogSections log={log} />
      ) : null}
    </Drawer>
  );
}

function LogSections({ log }: { log: RunLog }) {
  const fresh = log.warnings.filter((w) => w.new === true).length;
  return (
    <>
      <Section
        title="Warnings"
        aside={log.warnings.length ? <Chip kind={fresh ? 'warn' : 'flat'}>{fresh ? `${fresh} new` : 'none new'}</Chip> : <Chip kind="flat">none</Chip>}
      >
        <p className="line">
          {log.comparedWith
            ? `Compared with the ${log.comparedWith.runs} earlier run${log.comparedWith.runs === 1 ? '' : 's'} of this job the log still holds, ${span(log.comparedWith.from, log.comparedWith.to)}: a signature is new when none of them had it.`
            : 'The log holds no earlier run of this job: no signature can be called new or known.'}
        </p>
        {log.warnings.length ? (
          <ul className="wlist">
            {log.warnings.map((w) => (
              <li key={w.sig}>
                <span className="wh">
                  <code>{w.sig}</code>
                  <span className="mut">×{w.n}</span>
                  {w.new === true ? <Chip kind="warn">new: no earlier run had it</Chip> : w.new === false ? <Chip kind="flat">known</Chip> : null}
                </span>
                {w.line ? <code className="wline">{w.line}</code> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="line">No warning signature in this run&apos;s log slice.</p>
        )}
        {isNum(log.noiseFiltered) && log.noiseFiltered ? (
          <p className="line">
            {log.noiseFiltered} other line{log.noiseFiltered === 1 ? '' : 's'} a plain grep for &quot;warn&quot; or &quot;error&quot; would catch, left out as known noise.
          </p>
        ) : null}
      </Section>
      <Section title="Its log" aside={log.running ? <Chip kind="flat">still running</Chip> : null}>
        <p className="line">
          The last {log.lines.shown} of {isNum(log.lines.of) ? log.lines.of : DASH} lines of its slice
          {log.log ? ` of ${log.log}` : ''}
          {log.lines.whole ? '' : `, read from the end (${log.lines.readKb} KB)`}
          {log.inKeptRuns ? '' : ' · older than the runs kept, found by its markers'}.
        </p>
        {log.error ? <p className="line">{log.error}</p> : null}
        <pre className="logtail tall">{log.tail.length ? log.tail.join('\n') : 'no lines'}</pre>
      </Section>
    </>
  );
}
