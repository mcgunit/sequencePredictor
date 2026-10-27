import { Link } from 'react-router-dom';
import type { RunLine, WeekView } from '../../api';
import { DASH, isNum, span, stamp, weekday } from '../../format';
import Drawer, { Section } from '../../components/admin/Drawer';
import Chip from '../../components/admin/Chip';
import { Legend } from '../../components/admin/Card';
import { DurationBar } from '../../components/admin/Bars';
import { PART_LABEL, controlOutputs, durationText, jobLabel, runId, usualBasis, usualText } from './week';

// One run of This week (?run=<job>@<start>): when and how long against its usual, its warning signatures - which
// are new, which its job's earlier runs had - and what it wrote, as far as the page can tell from the files on disk.
// The log text itself is not sent; the signatures and their counts are.
export default function RunDrawer({ id, d, onClose }: { id: string; d: WeekView; onClose: () => void }) {
  const r = [...d.runs, ...d.since.predictorRuns, ...(d.tiles.today.run ? [d.tiles.today.run] : [])].find((x) => runId(x) === id) ?? null;
  return (
    <Drawer title={r ? jobLabel(r.key, r.job) : 'Run'} sub={r ? `${r.job} · ${weekday(r.start)} · Brussels time` : 'Brussels time'} onClose={onClose}>
      {!r ? (
        <section className="dsec">
          <p className="line">
            This run is not on the page any more: it shows the runs of the weekend of {weekday(d.weekend.saturday)} and the predictor&apos;s since{' '}
            {weekday(d.since.from)}.
          </p>
        </section>
      ) : (
        <>
          <RunSection r={r} />
          <WarningsSection r={r} />
          <OutputsSection r={r} d={d} />
        </>
      )}
    </Drawer>
  );
}

function RunSection({ r }: { r: RunLine }) {
  const u = r.usual;
  return (
    <Section title="Run" aside={<Chip kind={r.result === 'ok' ? 'flat' : 'down'}>{r.result}</Chip>}>
      <p className="line">
        {span(r.start, r.end)} · <b>{durationText(r.minutes, u)}</b>
        {r.end ? '' : ' · still running at this build'}
        {r.note ? ` · ${r.note}` : ''}
      </p>
      <p className="line">
        {u ? (
          <>
            {usualText(u, r.minutes)}, median {durationText(u.median, u)}: {usualBasis(u)}, {span(u.from, u.to)}
            {isNum(r.vsMedian) ? (
              <>
                {' '}
                · this run <b>{r.vsMedian.toFixed(2)}×</b> the median
              </>
            ) : null}
            {r.overUsual ? ' · longer than the usual maximum' : ''}
          </>
        ) : (
          'No earlier successful run of this job to compare with.'
        )}
      </p>
      <DurationBar minutes={r.minutes} usual={u} label={`${durationText(r.minutes, u)} against ${usualText(u, r.minutes)}`} />
      <Legend
        items={[
          { mark: 'fill', label: 'this run' },
          { mark: 'band', label: 'usual range' },
          { mark: 'tick', label: 'usual median' },
        ]}
      />
    </Section>
  );
}

function WarningsSection({ r }: { r: RunLine }) {
  const ws = r.warnings;
  const fresh = (ws ?? []).filter((x) => x.new === true).length;
  return (
    <Section
      title="Warnings"
      aside={ws === null ? <Chip kind="dim">log slice gone</Chip> : fresh ? <Chip kind="warn">{fresh} new</Chip> : <Chip kind="flat">{ws.length ? 'none new' : 'none'}</Chip>}
    >
      {ws === null ? (
        <p className="line">The run&apos;s slice of its log is no longer in the log or its rotated copy, so its warnings cannot be read.</p>
      ) : (
        <>
          <p className="line">
            {r.comparedWith
              ? `Compared with the ${r.comparedWith.runs} earlier run${r.comparedWith.runs === 1 ? '' : 's'} of this job the logs still hold, ${span(r.comparedWith.from, r.comparedWith.to)}: a signature is new when none of them had it.`
              : 'The logs hold no earlier run of this job: no signature can be called new or known.'}
          </p>
          {ws.length ? (
            <table className="tbl">
              <thead>
                <tr>
                  <th>Signature</th>
                  <th className="num">Lines</th>
                  <th>Earlier runs</th>
                </tr>
              </thead>
              <tbody>
                {ws.map((x) => (
                  <tr key={x.sig}>
                    <td>
                      <code>{x.sig}</code>
                    </td>
                    <td className="num">{x.n}</td>
                    <td>{x.new === true ? <Chip kind="warn">new: none had it</Chip> : x.new === false ? <Chip kind="flat">known: seen before</Chip> : <span className="mut">nothing to compare</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="line">No warning signature in this run&apos;s log slice.</p>
          )}
          <p className="line" style={{ marginTop: 8 }}>
            {r.noiseFiltered ?? DASH} other line{r.noiseFiltered === 1 ? '' : 's'} a plain grep for &quot;warn&quot; or &quot;error&quot; would catch, filtered as known noise (TF/XLA
            start-up chatter, the gate&apos;s lines on &quot;standard error&quot;). A signature is a Python warning, error or exception class (with the file that
            raised it, when the line names one), LOCKBOX or Traceback.
          </p>
        </>
      )}
    </Section>
  );
}

const SLACK_MS = 60 * 1000;
function OutputsSection({ r, d }: { r: RunLine; d: WeekView }) {
  const controls = controlOutputs(r, d.matrix.games, d.builtAt);
  if (controls) {
    const mine = controls.lines.filter((l) => l.how === 'this run');
    // a record rewritten after this run (a later run, by hand) is accounted for: this run did write it - red is for a
    // record from before the run, none, or one that could not be read, as on the page's status line
    const later = controls.lines.filter((l) => l.how === 'after this run');
    const times = mine.map((l) => l.at).filter((x): x is string => Boolean(x)).sort();
    const label = PART_LABEL[controls.part];
    return (
      <Section
        title="Outputs"
        aside={
          <Chip kind={mine.length + later.length === controls.lines.length ? 'flat' : 'down'}>
            {`${mine.length} of ${controls.lines.length} from this run${later.length ? ` · ${later.length} rewritten later` : ''}`}
          </Chip>
        }
      >
        <p className="line">
          The {label} records on disk at this build, one per game
          {controls.part === 'q0'
            ? ' (the Q0 band, context only since 27 Oct 2026 - Q0 itself is the daily report\'s row test: this job writes a synthetic and a shuffled record a game; both are judged, and a line names the one not from this run)'
            : ''}: a
          record whose time lies inside this run was written by it.
          {times.length ? ` This run's: ${span(times[0], times[times.length - 1])}.` : ''}
        </p>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th>Record</th>
              <th>Written</th>
            </tr>
          </thead>
          <tbody>
            {controls.lines.map((l) => (
              <tr key={l.game}>
                <td className="sym">
                  <Link className="gbtn" to={`/admin/controls?game=${encodeURIComponent(l.game)}`}>
                    {l.game}
                  </Link>
                </td>
                <td>{l.at ? stamp(l.at) : DASH}</td>
                <td>
                  {l.how === 'this run' ? (
                    <span className="mut">this run</span>
                  ) : l.how === 'after this run' ? (
                    <Chip kind="flat">{l.mode ? `${l.mode}: ` : ''}rewritten after this run</Chip>
                  ) : (
                    <Chip kind="down">{l.how === 'before this run' ? `${l.mode ? `${l.mode}: ` : ''}older, record of ${weekday(l.at)}: not from this run` : l.how}</Chip>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    );
  }
  const tn = d.tiles.tuners;
  const tuner = tn?.byTuner.find((t) => t.job === r.key) ?? null;
  if (tuner && tn) {
    const thisWeekend = tn.weekend === d.weekend.saturday && d.runs.some((x) => runId(x) === runId(r));
    return (
      <Section title="Outputs">
        {thisWeekend ? (
          <p className="line">
            This weekend&apos;s gate decisions by the {tuner.tuner} tuner, in bestParams_&lt;game&gt;.json: <b>{tuner.n}</b> in {tuner.games ?? DASH} games ·{' '}
            {tuner.replaced} replaced · {tuner.kept} kept · {tuner.reverted} reverted.
          </p>
        ) : (
          <p className="line">
            The gate records on disk are of the weekend of {weekday(tn.weekend)}; this run&apos;s are not among them any more.
          </p>
        )}
        <Link className="linkbtn more" to="/admin/tuning?by=tuner">
          Tuning gates ›
        </Link>
      </Section>
    );
  }
  if (r.key === 'predictor') {
    const at = d.rankingAt ? Date.parse(d.rankingAt) : NaN;
    const start = r.start ? Date.parse(r.start) : NaN;
    const end = r.end ? Date.parse(r.end) : NaN;
    const ours = Number.isFinite(at) && Number.isFinite(start) && at >= start - SLACK_MS && (!Number.isFinite(end) || at <= end + SLACK_MS);
    return (
      <Section title="Outputs">
        <p className="line">
          The predictor writes each game&apos;s day file and the ranking (modelPerformance.json); this page reads the ranking only.{' '}
          {ours ? `The ranking on disk, ${stamp(d.rankingAt)}, was written by this run.` : `The ranking on disk is from ${stamp(d.rankingAt)}, not from this run.`}
        </p>
      </Section>
    );
  }
  return (
    <Section title="Outputs">
      <p className="line">This page does not read what this job writes{r.key === 'trainMetaLearner' ? ' (the meta-learner’s model files)' : ''}.</p>
    </Section>
  );
}
