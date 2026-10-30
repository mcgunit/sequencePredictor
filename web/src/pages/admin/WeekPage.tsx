import { Fragment, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import type { MatrixGame, RunLine, WeekTuner, WeekView } from '../../api';
import { DASH, count, fixed, isNum, range, short, span, stamp, time, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam, useParamDrawer } from '../../hooks/useUrlState';
import { fetchWeek } from '../../hooks/useWeekStatus';
import Tile from '../../components/Tile';
import { Card, Legend } from '../../components/admin/Card';
import Chip, { type ChipKind } from '../../components/admin/Chip';
import { DurationBar, StackBar } from '../../components/admin/Bars';
import { historyNote, readingNote, ruleNote, streakWords } from '../../components/admin/Streak';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import ControlsDrawer from './ControlsDrawer';
import RunDrawer from './RunDrawer';
import { GameButton, GroupRow, earliest, joinParts, q0TileFoot, seedsText, units } from './bits';
import { marginText } from './gates';
import { durationText, jobLabel, runId, usualBasis, usualText, weekStatus, type WeekStatus } from './week';

// /admin, the admin landing page: the first screen of the Monday reading - did the weekend run, what did the tuners
// change, does anything clear a band, what is still open - and from Tuesday to Friday "did today's run go through".
// A run opens its drawer (?run=), a control cell its game's Controls drawer (?game=), a gates cell the Tuning gates
// drawer. Every time is Brussels time.
export default function WeekPage() {
  const load = useLoad(fetchWeek, []);
  const controls = useDrawerParam();
  const run = useParamDrawer('run', ['game']);

  useEffect(() => {
    document.title = 'This week · Predictor';
  }, []);

  const status = useMemo(() => (load.kind === 'ok' ? weekStatus(load.data) : null), [load]);

  if (load.kind !== 'ok' || !status) {
    return (
      <>
        <PageHead title="This week" />
        {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} /> : <Loading />}
      </>
    );
  }
  const d = load.data;
  return (
    <div className="admin">
      <PageHead
        title="This week"
        sub={joinParts([`Weekend of ${weekday(d.weekend.saturday)}`, d.since.from ? `since ${weekday(d.since.from)}` : ''])}
        meta={[`built ${stamp(d.builtAt)}`, 'Brussels time']}
      />
      <StatusLine status={status} onOpen={run.open} />
      <Tiles d={d} />
      <div className="grid">
        <RunsCard d={d} onOpen={run.open} />
        <div className="stack">
          <TuningCard d={d} />
          <SinceCard d={d} onOpen={run.open} />
        </div>
        <MatrixCard d={d} onOpen={controls.open} />
      </div>
      {run.value ? (
        <RunDrawer key={run.value} id={run.value} d={d} onClose={run.close} />
      ) : controls.game ? (
        <ControlsDrawer key={controls.game} game={controls.game} onClose={controls.close} />
      ) : null}
    </div>
  );
}

// --- what needs a look: the sidebar dot's reasons, in words ---------------------------------------------------
function StatusLine({ status, onOpen }: { status: WeekStatus; onOpen: (id: string) => void }) {
  if (!status.level) {
    return <p className="statusline mut">No run failed, no control record missing, no new warning signature.</p>;
  }
  return (
    <div className="statusline" role="status">
      <span className={`sdot ${status.level}`} aria-hidden="true" />
      <b>Needs a look</b>
      <span className="chips">
        {status.reasons.map((r, i) =>
          r.run ? (
            <button key={i} type="button" className="chipbtn" onClick={() => onOpen(r.run as string)} aria-haspopup="dialog">
              <Chip kind={r.level === 'red' ? 'down' : 'warn'}>{r.text}</Chip>
            </button>
          ) : (
            <Chip key={i} kind={r.level === 'red' ? 'down' : 'warn'}>
              {r.text}
            </Chip>
          ),
        )}
      </span>
    </div>
  );
}

// --- tiles ---------------------------------------------------------------------------------------------------
function lockText(lock: WeekView['since']['lock']): string {
  if (!lock) return 'lock not known here';
  if (!lock.held) return 'lock free';
  return `lock held by ${lock.script ?? 'a job'}${lock.pid !== null ? ` (pid ${lock.pid})` : ''}`;
}

// While NullControls rewrites the Q0 band (context since 27 Oct 2026) game by game, Sunday noon into Monday: how many
// bands are from this run - the row tests on the tile are the daily report's and do not move with it.
function q0BandRunNote(games: MatrixGame[], running: WeekView['tiles']['weekend']['running']): string {
  if (!running || running.key !== 'nullControls' || !running.start) return '';
  const cells = games.filter((g) => g.q0?.band);
  // a game's band is from a run only once both its control records are (NullControls writes the synthetic one first)
  const oldest = (g: MatrixGame) => earliest(g.q0?.band?.modes.map((m) => m.at) ?? []) ?? g.q0?.band?.at ?? null;
  const from = Date.parse(running.start) - 60 * 1000;
  const others = cells.filter((g) => {
    const at = oldest(g);
    return !at || Date.parse(at) < from;
  });
  return `band (context) being rewritten since ${stamp(running.start)}: ${cells.length - others.length} of ${cells.length} from this run`;
}

function Tiles({ d }: { d: WeekView }) {
  const we = d.tiles.weekend;
  const today = d.tiles.today;
  const tn = d.tiles.tuners;
  const t = d.tiles;
  const games = d.matrix.games;
  const closed = Boolean(d.weekend.until && d.builtAt && Date.parse(d.builtAt) >= Date.parse(d.weekend.until));
  const predictorRuns = d.runs.filter((r) => r.key === 'predictor').length;
  const unitOf = (game: string) => games.find((g) => g.game === game)?.unit ?? 'draws';
  const q2Above = (group: string) => games.filter((g) => g.group === group && g.q2?.status === 'above').length;
  const rowsGames = games.filter((g) => g.rows);
  const scored = rowsGames.reduce((s, g) => s + (g.rows?.scored ?? 0), 0);
  const reference = short(rowsGames[0]?.rows?.reference ?? 'GARCH Model');
  const byUnit = new Map<string, number[]>();
  t.rows.windows.forEach((w) => {
    if (w.window !== null) byUnit.set(w.unit, [...(byUnit.get(w.unit) ?? []), w.window]);
  });
  const windows = [...byUnit.entries()].map(([unit, xs]) => `${range({ min: Math.min(...xs), max: Math.max(...xs) })} ${units(unit)}`).join(' · ');
  const held = tn?.heldFromLastWeekend;
  return (
    <div className="kpis">
      <Tile
        href="#runs"
        label="Weekend runs"
        value={`${we.ok}/${we.runs}`}
        unit="ok"
        foot={[
          `${we.runs} runs: the predictor ${predictorRuns === 1 ? 'once' : predictorRuns === 2 ? 'twice' : `${predictorRuns} times`}, ${we.runs - predictorRuns} weekly jobs · ${we.expected} expected`,
          we.from ? `${span(we.from, we.to)}${isNum(we.hours) ? ` · ${we.hours.toFixed(1)} h` : ''}` : 'nothing has run this weekend yet',
          ...we.notOk.map((x) => ({ text: `${jobLabel(x.key)} ${x.result}${x.note ? `: ${x.note}` : ''}`, warn: true })),
          we.running ? `running: ${jobLabel(we.running.key)} since ${stamp(we.running.start)}` : '',
          we.missing.length
            ? { text: `${closed ? 'no run' : 'not run yet'}: ${we.missing.map((m) => `${jobLabel(m.key)}${m.day ? ` ${weekday(m.day)}` : ''}`).join(', ')}`, warn: closed }
            : '',
          we.notDue.length ? `not due this weekend: ${we.notDue.map((m) => `${jobLabel(m.key)}${m.next ? `, next ${weekday(m.next)}` : ''}`).join('; ')}` : '',
          we.slowest
            ? { text: `${jobLabel(we.slowest.key)} ${durationText(we.slowest.minutes, we.slowest.usual)} vs ${usualText(we.slowest.usual, we.slowest.minutes)} · ${usualBasis(we.slowest.usual)}`, warn: true }
            : '',
          ...we.newWarnings.map((x) => ({ text: `new warning: ${jobLabel(x.key)} ${x.sig} ×${x.n}`, warn: true })),
        ]}
      />
      <Tile
        href="#since"
        label="Today"
        value={today.run ? durationText(today.run.minutes, today.run.usual) : today.running ? 'running' : DASH}
        unit={today.run ? `predictor ${today.run.result}` : today.running ? `since ${time(today.running.start)}` : 'no predictor run yet'}
        foot={[
          today.run ? `${span(today.run.start, today.run.end)} · ${usualText(today.run.usual, today.run.minutes)} · ${usualBasis(today.run.usual)}` : `${weekday(today.day)}`,
          today.run && today.run.overUsual ? { text: 'longer than its usual maximum', warn: true } : '',
          ...(today.run?.warnings ?? []).filter((x) => x.new).map((x) => ({ text: `new warning: ${x.sig} ×${x.n}`, warn: true })),
          `queue ${d.since.queue.length} · ${lockText(d.since.lock)}`,
        ]}
      />
      {tn ? (
        <Tile
          to="/admin/tuning"
          label="Tuners"
          value={String(tn.replaced)}
          unit={`of ${tn.n} replaced`}
          foot={[
            `${tn.kept} kept · ${tn.reverted} reverted · ${tn.games} games · weekend of ${weekday(tn.weekend)}`,
            `${tn.marginZero} at margin 0 · ${tn.replacedUnder1Se} replaced on a gain under 1 SE over the incumbent (${tn.replacedUnder1SeVsBetterReference} over the better reference)`,
            held ? `of the ${held.replaced} replaced ${weekday(held.weekend)}: ${held.kept} kept, ${held.replacedAgain} replaced again, ${held.reverted} reverted${held.notDecided ? `, ${held.notDecided} not decided` : ''}` : '',
            tn.unattributed ? { text: `${tn.unattributed} unattributed`, warn: true } : '',
            tn.notThisWeekend ? `${tn.notThisWeekend} rows still from an earlier weekend, not counted` : '',
            `decided ${span(tn.from, tn.to)}`,
          ]}
        />
      ) : (
        <Tile to="/admin/tuning" label="Tuners" value={DASH} unit="gate records not read" foot={['the tuning gates could not be read']} />
      )}
      <Tile
        to="/admin/controls#q0"
        label="Q0 row tests"
        value={String(t.q0.confirmed)}
        unit="confirmed on new draws"
        foot={[...q0TileFoot(t.q0, unitOf), q0BandRunNote(games, we.running)]}
      />
      <Tile
        to="/admin/controls#q2"
        label="Q2 randomness"
        value={String(t.q2.above)}
        unit={`of ${t.q2.of} above`}
        foot={[
          joinParts([`lottery ${q2Above('Lottery')}`, seedsText(t.q2.seeds.Lottery), `markets ${q2Above('Markets')}`, seedsText(t.q2.seeds.Markets)]),
          t.q2.tooFewGames.length ? `${t.q2.tooFewGames.length} too few draws, under ${d.matrix.fewWindows ?? DASH} test windows (${t.q2.tooFewGames.join(', ')})` : '',
          t.q2.failedGames.length ? `${t.q2.failedGames.length} failed: a comparison scored under 2 seeds (${t.q2.failedGames.join(', ')})` : '',
          joinParts([t.q2.missing.length ? `no record: ${t.q2.missing.join(', ')}` : '', t.q2.unreadable.length ? `unreadable: ${t.q2.unreadable.join(', ')}` : '']),
          span(t.q2.from, t.q2.at),
        ]}
      />
      <Tile
        to="/admin/controls#rows"
        label="Market rows"
        value={String(t.rows.beatReference)}
        unit={`of ${scored} rows beat ${reference}`}
        foot={[
          joinParts([`${t.rows.games} games`, windows]),
          `${reference} beats uniform in ${t.rows.referenceAboveUniform} of ${t.rows.games}`,
          t.rows.frequencyGames ? `${reference} beats the frequencies in ${t.rows.referenceAboveFrequency} of ${t.rows.frequencyGames}` : '',
          t.rows.missing.length ? `no report: ${t.rows.missing.join(', ')}` : '',
          span(t.rows.from, t.rows.at),
        ]}
      />
    </div>
  );
}

// --- the weekend's runs -----------------------------------------------------------------------------------------
function RunsCard({ d, onOpen }: { d: WeekView; onOpen: (id: string) => void }) {
  const we = d.tiles.weekend;
  const closed = Boolean(d.weekend.until && d.builtAt && Date.parse(d.builtAt) >= Date.parse(d.weekend.until));
  const runningFor = we.running?.start && d.builtAt ? (Date.parse(d.builtAt) - Date.parse(we.running.start)) / 60000 : null;
  return (
    <Card
      id="runs"
      className="span-2"
      title="Weekend runs"
      flush
      help={
        <>
          The weekend: the predictor on Saturday and on Sunday, and every job of the two weekly plans (the tuning chain, the Sunday controls) started from
          Saturday 00:00 to Tuesday 09:00, where the controls&apos; catch-up ends. Every bar is on one scale: the run&apos;s time as a multiple of its usual
          median (the tick), from 0 to 2.5×. The shaded band is the usual range of the earlier successful runs: a weekly job&apos;s from the markers in its own
          log, the predictor&apos;s from the scheduler&apos;s history; the count is on the line. Amber: a run longer than its usual maximum, and a warning
          signature none of the job&apos;s earlier runs the logs still hold had. A signature is a Python warning or error class (with the file that raised it),
          LOCKBOX or Traceback; the other lines a plain grep for &quot;warn&quot; or &quot;error&quot; would catch (TF start-up chatter, the gate&apos;s lines on
          &quot;standard error&quot;) are filtered and counted in the run&apos;s drawer. A run not started yet counts as missing only once the window has closed. The Q0
          band (context) runs on the first Sunday of the month only (since 1 Nov 2026, NullControls first, then the three weekly controls); on the other
          weekends it is listed as not due, never missing.
        </>
      }
      meta={[`${we.runs} runs · ${we.expected} expected`, we.from ? span(we.from, we.to) : '', 'Brussels time']}
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
        {d.runs.map((r) => (
          <RunRow key={runId(r)} r={r} onOpen={onOpen} />
        ))}
        {we.running ? (
          <div className="run">
            <span className="job">
              {jobLabel(we.running.key)}
              <small>{stamp(we.running.start)}</small>
            </span>
            <DurationBar minutes={null} usual={null} label={`${jobLabel(we.running.key)} is running`} />
            <span className="dur">
              <b>running</b>
              <small>for {durationText(runningFor)} at this build</small>
            </span>
            <span className="chips">
              <Chip kind="flat">running</Chip>
            </span>
          </div>
        ) : null}
        {we.missing.map((m) => (
          <div className="run" key={`${m.key}${m.day ?? ''}`}>
            <span className="job mut">
              {jobLabel(m.key)}
              <small>{m.day ? weekday(m.day) : 'this weekend'}</small>
            </span>
            <DurationBar minutes={null} usual={null} label={`${jobLabel(m.key)}: no run`} />
            <span className="dur mut">{closed ? 'no run in the window' : 'not run yet'}</span>
            <span className="chips">
              <Chip kind={closed ? 'down' : 'dim'}>{closed ? 'missing' : 'not run yet'}</Chip>
            </span>
          </div>
        ))}
        {we.notDue.map((m) => (
          <div className="run" key={`notdue-${m.key}`}>
            <span className="job mut">
              {jobLabel(m.key)}
              <small>first Sunday of the month</small>
            </span>
            <DurationBar minutes={null} usual={null} label={`${jobLabel(m.key)}: not due this weekend`} />
            <span className="dur mut">{m.next ? `next ${weekday(m.next)}` : 'monthly'}</span>
            <span className="chips">
              <Chip kind="dim">not due</Chip>
            </span>
          </div>
        ))}
        {!d.runs.length && !we.running && !we.missing.length ? <div className="empty">No run this weekend.</div> : null}
      </div>
    </Card>
  );
}

const MAX_CHIPS = 2;
function RunRow({ r, onOpen }: { r: RunLine; onOpen: (id: string) => void }) {
  // a new signature first, so it is never the one folded into "+n more"
  const warnings = [...(r.warnings ?? [])].sort((a, b) => Number(b.new === true) - Number(a.new === true));
  const shown = warnings.slice(0, MAX_CHIPS);
  const label = jobLabel(r.key, r.job);
  return (
    <div className="run">
      <span className="job">
        <button type="button" className="gbtn row" onClick={() => onOpen(runId(r))} aria-haspopup="dialog">
          {label}
        </button>
        <small>{stamp(r.start)}</small>
      </span>
      <DurationBar
        minutes={r.minutes}
        usual={r.usual}
        label={`${label}: ${durationText(r.minutes, r.usual)}, ${usualText(r.usual, r.minutes)}${r.usual ? `, median ${durationText(r.usual.median, r.usual)}` : ''}`}
      />
      <span className={r.overUsual ? 'dur warn' : 'dur'}>
        <b>{durationText(r.minutes, r.usual)}</b>
        <small>
          {usualText(r.usual, r.minutes)}
          {r.usual ? ` · ${usualBasis(r.usual)}` : ''}
        </small>
      </span>
      <span className="chips">
        <Chip kind={r.result === 'ok' ? 'flat' : 'down'}>{r.result}</Chip>
        {r.warnings === null ? <Chip kind="dim">log slice gone</Chip> : null}
        {shown.map((x) => (
          <Chip key={x.sig} kind={x.new ? 'warn' : 'flat'}>
            {x.sig} ×{x.n}
            {x.new ? ' · new' : ''}
          </Chip>
        ))}
        {warnings.length > MAX_CHIPS ? (
          <Chip kind="flat">
            +{warnings.length - MAX_CHIPS} more{warnings.slice(MAX_CHIPS).some((x) => x.new) ? ', new among them' : ''}
          </Chip>
        ) : null}
      </span>
    </div>
  );
}

// --- the tuners -------------------------------------------------------------------------------------------------
function TuningCard({ d }: { d: WeekView }) {
  const tn = d.tiles.tuners;
  const top = Math.max(1, ...(tn?.byTuner ?? []).map((t) => t.n));
  const other = tn && tn.weekend && d.weekend.saturday && tn.weekend !== d.weekend.saturday;
  return (
    <Card
      title="Tuning"
      help={
        <>
          The champion/challenger gate&apos;s decisions of the newest weekend, per tuner. The bars share one scale, in gate decisions. Replaced: the challenger
          beat the better of the incumbent and the default by the margin. Kept: it did not, and the served set stays. Reverted: the untuned default is served
          again. Margin 0 means any gain replaces; 1 SE is one standard error of the challenger&apos;s score. Under 1 SE counts replacements whose gain over the
          incumbent was less than one standard error.
        </>
      }
      meta={tn ? [`weekend of ${weekday(tn.weekend)}${other ? ', not this one' : ''}`, `decided ${span(tn.from, tn.to)}`, 'Brussels time'] : []}
      foot={
        <>
          <Legend
            items={[
              { mark: 'seg-a', label: 'replaced' },
              { mark: 'seg-c', label: 'kept' },
              { mark: 'seg-b', label: 'reverted' },
            ]}
          />
          <Link className="linkbtn more" to="/admin/tuning">
            Tuning gates ›
          </Link>
        </>
      }
    >
      {!tn ? (
        <p className="note">The gate records could not be read.</p>
      ) : (
        <div className="tuners">
          {tn.byTuner.map((t) => (
            <TunerLine key={t.tuner} t={t} top={top} d={d} />
          ))}
        </div>
      )}
    </Card>
  );
}

// A tuner of the chain with no gate record of this weekend (mid-chain on Saturday): what its run is doing, never a
// zero bar under "margin mixed"
function tunerRunText(t: WeekTuner, d: WeekView): string {
  const we = d.tiles.weekend;
  if (t.job && we.running?.key === t.job) return `running since ${stamp(we.running.start)}`;
  const run = [...d.runs].reverse().find((r) => r.key === t.job);
  if (run) return `its run ${span(run.start, run.end)} ended ${run.result} without a gate record dated this weekend`;
  const closed = Boolean(d.weekend.until && d.builtAt && Date.parse(d.builtAt) >= Date.parse(d.weekend.until));
  return closed ? 'no run this weekend' : 'not run yet';
}

function TunerLine({ t, top, d }: { t: WeekTuner; top: number; d: WeekView }) {
  const margin = marginText(t);
  if (!t.n) {
    return (
      <div className="tn">
        <div className="tn-h">
          <b>{t.tuner}</b>
          <span className="n">no decision this weekend yet</span>
        </div>
        <div className="tn-f mut">{tunerRunText(t, d)} · its rows keep an earlier weekend&apos;s decision</div>
      </div>
    );
  }
  return (
    <div className="tn">
      <div className="tn-h">
        <b>{t.tuner}</b>
        {margin ? <Chip kind="flat">{margin}</Chip> : null}
        <span className="n">
          {t.n} gates{isNum(t.games) ? ` · ${t.games} games` : ''}
        </span>
      </div>
      <StackBar replaced={t.replaced} kept={t.kept} reverted={t.reverted} of={top} label={`${t.tuner}: ${t.replaced} replaced, ${t.kept} kept, ${t.reverted} reverted of ${t.n}`} />
      <div className="tn-f">
        <b>{t.replaced}</b> replaced · {t.kept} kept · {t.reverted} reverted
        {t.replacedUnder1Se ? (
          <>
            {' '}
            · <b>{t.replacedUnder1Se}</b> of them on a gain under 1 SE over the incumbent
          </>
        ) : null}
      </div>
    </div>
  );
}

// --- since Monday ------------------------------------------------------------------------------------------------
function SinceCard({ d, onOpen }: { d: WeekView; onOpen: (id: string) => void }) {
  const s = d.since;
  const c = s.council;
  const sign = s.signIns;
  const known = s.predictorRuns.flatMap((r) => r.warnings ?? []).filter((x) => !x.new).map((x) => x.sig);
  const newOnes = s.predictorRuns.flatMap((r) => (r.warnings ?? []).filter((x) => x.new));
  const truncated = sign && sign.oldestScanned && s.from && Date.parse(sign.oldestScanned) > Date.parse(s.from);
  return (
    <Card id="since" title="Since Monday" meta={[`from ${stamp(s.from)}`, `to ${stamp(d.builtAt)}`, 'Brussels time']}>
      <dl className="kv">
        <dt>Predictor</dt>
        <dd>
          {s.predictorRuns.length ? (
            s.predictorRuns.map((r) => (
              <span className="ln" key={runId(r)}>
                <button type="button" className="gbtn row" onClick={() => onOpen(runId(r))} aria-haspopup="dialog">
                  {weekday(r.start)}
                </button>
                <span>
                  {time(r.start)} → {time(r.end) || '…'} · <b>{durationText(r.minutes, r.usual)}</b>
                </span>
                <Chip kind={r.result === 'ok' ? 'flat' : 'down'}>{r.result}</Chip>
                {r.overUsual ? <Chip kind="warn">over its usual maximum</Chip> : null}
              </span>
            ))
          ) : (
            <span className="mut">no predictor run since Monday 00:00 yet</span>
          )}
          {s.predictorRuns.length ? (
            <span className="sub">
              {usualText(s.predictorRuns[s.predictorRuns.length - 1].usual, null)}, each against its own earlier runs (scheduler history) ·{' '}
              {newOnes.length
                ? `new warning signature: ${newOnes.map((x) => `${x.sig} ×${x.n}`).join(', ')}`
                : known.length
                  ? `only known warning signatures (${[...new Set(known)].join(', ')})`
                  : 'no warning signature'}
            </span>
          ) : null}
        </dd>
        <dt>Now</dt>
        <dd>
          {s.running ? (
            <>
              running: {jobLabel(s.running.key, s.running.name)} since {stamp(s.running.start)}
            </>
          ) : (
            'nothing running'
          )}
          <span className="sub">
            {s.queue.length ? `${s.queue.length} queued: ${s.queue.map((q) => `${jobLabel(q.key, q.name)}${q.manual ? ' (by hand)' : ''}`).join(', ')}` : 'queue empty'} · {lockText(s.lock)}
          </span>
        </dd>
        <dt>Council API</dt>
        <dd>
          {!c ? (
            <span className="mut">not known here</span>
          ) : c.error ? (
            <>
              <Chip kind="down">no answer</Chip> {c.error}
              <span className="sub">checked {stamp(c.checkedAt)}</span>
            </>
          ) : (
            <>
              {c.status || DASH}
              {c.healthy === false ? (
                <>
                  {' '}
                  <Chip kind="down">health check failed</Chip>
                </>
              ) : c.healthy ? ' · healthy' : ''}
              {c.since ? ` · since ${stamp(c.since)}` : ''}
              <span className="sub">
                {joinParts([
                  c.summary,
                  isNum(c.restarts) ? `${c.restarts} restart${c.restarts === 1 ? '' : 's'}` : '',
                  c.lastExit ? `last exit ${c.lastExit.code ?? c.lastExit.signal ?? DASH} ${stamp(c.lastExit.at)}` : '',
                  c.lastError ? `last error: ${c.lastError}` : '',
                  `checked ${stamp(c.checkedAt)}`,
                ])}
              </span>
            </>
          )}
        </dd>
        <dt>Sign-ins</dt>
        <dd>
          {!sign ? (
            <span className="mut">not known here</span>
          ) : (
            <>
              {sign.ok} ok · {sign.failed} failed · {sign.locked} locked
              <span className="sub">
                since Monday 00:00 · 7 days: {sign.last7Days.failed} failed · {sign.last7Days.locked} locked · counted in the newest {sign.scanned} audit events, back to{' '}
                {stamp(sign.oldestScanned)}
                {truncated ? ', so the counts may be short' : ''}
              </span>
            </>
          )}
        </dd>
      </dl>
    </Card>
  );
}

// --- the controls matrix ---------------------------------------------------------------------------------------
// notes: further lines under the chip, each a caveat on the figure it carries
type Cell = { text: string; kind: ChipKind; words: ReturnType<typeof streakWords>; notes: string[] };
const withHistory = (base: ChipKind, text: string, h: Parameters<typeof streakWords>[0], what = '', notes: string[] = []): Cell => {
  const words = streakWords(h, what);
  return { text, kind: words?.changed ? 'chg' : base, words, notes: notes.filter(Boolean) };
};

// Q0 (27 Oct 2026): the row tests of the report on disk - confirmed, holds, under 5% next to what chance puts there,
// followed. The chip is outlined only when a row is confirmed on new draws; holds and under 5% are this week's figures
// and colour nothing. Its weeks are the readings' (the rows confirmed and followed, each reading at its own Sunday's
// report), and the line under it says where the report on disk differs. A payout game is tested on hits, out of the tile.
function q0Cell(g: MatrixGame, rankingAt: string | null): Cell | null {
  const q = g.q0;
  const t = q?.test;
  if (!q || !t) return null;
  if (!t.family) return withHistory('dim', `too few live ${units(g.unit)}: ${t.liveMax} of ${t.minDraws ?? DASH}`, q.history);
  const prefix = q.ranking === 'profit' ? 'on hits: ' : '';
  const text = `${prefix}${t.confirmed} confirmed · ${t.holds} hold · ${t.under} < 5% (≈${fixed(t.expected, 1)}) · ${t.followed} followed`;
  return withHistory(t.confirmed ? 'hi' : 'flat', `${text}${q.inTile ? '' : ' · not in the tile'}`, q.history, '', [readingNote(q.history, rankingAt)]);
}
// Q2: the status first, in words (the outline is never the only sign), then the numbers named. The first week under a
// new verdict rule (1 Nov 2026: 20 seeds and the windows rule) every cell says the week before was read under the old
// one - and a market game's that its shuffled control's bins were re-cut on 26 Oct 2026 too - so a change that week
// may be the rule's rather than the draws'.
function q2Cell(g: MatrixGame): Cell | null {
  const q = g.q2;
  if (!q) return null;
  const seeds = `${q.seeds ?? DASH} seeds`;
  const rule = ruleNote(q.history);
  const notes = [rule && g.group === 'Markets' && q.history?.ruleChanged?.rule === 1 ? `${rule}, and the shuffled control's bins were re-cut on 26 Oct` : rule];
  if (q.status === 'too few draws') {
    return withHistory('dim', `too few draws · ${q.windowsTest ?? DASH} test windows (under ${q.minWindows ?? DASH}) · ${seeds}`, q.history, q.status, notes);
  }
  if (q.status === 'failed') return withHistory('flat', `failed: a comparison scored under 2 seeds · ${seeds}`, q.history, q.status, notes);
  return withHistory(q.status === 'above' ? 'hi' : 'flat', `${q.status} · AUC ${fixed(q.real)} vs threshold ${fixed(q.threshold)} · ${seeds}`, q.history, q.status, notes);
}
// the feature control's verdicts in full: a variant that "fits noise" fits the noise columns, a bad sign
function fcCell(g: MatrixGame): Cell | null {
  const f = g.fc;
  if (!f) return null;
  const text = joinParts([
    `${f.fits} fit noise`,
    f.inconclusive ? `${f.inconclusive} inconclusive` : '',
    f.ignored ? `${f.ignored} noise ignored` : '',
    f.failed ? `${f.failed} failed` : '',
  ]);
  return withHistory('flat', `${text} of ${f.variants} variants`, f.history);
}
function rowsCell(g: MatrixGame): Cell | null {
  const r = g.rows;
  if (!r) return null;
  const ref = short(r.reference);
  // vs the frequencies only from a report that has them (MarketRows.py, 26 Oct 2026)
  const frequencies = r.referenceVsFrequency ? ` · vs the frequencies: ${r.referenceVsFrequency}` : '';
  return withHistory('flat', `${r.beat} of ${r.scored} beat ${ref} · ${ref} vs uniform: ${r.referenceVsUniform ?? DASH}${frequencies} · ${count(r.window, g.unit)}`, r.history);
}

function CellButton({ game, cell, missing, onOpen }: { game: string; cell: Cell | null; missing: string; onOpen: (g: string) => void }) {
  return (
    <button type="button" className="cellbtn" onClick={() => onOpen(game)} aria-haspopup="dialog">
      {cell ? <Chip kind={cell.kind}>{cell.text}</Chip> : <Chip kind="dim">{missing}</Chip>}
      {cell?.words?.text ? <span className="was">{cell.words.text}</span> : null}
      {cell?.words?.behind ? <span className="was">{cell.words.behind}</span> : null}
      {cell?.notes.map((n) => (
        <span className="was" key={n}>
          {n}
        </span>
      ))}
    </button>
  );
}

function MatrixCard({ d, onOpen }: { d: WeekView; onOpen: (g: string) => void }) {
  const groups = (['Lottery', 'Markets'] as const).map((name) => ({ name, games: d.matrix.games.filter((g) => g.group === name) })).filter((g) => g.games.length);
  const histories = d.matrix.games.flatMap((g) => [g.q0?.history, g.q2?.history, g.fc?.history, g.rows?.history]);
  const tn = d.tiles.tuners;
  const missing = (g: MatrixGame, part: string) => (g.errors.includes(part) ? 'record unreadable' : 'no record');
  return (
    <Card
      id="matrix"
      className="span-3"
      title="Controls matrix"
      flush
      help={
        <>
          One chip per control and game; a control opens that game&apos;s Controls drawer, the gates cell its Tuning gates drawer. Outlined: Q0 with a row
          confirmed on new draws, Q2&apos;s real AUC above the threshold and the shuffled control&apos;s bound. Dashed: too few draws to judge (Q0: no row with 10
          live draws, days or weeks yet; Q2: under {d.matrix.fewWindows ?? DASH} test windows, fixed by the history before the run). Colour marks a
          change only: a control that reads differently from the week before, in the stored weekly readings, is coloured and says what it read then; a state
          that holds is grey, with its weeks in a row. Q0 (since 27 Oct 2026, the daily report&apos;s row test): each ranked row on its live draws against the
          fair draw (lottery) or its own tickets shuffled within weeks (markets) - the rows confirmed on new draws, those whose p holds after the correction
          for the rows compared, those under 5% with what chance puts there, and those followed; only a confirmed row outlines the chip. Its weeks count the
          rows confirmed and followed in the stored readings (a reading from before 1 Nov 2026 read the old band rule), and the line under it names the
          rows where today&apos;s report differs. jokerplus, pick3 and keno rank on profit and are tested on hits here, outside the tile. Q2: its status, the real AUC against
          the threshold and its seeds, or its test windows when there are too few; failed when a comparison scored under 2 seeds; the first week under a new
          verdict rule (1 Nov 2026), that the week before was read under the old one. Feature control: the variants that fit the noise columns (a
          bad sign), are inconclusive, or ignore the noise. Market rows: the rows that beat the reference, and the reference against uniform, over the scored
          window. Gates: replaced · kept · reverted on the newest weekend.
        </>
      }
      meta={[
        `controls of ${weekday(d.readings?.newestRecords ?? null)}`,
        `ranking ${stamp(d.rankingAt)}`,
        tn ? `gates: weekend of ${weekday(tn.weekend)}` : '',
        historyNote(d.readings, histories),
        'Brussels time',
      ]}
    >
      <table className="tbl mx">
        <thead>
          <tr>
            <th>Game</th>
            <th>Q0 · row tests</th>
            <th>Q2 · real vs threshold</th>
            <th>Feature control</th>
            <th>Market rows</th>
            <th>Gates R · K · V</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((grp) => (
            <Fragment key={grp.name}>
              <GroupRow cols={6}>
                {grp.name} · {grp.games.length} games
              </GroupRow>
              {grp.games.map((g) => (
                <tr key={g.game}>
                  <td className="sym g" data-l="Game">
                    <GameButton game={g.game} onOpen={onOpen} />
                  </td>
                  <td data-l="Q0 row tests">
                    <CellButton game={g.game} cell={q0Cell(g, d.rankingAt)} missing={`no row test${g.q0?.error ? `: ${g.q0.error}` : ''}`} onOpen={onOpen} />
                  </td>
                  <td data-l="Q2 AUC">
                    <CellButton game={g.game} cell={q2Cell(g)} missing={missing(g, 'q2')} onOpen={onOpen} />
                  </td>
                  <td data-l="FC">
                    <CellButton game={g.game} cell={fcCell(g)} missing={missing(g, 'fc')} onOpen={onOpen} />
                  </td>
                  <td data-l="Market rows">
                    {g.group === 'Markets' ? (
                      <CellButton game={g.game} cell={rowsCell(g)} missing={g.errors.includes('rows') ? 'report unreadable' : 'no report'} onOpen={onOpen} />
                    ) : (
                      <span className="fnt">{DASH}</span>
                    )}
                  </td>
                  <td data-l="Gates R·K·V">
                    {g.gates ? (
                      <Link className="cellbtn" to={`/admin/tuning?game=${encodeURIComponent(g.game)}`}>
                        <Chip kind="flat">
                          {g.gates.replaced} R · {g.gates.kept} K · {g.gates.reverted} V
                        </Chip>
                        {g.gates.notThisWeekend ? <span className="was">{g.gates.notThisWeekend} decided on an earlier weekend</span> : null}
                      </Link>
                    ) : (
                      <span className="fnt">{DASH}</span>
                    )}
                  </td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
