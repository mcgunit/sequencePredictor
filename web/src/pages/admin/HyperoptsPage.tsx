import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getAdminHyperopt, getAdminHyperoptLog, getAdminHyperopts, type Hyperopt, type HyperoptApplySide, type HyperoptList, type ParamValue } from '../../api';
import { DASH, fixed, short, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { usePoll } from '../../hooks/usePoll';
import { useSend, type Sender } from '../../hooks/useSend';
import { useParamDrawer } from '../../hooks/useUrlState';
import { Card } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Confirm from '../../components/admin/Confirm';
import Drawer, { Section } from '../../components/admin/Drawer';
import Toast from '../../components/admin/Toast';
import { Failed, Loading, PageHead } from '../../components/admin/States';

// /admin/hyperopts (owner, 11 Nov 2026): the deep-learning hyperopts administrators started - HyperoptDeepLearning.py
// for one model on one game, on the GPU worker (hyperopts.js, worker/remote-hyperopt.sh: the deep-learning backtests'
// queue, one at a time, beside the schedule). Record only: a run writes nothing that is served. Its drawer (?ho=) shows
// the gate's verdict and the values it would write; Apply writes them into bestParams_<game>.json on this server.
const LIVE_MS = 10000;
const IDLE_MS = 30000;

const hours = (s: number | null) => (s === null ? DASH : s < 5400 ? `${Math.round(s / 60)} min` : s < 2 * 86400 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} days`);
const val = (v: ParamValue) => (v === null || v === undefined ? 'not set' : typeof v === 'number' && !Number.isInteger(v) ? String(Number(v.toPrecision(4))) : String(v));
const decisionText = (d: string | null) =>
  d === 'replaced' ? 'the gate would replace the served values' : d === 'reverted' ? 'the gate would serve the untuned defaults' : d === 'kept' ? 'the gate keeps what is served' : d ?? DASH;

export default function HyperoptsPage() {
  const [every, setEvery] = useState(LIVE_MS);
  const poll = usePoll(getAdminHyperopts, every);
  const drawer = useParamDrawer('ho');
  const sender = useSend(poll.refresh);
  const [starting, setStarting] = useState<Partial<Hyperopt> | null>(null);
  const [stop, setStop] = useState<Hyperopt | null>(null);

  useEffect(() => {
    document.title = 'Hyperopts · Predictor';
  }, []);
  const d = poll.data;
  const live = Boolean(d && d.hyperopts.some((h) => h.status === 'queued' || h.status === 'running'));
  useEffect(() => setEvery(live ? LIVE_MS : IDLE_MS), [live]);

  const head = (
    <PageHead
      title="Hyperopts"
      sub={d ? `${d.hyperopts.filter((h) => h.status === 'running').length} running · ${d.hyperopts.filter((h) => h.status === 'queued').length} queued · ${d.hyperopts.length} kept` : undefined}
      meta={['Brussels time']}
      aside={
        <button type="button" className="btn" onClick={() => { sender.reset(); setStarting({}); }} disabled={!d || !d.gpu.configured}>
          New hyperopt
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
        The deep-learning tuner for one model on one game, run on the GPU worker
        {d.gpu.configured ? ` (${d.gpu.host ?? 'set up'}${d.gpu.running ? ', busy' : ''}${d.gpu.queued ? `, ${d.gpu.queued} waiting` : ''})` : ' - none is set up on this server (config/workers.json), so none can start'}: in the same queue as
        the deep-learning <Link to="/admin/backtests">backtests</Link>, one at a time, beside the schedule. A run can take hours to days, and backtests wait behind it.
      </p>
      <p className="note">
        <b>Nothing a run finds is served until you apply it.</b> When a study ends, its drawer shows the gate&apos;s verdict (this run&apos;s best trial against the served
        values and the untuned defaults, re-scored on the same draws) and the values it would write. Apply writes them into bestParams for that game on this server. The
        next run serves them, and its commit includes the file.
      </p>
      {d.hyperopts.length ? (
        <HyperoptTable d={d} onOpen={drawer.open} onStop={(h) => { sender.reset(); setStop(h); }} onAgain={(h) => { sender.reset(); setStarting(h); }} />
      ) : (
        <p className="note">No hyperopt yet.</p>
      )}
      {drawer.value ? <HyperoptDrawer key={drawer.value} id={drawer.value} onClose={drawer.close} onChanged={() => void poll.refresh()} /> : null}
      {starting ? <HyperoptStart d={d} from={starting} sender={sender} onDone={() => setStarting(null)} /> : null}
      {stop ? (
        <Confirm
          title={stop.status === 'running' ? `Stop the hyperopt of ${short(stop.row)} on ${stop.game}` : `Cancel the hyperopt of ${short(stop.row)} on ${stop.game}`}
          action={stop.status === 'running' ? 'Stop' : 'Cancel hyperopt'}
          danger
          busy={sender.busy}
          error={sender.error}
          onConfirm={() => void sender.send(`/api/admin/hyperopts/${encodeURIComponent(stop.id)}/cancel`).then((done) => { if (done) setStop(null); })}
          onCancel={() => setStop(null)}
          lines={[
            stop.status === 'running'
              ? 'The trial that is running is lost. The finished trials stay in the study on the GPU worker, and Run again with the same settings (Fresh off) resumes the run.'
              : 'It is taken out of the queue.',
          ]}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

// What trains now, from the tuner's result file (written at the start of each trial and after each draw)
const ROLE: Record<string, string> = { incumbent: 'the served parameters', default: "Predictor.py's defaults" };
function nowText(h: Hyperopt): string | null {
  const c = h.study?.current ?? null;
  if (!c) return null;
  const draws = `${c.day} of ${c.days} draws`;
  if (c.role === 'trial') return `trial ${Math.min((h.study?.progress?.done ?? 0) + 1, h.trials)} of ${h.trials}${c.trial !== null ? ` (study trial #${c.trial})` : ''} · ${draws}`;
  return `the gate: scoring ${ROLE[c.role] ?? c.role} · ${draws}`;
}

function StatusCell({ h }: { h: Hyperopt }) {
  const p = h.study?.progress ?? null;
  if (h.status === 'running') {
    const done = p ? p.done : 0;
    const c = h.study?.current ?? null;
    const gate = h.study?.status === 'gate' || (c !== null && c.role !== 'trial');
    // trials: the finished ones plus the share of the running one's draws; the gate: its reference's draws
    const frac = c && c.days ? Math.min(1, c.day / c.days) : 0;
    const max = gate ? 1 : h.trials || 1;
    const value = gate ? frac : Math.min(max, done + (c && c.role === 'trial' ? frac : 0));
    const elapsed = h.startedAt ? ` · running ${hours((Date.now() - Date.parse(h.startedAt)) / 1000)}` : '';
    return (
      <div className="progress">
        <progress max={max} value={value} aria-label={nowText(h) ?? `${done} of ${h.trials} trials`} />
        <span className="sub">
          {h.phase ??
            (nowText(h) ? `${nowText(h)}${elapsed}` : gate ? `${done} of ${h.trials} trials · the gate is scoring the references` : `${done} of ${h.trials} trials${elapsed}`)}
        </span>
      </div>
    );
  }
  if (h.status === 'queued') return <span>{h.phase ?? `queued for the GPU worker${h.place ? `, number ${h.place}` : ''}`}</span>;
  return (
    <span>
      <Chip kind={h.status === 'done' ? 'flat' : h.status === 'failed' || h.status === 'ended' || h.status === 'lost' ? 'down' : 'dim'}>{h.status}</Chip>
      {h.error ? <span className="sub">{h.error}</span> : null}
    </span>
  );
}

function ResultCell({ h }: { h: Hyperopt }) {
  const s = h.study;
  if (h.applied) return <span>applied by {h.applied.by} · {stamp(h.applied.at)} <span className="sub">{h.applied.what === 'gate' ? "the gate's result" : "this run's best, against the gate"}, {h.applied.keys} values</span></span>;
  if (!s) return <span className="mut">{DASH}</span>;
  const best = s.progress?.best ?? null;
  if (s.status === 'done') {
    return (
      <span>
        <b>{s.decision ?? DASH}</b> <span className="mut">not applied</span>
        {best ? <span className="sub">best trial {best.trial}: {best.score === null ? 'no score' : fixed(best.score, 4)}</span> : null}
      </span>
    );
  }
  return best ? <span>best so far {best.score === null ? 'no score' : fixed(best.score, 4)} <span className="sub">trial {best.trial}</span></span> : <span className="mut">no scored trial yet</span>;
}

function HyperoptTable({ d, onOpen, onStop, onAgain }: { d: HyperoptList; onOpen: (id: string) => void; onStop: (h: Hyperopt) => void; onAgain: (h: Hyperopt) => void }) {
  return (
    <Card title="Hyperopts" meta={[`${d.hyperopts.length} kept`, 'newest first']} flush>
      <table className="tbl stk">
        <thead>
          <tr>
            <th>Model</th>
            <th>Game</th>
            <th>Settings</th>
            <th>Status</th>
            <th>Result</th>
            <th>Started by</th>
            <th aria-label="Action" />
          </tr>
        </thead>
        <tbody>
          {d.hyperopts.map((h) => (
            <tr key={h.id}>
              <td data-l="Model">
                <button type="button" className="linkbtn" onClick={() => onOpen(h.id)}>
                  {short(h.row)}
                </button>
              </td>
              <td data-l="Game">{h.game}</td>
              <td data-l="Settings" className="mut">
                {h.trials} trials × {h.days} draws · {h.trainSeconds} s
                <span className="sub">
                  gate {h.gate ? `${h.gateSeMargin} SE` : 'off'}
                  {h.fresh ? ' · fresh' : ''}
                </span>
              </td>
              <td data-l="Status">
                <StatusCell h={h} />
              </td>
              <td data-l="Result">
                <ResultCell h={h} />
              </td>
              <td data-l="Started by" className="mut">
                {h.by} · {stamp(h.requestedAt)}
              </td>
              <td>
                {h.status === 'running' || h.status === 'queued' ? (
                  <button type="button" className="linkbtn" onClick={() => onStop(h)}>
                    {h.status === 'running' ? 'Stop' : 'Cancel'}
                  </button>
                ) : (
                  <button type="button" className="linkbtn" onClick={() => onAgain(h)} disabled={!d.gpu.configured}>
                    Run again
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

// The form: the tuner's own arguments, with its defaults. `from` prefills it (Run again: the same settings, Fresh off,
// so a stopped run resumes).
function HyperoptStart({ d, from, sender, onDone }: { d: HyperoptList; from: Partial<Hyperopt>; sender: Sender; onDone: () => void }) {
  const models = Object.keys(d.models);
  const f = d.fields;
  const init = (k: keyof Hyperopt & string) => String(from[k] ?? f[k]?.default ?? '');
  const [model, setModel] = useState(from.model ?? 'transformer_model');
  const [game, setGame] = useState(from.game ?? 'lotto');
  const [v, setV] = useState<Record<string, string>>({
    trials: init('trials'), days: init('days'), trainSeconds: init('trainSeconds'), pruneAfterDays: init('pruneAfterDays'),
    trialTimeout: init('trialTimeout'), gateSeMargin: init('gateSeMargin'),
  });
  const [gate, setGate] = useState(from.gate ?? true);
  const [fresh, setFresh] = useState(false);
  const set = (k: string) => (e: { target: { value: string } }) => setV((x) => ({ ...x, [k]: e.target.value }));
  const bad = Object.entries(v)
    .map(([k, s]) => {
      const fl = f[k];
      const n = Number(s);
      if (!fl) return null;
      return s === '' || !Number.isFinite(n) || (fl.integer && !Number.isInteger(n)) || n < fl.min || n > fl.max ? `${LABEL[k]}: ${fl.integer ? 'a whole number' : 'a number'} from ${fl.min} to ${fl.max}` : null;
    })
    .find(Boolean) ?? (Number(v.pruneAfterDays) > Number(v.days) ? 'Prune after: at most the draws per trial (0 = no pruning)' : null);
  const budget = Math.min(d.budgetMaxSeconds, (Number(v.trials) + (gate ? 2 : 0)) * Number(v.trialTimeout) + 3600);
  const start = async () => {
    if (bad) return;
    const body: Record<string, unknown> = { model, game, gate, fresh };
    Object.entries(v).forEach(([k, s]) => { body[k] = Number(s); });
    const done = await sender.send<{ ok: true; message: string; id?: string }>('/api/admin/hyperopts/start', body);
    if (done) onDone();
  };
  const field = (k: string, help: string) => (
    <label className="field" key={k}>
      <span>{LABEL[k]}</span>
      <input value={v[k]} onChange={set(k)} inputMode="decimal" autoComplete="off" />
      <small>{help}</small>
    </label>
  );
  return (
    <Confirm
      title={from.id ? `Run again: ${short(from.row ?? '')} on ${from.game}` : 'Start a deep-learning hyperopt'}
      action="Start hyperopt"
      busy={sender.busy}
      error={sender.error ?? bad}
      disabled={Boolean(bad)}
      onConfirm={() => void start()}
      onCancel={onDone}
      lines={[
        'HyperoptDeepLearning.py on the GPU worker: each trial retrains the model on the newest draws with its suggested values and scores how much better than the counts it predicts (0 = no better). Nothing is served until you apply the result.',
        `At most ${hours(budget)} (every trial at its time limit${gate ? ', plus the gate’s two reference trials' : ''}). The deep-learning backtests wait behind it.`,
      ]}
    >
      <label className="field">
        <span>Model</span>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {short(d.models[m])}
            </option>
          ))}
        </select>
        <small>The LSTM, TCN and unified models are the slow ones: hours a study. The transformer, GNN and autoencoder are an order of magnitude faster.</small>
      </label>
      <label className="field">
        <span>Game</span>
        <select value={game} onChange={(e) => setGame(e.target.value)}>
          {d.games.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </label>
      {field('trials', 'Trials in this run. The study on the GPU worker keeps every earlier trial, which guides the sampler.')}
      {field('days', 'Draws retrained per trial (3 is the gate’s window; the warm-started draws barely differ, so more mostly cost time).')}
      {field('trainSeconds', 'Training budget of each retrain, in seconds. 240 is what the daily run gives each model.')}
      {field('pruneAfterDays', 'Stop a trial whose score after this many draws is below the median of the earlier trials (the first 3 never are). 0 = no pruning.')}
      {field('trialTimeout', 'Wall-clock limit of one trial, in seconds: a trial over it is recorded as pruned.')}
      <label className="field">
        <span>Gate</span>
        <select value={gate ? 'on' : 'off'} onChange={(e) => setGate(e.target.value === 'on')}>
          <option value="on">On: score the served values and the defaults on the same draws (2 extra trials)</option>
          <option value="off">Off: report this run’s best trial only</option>
        </select>
      </label>
      {gate ? field('gateSeMargin', 'How many standard errors of its own score the best trial must win by.') : null}
      <label className="field">
        <span>Resume</span>
        <select value={fresh ? 'fresh' : 'resume'} onChange={(e) => setFresh(e.target.value === 'fresh')}>
          <option value="resume">Resume: an unfinished run with these settings on the GPU worker carries on</option>
          <option value="fresh">Fresh: start a new run even if one with these settings is unfinished</option>
        </select>
        <small>Only the latest run can be resumed: a hyperopt with other settings starts a new one.</small>
      </label>
    </Confirm>
  );
}
const LABEL: Record<string, string> = {
  trials: 'Trials', days: 'Draws per trial', trainSeconds: 'Training seconds', pruneAfterDays: 'Prune after', trialTimeout: 'Trial time limit', gateSeMargin: 'Gate margin (SE)',
};

function HyperoptDrawer({ id, onClose, onChanged }: { id: string; onClose: () => void; onChanged: () => void }) {
  const load = useLoad(() => getAdminHyperopt(id), [id]);
  const h = load.kind === 'ok' ? load.data : null;
  return (
    <Drawer title={h ? `${short(h.row)} · ${h.game}` : 'Hyperopt'} sub={h ? `${h.trials} trials × ${h.days} draws · ${h.status}` : undefined} onClose={onClose}>
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {h ? <DrawerBody h={h} onChanged={() => { load.retry(); onChanged(); }} /> : null}
    </Drawer>
  );
}

function DrawerBody({ h, onChanged }: { h: Hyperopt; onChanged: () => void }) {
  const s = h.study;
  const r = s?.record ?? null;
  const n = (k: string) => (r && typeof r[k] === 'number' ? (r[k] as number) : null);
  const states = s?.progress?.states ?? {};
  return (
    <>
      <Section title="Run">
        <p className="line">
          Started by {h.by} {stamp(h.requestedAt)}
          {h.startedAt ? ` · ran from ${stamp(h.startedAt)}` : ''}
          {h.finishedAt ? ` to ${stamp(h.finishedAt)}` : ''}
          {h.error ? <span className="mut"> · {h.error}</span> : null}
        </p>
        <p className="line mut">
          {h.trials} trials × {h.days} draws, {h.trainSeconds} s of training a retrain, pruning {h.pruneAfterDays ? `after ${h.pruneAfterDays} draws` : 'off'}, {Math.round(h.trialTimeout / 60)} min a
          trial at most, gate {h.gate ? `on (${h.gateSeMargin} SE)` : 'off'}
          {h.fresh ? ', fresh' : ''} · budget {hours(h.budget)}
        </p>
        {s?.progress ? (
          <p className="line">
            {s.progress.done} of {h.trials} trials done
            {Object.keys(states).length ? ` (${Object.entries(states).map(([k, x]) => `${x} ${k.toLowerCase()}`).join(', ')})` : ''}
            {s.progress.best ? ` · best so far trial ${s.progress.best.trial} at ${s.progress.best.score === null ? 'no score' : fixed(s.progress.best.score, 4)}` : ''}
          </p>
        ) : null}
        {h.status === 'running' && nowText(h) ? (
          <p className="line">
            Now: {nowText(h)}
            {s?.current?.at ? <span className="mut"> · updated {stamp(s.current.at)}</span> : null}
          </p>
        ) : null}
      </Section>
      {s && s.status === 'done' ? (
        <Section title="The gate">
          <p className="line">
            <b>{decisionText(s.decision)}</b>
            {s.reason ? <span className="mut"> · {s.reason}</span> : null}
          </p>
          {r ? (
            <p className="line mut">
              Best trial of this run {n('challenger') === null ? DASH : fixed(n('challenger'), 4)} · served values {n('incumbent') === null ? 'not scored' : fixed(n('incumbent'), 4)} · untuned
              defaults {n('default') === null ? 'not scored' : fixed(n('default'), 4)} (minus the mean excess loss over the counts on the same {n('window_days') ?? h.days} draws; 0 = as good as the counts,
              above 0 = better). The standard error is over retrains on almost the same history: a floor, not a test. The validation split contains the lockbox period.
            </p>
          ) : null}
        </Section>
      ) : null}
      {h.applied ? (
        <Section title="Applied">
          <p className="line">
            {h.applied.keys} values written to bestParams_{h.game}.json by {h.applied.by} {stamp(h.applied.at)} ({h.applied.what === 'gate' ? "the gate's result" : "this run's best trial, against the gate"}).
          </p>
        </Section>
      ) : h.apply ? (
        <ApplySection h={h} onChanged={onChanged} />
      ) : null}
      <LogSection id={h.id} live={h.status === 'running'} />
    </>
  );
}

function Changes({ side }: { side: HyperoptApplySide }) {
  if (!side) return null;
  if (!side.changes.length) return <p className="line mut">Every value is already served: writing changes nothing but the gate&apos;s record.</p>;
  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>Value</th>
          <th>Served now</th>
          <th>Would be</th>
        </tr>
      </thead>
      <tbody>
        {side.changes.map((c) => (
          <tr key={c.key}>
            <td>
              <code>{c.key}</code>
              {side.changedSince.includes(c.key) ? <span className="sub">changed since the run started</span> : null}
            </td>
            <td className="mut">{val(c.served)}</td>
            <td>{val(c.value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ApplySection({ h, onChanged }: { h: Hyperopt; onChanged: () => void }) {
  const sender = useSend(onChanged);
  const [ask, setAsk] = useState<'gate' | 'challenger' | null>(null);
  const a = h.apply;
  if (!a) return null;
  const side = ask === 'gate' ? a.gate : ask === 'challenger' ? a.challenger : null;
  const since = side ? side.changedSince : [];
  return (
    <Section title="Apply">
      {a.gate ? (
        <>
          <p className="line">If you apply the gate&apos;s result ({h.study?.decision === 'reverted' ? 'the untuned defaults' : "this run's best trial"}):</p>
          <Changes side={a.gate} />
          <p className="line">
            <button type="button" className="btn" onClick={() => { sender.reset(); setAsk('gate'); }}>
              Apply the gate&apos;s result
            </button>
          </p>
        </>
      ) : (
        <p className="line mut">The gate keeps what is served: it has nothing to write.</p>
      )}
      {a.challenger ? (
        <>
          <p className="line">If you apply this run&apos;s best trial anyway, against the gate:</p>
          <Changes side={a.challenger} />
          <p className="line">
            <button type="button" className="btn" onClick={() => { sender.reset(); setAsk('challenger'); }}>
              Apply this run&apos;s best
            </button>
          </p>
        </>
      ) : null}
      {ask && side ? (
        <Confirm
          title={ask === 'gate' ? `Apply the gate's result to ${short(h.row)} on ${h.game}` : `Apply this run's best to ${short(h.row)} on ${h.game}`}
          action="Apply"
          danger={ask === 'challenger' || since.length > 0}
          busy={sender.busy}
          error={sender.error}
          onConfirm={() =>
            void sender.send(`/api/admin/hyperopts/${encodeURIComponent(h.id)}/apply`, { what: ask, anyway: since.length > 0 }).then((done) => {
              if (done) setAsk(null);
            })
          }
          onCancel={() => setAsk(null)}
          lines={[
            `${side.changes.length} values change in bestParams_${h.game}.json on this server, with the gate's record${ask === 'challenger' ? ' marked "applied by hand against the gate"' : ''}. The next run serves them, and its commit includes the file.`,
            since.length ? `${since.length} of them were changed on this server since the run started (${since.slice(0, 4).join(', ')}${since.length > 4 ? ', ...' : ''}): the gate compared against the old values.` : '',
            'Refused while a job is running: it may be writing the same file.',
          ]}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </Section>
  );
}

function LogSection({ id, live }: { id: string; live: boolean }) {
  const poll = usePoll(() => getAdminHyperoptLog(id), live ? LIVE_MS : 3600 * 1000);
  const l = poll.data;
  return (
    <Section title="Log">
      {!l ? <Loading /> : l.text ? <pre className="logtail">{l.text.split('\n').slice(-200).join('\n')}</pre> : <p className="line mut">No output yet.</p>}
      {l && l.truncated ? <p className="line mut">The last part only: log/hyperopts/{id}.log on this server has it all.</p> : null}
    </Section>
  );
}
