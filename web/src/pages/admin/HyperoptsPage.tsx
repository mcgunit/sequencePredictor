import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getAdminHyperopt, getAdminHyperoptLog, getAdminHyperopts, type Hyperopt, type HyperoptApplySide, type HyperoptList, type HyperoptStudy, type ParamValue } from '../../api';
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

// /admin/hyperopts (owner, 11-12 Nov 2026): the hyperopts administrators started - one tuner on one game (hyperopts.js,
// worker/remote-hyperopt.sh): deep learning on the GPU worker, the other five tuners on the DL360 (or this server when
// it does not come up), each in its worker's backtest queue, one at a time, beside the schedule. Record only: a run
// writes nothing that is served. Its drawer (?ho=) shows each study's gate verdict and the values it would write; Apply
// writes them into bestParams_<game>.json on this server. The variety line card sets the owner's line.
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
        <button type="button" className="btn" onClick={() => { sender.reset(); setStarting({}); }} disabled={!d}>
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
  const laneText = (l: HyperoptList['gpu'], none: string) =>
    l.configured ? ` (${l.host ?? 'set up'}${l.running ? (l.local ? ', running on this server' : ', busy') : ''}${l.queued ? `, ${l.queued} waiting` : ''})` : none;
  return (
    <div className="admin">
      {head}
      <p className="note">
        One tuner on one game, run beside the schedule. Deep learning runs on the GPU worker{laneText(d.gpu, ' - none is set up on this server, so it cannot start')}, in the same queue as the
        deep-learning <Link to="/admin/backtests">backtests</Link>. Statistics, boosting, ensemble, quantum and RL run on the DL360{laneText(d.cpu, ' - none is set up, so they run on this server')},
        in the same queue as its backtests. The DL360 is switched on for them; when it does not come up, the run moves to this server. One run at a time per worker; a run can take hours.
      </p>
      <p className="note">
        <b>Nothing a run finds is served until you apply it.</b> When a study ends, its drawer shows the gate&apos;s verdict (this run&apos;s best trial against the served
        values and the untuned defaults, re-scored on the same draws) and the values it would write. Apply writes them into bestParams for that game on this server. The
        next run serves them, and its commit includes the file.
      </p>
      <VarietySetting d={d} onChanged={() => void poll.refresh()} />
      {d.hyperopts.length ? (
        <HyperoptTable d={d} onOpen={drawer.open} onStop={(h) => { sender.reset(); setStop(h); }} onAgain={(h) => { sender.reset(); setStarting(h); }} />
      ) : (
        <p className="note">No hyperopt yet.</p>
      )}
      {drawer.value ? <HyperoptDrawer key={drawer.value} id={drawer.value} onClose={drawer.close} onChanged={() => void poll.refresh()} /> : null}
      {starting ? <HyperoptStart d={d} from={starting} sender={sender} onDone={() => setStarting(null)} /> : null}
      {stop ? (
        <Confirm
          title={stop.status === 'running' ? `Stop the hyperopt of ${rowText(stop)} on ${stop.game}` : `Cancel the hyperopt of ${rowText(stop)} on ${stop.game}`}
          action={stop.status === 'running' ? 'Stop' : 'Cancel hyperopt'}
          danger
          busy={sender.busy}
          error={sender.error}
          onConfirm={() => void sender.send(`/api/admin/hyperopts/${encodeURIComponent(stop.id)}/cancel`).then((done) => { if (done) setStop(null); })}
          onCancel={() => setStop(null)}
          lines={[
            stop.status === 'running'
              ? 'The trial that is running is lost. The finished trials stay in the study on the worker, and a run with the same settings carries on from them.'
              : 'It is taken out of the queue.',
          ]}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

const rowText = (h: Hyperopt) => (h.tuner === 'deeplearning' ? short(h.row) : `${h.tunerLabel}${h.model ? `: ${short(h.row)}` : ''}`);
const workerText = (h: Hyperopt) => h.where ?? (h.worker === 'cpu' ? 'the DL360' : 'the GPU worker');

// The variety line (owner, 12 Nov 2026): the setting the weekly tuning uses, and what a new hyperopt starts with
function VarietySetting({ d, onChanged }: { d: HyperoptList; onChanged: () => void }) {
  const sender = useSend(onChanged);
  const [v, setV] = useState(String(d.variety.varietyLine));
  const [ask, setAsk] = useState(false);
  const n = Number(v);
  const bad = v === '' || !Number.isFinite(n) || n < d.variety.min || n > d.variety.max ? `A percentage from ${d.variety.min} to ${d.variety.max}` : null;
  return (
    <Card title="Variety rule" meta={[`now ${d.variety.varietyLine}%`]}>
      <p className="line">
        A usable parameter set predicts new tickets; recurring numbers are fine. A candidate that gives the same ticket on this share or more is <b>input-blind</b>: the gate
        neither writes it nor serves it back, and it blocks nothing. Deep learning counts its validation windows (blind on at least half of its draws); the other tuners count
        their scored backtest draws. The weekly tuning uses this line from its next run; a new hyperopt starts with it and can set its own.
      </p>
      <div className="vline">
        <label className="field">
          <span>Line (%)</span>
          <input value={v} onChange={(e) => setV(e.target.value)} inputMode="decimal" autoComplete="off" />
        </label>
        <button type="button" className="btn" disabled={Boolean(bad) || n === d.variety.varietyLine} onClick={() => { sender.reset(); setAsk(true); }}>
          Save
        </button>
        {bad ? <span className="mut">{bad}</span> : null}
      </div>
      {ask ? (
        <Confirm
          title={`Set the variety line to ${n}%`}
          action="Save"
          busy={sender.busy}
          error={sender.error}
          onConfirm={() => void sender.send('/api/admin/hyperopts/settings', { varietyLine: n }).then((done) => { if (done) setAsk(false); })}
          onCancel={() => setAsk(false)}
          lines={[
            `From the next weekly tuning on, every tuner refuses a candidate that gives the same ticket on ${n}% or more${n < d.variety.varietyLine ? ' - a stricter line than now: more candidates are refused' : ''}.`,
            'What is served now does not change until a tuner runs.',
          ]}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </Card>
  );
}

// What is scored now, from the tuner's result file (written per trial, and per draw where the tuner backtests draw by draw)
const ROLE: Record<string, string> = {
  incumbent: 'the served parameters', default: 'the untuned defaults', 'day table': 'building the keno day table', table: "backtesting the base models' score table",
};
function nowText(h: Hyperopt): string | null {
  const c = h.study?.current ?? null;
  if (!c) return null;
  const ev = (c as { evaluated?: unknown; of?: unknown }).evaluated;
  const of = (c as { evaluated?: unknown; of?: unknown }).of;
  const draws = c.day !== null && c.days ? ` · ${c.day} of ${c.days} draws` : typeof ev === 'number' ? ` · ${ev}${typeof of === 'number' ? ` of ${of}` : ''} combinations evaluated` : '';
  const which = h.studies.length > 1 && h.study?.row ? `${short(h.study.row)}: ` : '';
  if (c.role === 'trial') return `${which}trial ${Math.min((h.study?.progress?.done ?? 0) + 1, h.trials)} of ${h.trials}${c.trial !== null ? ` (study trial #${c.trial})` : ''}${draws}`;
  if (c.role === 'day table' || c.role === 'table') return `${which}${ROLE[c.role]}${draws}`;
  return `${which}the gate: scoring ${ROLE[c.role] ?? c.role}${draws}`;
}

function StatusCell({ h }: { h: Hyperopt }) {
  const p = h.study?.progress ?? null;
  if (h.status === 'running') {
    const done = p ? p.done : 0;
    const c = h.study?.current ?? null;
    const gate = h.study?.status === 'gate' || (c !== null && c.role !== 'trial' && c.role !== 'day table' && c.role !== 'table');
    // trials: the finished ones plus the share of the running one's draws; the gate: its reference's draws
    const frac = c && c.days && c.day !== null ? Math.min(1, c.day / c.days) : 0;
    const max = gate ? 1 : h.trials || 1;
    const value = gate ? frac : Math.min(max, done + (c && c.role === 'trial' ? frac : 0));
    const elapsed = h.startedAt ? ` · running ${hours((Date.now() - Date.parse(h.startedAt)) / 1000)}` : '';
    const where = h.where === 'this server' ? ' · on this server' : '';
    return (
      <div className="progress">
        <progress max={max} value={value} aria-label={nowText(h) ?? `${done} of ${h.trials} trials`} />
        <span className="sub">
          {h.phase ??
            (nowText(h) ? `${nowText(h)}${elapsed}${where}` : gate ? `${done} of ${h.trials} trials · the gate is scoring the references` : `${done} of ${h.trials} trials${elapsed}${where}`)}
        </span>
      </div>
    );
  }
  if (h.status === 'queued') return <span>{h.phase ?? `queued for ${workerText(h)}${h.place ? `, number ${h.place}` : ''}`}</span>;
  return (
    <span>
      <Chip kind={h.status === 'done' ? 'flat' : h.status === 'failed' || h.status === 'ended' || h.status === 'lost' ? 'down' : 'dim'}>{h.status}</Chip>
      {h.error ? <span className="sub">{h.error}</span> : null}
    </span>
  );
}

function ResultCell({ h }: { h: Hyperopt }) {
  const s = h.study;
  const nApplied = Object.keys(h.appliedStudies).length;
  if (h.applied && h.studies.length <= 1) return <span>applied by {h.applied.by} · {stamp(h.applied.at)} <span className="sub">{h.applied.what === 'gate' ? "the gate's result" : "this run's best, against the gate"}, {h.applied.keys} values</span></span>;
  if (!s) return <span className="mut">{DASH}</span>;
  if (h.studies.length > 1 && h.studies.every((x) => x.status === 'done')) {
    return (
      <span>
        {h.studies.map((x) => `${short(x.row ?? x.key)} ${x.decision ?? DASH}`).join(' · ')}
        <span className="sub">{nApplied ? `${nApplied} of ${h.studies.length} applied` : 'not applied'}</span>
      </span>
    );
  }
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

const settingsText = (h: Hyperopt) =>
  h.tuner === 'deeplearning'
    ? `${h.trials} trials × ${h.days} draws · ${h.trainSeconds} s`
    : h.tuner === 'ensemble'
      ? `${h.trials} combinations · ${h.days} draws`
      : `${h.trials} trials × ${h.days} draws`;

function HyperoptTable({ d, onOpen, onStop, onAgain }: { d: HyperoptList; onOpen: (id: string) => void; onStop: (h: Hyperopt) => void; onAgain: (h: Hyperopt) => void }) {
  return (
    <Card title="Hyperopts" meta={[`${d.hyperopts.length} kept`, 'newest first']} flush>
      <table className="tbl stk">
        <thead>
          <tr>
            <th>Tuner</th>
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
              <td data-l="Tuner">
                <button type="button" className="linkbtn" onClick={() => onOpen(h.id)}>
                  {rowText(h)}
                </button>
                {h.tuner === 'deeplearning' ? <span className="sub">deep learning</span> : null}
              </td>
              <td data-l="Game">{h.game}</td>
              <td data-l="Settings" className="mut">
                {settingsText(h)}
                <span className="sub">
                  gate {h.gate ? `${h.gateSeMargin} SE` : 'off'}
                  {h.varietyLine !== null ? ` · variety ${h.varietyLine}%` : ''}
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
                  <button type="button" className="linkbtn" onClick={() => onAgain(h)} disabled={h.worker === 'gpu' && !d.gpu.configured}>
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

// What each tuner's run does, for the form
const ABOUT: Record<string, string> = {
  deeplearning: 'HyperoptDeepLearning.py on the GPU worker: each trial retrains the model on the newest draws with its suggested values and scores how much better than the counts it predicts (0 = no better).',
  statistics: 'HyperoptStatistics.py on the DL360: each trial backtests the strategy over the last draws with its suggested values and scores a cautious lower bound of its hits or capped profit.',
  boost: 'HyperoptBoost.py on the DL360: each trial backtests the boosting model walk-forward over the last draws (refitting every few draws) and scores a cautious lower bound of its hits or capped profit. Slow trials are pruned.',
  ensemble: 'HyperoptEnsemble.py on the DL360: searches which rows SubsetEnsemble Model votes over, scored on the stored predictions of the last draws. Minutes, not hours.',
  quantum: 'HyperoptQuantum.py on the DL360: backtests the base models once (cached between runs), then tunes the meta-learner rows (quantum kernel, quantum VQC, classical SVM) on a holdout of those draws.',
  rlticket: 'HyperoptRLTicket.py on the DL360: tunes the RL Ticket row on a lottery game, or RL Position on a market, walking the last draws of the stored predictions. RL Position sizes positions: the variety rule does not apply to it.',
};
const HELP: Record<string, string> = {
  trials: 'Trials in this run. The study on the worker keeps every earlier trial, which guides the sampler.',
  days: 'Draws per trial.',
  trainSeconds: 'Training budget of each retrain, in seconds. 240 is what the daily run gives each model.',
  pruneAfterDays: 'Stop a trial whose score after this many draws is below the median of the earlier trials (the first 3 never are). 0 = no pruning.',
  trialTimeout: 'Wall-clock limit of one trial, in seconds: a trial over it is recorded as pruned.',
  gateSeMargin: 'How many standard errors of its own score the best trial must win by.',
};
const HELP_BY: Record<string, Record<string, string>> = {
  deeplearning: { days: 'Draws retrained per trial (3 is the gate’s window; the warm-started draws barely differ, so more mostly cost time).' },
  ensemble: { trials: 'Row combinations the search evaluates (the stored predictions are scored, nothing is retrained).' },
};

// The form: the tuner's own arguments, with its defaults. `from` prefills it (Run again: the same settings, Fresh off,
// so a stopped run resumes).
function HyperoptStart({ d, from, sender, onDone }: { d: HyperoptList; from: Partial<Hyperopt>; sender: Sender; onDone: () => void }) {
  const [tuner, setTuner] = useState(from.tuner ?? (d.gpu.configured ? 'deeplearning' : 'statistics'));
  const t = d.tuners[tuner];
  const f = t ? t.fields : {};
  const initFor = (tk: string, src: Partial<Hyperopt>) =>
    Object.fromEntries(Object.entries(d.tuners[tk]?.fields ?? {}).map(([k, fl]) => [k, String((src as Record<string, unknown>)[k] ?? fl.default)]));
  const [v, setV] = useState<Record<string, string>>(initFor(tuner, from));
  const [game, setGame] = useState(from.game ?? 'lotto');
  const rowsFor = (tk: string, g: string) => Object.keys(d.tuners[tk]?.rows ?? {}).filter((r) => !d.tuners[tk].rowGames[r] || d.tuners[tk].rowGames[r].includes(g));
  const [model, setModel] = useState(from.model ?? (tuner === 'deeplearning' ? 'transformer_model' : rowsFor(tuner, game)[0] ?? ''));
  const [variety, setVariety] = useState(String(from.varietyLine ?? d.variety.varietyLine));
  const [gate, setGate] = useState(from.gate ?? true);
  const [fresh, setFresh] = useState(false);
  const pickTuner = (tk: string) => {
    setTuner(tk);
    setV(initFor(tk, {}));
    const games = d.tuners[tk].games;
    const g = games.includes(game) ? game : games[0];
    setGame(g);
    setModel(tk === 'deeplearning' ? 'transformer_model' : rowsFor(tk, g)[0] ?? '');
  };
  const pickGame = (g: string) => {
    setGame(g);
    if (!rowsFor(tuner, g).includes(model)) setModel(rowsFor(tuner, g)[0] ?? '');
  };
  const set = (k: string) => (e: { target: { value: string } }) => setV((x) => ({ ...x, [k]: e.target.value }));
  const vn = Number(variety);
  const bad =
    Object.entries(v)
      .map(([k, s]) => {
        const fl = f[k];
        const n = Number(s);
        if (!fl) return null;
        return s === '' || !Number.isFinite(n) || (fl.integer && !Number.isInteger(n)) || n < fl.min || n > fl.max ? `${LABEL[k] ?? k}: ${fl.integer ? 'a whole number' : 'a number'} from ${fl.min} to ${fl.max}` : null;
      })
      .find(Boolean) ??
    (tuner === 'deeplearning' && Number(v.pruneAfterDays) > Number(v.days) ? 'Prune after: at most the draws per trial (0 = no pruning)' : null) ??
    (variety === '' || !Number.isFinite(vn) || vn < d.variety.min || vn > d.variety.max ? `Variety line: a percentage from ${d.variety.min} to ${d.variety.max}` : null) ??
    (!t ? 'Pick a tuner' : t.lane === 'gpu' && !d.gpu.configured ? 'No GPU worker is set up: the deep-learning tuner runs there' : !rowsFor(tuner, game).includes(model) ? `Pick a ${t.rowWord.toLowerCase()}` : null);
  const wake = t && t.lane === 'cpu' ? 1800 : 0;
  const budget = !t ? 0 : t.fixedBudget !== null ? Math.min(d.budgetMaxSeconds, t.fixedBudget + wake)
    : Math.min(d.budgetMaxSeconds, (Number(v.trials) + (gate ? t.references : 0)) * Number(v.trialTimeout) + 3600 + wake);
  const start = async () => {
    if (bad) return;
    const body: Record<string, unknown> = { tuner, model, game, gate, varietyLine: vn, ...(tuner === 'deeplearning' ? { fresh } : {}) };
    Object.entries(v).forEach(([k, s]) => { body[k] = Number(s); });
    const done = await sender.send<{ ok: true; message: string; id?: string }>('/api/admin/hyperopts/start', body);
    if (done) onDone();
  };
  const field = (k: string) => (
    <label className="field" key={k}>
      <span>{LABEL[k] ?? k}</span>
      <input value={v[k] ?? ''} onChange={set(k)} inputMode="decimal" autoComplete="off" />
      <small>{HELP_BY[tuner]?.[k] ?? HELP[k] ?? ''}</small>
    </label>
  );
  const rows = rowsFor(tuner, game);
  return (
    <Confirm
      title={from.id ? `Run again: ${rowText(from as Hyperopt)} on ${from.game}` : 'Start a hyperopt'}
      action="Start hyperopt"
      busy={sender.busy}
      error={sender.error ?? bad}
      disabled={Boolean(bad)}
      onConfirm={() => void start()}
      onCancel={onDone}
      lines={[
        `${ABOUT[tuner] ?? ''} Nothing is served until you apply the result.`,
        `At most ${hours(budget)}${t && t.fixedBudget === null ? ` (every trial at its time limit${gate && t.references ? ', plus the gate’s reference runs' : ''}${wake ? ', plus half an hour to switch the DL360 on' : ''})` : ''}. The ${t && t.lane === 'cpu' ? 'statistical and boosting' : 'deep-learning'} backtests wait behind it.`,
      ]}
    >
      <label className="field">
        <span>Tuner</span>
        <select value={tuner} onChange={(e) => pickTuner(e.target.value)} disabled={Boolean(from.id)}>
          {Object.entries(d.tuners).map(([k, x]) => (
            <option key={k} value={k}>
              {x.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Game</span>
        <select value={game} onChange={(e) => pickGame(e.target.value)}>
          {(t?.games ?? []).map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </label>
      {t && Object.keys(t.rows).length > 1 ? (
        <label className="field">
          <span>{t.rowWord}</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {rows.map((m) => (
              <option key={m} value={m}>
                {short(t.rows[m])}
              </option>
            ))}
          </select>
          {tuner === 'deeplearning' ? <small>The LSTM, TCN and unified models are the slow ones: hours a study. The transformer, GNN and autoencoder are an order of magnitude faster.</small> : null}
        </label>
      ) : t ? (
        <p className="line mut">Rows: {t.rows[model] ?? Object.values(t.rows)[0]} (every one the tuner has for this game).</p>
      ) : null}
      {Object.keys(f).filter((k) => k !== 'gateSeMargin').map((k) => field(k))}
      <label className="field">
        <span>Gate</span>
        <select value={gate ? 'on' : 'off'} onChange={(e) => setGate(e.target.value === 'on')}>
          <option value="on">On: score the served values and the defaults on the same draws</option>
          <option value="off">Off: report this run’s best trial only</option>
        </select>
      </label>
      {gate && f.gateSeMargin ? field('gateSeMargin') : null}
      {!(tuner === 'rlticket' && !['lotto', 'euromillions', 'eurodreams', 'vikinglotto', 'jokerplus', 'pick3', 'keno'].includes(game)) ? (
        <label className="field">
          <span>Variety line (%)</span>
          <input value={variety} onChange={(e) => setVariety(e.target.value)} inputMode="decimal" autoComplete="off" />
          <small>
            A candidate that gives the same ticket on this share or more{tuner === 'deeplearning' ? ' of its validation windows, on at least half of its draws,' : ' of its scored draws'} is input-blind: never written or served back. The setting is{' '}
            {d.variety.varietyLine}%.
          </small>
        </label>
      ) : null}
      {tuner === 'deeplearning' ? (
        <label className="field">
          <span>Resume</span>
          <select value={fresh ? 'fresh' : 'resume'} onChange={(e) => setFresh(e.target.value === 'fresh')}>
            <option value="resume">Resume: an unfinished run with these settings on the GPU worker carries on</option>
            <option value="fresh">Fresh: start a new run even if one with these settings is unfinished</option>
          </select>
          <small>Only the latest run can be resumed: a hyperopt with other settings starts a new one.</small>
        </label>
      ) : null}
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
    <Drawer title={h ? `${rowText(h)} · ${h.game}` : 'Hyperopt'} sub={h ? `${settingsText(h)} · ${h.status}` : undefined} onClose={onClose}>
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {h ? <DrawerBody h={h} onChanged={() => { load.retry(); onChanged(); }} /> : null}
    </Drawer>
  );
}

// the score's words per tuner, for the gate's line
const SCORE_TEXT: Record<string, string> = {
  deeplearning: 'minus the mean excess loss over the counts; 0 = as good as the counts, above 0 = better. The standard error is over retrains on almost the same history: a floor, not a test. The validation split contains the lockbox period.',
  statistics: 'a cautious lower bound of hits, capped profit per bet or (markets) log-score over the scored draws, lockbox draws left out.',
  boost: 'a cautious lower bound of hits or capped profit per bet over the scored draws, lockbox draws left out.',
  ensemble: 'the vote’s cautious lower bound over the stored draws, lockbox draws left out.',
  quantum: 'hits on the holdout draws of the base models’ table.',
  rlticket: 'the paper money of the stakes over the stored draws.',
};

function DrawerBody({ h, onChanged }: { h: Hyperopt; onChanged: () => void }) {
  return (
    <>
      <Section title="Run">
        <p className="line">
          Started by {h.by} {stamp(h.requestedAt)}
          {h.startedAt ? ` · ran from ${stamp(h.startedAt)}` : ''}
          {h.finishedAt ? ` to ${stamp(h.finishedAt)}` : ''}
          {h.where ? ` · on ${h.where}` : ''}
          {h.error ? <span className="mut"> · {h.error}</span> : null}
        </p>
        <p className="line mut">
          {h.tunerLabel}: {settingsText(h)}
          {h.tuner === 'deeplearning' ? `, pruning ${h.pruneAfterDays ? `after ${h.pruneAfterDays} draws` : 'off'}` : ''}
          {h.trialTimeout ? `, ${Math.round(h.trialTimeout / 60)} min a trial at most` : ''}, gate {h.gate ? `on (${h.gateSeMargin} SE)` : 'off'}
          {h.varietyLine !== null ? `, variety line ${h.varietyLine}%` : ''}
          {h.fresh ? ', fresh' : ''} · budget {hours(h.budget)}
        </p>
        {h.status === 'running' && nowText(h) ? (
          <p className="line">
            Now: {nowText(h)}
            {h.study?.current?.at ? <span className="mut"> · updated {stamp(h.study.current.at)}</span> : null}
          </p>
        ) : null}
      </Section>
      {h.studies.length ? h.studies.map((s) => <StudySection key={s.key} h={h} s={s} many={h.studies.length > 1} onChanged={onChanged} />) : null}
      <LogSection id={h.id} live={h.status === 'running'} />
    </>
  );
}

function StudySection({ h, s, many, onChanged }: { h: Hyperopt; s: HyperoptStudy; many: boolean; onChanged: () => void }) {
  const row = s.row ?? h.row;
  const records = s.rows && Object.keys(s.rows).length ? Object.entries(s.rows) : s.record ? [[row, s.record] as [string, Record<string, unknown>]] : [];
  const states = s.progress?.states ?? {};
  const applied = h.appliedStudies[s.key] ?? (!many ? h.applied : null);
  const apply = h.apply[s.key] ?? null;
  return (
    <>
      <Section title={many ? short(row) : 'Study'}>
        <p className="line">
          {s.status === 'done' ? 'Done' : s.status === 'gate' ? 'The gate is scoring the references' : s.status === 'failed' ? 'Failed' : s.status === 'skipped' ? `Skipped${s.reason ? `: ${s.reason}` : ''}` : s.status}
          {s.progress ? ` · ${s.progress.done} of ${s.wanted ?? h.trials} trials done` : ''}
          {Object.keys(states).length ? ` (${Object.entries(states).map(([k, x]) => `${x} ${k.toLowerCase()}`).join(', ')})` : ''}
          {s.progress?.best ? ` · best trial ${s.progress.best.trial} at ${s.progress.best.score === null ? 'no score' : fixed(s.progress.best.score, 4)}` : ''}
          {s.error ? <span className="mut"> · {s.error}</span> : null}
        </p>
        {s.status === 'done' ? (
          <>
            <p className="line">
              <b>{decisionText(s.decision)}</b>
              {s.reason ? <span className="mut"> · {s.reason}</span> : null}
            </p>
            {records.map(([name, r]) => (
              <div key={name}>
                {records.length > 1 ? <p className="line"><b>{short(name)}</b>: {String(r.decision ?? DASH)}</p> : null}
                <GateLine h={h} r={r} />
                <Variety r={r} />
              </div>
            ))}
          </>
        ) : null}
      </Section>
      {applied ? (
        <Section title={many ? `Applied: ${short(row)}` : 'Applied'}>
          <p className="line">
            {applied.keys} values written to bestParams_{h.game}.json by {applied.by} {stamp(applied.at)} ({applied.what === 'gate' ? "the gate's result" : "this run's best trial, against the gate"}).
          </p>
        </Section>
      ) : apply ? (
        <ApplySection h={h} s={s} a={apply} onChanged={onChanged} />
      ) : null}
    </>
  );
}

function GateLine({ h, r }: { h: Hyperopt; r: Record<string, unknown> }) {
  const n = (k: string) => (typeof r[k] === 'number' ? (r[k] as number) : null);
  return (
    <p className="line mut">
      Best trial of this run {n('challenger') === null ? DASH : fixed(n('challenger'), 4)} · served values {n('incumbent') === null ? 'not scored' : fixed(n('incumbent'), 4)} · untuned
      defaults {n('default') === null ? 'not scored' : fixed(n('default'), 4)}, on the same {n('window_days') ?? h.days} draws: {SCORE_TEXT[h.tuner] ?? ''}
    </p>
  );
}

// Variety (owner, 11-12 Nov 2026): how often each candidate gave the same ticket. Deep learning: across a draw's
// validation windows (input-blind at the line or more on at least half of its draws) and, since 13 Nov 2026, across
// its backtested draws as well (input-blind at the line or more, the other tuners' measure); the other tuners: across the
// scored backtest draws (input-blind at the line or more). The line is the owner's setting (the run's own value is in
// the record). An input-blind candidate is neither written nor served back. Records from before the rule show nothing.
type Side = {
  same_ticket_mean?: unknown; blind_days?: unknown; variety_days?: unknown; input_blind?: unknown; distinct_tickets?: unknown; tickets?: unknown;
  variety_measure?: unknown; variety_line?: unknown; draws_same_ticket?: unknown; blind_by?: unknown;
};
const pct = (x: number) => `${Number((x * 100).toFixed(1))}%`;
function Variety({ r }: { r: Record<string, unknown> }) {
  const sides: [string, Side | null][] = [
    ['Best trial of this run', (r.challenger_tuning as Side) ?? null],
    ['Served values', (r.incumbent_tuning as Side) ?? null],
    ['Untuned defaults', (r.default_tuning as Side) ?? null],
  ];
  const shown = sides.filter(([, s]) => s && typeof s.same_ticket_mean === 'number');
  const v = r.variety as { rule?: string; blind_trials?: number[] } | undefined;
  if (!shown.length && !v) return null;
  const num = (x: unknown) => (typeof x === 'number' ? x : null);
  return (
    <>
      {shown.map(([label, s]) => (
        <p className="line" key={label}>
          {s!.variety_measure === 'draws' ? (
            <>
              {label}: the same ticket on {pct(num(s!.same_ticket_mean) ?? 0)} of its {num(s!.tickets) ?? 0} scored draws · {num(s!.distinct_tickets) ?? 0} different tickets
            </>
          ) : (
            <>
              {label}: the same ticket on {pct(num(s!.same_ticket_mean) ?? 0)} of its validation windows on average, input-blind on {num(s!.blind_days) ?? 0} of{' '}
              {num(s!.variety_days) ?? 0} draws{num(s!.tickets) ? ` · ${num(s!.distinct_tickets)} different tickets over ${num(s!.tickets)} draws` : ''}
              {num(s!.draws_same_ticket) !== null ? ` · the same ticket on ${pct(num(s!.draws_same_ticket)!)} of its draws` : ''}
            </>
          )}
          {s!.input_blind === true ? (
            <Chip kind="down">
              {Array.isArray(s!.blind_by) && s!.blind_by.length === 1 ? `input-blind (${s!.blind_by[0] === 'draws' ? 'across draws' : 'within draws'})` : 'input-blind'}
            </Chip>
          ) : null}
        </p>
      ))}
      {v ? (
        <p className="line mut">
          Variety rule: {v.rule ?? 'a candidate that gives the same ticket on 95% or more of its validation windows on at least half of its draws is neither written nor served back, and blocks nothing'}.
          {v.blind_trials && v.blind_trials.length ? ` Left out as input-blind: trial${v.blind_trials.length > 1 ? 's' : ''} ${v.blind_trials.join(', ')}.` : ''}
        </p>
      ) : null}
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
            <td>{c.removed ? <span className="mut">removed</span> : val(c.value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ApplySection({ h, s, a, onChanged }: { h: Hyperopt; s: HyperoptStudy; a: { gate: HyperoptApplySide; challenger: HyperoptApplySide }; onChanged: () => void }) {
  const sender = useSend(onChanged);
  const [ask, setAsk] = useState<'gate' | 'challenger' | null>(null);
  const row = s.row ?? h.row;
  const side = ask === 'gate' ? a.gate : ask === 'challenger' ? a.challenger : null;
  const since = side ? side.changedSince : [];
  return (
    <Section title={h.studies.length > 1 ? `Apply: ${short(row)}` : 'Apply'}>
      {a.gate ? (
        <>
          <p className="line">If you apply the gate&apos;s result ({s.decision === 'reverted' ? 'the untuned defaults' : "this run's best trial"}):</p>
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
          title={ask === 'gate' ? `Apply the gate's result to ${short(row)} on ${h.game}` : `Apply this run's best to ${short(row)} on ${h.game}`}
          action="Apply"
          danger={ask === 'challenger' || since.length > 0}
          busy={sender.busy}
          error={sender.error}
          onConfirm={() =>
            void sender.send(`/api/admin/hyperopts/${encodeURIComponent(h.id)}/apply`, { what: ask, study: s.key, anyway: since.length > 0 }).then((done) => {
              if (done) setAsk(null);
            })
          }
          onCancel={() => setAsk(null)}
          lines={[
            `${side.changes.length} values change in bestParams_${h.game}.json on this server, with the gate's record${side.rows.length > 1 ? `s of ${side.rows.join(', ')}` : ''}${ask === 'challenger' ? ' marked "applied by hand against the gate"' : ''}. The next run serves them, and its commit includes the file.`,
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
