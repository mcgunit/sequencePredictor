import { useEffect, useState, type ReactNode } from 'react';
import {
  GAMES,
  LOTTERY_GAMES,
  MARKET_GAMES,
  getAdminParameters,
  type ParamCard,
  type ParamKind,
  type ParamOverride,
  type ParamValue,
  type ParametersView,
} from '../../api';
import { DASH, short, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useSend } from '../../hooks/useSend';
import { useSwitchParam } from '../../hooks/useUrlState';
import { Card } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Confirm from '../../components/admin/Confirm';
import Toast from '../../components/admin/Toast';
import { Failed, Loading, PageHead } from '../../components/admin/States';

// /admin/parameters (owner's roadmap steps 2 and 3, 8 Nov 2026): every setting in one game's bestParams_<game>.json,
// one folding card per model (?game=), the settings several models read in a card of their own - and the manual
// overrides: an administrator sets a value, it wins over the tuners until someone releases it (config/
// paramOverrides.json, never the tuners' file). A key the file does not have is "not set": the model runs on its code
// default. A deep-learning row whose setting is overridden or released trains once from new weights on its next run.

// What an action is about: one setting of the game, with what the dialog says about it.
type Target = {
  key: string;
  label: string; // the setting's name as the dialog shows it
  type: ParamKind;
  tuned: ParamValue;
  tunedSet: boolean;
  override: ParamOverride | null;
  rows: string[];
  retrains: string[]; // deep-learning rows that train from new weights after the change
};
type Ask = { mode: 'set' | 'release'; t: Target };

export default function ParametersPage() {
  const [game, setGame] = useSwitchParam('game', GAMES, 'lotto');
  const load = useLoad(() => getAdminParameters(game), [game]);
  const sender = useSend(() => load.retry());
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const [ask, setAsk] = useState<Ask | null>(null);
  const [value, setValue] = useState('');
  const [note, setNote] = useState('');

  useEffect(() => {
    document.title = 'Parameters · Predictor';
  }, []);
  useEffect(() => setOpen(new Set()), [game]);

  const d = load.kind === 'ok' ? load.data : null;
  const start = (mode: 'set' | 'release', t: Target) => {
    sender.reset();
    const current = t.override ? t.override.value : t.tuned;
    setValue(current === null || current === undefined ? (t.type === 'bool' ? 'true' : '') : String(current));
    setNote(t.override?.note ?? '');
    setAsk({ mode, t });
  };
  const parsed = ask ? parseValue(ask.t.type, value) : null;
  const confirm = async () => {
    if (!ask) return;
    const url = `/api/admin/parameters/${encodeURIComponent(game)}/${ask.mode === 'set' ? 'override' : 'release'}`;
    if (ask.mode === 'set' && (!parsed || 'error' in parsed)) return;
    const body = ask.mode === 'set' && parsed && 'value' in parsed ? { key: ask.t.key, value: parsed.value, note } : { key: ask.t.key };
    if (await sender.send(url, body)) setAsk(null);
  };

  const head = (
    <PageHead
      title="Parameters"
      sub={d ? `${game} · ${d.file.name} · ${d.file.keys} keys${d.overrides.length ? ` · ${d.overrides.length} manual` : ''}` : game}
      meta={d && d.file.modified ? [`file written ${stamp(d.file.modified)}`, 'Brussels time'] : []}
    />
  );
  const picker = (
    <div className="segbar">
      {[LOTTERY_GAMES, MARKET_GAMES].map((list, i) => (
        <div key={i} className="seg" role="group" aria-label={i ? 'Market games' : 'Lottery games'}>
          {list.map((g) => (
            <button key={g} type="button" aria-pressed={g === game} onClick={() => setGame(g)}>
              {g}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
  if (!d) {
    return (
      <div className="admin">
        {head}
        {picker}
        {load.kind === 'loading' ? <Loading /> : load.kind === 'error' ? <Failed error={load.error} retry={load.retry} /> : null}
      </div>
    );
  }
  const all = d.groups.flatMap((g) => g.cards.map((c) => c.id));
  const toggle = (id: string) =>
    setOpen((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  return (
    <div className="admin">
      {head}
      {picker}
      {!d.file.exists ? (
        <p className="note">No {d.file.name} yet: no tuner has written this game&apos;s settings, so every model runs on its code defaults until the next weekend&apos;s tuning.</p>
      ) : null}
      <p className="note">
        The values the tuners last wrote for {game}. A setting marked <i>not set</i> is missing from the file, and the model uses the default in its code. A manual
        override wins over the tuners from {d.nextRun} until someone releases it; the tuners keep tuning underneath. The meta-learner training uses the manual values too;
        the weekend tuners and the controls keep scoring the tuned ones.{' '}
        <button type="button" className="linkbtn" onClick={() => setOpen(new Set(all))}>
          Open all
        </button>{' '}
        ·{' '}
        <button type="button" className="linkbtn" onClick={() => setOpen(new Set())}>
          Fold all
        </button>
      </p>
      <div className="stack">
        {d.overrides.length ? <OverridesCard d={d} onRelease={(t) => start('release', t)} /> : null}
        <SharedCard d={d} onAct={start} />
        {d.groups.map((g) => (
          <div key={g.label} className="stack">
            <div className="grp-label">{g.label}</div>
            {g.cards.map((c) => (
              <ModelCard key={c.id} c={c} open={open.has(c.id)} onToggle={() => toggle(c.id)} onAct={start} />
            ))}
          </div>
        ))}
        {d.empty.length ? (
          <p className="note">
            Nothing stored for: {d.empty.map(short).join(', ')}. Where {game} runs these models, they run on their code defaults.
          </p>
        ) : null}
        {d.unread.length ? (
          <Card title="Stored but not read" meta={[`${d.unread.length} keys`]} help="Keys from an older design of the LSTM rows. They are still in the file, but the predictor does not read them, so changing them changes nothing.">
            <ParamTable game={game} lines={d.unread.map((u) => ({ key: u.key, value: u.value, read: false, note: u.note, override: null, editable: false, type: null }))} onAct={start} rows={[]} retrains={[]} />
          </Card>
        ) : null}
        {d.unassigned.length ? (
          <Card title="Not assigned" meta={[`${d.unassigned.length} keys`]} help="Keys this page does not know a model for (params.js FAMILIES). Shown so nothing in the file is hidden.">
            <ParamTable game={game} lines={d.unassigned.map((u) => ({ key: u.key, value: u.value, read: true, note: u.note, override: null, editable: false, type: null }))} onAct={start} rows={[]} retrains={[]} />
          </Card>
        ) : null}
      </div>
      {ask ? (
        <Confirm
          title={ask.mode === 'set' ? `${ask.t.override ? 'Change' : 'Set'} a manual value: ${ask.t.label}` : `Release the manual value: ${ask.t.label}`}
          action={ask.mode === 'set' ? 'Set manual value' : 'Release'}
          danger={ask.mode === 'release'}
          busy={sender.busy}
          error={sender.error ?? (ask.mode === 'set' && parsed && 'error' in parsed && value !== '' ? parsed.error : null)}
          disabled={ask.mode === 'set' && (!parsed || 'error' in parsed)}
          onConfirm={() => void confirm()}
          onCancel={() => setAsk(null)}
          lines={[
            `${game}${ask.t.rows.length ? ` · ${ask.t.rows.map(short).join(', ')}` : ''}`,
            `Tuned value: ${ask.t.tunedSet ? shownValue(ask.t.tuned) : 'not set (the code default)'}${ask.t.override ? ` · manual now: ${shownValue(ask.t.override.value)}` : ''}`,
            ask.mode === 'set'
              ? `It wins over the tuners from ${d.nextRun} until someone releases it.`
              : `The tuned value is used again from ${d.nextRun}.`,
            ask.t.retrains.length
              ? `${ask.t.retrains.map(short).join(', ')} ${ask.t.retrains.length === 1 ? 'trains' : 'train'} from new weights on that run, without the weights it has now: the run takes longer than usual.`
              : null,
          ]}
        >
          {ask.mode === 'set' ? (
            <>
              <label className="field">
                <span>Value</span>
                {ask.t.type === 'bool' ? (
                  <select value={value} onChange={(e) => setValue(e.target.value)}>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : (
                  <input
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    inputMode={ask.t.type === 'int' ? 'numeric' : ask.t.type === 'float' ? 'decimal' : 'text'}
                    autoComplete="off"
                  />
                )}
                <small>{kindText(ask.t.type)}</small>
              </label>
              <label className="field">
                <span>Why (optional)</span>
                <input value={note} onChange={(e) => setNote(e.target.value)} maxLength={200} autoComplete="off" />
              </label>
            </>
          ) : null}
        </Confirm>
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

const kindText = (t: ParamKind) =>
  t === 'int' ? 'A whole number, 0 or more.' : t === 'float' ? 'A number, 0 or more (a temperature: above 0).' : t === 'bool' ? 'On (true) or off (false).' : 'A word, as the tuners write it.';

// The typed value from the field, or why it is not one.
function parseValue(t: ParamKind, raw: string): { value: ParamValue } | { error: string } {
  const s = raw.trim();
  if (t === 'bool') return s === 'true' || s === 'false' ? { value: s === 'true' } : { error: 'choose true or false' };
  if (t === 'string') return s ? { value: s } : { error: 'type a value' };
  if (!s) return { error: 'type a number' };
  const n = Number(s.replace(',', '.'));
  if (!Number.isFinite(n)) return { error: 'not a number' };
  if (t === 'int' && !Number.isInteger(n)) return { error: 'a whole number' };
  if (n < 0) return { error: 'not below 0' };
  return { value: n };
}

function shownValue(v: ParamValue): string {
  if (v === null || v === undefined) return DASH;
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  return typeof v === 'string' ? v : JSON.stringify(v);
}

function OverridesCard({ d, onRelease }: { d: ParametersView; onRelease: (t: Target) => void }) {
  return (
    <Card title="Manual values" meta={[`${d.overrides.length} in force`, `from ${d.nextRun}`]} help="Every manual value of this game. Each wins over the tuners until it is released; the tuned value underneath keeps being tuned.">
      <table className="tbl">
        <thead>
          <tr>
            <th>Setting</th>
            <th>Manual</th>
            <th>Tuned</th>
            <th>Set by</th>
            <th aria-label="Action" />
          </tr>
        </thead>
        <tbody>
          {d.overrides.map((o) => (
            <tr key={o.key}>
              <td>
                <code>{o.key}</code>
                {o.rows.length ? <span className="mut"> · {o.rows.map(short).join(', ')}</span> : null}
                {o.orphan ? <span className="mut"> · no longer a setting this page shows</span> : null}
              </td>
              <td>
                <b>{shownValue(o.value)}</b>
              </td>
              <td className="mut">{o.tuned === null || o.tuned === undefined ? 'not set' : shownValue(o.tuned)}</td>
              <td className="mut">
                {o.by ?? DASH} · {stamp(o.at)}
                {o.note ? ` · ${o.note}` : ''}
              </td>
              <td>
                <button
                  type="button"
                  className="linkbtn"
                  onClick={() => onRelease({ key: o.key, label: o.key, type: null, tuned: o.tuned, tunedSet: o.tuned !== null && o.tuned !== undefined, override: o, rows: o.rows, retrains: o.retrains })}
                >
                  Release
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function SharedCard({ d, onAct }: { d: ParametersView; onAct: (mode: 'set' | 'release', t: Target) => void }) {
  return (
    <Card title="Shared settings" meta={['read by several models or by the run itself']}>
      <ParamTable
        game={d.game}
        lines={d.shared.map((s) => ({ key: s.key, value: s.value, read: true, note: s.note, override: s.override, editable: s.editable, type: s.type, unset: !s.set, retrains: s.retrains }))}
        onAct={onAct}
        rows={[]}
        retrains={[]}
      />
    </Card>
  );
}

function ModelCard({ c, open, onToggle, onAct }: { c: ParamCard; open: boolean; onToggle: () => void; onAct: (mode: 'set' | 'release', t: Target) => void }) {
  const off = c.switches.filter((s) => (s.override ? s.override.value === false : s.set && s.value === false));
  const newest = [...c.gates].sort((a, b) => (b.date ?? '').localeCompare(a.date ?? ''))[0];
  const pending = c.retrain.filter((r) => r.pending);
  const dlRows = c.deepLearning ? c.rows : [];
  const act = (mode: 'set' | 'release', key: string, label: string, type: ParamKind, tuned: ParamValue, tunedSet: boolean, override: ParamOverride | null, rows = c.rows) =>
    onAct(mode, { key, label, type, tuned, tunedSet, override, rows, retrains: key.startsWith('sampleTemperatures/') ? [] : key === 'labelSmoothing' ? ['LSTM Smoothed Model'] : dlRows });
  return (
    <div className="card" id={`pm-${c.id}`}>
      <h2 className="fold-t">
        <button type="button" className="fold-h" aria-expanded={open} aria-controls={`pm-${c.id}-b`} onClick={onToggle}>
          <span className="nm">{c.title}</span>
          <span className="meta">
            {c.params.length} {c.params.length === 1 ? 'setting' : 'settings'}
            {newest ? ` · last gate ${stamp(newest.date)}${newest.tuner ? ` by ${newest.tuner}` : ''}${newest.decision ? `: ${newest.decision}` : ''}` : ' · no gate record'}
          </span>
          <span className="chips">
            {c.manual ? <Chip>manual · {c.manual}</Chip> : null}
            {pending.length ? <Chip>fresh retrain next run</Chip> : null}
            {off.map((s) => (
              <Chip key={s.key}>off ({s.key})</Chip>
            ))}
            {c.variants.memorise.state === 'on' ? <Chip>memorise twin</Chip> : null}
            {c.variants.sampled.length ? <Chip>sampled twin</Chip> : null}
          </span>
        </button>
      </h2>
      {open ? (
        <div id={`pm-${c.id}-b`} className="card-b">
          {c.switches.length ? (
            <p className="line">
              <b>Switch:</b>{' '}
              {c.switches.map((s, i) => (
                <span key={s.key}>
                  {i ? ' · ' : ''}
                  <code>{s.key}</code>{' '}
                  <Shown value={s.value} unset={!s.set} unsetText="not set (code default)" override={s.override} />{' '}
                  {s.editable ? <Actions override={s.override} onSet={() => act('set', s.key, s.key, 'bool', s.value, s.set, s.override)} onRelease={() => act('release', s.key, s.key, 'bool', s.value, s.set, s.override)} /> : null}
                </span>
              ))}
            </p>
          ) : null}
          {c.retrain.map((r) => (
            <p key={r.row} className="line">
              <b>{short(r.row)}:</b> {r.pending ? `trains from new weights on the next run (asked ${stamp(r.asked)})` : `trained from new weights after the change of ${stamp(r.asked)}`}
            </p>
          ))}
          {c.gates.map((g) => (
            <p key={g.row} className="line">
              <b>{c.gates.length > 1 || c.rows.length > 1 ? `${short(g.row)}: ` : ''}Last gate</b> {stamp(g.date)}
              {g.tuner ? ` · ${g.tuner}` : ''}
              {g.decision ? ` · ${g.decision}` : ''}
              {g.windowDays !== null ? ` · window ${g.windowDays}` : ''}
              {g.runTrials !== null ? ` · ${g.runTrials} trials this run` : ''}
              {g.studyTrials !== null ? ` of ${g.studyTrials} in the study` : ''}
              {g.reason ? <span className="mut"> · {g.reason}</span> : null}
            </p>
          ))}
          <p className="line">
            <b>Variants:</b> memorise {c.variants.memorise.state}
            <span className="mut"> ({c.variants.memorise.why})</span>
            {' · '}
            {c.variants.sampled.length ? (
              <>
                sampled at temperature{' '}
                {c.variants.sampled.map((s, i) => (
                  <span key={s.row}>
                    {i ? ', ' : ''}
                    {c.variants.sampled.length > 1 ? `${short(s.row)} ` : ''}
                    <Shown value={s.temperature} unset={false} override={s.override} />
                    {s.set || s.override ? '' : <span className="mut"> (not set)</span>}{' '}
                    <Actions
                      override={s.override}
                      onSet={() => act('set', s.key, `sample temperature of ${short(s.row)}`, 'float', s.temperature, s.set, s.override, [s.row])}
                      onRelease={() => act('release', s.key, `sample temperature of ${short(s.row)}`, 'float', s.temperature, s.set, s.override, [s.row])}
                    />
                  </span>
                ))}
              </>
            ) : (
              <>
                no sampled twin <span className="mut">({c.variants.sampledWhy ?? 'none for this model'})</span>
              </>
            )}
          </p>
          {c.params.length ? (
            <ParamTable game="" lines={c.params} onAct={onAct} rows={c.rows} retrains={dlRows} />
          ) : (
            <p className="line mut">No settings stored: the model runs on its code defaults.</p>
          )}
        </div>
      ) : null}
    </div>
  );
}

// A value as served: the manual one in bold with "manual", the tuned one struck through beside it.
function Shown({ value, unset, unsetText = 'not set', override }: { value: ParamValue; unset: boolean; unsetText?: string; override: ParamOverride | null }) {
  if (override) {
    return (
      <span title={`set by ${override.by ?? 'an administrator'} ${stamp(override.at)}${override.note ? ` · ${override.note}` : ''}`}>
        <b>
          <Value v={override.value} />
        </b>{' '}
        <Chip>manual</Chip> <span className="mut">tuned {unset ? 'not set' : <s><Value v={value} /></s>}</span>
      </span>
    );
  }
  return unset ? <span className="mut">{unsetText}</span> : <Value v={value} />;
}

function Actions({ override, onSet, onRelease }: { override: ParamOverride | null; onSet: () => void; onRelease: () => void }) {
  return (
    <span className="mut">
      <button type="button" className="linkbtn" onClick={onSet}>
        {override ? 'Change' : 'Set'}
      </button>
      {override ? (
        <>
          {' · '}
          <button type="button" className="linkbtn" onClick={onRelease}>
            Release
          </button>
        </>
      ) : null}
    </span>
  );
}

type Line = { key: string; value: ParamValue; read: boolean; note: string | null; override: ParamOverride | null; editable: boolean; type: ParamKind; unset?: boolean; retrains?: string[] };

function ParamTable({
  lines,
  onAct,
  rows,
  retrains,
}: {
  game: string;
  lines: Line[];
  onAct: (mode: 'set' | 'release', t: Target) => void;
  rows: string[];
  retrains: string[]; // the card's deep-learning rows: they train from new weights after a change (a line can name its own)
}) {
  const target = (l: Line): Target => ({
    key: l.key,
    label: l.key,
    type: l.type,
    tuned: l.value,
    tunedSet: !l.unset,
    override: l.override,
    rows: l.key === 'labelSmoothing' ? ['LSTM Smoothed Model'] : rows,
    retrains: l.retrains ?? (l.key === 'labelSmoothing' && retrains.length ? ['LSTM Smoothed Model'] : retrains),
  });
  const cell = (l: Line): ReactNode => <Shown value={l.value} unset={!!l.unset} override={l.override} />;
  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>Setting</th>
          <th>Value</th>
          <th>Note</th>
          <th aria-label="Action" />
        </tr>
      </thead>
      <tbody>
        {lines.map((l) => (
          <tr key={l.key} className={l.read ? undefined : 'mut'}>
            <td>
              <code>{l.key}</code>
            </td>
            <td>{cell(l)}</td>
            <td className="mut">{l.note ?? ''}</td>
            <td>{l.editable ? <Actions override={l.override} onSet={() => onAct('set', target(l))} onRelease={() => onAct('release', target(l))} /> : null}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// A stored value as it reads best: integers as they are, other numbers to four significant digits (the exact value on
// hover), lists joined, objects as compact JSON.
function Value({ v }: { v: ParamValue }) {
  if (v === null || v === undefined) return <span className="mut">{DASH}</span>;
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return <span className="num">{v}</span>;
    return (
      <span className="num" title={String(v)}>
        {Number(v.toPrecision(4))}
      </span>
    );
  }
  if (typeof v === 'boolean') return <span>{v ? 'true' : 'false'}</span>;
  if (typeof v === 'string') return <span>{v}</span>;
  if (Array.isArray(v) && v.every((x) => typeof x !== 'object' || x === null)) return <span>{v.length ? v.map(String).join(', ') : '(empty list)'}</span>;
  const text = JSON.stringify(v);
  return <code style={{ overflowWrap: 'anywhere' }}>{text.length > 240 ? `${text.slice(0, 240)}…` : text}</code>;
}
