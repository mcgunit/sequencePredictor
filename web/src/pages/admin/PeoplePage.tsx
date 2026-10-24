import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { ACTIONS, getAdminPeople, type AuditEntry, type PeopleView, type PersonLine } from '../../api';
import { DASH, span, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useSend, type Sender } from '../../hooks/useSend';
import Tile from '../../components/Tile';
import { Card } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Confirm from '../../components/admin/Confirm';
import Toast from '../../components/admin/Toast';
import { Failed, Loading, PageHead } from '../../components/admin/States';

// /admin/people: who has an account and when each last signed in, the sign-ins of the last 7 days that deserve a look
// (failed, locked out, a device or an address the name had not used before), and the whole activity behind "Show all".
// Add user, Set password and Delete each go through an in-page confirmation. Every time is Brussels time.
const USER_NAME = /^[A-Za-z0-9._-]{3,32}$/;

type PeopleAsk = { kind: 'add' } | { kind: 'password'; name: string } | { kind: 'delete'; person: PersonLine };

export default function PeoplePage() {
  const [all, setAll] = useState(false);
  const load = useLoad(() => getAdminPeople(all), [all]);
  const sender = useSend(() => load.retry());
  const [ask, setAsk] = useState<PeopleAsk | null>(null);
  const { hash } = useLocation();

  useEffect(() => {
    document.title = 'People · Predictor';
  }, []);
  useEffect(() => {
    if (load.kind === 'ok' && hash) document.getElementById(hash.slice(1))?.scrollIntoView({ block: 'start' });
  }, [hash, load.kind]);

  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title="People" />
        {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} /> : <Loading />}
      </>
    );
  }
  const d = load.data;
  const open = (a: PeopleAsk) => {
    sender.reset();
    setAsk(a);
  };
  const close = () => setAsk(null);
  return (
    <div className="admin">
      <PageHead title="People" sub={d.auth.enabled ? `${d.users.length} accounts` : 'Authentication is off'} meta={[`as of ${stamp(d.builtAt)}`, 'Brussels time']} />
      {!d.auth.enabled ? (
        <p className="statusline" role="status">
          <Chip kind="warn">read-only</Chip> No WEB_USER and WEB_PASSWORD on this server, so nobody signs in and accounts cannot be changed here.
        </p>
      ) : null}
      <Tiles d={d} />
      <div className="grid">
        <UsersCard d={d} onAsk={open} />
        <ExceptionsCard d={d} />
        <ActivityCard d={d} all={all} setAll={setAll} busy={load.busy} />
      </div>
      {ask?.kind === 'add' ? <AddDialog d={d} sender={sender} onClose={close} /> : null}
      {ask?.kind === 'password' ? <PasswordDialog d={d} name={ask.name} sender={sender} onClose={close} /> : null}
      {ask?.kind === 'delete' ? (
        <Confirm
          title={`Delete ${ask.person.name}?`}
          lines={[
            `Their sessions end at once and they can no longer sign in${ask.person.lastSignIn?.at ? ` (last signed in ${stamp(ask.person.lastSignIn.at)})` : ''}.`,
            'What they did stays in the activity log; the name can be added again later.',
          ]}
          action="Delete"
          danger
          tz={Boolean(ask.person.lastSignIn?.at)}
          busy={sender.busy}
          error={sender.error}
          onConfirm={() => void sender.send(ACTIONS.deleteUser, { username: ask.person.name }).then((r) => r && close())}
          onCancel={close}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

// --- tiles ---------------------------------------------------------------------------------------------------------
function Tiles({ d }: { d: PeopleView }) {
  const t = d.tiles;
  const admins = d.users.filter((u) => u.role === 'admin').length;
  const window = `last ${t.days} days · ${span(t.from, t.to)}`;
  const short = t.scanned.complete ? '' : { text: `read back to ${stamp(t.scanned.oldest)} only (the newest ${t.scanned.events} events): may be short`, warn: true };
  return (
    <div className="kpis k4">
      <Tile href="#users" label="Users" value={String(t.users)} unit="accounts" foot={[`${admins} administrator (from .env) · ${t.users - admins} user${t.users - admins === 1 ? '' : 's'}`]} />
      <Tile href="#exceptions" label="Failed sign-ins" value={String(t.failed)} unit={`in ${t.days} days`} foot={[window, short]} />
      <Tile
        href="#exceptions"
        label="Lockouts"
        value={String(t.lockouts)}
        unit={`in ${t.days} days`}
        foot={[
          window,
          'too many failed attempts on one name from one address',
          t.refusedWhileLocked ? `${t.refusedWhileLocked} attempt${t.refusedWhileLocked === 1 ? '' : 's'} refused while locked` : '',
          short,
        ]}
      />
      <Tile
        href="#exceptions"
        label="New device/IP"
        value={String(t.newDeviceOrAddress)}
        unit={`sign-in${t.newDeviceOrAddress === 1 ? '' : 's'} in ${t.days} days`}
        foot={[window, 'from a device or an address the name had not used before; a first sign-in on record counts', short]}
      />
    </div>
  );
}

// --- the accounts ----------------------------------------------------------------------------------------------------
function UsersCard({ d, onAsk }: { d: PeopleView; onAsk: (a: PeopleAsk) => void }) {
  const writable = d.auth.writable;
  return (
    <Card
      id="users"
      className="span-3"
      title="Accounts"
      flush
      help={
        <>
          The administrator signs in with WEB_USER and WEB_PASSWORD from the server&apos;s .env; that account cannot be changed here. Users sign in with their own
          password and see Home, Lottery (the top 5 models only, as ranked before each draw), Crypto, Shares, the Council (asking it, and their own sessions),
          their own account and What&apos;s new. Last sign-in: the newest in the activity log (with the browser), or the account&apos;s own record when that is newer. Setting a
          password ends that user&apos;s open sessions.
        </>
      }
      meta={['Brussels time']}
      foot={
        writable ? (
          <button type="button" className="btn primary" onClick={() => onAsk({ kind: 'add' })}>
            Add user…
          </button>
        ) : (
          <span className="fnote">Read-only while authentication is off.</span>
        )
      }
    >
      <table className="tbl stk">
        <thead>
          <tr>
            <th>User</th>
            <th>Role</th>
            <th>Created</th>
            <th>Last sign-in</th>
            {writable ? <th>Change</th> : null}
          </tr>
        </thead>
        <tbody>
          {d.users.map((u) => (
            <tr key={u.name}>
              <td className="lead">
                <b>{u.name}</b>
                <span className="sub">{u.source === 'environment' ? 'from .env' : 'users.json'}</span>
              </td>
              <td data-l="Role">
                <Chip kind="flat">{u.role === 'admin' ? 'administrator' : 'user'}</Chip>
              </td>
              <td data-l="Created">
                {u.created ? stamp(u.created) : <span className="mut">{u.source === 'environment' ? 'set in .env' : DASH}</span>}
                {u.passwordChangedAt ? <span className="sub">password set {stamp(u.passwordChangedAt)}</span> : null}
              </td>
              <td data-l="Last sign-in">
                {u.lastSignIn?.at ? stamp(u.lastSignIn.at) : <span className="mut">never</span>}
                {u.lastSignIn ? <span className="sub">{[u.lastSignIn.device, u.lastSignIn.address].filter(Boolean).join(' · ') || DASH}</span> : null}
                {u.signIns !== null ? <span className="sub">{u.signIns} sign-in{u.signIns === 1 ? '' : 's'} in all</span> : null}
              </td>
              {writable ? (
                <td data-l="Change">
                  {u.role === 'admin' ? (
                    <span className="mut">in .env</span>
                  ) : (
                    <span className="acts">
                      <button type="button" className="btn" onClick={() => onAsk({ kind: 'password', name: u.name })}>
                        Set password…
                      </button>
                      <button type="button" className="btn danger" onClick={() => onAsk({ kind: 'delete', person: u })}>
                        Delete…
                      </button>
                    </span>
                  )}
                </td>
              ) : null}
            </tr>
          ))}
          {!d.users.length ? (
            <tr>
              <td colSpan={writable ? 5 : 4} className="mut">
                No accounts.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </Card>
  );
}

// --- the exceptions of the last 7 days ---------------------------------------------------------------------------------
const WARN_KINDS = ['failed sign-in', 'locked out', 'refused while locked'];
function EntryRows({ items, kind }: { items: (AuditEntry & { kind?: string })[]; kind?: boolean }) {
  return (
    <>
      {items.map((e, i) => (
        <tr key={`${e.at}-${i}`}>
          <td className="lead">{stamp(e.at)}</td>
          <td data-l="User">{e.user ?? DASH}</td>
          <td data-l="What">
            {kind && e.kind ? <Chip kind={WARN_KINDS.includes(e.kind) ? 'warn' : 'flat'}>{e.kind}</Chip> : e.label}
            {e.detail ? <span className="sub">{e.detail}</span> : null}
          </td>
          <td data-l="From">
            {e.address ?? DASH}
            {e.device ? <span className="sub">{e.device}</span> : null}
          </td>
        </tr>
      ))}
    </>
  );
}

function ExceptionsCard({ d }: { d: PeopleView }) {
  const ex = d.exceptions;
  const t = d.tiles;
  return (
    <Card
      id="exceptions"
      className="span-3"
      title="Sign-in exceptions"
      flush
      help={
        <>
          Failed sign-ins; a lockout when they reach the limit on one name from one address, and each attempt refused while it lasts; and sign-ins from a
          device or an address the same name had not used before: {ex.compared}. A browser update is not a new device.
        </>
      }
      meta={[`last ${t.days} days`, span(t.from, t.to), ex.of > ex.shown ? `the newest ${ex.shown} of ${ex.of}` : `${ex.of} found`, 'Brussels time']}
    >
      {!t.scanned.complete ? (
        <p className="note pad">
          Read back to {stamp(t.scanned.oldest)} only (the newest {t.scanned.events} events), not to the window&apos;s start: the list may be short.
        </p>
      ) : null}
      {ex.items.length ? (
        <table className="tbl stk">
          <thead>
            <tr>
              <th>When</th>
              <th>User</th>
              <th>What</th>
              <th>From</th>
            </tr>
          </thead>
          <tbody>
            <EntryRows items={ex.items} kind />
          </tbody>
        </table>
      ) : (
        <div className="empty">No failed sign-in, lockout, new device or new address in the last {t.days} days.</div>
      )}
    </Card>
  );
}

// --- the activity log ------------------------------------------------------------------------------------------------
function ActivityCard({ d, all, setAll, busy }: { d: PeopleView; all: boolean; setAll: (v: boolean) => void; busy: boolean }) {
  const a = d.activity;
  const total = a.of ?? a.shown;
  const more = !all && total > a.shown;
  return (
    <Card
      id="activity"
      className="span-3"
      title="Activity"
      flush
      help={
        <>
          Every sign-in, sign-out and account change recorded in the audit log, newest first: who, what, from which address and browser. The log is rotated at
          2 MB with one older file kept, so it reaches back only to the date shown. Addresses are the caller&apos;s as seen through the reverse proxy.
        </>
      }
      meta={[`the newest ${a.shown} of ${total} events`, a.oldest ? `log since ${stamp(a.oldest)}` : '', 'Brussels time']}
      foot={
        more || all ? (
          <button type="button" className="linkbtn more" disabled={busy} onClick={() => setAll(!all)}>
            {busy ? 'Loading…' : all ? 'Show the newest 20' : total > 1000 ? `Show the newest 1000 of ${total}` : `Show all ${total}`}
          </button>
        ) : null
      }
    >
      {a.items.length ? (
        <table className="tbl stk">
          <thead>
            <tr>
              <th>When</th>
              <th>User</th>
              <th>What</th>
              <th>From</th>
            </tr>
          </thead>
          <tbody>
            <EntryRows items={a.items} />
          </tbody>
        </table>
      ) : (
        <div className="empty">Nothing recorded yet.</div>
      )}
    </Card>
  );
}

// --- the dialogs with fields ----------------------------------------------------------------------------------------------
function PasswordFields({ min, password, repeat, setPassword, setRepeat }: { min: number; password: string; repeat: string; setPassword: (v: string) => void; setRepeat: (v: string) => void }) {
  return (
    <>
      <label className="field">
        <span>Password</span>
        <input type="password" autoComplete="new-password" minLength={min} value={password} onChange={(e) => setPassword(e.target.value)} required />
        <small>at least {min} characters</small>
      </label>
      <label className="field">
        <span>Password again</span>
        <input type="password" autoComplete="new-password" minLength={min} value={repeat} onChange={(e) => setRepeat(e.target.value)} required />
      </label>
    </>
  );
}

// The same rules as auth.js, so a typo is caught before anything is sent; the server checks them again.
function passwordProblem(min: number, password: string, repeat: string): string | null {
  if (password.length < min) return `Password must be at least ${min} characters.`;
  if (password !== repeat) return 'The two passwords do not match.';
  return null;
}

function AddDialog({ d, sender, onClose }: { d: PeopleView; sender: Sender; onClose: () => void }) {
  const min = d.auth.minPasswordLength;
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [repeat, setRepeat] = useState('');
  const [problem, setProblem] = useState<string | null>(null);
  const submit = async () => {
    const username = name.trim();
    const p = !USER_NAME.test(username) ? `User name must be ${d.auth.userName}.` : passwordProblem(min, password, repeat);
    setProblem(p);
    if (p) return;
    if (await sender.send(ACTIONS.addUser, { username, password })) onClose();
  };
  return (
    <Confirm
      title="Add a user"
      lines={[
        "The user signs in with this name and password and sees Home, Lottery (the top 5 models only, as ranked before each draw), Crypto, Shares, the Council (asking it, and their own sessions), their own account and What's new.",
      ]}
      action="Add user"
      busy={sender.busy}
      error={problem ?? sender.error}
      onConfirm={() => void submit()}
      onCancel={onClose}
    >
      <label className="field">
        <span>User name</span>
        <input type="text" autoComplete="off" autoCapitalize="none" spellCheck={false} maxLength={32} value={name} onChange={(e) => setName(e.target.value)} required />
        <small>{d.auth.userName}</small>
      </label>
      <PasswordFields min={min} password={password} repeat={repeat} setPassword={setPassword} setRepeat={setRepeat} />
    </Confirm>
  );
}

function PasswordDialog({ d, name, sender, onClose }: { d: PeopleView; name: string; sender: Sender; onClose: () => void }) {
  const min = d.auth.minPasswordLength;
  const [password, setPassword] = useState('');
  const [repeat, setRepeat] = useState('');
  const [problem, setProblem] = useState<string | null>(null);
  const submit = async () => {
    const p = passwordProblem(min, password, repeat);
    setProblem(p);
    if (p) return;
    if (await sender.send(ACTIONS.setPassword, { username: name, password })) onClose();
  };
  return (
    <Confirm
      title={`Set a new password for ${name}?`}
      lines={[`${name}'s open sessions end at once: they sign in again with the new password.`]}
      action="Set password"
      busy={sender.busy}
      error={problem ?? sender.error}
      onConfirm={() => void submit()}
      onCancel={onClose}
    >
      <PasswordFields min={min} password={password} repeat={repeat} setPassword={setPassword} setRepeat={setRepeat} />
    </Confirm>
  );
}
