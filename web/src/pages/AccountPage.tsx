import { useEffect, useState, type FormEvent } from 'react';
import { ACTIONS, getAccount, type AccountView, type Done } from '../api';
import { DASH, stamp } from '../format';
import { useLoad } from '../hooks/useLoad';
import { setCsrf } from '../hooks/useMe';
import { useSend, type Sender } from '../hooks/useSend';
import { Card } from '../components/admin/Card';
import Toast from '../components/admin/Toast';
import { Failed, Loading, PageHead } from '../components/admin/States';

// /account, every signed-in reader's own: who they are signed in as, their password (the current one required, as on
// the old page), and the sign-ins and changes recorded for their name, so a session they did not start shows here.
// Every time is Brussels time.
export default function AccountPage() {
  const load = useLoad(getAccount, []);
  const sender = useSend(() => load.retry());

  useEffect(() => {
    document.title = 'Your account · Predictor';
  }, []);

  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title="Your account" />
        {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} /> : <Loading />}
      </>
    );
  }
  const a = load.data;
  const role = a.role === 'admin' ? 'administrator' : 'user';
  return (
    <div className="admin">
      <PageHead title="Your account" sub={a.open ? 'Open access: nobody signs in on this server' : `${a.name} · ${role}`} />
      <div className="grid">
        <Card title="Account" meta={['Brussels time']}>
          <dl className="kv">
            <dt>Name</dt>
            <dd>{a.name}</dd>
            <dt>Role</dt>
            <dd>
              {role}
              <span className="sub">{a.source === 'environment' ? 'from the server’s .env' : a.source === 'users.json' ? 'an account on this server' : a.source}</span>
            </dd>
            <dt>Created</dt>
            <dd>{a.created ? stamp(a.created) : a.source === 'environment' ? 'set in .env' : DASH}</dd>
            <dt>Password</dt>
            <dd>
              {a.passwordChangedAt ? `changed ${stamp(a.passwordChangedAt)}` : a.created ? `set ${stamp(a.created)}` : a.source === 'environment' ? 'in .env' : DASH}
            </dd>
            <dt>Last sign-in</dt>
            <dd>
              {a.lastSignIn?.at ? stamp(a.lastSignIn.at) : DASH}
              {a.lastSignIn ? <span className="sub">{[a.lastSignIn.device, a.lastSignIn.address].filter(Boolean).join(' · ') || DASH}</span> : null}
            </dd>
          </dl>
        </Card>
        <PasswordCard a={a} sender={sender} />
        <ActivityCard a={a} />
      </div>
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

function PasswordCard({ a, sender }: { a: AccountView; sender: Sender }) {
  const min = a.password.minLength;
  const [current, setCurrent] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [problem, setProblem] = useState<string | null>(null);

  // the same rules as auth.js changeOwnPassword, so a typo is caught before anything is sent; the server checks again
  const check = (): string | null => {
    if (!current) return 'Type your current password.';
    if (password.length < min) return `The new password must be at least ${min} characters.`;
    if (password !== confirm) return 'The two new passwords do not match.';
    if (password === current) return 'The new password is the same as the current one.';
    return null;
  };
  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    const p = check();
    setProblem(p);
    if (p || sender.busy) return;
    const done = await sender.send<Done & { csrf?: unknown }>(ACTIONS.ownPassword, { current, password, confirm });
    if (!done) return;
    // the change re-issued this session: its new form token is the one every later action sends
    if (typeof done.csrf === 'string' && done.csrf) setCsrf(done.csrf);
    setCurrent('');
    setPassword('');
    setConfirm('');
  };

  if (!a.password.canChange) {
    return (
      <Card title="Password">
        <p className="note">{a.password.why ?? 'This password cannot be changed here.'}</p>
        {a.role === 'admin' && !a.open ? (
          <p className="note" style={{ marginTop: 8 }}>
            To change it, edit WEB_PASSWORD in the server&apos;s .env and restart the server. Every administrator session ends the moment it changes.
          </p>
        ) : null}
      </Card>
    );
  }
  return (
    <Card title="Password">
      <form className="pwform" onSubmit={(e) => void submit(e)} noValidate>
        <input type="text" name="username" autoComplete="username" value={a.name} readOnly hidden />
        <label className="field">
          <span>Current password</span>
          <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} required />
        </label>
        <label className="field">
          <span>New password</span>
          <input type="password" autoComplete="new-password" minLength={min} value={password} onChange={(e) => setPassword(e.target.value)} required />
          <small>at least {min} characters, not the current one</small>
        </label>
        <label className="field">
          <span>New password again</span>
          <input type="password" autoComplete="new-password" minLength={min} value={confirm} onChange={(e) => setConfirm(e.target.value)} required />
        </label>
        {problem ?? sender.error ? (
          <p className="ferr" role="alert">
            {problem ?? sender.error}
          </p>
        ) : null}
        <div className="acts">
          <button type="submit" className="btn primary" disabled={sender.busy} aria-busy={sender.busy || undefined}>
            {sender.busy ? 'Sending…' : 'Change my password'}
          </button>
        </div>
        <p className="note">Your other browsers are signed out; this one stays signed in.</p>
      </form>
    </Card>
  );
}

function ActivityCard({ a }: { a: AccountView }) {
  return (
    <Card
      title="Your recent activity"
      className="span-3"
      flush
      meta={[a.open ? '' : `the newest ${a.activity.shown}${a.activity.shown >= a.activity.max ? ` (at most ${a.activity.max})` : ''}`, 'Brussels time']}
    >
      {a.activity.items.length ? (
        <table className="tbl stk">
          <thead>
            <tr>
              <th>When</th>
              <th>What</th>
              <th>From</th>
            </tr>
          </thead>
          <tbody>
            {a.activity.items.map((e, i) => (
              <tr key={`${e.at}-${i}`}>
                <td className="lead">{stamp(e.at)}</td>
                <td data-l="What">
                  {e.label}
                  {e.detail ? <span className="sub">{e.detail}</span> : null}
                </td>
                <td data-l="From">
                  {e.address ?? DASH}
                  {e.device ? <span className="sub">{e.device}</span> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="empty">{a.open ? 'Open access: nothing is recorded for a name.' : 'Nothing recorded for your name yet.'}</div>
      )}
    </Card>
  );
}
