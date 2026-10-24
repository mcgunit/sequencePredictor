import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ApiError } from '../api';
import {
  MAX_CONTEXT,
  MAX_QUESTION,
  SESSION_ID,
  askCouncil,
  deleteSession,
  getEndpoints,
  getJob,
  getSession,
  getSessions,
  getStatus,
  type CouncilResult,
  type CouncilStatus,
  type Endpoints,
  type Job,
  type Seat,
  type Session,
  type SessionSummary,
  type Turn,
} from '../council';
import { DASH, day, stamp } from '../format';
import { useLoad } from '../hooks/useLoad';
import { csrfToken, refreshMe, useMe } from '../hooks/useMe';
import { Card } from '../components/admin/Card';
import Confirm from '../components/admin/Confirm';
import { Failed, Loading, PageHead } from '../components/admin/States';
import Toast, { type ToastMessage } from '../components/admin/Toast';
import Markdown, { excerpt } from '../components/council/Markdown';

// /council: the question box and the reader's own sessions (kept for their account on the server, councilSessions.js).
// A question is filed in the open session, or starts one; the server follows the run to its end, so closing the tab
// loses nothing, and this page shows its progress from the run's own job route while it runs. Who sits on the council
// and how to reach them (the members, labs, temperatures, head, preset, reachability) is the administrator's: it shows
// only on their page, in a folded card. A run's answer names the members whose answers it weighs, as before.

const PHASE: Record<string, string> = {
  starting: 'Summoning the council…',
  members: 'The members are answering…',
  head: 'The head is weighing the answers…',
  done: 'The council has reported.',
};
const SEAT_WORDS: Record<Seat['state'], string> = { waiting: 'waiting', asking: 'answering…', answered: 'answered', cached: 'answered (from the cache)', failed: 'no answer' };

// A user's page says a failure in the server's words, without an address in them (an endpoint's URL or host:port is
// the administrator's to see; an error body can carry one).
function forReader(text: string, admin: boolean): string {
  if (admin) return text;
  return text
    .replace(/\bhttps?:\/\/[^\s)'"]+/gi, 'an endpoint')
    .replace(/\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b/g, 'an endpoint')
    .replace(/\blocalhost(?::\d+)?\b/gi, 'an endpoint');
}
// why an ask was refused, for the reader
function askRefusal(e: unknown, admin: boolean): string {
  if (e instanceof ApiError) {
    if (e.code === 'busy') return 'The council is already answering another question. Try again shortly.';
    if (e.status === 503 && !admin) return 'The council cannot sit right now: its service is not answering. Your earlier sessions can still be read.';
    if (e.status === 403 && /form token/i.test(e.message)) return 'The form token was out of date (your session was renewed). Try again.';
    if (e.status === 404) return 'That session is no longer there. Start a new one.';
    return forReader(e.detail ?? e.message, admin);
  }
  return (e as Error)?.message || 'The question could not be sent.';
}

// Live state of a run this page follows: its newest job answer, or that the page lost contact, or that the council no
// longer knows it (restarted): then the session's own record says how it ended.
interface Live {
  job: Job | null;
  lost: boolean;
  gone: boolean;
}

export default function CouncilPage() {
  const me = useMe();
  const admin = Boolean(me?.user && (me.user.admin || me.user.open));
  const [params, setParams] = useSearchParams();
  const raw = params.get('session');
  const sid = raw && SESSION_ID.test(raw) ? raw : null;

  const [listKey, setListKey] = useState(0);
  const sessions = useLoad(getSessions, [listKey]);
  const reloadList = useCallback(() => setListKey((n) => n + 1), []);

  // the open session, read again whenever a run in it ends
  const [session, setSession] = useState<Session | null>(null);
  const [sessionError, setSessionError] = useState<Error | null>(null);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [sessionAttempt, setSessionAttempt] = useState(0);
  useEffect(() => {
    setSessionError(null);
    if (!sid) {
      setSession(null);
      return undefined;
    }
    let alive = true;
    setSessionBusy(true);
    getSession(sid)
      .then((s) => alive && setSession(s))
      .catch((e: Error) => {
        if (!alive) return;
        setSession(null);
        setSessionError(e);
      })
      .finally(() => alive && setSessionBusy(false));
    return () => {
      alive = false;
    };
  }, [sid, sessionAttempt]);
  const shown = session && session.id === sid ? session : null;

  // the members and their endpoints: the administrator's only (a user's request would be a 403)
  const endpoints = useLoad<Endpoints | null>(() => (admin ? getEndpoints() : Promise.resolve(null)), [admin]);
  // a user's: only whether the council can sit now, looked at again every 30 s while it cannot
  const [statusKey, setStatusKey] = useState(0);
  const status = useLoad<CouncilStatus | null>(() => (admin ? Promise.resolve(null) : getStatus()), [admin, statusKey]);
  const notSitting = !admin && status.kind === 'ok' && status.data !== null && !status.data.ready;
  useEffect(() => {
    if (!notSitting) return undefined;
    const timer = setInterval(() => setStatusKey((n) => n + 1), 30000);
    return () => clearInterval(timer);
  }, [notSitting]);

  // on a phone the conversation comes after the ask box and the sessions: brought into view when a session opens or a
  // question has been sent, or the tap would seem to do nothing
  const log = useRef<HTMLDivElement>(null);
  const showLog = () => {
    if (window.matchMedia('(max-width: 1100px)').matches) requestAnimationFrame(() => log.current?.scrollIntoView({ block: 'start' }));
  };

  // follow the runs still pending in the open session
  const [live, setLive] = useState<Record<string, Live>>({});
  const pending = useMemo(() => (shown ? shown.turns.filter((t) => t.state === 'pending' && t.jobId).map((t) => ({ job: t.jobId as string, turn: t.id })) : []), [shown]);
  const pendingKey = pending.map((p) => p.job).join(',');
  useEffect(() => {
    if (!pending.length || !sid) return undefined;
    let alive = true;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const later = (fn: () => void, ms: number) => {
      timers.push(setTimeout(fn, ms));
    };
    // once a run has ended, read the session again until the server's own tracker has filed its end (it looks every 3 s)
    const settle = (turnId: string, n: number) =>
      later(() => {
        getSession(sid)
          .then((s) => {
            if (!alive) return;
            const t = s.turns.find((x) => x.id === turnId);
            if (t && t.state === 'pending' && n < 6) return settle(turnId, n + 1);
            setSession(s);
            reloadList();
            return undefined;
          })
          .catch(() => alive && n < 6 && settle(turnId, n + 1));
      }, n === 0 ? 1500 : 3000);
    pending.forEach(({ job, turn }) => {
      const step = () => {
        if (!alive) return;
        getJob(job).then(
          (j) => {
            if (!alive) return;
            setLive((m) => ({ ...m, [job]: { job: j, lost: false, gone: false } }));
            if (j.state === 'done' || j.state === 'failed') return settle(turn, 0);
            // while the replies stream in, every second so the text grows at reading speed
            const working = j.progress?.phase === 'members' || j.progress?.phase === 'head';
            return later(step, working ? 1000 : 2000);
          },
          (e: Error) => {
            if (!alive) return;
            if (e instanceof ApiError && (e.status === 404 || e.status === 400)) {
              setLive((m) => ({ ...m, [job]: { job: m[job]?.job ?? null, lost: false, gone: true } }));
              settle(turn, 0);
              return;
            }
            // one failed read is not a failed run: the run goes on
            setLive((m) => ({ ...m, [job]: { job: m[job]?.job ?? null, lost: true, gone: false } }));
            later(step, 5000);
          },
        );
      };
      step();
    });
    return () => {
      alive = false;
      timers.forEach(clearTimeout);
    };
    // restarted when the set of pending runs changes (pendingKey names them)
  }, [pendingKey, sid]);

  // --- asking ---------------------------------------------------------------------------------------------------------
  const [question, setQuestion] = useState('');
  const [context, setContext] = useState('');
  const [asking, setAsking] = useState<{ question: string; at: string } | null>(null);
  const [askError, setAskError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const running = pending.length > 0 || asking !== null;
  const summon = endpoints.kind === 'ok' && endpoints.data?.status && !endpoints.data.status.ready ? endpoints.data.status : null;
  const canAsk = !running && !summon && !notSitting && question.trim().length > 0 && question.length <= MAX_QUESTION && context.length <= MAX_CONTEXT;

  const ask = async (ev?: FormEvent) => {
    ev?.preventDefault();
    const q = question.trim();
    if (!q || !canAsk) return;
    const token = csrfToken();
    setAskError(null);
    if (!token) {
      void refreshMe();
      setAskError('The form token is not loaded yet - try again in a moment.');
      return;
    }
    setAsking({ question: q, at: new Date().toISOString() });
    try {
      const r = await askCouncil(token, { question: q, context: context.trim() || null, session: sid });
      setQuestion('');
      const s = await getSession(r.session).catch(() => null);
      if (s) setSession(s);
      if (r.session !== sid) setParams({ session: r.session });
      else if (!s) setSessionAttempt((n) => n + 1);
      reloadList();
      showLog();
    } catch (e) {
      if (e instanceof ApiError && e.status === 403) void refreshMe();
      setAskError(askRefusal(e, admin));
    } finally {
      setAsking(null);
    }
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void ask();
  };
  const newSession = () => {
    setAskError(null);
    setParams({});
    box.current?.focus();
  };
  const open = (id: string) => {
    setAskError(null);
    setParams({ session: id });
    showLog();
  };

  // --- deleting a session (an in-page confirmation) ------------------------------------------------------------------
  const [doomed, setDoomed] = useState<SessionSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastMessage | null>(null);
  const confirmDelete = async () => {
    if (!doomed) return;
    const token = csrfToken();
    if (!token) {
      void refreshMe();
      setDeleteError('The form token is not loaded yet - try again in a moment.');
      return;
    }
    setDeleting(true);
    setDeleteError(null);
    try {
      const removed = await deleteSession(token, doomed.id);
      setToast({ id: Date.now(), text: removed ? `Deleted “${doomed.title || 'untitled'}”.` : 'That session was already gone.' });
      if (doomed.id === sid) setParams({});
      setDoomed(null);
      reloadList();
    } catch (e) {
      if (e instanceof ApiError && e.status === 403) void refreshMe();
      setDeleteError(askRefusal(e, admin));
    } finally {
      setDeleting(false);
    }
  };

  useEffect(() => {
    document.title = 'Council · Predictor';
  }, []);

  const list = sessions.kind === 'ok' ? sessions.data : [];
  const title = shown?.title || (sid ? list.find((s) => s.id === sid)?.title : null) || null;
  return (
    <div className="council">
      <PageHead title="Council" sub="one question to several language models at once, weighed by a head model" meta={['Brussels time']} />
      <div className="cgrid">
        <div className="c-ask">
          <Card
            title="Ask the council"
            help="Every member answers the question on its own; the head reads their answers without their names and writes the council's answer. A run takes minutes. It goes on when you close the page, and its answer is kept in your session. Ctrl+Enter sends."
            meta={[sid ? 'in this session' : 'starts a new session']}
          >
            <form className="askform" onSubmit={ask}>
              <label className="field">
                <span>Your question</span>
                <textarea
                  ref={box}
                  rows={3}
                  value={question}
                  maxLength={MAX_QUESTION}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={onKey}
                  placeholder="Ask a question…"
                  aria-describedby="ask-state"
                />
              </label>
              <details className="fold">
                <summary>Context (optional){context.trim() ? <span className="mut"> · {context.trim().length} characters</span> : null}</summary>
                <label className="field">
                  <span>Sent to every member with the question</span>
                  <textarea rows={4} value={context} maxLength={MAX_CONTEXT} onChange={(e) => setContext(e.target.value)} placeholder="Earlier answers, constraints…" />
                  <small>Every member reads the same text, so a wrong premise here misleads all of them at once.</small>
                </label>
              </details>
              <div className="acts">
                <button type="submit" className="btn primary" disabled={!canAsk} aria-busy={asking !== null || undefined}>
                  {asking ? 'Sending…' : 'Ask'}
                </button>
                <button type="button" className="btn" onClick={newSession} disabled={running || !sid}>
                  New session
                </button>
                <span id="ask-state" className="note" role="status">
                  {summon
                    ? `The members have to be summoned: ${summonText(summon)}.`
                    : notSitting
                      ? 'The council is not sitting right now; your earlier sessions can still be read.'
                      : running
                        ? 'Running: this can take several minutes.'
                        : sid
                          ? `Asked in “${title ?? 'this session'}”.`
                          : 'A new session starts with this question.'}
                </span>
              </div>
              {askError ? (
                <p className="ferr" role="alert">
                  {askError}
                </p>
              ) : null}
            </form>
          </Card>
        </div>

        <div className="c-log" ref={log}>
          {sid && sessionError instanceof ApiError && sessionError.status === 404 ? (
            <div className="card">
              <div className="card-b">
                <p className="note">That session is not there any more: it was deleted, or it belongs to another account.</p>
                <div className="acts">
                  <button type="button" className="btn" onClick={newSession}>
                    Start a new session
                  </button>
                </div>
              </div>
            </div>
          ) : sid && sessionError ? (
            <Failed error={sessionError} retry={() => setSessionAttempt((n) => n + 1)} />
          ) : sid && !shown ? (
            sessionBusy ? <Loading /> : null
          ) : null}
          {shown || asking ? (
            <div className="turns">
              {shown?.turns.map((t) => <TurnView key={t.id} turn={t} live={t.jobId ? (live[t.jobId] ?? null) : null} admin={admin} />)}
              {asking ? (
                <article className="card turn">
                  <div className="turn-q">
                    <p>{asking.question}</p>
                    <small>{stamp(asking.at)}</small>
                  </div>
                  <p className="note">Sending the question…</p>
                </article>
              ) : null}
            </div>
          ) : !sid ? (
            <div className="card">
              <div className="card-b empty">Your question and the council&apos;s answer show here.</div>
            </div>
          ) : null}
        </div>

        <div className="c-side">
          <Card title="Your sessions" meta={[sessions.kind === 'ok' ? `${list.length}` : '']} help="Kept for your account only; nobody else can read them. A session starts with its first question." flush>
            {sessions.kind === 'error' ? (
              <div className="card-b">
                <Failed error={sessions.error} retry={sessions.retry} bare />
              </div>
            ) : sessions.kind === 'loading' ? (
              <div className="card-b empty">Loading…</div>
            ) : list.length ? (
              <ul className="slist">
                {list.map((s) => (
                  <li key={s.id} className={s.id === sid ? 'on' : undefined}>
                    <button type="button" className="s-open" onClick={() => open(s.id)} aria-current={s.id === sid ? 'true' : undefined}>
                      <span className="s-t">{s.title || '(untitled)'}</span>
                      <small>
                        {s.turns} {s.turns === 1 ? 'question' : 'questions'} · {s.updated ? day(s.updated) : DASH}
                        {s.pending ? ' · running' : s.lastState === 'failed' ? ' · last run failed' : ''}
                      </small>
                    </button>
                    <button
                      type="button"
                      className="btn s-del"
                      onClick={() => {
                        setDeleteError(null);
                        setDoomed(s);
                      }}
                      disabled={s.pending}
                      aria-label={`Delete the session “${s.title || 'untitled'}”`}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="card-b">
                <p className="note">No sessions yet: your first question starts one.</p>
              </div>
            )}
          </Card>
          {admin ? <PanelCard load={endpoints} /> : null}
        </div>
      </div>
      {doomed ? (
        <Confirm
          title="Delete this session?"
          lines={[`“${doomed.title || 'untitled'}” and its ${doomed.turns} ${doomed.turns === 1 ? 'question' : 'questions'} with their answers are removed for good.`]}
          action="Delete"
          danger
          busy={deleting}
          error={deleteError}
          onConfirm={() => void confirmDelete()}
          onCancel={() => setDoomed(null)}
        />
      ) : null}
      <Toast toast={toast} onClose={() => setToast(null)} />
    </div>
  );
}

function summonText(s: NonNullable<Endpoints['status']>): string {
  const down = s.members.filter((m) => !m.ok).map((m) => m.name);
  if (s.head && !s.head.ok) down.push(s.head.name);
  return down.length ? `not answering: ${down.join(', ')}` : 'no member is reachable';
}

// One question of the session and its answer: from the session's record once the run has ended, from the live job
// while it runs (and in the moment between its end and the record).
function TurnView({ turn, live, admin }: { turn: Turn; live: Live | null; admin: boolean }) {
  const job = live?.job ?? null;
  const result = turn.state === 'done' ? turn.result : job?.state === 'done' ? job.result : null;
  const error = turn.state === 'failed' ? (turn.error ?? 'the run failed') : job?.state === 'failed' ? (job.error ?? 'the run failed') : null;
  return (
    <article className="card turn">
      <div className="turn-q">
        <p>{turn.question}</p>
        <small>{stamp(turn.asked)}</small>
      </div>
      {turn.context ? (
        <details className="fold ctx">
          <summary>Context sent with it</summary>
          <p>{turn.context}</p>
        </details>
      ) : null}
      {result ? (
        <ResultView result={result} admin={admin} />
      ) : error ? (
        <p className="ferr">The run failed: {forReader(error, admin)}</p>
      ) : (
        <ProgressView job={job} lost={live?.lost ?? false} gone={live?.gone ?? false} />
      )}
    </article>
  );
}

// "Member N" as the head numbered the answers it read (it reads them without names, in this order)
function seatLabel(order: string[], name: string): string | null {
  const i = order.indexOf(name);
  return i >= 0 ? `Member ${i + 1}` : null;
}
const secs = (v: number | null) => (v === null ? '' : ` in ${Math.round(v)} s`);

function ResultView({ result, admin }: { result: CouncilResult; admin: boolean }) {
  const head = result.head;
  const order = head?.order ?? [];
  const answered = result.members.filter((m) => m.ok).length;
  return (
    <>
      {head ? (
        <div className="verdict">
          <div className="verdict-h">
            <b>The council&apos;s answer</b>
            <small>
              by the head, {head.name}, from {answered} of {result.members.length} members&apos; answers{secs(head.seconds)}
            </small>
          </div>
          {head.ok ? <Markdown text={head.answer} /> : <p className="ferr">The head did not answer{head.error ? `: ${forReader(head.error, admin)}` : ''}.</p>}
          {head.ok && head.note ? <p className="note">{head.note}</p> : null}
          {head.ok && head.value ? <p className="value">ANSWER: {head.value}</p> : head.ok && head.expectsValue ? <p className="ferr">No “ANSWER:” line in it.</p> : null}
        </div>
      ) : (
        <p className="note">{answered ? 'No head model weighed the answers: the members’ answers are below.' : 'No member answered, so there is no answer.'}</p>
      )}
      {result.members.length ? (
        <details className="fold members">
          <summary>
            The members&apos; answers <span className="mut">· {answered} of {result.members.length} answered</span>
          </summary>
          <div className="mlist">
            {result.members.map((m) => {
              const seat = seatLabel(order, m.name);
              return (
                <details key={m.name} className="member">
                  <summary>
                    <span className="m-h">
                      {seat ? <span className="chip flat">{seat}</span> : null}
                      <b>{m.name}</b>
                      <small>
                        {admin && m.lab ? `${m.lab} · ` : ''}
                        {m.ok ? `${m.cached ? 'from the cache' : 'answered'}${secs(m.seconds)}` : 'no answer'}
                      </small>
                    </span>
                    <span className="m-x">{m.ok ? excerpt(m.answer, 140) : ''}</span>
                  </summary>
                  {m.ok ? <Markdown text={m.answer} /> : <p className="ferr">No answer{m.error && admin ? `: ${m.error}` : ''}.</p>}
                </details>
              );
            })}
          </div>
        </details>
      ) : null}
    </>
  );
}

function ProgressView({ job, lost, gone }: { job: Job | null; lost: boolean; gone: boolean }) {
  if (gone) return <p className="note">The council no longer knows this run; reading how it ended…</p>;
  const p = job?.progress ?? null;
  const done = p ? p.seats.filter((s) => s.state === 'answered' || s.state === 'cached' || s.state === 'failed').length : 0;
  const head = p?.head ?? null;
  return (
    <div className="progress" aria-live="polite">
      <p className="note">
        <b>{p ? (PHASE[p.phase] ?? 'Working…') : job ? 'Waiting for the council to start…' : 'Still running…'}</b>
        {p && p.seats.length ? ` ${done} of ${p.seats.length} members done.` : ''}
        {lost ? ' Lost contact with the server; trying again…' : ''}
      </p>
      {p && p.seats.length ? (
        <ul className="seats">
          {p.seats.map((s) => (
            <li key={s.name} className={s.state}>
              <span className="s-n">
                <b>{s.name}</b>
                <small>
                  {SEAT_WORDS[s.state]}
                  {s.state === 'answered' || s.state === 'cached' ? secs(s.seconds) : ''}
                </small>
              </span>
              {s.state === 'asking' && s.partial ? <span className="m-x">…{s.partial.replace(/\s+/g, ' ').trim().slice(-160)}</span> : null}
              {(s.state === 'answered' || s.state === 'cached') && s.answer ? <span className="m-x">{excerpt(s.answer, 160)}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}
      {head && head.state === 'asking' ? (
        <div className="verdict">
          <div className="verdict-h">
            <b>The council&apos;s answer</b>
            <small>the head, {head.name}, is writing it</small>
          </div>
          {head.partial ? <Markdown text={head.partial} typing /> : <p className="note">Reading the members&apos; answers…</p>}
        </div>
      ) : null}
    </div>
  );
}

// The administrator's: who sits on the council and whether their endpoints answer (api.py /endpoints), folded
function PanelCard({ load }: { load: ReturnType<typeof useLoad<Endpoints | null>> }) {
  const e = load.kind === 'ok' ? load.data : null;
  const s = e?.status ?? null;
  return (
    <div className="card">
      <details className="fold panel">
        <summary>
          Members and endpoints <span className="mut">· admin{s ? ` · ${s.reachable} of ${s.total} members reachable${s.head ? `, head ${s.head.ok ? 'reachable' : 'not answering'}` : ''}` : ''}</span>
        </summary>
        {load.kind === 'loading' ? (
          <p className="note">Loading…</p>
        ) : load.kind === 'error' ? (
          <Failed error={load.error} retry={load.retry} bare />
        ) : e ? (
          <>
            <ul className="plist">
              {e.members.map((m) => {
                const st = s?.members.find((x) => x.name === m.name) ?? null;
                return (
                  <li key={m.name}>
                    <b>{m.name}</b>
                    <small>
                      {[m.lab, m.temperature !== null ? `temperature ${m.temperature}` : '', st ? (st.ok ? 'reachable' : `not answering${st.detail ? `: ${st.detail}` : ''}`) : '']
                        .filter(Boolean)
                        .join(' · ')}
                    </small>
                  </li>
                );
              })}
              {e.head ? (
                <li>
                  <b>Head: {e.head.name}</b>
                  <small>
                    {[e.head.lab, s?.head ? (s.head.ok ? 'reachable' : `not answering${s.head.detail ? `: ${s.head.detail}` : ''}`) : ''].filter(Boolean).join(' · ')}
                  </small>
                </li>
              ) : null}
            </ul>
            <p className="note">
              Preset {e.preset} · members {e.askMembers === 'parallel' ? 'asked all at once' : 'asked one after another'} · replies {e.streams ? 'stream' : 'arrive whole'}
              {s?.checked ? ` · checked ${stamp(s.checked)}` : ''}
            </p>
          </>
        ) : null}
      </details>
    </div>
  );
}
