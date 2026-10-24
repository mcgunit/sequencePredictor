// The Council's JSON routes (council.js): every signed-in reader asks the council and reads their own sessions; who sits on
// the council and how to reach them (/council/api/endpoints) is the administrator's. A run is answered only to its asker.
// The answers are the routes' own (not system.js's {ok, message}), so the actions here send the form token themselves.
// Typed and normalised like api.ts: every container filled, every figure a number or null.
import { ApiError, getJson } from './api';

type Raw = Record<string, unknown>;
const o = (v: unknown): Raw => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Raw) : {});
const has = (v: unknown): boolean => Boolean(v) && typeof v === 'object' && !Array.isArray(v);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const opt = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const num = (v: unknown, fallback: number): number => opt(v) ?? fallback;
const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const ostr = (v: unknown): string | null => (typeof v === 'string' && v ? v : null);
const strs = (v: unknown): string[] => arr(v).filter((x): x is string => typeof x === 'string');

export const SESSION_ID = /^[a-f0-9]{12}$/; // councilSessions.js ID
const JOB_ID = /^[a-f0-9]{1,32}$/;
export const MAX_QUESTION = 4000; // api.py MAX_QUESTION
export const MAX_CONTEXT = 20000;

// --- a run's answer (what run_council returns) --------------------------------------------------------------------------
export interface MemberAnswer {
  name: string;
  lab: string | null;
  ok: boolean;
  cached: boolean;
  seconds: number | null;
  answer: string;
  error: string | null;
}
export interface HeadAnswer {
  name: string;
  ok: boolean;
  answer: string;
  value: string | null; // the "ANSWER:" line of a preset that asks for one
  expectsValue: boolean;
  note: string | null; // what the orchestrator had to do to get the answer
  error: string | null;
  order: string[]; // the answered members as the head read them: "Member 1" is order[0]
  seconds: number | null;
}
export interface CouncilResult {
  members: MemberAnswer[];
  head: HeadAnswer | null;
}
function headOf(v: unknown): HeadAnswer | null {
  if (!has(v)) return null;
  const h = o(v);
  return {
    name: str(h.name, 'head'), ok: h.ok !== false && h.state !== 'failed', answer: str(h.answer), value: ostr(h.value), expectsValue: h.expects_value === true,
    note: ostr(h.note), error: ostr(h.error), order: strs(h.order), seconds: opt(h.seconds),
  };
}
function resultOf(v: unknown): CouncilResult | null {
  if (!has(v)) return null;
  const r = o(v);
  return {
    members: arr(r.members).filter(has).map((x) => {
      const m = o(x);
      return { name: str(m.name), lab: ostr(m.lab), ok: m.ok !== false, cached: m.cached === true, seconds: opt(m.seconds), answer: str(m.answer), error: ostr(m.error) };
    }),
    head: headOf(r.head),
  };
}

// --- sessions (councilSessions.js) -------------------------------------------------------------------------------------
export type TurnState = 'pending' | 'done' | 'failed';
export interface SessionSummary {
  id: string;
  title: string;
  turns: number;
  created: string | null;
  updated: string | null;
  pending: boolean;
  lastState: TurnState | null;
}
export interface Turn {
  id: string;
  jobId: string | null;
  question: string;
  context: string | null;
  state: TurnState;
  asked: string | null;
  finished: string | null;
  result: CouncilResult | null;
  error: string | null;
}
export interface Session {
  id: string;
  title: string;
  created: string | null;
  updated: string | null;
  turns: Turn[];
}
const turnState = (v: unknown): TurnState => (v === 'done' || v === 'failed' ? v : 'pending');
function summaryOf(v: unknown): SessionSummary {
  const s = o(v);
  return {
    id: str(s.id), title: str(s.title), turns: num(s.turns, 0), created: ostr(s.created), updated: ostr(s.updated), pending: s.pending === true,
    lastState: s.lastState === null || s.lastState === undefined ? null : turnState(s.lastState),
  };
}
function sessionOf(v: unknown): Session {
  const s = o(v);
  return {
    id: str(s.id), title: str(s.title), created: ostr(s.created), updated: ostr(s.updated),
    turns: arr(s.turns).filter(has).map((x) => {
      const t = o(x);
      return {
        id: str(t.id), jobId: ostr(t.jobId), question: str(t.question), context: ostr(t.context), state: turnState(t.state), asked: ostr(t.asked),
        finished: ostr(t.finished), result: resultOf(t.result), error: ostr(t.error),
      };
    }),
  };
}

// --- a running job (api.py: the seats as the members answer) -----------------------------------------------------------
export type SeatState = 'waiting' | 'asking' | 'answered' | 'cached' | 'failed';
const SEATS: SeatState[] = ['waiting', 'asking', 'answered', 'cached', 'failed'];
export interface Seat {
  name: string;
  state: SeatState;
  partial: string | null; // the reply so far, while it streams
  answer: string | null;
  seconds: number | null;
  error: string | null;
}
export interface Progress {
  phase: 'starting' | 'members' | 'head' | 'done' | string;
  seats: Seat[];
  head: (Seat & { order: string[] }) | null;
}
export interface Job {
  id: string;
  state: 'pending' | 'running' | 'done' | 'failed';
  progress: Progress | null;
  result: CouncilResult | null;
  error: string | null;
}
const seatState = (v: unknown): SeatState => (SEATS.includes(v as SeatState) ? (v as SeatState) : 'waiting');
function seatOf(name: string, v: unknown): Seat {
  const s = o(v);
  return { name, state: seatState(s.state), partial: ostr(s.partial), answer: ostr(s.answer), seconds: opt(s.seconds), error: ostr(s.error) };
}
function jobOf(v: unknown, id: string): Job {
  const j = o(v);
  const p = o(j.progress);
  const state = j.state === 'done' || j.state === 'failed' || j.state === 'running' ? j.state : 'pending';
  return {
    id: str(j.id, id), state,
    progress: has(j.progress)
      ? {
          phase: str(p.phase, 'starting'),
          seats: Object.entries(o(p.seats)).map(([name, s]) => seatOf(name, s)),
          head: has(p.head) ? { ...seatOf(str(o(p.head).name, 'head'), p.head), order: strs(o(p.head).order) } : null,
        }
      : null,
    result: resultOf(j.result),
    error: ostr(j.error),
  };
}

// --- who sits on the council (the administrator's) ----------------------------------------------------------------------
export interface EndpointState {
  name: string;
  ok: boolean;
  detail: string | null;
}
export interface Endpoints {
  members: { name: string; lab: string | null; temperature: number | null }[];
  head: { name: string; lab: string | null } | null;
  preset: string;
  askMembers: string; // 'parallel' | 'sequential'
  streams: boolean;
  status: { checked: string | null; ready: boolean; reachable: number; total: number; members: EndpointState[]; head: EndpointState | null } | null;
}
const endpointOf = (v: unknown): EndpointState => ({ name: str(o(v).name), ok: o(v).ok === true, detail: ostr(o(v).detail) });
function endpointsOf(v: unknown): Endpoints {
  const e = o(v);
  const s = o(e.status);
  return {
    members: arr(e.members).filter(has).map((x) => ({ name: str(o(x).name), lab: ostr(o(x).lab), temperature: opt(o(x).temperature) })),
    head: has(e.head) ? { name: str(o(e.head).name), lab: ostr(o(e.head).lab) } : null,
    preset: str(e.preset, 'default'), askMembers: str(e.ask_members), streams: e.stream_answers !== false,
    status: has(e.status)
      ? { checked: ostr(s.checked), ready: s.ready === true, reachable: num(s.reachable, 0), total: num(s.total, 0),
          members: arr(s.members).filter(has).map(endpointOf), head: has(s.head) ? endpointOf(s.head) : null }
      : null,
  };
}

const enc = encodeURIComponent;
export const getSessions = async (): Promise<SessionSummary[]> => arr(o(await getJson<unknown>('/council/api/sessions')).sessions).filter(has).map(summaryOf);
export const getSession = async (id: string): Promise<Session> => sessionOf(o(await getJson<unknown>(`/council/api/sessions/${enc(id)}`)).session);
export const getJob = async (id: string): Promise<Job> => {
  if (!JOB_ID.test(id)) throw new ApiError(`/council/api/job/${id}`, 400, 'bad job id');
  return jobOf(await getJson<unknown>(`/council/api/job/${enc(id)}`), id);
};
export const getEndpoints = async (): Promise<Endpoints> => endpointsOf(await getJson<unknown>('/council/api/endpoints'));

// Whether the council can sit now, for every reader (/council/api/status: api.py /health's verdict without who or where;
// the members and their endpoints stay the administrator's)
export interface CouncilStatus {
  ready: boolean; // at least one member and the head answer
  busy: boolean; // another question is being answered
  apiError: string | null; // the administrator's only: why the Council API itself did not answer (its /endpoints is then a 503)
}
export const getStatus = async (): Promise<CouncilStatus> => {
  const s = o(await getJson<unknown>('/council/api/status'));
  return { ready: s.ready === true, busy: s.busy === true, apiError: ostr(s.apiError) };
};

// A council POST with the session's form token as _csrf. A refusal is an ApiError in the server's own words; a run the
// council cannot take because another is running (api.py 409 {busy}) carries code 'busy'.
async function post(url: string, csrf: string, body: Record<string, unknown>): Promise<Raw> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ ...body, _csrf: csrf }),
    });
  } catch (e) {
    throw new ApiError(url, null, `Could not reach the server (${(e as Error).message || 'network error'}). Nothing was sent.`);
  }
  if (res.status === 401) {
    window.location.assign(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
    throw new ApiError(url, 401, 'signed out - going to the login page');
  }
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  const answer = o(data);
  if (!res.ok) {
    const busy = answer.busy === true;
    const message = typeof answer.error === 'string' && answer.error ? answer.error : `The server answered ${res.status} ${res.statusText}`.trim();
    throw new ApiError(url, res.status, message, busy ? 'busy' : null, message);
  }
  return answer;
}

// Ask: the run is filed in the caller's session (a new one when none is given); 202 {id, state, session, turn}
export async function askCouncil(csrf: string, q: { question: string; context: string | null; session: string | null }): Promise<{ job: string; session: string; turn: string }> {
  const a = await post('/council/api/ask', csrf, { question: q.question, context: q.context, session: q.session });
  return { job: str(a.id), session: str(a.session), turn: str(a.turn) };
}
// Delete one of the caller's sessions: removed false when it was not there (already gone, or not theirs)
export async function deleteSession(csrf: string, id: string): Promise<boolean> {
  const a = await post(`/council/api/sessions/${enc(id)}/delete`, csrf, {});
  return a.removed === true;
}
