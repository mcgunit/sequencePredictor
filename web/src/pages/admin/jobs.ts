import { ACTIONS, type JobLine, type JobsView, type QueueItem, type Running, type ServiceLine, type Usual } from '../../api';
import { isNum, stamp, weekday } from '../../format';
import { durationText, jobLabel, usualRange } from './week';

// Jobs & services' pure helpers: the words of the tiles and of every confirmation. A confirmation says what happens
// and what it costs, in the page's own figures (the job's usual duration, what waits behind it), so the reader never
// has to look elsewhere before pressing the button.

export const POLL_MS = 15 * 1000;

export const MODE_WORD = { on: 'On', dry: 'Dry run', off: 'Off' } as const;
export const MODE_NOTE = {
  on: 'jobs start on their schedule',
  dry: 'the schedule only records what it would start; Run now still runs a job',
  off: 'nothing starts, Run now included',
} as const;

// A service's state in one word, for its tile's figure (the card says it in full).
export const SERVICE_SHORT: Record<string, string> = {
  running: 'Running',
  external: 'Outside',
  restarting: 'Restarting',
  stopped: 'Stopped',
  disabled: 'Off',
  failed: 'Failed',
};

// The supervised services' states as the old page named them.
export const SERVICE_WORD: Record<string, string> = {
  running: 'running',
  external: 'running, started outside',
  restarting: 'restarting',
  stopped: 'stopped',
  disabled: 'autostart off',
  failed: 'failed',
};

// The two weekly plans and the daily job, as the schedule groups them; day: the weekday whose predictor starts the plan
// (0 Sunday … 6 Saturday, jobs.js PLANS).
export const PLAN_GROUPS: { plan: string | null; label: string; name: string; day?: number }[] = [
  { plan: null, label: 'Daily', name: 'the daily run' },
  { plan: 'weeklyTuning', label: 'Saturday · tuning chain', name: "Saturday's tuning chain", day: 6 },
  { plan: 'weeklyControls', label: 'Sunday · controls', name: "Sunday's controls", day: 0 },
];

// When a job of a weekly plan starts: the first when its weekday's predictor finishes (the schedule's own words), each
// other one after the job before it in the plan; with its place in the plan. null for the daily job.
export function afterText(job: JobLine, d: JobsView): { when: string; place: string } | null {
  if (!job.plan) return null;
  const plan = d.jobs.filter((j) => j.plan === job.plan);
  const at = plan.findIndex((j) => j.key === job.key);
  const name = PLAN_GROUPS.find((g) => g.plan === job.plan)?.name ?? 'its plan';
  const before = at > 0 ? plan[at - 1] : null;
  return { when: before ? `after ${jobLabel(before.key, before.name)}` : job.trigger ?? 'with its plan', place: `${at + 1} of ${plan.length} in ${name}` };
}

const s = (n: number) => (n === 1 ? '' : 's');

// "3 earlier runs (job log)": whose runs a usual duration comes from
export function basisText(u: Usual | null): string {
  if (!u) return '';
  const where = u.basis.includes('job log') ? ' (job log)' : u.basis.includes('scheduler') ? ' (scheduler history)' : '';
  return `${u.n} earlier run${s(u.n)}${where}`;
}

// "usually takes 12.1-14.3 h (median 13.0 h of 3 earlier runs)"
export function takesText(u: Usual | null): string {
  const r = usualRange(u);
  if (!u || !r) return 'has no earlier successful run that says how long it takes';
  return `usually takes ${r} (median ${durationText(u.median, u)} of ${u.n} earlier run${s(u.n)})`;
}

// Who holds process.lock, in words: the running job's own lock says nothing new. The schedule starts a job only on a
// free lock and runs one at a time, so a lock its running job does not hold - a pipeline script started by hand or by
// cron included - was not started by it.
export function lockText(lock: JobsView['lock']): string {
  if (!lock) return 'the lock is not known here';
  if (!lock.held) return 'process.lock free';
  if (lock.ours) return 'process.lock held by the running job';
  const who = `${lock.script ?? 'a script'}${lock.pid !== null ? ` (pid ${lock.pid})` : ''}`;
  return `process.lock held by ${who}, not started by the schedule`;
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
// The Brussels weekday and hour of an instant (ms): ['Sat', 14]
function brussels(ms: number): [string, number] {
  const parts = new Intl.DateTimeFormat('en-GB', { weekday: 'short', hour: '2-digit', hour12: false, timeZone: 'Europe/Brussels' }).formatToParts(new Date(ms));
  const part = (t: string) => parts.find((p) => p.type === t)?.value ?? '';
  return [part('weekday'), Number(part('hour')) % 24];
}
// The Brussels calendar day of an instant, "2026-10-24", n days later
function brusselsDay(iso: string, n = 0): string {
  const day = new Intl.DateTimeFormat('en-CA', { year: 'numeric', month: '2-digit', day: '2-digit', timeZone: 'Europe/Brussels' }).format(new Date(iso));
  const [y, m, dd] = day.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, dd + n)).toISOString().slice(0, 10);
}

// Minutes still to go for the running job, by its usual longest run; 0 when it is past it or nothing runs.
function aheadMinutes(r: Running | null): number {
  if (!r || !r.usual || !isNum(r.usual.max) || !isNum(r.minutes)) return 0;
  return Math.max(0, r.usual.max - r.minutes);
}

// The queue a new job joins at the back of (FIFO): the queued jobs' usual longest runs added up, and how many have no
// earlier run to time them by.
function queueAhead(d: JobsView): { minutes: number; untimed: number } {
  let minutes = 0;
  let untimed = 0;
  d.queue.forEach((q) => {
    const u = d.jobs.find((j) => j.key === q.key)?.usual ?? null;
    if (u && isNum(u.max)) minutes += u.max;
    else untimed += 1;
  });
  return { minutes, untimed };
}

// A confirmation: its words, the action's address and body; tz: a line holds a time (the dialog says Brussels time).
export interface Ask {
  title: string;
  lines: string[];
  action: string;
  danger?: boolean;
  tz?: boolean;
  url: string;
  body: Record<string, unknown>;
}

// Run now: how long it takes, where it goes (straight away, or behind what), what waits behind it, and what a
// predictor finishing at the weekend starts. Its start counts the running job's rest and every queued job ahead of it,
// each by its usual longest run.
export function runAsk(job: JobLine, d: JobsView): Ask {
  const label = jobLabel(job.key, job.name);
  const lines: string[] = [`${label} ${takesText(job.usual)}.`];
  const r = d.running;
  const queued = d.queue.length;
  const q = queueAhead(d);
  const behind = queued ? `${queued} queued job${s(queued)}${q.minutes ? ` (up to ${durationText(q.minutes)} by their usual longest runs)` : ''}` : '';
  if (r) {
    const range = usualRange(r.usual, r.minutes);
    lines.push(`It waits behind ${jobLabel(r.key, r.name)}, running ${durationText(r.minutes, r.usual)}${range ? ` of its usual ${range}` : ''}${queued ? `, and ${behind}` : ''}.`);
  } else if (d.lock?.held && !d.lock.ours) {
    lines.push(`It waits until ${d.lock.script ?? 'the script'}${d.lock.pid !== null ? ` (pid ${d.lock.pid})` : ''} lets go of process.lock${queued ? `, then behind ${behind}` : ''}.`);
  } else if (queued) {
    lines.push(`It waits behind ${behind}.`);
  } else {
    lines.push('It starts at once.');
  }
  const ahead = aheadMinutes(r) + q.minutes;
  const timed: string[] = [];
  // one job at a time: the predictor due before this one would end waits behind it
  const predictor = d.jobs.find((j) => j.daily) ?? null;
  const longest = job.usual && isNum(job.usual.max) ? job.usual.max : null;
  if (predictor && predictor.key !== job.key && predictor.nextAt && longest !== null && d.builtAt) {
    const ends = Date.parse(d.builtAt) + (ahead + longest) * 60000;
    if (Date.parse(predictor.nextAt) < ends) timed.push(`The predictor due ${stamp(predictor.nextAt)} would wait behind it: one job runs at a time.`);
  }
  // a predictor finishing on a Saturday (Sunday) after 09:00 starts the week's tuning chain (controls) if it has not run
  // yet (jobs.js planTriggerAt): when this one would finish, by its median
  if (job.daily && d.mode?.mode === 'on' && d.builtAt && job.usual && isNum(job.usual.median)) {
    const finish = Date.parse(d.builtAt) + (ahead + job.usual.median) * 60000;
    const [day, hour] = brussels(finish);
    const plan = hour >= 9 ? (day === 'Sat' ? 'weeklyTuning' : day === 'Sun' ? 'weeklyControls' : null) : null;
    if (plan) {
      const jobs = d.jobs.filter((j) => j.plan === plan);
      const hours = jobs.reduce((sum, j) => sum + (j.usual && isNum(j.usual.median) ? j.usual.median : 0), 0) / 60;
      const what = plan === 'weeklyTuning' ? "this week's tuning chain" : "this week's Sunday controls";
      timed.push(`Finishing about ${stamp(new Date(finish).toISOString())}, it starts ${what} if it has not run yet (${jobs.length} jobs${hours >= 1 ? `, about ${Math.round(hours)} h by their medians` : ''}).`);
    }
  }
  lines.push(...timed);
  if (timed.length && q.untimed) lines.push(`${q.untimed} queued job${s(q.untimed)} ${q.untimed === 1 ? 'has' : 'have'} no earlier run to time: these times leave ${q.untimed === 1 ? 'it' : 'them'} out.`);
  if (d.mode?.mode === 'dry') lines.push('The schedule is in dry run, but Run now really runs it.');
  return { title: `Run ${label} now?`, lines, action: 'Run now', tz: timed.length > 0, url: ACTIONS.run, body: { key: job.key } };
}

// Stop: how far it got, that nothing resumes it, and what starts next.
export function stopAsk(r: Running, d: JobsView): Ask {
  const label = jobLabel(r.key, r.name);
  const range = usualRange(r.usual, r.minutes);
  const lines = [
    `It has run ${durationText(r.minutes, r.usual)}${range ? ` of its usual ${range}` : ''}, since ${stamp(r.start)}.`,
    'Its whole process group gets SIGTERM. Nothing resumes it: what it has not finished waits for its next run.',
  ];
  if (d.queue.length) lines.push(`Then the queue moves on: ${d.queue.map((q) => jobLabel(q.key, q.name)).join(', ')}.`);
  return { title: `Stop ${label}?`, lines, action: 'Stop it', danger: true, tz: true, url: ACTIONS.stop, body: { runId: r.runId } };
}

// Cancel: when it was queued and why, and what it costs. The schedule remembers an occurrence when it queues it, so a
// job it queued is not queued again for that slot: the day's predictions are not made, and a weekly job waits for next
// week's plan. A job queued by Run now leaves its scheduled runs as they were.
export function cancelAsk(q: QueueItem, d: JobsView): Ask {
  const label = jobLabel(q.key, q.name);
  const job = d.jobs.find((j) => j.key === q.key) ?? null;
  const plan = PLAN_GROUPS.find((g) => g.plan !== null && g.plan === job?.plan) ?? null;
  let cost: string;
  if (q.manual) {
    cost = 'It was queued by Run now: its scheduled runs stay as they are.';
  } else if (job?.daily) {
    const day = q.queuedAt ? brusselsDay(q.queuedAt) : null;
    const whose = day && d.builtAt && day === brusselsDay(d.builtAt) ? "Today's" : day ? `${weekday(day)}'s` : 'These';
    cost = `${whose} predictions are not made unless you use Run now${job.nextAt ? `; the next slot is ${stamp(job.nextAt)}` : ''}.`;
  } else if (plan && isNum(plan.day) && d.builtAt) {
    const [today] = brussels(Date.parse(d.builtAt));
    const days = (plan.day - WEEKDAYS.indexOf(today) + 7) % 7 || 7;
    cost = `This week's run is skipped: next on ${weekday(brusselsDay(d.builtAt, days))}, when that day's predictor finishes (or by Run now).`;
  } else {
    cost = `It runs again ${job?.trigger ?? 'with its plan next week'}, or by Run now.`;
  }
  return {
    title: `Take ${label} out of the queue?`,
    lines: [`Queued ${stamp(q.queuedAt)}${q.reason ? ` · ${q.reason}` : ''} · waiting ${durationText(q.waitedMinutes)}.`, cost, 'The rest of the queue stays.'],
    action: 'Take it out',
    tz: true,
    url: ACTIONS.cancel,
    body: { key: q.key },
  };
}

// Start, restart, stop a service: what the reader of its page sees meanwhile.
export function serviceAsk(sv: ServiceLine, action: 'start' | 'stop' | 'restart'): Ask {
  const council = sv.key === 'councilApi';
  const port = sv.port !== null ? ` on port ${sv.port}` : '';
  const up = isNum(sv.upMinutes) ? `Up ${durationText(sv.upMinutes)}, since ${stamp(sv.since)}.` : '';
  const url = ACTIONS.service(sv.key, action);
  if (action === 'start') {
    return { title: `Start the ${sv.name}?`, lines: [`The server starts it${port} and restarts it when it crashes.`], action: 'Start', url, body: {} };
  }
  if (action === 'restart') {
    return {
      title: `Restart the ${sv.name}?`,
      lines: [up, `It stops and starts again: an answer it is giving now is cut off${council ? ', and the Council page gets no answer for a few seconds' : ''}.`].filter(Boolean),
      action: 'Restart',
      tz: Boolean(up),
      url,
      body: {},
    };
  }
  return {
    title: `Stop the ${sv.name}?`,
    lines: [up, `Nothing answers${port} until it is started here again or the server restarts${council ? ': the Council page cannot reach it' : ''}.`].filter(Boolean),
    action: 'Stop it',
    danger: true,
    tz: Boolean(up),
    url,
    body: {},
  };
}

// The runs kept (or the ones given: a job's), from the oldest start to the newest: the window a "none in 40 runs" is over.
export function runsWindow(d: JobsView, items = d.runs.items): { from: string | null; to: string | null } {
  const starts = items.filter((x) => x.kind === 'run').map((x) => x.start).filter((x): x is string => Boolean(x)).sort();
  return { from: starts[0] ?? null, to: starts[starts.length - 1] ?? null };
}

// "Wed 21 Oct (Brussels day)"
export const dayText = (day: string | null) => (day ? weekday(day) : '');

// A run id's job and start (<job>-<epoch ms>), for a run the page no longer lists.
export function fromRunId(id: string): { key: string; start: string | null } {
  const m = /^([A-Za-z]+)-(\d{10,16})$/.exec(id);
  if (!m) return { key: '', start: null };
  const ms = Number(m[2]);
  return { key: m[1], start: Number.isFinite(ms) ? new Date(ms).toISOString() : null };
}
