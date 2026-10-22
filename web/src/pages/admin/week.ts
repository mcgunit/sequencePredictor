import type { MatrixGame, RunLine, Usual, WeekView } from '../../api';
import { DASH, isNum } from '../../format';

// This week's pure helpers: the jobs' short names, durations, what a run wrote, and the sidebar's status dot.

// The scheduler jobs (jobs.js JOBS) as the runs card names them; an unknown job keeps its own name.
const JOB_LABEL: Record<string, string> = {
  predictor: 'Predictor',
  hyperoptStatistics: 'Tuning: statistical',
  hyperoptBoost: 'Tuning: boosting',
  hyperoptRLTicket: 'Tuning: RL ticket',
  hyperoptEnsemble: 'Tuning: ensemble',
  hyperoptQuantum: 'Tuning: quantum',
  trainMetaLearner: 'Retrain: meta-learner',
  randomnessDiscrimination: 'Control: Q2',
  featureControl: 'Control: feature (FC)',
  marketRows: 'Control: market rows',
  nullControls: 'Control: Q0',
};
export const jobLabel = (key: string, name?: string) => JOB_LABEL[key] ?? name ?? key;

// The control a Sunday job writes, as the matrix names its parts.
export type Part = 'q0' | 'q2' | 'fc' | 'rows';
export const CONTROL_PART: Record<string, Part> = { nullControls: 'q0', randomnessDiscrimination: 'q2', featureControl: 'fc', marketRows: 'rows' };
export const PART_LABEL: Record<Part, string> = { q0: 'Q0', q2: 'Q2', fc: 'Feature control', rows: 'Market rows' };

// A run's address in the URL (?run=): its job and its start, which no two runs share.
export const runId = (r: { key: string; start: string | null }) => `${r.key}@${r.start ?? ''}`;

// Minutes in the unit a reader takes in: "2.9 min", "43 min", "9.9 h"; inHours forces hours (a range read with them).
function minutes(m: number, inHours: boolean): string {
  if (inHours) return (m / 60).toFixed(1);
  return m < 10 ? m.toFixed(1) : String(Math.round(m));
}
export function durationText(m: number | null, usual?: Usual | null): string {
  if (!isNum(m)) return DASH;
  const hours = Math.max(m, usual && isNum(usual.max) ? usual.max : 0) >= 60;
  return `${minutes(m, hours)} ${hours ? 'h' : 'min'}`;
}
// "7.7-8.2 h": the usual range in the run's own unit; '' without one
export function usualRange(u: Usual | null, m: number | null = null): string {
  if (!u || !isNum(u.min) || !isNum(u.max)) return '';
  const hours = Math.max(isNum(m) ? m : 0, u.max) >= 60;
  const lo = minutes(u.min, hours);
  const hi = minutes(u.max, hours);
  return `${lo === hi ? lo : `${lo}-${hi}`} ${hours ? 'h' : 'min'}`;
}
// "usual 7.7-8.2 h" in the run's own unit
export function usualText(u: Usual | null, m: number | null): string {
  const r = usualRange(u, m);
  return r ? `usual ${r}` : 'no earlier run to compare';
}
// "3 earlier runs (job log)" / "12 earlier runs (scheduler history)"
export function usualBasis(u: Usual | null): string {
  if (!u) return '';
  const where = u.basis.includes('job log') ? 'job log' : u.basis.includes('scheduler') ? 'scheduler history' : u.basis;
  return `${u.n} earlier run${u.n === 1 ? '' : 's'} (${where})`;
}

// What a Sunday control run wrote, read from the records on disk now: a game's record whose time lies inside the
// run (a minute of slack) was written by it; one from before its start was not (the job skipped or failed that game,
// so last week's record is still there); one after its end was rewritten since (a later run, by hand). A game with
// no record at all, or one that could not be read, is none. Q0 writes two records a game (synthetic, shuffled): each
// is judged, and a game's line is the record that is not from the run, named by its mode - a band built from a new
// record and a stale one is not this run's.
const SLACK_MS = 60 * 1000;
type How = 'this run' | 'before this run' | 'after this run' | 'no record' | 'unreadable';
export interface Written {
  game: string;
  at: string | null;
  how: How;
  mode?: string; // Q0: the control whose record this line is, when it is not from this run
}
export function controlOutputs(run: RunLine, games: MatrixGame[], builtAt: string | null): { part: Part; lines: Written[] } | null {
  const part = CONTROL_PART[run.key];
  if (!part) return null;
  const start = run.start ? Date.parse(run.start) : NaN;
  const end = run.end ? Date.parse(run.end) : builtAt ? Date.parse(builtAt) : NaN;
  const howOf = (at: string): How => {
    const ms = Date.parse(at);
    if (Number.isFinite(start) && ms < start - SLACK_MS) return 'before this run';
    if (Number.isFinite(end) && ms > end + SLACK_MS) return 'after this run';
    return 'this run';
  };
  const lines = games
    .filter((g) => part !== 'rows' || g.group === 'Markets')
    .map((g): Written => {
      const cell = g[part];
      const at = cell ? cell.at : null;
      if (!at) return { game: g.game, at: null, how: g.errors.includes(part) ? 'unreadable' : 'no record' };
      const modes = part === 'q0' && g.q0 ? g.q0.modes.filter((m): m is { mode: string; at: string } => Boolean(m.at)) : [];
      const off = modes.map((m) => ({ ...m, how: howOf(m.at) })).sort((a, b) => Number(b.how === 'before this run') - Number(a.how === 'before this run')).find((m) => m.how !== 'this run');
      if (off) return { game: g.game, at: off.at, how: off.how, mode: off.mode };
      return { game: g.game, at, how: howOf(at) };
    });
  return { part, lines };
}

// The sidebar's dot and the page's "needs a look" line: red when a run failed or an output is missing, amber when a
// run has a warning signature none of its job's earlier runs had. A run not started yet is missing only once the
// weekend's window has closed (Tuesday 09:00): until then it may simply not be due.
export interface StatusReason {
  level: 'red' | 'amber';
  text: string;
  run?: string; // the run it is about (its runId), for a link to its drawer
}
export interface WeekStatus {
  level: 'red' | 'amber' | null;
  reasons: StatusReason[];
}
export function weekStatus(w: WeekView): WeekStatus {
  const reasons: StatusReason[] = [];
  const seen = new Set<string>();
  const all = [...w.runs, ...w.since.predictorRuns, ...(w.tiles.today.run ? [w.tiles.today.run] : [])].filter((r) => {
    const id = runId(r);
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
  all.filter((r) => r.result !== 'ok').forEach((r) => reasons.push({ level: 'red', text: `${jobLabel(r.key, r.job)} ${r.result}`, run: runId(r) }));
  const closed = Boolean(w.weekend.until && w.builtAt && Date.parse(w.builtAt) >= Date.parse(w.weekend.until));
  if (closed && w.tiles.weekend.missing.length) {
    reasons.push({ level: 'red', text: `no run: ${w.tiles.weekend.missing.map((m) => jobLabel(m.key)).join(', ')}` });
  }
  w.runs
    .filter((r) => r.result === 'ok' && r.end && CONTROL_PART[r.key])
    .forEach((r) => {
      const out = controlOutputs(r, w.matrix.games, w.builtAt);
      const lacking = out ? out.lines.filter((l) => l.how === 'before this run' || l.how === 'no record' || l.how === 'unreadable') : [];
      if (lacking.length) reasons.push({ level: 'red', text: `${jobLabel(r.key)}: no record from this run for ${lacking.map((l) => (l.mode ? `${l.game} ${l.mode}` : l.game)).join(', ')}`, run: runId(r) });
    });
  all.forEach((r) =>
    (r.warnings ?? []).filter((x) => x.new === true).forEach((x) => reasons.push({ level: 'amber', text: `new warning: ${jobLabel(r.key, r.job)} ${x.sig} ×${x.n}`, run: runId(r) })),
  );
  const level = reasons.some((r) => r.level === 'red') ? 'red' : reasons.length ? 'amber' : null;
  return { level, reasons };
}
