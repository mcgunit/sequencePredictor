import type { Usual } from '../../api';
import { isNum } from '../../format';

// Two bars drawn in CSS on the tokens: the gate decisions of a set of rows as one stacked bar (replaced, kept,
// reverted), and a run's time against its usual duration. Positions are percentages, so both stretch with their
// column; each carries its numbers in its aria-label and on the line next to it.

const pc = (v: number) => `${Math.max(0, Math.min(100, v)).toFixed(2)}%`;

// replaced / kept / reverted; of: the largest count of the set the bar is compared with (its width is n / of), or
// none for a bar that fills its column end to end
export function StackBar({ replaced, kept, reverted, of, label }: { replaced: number; kept: number; reverted: number; of?: number; label: string }) {
  const n = replaced + kept + reverted;
  const width = of && of > 0 ? Math.max(n ? 2 : 0, (n / of) * 100) : 100;
  const part = (x: number) => (n ? pc((x / n) * 100) : '0%');
  return (
    <div className="sbar" role="img" aria-label={label} style={{ width: pc(width) }}>
      <i className="r" style={{ width: part(replaced) }} />
      <i className="k" style={{ width: part(kept) }} />
      <i className="v" style={{ width: part(reverted) }} />
    </div>
  );
}

// One scale for every run: its time as a multiple of its usual median, 0 to 2.5x - the tick is the median (40%), the
// shaded band the usual range, the faint line 2x. A run longer than the usual maximum is drawn amber; one past 2.5x
// runs to the end. Without an earlier run to compare with, only the base line is drawn.
export const DURATION_SPAN = 2.5;
export function DurationBar({ minutes, usual, label }: { minutes: number | null; usual: Usual | null; label: string }) {
  const median = usual?.median;
  if (!usual || !isNum(median) || median <= 0) {
    return (
      <span className="rbar" role="img" aria-label={label}>
        <i className="base" />
      </span>
    );
  }
  const at = (m: number) => (m / (DURATION_SPAN * median)) * 100;
  const lo = isNum(usual.min) ? at(usual.min) : null;
  const hi = isNum(usual.max) ? at(usual.max) : null;
  const over = isNum(minutes) && isNum(usual.max) && minutes > usual.max;
  return (
    <span className="rbar" role="img" aria-label={label}>
      <i className="base" />
      {lo !== null && hi !== null ? <i className="band" style={{ left: pc(lo), width: pc(Math.max(hi - lo, 0.6)) }} /> : null}
      <i className="g2" />
      {isNum(minutes) ? <i className={over ? 'fill warn' : 'fill'} style={{ width: pc(at(minutes)) }} /> : null}
      <i className="tick" />
    </span>
  );
}
