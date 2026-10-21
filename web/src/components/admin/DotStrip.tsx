// Values on one axis, drawn to scale as inline SVG: positions are percentages of the strip's width, so the
// strip stretches with its column while the dots stay round. Every colour and size comes from the CSS tokens
// (styles.css .strip / .axis), so the strips follow light and dark mode.

export type MarkKind = 'dot' | 'hollow' | 'small' | 'smallHollow' | 'null' | 'diamond' | 'tick' | 'tickOwn';
export interface Mark {
  at: number | null;
  kind: MarkKind;
}
// zone: a shaded part of the axis (the null band); se: a thin whisker (±1 sd, ±2 sd); ci: a thicker interval
export interface SpanMark {
  from: number | null;
  to: number | null;
  kind: 'zone' | 'se' | 'ci';
}

const H = 22;
const MID = H / 2;

export interface Scale {
  min: number;
  max: number;
}

export function pos(v: number, s: Scale): number {
  const span = s.max - s.min;
  if (!(span > 0)) return 50;
  return Math.max(0, Math.min(100, ((v - s.min) / span) * 100));
}
const pc = (v: number) => `${v.toFixed(2)}%`;

export default function DotStrip({
  scale,
  grid = [],
  spans = [],
  marks = [],
  label,
  dim,
}: {
  scale: Scale;
  grid?: number[];
  spans?: SpanMark[];
  marks?: Mark[];
  label: string;
  dim?: boolean;
}) {
  return (
    <svg className={`strip${dim ? ' dim' : ''}`} role="img" aria-label={label} width="100%" height={H} overflow="visible">
      {grid.map((g, i) => (
        <line key={`g${i}`} className={g === 0 ? 'gl zero' : 'gl'} x1={pc(pos(g, scale))} x2={pc(pos(g, scale))} y1={0} y2={H} />
      ))}
      {spans.map((s, i) => {
        if (s.from === null || s.to === null) return null;
        const a = pos(Math.min(s.from, s.to), scale);
        const b = pos(Math.max(s.from, s.to), scale);
        const w = pc(Math.max(b - a, 0.4));
        if (s.kind === 'zone') return <rect key={`s${i}`} className="zone" x={pc(a)} width={w} y={4} height={H - 8} rx={3} />;
        if (s.kind === 'ci') return <rect key={`s${i}`} className="ci" x={pc(a)} width={w} y={MID - 2} height={4} rx={2} />;
        return <rect key={`s${i}`} className="se" x={pc(a)} width={w} y={MID - 1} height={2} />;
      })}
      {marks.map((m, i) => {
        if (m.at === null) return null;
        const x = pc(pos(m.at, scale));
        switch (m.kind) {
          case 'tick':
          case 'tickOwn':
            return <line key={`m${i}`} className={m.kind === 'tick' ? 'tk' : 'tk own'} x1={x} x2={x} y1={1} y2={H - 1} />;
          case 'hollow':
            return <circle key={`m${i}`} className="dot o" cx={x} cy={MID} r={4} />;
          case 'small':
            return <circle key={`m${i}`} className="dot" cx={x} cy={MID} r={3.5} />;
          case 'smallHollow':
            return <circle key={`m${i}`} className="dot o" cx={x} cy={MID} r={3} />;
          case 'null':
            return <circle key={`m${i}`} className="dot nul" cx={x} cy={MID} r={3.5} />;
          case 'diamond':
            // a nested svg puts the diamond's own origin on the value, whatever the strip's width
            return (
              <svg key={`m${i}`} x={x} y={MID} overflow="visible">
                <polygon className="dia" points="0,-5 5,0 0,5 -5,0" />
              </svg>
            );
          default:
            return <circle key={`m${i}`} className="dot" cx={x} cy={MID} r={5} />;
        }
      })}
    </svg>
  );
}

// The axis under (or over) a column of strips: labelled ticks on the same scale.
export function Axis({ scale, ticks }: { scale: Scale; ticks: { at: number; label: string }[] }) {
  return (
    <svg className="axis" width="100%" height={14} overflow="visible" aria-hidden="true">
      {ticks.map((t, i) => {
        const p = pos(t.at, scale);
        const anchor = p <= 0.5 ? 'start' : p >= 99.5 ? 'end' : 'middle';
        return (
          <text key={i} x={pc(p)} y={11} textAnchor={anchor}>
            {t.label}
          </text>
        );
      })}
    </svg>
  );
}

// Round tick values from min to max at a step (floating point kept off the labels).
export function ticks(min: number, max: number, step: number, label: (v: number) => string): { at: number; label: string }[] {
  const out: { at: number; label: string }[] = [];
  const first = Math.ceil(min / step - 1e-9) * step;
  for (let v = first; v <= max + 1e-9; v += step) {
    const r = Number(v.toFixed(6));
    out.push({ at: r, label: label(r) });
  }
  return out;
}

// A scale around a set of values with a little room at both ends, and round ticks on it (about five).
export function niceScale(values: (number | null | undefined)[], pad = 0.08, target = 5): { scale: Scale; ticks: { at: number; label: string }[] } {
  const v = values.filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
  const lo = v.length ? Math.min(...v) : 0;
  const hi = v.length ? Math.max(...v) : 1;
  const width = hi - lo || Math.abs(hi) || 1;
  const scale = { min: lo - width * pad, max: hi + width * pad };
  const raw = (scale.max - scale.min) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return { scale, ticks: ticks(scale.min, scale.max, step, (x) => x.toFixed(decimals).replace('-', '−')) };
}
