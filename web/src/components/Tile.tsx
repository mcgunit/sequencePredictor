import { Link } from 'react-router-dom';

// One KPI tile: label, one figure with its unit, and one or more lines of context under it. The market
// pages' status strip and the admin pages share it. Given a route (to) or an anchor (href) the whole tile
// is a link; a foot line can be marked as a warning (the amber of a run over its usual time).
export interface TileProps {
  label: string;
  value: string;
  cls?: string;
  unit?: string;
  foot: string | (string | { text: string; warn?: boolean })[];
  title?: string;
  to?: string;
  href?: string;
}

export default function Tile({ label, value, cls, unit, foot, title, to, href }: TileProps) {
  const lines = Array.isArray(foot) ? foot.filter((f) => (typeof f === 'string' ? f : f.text)) : [foot];
  const body = (
    <>
      <span className="k">{label}</span>
      <span className={`v ${cls ?? ''}`}>
        {value}
        {unit ? <small>{unit}</small> : null}
      </span>
      {lines.map((f, i) =>
        typeof f === 'string' ? (
          <span key={i} className="f">
            {f}
          </span>
        ) : (
          <span key={i} className={f.warn ? 'f warn' : 'f'}>
            {f.text}
          </span>
        ),
      )}
    </>
  );
  if (to) {
    return (
      <Link className="kpi" to={to} title={title}>
        {body}
      </Link>
    );
  }
  if (href) {
    return (
      <a className="kpi" href={href} title={title}>
        {body}
      </a>
    );
  }
  return (
    <div className="kpi" title={title}>
      {body}
    </div>
  );
}
