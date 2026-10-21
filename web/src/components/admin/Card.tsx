import type { ReactNode } from 'react';

// "?" next to a card's title: the one place a card explains itself (a native details, no hover).
export function Help({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="help">
      <summary aria-label={`How to read ${title}`}>?</summary>
      <p>{children}</p>
    </details>
  );
}

// Meta items, separated by dots; empty ones are left out.
export function Meta({ items }: { items: (ReactNode | null | undefined | false | '')[] }) {
  const shown = items.filter((x) => x !== null && x !== undefined && x !== false && x !== '');
  if (!shown.length) return null;
  return (
    <div className="meta dots">
      {shown.map((x, i) => (
        <span key={i}>{x}</span>
      ))}
    </div>
  );
}

// A card with one header (title, "?", meta), a body and an optional footer, as on the market pages.
export function Card({
  id,
  title,
  help,
  meta,
  head,
  foot,
  className,
  flush,
  children,
}: {
  id?: string;
  title: ReactNode;
  help?: ReactNode;
  meta?: (ReactNode | null | undefined | false | '')[];
  head?: ReactNode; // extra header content (a switch)
  foot?: ReactNode;
  className?: string;
  flush?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={`card${className ? ` ${className}` : ''}`} id={id}>
      <div className="card-h">
        <h2>{title}</h2>
        {help ? <Help title={typeof title === 'string' ? title : 'this card'}>{help}</Help> : null}
        {meta ? <Meta items={meta} /> : null}
        {head}
      </div>
      <div className={flush ? 'card-b flush' : 'card-b'}>{children}</div>
      {foot ? <div className="card-f">{foot}</div> : null}
    </div>
  );
}

// A wide table keeps its own sideways scroll, never the page's.
export function Scroll({ children }: { children: ReactNode }) {
  return <div className="scroll">{children}</div>;
}

// The legend under a chart: each mark drawn as the strip draws it.
export function Legend({ items }: { items: { mark: string; label: ReactNode }[] }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <span key={i} className="swatch">
          <i className={it.mark} />
          {it.label}
        </span>
      ))}
    </div>
  );
}
