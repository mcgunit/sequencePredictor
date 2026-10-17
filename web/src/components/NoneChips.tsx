import { useState } from 'react';
import type { NoneCall } from '../api';

const FOLD_ABOVE = 6;

// "Not traded today 4/5" followed by a chip per instrument; folded to the count with a
// "show" toggle when there are more than six, so a longer list never runs off the row.
export default function NoneChips({ label, calls, total, title }: { label: string; calls: NoneCall[]; total: number; title?: string }) {
  const [shown, setShown] = useState(false);
  if (!calls.length) return null;
  const foldable = calls.length > FOLD_ABOVE;
  return (
    <div className="none-row">
      <span title={title}>
        {label} <b>{calls.length}{total > 0 ? `/${total}` : ''}</b>
      </span>
      {foldable && !shown ? (
        <button type="button" className="linkbtn" onClick={() => setShown(true)}>
          Show
        </button>
      ) : (
        <>
          {calls.map((c, i) => (
            <span key={`${c.symbol}-${i}`} className={`chip ${c.call === 'down' ? 'down' : 'flat'}`} title={`call ${c.call}`}>
              {c.symbol} {c.call === 'down' ? '▼' : '■'}
            </span>
          ))}
          {foldable ? (
            <button type="button" className="linkbtn" onClick={() => setShown(false)}>
              Hide
            </button>
          ) : null}
        </>
      )}
    </div>
  );
}
