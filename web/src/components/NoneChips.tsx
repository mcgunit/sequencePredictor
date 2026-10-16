import { useState } from 'react';
import type { NoneCall } from '../api';

const FOLD_ABOVE = 6;

// "No day position today: 4 of 5" followed by a chip per instrument; folded to
// the count with a "show" toggle when there are more than six.
export default function NoneChips({ label, calls, total }: { label: string; calls: NoneCall[]; total: number }) {
  const [shown, setShown] = useState(false);
  if (!calls.length) return null;
  const foldable = calls.length > FOLD_ABOVE;
  return (
    <div className="none-row">
      <span>
        {label}: <b>{calls.length}{total > 0 ? ` of ${total}` : ''}</b>
      </span>
      {foldable && !shown ? (
        <button type="button" className="linkbtn" onClick={() => setShown(true)}>
          show
        </button>
      ) : (
        <>
          {calls.map((c, i) => (
            <span key={`${c.symbol}-${i}`} className={`chip ${c.call === 'down' ? 'down' : 'flat'}`}>
              {c.symbol} {c.call === 'down' ? '▼' : '■'}
            </span>
          ))}
          {foldable ? (
            <button type="button" className="linkbtn" onClick={() => setShown(false)}>
              hide
            </button>
          ) : null}
        </>
      )}
    </div>
  );
}
