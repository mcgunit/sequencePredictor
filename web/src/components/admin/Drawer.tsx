import { useId, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { useModal } from '../../hooks/useModal';

// A side sheet over the page (full screen under 640px), for the details that used to hide in hover titles.
// It closes on its button, the scrim and Escape; focus moves into it when it opens, stays inside while it
// is open and goes back to what opened it when it closes (useModal: a confirmation opened over it takes the
// keys until it closes).
export default function Drawer({ title, sub, onClose, children }: { title: ReactNode; sub?: ReactNode; onClose: () => void; children: ReactNode }) {
  const sheet = useRef<HTMLElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  const closeRef = useModal(sheet, onClose, () => closeBtn.current);
  const id = useId();

  return createPortal(
    <>
      <div className="scrim" onClick={() => closeRef.current()} />
      <aside ref={sheet} className="drawer admin" role="dialog" aria-modal="true" aria-labelledby={id}>
        <div className="drawer-h">
          <h2 id={id}>
            {title}
            {sub ? <small>{sub}</small> : null}
          </h2>
          <button ref={closeBtn} type="button" className="xbtn" onClick={() => closeRef.current()}>
            Close ×
          </button>
        </div>
        <div className="drawer-b" tabIndex={0}>
          {children}
        </div>
      </aside>
    </>,
    document.body,
  );
}

// One section of a drawer: a small heading (with chips or a switch beside it) and its lines.
export function Section({ title, aside, children }: { title: ReactNode; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="dsec">
      <h3>
        {title}
        {aside}
      </h3>
      {children}
    </section>
  );
}
