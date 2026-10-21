import { useEffect, useId, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

const FOCUSABLE = 'button, [href], input, select, summary, [tabindex]:not([tabindex="-1"])';

// A side sheet over the page (full screen under 640px), for the details that used to hide in hover titles.
// It closes on its button, the scrim and Escape; focus moves into it when it opens, stays inside while it
// is open and goes back to what opened it when it closes.
export default function Drawer({ title, sub, onClose, children }: { title: ReactNode; sub?: ReactNode; onClose: () => void; children: ReactNode }) {
  const sheet = useRef<HTMLElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const id = useId();

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const root = document.documentElement;
    root.classList.add('locked');
    closeBtn.current?.focus();
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape' || ev.key === 'Esc') {
        ev.preventDefault();
        closeRef.current();
        return;
      }
      if (ev.key !== 'Tab' || !sheet.current) return;
      const items = Array.from(sheet.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((n) => !n.hasAttribute('disabled') && n.offsetParent !== null);
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      const inside = sheet.current.contains(document.activeElement);
      if (ev.shiftKey && (document.activeElement === first || !inside)) {
        ev.preventDefault();
        last.focus();
      } else if (!ev.shiftKey && (document.activeElement === last || !inside)) {
        ev.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      root.classList.remove('locked');
      if (opener && document.body.contains(opener)) opener.focus();
    };
  }, []);

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
