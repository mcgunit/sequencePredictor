import { useEffect, useRef, type RefObject } from 'react';

const FOCUSABLE = 'button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])';

// The modals open now, the newest last: a confirmation can open over a drawer, and only the top one answers Escape and
// keeps Tab inside it. The page behind stays put (html.locked) while any is open.
const open: object[] = [];

// A modal's keyboard and focus: focus moves into it when it opens (to what `first` names, else the sheet), stays inside
// while it is the top one, and goes back to what opened it when it closes; Escape closes it.
export function useModal(sheet: RefObject<HTMLElement>, onClose: () => void, first?: () => HTMLElement | null | undefined) {
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const me = {};
    open.push(me);
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const root = document.documentElement;
    root.classList.add('locked');
    (first?.() ?? sheet.current)?.focus();
    const onKey = (ev: KeyboardEvent) => {
      if (open[open.length - 1] !== me) return;
      if (ev.key === 'Escape' || ev.key === 'Esc') {
        ev.preventDefault();
        closeRef.current();
        return;
      }
      if (ev.key !== 'Tab' || !sheet.current) return;
      const items = Array.from(sheet.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((n) => !n.hasAttribute('disabled') && n.offsetParent !== null);
      if (!items.length) return;
      const firstItem = items[0];
      const last = items[items.length - 1];
      const inside = sheet.current.contains(document.activeElement);
      if (ev.shiftKey && (document.activeElement === firstItem || !inside)) {
        ev.preventDefault();
        last.focus();
      } else if (!ev.shiftKey && (document.activeElement === last || !inside)) {
        ev.preventDefault();
        firstItem.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      const at = open.indexOf(me);
      if (at >= 0) open.splice(at, 1);
      if (!open.length) root.classList.remove('locked');
      if (opener && document.body.contains(opener)) opener.focus();
    };
    // opened once: what to focus is read when it opens, the close goes through closeRef
  }, []);

  return closeRef;
}
