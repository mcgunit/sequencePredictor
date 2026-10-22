import { useId, useRef, type FormEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { useModal } from '../../hooks/useModal';

// The in-page confirmation every action passes through (never window.confirm): what it does and what it costs, in a
// few lines, then the action's own words on the button. A form (Add user, Set password) puts its fields in children;
// Enter sends it. While the request is on its way the button says so; a refusal stays in the dialog, in the server's
// own words, so the reader can correct a field or cancel. Centred on a wide screen, a bottom sheet on a phone. Focus
// starts on the first field, or on Cancel - never on the action. tz: a line holds a time - the dialog is a card of its
// own (on a phone a sheet over the page head), so it says Brussels time itself.
export default function Confirm({
  title,
  lines = [],
  action,
  danger,
  tz,
  busy,
  error,
  disabled,
  onConfirm,
  onCancel,
  children,
}: {
  title: string;
  lines?: (ReactNode | null | false | '')[];
  action: string;
  danger?: boolean;
  tz?: boolean;
  busy: boolean;
  error: string | null;
  disabled?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  children?: ReactNode;
}) {
  const sheet = useRef<HTMLFormElement>(null);
  const cancelBtn = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const bodyId = useId();
  const closeRef = useModal(sheet, onCancel, () => sheet.current?.querySelector<HTMLElement>('input, select, textarea') ?? cancelBtn.current);

  const submit = (ev: FormEvent) => {
    ev.preventDefault();
    if (!busy && !disabled) onConfirm();
  };
  const shown = lines.filter((l) => l !== null && l !== false && l !== '');

  return createPortal(
    <>
      <div className="scrim top" onClick={() => !busy && closeRef.current()} />
      <form ref={sheet} className="dialog" role="alertdialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={bodyId} onSubmit={submit} noValidate>
        <h2 id={titleId}>{title}</h2>
        {tz ? <p className="dialog-tz">Brussels time</p> : null}
        <div id={bodyId} className="dialog-b">
          {shown.map((l, i) => (
            <p key={i}>{l}</p>
          ))}
          {children}
          {error ? (
            <p className="ferr" role="alert">
              {error}
            </p>
          ) : null}
        </div>
        <div className="dialog-f">
          <button ref={cancelBtn} type="button" className="btn" onClick={() => closeRef.current()}>
            {error ? 'Close' : 'Cancel'}
          </button>
          <button type="submit" className={danger ? 'btn danger' : 'btn primary'} disabled={busy || disabled} aria-busy={busy || undefined}>
            {busy ? 'Sending…' : action}
          </button>
        </div>
      </form>
    </>,
    document.body,
  );
}
