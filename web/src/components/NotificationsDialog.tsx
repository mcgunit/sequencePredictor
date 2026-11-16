import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link, useLocation } from 'react-router-dom';
import { ACTIONS, type Alert } from '../api';
import { stamp } from '../format';
import { useMe } from '../hooks/useMe';
import { useModal } from '../hooks/useModal';
import { useNotifications } from '../hooks/useNotifications';
import { useSend } from '../hooks/useSend';
import { useWhatsNew } from '../hooks/useWhatsNew';
import Chip from './admin/Chip';
import Toast from './admin/Toast';
import { onWhatsNewLater, whatsNewUp } from './WhatsNewDialog';

// What an administrator is shown on the first page after signing in (16 Nov 2026): the notifications they have not
// read - a failed job, a run without data, a DL360 that did not come up. Once per load of the app: what arrives later
// moves the bell's count, it does not open a dialog over the work. It waits for the What's new dialog rather than
// opening over it. "Later" leaves them unread; "Mark all as read"
// records them read. Not on the Notifications page itself, and not with open access (no account to record it against).
let closed = false; // this load of the app has shown it, or had nothing to show
const SHOWN = 8;

export const LEVEL_CHIP = { error: 'down', warning: 'warn', info: 'flat' } as const;
export const LEVEL_WORD = { error: 'failed', warning: 'warning', info: 'note' } as const;
export const alertTime = (n: Alert) => (n.at === null ? '' : stamp(new Date(n.at).toISOString()));

export default function NotificationsDialog() {
  const me = useMe();
  const user = me?.user ?? null;
  const admin = Boolean(user && user.admin && !user.open);
  const { view, refresh } = useNotifications(admin ? 'once' : 'never');
  const { pathname } = useLocation();
  const sender = useSend(refresh);
  const [, setTick] = useState(0);
  // What's new goes first: until its answer is in, and while its dialog is up
  const wn = useWhatsNew(admin ? 'once' : 'never');
  useEffect(() => onWhatsNewLater(() => setTick((n) => n + 1)), []);
  const waits = (!wn.view && !wn.error) || whatsNewUp(wn.view, pathname);
  const close = () => {
    closed = true;
    setTick((n) => n + 1);
  };
  // the first answer decides: nothing unread then means no dialog for this load
  if (admin && view && !closed && !view.unread) closed = true;
  const unread = view ? view.items.filter((n) => !n.read) : [];
  const shown = admin && !closed && !waits && unread.length > 0 && pathname !== '/admin/notifications';
  return (
    <>
      {shown ? (
        <Sheet
          unread={unread}
          busy={sender.busy}
          error={sender.error}
          onLater={close}
          onRead={() =>
            void sender.send(ACTIONS.notificationsRead, { all: true }).then((done) => {
              if (done) close();
            })
          }
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </>
  );
}

function Sheet({ unread, busy, error, onLater, onRead }: { unread: Alert[]; busy: boolean; error: string | null; onLater: () => void; onRead: () => void }) {
  const sheet = useRef<HTMLDivElement>(null);
  const laterBtn = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const closeRef = useModal(sheet, onLater, () => laterBtn.current);
  const more = unread.length - SHOWN;
  return createPortal(
    <>
      <div className="scrim top" onClick={() => !busy && closeRef.current()} />
      <div ref={sheet} className="dialog wide" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <h2 id={titleId}>
          {unread.length} unread notification{unread.length === 1 ? '' : 's'}
        </h2>
        <div className="dialog-b">
          {unread.slice(0, SHOWN).map((n) => (
            <section key={n.id} className="alert-s">
              <h3>
                <Chip kind={LEVEL_CHIP[n.level]}>{LEVEL_WORD[n.level]}</Chip> {n.title} <span className="mut">{alertTime(n)}</span>
              </h3>
              {n.body.map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </section>
          ))}
          <p className="mut">
            {more > 0 ? `And ${more} more. ` : ''}All of them, with their links, are on{' '}
            <Link to="/admin/notifications" onClick={() => closeRef.current()}>
              Notifications
            </Link>
            .
          </p>
          {error ? (
            <p className="ferr" role="alert">
              {error}
            </p>
          ) : null}
        </div>
        <div className="dialog-f">
          <button ref={laterBtn} type="button" className="btn" onClick={() => closeRef.current()}>
            Later
          </button>
          <button type="button" className="btn primary" disabled={busy} aria-busy={busy || undefined} onClick={onRead}>
            {busy ? 'Sending…' : 'Mark all as read'}
          </button>
        </div>
      </div>
    </>,
    document.body,
  );
}
