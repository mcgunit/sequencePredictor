import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link, useLocation } from 'react-router-dom';
import { ACTIONS, type WhatsNewView } from '../api';
import { dayYear } from '../format';
import { useMe } from '../hooks/useMe';
import { useModal } from '../hooks/useModal';
import { useSend, type Sender } from '../hooks/useSend';
import { useWhatsNew } from '../hooks/useWhatsNew';
import NoteLink from './NoteLink';
import Toast from './admin/Toast';

// What the old pages pop up on whichever page a reader lands on (whatsnew.js dialog), in the app shell: the
// introduction for an account that never finished it, else the unseen major notes (GET /api/whats-new's pending).
// "Start using it" / "Got it" records them seen - the old pages read the same record; "Later" hides it until the app is
// loaded again. Not on What's new itself, which holds all of it.
let later: string | null = null; // the pending set "Later" was pressed on, for this load of the app

// "Show it again" on What's new: the introduction opens again on the next page, even after a "Later".
export function showAgain() {
  later = null;
}

const keyOf = (p: NonNullable<WhatsNewView['pending']>) => `${p.kind}:${p.ids.join(',')}`;

export default function WhatsNewDialog() {
  const me = useMe();
  const user = me?.user ?? null;
  const { view, refresh } = useWhatsNew(user && !user.open ? 'once' : 'never');
  const { pathname } = useLocation();
  const sender = useSend(refresh);
  const [, setHidden] = useState(0);
  const p = view?.pending ?? null;
  const shown = Boolean(user && !user.open && view && p && pathname !== '/whats-new' && later !== keyOf(p));
  return (
    <>
      {shown && view && p && user ? (
        <PendingSheet
          view={view}
          name={user.name}
          sender={sender}
          onLater={() => {
            later = keyOf(p);
            setHidden((n) => n + 1);
          }}
        />
      ) : null}
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </>
  );
}

function PendingSheet({ view, name, sender, onLater }: { view: WhatsNewView; name: string; sender: Sender; onLater: () => void }) {
  const sheet = useRef<HTMLDivElement>(null);
  const laterBtn = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const closeRef = useModal(sheet, onLater, () => laterBtn.current);
  const p = view.pending;
  if (!p) return null;
  const tour = p.kind === 'tour';
  const notes = view.items.filter((n) => p.ids.includes(n.id));
  // the introduction marks every unseen note with it, as the old dialog does (all: past the 50 ids a request may name)
  const done = () => void sender.send(ACTIONS.seen, tour ? { all: true, tour: true } : { ids: p.ids });
  return createPortal(
    <>
      <div className="scrim top" onClick={() => !sender.busy && closeRef.current()} />
      <div ref={sheet} className="dialog wide" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <h2 id={titleId}>{tour ? `${view.tour.title}, ${name}` : "What's new"}</h2>
        <div className="dialog-b">
          {tour ? (
            <>
              {view.tour.intro ? <p>{view.tour.intro}</p> : null}
              {view.tour.steps.map((s) => (
                <section key={s.title}>
                  <h3>{s.title}</h3>
                  <p>{s.body}</p>
                </section>
              ))}
            </>
          ) : (
            notes.map((n) => (
              <section key={n.id}>
                <h3>
                  {n.title} <span className="mut">{dayYear(n.date)}</span>
                </h3>
                {n.body.map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
                {n.link ? <NoteLink href={n.link.href} label={n.link.label} /> : null}
              </section>
            ))
          )}
          <p className="mut">
            Everything here stays on <Link to="/whats-new">What&apos;s new</Link>.
          </p>
          {sender.error ? (
            <p className="ferr" role="alert">
              {sender.error}
            </p>
          ) : null}
        </div>
        <div className="dialog-f">
          <button ref={laterBtn} type="button" className="btn" onClick={() => closeRef.current()}>
            Later
          </button>
          <button type="button" className="btn primary" disabled={sender.busy} aria-busy={sender.busy || undefined} onClick={done}>
            {sender.busy ? 'Sending…' : tour ? 'Start using it' : 'Got it'}
          </button>
        </div>
      </div>
    </>,
    document.body,
  );
}
