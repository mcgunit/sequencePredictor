import { useEffect } from 'react';
import { ACTIONS, type Note, type WhatsNewView } from '../api';
import { dayYear } from '../format';
import { useSend, type Sender } from '../hooks/useSend';
import { useWhatsNew } from '../hooks/useWhatsNew';
import NoteLink from '../components/NoteLink';
import { showAgain } from '../components/WhatsNewDialog';
import Chip from '../components/admin/Chip';
import Toast from '../components/admin/Toast';
import { Failed, Loading, PageHead } from '../components/admin/States';

// /whats-new, every signed-in reader's own: the notes their audience may see, newest first - the unseen ones open and
// marked, the seen ones folded to their title - and the introduction, which can be read again. "Mark as seen" records
// it against the account (the old pages read the same record), and the sidebar's count follows. A refusal also shows
// as a toast at the foot of the screen: the button pressed may be far down the list.
export default function WhatsNewPage() {
  const { view, error, refresh } = useWhatsNew('now');
  const sender = useSend(refresh, { errorToast: 'Not recorded: ' });

  useEffect(() => {
    document.title = "What's new · Predictor";
  }, []);

  if (!view) {
    return (
      <>
        <PageHead title="What's new" />
        {error ? <Failed error={error} retry={() => void refresh()} /> : <Loading />}
      </>
    );
  }
  const unseen = view.items.filter((n) => n.seen === false);
  return (
    <div className="admin">
      <PageHead
        title="What's new"
        sub={`${view.of} note${view.of === 1 ? '' : 's'}${view.unseen ? ` · ${view.unseen} not seen yet` : ''}`}
        aside={
          unseen.length ? (
            <button type="button" className="btn" disabled={sender.busy} onClick={() => void sender.send(ACTIONS.seen, { all: true })}>
              Mark all as seen
            </button>
          ) : null
        }
      />
      {sender.error ? (
        <p className="statusline" role="alert">
          <Chip kind="down">not recorded</Chip> {sender.error}
        </p>
      ) : null}
      <div className="notes">
        <TourCard view={view} sender={sender} />
        {view.items.map((n) => (
          <NoteCard key={`${n.id}-${n.seen === false ? 'new' : 'seen'}`} n={n} sender={sender} />
        ))}
        {!view.items.length ? <p className="note">No notes yet.</p> : null}
      </div>
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

function NoteCard({ n, sender }: { n: Note; sender: Sender }) {
  const fresh = n.seen === false;
  return (
    <details className={`card note-c${fresh ? ' fresh' : ''}`}>
      <summary className="note-h">
        <h2>{n.title}</h2>
        <span className="note-m">
          {dayYear(n.date)}
          {fresh ? <Chip kind="chg">new</Chip> : null}
          {n.audience === 'admin' ? <Chip kind="flat">admin only</Chip> : null}
        </span>
      </summary>
      <div className="card-b">
        {n.body.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
        <div className="acts">
          {n.link ? <NoteLink href={n.link.href} label={n.link.label} /> : null}
          {fresh ? (
            <button type="button" className="btn" disabled={sender.busy} onClick={() => void sender.send(ACTIONS.seen, { ids: [n.id] })}>
              Mark as seen
            </button>
          ) : null}
        </div>
      </div>
    </details>
  );
}

// The introduction every account is shown once: readable again here. "Show it again" puts it back as unread (the app
// and the old pages then open it once more, on the next page); "Mark as read" records it read.
function TourCard({ view, sender }: { view: WhatsNewView; sender: Sender }) {
  const t = view.tour;
  const unread = t.done === false;
  return (
    <details className={`card note-c${unread ? ' fresh' : ''}`}>
      <summary className="note-h">
        <h2>Introduction · {t.title}</h2>
        <span className="note-m">
          {t.steps.length} parts
          {unread ? <Chip kind="chg">not read yet</Chip> : null}
        </span>
      </summary>
      <div className="card-b">
        {t.intro ? <p>{t.intro}</p> : null}
        {t.steps.map((s) => (
          <section key={s.title} className="step">
            <h3>{s.title}</h3>
            <p>{s.body}</p>
          </section>
        ))}
        {t.done !== null ? (
          <div className="acts">
            {unread ? (
              <button type="button" className="btn" disabled={sender.busy} onClick={() => void sender.send(ACTIONS.seen, { tour: true })}>
                Mark as read
              </button>
            ) : (
              <button
                type="button"
                className="btn"
                disabled={sender.busy}
                onClick={() => {
                  showAgain();
                  void sender.send(ACTIONS.replay);
                }}
              >
                Show it again
              </button>
            )}
          </div>
        ) : null}
      </div>
    </details>
  );
}
