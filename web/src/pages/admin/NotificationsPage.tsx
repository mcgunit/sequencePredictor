import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ACTIONS, type Alert } from '../../api';
import { useMe } from '../../hooks/useMe';
import { useNotifications } from '../../hooks/useNotifications';
import { useSend, type Sender } from '../../hooks/useSend';
import { LEVEL_CHIP, LEVEL_WORD, alertTime } from '../../components/NotificationsDialog';
import Chip from '../../components/admin/Chip';
import Toast from '../../components/admin/Toast';
import { Failed, Loading, PageHead } from '../../components/admin/States';

// /admin/notifications (16 Nov 2026): what went wrong while nobody was looking - a job that failed, a lottery run
// without data and its retries, a control that fell back from the DL360, a DL360 that did not come up or stayed on,
// a week's reading that was not written, a job far slower than usual. Newest first, the unread ones outlined; "Mark
// as read" records it against this administrator's account and the sidebar's bell follows.
export default function NotificationsPage() {
  const { view, error, refresh } = useNotifications('now');
  const sender = useSend(refresh, { errorToast: 'Not recorded: ' });
  const open = useMe()?.user?.open === true;

  useEffect(() => {
    document.title = 'Notifications · Predictor';
  }, []);

  if (!view) {
    return (
      <>
        <PageHead title="Notifications" />
        {error ? <Failed error={error} retry={() => void refresh()} /> : <Loading />}
      </>
    );
  }
  return (
    <div className="admin">
      <PageHead
        title="Notifications"
        sub={`${view.of} kept${view.kept ? ` (the newest ${view.kept})` : ''}${view.unread ? ` · ${view.unread} unread` : ' · all read'}`}
        aside={
          view.unread && !open ? (
            <button type="button" className="btn" disabled={sender.busy} onClick={() => void sender.send(ACTIONS.notificationsRead, { all: true })}>
              Mark all as read
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
        {view.items.map((n) => (
          <AlertCard key={n.id} n={n} sender={sender} canMark={!open} />
        ))}
        {!view.items.length ? <p className="note">Nothing to report: no job failed, no run went without data, the workers did what they were asked.</p> : null}
      </div>
      <Toast toast={sender.toast} onClose={sender.dismiss} />
    </div>
  );
}

function AlertCard({ n, sender, canMark }: { n: Alert; sender: Sender; canMark: boolean }) {
  return (
    <section className={`card alert-c${n.read ? '' : ' fresh'}`}>
      <div className="alert-h">
        <h2>{n.title}</h2>
        <span className="note-m">
          {alertTime(n)}
          <Chip kind={LEVEL_CHIP[n.level]}>{LEVEL_WORD[n.level]}</Chip>
          {n.read ? null : <Chip kind="chg">unread</Chip>}
        </span>
      </div>
      <div className="card-b">
        {n.body.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
        <div className="acts">
          {n.link ? (
            <Link className="linkbtn more" to={n.link.href}>
              {n.link.label} ›
            </Link>
          ) : null}
          {!n.read && canMark ? (
            <button type="button" className="btn" disabled={sender.busy} onClick={() => void sender.send(ACTIONS.notificationsRead, { ids: [n.id] })}>
              Mark as read
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}
