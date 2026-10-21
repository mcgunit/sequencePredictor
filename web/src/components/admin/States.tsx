import type { ReactNode } from 'react';
import { ApiError } from '../../api';
import { retryMe, useMe } from '../../hooks/useMe';

// The admin pages' page head: a title with its subtitle, the meta on the right (the run times, "Brussels
// time" once), and room for the All / Top 5 switch.
export function PageHead({ title, sub, meta, aside }: { title: string; sub?: ReactNode; meta?: ReactNode[]; aside?: ReactNode }) {
  const items = (meta ?? []).filter(Boolean);
  return (
    <div className="page-head">
      <h1>
        {title}
        {sub ? <small>{sub}</small> : null}
      </h1>
      {items.length || aside ? (
        <div className="head-r">
          {items.length ? (
            <div className="meta">
              {items.map((m, i) => (
                <span key={i}>{m}</span>
              ))}
            </div>
          ) : null}
          {aside}
        </div>
      ) : null}
    </div>
  );
}

export function AdminOnly() {
  return (
    <div className="card">
      <div className="card-h">
        <h2>This page is for the admin</h2>
      </div>
      <div className="card-b">
        <p className="note">Lottery, Crypto, Shares and Council are in the menu.</p>
      </div>
    </div>
  );
}

export function Loading({ what = 'Loading…' }: { what?: string }) {
  return (
    <div className="card">
      <div className="card-b empty">{what}</div>
    </div>
  );
}

// A failed load: a 403 is "not for you", anything else says what failed and offers a retry.
export function Failed({ error, retry, bare }: { error: Error; retry?: () => void; bare?: boolean }) {
  if (error instanceof ApiError && error.status === 403) return <AdminOnly />;
  const body = (
    <>
      <p className="mut">
        <code>{error.message}</code>
      </p>
      {retry ? (
        <button type="button" className="linkbtn more" onClick={retry}>
          Try again
        </button>
      ) : null}
    </>
  );
  if (bare) return <div className="failed">{body}</div>;
  return (
    <div className="card err">
      <div className="card-h">
        <h2>Could not load this page</h2>
      </div>
      <div className="card-b">{body}</div>
    </div>
  );
}

// The admin routes: their pages render only for the administrator (the API answers a JSON 403 anyway). A
// failed "who is reading" says it failed, with a retry - it is not an answer that the reader is not the admin.
export default function AdminGuard({ children }: { children: ReactNode }) {
  const me = useMe();
  if (me === undefined) return <Loading />;
  if (me.failed) return <Failed error={me.failed} retry={retryMe} />;
  if (!me.user?.admin) return <AdminOnly />;
  return <>{children}</>;
}
