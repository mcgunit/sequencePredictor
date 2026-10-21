import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { retryMe, useMe } from '../hooks/useMe';

const NARROW = '(max-width: 860px)';

// The menu as data. An item is a route of this app (to) or a server page (href, a full reload); a server
// page that the app will replace carries the ↗ "old page" mark. The users see the four top items; the
// administrator also the three groups.
interface Item {
  label: string;
  to?: string; // a route, possibly with an anchor (#q0)
  href?: string;
  old?: boolean;
  nested?: boolean; // an anchor inside the page above it
  links?: { label: string; href: string }[]; // one line, several old pages (the market workbench)
}
interface Group {
  label: string;
  items: Item[];
}

const USER_ITEMS: Item[] = [
  { label: 'Lottery', href: '/lottery', old: true },
  { label: 'Crypto', to: '/crypto' },
  { label: 'Shares', to: '/shares' },
  { label: 'Council', href: '/council', old: true },
];

const ADMIN_GROUPS: Group[] = [
  {
    label: 'Controls',
    items: [
      { label: 'All controls', to: '/admin/controls' },
      { label: 'Q0 null band', to: '/admin/controls#q0', nested: true },
      { label: 'Q2 randomness', to: '/admin/controls#q2', nested: true },
      { label: 'Feature control', to: '/admin/controls#fc', nested: true },
      { label: 'Market rows', to: '/admin/controls#rows', nested: true },
    ],
  },
  {
    label: 'Models',
    items: [
      { label: 'Best model & rankings', to: '/admin/models' },
      { label: 'Phase-shift', to: '/admin/phase-shift' },
      { label: 'Randomness watch', to: '/admin/randomness' },
      {
        label: 'Market workbench',
        links: [
          { label: 'crypto', href: '/markets/crypto' },
          { label: 'shares', href: '/markets/shares' },
        ],
      },
    ],
  },
  {
    label: 'System',
    items: [
      { label: 'Jobs & services', href: '/admin/jobs', old: true },
      { label: 'Users', href: '/admin/users', old: true },
      { label: "What's new", href: '/whats-new', old: true },
    ],
  },
];

// What the compact bar names on a phone.
const HERE: [string, string][] = [
  ['/crypto', 'Crypto'],
  ['/shares', 'Shares'],
  ['/admin/controls', 'Controls'],
  ['/admin/models', 'Models'],
  ['/admin/phase-shift', 'Phase-shift'],
  ['/admin/randomness', 'Randomness'],
];

function Old() {
  return (
    <span className="old">
      <span aria-hidden="true">↗</span>
      <span className="vh"> old page</span>
    </span>
  );
}

function ItemLink({ item, sub, pathname, hash, onNavigate }: { item: Item; sub: boolean; pathname: string; hash: string; onNavigate: () => void }) {
  const cls = [sub ? 'sub' : '', item.nested ? 'sub2' : ''].filter(Boolean).join(' ');
  if (item.links) {
    return (
      <div className="pair">
        {item.label}
        {item.links.map((l) => (
          <a key={l.href} href={l.href}>
            {l.label}
            <Old />
          </a>
        ))}
      </div>
    );
  }
  if (item.href) {
    return (
      <a className={cls || undefined} href={item.href}>
        {item.label}
        {item.old ? <Old /> : null}
      </a>
    );
  }
  const [path, anchor = ''] = (item.to ?? '/').split('#');
  // a page's own entry is "on" unless one of its anchors is; an anchor only when it is the one in the URL
  const anchored = hash !== '' && ADMIN_GROUPS.some((g) => g.items.some((i) => i.to === `${path}${hash}`));
  const on = pathname === path && (anchor ? hash === `#${anchor}` : !anchored);
  return (
    <Link className={[cls, on ? 'on' : ''].filter(Boolean).join(' ') || undefined} to={item.to ?? '/'} aria-current={on ? 'page' : undefined} onClick={onNavigate}>
      {item.label}
    </Link>
  );
}

export default function Sidebar() {
  const { pathname, hash } = useLocation();
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW).matches);
  const [open, setOpen] = useState(!narrow);
  const me = useMe();
  const user = me?.user ?? null;

  // Open as a sidebar on wide screens, closed under the compact bar on phones.
  useEffect(() => {
    const mq = window.matchMedia(NARROW);
    const onChange = () => {
      setNarrow(mq.matches);
      setOpen(!mq.matches);
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  // Close the drawer after navigating on a phone.
  useEffect(() => {
    if (narrow) setOpen(false);
  }, [pathname, narrow]);

  const here = (HERE.find(([p]) => pathname.startsWith(p)) ?? ['', ''])[1];
  const active = ({ isActive }: { isActive: boolean }) => (isActive ? 'on' : undefined);
  const closeDrawer = () => {
    if (narrow) setOpen(false);
  };

  return (
    <nav className="side" aria-label="Main">
      <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary className="bar">
          <span className="brand">Predictor</span>
          <span className="here">{here}</span>
          <span className="burger" aria-hidden="true">
            ☰
          </span>
        </summary>
        <div className="menu">
          <div className="brand">Predictor</div>
          {USER_ITEMS.map((it) =>
            it.to ? (
              <NavLink key={it.label} to={it.to} className={active} onClick={closeDrawer}>
                {it.label}
              </NavLink>
            ) : (
              <ItemLink key={it.label} item={it} sub={false} pathname={pathname} hash={hash} onNavigate={closeDrawer} />
            ),
          )}
          {user?.admin ? (
            <>
              <div className="grp">Admin</div>
              {ADMIN_GROUPS.map((g) => {
                const holds = g.items.some((i) => i.to && pathname === i.to.split('#')[0]);
                return (
                  // every group open on a wide screen; on a phone only the one holding the page
                  <details key={`${g.label}-${narrow ? 'n' : 'w'}`} className="grpd" open={!narrow || holds}>
                    <summary>{g.label}</summary>
                    {g.items.map((it) => (
                      <ItemLink key={it.label} item={it} sub pathname={pathname} hash={hash} onNavigate={closeDrawer} />
                    ))}
                  </details>
                );
              })}
            </>
          ) : null}
          {/* a failed "who is reading" (a restart, the network) is said, with a retry, instead of a user-only menu */}
          {me?.failed ? (
            <div className="foot">
              <span>Could not tell who is signed in</span>
              <span>
                <button type="button" className="linkbtn" onClick={retryMe}>
                  Try again
                </button>
              </span>
            </div>
          ) : null}
          {user ? (
            <div className="foot">
              {user.open ? (
                <span>Open access: no sign-in on this server</span>
              ) : (
                <>
                  <span>Signed in as {user.name}</span>
                  <span>
                    <a href="/account">
                      Account
                      <Old />
                    </a>
                    {' · '}
                    {/* /logout clears the session and answers with the login page (auth.js), as the old navbar's form */}
                    <form method="post" action="/logout">
                      <button type="submit" className="linkbtn">
                        Sign out
                      </button>
                    </form>
                  </span>
                </>
              )}
            </div>
          ) : null}
        </div>
      </details>
    </nav>
  );
}
