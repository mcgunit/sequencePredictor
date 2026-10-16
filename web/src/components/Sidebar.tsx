import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { getMe, type Me } from '../api';

const NARROW = '(max-width: 860px)';

interface Item {
  label: string;
  href: string;
}

// Server pages are plain anchors (full reload); the two market pages are router links. The admin
// groups point at the pages that exist today (the server-rendered ones); the market page follows
// the market the reader is on.
function adminGroups(market: 'crypto' | 'shares'): { label: string; items: Item[] }[] {
  return [
    {
      label: 'Models',
      items: [
        { label: 'Market page: books, charts, background', href: `/markets/${market}` },
        { label: 'Day by day', href: `/markets/${market}#daily-${market}` },
        { label: 'History & best model', href: '/database' },
      ],
    },
    {
      label: 'Controls',
      items: [{ label: 'Null band, randomness watch, feature control', href: '/database' }],
    },
    {
      label: 'System',
      items: [
        { label: 'Jobs & services', href: '/admin/jobs' },
        { label: 'Users', href: '/admin/users' },
        { label: "What's new", href: '/whats-new' },
      ],
    },
  ];
}

function Group({ label, items }: { label: string; items: Item[] }) {
  return (
    <details className="grpd">
      <summary>{label}</summary>
      {items.map((it) => (
        <a key={it.href} className="sub" href={it.href}>
          {it.label}
        </a>
      ))}
    </details>
  );
}

export default function Sidebar() {
  const { pathname } = useLocation();
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW).matches);
  const [open, setOpen] = useState(!narrow);
  const [user, setUser] = useState<Me['user']>(null);

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

  useEffect(() => {
    let alive = true;
    getMe()
      .then((me) => {
        if (alive) setUser(me.user);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const here = pathname.startsWith('/shares') ? 'Shares' : pathname.startsWith('/crypto') ? 'Crypto' : '';
  const market = pathname.startsWith('/shares') ? 'shares' : 'crypto';
  const active = ({ isActive }: { isActive: boolean }) => (isActive ? 'on' : undefined);
  const closeDrawer = () => {
    if (narrow) setOpen(false);
  };

  return (
    <nav className="side">
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
          <a href="/lottery">Lottery</a>
          <NavLink to="/crypto" className={active} onClick={closeDrawer}>
            Crypto
          </NavLink>
          <NavLink to="/shares" className={active} onClick={closeDrawer}>
            Shares
          </NavLink>
          <a href="/council">Council</a>
          {user?.admin ? (
            <>
              <div className="grp">Admin</div>
              {adminGroups(market).map((g) => (
                <Group key={g.label} label={g.label} items={g.items} />
              ))}
            </>
          ) : null}
          {user?.name ? <div className="foot">Signed in as {user.name}</div> : null}
        </div>
      </details>
    </nav>
  );
}
