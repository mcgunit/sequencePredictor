import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { retryMe, useMe } from '../hooks/useMe';
import { useWeekStatus } from '../hooks/useWeekStatus';
import { useWhatsNew } from '../hooks/useWhatsNew';
import { readTheme, setTheme, type ThemeChoice } from '../theme';

const NARROW = '(max-width: 860px)';

// The menu as data: every item is a route of this app. The users see the five top items (Home, Lottery,
// Crypto, Shares, Council: the slim app of 21-23 Oct 2026), and in the footer their account, What's new and Sign out;
// the administrator the same and also This week (with its status dot) and the three groups, What's new among the
// System items.
interface Item {
  label: string;
  to?: string; // a route, possibly with an anchor (#q0)
  nested?: boolean; // an anchor inside the page above it
  prefix?: boolean; // "on" for the pages under it too (/admin/games/lotto/2026-10-21)
  links?: { label: string; to: string }[]; // one line, several routes (the market workbench: crypto, shares)
  news?: boolean; // carries What's new's unseen count
}
interface Group {
  label: string;
  items: Item[];
}

const USER_ITEMS: Item[] = [
  { label: 'Home', to: '/' },
  { label: 'Lottery', to: '/lottery', prefix: true },
  { label: 'Crypto', to: '/crypto' },
  { label: 'Shares', to: '/shares' },
  { label: 'Council', to: '/council' },
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
      { label: 'Tuning gates', to: '/admin/tuning' },
      { label: 'Best model & rankings', to: '/admin/models' },
      { label: 'Phase-shift', to: '/admin/phase-shift' },
      { label: 'Randomness watch', to: '/admin/randomness' },
      { label: 'Games', to: '/admin/games', prefix: true },
      {
        label: 'Market workbench',
        links: [
          { label: 'crypto', to: '/admin/markets/crypto' },
          { label: 'shares', to: '/admin/markets/shares' },
        ],
      },
    ],
  },
  {
    label: 'System',
    items: [
      { label: 'Jobs & services', to: '/admin/jobs' },
      { label: 'People', to: '/admin/people' },
      { label: "What's new", to: '/whats-new', news: true },
    ],
  },
];

// What the compact bar names on a phone ('/' only on Home itself).
const HERE: [string, string][] = [
  ['/lottery', 'Lottery'],
  ['/council', 'Council'],
  ['/crypto', 'Crypto'],
  ['/shares', 'Shares'],
  ['/admin/tuning', 'Tuning gates'],
  ['/admin/controls', 'Controls'],
  ['/admin/models', 'Models'],
  ['/admin/phase-shift', 'Phase-shift'],
  ['/admin/randomness', 'Randomness'],
  ['/admin/jobs', 'Jobs & services'],
  ['/admin/people', 'People'],
  ['/admin/games', 'Games'],
  ['/admin/markets', 'Market workbench'],
  ['/admin', 'This week'],
  ['/account', 'Account'],
  ['/whats-new', "What's new"],
];

// This week's dot: red when a run failed or an output is missing, amber when a run has a new warning signature
// (pages/admin/week.ts weekStatus); the page itself says which. No dot while the status is not known.
const DOT_TEXT = { red: 'a run failed or an output is missing', amber: 'a new warning signature' };
function WeekLink({ on, onNavigate }: { on: boolean; onNavigate: () => void }) {
  const status = useWeekStatus(true);
  const level = status?.level ?? null;
  return (
    <Link className={on ? 'wk on' : 'wk'} to="/admin" aria-current={on ? 'page' : undefined} onClick={onNavigate}>
      This week
      {level ? (
        <>
          <span className={`sdot ${level}`} aria-hidden="true" />
          <span className="vh">: {DOT_TEXT[level]}</span>
        </>
      ) : null}
    </Link>
  );
}

// What's new's unseen notes and an unread introduction, as a count; nothing at 0 or while not known.
function NewCount({ n }: { n: number | null | undefined }) {
  if (!n) return null;
  return (
    <span className="badge">
      {n}
      <span className="vh"> new in What&apos;s new</span>
    </span>
  );
}

function ItemLink({ item, sub, pathname, hash, news, onNavigate }: { item: Item; sub: boolean; pathname: string; hash: string; news?: number | null; onNavigate: () => void }) {
  const cls = [sub ? 'sub' : '', item.nested ? 'sub2' : ''].filter(Boolean).join(' ');
  if (item.links) {
    return (
      <div className="pair">
        {item.label}
        {item.links.map((l) => (
          <Link key={l.to} to={l.to} className={pathname === l.to ? 'on' : undefined} aria-current={pathname === l.to ? 'page' : undefined} onClick={onNavigate}>
            {l.label}
          </Link>
        ))}
      </div>
    );
  }
  const [path, anchor = ''] = (item.to ?? '/').split('#');
  // a page's own entry is "on" unless one of its anchors is; an anchor only when it is the one in the URL
  const anchored = hash !== '' && ADMIN_GROUPS.some((g) => g.items.some((i) => i.to === `${path}${hash}`));
  const on = (pathname === path || (item.prefix === true && pathname.startsWith(`${path}/`))) && (anchor ? hash === `#${anchor}` : !anchored);
  return (
    <Link className={[cls, on ? 'on' : '', item.news ? 'wn' : ''].filter(Boolean).join(' ') || undefined} to={item.to ?? '/'} aria-current={on ? 'page' : undefined} onClick={onNavigate}>
      {item.label}
      {item.news ? <NewCount n={news} /> : null}
    </Link>
  );
}

export default function Sidebar() {
  const { pathname, hash } = useLocation();
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW).matches);
  const [open, setOpen] = useState(!narrow);
  const me = useMe();
  const user = me?.user ?? null;
  // the unseen notes, and the introduction while it is unread (the old navbar's dot lit for either)
  const wn = useWhatsNew(user && !user.open ? 'once' : 'never').view;
  const news = wn && wn.unseen !== null ? wn.unseen + (wn.tour.done === false ? 1 : 0) : null;

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

  const here = pathname === '/' ? 'Home' : (HERE.find(([p]) => pathname.startsWith(p)) ?? ['', ''])[1];
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
          {/* on a phone the menu starts closed: the count shows on the bar, as the old navbar's dot did */}
          <NewCount n={news} />
          <span className="burger" aria-hidden="true">
            ☰
          </span>
        </summary>
        <div className="menu">
          <div className="brand">Predictor</div>
          {USER_ITEMS.map((it) =>
            it.to ? (
              // Home is "on" on Home only; Lottery also on a game's history and its draws (prefix)
              <NavLink key={it.label} to={it.to} end={!it.prefix} className={active} onClick={closeDrawer}>
                {it.label}
              </NavLink>
            ) : (
              <ItemLink key={it.label} item={it} sub={false} pathname={pathname} hash={hash} onNavigate={closeDrawer} />
            ),
          )}
          {user?.admin ? (
            <>
              <div className="grp">Admin</div>
              <WeekLink on={pathname === '/admin' || pathname === '/admin/'} onNavigate={closeDrawer} />
              {ADMIN_GROUPS.map((g) => {
                const holds = g.items.some(
                  (i) => (i.to && (pathname === i.to.split('#')[0] || (i.prefix && pathname.startsWith(`${i.to}/`)))) || i.links?.some((l) => pathname === l.to),
                );
                return (
                  // folded, except the one holding the page
                  <details key={g.label} className="grpd" open={holds}>
                    <summary>
                      <span>{g.label}</span>
                      {/* the count of a folded group's What's new (hidden while the group is open: the item has it) */}
                      {g.items.some((i) => i.news) ? <NewCount n={news} /> : null}
                    </summary>
                    {g.items.map((it) => (
                      <ItemLink key={it.label} item={it} sub pathname={pathname} hash={hash} news={news} onNavigate={closeDrawer} />
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
                  <Link className={pathname === '/account' ? 'acct on' : 'acct'} to="/account" aria-current={pathname === '/account' ? 'page' : undefined} onClick={closeDrawer}>
                    Signed in as <b>{user.name}</b>
                    <span className="vh">: your account</span>
                  </Link>
                  <span>
                    {/* a user has no System group: What's new is here */}
                    {!user.admin ? (
                      <>
                        <Link className={pathname === '/whats-new' ? 'wn on' : 'wn'} to="/whats-new" aria-current={pathname === '/whats-new' ? 'page' : undefined} onClick={closeDrawer}>
                          What&apos;s new
                          <NewCount n={news} />
                        </Link>
                        {' · '}
                      </>
                    ) : null}
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
          <ThemeSwitch />
        </div>
      </details>
    </nav>
  );
}

// Auto (the system's scheme), Light or Dark, remembered in this browser (theme.ts)
function ThemeSwitch() {
  const [choice, setChoice] = useState<ThemeChoice>(() => readTheme());
  const pick = (c: ThemeChoice) => {
    setTheme(c);
    setChoice(c);
  };
  return (
    <div className="foot theme">
      <span id="theme-label">Colours</span>
      <div className="seg" role="group" aria-labelledby="theme-label">
        {(['auto', 'light', 'dark'] as const).map((c) => (
          <button key={c} type="button" aria-pressed={choice === c} onClick={() => pick(c)} title={c === 'auto' ? 'follow the system setting' : undefined}>
            {c === 'auto' ? 'Auto' : c === 'light' ? 'Light' : 'Dark'}
          </button>
        ))}
      </div>
    </div>
  );
}
