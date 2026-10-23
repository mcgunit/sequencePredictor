import { Link } from 'react-router-dom';

// The old pages the app now has (21 Oct 2026; the Lottery and the Council 23 Oct): a note's link to one of them opens the
// app's page instead.
const IN_APP: Record<string, string> = {
  '/lottery': '/lottery',
  '/council': '/council',
  '/admin/jobs': '/admin/jobs',
  '/admin/users': '/admin/people',
  '/admin/activity': '/admin/people',
  '/account': '/account',
  '/whats-new': '/whats-new',
};

// A note's link: inside the app through the router; the server's other pages are a full load.
export default function NoteLink({ href, label }: { href: string; label: string }) {
  const to = href === '/app' || href.startsWith('/app/') ? href.slice(4) || '/' : IN_APP[href.split(/[?#]/)[0]] ?? null;
  if (to !== null) {
    return (
      <Link className="linkbtn more" to={to}>
        {label} ›
      </Link>
    );
  }
  return (
    <a className="linkbtn more" href={href}>
      {label} ›
    </a>
  );
}
