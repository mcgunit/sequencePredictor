import { useEffect, useMemo, type ReactNode } from 'react';
import { Link, Navigate, useLocation, useParams } from 'react-router-dom';
import { getAdminWiki, getAdminWikiPage } from '../../api';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import { useLoad } from '../../hooks/useLoad';
import { parse, target, type Block, type Inline } from '../../wiki/markdown';

// Admin › Wiki: the project's documentation (README.md and docs/*.md) read inside the app. The pages are the files of
// the checkout the server runs from, so the wiki always matches the code that is live. Read only.

const day = (iso: string | null): string | null => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'Europe/Brussels' });
};

function Spans({ c, pages }: { c: Inline[]; pages: string[] }): ReactNode {
  return c.map((x, i) => {
    if (x.t === 'text') return x.v;
    if (x.t === 'code') return <code key={i}>{x.v}</code>;
    if (x.t === 'b') return <strong key={i}><Spans c={x.c} pages={pages} /></strong>;
    if (x.t === 'i') return <em key={i}><Spans c={x.c} pages={pages} /></em>;
    const to = target(x.href, pages);
    const body = <Spans c={x.c} pages={pages} />;
    if (to.kind === 'web') return <a key={i} href={to.href} target="_blank" rel="noreferrer noopener">{body}</a>;
    if (to.kind === 'hash') return <a key={i} href={`#${to.hash}`}>{body}</a>;
    if (to.kind === 'page') return <Link key={i} to={`/admin/wiki/${to.page}${to.hash ? `#${to.hash}` : ''}`}>{body}</Link>;
    // a file of the repository: the app has no page for it, so it reads as its path
    return <span key={i} className="wiki-file" title={to.path}>{body}</span>;
  });
}

function Blocks({ blocks, pages }: { blocks: Block[]; pages: string[] }): ReactNode {
  return blocks.map((b, i) => {
    if (b.t === 'h') {
      const H = `h${Math.min(6, b.level)}` as 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
      return (
        <H key={i} id={b.id}>
          <Spans c={b.c} pages={pages} />
          <a className="wiki-anchor" href={`#${b.id}`} aria-label={`Link to ${b.text}`}>#</a>
        </H>
      );
    }
    if (b.t === 'p') return <p key={i}><Spans c={b.c} pages={pages} /></p>;
    if (b.t === 'hr') return <hr key={i} />;
    if (b.t === 'code') return <pre key={i}><code>{b.v}</code></pre>;
    if (b.t === 'quote') return <blockquote key={i}><Blocks blocks={b.c} pages={pages} /></blockquote>;
    if (b.t === 'list') {
      const items = b.items.map((it, j) => (
        <li key={j} className={it.box === null ? undefined : 'wiki-box'}>
          {it.box === null ? null : <span className="wiki-tick" role="img" aria-label={it.box ? 'done' : 'open'}>{it.box ? '☑' : '☐'}</span>}
          <Spans c={it.c} pages={pages} />
          {it.sub.length ? <Blocks blocks={it.sub} pages={pages} /> : null}
        </li>
      ));
      return b.ordered ? <ol key={i} start={b.start}>{items}</ol> : <ul key={i}>{items}</ul>;
    }
    const cls = (k: number) => (b.align[k] === 'r' ? 'num' : b.align[k] === 'c' ? 'ctr' : undefined);
    return (
      <div key={i} className="scroll wiki-tbl">
        <table className="tbl">
          <thead>
            <tr>{b.head.map((h, k) => <th key={k} className={cls(k)}><Spans c={h} pages={pages} /></th>)}</tr>
          </thead>
          <tbody>
            {b.rows.map((r, j) => (
              <tr key={j}>{r.map((c, k) => <td key={k} className={cls(k)}><Spans c={c} pages={pages} /></td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  });
}

function Doc({ id, pages }: { id: string; pages: string[] }) {
  const doc = useLoad(() => getAdminWikiPage(id), [id]);
  const { hash } = useLocation();
  const text = doc.kind === 'ok' && doc.data.id === id ? doc.data.text : null;
  const blocks = useMemo(() => (text === null ? [] : parse(text)), [text]);
  const toc = useMemo(() => blocks.filter((b): b is Extract<Block, { t: 'h' }> => b.t === 'h' && b.level === 2), [blocks]);

  // "#section" in the address: go there once the page is on screen, else start at the top
  useEffect(() => {
    if (text === null) return;
    const el = hash.length > 1 ? document.getElementById(decodeURIComponent(hash.slice(1))) : null;
    if (el) el.scrollIntoView();
    else if (hash.length <= 1) window.scrollTo(0, 0);
  }, [text, hash, id]);

  if (doc.kind === 'error') return <Failed error={doc.error} retry={doc.retry} />;
  if (text === null || doc.kind !== 'ok') return <Loading />;
  const changed = day(doc.data.updatedAt);
  return (
    <div className="card wiki-doc">
      <div className="card-b">
        <p className="note wiki-src">
          <code>{doc.data.file}</code>
          {changed ? ` · last changed ${changed}` : ''}
        </p>
        {toc.length > 2 ? (
          <details className="wiki-toc">
            <summary>On this page ({toc.length} sections)</summary>
            <ul>
              {toc.map((h) => (
                <li key={h.id}><a href={`#${h.id}`}>{h.text}</a></li>
              ))}
            </ul>
          </details>
        ) : null}
        <article className="wiki">
          <Blocks blocks={blocks} pages={pages} />
        </article>
      </div>
    </div>
  );
}

export default function WikiPage() {
  const { page } = useParams();
  const list = useLoad(getAdminWiki, []);
  if (list.kind === 'loading') return <Loading />;
  if (list.kind === 'error') return <Failed error={list.error} retry={list.retry} />;
  const { name, pages } = list.data;
  const ids = pages.map((p) => p.id);
  if (!pages.length) {
    return (
      <>
        <PageHead title={name} sub="The project's documentation" />
        <div className="card"><div className="card-b empty">No documentation files were found next to the server (README.md, docs/*.md).</div></div>
      </>
    );
  }
  if (!page || !ids.includes(page)) return <Navigate to={`/admin/wiki/${ids[0]}`} replace />;
  return (
    <>
      <PageHead title={name} sub="The project's documentation, as it is in the running checkout" />
      <div className="wiki-wrap">
        <nav className="wiki-pages" aria-label="Wiki pages">
          {pages.map((p) => (
            <Link key={p.id} to={`/admin/wiki/${p.id}`} className={p.id === page ? 'on' : undefined} aria-current={p.id === page ? 'page' : undefined}>
              {p.title}
            </Link>
          ))}
        </nav>
        <Doc key={page} id={page} pages={ids} />
      </div>
    </>
  );
}
