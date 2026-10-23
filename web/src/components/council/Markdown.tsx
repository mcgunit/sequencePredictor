import type { ReactNode } from 'react';

// The members' and the head's answers are Markdown written by language models: untrusted text. It is rendered as React
// elements only (never innerHTML), so nothing in it can run or style the page. What the members actually write is
// covered, as on the old council page: headings, paragraphs, bold and italics, inline and fenced code, bullet and
// numbered lists (nested by indentation), block quotes, rules, pipe tables and http(s) links (opened in a new tab). It is
// re-read on every streamed chunk, so a half-finished construct renders as far as it goes and settles when it closes.

const INLINE = /(`+)([^`]*?)\1|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|\*\*(.+?)\*\*|__(.+?)__|(^|[^\w*])\*([^*\n]+?)\*(?=[^\w*]|$)/g;

function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = new RegExp(INLINE.source, 'g');
  let last = 0;
  let k = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    const lead = m[7] ?? '';
    if (m.index + lead.length > last) out.push(text.slice(last, m.index + lead.length));
    const id = `${key}.${k}`;
    k += 1;
    if (m[1] !== undefined) out.push(<code key={id}>{m[2]}</code>);
    else if (m[3] !== undefined)
      out.push(
        <a key={id} href={m[4]} target="_blank" rel="noopener noreferrer">
          {m[3]}
        </a>,
      );
    else if (m[5] !== undefined || m[6] !== undefined) out.push(<strong key={id}>{inline(m[5] ?? m[6] ?? '', id)}</strong>);
    else out.push(<em key={id}>{inline(m[8] ?? '', id)}</em>);
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// lines joined with a break, each read inline
function lines(xs: string[], key: string): ReactNode[] {
  return xs.flatMap((x, i) => (i ? [<br key={`${key}b${i}`} />, ...inline(x, `${key}l${i}`)] : inline(x, `${key}l${i}`)));
}

interface ListItem {
  text: string[];
  subs: List[];
}
interface List {
  ordered: boolean;
  indent: number;
  items: ListItem[];
}
const ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const FENCE_OPEN = /^\s*(```|~~~)\s*([\w+-]+)?\s*$/;
const FENCE_CLOSE = /^\s*(```|~~~)\s*$/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s*([-*_])\s*(\1\s*){2,}$/;
const QUOTE = /^\s*>/;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function cells(line: string): string[] {
  let t = line.trim();
  if (t.startsWith('|')) t = t.slice(1);
  if (t.endsWith('|')) t = t.slice(0, -1);
  return t.split('|').map((c) => c.trim());
}

function renderList(list: List, key: string): ReactNode {
  const items = list.items.map((it, i) => (
    <li key={i}>
      {lines(it.text, `${key}i${i}`)}
      {it.subs.map((s, j) => renderList(s, `${key}i${i}s${j}`))}
    </li>
  ));
  return list.ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>;
}

function blocks(text: string, key: string): ReactNode[] {
  const src = text.replace(/\r\n?/g, '\n').split('\n');
  const out: ReactNode[] = [];
  let para: string[] = [];
  let n = 0;
  const id = () => {
    n += 1;
    return `${key}-${n}`;
  };
  const flush = () => {
    if (!para.length) return;
    const k = id();
    out.push(<p key={k}>{lines(para, k)}</p>);
    para = [];
  };
  let i = 0;
  while (i < src.length) {
    const line = src[i];
    if (FENCE_OPEN.test(line)) {
      flush();
      const code: string[] = [];
      i += 1;
      while (i < src.length && !FENCE_CLOSE.test(src[i])) {
        code.push(src[i]);
        i += 1;
      }
      i += 1; // past the closing fence, or the end
      out.push(
        <pre key={id()}>
          <code>{code.join('\n')}</code>
        </pre>,
      );
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      flush();
      const level = heading[1].length;
      const k = id();
      const body = inline(heading[2], k);
      out.push(level <= 2 ? <h4 key={k}>{body}</h4> : level === 3 ? <h5 key={k}>{body}</h5> : <h6 key={k}>{body}</h6>);
      i += 1;
      continue;
    }
    if (RULE.test(line)) {
      flush();
      out.push(<hr key={id()} />);
      i += 1;
      continue;
    }
    if (QUOTE.test(line)) {
      flush();
      const quoted: string[] = [];
      while (i < src.length && QUOTE.test(src[i])) {
        quoted.push(src[i].replace(/^\s*>\s?/, ''));
        i += 1;
      }
      const k = id();
      out.push(<blockquote key={k}>{blocks(quoted.join('\n'), k)}</blockquote>);
      continue;
    }
    if (line.includes('|') && i + 1 < src.length && TABLE_SEP.test(src[i + 1])) {
      flush();
      const head = cells(line);
      const body: string[][] = [];
      i += 2;
      while (i < src.length && src[i].includes('|') && src[i].trim()) {
        body.push(cells(src[i]));
        i += 1;
      }
      const k = id();
      out.push(
        <div className="md-tbl" key={k}>
          <table>
            <thead>
              <tr>
                {head.map((c, j) => (
                  <th key={j}>{inline(c, `${k}h${j}`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {body.map((row, r) => (
                <tr key={r}>
                  {row.map((c, j) => (
                    <td key={j}>{inline(c, `${k}r${r}c${j}`)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (ITEM.test(line)) {
      flush();
      // nested lists by indentation: a stack of open lists, the root at the bottom
      const stack: List[] = [];
      let root: List | null = null;
      while (i < src.length) {
        const it = ITEM.exec(src[i]);
        if (!it) {
          // an indented line continues the previous item
          const top = stack[stack.length - 1];
          if (top && top.items.length && /^\s{2,}\S/.test(src[i])) {
            top.items[top.items.length - 1].text.push(src[i].trim());
            i += 1;
            continue;
          }
          break;
        }
        const indent = it[1].replace(/\t/g, '  ').length;
        while (stack.length > 1 && indent < stack[stack.length - 1].indent) stack.pop();
        const top = stack[stack.length - 1];
        if (!top || indent > top.indent) {
          const list: List = { ordered: /\d/.test(it[2]), indent, items: [] };
          if (top && top.items.length) top.items[top.items.length - 1].subs.push(list);
          else if (top) top.items.push({ text: [], subs: [list] });
          else root = list;
          stack.push(list);
        }
        stack[stack.length - 1].items.push({ text: [it[3]], subs: [] });
        i += 1;
      }
      if (root) out.push(renderList(root, id()));
      continue;
    }
    if (!line.trim()) {
      flush();
      i += 1;
      continue;
    }
    para.push(line);
    i += 1;
  }
  flush();
  return out;
}

export default function Markdown({ text, typing }: { text: string; typing?: boolean }) {
  return (
    <div className="md">
      {blocks(text, 'md')}
      {typing ? (
        <span className="cursor" aria-hidden="true">
          ▍
        </span>
      ) : null}
    </div>
  );
}

// One line of an answer without its Markdown marks, for a folded member's summary
export function excerpt(text: string, max: number): string {
  const one = text
    .replace(/```[\s\S]*?```/g, ' [code] ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s*>\s?/gm, '')
    .replace(/^\s*([-*+]|\d+[.)])\s+/gm, '')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(^|\W)[*_]([^*_\n]+)[*_](?=\W|$)/g, '$1$2')
    .replace(/^\s*[-*_]{3,}\s*$/gm, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return one.length > max ? `${one.slice(0, max - 1)}…` : one;
}
