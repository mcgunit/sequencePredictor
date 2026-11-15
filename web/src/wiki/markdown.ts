// A small markdown reader for the wiki (Admin › Wiki): the subset the project's own docs use, turned into a tree the
// page renders as React elements. Nothing here produces HTML text, so a doc can never inject markup into the app.
//
// Blocks: # headings, paragraphs, - and 1. lists (nested by indent, [ ] / [x] boxes), | tables |, ``` code ```,
// > quotes, --- rules. Inline: `code`, **bold**, *italic*, [text](link). HTML comments are skipped.

export type Inline =
  | { t: 'text'; v: string }
  | { t: 'code'; v: string }
  | { t: 'b'; c: Inline[] }
  | { t: 'i'; c: Inline[] }
  | { t: 'a'; href: string; c: Inline[] };

export type Item = { c: Inline[]; box: boolean | null; sub: Block[] };
export type Block =
  | { t: 'h'; level: number; id: string; c: Inline[]; text: string }
  | { t: 'p'; c: Inline[] }
  | { t: 'list'; ordered: boolean; start: number; items: Item[] }
  | { t: 'table'; align: ('l' | 'r' | 'c')[]; head: Inline[][]; rows: Inline[][][] }
  | { t: 'code'; lang: string; v: string }
  | { t: 'quote'; c: Block[] }
  | { t: 'hr' };

// --- inline ---
export function inline(src: string): Inline[] {
  const out: Inline[] = [];
  let text = '';
  const flush = () => {
    if (text) out.push({ t: 'text', v: text });
    text = '';
  };
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === '\\' && i + 1 < src.length && /[\\`*_[\](){}#+\-.!|<>~]/.test(src[i + 1])) {
      text += src[i + 1];
      i += 2;
      continue;
    }
    if (ch === '`') {
      let n = 1;
      while (src[i + n] === '`') n += 1;
      const fence = '`'.repeat(n);
      const end = src.indexOf(fence, i + n);
      if (end > 0) {
        flush();
        out.push({ t: 'code', v: src.slice(i + n, end).replace(/^ (.*) $/, '$1') });
        i = end + n;
        continue;
      }
    }
    if (ch === '*' || ch === '_') {
      const two = src[i + 1] === ch;
      const mark = two ? ch + ch : ch;
      // an underscore inside a word is a name (snake_case), not emphasis
      const wordBefore = ch === '_' && i > 0 && /\w/.test(src[i - 1]);
      const next = src[i + mark.length];
      if (!wordBefore && next !== undefined && !/\s/.test(next)) {
        let end = src.indexOf(mark, i + mark.length + 1);
        while (end > 0 && (/\s/.test(src[end - 1]) || (ch === '_' && /\w/.test(src[end + mark.length] ?? '')) || (!two && src[end + 1] === ch))) {
          end = src.indexOf(mark, end + mark.length + (!two && src[end + 1] === ch ? 1 : 0));
        }
        if (end > 0) {
          flush();
          out.push({ t: two ? 'b' : 'i', c: inline(src.slice(i + mark.length, end)) });
          i = end + mark.length;
          continue;
        }
      }
    }
    if (ch === '[') {
      const close = bracketEnd(src, i);
      if (close > 0 && src[close + 1] === '(') {
        const end = src.indexOf(')', close + 2);
        if (end > 0) {
          flush();
          const href = src.slice(close + 2, end).trim().split(/\s+/)[0] ?? '';
          out.push({ t: 'a', href, c: inline(src.slice(i + 1, close)) });
          i = end + 1;
          continue;
        }
      }
    }
    if (ch === '<') {
      const m = /^<(https?:\/\/[^>\s]+)>/.exec(src.slice(i));
      if (m) {
        flush();
        out.push({ t: 'a', href: m[1], c: [{ t: 'text', v: m[1] }] });
        i += m[0].length;
        continue;
      }
    }
    text += ch;
    i += 1;
  }
  flush();
  return out;
}

function bracketEnd(src: string, open: number): number {
  let depth = 0;
  for (let i = open; i < src.length; i += 1) {
    if (src[i] === '\\') i += 1;
    else if (src[i] === '`') {
      const end = src.indexOf('`', i + 1);
      if (end < 0) return -1;
      i = end;
    } else if (src[i] === '[') depth += 1;
    else if (src[i] === ']') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

export const plain = (c: Inline[]): string => c.map((x) => (x.t === 'text' || x.t === 'code' ? x.v : plain(x.c))).join('');

// The anchor of a heading, the way GitHub cuts it, so the docs' own "#section" links work in both places.
export function slug(text: string, seen: Map<string, number>): string {
  const base = text.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
  const n = seen.get(base) ?? 0;
  seen.set(base, n + 1);
  return n ? `${base}-${n}` : base;
}

// --- blocks ---
const FENCE = /^\s{0,3}(```+|~~~+)\s*([^\s`]*)/;
const HEAD = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const LI = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const SEP = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

function cells(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let code = false;
  const s = line.trim().replace(/^\|/, '');
  for (let i = 0; i < s.length; i += 1) {
    const ch = s[i];
    if (ch === '\\' && s[i + 1] === '|') {
      cur += '|';
      i += 1;
    } else if (ch === '`') {
      code = !code;
      cur += ch;
    } else if (ch === '|' && !code) {
      out.push(cur.trim());
      cur = '';
    } else cur += ch;
  }
  if (cur.trim() !== '') out.push(cur.trim());
  return out;
}

const startsBlock = (line: string, next: string | undefined): boolean =>
  FENCE.test(line) || HEAD.test(line) || RULE.test(line) || LI.test(line) || /^\s{0,3}>/.test(line) || (line.includes('|') && next !== undefined && SEP.test(next) && next.includes('-'));

export function parse(text: string): Block[] {
  const seen = new Map<string, number>();
  const lines = text.replace(/\r\n?/g, '\n').replace(/<!--[\s\S]*?-->/g, '').split('\n');
  return blocks(lines, seen);
}

function blocks(lines: string[], seen: Map<string, number>): Block[] {
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === '') {
      i += 1;
      continue;
    }
    const fence = FENCE.exec(line);
    if (fence) {
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !new RegExp(`^\\s{0,3}${fence[1][0]}{${fence[1].length},}\\s*$`).test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1;
      out.push({ t: 'code', lang: fence[2] ?? '', v: body.join('\n') });
      continue;
    }
    const head = HEAD.exec(line);
    if (head) {
      const c = inline(head[2]);
      const t = plain(c);
      out.push({ t: 'h', level: head[1].length, id: slug(t, seen), c, text: t });
      i += 1;
      continue;
    }
    if (RULE.test(line)) {
      out.push({ t: 'hr' });
      i += 1;
      continue;
    }
    if (/^\s{0,3}>/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^\s{0,3}>/.test(lines[i])) {
        body.push(lines[i].replace(/^\s{0,3}>\s?/, ''));
        i += 1;
      }
      out.push({ t: 'quote', c: blocks(body, seen) });
      continue;
    }
    if (line.includes('|') && i + 1 < lines.length && SEP.test(lines[i + 1]) && lines[i + 1].includes('-')) {
      const head2 = cells(line);
      const align = cells(lines[i + 1]).map((c) => (c.startsWith(':') && c.endsWith(':') ? 'c' : c.endsWith(':') ? 'r' : 'l'));
      const rows: Inline[][][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim() !== '' && lines[i].includes('|')) {
        const row = cells(lines[i]);
        while (row.length < head2.length) row.push('');
        rows.push(row.slice(0, head2.length).map(inline));
        i += 1;
      }
      out.push({ t: 'table', align, head: head2.map(inline), rows });
      continue;
    }
    const li = LI.exec(line);
    if (li) {
      const [list, next] = listAt(lines, i, seen);
      out.push(list);
      i = next;
      continue;
    }
    const para: string[] = [line.trim()];
    i += 1;
    while (i < lines.length && lines[i].trim() !== '' && !startsBlock(lines[i], lines[i + 1])) {
      para.push(lines[i].trim());
      i += 1;
    }
    out.push({ t: 'p', c: inline(para.join(' ')) });
  }
  return out;
}

const indentOf = (line: string): number => /^\s*/.exec(line)![0].replace(/\t/g, '    ').length;

// One list: the items at this indent; whatever sits deeper under an item (more text, a nested list, a code block)
// belongs to that item. A blank line ends the list unless an indented line or another item of it follows.
function listAt(lines: string[], from: number, seen: Map<string, number>): [Block, number] {
  const first = LI.exec(lines[from])!;
  const base = indentOf(lines[from]);
  const ordered = /\d/.test(first[2]);
  const items: Item[] = [];
  let i = from;
  while (i < lines.length) {
    const m = LI.exec(lines[i]);
    if (!m || indentOf(lines[i]) !== base || /\d/.test(m[2]) !== ordered) break;
    const lead: string[] = [m[3]];
    const body: string[] = [];
    const inner = base + m[2].length + 1;
    i += 1;
    while (i < lines.length) {
      const l = lines[i];
      if (l.trim() === '') {
        let j = i + 1;
        while (j < lines.length && lines[j].trim() === '') j += 1;
        if (j < lines.length && indentOf(lines[j]) > base) {
          body.push('');
          i += 1;
          continue;
        }
        break;
      }
      const ind = indentOf(l);
      if (ind <= base) {
        // a lazy continuation of the item's own paragraph, unless a new block starts here
        if (body.length || startsBlock(l, lines[i + 1])) break;
        lead.push(l.trim());
        i += 1;
        continue;
      }
      if (!body.length && !startsBlock(l, lines[i + 1])) lead.push(l.trim());
      else body.push(l.slice(Math.min(ind, inner)));
      i += 1;
    }
    let textLine = lead.join(' ');
    let box: boolean | null = null;
    const bm = /^\[([ xX])\]\s+/.exec(textLine);
    if (bm) {
      box = bm[1] !== ' ';
      textLine = textLine.slice(bm[0].length);
    }
    items.push({ c: inline(textLine), box, sub: body.length ? blocks(body, seen) : [] });
    while (i < lines.length && lines[i].trim() === '') {
      let j = i;
      while (j < lines.length && lines[j].trim() === '') j += 1;
      const mm = j < lines.length ? LI.exec(lines[j]) : null;
      if (mm && indentOf(lines[j]) === base && /\d/.test(mm[2]) === ordered) i = j;
      else break;
    }
  }
  return [{ t: 'list', ordered, start: ordered ? parseInt(first[2], 10) || 1 : 1, items }, i];
}

// Where a link in a doc leads inside the wiki: another page ("tuning.md#the-gate", "docs/tuning.md", "../README.md"),
// a section of this page ("#the-gate"), the web ("https://..."), or nowhere the app can show (a file in the repo).
export type Target = { kind: 'page'; page: string; hash: string } | { kind: 'hash'; hash: string } | { kind: 'web'; href: string } | { kind: 'file'; path: string };
export function target(href: string, pages: string[]): Target {
  if (/^https?:\/\//i.test(href)) return { kind: 'web', href };
  if (/^[a-z][a-z0-9+.-]*:/i.test(href)) return { kind: 'file', path: href };
  if (href.startsWith('#')) return { kind: 'hash', hash: href.slice(1) };
  const [path, hash = ''] = href.split('#');
  const m = /^(?:\.\.?\/)*(?:docs\/)?([A-Za-z0-9-]+)\.md$/.exec(path);
  const id = m ? m[1].toLowerCase() : null;
  if (id && pages.includes(id)) return { kind: 'page', page: id, hash };
  return { kind: 'file', path };
}
