'use strict';
// Admin › Wiki (docs.js): which files the wiki lists and serves, on a stand-in folder; the markdown reader the page
// uses (web/src/wiki/markdown.ts, when the web app's build tools are installed); and - read-only - the checkout's own
// documentation: every page listed, readable, with links that lead somewhere, and written without naming a person.
//   node test/docs.test.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const docs = require('../docs');

let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// --- the list and a page, on a stand-in folder ----------------------------------------------------------------------
{
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'docs-test-'));
  ok(docs.list(root).pages.length === 0, 'no README and no docs folder: an empty list');
  fs.writeFileSync(path.join(root, 'README.md'), '# The project\n\nHello.\n');
  fs.mkdirSync(path.join(root, 'docs'));
  for (const [name, body] of [['tuning.md', '# Tuning\n'], ['changelog.md', '# Changelog\n'], ['models.md', 'intro\n\n# Models and rows\n'],
    ['zebra.md', 'no heading here\n'], ['alpha.md', '# Alpha\n'], ['notes.txt', '# Not markdown\n'], ['Bad Name.md', '# Bad\n'], ['readme.md', '# Shadow\n']]) {
    fs.writeFileSync(path.join(root, 'docs', name), body);
  }
  fs.mkdirSync(path.join(root, 'docs', 'folder.md'));
  fs.writeFileSync(path.join(root, 'secret.md'), '# Not in docs\n');
  const l = docs.list(root);
  ok(l.name === 'sequencePredictor Wiki', 'the wiki\'s name');
  ok(JSON.stringify(l.pages.map((p) => p.id)) === JSON.stringify(['readme', 'models', 'tuning', 'alpha', 'zebra', 'changelog']),
    `the overview first, the known pages in the menu's order, others by name, the changelog last (${l.pages.map((p) => p.id)})`);
  ok(l.pages[0].title === 'Overview' && l.pages[0].file === 'README.md', 'README.md is the Overview page');
  ok(l.pages[1].title === 'Models and rows', 'a title is the first "# " line');
  ok(l.pages.find((p) => p.id === 'zebra').title === 'zebra', 'no heading: the file\'s name');
  ok(l.pages.every((p) => p.bytes > 0 && /^\d{4}-\d\d-\d\dT/.test(p.updatedAt)), 'size and last change on every entry');
  ok(docs.page(root, 'tuning').text === '# Tuning\n' && docs.page(root, 'tuning').file === path.join('docs', 'tuning.md'), 'a page: its text and file');
  ok(docs.page(root, 'readme').text.startsWith('# The project'), 'readme is README.md, not docs/readme.md');
  for (const bad of ['secret', '../secret', '..', 'docs/tuning', 'tuning.md', 'notes', 'Bad Name', 'folder', '', 'TUNING', '%2e%2e', null, undefined, 7, {}]) {
    ok(docs.page(root, bad) === null, `not a page: ${JSON.stringify(bad)}`);
  }
  // the routes: admin only (the guard is in front of both), 404 for a page that is not listed
  const routes = {};
  const guard = () => {};
  docs.install({ get: (p, g, h) => { ok(g === guard, `${p}: behind the guard`); routes[p] = h; } },
    { root, guard, safe: (fn) => fn, json: (res, body, status) => { res.body = body; res.status = status || 200; } });
  const call = (p, params) => { const res = {}; routes[p]({ params: params || {} }, res); return res; };
  ok(call('/api/admin/docs').body.pages.length === 6, 'GET /api/admin/docs: the list');
  ok(call('/api/admin/docs/:page', { page: 'alpha' }).body.text === '# Alpha\n', 'GET /api/admin/docs/alpha: the text');
  const miss = call('/api/admin/docs/:page', { page: '../secret' });
  ok(miss.status === 404 && miss.body.error === 'no such page', 'a page outside the list: 404');
  fs.rmSync(root, { recursive: true, force: true });
}

// --- the markdown reader ----------------------------------------------------------------------------------------------
let md = null;
try {
  const esbuild = require(path.join(__dirname, '..', 'web', 'node_modules', 'esbuild'));
  const src = fs.readFileSync(path.join(__dirname, '..', 'web', 'src', 'wiki', 'markdown.ts'), 'utf-8');
  const m = { exports: {} };
  new Function('module', 'exports', esbuild.transformSync(src, { loader: 'ts', format: 'cjs' }).code)(m, m.exports);
  md = m.exports;
} catch (e) {
  console.log(`docs.test.js: the markdown reader's checks are skipped (${String(e.message).split('\n')[0]})`);
}
if (md) {
  const { parse, inline, plain, target } = md;
  const b = parse(['# Title', '', 'One *two* **three** `four_five` snake_case_name and [a link](tuning.md#the-gate).', 'Same paragraph.', '',
    '## The gate', '<!-- entry 2026-11-15 tuning -->', '- first', '  continues', '- second', '  - nested `x`', '  - nested two', '- [x] done', '- [ ] open', '',
    '1. one', '2. two', '', '| Name | Value |', '|:-----|------:|', '| `a\\|b` | 1 |', '| c | 2 | extra |', '', '```bash', 'npm test', '# not a heading', '```', '',
    '> quoted **text**', '', '---', '## The gate'].join('\n'));
  ok(JSON.stringify(b.map((x) => x.t)) === JSON.stringify(['h', 'p', 'h', 'list', 'list', 'table', 'code', 'quote', 'hr', 'h']), `the blocks in order (${b.map((x) => x.t)})`);
  ok(b[0].level === 1 && b[0].id === 'title' && b[2].id === 'the-gate' && b[9].id === 'the-gate-1', 'heading anchors, a repeated one numbered');
  ok(JSON.stringify(b[1].c.map((x) => x.t)) === JSON.stringify(['text', 'i', 'text', 'b', 'text', 'code', 'text', 'a', 'text']), 'inline: italic, bold, code, a link');
  ok(plain(b[1].c) === 'One two three four_five snake_case_name and a link. Same paragraph.', 'snake_case is not emphasis; two lines are one paragraph');
  ok(b[1].c[7].href === 'tuning.md#the-gate', 'the link\'s address');
  ok(b[3].items.length === 4 && plain(b[3].items[0].c) === 'first continues', 'list items, a continued line joined');
  ok(b[3].items[1].sub.length === 1 && b[3].items[1].sub[0].t === 'list' && b[3].items[1].sub[0].items.length === 2, 'a nested list under its item');
  ok(b[3].items[2].box === true && b[3].items[3].box === false && b[3].items[0].box === null && plain(b[3].items[2].c) === 'done', 'check boxes');
  ok(b[4].ordered === true && b[4].items.length === 2 && b[3].ordered === false, 'an ordered list');
  ok(b[5].head.length === 2 && b[5].rows.length === 2 && b[5].align.join('') === 'lr', 'a table: head, rows, alignment');
  ok(b[5].rows[0][0][0].t === 'code' && b[5].rows[0][0][0].v === 'a|b' && b[5].rows[1].length === 2, 'an escaped pipe stays in its cell; an extra cell is dropped');
  ok(b[6].lang === 'bash' && b[6].v === 'npm test\n# not a heading', 'a code block keeps its lines as they are');
  ok(b[7].c[0].t === 'p' && plain(b[7].c[0].c) === 'quoted text', 'a quote');
  ok(!JSON.stringify(b).includes('entry 2026'), 'an HTML comment is skipped');
  ok(plain(inline('2 * 3 * 4 and a_b_c and **bold `code`**')) === '2 * 3 * 4 and a_b_c and bold code', 'loose stars and underscores are text');
  ok(JSON.stringify(inline('<script>alert(1)</script>')) === JSON.stringify([{ t: 'text', v: '<script>alert(1)</script>' }]), 'HTML in a doc is plain text');
  const pages = ['readme', 'tuning', 'changelog'];
  ok(JSON.stringify(target('tuning.md#the-gate', pages)) === JSON.stringify({ kind: 'page', page: 'tuning', hash: 'the-gate' }), 'a link to another page');
  ok(target('docs/tuning.md', pages).page === 'tuning' && target('../README.md', pages).page === 'readme' && target('README.md', pages).page === 'readme', 'docs/ and ../ forms');
  ok(target('#the-gate', pages).kind === 'hash' && target('https://example.org/x', pages).kind === 'web', 'a section, the web');
  ok(target('worker/README.md', pages).kind === 'file' && target('nope.md', pages).kind === 'file' && target('javascript:alert(1)', pages).kind === 'file', 'anything else is not a link');
}

// --- the checkout's own documentation (read-only) -------------------------------------------------------------------
{
  const root = path.join(__dirname, '..');
  const l = docs.list(root);
  const ids = l.pages.map((p) => p.id);
  for (const id of ['readme', 'models', 'scoring', 'tuning', 'research', 'markets', 'operations', 'roadmap', 'changelog']) ok(ids.includes(id), `the wiki has the ${id} page`);
  ok(ids[0] === 'readme' && ids[ids.length - 1] === 'changelog', 'the overview first, the changelog last');
  for (const p of l.pages) {
    const text = docs.page(root, p.id).text;
    ok(text.length > 200 && p.title !== p.id, `${p.file}: has text and a title`);
    // the documentation names no person: decisions are recorded as decisions ("Decided: ..."), not as somebody's
    const personal = text.split('\n').map((line, i) => [i + 1, line]).filter(([, line]) => /\bowner(?:'s|s)?\b/i.test(line));
    ok(personal.length === 0, `${p.file}: no reference to a person (line ${personal.slice(0, 5).map(([n]) => n).join(', ')})`);
    if (!md) continue;
    const blocks = md.parse(text);
    const anchors = (bs) => bs.flatMap((x) => (x.t === 'h' ? [x.id] : x.t === 'quote' ? anchors(x.c) : x.t === 'list' ? x.items.flatMap((it) => anchors(it.sub)) : []));
    p.anchors = anchors(blocks);
    p.blocks = blocks;
  }
  if (md) {
    const links = (c) => c.flatMap((x) => (x.t === 'a' ? [x.href, ...links(x.c)] : x.t === 'b' || x.t === 'i' ? links(x.c) : []));
    const walk = (bs) => bs.flatMap((x) => (x.t === 'h' || x.t === 'p' ? links(x.c) : x.t === 'quote' ? walk(x.c) : x.t === 'list' ? x.items.flatMap((it) => [...links(it.c), ...walk(it.sub)])
      : x.t === 'table' ? [...x.head, ...x.rows.flat()].flatMap(links) : []));
    for (const p of l.pages) {
      const dead = [];
      for (const href of walk(p.blocks)) {
        const t = md.target(href, ids);
        if (t.kind === 'hash' && !p.anchors.includes(t.hash)) dead.push(href);
        if (t.kind === 'page' && t.hash && !l.pages.find((q) => q.id === t.page).anchors.includes(t.hash)) dead.push(href);
        if (t.kind === 'file' && !/^[a-z][a-z0-9+.-]*:/i.test(t.path) && !fs.existsSync(path.resolve(root, p.id === 'readme' ? '.' : 'docs', t.path))) dead.push(href);
      }
      ok(dead.length === 0, `${p.file}: every link leads somewhere (${dead.slice(0, 8).join(', ')})`);
    }
  }
}

console.log(`docs.test.js: ${checks} checks passed`);
