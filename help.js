// help.js - the "?" every page keeps its explanations behind (13 Oct 2026, at
// the owner's request: the pages show the data, the prose that explains how
// to read it opens on demand).
//
// help(body, label) returns a <details class="help"> block: a small round "?"
// that opens the explanation underneath. It is plain HTML - no script, no ids
// - so it works inside a card body, inside the market page's Background
// sections and in a table cell alike, and the explanation stays in the page
// source (searchable, and the text the tests read is unchanged). A help block
// is block-level: it may not sit inside a <p>; put it before or after one.
//
// What goes behind a "?": how to read a table or chart, definitions, method,
// caveats and history notes. What stays visible: the data itself, headline
// results, the plan's orders, and a warning a reader must see before acting
// (a one-line version of it, with the reasoning behind the "?").
'use strict';

const escAttr = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const HELP_CSS = `
      /* THE "?" EXPLANATIONS (help.js) */
      details.help { margin: 6px 0 10px; }
      details.help > summary {
        list-style: none; display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none;
        color: #7f8c8d; font-size: 0.85em;
      }
      details.help > summary::-webkit-details-marker { display: none; }
      details.help > summary::marker { content: ''; }
      details.help > summary .help-q {
        display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%;
        background: #eaf2fb; color: #2980b9; border: 1px solid #aed6f1; font-weight: bold; font-size: 0.95em; flex: none;
      }
      details.help > summary:hover .help-q { background: #d6eaf8; }
      details.help[open] > summary .help-q { background: #2980b9; color: white; border-color: #2980b9; }
      details.help > .help-body {
        margin-top: 6px; padding: 10px 14px; background: #f8fbfe; border-left: 3px solid #aed6f1; border-radius: 4px;
        color: #555; font-size: 0.9em; line-height: 1.45; white-space: normal;
      }
      details.help > .help-body > p:first-child { margin-top: 0; }
      details.help > .help-body > p:last-child { margin-bottom: 0; }
      details.help.help-inline { display: inline-block; margin: 0 0 0 6px; vertical-align: middle; }
      details.help.help-inline[open] { display: block; margin: 6px 0; }
      th details.help.help-inline > .help-body, td details.help.help-inline > .help-body { text-align: left; font-weight: normal; max-width: 520px; }
`;

// body: HTML (already escaped by the caller, like every other card text).
// label: a few words shown next to the "?" (optional; the hover title says
// the same) - e.g. "how this is scored". inline: true for a "?" next to a
// heading or inside a table cell.
function help(body, label = '', { inline = false } = {}) {
  if (body === null || body === undefined || String(body).trim() === '') return '';
  const title = label || 'What this means';
  return `<details class="help${inline ? ' help-inline' : ''}"><summary title="${escAttr(title)}" aria-label="${escAttr(title)}"><span class="help-q">?</span>${label ? `<span>${escAttr(label)}</span>` : ''}</summary><div class="help-body">${body}</div></details>`;
}

// autoReload(ms): the self-refresh some pages run (the Jobs page every 15 s,
// a page with a pipeline run in progress every 60 s), postponed while a "?"
// is open - a reload would snap the explanation shut under the reader.
function autoReload(ms) {
  const wait = Math.max(1000, Number(ms) || 15000);
  return `<script>(function wait() { setTimeout(function () { if (document.querySelector('details.help[open]')) { wait(); } else { location.reload(); } }, ${wait}); })();</script>`;
}

module.exports = { help, HELP_CSS, autoReload };
