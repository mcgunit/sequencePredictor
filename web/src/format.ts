// Formatting helpers. Every helper accepts unknown input and answers "—" for
// anything that is not a finite number or a parsable date, so the page never
// shows "null", "undefined" or "NaN".

const MINUS = '−';
const THIN = ' ';
const TZ = 'Europe/Brussels';
const DATE_ONLY = /^\d{4}-\d{2}-\d{2}$/;

export const DASH = '—';

export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

// "+1.23", "−1.23" or "0.00" (rounding first so −0.001 does not become "−0.00").
function signed(v: number, decimals: number): string {
  const r = Number(v.toFixed(decimals));
  const body = Math.abs(r).toFixed(decimals);
  return r > 0 ? `+${body}` : r < 0 ? `${MINUS}${body}` : body;
}

export function money(v: unknown, decimals = 2): string {
  return isNum(v) ? signed(v, decimals) : DASH;
}

export function unsigned(v: unknown, decimals = 2): string {
  return isNum(v) ? Math.abs(v).toFixed(decimals) : DASH;
}

export function pct(v: unknown, decimals = 1): string {
  return isNum(v) ? `${signed(v, decimals)}%` : DASH;
}

// A 0..1 share as an unsigned percentage, "53.6%".
export function share(v: unknown): string {
  return isNum(v) ? `${(v * 100).toFixed(1)}%` : DASH;
}

export function int(v: unknown): string {
  return isNum(v) ? String(Math.round(v)) : DASH;
}

// A price: grouped thousands, decimals by size (82 310 · 2 536.94 · 0.5123).
export function rate(v: unknown): string {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  const d = a >= 10000 ? 0 : a >= 1 ? 2 : 4;
  return v
    .toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })
    .replace(/,/g, THIN)
    .replace('-', MINUS);
}

export function signClass(v: unknown): 'pos' | 'neg' | 'mut' {
  return isNum(v) ? (v > 0 ? 'pos' : v < 0 ? 'neg' : 'mut') : 'mut';
}

export function plural(n: unknown, word: string): string {
  const k = isNum(n) ? Math.round(n) : 0;
  return `${k} ${word}${k === 1 ? '' : 's'}`;
}

export function cap(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function parse(s: unknown): Date | null {
  if (typeof s !== 'string' || !s) return null;
  const d = new Date(DATE_ONLY.test(s) ? `${s}T12:00:00Z` : s);
  return Number.isNaN(d.getTime()) ? null : d;
}

// A date-only string is a calendar day and stays one; a full timestamp is
// shown in Belgian time.
function zone(s: unknown): string {
  return typeof s === 'string' && DATE_ONLY.test(s) ? 'UTC' : TZ;
}

// "14 Oct"
export function day(s: unknown): string {
  const d = parse(s);
  if (!d) return DASH;
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', timeZone: zone(s) }).format(d);
}

// "20:31" in Belgian time, or "" when there is no timestamp.
export function time(s: unknown): string {
  const d = parse(s);
  if (!d) return '';
  return new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TZ }).format(d);
}

// Axis labels: "1 Oct", "2", "3", … and the month again when it changes.
export function axisDates(dates: unknown[]): string[] {
  let lastMonth = '';
  return dates.map((s) => {
    const d = parse(s);
    if (!d) return '';
    const month = new Intl.DateTimeFormat('en-GB', { month: 'short', timeZone: zone(s) }).format(d);
    const dayNo = new Intl.DateTimeFormat('en-GB', { day: 'numeric', timeZone: zone(s) }).format(d);
    if (month === lastMonth) return dayNo;
    lastMonth = month;
    return `${dayNo} ${month}`;
  });
}

export function esc(s: unknown): string {
  return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c);
}
