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

// --- the admin pages ------------------------------------------------------------

// An unsigned figure to fixed decimals, "1.126"; "—" for anything that is not a number.
export function fixed(v: unknown, decimals = 3): string {
  return isNum(v) ? v.toFixed(decimals).replace('-', MINUS) : DASH;
}

// A signed figure to fixed decimals, "+0.117" / "−0.195".
export function signedFixed(v: unknown, decimals = 3): string {
  return isNum(v) ? signed(v, decimals) : DASH;
}

// "Sun 18 Oct 12:11" in Belgian time; a date-only string stays a calendar day, "Sun 18 Oct".
export function stamp(s: unknown): string {
  const d = parse(s);
  if (!d) return DASH;
  const date = new Intl.DateTimeFormat('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: zone(s) }).format(d).replace(',', '');
  return typeof s === 'string' && DATE_ONLY.test(s) ? date : `${date} ${time(s)}`;
}

// A run from one instant to another: "Sun 18 Oct 10:48 → 11:02", the second date left out when it is the same day.
export function span(from: unknown, to: unknown): string {
  if (!parse(from)) return stamp(to);
  if (!parse(to) || from === to) return stamp(from);
  const sameDay = day(from) === day(to);
  return `${stamp(from)} → ${sameDay ? time(to) : stamp(to)}`;
}

// "1 Mar 2026": a calendar day with its year (windows that span years).
export function dayYear(s: unknown): string {
  const d = parse(s);
  if (!d) return DASH;
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: zone(s) }).format(d);
}

// Row names come in full ("Transformer Model"); the pages show them without the trailing " Model".
export function short(name: unknown): string {
  return typeof name === 'string' ? name.replace(/ Model$/, '') : DASH;
}

// What one scored draw of a game is, singular: draw, day, trading day or week.
export function perUnit(unit: unknown): string {
  if (unit === 'weeks') return 'week';
  if (unit === 'trading days') return 'trading day';
  if (unit === 'days') return 'day';
  return 'draw';
}

// "14 draws", "1 week", "189 trading days": a count with its game's unit.
export function count(n: unknown, unit: unknown): string {
  if (!isNum(n)) return DASH;
  const k = Math.round(n);
  const one = perUnit(unit);
  return `${k} ${k === 1 ? one : `${one}s`}`;
}

// A range of numbers, "5-9", or the one number when both ends agree.
export function range(r: { min: number; max: number } | null | undefined, decimals = 0): string {
  if (!r || !isNum(r.min) || !isNum(r.max)) return DASH;
  return r.min === r.max ? r.min.toFixed(decimals) : `${r.min.toFixed(decimals)}-${r.max.toFixed(decimals)}`;
}

// "Sun 18 Oct": the weekday and the day of an instant in Belgian time, or of a calendar day.
export function weekday(s: unknown): string {
  const d = parse(s);
  if (!d) return DASH;
  return new Intl.DateTimeFormat('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: zone(s) }).format(d).replace(',', '');
}
