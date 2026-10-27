import type { PartHistory, ReadingsMeta } from '../../api';
import { short, stamp, weekday } from '../../format';
import Chip from './Chip';

// A control's run of weeks, from the weekly readings (data/readings/<sunday>.json). Colour marks a change only: a
// control that reads differently from last week gets the coloured "new this week" chip with what it read before; a
// state that holds is a grey "3rd week"; with no reading of the week before there is nothing to compare, and the card
// says so once (historyNote) instead of every cell.

// 1st, 2nd, 3rd, 4th ... 11th, 12th, 13th, 21st
export function ordinal(n: number): string {
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th';
  return `${n}${suffix}`;
}

// What changed, shortest first: in states made of parts ("lr fit · gb fit ..." or "beat GARCH: none · GARCH vs uniform:
// better") only the parts that differ, as "lr fit → inconcl."; a one-part state as "was within".
function partChange(before: string, now: string): string {
  const colon = before.indexOf(': ');
  if (colon > 0 && now.startsWith(before.slice(0, colon + 2))) return `${before.slice(0, colon + 2)}${before.slice(colon + 2)} → ${now.slice(colon + 2)}`;
  const space = before.indexOf(' ');
  if (space > 0 && now.startsWith(before.slice(0, space + 1))) return `${before} → ${now.slice(space + 1)}`;
  return `${before} → ${now}`;
}
export function changeText(h: PartHistory): string {
  if (h.last === null) return 'no record the week before';
  const before = h.last.split(' · ');
  const now = h.state.split(' · ');
  if (before.length > 1 && before.length === now.length) {
    const parts = now.map((x, i) => (x === before[i] ? '' : partChange(before[i], x))).filter(Boolean);
    if (parts.length) return parts.join(' · ');
  }
  return `was ${h.last}`;
}

// The words for a cell: changed (the cell's colour), the streak ("3rd week above", "new this week · was within") and,
// for a record older than the other parts', its own Sunday. null when there is nothing to say: the first week on
// record is said once per card (historyNote). A week before whose reading holds an older record of this control (it
// did not run that week) is said, grey: it ends the run, it is not a change.
export function streakWords(h: PartHistory | null | undefined, what = ''): { changed: boolean; text: string; behind: string } | null {
  if (!h) return null;
  const behind = h.behind ? `record of ${weekday(h.sunday)}, older than the others` : '';
  if (h.changed === true) return { changed: true, text: `new this week · ${changeText(h)}`, behind };
  if (h.weeks >= 2) return { changed: false, text: `${ordinal(h.weeks)} week${what ? ` ${what}` : ''}`, behind };
  if (h.noRunBefore) return { changed: false, text: 'no run the week before: nothing to compare', behind };
  return behind ? { changed: false, text: '', behind } : null;
}

// Q0's row tests come with the daily report, and its weeks are counted on the weekly readings, each fixed at its own
// Sunday's report (27 Oct 2026: the rows confirmed on new draws and the rows followed). The line under a Q0 figure that
// says so: the reading of its Sunday and what today's report reads differently (+ a row followed or confirmed only now,
// − one only then), or that there is no reading of that Sunday yet, so the state is the report on disk. '' when the
// report reads the same as the reading.
export function readingNote(h: PartHistory | null | undefined, rankingAt: string | null): string {
  if (!h) return '';
  if (h.live) return `no reading of ${weekday(h.sunday)} yet: the report of ${stamp(rankingAt)}`;
  const s = h.sinceReading;
  if (!s) return '';
  const rows = [...s.added.map((r) => `+${short(r)}`), ...s.dropped.map((r) => `−${short(r)}`)].join(', ');
  return `reading of ${weekday(h.sunday)}: ${h.state} · the report of ${stamp(rankingAt)}: ${rows || s.now}`;
}

// Q2 only (27 Oct 2026): the week before was read under another verdict rule - 3 seeds (5 for the markets) and a 0.8
// ceiling until 25 Oct 2026, 20 seeds and the windows rule from 1 Nov - so its run of weeks starts again, and a change
// from that week may be the rule's rather than the draws'. '' when the rule is the same.
export function ruleNote(h: PartHistory | null | undefined): string {
  const r = h?.ruleChanged;
  if (!r) return '';
  return `read under the ${r.seeds !== null ? `${r.seeds}-seed` : 'old'} rule the week before`;
}

// The chip and the words that go with it (the Controls page): a coloured "new this week" with what it read before, or
// a grey "3rd week"; for Q2, the week before read under another rule.
export default function Streak({ h, what = '' }: { h: PartHistory | null | undefined; what?: string }) {
  const w = streakWords(h, what);
  const rule = ruleNote(h);
  if ((!w && !rule) || !h) return null;
  return (
    <span className="streak">
      {w?.changed ? <Chip kind="chg">new this week</Chip> : w?.text ? <Chip kind="flat">{w.text}</Chip> : null}
      {w?.changed ? <span className="was">{changeText(h)}</span> : null}
      {rule ? <span className="was">{rule}</span> : null}
      {w?.behind ? <span className="was">{w.behind}</span> : null}
    </span>
  );
}

// One line per card: whose weeks the chips count, and - when no cell can be compared yet - why there are none.
export function historyNote(readings: ReadingsMeta | null, histories: (PartHistory | null | undefined)[]): string {
  const shown = histories.filter((h): h is PartHistory => Boolean(h));
  if (!readings) return '';
  const stored = readings.stored === 1 ? '1 stored weekly reading' : `${readings.stored} stored weekly readings`;
  const unreadable = readings.unreadable.length ? ` · ${readings.unreadable.length} unreadable (${readings.unreadable.join(', ')})` : '';
  if (shown.length && shown.every((h) => h.changed === null)) {
    const sundays = [...new Set(shown.map((h) => h.sunday).filter(Boolean))] as string[];
    return `week over week: ${stored}${unreadable} · no reading of the week before ${sundays.map(weekday).join(', ')}, so no control can be compared yet`;
  }
  return `week over week: ${stored}${unreadable} · colour only where a control reads differently from the week before`;
}
