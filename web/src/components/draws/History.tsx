import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { count, dayYear, perUnit, short, stamp } from '../../format';
import type { DayFigures, DayLine, History, Month, RankVersion, Shape } from '../../workbench';
import ResultStrip from './ResultStrip';
import { bestMeaning, bestText, eur, eurU, figureLines, monthTitle, prettyDates, type Mode } from './text';

// One game's history of draws (the old /database/:folder page): a folding card per month, newest first and the newest
// open, each draw a line with its date (a link to the draw), the drawn row and its figures over the rows in scope.
// Admin mode carries the ranking version each Top line was read against; user mode (the users' Lottery) only the
// figures over the top five as they were known that day.

const NONE_SHORT: Record<string, string> = {
  'not-ranked': 'the game was not in the ranking yet',
  'no-ranked-row': 'no row of this draw was in the ranking then',
  unavailable: 'the ranking before this draw could not be read',
};
// why a draw has no Top: first is the day the first ranking was saved (25 Aug 2026 on production)
export function noneShort(code: string | null, first?: string | null): string {
  if (!code) return '';
  if (code === 'before-first') return `no ranking recorded before this draw${first ? ` (the first was saved on ${dayYear(first)})` : ''}`;
  return NONE_SHORT[code] ?? prettyDates(code);
}

// One draw's figures over the rows in scope: the figure in bold, then over how many rows and tickets (the admin pages
// colour no figure by its sign)
export function FiguresLine({ shape, figures, mode }: { shape: Shape; figures: Pick<DayFigures, 'rows' | 'tickets' | 'profit' | 'stake' | 'best' | 'multi'>; mode: Mode }) {
  const f = figureLines(shape, figures, mode);
  return (
    <span className="fig">
      <b>{f.main}</b>
      {f.sub.map((x) => (
        <small key={x}>{x}</small>
      ))}
    </span>
  );
}

export function DayRow({ d, shape, mode, scope, version, first, href }: { d: DayLine; shape: Shape; mode: Mode; scope: 'all' | 'top'; version: RankVersion | null; first: string | null; href: string }) {
  return (
    <div className="dl">
      <div className="d">
        <Link to={href}>{dayYear(d.date)}</Link>
        <small>{d.weekday}</small>
      </div>
      <ResultStrip shape={shape} result={d.result} compact />
      {d.figures ? (
        <span className="fig">
          <FiguresLine shape={shape} figures={d.figures} mode={mode} />
          {mode === 'admin' && scope === 'top' && version ? <small>top 5 by {version.metricText}, ranking saved {stamp(version.committedAt ?? version.generatedAt)}</small> : null}
        </span>
      ) : (
        <span className="fig mut">{scope === 'top' ? noneShort(d.none, first) : 'no scored row'}</span>
      )}
    </div>
  );
}

// a month's header: its totals over the counted draws, with the counts on the same line. Its rows are said per draw
// ("5 rows a draw"), as the day lines under it count them: the month's sum of rows over its draws is no row count.
function monthLine(m: Month, shape: Shape, unit: string, scope: 'all' | 'top', mode: Mode) {
  const parts: string[] = [];
  const drawsText = count(m.counted, unit);
  const lo = m.rowsMin ?? 0;
  const hi = m.rowsMax ?? 0;
  const rowsText = `${lo === hi ? hi : `${lo}–${hi}`} ${hi === 1 ? 'row' : 'rows'}${mode === 'user' ? ' shown' : ''} a ${perUnit(unit)}`;
  if (m.counted && shape.money) {
    parts.push(`${eur(m.profit)} over ${drawsText}`, `net over ${m.tickets} ${m.tickets === 1 ? 'ticket' : 'tickets'} of ${rowsText} · stake ${eurU(m.stake)}`);
    if (shape.family === 'jokerplus' && m.best) parts.push(`best ${bestText(shape, m.best)} ${bestMeaning(shape)} by ${short(m.best.by)}`);
  } else if (m.counted && m.best) {
    parts.push(`best ${bestText(shape, m.best)} over ${drawsText}`, `${bestMeaning(shape)}, by ${short(m.best.by)} · ${rowsText}`);
    if (shape.family === 'lotto' && m.multi !== null) parts.push(`best of 9 numbers (multi-pick): ${m.multi}`);
  } else if (m.counted) {
    parts.push(`${drawsText}, no scored row`);
  }
  if (scope === 'top' && m.counted && m.counted < m.draws) parts.push(`${m.counted} of ${count(m.draws, unit)} with a recorded ranking`);
  if (!m.counted) parts.push(scope === 'top' ? `${count(m.draws, unit)}, none with a recorded ranking` : `${count(m.draws, unit)}, no row in scope`);
  return parts;
}

export function MonthCard({
  m,
  shape,
  unit,
  scope,
  mode,
  versions,
  first,
  loadMonth,
  drawHref,
}: {
  m: Month;
  shape: Shape;
  unit: string;
  scope: 'all' | 'top';
  mode: Mode;
  versions: RankVersion[];
  first: string | null; // when the first ranking was saved
  loadMonth: (month: string) => Promise<History>;
  drawHref: (date: string) => string;
}) {
  const [open, setOpen] = useState(false);
  const [got, setGot] = useState<{ days: DayLine[]; versions: RankVersion[] } | null>(m.days ? { days: m.days, versions } : null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || got) return;
    let alive = true;
    setError(null);
    loadMonth(m.month)
      .then((h) => {
        const mm = h.months.find((x) => x.month === m.month);
        if (alive) setGot({ days: mm?.days ?? [], versions: h.ranking.versions });
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [open, got, m.month, loadMonth, attempt]);
  const id = `m-${m.month}`;
  return (
    <div id={id} className="card mcard">
      <h2 className="fold-t">
        <button type="button" className="fold-h" aria-expanded={open} aria-controls={`${id}-b`} onClick={() => setOpen((x) => !x)}>
          <span className="nm">{monthTitle(m.month)}</span>
          <span className="meta">
            {monthLine(m, shape, unit, scope, mode).map((x, i) => (
              <span key={i}>
                {i ? ' · ' : ''}
                {i === 0 ? <b>{x}</b> : x}
              </span>
            ))}
          </span>
        </button>
      </h2>
      {open ? (
        <div id={`${id}-b`} className="dlist">
          {got ? (
            got.days.map((d) => (
              <DayRow
                key={d.date}
                d={d}
                shape={shape}
                mode={mode}
                scope={scope}
                version={d.version !== null ? (got.versions[d.version] ?? null) : null}
                first={first}
                href={drawHref(d.date)}
              />
            ))
          ) : error ? (
            <div className="empty">
              <span>
                Could not load {monthTitle(m.month)}: <code>{error}</code>{' '}
                <button type="button" className="linkbtn" onClick={() => setAttempt((n) => n + 1)}>
                  Try again
                </button>
              </span>
            </div>
          ) : (
            <div className="empty">Loading…</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

// The whole history: the months as folding cards. The scope's months reload from the parent (key the view on scope).
export default function HistoryView({ h, mode, loadMonth, drawHref }: { h: History; mode: Mode; loadMonth: (month: string) => Promise<History>; drawHref: (date: string) => string }) {
  return (
    <div className="stack">
      {h.months.map((m) => (
        <MonthCard
          key={`${h.scope}-${m.month}`}
          m={m}
          shape={h.shape}
          unit={h.unit}
          scope={h.scope}
          mode={mode}
          versions={h.ranking.versions}
          first={h.ranking.first ? (h.ranking.first.committedAt ?? h.ranking.first.generatedAt) : null}
          loadMonth={loadMonth}
          drawHref={drawHref}
        />
      ))}
    </div>
  );
}
