import { short, stamp } from '../../format';
import type { Coming, DrawScope, RankVersion } from '../../workbench';
import { prettyDates, type Mode } from './text';

// Whose rows a table shows and which ranking picked them: the line over every scored table. A past draw is read
// against the newest ranking saved before it (rankhistory.js, never today's); the coming draw against today's.

const names = (xs: string[]) => xs.map(short).join(', ');
const saved = (r: RankVersion) => `saved ${stamp(r.committedAt ?? r.generatedAt)}`;

export default function ScopeLine({ scope, mode }: { scope: DrawScope; mode: Mode }) {
  const r = scope.ranking;
  const none = scope.none ? prettyDates(scope.none.text) : null;
  if (mode === 'user') {
    if (scope.scope === 'top' && r) {
      return (
        <p className="scopeline">
          The <b>{scope.rows}</b> models with the best record by {r.metricText}, as ranked on {stamp(r.committedAt ?? r.generatedAt)}, before this draw.
        </p>
      );
    }
    // a user never gets every row: a draw without its ranking has no rows at all, and says why
    return (
      <p className="scopeline">
        <b>No top 5 for this draw</b>: {none ?? 'no ranking was saved before it'}.
      </p>
    );
  }
  if (scope.scope === 'top') {
    if (!r || none) {
      return (
        <p className="scopeline">
          <b>No Top 5 for this draw</b>: {none ?? 'no ranking before it'}. All shows its {scope.of} rows.
        </p>
      );
    }
    return (
      <p className="scopeline">
        <b>
          Top 5 as ranked before this draw: {scope.rows} of {scope.of} rows
        </b>{' '}
        · by {r.metricText}, the ranking {saved(r)} (the newest saved before {stamp(scope.cutoff)}) · ranked from {r.minDraws ?? '—'} draws
      </p>
    );
  }
  return (
    <p className="scopeline">
      <b>All {scope.of} rows</b>
      {r && scope.top.length ? (
        <>
          {' '}
          · top 5 before this draw ({r.metricText}, ranking {saved(r)}): {names(scope.top)}
        </>
      ) : (
        <> · {none ?? 'no ranking before this draw'}: no rank or top 5 marked</>
      )}
    </p>
  );
}

export function ComingScopeLine({ scope, mode }: { scope: Coming['scope']; mode: Mode }) {
  const r = scope.ranking;
  const none = scope.none ? prettyDates(scope.none.text) : null;
  if (mode === 'user' || scope.scope === 'top') {
    if (none || !r) return <p className="scopeline">{none ?? 'No ranking today'}: no top 5 for the coming draw.</p>;
    return (
      <p className="scopeline">
        <b>
          {mode === 'user' ? `The ${scope.rows} models with the best record` : `Today's top 5: ${scope.rows} of ${scope.of} rows`}
        </b>{' '}
        · by {r.metricText}, the ranking generated {stamp(r.generatedAt)}
        {mode === 'admin' ? ' · the same five a user sees' : ''}
      </p>
    );
  }
  return (
    <p className="scopeline">
      <b>All {scope.of} rows</b>
      {r && scope.top.length ? (
        <>
          {' '}
          · today&apos;s top 5 ({r.metricText}, generated {stamp(r.generatedAt)}, what a user sees): {names(scope.top)}
        </>
      ) : (
        <> · {none ?? 'no ranking today'}</>
      )}
    </p>
  );
}
