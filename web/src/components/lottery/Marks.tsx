import type { Backtest, Standing } from '../../workbench';

// The two marks beside a row, the best model or the best combination (24 Nov 2026: figures on chips instead of
// sentences; the sentences are the tooltips).
//   validated   the share of its backtested calendar months that ended in profit (roadmap item 13). Green when that is
//               more months than random tickets played the same way; "not validated" without a backtest record, "too slow to validate" for a row whose
//               backtest does not fit in the daily job.
//   held        a bar and the share of the checked draws on which it held the place, and the draws in a row up to the
//               coming one (roadmap item 16). `wide` adds the number of draws checked.
// An indicator, not proof: the tooltip carries the limits.
export default function Marks({ backtest, standing, wide = false, quiet = false }: { backtest: Backtest | null; standing: Standing | null; wide?: boolean; quiet?: boolean }) {
  const held = standing && standing.tracked && standing.score !== null && standing.streak !== null ? standing : null;
  if (!backtest && !standing) return null;
  return (
    <span className="marks">
      {backtest ? (
        backtest.pct !== null ? (
          <span className={`chip ${backtest.ahead ? 'up' : 'flat'}`} title={`${backtest.text}. ${backtest.detail}`}>
            {backtest.pct}% validated{backtest.changed ? ' *' : ''}
          </span>
        ) : (
          <span className="chip dim" title={`${backtest.text}. ${backtest.detail}`}>
            {backtest.slow ? 'too slow to validate' : 'not validated'}
          </span>
        )
      ) : null}
      {held ? (
        <>
          <span className="chip model" title={`${held.text}. ${held.detail === held.text ? '' : held.detail}`.trim()}>
            <i className="meter" style={{ ['--v' as string]: `${held.score}%` }} />
            {held.score}%{wide && held.of !== null ? ` · checked over ${held.of} draws` : ''}
          </span>
          <span className="chip flat" title={held.text}>
            {held.streak} in a row
          </span>
        </>
      ) : standing && !quiet ? (
        <span className="chip dim" title={standing.detail}>
          {standing.text}
        </span>
      ) : null}
    </span>
  );
}
