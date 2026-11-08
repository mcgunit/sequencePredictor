import type { ReactNode } from 'react';

// A chip. The admin pages colour no state by itself: flat is a plain state, hi an outlined "above its band", dim a
// dashed "too few draws to judge", model a row name. Colour marks a change only: chg a control that reads
// differently from last week (the weekly readings), warn a run longer than its usual or a warning signature none of
// the job's earlier runs had, down a run that failed or an output that is missing. The words on the chip (or on the
// line next to it) always say the same thing, so nothing rests on the colour alone.
export type ChipKind = 'flat' | 'hi' | 'dim' | 'model' | 'warn' | 'chg' | 'down';

export default function Chip({ kind = 'flat', children }: { kind?: ChipKind; children: ReactNode }) {
  return <span className={`chip ${kind}`}>{children}</span>;
}

// "manual": a row that ran (or runs, on the Models page) on a setting an administrator set by hand on Admin ›
// Parameters instead of the tuned value; the settings on hover. Admin pages only (owner, 8 Nov 2026).
export function ManualChip({ keys, now = false }: { keys: string[] | null; now?: boolean }) {
  if (!keys || !keys.length) return null;
  return (
    <>
      {' '}
      <span title={`${now ? 'Runs' : 'Ran'} on a manual setting (Admin › Parameters): ${keys.join(', ')}`}>
        <Chip>manual</Chip>
      </span>
    </>
  );
}
