import type { ReactNode } from 'react';

// A neutral chip. The admin pages colour nothing by itself (a colour will mark a change from last week once
// the weekly history exists): flat is a plain state, hi an outlined "above its band", dim a dashed "too
// few draws to judge", model a row name.
export type ChipKind = 'flat' | 'hi' | 'dim' | 'model' | 'warn';

export default function Chip({ kind = 'flat', children }: { kind?: ChipKind; children: ReactNode }) {
  return <span className={`chip ${kind}`}>{children}</span>;
}
