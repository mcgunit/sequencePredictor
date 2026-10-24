import { useEffect, useRef } from 'react';
import { stamp } from '../format';
import type { Running } from '../lottery';

// The old pages' run banner (server.js pipelineBanner) on Home and the Lottery: while a pipeline job holds process.lock
// (pipeline.js, `running` on /api/home and /api/lottery), one line says which job and since when, and the page reads its
// data again every minute until the job has gone. The answer that clears the line carries what the run made; the reads
// stop then, and when the page is left.
const RUN_POLL_MS = 60 * 1000;

// "just started", "12 min", "2.5 h": how long the job has been running, as the old banner said it
function runningFor(since: string | null): string | null {
  const at = since ? Date.parse(since) : NaN;
  if (!Number.isFinite(at)) return null;
  const minutes = Math.max(0, Math.round((Date.now() - at) / 60000));
  if (minutes < 1) return 'just started';
  return minutes < 90 ? `${minutes} min` : `${(minutes / 60).toFixed(1)} h`;
}

export default function RunBanner({ running, refresh }: { running: Running | null; refresh: () => void }) {
  const again = useRef(refresh);
  again.current = refresh;
  const on = running !== null;
  useEffect(() => {
    if (!on) return undefined;
    // a hidden tab is not read; the next minute after it comes back is
    const timer = setInterval(() => {
      if (document.visibilityState !== 'hidden') again.current();
    }, RUN_POLL_MS);
    return () => clearInterval(timer);
  }, [on]);
  if (!running) return null;
  const length = runningFor(running.since);
  return (
    <p className="runbanner">
      <span className="spin" aria-hidden="true" />
      <span>
        <b>{running.what}.</b> {running.since ? `Running since ${stamp(running.since)}${length ? ` (${length})` : ''}; this` : 'This'} page updates when it finishes.
      </span>
    </p>
  );
}
