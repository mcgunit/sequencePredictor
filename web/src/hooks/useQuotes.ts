import { useEffect, useState } from 'react';
import { getQuotes, type Market, type Quotes } from '../api';

// Live quotes for a market: fetched once, then every 60 s while the tab is
// visible. A failed fetch keeps the last answer; the page then shows the
// close-based values only.
export function useQuotes(market: Market, enabled: boolean): Quotes | null {
  const [quotes, setQuotes] = useState<Quotes | null>(null);

  useEffect(() => {
    setQuotes(null);
    if (!enabled) return;
    let alive = true;
    let timer: number | undefined;

    function load() {
      getQuotes(market)
        .then((q) => {
          if (alive) setQuotes(q);
        })
        .catch(() => {});
    }
    function stop() {
      if (timer !== undefined) {
        window.clearInterval(timer);
        timer = undefined;
      }
    }
    function start() {
      stop();
      load();
      timer = window.setInterval(load, 60_000);
    }
    function onVisibility() {
      if (document.visibilityState === 'visible') start();
      else stop();
    }

    if (document.visibilityState === 'visible') start();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      alive = false;
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [market, enabled]);

  return quotes;
}
