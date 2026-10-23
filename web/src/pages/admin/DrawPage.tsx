import { useCallback, useEffect } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { isGame } from '../../api';
import ScopeSwitch from '../../components/admin/ScopeSwitch';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import DrawView from '../../components/draws/DrawView';
import { gameTitle } from '../../components/draws/text';
import { dayYear, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useScope } from '../../hooks/useUrlState';
import { getDraw } from '../../workbench';
import { drawHref, historyHref, marketHref } from './GameHistoryPage';

// /admin/games/:game/:date: one draw, scored. The switch's Top is the five ranked first before this draw; the coming
// draw (newest day only) is read against today's ranking.
export default function DrawPage() {
  const { game = '', date = '' } = useParams();
  const [scope, setScope] = useScope();
  const known = isGame(game);
  const load = useLoad(() => (known ? getDraw(game, date, scope) : Promise.reject(new Error('unknown game'))), [game, date, scope]);
  const toDraw = useCallback((d: string) => drawHref(game, d, scope), [game, scope]);

  useEffect(() => {
    document.title = `${gameTitle(game)} ${date} · Predictor`;
  }, [game, date]);

  if (!known) return <Navigate to="/admin/games" replace />;
  const aside = <ScopeSwitch scope={scope} onChange={setScope} topLabel="Top 5 as ranked before this draw" />;
  // back to the history on this draw's month; the ends say "first" / "newest" only once the draw has loaded
  const nav = (loaded: { prev: string | null; next: string | null } | null) => {
    const back = !loaded ? <span className="off">‹</span> : loaded.prev ? <Link to={toDraw(loaded.prev)}>‹ {dayYear(loaded.prev)}</Link> : <span className="off">‹ first stored draw</span>;
    const on = !loaded ? <span className="off">›</span> : loaded.next ? <Link to={toDraw(loaded.next)}>{dayYear(loaded.next)} ›</Link> : <span className="off">newest draw ›</span>;
    return (
      <div className="drawnav statusline">
        {back}
        <Link to={historyHref(game, scope, date.slice(0, 7))}>All {gameTitle(game)} draws</Link>
        {on}
      </div>
    );
  };
  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title={`${gameTitle(game)} · ${dayYear(date)}`} aside={aside} />
        {nav(null)}
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  return (
    <div className="admin">
      <PageHead
        title={`${gameTitle(game)} · ${dayYear(d.date)}`}
        sub={`${d.weekday} · ${d.newest ? 'the newest draw' : 'a past draw'} · ${d.group}`}
        meta={[`built ${stamp(d.builtAt)}`, 'Brussels time']}
        aside={aside}
      />
      {nav(d)}
      <DrawView d={d} mode="admin" drawHref={toDraw} marketHref={marketHref} />
    </div>
  );
}
