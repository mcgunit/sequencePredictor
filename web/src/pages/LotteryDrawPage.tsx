import { useCallback, useEffect } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { ApiError } from '../api';
import { dayYear } from '../format';
import { useLoad } from '../hooks/useLoad';
import { getUserDraw, isUserGame } from '../lottery';
import { Loading, PageHead } from '../components/admin/States';
import DrawView from '../components/draws/DrawView';
import { gameTitle } from '../components/draws/text';
import { LotteryFailed, drawHref, historyHref } from '../components/lottery/bits';

// /lottery/:game/:date: one counted draw - its result and the five models ranked best before it, their tickets with the
// hits marked and the money per ticket; on the newest draw also the coming draw with today's five. The previous and next
// links stay inside the counted history (a draw before the first saved ranking is left out).
const DAY = /^\d{4}-\d{2}-\d{2}$/;

export default function LotteryDrawPage() {
  const { game = '', date = '' } = useParams();
  const known = isUserGame(game);
  const isDay = DAY.test(date);
  const load = useLoad(() => (known && isDay ? getUserDraw(game, date) : Promise.reject(new Error('not a draw'))), [game, date]);
  const toDraw = useCallback((d: string) => drawHref(game, d), [game]);
  const title = gameTitle(game);

  useEffect(() => {
    document.title = `${title} ${date} · Predictor`;
  }, [title, date]);

  if (!known) return <Navigate to="/lottery" replace />;
  // not a date (the server answers 400): the game's history instead
  if (!isDay || (load.kind === 'error' && load.error instanceof ApiError && load.error.status === 400)) return <Navigate to={historyHref(game)} replace />;
  const nav = (loaded: { prev: string | null; next: string | null } | null) => {
    const back = !loaded ? <span className="off">‹</span> : loaded.prev ? <Link to={toDraw(loaded.prev)}>‹ {dayYear(loaded.prev)}</Link> : <span className="off">‹ the first counted draw</span>;
    const on = !loaded ? <span className="off">›</span> : loaded.next ? <Link to={toDraw(loaded.next)}>{dayYear(loaded.next)} ›</Link> : <span className="off">the newest draw ›</span>;
    return (
      <div className="drawnav statusline">
        {back}
        <Link to={historyHref(game, date.slice(0, 7))}>Every counted {title} draw</Link>
        {on}
      </div>
    );
  };
  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title={`${title} · ${dayYear(date)}`} />
        {nav(null)}
        {load.kind === 'loading' ? <Loading /> : <LotteryFailed error={load.error} game={game} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  return (
    <div className="lottery">
      <PageHead title={`${title} · ${dayYear(d.date)}`} sub={`${d.weekday} · ${d.newest ? 'the newest draw' : 'a past draw'}`} meta={['Brussels time']} />
      {nav(d)}
      <DrawView d={d} mode="user" drawHref={toDraw} />
    </div>
  );
}
