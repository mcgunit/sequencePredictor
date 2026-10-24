import { useCallback, useEffect } from 'react';
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom';
import { DASH, count, short } from '../format';
import { useLoad } from '../hooks/useLoad';
import { getUserHistory, isUserGame, type HistoryTotal, type UserHistory } from '../lottery';
import type { Shape } from '../workbench';
import { Card } from '../components/admin/Card';
import { Loading, PageHead } from '../components/admin/States';
import HistoryView from '../components/draws/History';
import { bestMeaning, bestText, eur, eurU, gameTitle } from '../components/draws/text';
import { LotteryFailed, UserGamePicker, drawHref, fromTo, lotteryHref, userText } from '../components/lottery/bits';

// /lottery/:game: the game's draws from its first saved ranking on, a folding card per month, newest first, all folded
// (?month= opens that month: back from one of its draws). Every figure is over the five models ranked best
// BEFORE each draw - the newest ranking saved before 15:00 Brussels time that day - never today's five; the draws before
// the first saved ranking are left out, and one line says how many.
const MONTH = /^\d{4}-\d{2}$/;

export default function LotteryGamePage() {
  const { game = '' } = useParams();
  const [params] = useSearchParams();
  const raw = params.get('month');
  const month = raw && MONTH.test(raw) ? raw : null;
  const known = isUserGame(game);
  // a month without a counted draw opens the newest
  const load = useLoad(
    () => (known ? (month ? getUserHistory(game, month).catch(() => getUserHistory(game)) : getUserHistory(game)) : Promise.reject(new Error('unknown game'))),
    [game, month],
  );
  const loadMonth = useCallback((m: string) => getUserHistory(game, m), [game]);
  const toDraw = useCallback((date: string) => drawHref(game, date), [game]);

  useEffect(() => {
    document.title = `${gameTitle(game)} · every counted draw · Predictor`;
  }, [game]);
  const opened = load.kind === 'ok' ? load.data.month : null;
  useEffect(() => {
    if (month && opened === month) document.getElementById(`m-${month}`)?.scrollIntoView({ block: 'start' });
  }, [month, opened]);

  if (!known) return <Navigate to="/lottery" replace />;
  const title = gameTitle(game);
  const aside = <UserGamePicker game={game} />;
  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title={title} sub="every counted draw" aside={aside} />
        {load.kind === 'loading' ? <Loading /> : <LotteryFailed error={load.error} game={game} retry={load.retry} />}
      </>
    );
  }
  const h = load.data;
  return (
    <div className="lottery">
      <PageHead
        title={title}
        sub={h.total ? (h.draws > h.counted ? `${h.counted} of ${count(h.draws, h.unit)} counted` : `${count(h.counted, h.unit)} counted`) : 'no draw counted yet'}
        meta={['Brussels time']}
        aside={aside}
      />
      <p className="statusline">
        <span>
          <b>Each draw counts the 5 models ranked best before it</b> (the ranking saved before 15:00 Brussels time that day), never today&apos;s five.
          {h.leftOut ? ` ${userText(h.leftOut.text)}` : ''}
        </span>
      </p>
      {h.ranking.error ? (
        <p className="statusline">
          <span>
            <b>A saved ranking could not be read just now</b>: the draws it decides have no top 5 and say so; it is read again shortly.
          </span>
        </p>
      ) : null}
      <div className="stack">
        {h.total ? <TotalCard h={h} t={h.total} /> : null}
        {h.months.length ? (
          <HistoryView key={game} h={h} mode="user" open={month} loadMonth={loadMonth} drawHref={toDraw} />
        ) : (
          <div className="card">
            <div className="card-b empty">No draw of {title} has been counted yet.</div>
          </div>
        )}
      </div>
      <p className="note foot-note">
        <Link to={lotteryHref(game)}>{title}&apos;s next draw and today&apos;s top 5 ›</Link>
      </p>
    </div>
  );
}

// The whole history's figure over the same rows as the months': each draw's own five
function TotalCard({ h, t }: { h: UserHistory; t: HistoryTotal }) {
  const shape: Shape = h.shape;
  const lines = totalLines(shape, t, h.unit);
  return (
    <Card title="Since the first counted draw" meta={[fromTo(t.from, t.to), count(t.counted, h.unit)]}>
      <span className="fig big">
        <b>{lines.main}</b>
        {lines.sub.map((x) => (
          <small key={x}>{x}</small>
        ))}
      </span>
    </Card>
  );
}

function totalLines(shape: Shape, t: HistoryTotal, unit: string): { main: string; sub: string[] } {
  const over = `over ${count(t.counted, unit)}, each with the 5 ranked best before it`;
  // a draw without a top 5 (a ranking that could not be read, no row of it ranked then): its own line says why
  const n = t.draws - t.counted;
  const notCounted = n > 0 ? [`${count(n, unit)} without a top 5 ${n === 1 ? 'is' : 'are'} not counted (${n === 1 ? 'its line says' : 'their lines say'} why)`] : [];
  if (shape.money) {
    const sub = [`net over ${t.tickets} ${t.tickets === 1 ? 'ticket' : 'tickets'} · stake ${eurU(t.stake)}`, over];
    if (shape.family === 'jokerplus' && t.best) sub.push(`best ticket ${bestText(shape, t.best)}: ${bestMeaning(shape)}, by ${short(t.best.by)}`);
    return { main: eur(t.profit), sub: [...sub, ...notCounted] };
  }
  if (!t.best) return { main: DASH, sub: ['no scored row', over, ...notCounted] };
  const sub = [`best ticket: ${bestMeaning(shape)}, by ${short(t.best.by)}`, over];
  if (shape.family === 'lotto' && t.multi !== null) sub.push(`best of 9 numbers (multi-pick): ${t.multi}`);
  return { main: bestText(shape, t.best), sub: [...sub, ...notCounted] };
}
