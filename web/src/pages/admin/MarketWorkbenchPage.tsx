import { useEffect } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { isMarket, TITLES } from '../../api';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import { day, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { getWorkbench } from '../../workbench';
import AccuracyCard from './workbench/AccuracyCard';
import BooksCard from './workbench/BooksCard';
import FollowedCard from './workbench/FollowedCard';
import InstrumentsCard from './workbench/InstrumentsCard';
import ModelsCard from './workbench/ModelsCard';
import { BackgroundCard, DaysCard, GuardCard, RegimesCard } from './workbench/MoreCards';
import { PlanCard, WeekGameCard } from './workbench/PlanCards';

// /admin/markets/:market: the market workbench (the old /markets/:market page), everything /app/crypto and /app/shares
// and the other admin pages do not already show, one card per section. The record's own stamps are in the head; the
// draws of the day and week games are under Games.
export default function MarketWorkbenchPage() {
  const { market = '' } = useParams();
  const known = isMarket(market);
  const load = useLoad(() => (known ? getWorkbench(market) : Promise.reject(new Error('unknown market'))), [market]);

  useEffect(() => {
    document.title = `${known ? TITLES[market] : market} workbench · Predictor`;
  }, [market, known]);

  if (!known) return <Navigate to="/admin/games" replace />;
  const title = `${TITLES[market]} workbench`;
  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title={title} />
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const w = load.data;
  const active = w.instruments.filter((i) => i.active).length;
  return (
    <div className="admin">
      <PageHead
        title={title}
        sub={`${w.calendar} · ${active} ${w.noun}s · ${w.scoredDays ?? '—'} scored days`}
        meta={[`record ${stamp(w.generatedAt)}`, `newest game day ${day(w.newestGameDay)}`, 'Brussels time']}
      />
      <p className="statusline mut">
        <span>
          Each {w.noun}&apos;s next move is cut into {w.k} bins of its own past; every model calls one bin per {w.noun}. A predictor, not a trading bot: paper money,{' '}
          {w.stake ?? 100} {w.currency} a position.
        </span>
        <span className="wb-links">
          <Link to={`/admin/games/${market}`}>{market} draws ›</Link>
          {w.weekGame ? <Link to={`/admin/games/${w.weekGame}`}>{w.weekGame} draws ›</Link> : null}
          <Link to={`/${market}`}>the users&apos; {TITLES[market]} page ›</Link>
        </span>
      </p>
      <div className="stack">
        <PlanCard w={w} />
        <WeekGameCard w={w} />
        <FollowedCard w={w} />
        <BooksCard w={w} />
        <ModelsCard w={w} />
        <AccuracyCard w={w} />
        <InstrumentsCard w={w} />
        <DaysCard w={w} />
        <GuardCard w={w} />
        <RegimesCard w={w} />
        <BackgroundCard w={w} />
      </div>
    </div>
  );
}
