import { useEffect, useState, type ReactNode } from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { TITLES, getMarket, isMarket, type Market, type MarketView } from '../api';
import { day, time } from '../format';
import { useQuotes } from '../hooks/useQuotes';
import StatusStrip from '../components/StatusStrip';
import TraderCard from '../components/TraderCard';
import TradesLog from '../components/TradesLog';
import { useChartPanels } from '../components/Charts';
import ClosedPositions from '../components/ClosedPositions';

type State = { kind: 'loading' } | { kind: 'error'; error: Error } | { kind: 'ok'; data: MarketView };

// /crypto and /shares; anything else goes back to Home.
export default function MarketPage() {
  const { market } = useParams();
  if (!isMarket(market)) return <Navigate to="/" replace />;
  return <Market key={market} market={market} />;
}

function Head({ title, instruments, note }: { title: string; instruments: string[]; note?: ReactNode }) {
  return (
    <div className="page-head">
      <h1>
        {title}
        {instruments.length ? <small>{instruments.join(' · ')}</small> : null}
      </h1>
      {note ? <div className="meta">{note}</div> : null}
    </div>
  );
}

function Market({ market }: { market: Market }) {
  const [state, setState] = useState<State>({ kind: 'loading' });
  const [attempt, setAttempt] = useState(0);
  const quotes = useQuotes(market, state.kind === 'ok');

  useEffect(() => {
    let alive = true;
    setState({ kind: 'loading' });
    getMarket(market)
      .then((data) => alive && setState({ kind: 'ok', data }))
      .catch((error: Error) => alive && setState({ kind: 'error', error }));
    return () => {
      alive = false;
    };
  }, [market, attempt]);

  useEffect(() => {
    document.title = `${state.kind === 'ok' ? state.data.title : TITLES[market]} · Predictor`;
  }, [state, market]);

  if (state.kind === 'loading') {
    return (
      <>
        <Head title={TITLES[market]} instruments={[]} />
        <div className="card">
          <div className="card-b empty">Loading…</div>
        </div>
      </>
    );
  }

  if (state.kind === 'error') {
    return (
      <>
        <Head title={TITLES[market]} instruments={[]} />
        <div className="card err">
          <div className="card-h">
            <h2>Could not load this page</h2>
          </div>
          <div className="card-b">
            <p className="mut">
              <code>{state.error.message}</code>
            </p>
            <button type="button" className="linkbtn more" onClick={() => setAttempt((n) => n + 1)}>
              Try again
            </button>
          </div>
        </div>
      </>
    );
  }

  const { data } = state;
  return <Dashboard data={data} quotes={quotes} />;
}

function Dashboard({ data, quotes }: { data: MarketView; quotes: ReturnType<typeof useQuotes> }) {
  // everything under the head is the paper trader's record (18 Nov 2026): the card, then its figures
  const trader = data.trader;
  const { cumulativePanel, perCoinPanel, dailyPanel } = useChartPanels(trader, data.unit, data.noun);
  const paper = (data.status.paperNote || '').split('. ')[0].replace(/\.$/, '');
  return (
    <>
      <Head
        title={data.title}
        instruments={data.instruments}
        note={
          <>
            {paper ? <span title={data.status.paperNote}>{paper}</span> : null}
            {data.generatedAt ? (
              <span>
                updated {day(data.generatedAt)} {time(data.generatedAt)}
              </span>
            ) : null}
          </>
        }
      />
      {trader ? <StatusStrip status={trader.status} unit={data.unit} noun={data.noun} /> : null}
      <div className="grid">
        {/* the trader's card is the page's only instructions: the day and week cards left the user page on 18 Nov 2026 */}
        {trader ? (
          <TraderCard trader={trader} noun={data.noun} unit={data.unit} quotes={quotes} />
        ) : (
          <div className="card span-3">
            <div className="card-h">
              <h2>Trader</h2>
            </div>
            <div className="card-b empty">No instructions yet: the trader&apos;s first instructions come with the next run.</div>
          </div>
        )}
        {cumulativePanel}
        {perCoinPanel}
        {dailyPanel}
        <TradesLog closed={trader ? trader.closed : []} unit={data.unit} />
        <ClosedPositions closed={trader ? trader.closed : []} noun={data.noun} />
      </div>
    </>
  );
}
