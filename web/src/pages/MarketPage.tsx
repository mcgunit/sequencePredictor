import { useEffect, useState, type ReactNode } from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { TITLES, getMarket, isMarket, type Market, type MarketView } from '../api';
import { day, time } from '../format';
import { useQuotes } from '../hooks/useQuotes';
import StatusStrip from '../components/StatusStrip';
import OpenPositions from '../components/OpenPositions';
import TradesLog from '../components/TradesLog';
import { useChartPanels } from '../components/Charts';
import ClosedPositions from '../components/ClosedPositions';

type State = { kind: 'loading' } | { kind: 'error'; error: Error } | { kind: 'ok'; data: MarketView };

// /crypto and /shares; anything else goes back to /crypto.
export default function MarketPage() {
  const { market } = useParams();
  if (!isMarket(market)) return <Navigate to="/crypto" replace />;
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
  const pending = data.status.days > 0 && data.status.positions === 0;
  const { cumulativePanel, perCoinPanel, dailyPanel } = useChartPanels(data, pending);
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
      <StatusStrip status={data.status} unit={data.unit} noun={data.noun} />
      <div className="grid">
        <OpenPositions data={data} quotes={quotes} />
        {cumulativePanel}
        {perCoinPanel}
        {dailyPanel}
        <TradesLog log={data.log} unit={data.unit} pending={pending} />
        <ClosedPositions closed={data.closed} noun={data.noun} unit={data.unit} pending={pending} />
      </div>
    </>
  );
}
