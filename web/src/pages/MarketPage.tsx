import { useEffect, useState } from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { TITLES, getMarket, isMarket, type Market, type MarketView } from '../api';
import { useQuotes } from '../hooks/useQuotes';
import StatusStrip from '../components/StatusStrip';
import OpenPositions from '../components/OpenPositions';
import TradesLog from '../components/TradesLog';
import Charts from '../components/Charts';
import ClosedPositions from '../components/ClosedPositions';

type State = { kind: 'loading' } | { kind: 'error'; error: Error } | { kind: 'ok'; data: MarketView };

// /crypto and /shares; anything else goes back to /crypto.
export default function MarketPage() {
  const { market } = useParams();
  if (!isMarket(market)) return <Navigate to="/crypto" replace />;
  return <Market key={market} market={market} />;
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
        <h1>{TITLES[market]}</h1>
        <div className="card mut" style={{ marginTop: 10 }}>
          Loading…
        </div>
      </>
    );
  }

  if (state.kind === 'error') {
    return (
      <>
        <h1>{TITLES[market]}</h1>
        <div className="card err" style={{ marginTop: 10 }}>
          <strong>Could not load this page.</strong>
          <p className="mut">
            <code>{state.error.message}</code>
          </p>
          <button type="button" className="linkbtn more" onClick={() => setAttempt((n) => n + 1)}>
            try again
          </button>
        </div>
      </>
    );
  }

  const { data } = state;
  return (
    <>
      <h1>
        {data.title} {data.instruments.length ? <small>{data.instruments.join(' · ')}</small> : null}
      </h1>
      <StatusStrip status={data.status} unit={data.unit} />
      <div className="stack">
        <OpenPositions data={data} quotes={quotes} />
        <TradesLog log={data.log} pending={data.status.days > 0 && data.status.positions === 0} />
        <Charts data={data} />
        <ClosedPositions closed={data.closed} noun={data.noun} />
      </div>
    </>
  );
}
