import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { LOTTERY_GAMES, MARKET_GAMES } from '../../api';
import { PageHead } from '../../components/admin/States';
import { MARKET_DRAW, gameTitle } from '../../components/draws/text';

// /admin/games: the game picker of the Workbench. Each game opens its history of draws (/admin/games/:game); the two
// markets also open their market workbench (/admin/markets/:market).
const LOTTERY_NOTE: Record<string, string> = {
  lotto: '6 of 45 and a bonus ball · multi-pick rows',
  euromillions: '5 of 50 and 2 stars',
  eurodreams: '6 of 40 and a dream number',
  vikinglotto: '6 of 48 and a Viking number',
  jokerplus: '6 digits and a sign · runs from the left and the right',
  pick3: '3 digits · straight, box and pairs',
  keno: '20 of 70 · a ticket per size, 5 to 10',
};

export default function GamesPage() {
  useEffect(() => {
    document.title = 'Games · Predictor';
  }, []);
  return (
    <div className="admin">
      <PageHead title="Games" sub="every stored draw of each game, scored · All or the top 5 as ranked before each draw" />
      <div className="stack">
        <div className="grp-label">Lottery</div>
        <div className="games">
          {LOTTERY_GAMES.map((g) => (
            <Link key={g} className="card" to={`/admin/games/${g}`}>
              <b>{gameTitle(g)} ›</b>
              <small>{LOTTERY_NOTE[g] ?? ''}</small>
            </Link>
          ))}
        </div>
        <div className="grp-label">Markets</div>
        <div className="games">
          {MARKET_GAMES.map((g) => (
            <Link key={g} className="card" to={`/admin/games/${g}`}>
              <b>{gameTitle(g)} ›</b>
              <small>one draw is {MARKET_DRAW[g] ?? 'one game day'}</small>
            </Link>
          ))}
          {['crypto', 'shares'].map((m) => (
            <Link key={m} className="card" to={`/admin/markets/${m}`}>
              <b>{m === 'crypto' ? 'Crypto' : 'Shares'} workbench ›</b>
              <small>the plan in detail, paper books, models, prices, the guard, regimes</small>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
