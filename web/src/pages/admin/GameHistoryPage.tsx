import { useCallback, useEffect } from 'react';
import { Navigate, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { LOTTERY_GAMES, MARKET_GAMES, isGame, type Scope } from '../../api';
import ScopeSwitch from '../../components/admin/ScopeSwitch';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import { MarketNote } from '../../components/draws/DrawView';
import HistoryView from '../../components/draws/History';
import { gameTitle } from '../../components/draws/text';
import { count, dayYear, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useScope } from '../../hooks/useUrlState';
import { getHistory, type History } from '../../workbench';

// The Workbench's routes: a game's history and one draw, the switch carried in the links
const MONTH = /^\d{4}-\d{2}$/;
// month (YYYY-MM): the month card to open, as when going back from one of its draws
export const historyHref = (game: string, scope?: Scope, month?: string) => {
  const q = new URLSearchParams();
  if (scope) q.set('scope', scope);
  if (month && MONTH.test(month)) q.set('month', month);
  const search = q.toString();
  return `/admin/games/${game}${search ? `?${search}` : ''}`;
};
export const drawHref = (game: string, date: string, scope?: Scope) => `/admin/games/${game}/${date}${scope ? `?scope=${scope}` : ''}`;
export const marketHref = (market: string) => `/admin/markets/${market}`;

// The game picker in a page head: another game's history, the switch kept
export function GamePicker({ game, scope }: { game: string; scope: Scope }) {
  const navigate = useNavigate();
  return (
    <label className="pick">
      <span className="vh">Game</span>
      <select value={game} onChange={(e) => navigate(historyHref(e.target.value, scope))}>
        <optgroup label="Lottery">
          {LOTTERY_GAMES.map((g) => (
            <option key={g} value={g}>
              {gameTitle(g)}
            </option>
          ))}
        </optgroup>
        <optgroup label="Markets">
          {MARKET_GAMES.map((g) => (
            <option key={g} value={g}>
              {gameTitle(g)}
            </option>
          ))}
        </optgroup>
      </select>
    </label>
  );
}

// Whose rows the history's figures are over, once for the page: All, or each draw's top five as ranked before it.
// The first saved ranking (25 Aug 2026) is said, and when this game entered the ranking later, that day too.
function ScopeSummary({ h }: { h: History }) {
  const at = (v: { committedAt: string | null; generatedAt: string | null } | null) => (v ? dayYear(v.committedAt ?? v.generatedAt) : null);
  const first = at(h.ranking.first);
  const own = at(h.ranking.firstOfGame);
  const since = !first ? '' : !own ? `; ${gameTitle(h.game)} is in no saved ranking yet` : own !== first ? `; ${gameTitle(h.game)} was first ranked on ${own}` : '';
  if (h.ranking.error && !h.ranking.first) {
    return (
      <p className="statusline">
        <span>
          <b>The ranking history could not be read</b> ({h.ranking.error}): no draw has a Top 5{h.scope === 'all' ? '; All shows every row' : ''}.
        </span>
      </p>
    );
  }
  // a version git could not read: only the draws it decides lose their Top, each says so
  const partial = h.ranking.error ? (
    <p className="statusline">
      <span>
        <b>A saved ranking could not be read</b> ({h.ranking.error}): the draws it decides have no Top 5 and say so; it is read again shortly.
      </span>
    </p>
  ) : null;
  if (h.scope === 'top') {
    return (
      <>
        {partial}
        <p className="statusline">
          <span>
            <b>Top 5 as ranked before each draw</b>: {h.ranking.cutoff.text}. {h.counted} of {count(h.draws, h.unit)} have one
            {first ? `; no ranking was recorded before ${first}` : '; no ranking was recorded yet'}
            {since}. Every figure is over those five rows.
          </span>
        </p>
      </>
    );
  }
  return (
    <>
      {partial}
      <p className="statusline mut">
        <span>
          <b>All rows</b> of each of the {count(h.draws, h.unit)}. The switch keeps each draw&apos;s top 5 as ranked before it (never today&apos;s)
          {first ? `; none was recorded before ${first}` : ''}
          {since}.
        </span>
      </p>
    </>
  );
}

export default function GameHistoryPage() {
  const { game = '' } = useParams();
  const [scope, setScope] = useScope();
  const [params] = useSearchParams();
  const raw = params.get('month');
  const month = raw && MONTH.test(raw) ? raw : null;
  const known = isGame(game);
  // ?month= opens that month (back from one of its draws); a month without a stored draw leaves every month folded
  const load = useLoad(
    () => (known ? (month ? getHistory(game, scope, month).catch(() => getHistory(game, scope)) : getHistory(game, scope)) : Promise.reject(new Error('unknown game'))),
    [game, scope, month],
  );
  const loadMonth = useCallback((month: string) => getHistory(game, scope, month), [game, scope]);
  const toDraw = useCallback((date: string) => drawHref(game, date, scope), [game, scope]);

  useEffect(() => {
    document.title = `${gameTitle(game)} · every draw · Predictor`;
  }, [game]);
  // the month asked for is opened and scrolled to
  const opened = load.kind === 'ok' ? load.data.month : null;
  useEffect(() => {
    if (month && opened === month) document.getElementById(`m-${month}`)?.scrollIntoView({ block: 'start' });
  }, [month, opened]);

  if (!known) return <Navigate to="/admin/games" replace />;
  const aside = (
    <>
      <GamePicker game={game} scope={scope} />
      <ScopeSwitch scope={scope} onChange={setScope} topLabel="Top 5 as ranked before each draw" />
    </>
  );
  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title={gameTitle(game)} sub="every stored draw" aside={aside} />
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const h = load.data;
  return (
    <div className="admin">
      <PageHead
        title={gameTitle(game)}
        sub={`${count(h.draws, h.unit)} stored · ${h.months.length} months · ${h.group}`}
        meta={[`built ${stamp(h.builtAt)}`, 'Brussels time']}
        aside={aside}
      />
      <ScopeSummary h={h} />
      {h.shape.family === 'market' ? <MarketNote game={game} slots={h.shape.slots} mode="admin" marketHref={marketHref} /> : null}
      <HistoryView key={`${game}-${h.scope}`} h={h} mode="admin" open={month} loadMonth={loadMonth} drawHref={toDraw} />
    </div>
  );
}
