import { useEffect } from 'react';
import { getAdminModels, type ModelsGameSummary, type ModelsView } from '../../api';
import { DASH, count, day, fixed, perUnit, short, stamp, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam } from '../../hooks/useUrlState';
import { Card, Scroll } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import ModelsDrawer from './ModelsDrawer';
import { GameButton, GroupRow, isMarketGame, oneControlText, recordText, sinceValue, topWhose, valueCell } from './bits';

// /admin/models: the Best model per game card of the old History page, one row per game, with its top five (a
// lottery game's are the rows the users see today; a market game's users' page follows its best book); a game's drawer (?game=lotto) holds its whole ranking with the All / Top 5 switch, the lotto
// multi-pick table and the best combination.
export default function ModelsPage() {
  const load = useLoad(getAdminModels, []);
  const drawer = useDrawerParam();

  useEffect(() => {
    document.title = 'Best model & rankings · Predictor';
  }, []);

  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title="Best model & rankings" />
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  return (
    <div className="admin">
      <PageHead
        title="Best model & rankings"
        sub={`${d.games.length} games · every scored draw`}
        meta={[`ranking ${stamp(d.generatedAt)}`, 'Brussels time']}
      />
      <div className="stack">
        <BestCard d={d} onOpen={drawer.open} />
      </div>
      {drawer.game ? <ModelsDrawer key={drawer.game} game={drawer.game} onClose={drawer.close} /> : null}
    </div>
  );
}

function BestCard({ d, onOpen }: { d: ModelsView; onOpen: (g: string) => void }) {
  const since = d.since?.date ? day(d.since.date) : null;
  const groups = (['Lottery', 'Markets'] as const).map((name) => ({ name, games: d.games.filter((g) => g.group === name) })).filter((g) => g.games.length);
  return (
    <Card
      title="Best model per game"
      flush
      help={
        <>
          Best row: the first row of the game&apos;s History ranking over every scored draw, with its record (market games: the live days apart from the
          back-built or late ones). Lotto, EuroMillions, EuroDreams, Vikinglotto and the market games rank on average hits; jokerplus, pick3 and keno on a capped
          profit bound, the mean minus 1 SE per bet: keno&apos;s bound is measured against a fair ticket of its size (0 is a fair ticket, not money won), so the raw
          profit per bet stands under it. Since: the best row of the forward record from the design freeze, with its own minimum of draws. Q0 band: hits per
          draw on fair histories; a hit-ranked row within it is no better than the best of the control rows by chance, and for the payout games it is
          information only. Top 5: for a lottery game the five rows a user sees today, the first five ranked rows in the newest day&apos;s table (a ranked
          row missing from it is passed over); for a market game the first five of its History game view, while the users&apos; Crypto and Shares pages
          follow the row named under them, by best book. A game opens its whole ranking.
        </>
      }
      meta={[`ranking ${stamp(d.generatedAt)}`, d.window, since ? `since ${since}${d.since?.label ? `: ${d.since.label}` : ''}` : '']}
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th>Best row</th>
              <th className="num last-sm">Value</th>
              <th className="hide-sm">Since {since ?? DASH}</th>
              <th className="hide-sm last-1col">Q0 band</th>
              <th className="hide-1col">Top 5</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((grp) => (
              <BestGroup key={grp.name} name={grp.name} games={grp.games} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

function BestGroup({ name, games, onOpen }: { name: string; games: ModelsGameSummary[]; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={6}>{name}</GroupRow>
      {games.map((g) => {
        const b = g.best;
        const v = b ? valueCell(g, b.value, b.profitPerBet, b.bets) : null;
        const s = g.since;
        // a profit bound carries what it is measured against on its own line (keno: a fair ticket, not money won)
        const sinceText = !s ? DASH : s.none || !s.best ? 'no record since' : `${short(s.best.row)} ${sinceValue(g, s.best.value)}`;
        // the since record counts only live days in a market game (Since.py)
        const sinceSub = s && s.best ? `${count(s.best.draws, g.unit)}${isMarketGame(g.game) ? ' live' : ''}${s.best.tooFew ? ' · too few' : ''} · ranked from ${count(s.minDraws, g.unit)}` : '';
        const band = g.band;
        const bandSub = band ? (band.greys ? (b?.inBand ? 'best row within the band' : 'best row above the band') : `on hits, information only · best row ${fixed(b?.avgHits)}`) : '';
        // what the band is built on: the best of a few control rows on long fair histories, not this ranking
        const bandBasis = band ? `best of ${band.rows ?? DASH} rows · fair histories of ${count(band.history, g.unit)}` : '';
        return (
          <tr key={g.game} className={band?.greys && b?.inBand ? 'grey' : undefined}>
            <td className="sym">
              <GameButton game={g.game} onOpen={onOpen} />
              <span className="sub">{g.metric === 'avg_hits' ? 'average hits' : 'profit bound'}</span>
            </td>
            <td>
              {b ? short(b.row) : DASH}
              {b?.young ? (
                <>
                  {' '}
                  <Chip kind="dim">short record</Chip>
                </>
              ) : null}
              {b ? <span className="sub">{recordText(b.draws, g.unit, b.live)}</span> : null}
              <span className="sub show-sm">
                since: {sinceText}
                {sinceSub ? ` (${sinceSub})` : ''}
              </span>
              {band ? (
                <span className="sub show-sm">
                  band {fixed(band.ceiling)} · {bandSub} · {bandBasis} · {oneControlText(band) ? `${oneControlText(band)} · ` : ''}
                  {weekday(band.at)}
                </span>
              ) : null}
              <span className="sub show-1col">
                top 5: {g.top.map(short).join(', ') || DASH} ({topWhose(g)})
              </span>
            </td>
            <td className="num last-sm">
              <b>{v ? v.main : DASH}</b>
              {v?.sub.map((x) => (
                <span key={x} className="sub">
                  {x}
                </span>
              ))}
            </td>
            <td className="hide-sm">
              {sinceText}
              {sinceSub ? <span className="sub">{sinceSub}</span> : null}
            </td>
            <td className="hide-sm last-1col">
              {band ? (
                <>
                  <b>{fixed(band.ceiling)}</b> <span className="mut">hits per {perUnit(g.unit)}</span>
                  <span className="sub">{bandSub}</span>
                  <span className="sub">{bandBasis}</span>
                  <span className="sub">
                    {oneControlText(band) || band.from} · {weekday(band.at)}
                  </span>
                </>
              ) : (
                <span className="mut">no band yet</span>
              )}
            </td>
            <td className="hide-1col">
              <span className="chips">
                {g.top.map((n) => (
                  <Chip key={n} kind="model">
                    {short(n)}
                  </Chip>
                ))}
              </span>
              <span className="sub">{topWhose(g)}</span>
            </td>
          </tr>
        );
      })}
    </>
  );
}
