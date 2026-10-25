import { useEffect } from 'react';
import { getAdminModels, type Combination, type ModelsGameSummary, type ModelsView } from '../../api';
import { DASH, count, day, fixed, perUnit, short, signedFixed, stamp, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam } from '../../hooks/useUrlState';
import { Card, Scroll } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import ModelsDrawer, { LearningTag } from './ModelsDrawer';
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
        <CombinationCard d={d} onOpen={drawer.open} />
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
          profit per bet stands under it. Keno is tuned by this bound; pick3 and jokerplus are tuned by slot hits, with the bound only a tie-breaker. Until 13 Oct
          2026 the payout games ranked on the raw profit per bet, on which one winning ticket could crown a row. Since: the best row of the forward record from
          the design freeze, the same ranking over the draws on or after the declared date, with its own minimum of draws (the date is a filter, nothing is
          rebuilt); &ldquo;no record since&rdquo; until a game&apos;s first draw on or after the date is scored, &ldquo;not declared&rdquo; until a date is set in
          since.json. Q0 band: hits per draw on fair histories; a hit-ranked row within it is no better than the best of the control rows by chance, and for
          the payout games it is information only. The band is not applied to Since: it is measured over long control histories, and the best of the rows
          over a few draws clears it by luck. Top 5: for a lottery game the five rows a user sees today, the first five ranked rows in the newest day&apos;s table
          (a ranked row missing from it is passed over); for a market game the first five of its History game view, while the users&apos; Crypto and Shares pages
          follow the row named under them, by best book. No better than the frequencies (deep-learning rows, from 25 Oct 2026): on the newest day it ran, the
          row&apos;s validation loss did not get below that of the training split&apos;s frequencies per position by more than an output that ignores its input
          gets on the same windows; the tag&apos;s title gives the losses and whether the row ignores its input, and it is a label only, read by neither the
          ranking nor the users&apos; five. A game opens its whole ranking.
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

// a cell of a row greyed by the Q0 band that the band does not apply to (Since)
const NOT_GREYED = { color: 'var(--fg)' };
const KEEP_TOGETHER = { display: 'inline-flex', alignItems: 'center', maxWidth: '100%' };

function BestGroup({ name, games, onOpen }: { name: string; games: ModelsGameSummary[]; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={6}>{name}</GroupRow>
      {games.map((g) => {
        const b = g.best;
        const v = b ? valueCell(g, b.value, b.profitPerBet, b.bets) : null;
        const s = g.since;
        // a profit bound carries what it is measured against on its own line (keno: a fair ticket, not money won); no
        // since record at all means no date is declared in since.json
        const sinceText = !s ? 'not declared' : s.none || !s.best ? 'no record since' : `${short(s.best.row)} ${sinceValue(g, s.best.value)}`;
        // the since record counts only live days in a market game (Since.py)
        const sinceSub = s && s.best ? `${count(s.best.draws, g.unit)}${isMarketGame(g.game) ? ' live' : ''}${s.best.tooFew ? ' · too few' : ''} · ranked from ${count(s.minDraws, g.unit)}` : '';
        const band = g.band;
        const bandSub = band ? (band.greys ? (b?.inBand ? 'best row within the band' : 'best row above the band') : `on hits, information only · best row ${fixed(b?.avgHits)}`) : '';
        // what the band is built on: the best of a few control rows on long fair histories, not this ranking
        const bandBasis = band ? `best of ${band.rows ?? DASH} rows · fair histories of ${count(band.history, g.unit)}` : '';
        const grey = Boolean(band?.greys && b?.inBand);
        // the top five rows whose newest learning check reads no better than the frequencies
        const flagged = g.top.filter((n) => g.learning[n]?.verdict === 'counts-only');
        return (
          <tr key={g.game} className={grey ? 'grey' : undefined}>
            <td className="sym">
              <GameButton game={g.game} onOpen={onOpen} />
              <span className="sub">{g.metric === 'avg_hits' ? 'average hits' : 'profit bound'}</span>
            </td>
            <td>
              {b ? short(b.row) : DASH}
              {b ? <LearningTag l={g.learning[b.row]} /> : null}
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
                {flagged.length ? ` · no better than the frequencies: ${flagged.map(short).join(', ')}` : ''}
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
            {/* the band is not applied to Since, so a row greyed by the band leaves this cell in plain colour */}
            <td className="hide-sm" style={grey ? NOT_GREYED : undefined}>
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
                {g.top.map((n) =>
                  g.learning[n]?.verdict === 'counts-only' ? (
                    // the chip and its tag wrap as one, so a tag never starts a line under another row's chip
                    <span key={n} style={KEEP_TOGETHER}>
                      <Chip kind="model">{short(n)}</Chip>
                      <LearningTag l={g.learning[n]} />
                    </span>
                  ) : (
                    <Chip key={n} kind="model">
                      {short(n)}
                    </Chip>
                  ),
                )}
              </span>
              <span className="sub">{topWhose(g)}</span>
            </td>
          </tr>
        );
      })}
    </>
  );
}

// The old History page's Best combination per game card: per game the best set of rows played together, its value
// with its control's p value on the same line, the draws its members share and the best single row on those draws;
// a game opens its drawer, where the set's top ten are.
function CombinationCard({ d, onOpen }: { d: ModelsView; onOpen: (g: string) => void }) {
  const groups = (['Lottery', 'Markets'] as const).map((name) => ({ name, games: d.games.filter((g) => g.group === name) })).filter((g) => g.games.length);
  if (!d.games.some((g) => g.combination)) return null;
  return (
    <Card
      title="Best combination per game"
      flush
      help={
        <>
          The best set of established rows played together over the draws they all have: for the payout games the net profit per draw of playing every
          member (uncapped), for the others the best line held per draw (the average of the best member&apos;s hits). Its p value comes from the same search on
          shuffled draws: above 0.05 the set is no better than the best set luck finds. The best single row is scored on the same draws, so it can differ from
          its ranking value. The best line held grows with every extra line, so a set is at least as good as any of its own rows by construction: pairs and
          triples are ranked within their size, and the hit games give the best single row no verdict. A game opens the set&apos;s top ten.
        </>
      }
      meta={[`ranking ${stamp(d.generatedAt)}`, 'every established row, not filtered by a switch']}
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th>Rows played together</th>
              <th className="num last-sm">Value</th>
              <th className="num hide-sm">Shared draws</th>
              <th className="hide-sm">Best single row on the same draws</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((grp) => (
              <CombinationGroup key={grp.name} name={grp.name} games={grp.games} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

function comboValue(c: Combination, unit: string, v: number | null): string {
  const per = `per ${perUnit(unit)}`;
  return c.metric === 'profit_per_draw' ? `${signedFixed(v, 2)} EUR ${per}` : `${fixed(v)} hits ${per}`;
}

function CombinationGroup({ name, games, onOpen }: { name: string; games: ModelsGameSummary[]; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={5}>{name}</GroupRow>
      {games.map((g) => {
        const c = g.combination;
        const best = c?.best ?? null;
        const control = c?.control ?? null;
        // the value's control on the value's line: a set no better than luck says so where its number is
        const controlText = !control ? 'no control' : 'error' in control ? 'control not available' : `p = ${fixed(control.pValue)}${control.consistentWithLuck ? ' · consistent with luck' : ''}`;
        const single = c?.bestSingle ?? null;
        const singleText = single ? `${short(single.row)} ${comboValue(c as Combination, g.unit, single.value)}` : DASH;
        // the best line held never falls when a line is added, so the hit games get no verdict, and say why
        const beats =
          c?.beatsBestSingle === true
            ? 'the set beats it'
            : c?.beatsBestSingle === false
              ? 'the set does not beat it'
              : single && c?.metric === 'avg_best_hits'
                ? 'no verdict: a set scores at least its own rows by construction'
                : '';
        return (
          <tr key={g.game} className={control && !('error' in control) && control.consistentWithLuck ? 'grey' : undefined}>
            <td className="sym">
              <GameButton game={g.game} onOpen={onOpen} />
              <span className="sub">{c ? (c.metric === 'profit_per_draw' ? 'profit per draw' : 'best line held') : DASH}</span>
            </td>
            <td>
              {best ? (
                best.members.map(short).join(' + ')
              ) : (
                <span className="mut">
                  {c ? `${c.note ?? 'no combination scored'} (${c.candidateRows} established ${c.candidateRows === 1 ? 'row' : 'rows'})` : 'no combination scored'}
                </span>
              )}
              {best ? <span className="sub">{`${best.how}${best.size !== null ? `, ${best.size} rows` : ''}`}</span> : null}
              {best ? <span className="sub show-sm">{`${count(best.draws, g.unit)} shared · best single: ${singleText}${beats ? ` (${beats})` : ''}`}</span> : null}
            </td>
            <td className="num last-sm">
              <b>{best ? comboValue(c as Combination, g.unit, best.value) : DASH}</b>
              {best ? <span className="sub">{`${controlText}${c?.metric === 'profit_per_draw' && c.measure.capped === false ? ' · uncapped' : ''}`}</span> : null}
            </td>
            <td className="num hide-sm">{best ? count(best.draws, g.unit) : DASH}</td>
            <td className="hide-sm">
              {singleText}
              {beats ? <span className="sub">{beats}</span> : null}
            </td>
          </tr>
        );
      })}
    </>
  );
}
