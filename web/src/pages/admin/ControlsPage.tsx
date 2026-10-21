import { Fragment, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getAdminControls, type ControlsView, type GameControls } from '../../api';
import { DASH, count, dayYear, fixed, perUnit, range, short, signedFixed, span, stamp, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam } from '../../hooks/useUrlState';
import Tile from '../../components/Tile';
import { Card, Legend, Scroll } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import DotStrip, { Axis, ticks } from '../../components/admin/DotStrip';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import ControlsDrawer from './ControlsDrawer';
import { GameButton, GroupRow, earliest, isMarketGame, joinParts, latest, oneControlText, recordText, seedsText, tableNote, units } from './bits';

// /admin/controls: the four Sunday controls for all 11 games on one page (Q0 null band, Q2 randomness,
// feature control, market rows), games grouped Lottery / Markets, and per game a drawer with all four in
// full (?game=lotto). Every figure carries its unit and its record's time; nothing is coloured as good or bad.
export default function ControlsPage() {
  const load = useLoad(getAdminControls, []);
  const drawer = useDrawerParam();
  const { hash } = useLocation();

  useEffect(() => {
    document.title = 'Controls · Predictor';
  }, []);
  // the menu's anchors (#q0, #q2, #fc, #rows) scroll to their card once it is on the page
  useEffect(() => {
    if (load.kind === 'ok' && hash) document.getElementById(hash.slice(1))?.scrollIntoView({ block: 'start' });
  }, [hash, load.kind]);

  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title="Controls" />
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  const t = d.tiles;
  const all = d.groups.flatMap((g) => g.games);
  const from = earliest([t.q0.from, t.q2.from, t.fc.from, t.rows.from]);
  const to = latest([t.q0.at, t.q2.at, t.fc.at, t.rows.at]);
  return (
    <div className="admin">
      <PageHead title="Controls" sub={`Sunday controls of ${weekday(from)} · ${all.length} games`} meta={[span(from, to), 'Brussels time']} />
      <Tiles d={d} />
      <div className="stack">
        <Q0Card d={d} onOpen={drawer.open} />
        <Q2Card d={d} onOpen={drawer.open} />
        <FcCard d={d} onOpen={drawer.open} />
        <RowsCard d={d} onOpen={drawer.open} />
      </div>
      {drawer.game ? <ControlsDrawer key={drawer.game} game={drawer.game} onClose={drawer.close} /> : null}
    </div>
  );
}

// --- tiles ----------------------------------------------------------------------
function Tiles({ d }: { d: ControlsView }) {
  const t = d.tiles;
  const games = d.groups.flatMap((g) => g.games);
  const hitRanked = games.filter((g) => g.q0.band && g.q0.ranking === 'hits' && g.q0.above.n > 0);
  // the longest record among the rows above, per side: a lottery game's draws, a market game's live part only
  const lottery = hitRanked.filter((g) => !isMarketGame(g.game)).sort((a, b) => (b.q0.above.longest ?? 0) - (a.q0.above.longest ?? 0))[0];
  const market = hitRanked.filter((g) => isMarketGame(g.game)).sort((a, b) => (b.q0.above.longestLive ?? 0) - (a.q0.above.longestLive ?? 0))[0];
  const records = [
    lottery ? `lottery ≤ ${count(lottery.q0.above.longest, lottery.unit)}` : '',
    market ? `markets ≤ ${market.q0.above.longestLive ?? 0} live ${market.q0.above.longestLive === 1 ? perUnit(market.unit) : units(market.unit)}` : '',
  ].filter(Boolean).join(' · ');
  const q2Above = (group: string) => d.groups.find((g) => g.name === group)?.games.filter((g) => g.q2?.status === 'above').length ?? 0;
  const rowsGames = games.filter((g) => g.rows);
  const scored = rowsGames.reduce((s, g) => s + (g.rows?.rowsScored ?? 0), 0);
  const reference = short(rowsGames[0]?.rows?.reference ?? 'GARCH Model');
  // the market-rows windows per unit: "231 days · 189 trading days · 110-237 weeks"
  const byUnit = new Map<string, number[]>();
  t.rows.windows.forEach((w) => {
    if (w.window !== null) byUnit.set(w.unit, [...(byUnit.get(w.unit) ?? []), w.window]);
  });
  const windows = [...byUnit.entries()].map(([unit, xs]) => `${range({ min: Math.min(...xs), max: Math.max(...xs) })} ${units(unit)}`).join(' · ');
  // the Q0 run spans Sunday into Monday (Brussels time); records more than a day apart are two weeks' runs mixed
  const q0Mixed = Boolean(t.q0.from && t.q0.at && Date.parse(t.q0.at) - Date.parse(t.q0.from) > 24 * 3600 * 1000);
  // a game out of a tile's count, named: no record yet, or one that is there but could not be read
  const unreadable = (part: string) => (game: string) => (games.find((g) => g.game === game)?.errors.includes(part) ? `${game} (unreadable)` : game);
  const absent = (missing: string[], bad: string[], what: string) =>
    joinParts([missing.length ? `no ${what}: ${missing.join(', ')}` : '', bad.length ? `unreadable: ${bad.join(', ')}` : '']);
  return (
    <div className="kpis k4">
      <Tile
        href="#q0"
        label="Q0 null band"
        value={String(t.q0.rowsAbove)}
        unit="rows above band"
        foot={[
          `${t.q0.games} of ${t.q0.hitRankedGames} hit-ranked games`,
          t.q0.noBand.length ? `no band, not compared: ${t.q0.noBand.map(unreadable('q0')).join(', ')}` : '',
          t.q0.oneControl.length ? `band from one control: ${t.q0.oneControl.map((x) => `${x.game} ${oneControlText(x)}`).join(', ')}` : '',
          records,
          `rule open: ${t.q0.alternatives.allGamesOnHits.rowsAbove} with payout games · ${t.q0.alternatives.bestRowOnly.games.length} best rows only`,
          `bands ${span(t.q0.from, t.q0.at)}${q0Mixed ? ' · mixed weeks: run in progress' : ''}`,
          `ranking ${stamp(d.rankingAt)}`,
        ]}
      />
      <Tile
        href="#q2"
        label="Q2 randomness"
        value={String(t.q2.above)}
        unit={`of ${t.q2.of} above`}
        foot={[
          joinParts([`lottery ${q2Above('Lottery')}`, seedsText(t.q2.seeds.Lottery), `markets ${q2Above('Markets')}`, seedsText(t.q2.seeds.Markets)]),
          absent(t.q2.missing, t.q2.unreadable, 'record'),
          ...t.q2.aboveFewWindows.map((w) => `${w.game} above on ${w.windowsTest ?? DASH} test windows (under ${d.meta.q2.fewWindows})`),
          t.q2.tooFewGames.length ? `${t.q2.tooFewGames.length} too few draws (${t.q2.tooFewGames.join(', ')})` : '',
          span(t.q2.from, t.q2.at),
        ]}
      />
      <Tile
        href="#fc"
        label="Feature control"
        value={String(t.fc.fits)}
        unit={`of ${t.fc.variants} fit noise`}
        foot={[
          `${t.fc.inconclusive} inconclusive · ${t.fc.ignored} ignored · ${t.fc.failed} failed · ${t.fc.games} games`,
          absent(t.fc.missing, t.fc.unreadable, 'record'),
          span(t.fc.from, t.fc.at),
        ]}
      />
      <Tile
        href="#rows"
        label="Market rows"
        value={String(t.rows.beatReference)}
        unit={`of ${scored} rows beat ${reference}`}
        foot={[
          joinParts([`${t.rows.games} games`, windows]),
          absent(t.rows.missing.filter((g) => !t.rows.unreadable.includes(g)), t.rows.unreadable, 'report'),
          `${reference} beats uniform in ${t.rows.referenceAboveUniform} of ${t.rows.games}`,
          span(t.rows.from, t.rows.at),
        ]}
      />
    </div>
  );
}

// --- Q0 -------------------------------------------------------------------------------
function Q0Card({ d, onOpen }: { d: ControlsView; onOpen: (g: string) => void }) {
  const m = d.meta.q0;
  const tile = d.tiles.q0;
  return (
    <Card
      id="q0"
      title="Q0 null band"
      flush
      help={
        <>
          Band: the best of the tracked rows on fair histories plus 2 sd across seeds, from whichever control is higher, synthetic or shuffled; its time is that
          control&apos;s own record. A band from one control only (the other&apos;s record missing, or there but unreadable) is marked as such: it can be
          lower than the rule&apos;s. Best row: the ranked row with the most hits per draw, with its record; for market games the days whose ticket was not out
          before the close (back-built or late) are counted apart. Above: ranked rows ({'≥'} their game&apos;s minimum draws) whose hits per draw beat their
          game&apos;s band, and the longest record among them. Own level: the row&apos;s own mean on the fair histories in the control the band comes from, only
          for rows the control runs (on narrow screens it is in the game&apos;s drawer). jokerplus, pick3 and keno rank on a profit bound; here they are compared
          on hits, and they stay out of the headline count. The rule is still open: {tile.rowsAbove} rows on the hit-ranked games,{' '}
          {tile.alternatives.allGamesOnHits.rowsAbove} if the payout games count on hits, {tile.alternatives.bestRowOnly.games.length} if only each game&apos;s best
          row counts.
        </>
      }
      meta={[
        'hits per draw (week games: per week)',
        `best of ${range(m.rows)} rows + 2 sd on fair histories of ${range(m.history)} draws, days or weeks`,
        span(tile.from, tile.at),
        `ranking ${stamp(d.rankingAt)}`,
      ]}
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th className="hide-sm">Band</th>
              <th>Best row</th>
              <th className="last-1col">Above</th>
              <th className="hide-1col">Own level</th>
            </tr>
          </thead>
          <tbody>
            {d.groups.map((grp) => (
              <Q0Group key={grp.name} name={grp.name} seeds={m.seeds[grp.name]} games={grp.games} headline={tile.rowsAbove} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

function Q0Group({ name, seeds, games, headline, onOpen }: { name: string; seeds: number[] | undefined; games: GameControls[]; headline: number; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={5}>
        {name}
        {seeds?.length ? ` · ${seedsText(seeds)}` : ''}
      </GroupRow>
      {games.map((g) => {
        const q = g.q0;
        const b = q.band;
        const payout = q.ranking === 'profit';
        const market = isMarketGame(g.game);
        const owned = q.above.rows.filter((r) => r.own !== null);
        // no band: nothing was compared, which is not "none above"
        const noBand = g.errors.includes('q0') ? 'record unreadable' : 'no band yet';
        return (
          <tr key={g.game}>
            <td className="sym">
              <GameButton game={g.game} onOpen={onOpen} />
              {payout ? <span className="sub">compared on hits; ranks on profit</span> : null}
              {/* the phone line keeps what the band is built on and when, as the hidden Band column does */}
              <span className="sub show-sm">
                {b
                  ? joinParts([`band ${fixed(b.ceiling)}`, oneControlText(b) || b.from, `best of ${b.rows ?? DASH} rows, histories of ${count(b.history, g.unit)}`, weekday(b.at)])
                  : noBand}
              </span>
            </td>
            <td className="hide-sm">
              {b ? (
                <>
                  <b>{fixed(b.ceiling)}</b>
                  <span className="sub">
                    {oneControlText(b) || b.from} · {b.rows ?? DASH} rows · histories of {count(b.history, g.unit)} · {stamp(b.at)}
                  </span>
                </>
              ) : (
                <span className="mut">{noBand}</span>
              )}
            </td>
            <td>
              {q.best ? (
                <>
                  <b>{fixed(q.best.hits)}</b> {short(q.best.row)}
                  <span className="sub">{recordText(q.best.draws, g.unit, market ? q.best.live : null)}</span>
                </>
              ) : (
                <span className="mut">{DASH}</span>
              )}
              {q.leader ? <span className="sub">profit-bound leader: {short(q.leader.row)}</span> : null}
            </td>
            <td className="last-1col">
              {!b ? (
                <>
                  <span className="fnt">{DASH}</span>
                  <span className="sub">no band: not compared</span>
                </>
              ) : (
                <>
                  <b>{q.above.n}</b> of {q.rankedRows}
                  <span className="sub">
                    {q.above.n === 0
                      ? 'none above'
                      : `longest ${count(q.above.longest, g.unit)}${market ? `, ${q.above.sameLength ? '' : '≤'}${q.above.longestLive ?? 0} live` : ''}`}
                    {payout && q.above.n ? ` · not in the tile's ${headline}` : ''}
                  </span>
                </>
              )}
            </td>
            <td className="hide-1col">
              {!b ? (
                <span className="sub">no band: not compared</span>
              ) : q.above.n === 0 ? (
                <span className="fnt">{DASH}</span>
              ) : owned.length ? (
                <>
                  <b>{owned.length}</b> of {q.above.n}
                  <span className="sub">
                    {owned.map((r) => `${short(r.row)} ${fixed(r.own)}`).join(' · ')} · {b?.from} control
                  </span>
                </>
              ) : (
                <>
                  <span className="mut">none of {q.above.n}</span>
                  <span className="sub">the control does not run them</span>
                </>
              )}
            </td>
          </tr>
        );
      })}
    </>
  );
}

// --- Q2 -------------------------------------------------------------------------------
const Q2_CHIP: Record<string, 'hi' | 'dim' | 'flat'> = { above: 'hi', 'too few draws': 'dim', within: 'flat', 'above band, not shuffled': 'flat' };

function Q2Card({ d, onOpen }: { d: ControlsView; onOpen: (g: string) => void }) {
  const [lo, hi] = d.meta.q2.axis;
  const scale = { min: lo, max: hi };
  const grid = ticks(lo, hi, 0.1, (v) => v.toFixed(1));
  const games = d.groups.flatMap((g) => g.games).filter((g) => g.q2);
  const windows = [...new Set(games.map((g) => g.q2?.window))].filter((x) => x !== null);
  const caps = [...new Set(games.map((g) => g.q2?.drawsCap))].filter((x) => x !== null);
  return (
    <Card
      id="q2"
      title="Q2 randomness"
      flush
      help={
        <>
          Dot: how well the classifiers tell the real draws from fair synthetic ones (best AUC of the suite, mean over seeds). Tick: the threshold, the null mean
          + 2 sd, where the null is fair draws against fair draws; the shaded part is the null band. Hollow dot: the real draws reordered, which keeps every
          frequency and destroys time order. A game is above when its dot is right of the tick and right of the hollow dot. A threshold of 0.8 or more means too
          few draws to judge; the record&apos;s own power rule puts that case below about {d.meta.q2.fewWindows} test windows, so a game above on fewer windows is
          read with that in mind. {seedsText(d.tiles.q2.seeds.Lottery)} for lottery games, {seedsText(d.tiles.q2.seeds.Markets)} for markets: a band on few seeds is wide and moves from week to week.
        </>
      }
      meta={[
        'best classifier AUC, mean over seeds',
        windows.length === 1 && caps.length === 1 ? `windows of ${windows[0]} · at most the last ${caps[0]} draws, days or weeks` : '',
        span(d.tiles.q2.from, d.tiles.q2.at),
      ]}
      foot={
        <>
          <Legend
            items={[
              { mark: 'dot', label: 'real' },
              { mark: 'hol', label: 'shuffled' },
              { mark: 'tick', label: 'threshold' },
              { mark: 'zone', label: 'null band' },
            ]}
          />
          <span className="fnote">
            AUC axis {lo.toFixed(1)} – {hi.toFixed(1)}
          </span>
        </>
      }
    >
      <div className="q2rows">
        <div className="q2r head">
          <span>Game · status</span>
          <Axis scale={scale} ticks={grid} />
          <span>Real vs threshold</span>
        </div>
        {d.groups.map((grp) => (
          <Fragment key={grp.name}>
            <div className="q2r grp">{grp.name}</div>
            {grp.games.map((g) => {
              const q = g.q2;
              if (!q) {
                return (
                  <div key={g.game} className="q2r dim">
                    <span className="g">
                      <GameButton game={g.game} onOpen={onOpen} />
                    </span>
                    <span className="mut">{g.errors.includes('q2') ? 'record unreadable' : 'no record yet'}</span>
                    <span />
                  </div>
                );
              }
              const few = q.status === 'too few draws';
              return (
                <div key={g.game} className={few ? 'q2r dim' : 'q2r'}>
                  <span className="g">
                    <GameButton game={g.game} onOpen={onOpen} />
                    <Chip kind={Q2_CHIP[q.status] ?? 'flat'}>{q.status}</Chip>
                  </span>
                  <DotStrip
                    scale={scale}
                    dim={few}
                    grid={grid.map((x) => x.at)}
                    spans={[{ from: lo, to: q.threshold, kind: 'zone' }]}
                    marks={[
                      { at: q.threshold, kind: 'tick' },
                      { at: q.shuffled, kind: 'hollow' },
                      { at: q.real, kind: 'dot' },
                    ]}
                    label={`${g.game}: real ${fixed(q.real)}, threshold ${fixed(q.threshold)}, shuffled ${fixed(q.shuffled)}`}
                  />
                  <span className="txt">
                    <b>{fixed(q.real)}</b> vs {fixed(q.threshold)}
                    <small>
                      shuffled {fixed(q.shuffled)} · {q.seeds ?? DASH} seeds · {q.windowsTest ?? DASH} test windows
                      {q.fewWindows ? ` (under ${d.meta.q2.fewWindows})` : ''}
                    </small>
                  </span>
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </Card>
  );
}

// --- feature control ----------------------------------------------------------------------
const VARIANTS = ['lr', 'gb', 'qk', 'qv', 'svm'];
const markOf = (verdict: string) => (verdict === 'fits noise' ? 'm' : verdict === 'inconclusive' ? 'm o' : verdict === 'noise ignored' ? 'm x' : 'm f');

function FcCard({ d, onOpen }: { d: ControlsView; onOpen: (g: string) => void }) {
  const games = d.groups.flatMap((g) => g.games);
  const shares = games.flatMap((g) => (g.fc ? g.fc.variants.map((v) => v.share ?? 0) : []));
  const top = Math.max(0.5, Math.ceil(Math.max(0, ...shares) * 10) / 10);
  const scale = { min: 0, max: top };
  const axis = ticks(0, top, 0.2, (v) => `${Math.round(v * 100)}%`);
  const grid = ticks(0, top, 0.1, String).map((x) => x.at);
  const repeats = [...new Set(games.map((g) => g.fc?.repeats).filter((x) => x !== null && x !== undefined))];
  const noise = [...new Set(games.map((g) => g.fc?.noiseColumns).filter((x) => x !== null && x !== undefined))];
  return (
    <Card
      id="fc"
      title="Feature control"
      flush
      help={
        <>
          Each variant ({VARIANTS.map((k) => `${k} ${d.meta.fc.variants[k] ?? k}`).join(', ')}) is trained with pure-noise columns added. Fits noise (filled): a
          noise column matters at least a tenth as much as an average real column in training and the repeats agree (t &gt; 2). Inconclusive (hollow): the
          ratio is exceeded but the repeats disagree. Share: the noise columns&apos; share of training importance; the tick is this table&apos;s chance share,
          noise columns / (columns + noise columns). Above the noise band: real columns whose held-out importance beats the noise mean + 2 sd. Table: one row
          per draw, so its length is in draws, days, trading days or weeks; the lockbox rows are withheld from it. Behind: the table ends that many draws before
          the newest draw of the game&apos;s data file. Positional: the table is built per position (one classifier per slot).
        </>
      }
      meta={[`${VARIANTS.length} variants · ${noise.join('/') || DASH} noise columns · ${repeats.join('/') || DASH} repeats`, span(d.tiles.fc.from, d.tiles.fc.at)]}
      foot={
        <>
          <Legend
            items={[
              { mark: 'dot', label: 'fits noise' },
              { mark: 'hol', label: 'inconclusive' },
              { mark: 'dot nul', label: 'noise ignored' },
              { mark: 'tick', label: 'chance share' },
            ]}
          />
          <span className="fnote">Share axis 0 – {Math.round(top * 100)}%</span>
        </>
      }
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th>
                Variants
                <br />
                <span className="vd lab" aria-hidden="true">
                  {VARIANTS.map((k) => (
                    <span key={k}>{k}</span>
                  ))}
                </span>
              </th>
              <th className="w-strip">
                Noise share
                <Axis scale={scale} ticks={axis} />
              </th>
              <th className="hide-sm">Above the noise band</th>
              <th className="hide-sm">AUC held-out</th>
              <th className="hide-sm">Table</th>
            </tr>
          </thead>
          <tbody>
            {d.groups.map((grp) => (
              <FcGroup key={grp.name} name={grp.name} games={grp.games} scale={scale} grid={grid} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

function FcGroup({ name, games, scale, grid, onOpen }: { name: string; games: GameControls[]; scale: { min: number; max: number }; grid: number[]; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={6}>{name}</GroupRow>
      {games.map((g) => {
        const f = g.fc;
        if (!f) {
          return (
            <tr key={g.game}>
              <td className="sym">
                <GameButton game={g.game} onOpen={onOpen} />
              </td>
              <td colSpan={5} className="mut">
                {g.errors.includes('fc') ? 'record unreadable' : 'no record yet'}
              </td>
            </tr>
          );
        }
        const byKey = Object.fromEntries(f.variants.map((v) => [v.key, v]));
        // the base-model columns above the noise band, each with the variants that put it there
        const above = new Map<string, string[]>();
        f.variants.forEach((v) => v.above.forEach((row) => above.set(row, [...(above.get(row) ?? []), v.key])));
        const aucs = f.variants.flatMap((v) => v.auc).filter((x): x is number => x !== null);
        const auc = aucs.length ? range({ min: Math.min(...aucs), max: Math.max(...aucs) }, 2) : DASH;
        const counts = [f.fits ? `${f.fits} fit` : '', f.inconclusive ? `${f.inconclusive} inconcl.` : '', f.ignored ? `${f.ignored} ignored` : '', f.failed ? `${f.failed} failed` : '']
          .filter(Boolean)
          .join(' · ');
        return (
          <tr key={g.game}>
            <td className="sym">
              <GameButton game={g.game} onOpen={onOpen} />
              <span className="sub show-sm">{joinParts([`AUC ${auc}`, count(f.tableRows, g.unit), tableNote(f, g.unit)])}</span>
            </td>
            <td className="mid">
              <span className="vd" role="img" aria-label={VARIANTS.map((k) => `${k} ${byKey[k]?.verdict ?? 'not run'}`).join(', ')}>
                {VARIANTS.map((k) => (byKey[k] ? <i key={k} className={markOf(byKey[k].verdict)} /> : <i key={k} />))}
              </span>
              <span className="sub">{counts || DASH}</span>
            </td>
            <td className="w-strip mid last-sm">
              <DotStrip
                scale={scale}
                grid={grid}
                marks={[
                  { at: f.chanceShare, kind: 'tick' },
                  ...f.variants
                    .filter((v) => v.verdict !== 'failed')
                    .map((v) => ({ at: v.share, kind: v.verdict === 'fits noise' ? ('dot' as const) : v.verdict === 'inconclusive' ? ('hollow' as const) : ('null' as const) })),
                ]}
                label={`${g.game}: noise shares ${f.variants.map((v) => (v.share === null ? DASH : `${Math.round(v.share * 100)}%`)).join(', ')}; chance ${f.chanceShare === null ? DASH : `${Math.round(f.chanceShare * 100)}%`}`}
              />
              <span className="sub">
                chance {f.chanceShare === null ? DASH : `${Math.round(f.chanceShare * 100)}%`} · {f.columns ?? DASH} columns
              </span>
              {above.size ? <span className="sub show-sm">above: {[...above.keys()].map(short).join(', ')}</span> : null}
            </td>
            <td className="hide-sm">
              {above.size ? (
                <span className="chips">
                  {[...above.entries()].map(([row, keys]) => (
                    <Chip key={row} kind="model">
                      {short(row)} · {keys.join(' ')}
                    </Chip>
                  ))}
                </span>
              ) : (
                <span className="fnt">{DASH}</span>
              )}
            </td>
            <td className="hide-sm">
              {auc}
              <span className="sub">without / with noise</span>
            </td>
            <td className="hide-sm">
              {count(f.tableRows, g.unit)}
              <span className="sub">
                {f.lockboxRows ?? DASH} lockbox {f.lockboxRows === 1 ? perUnit(g.unit) : units(g.unit)} withheld
              </span>
              {tableNote(f, g.unit) ? <span className="sub">{tableNote(f, g.unit)}</span> : null}
            </td>
          </tr>
        );
      })}
    </>
  );
}

// --- market rows ------------------------------------------------------------------------------
function RowsCard({ d, onOpen }: { d: ControlsView; onOpen: (g: string) => void }) {
  // every market game keeps its row: one without a report says so instead of dropping out
  const all = d.groups.flatMap((g) => g.games).filter((g) => isMarketGame(g.game));
  const games = all.filter((g) => g.rows);
  const intervals = games.map((g) => g.rows?.referenceVsUniform).filter((x) => x !== null && x !== undefined);
  const los = intervals.map((x) => x.lo ?? 0);
  const his = intervals.map((x) => x.hi ?? 0);
  const lo0 = Math.min(0, ...los);
  const hi0 = Math.max(0, ...his);
  const pad = (hi0 - lo0) * 0.12 || 0.01;
  const scale = { min: lo0 - pad, max: hi0 + pad };
  const step = scale.max - scale.min > 0.3 ? 0.1 : 0.05;
  const axis = ticks(scale.min, scale.max, step, (v) => (v === 0 ? '0' : signedFixed(v, 2)));
  const reference = short(games[0]?.rows?.reference ?? 'GARCH Model');
  return (
    <Card
      id="rows"
      title="Market rows"
      flush
      help={
        <>
          Each row&apos;s mean log-score difference per day (per week for the week games) against {reference}, the reference, and against uniform, with a 95%
          interval. A row beats {reference} only when the whole interval is above zero. The last column shows {reference} itself against uniform. Window: the
          days (trading days for shares, weeks for the week games) scored; the lockbox days were withheld from those asked for. Profit and loss is left out: it
          is in price units, unlike every other figure here.
        </>
      }
      meta={[
        games.length ? `${games[0].rows?.rowsTotal ?? DASH} rows per game: ${reference}, ${games[0].rows?.rowsScored ?? DASH} scored, ${games[0].rows?.baselines ?? DASH} baselines` : '',
        span(d.tiles.rows.from, d.tiles.rows.at),
      ]}
      foot={
        <>
          <Legend
            items={[
              { mark: 'dot', label: 'mean' },
              { mark: 'ci', label: '95% interval' },
              { mark: 'tick', label: 'zero' },
            ]}
          />
          <span className="fnote">Log-score per day or week</span>
        </>
      }
    >
      {all.length === 0 ? (
        <div className="empty">No market-rows report yet.</div>
      ) : (
        <Scroll>
          <table className="tbl">
            <thead>
              <tr>
                <th>Game</th>
                <th className="hide-sm">Window</th>
                <th>Beat uniform</th>
                <th className="hide-sm">Beat {reference}</th>
                <th className="w-strip">
                  {reference} vs uniform
                  <Axis scale={scale} ticks={axis} />
                </th>
              </tr>
            </thead>
            <tbody>
              {all.map((g) => {
                const r = g.rows;
                if (!r) {
                  return (
                    <tr key={g.game}>
                      <td className="sym">
                        <GameButton game={g.game} onOpen={onOpen} />
                      </td>
                      <td colSpan={4} className="mut">
                        {g.errors.includes('rows') ? 'report unreadable' : 'no report yet'}
                      </td>
                    </tr>
                  );
                }
                const ref = r.referenceVsUniform;
                return (
                  <tr key={g.game}>
                    <td className="sym">
                      <GameButton game={g.game} onOpen={onOpen} />
                      <span className="sub show-sm">{count(r.window, g.unit)}</span>
                    </td>
                    <td className="hide-sm">
                      <b>{r.window ?? DASH}</b> {units(g.unit)}
                      <span className="sub">
                        {dayYear(r.firstDay)} → {dayYear(r.lastDay)} · {r.lockboxWithheld ?? DASH} lockbox {units(g.unit)} withheld
                      </span>
                    </td>
                    <td>
                      <b>{r.betterThanUniform.length}</b> of {r.rowsScored}
                      {r.betterThanUniform.length ? <span className="sub">{r.betterThanUniform.map(short).join(', ')}</span> : null}
                      <span className="sub show-sm">
                        beat {reference}: {r.betterThanReference.length} of {r.rowsScored} · {r.noDifferenceVsReference} no difference
                      </span>
                    </td>
                    <td className="hide-sm">
                      <b>{r.betterThanReference.length}</b> of {r.rowsScored}
                      <span className="sub">{r.noDifferenceVsReference} no difference</span>
                    </td>
                    <td className="w-strip mid">
                      {ref ? (
                        <>
                          <DotStrip
                            scale={scale}
                            grid={axis.map((x) => x.at)}
                            spans={[{ from: ref.lo, to: ref.hi, kind: 'ci' }]}
                            marks={[
                              { at: 0, kind: 'tick' },
                              { at: ref.mean, kind: 'dot' },
                            ]}
                            label={`${reference} vs uniform: ${signedFixed(ref.mean, 4)} (${signedFixed(ref.lo, 4)} to ${signedFixed(ref.hi, 4)}) per ${perUnit(g.unit)}`}
                          />
                          <span className="sub">
                            {signedFixed(ref.mean)} ({signedFixed(ref.lo)} to {signedFixed(ref.hi)}) per {perUnit(g.unit)} · {ref.verdict}
                          </span>
                        </>
                      ) : (
                        <span className="mut">{DASH}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Scroll>
      )}
    </Card>
  );
}
