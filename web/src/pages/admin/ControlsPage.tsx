import { Fragment, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getAdminControls, type ControlsView, type GameControls, type PartHistory } from '../../api';
import { DASH, count, dayYear, fixed, perUnit, range, short, signedFixed, span, stamp, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam } from '../../hooks/useUrlState';
import Tile from '../../components/Tile';
import { Card, Legend, Scroll } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import DotStrip, { Axis, ticks } from '../../components/admin/DotStrip';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import Streak, { historyNote, readingNote, streakWords } from '../../components/admin/Streak';
import ControlsDrawer, { chanceText } from './ControlsDrawer';
import { GameButton, GroupRow, chanceLine, earliest, freshText, isMarketGame, joinParts, latest, pText, q0TileFoot, seedsText, tableNote, units } from './bits';

// /admin/controls: the four controls for all 11 games on one page (Q0 row tests, Q2 randomness, feature control,
// market rows), games grouped Lottery / Markets, and per game a drawer with all four in full (?game=lotto). Every
// figure carries its unit and its record's time; nothing is coloured as good or bad. Each control carries its run of
// weeks from the weekly readings (data/readings/<sunday>.json): a grey "3rd week" for a state that holds, a coloured
// "new this week" only where it reads differently from the week before. Q0 is the daily report's row test since
// 27 Oct 2026 (the other three are the Sunday controls); its old null band is context in the drawer.
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
  // the Sunday controls' span: the three weekly ones; Q0's row tests are the daily report's, and its band (context) is
  // monthly since 1 Nov 2026 - named apart with its own day, so a band from an earlier Sunday does not stretch the span
  const from = earliest([t.q2.from, t.fc.from, t.rows.from]);
  const to = latest([t.q2.at, t.fc.at, t.rows.at]);
  const bandDay = t.q0.band.from ?? t.q0.band.at;
  const note = historyNote(d.readings, all.flatMap((g) => [g.history.q0, g.history.q2, g.history.fc, g.history.rows]));
  return (
    <div className="admin">
      <PageHead
        title="Controls"
        sub={`Sunday controls of ${weekday(from)} · ${all.length} games`}
        meta={[span(from, to), bandDay ? `Q0 band (monthly, context) of ${weekday(bandDay)}` : '', 'Brussels time']}
      />
      {note ? <p className="statusline mut">{note}</p> : null}
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
  const absent = (missing: string[], bad: string[], what: string) =>
    joinParts([missing.length ? `no ${what}: ${missing.join(', ')}` : '', bad.length ? `unreadable: ${bad.join(', ')}` : '']);
  return (
    <div className="kpis k4">
      <Q0Tile d={d} />
      <Tile
        href="#q2"
        label="Q2 randomness"
        value={String(t.q2.above)}
        unit={`of ${t.q2.of} above`}
        foot={[
          joinParts([`lottery ${q2Above('Lottery')}`, seedsText(t.q2.seeds.Lottery), `markets ${q2Above('Markets')}`, seedsText(t.q2.seeds.Markets)]),
          absent(t.q2.missing, t.q2.unreadable, 'record'),
          t.q2.tooFewGames.length ? `${t.q2.tooFewGames.length} too few draws, under ${d.meta.q2.fewWindows ?? DASH} test windows (${t.q2.tooFewGames.join(', ')})` : '',
          t.q2.failedGames.length ? `${t.q2.failedGames.length} failed: a comparison scored under 2 seeds (${t.q2.failedGames.join(', ')})` : '',
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
          t.rows.frequencyGames ? `${reference} beats the frequencies in ${t.rows.referenceAboveFrequency} of ${t.rows.frequencyGames}` : '',
          span(t.rows.from, t.rows.at),
        ]}
      />
    </div>
  );
}

// The Q0 tile (27 Oct 2026): the rows confirmed on new draws in the hit-ranked games, this week's figures under it
// (q0TileFoot, shared with the Week page).
function Q0Tile({ d }: { d: ControlsView }) {
  const t = d.tiles.q0;
  const unitOf = (game: string) => d.groups.flatMap((g) => g.games).find((g) => g.game === game)?.unit ?? 'draws';
  return <Tile href="#q0" label="Q0 row tests" value={String(t.confirmed)} unit="confirmed on new draws" foot={q0TileFoot(t, unitOf)} />;
}

// --- week over week ----------------------------------------------------------------------
// A control's streak as a sub line under the game's name, or nothing when there is nothing to say.
function StreakSub({ h, what }: { h: PartHistory | null; what?: string }) {
  return streakWords(h, what) ? (
    <span className="sub">
      <Streak h={h} what={what} />
    </span>
  ) : null;
}

// --- Q0 -------------------------------------------------------------------------------
function Q0Card({ d, onOpen }: { d: ControlsView; onOpen: (g: string) => void }) {
  const m = d.meta.q0;
  const tile = d.tiles.q0;
  return (
    <Card
      id="q0"
      title="Q0 chance per row"
      flush
      help={
        <>
          Each row is tested on its own live draws (the ticket saved before the sales close; a lottery ticket rewritten after the draw counts as it
          stood at the close), on this card&apos;s hit scale. Lottery: the chance that fair draws give tickets of the same shapes (6 or 7 numbers, distinct
          digits, Joker+ runs, numbers that cannot be drawn) at least as many hits over as many draws, computed exactly - the row&apos;s fair level is its
          own, not one band for the game. Markets: the row&apos;s own tickets shuffled among its own live days within blocks of 7 consecutive live days (weeks
          for the week games), the outcomes fixed, 20,000 times. A row that calls the same bins every day scores the same shuffled and cannot pass, and the
          bins a few days favour - the stretch&apos;s bin mix and the drift of its volatility regime - stay with those days, so they do not count. What it
          does not remove: a row that times the regime within a week can still pass; that is real, but it carries no direction, and the market rows&apos;
          score against GARCH (the Market rows card) stays the reference for it. Under 5%: rows with p &lt; 0.05; with m rows, fair draws put about the
          figure next to it there every week by chance alone. Holds: still under 5% after the Holm correction for the m rows compared - on fair draws a card
          shows one about one week in thirty, so it is a figure, not a colour or a streak. Followed: from the first Sunday a row&apos;s live record read p
          &lt; 0.05, its new live draws are kept apart and judged once, after as many of them as flagged it (at least 10, at most 52): confirmed when p
          &lt; 0.01, not confirmed otherwise; stopped after four weeks without a new live draw. Only a confirmed row counts - in the tile, the colours and
          the weeks. A row is tested once it has 10 live draws, days or weeks (the week games: about mid December 2026). jokerplus, pick3 and keno rank on a
          profit bound; they are tested on hits here, information only, out of the tile. Each p stands
          with the row&apos;s own hits per draw, live draws and null on its line. The old null band - the best of 8 untuned rows on 120 fair draws - is
          context in a game&apos;s drawer; nothing is read against it. Weeks, on every card: a control&apos;s run of weeks in the stored weekly readings, ending
          at its record&apos;s Sunday - a grey &quot;3rd week&quot; while it reads the same, a coloured &quot;new this week&quot; with what it read before
          when it changed (here: the rows confirmed and the rows followed). A reading from before 1 Nov 2026 was read under the band rule, so Q0&apos;s weeks
          start again then.
        </>
      }
      meta={['hits per draw (week games: per week), on live draws', "p one-sided; Holm at 5% over each game's rows", `report ${stamp(tile.at)}`, m.band ? 'band: context only' : '']}
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th>Lowest p</th>
              <th className="hide-sm">Under 5%</th>
              <th className="last-1col">Holds</th>
              <th className="hide-1col">New draws</th>
            </tr>
          </thead>
          <tbody>
            {d.groups.map((grp) => (
              <Q0Group key={grp.name} name={grp.name} games={grp.games} rankingAt={d.rankingAt} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

function Q0Group({ name, games, rankingAt, onOpen }: { name: string; games: GameControls[]; rankingAt: string | null; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={5}>{name}</GroupRow>
      {games.map((g) => {
        const q = g.q0;
        const t = q.test;
        const payout = q.ranking === 'profit';
        const under = q.rows.filter((r) => r.inFamily && r.p !== null && r.p < (t?.alpha ?? 0.05));
        const fresh = q.rows.filter((r) => r.fresh);
        const head = (
          <td className="sym">
            <GameButton game={g.game} onOpen={onOpen} />
            {payout ? <span className="sub">tested on hits; ranks on profit · information only</span> : null}
            {/* its weeks: the rows confirmed and followed read the same ("3rd week"), or what changed */}
            <StreakSub h={g.history.q0} />
            {readingNote(g.history.q0, rankingAt) ? <span className="sub">{readingNote(g.history.q0, rankingAt)}</span> : null}
          </td>
        );
        // no test: nothing was compared, which is never "none under"
        if (!t) {
          return (
            <tr key={g.game}>
              {head}
              <td colSpan={4} className="mut">
                no row test: {q.error ?? 'not in the report'}
              </td>
            </tr>
          );
        }
        if (!t.family) {
          return (
            <tr key={g.game}>
              {head}
              <td colSpan={4} className="mut">
                too few live {units(g.unit)}: {t.liveMax} of {t.minDraws ?? DASH} - no row is tested before it has {count(t.minDraws, g.unit)} live
              </td>
            </tr>
          );
        }
        const low = q.lowest;
        return (
          <tr key={g.game}>
            {head}
            <td>
              {low ? (
                <>
                  <b>p {pText(low.p)}</b> {short(low.row)}
                  <span className="sub">{chanceLine(low, g.unit, t.method, false)}</span>
                </>
              ) : (
                <span className="mut">{DASH}</span>
              )}
              <span className="sub">
                {t.family} of {q.allRows} rows with ≥ {count(t.minDraws, g.unit)} live
              </span>
              <span className="sub show-sm">
                {t.under} under 5%, ≈{fixed(t.expected, 2)} by chance · {t.holds} hold
              </span>
              {q.leader ? <span className="sub">profit-bound leader: {short(q.leader.row)}</span> : null}
            </td>
            <td className="hide-sm">
              <b>{t.under}</b> of {t.family} <span className="mut">· ≈{fixed(t.expected, 2)} by chance</span>
              {under.map((r) => (
                <span key={r.row} className="sub">
                  {chanceLine(r, g.unit, t.method)}
                </span>
              ))}
            </td>
            <td className="last-1col">
              <b>{t.holds}</b> of {t.family}
              {/* no bare corrected p here: each row's sits on its own line, with its figures (the plan-card trust rule) */}
              <span className="sub">under 5% after the correction for the {t.family} rows compared</span>
            </td>
            <td className="hide-1col">
              {t.followed} followed · {t.confirmed} confirmed · {t.notConfirmed} not
              {fresh.map((r) => (r.fresh ? (
                <span key={r.row} className="sub">
                  {short(r.row)}: {freshText(r.fresh, g.unit, t.method, t.verdictAlpha)}
                </span>
              ) : null))}
            </td>
          </tr>
        );
      })}
    </>
  );
}

// --- Q2 -------------------------------------------------------------------------------
const Q2_CHIP: Record<string, 'hi' | 'dim' | 'flat'> = { above: 'hi', 'too few draws': 'dim', within: 'flat', 'above band, not shuffled': 'flat', failed: 'flat' };
// "± 0.019" after a threshold read under rule 2 (27 Oct 2026); nothing after an older one, which has no error
const pmText = (x: number | null) => (x === null ? '' : ` ± ${fixed(x)}`);

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
          Dot: how well the classifiers tell the real draws from fair synthetic ones (best AUC of the suite, mean over seeds). Tick: the threshold, the highest
          AUC one more run of fair draws against fair draws should reach - the null mean + k sd, the 97.7% bound for one run, with k from the seed count (2.19
          at 20 seeds, wider on fewer); the short line through it is its own uncertainty (± its standard error), the shaded part the null band. Hollow dot: the
          real draws reordered, which keeps every frequency and destroys time order (for a market the returns are reordered and the bins re-cut from them, so
          its bin frequencies move a little); the faint tick is its bound, worked out the same way. A game is above when its dot is right of both ticks. Too few
          draws: under {d.meta.q2.fewWindows ?? DASH} test windows, fixed by the history&apos;s length before the run - the band there is too wide to show
          anything. Failed: a comparison scored fewer than 2 seeds, so there is no band. A record from before 1 Nov 2026 was read under the old rule (null mean
          + 2 sd of 3 seeds, 5 for the markets) and has no ± and no shuffled bound. {seedsText(d.tiles.q2.seeds.Lottery)} for lottery games,{' '}
          {seedsText(d.tiles.q2.seeds.Markets)} for markets. The suite: logistic regression, an RBF SVM, a random forest, gradient boosting and a small network,
          plus a quantum kernel and a VQC unless the run skipped them (a game&apos;s drawer says which).
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
              { mark: 'se', label: '± its error' },
              { mark: 'tick own', label: 'shuffled bound' },
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
              const se = q.thresholdSe;
              // the caveat first on the line that carries the numbers: why this record cannot be read as above or within
              const why =
                q.status === 'too few draws'
                  ? `too few draws · ${q.windowsTest ?? DASH} test windows (under ${q.minWindows ?? DASH})`
                  : q.status === 'failed'
                    ? 'failed: a comparison scored under 2 seeds'
                    : '';
              return (
                <div key={g.game} className={few ? 'q2r dim' : 'q2r'}>
                  <span className="g">
                    <GameButton game={g.game} onOpen={onOpen} />
                    <Chip kind={Q2_CHIP[q.status] ?? 'flat'}>{q.status}</Chip>
                    <Streak h={g.history.q2} />
                  </span>
                  <DotStrip
                    scale={scale}
                    dim={few}
                    grid={grid.map((x) => x.at)}
                    spans={[
                      { from: lo, to: q.threshold, kind: 'zone' },
                      { from: q.threshold !== null && se !== null ? q.threshold - se : null, to: q.threshold !== null && se !== null ? q.threshold + se : null, kind: 'se' },
                    ]}
                    marks={[
                      { at: q.threshold, kind: 'tick' },
                      { at: q.shuffledThreshold, kind: 'tickOwn' },
                      { at: q.shuffled, kind: 'hollow' },
                      { at: q.real, kind: 'dot' },
                    ]}
                    label={`${g.game}: real ${fixed(q.real)}, threshold ${fixed(q.threshold)}${pmText(se)}, shuffled ${fixed(q.shuffled)}${q.shuffledThreshold !== null ? `, its bound ${fixed(q.shuffledThreshold)}` : ''}`}
                  />
                  <span className="txt">
                    <b>{fixed(q.real)}</b> vs {fixed(q.threshold)}
                    {pmText(se)}
                    <small>
                      {why ? `${why} · ` : ''}shuffled {fixed(q.shuffled)}
                      {q.shuffledThreshold !== null ? `, bound ${fixed(q.shuffledThreshold)}` : ''} · {q.seeds ?? DASH} seeds
                      {q.status === 'too few draws' ? '' : ` · ${q.windowsTest ?? DASH} test windows`}
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
  // how often one of a game's base-model columns clears the band by chance alone (the old card's figure)
  const chance = chanceText(games.map((g) => g.fc?.columns));
  return (
    <Card
      id="fc"
      title="Feature control"
      flush
      help={
        <>
          Each variant ({VARIANTS.map((k) => `${k} ${d.meta.fc.variants[k] ?? k}`).join(', ')}) is trained with noise columns added: shuffled copies of real
          base-model scores, so they look exactly like a base model and mean nothing. Fits noise (filled): a noise column matters at least a tenth as much as an
          average real column in training and the repeats agree (t &gt; 2). Inconclusive (hollow): the ratio is exceeded but the repeats disagree. Share: the
          noise columns&apos; share of training importance; the tick is this table&apos;s chance share, noise columns / (columns + noise columns). Importance is
          the ranking power lost when a column is scrambled (the drop in |AUC − 0.5|), so a model whose probabilities came out inverted still shows what it
          leans on. Above the noise band: real columns whose held-out importance beats the noise mean + 2 sd{chance ? `; ${chance}` : ''}: a column counts
          when it stays above week after week, not when it appears once. Table: one row per draw, so its length is in draws, days, trading days or weeks; the
          lockbox rows are withheld from it. Behind: the table ends that many draws before the newest draw of the game&apos;s data file. Positional: the table is built
          per position (one classifier per slot).
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
              <StreakSub h={g.history.fc} />
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
              <span className="sub">range over the variants, without and with noise</span>
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
          interval. A row beats {reference} only when the whole interval is above zero. Beating uniform can be no more than knowing which bins the period
          favoured, so each row is also held against the frequencies - each instrument&apos;s bins as often as they came up before the day - and is credited
          only when it beats those too. The last column shows {reference} itself against uniform and against the frequencies. Window: the
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
                <th className="hide-sm">Beat the frequencies</th>
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
                      <td colSpan={5} className="mut">
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
                      <StreakSub h={g.history.rows} />
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
                      {r.frequency ? (
                        <span className="sub show-sm">
                          beat the frequencies: {r.betterThanFrequency.length} of {r.rowsScored}
                        </span>
                      ) : null}
                    </td>
                    <td className="hide-sm">
                      <b>{r.betterThanReference.length}</b> of {r.rowsScored}
                      <span className="sub">{r.noDifferenceVsReference} no difference</span>
                    </td>
                    <td className="hide-sm">
                      {/* the frequencies: the column-frequency baseline (MarketRows.py, 26 Oct 2026); a report from before has none */}
                      {r.frequency ? (
                        <>
                          <b>{r.betterThanFrequency.length}</b> of {r.rowsScored}
                          {r.betterThanFrequency.length ? <span className="sub">{r.betterThanFrequency.map(short).join(', ')}</span> : null}
                          {r.frequencyVsUniform ? (
                            <span className="sub">
                              the frequencies alone vs uniform: {signedFixed(r.frequencyVsUniform.mean)} · {r.frequencyVsUniform.verdict}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className="mut">not in this report</span>
                      )}
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
                          {r.referenceVsFrequency ? (
                            <span className="sub">
                              vs the frequencies: {signedFixed(r.referenceVsFrequency.mean)} ({signedFixed(r.referenceVsFrequency.lo)} to{' '}
                              {signedFixed(r.referenceVsFrequency.hi)}) · {r.referenceVsFrequency.verdict}
                            </span>
                          ) : null}
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
