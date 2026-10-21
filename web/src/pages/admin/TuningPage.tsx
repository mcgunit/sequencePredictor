import { useEffect } from 'react';
import { GAMES, getAdminTuning, type TunerSummary, type TuningView } from '../../api';
import { DASH, range, short, span, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam, useSwitchParam } from '../../hooks/useUrlState';
import Tile from '../../components/Tile';
import { Card, Legend } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import { StackBar } from '../../components/admin/Bars';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import TuningDrawer from './TuningDrawer';
import { GameButton, joinParts } from './bits';
import { kindsText, marginText, windowText } from './gates';
import { durationText } from './week';

// /admin/tuning: what the weekend's tuners changed, under which margin, and whether last weekend's winners held. Every
// row the champion/challenger gate decided on the newest weekend (src/TuningGate.py), by tuner or by game (?by=);
// a game opens its drawer (?game=) with one line per tuned row and its last four weekends.
const BY = ['tuner', 'game'] as const;

export default function TuningPage() {
  const load = useLoad(getAdminTuning, []);
  const drawer = useDrawerParam();
  const [by, setBy] = useSwitchParam('by', BY, 'tuner');

  useEffect(() => {
    document.title = 'Tuning gates · Predictor';
  }, []);

  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title="Tuning gates" />
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  const t = d.tiles;
  const h = d.history;
  return (
    <div className="admin">
      <PageHead
        title="Tuning gates"
        sub={`Weekend of ${weekday(d.weekend)} · ${t.n} gate decisions in ${t.games} games`}
        meta={[`decided ${span(t.from, t.to)}`, 'Brussels time']}
      />
      <Tiles d={d} />
      <div className="segbar">
        <h2>{by === 'tuner' ? 'By tuner' : 'By game'}</h2>
        <span className="note">
          {h.error
            ? `Earlier weekends: not available (${h.error})`
            : h.weekends.length > 1
              ? `Earlier weekends from git: ${h.weekends.length - 1} (${weekday(h.weekends[h.weekends.length - 1])} – ${weekday(h.weekends[1])}), each from the last tuner commit of its weekend`
              : 'No earlier weekend in git yet'}
        </span>
        <div className="seg" role="group" aria-label="Group the gates">
          <button type="button" aria-pressed={by === 'tuner'} onClick={() => setBy('tuner')}>
            By tuner
          </button>
          <button type="button" aria-pressed={by === 'game'} onClick={() => setBy('game')}>
            By game
          </button>
        </div>
      </div>
      {by === 'tuner' ? (
        <div className="grid">
          {d.byTuner.map((x, i) => (
            <TunerCard key={x.tuner} t={x} wide={i === 0} onOpen={drawer.open} />
          ))}
        </div>
      ) : (
        <div className="stack">
          <ByGameCard d={d} onOpen={drawer.open} />
        </div>
      )}
      {d.unattributed.length ? (
        <div className="stack" style={{ marginTop: 'var(--gap)' }}>
          <UnattributedCard d={d} onOpen={drawer.open} />
        </div>
      ) : null}
      {drawer.game ? <TuningDrawer key={drawer.game} game={drawer.game} onClose={drawer.close} /> : null}
    </div>
  );
}

// --- tiles -----------------------------------------------------------------------------------------------------
function Tiles({ d }: { d: TuningView }) {
  const t = d.tiles;
  const held = t.heldFromLastWeekend;
  const zeroBy = d.byTuner.filter((x) => x.marginZero).map((x) => `${x.tuner} ${x.marginZero}`).join(' · ');
  const under = d.byTuner.filter((x) => x.replacedUnder1Se).map((x) => `${x.tuner} ${x.replacedUnder1Se}`).join(' · ');
  return (
    <div className="kpis">
      <Tile
        label="Replaced"
        value={String(t.replaced)}
        unit={`of ${t.n}`}
        foot={[
          `${t.games} games · weekend of ${weekday(d.weekend)}`,
          'the challenger beat the better of incumbent and default by the margin',
          t.notThisWeekend ? `${t.notThisWeekend} rows last decided on an earlier weekend, not counted` : '',
        ]}
      />
      <Tile label="Kept" value={String(t.kept)} unit={`of ${t.n}`} foot={['the challenger did not clear the bar; the served set stays']} />
      <Tile label="Reverted" value={String(t.reverted)} unit={`of ${t.n}`} foot={['the untuned default is served again']} />
      <Tile label="At margin 0" value={String(t.marginZero)} unit={`of ${t.n} gates`} foot={[zeroBy || 'none', 'any gain replaces']} />
      <Tile
        label="Gained under 1 SE"
        value={String(t.replacedUnder1Se)}
        unit={`of ${t.replaced} replaced`}
        foot={[
          under ? `over the incumbent: ${under}` : 'over the incumbent: none',
          `${t.replacedUnder1SeVsBetterReference} over the better of incumbent and default`,
        ]}
      />
      {held ? (
        <Tile
          label="Held from last weekend"
          value={String(held.kept)}
          unit={`of ${held.replaced} kept`}
          foot={[
            `the rows replaced ${weekday(held.weekend)}: ${held.replacedAgain} replaced again · ${held.reverted} reverted${held.notDecided ? ` · ${held.notDecided} not decided this weekend` : ''}`,
            'kept: this weekend’s challenger did not clear them',
          ]}
        />
      ) : (
        <Tile label="Held from last weekend" value={DASH} foot={[d.history.error ? `no earlier weekend: ${d.history.error}` : 'no earlier weekend in git yet']} />
      )}
    </div>
  );
}

// --- by tuner ----------------------------------------------------------------------------------------------------
const HELP = (
  <>
    Replaced: the challenger beat the better of the incumbent and the default by the margin. Kept: it did not, and the served set stays. Reverted: the
    challenger did not clear the default while the served set lost to the default or the challenger, so the untuned default is served again. Margin 0
    (statistical, RL ticket) means any gain replaces; the other tuners need one standard error of the challenger&apos;s score. Under 1 SE counts the
    replacements whose gain over the incumbent was less than one standard error of the challenger&apos;s score. Gate window: the newest backtest draws the
    gate scores on, in each game&apos;s own unit (a draw, a day, a trading day, a week), so a card lists it per unit. Keno&apos;s capped profit is measured above
    a fair ticket of the same size: a positive score is not money won (the drawer gives the raw profit per bet next to it). Trials this run: what the tuner
    ran this weekend; each study also holds every earlier weekend&apos;s trials. A game opens one line per tuned row.
  </>
);

function TunerCard({ t, wide, onOpen }: { t: TunerSummary; wide: boolean; onOpen: (g: string) => void }) {
  const margin = marginText(t);
  return (
    <Card
      className={wide ? 'span-2' : undefined}
      title={
        <>
          {t.tuner} {margin ? <Chip kind="flat">{margin}</Chip> : null}
        </>
      }
      help={wide ? HELP : undefined}
      meta={[
        t.run ? `run ${span(t.run.start, t.run.end)} · ${durationText(t.run.minutes)}${t.run.result !== 'ok' ? ` · ${t.run.result}` : ''}` : 'no run of this weekend in the scheduler history',
        'Brussels time',
      ]}
    >
      {t.n ? <TunerBody t={t} wide={wide} onOpen={onOpen} /> : <NoDecision t={t} />}
    </Card>
  );
}

// A tuner of the chain with no gate record of this weekend yet (mid-chain on Saturday, or its run wrote none): said in
// words, never as a zero bar and "all 0 kept"
function NoDecision({ t }: { t: TunerSummary }) {
  return (
    <p className="note">
      No decision this weekend yet:{' '}
      {t.run ? `its run ended ${t.run.result} without a gate record dated this weekend` : 'no finished run of this weekend in the scheduler history (it may still be queued or running)'}. The rows it
      gates keep an earlier weekend&apos;s decision, not counted here.
    </p>
  );
}

function TunerBody({ t, wide, onOpen }: { t: TunerSummary; wide: boolean; onOpen: (g: string) => void }) {
  return (
    <>
      <StackBar replaced={t.replaced} kept={t.kept} reverted={t.reverted} label={`${t.tuner}: ${t.replaced} replaced, ${t.kept} kept, ${t.reverted} reverted of ${t.n}`} />
      <p className="tmeta">
        <b>{t.replaced}</b> replaced · {t.kept} kept · {t.reverted} reverted · {t.n} gates in {t.games} games
        {t.replacedUnder1Se ? (
          <>
            {' '}
            · <b>{t.replacedUnder1Se}</b> of {t.replaced} replaced on a gain under 1 SE over the incumbent ({t.replacedUnder1SeVsBetterReference} over the better reference)
          </>
        ) : null}
        <br />
        {joinParts([
          kindsText(t),
          Object.keys(t.windowByUnit).length ? `gate window ${windowText(t.windowByUnit)}` : '',
          t.runTrials ? `trials this run ${range(t.runTrials)}` : '',
          t.studyTrials ? `studies of ${range(t.studyTrials)}` : '',
        ])}
      </p>
      {t.replaced || t.reverted ? (
        <div className={wide ? 'rowsets two' : 'rowsets'}>
          <RowSet title="Replaced" n={t.replaced} rows={t.replacedRows} onOpen={onOpen} />
          <RowSet title="Reverted to default" n={t.reverted} rows={t.revertedRows} onOpen={onOpen} />
        </div>
      ) : (
        <p className="note">No row replaced or reverted: all {t.kept} kept.</p>
      )}
    </>
  );
}

function RowSet({ title, n, rows, onOpen }: { title: string; n: number; rows: Record<string, string[]>; onOpen: (g: string) => void }) {
  const games = GAMES.filter((g) => rows[g]?.length).concat(Object.keys(rows).filter((g) => !GAMES.includes(g) && rows[g].length));
  return (
    <div className="rowset">
      <h3>
        {title} · {n}
      </h3>
      {games.length ? (
        <div className="gl-list">
          {games.map((g) => (
            <div className="gline" key={g}>
              <span className="gn">
                <GameButton game={g} onOpen={onOpen} />
              </span>
              <span className="chips">
                {rows[g].map((r) => (
                  <Chip key={r} kind="model">
                    {short(r)}
                  </Chip>
                ))}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="note">none</p>
      )}
    </div>
  );
}

// --- by game -----------------------------------------------------------------------------------------------------
function ByGameCard({ d, onOpen }: { d: TuningView; onOpen: (g: string) => void }) {
  const top = Math.max(1, ...d.byGame.map((g) => g.n));
  const groups = (['Lottery', 'Markets'] as const).map((name) => ({ name, games: d.byGame.filter((g) => g.group === name) })).filter((g) => g.games.length);
  return (
    <Card
      title="By game"
      help={HELP}
      flush
      meta={[`weekend of ${weekday(d.weekend)}`, `bars on one scale, up to ${top} gates`]}
      foot={
        <Legend
          items={[
            { mark: 'seg-a', label: 'replaced' },
            { mark: 'seg-c', label: 'kept' },
            { mark: 'seg-b', label: 'reverted' },
          ]}
        />
      }
    >
      <div className="byg">
        {groups.map((grp) => (
          <div key={grp.name} className="byg">
            <div className="bg grp">{grp.name}</div>
            {grp.games.map((g) => (
              <div className="bg" key={g.game}>
                <span className="g">
                  <GameButton game={g.game} onOpen={onOpen} />
                </span>
                <div>
                  {g.noFile ? (
                    <span className="mut">no bestParams file</span>
                  ) : (
                    <>
                      <StackBar replaced={g.replaced} kept={g.kept} reverted={g.reverted} of={top} label={`${g.game}: ${g.replaced} replaced, ${g.kept} kept, ${g.reverted} reverted of ${g.n}`} />
                      <div className="cnt">
                        <b>{g.replaced}</b> replaced · {g.kept} kept · {g.reverted} reverted · {g.marginZero} at margin 0
                        {g.notThisWeekend ? ` · ${g.notThisWeekend} last decided on an earlier weekend` : ''}
                      </div>
                    </>
                  )}
                </div>
                <div className="sets">
                  <div className="set">
                    <span>replaced</span>
                    <span className="chips">
                      {g.replacedRows.length ? g.replacedRows.map((r) => <Chip key={r} kind="model">{short(r)}</Chip>) : <span className="fnt">{DASH}</span>}
                    </span>
                  </div>
                  <div className="set">
                    <span>reverted</span>
                    <span className="chips">
                      {g.revertedRows.length ? g.revertedRows.map((r) => <Chip key={r} kind="model">{short(r)}</Chip>) : <span className="fnt">{DASH}</span>}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </Card>
  );
}

// --- unattributed ------------------------------------------------------------------------------------------------
function UnattributedCard({ d, onOpen }: { d: TuningView; onOpen: (g: string) => void }) {
  return (
    <Card
      title="Unattributed"
      help={
        <>
          A record written before 21 Oct 2026 does not name its tuner. It is given to a tuner only when its row&apos;s family and an independent fact agree: the
          tuner commit that first carries the record, or the scheduler run whose window holds its date. Otherwise it stays here, never guessed from times alone.
        </>
      }
      meta={[`${d.unattributed.length} records`]}
      flush
    >
      <table className="tbl">
        <thead>
          <tr>
            <th>Game</th>
            <th>Row</th>
            <th>Why</th>
          </tr>
        </thead>
        <tbody>
          {d.unattributed.map((u) => (
            <tr key={`${u.game}-${u.row}`}>
              <td className="sym">
                <GameButton game={u.game} onOpen={onOpen} />
              </td>
              <td>
                {short(u.row)}
                <span className="sub">{joinParts([u.family ? `family: ${u.family}` : 'no family', `weekend of ${weekday(u.weekend)}`])}</span>
              </td>
              <td>{u.why}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
