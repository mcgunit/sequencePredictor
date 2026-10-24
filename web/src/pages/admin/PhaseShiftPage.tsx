import { useEffect, useRef, useState } from 'react';
import { getAdminPhaseShift, getAdminPhaseShiftGame, type LagRow, type PhaseShiftSummary, type Scope, type TrailStep } from '../../api';
import { DASH, cap, count, day, fixed, perUnit, range, share, short, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam, useScope } from '../../hooks/useUrlState';
import { Scroll } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Drawer, { Section } from '../../components/admin/Drawer';
import ScopeSwitch from '../../components/admin/ScopeSwitch';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import { isMarketGame, units } from './bits';

// Whose top five a game's are: a lottery game's are the rows its users see today; a market game's are the first
// five of its History game view, while the users' Crypto and Shares pages follow one row by best book (topWhose).
const topLabel = (game: string) => (isMarketGame(game) ? 'History top 5' : 'top 5');

// /admin/phase-shift: the old "Phase-shift check (tracked peaks)" card. Per game a collapsible card with each
// row's peak lag now, the lag that won most of the kept runs, and the peak trail; a row opens a drawer with
// every step's average hits and z (?game=lotto&row=...). "Tracked" is a rule, not a finding: no control
// measures what z or the tracked share reach by chance, so nothing here is coloured.
export default function PhaseShiftPage() {
  const load = useLoad(getAdminPhaseShift, []);
  const [scope, setScope] = useScope();
  const drawer = useDrawerParam('row');
  const [open, setOpen] = useState<Set<string>>(() => new Set(drawer.game ? [drawer.game] : []));

  useEffect(() => {
    document.title = 'Phase-shift · Predictor';
  }, []);
  // a link to one game (?game=lotto) opens its card (see the state above) and, once the cards are on the
  // page, scrolls to it; opening a row's drawer later does not move the page
  const linked = useRef(drawer.game);
  useEffect(() => {
    if (load.kind === 'ok' && linked.current) {
      document.getElementById(`ps-${linked.current}`)?.scrollIntoView({ block: 'start' });
      linked.current = null;
    }
  }, [load.kind]);

  const head = (
    <PageHead
      title="Phase-shift"
      sub="Tracked lag peaks per row · no control"
      meta={load.kind === 'ok' ? [`ranking ${stamp(load.data.generatedAt)}`, 'Brussels time'] : []}
      aside={<ScopeSwitch scope={scope} onChange={setScope} />}
    />
  );
  if (load.kind !== 'ok') {
    return (
      <>
        {head}
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  const toggle = (game: string) =>
    setOpen((s) => {
      const next = new Set(s);
      if (next.has(game)) next.delete(game);
      else next.add(game);
      return next;
    });
  return (
    <div className="admin">
      {head}
      <p className="note">
        Lags {d.meta.lags}. Tracked: {d.meta.tracked}. {cap(d.meta.control.replace(/^none: /, ''))}.
      </p>
      {/* the old card's reading of the peaks, folded as every help */}
      <details className="help" style={{ marginBottom: 'var(--gap)' }}>
        <summary>How to read the peaks</summary>
        <p>
          Each run scores every prediction against each lag and keeps only its best one. z near 1 means a flat lag profile, so the row&apos;s hits come from
          number frequency rather than timing. Several runs agreeing on the same lag is what a real phase shift would look like; a peak that moves every run is
          noise. The trail shows the peaks in run order: a repeated +30 ×5 means the peak is holding still, +26 → +28 → +30 that it is drifting. Pick3 is scored
          positionally (the digit in the right place), Joker+ as its left and right positional runs.
        </p>
      </details>
      <div className="stack">
        {(['Lottery', 'Markets'] as const).map((group) => {
          const games = d.games.filter((g) => g.group === group);
          if (!games.length) return null;
          return (
            <div key={group} className="stack">
              <div className="grp-label">{group}</div>
              {games.map((s) => (
                <GameCard key={s.game} s={s} open={open.has(s.game)} onToggle={() => toggle(s.game)} scope={scope} onRow={(row) => drawer.open(s.game, row)} />
              ))}
            </div>
          );
        })}
      </div>
      {drawer.game && drawer.extra ? <RowDrawer key={`${drawer.game}/${drawer.extra}`} game={drawer.game} row={drawer.extra} onClose={drawer.close} /> : null}
    </div>
  );
}

function GameCard({ s, open, onToggle, scope, onRow }: { s: PhaseShiftSummary; open: boolean; onToggle: () => void; scope: Scope; onRow: (row: string) => void }) {
  return (
    <div className="card" id={`ps-${s.game}`}>
      <h2 className="fold-t">
        <button type="button" className="fold-h" aria-expanded={open} aria-controls={`ps-${s.game}-b`} onClick={onToggle}>
          <span className="nm">{s.game}</span>
          <span className="meta">
            {s.tracked} of {s.rows} rows tracked (a rule, no control) · {s.topTracked} of the {topLabel(s.game)}
            {isMarketGame(s.game) ? ', not the users’ rows' : ''} · {range(s.runs)} runs per row · newest run {day(s.newestRun)}
            {s.scanned && s.scanned !== s.newestRun ? ` · scanned ${day(s.scanned)}` : ''}
          </span>
          {/* most frequent peak over every row, as the table's Most frequent column; of those, the tracked ones */}
          {s.commonLags.length ? (
            <span className="chips">
              {s.commonLags.map((l) => (
                <Chip key={l.lag}>
                  most often +{l.lag} · {l.rows} {l.rows === 1 ? 'row' : 'rows'}, {l.tracked} tracked
                </Chip>
              ))}
            </span>
          ) : null}
        </button>
      </h2>
      {open ? (
        <div id={`ps-${s.game}-b`}>
          <GameRows game={s.game} unit={s.unit} scope={scope} onRow={onRow} />
        </div>
      ) : null}
    </div>
  );
}

function Trail({ steps, earlier }: { steps: TrailStep[]; earlier: number }) {
  return (
    <span className="trail">
      {earlier ? <span className="ar">… →</span> : null}
      {steps.map((t, i) => (
        <span key={i} className="st">
          {i ? <span className="ar">→ </span> : null}
          <b>+{t.lag ?? DASH}</b>
          {t.runs > 1 ? ` ×${t.runs}` : ''}
        </span>
      ))}
    </span>
  );
}

function GameRows({ game, unit, scope, onRow }: { game: string; unit: string; scope: Scope; onRow: (row: string) => void }) {
  const load = useLoad(() => getAdminPhaseShiftGame(game, scope), [game, scope]);
  if (load.kind === 'loading') return <div className="card-b empty">Loading…</div>;
  if (load.kind === 'error') {
    return (
      <div className="card-b">
        <Failed error={load.error} retry={load.retry} bare />
      </div>
    );
  }
  const d = load.data;
  return (
    <>
      <div className="card-b flush">
        <Scroll>
          <table className="tbl">
            <thead>
              <tr>
                <th>Row</th>
                <th className="last-sm">Peak now</th>
                <th className="hide-sm">Most frequent</th>
                <th className="hide-sm">Trail, newest last</th>
              </tr>
            </thead>
            <tbody>
              {d.rows.map((r) => (
                <tr key={r.row}>
                  <td className="w-name">
                    <button type="button" className="gbtn row" onClick={() => onRow(r.row)} aria-haspopup="dialog">
                      {short(r.row)}
                    </button>
                    {r.top && d.scope === 'all' ? <span className="tag">{topLabel(game)}</span> : null}
                    <span className="sub show-sm">
                      most frequent +{r.mostFrequent.lag ?? DASH} in {r.mostFrequent.runs} of {r.runs} runs · {r.tracked ? 'tracked' : 'not tracked'}
                    </span>
                    <span className="sub show-sm">
                      <Trail steps={r.trail} earlier={r.earlierSteps} />
                    </span>
                  </td>
                  <td className="last-sm">
                    <b>+{r.peak.lag ?? DASH}</b>
                    <span className="sub">
                      z {fixed(r.peak.z, 2)} · {day(r.peak.run)}
                    </span>
                  </td>
                  <td className="hide-sm">
                    +{r.mostFrequent.lag ?? DASH} in {r.mostFrequent.runs} of {r.runs} runs
                    <span className="sub">
                      {share(r.mostFrequent.share)} · {r.tracked ? 'tracked' : 'not tracked'}
                    </span>
                  </td>
                  <td className="hide-sm">
                    <Trail steps={r.trail} earlier={r.earlierSteps} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Scroll>
      </div>
      <div className="card-f">
        <span className="fnote">
          {d.shown} of {d.of} rows
          {d.scope === 'top'
            ? isMarketGame(game)
              ? ' (History top 5)'
              : ' (the users’ top 5)'
            : isMarketGame(game)
              ? ' · tagged: the History top 5, not the users’ rows (their Crypto and Shares pages follow one row by best book)'
              : ' · tagged: the users’ top 5'}
          {d.scope === 'top' && d.topMissing.length ? ` · ${d.topMissing.length} of the top 5 ${d.topMissing.length === 1 ? 'has' : 'have'} no lag record: ${d.topMissing.map(short).join(', ')}` : ''} · in
          ranking order · lags in {units(unit)} · up to {d.meta.runsKept ?? DASH} runs kept
        </span>
      </div>
    </>
  );
}

// One row's peak record: this run's peak, the most frequent peak, and every step of the trail with its
// average hits and z (the newest run of the step) - the old card's hover facts.
function RowDrawer({ game, row, onClose }: { game: string; row: string; onClose: () => void }) {
  const load = useLoad(() => getAdminPhaseShiftGame(game, 'all'), [game]);
  const d = load.kind === 'ok' ? load.data : null;
  const r: LagRow | null = d ? (d.rows.find((x) => x.row === row) ?? null) : null;
  const per = d ? `hits per ${perUnit(d.unit)}` : 'hits';
  return (
    <Drawer title={short(row)} sub={`${game} · phase-shift · Brussels time`} onClose={onClose}>
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {d && !r ? <p className="line">{short(row)} has no phase-shift record in {game}.</p> : null}
      {d && r ? (
        <>
          <Section title="Peak now" aside={r.top ? <Chip>{topLabel(game)}</Chip> : null}>
            <p className="line">
              {r.peak.unchanged
                ? `Peak unchanged since the run of ${day(r.peak.run)} (the scan of ${day(r.peak.scanned)} added no draw, so it is no run)`
                : `Newest run ${day(r.peak.run)}`}
              : peak at <b>+{r.peak.lag ?? DASH}</b>, {fixed(r.peak.avgHits)} {per} over {count(r.peak.n, d.unit)}, z <b>{fixed(r.peak.z, 2)}</b> against
              the row&apos;s own lag profile (mean {fixed(r.peak.profileMean)}).
            </p>
            <p className="line">z: {d.meta.z}; no control says what z reaches by chance.</p>
          </Section>
          <Section title="Most frequent peak" aside={<Chip>{r.tracked ? 'tracked' : 'not tracked'}</Chip>}>
            <p className="line">
              <b>+{r.mostFrequent.lag ?? DASH}</b> in {r.mostFrequent.runs} of {r.runs} runs ({share(r.mostFrequent.share)}) · tracked means {d.meta.tracked}: a rule, not a
              finding.
            </p>
          </Section>
          <Section title="Peak trail">
            <p className="line">
              Each step: runs in a row that peaked at the same lag; its average hits and z are those of its newest run.
              {r.earlierSteps ? ` ${r.earlierSteps} earlier ${r.earlierSteps === 1 ? 'step' : 'steps'} not shown.` : ''}
            </p>
            <Scroll>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Lag</th>
                    <th className="num">Runs</th>
                    <th>Runs from → to</th>
                    <th className="num">Avg hits</th>
                    <th className="num">z</th>
                  </tr>
                </thead>
                <tbody>
                  {r.trail.map((t, i) => (
                    <tr key={i}>
                      <td>+{t.lag ?? DASH}</td>
                      <td className="num">{t.runs}</td>
                      <td>{t.from === t.to ? day(t.from) : `${day(t.from)} → ${day(t.to)}`}</td>
                      <td className="num">{fixed(t.avgHits)}</td>
                      <td className="num">{fixed(t.z, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Scroll>
          </Section>
        </>
      ) : null}
    </Drawer>
  );
}
