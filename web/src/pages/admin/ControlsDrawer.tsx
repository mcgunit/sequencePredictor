import { getAdminControlsGame, type ControlsGameView, type Interval } from '../../api';
import { DASH, count, dayYear, fixed, perUnit, short, signedFixed, stamp, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import Drawer, { Section } from '../../components/admin/Drawer';
import Chip from '../../components/admin/Chip';
import { Legend, Scroll } from '../../components/admin/Card';
import DotStrip, { Axis, niceScale, type Scale } from '../../components/admin/DotStrip';
import { Failed, Loading } from '../../components/admin/States';
import { isMarketGame, oneControlText, recordText, tableNote, units } from './bits';
import { hmmFeaturesText } from '../../components/draws/text';

// One game's four controls in full (GET /api/admin/controls/:game): Q0 with both fair-history controls and
// each row's own level, Q2 per seed, the feature-control variants as numbers, and for a market game the rows
// against the reference and uniform with the regimes.
export default function ControlsDrawer({ game, onClose }: { game: string; onClose: () => void }) {
  const load = useLoad(() => getAdminControlsGame(game), [game]);
  const d = load.kind === 'ok' ? load.data : null;
  return (
    <Drawer
      title={game}
      sub={
        d
          ? `${d.group} · all four controls · ${weekday(d.q2?.at ?? d.q0.band?.at ?? null)} · Brussels time`
          : isMarketGame(game)
            ? 'Markets · all four controls'
            : 'Lottery · all four controls'
      }
      onClose={onClose}
    >
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {d ? (
        <>
          <Q0Section d={d} />
          <Q2Section d={d} />
          <FcSection d={d} />
          <RowsSection d={d} />
        </>
      ) : null}
    </Drawer>
  );
}

const BASELINES = ['random', 'global_frequency', 'column_frequency'];

function Q0Section({ d }: { d: ControlsGameView }) {
  const q = d.q0;
  const b = q.band;
  const modes = Object.entries(q.modes);
  const market = isMarketGame(d.game);
  const rows = q.above.n ? q.above.rows : q.best ? [q.best] : [];
  const { scale, ticks } = niceScale([
    ...modes.flatMap(([, m]) => [m.random, m.best !== null && m.sd !== null ? m.best - 2 * m.sd : m.best, m.ceiling]),
    ...rows.map((r) => r.hits),
  ]);
  // every row the control runs, with its own level in each control, baselines last
  const names = [...new Set(modes.flatMap(([, m]) => Object.keys(m.perRow)))].sort((x, y) => Number(BASELINES.includes(x)) - Number(BASELINES.includes(y)) || x.localeCompare(y));
  const owned = q.above.rows.filter((r) => r.own !== null);
  const perUnitText = `hits per ${perUnit(d.unit)}`;
  return (
    <Section
      title="Q0 null band"
      aside={
        b ? (
          <>
            <Chip kind={q.above.n ? 'hi' : 'flat'}>{q.above.n ? `${q.above.n} rows above ${fixed(b.ceiling)}` : `none above ${fixed(b.ceiling)}`}</Chip>
            {b.missing.length ? <Chip kind="dim">one control only</Chip> : null}
          </>
        ) : (
          <Chip kind="dim">{d.errors.includes('q0') ? 'record unreadable' : 'no band yet'}</Chip>
        )
      }
    >
      {b ? (
        <p className="line">
          Band <b>{fixed(b.ceiling)}</b> {perUnitText} from the {b.from} control · {b.rows ?? DASH} rows · {b.seeds ?? DASH} seeds · histories of {count(b.history, d.unit)} ·
          records {modes.map(([name, m]) => `${name} ${stamp(m.at)}`).join(', ')}
          {b.missing.length
            ? ` · ${oneControlText(b)}: the rule takes the higher of synthetic and shuffled, so this band can be lower than the rule's`
            : ''}{' '}
          · ranking {stamp(d.rankingAt)}, {q.rankedRows} of {q.allRows} rows with ≥ {count(q.minDraws, d.unit)}
        </p>
      ) : (
        <p className="line">{d.errors.includes('q0') ? 'The Q0 record could not be read.' : 'No Q0 record for this game yet.'}</p>
      )}
      {q.ranking === 'profit' ? (
        <p className="line">
          Ranks on a profit bound; compared here on average hits, so no euro figure sits next to the band.
          {q.leader ? ` Profit-bound leader: ${short(q.leader.row)} (${count(q.leader.draws, d.unit)}).` : ''}
        </p>
      ) : null}
      {b && modes.length ? (
        <>
          <div className="lanes">
            <div className="lane axrow">
              <span />
              <Axis scale={scale} ticks={ticks} />
              <span />
            </div>
            {modes.map(([name, m]) => (
              <div className="lane" key={name}>
                <span className="lab">
                  {name} control
                  <small>
                    best ± 2 sd · {m.seeds ?? DASH} seeds · {m.rows ?? DASH} rows
                  </small>
                </span>
                <DotStrip
                  scale={scale}
                  grid={ticks.map((t) => t.at)}
                  spans={[{ from: m.best !== null && m.sd !== null ? m.best - 2 * m.sd : null, to: m.ceiling, kind: 'se' }]}
                  marks={[
                    { at: m.random, kind: 'diamond' },
                    { at: m.best, kind: 'hollow' },
                    { at: m.ceiling, kind: 'tickOwn' },
                    { at: b.ceiling, kind: 'tick' },
                  ]}
                  label={`${name}: best ${fixed(m.best)} plus or minus 2 sd, ceiling ${fixed(m.ceiling)}, random ${fixed(m.random)}`}
                />
                <span className="val">
                  {fixed(m.best)} → {fixed(m.ceiling)}
                </span>
              </div>
            ))}
            {rows.map((r) => (
              <div className="lane" key={r.row}>
                <span className="lab">
                  {short(r.row)}
                  <small>{recordText(r.draws, d.unit, market ? r.live : null)}</small>
                </span>
                <DotStrip
                  scale={scale}
                  grid={ticks.map((t) => t.at)}
                  marks={[
                    { at: b.ceiling, kind: 'tick' },
                    { at: r.hits, kind: 'dot' },
                  ]}
                  label={`${short(r.row)}: ${fixed(r.hits)} ${perUnitText} over ${count(r.draws, d.unit)}`}
                />
                <span className="val">
                  <b>{fixed(r.hits)}</b>
                </span>
              </div>
            ))}
          </div>
          <Legend
            items={[
              { mark: 'hol', label: 'control best' },
              { mark: 'se', label: '± 2 sd' },
              { mark: 'dia', label: 'random baseline' },
              { mark: 'dot', label: q.above.n ? 'ranked row above' : 'best ranked row' },
              { mark: 'tick', label: `band ${fixed(b.ceiling)}` },
              { mark: 'tick own', label: 'this control’s ceiling' },
            ]}
          />
        </>
      ) : null}
      {q.above.n ? (
        <p className="line">
          {owned.length
            ? `${owned.length} of the ${q.above.n} rows above ${owned.length === 1 ? 'has' : 'have'} an own level (${owned.map((r) => `${short(r.row)} ${fixed(r.own)}`).join(', ')}, ${b?.from} control).`
            : `None of the ${q.above.n} rows above has an own level: the control does not run them.`}{' '}
          Own levels of the rows it runs, mean {perUnitText} on the fair histories:
        </p>
      ) : names.length ? (
        <p className="line">Own levels of the rows the control runs, mean {perUnitText} on the fair histories:</p>
      ) : null}
      {names.length ? (
        <Scroll>
          <table className="tbl">
            <thead>
              <tr>
                <th>Row</th>
                {modes.map(([name]) => (
                  <th key={name} className="num">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {names.map((n) => (
                <tr key={n}>
                  <td>
                    {BASELINES.includes(n) ? n.replace(/_/g, ' ') : short(n)}
                    {BASELINES.includes(n) ? <span className="tag">baseline</span> : null}
                  </td>
                  {modes.map(([name, m]) => (
                    <td key={name} className="num">
                      {fixed(m.perRow[n])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Scroll>
      ) : null}
      {modes.length ? (
        <p className="line" style={{ marginTop: 8 }}>
          Winners per seed: {modes.map(([name, m]) => `${name} ${m.winners.map(short).join(', ') || DASH}`).join(' · ')}. Selection gain{' '}
          {modes.map(([, m]) => fixed(m.selectionGain)).join(' / ')}.
        </p>
      ) : null}
    </Section>
  );
}

function Q2Section({ d }: { d: ControlsGameView }) {
  const q = d.q2;
  if (!q) {
    return (
      <Section title="Q2 randomness">
        <p className="line">{d.errors.includes('q2') ? 'The Q2 record could not be read.' : 'No Q2 record for this game yet.'}</p>
      </Section>
    );
  }
  const lanes: { name: string; what: string; values: number[]; mean: number | null; sd: number | null; kind: 'small' | 'smallHollow' | 'null' }[] = [
    { name: 'real', what: 'real vs fair draws', values: q.perSeed.real, mean: q.realMean, sd: q.realSd, kind: 'small' },
    { name: 'shuffled', what: 'real draws reordered', values: q.perSeed.shuffled, mean: q.shuffledMean, sd: q.shuffledSd, kind: 'smallHollow' },
    { name: 'null', what: 'fair vs fair draws', values: q.perSeed.null, mean: q.nullMean, sd: q.nullSd, kind: 'null' },
  ];
  const { scale, ticks } = niceScale([...lanes.flatMap((l) => [...l.values, l.mean]), q.thresholdExact ?? q.threshold]);
  const threshold = q.thresholdExact ?? q.threshold;
  return (
    <Section
      title="Q2 randomness"
      aside={
        q.status === 'too few draws' ? (
          <Chip kind="dim">too few draws</Chip>
        ) : (
          <>
            <Chip kind={q.aboveBand ? 'hi' : 'flat'}>{q.aboveBand ? 'above the band' : 'within the band'}</Chip>
            <Chip kind={q.survivesShuffled ? 'hi' : 'flat'}>{q.survivesShuffled ? 'beats shuffled' : 'not above shuffled'}</Chip>
          </>
        )
      }
    >
      <p className="line">
        <b>{fixed(q.real)}</b> vs threshold <b>{fixed(q.threshold)}</b> (null mean + 2 sd) · windows of {q.window ?? DASH} · the newest{' '}
        {q.drawsUsed !== null ? count(q.drawsUsed, d.unit) : `${count(q.drawsCap, d.unit)} at most`}
        {q.drawsUsed !== null && q.drawsCap !== null && q.drawsUsed < q.drawsCap ? ` (of ${q.drawsCap} asked: the history is shorter)` : ''} ·{' '}
        {q.windowsTrain ?? DASH} train / {q.windowsTest ?? DASH} test windows{q.fewWindows ? ' (under 250: underpowered)' : ''} · {q.features ?? DASH} features ·{' '}
        {q.seeds ?? DASH} seeds · {q.quantum ? 'quantum suite included' : 'classical suite only'} · {stamp(q.at)}
      </p>
      <div className="lanes">
        <div className="lane axrow">
          <span />
          <Axis scale={scale} ticks={ticks} />
          <span />
        </div>
        {lanes.map((l) => (
          <div className="lane" key={l.name}>
            <span className="lab">
              {l.name}
              <small>{l.what}</small>
            </span>
            <DotStrip
              scale={scale}
              grid={ticks.map((t) => t.at)}
              spans={[{ from: l.mean !== null && l.sd !== null ? l.mean - l.sd : null, to: l.mean !== null && l.sd !== null ? l.mean + l.sd : null, kind: 'se' }]}
              marks={[...l.values.map((v) => ({ at: v, kind: l.kind })), { at: threshold, kind: 'tick' as const }]}
              label={`${l.name}: ${l.values.map((v) => fixed(v)).join(', ')}`}
            />
            <span className="val">
              {fixed(l.mean)} ± {fixed(l.sd)}
            </span>
          </div>
        ))}
      </div>
      <Legend
        items={[
          { mark: 'dot sm', label: 'one seed' },
          { mark: 'se', label: 'mean ± 1 sd' },
          { mark: 'tick', label: `threshold ${fixed(threshold)}` },
        ]}
      />
    </Section>
  );
}

const share = (v: number | null) => (v === null ? DASH : `${Math.round(v * 100)}%`);

// One week in how many at least one of n base-model columns clears the noise band (mean + 2 sd, one-sided: 2.28%
// each) by chance alone: 1 / (1 - 0.9772^n), at least 2 - the old card's weeksFor.
const chanceWeeks = (n: number) => Math.max(2, Math.round(1 / (1 - Math.pow(1 - 0.0228, n))));
// that figure for the column counts at hand (per variant, or per game): null when no count is known
export function chanceText(counts: (number | null | undefined)[]): string | null {
  const n = counts.filter((x): x is number => typeof x === 'number' && x > 0);
  if (!n.length) return null;
  const lo = Math.min(...n);
  const hi = Math.max(...n);
  return lo === hi
    ? `with ${hi} base-model columns per variant, one clears the band by chance about one week in ${chanceWeeks(hi)}`
    : `with ${lo} to ${hi} base-model columns per variant, one clears the band by chance about one week in ${chanceWeeks(hi)} to ${chanceWeeks(lo)}`;
}

function FcSection({ d }: { d: ControlsGameView }) {
  const f = d.fc;
  if (!f) {
    return (
      <Section title="Feature control">
        <p className="line">{d.errors.includes('fc') ? 'The feature-control record could not be read.' : 'No feature-control record for this game yet.'}</p>
      </Section>
    );
  }
  const counts = [`${f.fits} fit`, f.inconclusive ? `${f.inconclusive} inconcl.` : '', f.ignored ? `${f.ignored} ignored` : '', f.failed ? `${f.failed} failed` : ''].filter(Boolean).join(' · ');
  const chance = chanceText(f.variants.map((v) => v.columns));
  return (
    <Section title="Feature control" aside={<Chip>{counts}</Chip>}>
      <p className="line">
        {f.noiseColumns ?? DASH} noise columns · {f.repeats ?? DASH} repeats · {count(f.tableRows, d.unit)} in the table, {f.lockboxRows ?? DASH} lockbox{' '}
        {units(d.unit)} withheld{tableNote(f, d.unit) ? ` · ${tableNote(f, d.unit)}` : ''} · chance share {share(f.chanceShare)} · {stamp(f.at)}
      </p>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Variant</th>
              <th className="num">Noise share</th>
              <th className="num last-sm">Ratio</th>
              <th className="num hide-sm">AUC w/o → with</th>
              <th className="hide-sm">Above band</th>
            </tr>
          </thead>
          <tbody>
            {f.variants.map((v) => (
              <tr key={v.key}>
                <td>
                  <b>{v.label}</b>
                  <span className="sub">
                    <Chip kind={v.verdict === 'inconclusive' ? 'dim' : 'flat'}>{v.verdict}</Chip>
                  </span>
                  {v.error ? <span className="sub">error: {v.error}</span> : null}
                  {v.rule ? <span className="sub">rule: {v.rule}</span> : null}
                  <span className="sub show-sm">
                    AUC {fixed(v.aucWithout, 2)} → {fixed(v.aucWith, 2)} · cost {signedFixed(v.cost, 4)}
                  </span>
                  <span className="sub show-sm">above band: {v.above.length ? v.above.map(short).join(', ') : DASH}</span>
                </td>
                <td className="num">
                  {share(v.noiseShare)}
                  <span className="sub">chance {share(v.chanceShare)}</span>
                </td>
                <td className="num last-sm">
                  {fixed(v.ratio, 2)}
                  <span className="sub">
                    t {fixed(v.t, 2)} · band {fixed(v.band, 4)}
                  </span>
                </td>
                <td className="num hide-sm">
                  {fixed(v.aucWithout, 2)} → {fixed(v.aucWith, 2)}
                  <span className="sub">cost {signedFixed(v.cost, 4)}</span>
                </td>
                <td className="hide-sm">{v.above.length ? v.above.map(short).join(', ') : <span className="fnt">{DASH}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Scroll>
      {/* a column above the band is the figure read most; how often one gets there by chance stands with it */}
      {f.columnsAbove && chance ? (
        <p className="line" style={{ marginTop: 8 }}>
          Above band: {chance}. A column counts when it stays above week after week, not when it appears once.
        </p>
      ) : null}
      <details className="help">
        <summary>How to read this table</summary>
        <p>
          Noise columns: shuffled copies of real base-model scores, so they look exactly like a base model and mean nothing. Noise share: their share of the
          variant&apos;s training importance, against the chance share a model that cannot tell them apart would give them. Importance is the ranking power lost
          when a column is scrambled (the drop in |AUC − 0.5|), so a model whose probabilities came out inverted still shows what it leans on. Above band: base
          models whose held-out importance beats the noise columns&apos; mean + 2 sd. AUC: held-out, without the noise columns and with them; the cost is the
          difference.
        </p>
        {f.rule ? <p>Verdict rule: {f.rule}</p> : null}
      </details>
    </Section>
  );
}

const intervalText = (x: Interval | null) => (x ? `${signedFixed(x.mean)} (${signedFixed(x.lo)} to ${signedFixed(x.hi)}) · ${x.verdict}` : DASH);

// The forest plot's axis: the plot is there to show which intervals reach zero, so it is drawn over the rows
// within OFF_SCALE_WIDTHS median interval widths of zero. A row far beyond (Markov sits near -4 to -5.7 against
// GARCH while every other row is within -1) would squeeze every other interval onto the zero tick; it is named off
// the axis instead, its interval in the text line as always.
const OFF_SCALE_WIDTHS = 5;
function forestScale(intervals: Interval[]): { values: number[]; off: (x: Interval) => boolean } {
  const ends = (x: Interval) => [x.lo, x.hi, x.mean].filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  const widths = intervals
    .map((x) => (x.lo !== null && x.hi !== null ? x.hi - x.lo : null))
    .filter((w): w is number => w !== null && Number.isFinite(w))
    .sort((a, b) => a - b);
  const median = widths.length ? widths[Math.floor((widths.length - 1) / 2)] : 0;
  const limit = OFF_SCALE_WIDTHS * median;
  // how far the interval is from zero: 0 when it reaches zero, else its nearer end
  const distance = (x: Interval) => {
    if (x.lo !== null && x.hi !== null) return x.lo <= 0 && x.hi >= 0 ? 0 : Math.min(Math.abs(x.lo), Math.abs(x.hi));
    return x.mean !== null ? Math.abs(x.mean) : 0;
  };
  const far = (x: Interval) => limit > 0 && distance(x) > limit;
  const kept = intervals.filter((x) => !far(x));
  // every row far off (or no width to go by): the whole range, nothing named off the axis
  if (!kept.length) return { values: [0, ...intervals.flatMap(ends)], off: () => false };
  return { values: [0, ...kept.flatMap(ends)], off: far };
}

// a market-rows row's kind (MarketRows.py): a market model, a lottery model run on the market, or a baseline
const ROW_KIND: Record<string, string> = { market: 'market row', base: 'lottery row', baseline: 'baseline' };

function RowsSection({ d }: { d: ControlsGameView }) {
  if (!isMarketGame(d.game)) {
    return (
      <Section title="Market rows">
        <p className="line">{d.game} is not a market game: no market-rows report and no regimes.</p>
      </Section>
    );
  }
  const r = d.rows;
  const reference = short(r?.reference ?? 'GARCH Model');
  const scored = r ? r.table.filter((x) => x.vsReference) : [];
  const forest = forestScale(scored.map((x) => x.vsReference as Interval));
  const { scale, ticks } = niceScale(forest.values);
  const offScale = scored.filter((x) => x.vsReference && forest.off(x.vsReference));
  return (
    <>
      <Section title="Market rows">
        {!r ? (
          <p className="line">{d.errors.includes('rows') ? 'The market-rows report could not be read.' : 'No market-rows report for this game yet.'}</p>
        ) : (
          <>
            <p className="line">
              <b>{r.window ?? DASH}</b> {units(d.unit)} scored · {dayYear(r.firstDay)} → {dayYear(r.lastDay)} · {r.lockboxWithheld ?? DASH} lockbox {units(d.unit)} withheld
              {/* a game shorter than the request keeps its earliest draws as the models' history (MarketRows.py) */}
              {r.window !== null && r.lockboxWithheld !== null && r.requested !== null && r.window + r.lockboxWithheld < r.requested
                ? ` · ${r.window + r.lockboxWithheld} backtested of ${r.requested} asked: the game is shorter and keeps its earliest ${units(d.unit)} as history`
                : ` of ${r.requested ?? DASH} asked`}{' '}
              · reference {reference} · uniform log-score {signedFixed(r.uniform)} per {perUnit(d.unit)}
              {r.hmmFeatures ? ` · HMM features ${r.hmmFeatures}` : ''} · {stamp(r.at)}
            </p>
            <Scroll>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Row</th>
                    <th className="w-strip">
                      vs {reference}
                      <Axis scale={scale} ticks={ticks} />
                    </th>
                    <th className="last-sm">vs uniform</th>
                    <th className="hide-sm">Hit rates</th>
                  </tr>
                </thead>
                <tbody>
                  {r.table.map((x) => (
                    <tr key={x.row}>
                      <td className="w-name">
                        {short(x.row).replace(/_/g, ' ')}
                        {x.reference ? <span className="tag">reference</span> : null}
                        {x.kind ? <span className="tag">{ROW_KIND[x.kind] ?? x.kind}</span> : null}
                        <span className="sub">{x.logScore === null ? 'no log-score' : `log-score ${signedFixed(x.logScore)} ± ${fixed(x.se)}`}</span>
                        <span className="sub show-sm">
                          exact {share(x.exact)} · adjacent {share(x.adjacent)} · direction {share(x.direction)}
                        </span>
                      </td>
                      <td className="w-strip mid">
                        {x.vsReference && forest.off(x.vsReference) ? (
                          <>
                            <span className="mut">{(x.vsReference.mean ?? 0) < 0 ? '← off the axis' : 'off the axis →'}</span>
                            <span className="sub">{intervalText(x.vsReference)}</span>
                          </>
                        ) : x.vsReference ? (
                          <>
                            <Forest scale={scale} grid={ticks.map((t) => t.at)} x={x.vsReference} label={`${short(x.row)} vs ${reference}`} />
                            <span className="sub">{intervalText(x.vsReference)}</span>
                          </>
                        ) : (
                          <span className="mut">{x.reference ? 'the reference' : DASH}</span>
                        )}
                      </td>
                      <td className="last-sm">
                        {x.vsUniform ? signedFixed(x.vsUniform.mean) : DASH}
                        {x.vsUniform ? (
                          <span className="sub">
                            {signedFixed(x.vsUniform.lo)} to {signedFixed(x.vsUniform.hi)} · {x.vsUniform.verdict}
                          </span>
                        ) : null}
                      </td>
                      <td className="hide-sm">
                        exact {share(x.exact)}
                        <span className="sub">
                          adjacent {share(x.adjacent)} · direction {share(x.direction)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Scroll>
            <Legend
              items={[
                { mark: 'dot', label: 'mean' },
                { mark: 'ci', label: '95% interval' },
                { mark: 'tick', label: 'zero' },
              ]}
            />
            {r.failed.length ? (
              <p className="line">
                <b>
                  {r.failed.length} {r.failed.length === 1 ? 'row' : 'rows'} failed
                </b>
                :{' '}
                {r.failed
                  .map((f) => `${short(f.row).replace(/_/g, ' ')} ${f.reason ? `scored no ${perUnit(d.unit)} (${f.reason})` : f.days !== null ? `raised on ${f.days} ${f.days === 1 ? perUnit(d.unit) : units(d.unit)}` : '(no reason recorded)'}`)
                  .join('; ')}
                .
              </p>
            ) : r.errors ? (
              <p className="line">
                <b>{r.errors} failed</b> (the report names none).
              </p>
            ) : null}
            {offScale.length ? (
              <p className="line">
                Off the axis (more than {OFF_SCALE_WIDTHS} median interval widths from zero, so the others stay readable):{' '}
                {offScale.map((x) => `${short(x.row).replace(/_/g, ' ')} ${signedFixed(x.vsReference?.mean ?? null)}`).join(', ')} per {perUnit(d.unit)} against {reference}.
              </p>
            ) : null}
            <details className="help">
              <summary>How to read this table</summary>
              <p>
              Mean log-score difference per {perUnit(d.unit)} against {reference} and against uniform, with a 95% interval: a row beats {reference} only when the
              whole interval is above zero.{r.floor !== null ? ` A probability under ${r.floor} is scored as ${r.floor}.` : ''} Hit rates over the same{' '}
              {units(d.unit)}: the exact bin, the bin or a neighbour, and the direction. Profit and loss is left out (price units). Market rows are the market
              models, lottery rows the lottery models run on this market, baselines the random and frequency picks.
              </p>
              <p>
              The two ablation rows say what the Regime HMM&apos;s regimes are made of: if the full row is not better than ZeroMean, the regimes carry variance
              only; if it is not better than Single, there are no regimes worth the name. {hmmFeaturesText(r.hmmFeatures)}
              </p>
            </details>
          </>
        )}
      </Section>
      <Section title="Regimes">
        {d.regimes && d.regimes.length ? (
          <Scroll>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Row</th>
                  <th>Newest regime</th>
                  <th className="num">Regimes</th>
                  <th className="num">Readings kept</th>
                </tr>
              </thead>
              <tbody>
                {d.regimes.map((x) => (
                  <tr key={x.row}>
                    <td>{short(x.row)}</td>
                    <td>
                      {x.label || DASH}
                      <span className="sub">
                        p {fixed(x.probability)} · volatility rank {x.volatilityRank ?? DASH} · {x.date ?? DASH}
                      </span>
                    </td>
                    <td className="num">{x.regimes ?? DASH}</td>
                    <td className="num">{x.history ?? DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Scroll>
        ) : (
          <p className="line">{d.errors.includes('regimes') ? 'The regime record could not be read.' : 'No regime reading for this game yet.'}</p>
        )}
      </Section>
    </>
  );
}

function Forest({ scale, grid, x, label }: { scale: Scale; grid: number[]; x: Interval; label: string }) {
  return (
    <DotStrip
      scale={scale}
      grid={grid}
      spans={[{ from: x.lo, to: x.hi, kind: 'ci' }]}
      marks={[
        { at: 0, kind: 'tick' },
        { at: x.mean, kind: 'dot' },
      ]}
      label={`${label}: ${intervalText(x)}`}
    />
  );
}
