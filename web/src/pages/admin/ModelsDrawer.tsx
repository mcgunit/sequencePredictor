import { Fragment, useState } from 'react';
import { getAdminModelsGame, type Combination, type ModelsGameView, type MultiPick, type RankRow } from '../../api';
import { DASH, cap, count, day, fixed, perUnit, range, share, short, signedFixed, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useScope } from '../../hooks/useUrlState';
import Drawer, { Section } from '../../components/admin/Drawer';
import Chip from '../../components/admin/Chip';
import { Scroll } from '../../components/admin/Card';
import ScopeSwitch from '../../components/admin/ScopeSwitch';
import { Failed, Loading } from '../../components/admin/States';
import { boundWord, isMarketGame, oneControlText, recordText, sinceValue, topWhose, units, valueCell } from './bits';

// One game's ranking (GET /api/admin/models/:game?scope=all|top): every row, or its top five, with its
// value, record, since-freeze value and average hits or best day; the lotto multi-pick table; the best
// combination, whose value always travels with its control's p value.
export default function ModelsDrawer({ game, onClose }: { game: string; onClose: () => void }) {
  const [scope, setScope] = useScope();
  const load = useLoad(() => getAdminModelsGame(game, scope), [game, scope]);
  const d = load.kind === 'ok' ? load.data : null;
  return (
    <Drawer
      title={game}
      sub={d ? `${d.group} · ranked by ${d.measure.text} · ${d.window} · ranking ${stamp(d.generatedAt)} · Brussels time` : 'Ranking'}
      onClose={onClose}
    >
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {d ? (
        <>
          <RankingSection d={d} scope={scope} setScope={setScope} busy={load.busy} />
          {d.multiPick ? <MultiPickSection mp={d.multiPick} unit={d.unit} scope={d.scope} /> : null}
          {d.combination ? <CombinationSection c={d.combination} unit={d.unit} /> : null}
        </>
      ) : null}
    </Drawer>
  );
}

function RankingSection({ d, scope, setScope, busy }: { d: ModelsGameView; scope: 'all' | 'top'; setScope: (s: 'all' | 'top') => void; busy: boolean }) {
  const [sizes, setSizes] = useState<string | null>(null);
  const hits = d.metric === 'avg_hits';
  const band = d.band;
  const market = isMarketGame(d.game);
  const since = d.since && !d.since.none ? d.since : null;
  const modes = band ? Object.entries(band.modes) : [];
  return (
    <Section
      title="Ranking"
      aside={
        <>
          <Chip>
            {d.shown} of {d.of} rows{busy ? ' · updating' : ''}
          </Chip>
          <ScopeSwitch scope={scope} onChange={setScope} />
        </>
      }
    >
      <p className="line">
        {hits ? (
          <>
            Value: <b>average hits per {perUnit(d.unit)}</b> over every scored {perUnit(d.unit)}; ranked from {count(d.minDraws, d.unit)}, shorter records are
            marked &ldquo;short record&rdquo;.
          </>
        ) : (
          <>
            Value: <b>{d.measure.text}</b>
            {d.measure.bound ? `, ${d.measure.bound}` : ''}
            {d.measure.capEur !== null ? `, capped at ${d.measure.capEur} EUR` : ''}, measured against {d.measure.against ?? DASH}. The raw profit per bet stands
            under it; ranked from {count(d.minDraws, d.unit)}, shorter records are marked &ldquo;short record&rdquo;.
          </>
        )}
      </p>
      {band ? (
        <p className="line">
          Q0 band <b>{fixed(band.ceiling)}</b> hits per {perUnit(d.unit)} ({band.from} control, {stamp(band.at)}): the best of {band.rows ?? DASH} control rows on fair
          histories of {count(band.history, d.unit)}, not this ranking
          {band.greys ? '; rows within it are greyed.' : '; read on average hits: information only.'}{' '}
          {modes.map(([name, m]) => `${name} best ${fixed(m.best)} ± ${fixed(m.sd)} over ${m.seeds ?? DASH} seeds`).join(' · ')}
          {band.missing.length ? ` · ${oneControlText(band)}: the rule takes the higher of synthetic and shuffled, so this band can be lower than the rule's.` : ''}
        </p>
      ) : null}
      {since ? (
        <p className="line">
          Since {day(since.date)} ({since.label}): from {count(since.minDraws, d.unit)}
          {since.live ? `, ${since.live}` : ''}; best {short(since.bestModel)}.
        </p>
      ) : null}
      <p className="line">
        Top 5{d.scope === 'all' ? ' (tagged)' : ''}: {d.top.map(short).join(', ') || DASH}, {topWhose(d)}
        {isMarketGame(d.game) ? '.' : ': the first five ranked rows in the newest day\'s table, a ranked row missing from it passed over.'}
      </p>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th className="rk">#</th>
              <th>Row</th>
              <th className="num last-sm">Value</th>
              <th className="num hide-sm">{cap(units(d.unit))}</th>
              <th className="num hide-sm">Since {since ? day(since.date) : ''}</th>
              <th className="num hide-sm">{hits ? 'Best' : 'Avg hits'}</th>
            </tr>
          </thead>
          <tbody>
            {d.rows.map((r) => (
              <Fragment key={r.row}>
                <RankLine d={d} r={r} market={market} sizesOpen={sizes === r.row} onSizes={() => setSizes((x) => (x === r.row ? null : r.row))} />
                {sizes === r.row && r.bySize ? <SizesRow r={r} /> : null}
              </Fragment>
            ))}
          </tbody>
        </table>
      </Scroll>
    </Section>
  );
}

function RankLine({ d, r, market, sizesOpen, onSizes }: { d: ModelsGameView; r: RankRow; market: boolean; sizesOpen: boolean; onSizes: () => void }) {
  const hits = d.metric === 'avg_hits';
  const v = valueCell(d, r.value, r.profitPerBet, r.bets);
  const grey = Boolean(d.band?.greys && r.inBand);
  const sinceText = r.since === undefined ? '' : r.since === null ? 'no draw since' : hits ? fixed(r.since.value) : signedFixed(r.since.value);
  // a profit bound says what it is measured against under the number, as the Value column does
  const sinceBound = r.since && !hits ? boundWord(d) : '';
  const sinceSub = r.since ? `${count(r.since.draws, d.unit)}${market ? ' live' : ''}${r.since.young ? ' · few' : ''}${sinceBound ? ` · ${sinceBound}` : ''}` : '';
  const last = hits ? (r.bestDay === null ? DASH : `${r.bestDay} ${r.bestDay === 1 ? 'hit' : 'hits'}`) : fixed(r.avgHits);
  const lastSub = hits ? `on one ${perUnit(d.unit)}` : `best ${perUnit(d.unit)} ${r.bestDay ?? DASH}`;
  return (
    <tr className={grey ? 'grey' : undefined}>
      <td className="rk">{r.rank}</td>
      <td className="w-name">
        {short(r.row)}
        {r.top && d.scope === 'all' ? <span className="tag">top 5</span> : null}
        {r.young ? (
          <>
            {' '}
            <Chip kind="dim">short record</Chip>
          </>
        ) : null}
        {grey ? <span className="sub">within the band</span> : null}
        <span className="sub show-sm">
          {recordText(r.draws, d.unit, market ? r.live : null)}
          {sinceText ? ` · since ${r.since ? sinceValue(d, r.since.value) : sinceText}${r.since ? ` (${count(r.since.draws, d.unit)}${market ? ' live' : ''}${r.since.young ? ' · few' : ''})` : ''}` : ''} ·{' '}
          {hits ? `best ${last}` : `avg hits ${last}`}
        </span>
        {r.bySize ? (
          <span className="sub">
            <button type="button" className="linkbtn" aria-expanded={sizesOpen} onClick={onSizes}>
              {sizesOpen ? 'Hide ticket sizes' : 'Per ticket size'}
            </button>
          </span>
        ) : null}
      </td>
      <td className="num last-sm">
        <b>{v.main}</b>
        {hits ? null : <span className="sub">{`raw ${signedFixed(r.profitPerBet)} · ${r.bets ?? DASH} bets`}</span>}
      </td>
      <td className="num hide-sm">
        {count(r.draws, d.unit)}
        {market && r.live !== null && r.draws !== null ? <span className="sub">{r.live >= r.draws ? 'all live' : `${r.live} live`}</span> : null}
      </td>
      <td className="num hide-sm">
        {sinceText || DASH}
        {sinceSub ? <span className="sub">{sinceSub}</span> : null}
      </td>
      <td className="num hide-sm">
        {last}
        <span className="sub">{lastSub}</span>
      </td>
    </tr>
  );
}

// keno: the row's profit per bet at each playable ticket size, in place of the old one-line string
function SizesRow({ r }: { r: RankRow }) {
  return (
    <tr className="sizes">
      <td />
      <td colSpan={5}>
        <table className="tbl">
          <thead>
            <tr>
              <th>Ticket size</th>
              <th className="num">Profit per bet</th>
              <th className="num">Bets</th>
            </tr>
          </thead>
          <tbody>
            {(r.bySize ?? []).map((s) => (
              <tr key={s.size}>
                <td>{s.size} numbers</td>
                <td className="num">{signedFixed(s.profitPerBet)} EUR</td>
                <td className="num">{s.bets ?? DASH}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </td>
    </tr>
  );
}

function MultiPickSection({ mp, unit, scope }: { mp: MultiPick; unit: string; scope: 'all' | 'top' }) {
  const six = mp.chance[String(mp.draw)] ?? null;
  const nine = mp.chance[String(mp.draw + mp.extra)] ?? null;
  return (
    <Section title={`Multi-pick (${mp.draw} + ${mp.extra} extra numbers)`} aside={<Chip>{mp.draws ? `${range(mp.draws)} ${units(unit)} per row` : 'no draws yet'}</Chip>}>
      <p className="line">
        Chance of {mp.minHits ?? DASH} or more hits per {perUnit(unit)}: <b>{share(six)}</b> with {mp.draw} numbers, <b>{share(nine)}</b> with {mp.draw + mp.extra} (
        {mp.grids ?? DASH} grids, {mp.price ?? DASH} EUR). Each extra number hits {share(mp.perExtraChance)} by chance. {mp.rowsWithExtras} rows have extras
        {scope === 'top' ? `; ${mp.rows.length} of the top 5 shown` : ''}.
      </p>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Row</th>
              <th className="num">Avg hits {mp.draw} → {mp.draw + mp.extra}</th>
              <th className="num">
                ≥ {mp.minHits ?? DASH} hits, {mp.draw}
              </th>
              <th className="num last-sm">
                ≥ {mp.minHits ?? DASH} hits, {mp.draw + mp.extra}
              </th>
              <th className="hide-sm">Extras hit (7th · 8th · 9th)</th>
            </tr>
          </thead>
          <tbody>
            {mp.rows.map((r) => (
              <tr key={r.row}>
                <td className="w-name">
                  {short(r.row)}
                  {mp.draws && mp.draws.min !== mp.draws.max ? <span className="sub">{count(r.draws, unit)}</span> : null}
                  <span className="sub show-sm">extras hit {r.extraHits.map((x) => share(x)).join(' · ')}</span>
                </td>
                <td className="num">
                  {fixed(r.avgHits6, 2)} → {fixed(r.avgHits9, 2)}
                </td>
                <td className="num">{share(r.win6)}</td>
                <td className="num last-sm">{share(r.win9)}</td>
                <td className="hide-sm">{r.extraHits.map((x) => share(x)).join(' · ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Scroll>
    </Section>
  );
}

function CombinationSection({ c, unit }: { c: Combination; unit: string }) {
  const profit = c.metric === 'profit_per_draw';
  const per = `per ${perUnit(unit)}`;
  const value = (v: number | null) => (profit ? `${signedFixed(v, 2)} EUR ${per}` : `${fixed(v)} hits ${per}`);
  // the measure in the game's own unit, as one clause: "best line held per week, the average of the best member's hits"
  const what = c.measure.text.replace(/per draw/g, per).replace(' (', ', the ').replace(/\)$/, '');
  const control = c.control;
  const controlText = !control
    ? 'no control'
    : 'error' in control
      ? `control not available (${control.error})`
      : `p = ${fixed(control.pValue)} against ${control.shuffles ?? DASH} shuffles${control.consistentWithLuck ? ': consistent with luck' : ''}`;
  return (
    <Section title="Best combination" aside={<Chip>every established row · not filtered by the switch</Chip>}>
      {!c.best ? (
        <p className="line">{c.note ?? 'No combination could be scored.'}</p>
      ) : (
        <>
          {/* the value, its uncapped state and its p value on one line (plan decision 8) */}
          <p className="line">
            <b>{value(c.best.value)}</b>: {what}
            {profit && c.measure.capped === false ? ', uncapped' : ''} · over {count(c.best.draws, unit)} · {controlText}
          </p>
          <div className="chips" style={{ marginBottom: 10 }}>
            {c.best.members.map((m) => (
              <Chip key={m} kind="model">
                {short(m)}
              </Chip>
            ))}
            <span className="note">
              {c.best.how}
              {c.best.size !== null ? `, ${c.best.size} rows` : ''}
            </span>
          </div>
          {/* scored on the best set's own draws (those every member has), so it can differ from the row's ranking value */}
          {c.bestSingle ? (
            <p className="line">
              On the same {count(c.bestSingle.draws, unit)} the best set is scored on (those all its members have), the best single row is{' '}
              {short(c.bestSingle.row)} at {value(c.bestSingle.value)}
              {c.beatsBestSingle === true ? ' · the set beats it' : c.beatsBestSingle === false ? ' · the set does not beat it' : ''}; its ranking value above is over
              its own record.
            </p>
          ) : null}
          {profit ? (
            <p className="line">
              Profit per bet {signedFixed(c.best.profitPerBet, 2)} EUR · winning {units(unit)} {share(c.best.winDayRate)} · best {perUnit(unit)} {c.best.bestDay ?? DASH} hits
            </p>
          ) : null}
          <p className="line">
            {c.evaluated ?? DASH} sets searched over {c.candidateRows} established rows
            {control && !('error' in control)
              ? ` · on shuffled draws the best set averages ${value(control.bestMean)}, 95th percentile ${value(control.bestP95)}`
              : ''}
          </p>
          {c.ranking.length ? (
            <Scroll>
              <table className="tbl">
                <thead>
                  <tr>
                    <th className="rk">#</th>
                    <th>Members</th>
                    <th className="num">Value, {profit ? 'EUR' : 'hits'} {per}</th>
                    <th className="num hide-sm">{cap(units(unit))}</th>
                  </tr>
                </thead>
                <tbody>
                  {c.ranking.map((x, i) => (
                    <tr key={i}>
                      <td className="rk">{i + 1}</td>
                      <td>
                        {x.members.map(short).join(' + ')}
                        <span className="sub">{x.how}</span>
                        <span className="sub show-sm">{count(x.draws, unit)}</span>
                      </td>
                      <td className="num">
                        {profit ? signedFixed(x.value, 2) : fixed(x.value)}
                        {/* the old card's per-set figures: profit per bet and winning share, or average and best hits */}
                        {profit ? (
                          <span className="sub">
                            {signedFixed(x.profitPerBet, 2)} EUR per bet · winning {units(unit)} {share(x.winDayRate)}
                          </span>
                        ) : (
                          <span className="sub">
                            avg {fixed(x.avgHits)} per line · best {perUnit(unit)} {x.bestDay ?? DASH} hits
                          </span>
                        )}
                      </td>
                      <td className="num hide-sm">{x.draws ?? DASH}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Scroll>
          ) : null}
        </>
      )}
    </Section>
  );
}
