import { Fragment, useState } from 'react';
import { getAdminModelsGame, type Combination, type GameLiveness, type LearningCheck, type ModelsGameView, type MultiPick, type RankRow } from '../../api';
import { DASH, cap, count, day, fixed, isNum, perUnit, plural, range, share, short, signedFixed, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useScope } from '../../hooks/useUrlState';
import Drawer, { Section } from '../../components/admin/Drawer';
import Chip from '../../components/admin/Chip';
import { Scroll } from '../../components/admin/Card';
import ScopeSwitch from '../../components/admin/ScopeSwitch';
import { Failed, Loading } from '../../components/admin/States';
import { boundWord, chanceLine, freshText, isMarketGame, recordText, sinceValue, topWhose, units, valueCell } from './bits';

// One game's ranking (GET /api/admin/models/:game?scope=all|top): every row, or its top five, with its
// value, record, since-freeze value and average hits or best day, and its Q0 row test on its own line; the lotto
// multi-pick table; the best combination, whose value always travels with its control's p value.
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
  const t = d.chance.test;
  // the since record counts live draws only: a market game's days, a lottery game's tickets saved before the close
  const liveSince = Boolean(d.since && d.since.live);
  const live = d.rankedOn === 'live';
  const since = d.since && !d.since.none ? d.since : null;
  const checked = d.rows.filter((r) => r.learning);
  const flagged = checked.filter((r) => r.learning?.verdict === 'counts-only').length;
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
            Value: <b>average hits per {perUnit(d.unit)}</b> over {live ? `the live record: the ${units(d.unit)} whose ticket was saved before the sales close` : `every scored ${perUnit(d.unit)}`};
            ranked from {count(d.minDraws, d.unit)}{live ? ' live' : ''}, shorter records are marked &ldquo;short record&rdquo;.
          </>
        ) : (
          <>
            Value: <b>{d.measure.text}</b>
            {d.measure.bound ? `, ${d.measure.bound}` : ''}
            {d.measure.capEur !== null ? `, capped at ${d.measure.capEur} EUR` : ''}, measured against {d.measure.against ?? DASH}
            {live ? `, over the live record: the ${units(d.unit)} whose ticket was saved before the sales close` : ''}. The raw profit per bet stands under it; ranked
            from {count(d.minDraws, d.unit)}{live ? ' live' : ''}, shorter records are marked &ldquo;short record&rdquo;.
          </>
        )}
        {live
          ? ` Ranked on the live record since 27 Oct 2026 (the owner's decision): a ticket rewritten after the draw is scored as it stood at the close; every stored ${perUnit(d.unit)}'s value stands under the live one (ranked from ${count(d.allMinDraws, d.unit)} there). The users' Top 5 follows this order.`
          : ''}
      </p>
      {d.liveness ? <LivenessLine l={d.liveness} unit={d.unit} /> : null}
      <p className="line">
        {t ? (
          <>
            Q0 chance (under each row): its hits per {perUnit(d.unit)} on its live {units(d.unit)} against{' '}
            {t.method === 'permutation' ? `its own tickets shuffled within blocks of ${t.block ?? 7} live ${units(d.unit)}` : 'what fair draws give its own tickets'}, p
            and - for the {t.family} rows with ≥ {count(t.minDraws, d.unit)} live - corrected for the rows compared (Holm).{' '}
            {t.family ? `${t.under} under 5%, ≈${fixed(t.expected, 2)} by chance; ${t.holds} hold; ${t.followed} followed, ${t.confirmed} confirmed on new ${units(d.unit)}.` : `No row has ${count(t.minDraws, d.unit)} live yet.`}{' '}
            Only a confirmed row has shown something; nothing here is greyed{d.chance.inTile ? '' : ', and for this payout game the test is on hits, information only'}.
          </>
        ) : (
          `Q0 chance: no row test (${d.chance.error ?? 'not in the report'}).`
        )}
      </p>
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
      {checked.length ? (
        <p className="line">
          Learning check of the deep-learning rows (from 25 Oct 2026): {flagged} of {plural(checked.length, 'checked row')} shown {flagged === 1 ? 'is' : 'are'}{' '}
          tagged &ldquo;no better than the frequencies&rdquo;: on the newest day a tagged row ran, its validation loss did not get below that of the training
          split&apos;s frequencies per position by more than an output that ignores its input gets on the same windows; the tag&apos;s title gives the losses
          and whether the row ignores its input. A label only: the ranking does not read it, and the users&apos; pages do not show it.
        </p>
      ) : null}
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
                <RankLine d={d} r={r} liveSince={liveSince} sizesOpen={sizes === r.row} onSizes={() => setSizes((x) => (x === r.row ? null : r.row))} />
                {sizes === r.row && r.bySize ? <SizesRow r={r} /> : null}
              </Fragment>
            ))}
          </tbody>
        </table>
      </Scroll>
    </Section>
  );
}

// A lottery game's liveness, as the daily report judged it (src/Liveness.py, 27 Oct 2026): how many draws were live as
// stored, how many more with a ticket rewritten after the draw (the live record scores the ticket that stood at the close,
// and the audit says what the rewrites changed), the rows back-built or late and the newest, and how it was judged. The
// counts are the draw pages' markers.
function LivenessLine({ l, unit }: { l: GameLiveness; unit: string }) {
  const how =
    l.checked === 'stamp+git'
      ? 'judged by the tickets\' stamps and git'
      : l.checked === 'stamp only'
        ? `git could not be read (${l.error ?? DASH}): judged by the stamps and the last check`
        : `not judged (${l.error ?? 'no evidence'}): every row counts as not live and the ranking stays on every stored ${perUnit(unit)}`;
  // not judged, or 'unchecked' (no git and no earlier check): no counts - an unchecked row is no evidence of a late ticket
  if (l.draws === null || l.checked === 'unchecked') {
    return <p className="line">Live record: {how}.</p>;
  }
  const rewritten = (l.allCounted ?? 0) - (l.allLive ?? 0);
  const a = l.rewriteAudit;
  // each count with what it is over in its own sentence (27 Oct 2026 review: the rewritten rows' count stood next to
  // the draws rewritten with no other row late, though most of those rows sit on draws with a late row as well)
  const parts = [
    `Live record (${how}): ${l.allCounted ?? DASH} of ${count(l.draws, unit)} with every row in it - ${l.allLive ?? DASH} live as stored${
      rewritten > 0 ? `, ${rewritten} with a ticket rewritten after the draw, scored as it stood at the close` : ''}.`,
    l.rowsRecovered
      ? `${plural(l.rowsRecovered, 'row')} rewritten after the draw in all, scored as they stood at the close: ${a.hitsAtClose ?? DASH} hits, where the rewrites score ${a.hitsStored ?? DASH}.`
      : '',
    l.rowsNotLive
      ? `${plural(l.rowsNotLive, 'row')} not live (back-built or late, or a second row of one name) on ${count((l.draws ?? 0) - (l.allCounted ?? 0), unit)}, newest ${
        l.newestNotLive ? day(l.newestNotLive) : DASH}; every later ${perUnit(unit)} has every row in the live record.`
      : 'No row back-built or late.',
    `The users' history leaves a ${perUnit(unit)} out of its totals when a ticket of its five was not out before the close, a rewritten one included.`,
    l.unscheduled.length ? `${count(l.unscheduled.length, unit)} on a weekday src/drawClose.json does not list: check the schedule.` : '',
  ];
  return <p className="line">{parts.filter(Boolean).join(' ')}</p>;
}

function RankLine({ d, r, liveSince, sizesOpen, onSizes }: { d: ModelsGameView; r: RankRow; liveSince: boolean; sizesOpen: boolean; onSizes: () => void }) {
  const hits = d.metric === 'avg_hits';
  const v = valueCell(d, r.value, r.profitPerBet, r.bets);
  // its Q0 test on its own line (its live mean, draws and null with its p), and a followed row's new draws
  const t = d.chance.test;
  const chance = t && r.chance && r.chance.p !== null ? `Q0: ${chanceLine(r.chance, d.unit, t.method, false)}` : '';
  const fresh = t && r.chance?.fresh ? freshText(r.chance.fresh, d.unit, t.method, t.verdictAlpha) : '';
  const sinceText = r.since === undefined ? '' : r.since === null ? 'no draw since' : hits ? fixed(r.since.value) : signedFixed(r.since.value);
  // a profit bound says what it is measured against under the number, as the Value column does
  const sinceBound = r.since && !hits ? boundWord(d) : '';
  const sinceSub = r.since ? `${count(r.since.draws, d.unit)}${liveSince ? ' live' : ''}${r.since.young ? ' · few' : ''}${sinceBound ? ` · ${sinceBound}` : ''}` : '';
  // ranked on the live record: every stored draw's value under the live one
  const allText = r.allValue !== null ? `every ${perUnit(d.unit)} ${hits ? fixed(r.allValue) : signedFixed(r.allValue)}` : '';
  const last = hits ? (r.bestDay === null ? DASH : `${r.bestDay} ${r.bestDay === 1 ? 'hit' : 'hits'}`) : fixed(r.avgHits);
  const lastSub = hits ? `on one ${perUnit(d.unit)}` : `best ${perUnit(d.unit)} ${r.bestDay ?? DASH}`;
  return (
    <tr>
      <td className="rk">{r.rank}</td>
      <td className="w-name">
        {short(r.row)}
        {r.top && d.scope === 'all' ? <span className="tag">top 5</span> : null}
        <LearningTag l={r.learning} />
        {r.young ? (
          <>
            {' '}
            <Chip kind="dim">short record</Chip>
          </>
        ) : null}
        {chance ? <span className="sub">{chance}</span> : null}
        {fresh ? <span className="sub">{fresh}</span> : null}
        <span className="sub show-sm">
          {recordText(r.draws, d.unit, r.live, r.recovered)}
          {sinceText ? ` · since ${r.since ? sinceValue(d, r.since.value) : sinceText}${r.since ? ` (${count(r.since.draws, d.unit)}${liveSince ? ' live' : ''}${r.since.young ? ' · few' : ''})` : ''}` : ''} ·{' '}
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
        {allText ? <span className="sub">{allText}</span> : null}
      </td>
      <td className="num hide-sm">
        {count(r.draws, d.unit)}
        {r.live !== null && r.draws !== null ? (
          <span className="sub">
            {r.live >= r.draws ? 'all live' : `${r.live} live`}
            {r.recovered ? `, ${r.recovered} as saved before the close` : ''}
          </span>
        ) : null}
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

// The "no better than the frequencies" tag of a deep-learning row (25 Oct 2026, src/LearningCheck.py): on a
// counts-only verdict only, its title giving the row's validation loss against the frequencies' loss, the bar it had
// to clear, whether it ignores its input and the date its data runs to. A row that learns gets no tag, nor does a row
// without a check (every other row). The Models page tags its best row and top five with it too.
export function LearningTag({ l }: { l: LearningCheck | null | undefined }) {
  if (!l || l.verdict !== 'counts-only') return null;
  return (
    <span className="tag" title={learningTitle(l)}>
      no better than the frequencies
    </span>
  );
}

function learningTitle(l: LearningCheck): string {
  const bar = isNum(l.baselineLoss) && isNum(l.margin) ? l.baselineLoss - l.margin : null;
  // above the frequencies' own loss: the row memorises its training windows rather than collapsing onto the counts
  const worse = isNum(l.valLoss) && isNum(l.baselineLoss) && l.valLoss > l.baselineLoss;
  const ticket = `the same ticket on ${share(l.sameTicket)} of ${l.windows ?? DASH} validation windows`;
  return [
    `Validation loss ${fixed(l.valLoss, 4)} against ${fixed(l.baselineLoss, 4)} for the training split's frequencies per position${worse ? ' (worse than them)' : ''}.`,
    `A row that learns gets below ${fixed(bar, 4)}: the margin ${fixed(l.margin, 4)} is what an output that ignores its input gets below the frequencies on the same windows.`,
    l.inputBlind ? `Input-blind: ${ticket}.` : `Not input-blind: ${ticket}${worse ? ', so its ticket follows its input but does worse on held-out draws (memorising)' : ''}.`,
    `Checked on the data up to ${day(l.date)}; a label only, the ranking does not read it.`,
  ].join(' ');
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

// a multi-pick rate with a text tag when it is above its chance level (more numbers win more by arithmetic alone)
function Rate({ v, level }: { v: number | null; level: number | null }) {
  return (
    <>
      {share(v)}
      {v !== null && level !== null && v > level ? <span className="tag">above chance</span> : null}
    </>
  );
}

function MultiPickSection({ mp, unit, scope }: { mp: MultiPick; unit: string; scope: 'all' | 'top' }) {
  const six = mp.chance[String(mp.draw)] ?? null;
  const nine = mp.chance[String(mp.draw + mp.extra)] ?? null;
  const extras = (xs: (number | null)[]) =>
    xs.map((x, i) => (
      <Fragment key={i}>
        {i ? ' · ' : null}
        <Rate v={x} level={mp.perExtraChance} />
      </Fragment>
    ));
  return (
    <Section title={`Multi-pick (${mp.draw} + ${mp.extra} extra numbers)`} aside={<Chip>{mp.draws ? `${range(mp.draws)} ${units(unit)} per row` : 'no draws yet'}</Chip>}>
      <p className="line">
        Chance of {mp.minHits ?? DASH} or more hits per {perUnit(unit)}: <b>{share(six)}</b> with {mp.draw} numbers, <b>{share(nine)}</b> with {mp.draw + mp.extra} (
        {mp.grids ?? DASH} grids, {mp.price ?? DASH} EUR). Each extra number hits {share(mp.perExtraChance)} by chance. {mp.rowsWithExtras} rows have extras
        {scope === 'top' ? `; ${mp.rows.length} of the top 5 shown` : ''}.
      </p>
      <p className="line">
        More numbers win more by arithmetic alone, so each rate is read against its chance level, over the {units(unit)} with extras; a rate above it is tagged
        &ldquo;above chance&rdquo;. One row above over a few {units(unit)} is noise; a row that stays above for many {units(unit)} is the finding, and the ground
        for rethinking its first {mp.draw} numbers (README roadmap item 10).
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
                  <span className="sub show-sm">extras hit {extras(r.extraHits)}</span>
                </td>
                <td className="num">
                  {fixed(r.avgHits6, 2)} → {fixed(r.avgHits9, 2)}
                </td>
                <td className="num">
                  <Rate v={r.win6} level={six} />
                </td>
                <td className="num last-sm">
                  <Rate v={r.win9} level={nine} />
                </td>
                <td className="hide-sm">{extras(r.extraHits)}</td>
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
        <p className="line">
          {cap(c.note ?? 'no combination could be scored')} ({c.candidateRows} established {c.candidateRows === 1 ? 'row' : 'rows'}).
        </p>
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
              {profit
                ? ''
                : ' No verdict: the best line held grows with every extra line, so a set is at least as good as any of its own rows by construction.'}
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
            {profit ? '' : ' · the table holds the best pairs, then the best triples, each ranked within its size'}
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
