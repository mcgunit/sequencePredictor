import { getAdminTuningGame, type GateRow, type GateSide, type TuningGameView } from '../../api';
import { DASH, count, day, fixed, isNum, short, signedFixed, weekday } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import Drawer, { Section } from '../../components/admin/Drawer';
import Chip from '../../components/admin/Chip';
import { Legend } from '../../components/admin/Card';
import DotStrip, { Axis, niceScale } from '../../components/admin/DotStrip';
import { Failed, Loading } from '../../components/admin/States';
import { isMarketGame, joinParts } from './bits';
import { kindLong, letterOf, marginText } from './gates';

// One game's tuning gates (GET /api/admin/tuning/:game): one line per tuned row - the decision, the default ◆, the
// incumbent ○ with ± 1 SE and the challenger ● on one axis per tuner and score kind, the bar the challenger had to
// clear, the score kind with its window and trials, and the row's last four weekends as letters. The scores are the
// tuners' own gate scores, never hits per draw; keno's capped profit is above a fair ticket, with the raw profit per bet
// next to it.
export default function TuningDrawer({ game, onClose }: { game: string; onClose: () => void }) {
  const load = useLoad(() => getAdminTuningGame(game), [game]);
  const d = load.kind === 'ok' ? load.data : null;
  return (
    <Drawer
      title={game}
      sub={
        d
          ? `${d.group} · tuning gates · weekend of ${weekday(d.weekend)} · ${d.rows.length} rows`
          : `${isMarketGame(game) ? 'Markets' : 'Lottery'} · tuning gates`
      }
      onClose={onClose}
    >
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {d ? <Body d={d} /> : null}
    </Drawer>
  );
}

function Body({ d }: { d: TuningGameView }) {
  // one axis per tuner and score kind: two kinds are never on one scale
  const groups: { key: string; tuner: string; kind: string | null; rows: GateRow[] }[] = [];
  d.rows.forEach((r) => {
    const key = `${r.tuner}|${r.kind ?? ''}`;
    const g = groups.find((x) => x.key === key);
    if (g) g.rows.push(r);
    else groups.push({ key, tuner: r.tuner, kind: r.kind, rows: [r] });
  });
  return (
    <>
      <section className="dsec">
        <p className="line">
          <b>{d.replaced}</b> replaced · {d.kept} kept · {d.reverted} reverted on the weekend of {weekday(d.weekend)}
          {d.notThisWeekend ? ` · ${d.notThisWeekend} rows last decided on an earlier weekend (not counted)` : ''}
        </p>
        <Legend
          items={[
            { mark: 'dia', label: 'default' },
            { mark: 'hol', label: 'incumbent' },
            { mark: 'se', label: '± 1 SE of the incumbent' },
            { mark: 'dot', label: 'challenger' },
            { mark: 'tick', label: 'bar: the better of incumbent and default + margin' },
          ]}
        />
        <p className="line">
          Each score is its tuner&apos;s gate score over the gate window, not hits per draw: a bound is the mean minus one standard error, so it sits below the
          row&apos;s own mean; the quantum tuner&apos;s holdout scores are means on its own holdout. &quot;Over the bar&quot; is the gain that decided, in standard errors
          of the challenger&apos;s score; &quot;over the incumbent&quot; is the one the under-1-SE count reads. The gate window counts backtest days, one per {d.unit === 'draws' ? 'draw' : d.unit === 'weeks' ? 'week' : d.unit === 'trading days' ? 'trading day' : 'day'} of {d.game}.
          {d.rows.some((r) => isNum(r.challenger.fairPerBet))
            ? ' A capped-profit score here is the capped profit per bet above a fair ticket of the same size (its own capped value per bet is given): a positive score is not money won, and the raw, uncapped profit per bet stands next to it.'
            : ''}
        </p>
        <p className="line">
          Last weekends, newest first: {d.weekends.length ? d.weekends.map((w) => weekday(w)).join(' · ') : DASH} · R replaced · K kept · V reverted · – not decided
          that weekend · ? not read
          {d.historyError ? ` · earlier weekends not available: ${d.historyError}` : ''}
        </p>
      </section>
      {groups.map((g) => (
        <Group key={g.key} tuner={g.tuner} kind={g.kind} rows={g.rows} d={d} />
      ))}
    </>
  );
}

function Group({ tuner, kind, rows, d }: { tuner: string; kind: string | null; rows: GateRow[]; d: TuningGameView }) {
  const values = rows.flatMap((r) => [
    r.default.score,
    r.challenger.score,
    r.bar,
    isNum(r.incumbent.score) && isNum(r.incumbent.se) ? r.incumbent.score - r.incumbent.se : r.incumbent.score,
    isNum(r.incumbent.score) && isNum(r.incumbent.se) ? r.incumbent.score + r.incumbent.se : r.incumbent.score,
  ]);
  const { scale, ticks } = niceScale(values);
  const margins = { n: rows.length, marginZero: rows.filter((r) => !(isNum(r.margin) && r.margin > 0)).length, marginOneSe: rows.filter((r) => isNum(r.marginSe) && Math.abs(r.marginSe - 1) < 0.01).length };
  // a keno capped-profit group: every score above a fair ticket of this capped value per bet
  const fair = rows.map((r) => r.challenger.fairPerBet).find(isNum) ?? null;
  return (
    <Section
      title={`${tuner} · ${kindLong(kind, fair)}`}
      aside={
        <>
          <Chip kind="flat">{marginText(margins)}</Chip>
          <Chip kind="flat">
            {rows.length} row{rows.length === 1 ? '' : 's'}
          </Chip>
        </>
      }
    >
      <div className="gates">
        <div className="gate axrow">
          <span />
          <Axis scale={scale} ticks={ticks} />
        </div>
        {rows.map((r) => (
          <GateLine key={r.row} r={r} scale={scale} grid={ticks.map((t) => t.at)} d={d} />
        ))}
      </div>
    </Section>
  );
}

const sideDays = (name: string, s: GateSide, windowDays: number | null) => (isNum(s.days) && isNum(windowDays) && s.days !== windowDays ? `${name} scored on ${s.days} of ${windowDays}` : '');
const se = (v: number | null) => (isNum(v) ? `${signedFixed(v, 2)} SE` : DASH);

function GateLine({ r, scale, grid, d }: { r: GateRow; scale: { min: number; max: number }; grid: number[]; d: TuningGameView }) {
  const inc = r.incumbent;
  const under = r.decision === 'replaced' && isNum(r.gainSe) && r.gainSe < 1;
  const short1 = short(r.row);
  return (
    <div className="gate">
      <span className="gh">
        {short1}
        <Chip kind={r.decision === 'replaced' ? 'hi' : 'flat'}>{r.decision}</Chip>
        {under ? <Chip kind="flat">under 1 SE over the incumbent</Chip> : null}
        {!r.thisWeekend ? <Chip kind="dim">decided {weekday(r.weekend)}</Chip> : null}
      </span>
      <DotStrip
        scale={scale}
        grid={grid}
        spans={[{ from: isNum(inc.score) && isNum(inc.se) ? inc.score - inc.se : null, to: isNum(inc.score) && isNum(inc.se) ? inc.score + inc.se : null, kind: 'se' }]}
        marks={[
          { at: r.bar, kind: 'tick' },
          { at: inc.score, kind: 'hollow' },
          { at: r.default.score, kind: 'diamond' },
          { at: r.challenger.score, kind: 'dot' },
        ]}
        label={`${short1}: default ${fixed(r.default.score)}, incumbent ${fixed(inc.score)} plus or minus ${fixed(inc.se)}, challenger ${fixed(r.challenger.score)}, bar ${fixed(r.bar)}, ${r.decision}`}
      />
      <span className="gf">
        {joinParts([
          kindLong(r.kind, r.challenger.fairPerBet),
          isNum(r.windowDays) ? `window ${count(r.windowDays, d.unit)}` : '',
          sideDays('default', r.default, r.windowDays),
          sideDays('incumbent', inc, r.windowDays),
          sideDays('challenger', r.challenger, r.windowDays),
          isNum(r.runTrials) ? `${r.runTrials} trials this run${isNum(r.studyTrials) ? ` (${r.studyTrials} in the study)` : ''}` : '',
          r.date ? `decided ${day(r.date)}` : '',
        ])}
        <br />◆ {fixed(r.default.score)} ○ {fixed(inc.score)} ± {fixed(inc.se)} ● {fixed(r.challenger.score)} ± {fixed(r.challenger.se)} · bar {fixed(r.bar)}{' '}
        {isNum(r.margin) && r.margin > 0 ? `(margin ${fixed(r.margin)})` : '(margin 0)'}: <b>{se(r.gainBarSe)} over the bar</b> · {se(r.gainSe)} over the incumbent
        {isNum(r.challenger.fairPerBet) ? (
          <>
            <br />
            raw profit per bet, uncapped, in EUR: ● {signedFixed(r.challenger.profitPerBet, 2)} ○ {signedFixed(r.incumbent.profitPerBet, 2)}
          </>
        ) : null}
        <br />
        <Letters history={r.history} weekends={d.weekends} />
        {r.why ? (
          <>
            <br />
            unattributed: {r.why}
          </>
        ) : null}
      </span>
    </div>
  );
}

function Letters({ history, weekends }: { history: (string | null)[]; weekends: string[] }) {
  if (!weekends.length) return <span className="mut">no weekend in git</span>;
  const words = weekends.map((w, i) => `${weekday(w)} ${history[i] ?? 'not decided'}`).join(', ');
  return (
    <span className="letters" role="img" aria-label={`last weekends: ${words}`}>
      {weekends.map((w, i) => (
        <span key={w} className="lt">
          <b>{letterOf(history[i] ?? null)}</b>
          <small>{day(w)}</small>
        </span>
      ))}
    </span>
  );
}
