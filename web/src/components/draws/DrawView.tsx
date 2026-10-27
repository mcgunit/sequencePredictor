import { Link } from 'react-router-dom';
import { dayYear, isNum, pct, stamp, weekday } from '../../format';
import type { Draw, Row } from '../../workbench';
import { Card } from '../admin/Card';
import { FiguresLine, liveMarksText } from './History';
import MultiPickLegend from './MultiPickLegend';
import ResultStrip from './ResultStrip';
import ScopeLine, { ComingScopeLine } from './ScopeLine';
import TicketTable from './TicketTable';
import VotesChart, { slotNames } from './VotesChart';
import { CURRENCY_OF, MARKET_DRAW, MARKET_OF, MULTI_PICK_ORDER, howHitsCounted, prettyDates, type Mode } from './text';

// One draw (the old /database/:folder/:file page): the drawn row with the day's figures over the rows in scope, every
// row's tickets scored, lotto's multi-pick once, the vote, and - on the newest day only - the coming draw's tickets
// with today's ranking. An older day's next prediction is the next draw's table (the old page repeated it), so it
// links there instead.

// a Top table reads best in its ranking's order; All keeps the file's
const ordered = (rows: Row[], top: boolean) => (top ? [...rows].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99)) : rows);

// the coming draw's date, "today, Sat 24 Oct" on its own Brussels day (the old page's "for the draw of today, ...")
const BRUSSELS_DAY = new Intl.DateTimeFormat('en-CA', { timeZone: 'Europe/Brussels', year: 'numeric', month: '2-digit', day: '2-digit' });
const comingText = (date: string) => `${date === BRUSSELS_DAY.format(new Date()) ? 'today, ' : ''}${weekday(date)}`;

export function MarketNote({ game, slots, mode, marketHref }: { game: string; slots: string[]; mode: Mode; marketHref?: (market: string) => string }) {
  if (!MARKET_DRAW[game]) return null;
  const market = MARKET_OF[game];
  return (
    <p className="statusline mut">
      <span>
        One draw is {MARKET_DRAW[game]}{slots.length ? `; slots in order: ${slots.join(', ')}` : ''}. A blind pick is right 1 time in 10 per slot.
      </span>
      {mode === 'admin' && marketHref && market ? (
        <span className="wb-links">
          <Link to={marketHref(market)}>Market workbench ›</Link>
        </span>
      ) : null}
    </p>
  );
}

export default function DrawView({ d, mode, drawHref, marketHref }: { d: Draw; mode: Mode; drawHref: (date: string) => string; marketHref?: (market: string) => string }) {
  const s = d.scope;
  const top = s.scope === 'top';
  const currency = CURRENCY_OF[d.game];
  const slots = slotNames(d.shape);
  return (
    <div className="stack">
      {d.shape.family === 'market' ? <MarketNote game={d.game} slots={d.shape.slots} mode={mode} marketHref={marketHref} /> : null}
      <Card
        title="Result"
        meta={[weekday(d.date), mode === 'admin' && d.file ? `file ${d.file}` : '']}
        foot={
          d.scoredIn ? (
            <span className="fnote">
              Its next prediction was scored in the draw of <Link to={drawHref(d.scoredIn)}>{dayYear(d.scoredIn)}</Link>.
            </span>
          ) : undefined
        }
      >
        <div className="rstrip">
          <ResultStrip shape={d.shape} result={d.result} />
          {d.figures ? (
            <FiguresLine shape={d.shape} figures={d.figures} mode={mode} />
          ) : (
            <span className="fig mut">{top ? (s.none ? prettyDates(s.none.text) : 'no rows in scope') : 'no scored row'}</span>
          )}
        </div>
        {d.market ? (
          <p className="line">
            Close-to-close moves: {d.market.instruments.map((i) => `${i.symbol} ${pct(isNum(i.ret) ? i.ret * 100 : null, 2)} → bin ${i.bin ?? '—'}`).join(' · ')}
          </p>
        ) : null}
      </Card>
      <Card
        id="tickets"
        title={mode === 'admin' ? (top ? 'Top 5 rows scored' : 'Every row scored') : 'The models’ tickets'}
        help={howHitsCounted(d.shape, mode)}
        meta={[mode === 'user' ? `top ${s.rows} of ${s.of} models` : `${s.rows} of ${s.of} rows`, d.shape.money ? `${s.tickets} tickets` : '']}
        flush
      >
        <div className="card-b">
          <ScopeLine scope={s} mode={mode} />
          {mode === 'user' && d.afterClose ? <p className="scopeline mut">{d.afterCloseText ?? 'Tickets for this draw were made after its sales closed - shown, not counted in the totals.'}</p> : null}
          {mode === 'admin' && d.liveness?.marks ? (
            <p className="scopeline">
              Not live as stored: {liveMarksText(d.liveness.marks)} (lotteryLiveness.json; each row says why). The admin&apos;s live record scores a rewritten ticket as
              it stood at the close{d.liveness.afterClose ? '; the users\' history shows this draw greyed and leaves it out of its totals' : ''}.
            </p>
          ) : null}
          {d.market && mode === 'admin' && d.market.exactMean !== null ? (
            <p className="scopeline">
              Exact rate over every row this day: {(d.market.exactMean * 100).toFixed(1)}%, chance 10%{d.market.best ? ` · most exact: ${d.market.best.replace(/ Model$/, '')}` : ''}.
            </p>
          ) : null}
        </div>
        {d.rows.length ? (
          <TicketTable shape={d.shape} rows={ordered(d.rows, top)} result={d.result} market={d.market} mode={mode} currency={currency} ranked={s.ranking !== null} showTop={!top} />
        ) : (
          <div className="card-b">
            <div className="empty">
              {top ? `No ${mode === 'user' ? 'top' : 'Top'} 5 for this draw: ${s.none ? prettyDates(s.none.text) : 'no ranking before it'}.` : 'No predictions stored for this draw.'}
            </div>
          </div>
        )}
      </Card>
      {d.multiPick ? (
        <Card title="Multi-pick" help={MULTI_PICK_ORDER}>
          <MultiPickLegend data={d.multiPick} mode={mode} />
        </Card>
      ) : null}
      <Card
        title="Vote"
        help={
          d.shape.family === 'keno'
            ? 'Per number, how many of the shown rows carry it in their ranking line (20 numbers on most rows; not a played ticket).'
            : "Per number, how many of the shown rows' first tickets carry it; the positional games per slot, so a bar lights only for the value drawn in its own slot."
        }
        meta={[`${d.votes.rows} rows`]}
      >
        <VotesChart votes={d.votes} shape={d.shape} game={d.game} slotLabels={slots} mode={mode} />
      </Card>
      {d.coming ? (
        <Card
          id="next"
          title={`Next draw${d.coming.date ? ` · ${comingText(d.coming.date)}` : ''}`}
          meta={[
            mode === 'user' ? `top ${d.coming.scope.rows} of ${d.coming.scope.of} models` : `${d.coming.scope.rows} of ${d.coming.scope.of} rows`,
            d.coming.close ? `sales close ${d.coming.close.time}${d.coming.close.scheduled ? '' : ' (not a usual draw day)'}` : '',
            d.coming.madeAt ? `${mode === 'user' ? 'tickets made' : 'ticket'} ${stamp(d.coming.madeAt)}` : '',
          ]}
          flush
        >
          <div className="card-b">
            <ComingScopeLine scope={d.coming.scope} mode={mode} />
            {d.coming.close?.afterClose ? (
              <p className="scopeline">These tickets were made after sales closed ({d.coming.close.time}): for reading, not for playing.</p>
            ) : null}
          </div>
          {d.coming.rows.length ? (
            <TicketTable
              shape={d.shape}
              rows={ordered(d.coming.rows, d.coming.scope.scope === 'top')}
              result={null}
              market={null}
              mode={mode}
              currency={currency}
              ranked={d.coming.scope.ranking !== null}
              showTop={d.coming.scope.scope !== 'top'}
              rankWhose="in today's ranking"
            />
          ) : (
            <div className="card-b">
              <div className="empty">{mode === 'user' ? `No top 5 for the coming draw${d.coming.scope.none ? `: ${prettyDates(d.coming.scope.none.text)}` : ''}.` : 'No rows in this scope for the coming draw.'}</div>
            </div>
          )}
          <div className="card-b">
            <h3 className="subh">Vote for the coming draw</h3>
            <VotesChart votes={d.coming.votes} shape={d.shape} game={d.game} slotLabels={slots} mode={mode} />
          </div>
        </Card>
      ) : null}
    </div>
  );
}
