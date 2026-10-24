import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { DASH, isNum, money, short, stamp, unsigned, weekday } from '../format';
import { useLoad } from '../hooks/useLoad';
import { getLottery, getUserDraw, type GameCard, type LastDraw } from '../lottery';
import type { Shape } from '../workbench';
import { Help, Scroll } from '../components/admin/Card';
import { Failed, Loading, PageHead } from '../components/admin/States';
import { FiguresLine } from '../components/draws/History';
import MultiPickLegend from '../components/draws/MultiPickLegend';
import ResultStrip from '../components/draws/ResultStrip';
import ScopeLine, { ComingScopeLine } from '../components/draws/ScopeLine';
import TicketTable from '../components/draws/TicketTable';
import { bestText, gameTitle, howHitsCounted, prettyDates } from '../components/draws/text';
import { bestWords, drawHref, historyHref, nextText, userText } from '../components/lottery/bits';

// /lottery: one folding card per game. The coming draw with today's five rows (the History ranking's top five) and their
// tickets, lotto's extras for a system play; the last draw with its result and how the five ranked BEFORE it did,
// together and row by row - two different fives, each named on its own line. Wide screens open every card; on a phone
// they start folded, and #game (the Home page's links) opens and scrolls to one.

const NARROW = '(max-width: 860px)';

export default function LotteryPage() {
  const load = useLoad(getLottery, []);
  const { hash } = useLocation();
  const navigate = useNavigate();
  const [wide] = useState(() => !window.matchMedia(NARROW).matches);
  const [opened, setOpened] = useState<Record<string, boolean>>({});
  const target = hash.replace(/^#/, '');

  useEffect(() => {
    document.title = 'Lottery · Predictor';
  }, []);
  // #game opens that card and brings it into view (also after the page has loaded)
  const loaded = load.kind === 'ok';
  useEffect(() => {
    if (!target || !loaded) return;
    setOpened((o) => ({ ...o, [target]: true }));
    requestAnimationFrame(() => document.getElementById(target)?.scrollIntoView({ block: 'start' }));
  }, [target, loaded]);

  const head = <PageHead title="Lottery" sub="per game, the 5 models with the best record: their tickets for the next draw" meta={['Brussels time']} />;
  if (load.kind !== 'ok') {
    return (
      <>
        {head}
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const games = load.data.games;
  const isOpen = (g: string) => opened[g] ?? (wide || g === target);
  return (
    <div className="lottery">
      {head}
      <nav className="jump" aria-label="Games on this page">
        {games.map((g) => (
          <a
            key={g.game}
            href={`#${g.game}`}
            onClick={(e) => {
              e.preventDefault();
              navigate({ hash: g.game }, { replace: true });
              setOpened((o) => ({ ...o, [g.game]: true }));
              document.getElementById(g.game)?.scrollIntoView({ block: 'start' });
            }}
          >
            {gameTitle(g.game)}
          </a>
        ))}
      </nav>
      <div className="stack">
        {games.map((g) => (
          <GameCardView key={g.game} c={g} open={isOpen(g.game)} onToggle={() => setOpened((o) => ({ ...o, [g.game]: !isOpen(g.game) }))} />
        ))}
      </div>
      <p className="note foot-note">Not advice: the models are not expected to beat the draw, and this page measures whether they do.</p>
    </div>
  );
}

// the header's summary line: the next draw, and the last draw's figure over its own five (ranked before it)
function summary(c: GameCard): string[] {
  const parts = [`next draw ${nextText(c.next)}`];
  const l = c.last;
  if (!l) return parts;
  if (l.leftOut) parts.push(`last draw ${weekday(l.date)}: left out`);
  else if (l.figures) parts.push(`last draw ${weekday(l.date)}, its top ${l.figures.rows}: ${lastFigure(c.shape, l)}`);
  return parts;
}
function lastFigure(shape: Shape, l: LastDraw): string {
  const f = l.figures;
  if (!f) return DASH;
  if (shape.money) return `${money(f.profit)} EUR net`;
  return f.best ? `best ticket ${bestWords(shape, f.best)}` : 'no scored row';
}

function GameCardView({ c, open, onToggle }: { c: GameCard; open: boolean; onToggle: () => void }) {
  const title = gameTitle(c.game);
  const id = c.game;
  if (c.empty) {
    return (
      <div id={id} className="card">
        <div className="card-h">
          <h2>{title}</h2>
        </div>
        <div className="card-b">
          <p className="note">{c.empty}</p>
        </div>
      </div>
    );
  }
  const s = c.coming.scope;
  return (
    <div id={id} className="card mcard lcard">
      <h2 className="fold-t">
        <button type="button" className="fold-h" aria-expanded={open} aria-controls={`${id}-b`} onClick={onToggle}>
          <span className="nm">{title}</span>
          <span className="meta">
            {summary(c).map((x, i) => (
              <span key={i}>
                {i ? ' · ' : ''}
                {x}
              </span>
            ))}
          </span>
        </button>
      </h2>
      {open ? (
        <div id={`${id}-b`} className="lbody">
          <div className="card-b">
            <div className="lsec-h">
              <h3 className="subh">Next draw · {nextText(c.next)}</h3>
              <Help title={`${title}'s tickets`}>{howHitsCounted(c.shape, 'user')}</Help>
            </div>
            <ComingScopeLine scope={s} mode="user" />
            {c.next?.madeAt ? <p className="scopeline">Tickets made {stamp(c.next.madeAt)}.</p> : null}
          </div>
          {c.coming.rows.length ? (
            <TicketTable shape={c.shape} rows={c.coming.rows} result={null} market={null} mode="user" ranked={s.ranking !== null} showTop={false} rankWhose="in today's ranking" />
          ) : (
            <div className="card-b">
              <div className="empty">No top 5 for the coming draw{s.none ? `: ${prettyDates(s.none.text)}` : ''}.</div>
            </div>
          )}
          {c.multiPick ? (
            <div className="card-b">
              <details className="fold">
                <summary>Multi-pick: the shaded numbers, sizes and prices</summary>
                <MultiPickLegend data={c.multiPick} mode="user" />
              </details>
            </div>
          ) : null}
          {c.last ? <LastSection c={c} l={c.last} /> : null}
          <div className="card-f">
            <span className="wb-links">
              <Link to={historyHref(c.game)}>Every counted {title} draw ›</Link>
              {c.last && !c.last.leftOut ? <Link to={drawHref(c.game, c.last.date)}>The last draw in full ›</Link> : null}
            </span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

// The last scored draw: its result, whose five it counts (the ranking before it), their figure together, then row by row
function LastSection({ c, l }: { c: GameCard; l: LastDraw }) {
  return (
    <>
      <div className="card-b lsec">
        <h3 className="subh">Last draw · {weekday(l.date)}</h3>
        <ResultStrip shape={c.shape} result={l.result} />
        {l.leftOut ? (
          <p className="scopeline">Left out: {userText(l.leftOut)}. The Lottery counts the draws from the first saved ranking on.</p>
        ) : l.none ? (
          <p className="scopeline">No top 5: {userText(l.none.text)}.</p>
        ) : (
          <>
            {l.scope ? <ScopeLine scope={l.scope} mode="user" /> : null}
            {l.figures ? <FiguresLine shape={c.shape} figures={l.figures} mode="user" /> : null}
          </>
        )}
      </div>
      {!l.leftOut && l.rows.length ? <LastTickets c={c} l={l} /> : null}
    </>
  );
}

// The last draw's tickets in the hits grid (the old History table: every number in its own coloured cell), loaded from
// the draw's own route - the same five as above, ranked before the draw; the per-row summary while it loads or when
// the draw cannot be read
function LastTickets({ c, l }: { c: GameCard; l: LastDraw }) {
  const load = useLoad(() => getUserDraw(c.game, l.date), [c.game, l.date]);
  if (load.kind === 'ok' && load.data.rows.length) {
    const d = load.data;
    return <TicketTable shape={d.shape} rows={d.rows} result={d.result} market={null} mode="user" ranked={d.scope.ranking !== null} showTop={false} rankWhose="in the ranking before this draw" />;
  }
  return <LastRows shape={c.shape} rows={l.rows} />;
}

// Each of those five with its own figure: the money games' profit, the others' best ticket as N (M) (the help and the
// table above say how it reads), lotto's best of its nine numbers
function LastRows({ shape, rows }: { shape: Shape; rows: LastDraw['rows'] }) {
  const lotto = shape.family === 'lotto';
  const hitsHead = shape.family === 'jokerplus' ? 'Best L/R (Z)' : 'Hits N (M)';
  const best = shape.family === 'lotto' || shape.family === 'specials' || shape.family === 'jokerplus';
  return (
    <Scroll>
      <table className="tbl stk">
        <thead>
          <tr>
            <th>Model</th>
            {shape.money ? <th className="num">Profit EUR</th> : null}
            {best ? <th>{hitsHead}</th> : null}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td className="lead nm">
                <span className="strong">{short(r.name)}</span>
              </td>
              {shape.money ? (
                <td className="num" data-l="Profit">
                  <span className={isNum(r.profit) && r.profit > 0 ? 'strong' : undefined}>{money(r.profit)}</span>
                  <span className="sub">
                    {r.tickets} {r.tickets === 1 ? 'ticket' : 'tickets'} · stake {unsigned(r.stake)}
                  </span>
                </td>
              ) : null}
              {best ? (
                <td data-l="Hits">
                  <span className="strong">{bestText(shape, r.best)}</span>
                  {lotto && r.multi !== null ? <span className="sub">of 9: {r.multi}</span> : null}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </Scroll>
  );
}
