import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { day, money, plural, short, signClass, unsigned, weekday } from '../format';
import { useLoad } from '../hooks/useLoad';
import { useMe } from '../hooks/useMe';
import { getHome, type HomeGame, type MarketHeadline } from '../lottery';
import { Card } from '../components/admin/Card';
import { Failed, Loading, PageHead } from '../components/admin/States';
import { gameTitle, prettyDates } from '../components/draws/text';
import { TicketNums, figureText, lotteryHref, nextText, userText } from '../components/lottery/bits';

// / : the three sections (the owner's rule of 1 Oct: Lottery games, Crypto, Shares), each with its headline and a link,
// and the Council below. One answer (/api/home, about 6 KB): the Lottery's figures are the Lottery page's, the markets'
// are /crypto's and /shares' own (api.js marketView), so the pages cannot disagree. Every figure says whose it is and
// over which days.
export default function HomePage() {
  const load = useLoad(getHome, []);
  const me = useMe();
  const admin = Boolean(me?.user?.admin);

  useEffect(() => {
    document.title = 'Predictor';
  }, []);

  const head = (
    <PageHead
      title="Predictor"
      sub="the same models on lottery draws, crypto and shares, and how their picks did"
      meta={['Brussels time']}
      aside={
        admin ? (
          <Link className="btn" to="/admin">
            This week ›
          </Link>
        ) : undefined
      }
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
  const h = load.data;
  return (
    <div className="home">
      {head}
      <div className="hgrid">
        <LotterySection games={h.lottery} />
        <MarketSection m={h.crypto} />
        <MarketSection m={h.shares} />
      </div>
      <div className="card hcouncil">
        <div className="card-b">
          <p>
            <b>Council</b>: put one question to several language models at once; a head model weighs their answers. Your sessions are kept for your account.
          </p>
          <Link className="btn" to="/council">
            Ask the council ›
          </Link>
        </div>
      </div>
      <p className="note foot-note">Not advice. The models are not expected to beat the draw or the market; these pages measure whether they do.</p>
    </div>
  );
}

function LotterySection({ games }: { games: HomeGame[] }) {
  return (
    <Card
      title="Lottery"
      className="hsec"
      meta={[plural(games.length, 'game')]}
      help="Per game the first of today's top 5: the 5 models with the best record in the History ranking. Under it, the last draw's figure over the 5 models ranked best before that draw, which can be other models than today's."
      foot={
        <Link className="more" to="/lottery">
          Every game&apos;s top 5 ›
        </Link>
      }
    >
      <div className="hgames">
        {games.map((g) => (
          <GameLine key={g.game} g={g} />
        ))}
      </div>
    </Card>
  );
}

function GameLine({ g }: { g: HomeGame }) {
  const title = gameTitle(g.game);
  if (g.empty) {
    return (
      <div className="hgame">
        <div className="hgame-h">
          <b>{title}</b>
        </div>
        <small>{g.empty}</small>
      </div>
    );
  }
  const l = g.last;
  const last = !l
    ? null
    : l.leftOut
      ? `last draw ${day(l.date)}: left out, ${userText(l.leftOut)}`
      : l.figures
        ? `last draw ${day(l.date)}: ${figureText(g.shape, l.figures, `the ${l.figures.rows} ranked best before it`)}`
        : `last draw ${day(l.date)}: no top 5${l.none ? ` (${prettyDates(l.none.text)})` : ''}`;
  return (
    <div className="hgame">
      <div className="hgame-h">
        <Link to={lotteryHref(g.game)}>{title}</Link>
        <small>next draw {nextText(g.next)}</small>
      </div>
      {g.top ? (
        <>
          <TicketNums shape={g.shape} row={g.top} />
          <small>
            {short(g.top.name)}&apos;s {g.top.tickets.filter((t) => t.kind !== 'ranking').length > 1 ? 'tickets' : 'ticket'}: 1st of today&apos;s top {g.rows}
            {g.ranking ? ` by ${g.ranking.metricText}` : ''}
          </small>
        </>
      ) : (
        <small>No top 5 today{g.none ? `: ${prettyDates(g.none.text)}` : ''}.</small>
      )}
      {last ? <small>{last}</small> : null}
    </div>
  );
}

// The plan's game day by its own date, not "Today": the weekend's runs plan Monday's session, and before the morning run
// the day shown is the one that has just closed
const BRUSSELS_DAY = new Intl.DateTimeFormat('en-CA', { timeZone: 'Europe/Brussels', year: 'numeric', month: '2-digit', day: '2-digit' });
function dayLabel(market: string, date: string | null): string {
  if (!date) return 'Today';
  const today = BRUSSELS_DAY.format(new Date());
  if (date === today) return `Today (${weekday(date)})`;
  if (date > today) return `Next ${market === 'crypto' ? 'day' : 'session'} (${weekday(date)})`;
  return weekday(date);
}

function MarketSection({ m }: { m: MarketHeadline }) {
  const link = (
    <Link className="more" to={`/${m.market}`}>
      Open {m.title} ›
    </Link>
  );
  if (m.empty || !m.plan) {
    return (
      <Card title={m.title} className="hsec" foot={link}>
        <p className="note">{m.empty ? `${m.empty[0].toUpperCase()}${m.empty.slice(1)}.` : 'No record yet.'}</p>
      </Card>
    );
  }
  const p = m.plan;
  const t = m.today;
  const ahead = p.ahead;
  return (
    <Card title={m.title} className="hsec" meta={['paper money']} help={p.entryNote ?? undefined} foot={link}>
      <div className="hfig">
        <span className="k">Following the plan</span>
        <span className={`v ${signClass(p.total)}`}>
          {money(p.total)}
          <small>{m.unit}</small>
        </span>
        <small>
          {p.since ? `${day(p.since)} – ${day(p.through)}` : 'no live day yet'} · {plural(p.days, 'live day')} · {plural(p.positions, 'position')}, day and week together
        </small>
      </div>
      <p className="line">
        Buying everything over the same days: <b>{money(p.market)}</b> {m.unit}
        {ahead !== null ? (
          <>
            {' '}
            · the plan <b className={signClass(ahead)}>{unsigned(ahead)}</b> {m.unit} {ahead >= 0 ? 'better' : 'worse'}
          </>
        ) : null}
      </p>
      {t ? (
        <p className="line">
          {dayLabel(m.market, t.date)}: {t.state ?? 'no game day yet'} · <b>{t.toBuy}</b> to buy · <b>{t.held}</b> held
        </p>
      ) : null}
      {p.paperNote ? <p className="note">{p.paperNote}</p> : null}
    </Card>
  );
}
