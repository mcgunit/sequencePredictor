import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { DASH, day, fixed, money, plural, short, signClass, unsigned, weekday } from '../format';
import { useLoad } from '../hooks/useLoad';
import { useMe } from '../hooks/useMe';
import { getHome, type HomeGame, type HomeSummary, type MarketHeadline } from '../lottery';
import { Card } from '../components/admin/Card';
import { Failed, Loading, PageHead } from '../components/admin/States';
import { gameTitle } from '../components/draws/text';
import { TicketNums, lotteryHref, nextText } from '../components/lottery/bits';
import RunBanner from '../components/RunBanner';

// / : the three sections (the owner's rule of 1 Oct: Lottery games, Crypto, Stocks), each with its headline and a link,
// and the Council below. One answer (/api/home, about 6 KB): the Lottery's figures are the Lottery page's, the markets'
// are /crypto's and /shares' own (api.js marketView), so the pages cannot disagree. Every figure says whose it is and
// over which days. While a pipeline job runs, a line says so and the page is read again every minute until it ends.
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
      sub="the same models on lottery draws, crypto and stocks, and how their picks did"
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
      <RunBanner running={h.running} refresh={load.retry} />
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
      help="Per game today's top 5: the tickets of the 5 models with the best record in the History ranking, best first. Each game's page has the models' names and how they did."
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

// One game: its name, the next draw and today's five tickets, each with its model's name (owner 2 Nov 2026: the users'
// Home carries little text), and beside them the game's summary
function GameLine({ g }: { g: HomeGame }) {
  const title = gameTitle(g.game);
  return (
    <div className="hgame">
      <div className="hgame-h">
        {g.empty ? <b>{title}</b> : <Link to={lotteryHref(g.game)}>{title}</Link>}
        {g.next ? <small>next draw {nextText(g.next)}</small> : null}
      </div>
      <div className="hgame-b">
        {g.tops.length ? (
          <div className="htops">
            {g.tops.map((r) => (
              <div key={r.name} className="htop">
                <TicketNums shape={g.shape} row={r} numbers />
                <small>{short(r.name)}</small>
              </div>
            ))}
          </div>
        ) : (
          <small>{g.empty ?? 'No top 5 today.'}</small>
        )}
        {g.summary ? <GameSummary s={g.summary} /> : null}
      </div>
    </div>
  );
}

// The best row by total profit on its live tickets (the jackpot games: by average hits, they have no money record) and
// the best combination as the admin's Best combination per game card has it; the details are in the tooltips
function GameSummary({ s }: { s: HomeSummary }) {
  const euro = (v: number) => `${money(v, 0)} EUR`;
  const b = s.best;
  const c = s.combination;
  return (
    <dl className="hsum">
      <dt>Best model</dt>
      <dd
        title={b ? `${s.unit === 'EUR' ? 'total profit' : 'average hits per draw'} on the tickets saved before each draw's sales close, rows with at least ${s.minDraws} such draws` : undefined}
      >
        {b ? (
          <>
            {short(b.name)} <b className={s.unit === 'EUR' ? signClass(b.value) : ''}>{s.unit === 'EUR' ? euro(b.value) : `${fixed(b.value, 2)} hits`}</b>
            {b.draws !== null ? <small> · {plural(b.draws, 'draw')}</small> : null}
          </>
        ) : (
          DASH
        )}
      </dd>
      <dt>Best combination</dt>
      <dd
        title={
          c
            ? `as on Best combination per game: ${c.unit === 'EUR' ? 'total profit of playing every member' : 'the best line held per draw'} over the draws they share${c.pValue !== null ? `; luck control p ${fixed(c.pValue, 2)}${c.pValue > 0.05 ? ', no better than the best set luck finds' : ''}` : ''}`
            : undefined
        }
      >
        {c ? (
          <>
            {c.members.map(short).join(' + ')} <b className={c.unit === 'EUR' ? signClass(c.value) : ''}>{c.unit === 'EUR' ? euro(c.value) : `${fixed(c.value, 2)} hits`}</b>
            {c.draws !== null ? <small> · {plural(c.draws, 'draw')}</small> : null}
          </>
        ) : (
          DASH
        )}
      </dd>
    </dl>
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
