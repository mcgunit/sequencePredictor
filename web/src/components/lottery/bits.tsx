import { Link, useNavigate } from 'react-router-dom';
import { ApiError } from '../../api';
import { DASH, dayYear, weekday } from '../../format';
import { USER_GAMES, type NextDraw } from '../../lottery';
import type { Best, DayFigures, Row, Shape } from '../../workbench';
import { Failed } from '../admin/States';
import { Nb } from '../draws/ResultStrip';
import { eur, gameTitle, prettyDates, specialsLabel } from '../draws/text';

// The users' Lottery routes inside the app
export const lotteryHref = (game?: string) => (game ? `/lottery#${game}` : '/lottery');
export const historyHref = (game: string, month?: string) => `/lottery/${game}${month ? `?month=${month}` : ''}`;
export const drawHref = (game: string, date: string) => `/lottery/${game}/${date}`;

// Another game's history, from a page head
export function UserGamePicker({ game }: { game: string }) {
  const navigate = useNavigate();
  return (
    <label className="pick">
      <span className="vh">Game</span>
      <select value={game} onChange={(e) => navigate(historyHref(e.target.value))}>
        {USER_GAMES.map((g) => (
          <option key={g} value={g}>
            {gameTitle(g)}
          </option>
        ))}
      </select>
    </label>
  );
}

// "Sat 24 Oct" ("today, Sat 24 Oct" on the day, as the old pages said it), or that the date has gone and the next run
// has not scored it yet (the records hold no time of day)
export function nextText(n: NextDraw | null): string {
  if (!n || !n.date) return 'not known yet';
  return `${n.today ? 'today, ' : ''}${weekday(n.date)}${n.past ? ' (that day has passed; the next run scores it)' : ''}`;
}

// One coming ticket as cells: the numbers or digits, then the stars, dream, Viking number or Joker+'s sign as their own
// group. Keno says how many tickets a row plays instead (a ranking line and one ticket per size).
export function TicketNums({ shape, row }: { shape: Shape; row: Row }) {
  const played = row.tickets.filter((t) => t.kind !== 'ranking');
  if (shape.family === 'keno') {
    const sizes = played.map((t) => t.mains.length).filter((n) => n > 0);
    const lo = Math.min(...sizes);
    const hi = Math.max(...sizes);
    return <span className="note">{sizes.length ? `${sizes.length} ${sizes.length === 1 ? 'ticket' : 'tickets'} of ${lo === hi ? lo : `${lo} to ${hi}`} numbers` : 'no ticket'}</span>;
  }
  const t = played[0] ?? row.tickets[0];
  if (!t) return <span className="note">no ticket</span>;
  const sign = shape.family === 'jokerplus' ? (t.signName ?? (t.specials && t.specials.length ? String(t.specials[0]) : null)) : null;
  const specials = shape.family === 'specials' && t.specials ? t.specials : [];
  return (
    <span className="nums sm">
      {t.mains.map((v, i) => (
        <Nb key={i} v={v} />
      ))}
      {sign || specials.length ? (
        <>
          <span className="gap" />
          <span className="vh">{specialsLabel(shape)}: </span>
          {sign ? <Nb v={sign} word /> : specials.map((v, i) => <Nb key={`s${i}`} v={v} />)}
        </>
      ) : null}
    </span>
  );
}

// The best ticket in words, for a line without the table's N (M) legend: "2 numbers right and the bonus ball",
// "1 number right and 2 stars", "3 from the left, 1 from the right and the sign"
export function bestWords(shape: Shape, b: Best): string {
  const count = (v: number | null, one: string, many: string) => (v ? `${v} ${v === 1 ? one : many}` : `no ${one}`);
  if (shape.family === 'jokerplus') return `${b.left ?? 0} from the left, ${b.right ?? 0} from the right${b.sign ? ' and the sign' : ''}`;
  const mains = `${count(b.mains, 'number', 'numbers')} right`;
  if (!b.specials) return mains;
  if (shape.family === 'lotto') return `${mains} and the bonus ball`;
  const label = specialsLabel(shape);
  if (label === 'Stars') return `${mains} and ${count(b.specials, 'star', 'stars')}`;
  return `${mains} and the ${label === 'Viking' ? 'Viking' : label.toLowerCase()} number`;
}

// A draw's figure over the rows named by whose, in one line: "best ticket: 1 number right and the bonus ball, among the
// 5 ...", "−23.00 EUR net over 30 tickets of the 5 ..."
export function figureText(shape: Shape, f: DayFigures, whose: string): string {
  if (shape.money) return `${eur(f.profit)} net over ${f.tickets} ${f.tickets === 1 ? 'ticket' : 'tickets'} of ${whose}`;
  if (!f.best) return `no scored row among ${whose}`;
  return `best ticket: ${bestWords(shape, f.best)}, among ${whose}`;
}

// A server line in the reader's words: its ISO days as "25 Aug 2026", its game ids ("jokerplus") as the game's title
export function userText(text: string): string {
  return prettyDates(text).replace(/\b(lotto|euromillions|eurodreams|vikinglotto|jokerplus|pick3|keno)\b/g, (g) => gameTitle(g));
}
const sentence = (text: string) => text.replace(/^./, (c) => c.toUpperCase()).replace(/([^.])$/, '$1.');

// A failed Lottery load in the reader's words: a draw left out (before the game's first saved ranking), a game no saved
// ranking ranks yet, a draw that is not there; anything else as the admin pages say it, with a retry
export function LotteryFailed({ error, game, retry }: { error: Error; game: string; retry?: () => void }) {
  const api = error instanceof ApiError ? error : null;
  const title = gameTitle(game);
  let head: string | null = null;
  let text: string | null = null;
  if (api?.code === 'before-first') {
    head = 'This draw is left out';
    text = `${sentence(userText(api.detail ?? 'it comes before the first saved ranking'))} The Lottery counts the draws from the first saved ranking on, never one read against a ranking rebuilt later.`;
  } else if (api?.code === 'not-ranked') {
    head = `No ranking of ${title} yet`;
    text = 'No saved ranking ranks this game yet, so its draws are left out until one does.';
  } else if (api?.status === 404) {
    head = 'Not found';
    text = api.detail ? sentence(userText(api.detail)) : 'There is nothing at this address.';
  }
  // anything else in the server's own words, without the request (the admin pages show it whole)
  if (!head) return <Failed error={api?.detail ? new Error(api.detail) : error} retry={retry} />;
  return (
    <div className="card">
      <div className="card-h">
        <h2>{head}</h2>
      </div>
      <div className="card-b">
        <p className="note">{text}</p>
      </div>
      <div className="card-f">
        <span className="wb-links">
          <Link to={historyHref(game)}>Every counted {title} draw ›</Link>
          <Link to={lotteryHref(game)}>{title}&apos;s next draw ›</Link>
        </span>
      </div>
    </div>
  );
}

// "1 Mar 2026 – 22 Oct 2026"
export const fromTo = (from: string | null, to: string | null) => (from && to ? (from === to ? dayYear(from) : `${dayYear(from)} – ${dayYear(to)}`) : DASH);
