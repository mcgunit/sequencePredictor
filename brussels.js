// brussels.js - Brussels wall time on the server (21 Oct 2026: the admin pages give every time in Brussels time,
// and a week, a weekend or "since Monday" is counted in Brussels days). The rules here never read the process's
// own zone, so a test box or a pm2 started with another TZ gets the same days; DST is the zone database's.
//
// Days are 'YYYY-MM-DD' strings, instants are epoch milliseconds.
'use strict';

const TZ = 'Europe/Brussels';
const WEEKDAYS = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };
const formatter = new Intl.DateTimeFormat('en-US', {
  timeZone: TZ, hourCycle: 'h23', weekday: 'short', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
});
const DAY = /^(\d{4})-(\d{2})-(\d{2})$/;

// The Brussels wall clock at an instant: {year, month, day, hour, minute, second, weekday (0 = Sunday)}.
function parts(ms) {
  const p = {};
  formatter.formatToParts(new Date(ms)).forEach((x) => { p[x.type] = x.value; });
  return { year: Number(p.year), month: Number(p.month), day: Number(p.day), hour: Number(p.hour) % 24, minute: Number(p.minute),
    second: Number(p.second), weekday: WEEKDAYS[p.weekday] };
}
const pad = (n) => String(n).padStart(2, '0');
const dayOf = (ms) => { const p = parts(ms); return `${p.year}-${pad(p.month)}-${pad(p.day)}`; };
const isDay = (text) => {
  const m = DAY.exec(String(text || ''));
  return Boolean(m) && new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]))).toISOString().slice(0, 10) === text;
};
// A calendar day n days on (calendar arithmetic, no zone involved).
const addDays = (day, n) => new Date(Date.parse(`${day}T00:00:00Z`) + n * 86400000).toISOString().slice(0, 10);
const weekdayOfDay = (day) => new Date(`${day}T00:00:00Z`).getUTCDay();
// The most recent <weekday> (0 = Sunday) on or before the Brussels day of an instant.
function lastWeekday(ms, weekday) {
  const day = dayOf(ms);
  return addDays(day, -((weekdayOfDay(day) - weekday + 7) % 7));
}
// The instant of a Brussels wall time: the day at hour:minute. On the hour the clocks skip (spring) it is the
// instant an hour later; on the one they repeat (autumn), the first of the two.
function instant(day, hour = 0, minute = 0) {
  const [y, m, d] = day.split('-').map(Number);
  const wall = Date.UTC(y, m - 1, d, hour, minute);
  let guess = wall;
  for (let i = 0; i < 3; i += 1) {
    const p = parts(guess);
    const shown = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second);
    if (shown === wall) break;
    guess += wall - shown;
  }
  const earlier = guess - 3600000;
  const q = parts(earlier);
  return Date.UTC(q.year, q.month - 1, q.day, q.hour, q.minute, q.second) === wall ? earlier : guess;
}

module.exports = { TZ, parts, dayOf, isDay, addDays, weekdayOfDay, lastWeekday, instant };
