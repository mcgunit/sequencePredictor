"""
The paper trader (README roadmap item 11, decided 15 November 2026; first
part built 17 November 2026).

Until now a market card's instruction ended with the game it came from: buy
at the open of the day or the week, sell at its close. The holding period was
the prediction interval. The paper trader separates the two. The day and week
games stay what they are - scored predictions - and become INPUTS; one trader
per instrument is in or out of a position for as long as its rules say.

What the trader reads, per instrument and game day D:

  the DAY call   the followed day row's call for D (MarketSettle.followed_plan:
                 the row the page follows, picked from the record before D),
                 as up / flat / down (MarketSettle.call_word)
  the WEEK call  the followed week row's call for the week D lies in, the same
                 three words; unknown while that week's ticket does not exist
                 (week_call_known: a shares week's ticket is made on Monday
                 evening, so Monday's decision has no week call)

The month call joins these when the month game exists (not built yet).

The rules (RULES, per market; trade() is the whole state machine):

  enter    out of a position, the week call is up and the day call has been
           up `persist` game days in a row -> buy. With shorts on (crypto) the
           mirror: week down and the day call down `persist` days -> sell short.
  leave    the week call turned against the position; or the day call was
           against it `turn` days in a row; or the position has been held
           `longest` game days; or the stop.
  stop     a move of `stop` against the entry price. Shares: read on the
           close, sold at the next open - never a buy and a sale of one
           position on one day. Crypto: inside the day, on the day's low (a
           long) or high (a short), booked at the stop price itself - which is
           how a buy and a sale can fall on one day, and only so.
  after a stop or a longest-hold sale the day call has to persist anew.

A flat or a missing call never opens a position and never closes one by
itself: flat breaks a run of up or down calls, a missing call leaves the runs
as they are.

When an order is booked (18 Nov 2026): a reader looks at the card once a day,
between 9 and 10 in the morning, and every instruction has to be one they can
carry out then. There is one decision per instrument and day, and it is
booked at EXEC_HOUR (09:00 Brussels time) of that day:

  shares   the Euronext open of the day, 09:00 (the card is up the evening
           before; the store has no intraday share prices, so a reader who
           orders at 9:40 gets another price than the record)
  crypto   the open of the hourly Binance bar that starts at 09:00 Brussels
           time (the card is up from the 00:05 UTC run). A day without that
           hourly bar - the history before the hourly bars were stored - is
           booked at the close before the day, 00:00 UTC, which no reader
           could get: Prices.followable says which.

The crypto stop is a standing stop order the reader places with the buy; it
works around the clock, so it is read on the hourly bars before and after the
day's decision.

The record (trader_export): the decisions are written to the tracked file
data/markets/<market>-trader.json by every daily run - what the trader said
for a day, and when. A decision can still change until EXEC_HOUR of its day
(the base game's export runs before the week game's ticket exists) and is
frozen after it. The trader starts without a position on the first day it
has a stored decision for; a position counts as live only when the decisions
that opened and closed it were stored before their days' EXEC_HOUR.

Money: MarketSettle.Costs, the stake and costs of every other paper book
(100 USDT at 0.1% a leg; 1,000 EUR in whole shares at Bolero's fee plus the
stock exchange tax). One cost pair per position.

Two comparisons, without which a result cannot be told from a rising market
or from holding less:

  hold     every instrument bought on the trader's first day and kept
  random   the trader's own positions - as many, as long, the same side - laid
           at random over the same days (RANDOM_RUNS times): the mean, and
           the share of those runs the trader beat

This module only reads. Nothing here is on a page yet: `python3 -m
src.MarketTrader --replay crypto --db <store> --root <checkout>` prints what
the trader would have done over the stored record.

Self-check: python3 -m src.MarketTrader (part of npm test; no network).
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

try:
    from src.MarketData import instruments, base_instrument_id, base_market
    from src.MarketGame import week_of, K_BINS
    from src import MarketSettle as settle
except ImportError:  # imported from within src/
    from MarketData import instruments, base_instrument_id, base_market
    from MarketGame import week_of, K_BINS
    import MarketSettle as settle


# The trader's rules per market (17 Nov 2026: first values, chosen by hand and
# NOT tuned on the record - the record is 77 crypto days and 42 sessions, too
# short to choose from; --grid prints how much the result leans on them).
#   persist  game days in a row the day call has to agree before an entry
#   turn     game days in a row the day call has to be against a position before it is left
#   stop     the move against the entry price that closes the position (None: no stop)
#   longest  game days a position is held at most
#   shorts   whether the trader may sell short
#   intraday whether the stop works inside the day (else on the close, sold at the next open)
RULES = {
    "crypto": {"persist": 2, "turn": 2, "stop": 0.10, "longest": 30, "shorts": True, "intraday": True},
    "shares": {"persist": 2, "turn": 3, "stop": 0.08, "longest": 60, "shorts": False, "intraday": False},
}
WEEK_GAME = {"crypto": "cryptoweek", "shares": "sharesweek"}
# One decision a day, booked at this hour in Brussels time: when a reader looks at the card (18 Nov 2026).
EXEC_HOUR = 9
BRUSSELS = ZoneInfo("Europe/Brussels")
TRADER_SCHEMA = 1
RANDOM_RUNS = 2000
RANDOM_SEED = 20261117

REASONS = {
    "week": "the week call turned",
    "day": "the day calls turned",
    "stop": "the stop",
    "longest": "the longest hold",
}


def rules_for(market, **over):
    """The market's rules, with any of them overridden."""
    out = dict(RULES[base_market(market)])
    out.update({k: v for k, v in over.items() if k in out})
    return out


def week_call_known(market, day):
    """
    Whether the week call for the week `day` lies in exists when the decision
    for `day` is made. Crypto: always - the Monday 00:05 UTC run makes the
    week's ticket and Monday's day ticket together. Shares: from Tuesday - the
    week is cut by the Monday 18:05 run, after Monday's open (docs/markets.md,
    "When the week ticket is made").
    """
    if base_market(market) == "crypto":
        return True
    return date.fromisoformat(day).weekday() >= 1


def exec_instant(day):
    """The moment a day's decision is booked, as a UTC datetime: EXEC_HOUR in Brussels on that date."""
    return datetime.combine(date.fromisoformat(day), time(EXEC_HOUR), tzinfo=BRUSSELS).astimezone(timezone.utc)


class Prices:
    """
    One instrument's prices as the trader books them. `daily` is {date:
    (open, high, low, close)}; `hourly`, crypto only, {UTC start of the hour
    (datetime): (open, high, low)}.

      exec_price(day)  the price a decision for `day` is booked at
      before(day)      (high, low) between the start of the day and the
                       decision, for a standing stop; None when unknown
      after(day)       (high, low) from the decision to the end of the day
      close(day)       the day's close
      followable(day)  whether exec_price is a price a reader could get
    """

    def __init__(self, market, daily, hourly=None):
        self.crypto = base_market(market) == "crypto"
        self.daily = daily
        self.days = sorted(daily)
        self.hourly = hourly or {}

    def _hours(self, day):
        """The stored hourly bars of the UTC day, split at the decision: (bar at the decision or None, bars before, bars from it on)."""
        d = date.fromisoformat(day)
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        at = exec_instant(day)
        before, after = [], []
        for h in range(24):
            stamp = start + timedelta(hours=h)
            bar = self.hourly.get(stamp)
            if bar is not None:
                (before if stamp < at else after).append(bar)
        return self.hourly.get(at), before, after

    def followable(self, day):
        if not self.crypto:
            return day in self.daily
        return self._hours(day)[0] is not None

    def exec_price(self, day):
        if not self.crypto:
            return self.daily[day][0] if day in self.daily else None
        bar = self._hours(day)[0]
        if bar is not None:
            return bar[0]
        j = bisect.bisect_left(self.days, day) - 1
        return self.daily[self.days[j]][3] if j >= 0 else None

    @staticmethod
    def _range(bars):
        return (max(b[1] for b in bars), min(b[2] for b in bars)) if bars else None

    def before(self, day):
        if not self.crypto:
            return None
        at, before, _after = self._hours(day)
        return self._range(before) if at is not None else None

    def after(self, day):
        if not self.crypto:
            return None
        at, _before, after = self._hours(day)
        if at is not None:
            return self._range(after)
        bar = self.daily.get(day)
        return (bar[1], bar[2]) if bar else None

    def close(self, day):
        bar = self.daily.get(day)
        return bar[3] if bar else None


def stop_price(entry, side, stop):
    """The price at which a position entered at `entry` is stopped: `stop` below it for a long, above it for a short."""
    return entry * (1.0 - stop) if side > 0 else entry * (1.0 + stop)


class Runs:
    """How many game days in a row the day call has been up, and down (flat breaks both, a missing call leaves them)."""

    def __init__(self):
        self.up = self.down = 0

    def read(self, word):
        if word == "up":
            self.up, self.down = self.up + 1, 0
        elif word == "down":
            self.up, self.down = 0, self.down + 1
        elif word == "flat":
            self.up = self.down = 0

    def reset(self):
        self.up = self.down = 0

    def with_side(self, side):
        return self.up if side > 0 else self.down

    def against(self, side):
        return self.down if side > 0 else self.up


def leave_reason(position, runs, week_word, rules, last_close=None):
    """
    Why an open position is left at the decision for a day, or None. The
    order is the order the card names a reason in: the stop (close-read stops
    only: `last_close` is the close before the day), the week call, the day
    calls, the longest hold.
    """
    side = position["side"]
    if rules["stop"] is not None and not rules["intraday"] and last_close is not None:
        level = stop_price(position["entry_price"], side, rules["stop"])
        if (side > 0 and last_close <= level) or (side < 0 and last_close >= level):
            return "stop"
    if (side > 0 and week_word == "down") or (side < 0 and week_word == "up"):
        return "week"
    if runs.against(side) >= rules["turn"]:
        return "day"
    if position["held"] >= rules["longest"]:
        return "longest"
    return None


def enter_side(runs, week_word, rules):
    """The side to enter on at the decision for a day: 1 (buy), -1 (sell short) or 0 (stay out)."""
    if week_word == "up" and runs.up >= rules["persist"]:
        return 1
    if rules["shorts"] and week_word == "down" and runs.down >= rules["persist"]:
        return -1
    return 0


def trade(days, signal, prices, rules, running=None, start=None):
    """
    One instrument's trader over `days` (its game days, oldest first).

      signal(day)  -> (day word, week word): 'up', 'flat', 'down' or None
      prices       -> a Prices (or anything with its five methods)
      running      -> a day after the last of `days` that has signals but no
                      prices yet: the day the card is for
      start        -> the first day a position may be opened on (the calls
                      before it are still read: a signal can have persisted)

    Returns {positions, open, now}:
      positions  closed positions, each {side, entry, entry_price, exit,
                 exit_price, reason, held}; `held` counts game days
      open       the position still held after the last day (the same keys
                 without the exit), or None
      now        for `running`: {action, reason, side}: 'buy', 'short',
                 'sell' (a long), 'cover' (a short), 'hold' or 'wait'; None
                 without `running`

    A day without an execution price changes nothing (nothing can be booked
    on it); the runs of day calls still read its call.
    """
    runs = Runs()
    positions, position = [], None

    def close(day, price, reason):
        nonlocal position
        positions.append({**position, "exit": day, "exit_price": price, "reason": reason})
        position = None
        if reason in ("stop", "longest"):
            runs.reset()       # the signal has to persist anew

    def standing_stop(day, span):
        """A standing stop order over a (high, low) span of `day`: closes the position at the stop price when the span reaches it."""
        if position is None or span is None or rules["stop"] is None or not rules["intraday"]:
            return
        level = stop_price(position["entry_price"], position["side"], rules["stop"])
        if (position["side"] > 0 and span[1] <= level) or (position["side"] < 0 and span[0] >= level):
            close(day, level, "stop")

    previous_close = None
    for day in days:
        day_word, week_word = signal(day)
        runs.read(day_word)
        price = prices.exec_price(day)
        left_side = 0
        if position is not None:
            before = position
            standing_stop(day, prices.before(day))
            if position is None:
                left_side = before["side"]
        if position is not None and price is not None:
            reason = leave_reason(position, runs, week_word, rules, previous_close)
            if reason:
                left_side = position["side"]
                close(day, price, reason)
        if position is None and price is not None and (start is None or day >= start):
            side = enter_side(runs, week_word, rules)
            # no new position of the side just left on the same day; and with whole-day decisions only
            # (shares) none at all: never a buy and a sale on one day
            if side and side != left_side and not (left_side and not rules["intraday"]):
                position = {"side": side, "entry": day, "entry_price": price, "held": 0}
        if position is not None:
            position["held"] += 1
            standing_stop(day, prices.after(day))
        if prices.close(day) is not None:
            previous_close = prices.close(day)

    now = None
    if running is not None:
        day_word, week_word = signal(running)
        runs.read(day_word)
        if position is not None:
            reason = leave_reason(position, runs, week_word, rules, previous_close)
            if reason:
                now = {"action": "sell" if position["side"] > 0 else "cover", "reason": reason, "side": position["side"]}
            else:
                now = {"action": "hold", "reason": None, "side": position["side"]}
        else:
            side = enter_side(runs, week_word, rules) if (start is None or running >= start) else 0
            now = {"action": "buy" if side > 0 else ("short" if side < 0 else "wait"), "reason": None, "side": side}
        now.update({"day_call": day_word, "week_call": week_word, "up_days": runs.up, "down_days": runs.down})
    return {"positions": positions, "open": position, "now": now}


def position_money(costs, stake, side, entry_price, exit_price):
    """(money, move in percent) of one position, or (None, None) when it cannot be priced or buys no whole share."""
    if not entry_price or not exit_price or entry_price <= 0 or exit_price <= 0:
        return None, None
    ret = math.log(exit_price / entry_price)
    money = costs.position(ret, stake, side, entry_price)
    return (None, None) if money is None else (money, settle.move_pct(ret) * (1 if side > 0 else -1))


def random_placements(spans, n_days, rng):
    """
    The trader's positions laid at random over `n_days`: `spans` are
    (side, length in days) in any order; returns [(side, first index, index
    of the exit decision)] without overlap, the order shuffled and the days
    outside a position split at random between the gaps. A position's exit
    index may equal n_days: still held at the end.
    """
    spans = list(spans)
    rng.shuffle(spans)
    free = n_days - sum(length for _side, length in spans)
    if free < 0:
        return []
    cuts = sorted(rng.randint(0, free) for _ in spans)
    out, at, used = [], 0, 0
    for (side, length), cut in zip(spans, cuts):
        at += cut - used
        used = cut
        out.append((side, at, at + length))
        at += length
    return out


def _bars(conn, instrument_id):
    rows = conn.execute("SELECT date, open, high, low, close FROM bars WHERE instrument_id = ? ORDER BY date", (instrument_id,)).fetchall()
    return {r[0]: (float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows}


def _hourly(conn, instrument_id):
    """{UTC start of the hour: (open, high, low)} of the instrument's stored 1h bars ({} when the store has no such table)."""
    try:
        rows = conn.execute("SELECT start, open, high, low FROM intraday_bars WHERE instrument_id = ? AND interval = '1h'", (instrument_id,)).fetchall()
    except Exception:
        return {}
    out = {}
    for r in rows:
        try:
            stamp = datetime.fromisoformat(str(r[0]))
        except ValueError:
            continue
        out[stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)] = (float(r[1]), float(r[2]), float(r[3]))
    return out


def inputs(conn, market, path, k=K_BINS):
    """
    What the trader reads, from the store and the day files under `path`:
    {days, running, symbols, signal(symbol) -> signal(day), live(day), bars,
    day_model, week_model}. `days` are the base market's settled game days;
    `running` the day the newest day ticket is for (None without one).
    """
    base = base_market(market)
    week_game = WEEK_GAME[base]
    day_followed = settle.followed_plan(conn, base, settle.ledger(conn, base, k), path=path, k=k) or {"points": [], "next_pick": None}
    week_followed = settle.followed_plan(conn, week_game, settle.ledger(conn, week_game, k), path=path, k=k) or {"points": [], "next_pick": None}
    day_model = {p[0]: p[1] for p in day_followed["points"]}
    day_live = {p[0]: bool(p[6]) for p in day_followed["points"]}
    week_model = {week_of(p[0]): p[1] for p in week_followed["points"]}
    week_date = {week_of(p[0]): p[0] for p in week_followed["points"]}
    week_live = {week_of(p[0]): bool(p[6]) for p in week_followed["points"]}

    def words_of(game):
        out = {}
        for r in settle._result_rows(conn, game):
            size = settle.size_of(r["meta"])
            if r["bin"] is None and size is None:
                continue
            out[(r["model"], r["symbol"], r["for_date"])] = settle.call_word(r["bin"], size, k)
        return out
    day_words, week_words = words_of(base), words_of(week_game)
    days = sorted(day_model)

    # the tickets not settled yet: the day the card is for, and the week in progress
    day_next = settle.next_day_predictions(conn, base, path)
    week_next = settle.next_day_predictions(conn, week_game, path)
    running = settle.next_session(base, day_next["made_on"]) if day_next else None
    if running is not None and days and running <= days[-1]:
        running = None

    def next_word(ticket, symbol, model):
        if not ticket or not model:
            return None
        q = next((q for q in (ticket.get("instruments") or {}).get(symbol, {}).get("predictions") or [] if q["model"] == model), None)
        return settle.call_word(q.get("bin"), q.get("size"), k) if q else None
    week_running = week_of(settle.next_session(base, week_next["made_on"])) if week_next else None

    def signal(symbol):
        def read(day):
            if day == running:
                day_word = next_word(day_next, symbol, day_followed.get("next_pick"))
            else:
                day_word = day_words.get((day_model.get(day), symbol, day))
            week_word = None
            if week_call_known(market, day):
                wk = week_of(day)
                if wk in week_model:
                    week_word = week_words.get((week_model[wk], symbol, week_date[wk]))
                elif wk == week_running:
                    week_word = next_word(week_next, symbol, week_followed.get("next_pick"))
            return day_word, week_word
        return read

    def live(day):
        """Whether the day's decision could have been followed: its day ticket was live, and its week's ticket when that week is settled."""
        wk = week_of(day)
        return bool(day_live.get(day)) and week_live.get(wk, True)

    members = instruments(conn, base, active_only=False)
    prices = {i["symbol"]: Prices(base, _bars(conn, base_instrument_id(conn, i)), _hourly(conn, base_instrument_id(conn, i)) if base == "crypto" else None)
              for i in members}
    return {"days": days, "running": running, "symbols": [i["symbol"] for i in members if i["active"]], "signal": signal, "live": live, "prices": prices,
            "day_model": day_model, "day_next_model": day_followed.get("next_pick"), "week_next_model": week_followed.get("next_pick")}


def _decided_in_time(decisions, symbol, day, actions):
    """Whether a decision of one of `actions` for the symbol and day was stored before the day's EXEC_HOUR."""
    got = (decisions or {}).get((symbol, day))
    if not got or got.get("action") not in actions or not got.get("made_at"):
        return False
    try:
        made = datetime.fromisoformat(str(got["made_at"]))
    except ValueError:
        return False
    made = made if made.tzinfo else made.replace(tzinfo=timezone.utc)
    return made < exec_instant(day)


def trader_record(conn, market, path, k=K_BINS, rules=None, runs=RANDOM_RUNS, seed=RANDOM_SEED, read=None, start=None, decisions=None):
    """
    The paper trader over the stored record of `market` ('crypto' or
    'shares'): per instrument its closed positions with their money, the open
    position valued at the newest close as if sold there, and what the rules
    say for the day the card is for; the totals; and the two comparisons
    (hold, random). `read` passes inputs() in (a grid reuses one reading).

    `start`: the trader holds nothing before this day (trader_export: the
    first day with a stored decision); the comparisons run from it too.
    `decisions`: {(symbol, day): {action, made_at}} as stored - a position is
    live when its entry, and its exit unless a standing stop made it, were
    decided before their days' EXEC_HOUR; without them nothing is live.

    totals["all"] counts every position; totals["long"] the long ones only -
    what a reader can follow, as no card gives a short order. For the shares
    the two are the same.
    """
    base = base_market(market)
    rules = rules or rules_for(base)
    data = read or inputs(conn, base, path, k)
    all_days = data["days"]
    days = [d for d in all_days if start is None or d >= start]
    costs, stake = settle.costs_for(base), settle.stake_for(base)
    out = {"market": base, "rules": rules, "from": start, "first_day": days[0] if days else None, "last_day": days[-1] if days else None, "days": len(days),
           "running": data["running"], "exec_hour": EXEC_HOUR, "symbols": {}, "stake": stake, "diverged": []}
    views = {"all": lambda side: True, "long": lambda side: side > 0}
    sums = {v: {"closed": 0.0, "open": 0.0, "positions": 0, "wins": 0, "days_in": 0, "live": 0.0, "live_positions": 0, "random": [0.0] * runs, "spans": False}
            for v in views}
    hold_total = 0.0
    index = {d: i for i, d in enumerate(days)}
    for n, symbol in enumerate(data["symbols"]):
        prices = data["prices"][symbol]
        result = trade(all_days, data["signal"](symbol), prices, rules, running=data["running"], start=start)
        closed = []
        for p in result["positions"]:
            money, move = position_money(costs, stake, p["side"], p["entry_price"], p["exit_price"])
            standing = p["reason"] == "stop" and rules["intraday"]
            closed.append({**p, "money": None if money is None else round(money, 4), "move": move, "reason_text": REASONS[p["reason"]],
                           "followable": prices.followable(p["entry"]) and (standing or prices.followable(p["exit"])),
                           "live": _decided_in_time(decisions, symbol, p["entry"], ("buy",) if p["side"] > 0 else ("short",))
                           and (standing or _decided_in_time(decisions, symbol, p["exit"], ("sell",) if p["side"] > 0 else ("cover",)))})
        still = None
        last_close = prices.close(days[-1]) if days else None
        if result["open"] is not None:
            p = result["open"]
            money, move = position_money(costs, stake, p["side"], p["entry_price"], last_close)
            still = {**p, "last_close": last_close, "money": None if money is None else round(money, 4), "move": move,
                     "stop_price": None if rules["stop"] is None else round(stop_price(p["entry_price"], p["side"], rules["stop"]), 6),
                     "followable": prices.followable(p["entry"]),
                     "live": _decided_in_time(decisions, symbol, p["entry"], ("buy",) if p["side"] > 0 else ("short",))}
        # a stored order the replay no longer makes (a rebuilt history, changed rules): said, never silently dropped
        made = {(q["entry"], "buy" if q["side"] > 0 else "short") for q in closed + ([still] if still else [])}
        made |= {(q["exit"], "sell" if q["side"] > 0 else "cover") for q in closed if not (q["reason"] == "stop" and rules["intraday"])}
        for (sym, day), got in sorted((decisions or {}).items()):
            if sym == symbol and day in index and got.get("action") in ("buy", "short", "sell", "cover") and (day, got["action"]) not in made:
                out["diverged"].append({"symbol": symbol, "day": day, "stored": got["action"]})
        # hold: bought on the trader's first day, kept to the newest close
        first = next((d for d in days if prices.exec_price(d) is not None), None)
        hold_money, hold_move = position_money(costs, stake, 1, prices.exec_price(first) if first else None, last_close)
        hold_total += hold_money or 0.0
        entry = {"positions": closed, "open": still, "now": result["now"],
                 "hold": None if hold_money is None else round(hold_money, 4), "hold_move": hold_move}
        for view, keep in views.items():
            mine = [q for q in closed if keep(q["side"])]
            mine_open = still if still is not None and keep(still["side"]) else None
            t = sums[view]
            closed_money = sum(q["money"] or 0.0 for q in mine)
            t["closed"] += closed_money
            t["open"] += (mine_open["money"] or 0.0) if mine_open else 0.0
            t["positions"] += len(mine) + (1 if mine_open else 0)
            t["wins"] += sum(1 for q in mine if (q["money"] or 0.0) > 0)
            days_in = sum(q["held"] for q in mine) + (mine_open["held"] if mine_open else 0)
            t["days_in"] += days_in
            t["live"] += sum(q["money"] or 0.0 for q in mine if q["live"]) + ((mine_open["money"] or 0.0) if mine_open and mine_open["live"] else 0.0)
            t["live_positions"] += sum(1 for q in mine if q["live"]) + (1 if mine_open and mine_open["live"] else 0)
            # random: these positions (side and length) at random places, each booked at the day prices
            spans = [(q["side"], max(1, index[q["exit"]] - index[q["entry"]])) for q in mine]
            if mine_open is not None:
                spans.append((mine_open["side"], len(days) - index[mine_open["entry"]]))
            rng = random.Random(seed * 1000 + n * 10 + (0 if view == "all" else 1))
            mean = 0.0
            for r in range(runs if spans else 0):
                total = 0.0
                for side, first_i, end_i in random_placements(spans, len(days), rng):
                    z = prices.exec_price(days[end_i]) if end_i < len(days) else last_close
                    total += position_money(costs, stake, side, prices.exec_price(days[first_i]), z)[0] or 0.0
                t["random"][r] += total
                mean += total / runs
            t["spans"] = t["spans"] or bool(spans)
            if view == "all":
                entry.update({"closed_money": round(closed_money, 4), "open_money": None if still is None else still["money"], "days_in": days_in,
                              "random_mean": round(mean, 4)})
        out["symbols"][symbol] = entry
    out["totals"] = {"hold": round(hold_total, 4), "days_possible": len(days) * len(data["symbols"])}
    for view, t in sums.items():
        total = t["closed"] + t["open"]
        out["totals"][view] = {
            "total": round(total, 4), "closed": round(t["closed"], 4), "open": round(t["open"], 4), "positions": t["positions"], "wins": t["wins"],
            "days_in": t["days_in"], "live_total": round(t["live"], 4), "live_positions": t["live_positions"],
            "random_mean": round(sum(t["random"]) / runs, 4) if t["spans"] and runs else None,
            "random_beaten": round(sum(1 for v in t["random"] if total > v) / runs, 4) if t["spans"] and runs else None,
            "random_runs": runs if t["spans"] else 0,
        }
    return out


# ---------------------------------------------------------------------------
# The record: the decisions as they were made, in a tracked file
# ---------------------------------------------------------------------------

def trader_file(path, market):
    return os.path.join(path, "data", "markets", f"{base_market(market)}-trader.json")


def load_trader(path, market):
    """The stored trader file of the market, or {} when there is none or it cannot be read."""
    try:
        with open(trader_file(path, market)) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def trader_export(conn, path, market, now=None, log=print):
    """
    The daily step (MarketSettle.settle_and_export, for a market and for its
    week game): store what the trader says for the day the card is for, and
    write data/markets/<market>-trader.json - the rules, the positions since
    the trader's first day, the totals and comparisons, and every stored
    decision.

    The trader's first day is the first day a decision was stored for: it
    starts without a position there. The running day's decision is stored
    when it is first made and replaced while it can still change - before
    EXEC_HOUR of its day (the market's own export runs before its week
    game's ticket exists, so the first reading of a Monday can lack the week
    call); after that it stands. Returns the record, or None without a
    running day.
    """
    base = base_market(market)
    now = now or datetime.now(timezone.utc)
    previous = load_trader(path, base)
    stored = [d for d in previous.get("decisions") or [] if isinstance(d, dict) and d.get("symbol") and d.get("day")]
    data = inputs(conn, base, path)
    running = data["running"]
    start = previous.get("from") or running
    if start is None:
        return None
    by_key = {(d["symbol"], d["day"]): d for d in stored}
    record = trader_record(conn, base, path, read=data, start=start, decisions=by_key)
    if running is not None:
        for symbol, entry in record["symbols"].items():
            said = entry["now"]
            if said is None:
                continue
            fresh = {"symbol": symbol, "day": running, "action": said["action"], "reason": said["reason"], "day_call": said["day_call"],
                     "week_call": said["week_call"], "made_at": now.isoformat(timespec="seconds")}
            old = by_key.get((symbol, running))
            if old is None:
                by_key[(symbol, running)] = fresh
            elif now < exec_instant(running) and (old.get("action"), old.get("reason")) != (fresh["action"], fresh["reason"]):
                by_key[(symbol, running)] = fresh
    record.update({"schema": TRADER_SCHEMA, "generated_at": now.isoformat(timespec="seconds"),
                   "decisions": [by_key[key] for key in sorted(by_key, key=lambda x: (x[1], x[0]))]})
    os.makedirs(os.path.dirname(trader_file(path, base)), exist_ok=True)
    tmp = trader_file(path, base) + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(record, handle, indent=1)
    os.replace(tmp, trader_file(path, base))
    log(f"{base}: paper trader - {record['totals']['all']['positions']} position(s) since {start}, the card is for {running or '-'}")
    return record


# ---------------------------------------------------------------------------
# The replay (read-only): what the trader would have done over the stored record
# ---------------------------------------------------------------------------

def _side(side):
    return "long" if side > 0 else "short"


def print_replay(record, out=print):
    t, rules = record["totals"], record["rules"]
    unit = "EUR" if record["market"] == "shares" else "USDT"
    out(f"== {record['market']}: {record['days']} game days, {record['first_day']} .. {record['last_day']}; the card is for {record['running'] or '-'}")
    out(f"   rules: enter after {rules['persist']} agreeing day calls with the week call, leave after {rules['turn']} day calls against, "
        f"stop {'none' if rules['stop'] is None else format(rules['stop'], '.0%')}{' inside the day' if rules['intraday'] else ' on the close'}, "
        f"longest hold {rules['longest']} days, shorts {'on' if rules['shorts'] else 'off'}")
    for symbol, s in record["symbols"].items():
        out(f"-- {symbol}: closed {s['closed_money']:+.2f}, open {0.0 if s['open_money'] is None else s['open_money']:+.2f} {unit}; "
            f"in the market {s['days_in']} of {record['days']} days; hold {0.0 if s['hold'] is None else s['hold']:+.2f} "
            f"({s['hold_move']:+.1f}%); random {s['random_mean']:+.2f}")
        for p in s["positions"]:
            out(f"     {_side(p['side']):5} {p['entry']} at {p['entry_price']:.4f} -> {p['exit']} at {p['exit_price']:.4f}  {p['held']:3}d  "
                f"{'' if p['move'] is None else format(p['move'], '+.2f') + '%':>8}  {0.0 if p['money'] is None else p['money']:+8.2f}  {p['reason_text']}"
                f"{'' if p['followable'] else '  (booked at 00:00 UTC: not followable)'}")
        if s["open"]:
            p = s["open"]
            out(f"     {_side(p['side']):5} {p['entry']} at {p['entry_price']:.4f} -> open, {p['held']}d, at the newest close "
                f"{'' if p['move'] is None else format(p['move'], '+.2f') + '%'} {0.0 if p['money'] is None else p['money']:+.2f} if closed there")
        n = s["now"]
        if n:
            out(f"     now: {n['action']}{' (' + REASONS[n['reason']] + ')' if n['reason'] else ''} - day call {n['day_call']}, week call {n['week_call']}, "
                f"{n['up_days']} up / {n['down_days']} down in a row")
    for view, label in (("all", "total"), ("long", "long positions only (what a reader can follow)")):
        v = t[view]
        if view == "long" and v["positions"] == t["all"]["positions"]:
            continue
        out(f"== {label}: {v['total']:+.2f} {unit} (closed {v['closed']:+.2f}, open {v['open']:+.2f}) over {v['positions']} position(s), "
            f"{v['wins']} closed with a gain; in the market {v['days_in']} of {t['days_possible']} instrument-days")
        if v["random_mean"] is not None:
            out(f"   the same positions at random ({v['random_runs']} runs): mean {v['random_mean']:+.2f} {unit}; the trader beat {v['random_beaten']:.0%} of them")
    out(f"   hold everything: {t['hold']:+.2f} {unit}")


def print_grid(conn, market, path, out=print):
    """How much the result leans on the rules: every combination of a few values, one line each."""
    data = inputs(conn, market, path)
    base = rules_for(market)
    out(f"== {market}: the same record under other rules (total, positions, share of random runs beaten)")
    for persist in (1, 2, 3):
        for turn in (1, 2, 3):
            cells = []
            for stop in (0.05, base["stop"], None):
                r = trader_record(conn, market, path, rules=rules_for(market, persist=persist, turn=turn, stop=stop), runs=300, read=data)["totals"]["all"]
                cells.append(f"stop {'none' if stop is None else format(stop, '.0%'):>4}: {r['total']:+8.2f} / {r['positions']:2} / "
                             f"{'-' if r['random_beaten'] is None else format(r['random_beaten'], '.0%'):>4}")
            out(f"   persist {persist}, turn {turn}:  " + "   ".join(cells))


# ---------------------------------------------------------------------------
# Self-check (no network, no store)
# ---------------------------------------------------------------------------

def _self_check():
    days = [f"2026-03-{d:02d}" for d in range(2, 22)]           # 20 days from Monday 2 March 2026
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
    bars = {d: (p, p + 0.5, p - 0.5, p + 0.4) for d, p in zip(days, prices)}
    shares, crypto = rules_for("shares"), rules_for("crypto")
    assert rules_for("sharesweek") == shares and rules_for("crypto", persist=5, nonsense=1)["persist"] == 5 and "nonsense" not in rules_for("crypto", nonsense=1)

    class Flat:
        """Test prices: decisions at the day's open; the stop span after the decision is the day's (high, low), none before it."""
        def __init__(self, table):
            self.table = table

        def exec_price(self, day):
            return self.table[day][0] if day in self.table else None

        def before(self, day):
            return None

        def after(self, day):
            return (self.table[day][1], self.table[day][2]) if day in self.table else None

        def close(self, day):
            return self.table[day][3] if day in self.table else None

    def run(day_words, week_words, rules, bar=None, running=None, price=None, start=None):
        words = dict(zip(days + ([running] if running else []), zip(day_words, week_words)))
        return trade(days[:len(day_words) - (1 if running else 0)], lambda d: words.get(d, (None, None)), Flat(bar or bars), rules,
                     running=running, start=start)

    # 1. enter: the week call up and the day call up two days in a row - bought on the second day, not the first
    r = run(["up", "up", "up", "up"], ["up"] * 4, shares)
    assert r["positions"] == [] and r["open"]["entry"] == days[1] and r["open"]["entry_price"] == 101 and r["open"]["held"] == 3, r
    # a flat call breaks the run; a missing call does not; without the week call nothing is bought
    assert run(["up", "flat", "up", "up"], ["up"] * 4, shares)["open"]["entry"] == days[3]
    assert run(["up", None, "up", "up"], ["up"] * 4, shares)["open"]["entry"] == days[2]
    assert run(["up"] * 4, [None, "flat", "down", None], shares)["open"] is None
    # the week call arrives later: bought on the first day it is up, the day calls having persisted already
    assert run(["up"] * 4, [None, None, "up", "up"], shares)["open"]["entry"] == days[2]

    # 2. leave on the day calls: three down in a row (shares), sold at the open of the third; one or two do nothing
    r = run(["up", "up", "down", "down", "up", "down", "down", "down", "up"], ["up"] * 9, shares)
    assert [(p["entry"], p["exit"], p["reason"], p["held"]) for p in r["positions"]] == [(days[1], days[7], "day", 6)] and r["open"] is None, r
    # leave on the week call: it turns down while the day calls are still up - sold that day, and not bought back the same day
    r = run(["up"] * 6, ["up", "up", "up", "down", "down", "up"], shares)
    assert [(p["entry"], p["exit"], p["reason"]) for p in r["positions"]] == [(days[1], days[3], "week")] and r["open"]["entry"] == days[5], r
    # a missing or flat week call keeps the position
    assert run(["up"] * 6, ["up", "up", None, "flat", None, None], shares)["positions"] == []

    # 3. longest hold: sold after `longest` game days, and the signal has to persist anew (2 days) before the next buy
    short_hold = rules_for("shares", longest=3)
    r = run(["up"] * 9, ["up"] * 9, short_hold)
    assert [(p["entry"], p["exit"], p["reason"], p["held"]) for p in r["positions"]] == [(days[1], days[4], "longest", 3)], r
    assert (r["open"]["entry"], r["open"]["held"]) == (days[6], 3), r

    # 4. the shares' stop reads the close and sells at the NEXT open: never on the entry day
    falling = dict(bars)
    falling[days[2]] = (102, 102, 90, 91)          # the day after the buy closes 9.9% under the entry of 101
    r = run(["up"] * 5, ["up"] * 5, shares, bar=falling)
    assert [(p["entry"], p["exit"], p["exit_price"], p["reason"]) for p in r["positions"]] == [(days[1], days[3], 103, "stop")], r
    assert r["open"] is None, "after a stop the day call has to persist anew: one up day is not enough"
    assert r["positions"][0]["entry"] != r["positions"][0]["exit"]

    # 5. crypto: the stop works inside the day at the stop price, also on the entry day (the one way a buy and a sale share a day)
    crash = dict(bars)
    crash[days[1]] = (101, 101, 85, 88)
    r = run(["up"] * 3, ["up"] * 3, crypto, bar=crash)
    p = r["positions"][0]
    assert (p["entry"], p["exit"], p["reason"]) == (days[1], days[1], "stop") and abs(p["exit_price"] - 101 * 0.9) < 1e-9 and p["held"] == 1, r
    # 6. crypto shorts: week down and two down days -> short; the week turning up covers it (the week call is named before the day
    # calls) and, the day calls up two days, buys the same day; with the week still down the two up days cover it and nothing is bought
    r = run(["down", "down", "down", "up", "up", "up"], ["down", "down", "down", "down", "up", "up"], crypto)
    assert [(p["side"], p["entry"], p["exit"], p["reason"]) for p in r["positions"]] == [(-1, days[1], days[4], "week")], r
    r2 = run(["down", "down", "down", "up", "up", "up"], ["down"] * 6, crypto)
    assert [(p["side"], p["exit"], p["reason"]) for p in r2["positions"]] == [(-1, days[4], "day")] and r2["open"] is None, r2
    assert r["open"]["side"] == 1 and r["open"]["entry"] == days[4], r
    assert run(["down"] * 4, ["down"] * 4, shares)["open"] is None, "the shares never sell short"
    # a short's stop is above the entry
    spike = dict(bars)
    spike[days[2]] = (102, 120, 101, 103)
    r = run(["down"] * 4, ["down"] * 4, crypto, bar=spike)
    assert r["positions"][0]["reason"] == "stop" and abs(r["positions"][0]["exit_price"] - 101 * 1.1) < 1e-9 and r["positions"][0]["side"] == -1, r

    # 7. the card's day: no bar yet, only the decision
    assert run(["up", "up"], ["up", "up"], shares, running=days[1])["now"]["action"] == "buy"
    assert run(["up", "flat"], ["up", "up"], shares, running=days[1])["now"]["action"] == "wait"
    now = run(["up", "up", "up", "up"], ["up", "up", "up", "down"], shares, running=days[3])["now"]
    assert (now["action"], now["reason"], now["week_call"]) == ("sell", "week", "down"), now
    assert run(["up", "up", "up", "down"], ["up"] * 4, shares, running=days[3])["now"]["action"] == "hold"
    now = run(["down", "down", "down", "up", "up"], ["down"] * 5, crypto, running=days[4])["now"]
    assert (now["action"], now["reason"]) == ("cover", "day"), now
    assert run(["down", "down"], ["down", "down"], crypto, running=days[1])["now"]["action"] == "short"

    # 7b. the first day: nothing is bought before it, the calls before it still count, and the card's day obeys it too
    r = run(["up"] * 6, ["up"] * 6, shares, start=days[4])
    assert r["positions"] == [] and r["open"]["entry"] == days[4], r
    assert run(["up", "up"], ["up", "up"], shares, running=days[1], start=days[2])["now"]["action"] == "wait"
    assert run(["up", "up"], ["up", "up"], shares, running=days[1], start=days[1])["now"]["action"] == "buy"

    # 7c. the booking hour: 09:00 in Brussels is 08:00 UTC in winter and 07:00 UTC in summer
    assert exec_instant("2026-11-18") == datetime(2026, 11, 18, 8, tzinfo=timezone.utc) and exec_instant("2026-07-01") == datetime(2026, 7, 1, 7, tzinfo=timezone.utc)
    # crypto with hourly bars: booked at the open of the 09:00 Brussels hour; the stop spans split there; without the bar the close before, not followable
    daily = {"2026-11-17": (100.0, 106.0, 94.0, 101.0), "2026-11-18": (101.0, 108.0, 90.0, 104.0)}
    hours = {datetime(2026, 11, 18, h, tzinfo=timezone.utc): (100.0 + h, 101.0 + h, 99.0 + h) for h in range(24)}
    hours[datetime(2026, 11, 18, 3, tzinfo=timezone.utc)] = (103.0, 104.0, 90.0)       # a dip in the night, before the decision
    coin = Prices("crypto", daily, hours)
    assert coin.exec_price("2026-11-18") == 108.0 and coin.followable("2026-11-18") and coin.close("2026-11-18") == 104.0
    assert coin.before("2026-11-18") == (108.0, 90.0) and coin.after("2026-11-18") == (124.0, 107.0)
    assert coin.exec_price("2026-11-17") is None and not coin.followable("2026-11-17") and coin.before("2026-11-17") is None and coin.after("2026-11-17") == (106.0, 94.0)
    assert Prices("crypto", daily, {}).exec_price("2026-11-18") == 101.0, "without hourly bars: the close before the day"
    # a standing stop is hit in the night before the day's decision: sold at the stop price, and nothing is bought back that morning
    night = Prices("crypto", {"2026-11-16": (99.0, 100.0, 98.0, 100.0), **daily}, hours)
    words = {"2026-11-16": ("up", "up"), "2026-11-17": ("up", "up"), "2026-11-18": ("up", "up")}
    r = trade(sorted(words), words.get, night, rules_for("crypto"))
    assert [(q["entry"], q["entry_price"], q["exit"], q["reason"]) for q in r["positions"]] == [("2026-11-17", 100.0, "2026-11-18", "stop")] and r["open"] is None, r
    share = Prices("shares", daily)
    assert share.exec_price("2026-11-18") == 101.0 and share.before("2026-11-18") is None and share.after("2026-11-18") is None and share.followable("2026-11-18")

    # 7d. a decision counts when it was stored before 09:00 Brussels of its day, with the action the position needed
    stored = {("BTC", "2026-11-18"): {"action": "buy", "made_at": "2026-11-18T00:06:00+00:00"}, ("ETH", "2026-11-18"): {"action": "buy", "made_at": "2026-11-18T08:00:00+00:00"}}
    assert _decided_in_time(stored, "BTC", "2026-11-18", ("buy",)) and not _decided_in_time(stored, "ETH", "2026-11-18", ("buy",))
    assert not _decided_in_time(stored, "BTC", "2026-11-18", ("sell",)) and not _decided_in_time(stored, "SOL", "2026-11-18", ("buy",)) and not _decided_in_time(None, "BTC", "2026-11-18", ("buy",))

    # 8. the week call is unknown on a shares Monday, known every crypto day
    assert not week_call_known("shares", "2026-03-02") and week_call_known("shares", "2026-03-03") and week_call_known("sharesweek", "2026-03-06")
    assert week_call_known("crypto", "2026-03-02") and week_call_known("cryptoweek", "2026-03-08")

    # 9. money: the market's own costs; a share position that buys no whole share has no money
    crypto_costs, share_costs = settle.costs_for("crypto"), settle.costs_for("shares")
    money, move = position_money(crypto_costs, 100.0, 1, 100.0, 110.0)
    assert abs(money - (110 * 0.999 - 100 * 1.001)) < 1e-9 and abs(move - 10.0) < 1e-6
    money, move = position_money(crypto_costs, 100.0, -1, 100.0, 90.0)
    assert abs(money - (100 * 0.999 - 90 * 1.001)) < 1e-9 and abs(move - 10.0) < 1e-6, "a short's move is shown as its gain"
    money, _ = position_money(share_costs, 1000.0, 1, 40.0, 44.0)       # 25 shares: 1,000 -> 1,100, 5.00 + 3.50 and 7.50 + 3.85
    assert abs(money - (100.0 - 8.5 - 11.35)) < 1e-9, money
    assert position_money(share_costs, 1000.0, 1, 1500.0, 1600.0) == (None, None) and position_money(crypto_costs, 100.0, 1, None, 5.0) == (None, None)

    # 10. random placements: no overlap, every length kept, inside the days
    rng = random.Random(1)
    for _ in range(200):
        placed = random_placements([(1, 3), (-1, 5), (1, 1)], 20, rng)
        assert sorted(e - s for _side, s, e in placed) == [1, 3, 5] and all(0 <= s < e <= 20 for _side, s, e in placed)
        ordered = sorted(placed, key=lambda x: x[1])
        assert all(a[2] <= b[1] for a, b in zip(ordered, ordered[1:])), placed
    assert random_placements([(1, 30)], 20, rng) == []
    print("paper trader: enters on a persisting day call with the week call, leaves on a turned week or day call, a stop or the longest hold; "
          "shares never buy and sell on one day, crypto only through its stop")


def main(argv):
    if "--replay" not in argv:
        _self_check()
        return 0
    import os
    import sqlite3
    market = argv[argv.index("--replay") + 1]
    root = argv[argv.index("--root") + 1] if "--root" in argv else os.getcwd()
    db = argv[argv.index("--db") + 1] if "--db" in argv else os.path.join(root, "data", "markets.sqlite3")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)      # read-only: the replay never writes
    conn.row_factory = sqlite3.Row
    try:
        print_replay(trader_record(conn, market, root))
        if "--grid" in argv:
            print_grid(conn, market, root)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
