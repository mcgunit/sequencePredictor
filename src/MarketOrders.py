"""
The crypto ORDER BOOK (30 Oct 2026, the owner's ask: "the user instructions
must match a paper book exactly"; design orderbook-DESIGN-30oct.md, decisions
D1-D7 confirmed by the owner on 30 Oct 2026). It books what the crypto day
and week cards tell a reader to do: buy at market when the card appears, put
a sell limit at X, sell at market at a deadline or on the week's daily-call
exit. The existing books (daily, hold, open, followed, guarded) are not
touched; this is a new book next to them, exported as trading["orders"] of
data/markets/crypto.json and cryptoweek.json.

Why intraday bars: a card appears at about 08:30-10:30 UTC. With daily bars
the price at that moment is unknown, and so is whether the day's high came
before or after it, so a day order could never be judged. The book reads
Binance 1h bars (continuously from INTRADAY_SINCE) and 1m bars around every
card publication (table intraday_bars, src/MarketData.py).

The rules (exact; the constants below are stated on the Admin card):
  entry     P = the open of the first 1m bar at or after the card's
            publication (ticket_published, first seen; before 30 Oct the
            ticket's madeAt + PUBLISH_LAG_FALLBACK, flagged "estimated").
            Never priced from an hourly or daily bar; a missing 1m bar is
            PENDING, never guessed.
  no order  log(X / base close) < MIN_GAP: "target below fees"; P >= X:
            "already reached"; log(X / P) < MIN_GAP: "too close to the
            limit" - a row with money 0 that is counted, never a position.
  fill      at X (never better, also on a gap open above X), at the first
            bar whose high is at least X x (1 + FILL_THROUGH) - a touch is
            not a fill. Scanned in 1m bars to the first full hour, then 1h
            bars, then the 1m bars of the exit's own partial hour up to the
            minute BEFORE the exit's minute: a fill and a card in the same
            minute - the card wins (the reader cannot know the limit filled).
  exit      the first of: the fill; a card-time exit (week: the day card
            that turns the coin flat or down - D2, kept by the owner); the
            deadline = the publication of the next card of the same game
            (D1), or DEADLINE_FALLBACK_HOURS after the interval's close
            (12:00 UTC the day after) if no card appeared by then; card-time
            exits at the open of the 1m bar at or after the card.
  money     cash_pnl(log(exit / P)), FEE_PER_LEG on both legs, scaled by a
            sized row's size (position_of), as every other book.
  day (crypto)   the followed day row's up calls (followed_plan's pick for
            the day, unchanged); X anchored at the previous game day's close
            (the bin is the return from it); D5 roll: the same coin called up
            again on the next card with its new X' at least MIN_GAP above
            the price keeps the position, the limit moves to X', no fees -
            one position spanning days, booked on the day it closes.
  week (cryptoweek)  the followed week row's up calls are the permission,
            Monday's followed day call the trigger (D4: not up -> no Monday
            buy); X anchored at the previous Sunday's close; one round trip
            per coin per week (D3, no buy-back); entry at the later of the
            week card and Monday's day card.
  late      (week only, a separate ledger, D7) on each day card Tuesday to
            Sunday: week call up, that day's followed daily call up, the price
            at least max(LATE_MIN_GAP, MIN_GAP) below X and the deadline at
            least LATE_MIN_LEFT away - each showing one cohort position (the
            same X, early exit and deadline). Never part of "Following this
            card so far".
  newcomer  (day only, a separate ledger; owner, 31 Oct 2026) on every card
            that ROLLS a held position the reader who did not buy on the
            earlier card is told to buy now: each roll card books a newcomer
            entry at that card's publication P, the same new limit X, the
            same deadline rule and the same no-order tests as a fresh day
            order. It then rolls exactly like the main position (the same D5
            test on the next card: called up again with X'' at least MIN_GAP
            above the price keeps it, else it is sold at market at that card).
            By construction the roll test IS the fresh-order test at the same
            card and price, so a newcomer entry is in practice never skipped
            (a card whose test fails sells the main position instead - "sell
            now" - and books no newcomer); the tests are kept so the ledger
            follows the fresh-order rule if the roll test ever changes.
            Never part of "Following this card so far": its own line.
  benchmark every active coin bought at every card's publication without a
            limit and sold at the same deadline: the market under the same
            timing.
  reference the literal end the owner first named (00:00 UTC, the bin's own
            horizon - 01:00 or 02:00 Brussels, which no reader executes): the
            same orders with the deadline at the interval's close and no
            roll, for Admin only.

X variants (all booked, Admin): edge (PRIMARY, pre-registered 30 Oct 2026 in
ORDER_VARIANT_LOG before the back-test was read: the predicted bin's lower
edge - a right call fills, since a close inside or above the bin is at least
X), mid (the bin's representative return, the price the pages draw) and fixed
(P x 1.01 a day, x 1.04 a week: a control that ignores the bin). The user line
books per card the variant that card showed (shown_variant); a switch is a
logged decision at the review point (REVIEW_RULE), never a "best so far".

trading["orders"] (crypto and cryptoweek only; times ISO UTC, prices in USDT,
money in USDT at STAKE per order):
  {
    "market": "crypto" | "cryptoweek", "game": "day" | "week",
    "generated_at": iso, "since": "2026-10-13", "rule": text, "late_rule": text (week only),
    "constants": {"fee_per_leg", "min_gap", "fill_through", "publish_lag_fallback_min", "deadline_fallback_hours",
                  "late_min_gap", "late_min_left_hours", "stake", "fixed_gap"},
    "primary": "edge", "variants": {"edge": text, "mid": text, "fixed": text},
    "variant_log": [[date, variant, reason]], "review": text,
    "books":     {variant: BOOK},          # D1 deadline (next card), roll on for the day game
    "reference": {variant: BOOK},          # literal 00:00 UTC end, no roll
    "late":      {variant: BOOK + {"refused": {reason: n}}}      (week only)
    "newcomer":  {variant: BOOK}, "newcomer_rule": text            (day only; kind "newcomer", 31 Oct 2026)
    "benchmark": {"points": [[interval, coins closed, money, all closed]], "stats": STATS},
    "record":    {"shown": [[interval, variant]], "stats": STATS},   # the user line: per interval the variant the card showed
    "publications": [[made_on, interval, published_at, estimated]],
    "coverage":  {symbol: {"1h_first", "1h_last", "1h_bars", "1m_bars"}},
    "now": NOW or null
  }
  BOOK  = {"orders": [ORDER], "stats": STATS}
  ORDER = {"interval": day D (day) or the week's Sunday (week), "days": [days the position spanned (day: rolls)],
           "symbol", "model", "bin", "scale", "variant", "kind": "main" | "late" | "reference" | "newcomer",
           "status": "closed" | "open" | "pending" | "skip" | "not followable",
           "reason": skip / not-followable / pending reason or null,
           "published_at": iso, "estimated": bool, "base_close", "limit", "limits": [every limit, rolls included],
           "entry_at", "entry_price", "limit_pct": 100 x log(limit / entry) or null,
           "exit_kind": "limit" | "early" | "next card" | "fallback" | "close" | null,
           "exit_at", "exit_price", "move": log(exit / entry) or null, "money": USDT or null (closed only),
           "rolls": n, "provisional": bool (a decision on the last closed minute's close, its own minute not stored yet),
           "live": bool (followable as booked: published before the interval's end)}
  STATS = {"orders" (positions: closed + open + pending), "closed", "open", "pending", "filled", "early", "time_stop",
           "skips": {reason: n}, "not_followable", "total" (closed money), "wins", "estimated" (closed with an estimated time),
           "rolls"}
  NOW (day)  = {"interval", "made_on", "published_at", "estimated", "model", "variant", "deadline_fallback",
                "deadline_text" (Brussels), "symbols": {symbol: {"status": "buy now" | "hold (rolled)" | "sell now" | "no order",
                "call", "bin", "limit", "entry_price", "limit_pct", "reason", "provisional", "held_since", "state",
                "newcomer": null | {"status": "buy now" | "no order", "state", "limit", "entry_price", "limit_pct", "reason", "provisional"}
                            (a roll card's newcomer entry),
                "newcomers_held": [[first day, entry price]] (earlier newcomer entries held into this card or sold at it)}}}
  NOW (week) = {"interval", "first_day", "made_on", "published_at", "estimated", "model", "variant", "deadline_fallback",
                "deadline_text", "running_day", "day_model",
                "symbols": {symbol: {"status": "buy now" | "not bought" | "no order" | "sell now", "call", "bin", "limit",
                "entry_price", "limit_pct", "reason", "state", "exit_kind", "exit_at", "money"}},
                "late": {symbol: {"shown": bool, "reason", "price", "gap_pct", "limit"}} (the running day's card)}

Self-check: python3 -m src.MarketOrders (part of npm test; no network).
Back-test into a scratch copy (never the production store):
    python3 -m src.MarketOrders --backfill 2026-10-13 --root <scratch> --db <scratch>/data/markets.sqlite3 [--export]
It is a module CLI, not a pipeline entry point: it writes only the store
and the root it is given (both required) and takes no process.lock. A root
or store inside a git checkout (the dev repo, production) is refused unless
--allow-checkout is given (31 Oct 2026; the approved production backfill).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

try:
    from src.MarketData import (DEFAULT_DB, INTRADAY_LOOKBACK, INTRADAY_SINCE, base_instrument_id, base_market, connect, instruments,
                                is_week_game, iso, update_intraday)
    from src.MarketGame import K_BINS, GAME_SCHEMA, bin_interval, representative_return
    from src.MarketSettle import (FEE_PER_LEG, ORDER_MARKETS, STAKE, cash_pnl, close_instant, day_file_date, followed_plan, ledger, next_session,
                                  position_of, call_word, ticket_written)
except ImportError:  # imported from within src/
    from MarketData import (DEFAULT_DB, INTRADAY_LOOKBACK, INTRADAY_SINCE, base_instrument_id, base_market, connect, instruments,
                            is_week_game, iso, update_intraday)
    from MarketGame import K_BINS, GAME_SCHEMA, bin_interval, representative_return
    from MarketSettle import (FEE_PER_LEG, ORDER_MARKETS, STAKE, cash_pnl, close_instant, day_file_date, followed_plan, ledger, next_session,
                              position_of, call_word, ticket_written)

# ORDER_MARKETS (crypto, cryptoweek) lives in MarketSettle, which exports trading["orders"] for them
ORDER_SINCE = "2026-10-13"      # the first card date the book starts from: day orders for the days after it, weeks closing after it
MIN_GAP = 0.003                 # X at least 0.30 % (log) above the entry: two fee legs plus 0.1 %, else a fill loses money
FILL_THROUGH = 0.0005           # a limit fills only on a high of at least X x 1.0005: a touch is not a fill (queue priority)
PUBLISH_LAG_FALLBACK = timedelta(minutes=15)    # madeAt -> page, where ticket_published has no row (one 12-minute gap seen on 29 Oct)
DEADLINE_FALLBACK_HOURS = 12    # no next card by 12:00 UTC the day after the interval's close: sold at market then
LATE_MIN_GAP = 0.010            # late week entries: the price at least 1 % (log) below X (D7)
LATE_MIN_LEFT = timedelta(hours=24)   # ... and at least 24 h before the deadline (pre-registered; less is a coin flip on fees)
FIXED_GAP = {"day": 0.01, "week": 0.04}   # the 'fixed' control variant: X = P x (1 + gap)
# the export-time publication record: a ticket first exported more than this long after its madeAt is not stamped
# (a hand-run --settle hours later, or the first export after a deploy, would book a time no reader saw); it then
# keeps the estimate madeAt + PUBLISH_LAG_FALLBACK. Since 31 Oct 2026 only the fallback when the page being replaced
# cannot be read: otherwise "did the old page already show this ticket" decides (record_publication)
PUBLISH_RECORD_WINDOW = timedelta(hours=3)

ORDER_VARIANTS = ("edge", "mid", "fixed")
ORDER_VARIANT_TEXT = {
    "edge": "the predicted bin's lower edge from the base close: when the call comes true the close is at least X, so a right call fills",
    "mid": "the predicted bin's representative return from the base close: the predicted price the pages draw",
    "fixed": "the entry price plus 1 % (day) or 4 % (week): a control that ignores the bin",
}
# Pre-registered on 30 Oct 2026, before the back-test was read (the design's in-memory mechanics check had shown all
# three on 16 orders; the principle, not that table, chose it). A switch is appended here with its date and reason.
ORDER_VARIANT_LOG = [("2026-10-30", "edge", "pre-registered: a right call fills")]
REVIEW_RULE = ("one review, at 60 live day orders or 31 Jan 2027, whichever is later: switch only if another variant beats "
               "the primary in a paired per-order comparison with the lower bound of the mean difference above 0, and with the owner's decision")

ORDER_RULE_DAY = ("buy at market when the day card appears, a sell limit at X (the predicted bin's lower edge from the previous close), "
                  "sold at the limit or at market when the next day card appears (12:00 UTC the next day at the latest); the same coin "
                  "called up again with its new limit at least 0.30 % above the price keeps the position and only moves the limit")
ORDER_RULE_WEEK = ("buy at market when the week card appears if the week call is up and Monday's daily call is up, a sell limit at X "
                   "(the bin's lower edge from Sunday's close), sold at the limit, at market when a daily card turns the coin flat or down "
                   "(cancel the limit first), or at market when next Monday's week card appears (12:00 UTC Monday at the latest); "
                   "one round trip per coin per week, no buy-back")
NEWCOMER_RULE = ("on a day card that rolls a held position (the same coin called up again, its new limit at least 0.30 % above the price), "
                 "a reader who did not buy on the earlier card is told to buy now: each such showing is booked as a newcomer entry - bought at "
                 "the price when that card appeared, the same new limit, sold at the limit or at market when the next day card appears "
                 "(12:00 UTC the next day at the latest), under the same no-order tests as a fresh day order, and rolled like the main "
                 "position when the next card calls the coin up again - in a separate record")
LATE_RULE = ("a late entry is shown on a day card Tuesday to Sunday when the week call is up, that day's daily call is up, the price is "
             "at least 1 % below the limit and the deadline is at least 24 h away; each showing is booked as its own position "
             "(the same limit, early exit and deadline) in a separate record")

SKIP_TARGET = "target below fees"
SKIP_REACHED = "already reached"
SKIP_CLOSE = "too close to the limit"
SKIP_NO_TARGET = "no target"              # an open-ended lowest bin has no lower edge (a sized row's up call on bin 0)
SKIP_DAILY = "daily call not up"          # D4: the week call is up, Monday's daily call is not
NOT_FOLLOWABLE = "published after the interval's end"
LATE_FAR = "price within 1 % of the limit"
LATE_SHORT = "less than 24 h left"
LATE_DAILY = "daily call not up"

POSITION_STATES = ("closed", "open", "pending")
CARD_ENDS = ("next card", "early")       # ends at a card that has appeared (never a clock time that may lie ahead)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def ts(dt):
    """An aware datetime as integer epoch seconds."""
    return int(dt.timestamp())


def at(seconds):
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def ceil_min(t):
    return -(-int(t) // 60) * 60


def floor_min(t):
    return int(t) // 60 * 60


def floor_hour(t):
    return int(t) // 3600 * 3600


def iso_ts(t):
    return None if t is None else iso(at(t))


def fallback_deadline(market, interval):
    """DEADLINE_FALLBACK_HOURS after the interval's close (close_instant: 00:00 UTC after the day, Monday 00:00 after the week)."""
    return ts(close_instant(market, interval)) + DEADLINE_FALLBACK_HOURS * 3600


def brussels_text(t):
    """A UTC instant as the reader's clock: 'Sun 25 Oct 12:16' in Brussels time (DST from the zone database, never hard-coded)."""
    try:
        from zoneinfo import ZoneInfo
        local = at(t).astimezone(ZoneInfo("Europe/Brussels"))
    except Exception:
        local = at(t)
    return f"{local:%a} {local.day} {local:%b %H:%M}"


# ---------------------------------------------------------------------------
# Prices: one coin's stored 1m and 1h bars (a "tape")
# ---------------------------------------------------------------------------

class Tape:
    """
    One coin's 1m and 1h bars keyed by their start (epoch seconds) as (open,
    high, low, close), and `now`: the frontier up to which the store is
    complete - a needed bar ending after it is not fetched yet (the order is
    OPEN), one ending before it is a hole (PENDING). load_tapes sets it to the
    end of the newest stored hour when that is earlier than the clock, so a
    store read some minutes after its last fetch is not taken for a hole.
    """

    def __init__(self, m1=None, h1=None, now=None):
        self.m1 = dict(m1 or {})
        self.h1 = dict(h1 or {})
        self.now = int(now) if now is not None else ts(datetime.now(timezone.utc))


def load_tapes(conn, market, start, now):
    """{symbol: Tape} of the base market's instruments from `start` (epoch seconds) on."""
    out = {}
    a = iso_ts(start)
    for member in instruments(conn, base_market(market), active_only=False):
        tape = Tape(now=now)
        for r in conn.execute("SELECT interval, start, open, high, low, close FROM intraday_bars WHERE instrument_id = ? AND start >= ?",
                              (base_instrument_id(conn, member), a)).fetchall():
            key = ts(datetime.fromisoformat(r["start"]))
            (tape.m1 if r["interval"] == "1m" else tape.h1)[key] = (float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
        if tape.h1:
            tape.now = min(tape.now, max(tape.h1) + 3600)
        out[member["symbol"]] = tape
    return out


def quote_before(tape, t, within=3 * 3600):
    """The close of the last closed bar before `t` (1m first, else 1h) within `within` seconds: a provisional price, never a booked one."""
    best = None
    for start, bar in tape.m1.items():
        if start + 60 <= t and start + 60 > t - within and (best is None or start > best[0]):
            best = (start, bar[3])
    if best is None:
        for start, bar in tape.h1.items():
            if start + 3600 <= t and start + 3600 > t - within and (best is None or start > best[0]):
                best = (start, bar[3])
    return best[1] if best else None


def price_at(tape, t):
    """
    The market price at instant `t`: ("ok", open of the 1m bar at or after
    t, its start). At an exact hour the 1h bar's open is the same trade and
    is used when the minute is not stored. ("open", provisional quote,
    start) while that minute has not closed yet; ("pending", None, start)
    when it should be stored and is not.
    """
    m = ceil_min(t)
    bar = tape.m1.get(m)
    if bar is not None:
        return "ok", bar[0], m
    if m % 3600 == 0 and m in tape.h1:
        return "ok", tape.h1[m][0], m
    if m + 60 > tape.now:
        return "open", quote_before(tape, t), m
    return "pending", None, m


def scan_fill(tape, start, stop, threshold):
    """
    The first bar in [start, stop) (epoch seconds on minute boundaries)
    whose high reaches `threshold`: 1h bars where a whole stored hour lies
    inside the span, 1m bars elsewhere. Returns ("fill", bar start),
    ("none", None), ("open", t) when the scan reached a bar that has not
    closed yet, or ("pending", t) at a bar that should be stored and is not.
    """
    cur = int(start)
    while cur < stop:
        whole_hour = cur % 3600 == 0 and cur + 3600 <= stop
        if whole_hour and cur in tape.h1:
            bar, span = tape.h1[cur], 3600
        elif cur in tape.m1:
            bar, span = tape.m1[cur], 60
        else:
            span = 3600 if whole_hour else 60
            return ("open" if cur + span > tape.now else "pending"), cur
        if bar[1] >= threshold:
            return "fill", cur
        cur += span
    return "none", None


# ---------------------------------------------------------------------------
# One order
# ---------------------------------------------------------------------------

def target_price(variant, base_close, bin_value, edges, entry_price, horizon):
    """X of a variant: edge = base x exp(the bin's lower edge), mid = base x exp(its representative return), fixed = P x (1 + FIXED_GAP)."""
    if variant == "fixed":
        return None if entry_price is None else float(entry_price) * (1.0 + FIXED_GAP[horizon])
    if base_close is None or bin_value is None or not edges:
        return None
    if variant == "edge":
        low = bin_interval(int(bin_value), edges)[0]
        return None if low is None else float(base_close) * math.exp(low)
    if variant == "mid":
        return float(base_close) * math.exp(representative_return(int(bin_value), edges))
    raise ValueError(f"unknown variant {variant!r}")


def skip_reason(X, P, base_close, min_gap=MIN_GAP):
    """Why no order is placed at entry price P under limit X, or None (order placed). base_close None skips the "target below fees" test (the fixed variant ignores the bin)."""
    if X is None:
        return SKIP_NO_TARGET
    if base_close is not None and math.log(X / base_close) < min_gap:
        return SKIP_TARGET
    if P >= X:
        return SKIP_REACHED
    if math.log(X / P) < min_gap:
        return SKIP_CLOSE
    return None


def close_position(order, kind, t, price, scale):
    order.update({"exit_kind": kind, "exit_at": t, "exit_price": price})
    order["move"] = math.log(price / order["entry_price"])
    order["money"] = scale * cash_pnl(order["move"])


def run_order(tape, t0, X, exits=(), deadline=None, base_close=None, scale=1.0, min_gap=MIN_GAP, check=True):
    """
    One order (section 3.1): bought at the 1m open at or after `t0`, a limit
    at X (None: no limit, the benchmark), the card-time `exits` [(instant,
    kind)] and the `deadline` (instant, kind). Pure: the tape is all it
    reads. `check` False skips the no-order tests (the benchmark, a late
    entry already tested). Returns the order dict (times as epoch seconds).
    """
    status, P, m0 = price_at(tape, t0)
    order = {"status": "open", "reason": None, "entry_at": m0, "entry_price": P, "limit": X, "exit_kind": None, "exit_at": None,
             "exit_price": None, "move": None, "money": None, "provisional": status == "open"}
    if status == "pending":
        order.update(status="pending", reason="no 1m bar at the entry")
        return order
    if P is None:
        return order                                  # the card is up, its minute not stored and no quote before it: open
    # 31 Oct 2026 (review): the no-order tests run whenever `check` is on, X None included. They used to be skipped
    # for X None, so an up call without a target (a sized row's up call on the open lowest bin: edge has no lower
    # edge) was booked as an order WITHOUT a limit, and the day book then crashed on held["limit"] * (...) - the
    # whole trading["orders"] block was lost for the market. skip_reason books it as SKIP_NO_TARGET.
    if check:
        reason = skip_reason(X, P, base_close, min_gap)
        if reason:
            order.update(status="skip", reason=reason, entry_at=m0)
            return order
    if status == "open":
        return order                                  # decided on a provisional quote; booked once the minute is stored
    ends = sorted([e for e in exits if e[0] > t0] + ([deadline] if deadline else []))
    if not ends:
        return order
    end_t, end_kind = ends[0]
    unsure = False
    if X is not None:
        found, when = scan_fill(tape, m0, floor_min(end_t), X * (1.0 + FILL_THROUGH))
        if found == "fill":
            order["status"] = "closed"
            close_position(order, "limit", when, X, scale)
            return order
        if found == "pending":
            order.update(status=found, reason="a bar is missing in the fill scan")
            return order
        if found == "open":
            # 31 Oct 2026 (review): a card that has appeared (an early exit, the next card) is a sale the reader makes
            # NOW if the limit has not filled by then. The minutes just before it are often not stored yet at the
            # export that publishes it (the intraday refresh runs seconds before the stamp), and the order used to
            # stay plainly open - the week card then never said "sell now". It is booked as a provisional exit at
            # that card (no money until the bars are in; the next export decides it for good). A clock end (the
            # fallback, 00:00 UTC) may still lie ahead and stays open.
            if end_kind not in CARD_ENDS:
                order["status"] = "open"
                return order
            unsure = True
    state, price, m = price_at(tape, end_t)
    if state == "ok" and not unsure:
        order["status"] = "closed"
        close_position(order, end_kind, m, price, scale)
    elif state == "ok":
        order.update(status="open", exit_kind=end_kind, exit_at=m, exit_price=price, provisional=True)
    else:
        order.update(status=state, reason=None if state == "open" else "no 1m bar at the exit", exit_kind=end_kind, exit_at=m,
                     provisional=state == "open" or unsure)
    return order


# ---------------------------------------------------------------------------
# Cards: what each published ticket said
# ---------------------------------------------------------------------------

def record_publication(conn, market, made_on, made_at, now=None, shown_before=None):
    """
    ticket_published, first seen only: the moment a ticket's first export
    put it on the page. `shown_before` (31 Oct 2026, review): whether the page
    file being replaced already showed this made_on (export_page_json reads
    it). True: not the first showing (the first export after the deploy, a
    hand re-run) - not stamped, the estimate stays. False: this export IS the
    first showing and is stamped however long after madeAt it comes - the
    3-hour window alone fell back to the estimate madeAt + 15 min, a time
    BEFORE the page appeared (lookahead) whenever a slow run exported more
    than 3 h after the ticket was made (runs have taken up to 3.5 h). None
    (the page file could not be read): the PUBLISH_RECORD_WINDOW rule as
    before. Returns True when a row was written.
    """
    if market not in ORDER_MARKETS or not made_on:
        return False
    now = now or datetime.now(timezone.utc)
    if shown_before is True:
        return False
    if made_at and shown_before is None:
        try:
            made = datetime.fromisoformat(str(made_at))
            made = made if made.tzinfo else made.replace(tzinfo=timezone.utc)
            if now - made > PUBLISH_RECORD_WINDOW:
                return False
        except ValueError:
            pass
    cursor = conn.execute("INSERT OR IGNORE INTO ticket_published (market, made_on, made_at, published_at) VALUES (?, ?, ?, ?)",
                          (market, made_on, made_at, iso(now)))
    conn.commit()
    return bool(cursor.rowcount)


def publications(conn, path, market, lag=PUBLISH_LAG_FALLBACK):
    """{made_on: (published instant in epoch seconds, estimated)}: ticket_published, else when the ticket was made (madeAt, else the file's mtime) + `lag`."""
    folder = os.path.join(path, "data", "database", market)
    stored = {r["made_on"]: r["published_at"] for r in conn.execute("SELECT made_on, published_at FROM ticket_published WHERE market = ?", (market,))}
    out = {}
    if not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        made_on = day_file_date(name)
        if not made_on:
            continue
        if made_on in stored:
            out[made_on] = (ts(datetime.fromisoformat(stored[made_on])), False)
            continue
        made = ticket_written(os.path.join(folder, name))
        if made is not None:
            out[made_on] = (ts(made + lag), True)
    return out


def interval_of(market, made_on):
    """The interval a ticket made on `made_on` is for: the next game day (day), the Sunday a week later (week)."""
    if is_week_game(market):
        return (date.fromisoformat(made_on) + timedelta(days=7)).isoformat()
    return next_session(market, made_on)


def load_cards(conn, path, market, since=ORDER_SINCE, lag=PUBLISH_LAG_FALLBACK):
    """
    Every card of the market whose interval ends after `since` (a week: its
    Sunday after since's week began), oldest first: {made_on, interval,
    pub, estimated, calls: {model: {symbol: (bin, size)}}, edges: {symbol:
    edges}, base: {symbol: close}}. The edges and the base close are the
    ones the page used when the card was made: the made_on game day's
    (latest_edges then) and that day's close.
    """
    conn.executescript(GAME_SCHEMA)
    folder = os.path.join(path, "data", "database", market)
    if not os.path.isdir(folder):
        return []
    pubs = publications(conn, path, market, lag)
    first = date.fromisoformat(since)
    if is_week_game(market):
        first -= timedelta(days=first.weekday() + 7)      # the card of the week `since` lies in was made the Sunday before
    members = {i["symbol"]: i for i in instruments(conn, market, active_only=False)}
    close_cache = {}

    def close_on(symbol, day):
        member = members.get(symbol)
        if member is None:
            return None
        iid = base_instrument_id(conn, member)
        if iid not in close_cache:
            close_cache[iid] = {r["date"]: float(r["close"]) for r in conn.execute("SELECT date, close FROM bars WHERE instrument_id = ?", (iid,))}
        return close_cache[iid].get(day)

    cards = []
    for name in sorted(os.listdir(folder)):
        made_on = day_file_date(name)
        if not made_on or date.fromisoformat(made_on) < first or made_on not in pubs:
            continue
        interval = interval_of(market, made_on)
        if interval <= since and not is_week_game(market):
            continue
        try:
            with open(os.path.join(folder, name)) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        row = conn.execute("SELECT MAX(date) FROM game_days WHERE market = ? AND date <= ?", (market, made_on)).fetchone()
        game_day = row[0] if row else None
        if not game_day:
            continue
        rows = conn.execute("SELECT position, symbol, edges FROM game_days WHERE market = ? AND date = ? ORDER BY position", (market, game_day)).fetchall()
        symbols = [r["symbol"] for r in rows]
        calls = {}
        for entry in data.get("newPrediction") or []:
            tickets = entry.get("predictions") or []
            if not tickets or not tickets[0]:
                continue
            sizes = entry.get("positions") if isinstance(entry.get("positions"), list) else None
            per = {}
            for pos, symbol in enumerate(symbols):
                if pos >= len(tickets[0]) or tickets[0][pos] is None:
                    continue
                size = None
                if sizes is not None and pos < len(sizes) and sizes[pos] is not None:
                    try:
                        size = float(sizes[pos])
                    except (TypeError, ValueError):
                        size = None
                per[symbol] = (int(tickets[0][pos]), size)
            calls[entry.get("name")] = per
        pub, estimated = pubs[made_on]
        cards.append({"made_on": made_on, "interval": interval, "pub": pub, "estimated": estimated, "calls": calls, "symbols": symbols,
                      "edges": {r["symbol"]: json.loads(r["edges"]) for r in rows},
                      "base": {s: close_on(s, game_day) for s in symbols}})
    cards.sort(key=lambda c: c["made_on"])      # by date: the file names are not zero-padded ('2026-10-9' sorts after '2026-10-28')
    return cards


def call_of(card, model, symbol):
    """(long, scale, bin, size) of a model's call for a coin on a card; None when it made none."""
    if card is None or model is None:
        return None
    got = (card["calls"].get(model) or {}).get(symbol)
    if got is None:
        return None
    b, size = got
    long, scale = position_of(b, {"size": size} if size is not None else None)
    return long, scale, b, size


def pick_of(followed, interval):
    """
    The followed row for an interval: its point's pick; for an interval
    without a point (not settled yet, or a settlement gap) the pick of the
    next point after it - the record before both is the same, nothing
    settled in between - else the plan's next_pick.
    """
    if not followed:
        return None
    later = None
    for p in followed.get("points") or []:
        if p[0] == interval:
            return p[1]
        if p[0] > interval and later is None:
            later = p[1]
    return later if later is not None else followed.get("next_pick")


def shown_variant(day):
    """The variant the card of `day` showed: the newest ORDER_VARIANT_LOG entry dated on or before it (the first one before the log began)."""
    chosen = ORDER_VARIANT_LOG[0][1]
    for when, variant, _reason in ORDER_VARIANT_LOG:
        if when <= day:
            chosen = variant
    return chosen


def _order_row(card, symbol, model, call, variant, kind, interval_end):
    long, scale, b, _size = call
    return {"interval": card["interval"], "days": [card["interval"]], "symbol": symbol, "model": model, "bin": b, "scale": scale,
            "variant": variant, "kind": kind, "published_at": card["pub"], "estimated": card["estimated"],
            "base_close": card["base"].get(symbol), "live": card["pub"] < interval_end, "rolls": 0}


# ---------------------------------------------------------------------------
# The day book (crypto)
# ---------------------------------------------------------------------------

def day_orders(cards, day_followed, tapes, variant, end="card", roll=True, market="crypto", newcomers=None):
    """
    The day orders of one variant over `cards` (load_cards of crypto). end
    "card": the D1 deadline (the next day card, else the fallback) with the
    D5 roll; end "close": the literal 00:00 UTC end, no roll (the reference
    ledger). Returns the orders, oldest first by first day.

    `newcomers` (a list, end "card" with the roll on; 31 Oct 2026, the owner's
    decision): on every card that ROLLS a held position the reader who did
    not buy on the earlier card is told to buy now - each such showing is
    booked into this list as a NEWCOMER entry (kind "newcomer"): bought at
    that card's publication, the same new limit X, the same deadline rule
    and the same no-order tests as a fresh day order, and from then on it
    rolls exactly as the main book does (a newcomer position is one more
    holder of the same coin: on the next card it is rolled, sold or filled
    by the same test - D5 - as the position it sits next to). Never part of
    the main book or its "Following this card so far".
    """
    by_interval = {c["interval"]: c for c in cards}
    orders = []
    symbols = sorted({s for c in cards for s in c["symbols"]}, key=lambda s: min(c["symbols"].index(s) for c in cards if s in c["symbols"]))
    for symbol in symbols:
        tape = tapes.get(symbol) or Tape()
        held, blocked = None, None
        fresh = []                 # the newcomer positions held next to `held` (rolled with it)

        def settle_held(held, end_t, end_kind):
            """The held position up to `end_t`: filled, open/pending, or still unfilled (returned as 'none')."""
            found, when = scan_fill(tape, held["_scan_from"], floor_min(end_t), held["limit"] * (1.0 + FILL_THROUGH))
            if found == "fill":
                held["status"] = "closed"
                close_position(held, "limit", when, held["limit"], held["scale"])
            elif found in ("open", "pending"):
                held.update(status=found, reason=None if found == "open" else "a bar is missing in the fill scan")
            return found

        def sell_at(held, end_t, end_kind):
            state, price, m = price_at(tape, end_t)
            if state == "ok":
                held["status"] = "closed"
                close_position(held, end_kind, m, price, held["scale"])
            else:
                held.update(status=state, exit_kind=end_kind, exit_at=m, provisional=state == "open",
                            reason=None if state == "open" else "no 1m bar at the exit")

        def step(pos, card, X, test_base):
            """A held position (main or newcomer) at the card of day D: True when it rolls on (D5), else it is closed (or open/pending) here."""
            D, t = card["interval"], card["pub"]
            prev = pos["_card"]
            fb = fallback_deadline(market, prev["interval"])
            at_card = D == next_session(market, prev["interval"]) and t <= fb
            end_t = t if at_card else fb
            found = settle_held(pos, end_t, "next card")
            # 31 Oct 2026 (review): at the card that publishes it the minutes just before it are often not stored
            # yet (the refresh runs seconds before the stamp), so the scan comes back "open". The position used to
            # be booked plainly open and a NEW order placed for D next to it: the card then said "buy now" (or "no
            # order") to a reader who already held the coin and had to roll or sell. The reader decides at the
            # card on what has happened so far: roll or sell provisionally, settled for good by the next export.
            unsure = found == "open" and at_card
            if unsure:
                pos["provisional"] = True
                found = "none"
            if found == "none":
                if at_card and roll and X is not None and t < ts(close_instant(market, D)):
                    st, P, m = price_at(tape, t)
                    if st == "pending":
                        pos.update(status="pending", reason="no 1m bar at the roll")
                        found = "pending"
                    elif P is not None and skip_reason(X, P, test_base) is None:
                        pos["limit"] = X
                        pos["limits"].append(X)
                        pos["days"].append(D)
                        pos["rolls"] += 1
                        pos["_scan_from"] = m
                        pos["_card"] = card
                        pos["provisional"] = pos["provisional"] or st == "open"
                        pos["estimated"] = bool(pos.get("estimated") or card["estimated"])   # 31 Oct 2026: the roll card's time too
                        return True            # the position goes on with the new limit
                if found == "none":
                    sell_at(pos, end_t, "next card" if at_card else "fallback")
                    if at_card:
                        pos["estimated"] = bool(pos.get("estimated") or card["estimated"])   # 31 Oct 2026: the selling card's time too
                    if unsure and pos["status"] == "closed":
                        pos.update(status="open", move=None, money=None, provisional=True)   # priced, but the fill question is not settled
            return False

        for card in cards:
            D, t = card["interval"], card["pub"]
            model = pick_of(day_followed, D)
            call = call_of(card, model, symbol)
            test_base = None if variant == "fixed" else card["base"].get(symbol)      # the fixed control ignores the bin and its base
            X = None
            if call and call[0] and call[1] > 0:
                X_fixed_P = None
                if variant == "fixed":
                    st, X_fixed_P, _m = price_at(tape, t)
                X = target_price(variant, card["base"].get(symbol), call[2], card["edges"].get(symbol), X_fixed_P, "day")
            if fresh and end == "card":
                kept = []
                for pos in fresh:
                    if step(pos, card, X, test_base):
                        kept.append(pos)
                    else:
                        newcomers.append(pos)
                fresh = kept
            if held is not None and end == "card":
                if step(held, card, X, test_base):
                    if newcomers is not None:
                        # the newcomer entry of this roll card: a reader who did not buy on the earlier card buys now
                        # at P, the same limit X and deadline rule, under the same no-order tests as a fresh order
                        nrow = newcomer_entry(tape, card, symbol, model, call, variant, X, test_base, market)
                        if nrow["status"] in ("skip", "pending") or nrow["entry_price"] is None:
                            newcomers.append(nrow)
                        else:
                            nrow["_scan_from"] = nrow["entry_at"]
                            nrow["_card"] = card
                            fresh.append(nrow)
                    continue                   # the position goes on with the new limit: no new order for D
                if held["status"] == "pending":
                    blocked = D
                orders.append(held)
                held = None
            if call is None or not call[0] or call[1] <= 0:
                continue
            row = _order_row(card, symbol, model, call, variant, "main" if end == "card" else "reference", ts(close_instant(market, D)))
            if not row["live"]:
                row.update(status="not followable", reason=NOT_FOLLOWABLE, limit=X, limits=[X], entry_at=None, entry_price=None, exit_kind=None,
                           exit_at=None, exit_price=None, move=None, money=None, provisional=False)
                orders.append(row)
                continue
            if blocked is not None:
                row.update(status="pending", reason=f"an earlier {symbol} position (from {blocked}) is pending", limit=X, limits=[X],
                           entry_at=None, entry_price=None, exit_kind=None, exit_at=None, exit_price=None, move=None, money=None, provisional=False)
                orders.append(row)
                continue
            if end == "close":
                got = run_order(tape, t, X, (), (ts(close_instant(market, D)), "close"), test_base, call[1])
                row.update(got)
                row["limits"] = [X]
                orders.append(row)
                continue
            got = run_order(tape, t, X, (), None, test_base, call[1])
            row.update(got)
            row["limits"] = [X]
            if row["status"] in ("skip", "pending") or row["entry_price"] is None:
                orders.append(row)
                if row["status"] == "pending":
                    blocked = D
                continue
            row["_scan_from"] = row["entry_at"]
            row["_card"] = card
            held = row
        for pos in ([held] if held is not None else []) + fresh:
            fb = fallback_deadline(market, pos["_card"]["interval"])
            if settle_held(pos, fb, "fallback") == "none":
                sell_at(pos, fb, "fallback")
            (orders if pos is held else newcomers).append(pos)
    for o in orders + (newcomers if newcomers is not None else []):
        o.pop("_scan_from", None)
        o.pop("_card", None)
    orders.sort(key=lambda o: (o["interval"], symbols.index(o["symbol"])))
    if newcomers is not None:
        newcomers.sort(key=lambda o: (o["interval"], symbols.index(o["symbol"]) if o["symbol"] in symbols else 0))
    return orders


def newcomer_entry(tape, card, symbol, model, call, variant, X, test_base, market="crypto"):
    """
    The newcomer entry of a roll card (31 Oct 2026): bought at the card's
    publication at the 1m open, limit X, under the fresh day order's
    no-order tests (skip_reason: no target, target below fees, already
    reached, too close) - a row with money 0 when one fails. The fill scan
    and the exit are day_orders' (step), as for the position it sits next to.
    """
    row = _order_row(card, symbol, model, call, variant, "newcomer", ts(close_instant(market, card["interval"])))
    row.update(run_order(tape, card["pub"], X, (), None, test_base, call[1]))
    row["limits"] = [X]
    return row


def day_benchmark(cards, tapes, market="crypto"):
    """Every coin of every day card bought at the publication, no limit, sold at the next day card (or the fallback). [[day, coins priced, money]], and the orders."""
    by_interval = {c["interval"]: c for c in cards}
    points, orders = [], []
    for card in cards:
        D = card["interval"]
        if card["pub"] >= ts(close_instant(market, D)):
            continue
        nxt = by_interval.get(next_session(market, D))
        fb = fallback_deadline(market, D)
        deadline = (nxt["pub"], "next card") if nxt and nxt["pub"] <= fb else (fb, "fallback")
        got = [run_order(tapes.get(s) or Tape(), card["pub"], None, (), deadline, check=False) for s in card["symbols"]]
        closed = [g for g in got if g["status"] == "closed"]
        orders.extend(got)
        points.append([D, len(closed), round(sum(g["money"] for g in closed), 4), len(closed) == len(got)])
    return points, orders


# ---------------------------------------------------------------------------
# The week book (cryptoweek)
# ---------------------------------------------------------------------------

def _mark_exit_estimate(order, est_at):
    """31 Oct 2026 (review): an order sold at a card whose time is estimated is flagged estimated too (the flag used to read the entry's card only)."""
    if order.get("exit_kind") in CARD_ENDS and est_at.get(order.get("exit_at")):
        order["estimated"] = True


def week_orders(week_cards, day_cards, week_followed, day_followed, tapes, variant, end="card", market="cryptoweek"):
    """
    The week orders of one variant, the late-entry ledger and its refusals:
    (orders, late orders, late checks). end "card": the D1 deadline (next
    Monday's week card, else Monday 12:00 UTC); end "close": the literal
    Monday 00:00 UTC (the reference ledger; late entries are not booked).
    """
    base = base_market(market)
    day_by = {c["interval"]: c for c in day_cards}
    week_by = {c["interval"]: c for c in week_cards}
    orders, late, checks = [], [], []
    est_at = {ceil_min(c["pub"]): c["estimated"] for c in list(day_cards) + list(week_cards)}
    for wc in week_cards:
        W = wc["interval"]
        monday = (date.fromisoformat(W) - timedelta(days=6)).isoformat()
        days = [(date.fromisoformat(monday) + timedelta(days=j)).isoformat() for j in range(7)]
        mc = day_by.get(monday)
        t0 = max(wc["pub"], mc["pub"]) if mc else wc["pub"]
        close_t = ts(close_instant(market, W))
        if end == "close":
            deadline = (close_t, "close")
        else:
            nxt = week_by.get((date.fromisoformat(W) + timedelta(days=7)).isoformat())
            fb = fallback_deadline(market, W)
            deadline = (nxt["pub"], "next card") if nxt and nxt["pub"] <= fb else (fb, "fallback")
        model = pick_of(week_followed, W)

        def verdict(d, symbol):
            card = day_by.get(d)
            c = call_of(card, pick_of(day_followed, d), symbol)
            return None if c is None else bool(c[0] and c[1] > 0)

        for symbol in wc["symbols"]:
            call = call_of(wc, model, symbol)
            if call is None or not call[0] or call[1] <= 0:
                continue
            tape = tapes.get(symbol) or Tape()
            exits = []
            for d in days[1:]:
                card = day_by.get(d)
                if card is not None and verdict(d, symbol) is False and card["pub"] < deadline[0]:
                    exits.append((card["pub"], "early"))
            P0 = price_at(tape, t0)[1] if variant == "fixed" else None
            X = target_price(variant, wc["base"].get(symbol), call[2], wc["edges"].get(symbol), P0, "week")
            row = _order_row(wc, symbol, model, call, variant, "main" if end == "card" else "reference",
                             ts(datetime.fromisoformat(days[1]).replace(tzinfo=timezone.utc)))
            row.update(published_at=t0, estimated=wc["estimated"] or bool(mc and mc["estimated"]), days=list(days), limits=[X])
            # 31 Oct 2026 (review): followable is judged at the entry t0 (the later of the week card and Monday's day
            # card), not at the week card alone - a Monday day card that appeared after Monday ended used to leave the
            # order "live" and buy it on Tuesday as if it were Monday's entry
            row["live"] = t0 < ts(datetime.fromisoformat(days[1]).replace(tzinfo=timezone.utc))
            blank = dict(limit=X, entry_at=None, entry_price=None, exit_kind=None, exit_at=None, exit_price=None, move=None, money=None,
                         provisional=False)
            if mc is None or verdict(monday, symbol) is not True:
                row.update(status="skip", reason=SKIP_DAILY, **blank)
                row["live"] = mc is not None and row["live"]
            elif not row["live"]:
                row.update(status="not followable", reason=NOT_FOLLOWABLE, **blank)
            else:
                row.update(run_order(tape, t0, X, exits, deadline, None if variant == "fixed" else wc["base"].get(symbol), call[1]))
                _mark_exit_estimate(row, est_at)
            orders.append(row)
            if end != "card" or X is None:
                continue
            # late entries (D7): each Tuesday-Sunday day card that shows one is a cohort position of its own
            for d in days[1:]:
                card = day_by.get(d)
                if card is None:
                    continue
                t_d = card["pub"]
                check = {"interval": W, "day": d, "symbol": symbol, "variant": variant, "published_at": t_d, "estimated": card["estimated"],
                         "shown": False, "reason": None, "price": None, "gap_pct": None, "limit": X}
                if t_d >= deadline[0] or t_d >= ts(close_instant(base, d)):
                    continue                       # not a card of this week's running orders
                if verdict(d, symbol) is not True:
                    check["reason"] = LATE_DAILY
                elif deadline[0] - t_d < LATE_MIN_LEFT.total_seconds():
                    check["reason"] = LATE_SHORT
                else:
                    st, P, m = price_at(tape, t_d)
                    check["price"] = P
                    if st == "pending" or P is None:
                        check["reason"] = "no 1m bar at the card"
                    else:
                        gap = math.log(X / P)
                        check["gap_pct"] = round(100 * gap, 3)
                        if gap < max(LATE_MIN_GAP, MIN_GAP):
                            check["reason"] = LATE_FAR
                        else:
                            check["shown"] = True
                checks.append(check)
                if not check["shown"]:
                    continue
                lrow = _order_row(card, symbol, model, call, variant, "late", ts(close_instant(base, d)))
                lrow.update(interval=W, days=days[days.index(d):], limits=[X], estimated=card["estimated"])
                lrow.update(run_order(tape, t_d, X, [e for e in exits if e[0] > t_d], deadline, None, call[1], check=False))
                _mark_exit_estimate(lrow, est_at)
                late.append(lrow)
    return orders, late, checks


def week_benchmark(week_cards, day_cards, tapes, market="cryptoweek"):
    """Every coin of every week card bought at the week's entry time (the later of the week card and Monday's day card), sold at the next week card (or the fallback)."""
    day_by = {c["interval"]: c for c in day_cards}
    week_by = {c["interval"]: c for c in week_cards}
    points, orders = [], []
    for wc in week_cards:
        W = wc["interval"]
        monday = (date.fromisoformat(W) - timedelta(days=6)).isoformat()
        mc = day_by.get(monday)
        t0 = max(wc["pub"], mc["pub"]) if mc else wc["pub"]
        nxt = week_by.get((date.fromisoformat(W) + timedelta(days=7)).isoformat())
        fb = fallback_deadline(market, W)
        deadline = (nxt["pub"], "next card") if nxt and nxt["pub"] <= fb else (fb, "fallback")
        got = [run_order(tapes.get(s) or Tape(), t0, None, (), deadline, check=False) for s in wc["symbols"]]
        closed = [g for g in got if g["status"] == "closed"]
        orders.extend(got)
        points.append([W, len(closed), round(sum(g["money"] for g in closed), 4), len(closed) == len(got)])
    return points, orders


# ---------------------------------------------------------------------------
# Stats and the export block
# ---------------------------------------------------------------------------

def order_stats(orders):
    """STATS of a list of orders (see the module docstring)."""
    positions = [o for o in orders if o.get("status") in POSITION_STATES]
    closed = [o for o in positions if o["status"] == "closed"]
    skips = {}
    for o in orders:
        if o.get("status") == "skip":
            skips[o["reason"]] = skips.get(o["reason"], 0) + 1
    return {"orders": len(positions), "closed": len(closed), "open": sum(o["status"] == "open" for o in positions),
            "pending": sum(o["status"] == "pending" for o in positions),
            "filled": sum(o["exit_kind"] == "limit" for o in closed), "early": sum(o["exit_kind"] == "early" for o in closed),
            "time_stop": sum(o["exit_kind"] in ("next card", "fallback", "close") for o in closed),
            "skips": skips, "not_followable": sum(o.get("status") == "not followable" for o in orders),
            "total": round(sum(o["money"] for o in closed), 4), "wins": sum(o["money"] > 0 for o in closed),
            "estimated": sum(bool(o.get("estimated")) for o in closed), "rolls": sum(o.get("rolls", 0) for o in positions)}


def _export_order(o):
    """An order with its instants as ISO strings and its numbers rounded for the page."""
    out = dict(o)
    for key in ("published_at", "entry_at", "exit_at"):
        out[key] = iso_ts(o.get(key))
    for key in ("entry_price", "exit_price", "limit", "base_close"):
        if out.get(key) is not None:
            out[key] = round(float(out[key]), 8)
    out["limits"] = [None if x is None else round(float(x), 8) for x in o.get("limits") or []]
    out["move"] = None if o.get("move") is None else round(o["move"], 6)
    out["money"] = None if o.get("money") is None else round(o["money"], 4)
    out["limit_pct"] = (round(100 * math.log(o["limit"] / o["entry_price"]), 3)
                        if o.get("limit") and o.get("entry_price") else None)
    out["estimated"] = bool(o.get("estimated"))
    out["provisional"] = bool(o.get("provisional"))
    return out


def compute_books(conn, path, market, followed, day_followed=None, lag=PUBLISH_LAG_FALLBACK, now=None, since=ORDER_SINCE):
    """
    Every book of one order market, unrounded (times in epoch seconds):
    {cards, day_cards, books, reference, late, checks, benchmark, tapes}.
    `followed` is the market's own followed_plan; a week game also needs
    the base market's (`day_followed`).
    """
    now_t = ts(now or datetime.now(timezone.utc))
    week = is_week_game(market)
    day_market = base_market(market)
    day_cards = load_cards(conn, path, day_market, since=(date.fromisoformat(since) - timedelta(days=8)).isoformat() if week else since, lag=lag)
    week_cards = load_cards(conn, path, market, since=since, lag=lag) if week else []
    first = min([c["pub"] for c in day_cards + week_cards], default=now_t) - 2 * 3600
    tapes = load_tapes(conn, market, first, now_t)
    out = {"day_cards": day_cards, "week_cards": week_cards, "tapes": tapes, "books": {}, "reference": {}, "late": {}, "checks": {}, "newcomer": {}}
    if week:
        for v in ORDER_VARIANTS:
            out["books"][v], out["late"][v], out["checks"][v] = week_orders(week_cards, day_cards, followed, day_followed, tapes, v, "card", market)
            out["reference"][v] = week_orders(week_cards, day_cards, followed, day_followed, tapes, v, "close", market)[0]
        out["benchmark"] = week_benchmark(week_cards, day_cards, tapes, market)
    else:
        for v in ORDER_VARIANTS:
            out["newcomer"][v] = []
            out["books"][v] = day_orders(day_cards, followed, tapes, v, "card", True, market, newcomers=out["newcomer"][v])
            out["reference"][v] = day_orders(day_cards, followed, tapes, v, "close", False, market)
        out["benchmark"] = day_benchmark(day_cards, tapes, market)
    return out


def _day_now(books, cards, followed, market, newcomer_books=None):
    """
    NOW for the day card: what the newest card told a reader, coin by coin,
    from the primary variant's book. On a roll ("hold (rolled)") `newcomer`
    is the newcomer entry of this card (the reader who did not buy on the
    earlier card) and `newcomers_held` the newcomer entries of earlier roll
    cards still held into this card or sold at it: [[their first day, entry
    price]].
    """
    if not cards:
        return None
    card = cards[-1]
    D = card["interval"]
    variant = shown_variant(D)
    orders = books[variant]
    fresh = (newcomer_books or {}).get(variant) or []
    model = pick_of(followed, D)
    fb = fallback_deadline(market, D)
    symbols = {}
    for symbol in card["symbols"]:
        call = call_of(card, model, symbol)
        word = None if call is None else call_word(call[2], call[3])
        entry = {"status": "no order", "call": word, "bin": None if call is None else call[2], "limit": None, "entry_price": None,
                 "limit_pct": None, "reason": None if call and call[0] else f"call {word or 'missing'}", "provisional": False,
                 "held_since": None, "state": None}
        sold = next((o for o in orders if o["symbol"] == symbol and D not in o["days"] and o.get("exit_at") == ceil_min(card["pub"])
                     and o.get("exit_kind") == "next card"), None)
        mine = next((o for o in orders if o["symbol"] == symbol and D in o["days"]), None)
        if mine is not None:
            rolled = mine["days"][0] != D
            entry.update(status="hold (rolled)" if rolled else ("buy now" if mine["status"] in POSITION_STATES else "no order"),
                         limit=mine.get("limit"), entry_price=mine.get("entry_price"), reason=mine.get("reason") if mine["status"] in ("skip", "not followable") else None,
                         provisional=bool(mine.get("provisional")), held_since=mine["days"][0], state=mine["status"])
            if mine.get("limit") and mine.get("entry_price"):
                entry["limit_pct"] = round(100 * math.log(mine["limit"] / mine["entry_price"]), 3)
        if sold is not None and (mine is None or mine["status"] not in POSITION_STATES):
            entry["status"] = "sell now"
            entry["held_since"] = sold["days"][0]
        entry["newcomer"] = None
        entry["newcomers_held"] = []
        if entry["status"] in ("hold (rolled)", "sell now"):
            pub_m = ceil_min(card["pub"])
            entry["newcomers_held"] = [[o["days"][0], None if o.get("entry_price") is None else round(o["entry_price"], 8)] for o in fresh
                                       if o["symbol"] == symbol and o["days"][0] != D and o["status"] != "skip"
                                       and (D in o["days"] or (o.get("exit_at") == pub_m and o.get("exit_kind") == "next card"))]
        if entry["status"] == "hold (rolled)":
            nc = next((o for o in fresh if o["symbol"] == symbol and o["days"][0] == D), None)
            if nc is not None:
                entry["newcomer"] = {"status": "buy now" if nc["status"] in POSITION_STATES else "no order", "state": nc["status"],
                                     "limit": nc.get("limit"), "entry_price": nc.get("entry_price"), "reason": nc.get("reason") if nc["status"] == "skip" else None,
                                     "limit_pct": round(100 * math.log(nc["limit"] / nc["entry_price"]), 3) if nc.get("limit") and nc.get("entry_price") else None,
                                     "provisional": bool(nc.get("provisional"))}
        symbols[symbol] = entry
    return {"interval": D, "made_on": card["made_on"], "published_at": iso_ts(card["pub"]), "estimated": card["estimated"], "model": model,
            "variant": variant, "deadline_fallback": iso_ts(fb), "deadline_text": brussels_text(fb), "symbols": symbols}


def _week_now(books, checks, week_cards, day_cards, followed, day_followed, market):
    """NOW for the week card: the newest week's orders and the running day's late-entry lines, from the primary variant."""
    if not week_cards:
        return None
    wc = week_cards[-1]
    W = wc["interval"]
    variant = shown_variant(W)
    model = pick_of(followed, W)
    newest_day = day_cards[-1] if day_cards else None
    monday = (date.fromisoformat(W) - timedelta(days=6)).isoformat()
    running = newest_day["interval"] if newest_day and monday <= newest_day["interval"] <= W else None
    fb = fallback_deadline(market, W)
    symbols = {}
    for symbol in wc["symbols"]:
        call = call_of(wc, model, symbol)
        word = None if call is None else call_word(call[2], call[3])
        entry = {"status": "no order", "call": word, "bin": None if call is None else call[2], "limit": None, "entry_price": None,
                 "limit_pct": None, "reason": None if call and call[0] else f"week call {word or 'missing'}", "state": None,
                 "exit_kind": None, "exit_at": None, "money": None}
        mine = next((o for o in books[variant] if o["symbol"] == symbol and o["interval"] == W), None)
        if mine is not None:
            status = {"skip": "no order", "not followable": "no order"}.get(mine["status"], "buy now")
            if mine["status"] == "skip" and mine["reason"] == SKIP_DAILY:
                status = "not bought"
            if (newest_day and mine.get("exit_kind") == "early" and mine.get("exit_at") == ceil_min(newest_day["pub"])):
                status = "sell now"
            entry.update(status=status, limit=mine.get("limit"), entry_price=mine.get("entry_price"), state=mine["status"],
                         reason=mine.get("reason"), exit_kind=mine.get("exit_kind"), exit_at=iso_ts(mine.get("exit_at")),
                         money=None if mine.get("money") is None else round(mine["money"], 4))
            if mine.get("limit") and mine.get("entry_price"):
                entry["limit_pct"] = round(100 * math.log(mine["limit"] / mine["entry_price"]), 3)
        symbols[symbol] = entry
    late = {}
    if running and running != monday:
        for c in checks[variant]:
            if c["interval"] == W and c["day"] == running:
                late[c["symbol"]] = {"shown": c["shown"], "reason": c["reason"], "price": c["price"], "gap_pct": c["gap_pct"],
                                     "limit": None if c["limit"] is None else round(c["limit"], 8)}
    return {"interval": W, "first_day": monday, "made_on": wc["made_on"], "published_at": iso_ts(wc["pub"]), "estimated": wc["estimated"],
            "model": model, "variant": variant, "deadline_fallback": iso_ts(fb), "deadline_text": brussels_text(fb),
            "running_day": running, "day_model": pick_of(day_followed, running) if running else None, "symbols": symbols, "late": late}


def orders_record(conn, path, market, followed=None, day_followed=None, lag=PUBLISH_LAG_FALLBACK, now=None, since=ORDER_SINCE):
    """
    trading["orders"] of a crypto order market (shape in the module
    docstring), or None for any other market. `followed` (the market's own
    followed_plan) and, for the week, `day_followed` (crypto's) are computed
    when not given.
    """
    if market not in ORDER_MARKETS:
        return None
    conn.executescript(GAME_SCHEMA)
    if followed is None:
        followed = followed_plan(conn, market, ledger(conn, market), path=path)
    week = is_week_game(market)
    if week and day_followed is None:
        base = base_market(market)
        day_followed = followed_plan(conn, base, ledger(conn, base), path=path)
    got = compute_books(conn, path, market, followed, day_followed, lag=lag, now=now, since=since)
    cards = got["week_cards"] if week else got["day_cards"]
    now_block = (_week_now(got["books"], got["checks"], got["week_cards"], got["day_cards"], followed, day_followed, market) if week
                 else _day_now(got["books"], got["day_cards"], followed, market, got["newcomer"]))
    shown = [[c["interval"], shown_variant(c["interval"])] for c in cards]
    record_orders = [o for interval, v in shown for o in got["books"][v] if o["interval"] == interval]
    bench_points, bench_orders = got["benchmark"]
    coverage = {}
    for symbol, tape in got["tapes"].items():
        hours = sorted(tape.h1)
        coverage[symbol] = {"1h_first": iso_ts(hours[0]) if hours else None, "1h_last": iso_ts(hours[-1]) if hours else None,
                            "1h_bars": len(hours), "1m_bars": len(tape.m1)}
    out = {
        "market": market, "game": "week" if week else "day", "generated_at": iso(datetime.now(timezone.utc)), "since": since,
        "rule": ORDER_RULE_WEEK if week else ORDER_RULE_DAY,
        "constants": {"fee_per_leg": FEE_PER_LEG, "min_gap": MIN_GAP, "fill_through": FILL_THROUGH,
                      "publish_lag_fallback_min": PUBLISH_LAG_FALLBACK.total_seconds() / 60, "deadline_fallback_hours": DEADLINE_FALLBACK_HOURS,
                      "late_min_gap": LATE_MIN_GAP, "late_min_left_hours": LATE_MIN_LEFT.total_seconds() / 3600, "stake": STAKE,
                      "fixed_gap": FIXED_GAP["week" if week else "day"]},
        "primary": ORDER_VARIANT_LOG[-1][1], "variants": dict(ORDER_VARIANT_TEXT), "variant_log": [list(e) for e in ORDER_VARIANT_LOG],
        "review": REVIEW_RULE,
        "books": {v: {"orders": [_export_order(o) for o in got["books"][v]], "stats": order_stats(got["books"][v])} for v in ORDER_VARIANTS},
        "reference": {v: {"orders": [_export_order(o) for o in got["reference"][v]], "stats": order_stats(got["reference"][v])} for v in ORDER_VARIANTS},
        "benchmark": {"points": bench_points, "stats": order_stats(bench_orders)},
        "record": {"shown": shown, "stats": order_stats(record_orders)},
        "publications": [[c["made_on"], c["interval"], iso_ts(c["pub"]), c["estimated"]] for c in cards],
        "coverage": coverage,
        "now": now_block,
    }
    if not week:
        # the newcomer ledger (31 Oct 2026): per variant, its own record, never in "record" (the user line) above
        out["newcomer_rule"] = NEWCOMER_RULE
        out["newcomer"] = {v: {"orders": [_export_order(o) for o in got["newcomer"][v]], "stats": order_stats(got["newcomer"][v])} for v in ORDER_VARIANTS}
    if week:
        out["late_rule"] = LATE_RULE
        out["late"] = {}
        for v in ORDER_VARIANTS:
            refused = {}
            for c in got["checks"][v]:
                if not c["shown"]:
                    refused[c["reason"]] = refused.get(c["reason"], 0) + 1
            out["late"][v] = {"orders": [_export_order(o) for o in got["late"][v]], "stats": order_stats(got["late"][v]), "refused": refused}
    return out


# ---------------------------------------------------------------------------
# Fetching (export time and the backfill)
# ---------------------------------------------------------------------------

def minute_windows(instants):
    """The 1m windows the book needs around card instants: [the instant's hour, + 2 h) - the entry's partial hour, the exit's own partial hour and the card minute - merged where they overlap."""
    spans = sorted((floor_hour(t), floor_hour(t) + 2 * 3600) for t in instants)
    merged = []
    for a, z in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], z)
        else:
            merged.append([a, z])
    return [(at(a), at(z)) for a, z in merged]


def card_instants(conn, path, since, lags=(PUBLISH_LAG_FALLBACK,)):
    """Every crypto and cryptoweek card publication on or after `since` (an epoch second), estimated ones at each lag in `lags`."""
    out = set()
    for market in ORDER_MARKETS:
        for lag in lags:
            for made_on, (pub, _est) in publications(conn, path, market, lag).items():
                if pub >= since:
                    out.add(pub)
    return sorted(out)


def refresh_intraday(conn, path, now=None, since=None, lags=(PUBLISH_LAG_FALLBACK,), log=print):
    """
    The intraday step of an export (MarketSettle.settle_and_export, fetch on)
    or of a backfill (`since` given): 1h bars up to now, then the 1m windows
    of every card publication since `since` (else INTRADAY_LOOKBACK back) and
    of the export moment itself (the provisional quote of the card being
    published). Never raises.
    """
    try:
        now = now or datetime.now(timezone.utc)
        floor = since if since else now - INTRADAY_LOOKBACK
        floor_t = ts(floor if isinstance(floor, datetime) else datetime.fromisoformat(str(floor)).replace(tzinfo=timezone.utc))
        instants = card_instants(conn, path, floor_t, lags) + [ts(now) - 3600, ts(now)]
        windows = [(a, z) for a, z in minute_windows(instants) if a < now]
        return update_intraday(conn, "crypto", windows=windows, now=now, since=since if since else None, log=log)
    except Exception as exc:          # update_intraday never raises; this guards the publications and the window arithmetic too
        log(f"intraday: FAILED - {type(exc).__name__}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Self-check (no network)
# ---------------------------------------------------------------------------

def _self_check():
    U = lambda *a: ts(datetime(*a, tzinfo=timezone.utc))
    base = U(2026, 10, 20, 10, 0)

    def tape(now=None, minutes=(), hours=(), flat=100.0, m_from=None, m_to=None, h_from=None, h_to=None, m2=None):
        """A flat tape at `flat` with 1m bars over [m_from, m_to) (and the span m2) and 1h bars over [h_from, h_to); `minutes`/`hours` override (start, open, high)."""
        t = Tape(now=now or base + 10 * 86400)
        if m_from is not None:
            for s in list(range(m_from, m_to, 60)) + (list(range(*m2, 60)) if m2 else []):
                t.m1[s] = (flat, flat, flat, flat)
        if h_from is not None:
            for s in range(h_from, h_to, 3600):
                t.h1[s] = (flat, flat, flat, flat)
        for s, o, h in minutes:
            t.m1[s] = (o, h, min(o, h), o)
        for s, o, h in hours:
            t.h1[s] = (o, h, min(o, h), o)
        return t

    t0 = base + 20 * 60 + 24                         # a card at 10:20:24 -> entry at the 10:21 open
    X = 101.0
    deadline = (base + 86400 + 15 * 60, "next card")    # tomorrow 10:15
    # the 1m windows of the two cards (the entry's and the deadline's hour, + 2 h), 1h bars throughout
    full = dict(m_from=base, m_to=base + 2 * 3600, h_from=base, h_to=base + 3 * 86400, m2=(base + 86400, base + 86400 + 2 * 3600))

    # 1. a fill in the entry's partial hour (1m), at X
    o = run_order(tape(minutes=[(base + 40 * 60, 100.0, 101.06)], **full), t0, X, (), deadline, base_close=99.0)
    assert o["status"] == "closed" and o["exit_kind"] == "limit" and o["exit_at"] == base + 40 * 60 and o["exit_price"] == X, o
    assert o["entry_at"] == base + 21 * 60 and o["entry_price"] == 100.0
    assert abs(o["money"] - cash_pnl(math.log(X / 100.0))) < 1e-12
    # 2. a fill only after the first full hour (a 1h bar)
    o = run_order(tape(hours=[(base + 3 * 3600, 100.0, 101.2)], **full), t0, X, (), deadline, base_close=99.0)
    assert o["status"] == "closed" and o["exit_at"] == base + 3 * 3600, o
    # 3. a touch below FILL_THROUGH is no fill: sold at the deadline at its minute's open
    t3 = tape(hours=[(base + 3 * 3600, 100.0, 101.04)], minutes=[(deadline[0], 100.5, 100.5)], **full)
    o = run_order(t3, t0, X, (), deadline, base_close=99.0)
    assert o["status"] == "closed" and o["exit_kind"] == "next card" and o["exit_price"] == 100.5 and o["exit_at"] == deadline[0], o
    # 4. a gap open above X fills at X, never at the open
    o = run_order(tape(minutes=[(base + 30 * 60, 102.0, 102.5)], **full), t0, X, (), deadline, base_close=99.0)
    assert o["exit_kind"] == "limit" and o["exit_price"] == X
    print("orders: fill in the entry's partial hour and after the first full hour, a touch is no fill, a gap open fills at X")

    # 5. no order: P >= X, target below fees from the base close, too close to the limit
    assert run_order(tape(**full, flat=101.5), t0, X, (), deadline, base_close=99.0)["reason"] == SKIP_REACHED
    assert run_order(tape(**full), t0, 100.2, (), deadline, base_close=100.1)["reason"] == SKIP_TARGET
    assert run_order(tape(**full), t0, 100.2, (), deadline, base_close=99.0)["reason"] == SKIP_CLOSE
    assert skip_reason(None, 100, 99) == SKIP_NO_TARGET
    print("orders: no order when the price is at the limit, the call aims below the fees, or the limit is within 0.30 %")

    # 6. the deadline at the next card; 7. the fallback when no card appears (12:00 UTC on the day after the close)
    fb = fallback_deadline("crypto", "2026-10-20")
    assert at(fb) == datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    o = run_order(tape(**full), t0, X, (), (fb, "fallback"), base_close=99.0)
    assert o["status"] == "closed" and o["exit_kind"] == "fallback" and o["exit_at"] == fb and o["exit_price"] == 100.0, o     # priced from the exact-hour open
    nominute = tape(**full)
    del nominute.m1[deadline[0]]
    o = run_order(nominute, t0, X, (), deadline, base_close=99.0)
    assert o["status"] == "pending" and o["reason"] == "no 1m bar at the exit", o      # 10:15 is not an exact hour: needs the minute
    print("orders: the deadline at the next card's minute, the 12:00 UTC fallback from the hour's open, a missing exit minute is pending")

    # 9. an early exit and a fill in the same hour, ordered by minute; 10. a same-minute tie: the card wins
    ex = base + 5 * 3600 + 30 * 60                         # a card at 15:30
    hourly = dict(m_from=base, m_to=base + 2 * 3600, h_from=base, h_to=base + 3 * 86400)
    minutes = [(s, 100.0, 100.0) for s in range(base + 5 * 3600, base + 6 * 3600, 60)]
    fill_first = tape(minutes=minutes + [(base + 5 * 3600 + 10 * 60, 100.0, 101.1)], **hourly)
    fill_first.h1[base + 5 * 3600] = (100.0, 101.1, 100.0, 100.0)
    o = run_order(fill_first, t0, X, [(ex, "early")], deadline, base_close=99.0)
    assert o["exit_kind"] == "limit" and o["exit_at"] == base + 5 * 3600 + 10 * 60, o
    exit_first = tape(minutes=minutes + [(base + 5 * 3600 + 40 * 60, 100.0, 101.1)], **hourly)
    exit_first.h1[base + 5 * 3600] = (100.0, 101.1, 100.0, 100.0)
    o = run_order(exit_first, t0, X, [(ex, "early")], deadline, base_close=99.0)
    assert o["exit_kind"] == "early" and o["exit_at"] == ex and o["exit_price"] == 100.0, o
    tie = tape(minutes=minutes + [(ex, 100.0, 101.1)], **hourly)
    o = run_order(tie, t0, X, [(ex, "early")], deadline, base_close=99.0)
    assert o["exit_kind"] == "early", o
    print("orders: a fill before a card in the same hour fills, a fill after it does not, a fill in the card's own minute loses to the card")

    # 13. a missing 1m bar at the entry, and a missing 1h bar in the scan: pending, never guessed
    gap = tape(**full)
    del gap.m1[base + 21 * 60]
    assert run_order(gap, t0, X, (), deadline, base_close=99.0)["status"] == "pending"
    hole = tape(**full)
    del hole.h1[base + 5 * 3600]
    o = run_order(hole, t0, X, (), deadline, base_close=99.0)
    assert o["status"] == "pending" and o["money"] is None
    # an order whose deadline is after `now`: open, nothing booked; a card just published: decided on the last closed minute
    live = tape(now=base + 3 * 3600 + 1800, m_from=base, m_to=base + 2 * 3600, h_from=base, h_to=base + 3 * 3600)
    assert run_order(live, t0, X, (), deadline, base_close=99.0)["status"] == "open"
    fresh = tape(now=t0 + 10, m_from=base, m_to=base + 20 * 60, flat=100.0)
    o = run_order(fresh, t0, X, (), deadline, base_close=99.0)
    assert o["status"] == "open" and o["provisional"] and o["entry_price"] == 100.0, o
    print("orders: a missing minute or hour is pending; a running order is open; a card being published is decided on the last closed minute, provisionally")

    # 12. the day book: a roll keeps one fee pair over two days, a roll refused under MIN_GAP sells at the card
    edges = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.005, 0.01, 0.02, 0.03]      # bin 6 = [0.5 %, 1 %)
    def card(day, pub, b, made_on=None, model="M", base_close=100.0, sym="BTC"):
        return {"made_on": made_on or (date.fromisoformat(day) - timedelta(days=1)).isoformat(), "interval": day, "pub": pub, "estimated": False,
                "calls": {model: {sym: (b, None)}}, "symbols": [sym], "edges": {sym: edges}, "base": {sym: base_close}}
    followed = {"points": [], "next_pick": "M"}
    d1, d2 = U(2026, 10, 20, 10, 0), U(2026, 10, 21, 10, 0)
    flat = Tape(now=U(2026, 10, 25, 0, 0))
    for s in range(U(2026, 10, 19, 0, 0), U(2026, 10, 24, 0, 0), 60):
        flat.m1[s] = (100.0, 100.0, 100.0, 100.0)
    for s in range(U(2026, 10, 19, 0, 0), U(2026, 10, 24, 0, 0), 3600):
        flat.h1[s] = (100.0, 100.0, 100.0, 100.0)
    # day 20: bin 6 from 99.85 -> X = 99.85 e^0.005 = 100.35, 0.35 % above the price 100 -> placed; day 21 bin 7 from 100 -> X' 101.01: rolled
    cards = [card("2026-10-20", d1, 6, base_close=99.85), card("2026-10-21", d2, 7, base_close=100.0)]
    book = day_orders(cards, followed, {"BTC": flat}, "edge")
    assert len(book) == 1 and book[0]["days"] == ["2026-10-20", "2026-10-21"] and book[0]["rolls"] == 1, book
    assert book[0]["exit_kind"] == "fallback" and book[0]["limits"][1] == 100.0 * math.exp(0.01), book[0]
    assert abs(book[0]["money"] - cash_pnl(0.0)) < 1e-12, "one fee pair over two days"
    refused = [card("2026-10-20", d1, 6, base_close=99.85), card("2026-10-21", d2, 6, base_close=99.6)]     # X' = 100.10: 0.1 % above
    book = day_orders(refused, followed, {"BTC": flat}, "edge")
    assert [o["status"] for o in book] == ["closed", "skip"] and book[0]["exit_kind"] == "next card" and book[0]["exit_at"] == d2, book
    assert book[1]["reason"] == SKIP_CLOSE
    # the literal 00:00 UTC reference: no roll, each day sold at its close
    ref = day_orders(cards, followed, {"BTC": flat}, "edge", end="close", roll=False)
    assert [o["exit_kind"] for o in ref] == ["close", "close"] and ref[0]["exit_at"] == U(2026, 10, 21, 0, 0), ref
    # a card published after its day ended is not followable; a flat call is no order at all
    late_card = [card("2026-10-20", U(2026, 10, 21, 0, 30), 6, base_close=99.85)]
    assert day_orders(late_card, followed, {"BTC": flat}, "edge")[0]["status"] == "not followable"
    assert day_orders([card("2026-10-20", d1, 4)], followed, {"BTC": flat}, "edge") == []
    print("day book: a roll keeps one fee pair, a refused roll sells at the next card, the 00:00 reference never rolls, a late card is not followable")

    # the newcomer ledger (31 Oct 2026): a roll card books a newcomer entry for the reader who did not buy on the earlier card
    fresh = []
    book = day_orders(cards, followed, {"BTC": flat}, "edge", newcomers=fresh)
    assert len(book) == 1 and book[0]["rolls"] == 1 and all(o["kind"] == "main" for o in book), book
    assert len(fresh) == 1 and fresh[0]["kind"] == "newcomer" and fresh[0]["days"] == ["2026-10-21"] and fresh[0]["entry_at"] == d2, fresh
    assert fresh[0]["limit"] == 100.0 * math.exp(0.01) and fresh[0]["status"] == "closed" and fresh[0]["exit_kind"] == "fallback", fresh
    assert abs(fresh[0]["money"] - cash_pnl(0.0)) < 1e-12 and fresh[0]["rolls"] == 0, "its own fee pair, from the roll card's price"
    #     never in the main book's stats: the main line counts one order, the newcomer its own
    assert order_stats(book)["orders"] == 1 and order_stats(book)["total"] == round(book[0]["money"], 4) and order_stats(fresh)["orders"] == 1
    #     no roll (refused under MIN_GAP: the card sells), no newcomer; the reference ledger never books one
    fresh_r = []
    day_orders(refused, followed, {"BTC": flat}, "edge", newcomers=fresh_r)
    assert fresh_r == [], fresh_r
    now_r = _day_now({"edge": day_orders(refused, followed, {"BTC": flat}, "edge")}, refused, followed, "crypto", {"edge": []})
    assert now_r["symbols"]["BTC"]["status"] == "sell now" and now_r["symbols"]["BTC"]["newcomer"] is None
    fresh_c = []
    day_orders(cards, followed, {"BTC": flat}, "edge", end="close", roll=False, newcomers=fresh_c)
    assert fresh_c == []
    #     the fresh-order tests at the roll card's price: skipped under MIN_GAP and when already reached (in the book the
    #     roll test is the same test at the same price, so a refused roll sells instead - these guard the rule itself)
    nc_card = card("2026-10-21", d2, 7, base_close=100.0)
    nc_call = call_of(nc_card, "M", "BTC")
    assert newcomer_entry(flat, nc_card, "BTC", "M", nc_call, "edge", 100.2, 99.0)["reason"] == SKIP_CLOSE
    assert newcomer_entry(flat, nc_card, "BTC", "M", nc_call, "edge", 100.2, 100.1)["reason"] == SKIP_TARGET
    got = newcomer_entry(flat, nc_card, "BTC", "M", nc_call, "edge", 99.9, 99.0)
    assert got["status"] == "skip" and got["reason"] == SKIP_REACHED and got["kind"] == "newcomer" and got["money"] is None
    #     a second consecutive roll day: the main position and yesterday's newcomer roll again, and today's newcomer is booked
    d3 = U(2026, 10, 22, 10, 0)
    three = cards + [card("2026-10-22", d3, 8, base_close=100.0)]       # bin 8 = [2 %, 3 %): X'' = 100 e^0.02, 2 % above 100
    fresh3 = []
    book3 = day_orders(three, followed, {"BTC": flat}, "edge", newcomers=fresh3)
    assert len(book3) == 1 and book3[0]["days"] == ["2026-10-20", "2026-10-21", "2026-10-22"] and book3[0]["rolls"] == 2, book3
    assert [(o["days"], o["rolls"]) for o in fresh3] == [(["2026-10-21", "2026-10-22"], 1), (["2026-10-22"], 0)], fresh3
    assert fresh3[0]["limits"] == [100.0 * math.exp(0.01), 100.0 * math.exp(0.02)] and all(o["exit_kind"] == "fallback" for o in fresh3)
    n3 = _day_now({"edge": book3}, three, followed, "crypto", {"edge": fresh3})["symbols"]["BTC"]
    assert n3["status"] == "hold (rolled)" and n3["newcomer"]["status"] == "buy now" and n3["newcomer"]["entry_price"] == 100.0
    assert n3["newcomers_held"] == [["2026-10-21", 100.0]] and n3["held_since"] == "2026-10-20", n3
    #     a newcomer filled with the position it sits next to (the same limit, scanned from the same minute)
    up = Tape(now=flat.now, m1=dict(flat.m1), h1=dict(flat.h1))
    up.h1[U(2026, 10, 21, 15, 0)] = (100.0, 101.2, 100.0, 100.0)
    fresh_f = []
    book_f = day_orders(cards, followed, {"BTC": up}, "edge", newcomers=fresh_f)
    assert book_f[0]["exit_kind"] == "limit" and fresh_f[0]["exit_kind"] == "limit" and book_f[0]["exit_at"] == fresh_f[0]["exit_at"] == U(2026, 10, 21, 15, 0)
    print("newcomer ledger: a roll card books a newcomer entry (its own fee pair, never in the main stats), none without a roll; "
          "skipped under the fresh-order tests; rolled again on a second roll day next to a new one; filled with the position")

    # 8. the week: an early exit on the third day card; 11. a day without a card keeps the position;
    # 14. late entries shown and booked as cohorts, refused under LATE_MIN_GAP or with less than 24 h left
    W = "2026-10-25"
    monday = date(2026, 10, 19)
    days = [(monday + timedelta(days=j)).isoformat() for j in range(7)]
    wk_tape = Tape(now=U(2026, 11, 1, 0, 0))
    price_path = {}
    for s in range(U(2026, 10, 19, 0, 0), U(2026, 10, 27, 0, 0), 60):
        p = 100.0 if s < U(2026, 10, 20, 0, 0) else 97.0      # Tuesday on: 3 % lower
        wk_tape.m1[s] = (p, p, p, p)
    for s in range(U(2026, 10, 19, 0, 0), U(2026, 10, 27, 0, 0), 3600):
        p = 100.0 if s < U(2026, 10, 20, 0, 0) else 97.0
        wk_tape.h1[s] = (p, p, p, p)
    wedges = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.01, 0.02, 0.04, 0.06]     # bin 7 = [2 %, 4 %)
    wcards = [{"made_on": "2026-10-18", "interval": W, "pub": U(2026, 10, 19, 8, 50), "estimated": True, "calls": {"WK": {"ETH": (7, None)}},
               "symbols": ["ETH"], "edges": {"ETH": wedges}, "base": {"ETH": 99.0}},
              {"made_on": W, "interval": "2026-11-01", "pub": U(2026, 10, 26, 8, 30), "estimated": False, "calls": {"WK": {"ETH": (4, None)}},
               "symbols": ["ETH"], "edges": {"ETH": wedges}, "base": {"ETH": 97.0}}]
    # daily calls: Mon up, Tue up, Wed (no card), Thu down (the third day card), Fri up, Sat up, Sun up
    calls = {"2026-10-19": 8, "2026-10-20": 8, "2026-10-22": 2, "2026-10-23": 8, "2026-10-24": 8, "2026-10-25": 8}
    dcards = [{"made_on": (date.fromisoformat(d) - timedelta(days=1)).isoformat(), "interval": d,
               "pub": ts(datetime.fromisoformat(d).replace(tzinfo=timezone.utc) + timedelta(hours=8, minutes=45)), "estimated": False,
               "calls": {"D": {"ETH": (b, None)}}, "symbols": ["ETH"], "edges": {"ETH": edges}, "base": {"ETH": 100.0}}
              for d, b in calls.items()]
    wf = {"points": [], "next_pick": "WK"}
    df = {"points": [], "next_pick": "D"}
    orders, late, checks = week_orders(wcards, dcards, wf, df, {"ETH": wk_tape}, "edge")
    main = [o for o in orders if o["interval"] == W]
    assert len(main) == 1 and main[0]["status"] == "closed" and main[0]["exit_kind"] == "early", main
    assert main[0]["exit_at"] == ceil_min(U(2026, 10, 22, 8, 45)), "sold at Thursday's card - Wednesday had none and kept it"
    assert main[0]["entry_at"] == U(2026, 10, 19, 8, 50) and main[0]["limit"] == 99.0 * math.exp(0.02)
    shown = [c for c in checks if c["shown"]]
    assert [c["day"] for c in shown] == ["2026-10-20", "2026-10-23", "2026-10-24"], checks       # 3 % below X (100.99 vs 97)
    assert {c["day"]: c["reason"] for c in checks if not c["shown"]} == {"2026-10-22": LATE_DAILY, "2026-10-25": LATE_SHORT}, checks
    assert len(late) == 3 and late[0]["exit_kind"] == "early" and late[1]["exit_kind"] == "next card" and late[1]["days"][0] == "2026-10-23", late
    assert late[1]["exit_at"] == U(2026, 10, 26, 8, 30), "the deadline is next Monday's week card (Sunday's 08:45 card is 23.75 h before it)"
    # refused under LATE_MIN_GAP: the same week with Tuesday on at 100.5 (X 100.99 is 0.5 % away)
    near = Tape(now=wk_tape.now, m1={s: ((100.5,) * 4 if s >= U(2026, 10, 20, 0, 0) else v) for s, v in wk_tape.m1.items()},
                h1={s: ((100.5,) * 4 if s >= U(2026, 10, 20, 0, 0) else v) for s, v in wk_tape.h1.items()})
    _o, late2, checks2 = week_orders(wcards, dcards, wf, df, {"ETH": near}, "edge")
    assert late2 == [] and {c["reason"] for c in checks2} == {LATE_FAR, LATE_DAILY, LATE_SHORT}, checks2
    # D4: Monday's daily call not up -> no Monday buy (a counted skip), the late path from Tuesday
    no_monday = [dict(c, calls={"D": {"ETH": (2, None)}}) if c["interval"] == "2026-10-19" else c for c in dcards]
    o3, late3, _c = week_orders(wcards, no_monday, wf, df, {"ETH": wk_tape}, "edge")
    assert o3[0]["status"] == "skip" and o3[0]["reason"] == SKIP_DAILY and len(late3) == 3
    # the literal reference: Monday 00:00 UTC, early exits still at their cards
    ref, ref_late, _c = week_orders(wcards, dcards, wf, df, {"ETH": wk_tape}, "edge", end="close")
    assert ref[0]["exit_kind"] == "early" and ref_late == []
    print("week book: early exit at the third day card, a day without a card keeps the position, no Monday buy without the daily trigger, "
          "late entries booked per showing and refused under 1 % or with less than 24 h left")

    # 15. DST: the week of 25 Oct 2026 (Brussels leaves summer time that Sunday): 00:00 UTC is 02:00 before, 01:00 after
    assert brussels_text(U(2026, 10, 24, 0, 0)) == "Sat 24 Oct 02:00", brussels_text(U(2026, 10, 24, 0, 0))
    assert brussels_text(U(2026, 10, 26, 0, 0)) == "Mon 26 Oct 01:00"
    assert brussels_text(U(2026, 10, 25, 10, 16)) == "Sun 25 Oct 11:16"
    assert at(fallback_deadline("cryptoweek", "2026-10-25")) == datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    print("time: 00:00 UTC reads 02:00 Brussels before 25 Oct 2026 and 01:00 after; the week's fallback is Monday 12:00 UTC")

    # 31 Oct 2026 (review) regressions
    # R1. an up call without a target (a sized row's up call on the open lowest bin) is a counted skip, never an order
    #     without a limit - the day book used to crash on it and the whole orders block was lost
    sized = [dict(card("2026-10-20", d1, 0), calls={"M": {"BTC": (0, 1.0)}}), card("2026-10-21", d2, 7, base_close=100.0)]
    book = day_orders(sized, followed, {"BTC": flat}, "edge")
    assert [(o["status"], o["reason"]) for o in book][0] == ("skip", SKIP_NO_TARGET) and book[1]["status"] == "closed", book
    assert run_order(flat, d1, None, (), (d2, "next card"), base_close=100.0)["reason"] == SKIP_NO_TARGET
    wk_sized = [dict(wcards[0], calls={"WK": {"ETH": (0, 1.0)}})]
    o_s, l_s, _c = week_orders(wk_sized, dcards, wf, df, {"ETH": wk_tape}, "edge")
    assert o_s[0]["status"] == "skip" and o_s[0]["reason"] == SKIP_NO_TARGET and l_s == [], o_s
    # R2. the export that publishes a card: the minutes just before it are not stored yet (the refresh ran at 10:31:50,
    #     the stamp is 10:32:05). The held position is sold (or rolled) provisionally at that card - the card says
    #     "sell now", never "no order"/"buy now" to a reader who holds the coin - and no money is booked until the bars are in
    race = Tape(now=U(2026, 10, 21, 10, 32))
    for s in range(U(2026, 10, 19, 0, 0), U(2026, 10, 21, 10, 31), 60):
        race.m1[s] = (100.0,) * 4
    for s in range(U(2026, 10, 19, 0, 0), U(2026, 10, 21, 10, 0), 3600):
        race.h1[s] = (100.0,) * 4
    race.now = min(race.now, max(race.h1) + 3600)
    pub_now = U(2026, 10, 21, 10, 32) + 5
    sell_cards = [card("2026-10-20", d1, 7, base_close=99.5), card("2026-10-21", pub_now, 6, base_close=99.0)]   # X' below the price: no roll
    book = day_orders(sell_cards, followed, {"BTC": race}, "edge")
    assert book[0]["status"] == "open" and book[0]["exit_kind"] == "next card" and book[0]["exit_at"] == ceil_min(pub_now), book[0]
    assert book[0]["provisional"] and book[0]["money"] is None and book[1]["reason"] == SKIP_REACHED, book
    assert _day_now({"edge": book}, sell_cards, followed, "crypto")["symbols"]["BTC"]["status"] == "sell now"
    roll_cards = [card("2026-10-20", d1, 7, base_close=99.5), card("2026-10-21", pub_now, 7, base_close=100.0)]   # X' 101.0: rolled
    book = day_orders(roll_cards, followed, {"BTC": race}, "edge")
    assert len(book) == 1 and book[0]["rolls"] == 1 and book[0]["status"] == "open" and book[0]["provisional"], book
    assert _day_now({"edge": book}, roll_cards, followed, "crypto")["symbols"]["BTC"]["status"] == "hold (rolled)"
    #     the same for a week order's early exit at the day card being published; a clock end (the fallback) stays open
    o = run_order(race, d1, 101.0, [(pub_now, "early")], (U(2026, 10, 26, 12, 0), "fallback"), base_close=99.0)
    assert o["status"] == "open" and o["exit_kind"] == "early" and o["exit_at"] == ceil_min(pub_now) and o["provisional"] and o["money"] is None, o
    o = run_order(race, d1, 101.0, (), (U(2026, 10, 26, 12, 0), "fallback"), base_close=99.0)
    assert o["status"] == "open" and o["exit_kind"] is None, o
    # R3. a week order is followable only when its entry (the later of the week card and Monday's day card) falls on Monday
    late_monday = [dict(c, pub=U(2026, 10, 20, 0, 30)) if c["interval"] == "2026-10-19" else c for c in dcards]
    o3, _l, _c = week_orders(wcards, late_monday, wf, df, {"ETH": wk_tape}, "edge")
    assert o3[0]["status"] == "not followable" and not o3[0]["live"], o3[0]
    # R4. an order sold at a card whose time is estimated is flagged estimated (the week's Thursday card here)
    est_d = [dict(c, estimated=c["interval"] == "2026-10-22") for c in dcards]
    o4, _l, _c = week_orders([dict(wcards[0], estimated=False), wcards[1]], est_d, wf, df, {"ETH": wk_tape}, "edge")
    assert o4[0]["exit_kind"] == "early" and o4[0]["estimated"], o4[0]
    # R5. the CLI refuses a root inside a git checkout (this module's own repo)
    assert _checkout_of(os.path.dirname(os.path.abspath(__file__))) is not None or not os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".git"))
    print("review 31 Oct: an up call without a target is a skip; at the card being published a held coin is sold or rolled "
          "provisionally; a week entry after Monday is not followable; an estimated exit card flags the order; checkouts refused")

    # windows, variants, stats, the variant log
    assert minute_windows([U(2026, 10, 20, 10, 20), U(2026, 10, 20, 10, 50), U(2026, 10, 21, 8, 5)]) == [
        (at(U(2026, 10, 20, 10, 0)), at(U(2026, 10, 20, 12, 0))), (at(U(2026, 10, 21, 8, 0)), at(U(2026, 10, 21, 10, 0)))]
    assert abs(target_price("edge", 100.0, 6, edges, None, "day") - 100 * math.exp(0.005)) < 1e-12
    assert abs(target_price("mid", 100.0, 6, edges, None, "day") - 100 * math.exp(0.0075)) < 1e-12
    assert abs(target_price("fixed", None, 6, edges, 200.0, "week") - 208.0) < 1e-12 and target_price("edge", 100.0, 0, edges, None, "day") is None
    st = order_stats(orders + late)
    assert st["orders"] == 4 and st["closed"] == 4 and st["early"] == 2 and st["time_stop"] == 2 and st["filled"] == 0
    assert abs(st["total"] - round(sum(o["money"] for o in orders + late), 4)) < 1e-9
    assert shown_variant("2026-10-14") == "edge" and shown_variant("2027-01-01") == ORDER_VARIANT_LOG[-1][1]
    assert pick_of({"points": [["2026-10-20", "A"], ["2026-10-22", "B"]], "next_pick": "C"}, "2026-10-21") == "B"
    assert pick_of({"points": [["2026-10-20", "A"]], "next_pick": "C"}, "2026-10-21") == "C"
    print("windows merge, the three variants' limits, stats add up, the variant log and the pick for a day without a point behave")

    # the store end to end: cards from day files, publications (recorded, estimated), the export block's counts
    import tempfile
    try:
        from src.MarketData import install_universe, upsert_intraday
    except ImportError:
        from MarketData import install_universe, upsert_intraday
    with tempfile.TemporaryDirectory() as root:
        conn = connect(os.path.join(root, "m.sqlite3"))
        install_universe(conn)
        conn.executescript(GAME_SCHEMA)
        for gd in ("2026-10-09", "2026-10-19", "2026-10-20"):
            for pos, sym in enumerate(["BTC", "ETH", "BNB", "XRP", "SOL"]):
                conn.execute("INSERT INTO game_days VALUES ('crypto', ?, ?, ?, 0.0, 5, ?)", (gd, pos, sym, json.dumps(edges)))
        btc = instruments(conn, "crypto")[0]
        for gd in ("2026-10-19", "2026-10-20"):
            conn.execute("INSERT INTO bars VALUES (?, ?, 1, 1, 1, 99.85, 1, 't')", (btc["id"], gd))
        conn.commit()
        folder = os.path.join(root, "data", "database", "crypto")
        os.makedirs(folder)
        for made_on, made_at, b in (("2026-10-19", "2026-10-20T10:00:00+00:00", 6), ("2026-10-20", "2026-10-21T10:00:00+00:00", 4),
                                    ("2026-10-9", "2026-10-10T10:00:00+00:00", 4)):        # Predictor's names are not zero-padded
            with open(os.path.join(folder, f"{made_on}.json"), "w") as handle:
                json.dump({"madeAt": made_at, "newPrediction": [{"name": "M", "predictions": [[b, 4, 4, 4, 4]]}]}, handle)
        bars = [{"start": iso(at(s)), "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}
                for s in range(U(2026, 10, 20, 0, 0), U(2026, 10, 22, 0, 0), 60)]
        upsert_intraday(conn, btc["id"], "1m", bars)
        assert record_publication(conn, "crypto", "2026-10-20", "2026-10-21T10:00:00+00:00", now=datetime(2026, 10, 21, 10, 12, tzinfo=timezone.utc))
        assert not record_publication(conn, "crypto", "2026-10-20", "2026-10-21T10:00:00+00:00", now=datetime(2026, 10, 21, 10, 40, tzinfo=timezone.utc)), "first seen only"
        assert not record_publication(conn, "crypto", "2026-10-19", "2026-10-20T10:00:00+00:00", now=datetime(2026, 10, 21, 10, 40, tzinfo=timezone.utc)), "too late to stamp"
        assert not record_publication(conn, "shares", "2026-10-19", None)
        # 31 Oct 2026 (review): the old page decides when it can be read - a first showing is stamped however late
        # (the estimate would lie before the page appeared), a ticket the old page already showed never is
        late_run = datetime(2026, 10, 25, 14, 0, tzinfo=timezone.utc)
        assert not record_publication(conn, "cryptoweek", "2026-10-24", "2026-10-25T10:00:00+00:00", now=late_run, shown_before=True)
        assert not record_publication(conn, "cryptoweek", "2026-10-24", "2026-10-25T10:00:00+00:00", now=late_run, shown_before=None)
        assert record_publication(conn, "cryptoweek", "2026-10-24", "2026-10-25T10:00:00+00:00", now=late_run, shown_before=False)
        assert not record_publication(conn, "crypto", "2026-10-21", "2026-10-22T10:00:00+00:00",
                                      now=datetime(2026, 10, 22, 10, 5, tzinfo=timezone.utc), shown_before=True), "already on the page"
        pubs = publications(conn, root, "crypto")
        assert pubs["2026-10-20"] == (U(2026, 10, 21, 10, 12), False) and pubs["2026-10-19"] == (U(2026, 10, 20, 10, 15), True), pubs
        assert [c["interval"] for c in load_cards(conn, root, "crypto", since="2026-10-08")] == ["2026-10-10", "2026-10-20", "2026-10-21"], "cards in date order"
        cards = load_cards(conn, root, "crypto", since="2026-10-19")
        assert [c["interval"] for c in cards] == ["2026-10-20", "2026-10-21"] and cards[0]["base"]["BTC"] == 99.85 and cards[0]["calls"]["M"]["BTC"] == (6, None)
        rec = orders_record(conn, root, "crypto", followed={"points": [["2026-10-20", "M"]], "next_pick": "M"}, now=datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc),
                            since="2026-10-19")
        edge = rec["books"]["edge"]
        assert len(edge["orders"]) == 1 and edge["orders"][0]["exit_kind"] == "next card" and edge["orders"][0]["exit_at"] == "2026-10-21T10:12:00+00:00", edge
        assert edge["orders"][0]["estimated"] and edge["orders"][0]["limit_pct"] == round(100 * math.log(99.85 * math.exp(0.005) / 100.0), 3)
        assert rec["record"]["stats"] == edge["stats"] and rec["record"]["shown"] == [["2026-10-20", "edge"], ["2026-10-21", "edge"]]
        assert rec["newcomer"]["edge"]["stats"]["orders"] == 0 and rec["newcomer_rule"] == NEWCOMER_RULE and rec["now"]["symbols"]["BTC"]["newcomer"] is None
        assert rec["now"]["interval"] == "2026-10-21" and rec["now"]["symbols"]["BTC"]["status"] == "sell now" and rec["now"]["symbols"]["ETH"]["status"] == "no order"
        assert rec["benchmark"]["points"][0][:2] == ["2026-10-20", 1] and rec["reference"]["edge"]["orders"][0]["exit_kind"] == "close"
        assert orders_record(conn, root, "shares") is None
        json.dumps(rec)
        conn.close()
    print("store: cards from the day files, publications recorded first-seen within the window else estimated, the export block's counts agree")
    print("MarketOrders self-check OK")


def _checkout_of(folder):
    """The git work tree `folder` lies in (a directory with a .git entry, itself or a parent), or None."""
    here = os.path.realpath(folder)
    while True:
        if os.path.exists(os.path.join(here, ".git")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m src.MarketOrders", description="The crypto order book: self-check, or a backfill into a SCRATCH store")
    parser.add_argument("--backfill", metavar="DATE", help="fetch 1h bars from the Monday of DATE's week and the 1m windows of every card since, then print the books")
    parser.add_argument("--root", help="checkout whose data/database/crypto(week) is read and, with --export, data/markets written (required with --backfill)")
    parser.add_argument("--db", help="the SQLite store to fetch into (required with --backfill)")
    parser.add_argument("--no-fetch", action="store_true", help="no network: book from what is stored")
    parser.add_argument("--export", action="store_true", help="also write <root>/data/markets/crypto.json and cryptoweek.json (MarketSettle.export_page_json)")
    parser.add_argument("--allow-checkout", action="store_true",
                        help="allow a --root or --db inside a git checkout (the production backfill, with the owner's go); refused otherwise")
    parser.add_argument("--lag", type=float, default=PUBLISH_LAG_FALLBACK.total_seconds() / 60, help="publication lag in minutes for estimated cards (default 15)")
    args = parser.parse_args(argv)
    if not args.backfill:
        _self_check()
        return 0
    if not args.root or not args.db:
        parser.error("--backfill needs --root and --db explicitly (a scratch copy; never the production store by default)")
    # 31 Oct 2026 (review): the root's data/markets/*.json are TRACKED files of a checkout and its data/ is the live
    # store - a mistyped --root or --db (the dev repo, production) would write them. Refused unless asked for.
    for given in (args.root, os.path.dirname(os.path.abspath(args.db))):
        inside = _checkout_of(given)
        if inside and not args.allow_checkout:
            parser.error(f"{given} lies in the git checkout {inside}: use a scratch copy (cp -a), or --allow-checkout for the approved production backfill")
    since_day = date.fromisoformat(args.backfill)
    monday = since_day - timedelta(days=since_day.weekday())
    conn = connect(args.db)
    if not args.no_fetch:
        # windows at the three lags the back-test reads (0, 15, 30 min after madeAt), so the sensitivity has its minutes
        summary = refresh_intraday(conn, args.root, since=datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc),
                                   lags=(timedelta(0), PUBLISH_LAG_FALLBACK, timedelta(minutes=30)))
        for symbol, s in summary.items():
            print(f"{symbol}: {s['1h']} new 1h bar(s), {s['1m']} new 1m bar(s){' - ' + s['error'] if s['error'] else ''}")
    lag = timedelta(minutes=args.lag)
    for market in ORDER_MARKETS:
        rec = orders_record(conn, args.root, market, lag=lag, since=args.backfill)
        for v in ORDER_VARIANTS:
            s = rec["books"][v]["stats"]
            r = rec["reference"][v]["stats"]
            print(f"{market} {v:5s}: {s['orders']} order(s), {s['closed']} closed ({s['filled']} filled, {s['early']} early, {s['time_stop']} time stop), "
                  f"{s['open']} open, {s['pending']} pending, skips {s['skips']}: {s['total']:+.2f} USDT | 00:00 end {r['total']:+.2f}")
        for v in ORDER_VARIANTS if "newcomer" in rec else ():
            s = rec["newcomer"][v]["stats"]
            print(f"{market} {v:5s} newcomer: {s['orders']} entr{'y' if s['orders'] == 1 else 'ies'}, {s['closed']} closed ({s['filled']} filled, "
                  f"{s['time_stop']} time stop), {s['open']} open, {s['pending']} pending, skips {s['skips']}, rolls {s['rolls']}: {s['total']:+.2f} USDT")
        for v in ORDER_VARIANTS if "late" in rec else ():
            s = rec["late"][v]["stats"]
            print(f"{market} {v:5s} late: {s['orders']} entr{'y' if s['orders'] == 1 else 'ies'}, {s['closed']} closed, {s['open']} open: {s['total']:+.2f} USDT")
        print(f"{market} benchmark: {rec['benchmark']['stats']['total']:+.2f} USDT over {rec['benchmark']['stats']['closed']} closed")
    if args.export:
        try:
            from src.MarketSettle import export_page_json
        except ImportError:
            from MarketSettle import export_page_json
        for market in ORDER_MARKETS:
            out, _record = export_page_json(conn, args.root, market)
            print(f"{market}: {out}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
