"""
Settlement and export for the market games (README roadmap item 4, phase M2).

Predictor.py tracks a market like any game: one day file per game day under
data/database/<market>/ with the real result (the day's return bins) and
every model's prediction for it, made the day before. This module turns
those files into what the design asks for and the lottery machinery cannot
express, because it only ever sees bins:

  results        per model, instrument and day: the predicted bin, the actual
                 bin and return, hit exact / adjacent / direction, and the
                 paper P&L of the fixed rule - long when the predicted bin is
                 in the upper half, otherwise flat, minus a fee - with the
                 REAL return of the day (table `results` in the store)
  predictions    the same predictions with the price they stood for (table
                 `predictions`)
  exports        data/markets/<market>/results-<year>.csv - the tracked text
                 record (the store is gitignored) - and
                 data/markets/<market>.json, everything the Crypto and Shares
                 pages draw: closes, the predicted course per model, the
                 next day's predictions as prices, per-model accuracy against
                 chance, the daily accuracy series

Chance levels for K = 10 equiprobable bins: exact 1/K = 10%, adjacent
(|p - a| <= 1) (3K - 2)/K^2 = 28%, direction about 50%. A model has shown
something when it sits above those over many days - and, as everywhere in
this project, only after the controls have had their say.

Self-check: python3 -m src.MarketSettle (part of npm test; no network).
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone

try:
    from src.MarketData import closes, instruments
    from src.MarketGame import (K_BINS, GAME_SCHEMA, bin_interval, direction_of, latest_edges, load_game_day,
                                predicted_price, price_interval, representative_return)
except ImportError:  # imported from within src/
    from MarketData import closes, instruments
    from MarketGame import (K_BINS, GAME_SCHEMA, bin_interval, direction_of, latest_edges, load_game_day,
                            predicted_price, price_interval, representative_return)

FEE = 0.001            # 0.1% per position taken - a taker fee on a large exchange; shares are cheaper, this is the conservative one
# Paper trading in money (2 Oct 2026, the owner's ask: "for each prediction 1 is
# bought and the next day sold"): a fixed stake per position in the quote
# currency, bought at the previous game day's close when a model's bin says up
# (the upper half), sold at the day's close, a fee on each leg; flat otherwise.
# A fixed stake rather than one coin or one share, because one BTC and one XRP
# are not comparable positions. Holding across days and shorts are the later
# extensions the owner named; this is the first, simplest rule.
STAKE = 100.0          # per position, in USDT (crypto) or USD (shares)
FEE_PER_LEG = 0.001    # 0.1% on the buy and 0.1% on the sell
CHART_DAYS = 365       # closes and predicted course the page draws (a year; the chart zooms)
TOP_MODELS = 3         # models whose predicted course is drawn (plus every model in the next-day table)
DAY_RECORDS = 30       # settled days the page lists day by day, newest first
DAY_FILE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})\.json$")

# Rows that do not predict on their own: aggregates of the others, never
# settled as models of their own on the market pages (they still are on the
# History page, like everywhere).
AGGREGATE_ROWS = ()


def day_file_date(name):
    """'2026-9-9.json' -> '2026-09-09', or None."""
    match = DAY_FILE.match(name)
    if not match:
        return None
    year, month, day = (int(v) for v in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def chance_levels(k=K_BINS):
    return {"exact": 1.0 / k, "adjacent": (3 * k - 2) / (k * k), "direction": 0.5}


def cash_pnl(ret, stake=STAKE, fee=FEE_PER_LEG):
    """
    Money made by one position: `stake` bought at the previous close with a
    fee on it, sold at the close (stake x exp(ret)) with a fee on the
    proceeds. Negative when the move does not cover the two fees.
    """
    proceeds = stake * math.exp(float(ret))
    return proceeds * (1.0 - fee) - stake * (1.0 + fee)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def settle_day(conn, market, day, real_result, rows, symbol_ids, fee=FEE, k=K_BINS, settled_at=None):
    """
    Settles one game day: `rows` are the predictions made for it (the day
    file's currentPrediction), `real_result` its bins. Needs the day's
    stored game day (returns and edges) - without it nothing is settled.
    Returns the number of (model, instrument) results written.
    """
    game_day = load_game_day(conn, market, day)
    if game_day is None:
        return 0
    stamp = settled_at or utc_now()
    written = 0
    half = k / 2
    for row in rows:
        model = row.get("name")
        tickets = row.get("predictions") or []
        if not model or not tickets or not tickets[0] or model in AGGREGATE_ROWS:
            continue
        ticket = [int(v) for v in tickets[0]]
        for pos, symbol in enumerate(game_day["symbols"]):
            if pos >= len(ticket) or pos >= len(real_result) or symbol not in symbol_ids:
                continue
            predicted = ticket[pos]
            actual = int(real_result[pos])
            ret = float(game_day["returns"][pos])
            edges = game_day["edges"][pos]
            direction = direction_of(predicted, edges)
            actual_direction = 1 if ret > 0 else (-1 if ret < 0 else 0)
            long = predicted >= half
            pnl = (ret - fee) if long else 0.0
            conn.execute(
                "INSERT INTO results (model, instrument_id, for_date, settled_at, actual_bin, actual_return, hit_exact, hit_adjacent, "
                "hit_direction, pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(model, instrument_id, for_date) DO UPDATE SET settled_at = excluded.settled_at, actual_bin = excluded.actual_bin, "
                "actual_return = excluded.actual_return, hit_exact = excluded.hit_exact, hit_adjacent = excluded.hit_adjacent, "
                "hit_direction = excluded.hit_direction, pnl = excluded.pnl",
                (model, symbol_ids[symbol], day, stamp, actual, ret, int(predicted == actual), int(abs(predicted - actual) <= 1),
                 int(direction == actual_direction and direction != 0), pnl))
            conn.execute(
                "INSERT INTO predictions (model, instrument_id, for_date, made_at, bin, direction, confidence, probabilities, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?) "
                "ON CONFLICT(model, instrument_id, for_date) DO UPDATE SET bin = excluded.bin, direction = excluded.direction, meta = excluded.meta",
                (model, symbol_ids[symbol], day, "before " + day, predicted, direction,
                 json.dumps({"representative_return": representative_return(predicted, edges), "long": long})))
            written += 1
    conn.commit()
    return written


def settle_market(conn, path, market, fee=FEE, k=K_BINS, log=print):
    """Settles every day file of the market that has a stored game day. Returns (days settled, results written)."""
    conn.executescript(GAME_SCHEMA)
    folder = os.path.join(path, "data", "database", market)
    if not os.path.isdir(folder):
        return 0, 0
    symbol_ids = {i["symbol"]: i["id"] for i in instruments(conn, market, active_only=False)}
    days = written = 0
    for name in sorted(os.listdir(folder)):
        day = day_file_date(name)
        if not day:
            continue
        try:
            with open(os.path.join(folder, name)) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log(f"{market}: {name} unreadable - {exc}")
            continue
        real = data.get("realResult")
        rows = data.get("currentPrediction") or []
        if not real or not rows:
            continue
        count = settle_day(conn, market, day, real, rows, symbol_ids, fee=fee, k=k)
        if count:
            days += 1
            written += count
    return days, written


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------

def _result_rows(conn, market):
    """Every settled result of the market with the bin that was played, oldest first."""
    return conn.execute(
        "SELECT r.model, r.for_date, i.symbol, i.position, r.actual_return, r.hit_exact, r.hit_adjacent, r.hit_direction, r.pnl, p.bin "
        "FROM results r JOIN instruments i ON i.id = r.instrument_id "
        "LEFT JOIN predictions p ON p.model = r.model AND p.instrument_id = r.instrument_id AND p.for_date = r.for_date "
        "WHERE i.market = ? ORDER BY r.for_date, r.model, i.position", (market,)).fetchall()


def model_summary(conn, market, k=K_BINS, stake=STAKE, fee_per_leg=FEE_PER_LEG):
    """
    Per model over every settled result of the market, sorted by exact rate
    then P&L: hit rates, the fixed rule's P&L in return units (pnl_total),
    and the same positions in money (pnl_cash_total: `stake` per position,
    a fee on each leg - see cash_pnl), with trades, wins and the win rate.
    """
    per = {}
    for r in _result_rows(conn, market):
        m = per.setdefault(r["model"], {"n": 0, "days": set(), "exact": 0, "adjacent": 0, "direction": 0, "trades": 0, "pnl": 0.0,
                                        "cash": 0.0, "wins": 0, "first": r["for_date"], "last": r["for_date"]})
        m["n"] += 1
        m["days"].add(r["for_date"])
        m["exact"] += int(r["hit_exact"] or 0)
        m["adjacent"] += int(r["hit_adjacent"] or 0)
        m["direction"] += int(r["hit_direction"] or 0)
        m["pnl"] += float(r["pnl"] or 0.0)
        m["last"] = max(m["last"], r["for_date"])
        m["first"] = min(m["first"], r["for_date"])
        if r["bin"] is not None and int(r["bin"]) >= k / 2 and r["actual_return"] is not None:
            m["trades"] += 1
            made = cash_pnl(r["actual_return"], stake, fee_per_leg)
            m["cash"] += made
            m["wins"] += int(made > 0)
    out = []
    for name, m in per.items():
        n, trades = m["n"], m["trades"]
        out.append({
            "name": name, "days": len(m["days"]), "positions": n,
            "exact_rate": m["exact"] / n if n else None,
            "adjacent_rate": m["adjacent"] / n if n else None,
            "direction_rate": m["direction"] / n if n else None,
            "trades": trades, "pnl_total": m["pnl"],
            "pnl_per_trade": (m["pnl"] / trades) if trades else None,
            "pnl_cash_total": m["cash"], "pnl_cash_per_trade": (m["cash"] / trades) if trades else None,
            "wins": m["wins"], "win_rate": (m["wins"] / trades) if trades else None,
            "first_day": m["first"], "last_day": m["last"],
        })
    out.sort(key=lambda m: (-(m["exact_rate"] or 0), -m["pnl_total"], m["name"]))
    return out


def hold_book(dates, calls, returns, stake=STAKE, fee=FEE_PER_LEG):
    """
    One instrument under the HOLD rule: a position is opened at the previous
    close on the first day the call is up and kept while the calls stay up,
    then sold at the close of the last up day (the morning after, when the
    next call is not up, the position is gone - we only have closes, so the
    sale is booked at that close); a position still open at the end is
    sold at the last close. The buy fee is paid once on entry, the sell fee
    once on exit, and the stake compounds while held. `calls[i]` is True
    when the model called day i up, False or None otherwise; `returns[i]` is
    the day's log return (None when unknown). Returns (daily money per
    date, number of positions, positions that made money).
    """
    daily = [0.0] * len(dates)
    value = None
    entry_cost = 0.0
    trades = wins = 0
    for i in range(len(dates)):
        up = bool(calls[i]) and returns[i] is not None
        if up:
            if value is None:
                value = stake
                entry_cost = stake * (1.0 + fee)
                daily[i] -= stake * fee
                trades += 1
            grown = value * math.exp(float(returns[i]))
            daily[i] += grown - value
            value = grown
            last = i + 1 >= len(dates) or not (bool(calls[i + 1]) and returns[i + 1] is not None)
            if last:
                daily[i] -= value * fee
                wins += int(value * (1.0 - fee) - entry_cost > 0)
                value = None
    return daily, trades, wins


def ledger(conn, market, k=K_BINS, stake=STAKE, fee_per_leg=FEE_PER_LEG):
    """
    The paper-trading book, day by day, under two rules. DAILY: every long
    call is a round trip - the stake bought at the previous close, sold at
    the day's close, a fee on each leg. HOLD: a position is kept while the
    calls stay up and sold when they stop, so consecutive up days cost one
    fee pair and compound (hold_book). Per model the money made each settled
    day (zero when it sat out, None when it had no result) and the running
    total; and the market benchmark for each rule - buying every instrument
    every day with the same stake and fees (daily), or buying everything on
    the first day and holding to the last (hold): what the instruments
    themselves gave over the same days. Dates are the settled days, oldest
    first.
    """
    rows = _result_rows(conn, market)
    dates = sorted({r["for_date"] for r in rows})
    index = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    # per model and instrument: the call (up or not) and the return per date
    calls, rets, seen = {}, {}, {}
    market_ret = {}
    for r in rows:
        i = index[r["for_date"]]
        key = (r["model"], r["symbol"])
        calls.setdefault(key, [None] * n)[i] = r["bin"] is not None and int(r["bin"]) >= k / 2
        rets.setdefault(key, [None] * n)[i] = None if r["actual_return"] is None else float(r["actual_return"])
        seen.setdefault(r["model"], set()).add(i)
        if r["actual_return"] is not None:
            market_ret.setdefault(r["symbol"], [None] * n)[i] = float(r["actual_return"])

    def series(daily, model_seen=None):
        running, out = 0.0, []
        for i, d in enumerate(dates):
            if model_seen is not None and i not in model_seen:
                out.append([d, None, None])
                continue
            running += daily[i]
            out.append([d, round(daily[i], 4), round(running, 4)])
        return out

    daily_models, hold_models, hold_stats = {}, {}, {}
    for (model, symbol), call in calls.items():
        ret = rets[(model, symbol)]
        day = daily_models.setdefault(model, [0.0] * n)
        for i in range(n):
            if call[i] and ret[i] is not None:
                day[i] += cash_pnl(ret[i], stake, fee_per_leg)
        held, trades, wins = hold_book(dates, call, ret, stake, fee_per_leg)
        hday = hold_models.setdefault(model, [0.0] * n)
        for i in range(n):
            hday[i] += held[i]
        stat = hold_stats.setdefault(model, {"trades": 0, "wins": 0})
        stat["trades"] += trades
        stat["wins"] += wins
    bench_daily, bench_hold = [0.0] * n, [0.0] * n
    for symbol, ret in market_ret.items():
        for i in range(n):
            if ret[i] is not None:
                bench_daily[i] += cash_pnl(ret[i], stake, fee_per_leg)
        held, _, _ = hold_book(dates, [r is not None for r in ret], ret, stake, fee_per_leg)
        for i in range(n):
            bench_hold[i] += held[i]
    return {
        "dates": dates,
        "models": {m: series(day, seen[m]) for m, day in daily_models.items()},
        "benchmark": series(bench_daily),
        "hold": {"models": {m: series(day, seen[m]) for m, day in hold_models.items()}, "benchmark": series(bench_hold),
                 "stats": {m: {"trades": st["trades"], "wins": st["wins"], "total": round(sum(hold_models[m]), 4),
                               "win_rate": (st["wins"] / st["trades"]) if st["trades"] else None,
                               "per_trade": (sum(hold_models[m]) / st["trades"]) if st["trades"] else None}
                           for m, st in hold_stats.items()}},
    }


def daily_series(conn, market):
    """Per settled day: mean exact / direction rate over every model and position, and the best model's exact rate."""
    rows = conn.execute(
        "SELECT r.for_date AS day, r.model, AVG(r.hit_exact) AS exact, AVG(r.hit_direction) AS direction, SUM(r.pnl) AS pnl "
        "FROM results r JOIN instruments i ON i.id = r.instrument_id WHERE i.market = ? GROUP BY r.for_date, r.model ORDER BY r.for_date",
        (market,)).fetchall()
    by_day = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(r)
    series = []
    for day in sorted(by_day):
        group = by_day[day]
        best = max(group, key=lambda g: g["exact"])
        series.append({"date": day, "models": len(group),
                       "exact_mean": sum(g["exact"] for g in group) / len(group),
                       "direction_mean": sum(g["direction"] for g in group) / len(group),
                       "best_exact": best["exact"], "best_model": best["model"],
                       "pnl_mean": sum(g["pnl"] for g in group) / len(group)})
    return series


def chart_series(conn, market, member, dates):
    """
    What an instrument's chart needs for the dates it draws, aligned to
    them: the game day's return and bin edges (None on a date that is not a
    game day of this market), and per model the bin it predicted for the
    date (None where it had none). The page turns a bin back into a price
    from the previous GAME day's close - close x exp(-return), exactly as the
    settlement does - which is not the instrument's own previous bar when
    another instrument lacked a day in between.
    """
    if not dates:
        return {"edges": [], "moves": [], "course": {}}
    since = dates[0]
    index = {d: i for i, d in enumerate(dates)}
    edges = [None] * len(dates)
    moves = [None] * len(dates)
    for r in conn.execute("SELECT date, ret, edges FROM game_days WHERE market = ? AND symbol = ? AND date >= ? ORDER BY date",
                          (market, member["symbol"], since)).fetchall():
        i = index.get(r["date"])
        if i is None:
            continue
        edges[i] = [round(float(e), 6) for e in json.loads(r["edges"])]    # 1e-6 of a return: nothing the page shows resolves finer, and it halves the record
        moves[i] = round(float(r["ret"]), 8)
    course = {}
    for r in conn.execute("SELECT model, for_date, bin FROM predictions WHERE instrument_id = ? AND for_date >= ? ORDER BY for_date",
                          (member["id"], since)).fetchall():
        i = index.get(r["for_date"])
        # a ticket for a date that is no game day (any more) has no edges to
        # stand on - a re-cut dropped the day but the settled rows stay - so
        # it is not drawn
        if i is None or r["bin"] is None or edges[i] is None:
            continue
        course.setdefault(r["model"], [None] * len(dates))[i] = int(r["bin"])
    return {"edges": edges, "moves": moves, "course": course}


def day_records(conn, market, days=CHART_DAYS, k=K_BINS):
    """
    The newest settled game days, newest first, in market terms - what the
    History page shows as digits, here as returns: per instrument the real
    return and its bin with the edges the day was cut with, and per model
    the predicted bin per instrument with the day's hits and P&L. This is
    the market's day-by-day record; the day files stay the game view.
    """
    conn.executescript(GAME_SCHEMA)
    # only dates that still have a stored game day count towards `days`: a
    # re-cut can remove a day's game day while its settled results stay
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT r.for_date FROM results r JOIN instruments i ON i.id = r.instrument_id "
        "JOIN game_days g ON g.market = i.market AND g.date = r.for_date "
        "WHERE i.market = ? ORDER BY r.for_date DESC LIMIT ?", (market, int(days))).fetchall()]
    out = []
    for day in dates:
        game_day = load_game_day(conn, market, day)
        if game_day is None:
            continue
        # game slots are dense over the ACTIVE instruments of the cut, by symbol;
        # instruments.position is the table's column and need not be dense
        slot_of = {symbol: pos for pos, symbol in enumerate(game_day["symbols"])}
        rows = conn.execute(
            "SELECT r.model, i.position, i.symbol, r.actual_bin, r.actual_return, r.hit_exact, r.hit_adjacent, r.hit_direction, r.pnl, p.bin "
            "FROM results r JOIN instruments i ON i.id = r.instrument_id "
            "LEFT JOIN predictions p ON p.model = r.model AND p.instrument_id = r.instrument_id AND p.for_date = r.for_date "
            "WHERE i.market = ? AND r.for_date = ? ORDER BY r.model, i.position", (market, day)).fetchall()
        per_model = {}
        for r in rows:
            pos = slot_of.get(r["symbol"])
            if pos is None:
                continue        # a result for an instrument the day's cut does not hold (retired since): not part of this draw
            entry = per_model.setdefault(r["model"], {"name": r["model"], "bins": [None] * len(game_day["symbols"]),
                                                      "exact": 0, "adjacent": 0, "direction": 0, "positions": 0, "pnl": 0.0, "pnl_cash": 0.0, "trades": 0})
            entry["bins"][pos] = None if r["bin"] is None else int(r["bin"])
            entry["exact"] += int(r["hit_exact"] or 0)
            entry["adjacent"] += int(r["hit_adjacent"] or 0)
            entry["direction"] += int(r["hit_direction"] or 0)
            entry["positions"] += 1
            entry["pnl"] += float(r["pnl"] or 0.0)
            if r["bin"] is not None and int(r["bin"]) >= k / 2 and r["actual_return"] is not None:
                entry["trades"] += 1
                entry["pnl_cash"] += cash_pnl(r["actual_return"])
        models = sorted(per_model.values(), key=lambda m: (-m["exact"], -m["direction"], m["name"]))
        # the same definition as daily_series (the chart): the mean over models of each model's own exact rate
        rates = [m["exact"] / m["positions"] for m in models if m["positions"]]
        out.append({
            "date": day,
            "instruments": [{"symbol": symbol, "return": float(game_day["returns"][pos]), "bin": int(game_day["bins"][pos]),
                             "edges": [float(e) for e in game_day["edges"][pos]]}
                            for pos, symbol in enumerate(game_day["symbols"])],
            "models": models,
            "best": models[0]["name"] if models else None,
            "exact_mean": (sum(rates) / len(rates)) if rates else None,
        })
    return out


def next_day_predictions(conn, market, path):
    """
    The newest day file's newPrediction (for the day after it) as prices:
    per instrument, per model the bin, its interval as prices, direction and
    the price the bin stands for, from the newest close and the newest edges.
    """
    folder = os.path.join(path, "data", "database", market)
    if not os.path.isdir(folder):
        return None
    dated = [(day_file_date(n), n) for n in os.listdir(folder)]
    dated = [(d, n) for d, n in dated if d]
    if not dated:
        return None
    newest_day, newest_name = max(dated)
    try:
        with open(os.path.join(folder, newest_name)) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    rows = data.get("newPrediction") or []
    edges_day = latest_edges(conn, market)
    if not rows or edges_day is None:
        return None
    # by symbol: game slots are dense over the cut's active instruments, the
    # table's position need not be (a retired instrument keeps its column)
    members = {i["symbol"]: i for i in instruments(conn, market, active_only=False)}
    # made_at: when the ticket was written - the day file's modification time
    # (settlement never rewrites a day file; the export's generated_at is
    # rewritten by every run, also one that made no new ticket)
    try:
        made_at = datetime.fromtimestamp(os.path.getmtime(os.path.join(folder, newest_name)), tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        made_at = None
    out = {"made_on": newest_day, "made_at": made_at, "for": "the next trading day", "instruments": {}}
    for pos, symbol in enumerate(edges_day["symbols"]):
        member = members.get(symbol)
        if member is None:
            continue
        dates, values = closes(conn, member["id"])
        close_by_date = dict(zip(dates, values))
        # the prediction stood on the newest GAME day's close; an instrument
        # may hold a newer bar already (the others lagging), which is then
        # said, not silently used as the base
        last_date = edges_day["date"]
        last_close = close_by_date.get(last_date)
        if last_close is None:
            continue
        traded_since = [d for d in dates if d > last_date]
        edges = edges_day["edges"][pos]
        predictions = []
        for row in rows:
            tickets = row.get("predictions") or []
            if not tickets or not tickets[0] or pos >= len(tickets[0]) or row.get("name") in AGGREGATE_ROWS:
                continue
            b = int(tickets[0][pos])
            low, high = price_interval(last_close, b, edges)
            predictions.append({"model": row["name"], "bin": b, "direction": direction_of(b, edges),
                                "price": round(predicted_price(last_close, b, edges), 6),
                                "low": None if low is None else round(low, 6), "high": None if high is None else round(high, 6)})
        out["instruments"][symbol] = {"last_close": last_close, "last_date": last_date, "predictions": predictions,
                                      "traded_since": traded_since[-1] if traded_since else None}
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results_csv(conn, path, market):
    """data/markets/<market>/results-<year>.csv, rewritten from the store (the tracked text record)."""
    rows = conn.execute(
        "SELECT r.for_date, r.model, i.symbol, p.bin AS predicted_bin, r.actual_bin, r.actual_return, r.hit_exact, r.hit_adjacent, "
        "r.hit_direction, r.pnl FROM results r JOIN instruments i ON i.id = r.instrument_id "
        "LEFT JOIN predictions p ON p.model = r.model AND p.instrument_id = r.instrument_id AND p.for_date = r.for_date "
        "WHERE i.market = ? ORDER BY r.for_date, r.model, i.position", (market,)).fetchall()
    folder = os.path.join(path, "data", "markets", market)
    os.makedirs(folder, exist_ok=True)
    by_year = {}
    for r in rows:
        by_year.setdefault(r["for_date"][:4], []).append(r)
    files = []
    for year, group in sorted(by_year.items()):
        file = os.path.join(folder, f"results-{year}.csv")
        with open(file, "w", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["date", "model", "symbol", "predicted_bin", "actual_bin", "return", "hit_exact", "hit_adjacent", "hit_direction", "pnl", "pnl_cash"])
            for r in group:
                long = r["predicted_bin"] is not None and int(r["predicted_bin"]) >= K_BINS / 2
                writer.writerow([r["for_date"], r["model"], r["symbol"], r["predicted_bin"], r["actual_bin"], f"{r['actual_return']:.8f}",
                                 r["hit_exact"], r["hit_adjacent"], r["hit_direction"], f"{r['pnl']:.8f}",
                                 f"{cash_pnl(r['actual_return']) if long else 0.0:.4f}"])
        files.append(file)
    return files


def export_page_json(conn, path, market, chart_days=CHART_DAYS, top_models=TOP_MODELS, fee=FEE, k=K_BINS):
    """data/markets/<market>.json: everything the market page draws."""
    models = model_summary(conn, market, k)
    book = ledger(conn, market, k)
    for m in models:
        st = book["hold"]["stats"].get(m["name"])
        m["hold_total"] = st["total"] if st else None
        m["hold_trades"] = st["trades"] if st else None
        m["hold_wins"] = st["wins"] if st else None
        m["hold_win_rate"] = st["win_rate"] if st else None
        m["hold_per_trade"] = st["per_trade"] if st else None
    drawn = [m["name"] for m in models[:top_models]]
    members = instruments(conn, market, active_only=False)
    instruments_out = []
    for member in members:
        dates, values = closes(conn, member["id"])
        window = dates[-chart_days:]
        series = chart_series(conn, market, member, window)
        instruments_out.append({
            "symbol": member["symbol"], "name": member["name"], "position": member["position"], "quote": member["quote"],
            "active": bool(member["active"]),
            "last_close": values[-1] if values else None, "last_date": dates[-1] if dates else None,
            "closes": [[d, v] for d, v in zip(window, values[-chart_days:])],
            # aligned to "closes": the game day's edges and return, and every model's predicted bin
            "edges": series["edges"], "moves": series["moves"], "course": series["course"],
        })
    newest = latest_edges(conn, market)
    record = {
        "market": market, "generated_at": utc_now(), "k": k, "fee": fee, "chance": chance_levels(k),
        "newest_game_day": newest["date"] if newest else None,
        "instruments": instruments_out, "models": models, "drawn_models": drawn, "best_model": drawn[0] if drawn else None,
        "trading": {"stake": STAKE, "fee_per_leg": FEE_PER_LEG, "currency": members[0]["quote"] if members else "",
                    "rule": "long when the predicted bin is in the upper half: the stake bought at the previous close, sold at the day's close, a fee on each leg; flat otherwise",
                    "hold_rule": "the same calls, but a position is kept while the calls stay up and sold at the close of the last up day: one fee pair per run, the stake compounds",
                    **book},
        "next": next_day_predictions(conn, market, path),
        "daily": daily_series(conn, market)[-chart_days:],
        "days": day_records(conn, market, days=DAY_RECORDS, k=k),
    }
    os.makedirs(os.path.join(path, "data", "markets"), exist_ok=True)
    out = os.path.join(path, "data", "markets", f"{market}.json")
    with open(out, "w") as handle:
        json.dump(record, handle, indent=1)
    return out, record


def settle_and_export(path, market, db_path=None, log=print, fee=FEE):
    """Predictor.py's hook after a market game's day: settle what can be settled, export the page and the CSVs."""
    try:
        from src.MarketData import DEFAULT_DB, connect
    except ImportError:
        from MarketData import DEFAULT_DB, connect
    conn = connect(db_path or os.path.join(path, DEFAULT_DB))
    try:
        days, written = settle_market(conn, path, market, fee=fee, log=log)
        files = export_results_csv(conn, path, market)
        out, record = export_page_json(conn, path, market, fee=fee)
        log(f"{market}: {written} result(s) settled over {days} day(s); {len(record['models'])} model(s) summarised; "
            f"page json {out}, {len(files)} results file(s)")
        return {"days": days, "results": written, "models": len(record["models"]), "json": out, "csv": files}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Self-check (no network)
# ---------------------------------------------------------------------------

def _self_check():
    import tempfile
    import numpy as np
    try:
        from src.MarketData import connect, install_universe, upsert_bars
        from src.MarketGame import build_game, store_game_days
    except ImportError:
        from MarketData import connect, install_universe, upsert_bars
        from MarketGame import build_game, store_game_days

    assert day_file_date("2026-9-9.json") == "2026-09-09" and day_file_date("2026-12-31.json") == "2026-12-31"
    assert day_file_date("notes.json") is None and day_file_date("2026-13-1.json") is None
    levels = chance_levels(10)
    assert abs(levels["exact"] - 0.1) < 1e-12 and abs(levels["adjacent"] - 0.28) < 1e-12
    print("day files parse, chance levels 10% / 28% / 50%")

    rng = np.random.default_rng(21)
    conn = connect(":memory:")
    install_universe(conn)
    crypto = instruments(conn, "crypto")
    days = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(400)]
    for member in crypto:
        walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, size=len(days))))
        upsert_bars(conn, member["id"], [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1.0} for d, c in zip(days, walk)], fetched_at="t")
    game_days, symbols = build_game(conn, "crypto", k=10, min_history=250)
    store_game_days(conn, "crypto", game_days, symbols)
    assert len(game_days) == 149

    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "data", "database", "crypto")
        os.makedirs(folder)
        # day files: the predictions for day t were "made" on t-1; make the
        # first model right on slot 0 always, the second always long and wrong
        for i, day in enumerate(game_days[-30:]):
            d = date.fromisoformat(day["date"])
            actual = day["bins"]
            perfect = list(actual)
            long_wrong = [9 if a < 5 else 0 for a in actual]
            data = {"realResult": actual,
                    "currentPrediction": [{"name": "Perfect Model", "predictions": [perfect]},
                                          {"name": "Contrarian Model", "predictions": [long_wrong]}],
                    "newPrediction": [{"name": "Perfect Model", "predictions": [[5, 6, 7, 8, 9]]},
                                      {"name": "Contrarian Model", "predictions": [[0, 0, 0, 0, 0]]}]}
            with open(os.path.join(folder, f"{d.year}-{d.month}-{d.day}.json"), "w") as handle:
                json.dump(data, handle)
        # a day file without a stored game day is skipped
        with open(os.path.join(folder, "2019-1-1.json"), "w") as handle:
            json.dump({"realResult": [1, 2, 3, 4, 5], "currentPrediction": [{"name": "Perfect Model", "predictions": [[1, 2, 3, 4, 5]]}]}, handle)

        settled_days, written = settle_market(conn, root, "crypto", log=lambda *_: None)
        assert settled_days == 30 and written == 30 * 2 * 5, (settled_days, written)
        assert settle_market(conn, root, "crypto", log=lambda *_: None) == (30, 300), "settling again rewrites the same rows"
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 300
        summary = {m["name"]: m for m in model_summary(conn, "crypto")}
        perfect, contrarian = summary["Perfect Model"], summary["Contrarian Model"]
        assert perfect["exact_rate"] == 1.0 and perfect["adjacent_rate"] == 1.0 and perfect["days"] == 30 and perfect["positions"] == 150
        assert contrarian["exact_rate"] == 0.0 and contrarian["adjacent_rate"] == 0.0
        # the contrarian goes long exactly when the actual bin is in the lower half: its P&L is those days' returns minus the fee
        expected = sum(day["returns"][p] - FEE for day in game_days[-30:] for p in range(5) if day["bins"][p] < 5)
        assert abs(contrarian["pnl_total"] - expected) < 1e-9, (contrarian["pnl_total"], expected)
        assert contrarian["trades"] == sum(1 for day in game_days[-30:] for p in range(5) if day["bins"][p] < 5)
        assert perfect["direction_rate"] > 0.5, perfect["direction_rate"]
        # the money: a fixed stake per long position, a fee on each leg; the perfect model is long exactly on the days the bin was in the upper half
        longs = [day["returns"][p] for day in game_days[-30:] for p in range(5) if day["bins"][p] >= 5]
        assert perfect["trades"] == len(longs) and abs(perfect["pnl_cash_total"] - sum(cash_pnl(r) for r in longs)) < 1e-9
        assert perfect["wins"] == sum(1 for r in longs if cash_pnl(r) > 0) and abs(perfect["win_rate"] - perfect["wins"] / perfect["trades"]) < 1e-12
        assert abs(cash_pnl(0.0) + 2 * STAKE * FEE_PER_LEG) < 1e-9 and cash_pnl(0.01) > 0 > cash_pnl(-0.01), "a flat day costs the two fees"
        book = ledger(conn, "crypto")
        assert len(book["dates"]) == 30 and abs(book["models"]["Perfect Model"][-1][2] - perfect["pnl_cash_total"]) < 1e-3
        assert abs(book["benchmark"][-1][2] - sum(cash_pnl(day["returns"][p]) for day in game_days[-30:] for p in range(5))) < 1e-3, "the benchmark buys everything every day"
        assert abs(sum(x[1] for x in book["models"]["Contrarian Model"]) - contrarian["pnl_cash_total"]) < 1e-3
        # the hold rule: one instrument by hand - up, up, down, up: two positions, the first compounding two days with one fee pair
        held, trades, wins = hold_book(["a", "b", "c", "d"], [True, True, False, True], [0.01, 0.02, -0.03, 0.005])
        first = STAKE * math.exp(0.03) * (1 - FEE_PER_LEG) - STAKE * (1 + FEE_PER_LEG)
        second = STAKE * math.exp(0.005) * (1 - FEE_PER_LEG) - STAKE * (1 + FEE_PER_LEG)
        assert trades == 2 and wins == 2 and abs(sum(held) - first - second) < 1e-9 and held[2] == 0.0, (held, trades, wins)
        assert abs(held[0] + held[1] - first) < 1e-9 and held[0] < held[1], "the sell fee lands on the last up day"
        assert hold_book(["a"], [True], [None]) == ([0.0], 0, 0) and hold_book(["a", "b"], [False, False], [0.01, 0.01])[1] == 0
        # ...and in the book: the perfect model's hold total is the sum over its runs of up days, per instrument
        expected_hold = 0.0
        for p in range(5):
            run = []
            for day in game_days[-30:] + [None]:
                if day is not None and day["bins"][p] >= 5:
                    run.append(day["returns"][p])
                elif run:
                    expected_hold += STAKE * math.exp(sum(run)) * (1 - FEE_PER_LEG) - STAKE * (1 + FEE_PER_LEG)
                    run = []
        assert abs(book["hold"]["stats"]["Perfect Model"]["total"] - expected_hold) < 1e-3, (book["hold"]["stats"]["Perfect Model"], expected_hold)
        assert abs(book["hold"]["models"]["Perfect Model"][-1][2] - expected_hold) < 1e-3
        # the hold benchmark buys everything on the first settled day and sells on the last: one fee pair per instrument
        whole = sum(STAKE * math.exp(sum(day["returns"][p] for day in game_days[-30:])) * (1 - FEE_PER_LEG) - STAKE * (1 + FEE_PER_LEG) for p in range(5))
        assert abs(book["hold"]["benchmark"][-1][2] - whole) < 1e-3, (book["hold"]["benchmark"][-1], whole)
        print(f"money: {STAKE:.0f} per position, {FEE_PER_LEG:.1%} a leg - the perfect model's book {perfect['pnl_cash_total']:+.2f} daily, "
              f"{book['hold']['stats']['Perfect Model']['total']:+.2f} holding; the market's {book['benchmark'][-1][2]:+.2f} daily, "
              f"{book['hold']['benchmark'][-1][2]:+.2f} buy-and-hold; win rate {perfect['win_rate']:.0%}")
        # a direction hit needs the representative return and the real return to share a sign
        one = conn.execute("SELECT r.hit_direction, r.actual_return, p.meta FROM results r JOIN predictions p "
                           "ON p.model = r.model AND p.instrument_id = r.instrument_id AND p.for_date = r.for_date "
                           "WHERE r.model = 'Perfect Model' LIMIT 20").fetchall()
        for r in one:
            rep = json.loads(r["meta"])["representative_return"]
            assert r["hit_direction"] == int((rep > 0) == (r["actual_return"] > 0) and rep != 0 and r["actual_return"] != 0) or \
                   r["hit_direction"] == int((rep < 0) == (r["actual_return"] < 0) and rep != 0 and r["actual_return"] != 0)
        print(f"settlement: 30 days x 2 models x 5 instruments; a perfect model reads 100%/100%, "
              f"the long-and-wrong model {contrarian['direction_rate']:.0%} direction and its P&L is the real returns minus the fee")

        series = daily_series(conn, "crypto")
        assert len(series) == 30 and series[-1]["best_model"] == "Perfect Model" and series[-1]["best_exact"] == 1.0 and abs(series[-1]["exact_mean"] - 0.5) < 1e-12
        files = export_results_csv(conn, root, "crypto")
        years = sorted({day["date"][:4] for day in game_days[-30:]})
        assert [os.path.basename(f) for f in files] == [f"results-{y}.csv" for y in years], files
        total_lines = 0
        for f in files:
            with open(f) as handle:
                lines = handle.read().strip().splitlines()
            assert lines[0].startswith("date;model;symbol;predicted_bin")
            total_lines += len(lines) - 1
        assert total_lines == 300, total_lines
        records = day_records(conn, "crypto", days=10)
        assert len(records) == 10 and records[0]["date"] == game_days[-1]["date"] and records[0]["date"] > records[-1]["date"], "newest first"
        first = records[0]
        assert [i["symbol"] for i in first["instruments"]] == symbols and first["instruments"][0]["bin"] == game_days[-1]["bins"][0]
        assert abs(first["instruments"][0]["return"] - game_days[-1]["returns"][0]) < 1e-12 and len(first["instruments"][0]["edges"]) == 9
        assert first["best"] == "Perfect Model" and first["models"][0]["exact"] == 5 and first["models"][0]["bins"] == game_days[-1]["bins"]
        assert first["models"][1]["name"] == "Contrarian Model" and first["models"][1]["exact"] == 0 and abs(first["exact_mean"] - 0.5) < 1e-12
        out, record = export_page_json(conn, root, "crypto", chart_days=40)
        assert len(record["days"]) == 30 and record["days"][0]["date"] == game_days[-1]["date"]
        assert os.path.exists(out) and record["market"] == "crypto" and record["k"] == 10
        assert [i["symbol"] for i in record["instruments"]] == symbols and len(record["instruments"][0]["closes"]) == 40
        assert record["drawn_models"] == ["Perfect Model", "Contrarian Model"] and record["best_model"] == "Perfect Model"
        btc = record["instruments"][0]
        last_day = game_days[-1]
        assert len(btc["edges"]) == 40 and len(btc["moves"]) == 40 and set(btc["course"]) == {"Perfect Model", "Contrarian Model"}
        assert sum(1 for b in btc["course"]["Perfect Model"] if b is not None) == 30, "every settled day carries the model's bin"
        assert btc["closes"][-1][0] == last_day["date"] and btc["course"]["Perfect Model"][-1] == last_day["bins"][0]
        assert all(abs(a - b) < 1e-6 for a, b in zip(btc["edges"][-1], last_day["edges"][0])) and abs(btc["moves"][-1] - last_day["returns"][0]) < 1e-8
        assert btc["course"]["Perfect Model"][0] is None and btc["edges"][0] is not None and btc["moves"][0] is not None, \
            "a game day without a ticket carries the edges and the move but no bin"
        # the page's base for a bin is close x exp(-move), the previous GAME day's close: the one the settlement's return spans
        btc_dates, btc_closes = closes(conn, crypto[0]["id"])
        base = btc["closes"][-1][1] * math.exp(-btc["moves"][-1])
        assert abs(base - btc_closes[btc_dates.index(last_day["date"]) - 1]) < 1e-6
        # ... also when another instrument lacked the day before: the base is the day before THAT, as the return spans it
        gap_day = game_days[-3]["date"]
        conn.execute("DELETE FROM bars WHERE instrument_id = ? AND date = ?", (crypto[4]["id"], gap_day))
        conn.commit()
        game_days2, symbols2 = build_game(conn, "crypto", k=10, min_history=250)
        store_game_days(conn, "crypto", game_days2, symbols2)
        after_gap = [d for d in game_days2 if d["date"] > gap_day][0]
        # re-settle so the predictions carry the re-cut bins for the day after the gap
        with open(os.path.join(folder, f"{date.fromisoformat(after_gap['date']).year}-{date.fromisoformat(after_gap['date']).month}-{date.fromisoformat(after_gap['date']).day}.json"), "w") as handle:
            json.dump({"realResult": after_gap["bins"], "currentPrediction": [{"name": "Perfect Model", "predictions": [after_gap["bins"]]}],
                       "newPrediction": [{"name": "Perfect Model", "predictions": [[5, 6, 7, 8, 9]]}]}, handle)
        settle_market(conn, root, "crypto", log=lambda *_: None)
        _, record2 = export_page_json(conn, root, "crypto", chart_days=40)
        btc2 = record2["instruments"][0]
        chart_dates = [p[0] for p in btc2["closes"]]
        i_gap, i_after = chart_dates.index(gap_day), chart_dates.index(after_gap["date"])
        assert btc2["edges"][i_gap] is None and btc2["moves"][i_gap] is None and btc2["course"]["Perfect Model"][i_gap] is None, "the gap day is no game day"
        idx = btc_dates.index(after_gap["date"])
        base_after = btc2["closes"][i_after][1] * math.exp(-btc2["moves"][i_after])
        assert abs(base_after - btc_closes[idx - 2]) < 1e-6, "the base must be the previous game day's close, two bars back across the gap"
        assert abs(base_after - btc_closes[idx - 1]) > 1e-9
        assert btc2["course"]["Perfect Model"][i_after] == after_gap["bins"][0] and all(abs(a - b) < 1e-6 for a, b in zip(btc2["edges"][i_after], after_gap["edges"][0]))
        # the next-day price stands on the newest game day's close; a newer bar is reported, not used
        upsert_bars(conn, crypto[0]["id"], [{"date": "2025-12-31", "open": 1, "high": 1, "low": 1, "close": 999.0, "volume": 1}], fetched_at="t")
        nxt2 = next_day_predictions(conn, "crypto", root)
        assert nxt2["instruments"]["BTC"]["last_close"] == btc_closes[-1] and nxt2["instruments"]["BTC"]["traded_since"] == "2025-12-31"
        conn.execute("DELETE FROM bars WHERE instrument_id = ? AND date = '2025-12-31'", (crypto[0]["id"],))
        conn.commit()
        nxt = record["next"]
        assert nxt["made_on"] == last_day["date"] and set(nxt["instruments"]) == set(symbols)
        assert nxt["made_at"] and nxt["made_at"].endswith("+00:00"), nxt["made_at"]
        btc_next = nxt["instruments"]["BTC"]
        assert btc_next["last_close"] == btc_closes[-1] and len(btc_next["predictions"]) == 2
        top = [p for p in btc_next["predictions"] if p["model"] == "Perfect Model"][0]
        assert top["bin"] == 5 and top["direction"] in (1, -1) and top["low"] < top["price"] < top["high"]
        assert len(record["daily"]) == 30 and record["chance"]["exact"] == 0.1
        assert record["trading"]["stake"] == STAKE and record["trading"]["currency"] == "USDT" and len(record["trading"]["dates"]) == 30
        assert record["models"][0]["hold_total"] is not None and len(record["trading"]["hold"]["benchmark"]) == 30
        assert record["days"][0]["models"][0]["pnl_cash"] is not None
        json.dumps(record)
        print("export: yearly results CSV, page json with closes, every model's bins with the day's edges and return, next-day prices and the daily series")

        # Day records after the re-cut: the gap day lost its game day but keeps
        # its settled results, so it is not listed and does not eat a slot of
        # the requested count.
        gap_records = day_records(conn, "crypto", days=5)
        assert len(gap_records) == 5 and all(r["date"] != gap_day for r in gap_records), [r["date"] for r in gap_records]
        assert conn.execute("SELECT COUNT(*) FROM results WHERE for_date = ?", (gap_day,)).fetchone()[0] > 0, "the results of the gap day stay"
        # A retired instrument: BTC (table position 0) goes inactive, the game is
        # re-cut over the four others (slots 0..3 by symbol, table positions
        # 1..4), the day re-settled - a model's bins must land under ETH, BNB,
        # XRP, SOL and not shift by the dead position.
        conn.execute("UPDATE instruments SET active = 0 WHERE id = ?", (crypto[0]["id"],))
        conn.commit()
        game_days3, symbols3 = build_game(conn, "crypto", k=10, min_history=250)
        assert symbols3 == ["ETH", "BNB", "XRP", "SOL"]
        store_game_days(conn, "crypto", game_days3, symbols3)
        newest3 = game_days3[-1]
        d3 = date.fromisoformat(newest3["date"])
        with open(os.path.join(folder, f"{d3.year}-{d3.month}-{d3.day}.json"), "w") as handle:
            json.dump({"realResult": newest3["bins"], "currentPrediction": [{"name": "Perfect Model", "predictions": [list(newest3["bins"])]}],
                       "newPrediction": [{"name": "Perfect Model", "predictions": [[1, 2, 3, 4]]}]}, handle)
        settle_market(conn, root, "crypto", log=lambda *_: None)
        retired = day_records(conn, "crypto", days=1)[0]
        assert [i["symbol"] for i in retired["instruments"]] == symbols3 and retired["date"] == newest3["date"]
        perfect3 = [m for m in retired["models"] if m["name"] == "Perfect Model"][0]
        assert perfect3["bins"] == newest3["bins"] and perfect3["exact"] == 4 and perfect3["positions"] == 4, perfect3
        # exact_mean is the mean of per-model rates, as the daily chart draws it
        rates = [m["exact"] / m["positions"] for m in retired["models"] if m["positions"]]
        assert abs(retired["exact_mean"] - sum(rates) / len(rates)) < 1e-12
        # the next-day prices stand on each slot's own instrument, found by symbol: ETH's on ETH's close, not on the retired column's
        nxt3 = next_day_predictions(conn, "crypto", root)
        assert set(nxt3["instruments"]) == set(symbols3), nxt3["instruments"].keys()
        eth_dates, eth_closes = closes(conn, crypto[1]["id"])
        assert nxt3["instruments"]["ETH"]["last_close"] == dict(zip(eth_dates, eth_closes))[newest3["date"]]
        assert nxt3["instruments"]["ETH"]["predictions"][0]["bin"] == 1, nxt3["instruments"]["ETH"]["predictions"]
        print("day records: newest first, a day without a game day is skipped without shortening the list, a retired instrument keeps every bin in its own slot")
    print("MarketSettle self-check OK")


if __name__ == "__main__":
    _self_check()
