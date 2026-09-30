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
CHART_DAYS = 120       # closes and predicted course the page draws
TOP_MODELS = 3         # models whose predicted course is drawn (plus every model in the next-day table)
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

def model_summary(conn, market, k=K_BINS):
    """Per model over every settled result of the market, sorted by exact rate then P&L."""
    rows = conn.execute(
        "SELECT r.model, COUNT(*) AS n, COUNT(DISTINCT r.for_date) AS days, SUM(r.hit_exact) AS exact, SUM(r.hit_adjacent) AS adjacent, "
        "SUM(r.hit_direction) AS direction, SUM(CASE WHEN p.bin >= ? THEN 1 ELSE 0 END) AS trades, SUM(r.pnl) AS pnl, "
        "MIN(r.for_date) AS first_day, MAX(r.for_date) AS last_day "
        "FROM results r JOIN instruments i ON i.id = r.instrument_id "
        "LEFT JOIN predictions p ON p.model = r.model AND p.instrument_id = r.instrument_id AND p.for_date = r.for_date "
        "WHERE i.market = ? GROUP BY r.model", (k / 2, market)).fetchall()
    out = []
    for r in rows:
        n = r["n"] or 0
        trades = r["trades"] or 0
        out.append({
            "name": r["model"], "days": r["days"], "positions": n,
            "exact_rate": (r["exact"] or 0) / n if n else None,
            "adjacent_rate": (r["adjacent"] or 0) / n if n else None,
            "direction_rate": (r["direction"] or 0) / n if n else None,
            "trades": trades, "pnl_total": float(r["pnl"] or 0.0),
            "pnl_per_trade": (float(r["pnl"] or 0.0) / trades) if trades else None,
            "first_day": r["first_day"], "last_day": r["last_day"],
        })
    out.sort(key=lambda m: (-(m["exact_rate"] or 0), -m["pnl_total"], m["name"]))
    return out


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


def predicted_history(conn, market, model_names, days):
    """
    The predicted course to draw: for each instrument and each of the newest
    `days` settled days, per model the price its predicted bin stood for
    (the previous close moved by the bin's representative return).
    """
    members = instruments(conn, market, active_only=False)
    out = {}
    if not model_names:
        return out
    placeholders = ",".join("?" for _ in model_names)
    for member in members:
        dates, values = closes(conn, member["id"])
        close_by_date = dict(zip(dates, values))
        rows = conn.execute(
            f"SELECT p.model, p.for_date, p.bin FROM predictions p WHERE p.instrument_id = ? AND p.model IN ({placeholders}) "
            "ORDER BY p.for_date DESC LIMIT ?", (member["id"], *model_names, days * len(model_names))).fetchall()
        per_model = {}
        for r in rows:
            game_day = load_game_day(conn, market, r["for_date"])
            close = close_by_date.get(r["for_date"])
            if game_day is None or close is None or member["position"] >= len(game_day["edges"]):
                continue
            # the base the bin was cut against is the previous GAME day's
            # close - the day's close undone by the day's aligned return -
            # which is not the instrument's own previous bar when another
            # instrument lacked a day in between
            base = close * math.exp(-float(game_day["returns"][member["position"]]))
            price = predicted_price(base, int(r["bin"]), game_day["edges"][member["position"]])
            per_model.setdefault(r["model"], []).append([r["for_date"], round(price, 6)])
        out[member["symbol"]] = {model: sorted(points) for model, points in per_model.items()}
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
    members = {i["position"]: i for i in instruments(conn, market, active_only=False)}
    out = {"made_on": newest_day, "for": "the next trading day", "instruments": {}}
    for pos, symbol in enumerate(edges_day["symbols"]):
        member = members.get(pos)
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
            writer.writerow(["date", "model", "symbol", "predicted_bin", "actual_bin", "return", "hit_exact", "hit_adjacent", "hit_direction", "pnl"])
            for r in group:
                writer.writerow([r["for_date"], r["model"], r["symbol"], r["predicted_bin"], r["actual_bin"], f"{r['actual_return']:.8f}",
                                 r["hit_exact"], r["hit_adjacent"], r["hit_direction"], f"{r['pnl']:.8f}"])
        files.append(file)
    return files


def export_page_json(conn, path, market, chart_days=CHART_DAYS, top_models=TOP_MODELS, fee=FEE, k=K_BINS):
    """data/markets/<market>.json: everything the market page draws."""
    models = model_summary(conn, market, k)
    drawn = [m["name"] for m in models[:top_models]]
    members = instruments(conn, market, active_only=False)
    instruments_out = []
    history = predicted_history(conn, market, drawn, chart_days)
    for member in members:
        dates, values = closes(conn, member["id"])
        instruments_out.append({
            "symbol": member["symbol"], "name": member["name"], "position": member["position"], "quote": member["quote"],
            "active": bool(member["active"]),
            "last_close": values[-1] if values else None, "last_date": dates[-1] if dates else None,
            "closes": [[d, v] for d, v in zip(dates[-chart_days:], values[-chart_days:])],
            "predicted_course": history.get(member["symbol"], {}),
        })
    newest = latest_edges(conn, market)
    record = {
        "market": market, "generated_at": utc_now(), "k": k, "fee": fee, "chance": chance_levels(k),
        "newest_game_day": newest["date"] if newest else None,
        "instruments": instruments_out, "models": models, "drawn_models": drawn,
        "next": next_day_predictions(conn, market, path),
        "daily": daily_series(conn, market)[-chart_days:],
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
        out, record = export_page_json(conn, root, "crypto", chart_days=40)
        assert os.path.exists(out) and record["market"] == "crypto" and record["k"] == 10
        assert [i["symbol"] for i in record["instruments"]] == symbols and len(record["instruments"][0]["closes"]) == 40
        assert record["drawn_models"] == ["Perfect Model", "Contrarian Model"]
        course = record["instruments"][0]["predicted_course"]["Perfect Model"]
        assert len(course) == 30 and course[0][0] < course[-1][0]
        # a perfect prediction's course is the previous GAME day's close moved by the actual bin's representative return
        last_day = game_days[-1]
        btc_dates, btc_closes = closes(conn, crypto[0]["id"])
        prev_close = btc_closes[btc_dates.index(last_day["date"]) - 1]
        assert abs(course[-1][1] - predicted_price(prev_close, last_day["bins"][0], last_day["edges"][0])) < 1e-6
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
        history = predicted_history(conn, "crypto", ["Perfect Model"], 40)
        point = [p for p in history["BTC"]["Perfect Model"] if p[0] == after_gap["date"]][0]
        idx = btc_dates.index(after_gap["date"])
        two_back = btc_closes[idx - 2]      # the previous game day: the gap day is not a game day
        assert abs(point[1] - predicted_price(two_back, after_gap["bins"][0], after_gap["edges"][0])) < 1e-6, "the base must be the previous game day's close"
        assert abs(point[1] - predicted_price(btc_closes[idx - 1], after_gap["bins"][0], after_gap["edges"][0])) > 1e-9
        # the next-day price stands on the newest game day's close; a newer bar is reported, not used
        upsert_bars(conn, crypto[0]["id"], [{"date": "2025-12-31", "open": 1, "high": 1, "low": 1, "close": 999.0, "volume": 1}], fetched_at="t")
        nxt2 = next_day_predictions(conn, "crypto", root)
        assert nxt2["instruments"]["BTC"]["last_close"] == btc_closes[-1] and nxt2["instruments"]["BTC"]["traded_since"] == "2025-12-31"
        conn.execute("DELETE FROM bars WHERE instrument_id = ? AND date = '2025-12-31'", (crypto[0]["id"],))
        conn.commit()
        nxt = record["next"]
        assert nxt["made_on"] == last_day["date"] and set(nxt["instruments"]) == set(symbols)
        btc_next = nxt["instruments"]["BTC"]
        assert btc_next["last_close"] == btc_closes[-1] and len(btc_next["predictions"]) == 2
        top = [p for p in btc_next["predictions"] if p["model"] == "Perfect Model"][0]
        assert top["bin"] == 5 and top["direction"] in (1, -1) and top["low"] < top["price"] < top["high"]
        assert len(record["daily"]) == 30 and record["chance"]["exact"] == 0.1
        json.dumps(record)
        print("export: yearly results CSV, page json with closes, predicted course, next-day prices and the daily series")
    print("MarketSettle self-check OK")


if __name__ == "__main__":
    _self_check()
