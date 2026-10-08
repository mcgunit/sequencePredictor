#!/usr/bin/env python3
"""
The market rows under a proper score (README roadmap item 4, phase M3).

The Regime HMM row is interesting only if it carries information beyond
GARCH - the variance part of a regime model is volatility clustering, which
GARCH already captures - and a proper scoring rule has to decide that before
any hit rate or paper P&L is read: tuning or judging on the P&L of the fixed
rule would quietly turn the row back into the allocator this track does not
build. So this script backtests the market rows (GARCH, Regime HMM and its
two ablations) together with the positional base rows the market games run,
walk-forward over the newest -d game days, exactly as TrainMetaLearner
collects its table (src/Backtester with the per-slot probabilities), and
scores every row by the MEAN LOG-SCORE of the probability it gave the bin
that then happened - higher is better, log(1/K) = -2.303 is the uniform
forecast, and a row below it is worse than saying nothing.

Every difference to the GARCH reference (and to uniform) comes with a 95%
paired bootstrap interval over days, so "adds information beyond GARCH" is
a claim about the interval's lower bound, not about a point estimate. Hit
rates (exact, adjacent, direction), the reliability of each row's
probabilities bin by bin, and the P&L of the fixed rule with the real
returns are reported next to it - as information, after the score.

The lockbox days (lockbox.json) leave the table before scoring, as
everywhere else. A probability below 1e-4 is scored as 1e-4 (a proper score
needs a finite log; the floor is stated in the record).

    python3 MarketRows.py                            # crypto and shares, newest 250 game days
    python3 MarketRows.py -g crypto -d 60 --market-only
    python3 MarketRows.py --root /elsewhere/checkout

Takes process.lock (it is a full backtest); writes
data/controls/markets/<market>-rows.json, which the market pages read.
"""

import os
# One BLAS thread per process, set before numpy loads: the Backtester forks a
# worker per core, and a numpy-heavy row (the HMM's EM) in fifteen workers
# each spinning sixteen OpenBLAS threads put the load average at 230 and the
# run at a crawl (measured 30 Sept 2026). The tuners pin it the same way.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone

import numpy as np

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.HyperoptRunner import install_sigterm_handler
from src.Lockbox import load as load_lockbox, split_rows as lockbox_split, describe as describe_lockbox, as_json as lockbox_json
from src.MarketGame import K_BINS, direction_of, quantile_edges
from src.MarketModels import GARCH_NAME, LOG_FLOOR, MARKET_MODEL_NAMES, MarketHistory, build_market_models
from src.MarketSettle import FEE
from src.ModelFactory import build_models, prepare_foundation_scores
from HyperoptStatistics import GAME_CONFIG, is_running, create_lock, remove_lock

helpers = Helpers()

DEFAULT_DAYS = 250
BOOTSTRAP = 2000
BASELINE_ROWS = ("random", "global_frequency", "column_frequency")
RESULT_DIR = os.path.join("data", "controls", "markets")


def result_path(root, market):
    return os.path.join(root, RESULT_DIR, f"{market}-rows.json")


# --- scoring one row ------------------------------------------------------------
def day_edges(history, upto):
    """The edges the game day at returns-row `upto` was cut with: quantiles of the rows before it."""
    return [quantile_edges(history[:upto, pos], K_BINS) for pos in range(history.shape[1])]


def bootstrap_interval(values, rng, resamples=BOOTSTRAP):
    """(mean, 2.5%, 97.5%) of the mean over days, by resampling days."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    if len(values) < 3:
        m = float(values.mean())
        return {"mean": m, "lo": m, "hi": m, "days": int(len(values))}
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)), "days": int(len(values))}


def verdict(interval):
    if interval is None:
        return "no days"
    if interval["lo"] > 0:
        return "better"
    if interval["hi"] < 0:
        return "worse"
    return "no difference"


def score_rows(results, dates, history, day_index, model_names, k=K_BINS, fee=FEE, floor=LOG_FLOOR):
    """
    Per row: the daily mean log-score, hit rates, direction, the fixed
    rule's P&L with the real returns, and the reliability by bin. `day_index`
    maps a game date to its row in the returns history.
    """
    per_row = {}
    names = list(model_names) + list(BASELINE_ROWS)
    for name in names:
        per_row[name] = {"log": {}, "exact": 0, "adjacent": 0, "direction": 0, "positions": 0, "days": 0,
                         "trades": 0, "pnl": 0.0, "predicted_mass": np.zeros(k), "realised": np.zeros(k), "scored_positions": 0}
    for row in results:
        i = row.get("index")
        date = dates[i] if isinstance(i, int) and 0 <= i < len(dates) else None
        key = None if date is None else str(date)[:10]
        if key not in day_index:
            continue
        upto = day_index[key]
        edges = day_edges(history, upto)
        real_returns = history[upto]
        actual = [int(v) for v in row.get("actual_ordered") or row.get("actual") or []]
        if len(actual) != history.shape[1]:
            continue
        for name in names:
            entry = per_row[name]
            prediction = row.get(f"{name}_prediction")
            if prediction and len(prediction) == len(actual):
                entry["days"] += 1
                for pos, (p, a) in enumerate(zip(prediction, actual)):
                    p, a = int(p), int(a)
                    entry["positions"] += 1
                    entry["exact"] += int(p == a)
                    entry["adjacent"] += int(abs(p - a) <= 1)
                    predicted_direction = direction_of(p, edges[pos])
                    real_direction = 1 if real_returns[pos] > 0 else (-1 if real_returns[pos] < 0 else 0)
                    entry["direction"] += int(predicted_direction == real_direction and real_direction != 0)
                    if p >= k / 2:
                        entry["trades"] += 1
                        entry["pnl"] += float(real_returns[pos]) - fee
            slots = row.get(f"{name}_position_scores")
            if slots and len(slots) == len(actual):
                probabilities = helpers.normalize_position_scores(slots)
                logs = []
                for pos, a in enumerate(actual):
                    slot = probabilities[pos] if pos < len(probabilities) else {}
                    if not slot:
                        continue
                    p_actual = float(slot.get(a, slot.get(str(a), 0.0)))
                    logs.append(math.log(max(p_actual, floor)))
                    for b in range(k):
                        entry["predicted_mass"][b] += float(slot.get(b, slot.get(str(b), 0.0)))
                    entry["realised"][a] += 1
                    entry["scored_positions"] += 1
                if logs:
                    entry["log"][key] = float(np.mean(logs))
    return per_row


def summarise(per_row, reference, rng, k=K_BINS):
    uniform = math.log(1.0 / k)
    reference_log = per_row.get(reference, {}).get("log", {})
    out = []
    for name, entry in per_row.items():
        if entry["days"] == 0 and not entry["log"]:
            continue
        logs = entry["log"]
        days = sorted(logs)
        common = [d for d in days if d in reference_log]
        record = {
            "name": name,
            "kind": "market" if name in MARKET_MODEL_NAMES else ("baseline" if name in BASELINE_ROWS else "base"),
            "days": int(entry["days"]), "scored_days": len(days),
            "log_score": float(np.mean([logs[d] for d in days])) if days else None,
            "log_score_se": float(np.std([logs[d] for d in days], ddof=1) / math.sqrt(len(days))) if len(days) > 1 else None,
            "vs_uniform": None, "vs_reference": None,
            "exact_rate": entry["exact"] / entry["positions"] if entry["positions"] else None,
            "adjacent_rate": entry["adjacent"] / entry["positions"] if entry["positions"] else None,
            "direction_rate": entry["direction"] / entry["positions"] if entry["positions"] else None,
            "trades": int(entry["trades"]), "pnl_return": float(entry["pnl"]),
            "pnl_per_trade": float(entry["pnl"] / entry["trades"]) if entry["trades"] else None,
            "reliability": None,
        }
        if days:
            record["vs_uniform"] = bootstrap_interval([logs[d] - uniform for d in days], rng)
            record["vs_uniform"]["verdict"] = verdict(record["vs_uniform"])
        if common and name != reference:
            record["vs_reference"] = bootstrap_interval([logs[d] - reference_log[d] for d in common], rng)
            record["vs_reference"]["verdict"] = verdict(record["vs_reference"])
        n = entry["scored_positions"]
        if n:
            record["reliability"] = {"predicted": [float(v / n) for v in entry["predicted_mass"]],
                                     "realised": [float(v / n) for v in entry["realised"]]}
        out.append(record)
    out.sort(key=lambda r: (r["log_score"] is None, -(r["log_score"] if r["log_score"] is not None else 0), -(r["exact_rate"] or 0)))
    return out


# --- one market -----------------------------------------------------------------
def run_market(market, cfg, root, days, market_only, lockbox, seed):
    dataPath = os.path.join(root, "data", "trainingData", market)
    bestParams = {}
    bestParamsPath = os.path.join(root, f"bestParams_{market}.json")
    if os.path.exists(bestParamsPath):
        with open(bestParamsPath) as handle:
            bestParams = json.load(handle)

    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(cfg["min"], cfg["max"])
    loader.setDrawSize(cfg["draw_size"])
    numbers, _, _ = loader.load_numbers(skipLastColumns=cfg["skip_last_columns"])
    total_rows = len(numbers)
    if total_rows == 0:
        print(f"{market}: no game history, skipping")
        return None
    dates = list(getattr(loader, "dates", []))
    # A game shorter than the window (a week game: 221 weeks against 250 days)
    # keeps a history floor rather than starting at week 0, where every row
    # fails for want of history (8 Oct 2026).
    start_index = max(min(100, total_rows // 2), total_rows - days)

    history_source = MarketHistory(dataPath)
    history_days, history = history_source.visible(0)
    day_index = {d: i for i, d in enumerate(history_days)}

    if market_only:
        models = build_market_models(dataPath, bestParams)
    else:
        models = build_models(dataPath, bestParams, is_positional=True, game=market)
    model_names = list(models)

    # The foundation rows (Chronos by default, TimesFM when enabled) forecast
    # here, in the parent, before the Backtester forks: a foundation model
    # carried into a forked worker refuses to start a worker of its own and
    # the row then scores nothing. Until 5 Oct 2026 this call was missing
    # and every Sunday report lacked the Chronos row - thirty "unavailable"
    # lines in the log, no row, no error (see errors below).
    foundation = prepare_foundation_scores(models, start_index, total_rows,
                                           skipLastColumns=cfg["skip_last_columns"],
                                           specialColumnCount=cfg["special_column_count"],
                                           label=f"{market}: ")

    backtester = Backtester(loader)
    for name, model in models.items():
        backtester.add_model(name, model)
    print(f"\n{market}: backtesting {total_rows - start_index} game days with {len(models)} rows "
          f"({', '.join(n.replace(' Model', '') for n in model_names)})...")
    started = time.time()
    results = backtester.backtest(start_index=start_index, end_index=total_rows, skipLastColumns=cfg["skip_last_columns"],
                                  special_column_count=cfg["special_column_count"], include_baselines=True,
                                  collect_scores=True, verbose=True, game=market)
    elapsed = time.time() - started
    results, locked = lockbox_split(results, dates, lockbox)
    if lockbox:
        print(f"{market}: {describe_lockbox(lockbox)} - {len(locked)} day(s) withheld, {len(results)} scored")
    if not results:
        print(f"{market}: nothing to score")
        return None
    errors = {}
    for row in results:
        for key, value in row.items():
            if key.endswith("_error"):
                errors[key[:-6]] = errors.get(key[:-6], 0) + 1
    for name, count in errors.items():
        print(f"{market}: {name} failed on {count} day(s): "
              f"{next(r[f'{name}_error'] for r in results if f'{name}_error' in r)}")

    rng = np.random.default_rng(seed)
    per_row = score_rows(results, dates, history, day_index, model_names)
    rows = summarise(per_row, GARCH_NAME, rng)
    # A row that scored no day at all is a defect, not a quiet absence: it
    # goes into errors (the page lists those names) and into the log.
    reported = {r["name"] for r in rows}
    for name in model_names:
        if name not in reported:
            why = "no forecast succeeded before the backtest" if foundation.get(name) == 0 else "no scored day"
            if isinstance(errors.get(name), int):      # it raised on every day: keep that count
                why = f"failed on {errors[name]} day(s), {why}"
            errors[name] = why
            print(f"{market}: {name} scored no day - {why}")
    scored_dates = sorted({str(dates[r['index']])[:10] for r in results if isinstance(r.get('index'), int)})
    record = {
        "market": market, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days_requested": days, "days_scored": len(results), "first_day": scored_dates[0] if scored_dates else None,
        "last_day": scored_dates[-1] if scored_dates else None, "k": K_BINS, "uniform_log_score": math.log(1.0 / K_BINS),
        "probability_floor": LOG_FLOOR, "reference": GARCH_NAME, "fee": FEE, "bootstrap_resamples": BOOTSTRAP,
        "market_only": bool(market_only), "rows": rows, "errors": errors, "foundation_days": foundation,
        "lockbox": lockbox_json(lockbox), "lockbox_days_withheld": len(locked),
        "backtest_seconds": round(elapsed, 1), "instruments": history_source.symbols,
    }
    os.makedirs(os.path.dirname(result_path(root, market)), exist_ok=True)
    with open(result_path(root, market), "w") as handle:
        json.dump(record, handle, indent=1)

    print(f"\n{market}: {len(results)} days scored in {elapsed:.0f}s; uniform = {record['uniform_log_score']:.3f}")
    print(f"{'row':<30}{'log-score':>10}{'vs GARCH':>42}{'exact':>8}{'adjacent':>10}{'direction':>10}{'trades':>8}{'P&L':>9}")
    for r in rows:
        ref = r["vs_reference"]
        ref_text = "reference" if r["name"] == GARCH_NAME else (
            f"{ref['mean']:+.3f} [{ref['lo']:+.3f}, {ref['hi']:+.3f}] {ref['verdict']}" if ref else "-")
        pct = lambda v: f"{v * 100:5.1f}%" if v is not None else "    -"
        log_text = f"{r['log_score']:+.3f}" if r["log_score"] is not None else "-"
        print(f"{r['name']:<30}{log_text:>10}{ref_text:>42}"
              f"{pct(r['exact_rate']):>8}{pct(r['adjacent_rate']):>10}{pct(r['direction_rate']):>10}{r['trades']:>8}{r['pnl_return']:>+9.4f}")
    print(f"Record written to {os.path.relpath(result_path(root, market), root)}")
    return record


def main():
    parser = argparse.ArgumentParser(prog="Market rows",
                                     description="Score the market rows under a proper scoring rule against GARCH (README roadmap item 4, M3)")
    parser.add_argument("-g", "--games", default="crypto,shares", help="Comma-separated market games (default both)")
    parser.add_argument("-d", "--days", type=int, default=DEFAULT_DAYS, help=f"Newest game days to backtest (default {DEFAULT_DAYS})")
    parser.add_argument("--market-only", action="store_true", help="Only the four market rows (fast); default adds the positional base rows")
    parser.add_argument("--root", default=os.getcwd(), help="Checkout whose data/ is read and written (default: the working directory)")
    parser.add_argument("--seed", type=int, default=1, help="Bootstrap seed")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown = [g for g in games if g not in GAME_CONFIG or not helpers.is_market_game(g)]
    if unknown:
        print(f"Not a market game: {', '.join(unknown)} (markets: {', '.join(sorted(Helpers.MARKET_GAMES))})")
        return 1
    if is_running():
        print("Another instance is already running. Exiting.")
        return 1
    if not create_lock():
        print("Failed to create lock file. Exiting.")
        return 1
    install_sigterm_handler()
    started = time.time()
    try:
        try:
            lockbox = load_lockbox(args.root)
        except ValueError as e:
            print(f"LOCKBOX ERROR: {e}")
            return 1
        if lockbox:
            print(f"Under the {describe_lockbox(lockbox)}: its days are withheld from every score")
        failures = 0
        for game in games:
            try:
                if run_market(game, GAME_CONFIG[game], args.root, args.days, args.market_only, lockbox, args.seed) is None:
                    failures += 1
            except Exception as e:
                failures += 1
                print(f"{game}: report failed - {e}")
    finally:
        remove_lock()
    print(f"\nDone in {time.time() - started:.0f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
