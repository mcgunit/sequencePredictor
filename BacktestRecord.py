#!/usr/bin/env python3
"""
The backtest record of the lottery rows (roadmap item 13, decided 21 and 22 Nov 2026): for every row a backtest
covers, how each of ten past calendar months ended had its ticket been played on every draw.

Per game one file, data/backtestRecord/<game>.json: the months, per row each month's draws, bets, stake and profit
with the settings that were backtested, and the same months played by random tickets (src/BacktestMonths.py). The
pages read it through backtestRecord.js: "profit in 3 of 10 backtested months; a random ticket: 2.1 on average".

  the backtest    the weekend tuner's own evaluation of the row with the served settings (RunBacktest.py: the tuned
                  file with the manual overrides), walked over whole months instead of the newest days. Each draw is
                  fitted on the history before it.
  the months      the ten newest complete calendar months outside the lockbox (its days stay shut).
  kept            a month backtested with the settings a row still runs on is not backtested again: a normal week
                  adds the month that ended and the rows whose settings changed. A row whose settings changed is
                  backtested anew, all ten months.
  not covered     the quantum rows, the ensembles, the meta-learners, the RL rows, the Council rows and - until they
                  run on the workers - the deep-learning rows: they have no entry, and the pages say "not tested yet".
  the limit       the tuners chose a row's settings with these months in view, so the count flatters a tuned row.

Runs last in the weekend chain (jobs.js), behind the tuners, and holds process.lock like every pipeline script. With
--budget it starts no new row once the time is up; the next run goes on where this one stopped.

    python3 BacktestRecord.py [-g lotto,keno] [--rows "Markov Model,XGBoost Model"] [--budget SECONDS] [--today YYYY-MM-DD]
    python3 BacktestRecord.py --self-check      (in npm test: two rows of one game on a copy of the history)
"""
import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join("data", "backtestRecord")
VERSION = 1
# How a game's money is booked, where that changed: a record booked the old way is thrown away and built anew.
# vikinglotto 2: a grid costs 10.00 EUR, five times the base price (PrizeTables.GRID_PLAYS, 23 Nov 2026).
MONEY = {"vikinglotto": 2}
# A row whose backtest ran out of time (27 Nov 2026: Keno's XGBoost row needs about 4.5 hours, the job has two; it was
# started, cut off and started again every day, with the other Keno boosting rows waiting behind it). Such a row is
# marked `slow` with the settings it ran on and is not started again until those settings change. Only a row that had
# at least SLOW_SECONDS of its own counts: one cut off because the job's time was nearly up is left for the next run.
SLOW_SECONDS = 3600


def log(text):
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {text}", flush=True)


def record_path(root, game):
    return os.path.join(root, RECORD_DIR, f"{game}.json")


def load_record(root, game):
    try:
        with open(record_path(root, game)) as handle:
            data = json.load(handle)
        return data if data.get("version") == VERSION and data.get("money", 1) == MONEY.get(game, 1) else {}
    except (OSError, ValueError):
        return {}


def write_record(root, game, data):
    path = record_path(root, game)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, separators=(",", ":"), sort_keys=True)
    os.replace(temp, path)


def served_now(values, from_default, params):
    """True when the settings a row was backtested with are the ones it is served with now."""
    for key, value in (values or {}).items():
        if key in params:
            if params[key] != value:
                return False
        elif key not in (from_default or []):
            return False   # it had a tuned or manual value then, and has none now
    return True


def full_draws(root, game, config):
    """[(ISO day, the draw's full row in drawn order)] - with Lotto's bonus number, which the backtest leaves out."""
    from src.DataLoader import DataLoader
    loader = DataLoader()
    loader.setDataPath(os.path.join(root, "data", "trainingData", game))
    loader.setGameRange(config["min"], config["max"])
    loader.setDrawSize(config["draw_size"])
    numbers, _, _ = loader.load_numbers(skipLastColumns=0)
    return [(str(day)[:10], [int(v) for v in row]) for day, row in zip(loader.dates, numbers)]


SPAN = {}       # the days the patched Backtester walks: {"first": ISO day, "last": ISO day}, and what it captured


def patch_backtester(tuner):
    """The tuner's Backtester walks SPAN's days instead of the newest ones, and keeps its rows, names and dates."""
    cls = tuner.Backtester
    if getattr(cls, "_backtest_record_patched", False):
        return
    original = cls.backtest

    def backtest(self, *args, **kwargs):
        dates = [str(day)[:10] for day in (getattr(self.data_loader_model, "dates", []) or [])]
        inside = [i for i, day in enumerate(dates) if SPAN["first"] <= day <= SPAN["last"]]
        SPAN.update(names=list(self.models), dates=dates, results=[])
        if not inside:
            return []
        kwargs["start_index"], kwargs["end_index"] = inside[0], inside[-1] + 1
        out = original(self, *args, **kwargs)
        SPAN["results"] = out
        return out

    cls.backtest = backtest
    cls._backtest_record_patched = True


def backtest_span(root, game, row, first, last, params, seconds):
    """(result rows, model name, dates, values used, values from the code default) of `row` over the days first..last."""
    import RunBacktest as rb
    from HyperoptStatistics import GAME_CONFIG
    from src.Lockbox import load_or_warn
    tuner, objective, defaults_for = rb.resolve(row, game)
    tuner.LOCKBOX = load_or_warn(root, out=lambda *_: None)
    tuner.SERVED_KENO_SUBSETS = tuner.keno_subsets_served(params) if game == "keno" else None
    tuner.TRIAL_TIMEOUT_SECONDS = max(60, int(seconds))
    patch_backtester(tuner)
    SPAN.clear()
    SPAN.update(first=first, last=last, results=[], names=[], dates=[])
    trial = rb.make_trial(params, defaults_for(params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            objective(trial, game, os.path.join(root, "data", "trainingData", game), GAME_CONFIG[game], 1, None)
        except Exception as error:  # noqa: BLE001 - the score is not what is read here; the walked days are
            if not SPAN.get("results"):
                # the settings it ran on travel with the failure: a row out of time is marked with them
                error.backtest_values, error.backtest_default = dict(trial.used), sorted(set(trial.from_default))
                raise
            log(f"    (the tuner's own score failed after the walk: {type(error).__name__}: {error})")
    scored = tuner.withhold_lockbox(SPAN.get("results") or [], SPAN.get("dates") or [])
    name = (SPAN.get("names") or [None])[0]
    return scored, name, SPAN.get("dates") or [], dict(trial.used), sorted(set(trial.from_default))


def run_row(root, game, row, months, params, context, seconds):
    """A row's months, backtested: ({month: totals}, plays, values, from default). `seconds` is for all of them."""
    from src import BacktestMonths as bm
    lines, plays, values, from_default = [], set(), {}, []
    until = time.time() + seconds
    for first, last, _run in bm.segments(months):
        results, name, dates, values, from_default = backtest_span(root, game, row, first, last, params, until - time.time())
        for result in results:
            index = result.get("index")
            day = dates[index] if isinstance(index, int) and 0 <= index < len(dates) else None
            if day is None:
                continue
            money = bm.draw_money(game, context["helpers"], name, result, context["prizes"].get(day), context["typical"],
                                  context["bonus"].get(day))
            if money is not None:
                plays.update(money[3])
            lines.append((day, money))
    return bm.month_totals(lines, months), sorted(plays, key=lambda p: (len(p), p)), values, from_default


def game_context(root, game, config):
    """What a game's draws need to be turned into money: helpers, prize tables, Lotto's bonus numbers, the draws."""
    from src import BacktestMonths as bm
    from src import PrizeTables
    from src.Helpers import Helpers
    draws = full_draws(root, game, config)
    prizes = PrizeTables.load(os.path.join(root, "data", "trainingData", game), game) if game in bm.PRIZE_GAMES else {}
    prizes = {PrizeTables.day_key(day) or day: entry for day, entry in prizes.items()}
    return {
        "helpers": Helpers(), "draws": draws, "prizes": prizes, "typical": PrizeTables.typical_amounts(prizes),
        "bonus": {day: values[-1] for day, values in draws} if game == "lotto" else {},
        "specials": bm.special_range([values for _, values in draws[-300:]], config["draw_size"]),
    }


def random_months(game, months, config, context, kept, sims):
    """{month: {play: [each random player's total]}} - a month simulated before with as many players is kept."""
    from src import BacktestMonths as bm
    out = {}
    for month in months:
        before = (kept or {}).get(month)
        if before and all(len(series) == sims for series in before.values()):
            out[month] = before
            continue
        first, last = bm.month_span(month)
        draws = [(day, values) for day, values in context["draws"] if first <= day <= last]
        out[month] = bm.random_month(game, context["helpers"], month, draws, config, context["prizes"], context["typical"],
                                     context["specials"], sims)
    return out


def run_game(root, game, months, lockbox, only_rows=None, deadline=None, sims=None, today=None):
    """Brings one game's record up to date; returns (rows backtested, rows kept, rows left for the next run)."""
    import RunBacktest as rb
    from HyperoptStatistics import GAME_CONFIG
    from src import BacktestMonths as bm
    config = GAME_CONFIG[game]
    record = load_record(root, game)
    context = game_context(root, game, config)
    _tuned, params, _manual = rb.served_params(root, game, "served")
    rows = [row for row in list(rb.STATISTICS_ROWS) + list(rb.BOOSTING_ROWS)
            if game in rb.games_for(row) and (not only_rows or row in only_rows)]
    old_rows = record.get("rows") or {}
    record = {"version": VERSION, "money": MONEY.get(game, 1), "game": game, "months": months, "lockbox": lockbox or None,
              "random": {"sims": sims or bm.SIMS,
                         "months": random_months(game, months, config, context, (record.get("random") or {}).get("months"), sims or bm.SIMS)},
              "rows": {name: entry for name, entry in old_rows.items() if name in rb.covered() and game in rb.games_for(name)}}
    done = kept = left = 0
    # a row that failed before goes behind the others: it cannot hold them up again
    rows = sorted(rows, key=lambda name: 1 if (record["rows"].get(name) or {}).get("error") else 0)
    for row in rows:
        entry = record["rows"].get(row) or {}
        same = bool(entry) and served_now(entry.get("values"), entry.get("fromDefault"), params)
        if entry.get("slow") and same:
            kept += 1   # too slow on these settings: not started again until they change
            continue
        have = {m: t for m, t in (entry.get("months") or {}).items() if m in months} if same else {}
        missing = [m for m in months if m not in have]
        if not missing:
            if entry.get("months") != have:
                entry["months"] = have
            kept += 1
            continue
        if deadline is not None and time.time() >= deadline:
            left += 1
            continue
        started = time.time()
        try:
            allowed = (deadline - time.time()) if deadline else 6 * 3600
            totals, plays, values, from_default = run_row(root, game, row, missing, params, context, allowed)
        except Exception as error:  # noqa: BLE001 - one row's failure is said on the row; the others go on
            failed = {"error": f"{type(error).__name__}: {error}", "errorAt": now_iso()}
            if type(error).__name__ == "TrialPruned":   # the walk was cut off at its time limit
                if allowed >= SLOW_SECONDS and hasattr(error, "backtest_values"):
                    failed.update(slow={"at": now_iso(), "seconds": round(time.time() - started), "allowed": round(allowed)},
                                  values=error.backtest_values, fromDefault=error.backtest_default)
                    log(f"  {game} {row}: too slow ({round(time.time() - started)} s was not enough) - not started again until its settings change")
                else:
                    left += 1
                    log(f"  {game} {row}: cut off after {round(time.time() - started)} s, the job's time was up - left for the next run")
            else:
                log(f"  {game} {row}: failed ({type(error).__name__}: {error})")
            # months backtested on the same settings stand; an older failure's marks do not
            base = {k: v for k, v in entry.items() if k not in ("error", "errorAt", "slow")} if same else {}
            record["rows"][row] = {**base, **failed}
            write_record(root, game, {**record, "updatedAt": now_iso()})
            continue
        if not served_now(values, from_default, params) or (same and entry.get("values") != values):
            have = {}   # the settings asked for differ from the ones kept: nothing of the old months stands
        all_months = {**have, **totals}
        if len(all_months) < len(months):   # the kept months were dropped just above: the next run backtests them
            left += 1
        record["rows"][row] = {"values": values, "fromDefault": from_default, "plays": plays or entry.get("plays") or [],
                               "months": {m: all_months[m] for m in months if m in all_months},
                               "testedAt": now_iso(), "seconds": round(time.time() - started)}
        won, played = bm.profit_months(record["rows"][row]["months"])
        log(f"  {game} {row}: {len(missing)} month(s) backtested in {round(time.time() - started)} s - profit in {won} of {played}")
        done += 1
        write_record(root, game, {**record, "updatedAt": now_iso()})
    write_record(root, game, {**record, "updatedAt": now_iso()})
    return done, kept, left


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(root, games, only_rows=None, budget=None, today=None, sims=None):
    from src import BacktestMonths as bm
    try:
        with open(os.path.join(root, "lockbox.json")) as handle:
            lockbox = {k: v for k, v in json.load(handle).items() if k in ("from", "to", "declared")}
    except (OSError, ValueError):
        lockbox = None
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    months = bm.wanted_months(today, lockbox)
    deadline = time.time() + budget if budget else None
    log(f"Backtest record over {months[0]} .. {months[-1]} ({len(months)} months"
        + (f", outside the lockbox {lockbox['from']} to {lockbox['to']}" if lockbox else "") + ")")
    left_total = 0
    for game in games:
        started = time.time()
        done, kept, left = run_game(root, game, months, lockbox, only_rows, deadline, sims, today)
        left_total += left
        log(f"{game}: {done} row(s) backtested, {kept} kept, {left} left for the next run ({round(time.time() - started)} s)")
    if left_total:
        log(f"out of time: {left_total} row(s) left for the next run")
    return 0


def _self_check():
    """Two cheap rows of Pick3 on a copy of its history: a first run, a kept second run, a changed setting."""
    import shutil
    scratch = tempfile.mkdtemp(prefix="backtest-record-")
    try:
        for part in (os.path.join("data", "trainingData", "pick3"),):
            shutil.copytree(os.path.join(ROOT, part), os.path.join(scratch, part))
        for name in ("bestParams_pick3.json", "lockbox.json"):
            if os.path.exists(os.path.join(ROOT, name)):
                shutil.copy(os.path.join(ROOT, name), os.path.join(scratch, name))
        cwd = os.getcwd()
        os.chdir(scratch)
        try:
            rows = ["Markov Model", "PoissonMonteCarlo Model"]
            quiet = globals()["log"]
            globals()["log"] = lambda text: None
            run(scratch, ["pick3"], rows, None, "2026-11-22", 40)
            first = load_record(scratch, "pick3")
            assert first["months"][0] == "2025-10" and first["months"][-1] == "2026-10" and len(first["months"]) == 10, first["months"]
            assert sorted(first["rows"]) == sorted(rows), sorted(first["rows"])
            entry = first["rows"]["Markov Model"]
            assert sorted(entry["months"]) == sorted(first["months"]) and entry["plays"] == ["ticket"], entry
            october = entry["months"]["2026-10"]
            assert october["draws"] == 31 and october["bets"] == 31 and 93 <= october["stake"] <= 124 and october["unscored"] == 0, october
            assert all(len(first["random"]["months"][m]["ticket"]) == 40 for m in first["months"])
            assert not any("2025-12" <= m <= "2026-02" for m in first["months"]), "no lockbox month"
            # a second run keeps every month: nothing is backtested again
            stamp = entry["testedAt"]
            done, kept, left = run_game(scratch, "pick3", first["months"], first["lockbox"], rows, None, 40)
            assert (done, kept, left) == (0, 2, 0), (done, kept, left)
            assert load_record(scratch, "pick3")["rows"]["Markov Model"]["testedAt"] == stamp
            # a changed setting: that row alone is backtested anew, all its months
            path = os.path.join(scratch, "bestParams_pick3.json")
            params = json.load(open(path)) if os.path.exists(path) else {}
            key = next(k for k in entry["values"] if isinstance(entry["values"][k], (int, float)) and not isinstance(entry["values"][k], bool))
            params[key] = entry["values"][key] * 2 + 1
            json.dump(params, open(path, "w"))
            others = load_record(scratch, "pick3")["rows"]["PoissonMonteCarlo Model"]
            shared = key in others["values"]
            done, kept, left = run_game(scratch, "pick3", first["months"], first["lockbox"], rows, None, 40)
            assert done >= 1 and done + kept == 2 and left == 0 and (shared or done == 1), (done, kept, left, shared)
            again = load_record(scratch, "pick3")["rows"]["Markov Model"]
            assert again["values"][key] == params[key] and len(again["months"]) == 10, again["values"]
            # out of time before the first row: everything is left, the file stays readable
            params[key] = entry["values"][key]
            json.dump(params, open(path, "w"))
            done, kept, left = run_game(scratch, "pick3", first["months"], first["lockbox"], ["Markov Model"], time.time() - 1, 40)
            assert (done, left) == (0, 1), (done, kept, left)
            # a row cut off at its time limit: with an hour or more of its own it is marked slow and passed over until
            # its settings change, and it goes behind the other rows; cut off with less, it is left for the next run
            class TrialPruned(Exception):
                pass
            real_row = globals()["run_row"]
            asked = []

            def cut_off(root, game, row, months, row_params, context, seconds):
                asked.append(row)
                if row != "Markov Model":
                    return real_row(root, game, row, months, row_params, context, seconds)
                error = TrialPruned()
                error.backtest_values, error.backtest_default = dict(entry["values"]), list(entry.get("fromDefault") or [])
                raise error
            globals()["run_row"] = cut_off
            try:
                os.remove(record_path(scratch, "pick3"))
                done, kept, left = run_game(scratch, "pick3", first["months"], first["lockbox"], rows, time.time() + 600, 40)
                marked = load_record(scratch, "pick3")["rows"]["Markov Model"]
                assert (done, left) == (1, 1) and "slow" not in marked and marked.get("error"), (done, kept, left, marked)
                asked.clear()
                done, kept, left = run_game(scratch, "pick3", first["months"], first["lockbox"], rows, time.time() + 7200, 40)
                marked = load_record(scratch, "pick3")["rows"]["Markov Model"]
                assert (done, kept, left) == (0, 1, 0) and marked["slow"]["allowed"] >= 3600 and not marked.get("months"), (done, kept, left, marked)
                assert asked == ["Markov Model"], asked
                asked.clear()
                done, kept, left = run_game(scratch, "pick3", first["months"], first["lockbox"], rows, time.time() + 7200, 40)
                assert (done, kept, left) == (0, 2, 0) and asked == [], ("a slow row is not started again", done, kept, left, asked)
                params[key] = entry["values"][key] * 2 + 1
                json.dump(params, open(path, "w"))
                run_game(scratch, "pick3", first["months"], first["lockbox"], rows, time.time() + 7200, 40)
                assert asked and asked[-1] == "Markov Model", ("changed settings: tried again, behind the others", asked)
                params[key] = entry["values"][key]
                json.dump(params, open(path, "w"))
            finally:
                globals()["run_row"] = real_row
            assert served_now({"a": 1}, [], {"a": 1}) and not served_now({"a": 1}, [], {"a": 2})
            assert served_now({"a": 1}, ["a"], {}) and not served_now({"a": 1}, [], {}), "a value that was tuned then and is gone now is a change"
            globals()["log"] = quiet
        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("BacktestRecord self-check OK")


def main():
    import RunBacktest as rb
    parser = argparse.ArgumentParser(description="The lottery rows' backtest record over ten calendar months")
    parser.add_argument("-g", "--games", default=",".join(rb.LOTTERY_GAMES))
    parser.add_argument("--rows", default="", help="only these rows (display names, comma separated)")
    parser.add_argument("--budget", type=float, default=None, help="seconds after which no new row is started")
    parser.add_argument("--today", default=None, help="the day the months are counted back from (default: today, UTC)")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return 0
    games = [g for g in args.games.split(",") if g in rb.LOTTERY_GAMES]
    os.chdir(ROOT)
    if not rb.create_lock():
        print("another pipeline run holds process.lock")
        return 3
    try:
        return run(ROOT, games, [r.strip() for r in args.rows.split(",") if r.strip()] or None, args.budget, args.today)
    finally:
        rb.remove_lock()


if __name__ == "__main__":
    sys.exit(main())
