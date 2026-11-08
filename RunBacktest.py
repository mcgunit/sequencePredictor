#!/usr/bin/env python3
"""
A backtest of one model on one game, started from Admin > Parameters (owner's roadmap step 4, 8 Nov 2026).

The server (backtests.js) writes the request to config/backtests/<id>.json and queues the scheduler's "backtest" job
(jobs.js), which runs this script with --id and a --budget that ends it well before the next scheduled run. The
backtest is the weekend tuner's own evaluation, so its number is the gate's number: the row's tuner objective
(HyperoptStatistics / HyperoptBoost STRATEGIES) is driven by a trial that answers every setting the objective asks
for with the game's served value (the tuned file with the manual overrides, src/ParamOverrides.py) - or, when the
request says "tuned", the tuned file alone - and the code default where the file has none (the gate's defaults,
TuningGate.make_defaults). The Backtester walks the newest `days` draws forward, each day fitted on the history
before it; the lockbox days (lockbox.json) are backtested but never scored and never shown, as in the tuners.

Progress and the result go to the same record file: status queued -> running (done/total days, every few seconds)
-> done | failed | out of time | stopped. Holds process.lock while it runs, like every pipeline script.

Covered: the rows a weekend tuner scores this way - the statistical rows, the market rows (GARCH, Regime HMM) and the
six boosting rows. Not yet: the deep-learning rows (a training per day; with the GPU worker), the quantum rows, the
ensembles, the meta-learners and the RL rows.

    python3 RunBacktest.py --id <id> [--budget SECONDS]
    python3 RunBacktest.py --rows        (the rows and games it covers, as JSON - backtests.js is checked against it)
"""
import argparse
import json
import os
import signal
import sys
import time
import warnings
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join("config", "backtests")
LOCK_FILE = os.path.join(ROOT, "process.lock")
MAX_DAYS = 1000
WRITE_EVERY_SECONDS = 3

# row display name -> (tuner module, strategy key). Kept literal so --rows needs no heavy import.
STATISTICS_ROWS = {
    "Markov Model": "Markov", "MarkovMonteCarlo Model": "MarkovMonteCarlo", "MarkovBayesian Model": "MarkovBayesian",
    "MarkovBayesianEnhanched Model": "MarkovBayesianEnhanced", "PoissonMonteCarlo Model": "PoissonMonteCarlo",
    "PoissonMarkov Model": "PoissonMarkov", "LaplaceMonteCarlo Model": "LaPlaceMonteCarlo",
    "HybridStatisticalModel": "HybridStatistical", "GARCH Model": "Garch", "Regime HMM Model": "RegimeHmm",
}
BOOSTING_ROWS = {
    "XGBoost Model": "XGBoost", "XGBoostMultiLabel Model": "XGBoostMultiLabel", "LightGBM Model": "LightGBM",
    "LightGBMMultiLabel Model": "LightGBMMultiLabel", "CatBoost Model": "CatBoost", "CatBoostMultiLabel Model": "CatBoostMultiLabel",
}
LOTTERY_GAMES = ["lotto", "euromillions", "eurodreams", "vikinglotto", "jokerplus", "pick3", "keno"]
MARKET_GAMES = ["crypto", "shares", "cryptoweek", "sharesweek"]
POSITIONAL = ["pick3", "jokerplus"]
# the rows a positional game does not run (HyperoptStatistics/ModelFactory DISABLED_FOR_PICK3, the multi-label
# boosting rows), and the market rows that only the market games run
NOT_POSITIONAL = {"MarkovBayesian Model", "MarkovBayesianEnhanched Model", "PoissonMarkov Model", "HybridStatisticalModel",
                  "XGBoostMultiLabel Model", "LightGBMMultiLabel Model", "CatBoostMultiLabel Model"}
MARKET_ONLY = {"GARCH Model", "Regime HMM Model"}
# the rows the market games run: their statistical rows on positions, the market rows (no boosting on the markets)
MARKET_ROWS = {"Markov Model", "MarkovMonteCarlo Model", "PoissonMonteCarlo Model", "LaplaceMonteCarlo Model"} | MARKET_ONLY


def games_for(row):
    if row in MARKET_ONLY:
        return list(MARKET_GAMES)
    games = [g for g in LOTTERY_GAMES if not (g in POSITIONAL and row in NOT_POSITIONAL)]
    return games + ([g for g in MARKET_GAMES] if row in MARKET_ROWS else [])


def covered():
    """{row: [games]} - what this script can backtest."""
    return {row: games_for(row) for row in list(STATISTICS_ROWS) + list(BOOSTING_ROWS)}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Record:
    """The request/progress/result file of one backtest, rewritten atomically."""

    def __init__(self, root, backtest_id):
        if not backtest_id or not all(c.isalnum() or c in "-_" for c in backtest_id):
            raise ValueError("a backtest id is letters, digits, - and _")
        self.path = os.path.join(root, RECORD_DIR, f"{backtest_id}.json")
        with open(self.path) as handle:
            self.data = json.load(handle)
        self.last_write = 0.0

    def update(self, force=True, **fields):
        self.data.update(fields)
        if not force and time.time() - self.last_write < WRITE_EVERY_SECONDS:
            return
        self.last_write = time.time()
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(self.data, handle, indent=1)
        os.replace(tmp, self.path)


def make_trial(params, defaults):
    """
    A FixedTrial that answers each setting the objective asks for with `params` (the served or tuned file), else the
    gate's default for it - and records what it answered, so the result says which values were backtested. A value
    outside the search range or the choices is used as it is (scoring what is served is the point; the gate does the
    same).
    """
    import optuna

    class ServedTrial(optuna.trial.FixedTrial):
        def __init__(self):
            super().__init__({})
            self.used = {}
            self.from_default = []

        def _suggest(self, name, distribution):
            if name in params:
                value = params[name]
            else:
                value = defaults(name)
                self.from_default.append(name)
            self._params[name] = value
            self._suggested_params[name] = value
            self._distributions[name] = distribution
            self.used[name] = value
            return value

    return ServedTrial()


def resolve(row, game):
    """(tuner module, objective, defaults table builder) for the row, or raises ValueError with the reason."""
    if row not in covered():
        raise ValueError(f"{row} cannot be backtested here yet (deep-learning, quantum, ensemble, meta-learner and RL rows come later)")
    if game not in covered()[row]:
        raise ValueError(f"{game} does not run {row}")
    if row in STATISTICS_ROWS:
        import HyperoptStatistics as tuner
        strategy = tuner.STRATEGIES[STATISTICS_ROWS[row]]
        return tuner, strategy["objective"], (lambda served: tuner.make_defaults(tuner.SERVED_DEFAULTS, served))
    import HyperoptBoost as tuner
    from src.BoostingBase import BOOSTING_PARAM_SUFFIXES
    objective = tuner.STRATEGIES[BOOSTING_ROWS[row]]["objective"]
    prefix = objective.prefix
    table = {f"{prefix}{suffix}": default for suffix, _, default in BOOSTING_PARAM_SUFFIXES}
    return tuner, objective, (lambda served: tuner.make_defaults(table, served))


def watch_backtests(tuner, record, captured):
    """Wraps the tuner's Backtester.backtest: progress into the record, the day rows and the loader kept for the result."""
    cls = tuner.Backtester
    original = cls.backtest

    def backtest(self, *args, **kwargs):
        captured["names"] = list(self.models)
        captured["loader"] = self.data_loader_model
        start = kwargs.get("start_index", 100)
        own = kwargs.get("progress_callback")
        started = time.time()

        def progress(done, results):
            captured["results"] = results
            total = captured.get("total") or done
            elapsed = time.time() - started
            record.update(force=done >= total, progress={"done": done, "total": total, "seconds": round(elapsed),
                                                         "eta": round(elapsed / done * (total - done)) if done else None})
            if own is not None:
                own(done, results)

        end = kwargs.get("end_index")
        if end is not None:
            captured["total"] = max(0, end - start)
        kwargs["progress_callback"] = progress
        out = original(self, *args, **kwargs)
        captured["results"] = out
        return out

    cls.backtest = backtest


def day_lines(tuner, captured, game):
    """The scored days (lockbox days left out entirely - never shown): date, hits, profit where the game pays."""
    results = captured.get("results") or []
    names = captured.get("names") or []
    loader = captured.get("loader")
    if not results or not names:
        return [], 0
    name = names[0]
    dates = list(getattr(loader, "dates", []) or [])
    scored = tuner.withhold_lockbox(results, dates)
    withheld = len(results) - len(scored)
    lines = []
    for row in scored:
        i = row.get("index")
        date = dates[i] if isinstance(i, int) and 0 <= i < len(dates) else None
        line = {"date": str(date)[:10] if date is not None else None, "hits": row.get(f"{name}_hits")}
        if isinstance(row.get(f"{name}_profit"), (int, float)):
            line["profit"] = round(float(row[f"{name}_profit"]), 2)
        lines.append(line)
    return lines, withheld


def create_lock():
    try:
        with open(LOCK_FILE, "x") as handle:
            handle.write(str(os.getpid()))
        return True
    except FileExistsError:
        try:
            with open(LOCK_FILE) as handle:
                pid = int(handle.read().strip())
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError):
            os.remove(LOCK_FILE)   # a stale lock of a dead run
            return create_lock()
        except PermissionError:
            return False


def remove_lock():
    try:
        with open(LOCK_FILE) as handle:
            if handle.read().strip() == str(os.getpid()):
                os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass


def run(record, budget):
    req = record.data
    game, row, days, values = req.get("game"), req.get("row"), int(req.get("days") or 0), req.get("values", "served")
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be 1 to {MAX_DAYS}")
    tuner, objective, defaults_for = resolve(row, game)
    from src import ParamOverrides as paramOverrides
    from src.Lockbox import load_or_warn
    from HyperoptStatistics import GAME_CONFIG

    root = os.getcwd()
    tuned_path = os.path.join(root, f"bestParams_{game}.json")
    tuned = json.load(open(tuned_path)) if os.path.exists(tuned_path) else {}
    if values == "tuned":
        params, manual = tuned, []
    else:
        params, applied = paramOverrides.apply(tuned, paramOverrides.game_entry(root, game)["overrides"])
        manual = [{"key": k, "value": v, "tuned": t} for k, v, t in applied]
    # the tuner's module globals its objective reads, as its own __main__ sets them
    tuner.LOCKBOX = load_or_warn(root)
    tuner.SERVED_KENO_SUBSETS = tuner.keno_subsets_served(params) if game == "keno" else None
    tuner.TRIAL_TIMEOUT_SECONDS = max(60, int(budget))
    cfg = GAME_CONFIG[game]
    data_path = os.path.join(root, "data", "trainingData", game)

    captured = {}
    watch_backtests(tuner, record, captured)
    trial = make_trial(params, defaults_for(params))
    record.update(status="running", startedAt=now_iso(), pid=os.getpid(), manual=manual,
                  progress={"done": 0, "total": days, "seconds": 0, "eta": None})
    started = time.time()
    import optuna
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = objective(trial, game, data_path, cfg, days, None)
    except optuna.TrialPruned:
        done = (record.data.get("progress") or {}).get("done", 0)
        record.update(status="out of time", finishedAt=now_iso(), seconds=round(time.time() - started),
                      error=f"stopped after {done} of {days} days: the backtest's budget of {round(budget / 60)} min ran out "
                            "(it must end before the next scheduled run); try fewer days")
        return 4
    lines, withheld = day_lines(tuner, captured, game)
    tuning = trial.user_attrs.get("tuning") or {}
    gate = (tuned.get("tuningGate") or {}).get(row) or {}
    finite = isinstance(score, (int, float)) and score == score and score not in (float("inf"), float("-inf"))
    record.update(
        status="done", finishedAt=now_iso(), seconds=round(time.time() - started),
        result={
            "score": float(score) if finite else None, "tuning": tuning, "days": lines, "lockboxWithheld": withheld,
            "values": trial.used, "fromDefault": sorted(set(trial.from_default)),
            # the weekend gate's scores of this row on its own window (comparable when the windows are the same size)
            "gate": {"date": gate.get("date"), "windowDays": gate.get("window_days"), "incumbent": gate.get("incumbent"),
                     "default": gate.get("default"), "decision": gate.get("decision")} if gate else None,
        })
    return 0


def main():
    parser = argparse.ArgumentParser(description="Backtest one model on one game (Admin > Parameters)")
    parser.add_argument("--id", help="the backtest's id (config/backtests/<id>.json)")
    parser.add_argument("--budget", type=float, default=3 * 3600, help="seconds the backtest may take")
    parser.add_argument("--rows", action="store_true", help="print the rows and games covered, as JSON")
    args = parser.parse_args()
    if args.rows:
        print(json.dumps(covered(), sort_keys=True))
        return 0
    os.chdir(ROOT)
    record = Record(ROOT, args.id)
    if not create_lock():
        record.update(status="failed", finishedAt=now_iso(), error="another pipeline run holds process.lock")
        return 3

    def stopped(signum, frame):
        record.update(status="stopped", finishedAt=now_iso(), error="stopped by an administrator")
        remove_lock()
        os._exit(130)

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        return run(record, args.budget)
    except Exception as e:  # noqa: BLE001 - every failure is said in the record, in words
        record.update(status="failed", finishedAt=now_iso(), error=f"{type(e).__name__}: {e}")
        print(f"Backtest failed: {e}")
        return 1
    finally:
        remove_lock()


if __name__ == "__main__":
    sys.exit(main())
