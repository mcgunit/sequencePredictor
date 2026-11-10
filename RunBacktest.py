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
six boosting rows - and, since 10 Nov 2026, the seven deep-learning rows on the lottery games. Not yet: the quantum
rows, the ensembles, the meta-learners, the RL rows and the deep-learning rows on the markets.

The deep-learning rows (GPU_ROWS) are the deep-learning tuner's own evaluation (HyperoptDeepLearning.py): every
backtested draw is a retrain on the history before it, with production's 240 s training budget, and the score is the
tuner's - minus the mean excess loss over the counts (src/DeepLearningTuning.day_score; 0 = as good as serving the
counts). They run on the GPU worker (workers.js copies this checkout there, runs this script and copies the record
back), 3 draws by default (the deep-learning gate's window) and 30 at most (owner, 10 Nov 2026). The lockbox draws
are left out of the score and the day lines; each retrain's own validation split still contains the lockbox period,
as in the tuner (LOCKBOX_NOTE), and the result says so.

    python3 RunBacktest.py --id <id> [--budget SECONDS]
    python3 RunBacktest.py --rows        (the rows and games it covers, as JSON - backtests.js is checked against it)
    python3 RunBacktest.py --gpu-rows    (the rows that run on the GPU worker, as JSON)
"""
import argparse
import json
import math
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
# the deep-learning rows (HyperoptDeepLearning.MODEL_DISPLAY_NAMES): display name -> model type. Lottery games only,
# as the deep-learning tuner (its datasets); the GPU worker runs them
GPU_ROWS = {
    "LSTM Base Model": "lstm_model", "TCN Base Model": "tcn_model", "UnifiedLstmTcn Model": "unified_lstm_tcn_model",
    "UnifiedLstmGruTcn Model": "unified_lstm_gru_tcn_model", "Transformer Model": "transformer_model",
    "GNN Model": "gnn_model", "Autoencoder Model": "autoencoder_model",
}
GPU_DAYS_DEFAULT = 3    # the deep-learning gate's window (src/DeepLearningTuning.DAYS)
GPU_DAYS_MAX = 30
# the columns the deep-learning tuner drops (its datasets): lotto's bonus number
GPU_SKIP_LAST_COLUMNS = {"lotto": 1}
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
    if row in GPU_ROWS:
        return list(LOTTERY_GAMES)
    if row in MARKET_ONLY:
        return list(MARKET_GAMES)
    games = [g for g in LOTTERY_GAMES if not (g in POSITIONAL and row in NOT_POSITIONAL)]
    return games + ([g for g in MARKET_GAMES] if row in MARKET_ROWS else [])


def covered():
    """{row: [games]} - what this script can backtest."""
    return {row: games_for(row) for row in list(STATISTICS_ROWS) + list(BOOSTING_ROWS) + list(GPU_ROWS)}


def days_max(row):
    return GPU_DAYS_MAX if row in GPU_ROWS else MAX_DAYS


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
        raise ValueError(f"{row} cannot be backtested here yet (quantum, ensemble, meta-learner and RL rows come later)")
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


def served_params(root, game, values):
    """(tuned file, the values the backtest runs with, the manual values among them)."""
    from src import ParamOverrides as paramOverrides
    tuned_path = os.path.join(root, f"bestParams_{game}.json")
    tuned = json.load(open(tuned_path)) if os.path.exists(tuned_path) else {}
    if values == "tuned":
        return tuned, tuned, []
    params, applied = paramOverrides.apply(tuned, paramOverrides.game_entry(root, game)["overrides"])
    return tuned, params, [{"key": k, "value": v, "tuned": t} for k, v, t in applied]


def gate_of(tuned, row):
    """The weekend gate's scores of this row on its own window (comparable when the windows are the same size)."""
    gate = (tuned.get("tuningGate") or {}).get(row) or {}
    return {"date": gate.get("date"), "windowDays": gate.get("window_days"), "incumbent": gate.get("incumbent"),
            "default": gate.get("default"), "decision": gate.get("decision")} if gate else None


def run(record, budget):
    record.data.pop("phase", None)   # the worker's word while it was starting (worker/remote-backtest.sh)
    req = record.data
    game, row, days, values = req.get("game"), req.get("row"), int(req.get("days") or 0), req.get("values", "served")
    if not 1 <= days <= days_max(row):
        raise ValueError(f"days must be 1 to {days_max(row)}")
    if row in GPU_ROWS:
        if game not in covered()[row]:
            raise ValueError(f"{game} does not run {row}")
        return run_deep_learning(record, budget, game, row, days, values)
    tuner, objective, defaults_for = resolve(row, game)
    from src.Lockbox import load_or_warn
    from HyperoptStatistics import GAME_CONFIG

    root = os.getcwd()
    tuned, params, manual = served_params(root, game, values)
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
    finite = isinstance(score, (int, float)) and score == score and score not in (float("inf"), float("-inf"))
    record.update(
        status="done", finishedAt=now_iso(), seconds=round(time.time() - started),
        result={
            "score": float(score) if finite else None, "tuning": tuning, "days": lines, "lockboxWithheld": withheld,
            "values": trial.used, "fromDefault": sorted(set(trial.from_default)),
            "gate": gate_of(tuned, row),
        })
    return 0


def run_deep_learning(record, budget, game, row, days, values):
    """
    A deep-learning row: the deep-learning tuner's trial with the served (or tuned) values - `days` retrains of the
    newest draws, each on the history before it with production's training budget, in a spawned child
    (HyperoptDeepLearning.runPredictInChild) - scored as the tuner scores (src/DeepLearningTuning.day_score).
    """
    import threading
    import HyperoptDeepLearning as hdl
    from src import DeepLearningTuning as dlt
    from src.Lockbox import load_or_warn, covers
    from src.TuningGate import make_defaults

    root = os.getcwd()
    tuned, params, manual = served_params(root, game, values)
    model_type = GPU_ROWS[row]
    # Predictor.py's values for a key the file lacks, as the deep-learning gate's defaults
    trial = make_trial(params, make_defaults(dlt.predictor_defaults(hdl.MODEL_PARAM_PREFIX.get(model_type), params), params))
    model_params = hdl.suggest_model_params(trial, model_type)
    data_path = os.path.join(root, "data", "trainingData", game)
    model_path = os.path.join(root, "data", "hyperOptCache", "models", model_type)
    os.makedirs(model_path, exist_ok=True)
    hdl.clearFolder(model_path)
    progress_path = os.path.join(root, "data", "hyperOptCache", f"backtest-progress-{record.data.get('id')}.json")
    hdl.TRIAL_TIMEOUT_SECONDS = max(60, int(budget))
    dates = [entry[0] for entry in (hdl.helpers.getLatestPrediction(data_path, dateRange=days) or [])]

    record.update(status="running", startedAt=now_iso(), pid=os.getpid(), manual=manual,
                  progress={"done": 0, "total": days, "seconds": 0, "eta": None})
    started = time.time()
    finished = threading.Event()

    def watch():
        # the child writes each rebuilt day's value to progress_path (src/DeepLearningTuning.write_progress)
        while not finished.wait(5):
            done = len(dlt.read_progress(progress_path))
            elapsed = time.time() - started
            record.update(force=False, progress={"done": done, "total": days, "seconds": round(elapsed),
                                                 "eta": round(elapsed / done * (days - done)) if done else None})

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    import optuna
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hdl.runPredictInChild(
                None, progress_path, game, model_type, data_path, model_path, f"{game}-gamedata-NL-{datetime.now().year}.csv",
                skipLastColumns=GPU_SKIP_LAST_COLUMNS.get(game, 0), years_back=model_params["yearsOfHistory"],
                daysToRebuild=days, modelParams=model_params, specialColumnCount=hdl.SPECIAL_COLUMN_COUNTS.get(game, 0),
                trainSeconds=dlt.TRAIN_SECONDS)
    except optuna.TrialPruned:
        finished.set()
        watcher.join()
        done = len(dlt.read_progress(progress_path))
        record.update(status="out of time", finishedAt=now_iso(), seconds=round(time.time() - started),
                      error=f"stopped after {done} of {days} draws: the backtest's budget of {round(budget / 60)} min ran out; "
                            "try fewer draws")
        return 4
    finally:
        finished.set()
        hdl.clearFolder(model_path)
        try:
            os.remove(progress_path)
        except OSError:
            pass
    watcher.join()
    if result is None:
        raise RuntimeError("no draw could be rebuilt (no history in data/trainingData, or the newest draw had a prediction already)")
    hits, _, day_excess, day_losses, verdicts = result
    # the lockbox draws: retrained (the tuner does not withhold them) but left out of the score and the day lines
    lockbox = load_or_warn(root)
    if len(dates) != len(day_excess):
        dates = [None] * len(day_excess)
    keep = [i for i, d in enumerate(dates) if not covers(lockbox, d)]
    withheld = len(day_excess) - len(keep)
    tuning = dlt.day_score([day_excess[i] for i in keep], hits=None if withheld else hits,
                           val_losses=[day_losses[i] for i in keep], verdicts=[verdicts[i] for i in keep])
    # the tuner's hits (Helpers.calculate_profit): each retrain's ticket against the next rebuilt draw, so N draws
    # check N - 1 tickets (none for one draw)
    tuning["hits_draws"] = 0 if withheld else max(0, len(day_excess) - 1)
    tuning["lockbox"] = dlt.LOCKBOX_NOTE
    tuning["train_seconds"] = dlt.TRAIN_SECONDS
    lines = [{"date": str(dates[i])[:10] if dates[i] is not None else None, "hits": None,
              "excess": day_excess[i], "verdict": verdicts[i]} for i in keep]
    score = tuning.get("score")
    finite = isinstance(score, (int, float)) and math.isfinite(score)
    record.update(
        status="done", finishedAt=now_iso(), seconds=round(time.time() - started),
        progress={"done": len(day_excess), "total": days, "seconds": round(time.time() - started), "eta": 0},
        result={"score": float(score) if finite else None, "tuning": tuning, "days": lines, "lockboxWithheld": withheld,
                "values": trial.used, "fromDefault": sorted(set(trial.from_default)), "gate": gate_of(tuned, row),
                "worker": os.uname().nodename})
    return 0


def main():
    parser = argparse.ArgumentParser(description="Backtest one model on one game (Admin > Parameters)")
    parser.add_argument("--id", help="the backtest's id (config/backtests/<id>.json)")
    parser.add_argument("--budget", type=float, default=3 * 3600, help="seconds the backtest may take")
    parser.add_argument("--rows", action="store_true", help="print the rows and games covered, as JSON")
    parser.add_argument("--gpu-rows", action="store_true", help="print the rows that run on the GPU worker, as JSON")
    args = parser.parse_args()
    if args.rows:
        print(json.dumps(covered(), sort_keys=True))
        return 0
    if args.gpu_rows:
        print(json.dumps(sorted(GPU_ROWS)))
        return 0
    os.chdir(ROOT)
    record = Record(ROOT, args.id)
    if not create_lock():
        record.update(status="failed", finishedAt=now_iso(), error="another pipeline run holds process.lock")
        return 3

    def stopped(signum, frame):
        # a deep-learning backtest trains in a spawned child: it goes too, or it would keep the GPU busy
        hdl = sys.modules.get("HyperoptDeepLearning")
        executor = getattr(hdl, "CURRENT_EXECUTOR", None) if hdl else None
        for child in list(getattr(executor, "_processes", {}).values()) if executor else []:
            try:
                child.kill()
            except Exception:  # noqa: BLE001 - gone already
                pass
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
