#!/usr/bin/env python3
"""
Q0, second benchmark (README roadmap item 6): the selection effect of TUNING,
measured on data with nothing in it.

NullControls.py measures what the best of the tracked rows scores on a
history that carries no signal - the selection effect over rows. This script
measures the layer on top of it: what the hyperopt itself gains on such a
history. The statistical tuner's own objectives (HyperoptStatistics.STRATEGIES,
the same robust score of src/TuningScore.py, the same served Keno subset
sizes) are run for -t trials on a control history, and the best trial's value
is compared with the untuned defaults scored on the same history - exactly
the comparison the champion/challenger gate makes on the real history every
week (src/TuningGate.py, bestParams_<game>.json["tuningGate"]).

The number this exists for: on a fair synthetic history the tuned best CANNOT
be better than the defaults for any real reason, so whatever it gains there is
the selection effect of picking the best of -t trials on a window. The gate's
real-history gain (challenger minus default, read from the game's bestParams
file) is put next to it: a tuning gain that does not exceed the control band -
mean plus two standard deviations of the control gains - is the effect of
choosing, not of finding.

Like with like, or not at all: the maximum of n trials grows with n, so the
band is only comparable when the control studies ran the same number of
trials on the same window as the gate record was made with (-t and -d default
to the weekly tuner's DEFAULT_TRIALS / DEFAULT_DAYS; a record made with other
values gets no verdict, and says why). Even then the band is a FLOOR on the
selection effect: the weekly study is warm - its sampler is guided by every
completed trial of earlier weeks (the record's study_trials) - while a
control study starts cold, and a guided search finds a lucky window more
surely than a random one.

Costs one study per strategy per control history, in-process (statistical
trials are seconds to a minute), so the defaults are modest: lotto, both
controls, one history each. Records go to data/controls/tuning/<game>-<mode>.json.

  python3 TuningControls.py -g lotto -m both -n 3
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
import warnings
from datetime import datetime

# HyperoptStatistics first: it pins the BLAS/OpenMP thread pools before numpy
# is imported (optuna imports numpy), and every forked Backtester worker
# inherits that pin.
import HyperoptStatistics as HS
import optuna

from src.ControlHistories import build_control_dataset, CONTROL_ROOT
from src.HyperoptRunner import install_sigterm_handler
from src.TuningGate import make_defaults, reference_params
from src.TuningScore import describe

RESULT_DIR = os.path.join(CONTROL_ROOT, "tuning")
SKIPPED_STRATEGIES = ("KenoSubsetTuning",)   # tunes the ensemble subset on a precomputed table, not a row


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strategies_for(game, wanted):
    names = []
    positional = HS.helpers.is_positional_game(game)
    for name in wanted:
        strategy = HS.STRATEGIES.get(name)
        if strategy is None or name in SKIPPED_STRATEGIES:
            continue
        if positional and name in HS.DISABLED_FOR_POSITIONAL:
            continue
        games = strategy.get("games")
        if games and game not in games:
            continue
        names.append(name)
    return names


def tune_on_control(game, name, control_dir, cfg, days, trials, seed, served):
    """
    One in-memory Optuna study of `trials` trials of the strategy's own
    objective on the control history, then the untuned defaults through the
    same objective (a FixedTrial, as the gate scores them). Returns the record
    for this seed, or None when no trial produced a finite value.
    """
    objective = HS.STRATEGIES[name]["objective"]

    def wrapped(trial):
        return objective(trial, game, control_dir, cfg, days, None)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    started = time.time()
    study.optimize(wrapped, n_trials=trials, catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and _finite(t.value)]
    if not done:
        return None
    best = max(done, key=lambda t: t.value)

    default_params = reference_params(list(best.params), {}, make_defaults(HS.SERVED_DEFAULTS, served))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # a default outside the search range is still the default
        try:
            default_value = wrapped(optuna.trial.FixedTrial(default_params))
        except Exception as exc:  # noqa: BLE001 - a pruned/failed default is "no default"
            print(f"    defaults could not be scored: {exc}")
            default_value = None
    if not _finite(default_value):
        default_value = None
    return {
        "seed": seed, "trials": len(done), "seconds": round(time.time() - started, 1),
        "best": float(best.value), "default": default_value,
        "gain": (float(best.value) - default_value) if default_value is not None else None,
        "best_params": best.params, "best_tuning": best.user_attrs.get("tuning"),
    }


def real_gain(served, display_name):
    """The gate's own real-history comparison for this row, when it has run."""
    record = (served.get("tuningGate") or {}).get(display_name) or {}
    challenger, default = record.get("challenger"), record.get("default")
    if _finite(challenger) and _finite(default):
        tuning = record.get("challenger_tuning") or {}
        return {"challenger": challenger, "default": default, "incumbent": record.get("incumbent"),
                "decision": record.get("decision"), "date": record.get("date"), "gain": challenger - default,
                "window_days": record.get("window_days") or tuning.get("days"),
                "run_trials": record.get("run_trials"), "study_trials": record.get("study_trials")}
    return None


def comparable(real, days, trials):
    """Why the real gain may not be held against this band, or None when it may."""
    if not real:
        return "no gate record for this row"
    reasons = []
    if real.get("window_days") and int(real["window_days"]) != int(days):
        reasons.append(f"window {days} here vs {real['window_days']} in the gate record")
    if real.get("run_trials") and int(trials) < int(real["run_trials"]):
        reasons.append(f"{trials} trials here vs {real['run_trials']} in the gate run")
    return "; ".join(reasons) or None


def summarise(per_seed, real, days, trials):
    gains = [r["gain"] for r in per_seed if r and r.get("gain") is not None]
    if not gains:
        return {"gain_mean": None, "gain_sd": None, "band": None, "seeds": 0, "real": real, "real_gain": None,
                "above_band": None, "reason": "no control gain"}
    mean = statistics.fmean(gains)
    sd = statistics.stdev(gains) if len(gains) > 1 else 0.0
    band = mean + 2 * sd
    real_gain_value = real["gain"] if real else None
    reason = comparable(real, days, trials)
    return {"gain_mean": round(mean, 4), "gain_sd": round(sd, 4), "band": round(band, 4), "seeds": len(gains),
            "real": real, "real_gain": round(real_gain_value, 4) if real_gain_value is not None else None,
            "above_band": (real_gain_value > band) if real_gain_value is not None and reason is None else None,
            "reason": reason}


def run(game, mode, seeds, days, trials, wanted, recent, path):
    cfg = HS.GAME_CONFIG[game]
    served_path = os.path.join(path, f"bestParams_{game}.json")
    served = {}
    if os.path.exists(served_path):
        with open(served_path, "r") as handle:
            served = json.load(handle)
    HS.SERVED_KENO_SUBSETS = HS.keno_subsets_served(served) if "keno" in game else None

    names = strategies_for(game, wanted)
    record = {"game": game, "mode": mode, "days": days, "trials": trials, "seeds": list(seeds), "recent": recent or 0,
              "generated_at": datetime.now().isoformat(timespec="seconds"), "strategies": {}}
    per_strategy = {name: [] for name in names}
    for seed in seeds:
        directory, rows = build_control_dataset(game, mode, seed, recent=recent)
        print(f"\n{game} / {mode} / seed {seed}: control history of {rows} draws at {os.path.relpath(directory)}")
        for name in names:
            print(f"  {name}: {trials} trials on {days} days")
            result = tune_on_control(game, name, directory, cfg, days, trials, seed, served)
            if result is None:
                print("    no trial produced a finite value")
                continue
            per_strategy[name].append(result)
            gain = f"{result['gain']:+.4f}" if result["gain"] is not None else "n/a"
            default = "n/a" if result["default"] is None else f"{result['default']:+.4f}"
            print(f"    best {result['best']:+.4f} | default {default} | tuning gain {gain} "
                  f"({result['seconds']}s) | {describe(result['best_tuning'])}")

    for name in names:
        display = HS.STRATEGY_DISPLAY_NAMES.get(name, name)
        summary = summarise(per_strategy[name], real_gain(served, display), days, trials)
        summary["per_seed"] = per_strategy[name]
        record["strategies"][display] = summary

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{game}-{mode}.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    return record


def main():
    parser = argparse.ArgumentParser(
        prog="Tuning controls",
        description="What the hyperopt gains on a history with nothing in it - the selection effect of tuning (README roadmap item 6, Q0)")
    parser.add_argument("-g", "--games", default="lotto", help='Comma-separated games (default lotto)')
    parser.add_argument("-m", "--mode", default="both", choices=["synthetic", "shuffled", "both"])
    parser.add_argument("-n", "--repetitions", type=int, default=1,
                        help="Control histories per game/mode (default 1; the band has no width below 3)")
    parser.add_argument("-t", "--trials", type=int, default=HS.DEFAULT_TRIALS,
                        help=f"Trials per study (default {HS.DEFAULT_TRIALS}, the weekly tuner's; fewer trials give a "
                             "narrower band than the gate's own choosing and the verdict is then withheld)")
    parser.add_argument("-d", "--days", type=int, default=HS.DEFAULT_DAYS,
                        help=f"Backtest window per trial (default {HS.DEFAULT_DAYS}, the weekly tuner's; another window "
                             "is not comparable with the gate record and the verdict is withheld)")
    parser.add_argument("-s", "--strategies", default=",".join(HS.STRATEGIES.keys()),
                        help="Comma-separated HyperoptStatistics strategies (default all that apply)")
    parser.add_argument("--seed", type=int, default=1, help="First control seed; repetitions count up from it")
    parser.add_argument("--recent", type=int, default=0,
                        help="Build the control from the newest N draws only (0 = the whole history)")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown = [g for g in games if g not in HS.GAME_CONFIG]
    if unknown:
        print(f"Unknown games: {', '.join(unknown)} (known: {', '.join(HS.GAME_CONFIG)})")
        return 1
    modes = ["synthetic", "shuffled"] if args.mode == "both" else [args.mode]
    seeds = list(range(args.seed, args.seed + max(1, args.repetitions)))
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    records = []
    for game in games:
        for mode in modes:
            records.append(run(game, mode, seeds, args.days, args.trials, wanted, args.recent, os.getcwd()))

    print("\n" + "=" * 120)
    print(f"{'game':<13}{'control':<11}{'strategy':<30}{'tuning gain on nothing':>24}{'band':>9}{'real gain':>11}   verdict")
    for record in records:
        for display, s in record["strategies"].items():
            if s["gain_mean"] is None:
                print(f"{record['game']:<13}{record['mode']:<11}{display:<30}{'-':>24}")
                continue
            gain = f"{s['gain_mean']:+.4f} +/- {s['gain_sd']:.4f} (n={s['seeds']})"
            real = f"{s['real_gain']:+.4f}" if s["real_gain"] is not None else "no gate record"
            verdict = (f"no verdict: {s['reason']}" if s["above_band"] is None else
                       "real gain ABOVE the control band" if s["above_band"] else "within what choosing alone gives")
            if s["seeds"] == 1 and s["above_band"] is not None:
                verdict += " (one history: the band has no width, run -n 3 for one)"
            print(f"{record['game']:<13}{record['mode']:<11}{display:<30}{gain:>24}{s['band']:>+9.4f}{real:>11}   {verdict}")
    print("=" * 120)
    print("tuning gain = best of the trials minus the untuned defaults, both scored with the tuner's own objective on the same history.")
    print("The band is a floor on the selection effect: the weekly study is warm (guided by earlier weeks' trials), a control study is cold.")
    print(f"Records written to {os.path.relpath(RESULT_DIR)}/<game>-<control>.json")
    return 0


if __name__ == "__main__":
    # The same process.lock every other entry point takes: a study here is a
    # backtest per trial, and the scheduler and a hand-run predictor decide
    # from the lock whether the box is busy. SIGTERM releases it like the
    # tuners do (the handler turns the signal into SystemExit).
    install_sigterm_handler()
    if HS.is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)
    if not HS.create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)
    try:
        sys.exit(main())
    finally:
        HS.remove_lock()
