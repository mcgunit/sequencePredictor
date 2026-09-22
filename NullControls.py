#!/usr/bin/env python3
"""
Q0 null controls (README roadmap item 6): how good does the BEST row look on
data that is known to carry no signal?

Every ranking on the History page is a maximum over rows, and the maximum of
N noisy estimates sits above their mean even when every single row is
worthless. Roughly twenty-five rows are tracked today, so "the best model
averaged 1.1 hits a draw" is unreadable until the same pipeline has been run
against data where the true answer is known to be "nothing". That number is
what this script produces, and it is the precondition for reading any
quantum-versus-classical comparison (Q2) or any foundation-model row.

TWO CONTROLS, both through the production evaluation path (src/ModelFactory.py
builds the same models, src/Backtester.py scores them the same way, with the
same random/global-frequency/column-frequency baselines):

  synthetic  draws generated under the game's own rules, uniformly at random.
             No structure of any kind exists to be found: not a trend, not a
             bias, not a repeat. Anything a row scores above the random
             baseline here is the selection effect and nothing else.

  shuffled   the real draws, reordered. Every marginal is preserved exactly -
             number frequencies, co-occurrence, how "hot" each number looks -
             and only the time order is destroyed. A row that beats this
             control is using time, which is the one thing a fair draw cannot
             offer. A row that does not is reading frequencies that a fair
             process reproduces anyway.

WHAT THIS VERSION DOES NOT DO. The full control the roadmap describes also
re-runs the hyperopt on the control history, which measures the selection
effect of TUNING on top of the selection effect over rows. That costs hours
per game per seed, so this runs the models on their untuned defaults - which
is the honest choice for a control, since parameters tuned on the real
history mean nothing on a synthetic one. The gap this measures is therefore a
LOWER bound on the real one. Same reason the foundation rows are off by
default (--foundation turns them on): they cost a precompute per seed.

  python3 NullControls.py -g lotto -m both -n 3 -d 120
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np
from dateutil.parser import parse as parse_date

from src.Backtester import Backtester
from src.DataLoader import DataLoader
from src.Helpers import Helpers
from src.ControlHistories import build_control_dataset, CONTROL_ROOT
from src.ModelFactory import build_models, prepare_foundation_scores
from HyperoptStatistics import GAME_CONFIG

helpers = Helpers()

RESULT_DIR = os.path.join(CONTROL_ROOT, "results")
BASELINE_ROWS = ("random", "global_frequency", "column_frequency")


# --- scoring it exactly like production --------------------------------------
def evaluate(game, dataPath, days, foundation=False):
    cfg = GAME_CONFIG[game]
    positional = helpers.is_positional_game(game)

    loader = DataLoader()
    loader.setDataPath(dataPath)
    loader.setGameRange(cfg["min"], cfg["max"])
    loader.setDrawSize(cfg["draw_size"])
    numbers, _, _ = loader.load_numbers(skipLastColumns=cfg["skip_last_columns"])
    total_rows = len(numbers)
    start_index = max(0, total_rows - days)

    # Untuned defaults on purpose - see the header. The foundation rows are
    # off unless asked for, and then they need the same pre-fork precompute
    # every other collector does.
    bestParams = {"useChronosFeature": bool(foundation), "useTimesFmFeature": False}
    models = build_models(dataPath, bestParams, is_positional=positional)
    if foundation:
        prepare_foundation_scores(models, start_index, total_rows,
                                  skipLastColumns=cfg["skip_last_columns"],
                                  specialColumnCount=cfg["special_column_count"],
                                  label=f"{game} control: ")

    backtester = Backtester(loader)
    for name, model in models.items():
        backtester.add_model(name, model)

    results = backtester.backtest(
        start_index=start_index,
        end_index=total_rows,
        skipLastColumns=cfg["skip_last_columns"],
        special_column_count=cfg["special_column_count"],
        include_baselines=True,
        collect_scores=False,
        verbose=False,
        game=game if (positional or game == "keno") else None,
    )
    summary = backtester.summarize(results)
    # Backtester.summarize: models[name]["main"]["hits"] is the mean hits of
    # the row's main ticket, which is the number the History page ranks on.
    scores = {}
    for name, entry in (summary.get("models") or {}).items():
        value = (entry.get("main") or {}).get("hits", entry.get("hits_avg"))
        if value is not None:
            scores[name] = float(value)
    return scores, len(results)


# --- the number this exists for ----------------------------------------------
def selection_report(per_seed):
    """
    per_seed: [{row name: hits_avg}, ...], one dict per repetition.

    The headline is the gap between the best row and the average row on data
    with nothing in it. That gap is what a leaderboard buys you for free.
    """
    model_rows = sorted({name for seed in per_seed for name in seed if name not in BASELINE_ROWS})
    winners, means, spreads = [], [], []
    for seed in per_seed:
        values = [seed[name] for name in model_rows if name in seed]
        if not values:
            continue
        winners.append(max(values))
        means.append(sum(values) / len(values))
        spreads.append(max(values) - min(values))
    baseline = [seed["random"] for seed in per_seed if "random" in seed]
    report = {
        "rows_compared": len(model_rows),
        "row_mean": round(statistics.fmean(means), 4) if means else None,
        "best_row_mean": round(statistics.fmean(winners), 4) if winners else None,
        "selection_gain": round(statistics.fmean(winners) - statistics.fmean(means), 4) if winners else None,
        "best_to_worst_spread": round(statistics.fmean(spreads), 4) if spreads else None,
        "random_baseline": round(statistics.fmean(baseline), 4) if baseline else None,
        "per_row_mean": {name: round(statistics.fmean([s[name] for s in per_seed if name in s]), 4)
                         for name in model_rows + [b for b in BASELINE_ROWS if any(b in s for s in per_seed)]},
    }
    if winners and len(winners) > 1:
        report["best_row_sd_across_seeds"] = round(statistics.stdev(winners), 4)
    return report


def run(game, mode, seeds, days, foundation, recent=None):
    per_seed, winners = [], []
    for seed in seeds:
        started = time.time()
        directory, rows_written = build_control_dataset(game, mode, seed, recent=recent)
        scores, scored_days = evaluate(game, directory, days, foundation)
        per_seed.append(scores)
        best = max(((v, k) for k, v in scores.items() if k not in BASELINE_ROWS), default=(0, "-"))
        winners.append(best[1])
        print(f"  {game} {mode} seed {seed}: {scored_days} days, {len(scores)} rows, "
              f"best {best[1]} at {best[0]:.3f} hits/draw, random baseline "
              f"{scores.get('random', float('nan')):.3f} ({time.time() - started:.0f}s)")

    report = selection_report(per_seed)
    report.update({"game": game, "mode": mode, "seeds": list(seeds), "days": days, "recent": recent or 0,
                   "foundation_rows": bool(foundation),
                   "winner_per_seed": winners, "generated_at": datetime.now().isoformat(timespec="seconds")})
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{game}-{mode}.json"), "w") as handle:
        json.dump({"report": report, "per_seed": per_seed}, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(
        prog="Null controls",
        description="How good the best tracked row looks on data known to carry no signal (README roadmap item 6, Q0)")
    parser.add_argument("-g", "--games", default="lotto",
                        help='Comma-separated games, e.g. "lotto,pick3" (default lotto)')
    parser.add_argument("-m", "--mode", default="both", choices=["synthetic", "shuffled", "both"])
    parser.add_argument("-n", "--repetitions", type=int, default=3, help="How many control histories per game/mode")
    parser.add_argument("-d", "--days", type=int, default=120, help="Days to backtest in each control history")
    parser.add_argument("--seed", type=int, default=1, help="First seed; repetitions count up from it")
    parser.add_argument("--recent", type=int, default=0,
                        help="Build the control from the newest N draws only, where the game's rules have "
                             "not changed (0 = the whole history). Matters most for the shuffled control: "
                             "permuting thirty years and then scoring its tail mixes every rule era.")
    parser.add_argument("--foundation", action="store_true",
                        help="Include the Chronos row (costs a forecast pass per control history)")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown = [g for g in games if g not in GAME_CONFIG]
    if unknown:
        print(f"Unknown games: {', '.join(unknown)} (known: {', '.join(GAME_CONFIG)})")
        return 1
    modes = ["synthetic", "shuffled"] if args.mode == "both" else [args.mode]
    seeds = list(range(args.seed, args.seed + max(1, args.repetitions)))

    reports = []
    for game in games:
        for mode in modes:
            print(f"\n{game} / {mode}: {len(seeds)} control histories, {args.days} days each")
            reports.append(run(game, mode, seeds, args.days, args.foundation, args.recent))

    print("\n" + "=" * 108)
    print(f"{'game':<14}{'control':<11}{'rows':>5}{'avg row':>10}{'best row':>10}{'selection':>11}"
          f"{'spread':>9}{'random':>9}   winners")
    for report in reports:
        if report.get("best_row_mean") is None:
            print(f"{report['game']:<14}{report['mode']:<11}{'-':>5}   no row produced a score - see the log above")
            continue
        number = lambda key: f"{report[key]:>10.3f}" if report.get(key) is not None else f"{'-':>10}"
        print(f"{report['game']:<14}{report['mode']:<11}{report['rows_compared']:>5}"
              f"{number('row_mean')}{number('best_row_mean')}{report['selection_gain']:>11.3f}"
              f"{report['best_to_worst_spread']:>9.3f}{report['random_baseline']:>9.3f}   "
              f"{', '.join(sorted(set(report['winner_per_seed'])))}")
    print("=" * 108)
    print("selection = what the BEST row scores above the AVERAGE row on data with nothing in it.")
    print(f"Records written to {os.path.relpath(RESULT_DIR)}/<game>-<control>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
