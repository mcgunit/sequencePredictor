import os, argparse, json, sys, time, re
# Pin the BLAS pools before numpy is imported - the vote is a handful of small
# array ops per day, and a 16-thread OpenBLAS pool per trial process would only
# add spin-wait overhead (measured on the other tuners, see HyperoptBoost.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import optuna
import numpy as np
from multiprocessing import cpu_count
from art import text2art
from datetime import datetime

from src.Helpers import Helpers
from src.HyperoptRunner import (open_study, fail_stale_running_trials, optimize_study,
                                install_sigterm_handler)

helpers = Helpers()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

ROW_NAME = "SubsetEnsemble Model"

# Like HyperoptRLTicket.py this tuner never touches the training CSVs or the
# Backtester: it scores candidate subsets on the pipeline's own stored day
# JSONs (data/database/<game>), the only place where EVERY tracked row -
# statistical, boosting, deep learning, meta-learner, quantum - exists side by
# side for the same draws. That is what makes the subset search possible over
# all of them, and it is the real-life record: each row's ticket is what the
# pipeline actually emitted that day, tuned as it was then.
#
# Positional games (pick3, jokerplus) are not tuned: the vote row is not
# served for them (see Predictor.addWeightedEnsemblePrediction); README
# roadmap item 7 adds the per-slot vote that will make them tunable here.
GAMES = ("euromillions", "lotto", "eurodreams", "keno", "vikinglotto")

# Only Keno has a real payout table among the served games - its objective
# is profit per bet; the others are scored by main-ticket hits.
PAYOUT_GAMES = ("keno",)

# Rows that can never be members: the two vote rows themselves (a vote over a
# vote), and the RL row, which Predictor.py appends AFTER the ensembles so it
# is never present when the vote is taken.
EXCLUDED_ROWS = ("WeightedEnsemble Model", ROW_NAME, "RL Ticket Model")

# A game needs at least this many scoreable days before tuning on it means
# anything, and a subset is scored only on days where all of its members
# exist - so a trial ends up with fewer days than this is pruned.
MIN_EVALUATION_DAYS = 10
# Every scored trial must sit on (nearly) the same draws, or trial values are
# not comparable: a subset is scored only on the days all of its members
# exist, and with 2^N subsets searched, a subset that happened to break even
# on the ten days its members coexisted WILL be found and outrank every
# subset that lost the house edge over the whole window (seen in testing with
# an absolute 30-day floor: six rows, ten shared days, mean 0.0 vs -0.65 for
# everything scored on 300 days). So: a row is a candidate only when present
# on MIN_ROW_COVERAGE of the window, and a trial counts only when its members
# coexist on MIN_TRIAL_COVERAGE of it. Younger rows (boosting, meta-learner,
# quantum - weeks old next to a year of statistical rows) join automatically
# as they age into the window, or right away with a shorter --days.
MIN_ROW_COVERAGE = 0.8
MIN_TRIAL_COVERAGE = 0.6
# Trial value = mean - penalty * std / sqrt(days) of the per-day score: a lower
# confidence bound that mildly favours the subsets scored on more of the
# window within the coverage band above. 0 = plain mean.
DEFAULT_CONFIDENCE_PENALTY = 1.0


def is_running():
    """
    Checks if another instance is running based on the lock file.

    The PID written into the lock is verified to still be alive: a lock whose
    owner is gone is stale - left behind by a crashed or killed run (a hung
    2026-08-17 run held the lock for two days and silently blocked every cron
    start after it). A stale lock is removed and treated as not running.
    """
    if not os.path.exists(LOCK_FILE):
        return False
    try:
        with open(LOCK_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # signal 0 = existence check only, nothing is sent
        return True
    except (ValueError, ProcessLookupError):
        print("Removing stale lock file (owner process no longer exists)")
        remove_lock()
        return False
    except PermissionError:
        # Process exists but belongs to another user - definitely running.
        return True


def create_lock():
    """Creates the lock file."""
    try:
        with open(LOCK_FILE, "x") as f:  # "x" mode creates the file, failing if it exists
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def remove_lock():
    """Removes the lock file."""
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass


def print_intro():
    ascii_art = text2art("Subset Ensemble Hyperopt")
    print("============================================================")
    print("Subset Ensemble Hyperopt")
    print("Licence : MIT License")
    print(ascii_art)
    print("Select the subset of rows the SubsetEnsemble Model votes over")


def get_keno_subset_sizes(name, bestParams_json_object):
    """
    Same lookup as Predictor.py's getKenoSubsetSizes: the use_5..use_10
    toggles are the hyperopt-tuned global choice every row respects, so the
    subset row is scored on exactly the sub-selections production would bet.
    """
    if "keno" not in name:
        return []
    return [size for size in (5, 6, 7, 8, 9, 10) if bestParams_json_object.get(f"use_{size}")]


def load_evaluation_days(historyDir, days):
    """
    The most recent `days` day JSONs that can be scored: a non-empty
    currentPrediction (the rows that were predicted for that draw) and a
    realResult (the draw they are scored against). `days` <= 0 means all of
    them. Returned chronologically as (date, rows, realResult).
    """
    entries = []
    if not os.path.isdir(historyDir):
        return entries

    for fileName in os.listdir(historyDir):
        if not fileName.endswith(".json"):
            continue
        try:
            # File names are "YYYY-M-D.json" (not zero padded), which
            # strptime accepts fine; anything unparsable isn't a day file.
            fileDate = datetime.strptime(fileName[:-5], "%Y-%m-%d")
        except ValueError:
            continue
        try:
            with open(os.path.join(historyDir, fileName), "r") as infile:
                dayData = json.load(infile)
        except Exception:
            continue

        rows = dayData.get("currentPrediction") or []
        realResult = dayData.get("realResult") or []
        if not rows or not realResult:
            continue
        entries.append((fileDate, rows, realResult))

    entries.sort(key=lambda entry: entry[0])
    return entries[-days:] if days and days > 0 else entries


def candidate_rows(evaluation_days):
    """
    Row names eligible for the vote: present (with a non-empty main ticket)
    on at least MIN_ROW_COVERAGE of the evaluation days and never one of
    EXCLUDED_ROWS. Sorted, so the Optuna parameter set is stable across runs.
    """
    presence = {}
    for _, rows, _ in evaluation_days:
        seen = set()
        for row in rows:
            name = row.get("name")
            if not name or name in EXCLUDED_ROWS or name in seen:
                continue
            if row.get("predictions") and row["predictions"][0]:
                presence[name] = presence.get(name, 0) + 1
                seen.add(name)
    needed = max(MIN_EVALUATION_DAYS, int(np.ceil(MIN_ROW_COVERAGE * len(evaluation_days))))
    return sorted(name for name, count in presence.items() if count >= needed)


def include_param(name):
    """Optuna parameter name of a row's include flag ("Markov Model" -> subsetEnsemble_include_MarkovModel)."""
    return "subsetEnsemble_include_" + re.sub(r"[^A-Za-z0-9]+", "", name)


def main_count_of(rows, special_column_count, fallback):
    """
    Main-ticket size from the stored rows (the first non-empty main ticket
    minus the trailing special columns), like Predictor.addRLTicketPrediction.
    """
    for row in rows:
        if row.get("predictions") and row["predictions"][0]:
            return len(row["predictions"][0]) - special_column_count
    return fallback


def score_day(predictions, realResult, dataset_name, mainCount):
    """
    One day's contribution in the report's own currency (see
    Helpers.generate_model_performance_report): for Keno the net profit and
    bet count over the playable sub-selections (predictions[1:] - the
    20-number main ticket has no payout), elsewhere the main-ticket hits
    against the drawn mains (realResult sliced with main_special_split, so
    stars/dream/viking and lotto's bonus never count as mains).
    Returns (score, bets).
    """
    if dataset_name in PAYOUT_GAMES:
        profit, bets = 0.0, 0
        for subset in predictions[1:]:
            p = helpers.keno_ticket_profit(subset, realResult)
            if p is not None:
                profit += float(p)
                bets += 1
        return profit, bets

    realMainCount, _ = helpers.main_special_split(dataset_name, realResult)
    realMains = set(int(n) for n in realResult[:realMainCount])
    ticketMains = set(int(n) for n in predictions[0][:mainCount])
    return float(len(ticketMains & realMains)), 1


def objective_subset_ensemble(trial, dataset_name, evaluation_days, candidates, model_scores,
                              specialColumnCount, fallbackMainCount, kenoSubsetSizes, subsetMode, subsetTemperature,
                              confidencePenalty=DEFAULT_CONFIDENCE_PENALTY):
    """
    One include flag per candidate row plus the weighted/flat choice. The
    selected rows are voted exactly as Predictor.py will serve them
    (Helpers.build_vote_ensemble_predictions) on every evaluation day where
    all of them exist, and scored per day like the report scores the served
    rows: profit per bet for Keno, main-ticket hits otherwise. Trial value is
    the lower confidence bound mean - confidencePenalty * std / sqrt(days) of
    that per-day series (see DEFAULT_CONFIDENCE_PENALTY); the plain mean is
    kept as user attribute "mean". Fewer than two members, or members that
    coexist on less than MIN_TRIAL_COVERAGE of the window (at least
    MIN_EVALUATION_DAYS), prunes the trial (recorded, never a score).
    """
    included = [name for name in candidates if trial.suggest_categorical(include_param(name), [True, False])]
    weighted = trial.suggest_categorical("subsetEnsembleWeighted", [True, False])
    trial.set_user_attr("members", included)
    if len(included) < 2:
        raise optuna.TrialPruned()  # a one-row "ensemble" is just that row

    # The softmax Keno sub-selection samples; a fixed seed keeps identical
    # subsets identically scored (the other tuners reseed for the same reason).
    np.random.seed(42)
    weights = model_scores if weighted else None
    memberSet = set(included)

    daily = []
    for _, rows, realResult in evaluation_days:
        memberRows = [row for row in rows
                      if row.get("name") in memberSet and row.get("predictions") and row["predictions"][0]]
        if len({row["name"] for row in memberRows}) < len(included):
            continue  # a member did not run that day - production skips the row too
        mainCount = main_count_of(memberRows, specialColumnCount, fallbackMainCount)
        predictions = helpers.build_vote_ensemble_predictions(
            memberRows, mainCount, specialColumnCount, model_scores=weights,
            keno_subset_sizes=kenoSubsetSizes, subset_mode=subsetMode, subset_temperature=subsetTemperature)
        if not predictions:
            continue
        dayScore, dayBets = score_day(predictions, realResult, dataset_name, mainCount)
        if dayBets > 0:
            daily.append(dayScore / dayBets)

    trial.set_user_attr("scored_days", len(daily))
    if len(daily) < max(MIN_EVALUATION_DAYS, int(np.ceil(MIN_TRIAL_COVERAGE * len(evaluation_days)))):
        raise optuna.TrialPruned()
    daily = np.array(daily, dtype=float)
    mean = float(daily.mean())
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    trial.set_user_attr("mean", mean)
    return mean - float(confidencePenalty) * std / np.sqrt(len(daily))


if __name__ == "__main__":
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    install_sigterm_handler()

    try:
        try:
            helpers.git_pull()
        except Exception as e:
            print("Failed to get latest changes")

        parser = argparse.ArgumentParser(
            prog='Sequence Predictor',
            description='Selects the rows the SubsetEnsemble Model votes over, per game',
            epilog='Check it out'
        )

        parser.add_argument('-d', '--days', type=int, default=120,
                            help='Most recent scoreable day JSONs to select on (0 = all). Rows present on fewer '
                                 'than 80%% of them are not candidates, so a shorter window lets younger rows '
                                 '(boosting, meta-learner, quantum) take part; a longer one gives more draws.')
        parser.add_argument('-t', '--trials', type=int, default=200,
                            help='Trials per game. The space is one include flag per row (~2^20), '
                                 'a trial takes about a second, so a few hundred is cheap.')
        parser.add_argument(
            '--parallel-trials', type=int, default=1,
            help='Trials evaluated at the same time, each in its own process. Default 1: trials take '
                 'about a second and the runner spaces launches 3 s apart, so more processes would be '
                 'slower here, not faster.')
        parser.add_argument(
            '--confidence-penalty', type=float, default=DEFAULT_CONFIDENCE_PENALTY,
            help='Trial value = mean - penalty * std / sqrt(days) of the per-day score, so a subset scored '
                 'on few (lucky) days does not outrank one scored on many. 0 = plain mean.')
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '-g', '--games',
            type=str,
            default=",".join(GAMES),
            help='Comma-separated list of games, e.g. "keno,lotto"'
        )

        args = parser.parse_args()

        print_intro()

        evaluationDayCount = int(args.days)
        n_trials = int(args.trials)
        pushToGit = bool(args.save)
        parallel = max(1, min(int(args.parallel_trials), max(1, cpu_count() - 1)))
        confidencePenalty = max(0.0, float(args.confidence_penalty))

        print("Push to git: ", pushToGit)
        print("Running ", n_trials, "trials")

        games = [g.strip() for g in args.games.split(',') if g.strip()]
        unknown_games = [g for g in games if g not in GAMES]
        if unknown_games:
            print(f"Unknown or positional game(s), ignoring: {unknown_games}")
        print("Selected games:", games)

        path = os.getcwd()
        optunaDatabase = "sqlite:///db.sqlite3"

        for dataset_name in GAMES:
            if dataset_name not in games:
                continue
            try:
                print(f"\n{dataset_name.capitalize()}")
                historyDir = os.path.join(path, "data", "database", dataset_name)

                evaluation_days = load_evaluation_days(historyDir, evaluationDayCount)
                if len(evaluation_days) < MIN_EVALUATION_DAYS:
                    print(f"Skipping {dataset_name}: only {len(evaluation_days)} day JSONs with both "
                          f"a currentPrediction and a realResult (need at least {MIN_EVALUATION_DAYS})")
                    continue
                print(f"Evaluating on {len(evaluation_days)} days "
                      f"({evaluation_days[0][0].date()} .. {evaluation_days[-1][0].date()})")

                candidates = candidate_rows(evaluation_days)
                if len(candidates) < 2:
                    print(f"Skipping {dataset_name}: only {len(candidates)} candidate row(s) present on "
                          f"at least {MIN_ROW_COVERAGE:.0%} of those days")
                    continue
                print(f"Candidate rows ({len(candidates)}): {', '.join(candidates)}")
                paramNames = [include_param(name) for name in candidates]
                if len(set(paramNames)) != len(paramNames):
                    print(f"Skipping {dataset_name}: two candidate rows map to the same parameter name")
                    continue

                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                kenoSubsetSizes = get_keno_subset_sizes(dataset_name, existingData)
                if dataset_name in PAYOUT_GAMES and not kenoSubsetSizes:
                    print(f"Skipping {dataset_name}: no use_5..use_10 subset sizes enabled in "
                          f"bestParams_{dataset_name}.json, nothing to score profit on")
                    continue

                specialColumnCount = next(
                    (count for game, count in Helpers.SPECIAL_COLUMN_COUNTS.items() if game in dataset_name), 0)
                fallbackMainCount = main_count_of(evaluation_days[-1][1], specialColumnCount, 0)
                modelScores = existingData.get("modelScores", {})
                # The served row slices its Keno subsets with the tuned
                # WeightedEnsemble mode/temperature unless it has its own.
                subsetMode = existingData.get("subsetEnsembleSubsetMode",
                                              existingData.get("weightedEnsembleSubsetMode", "softmax"))
                subsetTemperature = existingData.get("subsetEnsembleSubsetTemperature",
                                                     existingData.get("weightedEnsembleSubsetTemperature", 0.5))

                studyName = f"{dataset_name}-subset_ensemble"
                study = open_study(studyName, optunaDatabase, parallel=parallel)
                fail_stale_running_trials(study)
                # Baseline the search can only improve on: every candidate in,
                # score-weighted - the WeightedEnsemble vote over the same rows.
                study.enqueue_trial({**{p: True for p in paramNames}, "subsetEnsembleWeighted": True},
                                    skip_if_exists=True)

                objective = lambda trial, dataset_name=dataset_name, evaluation_days=evaluation_days, \
                                   candidates=candidates, modelScores=modelScores, \
                                   specialColumnCount=specialColumnCount, fallbackMainCount=fallbackMainCount, \
                                   kenoSubsetSizes=kenoSubsetSizes, subsetMode=subsetMode, \
                                   subsetTemperature=subsetTemperature, confidencePenalty=confidencePenalty: \
                    objective_subset_ensemble(
                        trial, dataset_name, evaluation_days, candidates, modelScores,
                        specialColumnCount, fallbackMainCount, kenoSubsetSizes, subsetMode, subsetTemperature,
                        confidencePenalty)

                runStart = datetime.now()
                studyStart = time.time()
                optimize_study(studyName, optunaDatabase, objective, n_trials, parallel=parallel, expected_trial_gb=0.3)
                print(f"Study {studyName} finished in {time.time() - studyStart:.1f}s")

                # Best of THIS run only: the study persists across weeks, but
                # every week scores on a longer (different) day window, so
                # trial values of different runs are not comparable.
                study = open_study(studyName, optunaDatabase, parallel=parallel, quiet=True)
                thisRun = [t for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
                           if t.datetime_start is not None and t.datetime_start >= runStart and t.value is not None]
                if not thisRun:
                    print(f"No completed trials for {studyName} in this run (all pruned/failed) - keeping existing params")
                    continue
                best = max(thisRun, key=lambda t: t.value)
                members = list(best.user_attrs.get("members", []))
                weighted = bool(best.params.get("subsetEnsembleWeighted", True))

                metric = "profit per bet" if dataset_name in PAYOUT_GAMES else "avg main hits"
                bestMean = float(best.user_attrs.get("mean", best.value))
                print(f"Best subset for {dataset_name} ({metric} {bestMean:.4f}, lower bound {best.value:.4f} on "
                      f"{best.user_attrs.get('scored_days')} days, {'weighted' if weighted else 'flat'} vote): "
                      f"{', '.join(members)}")

                # Only the derived selection is written - never the row's own
                # score into modelScores (it would feed back into its own
                # weights), and not the twenty include flags.
                existingData["subsetEnsembleModels"] = members
                existingData["subsetEnsembleWeighted"] = weighted
                existingData["subsetEnsembleObjective"] = round(float(best.value), 4)
                existingData["subsetEnsembleMean"] = round(bestMean, 4)
                existingData["subsetEnsembleTunedOn"] = {
                    "days": int(best.user_attrs.get("scored_days") or 0),
                    "from": evaluation_days[0][0].strftime("%Y-%m-%d"),
                    "to": evaluation_days[-1][0].strftime("%Y-%m-%d"),
                    "candidates": len(candidates),
                    "confidencePenalty": confidencePenalty,
                }

                with open(jsonBestParamsFilePath, "w+") as outfile:
                    json.dump(existingData, outfile, indent=4)

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            if pushToGit:
                helpers.git_push(commit_message="Saving latest subset ensemble hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
