import os, argparse, json, sys, time, re
# Pin the BLAS pools before numpy is imported - the vote is a handful of small
# array ops per day; a 16-thread OpenBLAS pool would only add spin-wait
# overhead (measured on the other tuners, see HyperoptBoost.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import optuna
import numpy as np
from art import text2art
from datetime import datetime

from src.Helpers import Helpers
from src.HyperoptRunner import open_study, fail_stale_running_trials, install_sigterm_handler
from src.TuningGate import challenge, make_defaults

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
# The positional games (pick3, Joker+) vote per slot (README roadmap item 7,
# Helpers.build_positional_vote_predictions - the same recipe Predictor.py
# serves) and are scored with their real payout tables. Joker+ is tuned only
# when bestParams_jokerplus.json sets JOKERPLUS_ENSEMBLE_FLAG, like the
# served rows: its digits are system-generated, the player only picks the
# sign.
# The market games (crypto, shares - README roadmap item 4, M4) vote per slot
# like pick3 and are scored with the fixed rule in bin space
# (Helpers.market_ticket_profit: long on an upper-half bin, +1/-1 on the
# actual half, minus a fee, in units) - the "profit" the positional machinery
# uses wherever it asks for one; the euro books live in the settlement.
GAMES = ("euromillions", "lotto", "eurodreams", "keno", "vikinglotto", "pick3", "jokerplus", "crypto", "shares", "cryptoweek", "sharesweek")
JOKERPLUS_ENSEMBLE_FLAG = "useJokerplusEnsemble"

# Games with a real payout table - their objective is profit per bet (Keno:
# the playable subsets; pick3: the cumulative four-bet ticket; Joker+: the
# 1.50 EUR ticket); the others are scored by main-ticket hits.
PAYOUT_GAMES = ("keno", "pick3", "jokerplus")
# Positional games add a small per-slot accuracy tie-breaker to the value:
# pick3/Joker+ payouts are sparse (most days every subset loses the stake),
# so many subsets tie on profit and the tie-breaker orders them by how often
# their slots were right.
SLOT_ACCURACY_WEIGHT = 0.01

# Rows that can never be members: the two vote rows themselves (a vote over a
# vote), and the RL row, which Predictor.py appends AFTER the ensembles so it
# is never present when the vote is taken.
EXCLUDED_ROWS = ("WeightedEnsemble Model", ROW_NAME, "RL Ticket Model", "RL Position Model")   # the position row too: appended after the vote, and its ticket IS the vote (8 Oct 2026)

# A game needs at least this many scoreable days before tuning on it means
# anything, and a subset is scored only on days where all of its members
# exist - a subset that ends up with fewer days than this is infeasible.
MIN_EVALUATION_DAYS = 10
# Every scored subset must sit on (nearly) the same draws, or values are not
# comparable: a subset is scored only on the days all of its members exist,
# and with 2^N subsets in the space, a subset that happened to break even on
# the ten days its members coexisted WILL be found and outrank every subset
# that lost the house edge over the whole window (seen in testing with an
# absolute 30-day floor: six rows, ten shared days, mean 0.0 vs -0.65 for
# everything scored on 300 days). So: a row is a candidate only when present
# on MIN_ROW_COVERAGE of the window, and a subset counts only when its
# members coexist on MIN_TRIAL_COVERAGE of it. Younger rows (boosting,
# meta-learner, quantum - weeks old next to a year of statistical rows) join
# automatically as they age into the window, or right away with a shorter
# --days.
MIN_ROW_COVERAGE = 0.8
MIN_TRIAL_COVERAGE = 0.6
# ...and the row must still be emitted: present on at least RECENT_MIN_DAYS
# of the RECENT_DAYS most recent scoreable days. History alone is not
# enough - the LSTM Base Model row stopped appearing in mid-August 2026 yet
# covered 79 of 87 lotto days, was selected, and the served row was then
# skipped every day for a missing member. Three of five tolerates a one-off
# DL child failure without dropping a live row for a week.
RECENT_DAYS = 5
RECENT_MIN_DAYS = 3
# Subset value = mean - penalty * std / sqrt(days) of the per-day score: a
# lower confidence bound that mildly favours the subsets scored on more of
# the window within the coverage band above. 0 = plain mean.
DEFAULT_CONFIDENCE_PENALTY = 1.0

# Search. One subset evaluation is 2-20 ms in-process (the vote over ~100
# stored days), so the space is simply enumerated whenever 2^k x 2 (include
# flags x weighted/flat) fits the evaluation budget - k <= 12 at the default.
# Beyond that a deterministic local search runs: hill-climbing over single
# flag toggles from the all-in vote (weighted and flat) and from the best
# pair, every subset evaluated once. Optuna's TPE was tried first and is the
# wrong tool here: sampling boolean flags independently, it collapses onto
# its mode and re-proposes the same subset trial after trial (a production
# run spent 162 trials on 37 distinct subsets, one of them 62 times), and
# the process-per-trial runner spends ~2 s launching a 5 ms evaluation.
DEFAULT_MAX_EVALUATIONS = 16384
# Evaluated subsets are recorded into the game's Optuna study (db.sqlite3,
# the dashboard) as completed trials, best first - all of them when the
# search evaluated at most this many, otherwise the top ones - so the study
# is the ranking of this run, without duplicates.
DEFAULT_RECORD_TRIALS = 256

# The champion/challenger gate (src/TuningGate.py, since 15 Oct 2026 - the last tuner of the Saturday chain that
# wrote its result without one): this run's best combination is written only if it beats the served selection
# and the plain vote (every candidate row, weighted - the WeightedEnsemble Model over the same rows), both
# re-scored on this run's days by this objective, and by GATE_SE_MARGIN standard errors of its own score; the
# plain vote is served back when it beats both. The record is bestParams["tuningGate"]["SubsetEnsemble Model"].
GATE_ENABLED = True
GATE_MARGIN = 0.0
GATE_SE_MARGIN = 1.0
GATE_TIMEOUT_SECONDS = 300    # one subset scores in milliseconds; the reference evaluations run in a forked child


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
    on at least MIN_ROW_COVERAGE of the evaluation days AND on at least
    RECENT_MIN_DAYS of the last RECENT_DAYS of them (a row the pipeline no
    longer emits cannot be a member of a served row), never one of
    EXCLUDED_ROWS. Sorted, so the parameter set is stable across runs.
    Returns (candidates, dropped_stale) with the rows that had the history
    but not the recent presence, for the log.
    """
    presence, recent = {}, {}
    recentDays = evaluation_days[-RECENT_DAYS:]
    for index, (_, rows, _) in enumerate(evaluation_days):
        isRecent = index >= len(evaluation_days) - len(recentDays)
        seen = set()
        for row in rows:
            name = row.get("name")
            if not name or name in EXCLUDED_ROWS or name in seen:
                continue
            if row.get("predictions") and row["predictions"][0]:
                presence[name] = presence.get(name, 0) + 1
                if isRecent:
                    recent[name] = recent.get(name, 0) + 1
                seen.add(name)
    needed = max(MIN_EVALUATION_DAYS, int(np.ceil(MIN_ROW_COVERAGE * len(evaluation_days))))
    recentNeeded = min(RECENT_MIN_DAYS, len(recentDays))
    withHistory = sorted(name for name, count in presence.items() if count >= needed)
    candidates = [name for name in withHistory if recent.get(name, 0) >= recentNeeded]
    stale = [name for name in withHistory if name not in candidates]
    return candidates, stale


def include_param(name):
    """Study parameter name of a row's include flag ("Markov Model" -> subsetEnsemble_include_MarkovModel)."""
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
    Helpers.generate_model_performance_report): pick3 and Joker+ the net
    profit of the positional ticket (one bet), for Keno the net profit and
    bet count over the playable sub-selections (predictions[1:] - the
    20-number main ticket has no payout), elsewhere the main-ticket hits
    against the drawn mains (realResult sliced with main_special_split, so
    stars/dream/viking and lotto's bonus never count as mains).
    Returns (score, bets).
    """
    if "pick3" in dataset_name:
        p = helpers.pick3_ticket_profit(predictions[0], realResult)
        return (float(p), 1) if p is not None else (0.0, 0)
    if helpers.is_jokerplus(dataset_name):
        p = helpers.jokerplus_ticket_profit(predictions[0], realResult)
        return (float(p), 1) if p is not None else (0.0, 0)
    if helpers.is_market_game(dataset_name):
        p = helpers.market_ticket_profit(predictions[0], realResult)
        return (float(p), 1) if p is not None else (0.0, 0)
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


def evaluate_subset(members, weighted, dataset_name, evaluation_days, model_scores, specialColumnCount,
                    fallbackMainCount, kenoSubsetSizes, subsetMode, subsetTemperature,
                    confidencePenalty=DEFAULT_CONFIDENCE_PENALTY):
    """
    Scores one subset: the selected rows are voted exactly as Predictor.py
    will serve them (Helpers.build_vote_ensemble_predictions) on every
    evaluation day where all of them exist, and scored per day like the
    report scores the served rows - profit per bet for Keno, main-ticket hits
    otherwise. Returns {"value", "mean", "days", "se"} with value the lower
    confidence bound mean - confidencePenalty * std / sqrt(days) of that
    per-day series (positional games add SLOT_ACCURACY_WEIGHT x the mean
    share of slots the ticket got right) and se that standard error
    std / sqrt(days), the unit of the gate's margin, or None when the subset is
    infeasible: fewer than two
    members, or members that coexist on less than MIN_TRIAL_COVERAGE of the
    window (and at least MIN_EVALUATION_DAYS).
    """
    if len(members) < 2:
        return None  # a one-row "ensemble" is just that row

    # The softmax Keno sub-selection samples; a fixed seed keeps identical
    # subsets identically scored (the other tuners reseed for the same reason).
    np.random.seed(42)
    weights = model_scores if weighted else None
    memberSet = set(members)
    positional = helpers.is_positional_game(dataset_name)

    daily, slotAccuracy = [], []
    for _, rows, realResult in evaluation_days:
        memberRows = [row for row in rows
                      if row.get("name") in memberSet and row.get("predictions") and row["predictions"][0]]
        if len({row["name"] for row in memberRows}) < len(members):
            continue  # a member did not run that day - production skips the row too
        if positional:
            # Per-slot vote over the full stored ticket (pick3: 3 digits;
            # Joker+: 6 digits + the zodiac code as slot 7).
            slotCount = len(memberRows[0]["predictions"][0])
            vote = helpers.build_positional_vote_predictions(memberRows, slotCount, model_scores=weights)
            if not vote:
                continue
            predictions = [vote["ticket"]]
            mainCount = slotCount
            slotAccuracy.append(sum(int(t) == int(r) for t, r in zip(vote["ticket"], realResult[:slotCount])) / slotCount)
        else:
            mainCount = main_count_of(memberRows, specialColumnCount, fallbackMainCount)
            predictions = helpers.build_vote_ensemble_predictions(
                memberRows, mainCount, specialColumnCount, model_scores=weights,
                keno_subset_sizes=kenoSubsetSizes, subset_mode=subsetMode, subset_temperature=subsetTemperature)
            if not predictions:
                continue
        dayScore, dayBets = score_day(predictions, realResult, dataset_name, mainCount)
        if dayBets > 0:
            daily.append(dayScore / dayBets)

    if len(daily) < max(MIN_EVALUATION_DAYS, int(np.ceil(MIN_TRIAL_COVERAGE * len(evaluation_days)))):
        return None
    daily = np.array(daily, dtype=float)
    mean = float(daily.mean())
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    se = float(std / np.sqrt(len(daily)))
    value = mean - float(confidencePenalty) * se
    if slotAccuracy:
        value += SLOT_ACCURACY_WEIGHT * float(np.mean(slotAccuracy))
    return {"value": value, "mean": mean, "days": len(daily), "se": se}


def search_subsets(candidates, evaluate, max_evaluations):
    """
    Exhaustive enumeration of every (subset, weighted) combination when
    2^k x 2 fits max_evaluations, otherwise a deterministic local search:
    hill-climbing over single include-flag toggles and the weighted flip,
    started from the all-in vote (weighted and flat) and from the best pair,
    until no neighbour improves or the budget is spent. Every combination is
    evaluated at most once (memoized), so the returned ranking has no
    duplicates. Returns (ranking best-first as [(value, members, weighted,
    mean, days, se)], evaluated combinations, infeasible combinations, how).
    """
    k = len(candidates)
    cache = {}

    def score(mask, weighted):
        key = (mask, weighted)
        if key not in cache:
            members = [name for name, on in zip(candidates, mask) if on]
            cache[key] = evaluate(members, weighted)
        return cache[key]

    def value_of(mask, weighted):
        result = score(mask, weighted)
        return result["value"] if result else float("-inf")

    if 2 ** k * 2 <= max_evaluations:
        how = "exhaustive"
        for bits in range(2 ** k):
            mask = tuple(bool(bits >> i & 1) for i in range(k))
            for weighted in (True, False):
                score(mask, weighted)
    else:
        how = "local search"
        allIn = tuple([True] * k)
        starts = [(allIn, True), (allIn, False)]
        # Best pair as the third start: k(k-1) evaluations, then the climb
        # from below meets the climb from the all-in vote above.
        bestPair = None
        for i in range(k):
            for j in range(i + 1, k):
                if len(cache) >= max_evaluations:
                    break
                mask = tuple(idx in (i, j) for idx in range(k))
                for weighted in (True, False):
                    v = value_of(mask, weighted)
                    if bestPair is None or v > bestPair[0]:
                        bestPair = (v, mask, weighted)
        if bestPair is not None:
            starts.append((bestPair[1], bestPair[2]))

        for mask, weighted in starts:
            current = value_of(mask, weighted)
            while len(cache) < max_evaluations:
                bestMove = None
                neighbours = [(mask[:i] + (not mask[i],) + mask[i + 1:], weighted) for i in range(k)]
                neighbours.append((mask, not weighted))
                for nMask, nWeighted in neighbours:
                    if len(cache) >= max_evaluations:
                        break
                    v = value_of(nMask, nWeighted)
                    if bestMove is None or v > bestMove[0]:
                        bestMove = (v, nMask, nWeighted)
                if bestMove is None or bestMove[0] <= current:
                    break
                current, mask, weighted = bestMove

    ranking = sorted(
        ((result["value"], [name for name, on in zip(candidates, mask) if on], weighted, result["mean"], result["days"], result.get("se"))
         for (mask, weighted), result in cache.items() if result),
        key=lambda item: (item[0], -len(item[1])), reverse=True)
    infeasible = sum(1 for result in cache.values() if result is None)
    return ranking, len(cache), infeasible, how


def record_results(study, candidates, ranking, limit, runTag):
    """
    Writes the ranking (best first, at most `limit` entries) into the Optuna
    study as completed trials - the include flags and the weighted flag as
    categorical parameters, members/mean/days/run as user attributes - so
    db.sqlite3 and the dashboard carry this run's ranking without duplicates.
    """
    distributions = {include_param(name): optuna.distributions.CategoricalDistribution([True, False])
                     for name in candidates}
    distributions["subsetEnsembleWeighted"] = optuna.distributions.CategoricalDistribution([True, False])
    recorded = 0
    for value, members, weighted, mean, days, se in ranking[:limit]:
        params = {include_param(name): (name in members) for name in candidates}
        params["subsetEnsembleWeighted"] = bool(weighted)
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=distributions, value=float(value),
            user_attrs={"members": list(members), "mean": float(mean), "scored_days": int(days), "run": runTag,
                        "tuning": subset_tuning(value, mean, days, se)}))
        recorded += 1
    return recorded


def subset_tuning(value, mean, days, se):
    """The gate's diagnostics for a scored combination (TuningGate.challenge reads a trial's "tuning" attr)."""
    return {"score": float(value), "kind": "subset_bound", "days": int(days), "mean": float(mean),
            "score_se": (float(se) if se is not None else None)}


def reference_flags(existingData, candidates):
    """
    The gate's two references as include-flag dicts over this run's
    candidates, and a note when the served selection is not what the gate
    scores: (served, defaults, note).

    The served selection is bestParams "subsetEnsembleModels" /
    "subsetEnsembleWeighted" (weighted by default, as Predictor.
    addSubsetEnsemblePrediction reads it). When nothing was ever written, or
    a served member is no longer a candidate (a row the pipeline stopped
    emitting: live, Predictor skips the row on every day that member is
    absent, so the selection serves nothing), the served flags are all False:
    the incumbent then scores -inf (evaluate_subset needs two members), so
    the run writes the challenger when it clears the plain vote by the
    margin and the plain vote otherwise, and the row is served again. The
    review of 15 Oct 2026 caught the first draft scoring "served minus the
    missing member" instead - a proxy the challenger could never beat by the
    margin, which would have kept a dead selection for good.

    The plain vote - every candidate, weighted - is the all-in vote over this
    run's candidates: what the WeightedEnsemble Model restricts to those rows.
    """
    defaults = {include_param(name): True for name in candidates}
    defaults["subsetEnsembleWeighted"] = True
    nothing = {include_param(name): False for name in candidates}
    nothing["subsetEnsembleWeighted"] = bool(existingData.get("subsetEnsembleWeighted", True))
    members = [name for name in (existingData.get("subsetEnsembleModels") or [])]
    if not members:
        return nothing, defaults, "nothing served before this run - the incumbent is scored as nothing"
    missing = [name for name in members if name not in candidates]
    if missing:
        return nothing, defaults, (f"served member(s) not among this run's candidates: {', '.join(missing)} - "
                                   f"the served selection cannot be served and is scored as nothing")
    served = {include_param(name): (name in set(members)) for name in candidates}
    served["subsetEnsembleWeighted"] = bool(existingData.get("subsetEnsembleWeighted", True))
    return served, defaults, None


def gate_objective(trial, candidates, evaluate):
    """The search's evaluation as a trial objective, for the gate's reference scoring (a FixedTrial of flags)."""
    members = [name for name in candidates if trial.suggest_categorical(include_param(name), [True, False])]
    weighted = bool(trial.suggest_categorical("subsetEnsembleWeighted", [True, False]))
    result = evaluate(members, weighted)
    if not result:
        return float("-inf")
    trial.set_user_attr("tuning", subset_tuning(result["value"], result["mean"], result["days"], result.get("se")))
    return float(result["value"])


def members_of(params, candidates):
    """(members, weighted) from a dict of include flags."""
    return [name for name in candidates if params.get(include_param(name))], bool(params.get("subsetEnsembleWeighted", True))


def tune_game(dataset_name, path, evaluationDayCount, maxEvaluations, recordLimit, confidencePenalty, optunaDatabase, runTag):
    """
    One game: the candidate rows, the search, this run's ranking into the
    study, the gate, the write. Extracted from __main__ on 15 Oct 2026 so a
    scratch root can exercise it.
    """
    print(f"\n{dataset_name.capitalize()}")
    historyDir = os.path.join(path, "data", "database", dataset_name)

    evaluation_days = load_evaluation_days(historyDir, evaluationDayCount)
    if len(evaluation_days) < MIN_EVALUATION_DAYS:
        print(f"Skipping {dataset_name}: only {len(evaluation_days)} day JSONs with both "
              f"a currentPrediction and a realResult (need at least {MIN_EVALUATION_DAYS})")
        return
    print(f"Evaluating on {len(evaluation_days)} days "
          f"({evaluation_days[0][0].date()} .. {evaluation_days[-1][0].date()})")

    candidates, stale = candidate_rows(evaluation_days)
    if stale:
        print(f"Not candidates (history but absent from {RECENT_DAYS - RECENT_MIN_DAYS + 1}+ of the "
              f"last {RECENT_DAYS} days - no longer emitted?): {', '.join(stale)}")
    if len(candidates) < 2:
        print(f"Skipping {dataset_name}: only {len(candidates)} candidate row(s) present on "
              f"at least {MIN_ROW_COVERAGE:.0%} of those days")
        return
    print(f"Candidate rows ({len(candidates)}): {', '.join(candidates)}")
    paramNames = [include_param(name) for name in candidates]
    if len(set(paramNames)) != len(paramNames):
        print(f"Skipping {dataset_name}: two candidate rows map to the same parameter name")
        return

    jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
    existingData = {}
    if os.path.exists(jsonBestParamsFilePath):
        with open(jsonBestParamsFilePath, "r") as infile:
            existingData = json.load(infile)

    if helpers.is_jokerplus(dataset_name) and not existingData.get(JOKERPLUS_ENSEMBLE_FLAG, False):
        print(f"Skipping {dataset_name}: the vote rows are switched off for Joker+ "
              f"(set \"{JOKERPLUS_ENSEMBLE_FLAG}\": true in bestParams_{dataset_name}.json to tune and serve them)")
        return

    kenoSubsetSizes = get_keno_subset_sizes(dataset_name, existingData)
    if "keno" in dataset_name and not kenoSubsetSizes:
        print(f"Skipping {dataset_name}: no use_5..use_10 subset sizes enabled in "
              f"bestParams_{dataset_name}.json, nothing to score profit on")
        return

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

    evaluate = lambda members, weighted, dataset_name=dataset_name, evaluation_days=evaluation_days, \
                      modelScores=modelScores, specialColumnCount=specialColumnCount, \
                      fallbackMainCount=fallbackMainCount, kenoSubsetSizes=kenoSubsetSizes, \
                      subsetMode=subsetMode, subsetTemperature=subsetTemperature: \
        evaluate_subset(members, weighted, dataset_name, evaluation_days, modelScores,
                        specialColumnCount, fallbackMainCount, kenoSubsetSizes, subsetMode,
                        subsetTemperature, confidencePenalty)

    searchStart = time.time()
    ranking, evaluated, infeasible, how = search_subsets(candidates, evaluate, maxEvaluations)
    print(f"{how}: {evaluated} combinations evaluated ({infeasible} infeasible) "
          f"in {time.time() - searchStart:.1f}s")
    if not ranking:
        print(f"No feasible subset for {dataset_name} (members never coexist on "
              f"{MIN_TRIAL_COVERAGE:.0%} of the window) - keeping existing params")
        return

    # The study is the record of this run's ranking (best first,
    # no duplicates); older runs' trials stay as history and
    # RUNNING leftovers of a killed run are marked failed.
    studyName = f"{dataset_name}-subset_ensemble"
    study = open_study(studyName, optunaDatabase, quiet=True)
    fail_stale_running_trials(study)
    known_trials = {t.number for t in study.get_trials(deepcopy=False)}   # before this run's ranking is added
    recordStart = time.time()
    recorded = record_results(study, candidates, ranking, recordLimit, runTag)
    print(f"Recorded the best {recorded} combinations into study {studyName} "
          f"in {time.time() - recordStart:.1f}s")

    value, members, weighted, mean, days, _se = ranking[0]
    baseline = next((item for item in ranking if len(item[1]) == len(candidates) and item[2]), None)
    metric = "profit per bet" if dataset_name in PAYOUT_GAMES else ("fixed-rule units per day" if helpers.is_market_game(dataset_name) else "avg main hits")
    if helpers.is_positional_game(dataset_name):
        metric += " (per-slot vote)"
    print(f"Best subset for {dataset_name} ({metric} {mean:.4f}, lower bound {value:.4f} on "
          f"{days} days, {'weighted' if weighted else 'flat'} vote): {', '.join(members)}")
    if baseline:
        print(f"  all-in weighted vote (the WeightedEnsemble Model over the same rows): "
              f"{metric} {baseline[3]:.4f}, lower bound {baseline[0]:.4f} on {baseline[4]} days")

    # The gate: this run's best combination against the served selection and the plain vote, re-scored by
    # the same evaluation (reference_flags, gate_objective). Until 15 Oct 2026 the best combination was
    # written as it was - the last tuner of the chain to do so.
    served, defaults, note = reference_flags(existingData, candidates)
    if note:
        print(f"  {studyName}: {note}")
    study = open_study(studyName, optunaDatabase, quiet=True)
    objective = lambda trial, candidates=candidates, evaluate=evaluate: gate_objective(trial, candidates, evaluate)
    outcome = challenge(study, known_trials, objective, served, make_defaults(defaults, existingData),
                        timeout_seconds=GATE_TIMEOUT_SECONDS, label=studyName, margin=GATE_MARGIN,
                        enabled=GATE_ENABLED, window_days=len(evaluation_days), se_margin=GATE_SE_MARGIN)
    print(outcome["summary"])
    record = outcome["record"]
    record["candidates"] = list(candidates)
    if note:
        record["incumbent_note"] = note
    existingData.setdefault("tuningGate", {})[ROW_NAME] = record

    if outcome["params"] is not None:
        members, weighted = members_of(outcome["params"], candidates)
        chosen = next((item for item in ranking if set(item[1]) == set(members) and item[2] == weighted), None)
        if chosen is None:   # the plain vote was not among the evaluated combinations (a local search can miss it)
            result = evaluate(members, weighted) or {"value": float("nan"), "mean": float("nan"), "days": 0}
            chosen = (result["value"], members, weighted, result["mean"], result["days"], result.get("se"))
        value, _members, _weighted, mean, days, _se = chosen
        # Only the derived selection is written - never the row's own
        # score into modelScores (it would feed back into its own
        # weights), and not the include flags.
        existingData["subsetEnsembleModels"] = list(members)
        existingData["subsetEnsembleWeighted"] = bool(weighted)
        existingData["subsetEnsembleObjective"] = round(float(value), 4)
        existingData["subsetEnsembleMean"] = round(float(mean), 4)
        existingData["subsetEnsembleTunedOn"] = {
            "days": int(days),
            "from": evaluation_days[0][0].strftime("%Y-%m-%d"),
            "to": evaluation_days[-1][0].strftime("%Y-%m-%d"),
            "candidates": len(candidates),
            "evaluated": int(evaluated),
            "search": how,
            "confidencePenalty": confidencePenalty,
            "decision": outcome["decision"],
        }

    with open(jsonBestParamsFilePath, "w+") as outfile:
        json.dump(existingData, outfile, indent=4)


def _self_check():
    """The gate's wiring on hand-made data: python3 -c 'import HyperoptEnsemble; HyperoptEnsemble._self_check()'."""
    candidates = ["A Model", "B Model", "C Model"]
    nothing = {**{include_param(n): False for n in candidates}, "subsetEnsembleWeighted": True}
    served, defaults, note = reference_flags({}, candidates)
    assert served == nothing and "nothing served" in note and defaults == {**{include_param(n): True for n in candidates}, "subsetEnsembleWeighted": True}
    # a served member the pipeline no longer emits: the selection serves nothing live, so it is scored as nothing
    served, _, note = reference_flags({"subsetEnsembleModels": ["A Model", "C Model", "Gone Model"], "subsetEnsembleWeighted": False}, candidates)
    assert served == {**nothing, "subsetEnsembleWeighted": False} and "Gone Model" in note, (served, note)
    assert members_of(served, candidates) == ([], False)
    served, _, note = reference_flags({"subsetEnsembleModels": ["A Model", "C Model"], "subsetEnsembleWeighted": False}, candidates)
    assert note is None and served == {include_param("A Model"): True, include_param("B Model"): False, include_param("C Model"): True, "subsetEnsembleWeighted": False}, served
    assert members_of(served, candidates) == (["A Model", "C Model"], False)
    def evaluate(members, weighted):
        if len(members) < 2:
            return None
        return {"value": 0.1 * len(members) - (0.05 if weighted else 0.0), "mean": 0.1 * len(members), "days": 20, "se": 0.02}
    trial = optuna.trial.FixedTrial(served)
    value = gate_objective(trial, candidates, evaluate)
    assert abs(value - 0.2) < 1e-9 and trial.user_attrs["tuning"]["score_se"] == 0.02 and trial.user_attrs["tuning"]["kind"] == "subset_bound", (value, trial.user_attrs)
    assert gate_objective(optuna.trial.FixedTrial({include_param("A Model"): True, include_param("B Model"): False, include_param("C Model"): False, "subsetEnsembleWeighted": True}), candidates, evaluate) == float("-inf")
    # the gate on a fake study whose one new trial is this run's best: the served selection loses to it
    class FakeTrial:
        def __init__(self, number, value, params):
            self.number, self.value, self.params = number, value, params
            self.user_attrs = {"tuning": subset_tuning(value, value, 20, 0.0)}
            self.state = optuna.trial.TrialState.COMPLETE
    class FakeStudy:
        def __init__(self, trials):
            self._trials = trials
            self.best_value = max(t.value for t in trials)
        def get_trials(self, deepcopy=False):
            return self._trials
    best = {include_param("A Model"): True, include_param("B Model"): True, include_param("C Model"): True, "subsetEnsembleWeighted": False}
    study = FakeStudy([FakeTrial(0, 0.0, served), FakeTrial(1, 0.3, best)])
    outcome = challenge(study, {0}, lambda trial: gate_objective(trial, candidates, evaluate), served, make_defaults(defaults, {}),
                        timeout_seconds=60, label="self-check", margin=0.0, enabled=True, window_days=20, se_margin=0.0)
    assert outcome["decision"] == "replaced" and members_of(outcome["params"], candidates) == (candidates, False), outcome["reason"]
    # the plain vote beats a weak challenger and a weaker served selection: reverted to the defaults
    study = FakeStudy([FakeTrial(0, 0.0, served), FakeTrial(1, 0.21, {**best, "subsetEnsembleWeighted": True})])
    weak = lambda members, weighted: ({"value": 0.25 if (len(members) == 3 and weighted) else 0.1, "mean": 0.2, "days": 20, "se": 0.01} if len(members) >= 2 else None)
    outcome = challenge(study, {0}, lambda trial: gate_objective(trial, candidates, weak), served, make_defaults(defaults, {}),
                        timeout_seconds=60, label="self-check", margin=0.0, enabled=True, window_days=20, se_margin=0.0)
    assert outcome["decision"] == "reverted" and members_of(outcome["params"], candidates) == (candidates, True), outcome["reason"]
    # a dead served selection (a member no longer a candidate) is scored -inf: the run revives the row - here the
    # challenger does not clear the plain vote by the margin, so the plain vote is served
    dead, _, _ = reference_flags({"subsetEnsembleModels": ["A Model", "Gone Model"]}, candidates)
    study = FakeStudy([FakeTrial(0, 0.0, dead), FakeTrial(1, 0.21, {**best, "subsetEnsembleWeighted": True})])
    outcome = challenge(study, {0}, lambda trial: gate_objective(trial, candidates, weak), dead, make_defaults(defaults, {}),
                        timeout_seconds=60, label="self-check", margin=0.0, enabled=True, window_days=20, se_margin=0.0)
    assert outcome["decision"] == "reverted" and members_of(outcome["params"], candidates) == (candidates, True) and outcome["incumbent"] is None or outcome["incumbent"] == float("-inf"), (outcome["reason"], outcome["incumbent"])
    study = FakeStudy([FakeTrial(0, 0.0, dead), FakeTrial(1, 0.3, best)])
    outcome = challenge(study, {0}, lambda trial: gate_objective(trial, candidates, evaluate), dead, make_defaults(defaults, {}),
                        timeout_seconds=60, label="self-check", margin=0.0, enabled=True, window_days=20, se_margin=0.0)
    assert outcome["decision"] == "replaced" and members_of(outcome["params"], candidates) == (candidates, False), outcome["reason"]
    print("HyperoptEnsemble self-check OK: the gate's references, objective and decisions on hand-made data")


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
        parser.add_argument('-t', '--trials', '--max-evaluations', dest='max_evaluations', type=int,
                            default=DEFAULT_MAX_EVALUATIONS,
                            help='Evaluation budget per game (subset x weighted/flat combinations). Every '
                                 'combination is enumerated when 2^rows x 2 fits the budget (12 rows at the '
                                 'default), otherwise a deterministic local search runs within it. One '
                                 'evaluation takes 2-20 ms.')
        parser.add_argument('--record', type=int, default=DEFAULT_RECORD_TRIALS,
                            help='How many of the best evaluated combinations are written into the Optuna '
                                 'study as trials (the dashboard ranking).')
        parser.add_argument(
            '--confidence-penalty', type=float, default=DEFAULT_CONFIDENCE_PENALTY,
            help='Subset value = mean - penalty * std / sqrt(days) of the per-day score, so a subset scored '
                 'on fewer days of the window does not outrank one scored on more. 0 = plain mean.')
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '--gate-margin', type=float, default=0.0,
            help="How much this run's best combination must beat the served selection and the plain vote by, "
                 "re-scored on this run's days, before it is written (src/TuningGate.py)")
        parser.add_argument(
            '--gate-se-margin', type=float, default=GATE_SE_MARGIN,
            help="The margin in standard errors of the challenger's own score; the larger of this and --gate-margin applies")
        parser.add_argument('--no-gate', action='store_true', help="Write this run's best combination without the comparison")
        parser.add_argument(
            '-g', '--games',
            type=str,
            default=",".join(GAMES),
            help='Comma-separated list of games, e.g. "keno,lotto"'
        )

        args = parser.parse_args()

        print_intro()

        evaluationDayCount = int(args.days)
        maxEvaluations = max(4, int(args.max_evaluations))
        recordLimit = max(1, int(args.record))
        pushToGit = bool(args.save)
        confidencePenalty = max(0.0, float(args.confidence_penalty))
        GATE_ENABLED = not args.no_gate
        GATE_MARGIN = float(args.gate_margin)
        GATE_SE_MARGIN = float(args.gate_se_margin)

        print("Push to git: ", pushToGit)
        print("Evaluation budget per game: ", maxEvaluations)

        games = [g.strip() for g in args.games.split(',') if g.strip()]
        unknown_games = [g for g in games if g not in GAMES]
        if unknown_games:
            print(f"Unknown or positional game(s), ignoring: {unknown_games}")
        print("Selected games:", games)

        path = os.getcwd()
        optunaDatabase = "sqlite:///db.sqlite3"
        runTag = datetime.now().strftime("%Y-%m-%d %H:%M")

        for dataset_name in GAMES:
            if dataset_name not in games:
                continue
            try:
                tune_game(dataset_name, path, evaluationDayCount, maxEvaluations, recordLimit, confidencePenalty,
                          optunaDatabase, runTag)
            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            if pushToGit:
                helpers.git_push(commit_message="Saving latest subset ensemble hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
