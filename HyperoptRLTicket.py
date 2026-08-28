import os, argparse, json, sys, time
import optuna
from art import text2art
from datetime import datetime

from src.RLTicketModel import RLTicketModel
from src.Helpers import Helpers

helpers = Helpers()

LOCK_FILE = os.path.join(os.getcwd(), "process.lock")

# Unlike the other tuners this one never touches the training CSVs - the RL
# Ticket Model consumes the pipeline's own day JSONs (data/database/<game>).
# Per game we only need the main-ticket size as a fallback (when a day's rows
# can't provide it) and the trailing special-column count, so stored rows and
# realResults can be cut back to the main numbers exactly like Predictor.py's
# addRLTicketPrediction does. Lotto's unplayed bonus needs no entry of its own:
# scoring slices realResult[:drawSize], which drops any trailing extra column
# (stars/dream/viking AND the bonus) in one move.
GAME_CONFIG = {
    "euromillions": {"draw_size": 5, "special_column_count": 2},
    "lotto":        {"draw_size": 6, "special_column_count": 0},
    "eurodreams":   {"draw_size": 6, "special_column_count": 1},
    "keno":         {"draw_size": 20, "special_column_count": 0},
    "pick3":        {"draw_size": 3, "special_column_count": 0},
    "vikinglotto":  {"draw_size": 6, "special_column_count": 1},
}

# A game needs at least this many scoreable days before tuning on it means
# anything - below that the mean-per-day objective is mostly draw luck.
MIN_EVALUATION_DAYS = 10


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
    ascii_art = text2art("RL Ticket Hyperopt")
    print("============================================================")
    print("RL Ticket Hyperopt")
    print("Licence : MIT License")
    print(ascii_art)
    print("Find best parameters for the RL Ticket Model")


def clear_folder(folderPath):
    """
    Empties (files only) the per-trial policy cache so every trial starts from
    a blank policy instead of warm-starting off whatever the previous trial's
    hyperparameters learned - that cross-contamination would make trial scores
    incomparable. Missing folder is fine: _savePolicy creates it on demand.
    """
    try:
        for filename in os.listdir(folderPath):
            filePath = os.path.join(folderPath, filename)
            if os.path.isfile(filePath):
                os.remove(filePath)
    except FileNotFoundError:
        pass


def get_keno_subset_sizes(name, bestParams_json_object):
    """
    Same lookup as Predictor.py's getKenoSubsetSizes: Keno is the only game
    with sub-selections, and the use_5..use_10 toggles are the hyperopt-tuned
    global choice every row respects. Reused here so the tuner scores the RL
    row on exactly the subset sizes production would actually bet.
    """
    if "keno" not in name:
        return []
    return [size for size in (5, 6, 7, 8, 9, 10) if bestParams_json_object.get(f"use_{size}")]


def load_evaluation_days(historyDir, days):
    """
    The most recent `days` day JSONs that can be honestly scored: they need a
    non-empty currentPrediction (the rows the RL model consumes as features -
    the same rows Predictor.py hands it live) AND a realResult (the draw the
    emitted ticket is scored against). Returned chronologically so the
    trial's day loop can warm-start the policy day-over-day in draw order,
    mirroring how production retrains it every day.
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
    return entries[-days:]


def derive_draw_size(rows, special_column_count, fallback_draw_size):
    """
    Same main-ticket-size derivation as addRLTicketPrediction: stored rows
    append the special columns after the mains, so the first non-empty main
    ticket's length minus the special count IS the main count. The static
    GAME_CONFIG size is only the fallback for a malformed day.
    """
    rowWithTicket = next((row for row in rows if row.get("predictions") and row["predictions"][0]), None)
    if rowWithTicket is not None:
        return len(rowWithTicket["predictions"][0]) - special_column_count
    return fallback_draw_size


def score_rl_row(rlRow, realResult, drawSize, is_pick3, is_keno):
    """
    One day's objective contribution, in the same currency the other tuners'
    score_from_summary uses: euro profit where a real payout table exists
    (Pick3 straight/box/pair, Keno subsets), main-ticket hit count elsewhere.
    realResult is sliced to drawSize first - stored realResults carry the
    trailing special columns (stars/dream/viking) and lotto's bonus, and
    counting those as scoreable mains would reward numbers the RL row never
    even constructs.
    """
    predictions = (rlRow or {}).get("predictions") or []
    mainTicket = predictions[0] if predictions else []
    if not mainTicket:
        # run() is documented never to raise and to always emit a ticket, so
        # an empty row is a hard "this day earned nothing" rather than a skip.
        return 0.0

    if is_pick3:
        profit = helpers.pick3_ticket_profit(mainTicket, realResult[:3])
        return float(profit) if profit is not None else 0.0

    if is_keno:
        # The 20-number Keno main ticket has no payout of its own - only the
        # 5-10-number subset rows (predictions[1:]) are playable bets, so the
        # day's profit is the sum over those.
        dayProfit = 0.0
        for subset in predictions[1:]:
            profit = helpers.keno_ticket_profit(subset, realResult)
            if profit is not None:
                dayProfit += float(profit)
        return dayProfit

    realMains = set(int(n) for n in realResult[:drawSize])
    return float(len(set(int(n) for n in mainTicket) & realMains))


def objective_rl_ticket(trial, dataset_name, game_cfg, evaluation_days, historyDir,
                        policyDir, numberRange, kenoSubsetSizes, maxTrainSeconds):
    """
    Honest walk-forward over the pipeline's own day JSONs: for each evaluation
    day D the model is trained with cutoffDate=D (so it only ever sees days
    strictly before D - the same look-ahead guard the history rebuild uses)
    and its emitted ticket is scored against D's realResult. Trial value is
    the mean score per day: profit/day for the payout games, avg hits
    otherwise - same spirit as score_from_summary in the other tuners.
    """
    is_pick3 = "pick3" in dataset_name
    is_keno = "keno" in dataset_name

    # Fresh policy per trial - and pointed at the hyperopt cache, NEVER at
    # data/models/rl_model: the live policy there is what production warm
    # starts from, and a tuning trial scribbling over it would poison every
    # real prediction until the next daily retrain.
    clear_folder(policyDir)

    # Fresh instance per trial (rather than Predictor.py's module-level
    # singleton) so no internal state (_mainCount etc.) bleeds between trials.
    rlTicket = RLTicketModel()
    rlTicket.setModelPath(policyDir)
    rlTicket.setLearningRate(trial.suggest_float('rlTicketLearningRate', 0.005, 0.2, log=True))
    rlTicket.setEpochs(trial.suggest_int('rlTicketEpochs', 10, 60, step=10))
    rlTicket.setSamplesPerDay(trial.suggest_categorical('rlTicketSamplesPerDay', [16, 32, 64]))
    rlTicket.setTrainDays(trial.suggest_categorical('rlTicketTrainDays', [60, 120, 240]))
    # Deliberately untuned and kept at the production cap: a high-epoch trial
    # must be scored as production would actually run it - truncated by the
    # wall clock - not as an uncapped idealization production never executes.
    rlTicket.setMaxTrainSeconds(maxTrainSeconds)
    # Fixed seed so identical params always produce identical scores - trial
    # differences should come from the hyperparameters, not REINFORCE sampling
    # luck (same reasoning as the Backtester's per-day reseed in the other
    # tuners).
    rlTicket.setSeed(42)

    scores = []
    for fileDate, rows, realResult in evaluation_days:
        drawSize = derive_draw_size(rows, game_cfg["special_column_count"], game_cfg["draw_size"])
        gameConfig = {
            "numberRange": numberRange,
            "drawSize": drawSize,
            "kenoSubsetSizes": kenoSubsetSizes,
            "isPick3": is_pick3,
            "perPositionClasses": 10,
            # The cutoff makes training see only days strictly before D, even
            # though every later day JSON already sits on disk with its
            # realResult - without it the trial would be scored on data the
            # policy peeked at.
            "cutoffDate": fileDate,
        }
        # Within the trial the policy warm-starts day-over-day chronologically
        # (run() persists to policyDir and reloads it next call) - kept on
        # purpose, since that is exactly production's online behavior: each
        # day retrains on top of yesterday's theta.
        rlRow = rlTicket.run(dataset_name, rows, historyDir, gameConfig)
        scores.append(score_rl_row(rlRow, realResult, drawSize, is_pick3, is_keno))

    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


if __name__ == "__main__":
    if is_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    if not create_lock():
        print("Failed to create lock file. Exiting.")
        sys.exit(1)

    try:
        try:
            helpers.git_pull()
        except Exception as e:
            print("Failed to get latest changes")

        parser = argparse.ArgumentParser(
            prog='Sequence Predictor',
            description='Tries to predict a sequence of numbers',
            epilog='Check it out'
        )

        # 40 scoreable days: enough draws that one lucky jackpot-tier payout
        # doesn't fully own the mean, while a full study still finishes in
        # minutes (run() is wall-clock capped per day).
        parser.add_argument('-d', '--days', type=int, default=40)
        parser.add_argument('-t', '--trials', type=int, default=20)
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '-g', '--games',
            type=str,
            default=",".join(GAME_CONFIG.keys()),
            help='Comma-separated list of games, e.g. "keno,pick3"'
        )

        args = parser.parse_args()

        print_intro()

        evaluationDayCount = int(args.days)
        n_trials = int(args.trials)
        pushToGit = bool(args.save)

        print("Push to git: ", pushToGit)
        print("Running ", n_trials, "trials")

        games = [g.strip() for g in args.games.split(',') if g.strip()]
        unknown_games = [g for g in games if g not in GAME_CONFIG]
        if unknown_games:
            print(f"Unknown game(s), ignoring: {unknown_games}")
        print("Selected games:", games)

        path = os.getcwd()
        optunaDatabase = "sqlite:///db.sqlite3"
        # One shared scratch policy dir is enough: games run sequentially and
        # every trial clears it before use.
        policyDir = os.path.join(path, "data", "hyperOptCache", "rl_model")

        for dataset_name, game_cfg in GAME_CONFIG.items():
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

                jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
                existingData = {}
                if os.path.exists(jsonBestParamsFilePath):
                    with open(jsonBestParamsFilePath, "r") as infile:
                        existingData = json.load(infile)

                kenoSubsetSizes = get_keno_subset_sizes(dataset_name, existingData)
                if "keno" in dataset_name and not kenoSubsetSizes:
                    # Without playable subset sizes every Keno day scores a
                    # constant 0 (the 20-number main ticket has no payout), so
                    # the study would just pick noise.
                    print(f"Skipping {dataset_name}: no use_5..use_10 subset sizes enabled in "
                          f"bestParams_{dataset_name}.json, nothing to score profit on")
                    continue

                # Same numberRange derivation as addRLTicketPrediction:
                # get_unique_labels only pattern-matches on the path string, and
                # eurodreams gets the 1-40 override because get_unique_labels
                # has no branch for it (changing it there would invalidate the
                # saved DL weight fingerprints for the game).
                dataPath = os.path.join(path, "data", "trainingData", dataset_name)
                mainLabels = helpers.get_unique_labels(dataPath)
                numberRange = (int(min(mainLabels)), int(max(mainLabels)))
                if "eurodreams" in dataset_name:
                    numberRange = (1, 40)

                maxTrainSeconds = existingData.get("rlTicketMaxTrainSeconds", 60)

                studyName = f"{dataset_name}-rl_ticket"
                study = optuna.create_study(
                    direction='maximize',
                    storage=optunaDatabase,
                    study_name=studyName,
                    load_if_exists=True
                )

                objective = lambda trial: objective_rl_ticket(
                    trial, dataset_name, game_cfg, evaluation_days, historyDir,
                    policyDir, numberRange, kenoSubsetSizes, maxTrainSeconds
                )

                studyStart = time.time()
                study.optimize(objective, n_trials=n_trials)
                print(f"Study {studyName} finished in {time.time() - studyStart:.1f}s")

                print(f"Best Parameters for {studyName}: ", study.best_params)
                print(f"Best Score for {studyName}: ", study.best_value)

                existingData.update(study.best_params)

                with open(jsonBestParamsFilePath, "w+") as outfile:
                    json.dump(existingData, outfile, indent=4)

                # Leave no half-trained trial policy behind - the next tool in
                # runHyperopt.sh shares the hyperOptCache tree.
                clear_folder(policyDir)

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            if pushToGit:
                helpers.git_push(commit_message="Saving latest RL ticket hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
