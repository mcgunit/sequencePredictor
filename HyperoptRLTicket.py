import os, argparse, json, math, sys, time, shutil
# Pin the BLAS pools before numpy is imported: the REINFORCE updates are small
# matrix ops, and with trials running in parallel (HyperoptRunner) every
# process would otherwise spin up a 16-thread OpenBLAS pool and thrash the box.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import optuna
from multiprocessing import cpu_count
from art import text2art
from datetime import datetime

from src.RLTicketModel import RLTicketModel
from src.RLPositionModel import RLPositionModel, ROW_NAME as POSITION_ROW
from src.MarketSettle import cash_pnl
from src.TuningScore import score_bets_by_day, attrs_for_trial, describe
from src.TuningGate import challenge, make_defaults
from src.Lockbox import (load_or_warn as load_lockbox, covers as lockbox_covers, split_by_date as lockbox_split_days,
                         describe as describe_lockbox, describe_withheld as describe_lockbox_withheld)
from src.Helpers import Helpers
from src.HyperoptRunner import (open_study, fail_stale_running_trials, has_completed_trials, optimize_study,
                                install_sigterm_handler)

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
    # The market games tune the RL POSITION model instead (README item 4, M4,
    # since 8 Oct 2026): the RL Ticket row is not served there; the position
    # row learns how much to put on the vote's call. draw_size = instruments.
    "crypto":       {"draw_size": 5, "special_column_count": 0, "position": True},
    "shares":       {"draw_size": 4, "special_column_count": 0, "position": True},
    "cryptoweek":   {"draw_size": 5, "special_column_count": 0, "position": True},
    "sharesweek":   {"draw_size": 4, "special_column_count": 0, "position": True},
}

# A game needs at least this many scoreable days before tuning on it means
# anything - below that the mean-per-day objective is mostly draw luck.
MIN_EVALUATION_DAYS = 10

# The untuned knobs the gate compares a ticket study's challenger with: what Predictor.addRLTicketPrediction
# serves when a key is missing (rlTicketMaxTrainSeconds is not searched - it stays at the production cap).
RL_TICKET_DEFAULTS = {"rlTicketLearningRate": 0.05, "rlTicketEpochs": 30, "rlTicketSamplesPerDay": 32, "rlTicketTrainDays": 120}
TICKET_ROW = "RL Ticket Model"
# The ticket studies were renamed when they came under the gate (13 Oct 2026): their earlier trials were scored on
# an uncapped mean per day (pick3's 2026 best was one 500 EUR straight; keno's a 13 Sep trial on the 1-80 game with
# one ticket size), a scale the capped bound below does not share - a fresh study keeps the sampler off them.
TICKET_STUDY_SUFFIX = "rl_ticket_gated"

# What Predictor.addRLPositionPrediction serves for a key bestParams_<market>.json
# does not have - the gate's untuned reference (src/TuningGate.make_defaults).
RL_POSITION_DEFAULTS = {
    "rlPositionLearningRate": 0.05,
    "rlPositionEpochs": 30,
    "rlPositionSamplesPerDay": 16,
    "rlPositionTrainDays": 120,
    "rlPositionRiskAversion": 0.1,
}
# The chain's champion/challenger gate (src/TuningGate.py), set from the CLI:
# every study (the lottery ticket studies since 13 Oct 2026, the market
# position studies since 8 Oct) writes this run's best trial only when it
# beats the served parameters and the untuned defaults, re-scored on the same days.
GATE_ENABLED = True
GATE_MARGIN = 0.0
# Both gates (ticket and position) also take GATE_SE_MARGIN standard errors of the challenger's own score (TuningGate.
# challenge se_margin, 27 Oct 2026; until then any gain replaced): on 17 and 24 Oct the ticket studies replaced 9 rows,
# every one by 0.00-0.59 SE (and on 24 Oct lotto's challenger, incumbent and default all scored 0.7607 - a flat
# objective), and at one SE none of the 12 ticket row-weeks would have been replaced. A margin also absorbs the
# trials' own noise: training is capped by the wall clock (maxTrainSeconds 60) and trials run 15 at a time while the
# references run alone, so the same knobs can score differently (the untuned default drifted 0.36 SE between the weeks).
GATE_SE_MARGIN = 1.0
# ... while the revert takes none (TuningGate.challenge revert_margin, 27 Oct 2026): the untuned default is served back
# whenever it scores above the incumbent. Not because these scores are exact - they are not (above) - but because the
# default is the unselected null: it is not picked from trials, and on a row without skill going back to it costs
# nothing in expectation, so part of these reverts will be noise and that is harmless. Replayed: 5 ticket reverts on
# 17 Oct (incumbents the ungated studies had written, which the defaults beat), 2 on 24 Oct.
GATE_REVERT_MARGIN = 0.0
# The position row's sit-out floor (position_sit_out, 27 Oct 2026, approved by the owner): the served sizing is
# served only when its score - already the lower bound, mean minus one SE - is above SIT_OUT_SE_MARGIN of its own
# standard errors, i.e. its mean money per position is more than two standard errors above the 0 of never betting.
# Until then any bound above 0 served: on 24 Oct crypto served a sizing at +0.0084 with an SE of 0.0128 (mean size
# 0.0225, t 1.66) after its incumbent had held the row at +3e-19 - a policy that bet on one day of 40, where the
# bound is 0 by construction and "above 0" was decided by rounding. None of the eight market-weeks of 17 and 24 Oct
# clears it (the served sizings' t was at most 1.66), so the row will mostly sit out. Picked from 15 trials, t > 2 is a
# minimum bar, not a calibrated 2.3% test. No floor on the exposure: the mean size is recorded (over all positions and
# over the non-zero ones) for the reader only.
SIT_OUT_SE_MARGIN = 1.0
# The lockbox (src/Lockbox.py), loaded in __main__ and handed to every study (26 Oct 2026; until then this tuner had
# no lockbox code - 6 of vikinglotto's 40 ticket days were lockbox draws). Its days stay in the walk-forward: the
# policy is retrained and warm-started through them as production does every day (fitting on history that holds
# lockbox draws is allowed - the daily predictor does it), but they are never scored - not in a trial, not in the
# gate's re-scores of the served and the default knobs, not in the plain rule the position studies are read against.
# The window is not lengthened to make up for them.


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


def rl_row_material(rlRow, realResult, drawSize, is_pick3, is_keno):
    """
    One day of the RL ticket row as the shared objective's material
    (src/TuningScore.score_bets_by_day), so a ticket study is scored like the
    statistical and boosting rows (13 Oct 2026; until then the mean of
    a per-day profit or hit count, uncapped): (bets, hits). Keno: one (profit, size)
    bet per served subset - capped, the fair ticket of its size subtracted;
    pick3: the ticket's profit as one bet and its digits in the right slot
    as the hits (scored positional, like the pick3 rows); the other games:
    the main hits, no bets. An empty row is a day with 0 hits for pick3 and
    the hits games; a keno row without a playable (5-10-number) subset places
    no bet and, as in TuningScore, is not a day.
    """
    predictions = (rlRow or {}).get("predictions") or []
    mainTicket = predictions[0] if predictions else []
    if is_keno:
        bets = []
        for subset in predictions[1:]:
            profit = helpers.keno_ticket_profit(subset, realResult)
            if profit is not None:
                bets.append((float(profit), len(subset)))
        return bets, None
    if is_pick3:
        if not mainTicket:
            return [], 0
        profit = helpers.pick3_ticket_profit(mainTicket, realResult[:3])
        slots = sum(1 for a, b in zip(mainTicket[:3], realResult[:3]) if int(a) == int(b))
        return ([float(profit)] if profit is not None else []), slots
    if not mainTicket:
        return [], 0
    realMains = set(int(n) for n in realResult[:drawSize])
    return [], len(set(int(n) for n in mainTicket) & realMains)


def position_bets(sizes, returns_of_day):
    """One day's paper money per instrument: the size times the daily rule's cash P&L (MarketSettle.cash_pnl)."""
    return [float(size) * cash_pnl(float(ret)) for size, ret in zip(sizes, returns_of_day)]


def plain_rule_score(dataset_name, evaluation_days, returns, modelScores, lockbox=None):
    """
    The reference every position study is read against: size 1 where the
    vote's call is up, 0 where it is not - the row's own fallback without a
    policy. Same days, same money, same objective. The vote is the model's
    own (_voteTicket: the WeightedEnsemble voters, the market tie-break,
    XGBoost and LightGBM as one voter), so since 25 Oct 2026 the stored days'
    stuck deep-learning rows, baseline and aggregates no longer vote here
    either, and the two boosting twins no longer vote twice - this reference
    and the trials (objective_rl_position runs the model itself, its
    upShare and consensus features too) follow production's rule. The
    lockbox days are not scored, as in the trials.
    """
    model = RLPositionModel()
    bets_by_day = []
    for fileDate, rows, _ in evaluation_days:
        if lockbox_covers(lockbox, fileDate):
            continue
        ret = returns.get(fileDate.strftime("%Y-%m-%d"))
        rows = [r for r in rows if r.get("predictions") and r["predictions"][0]]
        if ret is None or not rows:
            continue
        slots = len(rows[0]["predictions"][0])
        ticket = model._voteTicket(rows, slots, modelScores, dataset_name)
        if ticket is None:
            continue
        bets_by_day.append(position_bets([1.0 if b >= model.k / 2 else 0.0 for b in ticket], ret))
    return score_bets_by_day(bets_by_day, payout=True)


def objective_rl_position(trial, dataset_name, evaluation_days, historyDir, dataPath, policyDir,
                          modelScores, maxTrainSeconds, returns, lockbox=None):
    """
    The position model's walk-forward (README item 4, M4): for each evaluation
    day D the policy is retrained with cutoffDate=D on the market's day JSONs
    (warm-started day over day inside the trial, as production does), sizes
    the rows made FOR D, and is paid the daily rule's paper money times the
    size on D's real returns. Trial value: the lower confidence bound of the
    per-day capped money per position (src/TuningScore, the same objective
    as the other tuners; the 20 cap, in the market's own currency, touches a
    daily position only at double size on a ten-percent day but bites often
    on a week game, where ten-percent weeks are routine). The utility the policy trains on is
    not the objective: rlPositionRiskAversion is a knob here, the money is
    what is scored. A lockbox day is walked through (the policy trains and
    warm-starts on it) but not paid or scored (26 Oct 2026).
    """
    policyDir = os.path.join(policyDir, f"trial_{os.getpid()}")
    clear_folder(policyDir)
    model = RLPositionModel()
    model.setModelPath(policyDir)
    model.setLearningRate(trial.suggest_float('rlPositionLearningRate', 0.005, 0.2, log=True))
    model.setEpochs(trial.suggest_int('rlPositionEpochs', 10, 60, step=10))
    model.setSamplesPerDay(trial.suggest_categorical('rlPositionSamplesPerDay', [8, 16, 32]))
    model.setTrainDays(trial.suggest_categorical('rlPositionTrainDays', [60, 120, 240]))
    model.setRiskAversion(trial.suggest_float('rlPositionRiskAversion', 0.0, 0.5))
    # untuned, the production cap: a trial is scored as production would run it
    model.setMaxTrainSeconds(maxTrainSeconds)
    model.setSeed(42)
    bets_by_day = []
    sizes_taken = []
    for fileDate, rows, _ in evaluation_days:
        ret = returns.get(fileDate.strftime("%Y-%m-%d"))
        if ret is None:
            continue
        row = model.run(dataset_name, rows, historyDir, dataPath,
                        {"cutoffDate": fileDate, "modelScores": modelScores, "returns": returns})
        if not row or not row.get("positions") or lockbox_covers(lockbox, fileDate):
            continue
        bets_by_day.append(position_bets(row["positions"], ret))
        sizes_taken.extend(float(s) for s in row["positions"])
    shutil.rmtree(policyDir, ignore_errors=True)
    tuning = score_bets_by_day(bets_by_day, payout=True)
    # A policy that sits out everywhere scores exactly 0 - no money, no spread -
    # and under the bound that is the ceiling of a market whose calls lose
    # money net of fees; a sizing is served only when it beats that, which
    # is what sizing is for (tune_position's floor: rlPositionSitOut). The
    # mean size travels with the record so a reader can tell a sizing from a
    # near sit-out - over every position, and (27 Oct 2026) over the ones it
    # took: a policy betting size 1 on 8% of the instrument-days averages
    # 0.08 like one betting 0.08 everywhere, and only the second is near-flat.
    tuning["mean_size"] = float(sum(sizes_taken) / len(sizes_taken)) if sizes_taken else None
    taken = [s for s in sizes_taken if s > 0]
    tuning["mean_nonzero_size"] = float(sum(taken) / len(taken)) if taken else None
    trial.set_user_attr("tuning", attrs_for_trial(tuning))
    print(f"Trial {trial.number}: {describe(tuning)}")
    return tuning["score"]


def scored_days(dataset_name, evaluation_days, lockbox):
    """
    (withheld, scored) of a walk-forward window under the lockbox, logged in
    the other tuners' words - the days are walked, only `scored` are scored.
    """
    withheld = len(lockbox_split_days(evaluation_days, lockbox)[1])
    if lockbox:
        print(describe_lockbox_withheld(dataset_name, lockbox, withheld, len(evaluation_days) - withheld))
    return withheld, len(evaluation_days) - withheld


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def position_sit_out(served_score, served_tuning, enabled=True, previous=False, margin=0.0, se_margin=SIT_OUT_SE_MARGIN):
    """
    Whether the RL Position row sits out after the gate (rlPositionSitOut): (sit_out, reason, floor), with floor the
    score the served sizing had to beat (None when nothing was compared). A policy that never bets scores exactly 0,
    so the sit-out is a fourth reference the gate itself does not know. The served sizing - the challenger, the
    default or the incumbent, whichever the gate serves - is served only when its score (the lower bound, mean minus
    one SE) is above `margin` and above se_margin of its own standard errors (SIT_OUT_SE_MARGIN, 27 Oct 2026): with
    se_margin 1 its mean money per position is more than two standard errors above 0. Fail closed: a sizing without
    a standard error (None or 0: one day, no spread) sits out. se_margin 0 is the rule until then - any score above
    `margin`. The gate disabled: the written sizing is served; nothing scored this run: the last setting (`previous`)
    stays. The mean sizes in the reason are for the reader only - there is no floor on the exposure.
    """
    if not enabled:
        return False, "gate disabled - the written sizing is served", None
    if served_score is None:
        # nothing was scored this run (no finite trial, an unscorable reference): the last decision stands
        return bool(previous), f"no scored sizing this run - the sit-out setting stays {'on' if previous else 'off'}", None
    tuning = served_tuning or {}
    sizes = ""
    if _number(tuning.get("mean_size")):
        sizes = f" - mean size {tuning['mean_size']:.3f}" + (
            f" ({tuning['mean_nonzero_size']:.3f} on the positions it took)" if _number(tuning.get("mean_nonzero_size")) else "")
    if not se_margin or se_margin <= 0:
        if served_score > margin:
            return False, f"served sizing {served_score:+.4f} beats the 0 of never betting{sizes}", margin
        return True, f"served sizing {served_score:+.4f} does not beat the 0 of never betting{sizes} - the row sits out", margin
    se = tuning.get("score_se")
    if not (_number(se) and se > 0):
        return True, (f"served sizing {served_score:+.4f} has no standard error ({'none' if se is None else repr(se)}), "
                      f"so its lead over the 0 of never betting cannot be measured{sizes} - the row sits out"), None
    floor = max(margin, se_margin * se)
    errors = "one standard error" if se_margin == 1 else f"{se_margin:g} standard errors"
    # the mean in standard errors: the score is mean - 1 SE (TuningScore's penalty), so mean / SE = score / SE + 1
    t = served_score / se + 1
    lead = (f"its mean is {abs(t):.2f} standard errors {'above' if t >= 0 else 'below'} 0; "
            f"it needs more than {floor / se + 1:.3g} above")
    if served_score > floor:
        return False, (f"served sizing {served_score:+.4f} (SE {se:.4f}) beats the 0 of never betting by more than "
                       f"{errors} of its own: {lead}{sizes}"), floor
    return True, (f"served sizing {served_score:+.4f} (SE {se:.4f}) does not beat the 0 of never betting by {errors} "
                  f"of its own: {lead}{sizes} - the row sits out"), floor


def tune_position(dataset_name, path, policyDir, evaluationDayCount, n_trials, parallel, optunaDatabase, lockbox=None):
    """One market game's position study, gated, written to bestParams_<market>.json."""
    historyDir = os.path.join(path, "data", "database", dataset_name)
    dataPath = os.path.join(path, "data", "trainingData", dataset_name)
    returns = RLPositionModel()._loadReturns(dataPath)
    evaluation_days = [e for e in load_evaluation_days(historyDir, evaluationDayCount)
                       if e[0].strftime("%Y-%m-%d") in returns]
    withheld, scored = scored_days(dataset_name, evaluation_days, lockbox)
    if scored < MIN_EVALUATION_DAYS:
        print(f"Skipping {dataset_name}: only {scored} day JSONs with rows, a result and a return"
              f"{' outside the lockbox' if withheld else ''} (need at least {MIN_EVALUATION_DAYS})")
        return
    print(f"Evaluating the position model on {len(evaluation_days)} days "
          f"({evaluation_days[0][0].date()} .. {evaluation_days[-1][0].date()})"
          + (f", {scored} of them scored" if withheld else ""))
    jsonBestParamsFilePath = os.path.join(path, f"bestParams_{dataset_name}.json")
    existingData = {}
    if os.path.exists(jsonBestParamsFilePath):
        with open(jsonBestParamsFilePath, "r") as infile:
            existingData = json.load(infile)
    modelScores = existingData.get("modelScores")
    maxTrainSeconds = existingData.get("rlPositionMaxTrainSeconds", 60)
    plain = plain_rule_score(dataset_name, evaluation_days, returns, modelScores, lockbox)
    print(f"  plain rule (size 1 where the vote is up): {describe(plain)}")
    shutil.rmtree(policyDir, ignore_errors=True)
    studyName = f"{dataset_name}-rl_position"
    study = open_study(studyName, optunaDatabase, parallel=parallel)
    fail_stale_running_trials(study)
    known_trials = {t.number for t in study.get_trials(deepcopy=False)}
    objective = lambda trial: objective_rl_position(trial, dataset_name, evaluation_days, historyDir, dataPath,
                                                    policyDir, modelScores, maxTrainSeconds, returns, lockbox)
    studyStart = time.time()
    optimize_study(studyName, optunaDatabase, objective, n_trials, parallel=parallel, expected_trial_gb=0.3)
    print(f"Study {studyName} finished in {time.time() - studyStart:.1f}s")
    study = open_study(studyName, optunaDatabase, parallel=parallel, quiet=True)
    outcome = challenge(study, known_trials, objective, existingData, make_defaults(RL_POSITION_DEFAULTS, existingData),
                        timeout_seconds=len(evaluation_days) * (maxTrainSeconds + 5) + 120, label=studyName,
                        margin=GATE_MARGIN, enabled=GATE_ENABLED, window_days=len(evaluation_days),
                        lockbox_withheld=withheld, se_margin=GATE_SE_MARGIN, revert_margin=GATE_REVERT_MARGIN)
    print(outcome["summary"])
    record = outcome["record"]
    record["plain_rule"] = attrs_for_trial(plain)
    if outcome["params"] is not None:
        existingData.update(outcome["params"])
    # The floor the gate itself does not know: a policy that never bets scores exactly 0, so the sit-out is a
    # fourth reference. When what the gate serves (the challenger, the default or the incumbent) does not beat
    # it clearly - by one standard error of its own score on top of the bound since 27 Oct 2026 (position_sit_out)
    # - the row is served sitting out - the tuned knobs are still written, so the policy they describe is what the
    # next run compares - and the record says so. Until 13 Oct 2026 the comment above promised this floor but
    # nothing applied it: the 10 Oct gate served sizings below 0 on crypto and shares.
    served_tuning = outcome.get("served_tuning") or {}
    sit_out, reason, floor = position_sit_out(outcome["served_score"], served_tuning, enabled=GATE_ENABLED,
                                              previous=existingData.get("rlPositionSitOut", False), margin=GATE_MARGIN)
    record["sit_out"] = bool(sit_out)
    record["sit_out_reason"] = reason
    # the score the served sizing had to beat (None: nothing compared), and its exposure - for the reader only
    record["sit_out_floor"] = floor
    record["sit_out_se_margin"] = SIT_OUT_SE_MARGIN
    record["served_mean_size"] = served_tuning.get("mean_size")
    record["served_mean_nonzero_size"] = served_tuning.get("mean_nonzero_size")
    existingData["rlPositionSitOut"] = bool(sit_out)
    print(f"  {dataset_name}-rl_position: {record['sit_out_reason']}")
    existingData.setdefault("tuningGate", {})[POSITION_ROW] = record
    # paper money is not a slot-hit vote weight: never in modelScores (the market rows' rule)
    existingData.get("modelScores", {}).pop(POSITION_ROW, None)
    with open(jsonBestParamsFilePath, "w+") as outfile:
        json.dump(existingData, outfile, indent=4)
    shutil.rmtree(policyDir, ignore_errors=True)


def tune_ticket(dataset_name, game_cfg, path, policyDir, evaluationDayCount, n_trials, parallel, optunaDatabase,
                lockbox=None):
    """One lottery game's RL ticket study, gated, written to bestParams_<game>.json (a function since 13 Oct 2026)."""
    historyDir = os.path.join(path, "data", "database", dataset_name)

    evaluation_days = load_evaluation_days(historyDir, evaluationDayCount)
    withheld, scored = scored_days(dataset_name, evaluation_days, lockbox)
    if scored < MIN_EVALUATION_DAYS:
        print(f"Skipping {dataset_name}: only {scored} day JSONs with both a currentPrediction and a realResult"
              f"{' outside the lockbox' if withheld else ''} (need at least {MIN_EVALUATION_DAYS})")
        return
    print(f"Evaluating on {len(evaluation_days)} days "
          f"({evaluation_days[0][0].date()} .. {evaluation_days[-1][0].date()})"
          + (f", {scored} of them scored" if withheld else ""))

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
        return

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

    # Scratch policy dirs of trials a killed run left behind.
    shutil.rmtree(policyDir, ignore_errors=True)

    studyName = f"{dataset_name}-{TICKET_STUDY_SUFFIX}"
    study = open_study(studyName, optunaDatabase, parallel=parallel)
    fail_stale_running_trials(study)
    known_trials = {t.number for t in study.get_trials(deepcopy=False)}

    objective = lambda trial, dataset_name=dataset_name, game_cfg=game_cfg, evaluation_days=evaluation_days, \
                       historyDir=historyDir, numberRange=numberRange, kenoSubsetSizes=kenoSubsetSizes, \
                       maxTrainSeconds=maxTrainSeconds: objective_rl_ticket(
        trial, dataset_name, game_cfg, evaluation_days, historyDir,
        policyDir, numberRange, kenoSubsetSizes, maxTrainSeconds, lockbox)

    studyStart = time.time()
    optimize_study(studyName, optunaDatabase, objective, n_trials, parallel=parallel, expected_trial_gb=0.3)
    print(f"Study {studyName} finished in {time.time() - studyStart:.1f}s")

    study = open_study(studyName, optunaDatabase, parallel=parallel, quiet=True)
    if not has_completed_trials(study):
        print(f"No completed trials for {studyName} (all pruned/failed) - keeping existing params")
        return

    # The champion/challenger gate (src/TuningGate.py, since 13 Oct 2026 for the ticket studies too):
    # this run's best trial replaces the served knobs only if it beats them and the untuned ones,
    # both re-scored on this run's days with this objective, by one standard error of its own score
    # (GATE_SE_MARGIN, 27 Oct 2026); the untuned ones are served back when they score above the served
    # ones and the challenger did not clear them (GATE_REVERT_MARGIN). Until 13 Oct the study's all-time
    # best was written, ungated.
    outcome = challenge(study, known_trials, objective, existingData,
                        make_defaults(RL_TICKET_DEFAULTS, existingData),
                        timeout_seconds=len(evaluation_days) * (maxTrainSeconds + 5) + 120, label=studyName,
                        margin=GATE_MARGIN, enabled=GATE_ENABLED, window_days=len(evaluation_days),
                        lockbox_withheld=withheld, se_margin=GATE_SE_MARGIN, revert_margin=GATE_REVERT_MARGIN)
    print(outcome["summary"])
    existingData.setdefault("tuningGate", {})[TICKET_ROW] = outcome["record"]
    if outcome["params"] is not None:
        existingData.update(outcome["params"])

    with open(jsonBestParamsFilePath, "w+") as outfile:
        json.dump(existingData, outfile, indent=4)

    # Leave no half-trained trial policy behind (per-trial subdirs
    # included) - the next tool in runHyperopt.sh shares the
    # hyperOptCache tree.
    shutil.rmtree(policyDir, ignore_errors=True)


def objective_rl_ticket(trial, dataset_name, game_cfg, evaluation_days, historyDir,
                        policyDir, numberRange, kenoSubsetSizes, maxTrainSeconds, lockbox=None):
    """
    Honest walk-forward over the pipeline's own day JSONs: for each evaluation
    day D the model is trained with cutoffDate=D (so it only ever sees days
    strictly before D - the same look-ahead guard the history rebuild uses)
    and its emitted ticket is scored against D's realResult. Trial value is
    the tuners' common lower-bound objective (rl_row_material, 13 Oct 2026;
    the plain mean per day before, which one straight or keno tier decided).
    A lockbox day is walked through - the policy trains and warm-starts on
    it - but its ticket is not scored (26 Oct 2026).
    """
    is_pick3 = "pick3" in dataset_name
    is_keno = "keno" in dataset_name

    # Fresh policy per trial - and pointed at the hyperopt cache, NEVER at
    # data/models/rl_model: the live policy there is what production warm
    # starts from, and a tuning trial scribbling over it would poison every
    # real prediction until the next daily retrain.
    # Trials run in parallel processes (HyperoptRunner.optimize_study), so each
    # gets its own scratch policy dir - removed again at the end of the trial.
    policyDir = os.path.join(policyDir, f"trial_{os.getpid()}")
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

    bets_by_day, hits_by_day = [], []
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
        if lockbox_covers(lockbox, fileDate):
            continue
        bets, hits = rl_row_material(rlRow, realResult, drawSize, is_pick3, is_keno)
        bets_by_day.append(bets)
        if hits is not None:
            hits_by_day.append(hits)

    shutil.rmtree(policyDir, ignore_errors=True)
    # the tuners' common objective: the lower confidence bound over the days of the capped profit per bet (keno,
    # above a fair ticket of each size), of the slot hits with the capped profit as a tie-breaker (pick3), or of
    # the hits (the other games) - one lucky payout counts as one good bet, not as the study's best trial
    tuning = score_bets_by_day(bets_by_day, hits_by_day, payout=is_keno, positional=is_pick3)
    trial.set_user_attr("tuning", attrs_for_trial(tuning))
    print(f"Trial {trial.number}: {describe(tuning)}")
    return tuning["score"]


def _self_check():
    """
    The lockbox wiring (26 Oct 2026) with stub RL models, no policy trained and no database:
        python3 -c 'import HyperoptRLTicket; HyperoptRLTicket._self_check()'
    Twelve lotto days, the first four in the lockbox and each a perfect ticket: the walk-forward runs the model on
    all twelve (the policy warm-starts through the lockbox, as production does), the score sees the last eight only.
    """
    import tempfile
    import warnings
    from src.Lockbox import parse

    box = parse({"from": "2026-01-01", "to": "2026-01-04"})
    perfect, wrong = [1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]
    days = [(datetime(2026, 1, d), [{"name": "Markov Model", "predictions": [perfect if d <= 4 else wrong]}],
             perfect + [40]) for d in range(1, 13)]
    assert scored_days("lotto", days, box) == (4, 8) and scored_days("lotto", days, None) == (0, 12)

    class StubTicket:
        """Plays the stored row's ticket and remembers the days it was run for."""
        walked = []
        def __getattr__(self, name):
            return lambda *args, **kwargs: None    # the setters
        def run(self, dataset_name, rows, historyDir, gameConfig):
            StubTicket.walked.append(gameConfig["cutoffDate"])
            return {"predictions": [rows[0]["predictions"][0]]}

    module = globals()
    saved = module["RLTicketModel"], module["RLPositionModel"]
    module["RLTicketModel"] = StubTicket
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    try:
        with tempfile.TemporaryDirectory() as policyDir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            def score(lockbox):
                StubTicket.walked = []
                trial = optuna.trial.FixedTrial(dict(RL_TICKET_DEFAULTS))
                value = objective_rl_ticket(trial, "lotto", GAME_CONFIG["lotto"], days, policyDir, policyDir,
                                            (1, 45), [], 60, lockbox)
                return value, trial.user_attrs["tuning"], list(StubTicket.walked)
            value, tuning, walked = score(None)
            assert tuning["days"] == 12 and tuning["hits_mean"] == 2.0 and len(walked) == 12, tuning
            value, tuning, walked = score(box)
            assert tuning["days"] == 8 and tuning["hits_mean"] == 0.0 and value == 0.0, tuning
            assert walked == [d[0] for d in days], "every day is walked, the lockbox's too"

        # the position studies' plain rule, read against: a lockbox day pays nothing either way
        class StubPosition:
            k = 10
            def _voteTicket(self, rows, slots, modelScores, game):
                return rows[0]["predictions"][0]
        module["RLPositionModel"] = StubPosition
        market = [(datetime(2026, 1, d), [{"name": "Markov Model", "predictions": [[9, 9]]}], [9, 9]) for d in range(1, 13)]
        returns = {when.strftime("%Y-%m-%d"): [0.05, 0.05] if when.day <= 4 else [-0.01, -0.01] for when, _, _ in market}
        whole = plain_rule_score("crypto", market, returns, None)
        held = plain_rule_score("crypto", market, returns, None, box)
        assert whole["days"] == 12 and held["days"] == 8 and held["profit_per_bet"] < 0 < whole["profit_per_bet"], (whole, held)
    finally:
        module["RLTicketModel"], module["RLPositionModel"] = saved
        optuna.logging.set_verbosity(verbosity)
    _self_check_gate()
    print("HyperoptRLTicket self-check OK: the lockbox days are walked through (the policy trains on them) but never "
          "scored - not by a ticket trial, not by the position studies' plain rule; the gates take one standard error "
          "and revert without a margin; the position row sits out unless its sizing's mean is over two standard errors "
          "above 0 - every market of 24 Oct")


def _self_check_gate():
    """
    The margins of 27 Oct 2026 on production's own gate records (bestParams of 17 and 24 Oct), no policy trained:
    the ticket gate through TuningGate.challenge with the recorded scores, the position sit-out (position_sit_out) on
    the served sides of 24 Oct.
    """
    class Trial:
        def __init__(self, value, se):
            self.number, self.value, self.params = 1, value, {"rlTicketEpochs": 50}
            self.state = optuna.trial.TrialState.COMPLETE
            self.user_attrs = {"tuning": {"score": value, "kind": "hits", "days": 40, "score_se": se}}

    class Study:
        def __init__(self, trial):
            self.trial, self.best_value = trial, trial.value
        def get_trials(self, deepcopy=False, states=None):
            return [self.trial]

    served = {"rlTicketEpochs": 20}
    def gate(ch, se, inc, default, **margins):
        scores = {"incumbent": inc, "default": default}
        return challenge(Study(Trial(ch, se)), set(), None, served, make_defaults(RL_TICKET_DEFAULTS, served), 10, "self-check",
                         evaluate=lambda params, role: scores[role], **margins)["decision"]
    ours = {"se_margin": GATE_SE_MARGIN, "revert_margin": GATE_REVERT_MARGIN}
    # euromillions, 17 Oct: replaced at margin 0 (+0.5542 over +0.4569, 0.81 SE); now kept out, and the untuned knobs
    # (+0.4823), which score above the served ones, are served back - with the challenger's margin on the revert
    # too the weaker incumbent would have stayed
    assert gate(0.5542, 0.1208, 0.4569, 0.4823) == "replaced"
    assert gate(0.5542, 0.1208, 0.4569, 0.4823, **ours) == "reverted"
    assert gate(0.5542, 0.1208, 0.4569, 0.4823, se_margin=GATE_SE_MARGIN) == "kept"
    # lotto, 24 Oct: challenger, incumbent and default all at +0.7607 - kept, whatever the margins
    assert gate(0.7607, 0.1143, 0.7607, 0.7607, **ours) == "kept" and gate(0.7607, 0.1143, 0.7607, 0.7607) == "kept"
    # vikinglotto, 24 Oct: replaced on 0.22 SE; now the default (+0.8096 over +0.7990) is served back
    assert gate(0.8262, 0.1238, 0.7990, 0.8096) == "replaced" and gate(0.8262, 0.1238, 0.7990, 0.8096, **ours) == "reverted"

    # the position sit-out on the served sides of 24 Oct (score, its tuning): none clears two standard errors
    crypto = (0.008435, {"score_se": 0.012843, "mean_size": 0.0225})     # replaced and served today: mean 1.66 SE
    single = (3.25e-19, {"score_se": 0.000898, "mean_size": 0.0025})     # its incumbent: one day of 40 with a bet
    shares = (-0.041600, {"score_se": 0.018166, "mean_size": 0.025})     # below 0: sat out then, sits out now
    cryptoweek = (0.195558, {"score_se": 0.346159, "mean_size": 0.228})  # kept and served: mean 1.56 SE
    sharesweek = (0.085338, {"score_se": 0.155180, "mean_size": 0.159})  # replaced and served: mean 1.55 SE
    for name, (score, tuning) in {"crypto": crypto, "its incumbent": single, "shares": shares,
                                  "cryptoweek": cryptoweek, "sharesweek": sharesweek}.items():
        sit_out, reason, floor = position_sit_out(score, tuning)
        assert sit_out and floor == tuning["score_se"] and reason.endswith("the row sits out"), (name, reason)
    # under the rule until 27 Oct (any bound above 0) crypto, cryptoweek and sharesweek were served - the single-day
    # incumbent too, on rounding
    assert [position_sit_out(s, t, se_margin=0.0)[0] for s, t in (crypto, single, shares, cryptoweek, sharesweek)] \
        == [False, False, True, False, False]
    sit_out, reason, floor = position_sit_out(0.25, {"score_se": 0.10, "mean_size": 0.30, "mean_nonzero_size": 0.60})
    assert not sit_out and floor == 0.10 and "3.50 standard errors above 0; it needs more than 2 above" in reason \
        and "0.600 on the positions it took" in reason, reason
    sit_out, _, _ = position_sit_out(0.25, {"score_se": 0.10}, margin=0.3)
    assert sit_out, "the gate's own margin, when larger, is the floor"
    # fail closed: no standard error (none, or 0 - no spread), or no diagnostics at all, sits out
    for tuning in ({"score_se": None, "mean_size": 0.5}, {"score_se": 0.0}, {}, None):
        sit_out, reason, floor = position_sit_out(0.25, tuning)
        assert sit_out and floor is None and "no standard error" in reason, (tuning, reason)
    # the gate disabled serves the written sizing; nothing scored keeps the last setting
    assert position_sit_out(-1.0, None, enabled=False)[0] is False
    assert position_sit_out(None, None, previous=True)[:2] == (True, "no scored sizing this run - the sit-out setting stays on")
    assert position_sit_out(None, None, previous=False)[0] is False


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
            description='Tries to predict a sequence of numbers',
            epilog='Check it out'
        )

        # 40 scoreable days: enough draws that one lucky jackpot-tier payout
        # doesn't fully own the mean, while a full study still finishes in
        # minutes (run() is wall-clock capped per day).
        parser.add_argument('-d', '--days', type=int, default=40)
        parser.add_argument('-t', '--trials', type=int, default=20)
        parser.add_argument(
            '--parallel-trials', type=int, default=0,
            help='Upper bound on trials evaluated at the same time, each in its own process. 0 '
                 '(default) = auto: one per core (the REINFORCE loop is single-threaded and small), '
                 'launched only while the memory gate allows (see HyperoptRunner.optimize_study). '
                 '1 = off. The per-day wall-clock cap (rlTicketMaxTrainSeconds) is per process, '
                 'so as long as trials do not exceed the cores they do not slow each other down.')
        parser.add_argument('-s', '--save', type=helpers.str2bool, default=True)
        parser.add_argument(
            '-g', '--games',
            type=str,
            default=",".join(GAME_CONFIG.keys()),
            help='Comma-separated list of games, e.g. "keno,pick3"'
        )

        parser.add_argument(
            '--gate-margin', type=float, default=0.0,
            help='Every study (the lottery ticket studies and the market position studies): the challenger must '
                 'beat the served parameters and the untuned defaults by this much to be written (src/TuningGate.py); '
                 'the larger of this and --gate-se-margin standard errors applies.')
        parser.add_argument(
            '--gate-se-margin', type=float, default=GATE_SE_MARGIN,
            help="Every study: the margin in standard errors of the challenger's own score (since 27 Oct 2026). "
                 "0 = any gain replaces, the rule until then. The position row's sit-out floor is SIT_OUT_SE_MARGIN, not this.")
        parser.add_argument(
            '--gate-revert-margin', type=float, default=GATE_REVERT_MARGIN,
            help='How much the untuned knobs must score above the served ones to be served back when the challenger '
                 'is not written. 0 (default) = any amount: the untuned knobs are not picked from trials.')
        parser.add_argument('--no-gate', action='store_true', help="Write each study's best trial of this run without the comparison")
        args = parser.parse_args()
        GATE_ENABLED = not args.no_gate
        GATE_MARGIN = float(args.gate_margin)
        GATE_SE_MARGIN = float(args.gate_se_margin)
        GATE_REVERT_MARGIN = float(args.gate_revert_margin)

        print_intro()

        evaluationDayCount = int(args.days)
        n_trials = int(args.trials)
        pushToGit = bool(args.save)
        cores = max(1, cpu_count() - 1)
        parallel = cores if int(args.parallel_trials) <= 0 else max(1, min(int(args.parallel_trials), cores))

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
        # The lockbox's days are walked but never scored (the lockbox note
        # above is_running). A malformed declaration is a warning, said once -
        # the run then is not lockbox-safe and the warning says so.
        lockbox = load_lockbox(path)
        if lockbox:
            print(f"Tuning under the {describe_lockbox(lockbox)}: its days are walked through but never scored")

        for dataset_name, game_cfg in GAME_CONFIG.items():
            if dataset_name not in games:
                continue
            try:
                print(f"\n{dataset_name.capitalize()}")
                if game_cfg.get("position"):
                    tune_position(dataset_name, path, policyDir, evaluationDayCount, n_trials, parallel, optunaDatabase,
                                  lockbox=lockbox)
                    continue
                tune_ticket(dataset_name, game_cfg, path, policyDir, evaluationDayCount, n_trials, parallel, optunaDatabase,
                            lockbox=lockbox)

            except Exception as e:
                print(f"Failed to Hyperopt {dataset_name.capitalize()}: {e}")

        try:
            if pushToGit:
                helpers.git_push(commit_message="Saving latest RL ticket hyperopt")
        except Exception as e:
            print("Failed to push latest predictions:", e)
    finally:
        remove_lock()
