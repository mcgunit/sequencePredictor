"""
The run machinery of HyperoptDeepLearning.py that needs no TensorFlow (owner, 6 Nov 2026).

The deep-learning tuner is run by hand and takes days: on production's last study (Joker+ LSTM, 17 Sep 2026) a
trial took 41 minutes - eight trainings of up to 1,000 epochs, one per rebuilt draw - and the default run is
7 games x 7 model types x 15 trials. What this module adds, on the owner's go-ahead for A-E and the gate:

  A  resume       a run keeps a marker (data/hyperOptCache/deeplearning-run.json) with its settings, per study the
                  trials that existed when the run first opened it, and the studies it finished. The same command
                  after a kill (Ctrl+C, OOM, reboot) skips the finished studies and runs only the trials a study
                  still lacks; --fresh starts over. Until then a rerun added --trials MORE trials to every study.
  B  save per study  the gate's decision is written to bestParams_<game>.json and committed when each study ends,
                  not once at the end of the run.
  C  progress     each rebuilt day's validation loss reaches the parent as it is trained (a progress file the
                  training child writes), one log line per day.
  D  pruning      the parent reports the running score to Optuna after each day; a MedianPruner stops a trial
                  whose score after --prune-after-days days is below the median of the earlier trials at that day.
  E  production's budget  every training run is capped at --train-seconds (default 240, Predictor.py's
                  --dl-model-seconds): the daily predictor trains each model for 240 s, the tuner used to train it
                  unlimited, so it chose parameters for a regime production never runs.

The gate (src/TuningGate.py) is the other tuners' champion/challenger: this round's best trial replaces the served
parameters only if it beats them and the untuned defaults - both re-scored on the same days with the same budget -
by GATE_SE_MARGIN standard errors of its own score; the defaults are served back when they score above the
incumbent. The defaults are what Predictor.py trains when a key is missing (PREDICTOR_DEFAULTS below, kept in step
with Predictor.runUnifiedDeepLearningModels and LSTMModel's constructor).

What the score is, and what it is not (7 Nov 2026, the owner's go-ahead after the first production run): minus the
mean over the rebuilt days of the row's excess loss - its plain cross-entropy on its validation windows minus that
of the training frequencies, no label smoothing and no L2 on either (src/LearningCheck.py, excessLoss). 0 = as good
as serving the counts, above 0 = better. Until then it was minus the mean val_loss plus 0.05 x the hits (capped):
val_loss carries the smoothing and the L2 penalty, which are tuned knobs, so the first run (6 Nov, lotto) "won" by
lowering the smoothing - the transformer's 3.304 -> 3.241 and the GNN's 3.282 -> 3.210 are mostly the yardstick
moving (the counts-only baseline is ~3.27 at smoothing 0.05 and ~3.19 at 0), and every one of those rows still read
counts-only. The hits are on the record, not in the score: 0.05 a hit was ~100 standard errors. The old trials
live in the old studies (<game>-<model_type>); this score writes to <game>-<model_type>-skill (STUDY_SUFFIX), since
the sampler and the pruner's median would otherwise mix the two scales. Days default to 3 (DAYS): the 8 warm-started
days of the first run agreed to the third decimal. Still to know: on a game whose frequencies drift, an output that
only shrinks the counts toward uniform also gets above 0, without reading its input - so a positive score is not by
itself learning; the learning check's verdict, counted per day on the record (learns_days), is that test. Its
standard error is taken over those days, which are warm-started retrains on almost the same
history - strongly correlated, so it understates how much a retrain varies; one SE is a floor, not a test. The
validation split is the last 20% of each retrain's history (10 years), which contains the lockbox period
(src/Lockbox.py): this tuner does not withhold it, and its gate records say so (LOCKBOX_NOTE).

    python3 -m src.DeepLearningTuning     # self-check (in npm test)
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone

# Predictor.py's --dl-model-seconds default (DL_MODEL_SECONDS): the budget every DL training run gets in production
TRAIN_SECONDS = 240
# Prune after this many rebuilt days (the second day's report, step 1); the first PRUNE_STARTUP_TRIALS trials of a
# study are never pruned, so the median has something to stand on
PRUNE_AFTER_DAYS = 2
PRUNE_STARTUP_TRIALS = 3
GATE_SE_MARGIN = 1.0
GATE_REVERT_MARGIN = 0.0
# rebuilt days per trial (-d): the warm-started days barely differ (the 6 Nov lotto run: 3.2917 .. 3.2907 over 8)
DAYS = 3
SCORE_KIND = "skill_vs_counts"
# the studies of this score (old trials scored minus val_loss + hits; their scale must not feed the sampler/pruner)
STUDY_SUFFIX = "skill"
RUN_FILE = "deeplearning-run.json"
LOCKBOX_NOTE = ("not withheld: the score is the validation loss on the last 20% of each retrain's history, "
                "which contains the lockbox period")

# What Predictor.py trains a row with when bestParams_<game>.json lacks the key: the prefixed rows read
# bestParams.get(f"{prefix}_{name}", default) (runUnifiedDeepLearningModels, its per-model modelDefaults in
# UNIFIED_DL_MODELS); the LSTM rows keep LSTMModel's constructor values for a missing bare key.
_PREFIXED_DEFAULTS = {
    "batchSize": 16, "epochs": 1000, "lstmUnits": 64, "tcnUnits": 64, "numTcnLayers": 2, "gruUnits": 64,
    "dModel": 64, "numLayers": 2, "ffnFactor": 4, "gcnUnits": 32, "numGcnLayers": 2, "embeddingDim": 16,
    "decay": 0.999, "latentDim": 16, "encoderUnits": 64, "numEncoderLayers": 2, "dropout": 0.3,
    "l2Regularization": 0.0005, "learningRate": 0.001, "earlyStopPatience": 20, "reduceLearningRatePatience": 5,
    "reduceLearningRateFactor": 0.5, "windowSize": 20, "labelSmoothing": 0.05, "numHeads": 4, "keyDim": 32,
}
_MODEL_DEFAULTS = {"transformer": {"windowSize": 30}, "autoencoder": {"labelSmoothing": 0.0, "dropout": 0.2}}
_LSTM_DEFAULTS = {
    "epochs": 1000, "batchSize": 4, "num_lstm_layers": 1, "num_bidirectional_layers": 1, "lstm_units": 16,
    "bidirectional_lstm_units": 16, "dropout": 0.2, "l2Regularization": 0.0001, "earlyStopPatience": 20,
    "reduceLearningRatePatience": 5, "reduceLearningRateFactor": 0.9, "useFinalLSTMLayer": True,
    "outputActivation": "softmax", "optimizer_type": "adam", "learningRate": 0.005, "windowSize": 10,
    "useTopPrediction": False, "labelSmoothing": 0.1,
}


def predictor_defaults(prefix, served):
    """
    {tuner parameter name: what Predictor.py serves without it} for one model type (prefix None = the LSTM's bare
    keys). The LSTM rows read the bare yearsOfHistory; since 8 Nov 2026 every other row reads its own
    "<prefix>_yearsOfHistory" and falls back to the bare key without one (Predictor.rowYearsOfHistory), so a
    prefixed row's default is the served bare value: a revert of one study puts that row back on the shared history.
    """
    years = served.get("yearsOfHistory", 20)
    if prefix is None:
        return dict(_LSTM_DEFAULTS, yearsOfHistory=years)
    table = dict(_PREFIXED_DEFAULTS, **_MODEL_DEFAULTS.get(prefix, {}))
    out = {f"{prefix}_{name}": value for name, value in table.items()}
    out[f"{prefix}_yearsOfHistory"] = years
    return out


def study_name(game, model_type):
    return f"{game}-{model_type}-{STUDY_SUFFIX}"


def day_excess(check):
    """One rebuilt day's value from the row's learning check (src/LearningCheck.py): its excessLoss, None when the
    check was skipped or is not finite."""
    value = (check or {}).get("excessLoss") if isinstance(check, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def day_score(excess_losses, hits=None, val_losses=None, verdicts=None):
    """
    The trial's score and its diagnostics (the gate's "tuning" record): minus the mean finite excess loss over the
    rebuilt days (None or non-finite for a day without a learning check); score_se the standard error of that over
    the days (None with fewer than two). No finite day: score -inf, which the gate never writes. hits (the profit
    calculate_profit gives over the days), the days' val_loss and the learning-check verdicts are on the record only.
    """
    finite = [-float(v) for v in excess_losses if isinstance(v, (int, float)) and math.isfinite(v)]
    losses = [float(v) for v in (val_losses or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    out = {"kind": SCORE_KIND, "score": float("-inf"), "score_se": None, "days": len(finite), "excess_mean": None,
           "hits": None if hits is None else float(hits),
           "val_loss_mean": statistics.fmean(losses) if losses else None,
           "learns_days": sum(1 for v in (verdicts or []) if v == "learns")}
    if finite:
        mean = statistics.fmean(finite)
        out.update(score=mean, excess_mean=-mean,
                   score_se=statistics.stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else None)
    return out


def running_value(excess_losses):
    """What the parent reports to Optuna after a day: minus the mean finite excess so far (-1e9 when none is)."""
    finite = [float(v) for v in excess_losses if isinstance(v, (int, float)) and math.isfinite(v)]
    return -statistics.fmean(finite) if finite else -1e9


def write_progress(path, val_losses):
    """The child's per-day values (excess losses), replaced atomically so the parent never reads half a file."""
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump([v if isinstance(v, (int, float)) and math.isfinite(v) else None for v in val_losses], f)
    os.replace(tmp, path)


def read_progress(path):
    """The losses the child has written so far; [] when there is no file yet or it does not read."""
    try:
        with open(path) as f:
            values = json.load(f)
        return values if isinstance(values, list) else []
    except (OSError, ValueError):
        return []


# --- A: the run marker ---

def run_settings(games, models, trials, days, train_seconds, gate):
    """What makes two runs the same run: a resume only continues a run with these settings."""
    return {"games": list(games), "models": list(models), "trials": int(trials), "days": int(days),
            "trainSeconds": int(train_seconds or 0), "gate": bool(gate), "score": SCORE_KIND}


def open_run(folder, settings, fresh=False, log=print, now=None):
    """
    (marker, resumed). An unfinished marker with the same settings is resumed; anything else - no marker, a
    finished one, other settings, --fresh - starts a new run (and says why when one was there).
    """
    path = os.path.join(folder, RUN_FILE)
    old = None
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = None
    if old and not old.get("finished") and not fresh and old.get("settings") == settings:
        log(f"Resuming the run started {old.get('started')}: {len(old.get('done', []))} of "
            f"{len(settings['games']) * len(settings['models'])} studies done")
        return old, True
    if old and not old.get("finished"):
        log("An unfinished run is on disk " + ("- --fresh: starting over" if fresh else
            f"with other settings ({old.get('settings')}) - starting a new run"))
    marker = {"started": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
              "settings": settings, "known": {}, "done": [], "finished": False}
    save_run(folder, marker)
    return marker, False


def save_run(folder, marker):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, RUN_FILE)
    with open(path + ".tmp", "w") as f:
        json.dump(marker, f, indent=2)
    os.replace(path + ".tmp", path)


def known_trials(folder, marker, key, numbers):
    """The study's trials before this run: recorded the first time the run opens the study, read back on a resume."""
    if key not in marker["known"]:
        marker["known"][key] = sorted(int(n) for n in numbers)
        save_run(folder, marker)
    return set(marker["known"][key])


def remaining_trials(trials, known, target):
    """
    How many trials the study still needs this run: `target` minus the trials this run finished (complete or
    pruned - a failed trial measured nothing and is retried). `trials` are (number, state name) pairs.
    """
    finished = sum(1 for number, state in trials if number not in known and state in ("COMPLETE", "PRUNED"))
    return max(0, int(target) - finished)


def mark_done(folder, marker, key):
    if key not in marker["done"]:
        marker["done"].append(key)
    save_run(folder, marker)


def finish_run(folder, marker):
    marker["finished"] = True
    marker["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_run(folder, marker)


# --- the result file (--result-file; a run started from the admin pages, 11 Nov 2026) ---
# The owner's decision: a hyperopt started from the Hyperopts page runs on the GPU worker and writes nothing that is
# served - its copy of bestParams_<game>.json is thrown away - and the page shows this file instead: per study the
# trials done, the best of this run, and the gate's verdict with the parameters it would write (Apply on the page
# writes them on the main server). Replaced atomically, so a copy taken mid-write is never half a file.

def write_result(path, study_key, **fields):
    """Merges `fields` into the study's entry of the result file (no file: nothing written)."""
    if not path:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    studies = data.setdefault("studies", {})
    entry = studies.setdefault(study_key, {})
    entry.update(_plain(fields))
    data["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1, allow_nan=False)
    os.replace(path + ".tmp", path)
    return data


def _plain(v):
    """Plain JSON values: numpy scalars as numbers, a non-finite float as None (the file stays strict JSON)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if hasattr(v, "dtype") and getattr(v.dtype, "kind", "") in "iu":
        return int(v)
    return f if math.isfinite(f) else None


def trial_progress(trials, known):
    """This run's trials of a study, for the result file: counts per state and the best finite score so far."""
    mine = [t for t in trials if t.number not in known]
    counts = {}
    for t in mine:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1
    scored = [t for t in mine if t.state.name == "COMPLETE" and isinstance(t.value, (int, float)) and math.isfinite(t.value)]
    best = max(scored, key=lambda t: t.value) if scored else None
    return {"states": counts, "done": counts.get("COMPLETE", 0) + counts.get("PRUNED", 0),
            "best": None if best is None else {"trial": best.number, "score": float(best.value), "params": dict(best.params)}}


def _self_check():
    import tempfile
    lstm = predictor_defaults(None, {"yearsOfHistory": 10})
    assert lstm["batchSize"] == 4 and lstm["learningRate"] == 0.005 and lstm["yearsOfHistory"] == 10
    tr = predictor_defaults("transformer", {})
    assert tr["transformer_windowSize"] == 30 and tr["transformer_batchSize"] == 16 and tr["transformer_yearsOfHistory"] == 20
    ae = predictor_defaults("autoencoder", {"yearsOfHistory": 10})
    assert ae["autoencoder_labelSmoothing"] == 0.0 and ae["autoencoder_dropout"] == 0.2 and ae["autoencoder_yearsOfHistory"] == 10

    s = day_score([0.02, -0.01, None, float("nan")], hits=4, val_losses=[3.2, 3.3, None],
                  verdicts=["counts-only", "learns", None])
    assert s["kind"] == SCORE_KIND and s["days"] == 2 and abs(s["excess_mean"] - 0.005) < 1e-12
    assert abs(s["score"] + 0.005) < 1e-12 and abs(s["score_se"] - 0.015) < 1e-12
    assert s["hits"] == 4.0 and abs(s["val_loss_mean"] - 3.25) < 1e-12 and s["learns_days"] == 1
    # the hits never move the score
    assert day_score([0.02, -0.01], hits=100)["score"] == day_score([0.02, -0.01], hits=0)["score"]
    assert day_score([0.01])["score_se"] is None and day_score([None])["score"] == float("-inf")
    assert running_value([0.02, None, 0.04]) == -0.03 and running_value([]) == -1e9
    assert day_excess({"excessLoss": -0.004}) == -0.004 and day_excess(None) is None
    assert day_excess({"excessLoss": float("nan")}) is None and day_excess({}) is None
    assert study_name("lotto", "gnn_model") == "lotto-gnn_model-skill"

    folder = tempfile.mkdtemp()
    progress = os.path.join(folder, "p.json")
    assert read_progress(progress) == []
    write_progress(progress, [2.5, float("inf")])
    assert read_progress(progress) == [2.5, None]

    settings = run_settings(["lotto"], ["gnn_model", "transformer_model"], 15, DAYS, 240, True)
    assert settings["score"] == SCORE_KIND
    quiet = lambda message: None
    marker, resumed = open_run(folder, settings, log=quiet)
    assert not resumed and marker["done"] == []
    assert known_trials(folder, marker, "lotto-gnn_model", [0, 1, 2]) == {0, 1, 2}
    assert remaining_trials([(0, "COMPLETE"), (3, "COMPLETE"), (4, "PRUNED"), (5, "FAIL")], {0, 1, 2}, 15) == 13
    mark_done(folder, marker, "lotto-gnn_model")
    again, resumed = open_run(folder, settings, log=quiet)                        # the same command after a kill
    assert resumed and again["done"] == ["lotto-gnn_model"]
    assert known_trials(folder, again, "lotto-gnn_model", [0, 1, 2, 3, 4, 5]) == {0, 1, 2}   # not re-recorded
    other, resumed = open_run(folder, dict(settings, trials=5), log=quiet)         # other settings: a new run
    assert not resumed and other["done"] == []
    finish_run(folder, other)
    fresh, resumed = open_run(folder, dict(settings, trials=5), log=quiet)         # a finished run is not resumed
    assert not resumed and not fresh["finished"]
    forced, resumed = open_run(folder, dict(settings, trials=5), fresh=True, log=quiet)
    assert not resumed

    # The gate as HyperoptDeepLearning calls it: a real study, this module's defaults and score shape
    import optuna
    try:
        from src.TuningGate import challenge, make_defaults
        from src.TuningScore import attrs_for_trial
    except ImportError:
        from TuningGate import challenge, make_defaults
        from TuningScore import attrs_for_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna.distributions import IntDistribution
    dist = {"gnn_gcnUnits": IntDistribution(16, 64), "gnn_yearsOfHistory": IntDistribution(10, 20)}

    def study_with(score, se):
        study = optuna.create_study(direction="maximize")
        old = optuna.trial.create_trial(value=9.0, params={"gnn_gcnUnits": 16, "gnn_yearsOfHistory": 10}, distributions=dist)
        new = optuna.trial.create_trial(value=score, params={"gnn_gcnUnits": 64, "gnn_yearsOfHistory": 10}, distributions=dist,
                                        user_attrs={"tuning": attrs_for_trial({"kind": SCORE_KIND, "score": score, "score_se": se, "days": DAYS})})
        study.add_trial(old)
        study.add_trial(new)
        return study

    served = {"yearsOfHistory": 10, "gnn_gcnUnits": 48}
    def gate(study, scores):
        return challenge(study, {0}, None, served, make_defaults(predictor_defaults("gnn", served), served), 10, "lotto-gnn_model",
                         evaluate=lambda params, role: (scores[role], {"score_se": 0.01}), se_margin=GATE_SE_MARGIN,
                         revert_margin=GATE_REVERT_MARGIN)
    out = gate(study_with(-2.00, 0.02), {"incumbent": -2.10, "default": -2.20})
    assert out["decision"] == "replaced" and out["params"]["gnn_gcnUnits"] == 64, out["reason"]   # the old 9.0 is history only
    assert out["params"]["gnn_yearsOfHistory"] == 10
    out = gate(study_with(-2.09, 0.02), {"incumbent": -2.10, "default": -2.20})
    assert out["decision"] == "kept", out["reason"]                                              # inside one SE
    out = gate(study_with(-2.30, 0.02), {"incumbent": -2.25, "default": -2.20})
    assert out["decision"] == "reverted" and out["params"]["gnn_gcnUnits"] == 32, out["reason"]  # Predictor's default
    assert out["params"]["gnn_yearsOfHistory"] == 10                                             # the shared value, not 20
    # the result file: merged per study, strict JSON, numpy-free
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, "sub", "result.json")
        assert write_result(None, "x", a=1) is None
        write_result(rf, "lotto-gnn_model-skill", status="running", trials={"wanted": 5})
        write_result(rf, "lotto-gnn_model-skill", score=float("nan"), status="done")
        with open(rf) as f:
            got = json.load(f)["studies"]["lotto-gnn_model-skill"]
        assert got == {"status": "done", "trials": {"wanted": 5}, "score": None}, got

        class T:
            def __init__(self, number, state, value, params):
                self.number, self.value, self.params = number, value, params
                self.state = type("S", (), {"name": state})()
        prog = trial_progress([T(0, "COMPLETE", 9.0, {"a": 0}), T(1, "COMPLETE", -0.2, {"a": 1}), T(2, "PRUNED", None, {}),
                               T(3, "COMPLETE", -0.1, {"a": 3}), T(4, "FAIL", None, {})], {0})
        assert prog["done"] == 3 and prog["states"] == {"COMPLETE": 2, "PRUNED": 1, "FAIL": 1}, prog
        assert prog["best"] == {"trial": 3, "score": -0.1, "params": {"a": 3}}, prog   # trial 0 is from before
    print("DeepLearningTuning self-check OK")


if __name__ == "__main__":
    _self_check()
