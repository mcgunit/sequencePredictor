"""
Champion/challenger gate for the Hyperopt* tuners.

Until September 2026 every run wrote study.best_params - the best trial the
study had EVER seen - into bestParams_<game>.json. With the profit objective
of the time that was the trial that had caught the biggest payout, and it
stayed served until a later window happened to beat it: keno LightGBM's 5.0
of 10 Sept 2026 was never approached again (2.13, 1.26 on the next runs), so
its parameters stayed. Two things change here:

  1. The challenger is THIS run's best completed trial, scored on the same
     window as everything it is compared with. Older trials stay in the
     study as sampler history, nothing more.
  2. It replaces the served parameters only if it beats the incumbent (the
     parameters in bestParams_<game>.json today, re-scored on this run's
     window with this run's objective) and the untuned defaults (what
     Predictor.py serves when the keys are missing). A reference that could
     not be scored - timeout, error - cannot block; the tuner then behaves as
     before for that comparison and says so in the log and the record.

The references are scored by the strategy's own Optuna objective driven by an
optuna.trial.FixedTrial, so the incumbent goes through exactly the code that
scored the trials (same model configuration, subsets, timeout and
TuningScore objective), and each reference runs in a forked child
(HyperoptRunner.run_isolated): the coordinator must never run a library fit
itself before it forks the next study's trials.

Every decision is recorded under bestParams_<game>.json["tuningGate"][row]
- challenger, incumbent and default scores, each one's diagnostics (raw
profit per bet, lucky strikes, days), the window - so a reader of the file
can see why a row's parameters did or did not change, and whether what
stayed served was itself a hijack.

    python3 -m src.TuningGate       # self-check with a fake study
"""

import math
import re
from datetime import datetime, timezone

import optuna

try:
    from src.HyperoptRunner import run_isolated
    from src.TuningScore import attrs_for_trial, describe, json_safe
except ImportError:  # imported from inside src/
    from HyperoptRunner import run_isolated
    from TuningScore import attrs_for_trial, describe, json_safe

# The tuners' per-model Keno subset flags ("markov_use_5", "lightgbm_use_10").
KENO_USE_KEY = re.compile(r"^(?P<model>.+)_use_(?P<size>\d+)$")


def make_defaults(table, served):
    """
    What Predictor.py serves for a tuned key that bestParams_<game>.json does
    not have: `table` maps the tuner's parameter names to the code defaults;
    a per-model Keno subset flag "<model>_use_<size>" defaults to the game's
    global "use_<size>" flag (Predictor.getKenoSubsetSizes reads only those),
    True when even that is missing. Returns a callable(name) that raises
    KeyError for a name it knows nothing about.
    """
    def default_for(name):
        if name in table:
            return table[name]
        match = KENO_USE_KEY.match(name)
        if match:
            return bool(served.get(f"use_{match.group('size')}", True))
        raise KeyError(name)
    return default_for


def reference_params(param_names, served, defaults):
    """
    {name: the served value, else its default} for every parameter a trial of
    this strategy suggests. `defaults` is a dict or a callable(name); a
    KeyError names the first parameter that has neither.
    """
    lookup = defaults if callable(defaults) else (lambda name: defaults[name])
    return {name: served[name] if name in served else lookup(name) for name in param_names}


def is_tuned(param_names, served):
    """Whether bestParams_<game>.json carries any of this strategy's keys."""
    return any(name in served for name in param_names)


def _fmt(score):
    if score is None:
        return "unscored"
    return f"{score:+.4f}" if math.isfinite(score) else str(score)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decide(challenger, incumbent, default=None, margin=0.0):
    """
    (replace, reason). A reference given as None could not be scored and
    cannot block; with both unscored a finitely scored challenger is written,
    as before the gate existed. A challenger without a finite score (-inf: an
    objective that declined the trial, or a backtest that produced nothing)
    is never written. `margin` is how much better the challenger must be.
    """
    if challenger is None:
        return False, "no scored challenger"
    if not _finite(challenger):
        return False, f"challenger {_fmt(challenger)} has no finite score"
    references = [(name, score) for name, score in (("incumbent", incumbent), ("default", default))
                  if score is not None]
    blockers = [f"{name} {_fmt(score)}" for name, score in references if not challenger > score + margin]
    if blockers:
        return False, (f"challenger {_fmt(challenger)} does not beat " + " or ".join(blockers)
                       + (f" by {margin:g}" if margin else ""))
    if not references:
        return True, f"challenger {_fmt(challenger)} written - no reference could be scored"
    return True, f"challenger {_fmt(challenger)} beats " + " and ".join(f"{n} {_fmt(s)}" for n, s in references)


def run_trials(study, known_numbers):
    """
    The finitely scored, completed trials this run added: everything not in
    known_numbers. A -inf trial (an objective that declined it - a Keno mask
    selecting no subset - or a backtest that produced no result) is not a
    candidate: it never measured anything, so it can never be written.
    """
    return [t for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE and t.number not in known_numbers
            and _finite(t.value)]


def evaluate_reference(objective, params, timeout_seconds, label):
    """
    Scores `params` with the strategy's own objective, driven by a FixedTrial,
    in a forked child. Returns (score, diagnostics): the float and the
    "tuning" attribute the objective set on the trial (TuningScore.attrs_for_
    trial), or (None, None) when it could not be scored (timeout, exception,
    a pruned/skipped evaluation, a non-numeric result).
    """
    def run():
        import warnings
        with warnings.catch_warnings():
            # A served value outside the search range (an older default, a
            # hand-edited file) makes FixedTrial warn; it is still used as
            # is, which is exactly the point of scoring what is served.
            warnings.simplefilter("ignore")
            trial = optuna.trial.FixedTrial(dict(params))
            value = objective(trial)
            return value, trial.user_attrs.get("tuning")

    status, payload = run_isolated(run, timeout_seconds, label)
    value, tuning = payload if status == "ok" and isinstance(payload, tuple) and len(payload) == 2 else (None, None)
    if status == "ok" and isinstance(value, (int, float)) and not isinstance(value, bool) \
            and not (isinstance(value, float) and math.isnan(value)):
        return float(value), tuning
    detail = f": {payload}" if payload and status != "ok" else ""
    print(f"  {label}: could not be scored ({status}{detail})")
    return None, None


def _scored(result):
    """(score, diagnostics) from an evaluate() result that may be a bare float (tests, older callers)."""
    if isinstance(result, tuple):
        return result[0], (result[1] if len(result) > 1 else None)
    return result, None


def challenge(study, known_numbers, objective, served, defaults, timeout_seconds, label,
              margin=0.0, enabled=True, window_days=None, evaluate=None):
    """
    Runs the gate for one strategy after its study was optimised.

      study          the reopened Optuna study
      known_numbers  trial numbers that existed before this run's optimize
      objective      the callable the trials ran (trial -> score)
      served         the game's bestParams_<game>.json contents
      defaults       dict or callable(name) - see make_defaults
      evaluate       (params, role) -> (float|None, diagnostics|None), or a
                     bare float; tests inject one, the tuners use
                     evaluate_reference through run_isolated

    Returns a dict: replace (bool), reason, params (the dict to write, or
    None), challenger / incumbent / default scores (None = unscored),
    served_score (the score of what is served after this decision, for
    modelScores; None when unknown), trial (the challenger's number), record
    (for bestParams["tuningGate"]) and summary (log lines).
    """
    if evaluate is None:
        def evaluate(params, role):
            return evaluate_reference(objective, params, timeout_seconds, f"{label} {role}")

    record = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "window_days": window_days}
    try:
        record["study_best"] = json_safe(float(study.best_value))
    except Exception:  # no completed trial in the whole study
        record["study_best"] = None

    trials = run_trials(study, known_numbers)
    if not trials:
        record.update({"decision": "kept", "reason": "no finitely scored trial this run", "trial": None,
                       "challenger": None, "incumbent": None, "default": None,
                       "challenger_tuning": None, "incumbent_tuning": None, "default_tuning": None})
        return {"replace": False, "reason": record["reason"], "params": None, "challenger": None,
                "incumbent": None, "default": None, "served_score": None, "trial": None,
                "record": record, "summary": f"  {label}: no finitely scored trial this run - parameters kept"}

    best = max(trials, key=lambda t: t.value)
    challenger = float(best.value)
    param_names = list(best.params)
    tuning = best.user_attrs.get("tuning")
    notes = []

    incumbent_tuning = default_tuning = None
    if not enabled:
        incumbent = default = None
        replace, reason = True, "gate disabled - this run's best trial is written"
    else:
        try:
            incumbent_params = reference_params(param_names, served, defaults)
        except KeyError as e:
            incumbent_params = None
            notes.append(f"incumbent not scorable: no served value or default for {e}")
        try:
            default_params = reference_params(param_names, {}, defaults)
        except KeyError as e:
            default_params = None
            notes.append(f"default not scorable: no default for {e}")

        incumbent, incumbent_tuning = _scored(evaluate(incumbent_params, "incumbent")) \
            if incumbent_params is not None else (None, None)
        if not is_tuned(param_names, served):
            # nothing of this strategy was ever written: the served parameters
            # ARE the defaults, one reference covers both
            default, default_tuning = incumbent, incumbent_tuning
            notes.append("untuned so far - the incumbent is the default")
        elif default_params is None:
            default = None
        elif incumbent_params is not None and default_params == incumbent_params:
            default, default_tuning = incumbent, incumbent_tuning
        else:
            default, default_tuning = _scored(evaluate(default_params, "default"))
        replace, reason = decide(challenger, incumbent, default, margin)

    served_score = challenger if replace else incumbent
    if served_score is not None and not math.isfinite(served_score):
        served_score = None  # a -inf weight would reach modelScores and the JSON file
    record.update({
        "decision": "replaced" if replace else "kept",
        "reason": reason + (" (" + "; ".join(notes) + ")" if notes else ""),
        "trial": best.number,
        "challenger": json_safe(challenger),
        "incumbent": json_safe(incumbent),
        "default": json_safe(default),
        # what each score was made of - raw profit per bet, lucky strikes, days
        "challenger_tuning": attrs_for_trial(tuning),
        "incumbent_tuning": attrs_for_trial(incumbent_tuning),
        "default_tuning": attrs_for_trial(default_tuning),
    })
    lines = [f"  {label}: this run's best is trial {best.number} at {_fmt(challenger)}"
             f" (study's all-time best {_fmt(record['study_best'])}) | {describe(tuning)}",
             f"  {label}: incumbent {_fmt(incumbent)} ({describe(incumbent_tuning)}), default {_fmt(default)} -> "
             f"{'REPLACED' if replace else 'KEPT'}: {record['reason']}"]
    return {"replace": replace, "reason": record["reason"], "params": dict(best.params) if replace else None,
            "challenger": challenger, "incumbent": incumbent, "default": default,
            "served_score": served_score, "trial": best.number, "record": record, "summary": "\n".join(lines)}


if __name__ == "__main__":
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    class FakeTrial:
        def __init__(self, number, value, params, state=optuna.trial.TrialState.COMPLETE, attrs=None):
            self.number, self.value, self.params, self.state = number, value, params, state
            self.user_attrs = attrs or {}

    class FakeStudy:
        def __init__(self, trials):
            self._trials = trials

        def get_trials(self, deepcopy=False, states=None):
            return list(self._trials)

        @property
        def best_value(self):
            done = [t.value for t in self._trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not done:
                raise ValueError("no trials")
            return max(done)

    P = optuna.trial.TrialState.PRUNED
    old = [FakeTrial(0, 5.0, {"xEstimators": 10, "xDepth": 2, "m_use_5": True, "m_use_6": True}),  # the lucky one, kept as history
           FakeTrial(1, -0.5, {"xEstimators": 100, "xDepth": 3, "m_use_5": True, "m_use_6": False})]
    new = [FakeTrial(2, -0.7, {"xEstimators": 200, "xDepth": 4, "m_use_5": True, "m_use_6": True}),
           FakeTrial(3, -0.4, {"xEstimators": 150, "xDepth": 5, "m_use_5": False, "m_use_6": True},
                     attrs={"tuning": {"score": -0.4, "kind": "capped_profit", "days": 90, "bets": 90,
                                       "profit_per_bet": -0.6, "lucky_strikes": 0, "hits_mean": None}}),
           FakeTrial(4, 9.0, {"xEstimators": 1, "xDepth": 1, "m_use_5": True, "m_use_6": True}, state=P)]
    study = FakeStudy(old + new)
    known = {0, 1}
    table = {"xEstimators": 100, "xDepth": 3}
    served = {"xEstimators": 10, "xDepth": 2, "m_use_5": True, "m_use_6": True, "use_5": True, "use_6": False}

    calls = []

    def scorer(scores, diagnostics=None):
        def evaluate(params, role):
            calls.append((role, dict(params)))
            if diagnostics is not None:
                return scores.get(role), diagnostics.get(role)
            return scores.get(role)
        return evaluate

    # 1. the challenger is this run's best COMPLETED trial, not the study's all-time best
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t",
                    evaluate=scorer({"incumbent": -1.0, "default": -0.9}))
    check(out["trial"] == 3 and out["challenger"] == -0.4, f"challenger should be trial 3: {out['trial']}")
    check(out["replace"] and out["params"] == new[1].params, "beats both -> replaced with trial 3's params")
    check(out["served_score"] == -0.4 and out["record"]["decision"] == "replaced", "served score is the challenger's")
    check(out["record"]["study_best"] == 5.0 and out["record"]["challenger_tuning"]["days"] == 90, "record carries context")
    # incumbent = served values, default = table + global keno flags
    roles = dict(calls)
    check(roles["incumbent"] == {"xEstimators": 10, "xDepth": 2, "m_use_5": True, "m_use_6": True}, roles["incumbent"])
    check(roles["default"] == {"xEstimators": 100, "xDepth": 3, "m_use_5": True, "m_use_6": False}, roles["default"])

    # 2. a better incumbent blocks - and its diagnostics are recorded next to the challenger's
    inc_diag = {"score": -0.3, "kind": "capped_profit", "days": 90, "bets": 180, "profit_per_bet": 4.1, "lucky_strikes": 1}
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t",
                    evaluate=scorer({"incumbent": -0.3, "default": -0.9}, {"incumbent": inc_diag, "default": None}))
    check(not out["replace"] and out["params"] is None and out["served_score"] == -0.3, "incumbent blocks")
    check("does not beat incumbent" in out["reason"], out["reason"])
    check(out["record"]["incumbent_tuning"] == inc_diag and out["record"]["challenger_tuning"]["days"] == 90
          and out["record"]["default_tuning"] is None, "each side's diagnostics are recorded")
    check("1 lucky strike" in out["summary"], out["summary"])

    # 3. a better default blocks even when the incumbent is worse
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t",
                    evaluate=scorer({"incumbent": -1.0, "default": -0.2}))
    check(not out["replace"] and "default" in out["reason"], "default blocks")

    # 4. an unscorable reference cannot block
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t",
                    evaluate=scorer({"incumbent": None, "default": -0.9}))
    check(out["replace"] and out["incumbent"] is None, "unscored incumbent cannot block")
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t", evaluate=scorer({}))
    check(out["replace"] and "no reference could be scored" in out["reason"], out["reason"])

    # 5. untuned strategy: incumbent == default, scored once
    calls.clear()
    out = challenge(study, known, None, {"use_5": True}, make_defaults(table, {"use_5": True}), 10, "t",
                    evaluate=scorer({"incumbent": -0.5}))
    check([r for r, _ in calls] == ["incumbent"], f"one reference for an untuned strategy: {calls}")
    check(out["default"] == -0.5 and out["replace"] and "untuned" in out["reason"], out["reason"])
    check(calls[0][1] == {"xEstimators": 100, "xDepth": 3, "m_use_5": True, "m_use_6": True}, calls[0][1])

    # 6. margin
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t", margin=0.5,
                    evaluate=scorer({"incumbent": -0.7, "default": -0.9}))
    check(not out["replace"] and "by 0.5" in out["reason"], out["reason"])

    # 7. nothing completed this run
    out = challenge(FakeStudy(old + [new[2]]), known, None, served, make_defaults(table, served), 10, "t",
                    evaluate=scorer({}))
    check(not out["replace"] and out["params"] is None and out["trial"] is None
          and out["record"]["decision"] == "kept", "no new trial -> kept")

    # 8. gate disabled
    calls.clear()
    out = challenge(study, known, None, served, make_defaults(table, served), 10, "t", enabled=False,
                    evaluate=scorer({}))
    check(out["replace"] and not calls and out["incumbent"] is None, "disabled: write this run's best, score nothing")

    # 8b. a run whose only completed trial is -inf (a Keno mask that selected no
    # subset, an all-error backtest) writes nothing - gate on or off, references or not
    inf_only = FakeStudy(old + [FakeTrial(5, float("-inf"), {"xEstimators": 7, "xDepth": 9, "m_use_5": False, "m_use_6": False})])
    for enabled in (False, True):
        out = challenge(inf_only, known, None, served, make_defaults(table, served), 10, "t", enabled=enabled,
                        evaluate=scorer({}))
        check(not out["replace"] and out["params"] is None and out["trial"] is None
              and "no finitely scored trial" in out["reason"], f"-inf-only run, enabled={enabled}: {out['reason']}")

    # 9. a parameter with neither served value nor default
    out = challenge(study, known, None, served, make_defaults({"xEstimators": 100}, served), 10, "t",
                    evaluate=scorer({"incumbent": -1.0}))
    check(out["incumbent"] == -1.0 and out["default"] is None and "xDepth" in out["reason"], out["reason"])

    # decide() on its own
    check(decide(1.0, 0.5) == (True, "challenger +1.0000 beats incumbent +0.5000"), decide(1.0, 0.5))
    check(decide(0.5, 0.5)[0] is False, "equal does not replace")
    check(decide(None, 0.5) == (False, "no scored challenger"), "no challenger")
    check(decide(1.0, None, None)[0] is True, "nothing scorable -> replace")
    check(decide(float("-inf"), -3.0)[0] is False, "a -inf challenger never replaces a scored incumbent")
    check(decide(float("-inf"), None, None)[0] is False, "a -inf challenger is never written, even with nothing scorable")
    check(decide(float("nan"), None, None)[0] is False, "a NaN challenger is never written")

    # the decision record and summary survive JSON, also with non-finite scores
    import json
    json.dumps(out["record"])
    try:
        from src.TuningScore import score_rows as _score_rows
    except ImportError:
        from TuningScore import score_rows as _score_rows
    stored = attrs_for_trial(_score_rows([{"index": 0, "m_error": "boom"}], "m", payout=True))
    out = challenge(FakeStudy([FakeTrial(9, -2.0, {"xEstimators": 5, "xDepth": 1}, attrs={"tuning": stored})]), set(), None,
                    {}, make_defaults(table, {}), 10, "t", evaluate=scorer({"incumbent": float("-inf")}))
    check(out["replace"] and out["served_score"] == -2.0 and out["record"]["incumbent"] is None
          and "-Infinity" not in json.dumps(out["record"]), "non-finite scores never reach the JSON file")
    check("score -inf" in out["summary"], "a stored all-error diagnostics block describes instead of raising")
    check("KEPT" in out["summary"] or "REPLACED" in out["summary"], out["summary"])

    for message in failures:
        print("FAIL:", message)
    print(f"TuningGate self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    raise SystemExit(1 if failures else 0)
