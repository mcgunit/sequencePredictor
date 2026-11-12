"""
What every Hyperopt*.py tuner shares for the Hyperopts page and the variety rule (owner, 11-12 Nov 2026).

The result file (--result-file). A run started from the Hyperopts page writes nothing that is served: it runs on a
worker copy with -s false, and the page shows this JSON file instead - per study its status, the trials done, what is
being scored right now, and the gate's verdict with what it would write. Apply on the page writes that on the main
server (hyperopts.js). Replaced atomically under a lock, so trial processes that write at the same time (the
boosting tuner's parallel trials) never lose each other's fields and a copy taken mid-write is never half a file.

    {"studies": {<study key>: {"tuner", "game", "row", "status", "wanted", "days", "progress", "current",
                               "decision", "reason", "summary",
                               "params"           top-level bestParams keys the gate's decision sets (derived keys
                                                  included: Markov's sort flags, a boosting row's use flag, ...)
                               "remove"           top-level keys the decision deletes
                               "modelScores"      {row: value | None} vote weights it sets (None: removes the row's)
                               "rows"             {row: tuningGate record} - one or more rows per study
                               "record"           the main row's record (the deep-learning tuner's single row)
                               "challengerParams" what "Apply this run's best" writes, when the tuner offers it
                              }},
     "updatedAt": ...}

The variety rule. A usable parameter set predicts new sequences, recurring numbers allowed (owner, 11 Nov 2026). The
owner sets the line (12 Nov 2026): config/tuning.json {"varietyLine": 0.95} for the weekly chain, and a page run's own
value (--variety-line). Two measures:
  windows  the deep-learning rows: the share of a day's validation windows given the most common ticket
           (src/LearningCheck sameTicket); input-blind when at or over the line on at least half of the days
           (src/DeepLearningTuning.variety)
  draws    every other row that predicts a ticket: the share of the scored backtest draws that got the candidate's
           most common ticket; input-blind (stuck on one ticket, whatever the draws before) when at or over the
           line (variety_of_tickets). The record's flag is "input_blind" for both, which is what
           TuningGate.challenge(variety=...) reads.

Import-free beyond the standard library: the deep-learning tuner's parent process and the server's self-checks
import it without TensorFlow.

    python3 -m src.TuningResult     # self-check
"""

import json
import math
import os
from collections import Counter
from datetime import datetime, timezone

try:
    import fcntl
except ImportError:  # not on Linux: no lock, still atomic
    fcntl = None

DEFAULT_VARIETY_LINE = 0.95
VARIETY_LINE_MIN = 0.5
VARIETY_LINE_MAX = 1.0
SETTINGS_FILE = os.path.join("config", "tuning.json")
# keys the result's "params" never carries: the gate's records and the vote weights have their own fields
_OWN_FIELDS = ("tuningGate", "modelScores")


# --- the variety line ---------------------------------------------------------------------------------------------

def clamp_line(value, fallback=DEFAULT_VARIETY_LINE):
    """A line as a share in [0.5, 1.0]; a percentage (95) is read as 0.95; anything else is the fallback."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    if v > 1.0:
        v = v / 100.0
    return v if VARIETY_LINE_MIN <= v <= VARIETY_LINE_MAX else fallback


def load_variety_line(root="."):
    """The owner's line for the weekly chain (config/tuning.json "varietyLine"), DEFAULT_VARIETY_LINE without one."""
    try:
        with open(os.path.join(root, SETTINGS_FILE)) as f:
            data = json.load(f)
        return clamp_line((data or {}).get("varietyLine"))
    except (OSError, ValueError, AttributeError):
        return DEFAULT_VARIETY_LINE


def line_text(line):
    return f"{round(float(line) * 100, 1):g}%"


def variety_rule(line, measure="draws"):
    """The rule as the gate's record states it."""
    if measure == "windows":
        what = (f"gives the same ticket on {line_text(line)} or more of its validation windows on at least half of "
                f"its draws (input-blind)")
    else:
        what = f"gives the same ticket on {line_text(line)} or more of its scored backtest draws (input-blind)"
    return f"a candidate that {what} is neither written nor served back, and blocks nothing"


def ticket_key(ticket):
    """One ticket as a comparable string (a list, a tuple, a dict of lists - JSON with sorted keys)."""
    if ticket is None or ticket == [] or ticket == "":
        return None
    try:
        return json.dumps(ticket, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(ticket)


def variety_of_tickets(tickets, line=DEFAULT_VARIETY_LINE):
    """
    The draws measure over a candidate's scored backtest draws, one ticket per draw (None: the draw had none):
      variety_measure   "draws"
      variety_line      the line used
      same_ticket_mean  the share of the draws that got the most common ticket (1.0: always the same ticket)
      distinct_tickets  how many different tickets, of `tickets` draws with one
      input_blind       same_ticket_mean at or over the line (None: fewer than two draws - nothing to judge)
    """
    keys = [k for k in (ticket_key(t) for t in tickets or []) if k is not None]
    out = {"variety_measure": "draws", "variety_line": float(line), "same_ticket_mean": None,
           "distinct_tickets": len(set(keys)), "tickets": len(keys), "input_blind": None}
    if keys:
        top = Counter(keys).most_common(1)[0][1]
        out["same_ticket_mean"] = round(top / len(keys), 3)
        if len(keys) >= 2:
            out["input_blind"] = bool(top / len(keys) >= float(line) - 1e-12)
    return out


def is_input_blind(tuning):
    """The gate's test (TuningGate.challenge(variety=...)): True only when the record says so."""
    return bool(isinstance(tuning, dict) and tuning.get("input_blind") is True)


# --- the result file ----------------------------------------------------------------------------------------------

def plain(v):
    """Plain JSON values: numpy scalars as numbers, a non-finite float as None (the file stays strict JSON)."""
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
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
    if hasattr(v, "dtype") and getattr(v.dtype, "kind", "") == "b":
        return bool(v)
    return f if math.isfinite(f) else None


def write_result(path, study_key, **fields):
    """Merges `fields` into the study's entry of the result file (no path: nothing written). Returns the data."""
    if not path:
        return None
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    lock = None
    try:
        if fcntl is not None:
            lock = open(path + ".lock", "a")
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        studies = data.setdefault("studies", {})
        entry = studies.setdefault(study_key, {})
        entry.update(plain(fields))
        data["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1, allow_nan=False)
        os.replace(tmp, path)
        return data
    finally:
        if lock is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)
            finally:
                lock.close()


def safe_write_result(path, study_key, **fields):
    """write_result for the page's view only: a failed write is printed, never raised (it must not stop a study)."""
    if not path:
        return None
    try:
        return write_result(path, study_key, **fields)
    except Exception as e:  # noqa: BLE001 - the view, not the work
        print(f"{study_key}: could not write the result file ({e})")
        return None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current(role, day, days, trial=None, **extra):
    """The result's "current": what is being scored now - a trial (its number) or a gate reference - and its draw."""
    out = {"role": role, "trial": trial if role == "trial" else None, "day": day, "days": days, "at": now_iso()}
    out.update(extra)
    return out


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


def writes_between(before, after, rows=None):
    """
    What a study's decision changed in bestParams_<game>.json, between the copy taken before the study and the
    contents after its decision was applied - the fields Apply writes on the main server:
      params       top-level keys set (new or changed), tuningGate and modelScores excepted
      remove       top-level keys deleted
      modelScores  {row: value | None} vote weights set or removed
      rows         {row: record} the tuningGate records written or changed (`rows` limits them to these names)
    """
    before = before or {}
    after = after or {}
    params = {k: v for k, v in after.items() if k not in _OWN_FIELDS and (k not in before or before[k] != v)}
    remove = [k for k in before if k not in _OWN_FIELDS and k not in after]
    sb, sa = before.get("modelScores") or {}, after.get("modelScores") or {}
    scores = {k: sa.get(k) for k in set(sb) | set(sa) if sb.get(k) != sa.get(k)}
    gb, ga = before.get("tuningGate") or {}, after.get("tuningGate") or {}
    names = rows if rows is not None else [k for k in ga if gb.get(k) != ga.get(k)]
    return {"params": plain(params), "remove": sorted(remove), "modelScores": plain(scores),
            "rows": {k: plain(ga[k]) for k in names if k in ga}}


def _self_check():
    import tempfile
    assert clamp_line(95) == 0.95 and clamp_line("0.8") == 0.8 and clamp_line(0.2) == DEFAULT_VARIETY_LINE
    assert clamp_line(None) == DEFAULT_VARIETY_LINE and clamp_line(float("nan")) == DEFAULT_VARIETY_LINE
    assert clamp_line(1) == 1.0 and clamp_line(100) == 1.0
    v = variety_of_tickets([[1, 2, 3]] * 19 + [[4, 5, 6]], 0.95)
    assert v["same_ticket_mean"] == 0.95 and v["input_blind"] is True and v["distinct_tickets"] == 2 and v["tickets"] == 20
    v = variety_of_tickets([[1, 2, 3]] * 18 + [[4, 5, 6], [7, 8, 9]], 0.95)
    assert v["same_ticket_mean"] == 0.9 and v["input_blind"] is False
    assert variety_of_tickets([[1, 2, 3]] * 18 + [[4, 5, 6], [7, 8, 9]], 0.9)["input_blind"] is True
    assert variety_of_tickets([[1]], 0.5)["input_blind"] is None and variety_of_tickets([], 0.5)["same_ticket_mean"] is None
    assert variety_of_tickets([None, [], [1, 2]], 0.5)["tickets"] == 1
    assert "95%" in variety_rule(0.95) and "validation windows" in variety_rule(0.8, "windows") and "80%" in variety_rule(0.8, "windows")
    assert is_input_blind({"input_blind": True}) and not is_input_blind({"input_blind": None}) and not is_input_blind(None)

    with tempfile.TemporaryDirectory() as d:
        assert load_variety_line(d) == DEFAULT_VARIETY_LINE
        os.makedirs(os.path.join(d, "config"))
        with open(os.path.join(d, SETTINGS_FILE), "w") as f:
            json.dump({"varietyLine": 0.8}, f)
        assert load_variety_line(d) == 0.8
        p = os.path.join(d, "r", "x.json")
        write_result(p, "s", status="running", current=current("trial", 2, 10, trial=4))
        write_result(p, "s", progress={"done": 1}, value=float("inf"))
        data = json.load(open(p))
        assert data["studies"]["s"]["status"] == "running" and data["studies"]["s"]["progress"] == {"done": 1}
        assert data["studies"]["s"]["value"] is None and data["studies"]["s"]["current"]["trial"] == 4
        assert safe_write_result(None, "s", a=1) is None

    before = {"a": 1, "b": 2, "gone": 0, "modelScores": {"X": 1.0, "Y": 2.0}, "tuningGate": {"X": {"d": 1}, "Y": {"d": 1}}}
    after = {"a": 1, "b": 3, "new": True, "modelScores": {"Y": 2.0, "Z": 0.5}, "tuningGate": {"X": {"d": 2}, "Y": {"d": 1}}}
    w = writes_between(before, after)
    assert w["params"] == {"b": 3, "new": True} and w["remove"] == ["gone"]
    assert w["modelScores"] == {"X": None, "Z": 0.5} and w["rows"] == {"X": {"d": 2}}
    assert writes_between(before, after, rows=["Y"])["rows"] == {"Y": {"d": 1}}
    print("TuningResult self-check OK")


if __name__ == "__main__":
    _self_check()
