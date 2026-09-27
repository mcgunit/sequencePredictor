"""
The lockbox period (README "Lockbox period", roadmap item 6 / Q0).

A chronological period of draws that model selection never sees: the
meta-learner trainer and the quantum tuner drop its days from their tables
before splitting and fitting, and the frozen artifacts are scored on it once,
on request (TrainMetaLearner.py --lockbox-report). Repeatedly inspecting the
same holdout and changing the design afterwards turns the holdout into
development data; the lockbox is the period that does not get that treatment.

Declared by hand in lockbox.json at the repository root - no file, no lockbox:

    {"from": "2026-07-01", "to": "2026-09-26", "declared": "2026-09-27",
     "note": "designs frozen after the September 2026 tuning changes"}

Dates are inclusive. The period should end at or before the declaration: a
lockbox that extends into the future would lock out the newest draws from
every weekly retrain, and the daily predictor is never subject to it - it
predicts; the lockbox is about what the trainers learn from.

The row tuners (HyperoptStatistics.py, HyperoptBoost.py) score trials on the
newest draws by design, so a lockbox that overlaps their window is reported
as a warning rather than enforced: tuning on it would make the lockbox
evaluation of the meta-learners rest on rows tuned inside it. Keep the lockbox
older than the tuning window, or accept the caveat knowingly.

    python3 -m src.Lockbox         # self-check
"""

import json
import os
from datetime import date, datetime

FILE_NAME = "lockbox.json"


def _to_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    return datetime.strptime(text, "%Y-%m-%d").date()


def load(root=None):
    """The declared lockbox as {"from": date, "to": date, ...} or None."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, FILE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{FILE_NAME}: not valid JSON ({exc})") from exc
    return parse(raw)


def parse(raw):
    """Validate a declaration; raises ValueError on a malformed one."""
    if not raw:
        return None
    try:
        start, end = _to_date(raw["from"]), _to_date(raw["to"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"{FILE_NAME}: 'from' and 'to' must be YYYY-MM-DD dates ({exc})") from exc
    if end < start:
        raise ValueError(f"{FILE_NAME}: 'to' ({end}) lies before 'from' ({start})")
    return {"from": start, "to": end, "declared": raw.get("declared"), "note": raw.get("note")}


def covers(lockbox, when):
    """Whether a draw date falls inside the lockbox (inclusive)."""
    if not lockbox or when is None:
        return False
    when = _to_date(when)
    return lockbox["from"] <= when <= lockbox["to"]


def split_rows(rows, dates, lockbox):
    """
    Backtester rows carry "index", their position in the loader's numbers;
    `dates` are the loader's dates in the same order. Returns (kept, locked):
    rows outside and inside the lockbox, both in their original order. Rows
    whose index has no date are kept - nothing is silently dropped.
    """
    if not lockbox:
        return list(rows), []
    kept, locked = [], []
    for row in rows:
        index = row.get("index")
        when = dates[index] if isinstance(index, int) and 0 <= index < len(dates) else None
        (locked if covers(lockbox, when) else kept).append(row)
    return kept, locked


def window_overlap(dates, window_days, lockbox):
    """How many of the newest `window_days` dates fall inside the lockbox."""
    if not lockbox or not dates or window_days <= 0:
        return 0
    return sum(1 for when in list(dates)[-int(window_days):] if covers(lockbox, when))


def describe(lockbox):
    if not lockbox:
        return "no lockbox declared"
    text = f"lockbox {lockbox['from']} to {lockbox['to']}"
    if lockbox.get("declared"):
        text += f" (declared {lockbox['declared']})"
    return text


def as_json(lockbox):
    if not lockbox:
        return None
    return {"from": lockbox["from"].isoformat(), "to": lockbox["to"].isoformat(),
            "declared": lockbox.get("declared"), "note": lockbox.get("note")}


if __name__ == "__main__":
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    box = parse({"from": "2026-07-01", "to": "2026-09-26", "declared": "2026-09-27"})
    check(box["from"] == date(2026, 7, 1) and box["to"] == date(2026, 9, 26), "dates parse")
    check(covers(box, "2026-07-01") and covers(box, date(2026, 9, 26)) and covers(box, datetime(2026, 8, 15, 20, 0)),
          "inclusive on both ends, datetime accepted")
    check(not covers(box, "2026-06-30") and not covers(box, "2026-09-27") and not covers(None, "2026-08-01") and not covers(box, None),
          "outside, no lockbox, no date")
    for bad in ({"from": "2026-09-30", "to": "2026-09-01"}, {"from": "soon"}, {"to": "2026-09-01"}):
        try:
            parse(bad)
            failures.append(f"malformed declaration accepted: {bad}")
        except ValueError:
            pass
    check(parse(None) is None and parse({}) is None, "an empty declaration is no lockbox")

    dates = [datetime(2026, 6, 1) .__class__(2026, 6, 1 + i) if i < 29 else datetime(2026, 7, i - 28) for i in range(60)]
    rows = [{"index": i, "v": i} for i in range(60)] + [{"index": 999}, {"v": "no index"}]
    kept, locked = split_rows(rows, dates, box)
    check(len(locked) == 31 and all(covers(box, dates[r["index"]]) for r in locked), f"July rows locked: {len(locked)}")
    check(len(kept) == 31 and kept[-2:] == [{"index": 999}, {"v": "no index"}], "June rows and dateless rows kept, order preserved")
    check(split_rows(rows, dates, None) == (rows, []), "no lockbox: everything kept")
    check(window_overlap(dates, 10, box) == 10 and window_overlap(dates, 40, box) == 31 and window_overlap(dates, 10, None) == 0,
          "overlap counts the newest window's locked dates")
    check(describe(box).startswith("lockbox 2026-07-01 to 2026-09-26") and describe(None) == "no lockbox declared", describe(box))
    json.dumps(as_json(box))
    check(load("/nonexistent/root") is None, "no file, no lockbox")

    for message in failures:
        print("FAIL:", message)
    print(f"Lockbox self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    raise SystemExit(1 if failures else 0)
