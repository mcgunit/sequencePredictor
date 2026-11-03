#!/usr/bin/env python3
"""
The one-time trim of the lotto tickets of 8 Nov 2025 - 13 May 2026 to the 6
numbers a Lotto grid plays - owner decision of 3 Nov 2026. Until 13 May 2026
the statistical rows stored 7 values: the 7th was a guess at the bonus ball
(one Markov row even guessed 46 in a 1-45 game), which a played grid has no
slot for. Those rows lose their 7th value; the 6-number rows since and the
draws' own realResult (6 mains + the bonus) are not touched.

    python3 TrimLottoBonusGuess.py                                # dry run: what would change, nothing touched
    python3 TrimLottoBonusGuess.py --apply --backup /root/backups/lotto-7th-2026-11-03
    python3 TrimLottoBonusGuess.py --root /elsewhere/checkout ...    # another checkout (a scratch copy)

Run it in the production checkout while no pipeline job runs. --apply:

  1. refuses while process.lock names a live process, then takes it itself
  2. backup into --backup (a new folder outside the checkout): files.tar.gz
     with every day file it changes, read back and counted before any write
  3. per day file in data/database/lotto: every row of exactly 7 values in
     currentPrediction and newPrediction keeps its first 6; matchingNumbers
     is scored again on the trimmed currentPrediction (the same call as
     Predictor.update_matching_numbers). The file's modification time is
     kept, as update_matching_numbers does.

The next run's report reads the trimmed tickets (average hits on 6 numbers,
profit on the prize tables); the pipeline commit (`git add -A`) records the
change. A second run finds nothing to trim. Exit code: 0 done (or a dry
run), 1 refused or failed before any write.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone

GAME = "lotto"
GRID = 6
TRIMMED = GRID + 1   # the rows with the bonus guess; longer rows were never stored and are left alone
FIELDS = ("currentPrediction", "newPrediction")


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def day_files(root):
    """The lotto day files (names start with the draw date, sometimes unpadded: 2026-5-13.json), oldest first."""
    folder = os.path.join(root, "data", "database", GAME)
    names = [n for n in os.listdir(folder) if n.endswith(".json") and n[:1].isdigit()]
    return [os.path.join(folder, n) for n in sorted(names, key=lambda n: [int(x) for x in re.findall(r"\d+", n)])]


def trimmed(day):
    """The day file with its 7-value rows cut to 6, and how many rows were cut per field (no copy when none)."""
    counts = {}
    for field in FIELDS:
        for entry in day.get(field) or []:
            rows = entry.get("predictions") if isinstance(entry, dict) else None
            for i, row in enumerate(rows or []):
                if isinstance(row, list) and len(row) == TRIMMED:
                    rows[i] = row[:GRID]
                    counts[field] = counts.get(field, 0) + 1
    return counts


def plan_of(root):
    """Per day file to change: its relative path and the rows it would lose a value in, per field."""
    plan = []
    for path in day_files(root):
        with open(path, encoding="utf-8") as f:
            day = json.load(f)
        counts = trimmed(day)
        if counts:
            plan.append({"file": os.path.relpath(path, root), "rows": counts})
    return plan


# --- the steps -------------------------------------------------------------------------------------------------------

def lock_holder(lock):
    """The live PID in process.lock, or None (no lock, or a stale one - left in place for its owner's next start)."""
    try:
        with open(lock) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return None
    except PermissionError:
        return pid


def make_backup(root, backup, plan):
    os.makedirs(backup)
    archive = os.path.join(backup, "files.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for item in plan:
            tar.add(os.path.join(root, item["file"]), arcname=item["file"])
    with tarfile.open(archive, "r:gz") as tar:
        stored = sum(1 for m in tar.getmembers() if m.isfile())
    if stored != len(plan):
        raise RuntimeError(f"the backup holds {stored} file(s), {len(plan)} expected")
    manifest = {"made": now_iso(), "root": root, "files": [item["file"] for item in plan]}
    with open(os.path.join(backup, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(backup, "RESTORE.txt"), "w", encoding="utf-8") as f:
        f.write(f"Restore the lotto day files as they were before the trim (no pipeline job running):\n"
                f"  tar -xzpf {archive} -C {root}\n")
    return stored


def rescore(day, helpers):
    """matchingNumbers on the trimmed ticket - lotto's split of Predictor.matchingSplitArgs: 6 mains, bonus as special."""
    if day.get("realResult") and day.get("currentPrediction"):
        day["matchingNumbers"] = helpers.find_best_matching_prediction(
            day["realResult"], day["currentPrediction"], specialColumnCount=0, realMainCount=GRID, game=GAME)


def apply_trim(root, plan, helpers):
    for item in plan:
        path = os.path.join(root, item["file"])
        with open(path, encoding="utf-8") as f:
            day = json.load(f)
        if trimmed(day).get("currentPrediction"):
            rescore(day, helpers)
        stamp = os.stat(path)
        with open(path, "w") as f:
            json.dump(day, f, indent=2)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))


def report(plan, log=print):
    rows = {field: sum(item["rows"].get(field, 0) for item in plan) for field in FIELDS}
    log(f"lotto day files with 7-value rows: {len(plan)}"
        + (f" ({plan[0]['file']} .. {plan[-1]['file']})" if plan else ""))
    log("rows cut to 6: " + ", ".join(f"{field} {n}" for field, n in rows.items()))


def refusal(root, backup):
    if not backup:
        return "--apply needs --backup <new folder outside the checkout>"
    real_root, real_backup = os.path.realpath(root), os.path.realpath(backup)
    if real_backup == real_root or real_backup.startswith(real_root + os.sep):
        return f"the backup folder {backup} is inside the checkout - the pipeline commit would sweep it in"
    if os.path.exists(backup):
        return f"the backup folder {backup} exists - give a new one"
    return None


def run(root, backup=None, apply=False, helpers=None, log=print):
    root = os.path.abspath(root)
    plan = plan_of(root)
    report(plan, log)
    if not plan:
        log("nothing to trim")
        return 0
    why = refusal(root, backup)
    if not apply:
        log("dry run - nothing touched" + (f"; --apply would refuse: {why}" if why else ""))
        return 0
    if why:
        log(f"REFUSED: {why}")
        return 1
    lock = os.path.join(root, "process.lock")
    holder = lock_holder(lock)
    if holder:
        log(f"REFUSED: process.lock is held by PID {holder} (a pipeline job runs) - wait for it to finish")
        return 1
    if os.path.exists(lock):
        os.remove(lock)   # stale: its owner is gone (HyperoptStatistics.is_running does the same)
    try:
        with open(lock, "x") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        log("REFUSED: a job took process.lock just now - run again when it is done")
        return 1
    try:
        if helpers is None:
            from src.Helpers import Helpers
            helpers = Helpers()
        stored = make_backup(root, backup, plan)
        log(f"backup: {stored} file(s) in {backup}/files.tar.gz")
        apply_trim(root, plan, helpers)
        log(f"trimmed {len(plan)} day file(s)")
    finally:
        os.remove(lock)
    log("done - the next run's report scores the trimmed tickets; restart the server (pm2 restart sequencePredictor) "
        "if its pages should show them before that")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="Trim lotto bonus guess", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)), help="The checkout (default: this script's)")
    parser.add_argument("--apply", action="store_true", help="Do it (default: a dry run that lists what would change)")
    parser.add_argument("--backup", default=None, help="A new folder outside the checkout for the backup (required with --apply)")
    args = parser.parse_args()
    return run(args.root, backup=args.backup, apply=args.apply)


# --- self-check ------------------------------------------------------------------------------------------------------

def _self_check():
    """A made-up checkout with an old 7-value ticket and a new 6-value one: dry run, refusals, --apply, a second run."""
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.Helpers import Helpers
    tmp = tempfile.mkdtemp(prefix="trim-lotto-")
    try:
        root = os.path.join(tmp, "checkout")
        folder = os.path.join(root, "data", "database", GAME)
        os.makedirs(folder)
        old = [{"name": "Markov Model", "predictions": [[3, 15, 26, 28, 41, 45, 46]]},
               {"name": "LSTM Base Model", "predictions": [[1, 2, 3, 4, 5, 6, 7]]}]
        new = [{"name": "Markov Model", "predictions": [[3, 15, 26, 28, 41, 45]]}]
        days = {"2026-5-9.json": {"realResult": [1, 2, 3, 4, 5, 6, 7], "currentPrediction": old, "newPrediction": old},
                "2026-5-13.json": {"realResult": [3, 15, 20, 21, 22, 23, 26], "currentPrediction": old, "newPrediction": new,
                                   "matchingNumbers": {"match_count": 3}},
                "2026-11-2.json": {"realResult": [3, 15, 20, 21, 22, 23, 26], "currentPrediction": new, "newPrediction": new}}
        for name, day in days.items():
            with open(os.path.join(folder, name), "w") as f:
                json.dump(day, f, indent=2)
            os.utime(os.path.join(folder, name), (1_700_000_000, 1_700_000_000))
        logs = []
        assert run(root, log=logs.append) == 0 and "dry run" in logs[-1]
        assert [item["file"] for item in plan_of(root)] == ["data/database/lotto/2026-5-9.json", "data/database/lotto/2026-5-13.json"]
        assert plan_of(root)[1]["rows"] == {"currentPrediction": 2}
        assert run(root, apply=True, log=logs.append) == 1 and "--backup" in logs[-1]
        assert run(root, backup=os.path.join(root, "bk"), apply=True, log=logs.append) == 1 and "inside" in logs[-1]
        with open(os.path.join(root, "process.lock"), "w") as f:
            f.write(str(os.getpid()))
        assert run(root, backup=os.path.join(tmp, "bk"), apply=True, log=logs.append) == 1 and "PID" in logs[-1]
        os.remove(os.path.join(root, "process.lock"))
        assert not os.path.exists(os.path.join(tmp, "bk"))
        assert run(root, backup=os.path.join(tmp, "bk"), apply=True, helpers=Helpers(), log=logs.append) == 0
        assert not os.path.exists(os.path.join(root, "process.lock"))
        with open(os.path.join(folder, "2026-5-13.json")) as f:
            day = json.load(f)
        assert [r for e in day["currentPrediction"] for r in e["predictions"]] == [[3, 15, 26, 28, 41, 45], [1, 2, 3, 4, 5, 6]]
        assert day["realResult"] == [3, 15, 20, 21, 22, 23, 26]   # the bonus of the draw stays
        assert day["matchingNumbers"]["match_count"] == 2 and day["matchingNumbers"]["special_match_count"] == 1   # 3, 15 + the bonus 26
        assert day["matchingNumbers"]["prediction"] == [3, 15, 26, 28, 41, 45]
        with open(os.path.join(folder, "2026-5-9.json")) as f:
            assert all(len(r) == GRID for e in json.load(f)["newPrediction"] for r in e["predictions"])
        assert all(os.stat(os.path.join(folder, n)).st_mtime == 1_700_000_000 for n in days)
        with tarfile.open(os.path.join(tmp, "bk", "files.tar.gz")) as tar:
            backed = json.load(tar.extractfile("data/database/lotto/2026-5-13.json"))
        assert len(backed["currentPrediction"][0]["predictions"][0]) == TRIMMED
        assert run(root, log=logs.append) == 0 and logs[-1] == "nothing to trim"
        print("TrimLottoBonusGuess self-check OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
