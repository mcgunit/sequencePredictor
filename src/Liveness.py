"""
Lottery liveness (27 Oct 2026, B4 of the 26 Oct weekly reading; README "Model performance report"): was the ticket a
row scored on a draw saved before that draw's sales close?

Until 27 Oct 2026 every stored lottery draw counted as live in the History ranking, the users' Top 5, the Q0 card and
the users' draw history. Production's git history says otherwise: of 12,170 scored (draw, row) records, 7,301 were
saved as scored before the close, 2,018 were rewritten after the draw (the ticket that stood at the close is in git)
and 2,851 were back-built or late - gap recoveries committed days after the draw (vikinglotto 27 May and 3 Jun 2026,
first committed 13 Jun), Joker+'s launch back-build (10 Aug - 2 Sep, committed 10 Sep), rows added to July files on
1 Aug 2026. Vikinglotto MarkovBayesian, the one long-record lottery row above its Q0 band (1.021 hits over 48 draws),
reads 0.878 over its 41 live draws.

THE RULE, per scored draw D and row R. The ticket is R in the newPrediction of the day file P before D (by date), as
D.currentPrediction holds it - K, its predictions and multiPick (labels such as learningCheck are not the ticket). The
close is src/drawClose.json's sales close on D's date (close_instant). First match wins:
  (a) live, by the stamp: P.madeAt (Predictor.stampTicket, on every write path since 13 Oct 2026) is before the
      close, and the first commit of P at or after the stamp (by commit time) holds R with K - or P has not been
      committed since the stamp and the file on disk holds it. The coverage check catches a row edited later under an
      old stamp.
  (b) live, by a commit: the earliest commit of P holding R with K is before the close.
  (c) recovered: the version of P written by the LATEST commit before the close (by commit time) holds R with other
      content: the ticket stood at the close and was rewritten after the draw. The live record scores THAT ticket
      (Helpers.generate_model_performance_report), kept here with its commit and time. A commit that deleted P wrote
      no version and is passed over: development deleted whole day folders on its own branch (21 Nov 2025, "going for
      ai training") while production's commits of them stood - a deletion unsaves no ticket.
  (d) late ("back-built or late", the market pages' words): no version of P before the close ("first saved after the
      close"), or the version standing then has no R ("row added after the close"). The hours from the close to the
      first save of K are kept.
  A second row of one name on one draw is not live ("duplicate row": euromillions 16 Jan 2026 has two
  'LSTM Base Model' rows with different tickets); the first is judged as above. The report counts both, so a row's
  draws minus its live draws is always the number of its not-live markers.
A file's modification time is never evidence: re-links rewrote the lottery files (lotto/2026-5-23.json has the mtime
of 23 Aug 2026), so only the stamp and the commits count.

Commit time, not first-seen time: (b) takes the earliest commit holding K, (c) the latest commit before the close, the
coverage check the first commit at or after the stamp. A blob's first-seen time would misread content that comes back
(A, then B, then A again before the close: the version standing is A). A commit on a branch production did not run
(a development commit merged later) counts at its own time: the ticket existed then. Two commits in one second keep
git's order.

Git: one `git log --raw --full-history -m` per game folder - not --first-parent, which would walk master's
fast-forwards through development's merges and mark 2,900 live records late - and the blobs read lazily through one
`git cat-file --batch` per game. Before git is trusted, the repository is checked: the database folder must be the
checkout's own data/database, `git rev-parse --show-toplevel` must be that checkout (a scratch copy nested inside
another checkout would read an empty history there and turn every unstamped draw late), the clone must not be
shallow, and `git ls-files` must list the game's folder. One deadline (DEADLINE_SECONDS) covers the whole scan: each
git call gets what is left of it, and a watchdog kills the cat-file process when it passes; the process is closed
after each game whatever happens.

WITHOUT GIT (a check fails, git fails, the deadline passes): "stamp only". A stamped ticket file is judged by (a)
without the coverage check (the file on disk must hold K); an unstamped one takes its verdict from the previous
data/database/lotteryLiveness.json when that run judged the draw with git (checkedThrough: the newest draw whose
ticket file has no stamp - every file written since 13 Oct 2026 has one, so it stops moving); anything else is
"unchecked", counted as not live, and the game's liveness reads "unchecked". The report never ranks on an unchecked
game's live record, so a one-day git failure never becomes the users' point-in-time ranking.

THE DETAIL FILE, data/database/lotteryLiveness.json (write_detail): per game the draws with a row that is not live
as stored - late, recovered (with the ticket that stood at the close), duplicate - written with sorted keys and no run
time, and only when it changes, so the daily commit carries it only when a verdict changes. The admin draw pages
(workbench.js) and the users' history (lottery.js) read it; a game judged without git keeps its previous entry, with
the draws the stamps alone judged added.

    python3 -m src.Liveness     # self-check on temporary git repositories (no network, no production file)
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

SCHEDULE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawClose.json")
DETAIL_FILE = "lotteryLiveness.json"
DEADLINE_SECONDS = 60
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
RULE = ("live = the ticket was saved before the draw's sales close (src/drawClose.json): stamped (madeAt) before it and "
        "committed as stamped, or committed before it; recovered = rewritten after the draw - the live record scores the "
        "ticket that stood at the close; back-built or late = no version before the close, or the row added after it; "
        "a second row of one name on a draw is not live (duplicate row)")
LATE_FIRST = "first saved after the close"
LATE_ROW = "row added after the close"
DUPLICATE = "duplicate row"


# --- the schedule -----------------------------------------------------------------------------------------------------
def load_schedule(path=SCHEDULE_FILE):
    """src/drawClose.json as {"zone": ZoneInfo, "games": {game: {weekday 0-6 (Mon-Sun): (hour, minute)}}}."""
    with open(path) as handle:
        raw = json.load(handle)
    games = {}
    for game, days in (raw.get("games") or {}).items():
        table = {}
        for weekday, hhmm in days.items():
            hour, minute = (int(x) for x in str(hhmm).split(":"))
            table[WEEKDAYS.index(weekday)] = (hour, minute)
        if table:
            games[game] = table
    return {"zone": ZoneInfo(raw.get("timezone") or "Europe/Brussels"), "games": games}


_SCHEDULE = None


def default_schedule():
    global _SCHEDULE
    if _SCHEDULE is None:
        _SCHEDULE = load_schedule()
    return _SCHEDULE


def close_instant(game, day, schedule=None):
    """
    (the sales close of `game`'s draw on `day` as a UTC datetime, scheduled). The table's Brussels wall time on that
    date through zoneinfo, so the clock change is in it (lotto Sat 24 Oct 2026 19:00 = 17:00Z, Sat 31 Oct = 18:00Z).
    A weekday the table does not list gets the game's earliest close and scheduled False (the report warns: the
    National Lottery may have moved a draw). KeyError for a game the table does not list.
    """
    schedule = schedule or default_schedule()
    table = schedule["games"][game]
    hour_minute = table.get(day.weekday())
    scheduled = hour_minute is not None
    if hour_minute is None:
        hour_minute = min(table.values())
    local = datetime(day.year, day.month, day.day, hour_minute[0], hour_minute[1], tzinfo=schedule["zone"])
    return local.astimezone(timezone.utc), scheduled


# --- small helpers ----------------------------------------------------------------------------------------------------
def day_of(file_name):
    """'2026-9-5.json' -> date(2026, 9, 5), or None for anything that is not a day file."""
    if not str(file_name).endswith(".json"):
        return None
    try:
        year, month, day = (int(x) for x in str(file_name)[:-5].split("-"))
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def ticket_key(row):
    """What a ticket is, for comparing two versions: its predictions and its multi-pick, nothing else."""
    return json.dumps({"p": row.get("predictions"), "m": row.get("multiPick")}, sort_keys=True)


def scored(row):
    """A row the report scores: a dict with a name and a non-empty first ticket."""
    return isinstance(row, dict) and bool(row.get("name")) and bool(row.get("predictions")) and bool(row["predictions"][0])


def first_rows(obj, field):
    """{name: row} of a day file's field, the first row of each name."""
    out = {}
    for row in (obj or {}).get(field) or []:
        if scored(row):
            out.setdefault(row["name"], row)
    return out


def iso_utc(instant):
    """A UTC instant (datetime or epoch seconds) as '2026-10-03T17:00:00Z'."""
    if instant is None:
        return None
    if not isinstance(instant, datetime):
        instant = datetime.fromtimestamp(instant, tz=timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_of(obj):
    """A day file's madeAt as epoch seconds, or None (no stamp, or one that does not parse)."""
    made = (obj or {}).get("madeAt")
    if not made:
        return None
    try:
        stamp = datetime.fromisoformat(str(made).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


class GitUnavailable(Exception):
    """git cannot be the evidence for this game (a failed check, a git error, the deadline): the game is judged
    'stamp only'."""


# --- git --------------------------------------------------------------------------------------------------------------
def _git(root, args, deadline, text=True):
    left = deadline - time.monotonic()
    if left <= 0:
        raise GitUnavailable(f"the git scan ran past its {DEADLINE_SECONDS} s deadline")
    try:
        done = subprocess.run(["git", "-C", root] + args, capture_output=True, text=text, timeout=left)
    except subprocess.TimeoutExpired:
        raise GitUnavailable(f"the git scan ran past its {DEADLINE_SECONDS} s deadline (git {args[0]})")
    except OSError as exc:
        raise GitUnavailable(f"git could not run ({exc.__class__.__name__})")
    if done.returncode != 0:
        message = (done.stderr or "").strip().splitlines()
        raise GitUnavailable(f"git {args[0]} failed ({message[-1] if message else f'exit {done.returncode}'})")
    return done.stdout


class _Folder:
    """
    One game folder's git history: versions() from one `git log --raw`, rows(blob) through one `git cat-file --batch`
    process, killed by a watchdog at the scan's deadline and closed by close() (or the `with` block) after the game.
    """

    def __init__(self, root, folder, deadline):
        self.root, self.folder, self.deadline = root, folder, deadline
        self._cat = None
        self._watchdog = None
        self._cache = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        cat, self._cat = self._cat, None
        if cat is not None:
            try:
                cat.stdin.close()
            except OSError:
                pass
            try:
                cat.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cat.kill()
                cat.wait()
            if cat.stdout:
                cat.stdout.close()

    def versions(self):
        """
        {file name: [(commit time, sha, blob or None for a deletion)]}, oldest first by commit time (git's order on a
        tie). A merge listed once per parent (-m) shows the merge's own blob each time; it is kept once.
        """
        out = _git(self.root, ["log", "--raw", "--no-abbrev", "--no-renames", "--full-history", "-m",
                               "--format=C %H %ct", "--", self.folder + "/"], self.deadline)
        per_file = {}
        seen = set()
        sha = ct = None
        position = 0
        for line in out.splitlines():
            if line.startswith("C "):
                _, sha, stamp = line.split()
                ct = int(stamp)
                position += 1
            elif line.startswith(":") and sha:
                meta, path = line.split("\t", 1)
                fields = meta.split()
                name = os.path.basename(path)
                if (sha, name) in seen:
                    continue
                seen.add((sha, name))
                blob = None if fields[4].startswith("D") or set(fields[3]) == {"0"} else fields[3]
                # git log lists the newest first: a later position is an older commit of the same second
                per_file.setdefault(name, []).append((ct, -position, sha, blob))
        return {name: [(ct, sha, blob) for ct, _, sha, blob in sorted(entries)] for name, entries in per_file.items()}

    def rows(self, blob):
        """{name: (ticket_key, row)} of the newPrediction in a blob, the first row of each name; {} when the blob is no
        day file."""
        if blob in self._cache:
            return self._cache[blob]
        if time.monotonic() > self.deadline:
            raise GitUnavailable(f"the git scan ran past its {DEADLINE_SECONDS} s deadline (reading blobs)")
        if self._cat is None:
            try:
                self._cat = subprocess.Popen(["git", "-C", self.root, "cat-file", "--batch"], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except OSError as exc:
                raise GitUnavailable(f"git cat-file could not run ({exc.__class__.__name__})")
            # the watchdog: a blocked read ends when the process is killed at the deadline
            self._watchdog = threading.Timer(max(0.0, self.deadline - time.monotonic()), self._cat.kill)
            self._watchdog.daemon = True
            self._watchdog.start()
        try:
            self._cat.stdin.write((blob + "\n").encode())
            self._cat.stdin.flush()
            head = self._cat.stdout.readline().split()
        except (OSError, ValueError) as exc:
            raise GitUnavailable(f"git cat-file stopped ({exc.__class__.__name__})")
        if not head:
            raise GitUnavailable(f"git cat-file stopped (the {DEADLINE_SECONDS} s deadline, or git ended)")
        obj = None
        if len(head) == 3:
            size = int(head[2])
            data = self._cat.stdout.read(size)
            self._cat.stdout.read(1)
            if len(data) < size:
                raise GitUnavailable("git cat-file stopped mid-blob")
            try:
                obj = json.loads(data)
            except ValueError:
                obj = None
        rows = {name: (ticket_key(row), row) for name, row in first_rows(obj, "newPrediction").items()}
        self._cache[blob] = rows
        return rows


# --- the scan ---------------------------------------------------------------------------------------------------------
def read_detail(database_dir):
    """The previous data/database/lotteryLiveness.json, or {} (none yet, or unreadable)."""
    detail = read_json(os.path.join(database_dir, DETAIL_FILE))
    return detail if isinstance(detail, dict) and isinstance(detail.get("games"), dict) else {}


class Scan:
    """
    The liveness of every lottery game under one data/database folder: judge(game) per game, each within the one
    deadline; the git checks of the checkout run once. previous: the detail file as the last run wrote it (read_detail
    by default) - the fallback for unstamped draws when git cannot be used.
    """

    def __init__(self, database_dir, schedule=None, deadline_seconds=DEADLINE_SECONDS, previous=None):
        self.database_dir = os.path.abspath(database_dir)
        self.root = os.path.dirname(os.path.dirname(self.database_dir))
        self.schedule = schedule or default_schedule()
        self.deadline = time.monotonic() + deadline_seconds
        self.previous = read_detail(self.database_dir) if previous is None else previous
        self._checked = None   # None: not checked yet; "": the checkout is usable; else why not

    def _checkout_error(self):
        """Why git cannot be the evidence for this folder ('' when it can): the checks that do not depend on the game."""
        if self._checked is not None:
            return self._checked
        real = os.path.realpath
        try:
            if real(self.database_dir) != real(os.path.join(self.root, "data", "database")):
                raise GitUnavailable("not the data/database folder of a checkout")
            top = _git(self.root, ["rev-parse", "--show-toplevel"], self.deadline).strip()
            if real(top) != real(self.root):
                raise GitUnavailable("the folder sits inside another checkout, not its own")
            if _git(self.root, ["rev-parse", "--is-shallow-repository"], self.deadline).strip() == "true":
                raise GitUnavailable("a shallow clone: the history is cut short")
            self._checked = ""
        except GitUnavailable as exc:
            self._checked = str(exc)
        return self._checked

    def judge(self, game):
        """
        One game's verdicts: {"game", "checked" ('stamp+git' | 'stamp only' | 'unchecked'), "error", "checkedThrough",
        "unscheduled" (draw dates on a weekday the schedule does not list), "draws": {ISO date: {"close", "ticketFile",
        "rows": {name: verdict}, "duplicates": {name: extra rows}}}}. A verdict: {"status": 'live' | 'recovered' |
        'late' | 'unchecked', "evidence" ('stamp' | 'commit' | 'previous check', live only), "reason" (late),
        "firstSaved", "delayHours" (late), "ticket", "commit", "savedAt" (recovered)}.
        A game the schedule does not list is 'unchecked' with an error. Git problems fall back to 'stamp only'; any
        other exception is the caller's (the report leaves that game's live blocks out).
        """
        if game not in self.schedule["games"]:
            return self._judge(game, None, error=f"{game} is not in src/drawClose.json: no sales close to judge by",
                               unchecked=True)
        error = self._checkout_error()
        if not error:
            folder = _Folder(self.root, f"data/database/{game}", self.deadline)
            try:
                with folder:
                    if not _git(self.root, ["ls-files", "--", f"data/database/{game}/"], self.deadline).strip():
                        raise GitUnavailable(f"data/database/{game} is not tracked in git")
                    return self._judge(game, folder)
            except GitUnavailable as exc:
                error = str(exc)
        return self._judge(game, None, error=error)

    def _judge(self, game, folder, error=None, unchecked=False):
        path = os.path.join(self.database_dir, game)
        files = sorted((d, n) for d, n in ((day_of(n), n) for n in os.listdir(path)) if d)
        versions = folder.versions() if folder is not None else {}
        prev_game = (self.previous.get("games") or {}).get(game) or {}
        result = {"game": game, "checked": "stamp+git" if folder is not None else ("unchecked" if unchecked else "stamp only"),
                  "error": error, "checkedThrough": None if folder is not None else prev_game.get("checkedThrough"),
                  "unscheduled": [], "draws": {}}
        loaded = {}

        def load(name):
            if name not in loaded:
                loaded[name] = read_json(os.path.join(path, name))
            return loaded[name]

        for index, (dday, dname) in enumerate(files):
            dobj = load(dname)
            if not isinstance(dobj, dict) or not dobj.get("realResult") or not dobj.get("currentPrediction"):
                continue
            rows = [r for r in dobj["currentPrediction"] if scored(r)]
            if not rows:
                continue
            iso = dday.isoformat()
            pname = files[index - 1][1] if index else None
            pobj = load(pname) if pname else None
            entry = {"close": None, "ticketFile": pname, "rows": {}, "duplicates": {}}
            result["draws"][iso] = entry
            if unchecked:
                for row in rows:
                    self._add(entry, row["name"], {"status": "unchecked"})
                continue
            close, on_schedule = close_instant(game, dday, self.schedule)
            if not on_schedule:
                result["unscheduled"].append(iso)
            entry["close"] = iso_utc(close)
            close_ts = close.timestamp()
            stamp = stamp_of(pobj)
            on_disk = {name: ticket_key(row) for name, row in first_rows(pobj, "newPrediction").items()}
            vers = versions.get(pname, []) if pname else []
            # the blob written by the latest commit before the close; a deletion writes none (development deleted
            # whole folders on its own branch on 21 Nov 2025 while production kept them - that unsaved no ticket)
            standing = next((v for v in reversed(vers) if v[0] < close_ts and v[2]), None)
            covering = next((v for v in vers if stamp is not None and v[0] >= stamp), None)
            if folder is not None and pname and stamp is None:
                result["checkedThrough"] = iso
            for row in rows:
                name = row["name"]
                if name in entry["rows"]:
                    entry["duplicates"][name] = entry["duplicates"].get(name, 0) + 1
                    continue
                key = ticket_key(row)
                if pname is None:
                    verdict = {"status": "late", "reason": "no ticket file before this draw", "firstSaved": None, "delayHours": None}
                elif folder is not None:
                    verdict = self._by_git(folder, name, key, stamp, close_ts, vers, standing, covering, on_disk)
                else:
                    verdict = self._by_stamp(prev_game, iso, name, key, stamp, close_ts, on_disk)
                entry["rows"][name] = verdict
        if folder is None and not unchecked and any(v["status"] == "unchecked" for d in result["draws"].values() for v in d["rows"].values()):
            result["checked"] = "unchecked"
        return result

    @staticmethod
    def _add(entry, name, verdict):
        if name in entry["rows"]:
            entry["duplicates"][name] = entry["duplicates"].get(name, 0) + 1
        else:
            entry["rows"][name] = verdict

    @staticmethod
    def _late(reason, close_ts, first_saved):
        return {"status": "late", "reason": reason, "firstSaved": iso_utc(first_saved),
                "delayHours": round((first_saved - close_ts) / 3600, 1) if first_saved is not None else None}

    def _by_git(self, folder, name, key, stamp, close_ts, vers, standing, covering, on_disk):
        # (a) the stamp, with the coverage check: the first commit at or after it holds the scored ticket
        if stamp is not None and stamp < close_ts:
            held = folder.rows(covering[2]).get(name, (None,))[0] if covering and covering[2] else (
                None if covering else on_disk.get(name))
            if held == key:
                return {"status": "live", "evidence": "stamp"}
        # (b) the earliest commit holding the scored ticket
        first = next((v for v in vers if v[2] and folder.rows(v[2]).get(name, (None,))[0] == key), None)
        if first and first[0] < close_ts:
            return {"status": "live", "evidence": "commit"}
        # (c) the version standing at the close holds the row with other content
        if standing and name in folder.rows(standing[2]):
            ticket = folder.rows(standing[2])[name][1]
            return {"status": "recovered", "commit": standing[1][:8], "savedAt": iso_utc(standing[0]),
                    "ticket": {"predictions": ticket.get("predictions"), "multiPick": ticket.get("multiPick")}}
        # (d) late: the first save of the scored ticket (else the stamp) after the close
        first_saved = first[0] if first else (stamp if stamp is not None and stamp >= close_ts else None)
        return self._late(LATE_FIRST if standing is None else LATE_ROW, close_ts, first_saved)

    def _by_stamp(self, prev_game, iso, name, key, stamp, close_ts, on_disk):
        if stamp is not None and stamp < close_ts and on_disk.get(name) == key:
            return {"status": "live", "evidence": "stamp"}
        if stamp is not None and stamp >= close_ts:
            return self._late(LATE_FIRST, close_ts, stamp)
        through = prev_game.get("checkedThrough")
        if not through or iso > through:
            return {"status": "unchecked"}
        prev = (prev_game.get("draws") or {}).get(iso)
        if not prev:
            return {"status": "live", "evidence": "previous check"}
        late = prev.get("late")
        if late == "all" or (isinstance(late, list) and name in late):
            reason = (prev.get("lateReasons") or {}).get(name) or prev.get("lateReason") or LATE_FIRST
            return {"status": "late", "reason": reason, "firstSaved": prev.get("firstSaved"), "delayHours": prev.get("delayHours")}
        recovered = prev.get("recovered") or {}
        if name in (recovered.get("rows") or {}):
            return {"status": "recovered", "commit": recovered.get("commit"), "savedAt": recovered.get("savedAt"),
                    "ticket": recovered["rows"][name]}
        return {"status": "live", "evidence": "previous check"}


# --- reading a verdict ------------------------------------------------------------------------------------------------
def ticket_for(judged, iso, name, occurrence, model):
    """
    What the live record scores for one stored row of one draw: (status, the row to score or None). status is
    'live', 'recovered' (the row with the ticket that stood at the close), 'late', 'duplicate' (a second row of the
    name) or 'unchecked'. A draw or row the scan did not see (a file written after it) is 'unchecked'.
    """
    if occurrence:
        return "duplicate", None
    verdict = (((judged or {}).get("draws") or {}).get(iso) or {}).get("rows", {}).get(name)
    if not verdict:
        return "unchecked", None
    if verdict["status"] == "live":
        return "live", model
    if verdict["status"] == "recovered":
        return "recovered", {"name": name, "predictions": verdict["ticket"]["predictions"], "multiPick": verdict["ticket"].get("multiPick")}
    return verdict["status"], None


# --- the detail file ----------------------------------------------------------------------------------------------------
def detail_of(judged):
    """
    One game's entry of lotteryLiveness.json: {"checked", "checkedThrough", "draws": {ISO date: {...}}}, only the draws
    with a row that is not live as stored: close, ticketFile, late ('all' or the names) with lateReason (lateReasons
    when the rows differ), firstSaved and delayHours (the earliest late row's), recovered {commit, savedAt, rows:
    {name: {predictions, multiPick}}}, duplicates {name: extra rows}, unchecked ('all' or the names).
    """
    draws = {}
    for iso, entry in judged["draws"].items():
        rows = entry["rows"]
        late = sorted(n for n, v in rows.items() if v["status"] == "late")
        recovered = {n: v for n, v in rows.items() if v["status"] == "recovered"}
        unchecked = sorted(n for n, v in rows.items() if v["status"] == "unchecked")
        if not late and not recovered and not unchecked and not entry["duplicates"]:
            continue
        out = {"close": entry["close"], "ticketFile": entry["ticketFile"]}
        if late:
            out["late"] = "all" if len(late) == len(rows) else late
            reasons = {n: rows[n]["reason"] for n in late}
            if len(set(reasons.values())) == 1:
                out["lateReason"] = next(iter(reasons.values()))
            else:
                out["lateReasons"] = reasons
            saved = sorted((rows[n]["firstSaved"], rows[n]["delayHours"]) for n in late if rows[n].get("firstSaved"))
            out["firstSaved"], out["delayHours"] = saved[0] if saved else (None, None)
        if recovered:
            first = recovered[sorted(recovered)[0]]
            out["recovered"] = {"commit": first["commit"], "savedAt": first["savedAt"],
                                "rows": {n: {"predictions": v["ticket"]["predictions"], "multiPick": v["ticket"].get("multiPick")}
                                         for n, v in recovered.items()}}
        if entry["duplicates"]:
            out["duplicates"] = dict(entry["duplicates"])
        if unchecked:
            out["unchecked"] = "all" if len(unchecked) == len(rows) else unchecked
        draws[iso] = out
    return {"checked": judged["checked"], "checkedThrough": judged["checkedThrough"], "draws": draws}


def _detail_text(detail):
    """The detail file's text: sorted keys, one line per draw (about 230 KB instead of 900 KB indented; a changed
    verdict is a changed line in git)."""
    compact = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    lines = ["{"]
    for key in sorted(k for k in detail if k != "games"):
        lines.append(f" {json.dumps(key)}: {compact(detail[key])},")
    lines.append(' "games": {')
    games = sorted(detail["games"])
    for g_index, game in enumerate(games):
        entry = detail["games"][game]
        lines.append(f"  {json.dumps(game)}: {{")
        for key in sorted(k for k in entry if k != "draws"):
            lines.append(f"   {json.dumps(key)}: {compact(entry[key])},")
        draws = sorted((entry.get("draws") or {}).items())
        lines.append('   "draws": {' + ("" if draws else "}"))
        for d_index, (iso, item) in enumerate(draws):
            lines.append(f"    {json.dumps(iso)}: {compact(item)}{',' if d_index + 1 < len(draws) else ''}")
        if draws:
            lines.append("   }")
        lines.append("  }" + ("," if g_index + 1 < len(games) else ""))
    lines.append(" }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_detail(database_dir, judged_games, previous=None):
    """
    Writes data/database/lotteryLiveness.json from this run's judged games; returns True when the file changed. A game
    judged with git replaces its entry; a game judged without it keeps its previous entry, its draws judged by the stamps
    alone added (so the draw pages' markers still match the report's counts); a game this run could not judge at all
    (absent from judged_games) keeps its previous entry. Sorted keys, no run time, rewritten only when the text changes.
    """
    previous = read_detail(database_dir) if previous is None else previous
    games = dict((previous or {}).get("games") or {})
    for game, judged in judged_games.items():
        entry = detail_of(judged)
        if judged["checked"] != "stamp+git" and game in games:
            kept = dict(games[game])
            draws = dict(kept.get("draws") or {})
            for iso, item in entry["draws"].items():
                if iso not in draws or (iso > (kept.get("checkedThrough") or "")):
                    draws[iso] = item
            kept["draws"], kept["checked"] = draws, judged["checked"]
            entry = kept
        games[game] = entry
    detail = {"about": ("The lottery draws whose scored tickets were not all saved before the sales close, per game "
                        "(src/Liveness.py, written by Helpers.generate_model_performance_report). Only those draws are "
                        "listed: every other scored draw is live. Rewritten only when a verdict changes."),
              "rule": RULE, "schedule": "src/drawClose.json", "games": games}
    text = _detail_text(detail)
    path = os.path.join(database_dir, DETAIL_FILE)
    try:
        with open(path) as handle:
            if handle.read() == text:
                return False
    except OSError:
        pass
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return True


# --- self-check -------------------------------------------------------------------------------------------------------
def _self_check():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    utc = timezone.utc
    # the schedule, DST included
    check(close_instant("lotto", date(2026, 10, 24))[0] == datetime(2026, 10, 24, 17, 0, tzinfo=utc), "lotto Sat 24 Oct 19:00 = 17:00Z")
    check(close_instant("lotto", date(2026, 10, 31))[0] == datetime(2026, 10, 31, 18, 0, tzinfo=utc), "lotto Sat 31 Oct 19:00 = 18:00Z (winter time)")
    check(close_instant("keno", date(2026, 10, 25)) == (datetime(2026, 10, 25, 12, 0, tzinfo=utc), True), "keno Sun 25 Oct 13:00 = 12:00Z")
    check(close_instant("vikinglotto", date(2026, 10, 21))[0] == datetime(2026, 10, 21, 16, 0, tzinfo=utc), "vikinglotto Wed 18:00")
    check(close_instant("vikinglotto", date(2026, 10, 22)) == (datetime(2026, 10, 22, 16, 0, tzinfo=utc), False),
          "an unscheduled weekday gets the earliest close, flagged")
    try:
        close_instant("bingo", date(2026, 10, 22))
        failures.append("a game outside the table must raise")
    except KeyError:
        pass

    base = tempfile.mkdtemp(prefix="liveness-")
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", HOME=base)

    def git(repo, *args, at=None):
        e = dict(env)
        if at is not None:
            e["GIT_COMMITTER_DATE"] = e["GIT_AUTHOR_DATE"] = f"@{int(at.timestamp())} +0000"
        subprocess.run(["git", "-C", repo, "-c", "user.name=check", "-c", "user.email=check@example.invalid",
                        "-c", "commit.gpgsign=false"] + list(args), check=True, capture_output=True, env=e)

    def day_file(folder, day, current=None, new=None, real=(1, 2, 3, 4, 5, 6, 7), made=None):
        obj = {"realResult": list(real) if current else [], "currentPrediction": current or [], "newPrediction": new or []}
        if made is not None:
            obj["madeAt"] = made.isoformat(timespec="seconds")
        with open(os.path.join(folder, f"{day.year}-{day.month}-{day.day}.json"), "w") as handle:
            json.dump(obj, handle)

    def row(name, numbers, multi=None):
        out = {"name": name, "predictions": [list(numbers)]}
        if multi:
            out["multiPick"] = list(multi)
        return out

    try:
        repo = os.path.join(base, "repo")
        lotto = os.path.join(repo, "data", "database", "lotto")
        os.makedirs(lotto)
        git(repo, "init", "-q")
        # draws: Sat 3 Oct (close 17:00Z), Wed 7 Oct, Sat 10 Oct, Wed 14 Oct (all CEST); ticket files the day before each
        d0, d1, d2, d3, d4 = date(2026, 9, 30), date(2026, 10, 3), date(2026, 10, 7), date(2026, 10, 10), date(2026, 10, 14)
        hour = lambda d, h: datetime(d.year, d.month, d.day, h, 0, tzinfo=utc)
        A, B, C = [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]
        # d0's file holds the ticket for d1: Commit row (committed 10:00, before the close), Rewrite row (A at the close,
        # C after), Same row (re-committed after the close with the same content), ABA row (A, B, A before the close,
        # C stored), Moved row (A, then B, B before the close, C stored: the LATEST commit's B stood, not the first's A)
        day_file(lotto, d0, new=[row("Commit", A), row("Rewrite", A), row("Same", B), row("ABA", A), row("Moved", A)])
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "d0", at=hour(d1, 8))
        day_file(lotto, d0, new=[row("Commit", A), row("Rewrite", A), row("Same", B), row("ABA", B), row("Moved", B)])
        git(repo, "commit", "-q", "-am", "d0 b", at=hour(d1, 9))
        day_file(lotto, d0, new=[row("Commit", A), row("Rewrite", A), row("Same", B), row("ABA", A), row("Moved", B)])
        git(repo, "commit", "-q", "-am", "d0 a again", at=hour(d1, 10))
        # after the close: Rewrite, ABA and Moved rewritten to C, Same re-committed unchanged, a row added; d1's draw file
        day_file(lotto, d0, new=[row("Commit", A), row("Rewrite", C), row("Same", B), row("ABA", C), row("Added", A), row("Moved", C)])
        day_file(lotto, d1, current=[row("Commit", A), row("Rewrite", C), row("Same", B), row("ABA", C), row("Added", A),
                                     row("Moved", C), row("Commit", B)],
                 new=[row("Stamped", A), row("Stamped2", B)], made=hour(d2, 8))
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "after close", at=hour(d1, 20))
        # d1's file holds d2's ticket, stamped 8:00 on d2; its first commit after the stamp (9:00, before the 17:00Z close)
        # holds it; after the close a row is added under the same stamp
        day_file(lotto, d1, current=[row("Commit", A), row("Rewrite", C), row("Same", B), row("ABA", C), row("Added", A),
                                     row("Moved", C), row("Commit", B)],
                 new=[row("Stamped", A), row("Stamped2", C)], made=hour(d2, 8))
        git(repo, "commit", "-q", "-am", "d1 at the stamp", at=hour(d2, 9))
        day_file(lotto, d1, current=[row("Commit", A), row("Rewrite", C), row("Same", B), row("ABA", C), row("Added", A),
                                     row("Moved", C), row("Commit", B)],
                 new=[row("Stamped", A), row("Stamped2", C), row("Extra", A)], made=hour(d2, 8))
        git(repo, "commit", "-q", "-am", "row added", at=hour(d2, 21))
        # never committed: d2's file (d3's ticket) stamped before the close, d3's (d4's ticket) stamped 1 h after it
        day_file(lotto, d2, current=[row("Stamped", A), row("Stamped2", C), row("Extra", A)], new=[row("Uncommitted", A)], made=hour(d3, 7))
        day_file(lotto, d3, current=[row("Uncommitted", A)], new=[row("StampedLate", B)], made=hour(d4, 18))
        day_file(lotto, d4, current=[row("StampedLate", B)])

        judged = Scan(os.path.join(repo, "data", "database"), previous={}).judge("lotto")
        status = lambda iso, name: judged["draws"][iso]["rows"][name]["status"]
        evidence = lambda iso, name: judged["draws"][iso]["rows"][name].get("evidence")
        check(judged["checked"] == "stamp+git" and judged["error"] is None, f"git is used: {judged['checked']} {judged['error']}")
        check(status("2026-10-03", "Commit") == "live" and evidence("2026-10-03", "Commit") == "commit", "a ticket committed before the close is live by its commit")
        rewrite = judged["draws"]["2026-10-03"]["rows"]["Rewrite"]
        check(rewrite["status"] == "recovered" and rewrite["ticket"]["predictions"] == [A] and len(rewrite["commit"]) == 8,
              f"rewritten after the close: recovered with the ticket at the close and its commit: {rewrite}")
        check(status("2026-10-03", "Same") == "live", "rewritten after the close with the same content: live")
        aba = judged["draws"]["2026-10-03"]["rows"]["ABA"]
        check(aba["status"] == "recovered" and aba["ticket"]["predictions"] == [A],
              f"A, B, A before the close: the latest commit's A stood at the close, not B: {aba}")
        moved = judged["draws"]["2026-10-03"]["rows"]["Moved"]
        check(moved["status"] == "recovered" and moved["ticket"]["predictions"] == [B],
              f"A, then B before the close: the latest commit's B stood at the close, not the first commit's A: {moved}")
        added = judged["draws"]["2026-10-03"]["rows"]["Added"]
        check(added["status"] == "late" and added["reason"] == LATE_ROW and added["delayHours"] == 3.0,
              f"a row only in the post-close version: late, row added, 3 h: {added}")
        check(judged["draws"]["2026-10-03"]["duplicates"] == {"Commit": 1}, "the second 'Commit' row is a duplicate")
        check(ticket_for(judged, "2026-10-03", "Commit", 1, row("Commit", B)) == ("duplicate", None), "a duplicate is not live")
        check(status("2026-10-07", "Stamped") == "live" and evidence("2026-10-07", "Stamped") == "stamp", "stamped before the close and committed as stamped: live by the stamp")
        check(status("2026-10-07", "Stamped2") == "live" and evidence("2026-10-07", "Stamped2") == "stamp",
              "a row edited after the stamp but before its first commit reads live: the stamp covers the file up to its first commit")
        extra = judged["draws"]["2026-10-07"]["rows"]["Extra"]
        check(extra["status"] == "late" and extra["reason"] == LATE_ROW, f"a row added after the close under an unchanged stamp is late: {extra}")
        check(status("2026-10-10", "Uncommitted") == "live" and evidence("2026-10-10", "Uncommitted") == "stamp", "stamped before the close, not committed yet: live by the stamp")
        late = judged["draws"]["2026-10-14"]["rows"]["StampedLate"]
        check(late["status"] == "late" and late["reason"] == LATE_FIRST and late["delayHours"] == 1.0, f"stamped after the close: late, 1 h: {late}")
        check(judged["checkedThrough"] == "2026-10-03", f"the newest unstamped draw judged with git: {judged['checkedThrough']}")
        check(ticket_for(judged, "2026-10-03", "Rewrite", 0, row("Rewrite", C))[1]["predictions"] == [A], "the live record scores the ticket at the close")

        # the report on this history (Helpers.generate_model_performance_report, 27 Oct 2026 review): the live record
        # scores the ticket that stood at the close, leaves the late rows and the second 'Commit' row out and ranks on
        # it - never on an unchecked game; a game whose judging raises loses its live blocks, never the report
        try:
            from src.Helpers import Helpers
        except ImportError:  # run from within src/
            from Helpers import Helpers
        database = os.path.join(repo, "data", "database")

        def report_of(folder):
            with contextlib.redirect_stdout(io.StringIO()):
                Helpers().generate_model_performance_report(folder, combinationShuffles=0)
            with open(os.path.join(folder, "modelPerformance.json")) as handle:
                game = json.load(handle)["games"]["lotto"]
            for name in os.listdir(folder):      # the report's files: the detail-file checks below start without them
                if not os.path.isdir(os.path.join(folder, name)):
                    os.remove(os.path.join(folder, name))
            return game, {m["name"]: m for m in game["models"]}

        game, models = report_of(database)
        check(game.get("rankedOn") == "live" and game["liveness"]["checked"] == "stamp+git",
              f"the report ranks on the live record: {game.get('rankedOn')} {game.get('liveness')}")
        check(models["Rewrite"]["avg_hits"] == 0.0 and models["Rewrite"]["live"]["avg_hits"] == 6.0 and models["Rewrite"]["live"]["recovered"] == 1,
              f"a recovered row is scored on the ticket that stood at the close (A, 6 hits), not the stored one (C, 0): {models['Rewrite']}")
        check(models["Commit"]["draws"] == 2 and models["Commit"]["live_draws"] == 1 and models["Commit"]["live"]["not_live"] == 1
              and models["Commit"]["live"]["avg_hits"] == 6.0, f"the first 'Commit' row (A) is live, the second (B) is not: {models['Commit']}")
        check(all(models[n]["live_draws"] == 0 and models[n]["live"]["not_live"] == 1 for n in ("Added", "Extra", "StampedLate")),
              "the late rows are left out of the live record")
        # the row tests (src/RowChance.py, 27 Oct 2026 review) read the same live record: the recovered ticket as it
        # stood at the close, the late rows and the second 'Commit' row left out
        check(all((m.get("chance") or {}).get("draws") == m["live_draws"] and (m.get("chance") or {}).get("notLive") == m["draws"] - m["live_draws"]
                  and (not m["live_draws"] or abs(m["chance"]["mean"] - m["live"]["avg_hits"]) < 1e-3) for m in models.values())
              and models["Rewrite"]["chance"]["mean"] == 6.0 and game.get("chance", {}).get("method") == "exact",
              f"the row tests read the live record: {models['Rewrite'].get('chance')} {models['Commit'].get('chance')}")
        tally = {k: game["liveness"][k] for k in ("draws", "rowsLive", "rowsRecovered", "rowsNotLive", "allLive", "allCounted")}
        check(tally == {"draws": 4, "rowsLive": 5, "rowsRecovered": 3, "rowsNotLive": 4, "allLive": 1, "allCounted": 1}
              and sum(m["draws"] - m["live_draws"] for m in models.values()) == tally["rowsNotLive"],
              f"the game's counts, draws - live draws = the rows not live: {tally}")
        order = [m["live"]["avg_hits"] for m in game["models"] if m["live_draws"] >= game["live"]["minDrawsForRanking"]]
        check(order == sorted(order, reverse=True) and game["bestModel"] == game["live"]["bestModel"], f"the models in the live block's order: {order}")
        loose = os.path.join(repo, "scratch-report", "data", "database")
        shutil.copytree(lotto, os.path.join(loose, "lotto"))
        game, models = report_of(loose)
        check("rankedOn" not in game and game["liveness"]["checked"] == "unchecked" and game["bestModel"] == "Commit",
              f"an unchecked game (a copy inside another checkout, no previous file) keeps the all-draws ranking: {game.get('rankedOn')} {game['liveness']}")
        check(str(game.get("chance", {}).get("error", "")).startswith("liveness not checked") and all("chance" not in m for m in models.values()),
              f"an unchecked game gets no row test, only why: {game.get('chance')}")
        module = sys.modules.get("src.Liveness") or sys.modules.get("Liveness")   # the copy Helpers imported
        judge = module.Scan.judge
        module.Scan.judge = lambda self, name: 1 / 0
        try:
            game, models = report_of(database)
        finally:
            module.Scan.judge = judge
        check(game["liveness"] == {"checked": "unchecked", "error": "ZeroDivisionError: division by zero"} and "live" not in game
              and "rankedOn" not in game and all("live" not in m for m in models.values()) and models["Commit"]["draws"] == 2,
              f"a game whose judging raises keeps its report, without live blocks: {game.get('liveness')}")

        # a deletion before the close (development's branch, 21 Nov 2025) unsaves nothing: the version before it stood
        dreams = os.path.join(repo, "data", "database", "eurodreams")
        os.makedirs(dreams)
        e0, e1 = date(2026, 10, 1), date(2026, 10, 5)            # Thu, then Mon 5 Oct (close 20:00 CEST = 18:00Z)
        day_file(dreams, e0, new=[row("Kept", A)])
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "e0", at=hour(e0, 9))
        os.remove(os.path.join(dreams, "2026-10-1.json"))
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "deleted", at=hour(e1, 10))
        day_file(dreams, e0, new=[row("Kept", B)])
        day_file(dreams, e1, current=[row("Kept", B)], real=(1, 2, 3, 4, 5, 6))
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "re-added", at=hour(e1, 21))
        kept = Scan(os.path.join(repo, "data", "database"), previous={}).judge("eurodreams")["draws"]["2026-10-05"]["rows"]["Kept"]
        check(kept["status"] == "recovered" and kept["ticket"]["predictions"] == [A], f"a deletion before the close is passed over: {kept}")

        detail = detail_of(judged)
        check(set(detail["draws"]) == {"2026-10-03", "2026-10-07", "2026-10-14"}, f"only the draws with a row not live as stored: {sorted(detail['draws'])}")
        check(detail["draws"]["2026-10-14"]["late"] == "all" and detail["draws"]["2026-10-03"]["late"] == ["Added"], "late: all or the names")
        check(write_detail(database, {"lotto": judged}, previous={}) and not write_detail(database, {"lotto": judged}),
              "the detail file is rewritten only when it changes")
        with open(os.path.join(database, DETAIL_FILE)) as handle:
            written = json.load(handle)
        check("generatedAt" not in json.dumps(written) and written["games"]["lotto"]["checkedThrough"] == "2026-10-03", "no run time in the detail file")

        # no git: the stamps alone, unstamped draws from the previous detail file, else unchecked
        copy = os.path.join(base, "copy", "data", "database")
        shutil.copytree(database, copy)
        fallback = Scan(copy).judge("lotto")
        check(fallback["checked"] == "stamp only" and bool(fallback["error"]),
              f"no git: stamp only, with the reason: {fallback['checked']} {fallback['error']}")
        rows = fallback["draws"]["2026-10-03"]["rows"]
        check(rows["Commit"]["status"] == "live" and rows["Commit"]["evidence"] == "previous check" and rows["Rewrite"]["status"] == "recovered"
              and rows["Rewrite"]["ticket"]["predictions"] == [A] and rows["Added"]["status"] == "late",
              f"an unstamped draw takes its verdict from the previous file: {rows}")
        check(fallback["draws"]["2026-10-14"]["rows"]["StampedLate"]["status"] == "late"
              and fallback["draws"]["2026-10-10"]["rows"]["Uncommitted"]["status"] == "live", "stamped draws are judged by the stamp")
        check(fallback["draws"]["2026-10-07"]["rows"]["Extra"]["status"] == "live",
              "without git the coverage check is gone: a row added under the stamp reads live (the documented limit)")
        # the stamp vouches only for the ticket its file holds: a ticket file changed under its stamp is no evidence
        changed_path = os.path.join(copy, "lotto", "2026-10-7.json")
        with open(changed_path) as handle:
            changed = json.load(handle)
        changed["newPrediction"] = [row("Uncommitted", B)]
        with open(changed_path, "w") as handle:
            json.dump(changed, handle)
        check(Scan(copy).judge("lotto")["draws"]["2026-10-10"]["rows"]["Uncommitted"]["status"] == "unchecked",
              "without git, a stamped file that no longer holds the scored ticket is not live by its stamp")
        os.remove(os.path.join(copy, DETAIL_FILE))
        bare = Scan(copy).judge("lotto")
        check(bare["checked"] == "unchecked" and bare["draws"]["2026-10-03"]["rows"]["Commit"]["status"] == "unchecked",
              f"no git and no previous file: unchecked, not live: {bare['checked']}")
        check(write_detail(copy, {"lotto": bare}) and read_detail(copy)["games"]["lotto"]["draws"]["2026-10-03"]["unchecked"] == "all",
              "unchecked draws are listed, so the draw pages can mark them")

        # a database nested inside another checkout: git answers for the outer one and must not be trusted
        nested = os.path.join(repo, "scratch", "data", "database")
        shutil.copytree(lotto, os.path.join(nested, "lotto"))
        inner = Scan(nested, previous={}).judge("lotto")
        check(inner["checked"] == "unchecked" and "another checkout" in (inner["error"] or ""),
              f"a scratch copy inside another checkout falls back: {inner['checked']} {inner['error']}")
        # the other checks before git is trusted: a database folder elsewhere in the checkout (git would answer for the
        # real data/database), a game folder git does not track, a shallow clone (its history is cut short)
        elsewhere = os.path.join(repo, "scratch2", "database")
        shutil.copytree(lotto, os.path.join(elsewhere, "lotto"))
        odd = Scan(elsewhere, previous={}).judge("lotto")
        check(odd["checked"] != "stamp+git" and "data/database folder" in (odd["error"] or ""),
              f"a database folder that is not the checkout's data/database falls back: {odd['checked']} {odd['error']}")
        shutil.copytree(lotto, os.path.join(database, "pick3"))
        untracked = Scan(database, previous={}).judge("pick3")
        check(untracked["checked"] != "stamp+git" and "not tracked" in (untracked["error"] or ""),
              f"a game folder git does not track falls back: {untracked['checked']} {untracked['error']}")
        shallow = os.path.join(base, "shallow")
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", shallow], check=True, capture_output=True, env=env)
        cut = Scan(os.path.join(shallow, "data", "database"), previous={}).judge("lotto")
        check(cut["checked"] != "stamp+git" and "shallow" in (cut["error"] or ""), f"a shallow clone falls back: {cut['checked']} {cut['error']}")
        # a game the schedule does not list, and the deadline
        os.makedirs(os.path.join(database, "bingo"))
        shutil.copy(os.path.join(lotto, "2026-10-3.json"), os.path.join(database, "bingo", "2026-10-3.json"))
        shutil.copy(os.path.join(lotto, "2026-9-30.json"), os.path.join(database, "bingo", "2026-9-30.json"))
        bingo = Scan(database, previous={}).judge("bingo")
        check(bingo["checked"] == "unchecked" and "drawClose" in (bingo["error"] or "")
              and all(v["status"] == "unchecked" for v in bingo["draws"]["2026-10-03"]["rows"].values()),
              f"a game outside the schedule is unchecked: {bingo['checked']} {bingo['error']}")
        expired = Scan(database, deadline_seconds=0, previous={}).judge("lotto")
        check(expired["checked"] == "unchecked" and "deadline" in (expired["error"] or ""), f"the deadline ends the git scan: {expired['error']}")
        folder = _Folder(repo, "data/database/lotto", time.monotonic() + 0.2)
        with folder:
            blob = folder.versions()["2026-9-30.json"][0][2]
            folder.rows(blob)
            time.sleep(0.4)
            check(folder._cat.poll() is not None, "the watchdog kills the cat-file process at the deadline")
            try:
                folder._cache.clear()
                folder.rows(blob)
                failures.append("a read after the deadline must stop")
            except GitUnavailable:
                pass
        check(folder._cat is None, "the cat-file process is closed after the game")
    finally:
        shutil.rmtree(base, ignore_errors=True)

    for message in failures:
        print("FAIL:", message)
    print(f"Liveness self-check: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    return not failures


if __name__ == "__main__":
    raise SystemExit(0 if _self_check() else 1)
