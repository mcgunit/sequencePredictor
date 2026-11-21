"""
The Council rows' job (docs/roadmap.md item 12, 17 Nov 2026): asks the Council's local LLMs for a ticket for each
lottery game's coming draw and adds the answers as rows to the game's newest day file, where the next predictor run
scores them like every other row. See src/CouncilTicket.py for the rows, the prompts and what a valid ticket is.

It runs after the lottery predictors (jobs.js queues it behind the last of them), never inside them: a slow or
failed LLM costs no game its predictions, and a row whose answer was no ticket is absent that day. It holds
process.lock while it works, because it writes day files and pushes them.

    python3 CouncilRows.py                    every lottery game
    python3 CouncilRows.py -g pick3,lotto     these games
    python3 CouncilRows.py --dry              ask and print, write and push nothing

Exit status: 0 when it ran (also with rows missing: the log says which), 75 when no Council model answered (the
machine is off or its services are down), 1 on an error.

The models are the row council's own (src/llmCouncil/rows.json, 20 Nov 2026), not the ones the Council pages ask.
Every member answers the draws question; the record question goes to one member (`record_member`), because every
model given the other rows' tickets answered with the leading row's. Members are asked one after the other, each
for every game, and a member on a machine whose card is taken (`guard`) is left out for the day.

A game is skipped when its newest day file already holds the rows (a second run the same day adds only what is
missing) or holds no predictions. What was asked and answered is kept beside the day files, in
data/council/<game>/<day>.json: the prompts, each attempt's answer and why it was refused.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "llmCouncil"))

from src import CouncilTicket as ct                     # noqa: E402
from council import client, health, sampling            # noqa: E402

UNREACHABLE_EXIT = 75
LOCK_FILE = os.path.join(os.getcwd(), "process.lock")
# One answer is one short line. The row models run with thinking off (their services' presets, 20 Nov 2026: with
# thinking on a model spent the whole answer thinking, or minutes a question on the DL360); the room is for a model
# that still writes a few sentences first. A model that goes on writing is cut off rather than waited for.
MAX_TOKENS = 400
# A model is loaded on its first question, which took up to 3.5 minutes in the trial of 20 Nov 2026: the wait covers
# that. An answer itself took 2 minutes at most.
CALL_TIMEOUT_S = 600
RETRY_TEMPERATURE = 0.5
# After a wake the models still load: how long the first health check waits for at least one of them.
WAIT_S = 600
BUDGET_S = 90 * 60


def log(text):
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {text}", flush=True)


# --- the lock (the same file and rule as Predictor.py) ------------------------------------------------------------
def is_running():
    if not os.path.exists(LOCK_FILE):
        return False
    try:
        with open(LOCK_FILE, "r") as handle:
            os.kill(int(handle.read().strip()), 0)
        return True
    except (ValueError, ProcessLookupError):
        try:
            os.remove(LOCK_FILE)
        except FileNotFoundError:
            pass
        return False
    except PermissionError:
        return True


def create_lock():
    try:
        with open(LOCK_FILE, "x") as handle:
            handle.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def remove_lock():
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass


# --- the Council's endpoints --------------------------------------------------------------------------------------
def endpoints(config):
    """(members, head): the enabled ones, each with its sampling settings."""
    members = []
    for member in config.get("members") or []:
        if member.get("enabled", True):
            members.append(dict(member, **{"_sampling": sampling.for_member(member)}))
    head = config.get("head") or None
    if head and head.get("enabled", True):
        head = dict(head, **{"_sampling": sampling.for_head(head)})
    else:
        head = None
    return members, head


def reachable(endpoint, timeout_s=5):
    try:
        health.check(endpoint["base_url"], timeout_s)
        return True
    except Exception:
        return False


def wait_for(members, head, wait_s, sleep=time.sleep):
    """The endpoints that answer, waiting up to wait_s for the first member (models load after a wake)."""
    until = time.time() + wait_s
    while True:
        up = [m for m in members if reachable(m)]
        if up or time.time() >= until:
            break
        sleep(15)
    if up and len(up) < len(members):
        # the others may still be loading: one more look, a little later
        sleep(min(30, max(0, until - time.time())))
        up = [m for m in members if reachable(m)]
    return up, (head if head and reachable(head) else None)


def model_of(endpoint):
    """The model the endpoint runs, as it names it (recorded with the row)."""
    if endpoint.get("model"):
        return endpoint["model"]
    ids = health.loaded_models(endpoint["base_url"])
    return ids[0] if ids else None


# --- asking ---------------------------------------------------------------------------------------------------------
def ask_ticket(game, day, name, endpoint, question, ask=client.ask, attempts=ct.ATTEMPTS):
    """
    Asks one endpoint until its answer is a ticket of the game, ATTEMPTS times at most.
    Returns (ticket or None, record): the record holds every attempt (seed, answer, why refused).
    """
    settings = endpoint["_sampling"]
    record = {"temperature": settings["temperature"], "attempts": []}
    again = ""
    for attempt in range(1, attempts + 1):
        # the config's own seed for the first try; a second try with the same seed would give the same answer
        seed = settings["seed"] if settings["seed"] is not None and attempt == 1 else ct.seed_for(game, day, name, attempt)
        # ... as would one at temperature 0: a retry is asked a little warmer than that
        temperature = settings["temperature"] if attempt == 1 else max(settings["temperature"], RETRY_TEMPERATURE)
        entry = {"seed": seed, "temperature": temperature}
        started = time.time()
        try:
            answer = ask(endpoint["base_url"], question + again, CALL_TIMEOUT_S, system_prompt=ct.SYSTEM, model=endpoint.get("model"),
                         temperature=temperature, max_tokens=MAX_TOKENS, seed=seed)
        except Exception as error:
            entry.update({"error": str(error)[:300], "seconds": round(time.time() - started, 1)})
            record["attempts"].append(entry)
            continue
        ticket, why = ct.read_answer(game, answer)
        entry.update({"answer": str(answer)[:600], "seconds": round(time.time() - started, 1)})
        if ticket is None:
            entry["refused"] = why
            # the next try is told what was wrong with this one
            said = " ".join(str(answer).split())[-200:]
            again = f"\n\nYou answered: {said}\nThat is not a valid ticket: {why}. Answer again, with one valid ticket."
        record["attempts"].append(entry)
        if ticket is not None:
            record["seed"] = seed
            record["temperature"] = temperature
            return ticket, record
    return None, record


def keno_sizes(rows):
    """The ticket sizes the day's other rows play (the tuned use_5..use_10): the Council rows play the same."""
    for row in rows:
        predictions = row.get("predictions") if isinstance(row, dict) else None
        if predictions and len(predictions) > 1 and not ct.is_council(row.get("name")):
            sizes = sorted({len(p) for p in predictions[1:] if isinstance(p, list) and 5 <= len(p) <= 10})
            if sizes:
                return tuple(sizes)
    return ct.KENO_SIZES


def ranked_names(root, game):
    try:
        with open(os.path.join(root, "data", "database", "modelPerformance.json"), "r") as handle:
            models = json.load(handle)["games"][game]["models"]
        return [m["name"] for m in models if isinstance(m, dict) and m.get("name")]
    except Exception:
        return None


# --- a shared card (20 Nov 2026) ----------------------------------------------------------------------------------
# Members on the GPU worker share its card with the deep-learning hyperopts and backtests started from the web app.
# A model loaded beside one of those does not fit and stalls both, so such a member is left out for the day while
# one runs. The config names the check: "guards": {"gpu": {"ssh": "root@host", "pid_files": [...]}} and, on a member,
# "guard": "gpu". A check that cannot be made counts as held: a row less is the cheaper mistake. A pid file older
# than the machine's boot is not believed: its number may belong to another process by now.
SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./-]{1,200}$")
SAFE_HOST = re.compile(r"^[A-Za-z0-9_.-]{1,64}@[A-Za-z0-9_.-]{1,200}$")


def held(guard, run=subprocess.run):
    """Why the guard's machine is taken (a text), or None when it is free."""
    if not isinstance(guard, dict):
        return "its guard is not in the config"
    host, files = guard.get("ssh"), guard.get("pid_files") or []
    if not (isinstance(host, str) and SAFE_HOST.match(host)) or not files or not all(isinstance(f, str) and SAFE_PATH.match(f) for f in files):
        return "its guard is not a host and a list of pid files"
    script = "; ".join(f'[ -f {f} ] && [ {f} -nt /proc/1 ] && kill -0 "$(cat {f})" 2>/dev/null && echo "BUSY {f}"' for f in files) + "; echo CHECKED"
    try:
        out = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, script], capture_output=True, text=True, timeout=40)
    except Exception as error:
        return f"the check on {host} failed ({str(error)[:120]})"
    text = out.stdout or ""
    if "CHECKED" not in text:
        return f"the check on {host} gave no answer"
    busy = [line.split(" ", 1)[1] for line in text.splitlines() if line.startswith("BUSY ")]
    return f"{os.path.basename(busy[0])} on {host} is alive" if busy else None


# --- one game's rows ----------------------------------------------------------------------------------------------
def asks(kind, member_name, record_member):
    """Whether this member answers this kind: with a record member set, the record question goes to that one only."""
    return kind != "record" or not record_member or member_name == record_member


def combines(kind, head, record_member):
    """Whether the kind gets a combined row: not the record kind when one member alone answers it."""
    return bool(head) and not (kind == "record" and record_member)


def game_context(root, game, members, head, record_member=None):
    """
    What a game's rows are made from: its newest day file, the questions and the rows it still misses. None when
    there is nothing to do. Nothing is asked or written here.
    """
    files = ct.day_files(os.path.join(root, "data", "database", game))
    if not files:
        log(f"{game}: no day file")
        return None
    days = []
    for _, path in files[-(max(ct.RECENT_DRAWS.values()) + 1):]:
        try:
            with open(path, "r") as handle:
                days.append(json.load(handle))
        except Exception as error:
            log(f"{game}: {os.path.basename(path)} could not be read ({error}): left out of the summary")
    date, day_file = files[-1]
    day = f"{date.year}-{date.month}-{date.day}"
    try:
        with open(day_file, "r") as handle:
            today = json.load(handle)
    except Exception as error:
        log(f"{game}: {day}.json could not be read ({error})")
        return None
    existing = today.get("newPrediction") or []
    if not ct.without_council(existing):
        log(f"{game}: {day}.json holds no predictions yet: nothing to read, no Council rows")
        return None
    have = {r.get("name") for r in existing if isinstance(r, dict)}
    wanted = [ct.row_name(kind, m["name"]) for kind in ct.KINDS for m in members if asks(kind, m["name"], record_member)]
    wanted += [ct.row_name(kind, ct.COMBINED) for kind in ct.KINDS if combines(kind, head, record_member)]
    if all(name in have for name in wanted):
        log(f"{game}: {day}.json already holds the Council rows")
        return None
    if days and days[-1] is not today:
        days[-1] = today
    ranking = ranked_names(root, game)
    context = {"game": game, "day": day, "day_file": day_file, "have": have, "record_member": record_member,
               "sizes": keno_sizes(existing) if game == "keno" else ct.KENO_SIZES, "info": {}, "question": {},
               "tickets": {kind: [] for kind in ct.KINDS}, "rows": [],
               "record": {"game": game, "day": day, "askedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kinds": {}}}
    for kind in ct.KINDS:
        context["info"][kind] = ct.summary(game, days, ranked_names=ranking, kind=kind)
        context["question"][kind] = ct.prompt(game, kind, context["info"][kind])
        context["record"]["kinds"][kind] = {"prompt": context["question"][kind], "members": {}, "head": None}
    return context


def ask_member(context, member, models, ask=client.ask, out_of_time=lambda: False):
    """One member's rows for the game: every kind it answers and the day file does not hold yet."""
    game, day = context["game"], context["day"]
    for kind in ct.KINDS:
        name = ct.row_name(kind, member["name"])
        if name in context["have"] or not asks(kind, member["name"], context["record_member"]):
            continue
        if out_of_time():
            log(f"{game}: out of time before {name}")
            continue
        question = context["question"][kind]
        ticket, asked = ask_ticket(game, day, name, member, question, ask=ask)
        context["record"]["kinds"][kind]["members"][member["name"]] = dict(asked, ticket=ticket)
        if ticket is None:
            last = asked["attempts"][-1] if asked["attempts"] else {}
            log(f"{game}: {name}: no ticket after {len(asked['attempts'])} tries ({last.get('refused') or last.get('error') or 'no answer'}): the row is absent today")
            continue
        context["tickets"][kind].append((member["name"], ticket))
        context["rows"].append(ct.as_row(game, name, ticket, {
            "kind": kind, "member": member["name"], "lab": member.get("lab"), "model": models.get(member["name"]),
            "temperature": asked["temperature"], "seed": asked["seed"], "tries": len(asked["attempts"]),
            "prompt": ct.sha(question)}, keno_sizes=context["sizes"]))
        log(f"{game}: {name}: {ticket} (try {len(asked['attempts'])})")


def ask_head(context, head, models, ask=client.ask, out_of_time=lambda: False):
    """The combined rows: the head reads the members' tickets of a kind and answers with one."""
    game, day = context["game"], context["day"]
    for kind in ct.KINDS:
        name = ct.row_name(kind, ct.COMBINED)
        if not combines(kind, head, context["record_member"]) or name in context["have"] or out_of_time():
            continue
        tickets = context["tickets"][kind]
        if len(tickets) < 2:
            log(f"{game}: {name}: {len(tickets)} member ticket(s), nothing to combine: the row is absent today")
            continue
        question = ct.head_prompt(game, kind, context["info"][kind], tickets)
        ticket, asked = ask_ticket(game, day, name, head, question, ask=ask)
        context["record"]["kinds"][kind]["head"] = dict(asked, prompt=question, ticket=ticket, combines=[m for m, _ in tickets])
        if ticket is None:
            log(f"{game}: {name}: no ticket after {len(asked['attempts'])} tries: the row is absent today")
            continue
        context["rows"].append(ct.as_row(game, name, ticket, {
            "kind": kind, "member": ct.COMBINED, "lab": head.get("lab"), "model": models.get(head["name"]), "head": head["name"],
            "combines": [m for m, _ in tickets], "temperature": asked["temperature"], "seed": asked["seed"],
            "tries": len(asked["attempts"]), "prompt": ct.sha(question)}, keno_sizes=context["sizes"]))
        log(f"{game}: {name}: {ticket} (try {len(asked['attempts'])})")


def ordered_rows(context, members):
    """The game's rows in the order the tables show them: kind by kind, the members in the config's order, combined last."""
    place = {ct.row_name(kind, name): (k, i) for k, kind in enumerate(ct.KINDS)
             for i, name in enumerate([m["name"] for m in members] + [ct.COMBINED])}
    return sorted(context["rows"], key=lambda row: place.get(row.get("name"), (len(ct.KINDS), 0)))


def rows_for_game(root, game, members, head, models, ask=client.ask, out_of_time=lambda: False, record_member=None):
    """
    The game's Council rows for its newest day file. Returns (day, day_file, rows, record) or None when there is
    nothing to do. Nothing is written here.
    """
    context = game_context(root, game, members, head, record_member)
    if not context:
        return None
    for member in members:
        ask_member(context, member, models, ask=ask, out_of_time=out_of_time)
    ask_head(context, head, models, ask=ask, out_of_time=out_of_time)
    return context["day"], context["day_file"], ordered_rows(context, members), context["record"]


def write_rows(root, game, day, day_file, rows, record):
    """Adds the rows to the day file (read again: it is the file as it is now) and keeps what was asked beside it."""
    with open(day_file, "r") as handle:
        today = json.load(handle)
    added = ct.put_back(today.setdefault("newPrediction", []), rows)
    tmp = f"{day_file}.tmp"
    with open(tmp, "w") as handle:
        json.dump(today, handle, indent=2)
    os.replace(tmp, day_file)
    folder = os.path.join(root, "data", "council", game)
    os.makedirs(folder, exist_ok=True)
    kept_file = os.path.join(folder, f"{day}.json")
    earlier = []
    try:
        with open(kept_file, "r") as handle:
            earlier = json.load(handle).get("runs") or []
    except Exception:
        pass
    with open(kept_file, "w") as handle:
        json.dump({"game": game, "day": day, "runs": earlier + [record]}, handle, indent=2)
    return added, [day_file, kept_file]


def push(root, game, files):
    """Commits the game's two files and pushes them: only these, never the whole checkout."""
    try:
        subprocess.run(["git", "add", "--"] + files, check=True, cwd=root)
        subprocess.run(["git", "commit", "-m", f"council rows: {game}", "--"] + files, check=True, cwd=root)
        subprocess.run(["git", "push"], check=True, cwd=root)
        log(f"{game}: pushed")
        return True
    except Exception as error:
        log(f"{game}: the rows are in the day file but were not pushed ({error}): the next run that pushes takes them along")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adds the Council rows to the lottery games' newest day files.")
    parser.add_argument("-g", "--games", default=",".join(ct.GAMES), help="comma separated; default: every lottery game")
    parser.add_argument("--config", default=os.path.join(ROOT, "src", "llmCouncil", "rows.json"),
                        help="the row models (default: src/llmCouncil/rows.json; the Council pages have their own config.json)")
    parser.add_argument("--budget", type=int, default=BUDGET_S, help="seconds after which no further row is asked")
    parser.add_argument("--wait", type=int, default=WAIT_S, help="seconds to wait for the first model to answer")
    parser.add_argument("--dry", action="store_true", help="ask and print; write and push nothing")
    parser.add_argument("--no-push", action="store_true", help="write the rows, leave the commit and push out")
    args = parser.parse_args(argv)

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    unknown = [g for g in games if g not in ct.GAMES]
    if unknown:
        print(f"Unknown game(s): {', '.join(unknown)}. Known: {', '.join(ct.GAMES)}")
        return 1
    try:
        with open(args.config, "r") as handle:
            config = json.load(handle)
    except Exception as error:
        print(f"The Council's config could not be read ({error})")
        return 1

    if not args.dry:
        if is_running():
            print("Another job holds process.lock: not started.")
            return 1
        if not create_lock():
            print("process.lock could not be taken: not started.")
            return 1
    try:
        members, head = endpoints(config)
        record_member = config.get("record_member") or None
        log(f"Council rows for {', '.join(games)}: {len(members)} members, {'a head' if head else 'no head'}")
        members, head = wait_for(members, head, args.wait)
        if not members:
            log("No Council model answers (is the machine on, are its services up?): no rows today.")
            return UNREACHABLE_EXIT
        log(f"answering: {', '.join(m['name'] for m in members)}{' and the head ' + head['name'] if head else '; the head does not answer: no combined rows'}")
        models = {e["name"]: model_of(e) for e in members + ([head] if head else [])}
        started = time.time()
        out_of_time = lambda: time.time() - started > args.budget
        made = 0
        # One member at a time over every game (20 Nov 2026): a machine that holds one model at a time loads each
        # model once a morning instead of once a question. The head combines when the members have answered.
        contexts = {}
        for game in games:
            try:
                context = game_context(ROOT, game, members, head, record_member)
                if context:
                    contexts[game] = context
            except Exception as error:
                log(f"{game}: failed ({error}): the other games go on")
        for member in (members if contexts else []):
            if member.get("guard"):
                why = held((config.get("guards") or {}).get(member["guard"]))
                if why:
                    log(f"{member['name']}: left out today, {why}")
                    continue
            for game, context in contexts.items():
                try:
                    ask_member(context, member, models, out_of_time=out_of_time)
                except Exception as error:
                    log(f"{game}: {member['name']} failed ({error}): the others go on")
        for game, context in contexts.items():
            try:
                ask_head(context, head, models, out_of_time=out_of_time)
                rows = ordered_rows(context, members)
                if args.dry:
                    log(f"{game}: dry run, {len(rows)} rows not written")
                    continue
                added, files = write_rows(ROOT, game, context["day"], context["day_file"], rows, context["record"])
                made += added
                log(f"{game}: {added} Council rows added to {context['day']}.json")
                if added and not args.no_push:
                    push(ROOT, game, files)
            except Exception as error:
                log(f"{game}: failed ({error}): the other games go on")
        log(f"done: {made} rows in {round(time.time() - started)} s")
        return 0
    finally:
        if not args.dry:
            remove_lock()


def _self_check():
    """The order of asking, the single record row and the guard, on a temporary root with a stand-in for the models."""
    import tempfile
    checks = 0
    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "data", "database", "pick3")
        os.makedirs(folder)
        row = lambda name, digits: {"name": name, "predictions": [digits]}
        for day, drawn in (("2026-11-17", [1, 2, 3]), ("2026-11-18", [4, 5, 6]), ("2026-11-19", None)):
            body = {"currentPrediction": [row("A Model", [1, 2, 3])], "newPrediction": [row("A Model", [7, 8, 9])]}
            if drawn:
                body["realResult"] = drawn
            with open(os.path.join(folder, f"{day}.json"), "w") as handle:
                json.dump(body, handle)
        members = [{"name": n, "base_url": "http://x", "_sampling": {"temperature": 0.7, "seed": None}} for n in ("One", "Two", "Three")]
        head = {"name": "head", "base_url": "http://x", "_sampling": {"temperature": 0.0, "seed": 42}}
        models = {e["name"]: e["name"] for e in members + [head]}
        asked = []

        def ask(base_url, question, timeout, **kwargs):
            asked.append(question)
            return "TICKET: 1 2 3"

        # every member answers both kinds, the head combines both: eight rows, kind by kind
        day, _file, rows, record = rows_for_game(root, "pick3", members, head, models, ask=ask)
        names = [r["name"] for r in rows]
        assert names == [ct.row_name(k, n) for k in ct.KINDS for n in ("One", "Two", "Three", ct.COMBINED)], names
        checks += 1
        # a record member: one record row and no combined record row; the draws rows as before
        asked.clear()
        day, _file, rows, record = rows_for_game(root, "pick3", members, head, models, ask=ask, record_member="Two")
        names = [r["name"] for r in rows]
        assert names == [ct.row_name("record", "Two")] + [ct.row_name("draws", n) for n in ("One", "Two", "Three", ct.COMBINED)], names
        assert len(asked) == 5 and record["kinds"]["record"]["head"] is None
        checks += 2
        # one member at a time: the contexts of two games, a member asked for both before the next member
        contexts = {g: game_context(root, "pick3", members, head, "Two") for g in ("first", "second")}
        order = []
        for member in members:
            for key, context in contexts.items():
                ask_member(context, member, models, ask=lambda *a, **k: order.append((member["name"], key)) or "TICKET: 1 2 3")
        assert [m for m, _ in order] == ["One", "One", "Two", "Two", "Two", "Two", "Three", "Three"], order
        checks += 1
        # a member left out (its machine taken) leaves the head two tickets to combine; with one ticket there is none
        context = game_context(root, "pick3", members, head, "Two")
        for member in members[:2]:
            ask_member(context, member, models, ask=ask)
        ask_head(context, head, models, ask=ask)
        assert ct.row_name("draws", ct.COMBINED) in [r["name"] for r in context["rows"]]
        context = game_context(root, "pick3", members, head, "Two")
        ask_member(context, members[0], models, ask=ask)
        ask_head(context, head, models, ask=ask)
        assert [r["name"] for r in context["rows"]] == [ct.row_name("draws", "One")]
        checks += 2
        # a day file that holds the wanted rows is left alone, also when the record member is the only record row
        path = os.path.join(folder, "2026-11-19.json")
        with open(path) as handle:
            body = json.load(handle)
        body["newPrediction"] += [row(ct.row_name("record", "Two"), [1, 2, 3])] + [row(ct.row_name("draws", n), [1, 2, 3]) for n in ("One", "Two", "Three", ct.COMBINED)]
        with open(path, "w") as handle:
            json.dump(body, handle)
        assert game_context(root, "pick3", members, head, "Two") is None and game_context(root, "pick3", members, head, None) is not None
        checks += 1

    # the guard: free, taken, and every way of not knowing counts as taken
    class Out:
        def __init__(self, text):
            self.stdout = text
    guard = {"ssh": "root@192.168.0.1", "pid_files": ["/opt/w/hyperopt/hyperopt.pid", "/opt/w/app/backtest.pid"]}
    seen = []
    assert held(guard, run=lambda cmd, **k: seen.append(cmd) or Out("CHECKED\n")) is None
    assert seen[0][:1] == ["ssh"] and "BatchMode=yes" in seen[0] and "hyperopt.pid" in seen[0][-1]
    assert "hyperopt.pid" in held(guard, run=lambda cmd, **k: Out("BUSY /opt/w/hyperopt/hyperopt.pid\nCHECKED\n"))
    assert held(guard, run=lambda cmd, **k: Out("")) and held(None) and held({"ssh": "root@h", "pid_files": []})
    assert held({"ssh": "root@h; rm -rf /", "pid_files": ["/a"]}, run=lambda cmd, **k: Out("CHECKED")) is not None
    assert held({"ssh": "root@h", "pid_files": ["/a; reboot"]}, run=lambda cmd, **k: Out("CHECKED")) is not None

    def boom(cmd, **k):
        raise OSError("no ssh")
    assert "failed" in held(guard, run=boom)
    checks += 7
    # the shipped config: readable, its record member and guards exist
    with open(os.path.join(ROOT, "src", "llmCouncil", "rows.json")) as handle:
        config = json.load(handle)
    members, head = endpoints(config)
    assert head and len(members) >= 2 and config["record_member"] in [m["name"] for m in members]
    assert all(m["guard"] in config["guards"] for m in members if m.get("guard")) and len({m["name"] for m in members}) == len(members)
    checks += 2
    print(f"CouncilRows: {checks} checks passed")


if __name__ == "__main__":
    sys.exit(main())
