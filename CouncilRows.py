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

A game is skipped when its newest day file already holds the rows (a second run the same day adds only what is
missing) or holds no predictions. What was asked and answered is kept beside the day files, in
data/council/<game>/<day>.json: the prompts, each attempt's answer and why it was refused.
"""

import argparse
import json
import os
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
# One answer is one short line, but the head thinks before it answers (about 150 tokens, 17 Nov 2026) and an answer
# cut off in its thinking is no ticket: room for both. A model that goes on writing is cut off rather than waited for.
MAX_TOKENS = 400
CALL_TIMEOUT_S = 300
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


def rows_for_game(root, game, members, head, models, ask=client.ask, out_of_time=lambda: False):
    """
    The game's Council rows for its newest day file. Returns (day, day_file, rows, record) or None when there is
    nothing to do. Nothing is written here.
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
    wanted = [ct.row_name(kind, m["name"]) for kind in ct.KINDS for m in members] + [ct.row_name(kind, ct.COMBINED) for kind in ct.KINDS if head]
    if all(name in have for name in wanted):
        log(f"{game}: {day}.json already holds the Council rows")
        return None
    if days and days[-1] is not today:
        days[-1] = today
    sizes = keno_sizes(existing) if game == "keno" else ct.KENO_SIZES
    ranking = ranked_names(root, game)
    rows, record = [], {"game": game, "day": day, "askedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kinds": {}}
    for kind in ct.KINDS:
        info = ct.summary(game, days, ranked_names=ranking, kind=kind)
        question = ct.prompt(game, kind, info)
        kept = {"prompt": question, "members": {}, "head": None}
        record["kinds"][kind] = kept
        tickets = []
        for member in members:
            name = ct.row_name(kind, member["name"])
            if name in have:
                continue
            if out_of_time():
                log(f"{game}: out of time before {name}")
                continue
            ticket, asked = ask_ticket(game, day, name, member, question, ask=ask)
            kept["members"][member["name"]] = dict(asked, ticket=ticket)
            if ticket is None:
                last = asked["attempts"][-1] if asked["attempts"] else {}
                log(f"{game}: {name}: no ticket after {len(asked['attempts'])} tries ({last.get('refused') or last.get('error') or 'no answer'}): the row is absent today")
                continue
            tickets.append((member["name"], ticket))
            rows.append(ct.as_row(game, name, ticket, {
                "kind": kind, "member": member["name"], "lab": member.get("lab"), "model": models.get(member["name"]),
                "temperature": asked["temperature"], "seed": asked["seed"], "tries": len(asked["attempts"]),
                "prompt": ct.sha(question)}, keno_sizes=sizes))
            log(f"{game}: {name}: {ticket} (try {len(asked['attempts'])})")
        name = ct.row_name(kind, ct.COMBINED)
        if head and name not in have and not out_of_time():
            if len(tickets) < 2:
                log(f"{game}: {name}: {len(tickets)} member ticket(s), nothing to combine: the row is absent today")
                continue
            question = ct.head_prompt(game, kind, info, tickets)
            ticket, asked = ask_ticket(game, day, name, head, question, ask=ask)
            kept["head"] = dict(asked, prompt=question, ticket=ticket, combines=[m for m, _ in tickets])
            if ticket is None:
                log(f"{game}: {name}: no ticket after {len(asked['attempts'])} tries: the row is absent today")
                continue
            rows.append(ct.as_row(game, name, ticket, {
                "kind": kind, "member": ct.COMBINED, "lab": head.get("lab"), "model": models.get(head["name"]), "head": head["name"],
                "combines": [m for m, _ in tickets], "temperature": asked["temperature"], "seed": asked["seed"],
                "tries": len(asked["attempts"]), "prompt": ct.sha(question)}, keno_sizes=sizes))
            log(f"{game}: {name}: {ticket} (try {len(asked['attempts'])})")
    return day, day_file, rows, record


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
    parser.add_argument("--config", default=os.path.join(ROOT, "src", "llmCouncil", "config.json"))
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
        for game in games:
            if out_of_time():
                log(f"{game}: out of time ({args.budget} s): not asked today")
                continue
            try:
                result = rows_for_game(ROOT, game, members, head, models, out_of_time=out_of_time)
                if not result:
                    continue
                day, day_file, rows, record = result
                if args.dry:
                    log(f"{game}: dry run, {len(rows)} rows not written")
                    continue
                added, files = write_rows(ROOT, game, day, day_file, rows, record)
                made += added
                log(f"{game}: {added} Council rows added to {day}.json")
                if added and not args.no_push:
                    push(ROOT, game, files)
            except Exception as error:
                log(f"{game}: failed ({error}): the other games go on")
        log(f"done: {made} rows in {round(time.time() - started)} s")
        return 0
    finally:
        if not args.dry:
            remove_lock()


if __name__ == "__main__":
    sys.exit(main())
