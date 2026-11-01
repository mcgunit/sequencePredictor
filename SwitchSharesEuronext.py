#!/usr/bin/env python3
"""
The one-time switch of the shares games from the four Nasdaq names of the
kickoff (NVDA, AAPL, MSFT, ASML in USD) to the five Euronext shares (Shell,
ING, AB InBev, KBC, UCB in EUR) - owner decision of 31 Oct 2026: the old
record is removed, not archived, after a backup.

    python3 SwitchSharesEuronext.py                               # dry run: what would go, nothing touched
    python3 SwitchSharesEuronext.py --apply --backup /root/backups/shares-nasdaq-2026-11-02
    python3 SwitchSharesEuronext.py --root /elsewhere/checkout ...   # another checkout (a scratch copy)

Run it in the production checkout AFTER the Euronext code is deployed (git
pull + npm run build:web) and BEFORE the 18:30 evening run, then restart the
server (pm2 restart sequencePredictor): its pages keep what they read in
memory. --apply, in this order:

  1. refuses while process.lock names a live process (a pipeline job runs),
     when the deployed code is not the Euronext universe, or when the store
     says the switch was already made - a second run would delete the new
     Euronext record; then takes process.lock itself until it is done
  2. backup into --backup (a new folder outside the checkout): files.tar.gz
     with every file below and the three summary files before their edit,
     copies of data/markets.sqlite3 and db.sqlite3 (SQLite's backup, so a
     consistent copy), manifest.json and RESTORE.txt; the archive is read
     back and counted before anything is removed
  3. removes every path named after shares or sharesweek in data/ (game
     files, page exports, the database folders, models, controls, caches)
     and the two bestParams files; data/readings/ is kept - a weekly reading
     is a dated document of what was read that week
  4. markets store: the shares' and sharesweek's results, predictions, bars,
     game days, ticket publications, model performance and instruments
  5. db.sqlite3: the Optuna studies of shares and sharesweek
  6. data/database/modelPerformance.json, rowChance.json, lagPeakHistory.json:
     the shares and sharesweek entries
  7. installs the Euronext universe (FetchMarkets --init), fetches Euronext's
     two years (--update --market shares) and back-builds Yahoo's history
     since 2016 (--back-build), unless --no-fetch; marks the store switched

The pipeline commit (`git add -A`) records the deletions with the next run.
Exit code: 0 done (or a dry run), 1 refused or failed before any removal,
2 removed and installed but a source failed - rerun the FetchMarkets
commands it prints.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

from src.MarketData import (DEFAULT_UNIVERSE, WINDOW_SOURCES, back_build_history, connect, fetch_history, history_symbol,
                            install_universe, instruments, update_all)

GAMES = ("shares", "sharesweek")
# a path segment named after one of the games: shares/, shares.json, shares-regimes.json, model_shares.keras,
# catboost_sharesweek_models, meta_position_table_shares.joblib, shares_position_policy.json - never "sharesX"
NAMED = re.compile(r"(^|[/_.-])shares(week)?([/_.-]|$)")
SKIPPED_DIRS = ("data/readings",)
SUMMARY_FILES = ("data/database/modelPerformance.json", "data/database/rowChance.json", "data/database/lagPeakHistory.json")
SWITCH_KEY = "shares_switched_to_euronext"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --- what goes -------------------------------------------------------------------------------------------------------

def named_paths(root):
    """The topmost paths named after the games: the two bestParams files in the root, everything under data/."""
    found = [name for name in sorted(os.listdir(root)) if name in (f"bestParams_{g}.json" for g in GAMES)]
    data = os.path.join(root, "data")
    for folder, dirs, files in os.walk(data):
        rel = os.path.relpath(folder, root)
        if rel in SKIPPED_DIRS:
            dirs[:] = []
            continue
        keep = []
        for d in sorted(dirs):
            path = os.path.join(rel, d)
            (found.append(path) if NAMED.search("/" + d + "/") and path not in SKIPPED_DIRS else keep.append(d))
        dirs[:] = keep
        found += [os.path.join(rel, f) for f in sorted(files) if NAMED.search("/" + f)]
    return found


def files_under(root, paths):
    """Every file the named paths hold (a folder by its files), relative to root."""
    out = []
    for path in paths:
        full = os.path.join(root, path)
        if os.path.isdir(full) and not os.path.islink(full):
            for folder, _, files in os.walk(full):
                out += [os.path.relpath(os.path.join(folder, f), root) for f in sorted(files)]
        else:
            out.append(path)
    return out


def store_rows(conn):
    """Rows per table that belong to the games, and the shares instruments installed."""
    ids = [r[0] for r in conn.execute(f"SELECT id FROM instruments WHERE market IN ({','.join('?' * len(GAMES))})", GAMES)]
    marks = ",".join("?" * len(ids)) or "NULL"
    counts = {}
    for table in ("results", "predictions", "bars", "intraday_bars"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE instrument_id IN ({marks})", ids).fetchone()[0]
    for table in ("game_days", "ticket_published", "model_performance"):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE market IN ({','.join('?' * len(GAMES))})",
                                         GAMES).fetchone()[0]
    counts["instruments"] = len(ids)
    return ids, counts


def installed_shares(conn):
    return [(r["market"], r["symbol"], r["source"], r["position"])
            for r in conn.execute("SELECT market, symbol, source, position FROM instruments WHERE market = 'shares' ORDER BY position")]


def wanted_shares():
    return [(r["market"], r["symbol"], r["source"], r["position"]) for r in DEFAULT_UNIVERSE if r["market"] == "shares"]


def studies_of(db_path):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute("SELECT study_name FROM studies ORDER BY study_name")]
    except sqlite3.OperationalError:
        names = []
    conn.close()
    return [n for n in names if re.match(r"^shares(week)?[_-]", n)]


def summary_keys(root):
    """(file, key path) of the games' entries in the three summary files: top level or under "games"."""
    out = []
    for rel in SUMMARY_FILES:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            continue
        with open(full, encoding="utf-8") as f:
            data = json.load(f)
        out += [(rel, (g,)) for g in GAMES if g in data]
        if isinstance(data.get("games"), dict):
            out += [(rel, ("games", g)) for g in GAMES if g in data["games"]]
    return out


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


def sqlite_copy(src, dst):
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    target = sqlite3.connect(dst)
    source.backup(target)
    target.close()
    source.close()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(root):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def make_backup(root, backup, files, plan):
    os.makedirs(backup)
    archive = os.path.join(backup, "files.tar.gz")
    members = files + [rel for rel in SUMMARY_FILES if os.path.exists(os.path.join(root, rel))]
    with tarfile.open(archive, "w:gz") as tar:
        for rel in members:
            tar.add(os.path.join(root, rel), arcname=rel, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        read_back = sum(1 for m in tar.getmembers() if m.isfile())
    if read_back != len(members):
        raise RuntimeError(f"the archive holds {read_back} file(s), {len(members)} were written - nothing removed")
    copies = {}
    for name, rel in (("markets.sqlite3", "data/markets.sqlite3"), ("db.sqlite3", "db.sqlite3")):
        if os.path.exists(os.path.join(root, rel)):
            sqlite_copy(os.path.join(root, rel), os.path.join(backup, name))
            copies[name] = rel
    manifest = dict(plan, made_at=now_iso(), root=os.path.abspath(root), git_head=git_head(root), archive_files=len(members),
                    archive_sha256=sha256(archive), database_copies=copies)
    with open(os.path.join(backup, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(backup, "RESTORE.txt"), "w", encoding="utf-8") as f:
        f.write(f"The Nasdaq shares record as it stood on {manifest['made_at']} in {manifest['root']} (git {manifest['git_head']}).\n\n"
                "To put it back (server stopped, no pipeline job running, the pre-Euronext code checked out):\n"
                f"  tar -xzf {archive} -C {manifest['root']}\n"
                + "".join(f"  cp {os.path.join(backup, name)} {os.path.join(manifest['root'], rel)}\n" for name, rel in copies.items())
                + "The database copies are whole files: restoring them also undoes everything else written to them since.\n")
    return manifest


def remove_paths(root, paths):
    for rel in paths:
        full = os.path.join(root, rel)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        elif os.path.lexists(full):
            os.remove(full)


def clear_store(conn, ids):
    marks = ",".join("?" * len(ids)) or "NULL"
    games = ",".join("?" * len(GAMES))
    with conn:
        for table in ("results", "predictions", "bars", "intraday_bars"):
            conn.execute(f"DELETE FROM {table} WHERE instrument_id IN ({marks})", ids)
        for table in ("game_days", "ticket_published", "model_performance"):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
                conn.execute(f"DELETE FROM {table} WHERE market IN ({games})", GAMES)
        conn.execute(f"DELETE FROM instruments WHERE id IN ({marks})", ids)


def delete_studies(db_path, names):
    if not names:
        return
    import optuna   # here, not at the top: a dry run and the self-check of the files do not need it
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.abspath(db_path)}"
    for name in names:
        optuna.delete_study(study_name=name, storage=storage)


def strip_summaries(root, keys):
    by_file = {}
    for rel, key in keys:
        by_file.setdefault(rel, []).append(key)
    for rel, entries in by_file.items():
        full = os.path.join(root, rel)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        data = json.loads(text)
        for key in entries:
            (data["games"] if key[0] == "games" else data).pop(key[-1], None)
        indent = re.match(r"\{\n( +)", text)
        out = json.dumps(data, indent=len(indent.group(1)) if indent else None, ensure_ascii=text.isascii())
        tmp = full + ".switch-tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if text.endswith("\n") else ""))
        os.replace(tmp, full)


def install_and_fetch(conn, fetch=True, log=print):
    """FetchMarkets --init, then (fetch) --update --market shares and --back-build: the number of sources that failed."""
    added, conflicts = install_universe(conn)
    log(f"universe: {len(added)} instrument(s) added ({', '.join(added) or 'none'})")
    for conflict in conflicts:
        log(f"universe CONFLICT: {conflict}")
    if conflicts:
        return len(conflicts)
    if not fetch:
        return 0
    failed = sum(1 for s in update_all(conn, market="shares", log=log) if s.get("error"))
    for instrument in instruments(conn, "shares"):
        listing = history_symbol(instrument["market"], instrument["symbol"])
        if instrument["source"] not in WINDOW_SOURCES or not listing:
            continue
        try:
            built = back_build_history(conn, instrument, fetch_history(listing))
        except Exception as exc:
            failed += 1
            log(f"{instrument['symbol']}: back-build FAILED - {exc}")
            continue
        log(f"{instrument['symbol']}: {built['inserted']} older bar(s) from {listing}, joined at {built['joined_at']} "
            f"(factor {built['factor']:.4f}); stored from {built['first']}")
    return failed


# --- the run ---------------------------------------------------------------------------------------------------------

def plan_of(root):
    markets_db = os.path.join(root, "data", "markets.sqlite3")
    conn = sqlite3.connect(f"file:{markets_db}?mode=ro", uri=True)   # a dry run writes nothing, not even the schema
    conn.row_factory = sqlite3.Row
    try:
        ids, rows = store_rows(conn)
        installed = installed_shares(conn)
        switched = conn.execute("SELECT value FROM meta WHERE key = ?", (SWITCH_KEY,)).fetchone()
    finally:
        conn.close()
    paths = named_paths(root)
    files = files_under(root, paths)
    return dict(paths=paths, files=files, instrument_ids=ids, store_rows=rows, installed=installed,
                studies=studies_of(os.path.join(root, "db.sqlite3")),
                summary_keys=[[rel, "/".join(key)] for rel, key in summary_keys(root)],
                switched=switched[0] if switched else None)


def refusal(root, plan, backup):
    """Why --apply must not run, or None."""
    wanted = wanted_shares()
    if len(wanted) != 5 or any(source != "euronext" for _, _, source, _ in wanted):
        return "the code in this checkout is not the Euronext universe - deploy it first (git pull)"
    if plan["switched"]:
        return f"the store says the switch was made on {plan['switched']}; a second run would delete the Euronext record"
    if plan["installed"] and all(source == "euronext" for _, _, source, _ in plan["installed"]):
        return "the shares installed are already the Euronext ones; nothing to switch"
    if not backup:
        return "--apply needs --backup <new folder outside the checkout>"
    real_root, real_backup = os.path.realpath(root), os.path.realpath(backup)
    if real_backup == real_root or real_backup.startswith(real_root + os.sep):
        return f"the backup folder {backup} is inside the checkout - the pipeline commit would sweep it in"
    if os.path.exists(backup):
        return f"the backup folder {backup} exists - give a new one"
    return None


def report(plan, log=print):
    log(f"paths named after {' / '.join(GAMES)}: {len(plan['paths'])} ({len(plan['files'])} file(s))")
    for path in plan["paths"]:
        log(f"  {path}")
    log("markets store, rows of the games: " + ", ".join(f"{t} {n}" for t, n in plan["store_rows"].items()))
    log("shares installed: " + (", ".join(f"{s} ({src}, {p})" for _, s, src, p in plan["installed"]) or "none"))
    log(f"Optuna studies ({len(plan['studies'])}): {', '.join(plan['studies']) or 'none'}")
    log("summary entries: " + (", ".join(f"{rel}:{key}" for rel, key in plan["summary_keys"]) or "none"))
    log("kept: data/readings/ (dated weekly readings), config/rankhistory-cache.json (keyed by commit), the logs")


def run(root, backup=None, apply=False, fetch=True, log=print):
    root = os.path.abspath(root)
    plan = plan_of(root)
    report(plan, log)
    why = refusal(root, plan, backup)
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
        manifest = make_backup(root, backup, plan["files"], plan)
        log(f"backup: {manifest['archive_files']} file(s) in {backup}/files.tar.gz, databases {', '.join(manifest['database_copies'])}")
        remove_paths(root, plan["paths"])
        log(f"removed {len(plan['paths'])} path(s)")
        conn = connect(os.path.join(root, "data", "markets.sqlite3"))
        try:
            clear_store(conn, plan["instrument_ids"])
            log("markets store: the Nasdaq instruments and their rows deleted")
            delete_studies(os.path.join(root, "db.sqlite3"), plan["studies"])
            log(f"Optuna: {len(plan['studies'])} studies deleted")
            strip_summaries(root, [(rel, tuple(key.split("/"))) for rel, key in plan["summary_keys"]])
            log(f"summary files: {len(plan['summary_keys'])} entries removed")
            failed = install_and_fetch(conn, fetch=fetch, log=log)
            with conn:
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (SWITCH_KEY, now_iso()))
        finally:
            conn.close()
    finally:
        os.remove(lock)
    if failed:
        log(f"{failed} source(s) failed - rerun: python3 FetchMarkets.py --update --market shares, then "
            "python3 FetchMarkets.py --back-build --market shares")
        return 2
    log("done - restart the server (pm2 restart sequencePredictor); the 18:30 run writes the first Euronext cards"
        + ("" if fetch else "; --no-fetch: run FetchMarkets --update and --back-build for the shares first"))
    return 0


def main():
    parser = argparse.ArgumentParser(prog="Switch shares to Euronext", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)), help="The checkout (default: this script's)")
    parser.add_argument("--apply", action="store_true", help="Do it (default: a dry run that lists what would go)")
    parser.add_argument("--backup", default=None, help="A new folder outside the checkout for the backup (required with --apply)")
    parser.add_argument("--no-fetch", action="store_true", help="Install the Euronext instruments but fetch no bars")
    args = parser.parse_args()
    return run(args.root, backup=args.backup, apply=args.apply, fetch=not args.no_fetch)


# --- self-check ------------------------------------------------------------------------------------------------------

def _self_check():
    """A made-up checkout with an old Nasdaq record: dry run, the refusals, --apply --no-fetch, the backup read back."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="switch-shares-")
    try:
        root = os.path.join(tmp, "checkout")
        kept = ["data/database/crypto/2026-10-30.json", "data/trainingData/crypto/crypto-gamedata-NL-2026.csv",
                "data/models/lstm_model/model_crypto.keras", "bestParams_crypto.json", "data/readings/2026-10-25.json",
                "data/models/catboost_cryptoweek_models/m.cbm", "data/markets/crypto.json"]
        gone = ["bestParams_shares.json", "bestParams_sharesweek.json", "data/database/shares/2026-10-30.json",
                "data/database/sharesweek/2026-10-30.json", "data/trainingData/shares/shares-returns.tsv",
                "data/markets/shares.json", "data/markets/shares-regimes.json", "data/markets/sharesweek/results-2026.csv",
                "data/models/lstm_model/model_shares.keras", "data/models/lstm_model/model_sharesweek.keras.fingerprint.json",
                "data/models/catboost_shares_models/m.cbm", "data/models/rl_position/shares_position_policy.json",
                "data/hyperOptCache/meta_position_table_sharesweek.joblib", "data/controls/results/shares-shuffled.json",
                "data/controls/discrimination/sharesweek-w10.json", "data/controls/shares/shuffled-seed1/shares-control.csv"]
        for rel in kept + gone:
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w") as f:
                f.write(rel)
        os.makedirs(os.path.join(root, "data", "readings"), exist_ok=True)
        with open(os.path.join(root, "data", "readings", "2026-10-25.json"), "w") as f:
            json.dump({"shares": "kept"}, f)
        perf = {"generatedAt": "x", "games": {"crypto": {"a": 1}, "shares": {"a": 2}, "sharesweek": {"a": 3}}}
        with open(os.path.join(root, "data/database/modelPerformance.json"), "w") as f:
            json.dump(perf, f, indent=2)
            f.write("\n")
        with open(os.path.join(root, "data/database/lagPeakHistory.json"), "w") as f:
            json.dump({"crypto": {}, "shares": {}, "sharesweek": {}}, f, indent=2)
        conn = connect(os.path.join(root, "data", "markets.sqlite3"))
        conn.executescript("CREATE TABLE IF NOT EXISTS game_days (market TEXT, date TEXT, position INTEGER, symbol TEXT, ret REAL, "
                           "bin INTEGER, edges TEXT)")
        crypto = [r for r in DEFAULT_UNIVERSE if r["market"] in ("crypto", "cryptoweek")]
        install_universe(conn, crypto)
        old = [dict(market=m, symbol=s, name=s, source="nasdaq", source_symbol=s, quote="USD", position=p, added_on="2026-09-30")
               for m in GAMES for p, s in enumerate(("NVDA", "AAPL", "MSFT", "ASML"))]
        install_universe(conn, old)
        for r in conn.execute("SELECT id, market FROM instruments").fetchall():
            conn.execute("INSERT INTO bars VALUES (?, '2026-10-30', 1, 1, 1, 1, 1, 'x')", (r["id"],))
            conn.execute("INSERT INTO predictions (model, instrument_id, for_date, made_at) VALUES ('m', ?, '2026-10-30', 'x')", (r["id"],))
            conn.execute("INSERT INTO results (model, instrument_id, for_date, settled_at) VALUES ('m', ?, '2026-10-30', 'x')", (r["id"],))
            conn.execute("INSERT INTO game_days VALUES (?, '2026-10-30', 0, 'X', 0, 0, '')", (r["market"],))
        conn.commit()
        conn.close()
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{os.path.join(root, 'db.sqlite3')}"
        for name in ("shares_Markov", "sharesweek-rl_position", "crypto_Markov", "cryptoweek-rl_position"):
            optuna.create_study(study_name=name, storage=storage)

        lines = []
        assert run(root, log=lines.append) == 0
        plan = plan_of(root)
        assert sorted(plan["files"]) == sorted(gone), sorted(set(plan["files"]) ^ set(gone))
        assert plan["store_rows"] == {"results": 8, "predictions": 8, "bars": 8, "intraday_bars": 0, "game_days": 8,
                                      "ticket_published": 0, "model_performance": 0, "instruments": 8}, plan["store_rows"]
        assert plan["studies"] == ["shares_Markov", "sharesweek-rl_position"], plan["studies"]
        assert sorted(plan["summary_keys"]) == sorted([["data/database/modelPerformance.json", "games/shares"],
                                                        ["data/database/modelPerformance.json", "games/sharesweek"],
                                                        ["data/database/lagPeakHistory.json", "shares"],
                                                        ["data/database/lagPeakHistory.json", "sharesweek"]]), plan["summary_keys"]
        assert all(os.path.exists(os.path.join(root, rel)) for rel in gone), "a dry run touched a file"

        assert run(root, apply=True, log=lines.append) == 1 and "--backup" in lines[-1], lines[-1]
        assert run(root, backup=os.path.join(root, "bk"), apply=True, log=lines.append) == 1 and "inside" in lines[-1], lines[-1]
        with open(os.path.join(root, "process.lock"), "w") as f:
            f.write(str(os.getpid()))
        backup = os.path.join(tmp, "backup")
        assert run(root, backup=backup, apply=True, fetch=False, log=lines.append) == 1 and "held by" in lines[-1], lines[-1]
        assert not os.path.exists(backup)
        with open(os.path.join(root, "process.lock"), "w") as f:
            f.write("999999999")   # a stale lock: removed and taken
        before = {rel: open(os.path.join(root, rel)).read() for rel in gone}

        assert run(root, backup=backup, apply=True, fetch=False, log=lines.append) == 0, lines[-8:]
        assert not os.path.exists(os.path.join(root, "process.lock"))
        assert not any(os.path.exists(os.path.join(root, rel)) for rel in gone)
        assert all(os.path.exists(os.path.join(root, rel)) for rel in kept)
        with open(os.path.join(root, "data/database/modelPerformance.json")) as f:
            text = f.read()
        assert json.loads(text) == {"generatedAt": "x", "games": {"crypto": {"a": 1}}} and text.startswith('{\n  "') and text.endswith("}\n")
        with open(os.path.join(root, "data/database/lagPeakHistory.json")) as f:
            assert json.load(f) == {"crypto": {}}
        conn = connect(os.path.join(root, "data", "markets.sqlite3"))
        assert installed_shares(conn) == wanted_shares()
        weeks = conn.execute("SELECT symbol, added_on FROM instruments WHERE market = 'sharesweek' ORDER BY position").fetchall()
        assert [w[0] for w in weeks] == [s for _, s, _, _ in wanted_shares()] and all(w[1] for w in weeks)
        assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 10   # crypto's own, untouched
        assert conn.execute("SELECT COUNT(*) FROM game_days WHERE market LIKE 'shares%'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM game_days").fetchone()[0] == 10
        conn.close()
        assert studies_of(os.path.join(root, "db.sqlite3")) == []
        assert {s.study_name for s in optuna.get_all_study_summaries(storage)} == {"crypto_Markov", "cryptoweek-rl_position"}

        # the backup holds every removed file as it was, and the databases before the edit
        with tarfile.open(os.path.join(backup, "files.tar.gz")) as tar:
            names = {m.name for m in tar.getmembers()}
            assert names == set(gone) | {"data/database/modelPerformance.json", "data/database/lagPeakHistory.json"}, names
            assert all(tar.extractfile(rel).read().decode() == before[rel] for rel in gone)
            assert len(json.load(tar.extractfile("data/database/modelPerformance.json"))["games"]) == 3
        saved = sqlite3.connect(os.path.join(backup, "markets.sqlite3"))
        assert saved.execute("SELECT COUNT(*) FROM instruments WHERE source = 'nasdaq'").fetchone()[0] == 8
        saved.close()
        assert len(studies_of(os.path.join(backup, "db.sqlite3"))) == 2
        with open(os.path.join(backup, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["archive_files"] == len(gone) + 2 and manifest["archive_sha256"] == sha256(os.path.join(backup, "files.tar.gz"))

        # a second run is refused: it would delete the Euronext record
        assert run(root, backup=os.path.join(tmp, "backup2"), apply=True, log=lines.append) == 1 and "switch was made" in lines[-1], lines[-1]
        assert not os.path.exists(os.path.join(tmp, "backup2"))
        assert NAMED.search("/model_shares.keras") and NAMED.search("/sharesweek-w10.json") and not NAMED.search("/sharesX.json")
        print("SwitchSharesEuronext self-check OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
