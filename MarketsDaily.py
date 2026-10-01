#!/usr/bin/env python3
"""
The market games' daily refresh by hand (README roadmap item 4, phase M2):
bring the bars up to date, re-cut the return bins and write the yearly game
files Predictor.py reads, plus the returns file next to them that the GARCH
and Regime HMM rows model (phase M3) - the same call Predictor.py makes for a
market game before predicting it (src/MarketGame.daily_refresh).

    python3 MarketsDaily.py                      # both markets: fetch + cut + write
    python3 MarketsDaily.py --market crypto
    python3 MarketsDaily.py --no-fetch           # re-cut from the stored bars only
    python3 MarketsDaily.py --no-fetch --settle  # ... and settle every stored day, rewrite data/markets/<market>.json
    python3 MarketsDaily.py --db /elsewhere/markets.sqlite3 --root /elsewhere/checkout

--settle runs the settlement and the page export Predictor.py runs after a
day (src/MarketSettle.settle_and_export) without predicting anything: the
way to refresh the Crypto and Shares pages after a deploy that changed what
the page record carries, instead of waiting for the next 09:00 run.

Takes process.lock: it rewrites data/trainingData/<market>/ files that a
running Predictor.py reads, so it waits its turn like every pipeline job.
Exit code 2 when a source failed (the game is still written from what is
stored), 1 when a market could not be cut at all.
"""

import argparse
import os
import sys

from HyperoptStatistics import is_running, create_lock, remove_lock
from src.MarketData import DEFAULT_DB, MARKETS
from src.MarketGame import daily_refresh
from src.MarketSettle import settle_and_export


def main():
    parser = argparse.ArgumentParser(prog="Markets daily", description="Fetch the bars and rewrite the market games' history")
    parser.add_argument("--market", choices=MARKETS, default=None, help="One market only (default both)")
    parser.add_argument("--no-fetch", action="store_true", help="Do not contact the sources; cut from the stored bars")
    parser.add_argument("--settle", action="store_true", help="Also settle every stored day and rewrite data/markets/<market>.json and the results CSVs")
    parser.add_argument("--db", default=None, help=f"SQLite store (default <root>/{DEFAULT_DB})")
    parser.add_argument("--root", default=os.getcwd(), help="Checkout whose data/trainingData/<market>/ is written (default: the working directory)")
    args = parser.parse_args()

    if is_running():
        print("Another instance is already running. Exiting.")
        return 1
    if not create_lock():
        print("Failed to create lock file. Exiting.")
        return 1

    failed_sources = 0
    empty = 0
    try:
        for market in ([args.market] if args.market else list(MARKETS)):
            try:
                summaries, game_days = daily_refresh(args.root, market, fetch=not args.no_fetch, db_path=args.db)
            except Exception as exc:
                print(f"{market}: refresh failed - {exc}")
                empty += 1
                continue
            failed_sources += sum(1 for s in summaries if s.get("error"))
            if not game_days:
                empty += 1
                continue
            if args.settle:
                try:
                    settle_and_export(args.root, market, db_path=args.db)
                except Exception as exc:
                    print(f"{market}: settlement/export failed - {exc}")
                    empty += 1
    finally:
        remove_lock()
    return 1 if empty else (2 if failed_sources else 0)


if __name__ == "__main__":
    sys.exit(main())
