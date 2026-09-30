#!/usr/bin/env python3
"""
Daily bars for the markets track (README roadmap item 4, phase M1): keeps
data/markets.sqlite3 up to date from the keyless sources and reports on it.

    python3 FetchMarkets.py --init                 # schema + the frozen universe (idempotent)
    python3 FetchMarkets.py --update               # incremental fetch for every active instrument
    python3 FetchMarkets.py --update --market crypto
    python3 FetchMarkets.py --check                # gaps, bad prices, staleness per instrument
    python3 FetchMarkets.py --status
    python3 FetchMarkets.py --set-source shares:ASML yahoo   # another source, same listing, bars reloaded
    python3 FetchMarkets.py --refetch-all --market shares    # rewrite a history from scratch
    python3 FetchMarkets.py --db /elsewhere/markets.sqlite3 ...

This script does NOT take process.lock: it is network-bound, touches only
the markets database, and must be able to run while a tuner holds the lock.
It is idempotent - running it twice fetches nothing new the second time
beyond the three-day revision window (Binance) or rewrites the same bars
(Nasdaq and Yahoo, whose adjusted history is always read whole) - and never
raises on a source that does not answer: the failure is printed and the
exit code says so.

Exit code: 0 when every requested instrument updated, 2 when at least one
source failed (the others are still stored), 1 on a usage error.
"""

import argparse
import sys

from src.MarketData import DEFAULT_DB, MARKETS, check_bars, connect, install_universe, instruments, set_source, status, update_all


def main():
    parser = argparse.ArgumentParser(prog="Fetch markets", description="Daily bars for the markets track")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite store (default {DEFAULT_DB})")
    parser.add_argument("--init", action="store_true", help="Create the schema and install the frozen universe")
    parser.add_argument("--update", action="store_true", help="Fetch the bars each active instrument is missing")
    parser.add_argument("--market", choices=MARKETS, default=None, help="Restrict --update/--check to one market")
    parser.add_argument("--check", action="store_true", help="Report gaps, bad prices and staleness")
    parser.add_argument("--status", action="store_true", help="One line per instrument")
    parser.add_argument("--set-source", nargs=2, metavar=("MARKET:SYMBOL", "SOURCE[:SOURCE_SYMBOL]"), default=None,
                        help="Point an instrument at another source, e.g. shares:ASML yahoo - identity unchanged, the source symbol "
                             "must name the same listing (never ASML.AS for the USD listing); the stored bars are cleared and reloaded")
    parser.add_argument("--refetch-all", action="store_true", help="With --update: read every instrument's whole history and rewrite it")
    args = parser.parse_args()
    if not (args.init or args.update or args.check or args.status or args.set_source):
        parser.print_help()
        return 1

    conn = connect(args.db)
    failed = 0
    if args.set_source:
        target, source = args.set_source
        try:
            market, symbol = target.split(":", 1)
            source_name, _, source_symbol = source.partition(":")
            cleared = set_source(conn, market, symbol, source_name, source_symbol or None)
        except ValueError as exc:
            print(f"--set-source: {exc}")
            return 1
        print(f"{target} now fetched from {source}" + (f"; {cleared} stored bar(s) cleared, run --update to reload" if cleared else ""))
    if args.init:
        added, conflicts = install_universe(conn)
        print(f"universe: {len(added)} instrument(s) added" + (f" ({', '.join(added)})" if added else " (already installed)"))
        for conflict in conflicts:
            print(f"universe CONFLICT: {conflict}")
    if args.update:
        if not instruments(conn, args.market):
            print("no instruments - run with --init first")
            return 1
        summaries = update_all(conn, market=args.market, full=args.refetch_all)
        failed = sum(1 for s in summaries if s.get("error"))
        print(f"update: {len(summaries) - failed} of {len(summaries)} instrument(s) updated" + (f", {failed} failed" if failed else ""))
    if args.check:
        for instrument in instruments(conn, args.market):
            report = check_bars(conn, instrument)
            gaps = "; ".join(f"{g['from']}..{g['to']} ({g['days']} day(s))" for g in report["gaps"][:5])
            more = f" and {len(report['gaps']) - 5} more" if len(report["gaps"]) > 5 else ""
            print(f"{instrument['symbol']:5s}: {report['rows']} bars {report['first']} .. {report['last']}, "
                  f"stale {report['stale_days']} day(s), {report['bad_prices']} bad price(s), {report['jumps']} jump(s), "
                  f"gaps: {gaps or 'none'}{more}")
    if args.status:
        for line in status(conn):
            print(line)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
