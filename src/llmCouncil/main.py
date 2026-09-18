"""Command-line front end for the council.

The orchestration itself lives in council/orchestrator.py, so a parent process
can import and call run_council() directly instead of shelling out to this.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from council import chat, orchestrator
from council.orchestrator import (
    ConfigError,
    EndpointsUnavailable,
    check_endpoints,
    enabled_members,
    head_config,
    load_config,
    make_cache,
    run_council,
    succeeded,
)

log = logging.getLogger("council")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM council.")
    parser.add_argument("question", nargs="?",
                        help="the question to put to the council")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--out", type=Path, default=Path("answers.json"))
    parser.add_argument("--chat", action="store_true",
                        help="interactive terminal chat with the council")
    parser.add_argument("--check", action="store_true",
                        help="check endpoints and exit without asking anything")
    parser.add_argument("--no-head", action="store_true",
                        help="collect member answers but skip aggregation")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached answers and do not write new ones")
    parser.add_argument("--clear-cache", action="store_true",
                        help="delete every cached answer and exit")
    parser.add_argument("--context", default=None,
                        help="per-question context sent to every member")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="read the context from a file instead")
    parser.add_argument("--retries", type=int, default=None,
                        help="extra attempts per request (default from config)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def report(result: dict) -> None:
    for member in result["members"]:
        note = " (cached)" if member.get("cached") else ""
        log.info("  %-18s %-7s %6.1fs%s", member["name"],
                 "ok" if member["ok"] else "FAILED", member["seconds"], note)
    head_result = result.get("head")
    if head_result:
        log.info("  %-18s %-7s %6.1fs", head_result["name"],
                 "ok" if head_result["ok"] else "FAILED",
                 head_result["seconds"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    if args.clear_cache:
        removed = make_cache(config, use_cache=True).clear()
        log.info("removed %d cache entrie(s)", removed)
        return 0

    context = args.context
    if args.context_file:
        try:
            context = args.context_file.read_text(encoding="utf-8")
        except OSError as exc:
            log.error("could not read %s: %s", args.context_file, exc)
            return 2

    if args.chat:
        return chat.run(
            config,
            context=context,
            use_head=not args.no_head,
            use_cache=not args.no_cache,
            retries=args.retries,
        )

    if args.check:
        members = enabled_members(config)
        head_cfg = None if args.no_head else head_config(config)
        endpoints = members + ([head_cfg] if head_cfg else [])
        log.info("checking %d endpoint(s)", len(endpoints))
        return 1 if check_endpoints(config, endpoints) else 0

    if not args.question:
        log.error("a question is required unless --check or --chat is given")
        return 2

    try:
        result = run_council(
            args.question, config,
            context=context,
            use_head=not args.no_head,
            use_cache=not args.no_cache,
            retries=args.retries,
        )
    except EndpointsUnavailable as exc:
        log.error("aborting: %d endpoint(s) unavailable", len(exc.problems))
        return 1
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    answered = sum(1 for m in result["members"] if m["ok"])
    log.info("%d/%d member(s) succeeded, written to %s",
             answered, len(result["members"]), args.out)
    report(result)

    if answered != len(result["members"]):
        log.warning("rerun to retry failed member(s); "
                    "successful answers are cached")
    return 0 if succeeded(result) else 1


if __name__ == "__main__":
    sys.exit(main())