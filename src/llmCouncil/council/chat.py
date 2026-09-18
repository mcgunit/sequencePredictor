"""A minimal terminal chat loop for poking at the council by hand.

Each question is an independent council run: members are stateless and never
see prior turns. This is deliberate - it mirrors how the nightly job behaves,
so what you see here is what the batch job would produce. It is a debugging
tool, not a conversation.
"""

from __future__ import annotations

import logging
import textwrap
import time

from council import orchestrator
from council.orchestrator import EndpointsUnavailable, run_council

log = logging.getLogger(__name__)

BANNER = """\
council chat - each question is an independent run, no conversation history

  /help      show this
  /members   show the last run's member answers in full
  /head      show the last run's final answer in full
  /json      show the last run as raw JSON
  /config    show the current endpoints and sampling settings
  /context   show the current context
  /setcontext <text>   set the context sent with every question
  /nocontext clear the context
  /nohead    toggle the aggregation step
  /nocache   toggle use of the answer cache
  /clear     drop every cached answer
  /exit      leave (Ctrl-D or Ctrl-C also work)
"""

RULE = "-" * 72


def _wrap(text: str, indent: str = "  ") -> str:
    paragraphs = text.strip().split("\n\n")
    return "\n\n".join(
        textwrap.fill(p, width=78, initial_indent=indent,
                      subsequent_indent=indent)
        for p in paragraphs
    )


def _show_members(result: dict | None, full: bool = True) -> None:
    if not result:
        print("nothing to show yet")
        return
    for member in result["members"]:
        mark = "cached" if member.get("cached") else f"{member['seconds']}s"
        print(f"\n{member['name']} ({member.get('lab', '?')}, {mark})")
        if member["ok"]:
            print(_wrap(member["answer"] if full
                        else member["answer"][:200] + "..."))
        else:
            print(f"  FAILED: {member['error']}")


def _show_head(result: dict | None) -> None:
    if not result:
        print("nothing to show yet")
        return
    head = result.get("head")
    if not head:
        print("no head in the last run")
    elif head["ok"]:
        print(f"\n{head['name']} ({head['seconds']}s)")
        print(_wrap(head["answer"]))
        if head.get("value"):
            print(f"\n  parsed value: {head['value']}")
        elif head.get("value") is None:
            print("\n  (no ANSWER: line found)")
    else:
        print(f"head FAILED: {head['error']}")


def _show_config(config: dict, use_head: bool, use_cache: bool) -> None:
    from council import sampling
    print(f"\ncache: {'on' if use_cache else 'off'}   "
          f"head: {'on' if use_head else 'off'}")
    for member in orchestrator.enabled_members(config):
        params = sampling.for_member(member)
        print(f"  {member['name']:<18} {member['base_url']:<28} "
              f"temp={params['temperature']} seed={params['seed']}")
    head = orchestrator.head_config(config)
    if head:
        params = sampling.for_head(head)
        print(f"  {head['name']:<18} {head['base_url']:<28} "
              f"temp={params['temperature']} seed={params['seed']}  [head]")


def run(config: dict, *, context: str | None = None,
        use_head: bool = True, use_cache: bool = True,
        retries: int | None = None) -> int:
    """Run the chat loop until the user leaves. Returns an exit code."""
    import json as json_mod

    print(BANNER)
    _show_config(config, use_head, use_cache)
    last: dict | None = None
    checked = False

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        if line.startswith("/"):
            command = line[1:].lower()
            if command in ("exit", "quit"):
                return 0
            if command == "help":
                print(BANNER)
            elif command == "members":
                _show_members(last)
            elif command == "head":
                _show_head(last)
            elif command == "json":
                print(json_mod.dumps(last, indent=2, ensure_ascii=False)
                      if last else "nothing to show yet")
            elif command == "config":
                _show_config(config, use_head, use_cache)
            elif command == "context":
                print(context if context else "no context set")
            elif command.startswith("setcontext "):
                context = line[len("/setcontext "):].strip()
                print(f"context set ({len(context)} chars)")
            elif command == "nocontext":
                context = None
                print("context cleared")
            elif command == "nohead":
                use_head = not use_head
                print(f"head {'on' if use_head else 'off'}")
            elif command == "nocache":
                use_cache = not use_cache
                checked = False
                print(f"cache {'on' if use_cache else 'off'}")
            elif command == "clear":
                removed = orchestrator.make_cache(config, True).clear()
                print(f"removed {removed} cache entrie(s)")
            else:
                print(f"unknown command: /{command}  (try /help)")
            continue

        started = time.monotonic()
        try:
            last = run_council(
                line, config,
                context=context,
                use_head=use_head,
                use_cache=use_cache,
                retries=retries,
                preflight=not checked,
            )
            checked = True
        except EndpointsUnavailable as exc:
            print(f"endpoints unavailable: {exc}")
            checked = False
            continue

        print(RULE)
        _show_members(last, full=False)
        print(f"\n{RULE}")
        _show_head(last)
        print(f"\n({time.monotonic() - started:.1f}s total - "
              f"/members, /head or /json for more)")