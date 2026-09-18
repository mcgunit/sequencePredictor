"""Example of a parent process driving the council as a library.

Models a nightly job: one question, with context assembled from earlier runs,
whose final answer is then parsed by the caller.
"""

import logging
import sys

from council import (
    ConfigError,
    EndpointsUnavailable,
    is_undetermined,
    load_config,
    run_council,
    succeeded,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

QUESTION = "Solve for x: 3x + 7 = 25. Give the value of x."

# In the real job this is assembled from stored history.
CONTEXT = """\
Earlier questions in this series and their accepted answers:
  2x + 3 = 11  ->  x = 4
  5x - 2 = 23  ->  x = 5
"""


def main() -> int:
    try:
        config = load_config("config.json")
    except ConfigError as exc:
        print(f"bad config: {exc}", file=sys.stderr)
        return 2

    try:
        result = run_council(QUESTION, config, context=CONTEXT)
    except EndpointsUnavailable as exc:
        print(f"council unavailable: {exc}", file=sys.stderr)
        return 1

    for member in result["members"]:
        if member["ok"]:
            print(f"{member['name']}: {member['answer'][:80]}...")
        else:
            print(f"{member['name']}: FAILED - {member['error']}")

    head = result.get("head")
    if head and head["ok"]:
        print(f"\nfinal answer:\n{head['answer']}")
        # The caller parses this string; see README on making that reliable.

    return 0 if succeeded(result) else 1


if __name__ == "__main__":
    sys.exit(main())