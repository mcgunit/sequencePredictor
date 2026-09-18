"""Pull the machine-readable result out of the head's reply.

The maths head preset ends its reply with a line of the form

    ANSWER: 42

so a caller reads `result["head"]["value"]` instead of running a regex over
prose. When no marker is present, `value` is None and the caller should fall
back to the full text (or treat the run as failed - its choice).
"""

from __future__ import annotations

import logging

from council.prompts import ANSWER_MARKER

log = logging.getLogger(__name__)

UNDETERMINED = "UNDETERMINED"


def extract(text: str) -> str | None:
    """Return the value after the last ANSWER: marker, or None if absent.

    The last marker wins: a model that restates the format while explaining
    itself would otherwise have its example picked up instead of its answer.
    """
    if not text:
        return None

    found = None
    for line in text.splitlines():
        stripped = line.strip().lstrip("*# ").rstrip("*")
        if stripped.upper().startswith(ANSWER_MARKER):
            value = stripped[len(ANSWER_MARKER):].strip()
            if value:
                found = value

    if found is None:
        log.warning("head reply has no %s line", ANSWER_MARKER)
    return found


def is_undetermined(value: str | None) -> bool:
    return value is not None and value.strip().upper() == UNDETERMINED