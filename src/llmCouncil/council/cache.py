"""Per-member answer cache, so a retried run does not redo completed work.

A cache entry is keyed by the question and the member's identity. If either
changes, the key changes and the member is asked again.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _key(question: str, member: dict, prompt_version: str,
         system_prompt: str = "") -> str:
    """Stable key for one member's answer to one question.

    The resolved system prompt is part of the key, so changing a role or a
    prompt in config invalidates affected entries without touching the rest.
    Sampling settings are included too: a different seed or temperature is a
    different answer.
    """
    params = member.get("seed"), member.get("temperature")
    material = "\x1f".join([
        question,
        member["name"],
        member.get("base_url", ""),
        member.get("model") or "",
        system_prompt,
        repr(params),
        prompt_version,
    ])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{member['name']}-{digest}"


class Cache:
    """Answers stored one JSON file per member per question."""

    def __init__(self, directory: Path | None, prompt_version: str = "v1"):
        self.directory = directory
        self.prompt_version = prompt_version
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _path(self, question: str, member: dict, system_prompt: str) -> Path:
        key = _key(question, member, self.prompt_version, system_prompt)
        return self.directory / f"{key}.json"

    def get(self, question: str, member: dict,
            system_prompt: str = "") -> dict | None:
        """Return a cached successful answer, or None."""
        if not self.enabled:
            return None
        path = self._path(question, member, system_prompt)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("ignoring unreadable cache entry %s: %s", path, exc)
            return None
        if not entry.get("ok"):
            return None
        log.info("%s answered from cache", member["name"])
        return entry

    def put(self, question: str, member: dict, result: dict,
            system_prompt: str = "") -> None:
        """Store a successful result. Failures are not cached."""
        if not self.enabled or not result.get("ok"):
            return
        path = self._path(question, member, system_prompt)
        try:
            path.write_text(
                json.dumps({**result, "cached": True}, indent=2,
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("could not write cache entry %s: %s", path, exc)

    def clear(self) -> int:
        """Remove every cache entry. Returns how many were removed."""
        if not self.enabled:
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                log.warning("could not remove %s: %s", path, exc)
        return removed