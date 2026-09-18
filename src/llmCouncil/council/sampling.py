"""Per-endpoint sampling settings.

Each member and the head may set `temperature`, `max_tokens` and `seed` in
config. Anything unset falls back to the role's default.

`seed` is null by default, which means the request omits it and the server
picks one. Set it to reproduce a run: useful when comparing prompt changes,
since otherwise two runs on identical input can differ enough to hide the
effect you are measuring. Reproducibility holds only while the prompt, model,
quant, server flags and backend stay the same.
"""

from __future__ import annotations

MEMBER_DEFAULTS = {"temperature": 0.7, "max_tokens": 2048, "seed": None}
HEAD_DEFAULTS = {"temperature": 0.3, "max_tokens": 4096, "seed": None}


def settings(endpoint: dict, defaults: dict) -> dict:
    """Resolve sampling settings for one endpoint against its role defaults."""
    return {
        key: endpoint.get(key, default) if endpoint.get(key) is not None
        else default
        for key, default in defaults.items()
    }


def for_member(member: dict) -> dict:
    return settings(member, MEMBER_DEFAULTS)


def for_head(head: dict) -> dict:
    return settings(head, HEAD_DEFAULTS)