"""Aggregation: put the members' answers to the head model."""

from __future__ import annotations

import logging

from council import client, prompts, sampling

log = logging.getLogger(__name__)


def synthesise(head: dict, question: str, members: list[dict],
               timeout_s: int, config: dict | None = None,
               context: str | None = None, retries: int = 0,
               on_chunk=None) -> str:
    """Ask the head to aggregate. Raises CompletionError on failure."""
    answered = [m for m in members if m.get("ok")]
    if not answered:
        raise client.CompletionError("no member answers to aggregate")

    params = sampling.for_head(head)
    log.info("head aggregating %d answer(s) (temp=%s, seed=%s)",
             len(answered), params["temperature"], params["seed"])
    return client.ask(
        head["base_url"],
        prompts.head_prompt(question, answered, context),
        timeout_s=timeout_s,
        system_prompt=prompts.head_system_prompt(head, config or {}),
        model=head.get("model"),
        retries=retries,
        on_chunk=on_chunk,
        **params,
    )