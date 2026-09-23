"""Council orchestration. This is the importable entry point.

    from council import load_config, run_council

    config = load_config(Path("config.json"))
    result = run_council("Which city is the capital of Belgium?", config)
    print(result["head"]["answer"])

`run_council` returns the same structure that the CLI writes to answers.json:

    {
      "question": str,
      "members": [ {name, lab, seconds, ok, answer|error, cached?}, ... ],
      "head":    {name, lab, seconds, ok, answer|error}        # may be absent
    }

It does not raise on model failure; inspect the `ok` flags. It raises only on
programming or configuration errors.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from council import cache as cache_mod
from council import answer as answer_mod
from council import client, head as head_mod, health, prompts, sampling

log = logging.getLogger(__name__)

# Bump when a default prompt changes, so stale answers are not reused. Prompts
# set in config are folded into the cache key automatically.
PROMPT_VERSION = "v2"


class ConfigError(ValueError):
    """The configuration is unusable."""


class EndpointsUnavailable(RuntimeError):
    """One or more endpoints failed preflight."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def load_config(path: Path | str) -> dict:
    """Read a config file. Raises ConfigError if it is unusable."""
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    if not config.get("members"):
        raise ConfigError(f"{path} defines no members")
    return config


def enabled_members(config: dict) -> list[dict]:
    return [m for m in config["members"] if m.get("enabled", True)]


def order_for_head(members: list[dict], results: list[dict],
                   config: dict) -> list[dict]:
    """Order the answers shown to the head.

    Heads show position bias: in testing, a 3B head adopted the last member's
    answer and discarded two correct earlier ones. Shuffling per run does not
    remove the bias but stops it favouring the same member every night, so a
    systematic error does not always land on the same model. Set
    `shuffle_members` to false for a fully deterministic run.
    """
    if not config.get("shuffle_members", True):
        return results
    seed = config.get("shuffle_seed")
    rng = random.Random(seed) if seed is not None else random.Random()
    shuffled = list(results)
    rng.shuffle(shuffled)
    return shuffled


ASK_SEQUENTIAL = "sequential"
ASK_PARALLEL = "parallel"


def ask_mode(config: dict) -> str:
    """How the members are asked: one after another (default) or all at once.

    Every member is its own llama-server, so asking them at the same time
    does not queue inside one server - it shares the model box's CPU and
    memory between them instead. Whether that is a win depends on the box:
    four small models on a machine with headroom finish in the time of the
    slowest one; four models that each need most of the RAM thrash. That is
    an operator's call, hence a config key rather than a default.

        "ask_members": "sequential" | "parallel"

    Anything else is treated as sequential and logged once, so a typo cannot
    silently change how a run behaves.
    """
    value = str(config.get("ask_members", ASK_SEQUENTIAL) or ASK_SEQUENTIAL).strip().lower()
    if value not in (ASK_SEQUENTIAL, ASK_PARALLEL):
        log.warning("ask_members=%r is not %r or %r - asking sequentially", value, ASK_SEQUENTIAL, ASK_PARALLEL)
        return ASK_SEQUENTIAL
    return value


def head_config(config: dict) -> dict | None:
    candidate = config.get("head")
    return candidate if candidate and candidate.get("enabled", True) else None


def make_cache(config: dict, use_cache: bool = True) -> cache_mod.Cache:
    directory = config.get("cache_dir")
    return cache_mod.Cache(
        Path(directory) if use_cache and directory else None, PROMPT_VERSION
    )


def check_endpoints(config: dict, endpoints: list[dict]) -> list[str]:
    """Check every endpoint. Returns a list of problems; empty means all good."""
    problems = []
    timeout = config.get("health_timeout_s", 5)
    for endpoint in endpoints:
        try:
            health.check(endpoint["base_url"], timeout)
            log.info("  %-18s ok   %s", endpoint["name"], endpoint["base_url"])
        except health.EndpointUnavailable as exc:
            log.error("  %-18s DOWN %s", endpoint["name"], exc)
            problems.append(f"{endpoint['name']}: {exc}")
    return problems


def ask_member(config: dict, member: dict, question: str,
               cache: cache_mod.Cache, retries: int,
               context: str | None = None) -> dict:
    """Ask one member, or return its cached answer. Never raises."""
    system = prompts.member_system_prompt(member, config)
    user = prompts.with_context(question, context)

    cached = cache.get(user, member, system)
    if cached:
        return cached

    started = time.monotonic()
    log.info("asking %s", member["name"])
    try:
        answer = client.ask(
            member["base_url"],
            user,
            timeout_s=config["request_timeout_s"],
            system_prompt=system,
            model=member.get("model"),
            retries=retries,
            retry_delay_s=config.get("retry_delay_s", 5.0),
            **sampling.for_member(member),
        )
        outcome = {"ok": True, "answer": answer}
    except client.CompletionError as exc:
        log.error("%s failed: %s", member["name"], exc)
        outcome = {"ok": False, "error": str(exc)}

    result = {
        "name": member["name"],
        "lab": member.get("lab"),
        "seconds": round(time.monotonic() - started, 1),
        **outcome,
    }
    cache.put(user, member, result, system)
    return result


def run_head(config: dict, head_cfg: dict, question: str,
             members: list[dict], retries: int,
             context: str | None = None) -> dict:
    """Run the aggregation step. Never raises."""
    started = time.monotonic()
    try:
        answer = head_mod.synthesise(
            head_cfg, question, members, config["request_timeout_s"],
            config=config, context=context, retries=retries,
        )
        outcome = {"ok": True, "answer": answer,
                   "value": answer_mod.extract(answer)}
    except client.CompletionError as exc:
        log.error("head failed: %s", exc)
        outcome = {"ok": False, "error": str(exc)}

    return {
        "name": head_cfg["name"],
        "lab": head_cfg.get("lab"),
        "seconds": round(time.monotonic() - started, 1),
        **outcome,
    }


def run_council(question: str, config: dict, *,
                context: str | None = None,
                use_head: bool = True,
                use_cache: bool = True,
                retries: int | None = None,
                preflight: bool = True,
                on_event=None) -> dict:
    """Put one question to the council and return the collected result.

    `context` is per-question data (prior answers, working notes, constraints)
    prepended to the question for every member and for the head. It is shared
    verbatim, so it does not make members correlated the way per-member framing
    would - but note that every member does see it, so a wrong premise in the
    context can mislead all of them at once.

    `on_event` is called as the run proceeds, for progress display. It gets
    (kind, payload) where kind is one of "start", "member_start", "member_done",
    "head_start", "head_done". A "member_done" payload carries the member's
    answer (or error) so a page can show what each member said the moment it
    said it, and "head_done" carries the head's. Calls are serialised with a
    lock, because in parallel mode (see ask_mode) they arrive from several
    threads. Anything it raises is swallowed: a progress display must never
    be able to fail a run.

    Raises ConfigError if no members are enabled, and EndpointsUnavailable if
    preflight is on and any endpoint is down. Model failures are reported in
    the returned dict rather than raised.
    """
    emit_lock = threading.Lock()

    def emit(kind: str, payload: dict) -> None:
        if on_event is None:
            return
        with emit_lock:
            try:
                on_event(kind, payload)
            except Exception:                            # noqa: BLE001
                log.debug("progress callback failed", exc_info=True)
    members = enabled_members(config)
    if not members:
        raise ConfigError("no enabled members in config")

    head_cfg = head_config(config) if use_head else None
    cache = make_cache(config, use_cache)
    if retries is None:
        retries = config.get("retries", 2)

    if preflight:
        endpoints = members + ([head_cfg] if head_cfg else [])
        log.info("checking %d endpoint(s)", len(endpoints))
        problems = check_endpoints(config, endpoints)
        if problems:
            raise EndpointsUnavailable(problems)

    emit("start", {
        "members": [m["name"] for m in members],
        "head": head_cfg["name"] if head_cfg else None,
    })

    def ask_one(member: dict) -> dict:
        emit("member_start", {"name": member["name"]})
        result = ask_member(config, member, question, cache, retries, context)
        emit("member_done", {
            "name": member["name"],
            "ok": result["ok"],
            "cached": bool(result.get("cached")),
            "seconds": result["seconds"],
            # What was said, not just that something was: the page shows a
            # member's answer at its seat as soon as it lands.
            "answer": result.get("answer"),
            "error": result.get("error"),
        })
        return result

    if ask_mode(config) == ASK_PARALLEL and len(members) > 1:
        # One thread per member. The results are put back in config order so
        # the output - and the cache, and the head's shuffled view of it -
        # is the same shape as a sequential run.
        with ThreadPoolExecutor(max_workers=len(members)) as pool:
            results = list(pool.map(ask_one, members))
    else:
        results = [ask_one(member) for member in members]

    output = {"question": question, "members": results}
    if context:
        output["context"] = context

    if head_cfg:
        if any(r["ok"] for r in results):
            emit("head_start", {"name": head_cfg["name"]})
            output["head"] = run_head(
                config, head_cfg, question,
                order_for_head(members, results, config),
                retries, context,
            )
            emit("head_done", {
                "name": head_cfg["name"],
                "ok": output["head"]["ok"],
                "seconds": output["head"]["seconds"],
                "answer": output["head"].get("answer"),
                "value": output["head"].get("value"),
                "error": output["head"].get("error"),
            })
        else:
            log.error("skipping head: no member answered")

    return output


def succeeded(result: dict) -> bool:
    """True when every member answered and the head, if run, answered too."""
    if not all(m["ok"] for m in result["members"]):
        return False
    head_result = result.get("head")
    return not (head_result and not head_result["ok"])