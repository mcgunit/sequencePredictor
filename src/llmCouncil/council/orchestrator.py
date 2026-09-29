"""Council orchestration. This is the importable entry point.

    from council import load_config, run_council

    config = load_config(Path("config.json"))
    result = run_council("Which city is the capital of Belgium?", config)
    print(result["head"]["answer"])

`run_council` returns the same structure that the CLI writes to answers.json:

    {
      "question": str,
      "members": [ {name, lab, seconds, ok, answer|error, cached?}, ... ],
      "head":    {name, lab, seconds, ok, answer|error, order}  # may be absent
    }

`head.order` lists the names of the answered members in the order the head
read them ("Member 1" in the head's text is order[0]): the head sees
anonymised, shuffled answers, so without it a reader could not tell which
member the head means.

It does not raise on model failure; inspect the `ok` flags. It raises only on
programming or configuration errors.
"""

from __future__ import annotations

import difflib
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


def stream_answers(config: dict) -> bool:
    """Whether member and head replies are streamed token by token so a page
    can show them growing (`"stream_answers": true`, the default). Turn it
    off for a server that does not speak server-sent events."""
    return bool(config.get("stream_answers", True))


# How often a growing reply is reported. Every emit copies the whole progress
# state for the page, so per-token would be wasteful; a quarter second is
# faster than anyone reads.
CHUNK_INTERVAL_S = 0.25


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
               context: str | None = None, on_chunk=None) -> dict:
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
            on_chunk=on_chunk,
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


COPY_RATIO = 0.9   # SequenceMatcher ratio above which a head reply is a member's reply
COPY_SHARE = 0.8   # a text contained in the other counts as a copy only when it IS most of it


def copied_member(answer: str, members: list[dict]) -> int | None:
    """
    The 1-based number (as the head saw it: "Member N") of the member whose
    answer the head's reply is a copy of, or None. Whitespace and case are
    ignored. A copy is: the same text; one text contained in the other and
    making up at least COPY_SHARE of it (a preamble or sign-off around a
    member's answer is still a copy; a member's sentence quoted inside a
    longer synthesis is not); or a SequenceMatcher ratio of COPY_RATIO or
    better - autojunk off, since with it on difflib scores a near-verbatim
    copy of a long answer close to zero. Short answers ("42", "yes") are
    exempt: agreeing in two words is not copying.
    """
    def norm(text):
        return " ".join((text or "").lower().split())

    reply = norm(answer)
    if len(reply) < 40:
        return None
    for index, member in enumerate([m for m in members if m.get("ok")], start=1):
        text = norm(member.get("answer"))
        if len(text) < 40:
            continue
        if reply == text:
            return index
        if text in reply and len(text) >= COPY_SHARE * len(reply):
            return index
        if reply in text and len(reply) >= COPY_SHARE * len(text):
            return index
        if difflib.SequenceMatcher(None, reply, text, autojunk=False).ratio() >= COPY_RATIO:
            return index
    return None


def head_order(members: list[dict]) -> list[str]:
    """Names of the members whose answers the head is shown, numbered as the
    head numbers them: "Member N" is head_order(members)[N - 1]. Failed
    members are left out, exactly as head.synthesise leaves them out."""
    return [m["name"] for m in members if m.get("ok")]


def run_head(config: dict, head_cfg: dict, question: str,
             members: list[dict], retries: int,
             context: str | None = None, on_chunk=None) -> dict:
    """Run the aggregation step. Never raises.

    A head that hands back one member's answer word for word has not chaired
    anything - a small head does this readily when the members contradict
    each other. The reply is checked against the members' answers and, once,
    asked again with the copy named; the second reply stands when it arrives,
    the first is kept if the second call fails, and the head result says what
    happened ("note").
    """
    started = time.monotonic()
    # Only a head that was told to end with the marker (the math, research
    # and decision presets, or a custom prompt that mentions it) is expected
    # to; the default preset answers in prose and has no value to parse.
    expects_value = prompts.ANSWER_MARKER in prompts.head_system_prompt(head_cfg, config)
    try:
        answer = head_mod.synthesise(
            head_cfg, question, members, config["request_timeout_s"],
            config=config, context=context, retries=retries, on_chunk=on_chunk,
        )
        note = None
        copied = copied_member(answer, members)
        if copied is not None:
            log.warning("head repeated Member %d word for word - asking again", copied)
            # The two rules a small head breaks most on a question that makes
            # it copy: restated here, once, because the copy shows the
            # system prompt alone did not hold.
            nudge = (f"Your previous reply repeated Member {copied} word for word. A chair does not "
                     "copy a member: weigh all the answers, say where they disagree and which is "
                     "right or that it cannot be decided, and write your own answer in your own words. "
                     "If the question is about the responder itself - who made you, what model you are - "
                     "the members are different models from different makers, so the panel has no single "
                     "answer: say that, and do not adopt one member's identity as yours.")
            try:
                # No client retries here: the first reply is the fallback,
                # so a busy box must not keep the page waiting a second cycle.
                answer = head_mod.synthesise(
                    head_cfg, question, members, config["request_timeout_s"],
                    config=config, context=context, retries=0, on_chunk=on_chunk, nudge=nudge,
                )
            except client.CompletionError as exc:
                log.error("head re-ask failed, keeping the first reply: %s", exc)
                note = (f"the head first repeated Member {copied} word for word and was asked again; "
                        f"the second attempt failed ({exc}), so the first reply is shown")
            else:
                again = copied_member(answer, members)
                note = (f"the head first repeated Member {copied} word for word and was asked again"
                        + (f"; it repeated Member {again} again" if again is not None else ""))
        outcome = {"ok": True, "answer": answer,
                   "value": answer_mod.extract(answer, expected=expects_value),
                   "expects_value": expects_value, "note": note}
    except client.CompletionError as exc:
        log.error("head failed: %s", exc)
        outcome = {"ok": False, "error": str(exc)}

    return {
        "name": head_cfg["name"],
        "lab": head_cfg.get("lab"),
        "seconds": round(time.monotonic() - started, 1),
        "order": head_order(members),
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
    (kind, payload) where kind is one of "start", "member_start",
    "member_chunk", "member_done", "head_start", "head_chunk", "head_done".
    A "member_done" payload carries the member's answer (or error) so a page
    can show what each member said the moment it said it, and "head_done"
    carries the head's; the "*_chunk" events carry the text so far while a
    reply is still being generated (see stream_answers), at most every
    CHUNK_INTERVAL_S. Calls are serialised with a
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

    streaming = stream_answers(config) and on_event is not None

    def chunker(kind: str, name: str):
        """A throttled reporter of a growing reply, or None when not streaming.

        It also keeps two diagnostics on itself - when the first token came
        and how many chunks arrived - because "it does not stream" has two
        very different causes that only the log can tell apart: a server that
        sends nothing until the reply is complete (zero chunks before done),
        and a server that is merely slow to produce its first token (a long
        prompt on a big CPU model can take minutes before the first chunk,
        and then streams normally).
        """
        if not streaming:
            return None
        last = [0.0]
        started = time.monotonic()

        def report(partial: str) -> None:
            report.chunks += 1
            if report.first_after is None:
                report.first_after = round(time.monotonic() - started, 1)
                log.info("%s: first token after %.1fs", name, report.first_after)
            now = time.monotonic()
            if now - last[0] < CHUNK_INTERVAL_S:
                return
            last[0] = now
            emit(kind, {"name": name, "partial": partial})
        report.chunks = 0
        report.first_after = None
        return report

    def note_streaming(name: str, reporter, result: dict) -> None:
        if reporter is None or result.get("cached") or not result.get("ok"):
            return
        length = len(result.get("answer") or "")
        if reporter.chunks <= 1 and length > 200:
            # One event carrying a long reply is a server that generated the
            # whole answer first and sent it in a piece - a buffering proxy,
            # or a backend that only pretends to stream. The page then shows
            # nothing until the end, which looks like "streaming is off".
            log.warning("%s: a %d-character reply arrived in %d piece(s) after %ss - that server "
                        "does not really stream (a buffering proxy, or streaming unsupported)",
                        name, length, reporter.chunks, reporter.first_after)
        else:
            log.info("%s: %d chunk(s), first after %ss", name, reporter.chunks, reporter.first_after)

    def ask_one(member: dict) -> dict:
        emit("member_start", {"name": member["name"]})
        reporter = chunker("member_chunk", member["name"])
        result = ask_member(config, member, question, cache, retries, context, on_chunk=reporter)
        note_streaming(member["name"], reporter, result)
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
            ordered = order_for_head(members, results, config)
            # The order goes out with head_start, so the page can number the
            # member cards while the head is still reading them.
            emit("head_start", {"name": head_cfg["name"], "order": head_order(ordered)})
            head_reporter = chunker("head_chunk", head_cfg["name"])
            output["head"] = run_head(
                config, head_cfg, question, ordered,
                retries, context, on_chunk=head_reporter,
            )
            note_streaming(head_cfg["name"], head_reporter, output["head"])
            emit("head_done", {
                "name": head_cfg["name"],
                "ok": output["head"]["ok"],
                "order": output["head"]["order"],
                "seconds": output["head"]["seconds"],
                "answer": output["head"].get("answer"),
                "value": output["head"].get("value"),
                "expects_value": output["head"].get("expects_value"),
                "note": output["head"].get("note"),
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