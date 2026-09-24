"""HTTP API in front of run_council(), for the web chat page.

Binds to 127.0.0.1 by default. The Node front end reverse-proxies to it, so
the API is never reachable from the network directly and authentication stays
in one place - the existing Express login - rather than being reimplemented
here. Do not bind this to 0.0.0.0 without putting auth in front of it: a
council run costs minutes of CPU, so an open endpoint is a denial-of-service
waiting to happen.

Endpoints:

  POST /ask           {"question": str, "context": str?}  -> 202 {job}
                      409 when a job is already running
  GET  /job/<id>                                          -> 200 {job}
  GET  /jobs                                              -> 200 {jobs: [...]}
  GET  /health                                            -> 200 {ok, busy, status}
  GET  /endpoints                                         -> 200 {members, head, status}

`status` is the live reachability of the llama.cpp endpoints (see
endpoint_status): the API stays up while the model boxes are switched off, and
the page uses this to say the council has to be summoned instead of accepting
a question that could not be answered.

A job is {id, state, question, progress, created} plus `result` once state is
"done" or `error` when "failed". `result` is exactly what run_council returns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from council import health
from council import jobs as jobs_mod
from council import orchestrator, sampling
from council.orchestrator import ConfigError, EndpointsUnavailable, load_config

log = logging.getLogger("council.api")

MAX_BODY = 64 * 1024
MAX_QUESTION = 4000
MAX_CONTEXT = 20000

# The model boxes are not on 24/7 (they cost electricity), so the API must
# survive them being down and simply report that the council cannot sit. The
# probe is therefore live rather than a start-up check - but it must not make
# the page wait: every endpoint is probed in parallel with a short timeout and
# the verdict is cached for STATUS_TTL_S, so a dark model box costs one short
# timeout per quarter minute, not one per page load.
STATUS_TTL_S = 15
STATUS_TIMEOUT_S = 2
_status_lock = threading.Lock()
_status_cache: dict = {"at": 0.0, "value": None}


def _probe(endpoint: dict, timeout_s: int) -> dict:
    """One endpoint's reachability. Never raises - a probe cannot break a page."""
    entry = {"name": endpoint.get("name"), "ok": False, "detail": None}
    try:
        health.check(endpoint["base_url"], timeout_s)
        entry["ok"] = True
    except health.EndpointUnavailable as exc:
        entry["detail"] = str(exc)
    except Exception as exc:  # malformed config, DNS, anything else
        entry["detail"] = f"{type(exc).__name__}: {exc}"
    return entry


def endpoint_status(config: dict, force: bool = False) -> dict:
    """
    Live view of the llama.cpp endpoints behind this API, cached for
    STATUS_TTL_S seconds:

        {checked, ready, reachable, total, members: [...], head: {...}|null}

    `ready` is what the page gates on: at least one member answers and, when a
    head is configured, the head answers too - without the head there is
    nobody to synthesise the members' answers into a verdict.
    """
    now = time.monotonic()
    with _status_lock:
        cached = _status_cache["value"]
        if cached is not None and not force and now - _status_cache["at"] < STATUS_TTL_S:
            return cached

    members = orchestrator.enabled_members(config)
    head = orchestrator.head_config(config)
    timeout_s = min(int(config.get("health_timeout_s", 5)), STATUS_TIMEOUT_S)
    endpoints = list(members) + ([head] if head else [])

    results = []
    if endpoints:
        with ThreadPoolExecutor(max_workers=min(8, len(endpoints))) as pool:
            results = list(pool.map(lambda e: _probe(e, timeout_s), endpoints))
    member_results = results[:len(members)]
    head_result = results[len(members)] if head else None

    reachable = sum(1 for entry in member_results if entry["ok"])
    status = {
        "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ready": bool(reachable and (head_result is None or head_result["ok"])),
        "reachable": reachable,
        "total": len(member_results),
        "members": member_results,
        "head": head_result,
    }
    with _status_lock:
        _status_cache["at"] = time.monotonic()
        _status_cache["value"] = status
    return status


def make_runner(config: dict) -> jobs_mod.JobRunner:
    """Turn run_council's events into a seat-state map the page can render.

    The page draws a seat per member, so it needs to know which member is
    being asked right now - not just that the run is in progress. Every seat
    carries one of: waiting, asking, answered, failed, cached.
    """
    def run(question: str, context: str | None, progress) -> dict:
        state = {"seats": {}, "head": None, "phase": "starting",
                 "mode": orchestrator.ask_mode(config)}

        def on_event(kind: str, payload: dict) -> None:
            # run_council serialises these calls, so in parallel mode the
            # seats of several members can be "asking" at once without two
            # threads writing this dict together.
            if kind == "start":
                state["seats"] = {n: {"state": "waiting"} for n in payload["members"]}
                state["head"] = {"name": payload["head"], "state": "waiting"} \
                    if payload["head"] else None
                state["phase"] = "members"
            elif kind == "member_start":
                state["seats"][payload["name"]] = {"state": "asking"}
            elif kind == "member_chunk":
                # The reply so far, so the page can show it typing.
                state["seats"][payload["name"]] = {"state": "asking", "partial": payload["partial"]}
            elif kind == "head_chunk":
                state["head"] = {"name": payload["name"], "state": "asking", "partial": payload["partial"]}
            elif kind == "member_done":
                seat = "cached" if payload["cached"] else (
                    "answered" if payload["ok"] else "failed")
                # The answer travels with the seat: the page shows what each
                # member said the moment it said it, instead of after the
                # head has finished minutes later.
                state["seats"][payload["name"]] = {
                    "state": seat, "seconds": payload["seconds"],
                    "answer": payload.get("answer"), "error": payload.get("error")}
            elif kind == "head_start":
                state["phase"] = "head"
                state["head"] = {"name": payload["name"], "state": "asking"}
            elif kind == "head_done":
                state["head"] = {
                    "name": payload["name"],
                    "state": "answered" if payload["ok"] else "failed",
                    "seconds": payload["seconds"],
                    "answer": payload.get("answer"), "value": payload.get("value"),
                    "error": payload.get("error"),
                }
            # A fresh dict each time: the job holds a reference, and mutating
            # one in place could be serialised mid-update by a polling request.
            progress(json.loads(json.dumps(state)))

        return orchestrator.run_council(
            question, config, context=context, on_event=on_event
        )

    return jobs_mod.JobRunner(run)


def describe_endpoints(config: dict) -> dict:
    """What the page shows in its header. No secrets here - names only.

    Carries the live endpoint status as well, so the page learns in its one
    start-up call both who sits on the council and whether they can be
    reached at all - and whether replies stream, so it knows to poll faster.
    """
    members = [
        {"name": m["name"], "lab": m.get("lab"),
         "temperature": sampling.for_member(m)["temperature"]}
        for m in orchestrator.enabled_members(config)
    ]
    head = orchestrator.head_config(config)
    return {
        "members": members,
        "head": {"name": head["name"], "lab": head.get("lab")} if head else None,
        "preset": config.get("head_preset", "default"),
        "ask_members": orchestrator.ask_mode(config),
        "stream_answers": orchestrator.stream_answers(config),
        "status": endpoint_status(config),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "council-api"
    runner: jobs_mod.JobRunner
    config: dict

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "bad Content-Length"})
            return None
        if length <= 0:
            self._send(400, {"error": "empty body"})
            return None
        if length > MAX_BODY:
            self._send(413, {"error": "body too large"})
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send(400, {"error": f"bad JSON: {exc}"})
            return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = dict(pair.split("=", 1) for pair in parsed.query.split("&") if "=" in pair)

        if path == "/health":
            self._send(200, {
                "ok": True,
                "busy": self.runner.active() is not None,
                "status": endpoint_status(self.config, force=query.get("force") == "1"),
            })
        elif path == "/endpoints":
            self._send(200, describe_endpoints(self.config))
        elif path == "/jobs":
            self._send(200, {"jobs": self.runner.recent()})
        elif path.startswith("/job/"):
            job = self.runner.get(path[len("/job/"):])
            if job is None:
                self._send(404, {"error": "no such job"})
            else:
                self._send(200, job.public())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path.rstrip("/") != "/ask":
            self._send(404, {"error": "not found"})
            return

        body = self._read_json()
        if body is None:
            return

        question = (body.get("question") or "").strip()
        context = (body.get("context") or "").strip() or None

        if not question:
            self._send(400, {"error": "question is required"})
            return
        if len(question) > MAX_QUESTION:
            self._send(413, {"error": f"question over {MAX_QUESTION} characters"})
            return
        if context and len(context) > MAX_CONTEXT:
            self._send(413, {"error": f"context over {MAX_CONTEXT} characters"})
            return

        try:
            job = self.runner.submit(question, context)
        except jobs_mod.Busy as exc:
            self._send(409, {"error": str(exc), "busy": True})
            return

        self._send(202, job.public())

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def serve(config: dict, host: str, port: int) -> int:
    Handler.runner = make_runner(config)
    Handler.config = config

    httpd = ThreadingHTTPServer((host, port), Handler)
    log.info("council api on http://%s:%d", host, port)
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("bound to %s - put authentication in front of this", host)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HTTP API for the council.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--check", action="store_true",
                        help="verify the model endpoints once and refuse to serve if any is "
                             "down. For a manual pre-flight only: leave it off for the API "
                             "the web page talks to, which must stay up while the model "
                             "boxes are off and report that through /health and /endpoints.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

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

    if args.check:
        members = orchestrator.enabled_members(config)
        head = orchestrator.head_config(config)
        endpoints = members + ([head] if head else [])
        log.info("checking %d endpoint(s)", len(endpoints))
        if orchestrator.check_endpoints(config, endpoints):
            log.error("not serving: endpoints unavailable")
            return 1

    return serve(config, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())