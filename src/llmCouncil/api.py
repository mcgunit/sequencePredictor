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
  GET  /health                                            -> 200 {ok, busy}
  GET  /endpoints                                         -> 200 {members, head}

A job is {id, state, question, progress, created} plus `result` once state is
"done" or `error` when "failed". `result` is exactly what run_council returns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from council import jobs as jobs_mod
from council import orchestrator, sampling
from council.orchestrator import ConfigError, EndpointsUnavailable, load_config

log = logging.getLogger("council.api")

MAX_BODY = 64 * 1024
MAX_QUESTION = 4000
MAX_CONTEXT = 20000


def make_runner(config: dict) -> jobs_mod.JobRunner:
    """Turn run_council's events into a seat-state map the page can render.

    The page draws a seat per member, so it needs to know which member is
    being asked right now - not just that the run is in progress. Every seat
    carries one of: waiting, asking, answered, failed, cached.
    """
    def run(question: str, context: str | None, progress) -> dict:
        state = {"seats": {}, "head": None, "phase": "starting"}

        def on_event(kind: str, payload: dict) -> None:
            if kind == "start":
                state["seats"] = {n: {"state": "waiting"} for n in payload["members"]}
                state["head"] = {"name": payload["head"], "state": "waiting"} \
                    if payload["head"] else None
                state["phase"] = "members"
            elif kind == "member_start":
                state["seats"][payload["name"]] = {"state": "asking"}
            elif kind == "member_done":
                seat = "cached" if payload["cached"] else (
                    "answered" if payload["ok"] else "failed")
                state["seats"][payload["name"]] = {
                    "state": seat, "seconds": payload["seconds"]}
            elif kind == "head_start":
                state["phase"] = "head"
                state["head"] = {"name": payload["name"], "state": "asking"}
            elif kind == "head_done":
                state["head"] = {
                    "name": payload["name"],
                    "state": "answered" if payload["ok"] else "failed",
                    "seconds": payload["seconds"],
                }
            # A fresh dict each time: the job holds a reference, and mutating
            # one in place could be serialised mid-update by a polling request.
            progress(json.loads(json.dumps(state)))

        return orchestrator.run_council(
            question, config, context=context, on_event=on_event
        )

    return jobs_mod.JobRunner(run)


def describe_endpoints(config: dict) -> dict:
    """What the page shows in its header. No secrets here - names only."""
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
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path == "/health":
            self._send(200, {"ok": True, "busy": self.runner.active() is not None})
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
                        help="verify the model endpoints before serving")
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