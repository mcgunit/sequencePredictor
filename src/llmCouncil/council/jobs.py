"""A single-slot job queue for council runs.

A council run takes minutes with real models, which is far longer than a
browser will hold a request open. So the web API starts a job, returns its id
immediately, and the caller polls for the result.

Exactly one job runs at a time. The llama-servers are started with
`--parallel 1` and the council asks its members sequentially, so concurrent
runs would queue inside llama-server anyway - and each one costs minutes of
CPU. Refusing a second job with a clear "busy" is more honest than silently
doubling everyone's wait, and it stops one page from monopolising the box.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass
class Job:
    id: str
    question: str
    context: str | None
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    progress: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict:
        """The shape sent to the client."""
        body = {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "progress": self.progress,
            "created": self.created,
        }
        if self.state == DONE:
            body["result"] = self.result
        elif self.state == FAILED:
            body["error"] = self.error
        return body


class Busy(RuntimeError):
    """A job is already running."""


class JobRunner:
    """Runs one council job at a time and keeps the last few results."""

    def __init__(self, run: Callable[[str, str | None, Callable[[dict], None]], dict],
                 history: int = 20):
        self._run = run
        self._history = history
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._active: str | None = None

    def submit(self, question: str, context: str | None = None) -> Job:
        """Queue a job. Raises Busy if one is already in flight."""
        with self._lock:
            if self._active is not None:
                raise Busy(f"job {self._active} is still running")
            job = Job(id=uuid.uuid4().hex[:12], question=question, context=context)
            self._jobs[job.id] = job
            while len(self._jobs) > self._history:
                self._jobs.popitem(last=False)
            self._active = job.id

        threading.Thread(target=self._work, args=(job,), daemon=True).start()
        return job

    def _work(self, job: Job) -> None:
        def progress(state: dict) -> None:
            job.progress = state

        job.state = RUNNING
        try:
            job.result = self._run(job.question, job.context, progress)
            job.state = DONE
        except Exception as exc:                     # noqa: BLE001
            # A failure here must not wedge the queue: any exception is
            # recorded on the job and the slot is released either way.
            log.exception("job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
            job.state = FAILED
        finally:
            job.finished = time.time()
            job.progress = None
            with self._lock:
                self._active = None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> str | None:
        with self._lock:
            return self._active

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.public() for j in reversed(jobs[-limit:])]