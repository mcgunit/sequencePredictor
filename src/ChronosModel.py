# Chronos Model - a pretrained time-series foundation model as a per-position
# predictor (README roadmap item 5).
#
# Every drawn position is treated as its own univariate series (position 1 of
# lotto over time, position 2, ...) and amazon/chronos-2 is asked, zero-shot,
# for the next value's distribution; that distribution over the game's labels
# is the per-position score shape every positional consumer here already
# understands, so this plugs in as one more tracked row and as a feature for
# the positional meta-learner.
#
# HOW TO READ THIS ROW. For the sorted set games the per-position series are
# order statistics - position 1 is the minimum of the draw, so its
# distribution is genuinely predictable without the draw being predictable at
# all. The row must therefore be compared against an order-statistics
# baseline, never against chance. For the positional games (pick3, Joker+) an
# honest process gives near-uniform forecasts, which makes this row a useful
# negative control: a strong, stable deviation there is the interesting
# outcome, not a good hit rate.
#
# WHERE THE MODEL RUNS. Not in this interpreter. torch and chronos need
# numpy 2.x while the pipeline runs TensorFlow 2.16 on numpy 1.26, so they
# live in their own library directory and this class talks to a small worker
# process (src/foundation/chronos_forecast.py) over JSON lines. The worker is
# kept warm for the life of this object, because a backtest asks for one
# forecast per day and paying the model load each time would dominate. Any
# failure - libraries missing, worker crash, timeout - degrades to "no
# scores", which costs this row for the day and nothing else.

import atexit
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Helpers import Helpers

helpers = Helpers()

WORKER = os.path.join(current_dir, "foundation", "chronos_forecast.py")
LIBS = os.environ.get("FOUNDATION_LIBS", "/root/.foundation-libs")


def _die_with_parent():
    """
    PR_SET_PDEATHSIG: the worker gets SIGTERM the moment its parent dies.
    The history rebuild runs the statistical step in a pool of spawned
    processes, each of which would otherwise leave a ~0.4 GB forecasting
    worker behind when it exits - eight of those on a 16 GB box next to
    TensorFlow is how a machine runs out of memory quietly. The same guard
    src/HyperoptRunner.py uses for its trial processes.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:
        pass


class ChronosModel:
    def __init__(self):
        self.dataPath = ""
        self.modelPath = ""
        self.recent_draws = 512          # context handed to the model
        self.quantiles = 199             # resolution of the forecast distribution
        self.startup_timeout = 300       # first call may download the weights
        self.request_timeout = 120
        self.sorted_prediction = True    # set games are played as a sorted ticket
        self.min_number = None
        self.max_number = None
        self._worker = None
        self._lock = threading.Lock()
        self._unavailable = None         # reason, once known
        self._cache = {}
        # Belt and braces next to PR_SET_PDEATHSIG: a clean interpreter exit
        # closes the worker instead of relying on the signal.
        atexit.register(self.close)

    # --- configuration (same shape as the statistical models) ---------------
    def setDataPath(self, dataPath): self.dataPath = dataPath
    def setModelPath(self, modelPath): self.modelPath = modelPath
    def setRecentDraws(self, n): self.recent_draws = max(8, int(n))
    def setQuantiles(self, n): self.quantiles = max(9, int(n))
    def setRequestTimeout(self, seconds): self.request_timeout = max(5, int(seconds))
    def setSortedPrediction(self, use): self.sorted_prediction = bool(use)

    def setGameRange(self, min_number, max_number):
        self.min_number = int(min_number)
        self.max_number = int(max_number)

    def clear(self):
        self._cache = {}

    # --- the worker ----------------------------------------------------------
    def available(self):
        return self._unavailable is None

    def _start_worker(self):
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        if not os.path.isdir(LIBS):
            self._unavailable = (f"the foundation libraries are not installed at {LIBS} - "
                                 f"see README 'Chronos Model'")
            return None
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/root"),
            "PYTHONPATH": LIBS,
            "PYTHONUNBUFFERED": "1",
            "FOUNDATION_LIBS": LIBS,
            "CHRONOS_CONTEXT": str(self.recent_draws),
            # The 6 GB card is TensorFlow's; this model is small and runs on
            # the CPU in a fraction of a second.
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "CHRONOS_MODEL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        try:
            worker = subprocess.Popen(
                [sys.executable, WORKER, "--warm"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=env, text=True, bufsize=1, preexec_fn=_die_with_parent)
        except Exception as exc:
            self._unavailable = f"could not start the Chronos worker: {exc}"
            return None

        ready = self._read_line(worker, self.startup_timeout)
        if not ready or not ready.get("ok"):
            self._unavailable = f"the Chronos worker did not start: {(ready or {}).get('error', 'no reply')}"
            try:
                worker.kill()
            except Exception:
                pass
            return None
        self._worker = worker
        return worker

    @staticmethod
    def _read_line(worker, timeout):
        """One JSON reply, or None when the worker is silent for too long."""
        result = {}

        def read():
            try:
                result["line"] = worker.stdout.readline()
            except Exception as exc:
                result["error"] = str(exc)

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive() or not result.get("line"):
            return None
        try:
            return json.loads(result["line"])
        except json.JSONDecodeError:
            return None

    def _ask(self, series, labels):
        """One forecast request; None (and this model disabled) on any failure."""
        with self._lock:
            if self._unavailable is not None:
                return None
            worker = self._start_worker()
            if worker is None:
                print(f"Chronos Model unavailable: {self._unavailable}")
                return None
            request = {"series": series, "labels": labels, "quantiles": self.quantiles}
            try:
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
            except Exception as exc:
                self._unavailable = f"the Chronos worker stopped accepting requests: {exc}"
                return None
            reply = self._read_line(worker, self.request_timeout)
            if reply is None:
                self._unavailable = f"the Chronos worker did not answer within {self.request_timeout}s"
                self.close()
                return None
            if not reply.get("ok"):
                print(f"Chronos Model could not forecast: {reply.get('error')}")
                return None
            return reply

    def close(self):
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            worker.stdin.close()
        except Exception:
            pass
        try:
            worker.wait(timeout=5)
        except Exception:
            try:
                worker.kill()
            except Exception:
                pass

    # --- scores ---------------------------------------------------------------
    def _labels(self, numbers):
        if self.min_number is not None and self.max_number is not None:
            return list(range(self.min_number, self.max_number + 1))
        values = [int(v) for draw in numbers for v in draw]
        return list(range(min(values), max(values) + 1)) if values else []

    def _forecast(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-position label distributions for one point in time, cached per
        (skipRows, skipLastColumns, specialColumnCount) so run() and the two
        score_* methods share a single request.
        """
        key = (skipRows, skipLastColumns, specialColumnCount)
        if key in self._cache:
            return self._cache[key]

        _, _, _, _, _, numbers, _, _ = helpers.load_data(
            self.dataPath, skipRows=skipRows, skipLastColumns=skipLastColumns,
            specialColumnCount=specialColumnCount)
        numbers = [[int(value) for value in draw] for draw in numbers]
        if len(numbers) < 8 or not numbers[-1]:
            self._cache[key] = []
            return []

        positions = len(numbers[-1])
        window = numbers[-self.recent_draws:]
        series = [[float(draw[pos]) for draw in window if len(draw) > pos] for pos in range(positions)]
        labels = self._labels(window)
        if not labels:
            self._cache[key] = []
            return []

        reply = self._ask(series, labels)
        if reply is None:
            self._cache[key] = []
            return []

        # The worker rounds its masses to keep the reply small, so each slot
        # is renormalized here: consumers may rely on a slot being a
        # distribution, and "sums to one" should not depend on the transport.
        scores = []
        for slot in reply["scores"]:
            parsed = {int(label): float(mass) for label, mass in slot.items()}
            total = sum(parsed.values())
            scores.append({label: mass / total for label, mass in parsed.items()} if total > 0
                          else {label: 1.0 / len(parsed) for label in parsed})
        self._cache[key] = scores
        return scores

    def score_positions(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        One {value: probability} dict per drawn position, in drawn order - the
        feature the positional meta-learner consumes. [] when the model could
        not be reached, which the consumers already treat as "no signal".
        """
        return self._forecast(skipRows, skipLastColumns, specialColumnCount)

    def score_numbers(self, skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Per-number score for the set games: the per-position distributions
        pooled, i.e. P(value appears somewhere in the draw) under the model's
        independent per-position forecasts, normalized. {} when unavailable.
        """
        slots = self._forecast(skipRows, skipLastColumns, specialColumnCount)
        if not slots:
            return {}
        pooled = {}
        for slot in slots:
            for value, mass in slot.items():
                pooled[value] = pooled.get(value, 0.0) + mass
        total = sum(pooled.values())
        return {value: mass / total for value, mass in pooled.items()} if total else {}

    # --- ticket ----------------------------------------------------------------
    def _positional_ticket(self, slots):
        """Positional games: the most likely value per slot, in drawn order, duplicates allowed."""
        return [max(sorted(slot), key=lambda value: slot[value]) for slot in slots]

    def _set_ticket(self, slots):
        """
        Set games: one distinct number per position, kept ascending. The
        per-position forecasts are order statistics, so taking each slot's
        best value in turn - skipping anything already used or below the
        previous pick - respects that structure instead of flattening it into
        one pooled top-N.
        """
        ticket = []
        used = set()
        for slot in slots:
            candidates = sorted(slot, key=lambda value: (-slot[value], value))
            choice = next((value for value in candidates if value not in used and (not ticket or value > ticket[-1])), None)
            if choice is None:
                choice = next((value for value in candidates if value not in used), None)
            if choice is None:
                continue
            ticket.append(choice)
            used.add(choice)
        return sorted(ticket) if self.sorted_prediction else ticket

    def generate_best_subset(self, predicted_numbers, nSubset):
        """The nSubset highest-scoring numbers of an already-built ticket (Keno)."""
        pooled = self.score_numbers()
        ranked = sorted(predicted_numbers, key=lambda value: (-pooled.get(int(value), 0.0), int(value)))
        return sorted(int(value) for value in ranked[:nSubset])

    def run(self, generateSubsets=[], skipRows=0, skipLastColumns=0, specialColumnCount=0):
        """
        Same contract as the statistical models: (prediction, subsets).
        An empty prediction means the model could not be reached - callers
        already skip a model that returns nothing.
        """
        slots = self._forecast(skipRows, skipLastColumns, specialColumnCount)
        if not slots:
            return [], {}

        positional = helpers.is_positional_game(self.dataPath)
        prediction = self._positional_ticket(slots) if positional else self._set_ticket(slots)

        subsets = {}
        if generateSubsets and prediction:
            pooled = self.score_numbers(skipRows, skipLastColumns, specialColumnCount)
            for size in generateSubsets:
                ranked = sorted(prediction, key=lambda value: (-pooled.get(int(value), 0.0), int(value)))
                subsets[size] = sorted(int(value) for value in ranked[:size])
        return prediction, subsets
