# The plumbing shared by every pretrained time-series foundation model used
# as a per-position predictor (README roadmap item 5): Chronos-2
# (src/ChronosModel.py) and TimesFM-3 (src/TimesFmModel.py). A subclass only
# names its worker script and its defaults.
#
# WHERE THE MODEL RUNS. Not in this interpreter. torch and the model packages
# want numpy 2.x while the pipeline runs TensorFlow 2.16 on numpy 1.26, so
# they live in their own library directory (FOUNDATION_LIBS, default
# /root/.foundation-libs) and this class talks to a small worker process over
# JSON lines. The worker is kept warm for the life of the object, because a
# backtest asks for one forecast per day and paying the model load each time
# would dominate. Any failure - libraries missing, worker crash, timeout -
# degrades to "no scores", which costs that row for the day and nothing else.
#
# WHY THERE IS A PRECOMPUTE MODE. The meta-learner's training table is
# collected by src/Backtester.py, which forks a pool of workers and shares
# the model objects copy-on-write. A forked child must never talk to the
# parent's worker (both would write into the same pipe) and must never start
# one of its own (fifteen children x ~0.9 GB is how a 16 GB box dies), so the
# parent computes every day it will need up front, closes the worker, and
# switches to cache-only: the children then read the inherited dictionary and
# start nothing. The PID guard below is the belt to that braces.

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

WORKER_DIR = os.path.join(current_dir, "foundation")
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


class FoundationModel:
    # Identity by attribute, not by isinstance: src/ modules here are imported
    # both as `src.X` (Predictor.py, ModelFactory.py) and as `X` (the sibling
    # imports inside src/), so the interpreter holds two distinct copies of
    # this class and isinstance() silently answers False across them.
    IS_FOUNDATION_MODEL = True

    # --- what a subclass defines --------------------------------------------
    WORKER_SCRIPT = None          # file name inside src/foundation/
    NAME = "Foundation Model"     # for log lines
    WORKER_ENV = {}               # extra environment for the worker
    PASSTHROUGH_ENV = ()          # env names copied through when set

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
        self._cache_only = False         # precomputed: never start a worker
        self._owner_pid = os.getpid()    # who owns the worker handle
        self._creator_pid = os.getpid()  # who built this object
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
        # A precomputed cache is the model: dropping it in cache-only mode
        # would silently turn every score into "no signal" for the rest of
        # the run, with no worker left to recompute it.
        if not self._cache_only:
            self._cache = {}

    @classmethod
    def worker_path(cls):
        return os.path.join(WORKER_DIR, cls.WORKER_SCRIPT)

    @classmethod
    def installed(cls):
        """
        Cheap, no-side-effect answer to "could this model run here at all?" -
        used by src/ModelFactory.py to decide whether the meta-learner gets a
        column for it, which must not depend on a worker starting.
        """
        return os.path.isdir(LIBS) and os.path.isfile(cls.worker_path())

    # --- the worker ----------------------------------------------------------
    def available(self):
        return self._unavailable is None

    def _start_worker(self):
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        # A model built in another process and reached here has been carried
        # across a fork - src/Backtester.py's pool is the case that matters.
        # Starting a worker per child is how fifteen of them each take
        # 0.8-2.8 GB, so this refuses instead, and says what to do about it.
        # The only supported way to use a foundation model inside that pool
        # is ModelFactory.prepare_foundation_scores before the fork.
        if os.getpid() != self._creator_pid:
            self._unavailable = (f"{self.NAME} was carried into another process (pid {os.getpid()}) without "
                                 f"being precomputed - call ModelFactory.prepare_foundation_scores "
                                 f"before the backtest forks")
            return None
        if not os.path.isdir(LIBS):
            self._unavailable = (f"the foundation libraries are not installed at {LIBS} - "
                                 f"see README 'Foundation models'")
            return None
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/root"),
            "PYTHONPATH": LIBS,
            "PYTHONUNBUFFERED": "1",
            "FOUNDATION_LIBS": LIBS,
            "FOUNDATION_CONTEXT": str(self.recent_draws),
            # The 6 GB card is TensorFlow's; these models are small and run on
            # the CPU in a fraction of a second.
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        env.update(self.WORKER_ENV)
        for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE") + tuple(self.PASSTHROUGH_ENV):
            if os.environ.get(name):
                env[name] = os.environ[name]
        try:
            worker = subprocess.Popen(
                [sys.executable, self.worker_path(), "--warm"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=env, text=True, bufsize=1, preexec_fn=_die_with_parent)
        except Exception as exc:
            self._unavailable = f"could not start the {self.NAME} worker: {exc}"
            return None

        ready = self._read_line(worker, self.startup_timeout)
        if not ready or not ready.get("ok"):
            self._unavailable = f"the {self.NAME} worker did not start: {(ready or {}).get('error', 'no reply')}"
            try:
                worker.kill()
            except Exception:
                pass
            return None
        self._worker = worker
        self._owner_pid = os.getpid()
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
            # Inherited by fork: the handle in this object belongs to the
            # parent process. Writing to it here would interleave two
            # conversations in one pipe and corrupt both.
            if self._worker is not None and os.getpid() != self._owner_pid:
                self._worker = None
            if self._cache_only:
                self._unavailable = (f"{self.NAME} is running from a precomputed cache and was asked for "
                                     f"a day that is not in it")
                return None
            worker = self._start_worker()
            if worker is None:
                print(f"{self.NAME} unavailable: {self._unavailable}")
                return None
            request = {"series": series, "labels": labels, "quantiles": self.quantiles}
            try:
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
            except Exception as exc:
                self._unavailable = f"the {self.NAME} worker stopped accepting requests: {exc}"
                return None
            reply = self._read_line(worker, self.request_timeout)
            if reply is None:
                self._unavailable = f"the {self.NAME} worker did not answer within {self.request_timeout}s"
                self.close()
                return None
            if not reply.get("ok"):
                print(f"{self.NAME} could not forecast: {reply.get('error')}")
                return None
            return reply

    def close(self):
        # Closing re-arms the model. _unavailable is a per-worker verdict (a
        # timeout, a crashed start), not a permanent one: left set, a single
        # transient failure on the first game would silently drop the row for
        # every later game of the same run, since Predictor closes between
        # games. The refusals that MUST stay permanent - missing libraries, a
        # model carried across a fork, cache-only mode - are re-checked on
        # every call anyway.
        self._unavailable = None
        worker, self._worker = self._worker, None
        if worker is None or os.getpid() != self._owner_pid:
            return                       # not ours to close (inherited by fork)
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
        score_* methods share a single request - and so a precomputed run can
        answer from the same dictionary without a worker at all.
        """
        key = (skipRows, skipLastColumns, specialColumnCount)
        if key in self._cache:
            return self._cache[key]
        if self._cache_only:
            return []                    # a day nobody precomputed: no signal, no spawn

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

    def precompute(self, keys, label=None):
        """
        Fill the cache for every (skipRows, skipLastColumns, specialColumnCount)
        the caller will ask for, then close the worker and refuse to start
        another. This is what makes the model usable inside src/Backtester.py's
        forked pool - see the header. Returns the number of days that produced
        scores.
        """
        done = 0
        failed = 0
        try:
            for index, key in enumerate(keys):
                # One unusable day must not cost the game its whole table.
                # src/Backtester.py already tolerates exactly this per model
                # per day (it records a *_error and moves on), and the first
                # key is the likeliest to raise: when the window covers the
                # whole history, skipRows equals the row count and
                # Helpers.load_data is handed an empty slice.
                try:
                    if self._forecast(*key):
                        done += 1
                except Exception as exc:
                    failed += 1
                    if failed == 1:
                        print(f"{self.NAME}: no forecast for day-slice {key} ({type(exc).__name__}: {exc}) - "
                              f"that day gets no scores, the rest continue")
                if label and index and index % 50 == 0:
                    print(f"  {label}: {index}/{len(keys)} days forecast")
        finally:
            # Whatever happened, the worker must not survive into the caller's
            # next game and its fork pool: 0.8-2.8 GB held for the rest of a
            # run is exactly what this method exists to prevent.
            self.close()
            self._cache_only = True
        if failed:
            print(f"{self.NAME}: {failed} of {len(keys)} day-slices produced no scores")
        return done

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
    @staticmethod
    def _tie_break(slot):
        """
        Sort key for one slot: most likely first, and among EXACT ties the
        value closest to the slot's own mean. Ties are not rare here - a
        quantile curve read as a piecewise-linear CDF is flat across whole
        stretches of labels - and "first wins" over ascending values makes
        every one of them resolve to the lowest label, a measured downward
        bias of about 0.8 of a label. The mean is the only thing the
        distribution says about where the tie sits.
        """
        centre = sum(value * mass for value, mass in slot.items())
        return lambda value: (-slot[value], abs(value - centre), value)

    def _positional_ticket(self, slots):
        """Positional games: the most likely value per slot, in drawn order, duplicates allowed."""
        return [min(slot, key=self._tie_break(slot)) for slot in slots]

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
            candidates = sorted(slot, key=self._tie_break(slot))
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


if __name__ == "__main__":
    # Self-check for the one property that cannot be allowed to regress: a
    # precomputed model must answer inside a forked child WITHOUT starting a
    # worker there. Fifteen backtest workers each holding a 0.8-2.8 GB
    # forecaster is how this box runs out of memory.
    #
    #   python3 src/FoundationModel.py [game]
    import glob
    import multiprocessing

    from ChronosModel import ChronosModel

    game = sys.argv[1] if len(sys.argv) > 1 else "lotto"
    dataPath = os.path.join(parent_dir, "data", "trainingData", game)

    def workers_alive():
        # Exact argv, never a substring scan: a shell (or this very script)
        # whose command line merely MENTIONS the worker would otherwise count
        # itself - the same trap that makes `pkill -f` kill its own caller.
        count = 0
        for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                with open(cmdline, "rb") as handle:
                    argv = handle.read().decode("utf-8", "replace").split("\0")
            except OSError:
                continue
            if len(argv) > 1 and os.path.basename(argv[1]) in {"chronos_forecast.py", "timesfm_forecast.py"}:
                count += 1
        return count

    if not ChronosModel.installed():
        print(f"Chronos is not installed at {LIBS} - see README 'Foundation models'")
        sys.exit(1)

    model = ChronosModel()
    model.setDataPath(dataPath)
    keys = [(skip, 0, 0) for skip in (1, 2, 3)]
    print(f"precomputing {len(keys)} day-slices of {game}...")
    done = model.precompute(keys)
    assert done == len(keys), f"only {done} of {len(keys)} day-slices produced scores"
    assert model._cache_only, "precompute must switch the model to cache-only"
    after = workers_alive()
    print(f"precomputed {done} day-slices; workers still alive: {after}")
    assert after == 0, "precompute must close its worker before the pool forks"

    def child(queue):
        cached = model.score_positions(*keys[0])
        missing = model.score_positions(999999, 0, 0)     # a day nobody precomputed
        queue.put({"pid": os.getpid(), "cached_slots": len(cached),
                   "missing_slots": len(missing), "workers": workers_alive()})

    queue = multiprocessing.get_context("fork").Queue()
    children = [multiprocessing.get_context("fork").Process(target=child, args=(queue,)) for _ in range(4)]
    for process in children:
        process.start()
    results = [queue.get(timeout=120) for _ in children]
    for process in children:
        process.join()

    # The same fork, with a model nobody precomputed: it must refuse to start
    # a worker rather than quietly costing a gigabyte per child.
    naive = ChronosModel()
    naive.setDataPath(dataPath)

    def naive_child(queue):
        slots = naive.score_positions(1, 0, 0)
        queue.put({"pid": os.getpid(), "slots": len(slots), "workers": workers_alive()})

    naive_process = multiprocessing.get_context("fork").Process(target=naive_child, args=(queue,))
    naive_process.start()
    naive_result = queue.get(timeout=120)
    naive_process.join()
    assert naive_result["slots"] == 0, "an un-precomputed model answered inside a fork"
    assert naive_result["workers"] == 0, "an un-precomputed model started a worker inside a fork"
    print("un-precomputed model inside a fork: refused, 0 workers")

    for result in results:
        assert result["cached_slots"] > 0, f"child {result['pid']} lost the inherited cache"
        assert result["missing_slots"] == 0, f"child {result['pid']} answered for a day it never had"
        assert result["workers"] == 0, f"child {result['pid']} started {result['workers']} worker(s)"
    print(f"{len(results)} forked children: served from the inherited cache, started 0 workers, "
          f"returned no signal for an unknown day")
    print("OK")
