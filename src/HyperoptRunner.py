"""
Shared Optuna run machinery for the Hyperopt* tuners.

HyperoptBoost, HyperoptStatistics, HyperoptRLTicket and HyperoptQuantum run
every tuning trial through optimize_study(); HyperoptDeepLearning keeps its
GPU-serial spawned child per trial and uses only the study helpers.

What one process per trial buys, measured on the 2026-09-09 production
HyperoptBoost run (keno: 14 hours, 7 of them timeouts) and its predecessors:

- Parallel trials where the objective is single-threaded (RL, quantum,
  XGBoost/LightGBM fits): how many is a per-tuner decision, and every launch
  beyond the first is gated on measured memory (PSS of the running trial
  trees), because this 16 GB container has been OOM-killed before.
- Crash isolation: a trial process that dies (exception, OOM kill) costs that
  trial, which is marked FAILED here; previously one exception aborted every
  remaining strategy of the game.
- Clean teardown: Ctrl+C or `kill <pid>` of the coordinator kills the trial
  trees (Backtester workers included, which would otherwise keep fitting as
  orphans) and the trial processes die with the parent (PR_SET_PDEATHSIG).
- Stale trials: a killed run leaves its trials RUNNING in db.sqlite3;
  fail_stale_running_trials marks them FAILED at the next start - required
  once TPESampler(constant_liar) is on, which treats RUNNING trials as live.
- A trial the tuner's own gate skipped (mark_trial_skipped, e.g. HyperoptBoost's
  predicted timeouts) is not counted toward n_trials, so a study still
  evaluates the requested number of real configurations.
- Optional wall-clock deadline per trial (trial_timeout_seconds): the tree is
  killed and the trial recorded as PRUNED with user_attr timeout=True - for
  objectives without an internal budget (HyperoptBoost enforces its own
  through the Backtester and records the same attribute).

Fork, not spawn: the objective is passed as a live callable (closures over
already-loaded data tables are the norm here) and the Backtester hands its
worker state to its pool through a module global - both need copy-on-write
inheritance. Nothing in a coordinator has run library fits before forking,
which is the same discipline Backtester.py relies on.
"""
import os
import sys
import time
import signal
import warnings
import multiprocessing

import optuna

# TPESampler(constant_liar=True) is flagged experimental by Optuna; the
# warning would otherwise print once per trial process.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

EXIT_TRIAL_SKIPPED = 3
_LAST_TRIAL_SKIPPED = False


def mark_trial_skipped():
    """
    Called by an objective, inside its trial process, right before it raises
    optuna.TrialPruned for a trial it decided not to evaluate at all; the
    coordinator then doesn't count that trial toward n_trials.
    """
    global _LAST_TRIAL_SKIPPED
    _LAST_TRIAL_SKIPPED = True


def make_storage(url):
    """
    One RDBStorage per process - a SQLAlchemy engine must not be used across a
    fork. The sqlite busy timeout is raised from Python's 5 s default: with
    concurrent trial processes each writing intermediate values, a write can
    briefly find the file locked and should wait, not raise.
    """
    return optuna.storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 60}})


def open_study(study_name, storage_url, parallel=1, pruner=None, quiet=False):
    """
    Study handle with the sampler this run uses (pruner as given, none by
    default). Sampler and pruner live in the process, not the db, so every
    trial process builds the same ones. constant_liar makes TPE treat trials
    other processes are still running as pessimistic observations, so
    concurrent trials don't sample the same neighbourhood (Optuna's documented
    setting for parallel optimization).
    """
    sampler = optuna.samplers.TPESampler(constant_liar=parallel > 1)
    verbosity = optuna.logging.get_verbosity()
    if quiet:
        # Every trial process re-opens the study; one "Using an existing
        # study" line per trial is noise.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        return optuna.create_study(
            direction='maximize',
            storage=make_storage(storage_url),
            study_name=study_name,
            load_if_exists=True,
            pruner=pruner if pruner is not None else optuna.pruners.NopPruner(),
            sampler=sampler,
        )
    finally:
        optuna.logging.set_verbosity(verbosity)


def fail_stale_running_trials(study):
    """
    process.lock guarantees a single hyperopt process at a time, so a trial
    still RUNNING when its study is opened was left behind by a killed run
    (Ctrl+C, reboot, OOM). Marked failed: constant_liar would otherwise treat
    it as live forever, and study.best_params ignores it either way.
    """
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        print(f"Marking stale trial {t.number} of {study.study_name} as failed (left by an earlier run)")
        study.tell(t.number, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)


def has_completed_trials(study):
    return any(t.state == optuna.trial.TrialState.COMPLETE for t in study.get_trials(deepcopy=False))


# --- signals -----------------------------------------------------------------

def _exit_on_sigterm(signum, frame):
    """
    SIGTERM (kill <pid>) as an exception instead of an instant death, so the
    coordinator's cleanup runs (trial trees killed, process.lock removed by
    the caller's finally) and a trial process unwinds through the
    Backtester's `with Pool`, terminating its workers. Inherited by the
    forked trial processes - and by their pool workers, where it must NOT
    fire: a worker is inside a library fit, and an exception raised from a C
    callback there corrupted the heap on the way out (glibc "corrupted size
    vs. prev_size"). Workers take the default instant death instead.
    """
    if multiprocessing.current_process().name.startswith("ForkPoolWorker"):
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    raise SystemExit(128 + signum)


def install_sigterm_handler():
    signal.signal(signal.SIGTERM, _exit_on_sigterm)


def _die_with_parent():
    """Linux prctl(PR_SET_PDEATHSIG, SIGTERM): the trial process ends when the coordinator does."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


# --- memory and process trees (ps is unreliable in this container) ------------

def _read_meminfo():
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    return values


def total_memory_gb():
    try:
        return _read_meminfo()["MemTotal"] / 2 ** 30
    except (OSError, KeyError, ValueError):
        return 0.0


def available_memory_gb():
    """MemAvailable, further capped by the cgroup v2 limit when one is set."""
    try:
        available = _read_meminfo()["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return float("inf")
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read().strip())
        if limit != "max":
            available = min(available, max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    return available / 2 ** 30


def _process_table():
    """{pid: (ppid, state)} straight from /proc."""
    table = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        ppid, state = None, "?"
        try:
            with open(f"/proc/{name}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("State:"):
                        state = line.split()[1]
        except (OSError, ValueError, IndexError):
            continue
        if ppid is not None:
            table[int(name)] = (ppid, state)
    return table


def _descendants(pid, table):
    found, stack = [], [pid]
    while stack:
        parent = stack.pop()
        kids = [p for p, (pp, _) in table.items() if pp == parent]
        found.extend(kids)
        stack.extend(kids)
    return found


def _process_memory_bytes(pid):
    """
    PSS (shared pages counted proportionally) when the kernel offers it, RSS
    otherwise. Forked trial processes share most of their pages with the
    coordinator copy-on-write; RSS would count those pages once per process
    and make one 300 MB parent look like N x 300 MB.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def tree_memory_gb(pid):
    """Memory of a trial process plus everything it forked (Backtester workers)."""
    table = _process_table()
    return sum(_process_memory_bytes(p) for p in [pid] + _descendants(pid, table)) / 2 ** 30


def kill_tree(pid, grace_seconds=5):
    """
    SIGTERM a trial process together with its workers - killed on their own,
    the workers would keep fitting as orphans - then SIGKILL what survives
    the grace period.
    """
    table = _process_table()
    pids = _descendants(pid, table) + [pid]
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        table = _process_table()
        if not any(p in table and not table[p][1].startswith("Z") for p in pids):
            return
        time.sleep(0.2)
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


# --- the coordinator ------------------------------------------------------------

def _trial_process(study_name, storage_url, objective, parallel, pruner_factory):
    """
    Body of one trial process (forked by optimize_study): open the study, run
    exactly one trial, exit. Exit code EXIT_TRIAL_SKIPPED tells the parent the
    objective's own gate skipped this trial without evaluating it.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    _die_with_parent()
    pruner = pruner_factory() if pruner_factory is not None else None
    study = open_study(study_name, storage_url, parallel=parallel, pruner=pruner, quiet=True)

    def run_trial(trial):
        # Lets the coordinator find this trial if the process dies (see mark_trial_of).
        trial.set_user_attr("worker_pid", os.getpid())
        return objective(trial)

    study.optimize(run_trial, n_trials=1)
    if _LAST_TRIAL_SKIPPED:
        sys.exit(EXIT_TRIAL_SKIPPED)


def optimize_study(study_name, storage_url, objective, n_trials, parallel=1, pruner_factory=None,
                   expected_trial_gb=0.5, memory_reserve_gb=2.0, memory_hard_floor_gb=1.0,
                   ramp_seconds=3, trial_timeout_seconds=None):
    """
    Runs n_trials evaluated trials of one study, up to `parallel` at a time,
    each in its own forked process.

    - `objective(trial)` is called inside the trial process; anything it
      closes over is inherited through the fork (read-only from its point of
      view - writes never reach the coordinator or other trials).
    - Trials the objective skipped via mark_trial_skipped don't count toward
      n_trials; attempts are capped at 4x n_trials so a study can't spin.
    - A further trial launches only when the memory gate allows: the largest
      footprint measured on the running trials (expected_trial_gb until
      measured) must fit with memory_reserve_gb to spare; under
      memory_hard_floor_gb the youngest trial is stopped and recorded as
      pruned. Launches are spaced ramp_seconds apart so a fresh trial has
      loaded its data before its footprint is read.
    - trial_timeout_seconds (optional): a trial older than this is killed and
      recorded as pruned with user_attr timeout=True.
    - A trial process that dies (exception, OOM kill) leaves its trial
      RUNNING; it is marked failed here, and three failures in a row give up
      on this study instead of the whole run.
    """
    footprint_seen = float(expected_trial_gb)
    print(f"{study_name}: {n_trials} trials, up to {parallel} at a time, "
          f"memory reserve {memory_reserve_gb:g} GB, {available_memory_gb():.1f} GB available")

    def mark_trial_of(pid, state, note, **attrs):
        study = open_study(study_name, storage_url, parallel=parallel, quiet=True)
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            if t.user_attrs.get("worker_pid") == pid:
                print(f"Trial {t.number} {note}")
                try:
                    for key, value in attrs.items():
                        study._storage.set_trial_user_attr(t._trial_id, key, value)
                except Exception as e:  # attributes are diagnostics only
                    print(f"Could not record attributes on trial {t.number}: {e}")
                study.tell(t.number, state=state, skip_if_finished=True)

    ctx = multiprocessing.get_context("fork")
    running = {}  # pid -> (Process, launched_at)
    evaluated = attempts = failures_in_a_row = 0
    max_attempts = n_trials * 4
    aborted = False

    def launch_allowed():
        if not running:
            return True
        newest = max(launched_at for _, launched_at in running.values())
        if time.time() - newest < ramp_seconds:
            return False
        return available_memory_gb() - footprint_seen >= memory_reserve_gb

    def stop_trial(pid, state, note, **attrs):
        nonlocal footprint_seen
        footprint_seen = max(footprint_seen, tree_memory_gb(pid))
        kill_tree(pid)
        running.pop(pid)[0].join()
        mark_trial_of(pid, state, note, **attrs)

    try:
        while True:
            for pid, (proc, _) in list(running.items()):
                if proc.is_alive():
                    continue
                proc.join()
                del running[pid]
                if proc.exitcode == EXIT_TRIAL_SKIPPED:
                    continue
                evaluated += 1
                if proc.exitcode == 0:
                    failures_in_a_row = 0
                    continue
                failures_in_a_row += 1
                mark_trial_of(pid, optuna.trial.TrialState.FAIL,
                              f"marked failed - its process exited with code {proc.exitcode}")
                if failures_in_a_row >= 3:
                    print(f"{study_name}: three trial processes failed in a row - giving up on this study")
                    aborted = True

            want_more = not aborted and evaluated + len(running) < n_trials and attempts < max_attempts
            if not want_more and not running:
                break

            if want_more and len(running) < parallel and launch_allowed():
                sys.stdout.flush()
                sys.stderr.flush()
                proc = ctx.Process(target=_trial_process, name=f"trial:{study_name}",
                                   args=(study_name, storage_url, objective, parallel, pruner_factory))
                proc.start()
                running[proc.pid] = (proc, time.time())
                attempts += 1
                continue

            if running:
                now = time.time()
                settled = [pid for pid, (_, launched_at) in running.items() if now - launched_at >= ramp_seconds]
                footprint_seen = max([footprint_seen] + [tree_memory_gb(pid) for pid in settled])

                if trial_timeout_seconds:
                    for pid, (_, launched_at) in list(running.items()):
                        if now - launched_at > trial_timeout_seconds:
                            evaluated += 1  # a real evaluation that ran out of budget
                            stop_trial(pid, optuna.trial.TrialState.PRUNED,
                                       f"pruned - exceeded the {trial_timeout_seconds:.0f}s trial budget",
                                       timeout=True, seconds=round(now - launched_at, 1))

                if len(running) > 1 and available_memory_gb() < memory_hard_floor_gb:
                    youngest = max(running, key=lambda p: running[p][1])
                    print(f"{study_name}: {available_memory_gb():.1f} GB available - stopping the youngest "
                          f"concurrent trial (process {youngest}) before the OOM killer does")
                    stop_trial(youngest, optuna.trial.TrialState.PRUNED, "pruned - stopped under memory pressure")
            time.sleep(2)
    except BaseException:
        # Ctrl+C or a crash of the coordinator: never leave trial processes
        # (and their worker pools) computing for a run that is gone.
        for pid, (proc, _) in running.items():
            kill_tree(pid)
            proc.join(timeout=10)
        raise
