#!/usr/bin/env bash
# One hyperopt started from the Hyperopts page, on a worker (docs/operations.md, The workers): the deep-learning tuner
# on the GPU worker (the 2080 Ti LXC; owner, 11 Nov 2026), the other five tuners on the CPU worker (the DL360; owner,
# 12 Nov 2026) - or, when the DL360 does not come up, on this server in a folder of its own (WORKER_HOST=local, the
# owner's fallback). workers.js runs this in its own process group with the worker's settings in the environment, as
# it runs remote-backtest.sh:
#   WORKER_HOST  ssh target, or "local"    WORKER_DIR  the worker's folder (venv/, app/ for the backtests, hyperopt/ for these)
#   WORKER_PYTHON  the worker's python    WORKER_REMOTE_ENV  NAME=value pairs for the run (the GPU and its memory cap)
#   WORKER_ILO_ENV  (the DL360) its iLO's settings: switched on when it does not answer (worker/power.sh wake)
#
#   remote-hyperopt.sh <hyperopt id> <budget seconds> <run id> <tuner script> <tuner arguments...>
#   (without a Hyperopt*.py fourth argument - a run queued before 12 Nov 2026 - the tuner is HyperoptDeepLearning.py)
#
# 0. (the DL360) switches it on when it does not answer; when it does not come up, or cannot be copied to, the script
#    ends with 75 before anything ran there, and workers.js runs the hyperopt on this server instead;
# 1. copies this checkout's code (with the served bestParams_<game>.json: the gate's incumbent), data/trainingData
#    and - for the tuners that read the stored predictions (ensemble, RL ticket, quantum) - data/database to
#    <dir>/hyperopt. What stays there between runs: db.sqlite3 (the Optuna studies - seeded once from this server's,
#    a consistent copy, so the sampler keeps the studies' history) and data/hyperOptCache (the deep-learning run
#    marker: the same settings after a stop resume the run, --fresh starts over);
# 2. starts the tuner there detached from the ssh session (a network blip does not kill a two-day run), with
#    --save false (no commit, no push) and --result-file, under `timeout <budget>`;
# 3. copies its result file back to config/hyperopts/<id>.result.json and its output into the run's log every 15 s,
#    and marks the record (config/hyperopts/<id>.json) done, failed, out of time or stopped at the end.
# Record only (owner): the worker's copy of bestParams_<game>.json is thrown away - the next copy replaces it - and
# nothing that is served changes until Apply on the Hyperopts page. The main server stays the only writer of data/.
# On SIGTERM (Stop on the page, to this script alone) it stops the tuner there (it kills its children and keeps
# its finished trials), copies the result back and ends with 130.
set -u
ID="${1:?hyperopt id}"; BUDGET="${2:?budget seconds}"; shift 3   # the run id ($3) only marks the process for workers.js
TUNER="HyperoptDeepLearning.py"
case "${1:-}" in Hyperopt*.py) TUNER="$1"; shift ;; esac
: "${WORKER_HOST:?WORKER_HOST not set}" "${WORKER_DIR:?WORKER_DIR not set}" "${WORKER_PYTHON:?WORKER_PYTHON not set}"
REMOTE_ENV="${WORKER_REMOTE_ENV:-}"
NOT_STARTED=75   # workers.js: the DL360 did not come up or could not be copied to - run it on this server
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REC="config/hyperopts/$ID.json"
RESULT="config/hyperopts/$ID.result.json"
APP="$WORKER_DIR/hyperopt"
LOST_AFTER="${WORKER_LOST_AFTER_SECONDS:-1800}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
if [ "$WORKER_HOST" = local ]; then
  WORD="this server"
  on() { bash -c "$1"; }                       # a command in the run's folder on this machine
  RSYNC_TO=""; RSYNC_E=()
else
  if [ -n "${WORKER_ILO_ENV:-}" ]; then WORD="the CPU worker"; else WORD="the GPU worker"; fi
  on() { $SSH "$WORKER_HOST" "$1"; }
  RSYNC_TO="$WORKER_HOST:"; RSYNC_E=(-e "$SSH")
fi
say() { printf '%s [%s] %s\n' "$(date -u '+%F %T')" "${WORD#the }" "$*"; }

case "$ID" in *[!A-Za-z0-9_-]*|'') say "not a hyperopt id: $ID"; exit 2 ;; esac
case "$TUNER" in HyperoptDeepLearning.py|HyperoptStatistics.py|HyperoptBoost.py|HyperoptEnsemble.py|HyperoptQuantum.py|HyperoptRLTicket.py) ;;
  *) say "not a tuner: $TUNER"; exit 2 ;; esac
case "$BUDGET" in *[!0-9]*|'') say "not a budget: $BUDGET"; exit 2 ;; esac
for a in "$@"; do case "$a" in *[!A-Za-z0-9,._-]*) say "an argument may hold letters, digits and , . _ - only: $a"; exit 2 ;; esac; done
case "$REMOTE_ENV" in *[!A-Za-z0-9_=.\ -]*) say "WORKER_REMOTE_ENV may hold NAME=value pairs only"; exit 2 ;; esac
[ -f "$ROOT/$REC" ] || { say "no record $REC"; exit 2; }

edit() {   # the local record: phase <text>, or <status> <text> (done, failed, out of time, stopped, lost)
  python3 - "$ROOT/$REC" "$1" "$2" <<'PY' 2>/dev/null
import json, os, sys
from datetime import datetime, timezone
path, kind, text = sys.argv[1:4]
with open(path) as f:
    rec = json.load(f)
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
if kind == "phase":
    rec["phase"] = text or None
elif kind == "running":
    rec.update(status="running", phase=None, startedAt=rec.get("startedAt") or now)
else:
    rec.update(status=kind, error=text or None, phase=None, finishedAt=now)
with open(path + ".edit", "w") as f:
    json.dump(rec, f, indent=1)
os.replace(path + ".edit", path)
PY
}
fail() { say "$1"; edit failed "$1"; exit "${2:-1}"; }
offset=0
fetch() {   # the new part of the tuner's output into this log, and its result file
  local size
  size="$(on "stat -c %s '$APP/hyperopt.out' 2>/dev/null || echo 0" 2>/dev/null)" || return 1
  if [ "${size:-0}" -gt "$offset" ] 2>/dev/null; then
    on "tail -c +$((offset + 1)) '$APP/hyperopt.out' | head -c $((size - offset))" 2>/dev/null && offset="$size"
  fi
  rsync -a --timeout=60 "${RSYNC_E[@]}" "$RSYNC_TO$APP/results/$ID.json" "$ROOT/$RESULT.worker" 2>/dev/null \
    && mv -f "$ROOT/$RESULT.worker" "$ROOT/$RESULT"
  return 0
}
started=""
stopping() {
  if [ -z "$started" ]; then say "stop asked before the tuner started"; edit stopped "stopped by an administrator before it started"; exit 130; fi
  say "stop asked: stopping the tuner on $WORD (its finished trials stay in the study)"
  on "kill -TERM \$(cat '$APP/hyperopt.pid' 2>/dev/null) 2>/dev/null"
  for _ in $(seq 1 45); do
    on "test -f '$APP/hyperopt.exit'" 2>/dev/null && break
    sleep 2
  done
  fetch
  edit stopped "stopped by an administrator - the same settings without Fresh resume the run"
  exit 130
}
trap stopping TERM INT

# 0. the DL360: switched on when it does not answer; not coming up = run it on this server (workers.js)
if [ -n "${WORKER_ILO_ENV:-}" ] && [ "$WORKER_HOST" != local ]; then
  if ! $SSH -o ConnectTimeout=8 "$WORKER_HOST" true 2>/dev/null; then
    edit phase "switching the DL360 on (5 to 10 minutes)"
    say "switching the DL360 on (5 to 10 minutes)"
    if ! bash "$ROOT/worker/power.sh" wake; then
      say "the DL360 did not come up (is a USB disk plugged in?) or its iLO did not answer: running it on this server instead"
      edit phase "the DL360 did not come up: moving to this server"
      exit "$NOT_STARTED"
    fi
  fi
fi
# a copy that fails before the tuner ran: on the DL360 the hyperopt moves to this server, elsewhere it fails
early() { if [ -n "${WORKER_ILO_ENV:-}" ] && [ "$WORKER_HOST" != local ]; then say "$1 - running it on this server instead"; edit phase "moving to this server"; exit "$NOT_STARTED"; fi; fail "$1" "$2"; }

# 1. the code, the data, the studies
edit phase "copying to $WORD"
say "copying the code and the data to ${RSYNC_TO}$APP"
on "mkdir -p '$APP/data/trainingData' '$APP/data/database' '$APP/data/hyperOptCache' '$APP/results'" || early "$WORD ($WORKER_HOST) does not answer" 90
if on "test -f '$APP/hyperopt.pid' && ! test -f '$APP/hyperopt.exit' && kill -0 \$(cat '$APP/hyperopt.pid') 2>/dev/null"; then
  fail "a hyperopt is still running on $WORD (started earlier and lost from view?) - see $APP/hyperopt.out there" 92
fi
rsync -a --delete --timeout=120 "${RSYNC_E[@]}" \
  --exclude=/.git --exclude=node_modules --exclude=/web --exclude=/log --exclude=/data --exclude=/config \
  --exclude=/test --exclude=/inspiration --exclude=__pycache__ --exclude='/db.sqlite3*' --exclude=/process.lock \
  --exclude=/results --exclude='/hyperopt.*' --exclude='*.keras' --exclude=/run.log --exclude='/wget*' \
  "$ROOT/" "$RSYNC_TO$APP/" || early "copying the code to $WORD failed" 91
rsync -a --delete --timeout=120 "${RSYNC_E[@]}" "$ROOT/data/trainingData/" "$RSYNC_TO$APP/data/trainingData/" \
  || early "copying the training data to $WORD failed" 91
if [ "$TUNER" != HyperoptDeepLearning.py ]; then
  # the stored predictions and their results: the ensemble, RL ticket and quantum tuners score from them
  rsync -a --delete --timeout=300 "${RSYNC_E[@]}" "$ROOT/data/database/" "$RSYNC_TO$APP/data/database/" \
    || early "copying the stored predictions to $WORD failed" 91
fi
if ! on "test -s '$APP/db.sqlite3'"; then
  if [ -f "$ROOT/db.sqlite3" ]; then
    say "first hyperopt there: copying this server's Optuna studies (db.sqlite3, a consistent copy)"
    SEED="$(mktemp "${TMPDIR:-/tmp}/hyperopt-seed.XXXXXX")"
    python3 - "$ROOT/db.sqlite3" "$SEED" <<'PY' || { rm -f "$SEED"; early "could not copy db.sqlite3" 91; }
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=60)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
PY
    rsync -a --timeout=300 "${RSYNC_E[@]}" "$SEED" "$RSYNC_TO$APP/db.sqlite3" || { rm -f "$SEED"; early "copying db.sqlite3 to $WORD failed" 91; }
    rm -f "$SEED"
  fi
fi

# 2. the run, detached from this session: hyperopt.pid (the tuner), hyperopt.out (its output), hyperopt.exit.
#    On this server it runs niced: the scheduler's jobs come first.
NICE=""; [ "$WORKER_HOST" = local ] && NICE="nice -n 10"
ARGS="$* --save false --result-file results/$ID.json"
say "starting $TUNER $ARGS on $WORD${REMOTE_ENV:+ ($REMOTE_ENV)}, $(( BUDGET / 3600 )) h at most"
on "cd '$APP' && rm -f hyperopt.exit hyperopt.out hyperopt.pid 'results/$ID.json' && { setsid nohup bash -c 'env $REMOTE_ENV PYTHONUNBUFFERED=1 $NICE timeout --signal=TERM --kill-after=300 $BUDGET \"$WORKER_PYTHON\" $TUNER $ARGS > hyperopt.out 2>&1 & echo \$! > hyperopt.pid; wait \$!; echo \$? > hyperopt.exit.tmp; mv hyperopt.exit.tmp hyperopt.exit' > /dev/null 2>&1 < /dev/null & }" \
  || fail "the tuner could not be started on $WORD" 91
started=1
edit running ""
contact=$(date +%s)
status=""
while :; do
  sleep 15 & wait $!
  if fetch; then contact=$(date +%s); fi
  code="$(on "cat '$APP/hyperopt.exit' 2>/dev/null" 2>/dev/null)" && contact=$(date +%s)
  if [ -n "$code" ]; then status="$code"; break; fi
  if [ $(( $(date +%s) - contact )) -ge "$LOST_AFTER" ]; then
    edit lost "no answer from $WORD for $(( LOST_AFTER / 60 )) minutes - the tuner may still be running there ($APP/hyperopt.out)"
    say "no answer from $WORD for $(( LOST_AFTER / 60 )) minutes: given up"
    exit 1
  fi
done
fetch
say "$TUNER on $WORD ended with status $status"

# 3. the record
case "$status" in
  0) edit done "" ;;
  124|137) edit "out of time" "over its budget of $(( BUDGET / 3600 )) h - the finished trials stay in the study; the same settings resume the run" ;;
  130|143) edit stopped "stopped on $WORD - the same settings without Fresh resume the run" ;;
  *) edit failed "the tuner ended with status $status - see its log" ;;
esac
exit "$status"
