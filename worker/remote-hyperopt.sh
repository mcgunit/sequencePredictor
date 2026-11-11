#!/usr/bin/env bash
# One deep-learning hyperopt on the GPU worker (the 2080 Ti LXC; the owner's decision of 11 Nov 2026, docs/operations.md,
# The workers). workers.js runs this in its own process group with the worker's settings in the environment, as it runs
# remote-backtest.sh:
#   WORKER_HOST  ssh target    WORKER_DIR  the worker's folder (venv/, app/ for the backtests, hyperopt/ for these)
#   WORKER_PYTHON  the worker's python    WORKER_REMOTE_ENV  NAME=value pairs for the run (the GPU and its memory cap)
#
#   remote-hyperopt.sh <hyperopt id> <budget seconds> <run id> <tuner arguments...>
#
# 1. copies this checkout's code (with the served bestParams_<game>.json: the gate's incumbent) and data/trainingData
#    to <dir>/hyperopt. What stays there between runs: db.sqlite3 (the Optuna studies - seeded once from this server's,
#    a consistent copy, so the sampler keeps the studies' history) and data/hyperOptCache (the run marker: the same
#    settings after a stop resume the run, HyperoptDeepLearning.py --fresh starts over);
# 2. starts HyperoptDeepLearning.py there detached from the ssh session (a network blip does not kill a two-day run),
#    with -s false (no commit, no push) and --result-file, under `timeout <budget>`;
# 3. copies its result file back to config/hyperopts/<id>.result.json and its output into the run's log every 15 s,
#    and marks the record (config/hyperopts/<id>.json) done, failed, out of time or stopped at the end.
# Record only (owner): the worker's copy of bestParams_<game>.json is thrown away - the next copy replaces it - and
# nothing that is served changes until Apply on the Hyperopts page. The main server stays the only writer of data/.
# On SIGTERM (Stop on the page, to this script alone) it stops the tuner there (it kills its training child and keeps
# its finished trials), copies the result back and ends with 130.
set -u
ID="${1:?hyperopt id}"; BUDGET="${2:?budget seconds}"; shift 3   # the run id ($3) only marks the process for workers.js
: "${WORKER_HOST:?WORKER_HOST not set}" "${WORKER_DIR:?WORKER_DIR not set}" "${WORKER_PYTHON:?WORKER_PYTHON not set}"
REMOTE_ENV="${WORKER_REMOTE_ENV:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REC="config/hyperopts/$ID.json"
RESULT="config/hyperopts/$ID.result.json"
APP="$WORKER_DIR/hyperopt"
LOST_AFTER="${WORKER_LOST_AFTER_SECONDS:-1800}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
say() { printf '%s [GPU worker] %s\n' "$(date -u '+%F %T')" "$*"; }

case "$ID" in *[!A-Za-z0-9_-]*|'') say "not a hyperopt id: $ID"; exit 2 ;; esac
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
  size="$($SSH "$WORKER_HOST" "stat -c %s '$APP/hyperopt.out' 2>/dev/null || echo 0" 2>/dev/null)" || return 1
  if [ "${size:-0}" -gt "$offset" ] 2>/dev/null; then
    $SSH "$WORKER_HOST" "tail -c +$((offset + 1)) '$APP/hyperopt.out' | head -c $((size - offset))" 2>/dev/null && offset="$size"
  fi
  rsync -a --timeout=60 -e "$SSH" "$WORKER_HOST:$APP/results/$ID.json" "$ROOT/$RESULT.worker" 2>/dev/null \
    && mv -f "$ROOT/$RESULT.worker" "$ROOT/$RESULT"
  return 0
}
started=""
stopping() {
  if [ -z "$started" ]; then say "stop asked before the tuner started"; edit stopped "stopped by an administrator before it started"; exit 130; fi
  say "stop asked: stopping the tuner on $WORKER_HOST (its finished trials stay in the study)"
  $SSH "$WORKER_HOST" "kill -TERM \$(cat '$APP/hyperopt.pid' 2>/dev/null) 2>/dev/null"
  for _ in $(seq 1 45); do
    $SSH "$WORKER_HOST" "test -f '$APP/hyperopt.exit'" 2>/dev/null && break
    sleep 2
  done
  fetch
  edit stopped "stopped by an administrator - the same settings without Fresh resume the run"
  exit 130
}
trap stopping TERM INT

# 1. the code, the data, the studies
edit phase "copying to the GPU worker"
say "copying the code and the training data to $WORKER_HOST:$APP"
$SSH "$WORKER_HOST" "mkdir -p '$APP/data/trainingData' '$APP/data/hyperOptCache' '$APP/results'" || fail "the GPU worker ($WORKER_HOST) does not answer" 90
if $SSH "$WORKER_HOST" "test -f '$APP/process.lock' && kill -0 \$(cat '$APP/process.lock') 2>/dev/null"; then
  fail "a hyperopt is still running on the GPU worker (started earlier and lost from view?) - see $APP/hyperopt.out there" 92
fi
rsync -a --delete --timeout=120 -e "$SSH" \
  --exclude=/.git --exclude=node_modules --exclude=/web --exclude=/log --exclude=/data --exclude=/config \
  --exclude=/test --exclude=/inspiration --exclude=__pycache__ --exclude='/db.sqlite3*' --exclude=/process.lock \
  --exclude=/results --exclude='/hyperopt.*' --exclude='*.keras' --exclude=/run.log --exclude='/wget*' \
  "$ROOT/" "$WORKER_HOST:$APP/" || fail "copying the code to the GPU worker failed" 91
rsync -a --delete --timeout=120 -e "$SSH" "$ROOT/data/trainingData/" "$WORKER_HOST:$APP/data/trainingData/" \
  || fail "copying the training data to the GPU worker failed" 91
if ! $SSH "$WORKER_HOST" "test -s '$APP/db.sqlite3'"; then
  if [ -f "$ROOT/db.sqlite3" ]; then
    say "first hyperopt there: copying this server's Optuna studies (db.sqlite3, a consistent copy)"
    SEED="$(mktemp "${TMPDIR:-/tmp}/hyperopt-seed.XXXXXX")"
    python3 - "$ROOT/db.sqlite3" "$SEED" <<'PY' || { rm -f "$SEED"; fail "could not copy db.sqlite3" 91; }
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=60)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
PY
    rsync -a --timeout=300 -e "$SSH" "$SEED" "$WORKER_HOST:$APP/db.sqlite3" || { rm -f "$SEED"; fail "copying db.sqlite3 to the GPU worker failed" 91; }
    rm -f "$SEED"
  fi
fi

# 2. the run, detached from this ssh session: hyperopt.pid (the tuner), hyperopt.out (its output), hyperopt.exit
ARGS="$* -s false --result-file results/$ID.json"
say "starting HyperoptDeepLearning.py $ARGS on $WORKER_HOST${REMOTE_ENV:+ ($REMOTE_ENV)}, $(( BUDGET / 3600 )) h at most"
$SSH "$WORKER_HOST" "cd '$APP' && rm -f hyperopt.exit hyperopt.out hyperopt.pid 'results/$ID.json' && { setsid nohup bash -c 'env $REMOTE_ENV PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=300 $BUDGET \"$WORKER_PYTHON\" HyperoptDeepLearning.py $ARGS > hyperopt.out 2>&1 & echo \$! > hyperopt.pid; wait \$!; echo \$? > hyperopt.exit.tmp; mv hyperopt.exit.tmp hyperopt.exit' > /dev/null 2>&1 < /dev/null & }" \
  || fail "the tuner could not be started on the GPU worker" 91
started=1
edit running ""
contact=$(date +%s)
status=""
while :; do
  sleep 15 & wait $!
  if fetch; then contact=$(date +%s); fi
  code="$($SSH "$WORKER_HOST" "cat '$APP/hyperopt.exit' 2>/dev/null" 2>/dev/null)" && contact=$(date +%s)
  if [ -n "$code" ]; then status="$code"; break; fi
  if [ $(( $(date +%s) - contact )) -ge "$LOST_AFTER" ]; then
    edit lost "no answer from the GPU worker for $(( LOST_AFTER / 60 )) minutes - the tuner may still be running there ($APP/hyperopt.out)"
    say "no answer from the GPU worker for $(( LOST_AFTER / 60 )) minutes: given up"
    exit 1
  fi
done
fetch
say "HyperoptDeepLearning.py on $WORKER_HOST ended with status $status"

# 3. the record
case "$status" in
  0) edit done "" ;;
  124|137) edit "out of time" "over its budget of $(( BUDGET / 3600 )) h - the finished trials stay in the study; the same settings resume the run" ;;
  130|143) edit stopped "stopped on the GPU worker - the same settings without Fresh resume the run" ;;
  *) edit failed "the tuner ended with status $status - see its log" ;;
esac
exit "$status"
