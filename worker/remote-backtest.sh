#!/usr/bin/env bash
# One backtest on a worker machine (owner, 10 Nov 2026; docs/operations.md, The workers): the GPU worker (the 2080
# Ti LXC) or the CPU worker (the DL360). workers.js runs this in its own process group with the worker's settings in
# the environment (config/workers.json):
#   WORKER_HOST        ssh target, e.g. root@<address>     WORKER_DIR     the worker's folder (holds venv/ and app/)
#   WORKER_PYTHON      the worker's python                  WORKER_REMOTE_ENV  NAME=value pairs for the remote run
#   WORKER_ILO_ENV     (the CPU worker) the iLO's settings: the machine is switched on when it does not answer
#                      (worker/power.sh wake), and switches itself off when idle
#
#   remote-backtest.sh <backtest id> <budget seconds> [run id]   (the run id only marks the process for workers.js)
#
# 0. (CPU worker) wakes the machine and waits for it (up to 15 minutes), the record saying so;
# 1. copies this checkout's code, data/trainingData, the manual values and the backtest's record to <dir>/app
#    (never config/ otherwise: it holds the accounts and the session secret; never data/ otherwise, nor .git);
# 2. runs RunBacktest.py there, over ssh, and copies the record back every 15 s (progress) and at the end (result);
# 3. on SIGTERM to this script alone (Stop on the Backtests page; not to its process group, which would take the ssh
#    session down first and leave the remote run without its stop) signals the remote run, which stops its training child and writes
#    "stopped", and copies that back.
# The main server stays the only writer of data/: the worker returns the record file and nothing else.
set -u
ID="${1:?backtest id}"
BUDGET="${2:?budget seconds}"
: "${WORKER_HOST:?WORKER_HOST not set}" "${WORKER_DIR:?WORKER_DIR not set}" "${WORKER_PYTHON:?WORKER_PYTHON not set}"
REMOTE_ENV="${WORKER_REMOTE_ENV:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REC="config/backtests/$ID.json"
APP="$WORKER_DIR/app"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
say() { printf '%s [worker] %s\n' "$(date -u '+%F %T')" "$*"; }
edit() {   # the local record: phase <text> (a word for the Backtests page while there is no progress yet), or failed <text>
  python3 - "$ROOT/$REC" "$1" "$2" <<'PY' 2>/dev/null
import json, os, sys
from datetime import datetime, timezone
path, kind, text = sys.argv[1:4]
with open(path) as f:
    rec = json.load(f)
if kind in ("failed", "stopped"):
    rec.update(status=kind, error=text, phase=None, finishedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"))
else:
    rec["phase"] = text or None
with open(path + ".edit", "w") as f:
    json.dump(rec, f, indent=1)
os.replace(path + ".edit", path)
PY
}
note() { edit phase "$1"; }
fail() { say "$2"; edit failed "$2"; exit "$1"; }
pull() {   # the worker's record -> ours, replaced in one step
  rsync -a --timeout=60 -e "$SSH" "$WORKER_HOST:$APP/$REC" "$ROOT/$REC.worker" 2>/dev/null && mv -f "$ROOT/$REC.worker" "$ROOT/$REC"
}
remote=""
stopping() {
  if [ -z "$remote" ]; then
    say "stop asked before the run started"
    edit stopped "stopped by an administrator before it started"
    exit 130
  fi
  say "stop asked: signalling the backtest on $WORKER_HOST"
  $SSH "$WORKER_HOST" "kill -TERM \$(cat '$APP/backtest.pid' 2>/dev/null) 2>/dev/null"
  for _ in $(seq 1 30); do
    [ -n "$remote" ] && kill -0 "$remote" 2>/dev/null || break
    sleep 2
  done
  pull
  exit 130
}
trap stopping TERM INT

if [ -n "${WORKER_ILO_ENV:-}" ]; then
  if ! $SSH -o ConnectTimeout=8 "$WORKER_HOST" true 2>/dev/null; then
    note "switching the worker on (5 to 10 minutes)"
    bash "$ROOT/worker/power.sh" wake || fail 93 "the worker did not come up within 15 minutes (is a USB disk plugged in on the DL360?), or its iLO did not answer - see log/backtest-cpu.log"
  fi
fi
note "copying to the worker"
say "copying the code and the training data to $WORKER_HOST:$APP"
$SSH "$WORKER_HOST" "mkdir -p '$APP/config/backtests' '$APP/data/trainingData'" || fail 90 "the worker ($WORKER_HOST) does not answer"
rsync -a --delete --timeout=120 -e "$SSH" \
  --exclude=/.git --exclude=node_modules --exclude=/web --exclude=/log --exclude=/data --exclude=/config \
  --exclude=/test --exclude=/inspiration --exclude=__pycache__ --exclude='/db.sqlite3*' --exclude=/process.lock \
  --exclude=/backtest.pid --exclude='*.keras' --exclude=/run.log \
  "$ROOT/" "$WORKER_HOST:$APP/" || fail 91 "copying the code to the worker failed"
rsync -a --delete --timeout=120 -e "$SSH" "$ROOT/data/trainingData/" "$WORKER_HOST:$APP/data/trainingData/" \
  || fail 91 "copying the training data to the worker failed"
note ""
rsync -a --timeout=60 -e "$SSH" "$ROOT/$REC" "$WORKER_HOST:$APP/config/backtests/" || fail 91 "copying the record to the worker failed"
if [ -f "$ROOT/config/paramOverrides.json" ]; then
  rsync -a --timeout=60 -e "$SSH" "$ROOT/config/paramOverrides.json" "$WORKER_HOST:$APP/config/" || fail 91 "copying the manual values to the worker failed"
else
  $SSH "$WORKER_HOST" "rm -f '$APP/config/paramOverrides.json'"
fi

case "$REMOTE_ENV" in *[!A-Za-z0-9_=.\ -]*) say "WORKER_REMOTE_ENV may hold NAME=value pairs only"; exit 2 ;; esac
say "starting RunBacktest.py --id $ID --budget $BUDGET on $WORKER_HOST${REMOTE_ENV:+ ($REMOTE_ENV)}"
$SSH "$WORKER_HOST" "cd '$APP' && echo \$\$ > backtest.pid && exec env $REMOTE_ENV PYTHONUNBUFFERED=1 '$WORKER_PYTHON' RunBacktest.py --id '$ID' --budget '$BUDGET'" &
remote=$!
while kill -0 "$remote" 2>/dev/null; do
  sleep 15 &
  wait $!
  pull
done
wait "$remote"
status=$?
pull
say "RunBacktest.py on $WORKER_HOST ended with status $status"
exit "$status"
