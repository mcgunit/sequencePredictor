#!/usr/bin/env bash
# One backtest on the GPU worker (owner, 10 Nov 2026; docs/workers.md). workers.js runs this in its own process group
# with the worker's settings in the environment (config/workers.json):
#   WORKER_HOST    ssh target, e.g. root@<the LXC>        WORKER_DIR     the worker's folder (holds venv/ and app/)
#   WORKER_PYTHON  the worker's python                    WORKER_GPU_MB  TensorFlow's cap on the shared card
#
#   gpu-backtest.sh <backtest id> <budget seconds> [run id]   (the run id only marks the process for workers.js)
#
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
GPU_MB="${WORKER_GPU_MB:-6144}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REC="config/backtests/$ID.json"
APP="$WORKER_DIR/app"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
say() { printf '%s [gpu worker] %s\n' "$(date -u '+%F %T')" "$*"; }
pull() {   # the worker's record -> ours, replaced in one step
  rsync -a --timeout=60 -e "$SSH" "$WORKER_HOST:$APP/$REC" "$ROOT/$REC.worker" 2>/dev/null && mv -f "$ROOT/$REC.worker" "$ROOT/$REC"
}
remote=""
stopping() {
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

say "copying the code and the training data to $WORKER_HOST:$APP"
$SSH "$WORKER_HOST" "mkdir -p '$APP/config/backtests' '$APP/data/trainingData'" || { say "cannot reach $WORKER_HOST"; exit 90; }
rsync -a --delete --timeout=120 -e "$SSH" \
  --exclude=/.git --exclude=node_modules --exclude=/web --exclude=/log --exclude=/data --exclude=/config \
  --exclude=/test --exclude=/inspiration --exclude=__pycache__ --exclude='/db.sqlite3*' --exclude=/process.lock \
  --exclude=/backtest.pid --exclude='*.keras' --exclude=/run.log \
  "$ROOT/" "$WORKER_HOST:$APP/" || { say "copying the code failed"; exit 91; }
rsync -a --delete --timeout=120 -e "$SSH" "$ROOT/data/trainingData/" "$WORKER_HOST:$APP/data/trainingData/" \
  || { say "copying the training data failed"; exit 91; }
rsync -a --timeout=60 -e "$SSH" "$ROOT/$REC" "$WORKER_HOST:$APP/config/backtests/" || { say "copying the record failed"; exit 91; }
if [ -f "$ROOT/config/paramOverrides.json" ]; then
  rsync -a --timeout=60 -e "$SSH" "$ROOT/config/paramOverrides.json" "$WORKER_HOST:$APP/config/" || { say "copying the manual values failed"; exit 91; }
else
  $SSH "$WORKER_HOST" "rm -f '$APP/config/paramOverrides.json'"
fi

say "starting RunBacktest.py --id $ID --budget $BUDGET on $WORKER_HOST (GPU memory cap ${GPU_MB} MB)"
$SSH "$WORKER_HOST" "cd '$APP' && echo \$\$ > backtest.pid && exec env CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=false SP_GPU_MEMORY_MB='$GPU_MB' PYTHONUNBUFFERED=1 '$WORKER_PYTHON' RunBacktest.py --id '$ID' --budget '$BUDGET'" &
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
