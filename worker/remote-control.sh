#!/usr/bin/env bash
# One Sunday control on the CPU worker (the HP DL360; the owner's decision of 11 Nov 2026, docs/operations.md, The
# workers). jobs.js runs this with the worker's settings in the environment (config/workers.json, as workers.js gives
# them to remote-backtest.sh):
#   WORKER_HOST  ssh target    WORKER_DIR  the worker's folder (venv/, app/ for the backtests, controls/ for these)
#   WORKER_PYTHON  the worker's python    WORKER_REMOTE_ENV  NAME=value pairs for the run    WORKER_ILO_ENV  the iLO
#
#   remote-control.sh <run id> <snapshot id> <script> [arguments...]
#
# 0. wakes the DL360 when it does not answer (worker/power.sh wake, up to 15 minutes);
# 1. copies the code to <dir>/controls, and - once per snapshot id (the plan's occurrence: the four controls of a
#    Sunday read the same draws, the same cached meta-learner tables) - data/trainingData, data/hyperOptCache and the
#    manual values; data/controls every time (the records the controls read and replace);
# 2. starts the control there detached from the ssh session (a network blip does not kill a 24-hour run) and copies
#    its output into the job's log every 15 s until it ends;
# 3. copies back the records that control writes (data/controls/<its folder>) and ends with the control's exit code.
# Anything failing before the control started there ends with 75 (jobs.js REMOTE_NOT_STARTED): nothing ran on the
# DL360, and the scheduler runs this control and the ones behind it on this server instead.
# On SIGTERM (Stop on the Jobs page, to this script alone) it stops the control there and copies back what it wrote.
# The main server stays the only writer of data/: the worker returns data/controls records and nothing else.
set -u
RUN_ID="${1:?run id}"; SNAP="${2:?snapshot id}"; SCRIPT="${3:?script}"; shift 3
: "${WORKER_HOST:?WORKER_HOST not set}" "${WORKER_DIR:?WORKER_DIR not set}" "${WORKER_PYTHON:?WORKER_PYTHON not set}"
REMOTE_ENV="${WORKER_REMOTE_ENV:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$WORKER_DIR/controls"
NOT_STARTED=75
LOST_AFTER="${WORKER_LOST_AFTER_SECONDS:-1800}"     # no answer from the DL360 for this long while it runs: given up
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
say() { printf '%s [DL360] %s\n' "$(date -u '+%F %T')" "$*"; }
early() { say "$1 - nothing ran on the DL360: the control runs on this server instead"; exit "$NOT_STARTED"; }

case "$SCRIPT" in *[!A-Za-z0-9_.]*|'') say "not a script name: $SCRIPT"; exit 2 ;; esac
for a in "$@"; do case "$a" in *[!A-Za-z0-9,._-]*) say "an argument may hold letters, digits and , . _ - only: $a"; exit 2 ;; esac; done
case "$REMOTE_ENV" in *[!A-Za-z0-9_=.\ -]*) say "WORKER_REMOTE_ENV may hold NAME=value pairs only"; exit 2 ;; esac
case "$SNAP" in *[!A-Za-z0-9_.-]*) say "not a snapshot id: $SNAP"; exit 2 ;; esac

# the records each control writes (data/controls/...): copied back with --delete inside that folder only
case "$SCRIPT" in
  NullControls.py) BACK=(--exclude=/discrimination --exclude=/features --exclude=/markets); FROM="data/controls/" ;;
  RandomnessDiscrimination.py) BACK=(); FROM="data/controls/discrimination/" ;;
  IrrelevantFeatureControl.py) BACK=(); FROM="data/controls/features/" ;;
  MarketRows.py) BACK=(); FROM="data/controls/markets/" ;;
  *) say "no copy-back folder known for $SCRIPT"; exit 2 ;;
esac

started=""
offset=0
fetch() {   # the new part of the run's output, into this log
  local size
  size="$($SSH "$WORKER_HOST" "stat -c %s '$APP/control.out' 2>/dev/null || echo 0" 2>/dev/null)" || return 1
  if [ "${size:-0}" -gt "$offset" ] 2>/dev/null; then
    $SSH "$WORKER_HOST" "tail -c +$((offset + 1)) '$APP/control.out' | head -c $((size - offset))" 2>/dev/null && offset="$size"
  fi
  return 0
}
pull_back() {
  $SSH "$WORKER_HOST" "test -d '$APP/$FROM'" 2>/dev/null || { say "no $FROM on the DL360: nothing copied back"; return 1; }
  mkdir -p "$ROOT/$FROM"
  rsync -a --delete --timeout=120 -e "$SSH" "${BACK[@]}" "$WORKER_HOST:$APP/$FROM" "$ROOT/$FROM" \
    && say "copied back $FROM" || { say "copying back $FROM failed"; return 1; }
}
stopping() {
  if [ -z "$started" ]; then say "stop asked before the control started on the DL360"; exit 130; fi
  say "stop asked: stopping $SCRIPT on the DL360"
  $SSH "$WORKER_HOST" "kill -TERM \$(cat '$APP/control.pid' 2>/dev/null) 2>/dev/null"
  for _ in $(seq 1 30); do
    $SSH "$WORKER_HOST" "test -f '$APP/control.exit'" 2>/dev/null && break
    sleep 2
  done
  fetch
  pull_back
  exit 130
}
trap stopping TERM INT

# 0. the machine
if ! $SSH -o ConnectTimeout=8 "$WORKER_HOST" true 2>/dev/null; then
  [ -n "${WORKER_ILO_ENV:-}" ] || early "$WORKER_HOST does not answer and has no iLO settings"
  say "switching the DL360 on (5 to 10 minutes)"
  bash "$ROOT/worker/power.sh" wake || early "the DL360 did not come up within 15 minutes (is a USB disk plugged in?), or its iLO did not answer"
fi

# 1. the code and the data
say "copying the code to $WORKER_HOST:$APP"
$SSH "$WORKER_HOST" "mkdir -p '$APP/config' '$APP/data/trainingData' '$APP/data/hyperOptCache' '$APP/data/controls'" || early "the DL360 does not answer"
rsync -a --delete --timeout=120 -e "$SSH" \
  --exclude=/.git --exclude=node_modules --exclude=/web --exclude=/log --exclude=/data --exclude=/config \
  --exclude=/test --exclude=/inspiration --exclude=__pycache__ --exclude='/db.sqlite3*' --exclude=/process.lock \
  --exclude=/control.pid --exclude=/control.exit --exclude=/control.out --exclude='*.keras' --exclude=/run.log \
  "$ROOT/" "$WORKER_HOST:$APP/" || early "copying the code failed"
have="$($SSH "$WORKER_HOST" "cat '$APP/data/.snapshot' 2>/dev/null")"
if [ "$have" != "$SNAP" ]; then
  say "copying the data (snapshot $SNAP)"
  rsync -a --delete --timeout=120 -e "$SSH" "$ROOT/data/trainingData/" "$WORKER_HOST:$APP/data/trainingData/" || early "copying the training data failed"
  if [ -d "$ROOT/data/hyperOptCache" ]; then
    rsync -a --delete --timeout=120 -e "$SSH" "$ROOT/data/hyperOptCache/" "$WORKER_HOST:$APP/data/hyperOptCache/" || early "copying the tuners' cache failed"
  fi
  if [ -f "$ROOT/config/paramOverrides.json" ]; then
    rsync -a --timeout=60 -e "$SSH" "$ROOT/config/paramOverrides.json" "$WORKER_HOST:$APP/config/" || early "copying the manual values failed"
  else
    $SSH "$WORKER_HOST" "rm -f '$APP/config/paramOverrides.json'"
  fi
  $SSH "$WORKER_HOST" "echo '$SNAP' > '$APP/data/.snapshot'" || early "the DL360 does not answer"
else
  say "the data of snapshot $SNAP is there already (the draws this Sunday's controls read)"
fi
if [ -d "$ROOT/data/controls" ]; then
  rsync -a --delete --timeout=120 -e "$SSH" "$ROOT/data/controls/" "$WORKER_HOST:$APP/data/controls/" || early "copying data/controls failed"
fi

# 2. the run, detached from this ssh session: control.pid (the run), control.out (its output), control.exit (its code)
ARGS="$*"
say "starting $SCRIPT $ARGS on $WORKER_HOST${REMOTE_ENV:+ ($REMOTE_ENV)}"
$SSH "$WORKER_HOST" "cd '$APP' && rm -f control.exit control.out control.pid && { setsid nohup bash -c 'env $REMOTE_ENV PYTHONUNBUFFERED=1 timeout 172800 \"$WORKER_PYTHON\" $SCRIPT $ARGS > control.out 2>&1 & echo \$! > control.pid; wait \$!; echo \$? > control.exit.tmp; mv control.exit.tmp control.exit' > /dev/null 2>&1 < /dev/null & }" \
  || early "the control could not be started on the DL360"
started=1
contact=$(date +%s)
status=""
while :; do
  sleep 15 & wait $!
  if fetch; then contact=$(date +%s); fi
  code="$($SSH "$WORKER_HOST" "cat '$APP/control.exit' 2>/dev/null" 2>/dev/null)" && contact=$(date +%s)
  if [ -n "$code" ]; then status="$code"; break; fi
  if [ $(( $(date +%s) - contact )) -ge "$LOST_AFTER" ]; then
    say "no answer from the DL360 for $(( LOST_AFTER / 60 )) minutes: given up (the control's records stay as they were here)"
    exit 1
  fi
done
fetch
say "$SCRIPT on $WORKER_HOST ended with status $status"

# 3. its records
pull_back || { [ "$status" = 0 ] && exit 1; }
exit "$status"
