#!/usr/bin/env bash
# Installed on the CPU worker (the HP DL360) as /usr/local/sbin/sp-worker-idle, run every minute by a systemd timer
# (sp-worker-idle.timer). It only acts on a boot the worker marked (worker/power.sh wake: /run/sequencePredictor-worker/
# woken - /run is emptied at every boot, so a start by hand is never switched off): it switches the machine off after
# IDLE_MINUTES (10, /etc/default/sp-worker-idle) without
#   - a backtest running (the worker's backtest.pid alive), a Sunday control (controls/control.pid) or a hyperopt
#     started from the Hyperopts page (hyperopt/hyperopt.pid, since 12 Nov 2026),
#   - a council request (a "slot" line from a llama@ service in the journal), or
#   - an active login (who -u: idle less than IDLE_MINUTES).
set -u
[ -f /etc/default/sp-worker-idle ] && . /etc/default/sp-worker-idle
IDLE="${IDLE_MINUTES:-10}"
APP="${WORKER_APP:-/opt/sequencePredictor-worker/app}"
MARK="${WORKER_MARK:-/run/sequencePredictor-worker}"   # WORKER_MARK and SP_IDLE_DRY_RUN=1: for a test only
[ -f "$MARK/woken" ] || exit 0
now=$(date +%s)
busy=""
if [ -f "$APP/backtest.pid" ] && kill -0 "$(cat "$APP/backtest.pid" 2>/dev/null)" 2>/dev/null; then busy="a backtest is running"; fi
# a Sunday control (worker/remote-control.sh: <dir>/controls/control.pid, since 11 Nov 2026)
for pidfile in "$(dirname "$APP")"/controls/control.pid "$(dirname "$APP")"/*/controls/control.pid; do
  [ -z "$busy" ] && [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null && busy="a control is running"
done
# a hyperopt from the Hyperopts page (worker/remote-hyperopt.sh: <dir>/hyperopt/hyperopt.pid, since 12 Nov 2026)
pidfile="$(dirname "$APP")/hyperopt/hyperopt.pid"
[ -z "$busy" ] && [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null && busy="a hyperopt is running"
if [ -z "$busy" ] && journalctl -u 'llama@*' --since "-${IDLE} min" -q --no-pager -o cat 2>/dev/null | grep -q ' slot '; then busy="the council answered a request"; fi
if [ -z "$busy" ]; then
  # who -u: the 5th field is the idle time - "." (active this minute), "HH:MM", or "old" (a day)
  while read -r idle; do
    case "$idle" in
      .) busy="someone is logged in" ;;
      [0-9]*:[0-9]*) [ $(( 10#${idle%%:*} * 60 + 10#${idle##*:} )) -lt "$IDLE" ] && busy="someone is logged in" ;;
    esac
  done < <(who -u | awk '{print $5}')
fi
if [ -n "$busy" ]; then
  [ "${SP_IDLE_DRY_RUN:-}" = 1 ] && echo "busy: $busy"
  echo "$now" > "$MARK/last-busy"
  exit 0
fi
last=$(cat "$MARK/last-busy" 2>/dev/null || cat "$MARK/woken")
if [ $(( now - last )) -ge $(( IDLE * 60 )) ]; then
  if [ "${SP_IDLE_DRY_RUN:-}" = 1 ]; then echo "would switch off (idle $(( (now - last) / 60 )) min)"; exit 0; fi
  logger -t sp-worker-idle "nothing for ${IDLE} minutes since the worker switched this machine on: switching off"
  systemctl poweroff
fi
