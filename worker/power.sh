#!/usr/bin/env bash
# The CPU worker's power (the HP DL360, owner, 8-10 Nov 2026; docs/operations.md, The workers). The machine is power
# hungry and off unless there is work: this switches it on through its iLO (IPMI), waits for it, and marks the boot as
# the worker's, so the machine switches itself off again after its idle time (worker/idle-shutdown.sh on the worker).
#
#   power.sh status    one line of JSON: {"power":"on|off|unknown","reachable":..,"woken":..,"lastBusy":..,"idleMinutes":..}
#   power.sh wake      on through the iLO when it does not answer, wait for ssh (15 min at most), mark the boot
#   power.sh off       shut it down cleanly (systemctl poweroff over ssh)
#
# Environment: WORKER_HOST (ssh target), WORKER_ILO_ENV (a root-only file with ILO_HOST, ILO_USER, ILO_PASSWORD; the
# password goes to ipmitool through IPMI_PASSWORD and -E, never on a command line), WORKER_IDLE_MINUTES (10).
set -u
: "${WORKER_HOST:?WORKER_HOST not set}"
IDLE="${WORKER_IDLE_MINUTES:-10}"
WAIT_SECONDS="${WORKER_BOOT_WAIT_SECONDS:-900}"
MARK=/run/sequencePredictor-worker
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=2"
say() { printf '%s [cpu worker] %s\n' "$(date -u '+%F %T')" "$*" >&2; }

ipmi() {
  [ -n "${WORKER_ILO_ENV:-}" ] && [ -r "$WORKER_ILO_ENV" ] || { echo "no iLO settings"; return 2; }
  ( set -a; . "$WORKER_ILO_ENV"; set +a
    IPMI_PASSWORD="$ILO_PASSWORD" exec timeout 30 ipmitool -I lanplus -H "$ILO_HOST" -U "$ILO_USER" -E "$@" )
}
reachable() { $SSH "$WORKER_HOST" true 2>/dev/null; }
power_state() {
  local out; out="$(ipmi chassis power status 2>&1)"
  case "$out" in *" is on"*) echo on ;; *" is off"*) echo off ;; *) echo unknown ;; esac
}

case "${1:-}" in
  status)
    p="$(power_state)"
    if [ "$p" != off ] && info="$($SSH "$WORKER_HOST" "cat $MARK/woken 2>/dev/null || echo -; cat $MARK/last-busy 2>/dev/null || echo -" 2>/dev/null)"; then
      woken="$(echo "$info" | sed -n 1p)"; busy="$(echo "$info" | sed -n 2p)"
      printf '{"power":"%s","reachable":true,"woken":%s,"wokenAt":%s,"lastBusy":%s,"idleMinutes":%s}\n' \
        "$p" "$([ "$woken" = - ] && echo false || echo true)" "$([ "$woken" = - ] && echo null || echo "$woken")" \
        "$([ "$busy" = - ] && echo null || echo "$busy")" "$IDLE"
    else
      printf '{"power":"%s","reachable":false,"woken":false,"wokenAt":null,"lastBusy":null,"idleMinutes":%s}\n' "$p" "$IDLE"
    fi
    ;;
  wake)
    if reachable; then
      say "$WORKER_HOST answers already"
      exit 0
    fi
    p="$(power_state)"
    case "$p" in
      off) say "switching $WORKER_HOST on through its iLO"; ipmi chassis power on >/dev/null 2>&1 || { say "the iLO refused to switch it on"; exit 92; } ;;
      on) say "$WORKER_HOST is on but does not answer yet (booting?): waiting" ;;
      *) say "the iLO did not answer ($(ipmi chassis power status 2>&1 | head -c 200))"; exit 92 ;;
    esac
    started=$(date +%s)
    until reachable; do
      if [ $(( $(date +%s) - started )) -ge "$WAIT_SECONDS" ]; then
        say "$WORKER_HOST did not come up within $(( WAIT_SECONDS / 60 )) minutes - is a USB disk plugged in? (its start-up stops at a red screen with one)"
        exit 93
      fi
      sleep 15 & wait $!
    done
    say "$WORKER_HOST answers after $(( $(date +%s) - started )) s"
    # the boot is the worker's: worker/idle-shutdown.sh switches it off after $IDLE idle minutes
    $SSH "$WORKER_HOST" "mkdir -p $MARK && date +%s > $MARK/woken && date +%s > $MARK/last-busy" || say "could not mark the boot"
    ;;
  off)
    say "shutting $WORKER_HOST down"
    $SSH "$WORKER_HOST" "systemctl poweroff" 2>/dev/null
    exit 0
    ;;
  *) echo "usage: power.sh status|wake|off" >&2; exit 2 ;;
esac
