#!/bin/bash
# Restart the cloudflared connector when the public site is unreachable but the
# local stack is healthy. cloudflared has twice (2026-09-22, 2026-09-28) kept
# running with no active tunnel connection and no log line, returning 530 to
# every visitor until someone restarted it.
#
# Runs every minute from ~/Library/LaunchAgents/com.cairogenizah.tunnel-watchdog.plist.
# Rules: two consecutive failed public probes while localhost:8000/health is 200,
# and at most one restart per 10 minutes (so a genuine Cloudflare outage does not
# turn into a restart loop). Everything is logged to ~/Library/Logs/tunnel-watchdog.log.
set -u

LABEL="com.cairogenizah.cloudflared"
PUBLIC_URL="https://api.cairogenizah.ai/health"
LOCAL_URL="http://localhost:8000/health"
LOG="$HOME/Library/Logs/tunnel-watchdog.log"
STATE_DIR="${TMPDIR:-/tmp}/tunnel-watchdog"
FAIL_FILE="$STATE_DIR/consecutive_failures"
STAMP_FILE="$STATE_DIR/last_restart"
MIN_RESTART_GAP_SECONDS=600

mkdir -p "$STATE_DIR"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

# --- Wi-Fi guard: the Mac Studio (192.168.8.122) is only reachable from the
# SOVWIFI network. If Wi-Fi is on and joined anything else (e.g. "Avalon
# Resident"), rejoin SOVWIFI. Networks are already in the keychain, so no
# password is needed.
WIFI_DEV="$(networksetup -listallhardwareports 2>/dev/null | awk '/Hardware Port: Wi-Fi/{getline; print $2}')"
if [ -n "$WIFI_DEV" ] && networksetup -getairportpower "$WIFI_DEV" 2>/dev/null | grep -q "On"; then
  # networksetup hides the SSID without Location Services; ipconfig does not.
  ssid="$(ipconfig getsummary "$WIFI_DEV" 2>/dev/null | awk -F': ' '/^ *SSID :/{print $2}')"
  case "$ssid" in
    SOVWIFI*|"") ;;
    *)
      log "wifi on '$ssid': rejoining SOVWIFI-5G"
      networksetup -setairportnetwork "$WIFI_DEV" "SOVWIFI-5G" >/dev/null 2>&1 \
        || networksetup -setairportnetwork "$WIFI_DEV" "SOVWIFI" >/dev/null 2>&1 \
        || log "rejoin FAILED"
      ;;
  esac
fi

public_code="$(curl -s -m 12 -o /dev/null -w '%{http_code}' "$PUBLIC_URL" || echo 000)"
if [ "$public_code" = "200" ]; then
  [ -s "$FAIL_FILE" ] && log "recovered: public health 200"
  : > "$FAIL_FILE"
  exit 0
fi

local_code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$LOCAL_URL" || echo 000)"
if [ "$local_code" != "200" ]; then
  log "public=$public_code local=$local_code: the stack itself is down, not the tunnel; no restart"
  exit 0
fi

failures=$(( $(cat "$FAIL_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$failures" > "$FAIL_FILE"
log "public=$public_code local=200 (consecutive failures: $failures)"
[ "$failures" -lt 2 ] && exit 0

now=$(date +%s)
last=$(cat "$STAMP_FILE" 2>/dev/null || echo 0)
if [ $(( now - last )) -lt "$MIN_RESTART_GAP_SECONDS" ]; then
  log "restart skipped: last restart $(( now - last ))s ago"
  exit 0
fi

echo "$now" > "$STAMP_FILE"
if launchctl kickstart -k "gui/$(id -u)/$LABEL"; then
  log "restarted $LABEL"
else
  log "restart of $LABEL FAILED"
fi
: > "$FAIL_FILE"
