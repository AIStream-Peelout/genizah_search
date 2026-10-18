#!/bin/bash
# Keep api.cairogenizah.ai up. Runs every minute from
# ~/Library/LaunchAgents/com.cairogenizah.tunnel-watchdog.plist (installed copy:
# ~/Library/Application Support/cairogenizah/tunnel_watchdog.sh — launchd may
# not execute from ~/Documents). Everything is logged to
# ~/Library/Logs/tunnel-watchdog.log. Tests: test_tunnel_watchdog.sh.
#
# 1. Tunnel: restart the cloudflared connector when the public site is
#    unreachable but the local stack is healthy. cloudflared has twice
#    (2026-09-22, 2026-09-28) kept running with no active tunnel connection and
#    no log line, returning 530 to every visitor until someone restarted it.
#    Two consecutive failed public probes while localhost:8000/health is 200;
#    at most one restart per 10 minutes.
#
# 2. Docker: on 2026-10-08 the Docker Desktop VM crashed ("Internal
#    Virtualization error") with the disk 99% full; the backend was gone, every
#    docker command hung, and the site was down ~3 h because this script only
#    logged "stack is down". The user has authorized force-restarting Docker
#    whenever it is down. After STACK_FAILS_BEFORE_ACTION consecutive failed
#    local probes:
#      - Docker Desktop not running          -> launch it;
#      - Docker daemon unresponsive or       -> kill Docker Desktop and relaunch
#        failing container lookups              it, unless its engine started
#                                               < DOCKER_BOOT_GRACE_SECONDS ago
#                                               (booting), at most once per 15
#                                               min and twice per outage;
#      - backend running but not answering  -> restart that container once both
#                                               the failures and the container
#                                               are BACKEND_FAILS_BEFORE_RESTART
#                                               min old; if that did not help,
#                                               escalate to a Docker restart;
#      - backend stopped/removed             -> nothing: with restart:
#                                               unless-stopped that only happens
#                                               on purpose (compose stop/down).
#    Containers come back on their own after a Docker restart (restart:
#    unless-stopped). Nothing here rebuilds, recreates, prunes or deletes.
#    A compose/build command under 15 min old pauses the container-level
#    actions (a deploy explains a down backend).
#
# 3. Disk: warn (log + macOS notification, hourly) when free space on the data
#    volume drops below DISK_WARN_GB; the Docker VM grows Docker.raw until the
#    host disk is full and then dies.
#
# Before quitting/restarting Docker Desktop or `compose down` on purpose:
#   touch "$HOME/Library/Application Support/cairogenizah/MAINTENANCE"
# and remove it afterwards — otherwise this script brings Docker back.
# DRY_RUN=1 logs actions instead of running them, into its own log and state.
set -u
export PATH="$PATH:/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin"

DRY_RUN="${DRY_RUN:-0}"
# A dry run must not read or advance the live counters and stamps.
DRY_SUFFIX=""; [ "$DRY_RUN" = "1" ] && DRY_SUFFIX="-dryrun"

LABEL="com.cairogenizah.cloudflared"
PUBLIC_URL="${WATCHDOG_PUBLIC_URL:-https://api.cairogenizah.ai/health}"
LOCAL_URL="${WATCHDOG_LOCAL_URL:-http://localhost:8000/health}"
LOG="${WATCHDOG_LOG:-$HOME/Library/Logs/tunnel-watchdog${DRY_SUFFIX}.log}"
STATE_DIR="${WATCHDOG_STATE_DIR:-${TMPDIR:-/tmp}/tunnel-watchdog${DRY_SUFFIX}}"
MAINTENANCE_FILE="${WATCHDOG_MAINTENANCE_FILE:-$HOME/Library/Application Support/cairogenizah/MAINTENANCE}"

BACKEND_CONTAINER="genizah_search-backend-1"
DOCKER_APP_PATTERN="^/Applications/Docker.app/Contents/"
DATA_VOLUME="/System/Volumes/Data"
DOCKER_RAW="$HOME/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
# Whole-word subcommands, so e.g. "rm" inside a "terraform_..." path never matches.
DEPLOY_PATTERN="docker[- ]compose( .*)? (up|down|stop|start|restart|rm|create|build|pull)( |$)|docker( buildx)? build( |$)"

MIN_RESTART_GAP_SECONDS=600          # tunnel restarts
STACK_FAILS_BEFORE_ACTION=3          # minutes of failed local probes before touching Docker
BACKEND_FAILS_BEFORE_RESTART=5       # minutes before restarting a hung backend container
MIN_DOCKER_RESTART_GAP_SECONDS=900   # between Docker Desktop launches/restarts
DOCKER_BOOT_GRACE_SECONDS=300        # never kill a Docker engine younger than this
MAX_DOCKER_RESTARTS_PER_OUTAGE=2     # then notify only: a restart is not fixing it
MIN_BACKEND_RESTART_GAP_SECONDS=900
DOCKER_CLI_TIMEOUT_SECONDS="${WATCHDOG_DOCKER_TIMEOUT:-20}"
DEPLOY_GRACE_SECONDS=900
DISK_WARN_GB=60
DISK_URGENT_GB=25
DISK_ALERT_GAP_SECONDS=3600

FAIL_FILE="$STATE_DIR/consecutive_failures"
STAMP_FILE="$STATE_DIR/last_restart"
STACK_FAIL_FILE="$STATE_DIR/stack_failures"
DOCKER_STAMP_FILE="$STATE_DIR/last_docker_restart"
DOCKER_RESTARTS_FILE="$STATE_DIR/docker_restarts_this_outage"
BACKEND_STAMP_FILE="$STATE_DIR/last_backend_restart"
BACKEND_RESTARTS_FILE="$STATE_DIR/backend_restarts_this_outage"
GAVE_UP_FILE="$STATE_DIR/gave_up_this_outage"
INSPECT_ERR_FILE="$STATE_DIR/inspect.err"
DISK_STAMP_FILE="$STATE_DIR/last_disk_alert"

mkdir -p "$STATE_DIR"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

# Run an action, or only log it under DRY_RUN=1.
act() {
  if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: $*"
    return 0
  fi
  "$@"
}

# Banner on this Mac; best effort, never fails the run.
notify() {
  act osascript -e "display notification \"$1\" with title \"cairogenizah watchdog\"" >/dev/null 2>&1 || true
}

# Integer from a state file, 0 when missing or garbled.
read_int() {
  local value
  value="$(cat "$1" 2>/dev/null)"
  case "$value" in ''|*[!0-9]*) echo 0 ;; *) echo "$value" ;; esac
}

# Seconds since the timestamp stored in $1 (huge when never stamped).
seconds_since() { echo $(( $(date +%s) - $(read_int "$1") )); }

stamp() { date +%s > "$1"; }

bump() { echo $(( $(read_int "$1") + 1 )) > "$1"; }

# The stack answered again: start the next outage from a clean slate.
reset_outage_state() { : > "$STACK_FAIL_FILE"; rm -f "$DOCKER_RESTARTS_FILE" "$BACKEND_RESTARTS_FILE" "$GAVE_UP_FILE"; }

# Run a command with a hard timeout (macOS has no timeout(1)). SIGKILL, because
# the docker CLI is a Go program and Go ignores SIGALRM. Exit 124 on timeout.
with_timeout() {
  perl -e '
    my $secs = shift @ARGV;
    my $pid = fork();
    exit 125 unless defined $pid;
    if ($pid == 0) { exec { $ARGV[0] } @ARGV or exit 127 }
    $SIG{ALRM} = sub { kill "KILL", $pid; waitpid($pid, 0); exit 124 };
    alarm $secs;
    waitpid($pid, 0);
    exit(($? & 127) ? 128 + ($? & 127) : $? >> 8);
  ' "$@"
}

# Elapsed seconds of a process, from ps's [[dd-]hh:]mm:ss etime.
process_age_seconds() {
  ps -o etime= -p "$1" 2>/dev/null | awk '{
    n = split($1, a, /[-:]/); s = 0
    if (n == 4) s = a[1]*86400 + a[2]*3600 + a[3]*60 + a[4]
    else if (n == 3) s = a[1]*3600 + a[2]*60 + a[3]
    else if (n == 2) s = a[1]*60 + a[2]
    print s }'
}

# Seconds since an ISO-8601 UTC time such as Docker's StartedAt.
seconds_since_iso() {
  local epoch
  epoch="$(date -j -u -f '%Y-%m-%dT%H:%M:%S' "${1%%.*}" +%s 2>/dev/null)" || { echo 0; return; }
  echo $(( $(date +%s) - epoch ))
}

docker_desktop_running() { pgrep -f "${DOCKER_APP_PATTERN}MacOS/com.docker.backend" >/dev/null 2>&1; }

# Age of the youngest Docker Desktop engine process, however it was started
# (by us, by hand, by an in-app restart, update or settings change). Empty if none.
docker_engine_age() {
  local pid
  pid="$(pgrep -n -f "${DOCKER_APP_PATTERN}MacOS/com.docker.(backend|virtualization)" 2>/dev/null)"
  [ -n "$pid" ] && process_age_seconds "$pid"
}

# `docker info --format` exits 0 even when the daemon is unreachable;
# `docker version` fails, and an empty server version is a failure too.
docker_responsive() {
  local version
  version="$(with_timeout "$DOCKER_CLI_TIMEOUT_SECONDS" docker version --format '{{.Server.Version}}' 2>/dev/null)" \
    && [ -n "$version" ]
}

# A compose/build command started in the last DEPLOY_GRACE_SECONDS explains a
# down backend. Older ones (a foreground `compose up` left in a terminal, or
# one stuck on a dead daemon) must not block healing forever.
deploy_in_progress() {
  local pid age
  for pid in $(pgrep -f "$DEPLOY_PATTERN" 2>/dev/null); do
    age="$(process_age_seconds "$pid")"
    [ -n "$age" ] && [ "$age" -lt "$DEPLOY_GRACE_SECONDS" ] && return 0
  done
  return 1
}

# Kill every Docker Desktop process and relaunch the app. Containers return via
# their restart: unless-stopped policy once the VM boots (1-3 min).
force_restart_docker() {
  stamp "$DOCKER_STAMP_FILE"
  bump "$DOCKER_RESTARTS_FILE"
  log "docker: force-restarting Docker Desktop ($1)"
  notify "Docker is down ($1) - force-restarting Docker Desktop"
  act pkill -TERM -f "$DOCKER_APP_PATTERN"
  act pkill -TERM -f "^com\.docker\."
  local waited=0
  while [ "$DRY_RUN" != "1" ] && [ "$waited" -lt 15 ] && pgrep -f "$DOCKER_APP_PATTERN" >/dev/null 2>&1; do
    sleep 1; waited=$(( waited + 1 ))
  done
  act pkill -KILL -f "$DOCKER_APP_PATTERN"
  act pkill -KILL -f "^com\.docker\."
  [ "$DRY_RUN" = "1" ] || sleep 2
  if act open -g -a Docker; then
    log "docker: Docker Desktop relaunched"
  else
    log "docker: relaunch FAILED (open -a Docker)"
  fi
}

# Force-restart Docker unless it is booting, was just restarted, or two
# restarts already failed to fix this outage. $1: why.
maybe_force_restart_docker() {
  local age
  age="$(docker_engine_age)"
  if [ -n "$age" ] && [ "$age" -lt "$DOCKER_BOOT_GRACE_SECONDS" ]; then
    log "docker: $1, but its engine started ${age}s ago; letting it boot"
    return
  fi
  if [ "$(seconds_since "$DOCKER_STAMP_FILE")" -lt "$MIN_DOCKER_RESTART_GAP_SECONDS" ]; then
    log "docker: $1; last restart $(seconds_since "$DOCKER_STAMP_FILE")s ago, waiting"
    return
  fi
  if [ "$(read_int "$DOCKER_RESTARTS_FILE")" -ge "$MAX_DOCKER_RESTARTS_PER_OUTAGE" ]; then
    log "docker: $1; still down after $MAX_DOCKER_RESTARTS_PER_OUTAGE restarts - not restarting again, needs a human"
    if [ ! -e "$GAVE_UP_FILE" ]; then
      touch "$GAVE_UP_FILE"
      notify "Docker still down after $MAX_DOCKER_RESTARTS_PER_OUTAGE restarts - needs manual attention (disk full?)"
    fi
    return
  fi
  force_restart_docker "$1"
}

check_disk() {
  local free_gb raw_gb level
  free_gb="$(df -g "$DATA_VOLUME" 2>/dev/null | awk 'NR==2 {print $4}')"
  case "$free_gb" in ''|*[!0-9]*) return ;; esac
  [ "$free_gb" -ge "$DISK_WARN_GB" ] && return
  [ "$(seconds_since "$DISK_STAMP_FILE")" -lt "$DISK_ALERT_GAP_SECONDS" ] && return
  stamp "$DISK_STAMP_FILE"
  level="low"; [ "$free_gb" -lt "$DISK_URGENT_GB" ] && level="URGENT"
  raw_gb="$(du -g "$DOCKER_RAW" 2>/dev/null | cut -f1)"
  log "disk $level: ${free_gb} GB free on $DATA_VOLUME (Docker.raw ${raw_gb:-?} GB); a full disk crashes the Docker VM"
  notify "Disk $level: ${free_gb} GB free (Docker.raw ${raw_gb:-?} GB). A full disk crashes Docker and the site."
}

# The backend container is running but its /health is failing. $1: failures.
heal_running_backend() {
  local failures="$1" uptime="$2"
  [ "$failures" -lt "$BACKEND_FAILS_BEFORE_RESTART" ] && return
  # Counted failures may predate this container (e.g. a Docker restart);
  # give a freshly started backend the same grace before calling it hung.
  [ "$uptime" -lt $(( BACKEND_FAILS_BEFORE_RESTART * 60 )) ] && return
  if [ "$(seconds_since "$BACKEND_STAMP_FILE")" -lt "$MIN_BACKEND_RESTART_GAP_SECONDS" ]; then
    log "backend: running but unhealthy; restarted $(seconds_since "$BACKEND_STAMP_FILE")s ago, waiting"
    return
  fi
  if [ "$(read_int "$BACKEND_RESTARTS_FILE")" -ge 1 ]; then
    # e.g. Docker's port forward to :8000 is broken, which a container restart cannot fix.
    maybe_force_restart_docker "backend still failing after a container restart"
    return
  fi
  stamp "$BACKEND_STAMP_FILE"
  bump "$BACKEND_RESTARTS_FILE"
  log "backend: container up ${uptime}s but /health failing for $failures min; restarting the container"
  notify "Backend container hung - restarting it"
  act with_timeout 90 docker restart "$BACKEND_CONTAINER" >/dev/null 2>&1 \
    || log "backend: docker restart FAILED"
}

# The local stack failed its health probe: decide whether to touch Docker.
heal_stack() {
  local public_code="$1" local_code="$2" failures inspect rc state

  # Paused on purpose: hold the count at zero so the grace restarts afterwards.
  if [ -e "$MAINTENANCE_FILE" ]; then
    log "public=$public_code local=$local_code: maintenance file present, no restarts"
    : > "$STACK_FAIL_FILE"
    return
  fi

  failures=$(( $(read_int "$STACK_FAIL_FILE") + 1 ))
  echo "$failures" > "$STACK_FAIL_FILE"
  log "public=$public_code local=$local_code: stack down (consecutive: $failures)"
  [ "$failures" -lt "$STACK_FAILS_BEFORE_ACTION" ] && return
  [ "$failures" -eq "$STACK_FAILS_BEFORE_ACTION" ] && notify "Site down: local backend failing for $failures min"

  if ! docker_desktop_running; then
    if [ "$(seconds_since "$DOCKER_STAMP_FILE")" -lt "$MIN_DOCKER_RESTART_GAP_SECONDS" ]; then
      log "docker: not running, but (re)started $(seconds_since "$DOCKER_STAMP_FILE")s ago; waiting"
      return
    fi
    stamp "$DOCKER_STAMP_FILE"
    log "docker: Docker Desktop not running; launching it"
    notify "Docker Desktop was not running - launching it"
    act open -g -a Docker || log "docker: launch FAILED (open -a Docker)"
    return
  fi

  if ! docker_responsive; then
    maybe_force_restart_docker "daemon unresponsive"
    return
  fi

  # Only a responsive daemon can be mid-deploy; leave the containers to it.
  if deploy_in_progress; then
    log "compose/build operation in progress: no container restarts"
    : > "$STACK_FAIL_FILE"
    return
  fi

  inspect="$(with_timeout "$DOCKER_CLI_TIMEOUT_SECONDS" docker inspect -f '{{.State.Status}} {{.State.StartedAt}}' "$BACKEND_CONTAINER" 2>"$INSPECT_ERR_FILE")"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    if grep -q "No such object" "$INSPECT_ERR_FILE" 2>/dev/null; then
      log "backend: container $BACKEND_CONTAINER not found (compose down?); not starting anything"
    else
      maybe_force_restart_docker "docker inspect failed (rc=$rc)"
    fi
    return
  fi
  state="${inspect%% *}"
  case "$state" in
    running)
      heal_running_backend "$failures" "$(seconds_since_iso "${inspect#* }")"
      ;;
    restarting)
      log "backend: container is crash-looping (restarting); see docker logs $BACKEND_CONTAINER"
      ;;
    *)
      log "backend: container is '$state' (stopped on purpose?); not starting it"
      ;;
  esac
}

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
      act networksetup -setairportnetwork "$WIFI_DEV" "SOVWIFI-5G" >/dev/null 2>&1 \
        || act networksetup -setairportnetwork "$WIFI_DEV" "SOVWIFI" >/dev/null 2>&1 \
        || log "rejoin FAILED"
      ;;
  esac
fi

check_disk

public_code="$(curl -s -m 12 -o /dev/null -w '%{http_code}' "$PUBLIC_URL")"; [ -n "$public_code" ] || public_code=000
if [ "$public_code" = "200" ]; then
  if [ -s "$FAIL_FILE" ] || [ -s "$STACK_FAIL_FILE" ]; then
    log "recovered: public health 200"
    [ "$(read_int "$STACK_FAIL_FILE")" -ge "$STACK_FAILS_BEFORE_ACTION" ] && notify "Site recovered"
  fi
  : > "$FAIL_FILE"
  reset_outage_state
  exit 0
fi

local_code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$LOCAL_URL")"; [ -n "$local_code" ] || local_code=000
if [ "$local_code" != "200" ]; then
  heal_stack "$public_code" "$local_code"
  exit 0
fi
if [ -s "$STACK_FAIL_FILE" ]; then
  log "local stack recovered (public still $public_code)"
fi
reset_outage_state

failures=$(( $(read_int "$FAIL_FILE") + 1 ))
echo "$failures" > "$FAIL_FILE"
log "public=$public_code local=200 (consecutive failures: $failures)"
[ "$failures" -lt 2 ] && exit 0

if [ "$(seconds_since "$STAMP_FILE")" -lt "$MIN_RESTART_GAP_SECONDS" ]; then
  log "restart skipped: last restart $(seconds_since "$STAMP_FILE")s ago"
  exit 0
fi

if [ -e "$MAINTENANCE_FILE" ]; then
  log "maintenance file present: no tunnel restart"
  exit 0
fi

stamp "$STAMP_FILE"
if act launchctl kickstart -k "gui/$(id -u)/$LABEL"; then
  log "restarted $LABEL"
else
  log "restart of $LABEL FAILED"
fi
: > "$FAIL_FILE"
