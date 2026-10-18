#!/bin/bash
# Dry-run tests for tunnel_watchdog.sh. Every external command the watchdog
# uses (curl, docker, pgrep, ps, df, du, networksetup, open, pkill, launchctl,
# osascript) is replaced by a stub on PATH, and DRY_RUN=1 makes the script log
# its actions instead of running them — nothing on this machine is touched.
#
#   bash scripts/mbp_migration/test_tunnel_watchdog.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
WATCHDOG="$HERE/tunnel_watchdog.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/watchdog-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
STUBS="$WORK/bin"
mkdir -p "$STUBS"

PUBLIC="https://public.invalid/health"
LOCAL="http://local.invalid/health"

# --- stubs: behavior comes from FAKE_* variables; side-effect commands only
# record that they were called (they must never be reached under DRY_RUN=1).
cat > "$STUBS/curl" <<EOF
#!/bin/bash
for url; do :; done
case "\$url" in
  "$PUBLIC") printf '%s' "\${FAKE_PUBLIC:-200}" ;;
  "$LOCAL") printf '%s' "\${FAKE_LOCAL:-200}" ;;
  *) printf '000' ;;
esac
EOF
cat > "$STUBS/docker" <<'EOF'
#!/bin/bash
case "$1" in
  version)
    case "${FAKE_DOCKER_INFO:-ok}" in
      ok) echo "27.5.1" ;;
      # Real CLI: error on stderr, nothing on stdout, exit 1.
      fail) echo "Cannot connect to the Docker daemon" >&2; exit 1 ;;
      empty) exit 0 ;;
      # A Go binary ignores SIGALRM; so does this stub.
      hang) exec perl -e '$SIG{ALRM} = "IGNORE"; sleep 60' ;;
    esac ;;
  inspect)
    case "${FAKE_INSPECT_MODE:-value}" in
      hang) exec perl -e '$SIG{ALRM} = "IGNORE"; sleep 60' ;;
      error) echo "request returned 500 Internal Server Error" >&2; exit 1 ;;
    esac
    [ -n "${FAKE_INSPECT:-}" ] || { echo "Error: No such object: $4" >&2; exit 1; }
    echo "$FAKE_INSPECT" ;;
  *) echo "docker $*" >> "$CALLS"; exit 0 ;;
esac
EOF
cat > "$STUBS/pgrep" <<'EOF'
#!/bin/bash
pattern="${@: -1}"
case "$pattern" in
  *virtualization*) [ "${FAKE_DOCKER_RUNNING:-1}" = "1" ] && echo "${FAKE_ENGINE_PID:-777}" ;;
  *com.docker.backend*) [ "${FAKE_DOCKER_RUNNING:-1}" = "1" ] ;;
  *compose*) [ -n "${FAKE_DEPLOY_PID:-}" ] && echo "$FAKE_DEPLOY_PID" ;;
  *) exit 1 ;;
esac
EOF
cat > "$STUBS/ps" <<'EOF'
#!/bin/bash
pid="${@: -1}"
if [ "$pid" = "${FAKE_DEPLOY_PID:-none}" ]; then echo "${FAKE_DEPLOY_ETIME:-00:10}"
else echo "${FAKE_ENGINE_ETIME:-2-00:00:00}"; fi
EOF
cat > "$STUBS/df" <<'EOF'
#!/bin/bash
echo "Filesystem 1G-blocks Used Available Capacity iused ifree %iused Mounted on"
echo "/dev/disk3s5 926 800 ${FAKE_DF_FREE:-200} 80% 1 1 0% /System/Volumes/Data"
EOF
printf '#!/bin/bash\nprintf "137\\t%%s\\n" "$2"\n' > "$STUBS/du"
printf '#!/bin/bash\nexit 0\n' > "$STUBS/networksetup"
for cmd in open pkill launchctl osascript; do
  printf '#!/bin/bash\necho "%s $*" >> "$CALLS"\n' "$cmd" > "$STUBS/$cmd"
done
chmod +x "$STUBS"/*

pass=0; fail=0
STATE="$WORK/state"; LOGFILE="$WORK/watchdog.log"; MAINT="$WORK/MAINTENANCE"
export CALLS="$WORK/calls"

# Fresh state for one scenario.
reset() {
  rm -rf "$STATE" "$LOGFILE" "$MAINT" "$CALLS"
  mkdir -p "$STATE"; : > "$LOGFILE"; : > "$CALLS"
  unset FAKE_PUBLIC FAKE_LOCAL FAKE_DOCKER_INFO FAKE_INSPECT FAKE_INSPECT_MODE FAKE_DOCKER_RUNNING \
        FAKE_DEPLOY_PID FAKE_DEPLOY_ETIME FAKE_ENGINE_ETIME FAKE_DF_FREE
}

# One watchdog run with the stubs first on PATH.
run() {
  PATH="$STUBS:/usr/bin:/bin:/usr/sbin:/sbin" DRY_RUN=1 \
    WATCHDOG_PUBLIC_URL="$PUBLIC" WATCHDOG_LOCAL_URL="$LOCAL" \
    WATCHDOG_LOG="$LOGFILE" WATCHDOG_STATE_DIR="$STATE" \
    WATCHDOG_MAINTENANCE_FILE="$MAINT" WATCHDOG_DOCKER_TIMEOUT="${TIMEOUT:-2}" \
    /bin/bash "$WATCHDOG"
}

# Run N times.
run_n() { local i; for i in $(seq "$1"); do run; done; }

expect_log() {
  if grep -q -- "$2" "$LOGFILE"; then pass=$((pass + 1))
  else fail=$((fail + 1)); echo "FAIL [$1]: log lacks '$2'"; sed 's/^/    /' "$LOGFILE"; fi
}

reject_log() {
  if grep -q -- "$2" "$LOGFILE"; then
    fail=$((fail + 1)); echo "FAIL [$1]: log unexpectedly has '$2'"; sed 's/^/    /' "$LOGFILE"
  else pass=$((pass + 1)); fi
}

# Side-effect stubs must never run under DRY_RUN=1.
expect_no_calls() {
  if [ -s "$CALLS" ]; then fail=$((fail + 1)); echo "FAIL [$1]: real side effects:"; sed 's/^/    /' "$CALLS"
  else pass=$((pass + 1)); fi
}

iso_ago() { date -u -v-"$1"S +%Y-%m-%dT%H:%M:%S.123456789Z; }

# Pretend the last Docker/backend restart was long ago (past every gap).
age_stamp() { echo $(( $(date +%s) - 7200 )) > "$STATE/$1"; }

# --- scenarios
reset; run
[ -s "$LOGFILE" ] && { fail=$((fail + 1)); echo "FAIL [healthy]: logged:"; cat "$LOGFILE"; } || pass=$((pass + 1))

reset; export FAKE_PUBLIC=502; run
expect_log "tunnel 1st failure" "consecutive failures: 1"
reject_log "tunnel 1st failure" "kickstart"
run
expect_log "tunnel 2nd failure restarts" "dry-run: launchctl kickstart -k"
run; run
expect_log "tunnel rate limit" "restart skipped: last restart"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_RUNNING=0; run_n 2
expect_log "stack: counts" "stack down (consecutive: 2)"
reject_log "stack: no action before threshold" "dry-run: open"
run
expect_log "stack: launches stopped Docker" "dry-run: open -g -a Docker"
run
expect_log "stack: launch rate-limited" "not running, but (re)started"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=hang; TIMEOUT=2
start=$(date +%s); run_n 3; took=$(( $(date +%s) - start ))
expect_log "hung daemon: force restart" "force-restarting Docker Desktop"
expect_log "hung daemon: pkill" "dry-run: pkill -KILL -f ^/Applications/Docker.app/Contents/"
expect_log "hung daemon: relaunch" "dry-run: open -g -a Docker"
if [ "$took" -le 12 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [hung daemon]: timeout did not kill a SIGALRM-ignoring docker (${took}s)"; fi
run
expect_log "hung daemon: rate-limited" "unresponsive; last restart"
unset TIMEOUT

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=fail; run_n 3
expect_log "daemon refusing: force restart" "force-restarting Docker Desktop"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 3600)"; run_n 4
reject_log "hung backend: waits for 5 min" "restarting the container"
run
expect_log "hung backend: restarts container" "dry-run: with_timeout 90 docker restart genizah_search-backend-1"
run
expect_log "hung backend: rate-limited" "restarted .*s ago, waiting"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 60)"; run_n 8
reject_log "fresh backend: not restarted" "restarting the container"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="exited 0001-01-01T00:00:00Z"; run_n 6
expect_log "stopped backend: left alone" "stopped on purpose"
reject_log "stopped backend: left alone" "dry-run: with_timeout 90 docker"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="restarting $(iso_ago 30)"; run_n 6
expect_log "crash loop: logged" "crash-looping"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT=""; run_n 6
expect_log "removed backend: left alone" "not found (compose down?)"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=fail; touch "$MAINT"; run_n 5
expect_log "maintenance" "maintenance file present, no restarts"
rm -f "$MAINT"; run
expect_log "maintenance resets the grace" "stack down (consecutive: 1)"
reject_log "maintenance" "force-restarting"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=200; touch "$MAINT"; run_n 3
reject_log "maintenance blocks tunnel restart" "kickstart"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 3600)" FAKE_DEPLOY_PID=4242 FAKE_DEPLOY_ETIME=02:30; run_n 6
expect_log "fresh deploy: hands off" "compose/build operation in progress"
reject_log "fresh deploy: hands off" "restarting the container"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 3600)" FAKE_DEPLOY_PID=4242 FAKE_DEPLOY_ETIME=1-02:00:00; run_n 5
expect_log "stale deploy: ignored" "restarting the container"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 60)"; run_n 3
export FAKE_PUBLIC=200; run
expect_log "recovery" "recovered: public health 200"
expect_log "recovery notifies" "dry-run: osascript"
: > "$LOGFILE"; run
reject_log "recovery: counters reset" "recovered"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 3600)"; run_n 2
export FAKE_LOCAL=200; run
expect_log "local recovery resets stack counter" "local stack recovered"

reset; export FAKE_DF_FREE=20; run
expect_log "disk urgent" "disk URGENT: 20 GB free"
: > "$LOGFILE"; run
reject_log "disk alert hourly" "disk"
reset; export FAKE_DF_FREE=45; run
expect_log "disk low" "disk low: 45 GB free"
reset; export FAKE_DF_FREE=200; run
reject_log "disk fine" "disk"

reset; printf 'garbage' > "$STATE/stack_failures"; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 60)"; run
expect_log "garbled state file" "stack down (consecutive: 1)"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=empty; run_n 3
expect_log "empty server version counts as down" "force-restarting Docker Desktop (daemon unresponsive)"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=fail FAKE_ENGINE_ETIME=01:30; run_n 5
expect_log "booting engine: left alone" "engine started 90s ago; letting it boot"
reject_log "booting engine: left alone" "force-restarting"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=fail; run_n 3
age_stamp last_docker_restart; run
age_stamp last_docker_restart; run
expect_log "restart cap" "still down after 2 restarts - not restarting again"
[ "$(grep -c "docker: force-restarting Docker Desktop" "$LOGFILE")" -eq 2 ] && pass=$((pass + 1)) \
  || { fail=$((fail + 1)); echo "FAIL [restart cap]: expected exactly 2 restarts"; cat "$LOGFILE"; }
export FAKE_PUBLIC=200; run
export FAKE_PUBLIC=502; age_stamp last_docker_restart; run_n 3
[ "$(grep -c "docker: force-restarting Docker Desktop" "$LOGFILE")" -eq 3 ] && pass=$((pass + 1)) \
  || { fail=$((fail + 1)); echo "FAIL [restart cap resets after recovery]"; cat "$LOGFILE"; }

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT_MODE=hang; TIMEOUT=2; run_n 3; unset TIMEOUT
expect_log "inspect hang escalates" "force-restarting Docker Desktop (docker inspect failed (rc=124))"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT_MODE=error; run_n 3
expect_log "inspect error escalates" "docker inspect failed (rc=1)"
reject_log "inspect error escalates" "not found"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 3600)"; run_n 5
expect_log "escalation: first a container restart" "restarting the container"
age_stamp last_backend_restart; run
expect_log "escalation: then Docker" "force-restarting Docker Desktop (backend still failing after a container restart)"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_INSPECT="running $(iso_ago 3600)" FAKE_DEPLOY_PID=4242 FAKE_DEPLOY_ETIME=02:30; run_n 3
expect_log "deploy resets the grace" "compose/build operation in progress"
unset FAKE_DEPLOY_PID; : > "$LOGFILE"; run
expect_log "deploy resets the grace" "stack down (consecutive: 1)"

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=hang FAKE_DEPLOY_PID=4242 FAKE_DEPLOY_ETIME=02:30; TIMEOUT=2; run_n 3; unset TIMEOUT
expect_log "hung daemon ignores deploy" "force-restarting Docker Desktop"

# Deploy regex, matched with the same ERE engine family pgrep uses.
eval "$(grep '^DEPLOY_PATTERN=' "$WATCHDOG")"
for cmd in "docker compose -f a.yml -f b.yml up -d backend" \
           "/Users/x/.docker/cli-plugins/docker-compose compose up -d" \
           "docker-compose stop" "docker compose build backend" \
           "docker build -t x ." "docker buildx build ." "docker compose restart"; do
  echo "$cmd" | grep -Eq "$DEPLOY_PATTERN" && pass=$((pass + 1)) \
    || { fail=$((fail + 1)); echo "FAIL [deploy regex]: missed '$cmd'"; }
done
for cmd in "docker compose -f /Users/x/terraform_gcp_search/docker-compose.yml logs -f backend" \
           "docker compose ps" "docker compose run --rm x ls" "vim docker-compose.mbp.yml" \
           "docker compose exec backend ls /startup" "docker ps"; do
  echo "$cmd" | grep -Eq "$DEPLOY_PATTERN" \
    && { fail=$((fail + 1)); echo "FAIL [deploy regex]: wrongly matched '$cmd'"; } || pass=$((pass + 1))
done

# DRY_RUN without explicit paths must not touch the live log or state.
reset; mkdir -p "$WORK/home/Library/Logs" "$WORK/tmp"
PATH="$STUBS:/usr/bin:/bin:/usr/sbin:/sbin" DRY_RUN=1 HOME="$WORK/home" TMPDIR="$WORK/tmp" \
  WATCHDOG_PUBLIC_URL="$PUBLIC" WATCHDOG_LOCAL_URL="$LOCAL" FAKE_PUBLIC=502 \
  WATCHDOG_MAINTENANCE_FILE="$MAINT" /bin/bash "$WATCHDOG"
if [ -s "$WORK/home/Library/Logs/tunnel-watchdog-dryrun.log" ] && [ -d "$WORK/tmp/tunnel-watchdog-dryrun" ] \
   && [ ! -e "$WORK/home/Library/Logs/tunnel-watchdog.log" ] && [ ! -e "$WORK/tmp/tunnel-watchdog" ]; then
  pass=$((pass + 1))
else fail=$((fail + 1)); echo "FAIL [dry-run isolation]"; ls -R "$WORK/home" "$WORK/tmp"; fi

reset; export FAKE_PUBLIC=502 FAKE_LOCAL=000 FAKE_DOCKER_INFO=hang; TIMEOUT=2; run_n 3; unset TIMEOUT
expect_no_calls "dry run touches nothing"

echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]
