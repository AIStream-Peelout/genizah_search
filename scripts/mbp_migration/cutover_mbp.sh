#!/bin/bash
# Point the public site at this MBP by starting a second cloudflared connector
# for the existing tunnel, as a user LaunchAgent that restarts on its own.
#
# Runs ON THE MBP after bootstrap_mbp.sh has passed. Cloudflare load-balances
# across all live connectors of one tunnel, so the Studio's connector can keep
# running until this one is verified; stopping the Studio's is a separate,
# manual step (see docs/MBP_MIGRATION.md).
set -euo pipefail

CONFIG="$HOME/.cloudflared/config.yml"
LABEL="com.cairogenizah.cloudflared"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/cloudflared.log"

[ -f "$CONFIG" ] || { echo "Missing $CONFIG (export_to_mbp.sh step 'cloudflared')."; exit 1; }
TUNNEL_ID="$(awk '/^tunnel:/ {print $2}' "$CONFIG")"
[ -n "$TUNNEL_ID" ] || { echo "No 'tunnel:' line in $CONFIG"; exit 1; }

if ! command -v cloudflared >/dev/null; then
  command -v brew >/dev/null || { echo "Install Homebrew, then: brew install cloudflared"; exit 1; }
  brew install cloudflared
fi
CLOUDFLARED="$(command -v cloudflared)"

# launchd pins a service to the code signature of the program it launches; a
# Homebrew upgrade of cloudflared then makes launchd refuse to spawn it (exit 78,
# no log line; seen 2026-09-28). Launch through a bash wrapper instead: bash is
# Apple-signed and exec() follows the Homebrew symlink to the current binary.
WRAPPER="$HOME/Library/Application Support/cairogenizah/run_cloudflared.sh"
mkdir -p "$(dirname "$WRAPPER")" "$(dirname "$PLIST")"
printf '#!/bin/bash\nexec %s tunnel --config "%s" run %s\n' "$CLOUDFLARED" "$CONFIG" "$TUNNEL_ID" > "$WRAPPER"
chmod +x "$WRAPPER"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$WRAPPER</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "cloudflared LaunchAgent loaded ($PLIST); log: $LOG"

sleep 8
echo
echo "Connectors registered for tunnel $TUNNEL_ID (expect this MBP and the Studio):"
"$CLOUDFLARED" tunnel info "$TUNNEL_ID" | sed -n '1,30p'

echo
for i in 1 2 3 4 5; do
  code="$(curl -s -o /dev/null -w '%{http_code}' https://api.cairogenizah.ai/health)"
  echo "api.cairogenizah.ai/health -> $code"
  sleep 1
done
echo
echo "If both connectors show and /health is 200, the Studio's cloudflared can be stopped"
echo "(on the Studio, with approval: pkill -f 'cloudflared tunnel run'). Then re-run"
echo "'cloudflared tunnel info $TUNNEL_ID' here and confirm only this MBP remains."
