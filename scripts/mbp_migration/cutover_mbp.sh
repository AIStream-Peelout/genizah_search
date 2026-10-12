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

mkdir -p "$(dirname "$PLIST")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$CLOUDFLARED</string>
    <string>tunnel</string>
    <string>--config</string><string>$CONFIG</string>
    <string>run</string>
    <string>$TUNNEL_ID</string>
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
