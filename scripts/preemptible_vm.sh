#!/bin/bash

# deploy-preemptible-vm-fixed.sh - Fixed version with proper disk size and metadata
VM_NAME="cairo-app-vm"
ZONE="us-central1-a"
MACHINE_TYPE="e2-highmem-4"  # 4 vCPUs, 32GB RAM

echo "Creating PREEMPTIBLE VM for Cairo Genizah POC..."
echo "Machine type: $MACHINE_TYPE (4 vCPUs, 32GB RAM)"
echo "Cost: ~$35-40/month (70% savings vs standard VM)"
echo "⚠️  Note: VM can be preempted with 30 second notice"
echo ""

# Create a separate startup script file first
cat > startup-script.sh << 'STARTUP_EOF'
#!/bin/bash

# Update system
echo "Setting up Cairo Genizah preemptible VM..." > /var/log/startup.log
apt update && apt upgrade -y

# Install Docker
echo "Installing Docker..." >> /var/log/startup.log
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
echo "Installing Docker Compose..." >> /var/log/startup.log
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install useful tools
apt install -y git curl htop jq

# Create app directory with proper permissions
mkdir -p /app
chown ubuntu:ubuntu /app

# Create deployment scripts directory
mkdir -p /app/scripts
chown ubuntu:ubuntu /app/scripts

# Create preemption handler script
cat > /app/scripts/handle-preemption.sh << 'EOF'
#!/bin/bash
echo "$(date): Preemption detected, shutting down gracefully..." >> /var/log/preemption.log

cd /app
if [ -f "docker-compose.yml" ]; then
    docker-compose down --timeout 20
    echo "$(date): Containers stopped" >> /var/log/preemption.log
fi

if [ -d "/app/src/backend/embedding_cache" ]; then
    echo "$(date): Embedding cache preserved" >> /var/log/preemption.log
fi

echo "$(date): Graceful shutdown complete" >> /var/log/preemption.log
EOF

chmod +x /app/scripts/handle-preemption.sh
chown ubuntu:ubuntu /app/scripts/handle-preemption.sh

# Create auto-restart script
cat > /app/scripts/auto-restart.sh << 'EOF'
#!/bin/bash
cd /app

while ! docker info > /dev/null 2>&1; do
    echo "Waiting for Docker to start..."
    sleep 5
done

if [ -f "docker-compose.yml" ]; then
    echo "$(date): Auto-restarting application after VM restart" >> /var/log/auto-restart.log
    docker-compose up -d
    echo "$(date): Application restarted" >> /var/log/auto-restart.log
else
    echo "$(date): No docker-compose.yml found, skipping auto-restart" >> /var/log/auto-restart.log
fi
EOF

chmod +x /app/scripts/auto-restart.sh
chown ubuntu:ubuntu /app/scripts/auto-restart.sh

# Set up auto-restart service
cat > /etc/systemd/system/cairo-auto-restart.service << 'EOF'
[Unit]
Description=Auto-restart Cairo app after VM restart
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/app/scripts/auto-restart.sh
User=ubuntu
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

systemctl enable cairo-auto-restart.service

# Create deployment script
cat > /app/scripts/deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting deployment on preemptible VM..."

cd /app && docker-compose down || true

if [ -d "/app/.git" ]; then
    git pull
fi

echo "Building images (this may take a few minutes for 7B model)..."
docker-compose build --no-cache

echo "Starting containers..."
docker-compose up -d

echo "Waiting for NOMICs 7B model to load (this can take 1-2 minutes)..."
sleep 120

echo "Running health checks..."
for i in {1..10}; do
    if curl -f http://localhost:8000/health; then
        echo "Backend is healthy"
        break
    else
        echo "Waiting for backend... (attempt $i/10)"
        sleep 10
    fi
done

curl -f http://localhost:3000 || echo "Frontend health check failed"

echo "Deployment complete!"
docker-compose ps

echo "Checking memory usage..."
free -h

echo "Model loading status..."
docker-compose logs backend | grep -i "model\|embedding\|loading" | tail -5 || echo "No model loading logs found"
EOF

chmod +x /app/scripts/deploy.sh
chown ubuntu:ubuntu /app/scripts/deploy.sh

# Create status script
cat > /app/scripts/status.sh << 'EOF'
#!/bin/bash
echo "=== Cairo Genizah App Status (Preemptible VM) ==="
echo "Date: $(date)"
echo ""
echo "=== VM Status ==="
echo "Preemptible: Yes (can be interrupted with 30s notice)"
uptime
echo ""
echo "=== Memory Usage (Important for 7B model) ==="
free -h
echo ""
echo "=== Docker Containers ==="
cd /app && docker-compose ps 2>/dev/null || echo "Docker Compose not running"
echo ""
echo "=== Disk Usage ==="
df -h /
echo ""
echo "=== Service Health ==="
curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Backend not responding"
curl -s -o /dev/null -w "Frontend HTTP: %{http_code}" http://localhost:3000 && echo ""
echo ""
echo "=== Recent Logs ==="
if [ -f "/var/log/preemption.log" ]; then
    echo "Last preemption events:"
    tail -3 /var/log/preemption.log 2>/dev/null || echo "No preemption events"
fi
echo ""
echo "Memory-intensive processes:"
ps aux --sort=-%mem | head -5
EOF

chmod +x /app/scripts/status.sh
chown ubuntu:ubuntu /app/scripts/status.sh

echo "Preemptible VM setup completed at $(date)" >> /var/log/startup.log
STARTUP_EOF

# Create the preemptible VM with fixed parameters
gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --preemptible \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=cairo-app-server \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script=startup-script.sh

# Create firewall rules
echo "Creating firewall rules..."
gcloud compute firewall-rules create allow-cairo-app \
  --allow tcp:3000,tcp:8000 \
  --source-ranges 0.0.0.0/0 \
  --target-tags cairo-app-server \
  --description "Allow Cairo app traffic" || echo "Firewall rule may already exist"

# Clean up the temporary startup script
rm startup-script.sh

echo ""
echo "✅ Preemptible VM created successfully!"
echo ""
echo "💾 DISK IMPROVEMENTS:"
echo "  - Increased to 200GB SSD for better I/O performance"
echo "  - Better for 7B model loading and Docker operations"
echo ""
echo "💰 COST BREAKDOWN:"
echo "  Compute (preemptible): ~$35/month"
echo "  Storage (200GB SSD): ~$8/month"
echo "  Network: ~$1-3/month"
echo "  Total: ~$44-46/month"
echo "  (Still 60% cheaper than standard VM!)"
echo ""
echo "⚠️  PREEMPTION HANDLING:"
echo "  - VM can be shut down with 30 second notice"
echo "  - Auto-restart service will restart your app"
echo "  - Embedding cache persists across restarts"
echo "  - Model reload takes ~1-2 minutes"
echo ""
echo "🔧 NEXT STEPS:"
VM_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "Getting IP...")
echo "  VM IP: $VM_IP"
echo "  Check setup: gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/startup.log'"
echo "  SSH: gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "  Status: gcloud compute ssh $VM_NAME --zone=$ZONE --command='/app/scripts/status.sh'"
echo ""
echo "🚀 The VM is being set up. Wait 3-5 minutes for setup to complete!"
echo "   Setup includes: Docker, Docker Compose, auto-restart services"