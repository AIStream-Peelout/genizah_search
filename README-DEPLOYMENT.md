# Manual Deployment Guide

This guide explains how to manually deploy the Cairo Genizah Search application to your MacBook Pro using Docker containers built by GitHub Actions.

## Prerequisites

### 1. Docker Desktop
Make sure Docker Desktop is installed and running on your MacBook Pro:
```bash
docker --version
docker-compose --version
```

### 2. Docker Hub Access
The GitHub Actions workflow builds and pushes images to Docker Hub. You'll need:
- A Docker Hub account
- Access to pull the images (they should be public or you need to login)

### 3. Environment Setup
Ensure you have the required environment variables set up in your `.env` file.

## Deployment Process

### Step 1: Pull Latest Images

After GitHub Actions successfully builds and pushes new images, pull them to your MacBook Pro:

```bash
# Navigate to your project directory
cd /Users/isaac/Documents/GitHub/genizah_search

# Login to Docker Hub (if needed)
docker login

# Pull the latest images
docker pull <your-dockerhub-username>/cairogenizah-backend:latest
docker pull <your-dockerhub-username>/cairogenizah-frontend:latest
```

### Step 2: Tag Images for Local Use

Tag the pulled images for use with docker-compose:

```bash
# Tag backend image
docker tag <your-dockerhub-username>/cairogenizah-backend:latest genizah_search-backend:latest

# Tag frontend image
docker tag <your-dockerhub-username>/cairogenizah-frontend:latest genizah_search-frontend:latest
```

### Step 3: Deploy Using Docker Compose

Stop existing containers and start new ones:

```bash
# Stop existing containers
docker-compose down

# Start new containers
docker-compose up -d
```

### Step 4: Verify Deployment

Check that all services are running:

```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8000/health  # Backend health check
curl http://localhost:3000         # Frontend
curl http://localhost:9200/_cluster/health  # Elasticsearch
```

## Using the Deployment Script

For easier deployment, you can use the provided deployment script:

```bash
# Make the script executable
chmod +x scripts/deploy.sh

# Deploy specific images
./scripts/deploy.sh deploy <your-dockerhub-username>/cairogenizah-backend:latest <your-dockerhub-username>/cairogenizah-frontend:latest

# Check deployment status
./scripts/deploy.sh status

# Check service health
./scripts/deploy.sh health

# Create backup
./scripts/deploy.sh backup

# Clean up old images
./scripts/deploy.sh cleanup
```

## Manual Deployment Process

The current `docker-compose.yml` is configured for local development (building containers locally). For production deployment using pre-built images from Docker Hub, you have two options:

### Option 1: Temporary Override (Recommended)
Create a temporary docker-compose override file for production:

```bash
# Create a production override file
cat > docker-compose.prod.yml << EOF
services:
  backend:
    image: genizah_search-backend:latest
    # Override the build section with image
  frontend:
    image: genizah_search-frontend:latest
    # Override the build section with image
EOF

# Deploy using the override
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Option 2: Manual Container Management
Skip docker-compose and manage containers directly:

```bash
# Stop existing containers
docker-compose down

# Run containers directly
docker run -d --name genizah-backend \
  -p 8000:8000 \
  --env-file .env \
  -v ./cache/embedding_cache:/app/embedding_cache \
  -v ./logs:/app/logs \
  genizah_search-backend:latest

docker run -d --name genizah-frontend \
  -p 3000:80 \
  -e REACT_APP_API_URL=https://api.cairogenizah.ai \
  genizah_search-frontend:latest
```

## Environment Variables

Ensure your `.env` file contains all necessary variables:

```bash
# GCP Configuration
GCP_PROJECT_ID=your-project-id
ENVIRONMENT=production

# CORS Configuration
ALLOWED_ORIGINS=https://cairogenizah.ai,https://api.cairogenizah.ai
CORS_ORIGINS=https://cairogenizah.ai,https://api.cairogenizah.ai

# Elasticsearch/Kibana
KIBANA_USER=cairo_user
KIBANA_SERVICE_PASSWORD=your-secure-password
```

## Monitoring and Maintenance

### Health Checks
Regularly check service health:
```bash
# Backend health
curl -f http://localhost:8000/health

# Frontend
curl -f http://localhost:3000

# Elasticsearch
curl -f http://localhost:9200/_cluster/health
```

### Logs
Monitor application logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f elasticsearch
```

### Backup
Create regular backups:
```bash
# Use the deployment script
./scripts/deploy.sh backup

# Or manually backup Elasticsearch data
cp -r backups/elasticsearch backups/elasticsearch-$(date +%Y%m%d-%H%M%S)
```

### Cleanup
Remove old Docker images to save space:
```bash
# Use the deployment script
./scripts/deploy.sh cleanup

# Or manually
docker image prune -f
docker system prune -f
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure ports 3000, 8000, 9200, and 5601 are available
2. **Permission issues**: Ensure Docker has proper permissions
3. **Memory issues**: Elasticsearch may need more memory - adjust `ES_JAVA_OPTS` in docker-compose.yml
4. **Network issues**: Check Docker network configuration

### Debugging Commands

```bash
# Check Docker status
docker info

# Check running containers
docker ps

# Check container logs
docker logs <container-name>

# Check resource usage
docker stats

# Check Docker networks
docker network ls
```

### Rollback

If deployment fails, you can rollback to previous images:

```bash
# Stop current containers
docker-compose down

# Pull previous image versions
docker pull <your-dockerhub-username>/cairogenizah-backend:<previous-commit-sha>
docker pull <your-dockerhub-username>/cairogenizah-frontend:<previous-commit-sha>

# Tag and restart
docker tag <your-dockerhub-username>/cairogenizah-backend:<previous-commit-sha> genizah_search-backend:latest
docker tag <your-dockerhub-username>/cairogenizah-frontend:<previous-commit-sha> genizah_search-frontend:latest

docker-compose up -d
```

## Security Considerations

1. **Environment Variables**: Never commit sensitive data to version control
2. **Docker Hub**: Use private repositories for sensitive applications
3. **Network Security**: Consider using reverse proxy (nginx) for production
4. **Updates**: Regularly update base images and dependencies

## Performance Optimization

1. **Resource Limits**: Set appropriate memory and CPU limits in docker-compose.yml
2. **Elasticsearch**: Tune JVM heap size based on available memory
3. **Caching**: Use Docker layer caching for faster builds
4. **Monitoring**: Implement proper monitoring and alerting

## Support

For issues or questions:
1. Check the GitHub Actions logs for build failures
2. Review Docker container logs
3. Check the deployment log: `cat deployment.log`
4. Verify environment variables and configuration
