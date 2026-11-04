# backend/Dockerfile
# Base image pinned by digest to the exact build production runs: Python 3.12.15
# on Debian trixie, built 2026-10-06 (its arm64 layers match the live image
# 718a87af5379). Tags move (`python:3.12` is now 3.12.16, and even
# `python:3.12.15-trixie` was rebuilt on 2026-10-21); the digest does not.
FROM python:3.12.15-trixie@sha256:63828510c8b5ccce3bf0d6fabd6f3d17d4effa1ffe690f80a67c8e2d394e03ee

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (Run from repo root: docker build -f src/backend/Dockerfile .)
# constraints.txt locks every transitive package to production's versions.
COPY src/backend/requirements.txt src/backend/constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy entire src package
COPY src/ /app/src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app

# Set PYTHONPATH to include /app so "import src.backend" works
ENV PYTHONPATH=/app

# Set working directory to backend specific folder for script compatibility
WORKDIR /app/src/backend

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

USER app

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]