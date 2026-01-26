# backend/Dockerfile
FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (Run from repo root: docker build -f src/backend/Dockerfile .)
COPY src/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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