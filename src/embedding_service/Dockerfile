# embedding_service/Dockerfile
FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create the non-root user and writable cache mount points. Named volumes copy
# these ownership bits on first use, so fresh Linux/CI hosts do not create
# root-owned cache directories that prevent Hugging Face model downloads.
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/embedding_cache /app/hf_home \
    && chown -R app:app /app
USER app

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

