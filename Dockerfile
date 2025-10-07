# ========================
# Stage 1 - Builder
# ========================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps into a venv (lighter runtime)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt


# ========================
# Stage 2 - Final Runtime
# ========================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy only source code (not your local .env)
COPY . .

# Create downloads folder
RUN mkdir -p /app/downloads

# Expose healthcheck + webhook ports
EXPOSE 8080 8443

# Default command
CMD ["python", "app.py"]
