# Multi-stage Dockerfile for Agent Mesh Federated Runtime

# =============================================================================
# Base stage - Common dependencies
# =============================================================================
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# =============================================================================
# Dependencies stage - Install Python dependencies
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# =============================================================================
# Development stage - For development environment
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# Expose ports
EXPOSE 4001 5001 8000 9090

# Development command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Testing stage - For running tests
# =============================================================================
FROM development as testing

USER root

# Install additional testing tools
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v"]

# =============================================================================
# Builder stage - Build the application
# =============================================================================
FROM dependencies as builder

# Copy source code
COPY src/ ./src/
COPY README.md LICENSE ./

# Build the package
RUN python -m build

# =============================================================================
# Production stage - Minimal production image
# =============================================================================
FROM python:3.13-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    libffi8 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/configs && \
    chown -R appuser:appuser /app

# Copy configuration files
COPY --chown=appuser:appuser configs/ ./configs/

# Switch to non-root user
USER appuser

# Create health check script
COPY --chown=appuser:appuser <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import sys
import requests
import os

def check_health():
    try:
        port = os.getenv('HEALTH_CHECK_PORT', '8080')
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        if response.status_code == 200:
            print("Health check passed")
            sys.exit(0)
        else:
            print(f"Health check failed with status {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Health check error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_health()
EOF

RUN chmod +x /app/healthcheck.py

# Expose ports
EXPOSE 4001 5001 8000 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python /app/healthcheck.py

# Production command
CMD ["agent-mesh", "--config", "/app/configs/production.yaml"]

# =============================================================================
# Edge stage - Optimized for edge devices
# =============================================================================
FROM python:3.13-alpine as edge

# Set edge environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=edge \
    LOG_LEVEL=WARNING

# Install minimal dependencies
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    openssl-dev \
    curl

# Create non-root user
RUN addgroup -g 1000 appuser && \
    adduser -u 1000 -G appuser -s /bin/sh -D appuser

WORKDIR /app

# Copy and install minimal package
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && \
    rm /tmp/*.whl

# Install only essential runtime dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    aiofiles

# Remove build dependencies
RUN apk del build-base

# Create minimal directories
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose minimal ports
EXPOSE 4001 8000

# Minimal health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Edge-optimized command
CMD ["agent-mesh", "--edge-mode", "--minimal-resources"]

# =============================================================================
# Monitoring stage - With monitoring tools
# =============================================================================
FROM production as monitoring

USER root

# Install monitoring tools
RUN apt-get update && apt-get install -y \
    prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

# Install Python monitoring dependencies
RUN pip install --no-cache-dir \
    prometheus-client \
    grafana-api \
    opentelemetry-api \
    opentelemetry-sdk

USER appuser

# Copy monitoring configurations
COPY --chown=appuser:appuser monitoring/ ./monitoring/

# Expose monitoring ports
EXPOSE 9100 9090

# Monitoring command
CMD ["sh", "-c", "node_exporter --web.listen-address=:9100 & agent-mesh --enable-monitoring"]

# =============================================================================
# GPU stage - With CUDA support for ML acceleration
# =============================================================================
FROM nvidia/cuda:12.9.1-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=gpu \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy and install package with GPU support
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Install GPU-specific dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN chown -R appuser:appuser /app
USER appuser

# GPU-optimized command
CMD ["agent-mesh", "--enable-gpu", "--cuda-memory-fraction", "0.8"]

# =============================================================================
# Multi-architecture support
# =============================================================================

# Build arguments for multi-arch
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Labels for metadata
LABEL org.opencontainers.image.title="Agent Mesh Federated Runtime"
LABEL org.opencontainers.image.description="Decentralized peer-to-peer runtime for federated learning and multi-agent systems"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Daniel Schmidt <daniel@terragon.ai>"
LABEL org.opencontainers.image.url="https://github.com/your-org/agent-mesh-federated-runtime"
LABEL org.opencontainers.image.source="https://github.com/your-org/agent-mesh-federated-runtime"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="2024-01-01T00:00:00Z"
LABEL org.opencontainers.image.revision="main"

# Security labels
LABEL security.non-root="true"
LABEL security.user="appuser"
LABEL security.capabilities="none"