# Multi-stage production Dockerfile for Agent Mesh
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r meshuser && useradd -r -g meshuser meshuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY deploy/entrypoint.sh ./entrypoint.sh

# Make scripts executable
RUN chmod +x ./entrypoint.sh ./scripts/*.py

# Create data directory
RUN mkdir -p /app/data && chown meshuser:meshuser /app/data

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV NODE_ID=""
ENV LISTEN_PORT=4001
ENV NODE_ROLE=worker
ENV BOOTSTRAP_PEERS=""
ENV REGION=us-east-1
ENV LOG_LEVEL=INFO
ENV DATABASE_URL=sqlite:///data/agent_mesh.db
ENV PROMETHEUS_PORT=9090
ENV HEALTH_CHECK_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${HEALTH_CHECK_PORT}/health || exit 1

# Switch to non-root user
USER meshuser

# Expose ports
EXPOSE 4001 8080 9090

# Run application
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python3", "-m", "examples.scaling_mesh_demo"]