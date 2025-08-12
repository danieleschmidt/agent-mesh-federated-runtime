FROM python:3.11-slim

LABEL maintainer="Terragon Labs"
LABEL version="1.0.0"
LABEL description="Agent Mesh - Autonomous Federated Learning System"

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Create non-root user
RUN useradd -m -s /bin/bash agentmesh
RUN chown -R agentmesh:agentmesh /app
USER agentmesh

# Environment variables
ENV PYTHONPATH=/app/src
ENV AGENT_MESH_CONFIG=/app/configs/production.json
ENV AGENT_MESH_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 scripts/health_check.py || exit 1

# Expose ports
EXPOSE 8080 8081 4001

# Run the application
CMD ["python3", "-m", "agent_mesh.core.mesh_node", "--config", "/app/configs/production.json"]
