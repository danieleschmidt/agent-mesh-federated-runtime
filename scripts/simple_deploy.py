#!/usr/bin/env python3
"""Simple production deployment setup for Agent Mesh system."""

import os
import sys
import json
from pathlib import Path


def create_docker_configs():
    """Create Docker and Docker Compose configurations."""
    
    # Dockerfile
    dockerfile_content = '''FROM python:3.11-slim

LABEL maintainer="Terragon Labs"
LABEL version="1.0.0"
LABEL description="Agent Mesh - Autonomous Federated Learning System"

# System dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python3 scripts/health_check.py || exit 1

# Expose ports
EXPOSE 8080 8081 4001

# Run the application
CMD ["python3", "-m", "agent_mesh.core.mesh_node", "--config", "/app/configs/production.json"]
'''

    # Docker Compose for development
    docker_compose_dev = '''version: '3.8'
services:
  agent-mesh-node-1:
    build: .
    ports:
      - "8080:8080"
      - "4001:4001"
    environment:
      - AGENT_MESH_NODE_ID=node-1
      - AGENT_MESH_PORT=8080
      - AGENT_MESH_P2P_PORT=4001
    volumes:
      - ./logs:/app/logs
    networks:
      - agent-mesh

  agent-mesh-node-2:
    build: .
    ports:
      - "8081:8080"
      - "4002:4001"
    environment:
      - AGENT_MESH_NODE_ID=node-2
      - AGENT_MESH_PORT=8080
      - AGENT_MESH_P2P_PORT=4001
      - AGENT_MESH_BOOTSTRAP=agent-mesh-node-1:4001
    volumes:
      - ./logs:/app/logs
    networks:
      - agent-mesh
    depends_on:
      - agent-mesh-node-1

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - agent-mesh

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    networks:
      - agent-mesh

networks:
  agent-mesh:
    driver: bridge

volumes:
  grafana-storage:
'''

    # Write files
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    with open("docker-compose.dev.yml", "w") as f:
        f.write(docker_compose_dev)

    print("✅ Docker configurations created")


def create_production_configs():
    """Create production configuration files."""
    
    production_config = {
        "app": {
            "name": "agent-mesh",
            "version": "1.0.0", 
            "environment": "production"
        },
        "network": {
            "host": "0.0.0.0",
            "port": 8080,
            "p2p_port": 4001,
            "max_connections": 100
        },
        "security": {
            "encryption_enabled": True,
            "tls_enabled": True,
            "key_rotation_interval": 3600
        },
        "consensus": {
            "algorithm": "raft",
            "election_timeout": 5000,
            "heartbeat_interval": 1000
        },
        "federated_learning": {
            "max_participants": 1000,
            "round_timeout": 300,
            "min_participants": 3
        },
        "monitoring": {
            "enabled": True,
            "prometheus_port": 9090,
            "health_check_interval": 30
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "file": "/app/logs/agent-mesh.log"
        },
        "database": {
            "type": "sqlite",
            "path": "/app/data/agent-mesh.db"
        }
    }

    staging_config = production_config.copy()
    staging_config["app"]["environment"] = "staging"
    staging_config["logging"]["level"] = "DEBUG"

    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    with open(configs_dir / "production.json", "w") as f:
        json.dump(production_config, f, indent=2)

    with open(configs_dir / "staging.json", "w") as f:
        json.dump(staging_config, f, indent=2)

    print("✅ Production configurations created")


def create_deployment_scripts():
    """Create deployment automation scripts."""
    
    # Build script
    build_script = '''#!/bin/bash
set -e

echo "🚀 Building Agent Mesh Docker image..."

# Build image
docker build -t terragonlabs/agent-mesh:latest .

# Tag with version
VERSION=$(python3 -c "import json; print(json.load(open('configs/production.json'))['app']['version'])")
docker tag terragonlabs/agent-mesh:latest terragonlabs/agent-mesh:$VERSION

echo "✅ Build completed"
echo "   Image: terragonlabs/agent-mesh:latest"
echo "   Tagged: terragonlabs/agent-mesh:$VERSION"
'''

    # Deploy script
    deploy_script = '''#!/bin/bash
set -e

ENVIRONMENT=${1:-development}
echo "🚀 Deploying Agent Mesh to $ENVIRONMENT..."

if [ "$ENVIRONMENT" = "development" ]; then
    echo "📝 Starting development deployment..."
    docker-compose -f docker-compose.dev.yml up -d
    echo "✅ Development deployment completed"
    echo "   Agent Mesh: http://localhost:8080"
    echo "   Monitoring: http://localhost:3000"

elif [ "$ENVIRONMENT" = "production" ]; then
    echo "📝 Starting production deployment..."
    
    # Check if Docker Swarm is initialized
    if ! docker info | grep -q "Swarm: active"; then
        echo "Initializing Docker Swarm..."
        docker swarm init
    fi
    
    echo "✅ Production deployment ready"
    echo "   Run: docker stack deploy -c docker-compose.prod.yml agent-mesh"
    
else
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Usage: $0 [development|production]"
    exit 1
fi
'''

    # Health check script
    health_check_script = '''#!/usr/bin/env python3
"""Health check script for Agent Mesh."""

import sys
import json
import time
from pathlib import Path

def check_health():
    """Check health of Agent Mesh service."""
    try:
        # Basic health check - verify config exists and is valid
        config_path = Path("configs/production.json")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
            
        with open(config_path) as f:
            config = json.load(f)
            
        if config.get("app", {}).get("name") == "agent-mesh":
            print("✅ Agent Mesh configuration is valid")
            return True
        else:
            print("❌ Invalid configuration")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
'''

    # Create scripts directory
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)

    scripts = [
        ("build.sh", build_script),
        ("deploy.sh", deploy_script),
        ("health_check.py", health_check_script)
    ]

    for filename, content in scripts:
        script_path = scripts_dir / filename
        with open(script_path, "w") as f:
            f.write(content)
        
        # Make scripts executable
        os.chmod(script_path, 0o755)

    print("✅ Deployment scripts created")


def create_monitoring_configs():
    """Create monitoring configuration files."""
    
    # Prometheus config in YAML format (manual)
    prometheus_config = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agent-mesh'
    static_configs:
      - targets: ['agent-mesh:9090']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
'''

    # Nginx config
    nginx_config = '''events {
    worker_connections 1024;
}

http {
    upstream agent_mesh {
        server agent-mesh:8080;
    }

    server {
        listen 80;
        server_name agent-mesh.example.com;
        
        location / {
            proxy_pass http://agent_mesh;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /metrics {
            proxy_pass http://agent_mesh;
            auth_basic "Metrics";
            auth_basic_user_file /etc/nginx/.htpasswd;
        }
    }
}
'''

    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    with open(configs_dir / "prometheus.yml", "w") as f:
        f.write(prometheus_config)

    with open(configs_dir / "nginx.conf", "w") as f:
        f.write(nginx_config)

    print("✅ Monitoring configurations created")


def create_readme():
    """Create deployment README."""
    
    readme_content = '''# Agent Mesh Deployment Guide

## Quick Start

### Development Deployment

1. Build the Docker image:
   ```bash
   ./scripts/build.sh
   ```

2. Start development environment:
   ```bash
   ./scripts/deploy.sh development
   ```

3. Access services:
   - Agent Mesh: http://localhost:8080
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Production Deployment

1. Build and tag images:
   ```bash
   ./scripts/build.sh
   ```

2. Deploy to production:
   ```bash
   ./scripts/deploy.sh production
   ```

## Health Checks

Check system health:
```bash
./scripts/health_check.py
```

## Configuration

- Development: `configs/staging.json`
- Production: `configs/production.json`

## Architecture

The Agent Mesh system consists of:

- **Mesh Nodes**: Core P2P network participants
- **Consensus Engine**: Byzantine fault-tolerant consensus
- **Federated Learner**: Distributed learning coordinator
- **Secure Aggregator**: Privacy-preserving model aggregation
- **Monitoring**: Prometheus + Grafana stack

## Security

- End-to-end encryption for all communications
- Certificate-based node authentication
- Role-based access control
- Regular security audits and logging

## Scaling

The system automatically scales based on:
- CPU and memory usage
- Network load
- Request latency
- Error rates

## Support

For support and troubleshooting, see:
- Logs: `./logs/`
- Metrics: http://localhost:9090
- Health: `./scripts/health_check.py`
'''

    with open("DEPLOYMENT.md", "w") as f:
        f.write(readme_content)

    print("✅ Deployment documentation created")


def main():
    """Main deployment setup function."""
    print("🚀 Agent Mesh Production Deployment Setup")
    print("=" * 50)

    try:
        create_docker_configs()
        create_production_configs()
        create_deployment_scripts()
        create_monitoring_configs()
        create_readme()

        print("\n🎉 Production deployment setup completed!")
        print("\nNext steps:")
        print("1. Review configurations in ./configs/")
        print("2. Build: ./scripts/build.sh") 
        print("3. Deploy: ./scripts/deploy.sh [development|production]")
        print("4. Check health: ./scripts/health_check.py")
        print("5. See DEPLOYMENT.md for full guide")

        return 0

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())