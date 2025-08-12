#!/usr/bin/env python3
"""Production deployment script for Agent Mesh system."""

import os
import sys
import subprocess
import argparse
import json
import yaml
from pathlib import Path


def create_docker_configs():
    """Create Docker and Docker Compose configurations."""
    
    # Dockerfile
    dockerfile_content = """
FROM python:3.11-slim

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
ENV AGENT_MESH_CONFIG=/app/configs/production.yaml
ENV AGENT_MESH_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python scripts/health_check.py || exit 1

# Expose ports
EXPOSE 8080 8081 4001

# Run the application
CMD ["python", "-m", "agent_mesh.core.mesh_node", "--config", "/app/configs/production.yaml"]
"""

    # Docker Compose for development
    docker_compose_dev = {
        "version": "3.8",
        "services": {
            "agent-mesh-node-1": {
                "build": ".",
                "ports": ["8080:8080", "4001:4001"],
                "environment": [
                    "AGENT_MESH_NODE_ID=node-1",
                    "AGENT_MESH_PORT=8080",
                    "AGENT_MESH_P2P_PORT=4001"
                ],
                "volumes": ["./logs:/app/logs"],
                "networks": ["agent-mesh"]
            },
            "agent-mesh-node-2": {
                "build": ".",
                "ports": ["8081:8080", "4002:4001"],
                "environment": [
                    "AGENT_MESH_NODE_ID=node-2", 
                    "AGENT_MESH_PORT=8080",
                    "AGENT_MESH_P2P_PORT=4001",
                    "AGENT_MESH_BOOTSTRAP=agent-mesh-node-1:4001"
                ],
                "volumes": ["./logs:/app/logs"],
                "networks": ["agent-mesh"],
                "depends_on": ["agent-mesh-node-1"]
            },
            "agent-mesh-node-3": {
                "build": ".",
                "ports": ["8082:8080", "4003:4001"],
                "environment": [
                    "AGENT_MESH_NODE_ID=node-3",
                    "AGENT_MESH_PORT=8080", 
                    "AGENT_MESH_P2P_PORT=4001",
                    "AGENT_MESH_BOOTSTRAP=agent-mesh-node-1:4001"
                ],
                "volumes": ["./logs:/app/logs"],
                "networks": ["agent-mesh"],
                "depends_on": ["agent-mesh-node-1"]
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": ["./configs/prometheus.yml:/etc/prometheus/prometheus.yml"],
                "networks": ["agent-mesh"]
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": ["GF_SECURITY_ADMIN_PASSWORD=admin"],
                "volumes": ["grafana-storage:/var/lib/grafana"],
                "networks": ["agent-mesh"]
            }
        },
        "networks": {
            "agent-mesh": {"driver": "bridge"}
        },
        "volumes": {
            "grafana-storage": {}
        }
    }

    # Production Docker Compose
    docker_compose_prod = {
        "version": "3.8",
        "services": {
            "agent-mesh": {
                "image": "terragonlabs/agent-mesh:latest",
                "deploy": {
                    "replicas": 3,
                    "restart_policy": {"condition": "on-failure", "max_attempts": 3},
                    "resources": {
                        "limits": {"cpus": "2.0", "memory": "4G"},
                        "reservations": {"cpus": "1.0", "memory": "2G"}
                    }
                },
                "ports": ["8080-8082:8080"],
                "environment": [
                    "AGENT_MESH_ENV=production",
                    "AGENT_MESH_LOG_LEVEL=INFO"
                ],
                "volumes": [
                    "/var/log/agent-mesh:/app/logs",
                    "/etc/agent-mesh:/app/configs"
                ],
                "networks": ["agent-mesh"],
                "healthcheck": {
                    "test": ["CMD", "python", "scripts/health_check.py"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            },
            "nginx": {
                "image": "nginx:alpine",
                "ports": ["80:80", "443:443"],
                "volumes": [
                    "./configs/nginx.conf:/etc/nginx/nginx.conf",
                    "/etc/letsencrypt:/etc/letsencrypt"
                ],
                "networks": ["agent-mesh"],
                "depends_on": ["agent-mesh"]
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": ["./configs/prometheus-prod.yml:/etc/prometheus/prometheus.yml"],
                "networks": ["agent-mesh"]
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": ["GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password"],
                "secrets": ["grafana_password"],
                "volumes": ["grafana-data:/var/lib/grafana"],
                "networks": ["agent-mesh"]
            }
        },
        "networks": {
            "agent-mesh": {"driver": "overlay", "attachable": True}
        },
        "volumes": {
            "grafana-data": {}
        },
        "secrets": {
            "grafana_password": {"file": "./secrets/grafana_password.txt"}
        }
    }

    # Write files
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    with open("docker-compose.dev.yml", "w") as f:
        yaml.dump(docker_compose_dev, f, default_flow_style=False, sort_keys=False)

    with open("docker-compose.prod.yml", "w") as f:
        yaml.dump(docker_compose_prod, f, default_flow_style=False, sort_keys=False)

    print("✅ Docker configurations created")


def create_kubernetes_configs():
    """Create Kubernetes deployment configurations."""
    
    # Namespace
    namespace = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": "agent-mesh"}
    }

    # ConfigMap
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "agent-mesh-config", "namespace": "agent-mesh"},
        "data": {
            "production.yaml": """
app:
  name: "agent-mesh"
  version: "1.0.0"
  environment: "production"

network:
  host: "0.0.0.0"
  port: 8080
  p2p_port: 4001

security:
  encryption_enabled: true
  tls_enabled: true

monitoring:
  enabled: true
  prometheus_port: 9090

logging:
  level: "INFO"
  format: "json"
"""
        }
    }

    # Deployment
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment", 
        "metadata": {"name": "agent-mesh", "namespace": "agent-mesh"},
        "spec": {
            "replicas": 3,
            "selector": {"matchLabels": {"app": "agent-mesh"}},
            "template": {
                "metadata": {"labels": {"app": "agent-mesh"}},
                "spec": {
                    "containers": [{
                        "name": "agent-mesh",
                        "image": "terragonlabs/agent-mesh:latest",
                        "ports": [
                            {"containerPort": 8080, "name": "http"},
                            {"containerPort": 4001, "name": "p2p"},
                            {"containerPort": 9090, "name": "metrics"}
                        ],
                        "env": [
                            {"name": "AGENT_MESH_CONFIG", "value": "/app/configs/production.yaml"},
                            {"name": "PYTHONPATH", "value": "/app/src"}
                        ],
                        "volumeMounts": [{
                            "name": "config",
                            "mountPath": "/app/configs"
                        }],
                        "resources": {
                            "limits": {"cpu": "2000m", "memory": "4Gi"},
                            "requests": {"cpu": "1000m", "memory": "2Gi"}
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8080},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/ready", "port": 8080},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5
                        }
                    }],
                    "volumes": [{
                        "name": "config",
                        "configMap": {"name": "agent-mesh-config"}
                    }]
                }
            }
        }
    }

    # Service
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "agent-mesh-service", "namespace": "agent-mesh"},
        "spec": {
            "selector": {"app": "agent-mesh"},
            "ports": [
                {"name": "http", "port": 80, "targetPort": 8080},
                {"name": "p2p", "port": 4001, "targetPort": 4001},
                {"name": "metrics", "port": 9090, "targetPort": 9090}
            ],
            "type": "LoadBalancer"
        }
    }

    # Ingress
    ingress = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": "agent-mesh-ingress",
            "namespace": "agent-mesh",
            "annotations": {
                "kubernetes.io/ingress.class": "nginx",
                "cert-manager.io/cluster-issuer": "letsencrypt-prod"
            }
        },
        "spec": {
            "tls": [{
                "hosts": ["agent-mesh.example.com"],
                "secretName": "agent-mesh-tls"
            }],
            "rules": [{
                "host": "agent-mesh.example.com", 
                "http": {
                    "paths": [{
                        "path": "/",
                        "pathType": "Prefix",
                        "backend": {
                            "service": {
                                "name": "agent-mesh-service",
                                "port": {"number": 80}
                            }
                        }
                    }]
                }
            }]
        }
    }

    # Create k8s directory
    k8s_dir = Path("k8s")
    k8s_dir.mkdir(exist_ok=True)

    # Write YAML files
    configs = [
        ("namespace.yaml", namespace),
        ("configmap.yaml", configmap),
        ("deployment.yaml", deployment),
        ("service.yaml", service),
        ("ingress.yaml", ingress)
    ]

    for filename, config in configs:
        with open(k8s_dir / filename, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("✅ Kubernetes configurations created")


def create_monitoring_configs():
    """Create monitoring configuration files."""
    
    # Prometheus config
    prometheus_config = {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s"
        },
        "scrape_configs": [
            {
                "job_name": "agent-mesh",
                "static_configs": [{
                    "targets": ["agent-mesh:9090"]
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "10s"
            },
            {
                "job_name": "prometheus",
                "static_configs": [{
                    "targets": ["localhost:9090"]
                }]
            }
        ]
    }

    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    with open(configs_dir / "prometheus.yml", "w") as f:
        yaml.dump(prometheus_config, f, default_flow_style=False)

    # Nginx config
    nginx_config = """
events {
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
"""

    with open(configs_dir / "nginx.conf", "w") as f:
        f.write(nginx_config)

    print("✅ Monitoring configurations created")


def create_deployment_scripts():
    """Create deployment automation scripts."""
    
    # Build script
    build_script = """#!/bin/bash
set -e

echo "🚀 Building Agent Mesh Docker image..."

# Build image
docker build -t terragonlabs/agent-mesh:latest .

# Tag with version
VERSION=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
docker tag terragonlabs/agent-mesh:latest terragonlabs/agent-mesh:$VERSION

echo "✅ Build completed"
echo "   Image: terragonlabs/agent-mesh:latest"
echo "   Tagged: terragonlabs/agent-mesh:$VERSION"
"""

    # Deploy script
    deploy_script = """#!/bin/bash
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
    
    # Deploy stack
    docker stack deploy -c docker-compose.prod.yml agent-mesh
    echo "✅ Production deployment completed"
    
elif [ "$ENVIRONMENT" = "kubernetes" ]; then
    echo "📝 Deploying to Kubernetes..."
    kubectl apply -f k8s/
    echo "✅ Kubernetes deployment completed"
    
else
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Usage: $0 [development|production|kubernetes]"
    exit 1
fi
"""

    # Health check script
    health_check_script = """#!/usr/bin/env python3
\"\"\"Health check script for Agent Mesh.\"\"\"

import sys
import requests
import time

def check_health():
    \"\"\"Check health of Agent Mesh service.\"\"\"
    try:
        response = requests.get("http://localhost:8080/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                print("✅ Agent Mesh is healthy")
                return True
            else:
                print(f"❌ Agent Mesh is unhealthy: {health_data}")
                return False
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
"""

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
        
        # Make shell scripts executable
        if filename.endswith('.sh'):
            os.chmod(script_path, 0o755)
        elif filename.endswith('.py'):
            os.chmod(script_path, 0o755)

    print("✅ Deployment scripts created")


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

    with open(configs_dir / "production.yaml", "w") as f:
        yaml.dump(production_config, f, default_flow_style=False)

    with open(configs_dir / "staging.yaml", "w") as f:
        yaml.dump(staging_config, f, default_flow_style=False)

    print("✅ Production configurations created")


def main():
    """Main deployment setup function."""
    parser = argparse.ArgumentParser(description="Agent Mesh Deployment Setup")
    parser.add_argument(
        "--target", 
        choices=["all", "docker", "kubernetes", "monitoring", "scripts", "configs"],
        default="all",
        help="What to deploy"
    )

    args = parser.parse_args()

    print("🚀 Agent Mesh Deployment Setup")
    print("=" * 50)

    if args.target in ["all", "docker"]:
        create_docker_configs()

    if args.target in ["all", "kubernetes"]:
        create_kubernetes_configs()

    if args.target in ["all", "monitoring"]:
        create_monitoring_configs()

    if args.target in ["all", "scripts"]:
        create_deployment_scripts()

    if args.target in ["all", "configs"]:
        create_production_configs()

    print("\n🎉 Deployment setup completed!")
    print("\nNext steps:")
    print("1. Build: ./scripts/build.sh")
    print("2. Deploy: ./scripts/deploy.sh [development|production|kubernetes]")
    print("3. Check health: ./scripts/health_check.py")


if __name__ == "__main__":
    main()