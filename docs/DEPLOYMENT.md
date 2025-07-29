# Deployment Guide

This guide covers deployment strategies for Agent Mesh Federated Runtime across different environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Edge Deployment](#edge-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker (optional)
- Redis (for clustering)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/agent-mesh-federated-runtime
cd agent-mesh-federated-runtime

# Setup development environment
python scripts/setup.py

# Start development servers
npm run dev
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Core configuration
MESH_NODE_ID=node-dev-001
MESH_ROLE=auto
MESH_LISTEN_ADDR=0.0.0.0:4001

# API configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./dev.db

# Redis (for clustering)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Docker Deployment

### Single Node

```bash
# Build image
docker build -t agent-mesh:latest .

# Run single node
docker run -d \
  --name agent-mesh-node \
  -p 4001:4001 \
  -p 5001:5001 \
  -p 8000:8000 \
  -e MESH_ROLE=auto \
  agent-mesh:latest
```

### Docker Compose Cluster

```bash
# Start cluster
docker-compose up -d

# Scale nodes
docker-compose up -d --scale mesh-node=5

# Check status
docker-compose ps
```

### Configuration

Create `docker-compose.override.yml` for custom configuration:

```yaml
version: '3.8'
services:
  mesh-node:
    environment:
      - MESH_ROLE=trainer
      - LOG_LEVEL=DEBUG
    volumes:
      - ./local-data:/app/data
      - ./local-config.yaml:/app/config.yaml
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3+ (optional)
- Persistent storage support

### Basic Deployment

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment
kubectl get pods -n agent-mesh
```

### Helm Deployment

```bash
# Add helm repository
helm repo add agent-mesh https://charts.agent-mesh.io
helm repo update

# Install with custom values
helm install agent-mesh agent-mesh/agent-mesh \
  --namespace agent-mesh \
  --create-namespace \
  --values values.yaml
```

### Custom Values (values.yaml)

```yaml
# Deployment configuration
replicaCount: 3
image:
  repository: ghcr.io/your-org/agent-mesh-federated-runtime
  tag: "latest"

# Resource limits
resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

# Storage
persistence:
  enabled: true
  size: 10Gi
  storageClass: "fast-ssd"

# Monitoring
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

# Service configuration
service:
  type: ClusterIP
  ports:
    p2p: 4001
    grpc: 5001
    http: 8000
    metrics: 9090

# Ingress
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: agent-mesh.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: agent-mesh-tls
      hosts:
        - agent-mesh.example.com
```

### Scaling

```bash
# Scale horizontally
kubectl scale statefulset agent-mesh --replicas=10 -n agent-mesh

# Vertical scaling (update deployment)
kubectl patch statefulset agent-mesh -n agent-mesh -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"mesh-node","resources":{"limits":{"cpu":"2000m","memory":"4Gi"}}}]}}}}'
```

## Edge Deployment

### Lightweight Configuration

For resource-constrained edge devices:

```yaml
# edge-config.yaml
mesh:
  network:
    max_connections: 5
    heartbeat_interval: 30s
  
  resource_limits:
    max_memory_mb: 256
    max_cpu_percent: 30
    
  features:
    consensus: false  # Disable for edge nodes
    training: true    # Enable training only
    
federated:
  batch_size: 8     # Smaller batches
  local_epochs: 2   # Fewer local epochs
  
monitoring:
  prometheus:
    enabled: false  # Disable metrics collection
```

### ARM64 Support

```dockerfile
# Multi-architecture build
FROM --platform=$BUILDPLATFORM python:3.11-slim as builder
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Build dependencies
RUN pip install build

# Copy source
COPY . /app
WORKDIR /app

# Build wheel
RUN python -m build

# Runtime image
FROM python:3.11-slim
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl
```

### Edge Deployment Script

```bash
#!/bin/bash
# deploy-edge.sh

# Set edge-specific environment
export MESH_ROLE=edge-trainer
export MESH_RESOURCE_LIMIT=low
export FEDERATED_BATCH_SIZE=8

# Start with reduced resource usage
docker run -d \
  --name agent-mesh-edge \
  --restart unless-stopped \
  --memory=256m \
  --cpus="0.5" \
  -e MESH_ROLE=$MESH_ROLE \
  -e MESH_RESOURCE_LIMIT=$MESH_RESOURCE_LIMIT \
  -v ./edge-data:/app/data \
  agent-mesh:edge
```

## Cloud Deployment

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name agent-mesh-cluster \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Deploy
kubectl apply -f k8s/
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create agent-mesh-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Deploy
kubectl apply -f k8s/
```

### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name agent-mesh-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Deploy
kubectl apply -f k8s/
```

## Monitoring Setup

### Prometheus & Grafana

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml

# Import Grafana dashboards
kubectl apply -f monitoring/grafana/dashboards/
```

### Custom Metrics

Agent Mesh exposes the following metrics:

- `agent_mesh_p2p_connected_peers` - Number of P2P connections
- `agent_mesh_federated_round_participants` - Federated learning participants
- `agent_mesh_consensus_duration_seconds` - Consensus latency
- `agent_mesh_api_requests_total` - API request count

### Alerting

Configure alerts in `monitoring/rules/agent_mesh.yml`:

```yaml
- alert: AgentMeshNodeDown
  expr: up{job="agent-mesh-nodes"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Agent Mesh node is down"
```

## Troubleshooting

### Common Issues

1. **P2P Connection Problems**
   ```bash
   # Check network connectivity
   kubectl exec -it agent-mesh-0 -- nc -zv agent-mesh-1.agent-mesh 4001
   
   # Check logs
   kubectl logs agent-mesh-0 -n agent-mesh
   ```

2. **Consensus Failures**
   ```bash
   # Check validator status
   kubectl exec -it agent-mesh-0 -- curl localhost:8000/consensus/status
   
   # Restart consensus
   kubectl exec -it agent-mesh-0 -- curl -X POST localhost:8000/consensus/restart
   ```

3. **Performance Issues**
   ```bash
   # Check resource usage
   kubectl top pods -n agent-mesh
   
   # Scale up
   kubectl scale statefulset agent-mesh --replicas=5 -n agent-mesh
   ```

### Debugging Commands

```bash
# Get cluster status
kubectl get all -n agent-mesh

# Check events
kubectl get events -n agent-mesh --sort-by=.metadata.creationTimestamp

# Describe problematic pod
kubectl describe pod agent-mesh-0 -n agent-mesh

# Check logs
kubectl logs agent-mesh-0 -n agent-mesh --previous

# Access pod shell
kubectl exec -it agent-mesh-0 -n agent-mesh -- /bin/bash

# Port forward for debugging
kubectl port-forward agent-mesh-0 8000:8000 -n agent-mesh
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# P2P status
curl http://localhost:8000/p2p/status

# Federated learning status
curl http://localhost:8000/federated/status

# Consensus status
curl http://localhost:8000/consensus/status
```

## Security Considerations

### Network Security

- Use network policies to restrict traffic
- Enable TLS for all communications
- Implement proper authentication/authorization

### Secrets Management

```bash
# Create TLS secret
kubectl create secret tls agent-mesh-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n agent-mesh

# Create API key secret
kubectl create secret generic agent-mesh-api-key \
  --from-literal=api-key="your-secure-api-key" \
  -n agent-mesh
```

### RBAC

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: agent-mesh
  name: agent-mesh-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

For more detailed deployment scenarios and advanced configurations, see the [Advanced Deployment Guide](ADVANCED_DEPLOYMENT.md).