# Deployment Documentation

This directory contains comprehensive deployment guides and configurations for the Agent Mesh Federated Runtime.

## 🚀 Deployment Options

### Local Development
- [Docker Compose Setup](./docker-compose.md) - Local development with containers
- [Native Installation](./native-setup.md) - Direct installation on development machine
- [Development Environment](./dev-environment.md) - Complete development setup

### Production Deployment
- [Kubernetes Deployment](./kubernetes.md) - Production Kubernetes deployment
- [Cloud Deployment](./cloud-providers.md) - AWS, GCP, Azure deployment guides
- [Edge Deployment](./edge-deployment.md) - Edge device and IoT deployments
- [High Availability](./ha-deployment.md) - Multi-region, fault-tolerant setups

### Container Orchestration
- [Docker Swarm](./docker-swarm.md) - Docker Swarm deployment
- [Nomad Deployment](./nomad.md) - HashiCorp Nomad orchestration
- [Container Security](./container-security.md) - Security best practices

## 📊 Deployment Architectures

### Single Node
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Single Agent Mesh Node                              │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Agent Mesh Runtime                         │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ P2P Network | Consensus | Federated Learning | API │  │  │
│  │  └────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│                     Ideal for: Development, Testing, PoC                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Node Mesh
```
         Bootstrap Node                 Trainer Nodes              Aggregator Nodes
     ┌────────────────────┐       ┌────────────────────┐       ┌────────────────────┐
     │   Network Discovery   │       │  Local Training    │       │ Model Aggregation  │
     │   Peer Coordination   │───────│  Model Updates     │───────│ Global Model      │
     │   Load Balancing     │       │  Data Privacy      │       │ Consensus         │
     └────────────────────┘       └────────────────────┘       └────────────────────┘
             │                          │                          │
             │                          │                          │
         ┌────────────────────────────────────────────────────────────────────────────────────────┐
         │                           P2P Mesh Network                            │
         │              (Encrypted, Byzantine Fault Tolerant)                  │
         └────────────────────────────────────────────────────────────────────────────────────────┘
         │                          │                          │
         │                          │                          │
     ┌────────────────────┐       ┌────────────────────┐       ┌────────────────────┐
     │   Validator Nodes     │       │   Edge Devices     │       │  Monitoring Stack  │
     │   Model Validation   │───────│   IoT Integration   │───────│  Prometheus        │
     │   Security Audits    │       │   Resource Limited  │       │  Grafana           │
     └────────────────────┘       └────────────────────┘       └────────────────────┘

                    Ideal for: Production, Research, Enterprise
```

## 🛠️ Quick Start Commands

### Docker Compose (Recommended for beginners)
```bash
# Start the complete mesh network
docker-compose up -d

# View logs
docker-compose logs -f

# Scale the network
docker-compose up -d --scale mesh-node-1=3

# Stop the network
docker-compose down
```

### Kubernetes (Production)
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=agent-mesh

# View logs
kubectl logs -l app=agent-mesh -f

# Scale deployment
kubectl scale deployment agent-mesh --replicas=10
```

### Native Installation
```bash
# Install dependencies
pip install -e ".[all]"

# Start a single node
agent-mesh --config configs/development.yaml

# Start multiple nodes
agent-mesh --node-id node-001 --port 4001 &
agent-mesh --node-id node-002 --port 4002 --bootstrap /ip4/127.0.0.1/tcp/4001 &
agent-mesh --node-id node-003 --port 4003 --bootstrap /ip4/127.0.0.1/tcp/4001 &
```

## 📋 Configuration Management

### Environment-Specific Configs
- `configs/development.yaml` - Development environment
- `configs/staging.yaml` - Staging environment  
- `configs/production.yaml` - Production environment
- `configs/edge.yaml` - Edge device configuration

### Configuration Hierarchy
1. **Default values** (hardcoded in application)
2. **Configuration files** (YAML/JSON)
3. **Environment variables** (`.env` files)
4. **Command line arguments** (highest priority)

### Secrets Management
- Use environment variables for sensitive data
- Never commit secrets to version control
- Consider using HashiCorp Vault, AWS Secrets Manager, etc.
- Rotate secrets regularly

## 📊 Monitoring and Health Checks

### Health Endpoints
- `http://localhost:8080/health` - Basic health check
- `http://localhost:8080/ready` - Readiness probe
- `http://localhost:9090/metrics` - Prometheus metrics

### Key Metrics to Monitor
- **Network**: Peer count, message latency, bandwidth usage
- **Consensus**: Round time, vote success rate, leader changes
- **Federated Learning**: Training rounds, model accuracy, aggregation time
- **System**: CPU, memory, disk usage, error rates

### Alerting Rules
```yaml
# Example Prometheus alerting rules
groups:
  - name: agent-mesh
    rules:
      - alert: MeshNodeDown
        expr: up{job="agent-mesh"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent Mesh node is down"
      
      - alert: HighConsensusLatency
        expr: consensus_round_duration_seconds > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Consensus rounds taking too long"
```

## 🔒 Security Considerations

### Network Security
- Use TLS 1.3 for all communications
- Implement proper certificate management
- Configure firewalls to restrict unnecessary access
- Use VPNs or private networks when possible

### Container Security
- Run containers as non-root users
- Use minimal base images (Alpine, Distroless)
- Scan images for vulnerabilities regularly
- Implement resource limits and network policies

### Secrets and Keys
- Generate strong cryptographic keys
- Store secrets securely (not in containers)
- Use hardware security modules (HSMs) for production
- Implement key rotation policies

## 📝 Performance Tuning

### Resource Requirements

| Node Type | CPU | Memory | Storage | Network |
|-----------|-----|---------|---------|----------|
| **Bootstrap** | 2 cores | 4 GB | 50 GB | 1 Gbps |
| **Trainer** | 4 cores | 8 GB | 100 GB | 1 Gbps |
| **Aggregator** | 8 cores | 16 GB | 200 GB | 10 Gbps |
| **Validator** | 2 cores | 4 GB | 50 GB | 1 Gbps |
| **Edge** | 1 core | 1 GB | 10 GB | 100 Mbps |

### Optimization Guidelines
- **Network**: Tune buffer sizes, enable compression
- **Consensus**: Adjust timeout values, batch size
- **ML Training**: Use GPU acceleration, optimize batch sizes
- **Storage**: Use SSD storage, implement data compression

## 🖄 Troubleshooting

### Common Issues

#### Node Connection Problems
```bash
# Check network connectivity
telnet <peer-ip> 4001

# Verify port availability
netstat -an | grep 4001

# Check firewall rules
sudo ufw status
```

#### Consensus Failures
```bash
# Check node roles and status
curl http://localhost:8080/status

# View consensus metrics
curl http://localhost:9090/metrics | grep consensus

# Examine logs for Byzantine behavior
docker logs agent-mesh-node-1 | grep byzantine
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check for memory leaks
valgrind --tool=memcheck agent-mesh

# Profile CPU usage
perf record -g agent-mesh
```

### Log Analysis
```bash
# Search for errors
grep -i error /app/logs/agent-mesh.log

# Monitor real-time logs
tail -f /app/logs/agent-mesh.log | grep -E "(ERROR|WARN)"

# Analyze consensus logs
jq '.consensus' /app/logs/agent-mesh.json
```

## 📞 Support and Resources

- 📚 [Documentation](https://docs.agent-mesh.io)
- 💬 [Discord Community](https://discord.gg/agent-mesh)
- 📝 [GitHub Discussions](https://github.com/your-org/agent-mesh/discussions)
- 🐛 [Issue Tracker](https://github.com/your-org/agent-mesh/issues)
- 📧 [Email Support](mailto:support@agent-mesh.io)

### Professional Services
- **Consulting**: Architecture design and optimization
- **Training**: Team training and certification
- **Support**: 24/7 production support contracts
- **Custom Development**: Feature development and integration

---

**Next Steps**: Choose your deployment method and follow the corresponding guide for detailed instructions.
