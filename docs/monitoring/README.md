# Monitoring and Observability

Comprehensive monitoring and observability setup for the Agent Mesh Federated Runtime.

## 📊 Overview

This directory contains configurations and documentation for monitoring the Agent Mesh system across all deployment environments.

## 🛠️ Monitoring Stack

### Core Components
- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization and dashboards
- **AlertManager** - Alert routing and management
- **Jaeger** - Distributed tracing
- **Loki** - Log aggregation
- **Node Exporter** - System metrics

### Architecture
```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            Agent Mesh Nodes                             │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  │
│  │    Bootstrap Node    │  │   Trainer Nodes    │  │ Aggregator Nodes │  │
│  │                    │  │                  │  │                  │  │
│  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │
│  │ │ Metrics Endpoint │ │  │ │ Metrics Endpoint │ │  │ │ Metrics Endpoint │ │  │
│  │ │ :9090/metrics    │ │  │ │ :9091/metrics    │ │  │ │ :9092/metrics    │ │  │
│  │ └──────────────────┘ │  │ └──────────────────┘ │  │ └──────────────────┘ │  │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                     ┌──────────────────────────────────────────────────────────────────────────────────────┐
                     │                      Monitoring Stack                       │
                     │                                                             │
                     │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
                     │  │                          Collection Layer                          │  │
                     │  │                                                                 │  │
                     │  │ ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐ │  │
                     │  │ │    Prometheus      │  │      Jaeger       │  │       Loki        │ │  │
                     │  │ │  (Metrics Store)  │  │  (Trace Store)   │  │   (Log Store)    │ │  │
                     │  │ └───────────────────┘  └───────────────────┘  └───────────────────┘ │  │
                     │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
                     │                                                             │
                     │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
                     │  │                       Visualization Layer                       │  │
                     │  │                                                                 │  │
                     │  │     ┌───────────────────────────────────────────────────────────────────────┐     │  │
                     │  │     │                       Grafana                        │     │  │
                     │  │     │              (Dashboards & Alerts)               │     │  │
                     │  │     └───────────────────────────────────────────────────────────────────────┘     │  │
                     │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
                     └─────────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Key Metrics

### System Metrics
- **CPU Usage**: `system_cpu_usage_percent`
- **Memory Usage**: `system_memory_usage_bytes`
- **Disk Usage**: `system_disk_usage_bytes`
- **Network I/O**: `system_network_bytes_total`

### P2P Network Metrics
- **Peer Count**: `mesh_peer_count`
- **Message Rate**: `mesh_messages_per_second`
- **Connection Duration**: `mesh_connection_duration_seconds`
- **Bandwidth Usage**: `mesh_bandwidth_bytes_per_second`

### Consensus Metrics
- **Round Duration**: `consensus_round_duration_seconds`
- **Vote Success Rate**: `consensus_vote_success_rate`
- **Leader Elections**: `consensus_leader_elections_total`
- **Byzantine Detection**: `consensus_byzantine_nodes_detected`

### Federated Learning Metrics
- **Training Rounds**: `fl_training_rounds_total`
- **Model Accuracy**: `fl_model_accuracy`
- **Aggregation Time**: `fl_aggregation_duration_seconds`
- **Privacy Budget**: `fl_privacy_budget_consumed`

### Security Metrics
- **Authentication Failures**: `security_auth_failures_total`
- **Encryption Overhead**: `security_encryption_overhead_ms`
- **Certificate Expiry**: `security_cert_expiry_days`
- **Attack Detection**: `security_attacks_detected_total`

## 🚨 Alerting Rules

### Critical Alerts
```yaml
# High-priority alerts requiring immediate attention
groups:
  - name: critical
    rules:
      - alert: MeshNodeDown
        expr: up{job="agent-mesh"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent Mesh node {{ $labels.instance }} is down"
          description: "Node has been down for more than 5 minutes"
      
      - alert: ConsensusFailure
        expr: rate(consensus_failed_rounds_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Consensus failure rate too high"
          description: "More than 10% of consensus rounds are failing"
      
      - alert: HighMemoryUsage
        expr: (system_memory_usage_bytes / system_memory_total_bytes) > 0.9
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is above 90% for 10 minutes"
```

### Warning Alerts
```yaml
groups:
  - name: warnings
    rules:
      - alert: HighConsensusLatency
        expr: consensus_round_duration_seconds > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High consensus latency"
          description: "Consensus rounds taking longer than 10 seconds"
      
      - alert: LowPeerCount
        expr: mesh_peer_count < 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low peer count in mesh network"
          description: "Less than 3 peers connected for 5 minutes"
      
      - alert: ModelAccuracyDrop
        expr: fl_model_accuracy < 0.8
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Model accuracy below threshold"
          description: "Model accuracy dropped below 80%"
```

## 📊 Dashboard Templates

### System Overview Dashboard
- **Node Health**: Status of all mesh nodes
- **Resource Usage**: CPU, memory, disk, network
- **Error Rates**: Application and system errors
- **Performance**: Latency and throughput metrics

### Network Dashboard
- **Topology Visualization**: Real-time network graph
- **Peer Connections**: Connection status and quality
- **Message Flow**: Inter-node communication patterns
- **Bandwidth Utilization**: Network usage statistics

### Consensus Dashboard
- **Round Performance**: Consensus timing and success rates
- **Leader Election**: Leadership changes and stability
- **Byzantine Detection**: Security event monitoring
- **Vote Analysis**: Voting patterns and participation

### Federated Learning Dashboard
- **Training Progress**: Round-by-round training metrics
- **Model Performance**: Accuracy, loss, and convergence
- **Data Distribution**: Dataset statistics across nodes
- **Privacy Metrics**: Differential privacy budget tracking

### Security Dashboard
- **Authentication Events**: Login attempts and failures
- **Certificate Status**: Certificate validity and expiration
- **Threat Detection**: Security alerts and incidents
- **Compliance Metrics**: Security policy adherence

## 🛠️ Setup Instructions

### 1. Start Monitoring Stack
```bash
# Using Docker Compose
docker-compose up -d prometheus grafana

# Using Kubernetes
kubectl apply -f monitoring/k8s/

# Verify services
curl http://localhost:9090/targets  # Prometheus
curl http://localhost:3000          # Grafana
```

### 2. Configure Data Sources
```bash
# Grafana datasources are auto-configured via:
# monitoring/grafana/datasources/prometheus.yml
# monitoring/grafana/datasources/jaeger.yml
# monitoring/grafana/datasources/loki.yml
```

### 3. Import Dashboards
```bash
# Dashboards are auto-imported from:
# monitoring/grafana/dashboards/

# Or import manually via Grafana UI
# Dashboard ID references in dashboard-config.yml
```

### 4. Set Up Alerting
```bash
# Configure alert destinations
edit monitoring/alertmanager/config.yml

# Restart AlertManager
docker-compose restart alertmanager
```

## 📝 Operational Procedures

### Daily Health Checks
1. **Node Status**: Verify all nodes are healthy
2. **Alert Review**: Check for any active alerts
3. **Performance**: Review key performance metrics
4. **Capacity**: Monitor resource utilization trends

### Weekly Reports
1. **System Performance**: Generate weekly performance report
2. **Security Events**: Review security incidents
3. **Capacity Planning**: Analyze growth trends
4. **Alert Tuning**: Adjust alert thresholds based on patterns

### Monthly Reviews
1. **Dashboard Updates**: Review and update dashboards
2. **Metric Retention**: Adjust data retention policies
3. **Alerting Rules**: Update alerting rules and runbooks
4. **Performance Optimization**: Identify optimization opportunities

## 🔍 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
curl http://localhost:9090/api/v1/query?query=system_memory_usage_bytes

# Identify memory-heavy processes
top -o %MEM

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full agent-mesh
```

#### Slow Consensus
```bash
# Check consensus metrics
curl http://localhost:9090/api/v1/query?query=consensus_round_duration_seconds

# Analyze network latency
ping -c 10 <peer-node-ip>

# Check for Byzantine nodes
grep "byzantine" /app/logs/consensus.log
```

#### Dashboard Not Loading
```bash
# Check Grafana status
docker logs agent-mesh-grafana

# Verify datasource connectivity
curl http://localhost:3000/api/datasources

# Test Prometheus connectivity
curl http://localhost:9090/api/v1/label/__name__/values
```

### Log Analysis
```bash
# Search for specific patterns
grep -E "(ERROR|WARN|FATAL)" /app/logs/agent-mesh.log

# Analyze consensus logs
jq '.consensus' /app/logs/agent-mesh.json | head -n 100

# Monitor real-time logs
tail -f /app/logs/agent-mesh.log | grep -E "(consensus|network)"
```

## 📝 Configuration Files

### Prometheus Configuration
- `monitoring/prometheus.yml` - Main Prometheus config
- `monitoring/rules/` - Alerting rules
- `monitoring/targets/` - Service discovery configs

### Grafana Configuration
- `monitoring/grafana/datasources/` - Data source definitions
- `monitoring/grafana/dashboards/` - Dashboard JSON files
- `monitoring/grafana/plugins/` - Custom plugins

### AlertManager Configuration
- `monitoring/alertmanager/config.yml` - Alert routing rules
- `monitoring/alertmanager/templates/` - Notification templates

## 📈 Performance Optimization

### Metric Collection Optimization
- **Scrape Intervals**: Adjust based on metric importance
- **Retention Policies**: Configure appropriate data retention
- **Aggregation Rules**: Pre-compute common queries
- **Sharding**: Distribute load across multiple Prometheus instances

### Dashboard Performance
- **Query Optimization**: Use efficient PromQL queries
- **Time Range Limits**: Set reasonable default time ranges
- **Auto-refresh**: Configure appropriate refresh intervals
- **Panel Optimization**: Limit the number of panels per dashboard

### Storage Optimization
- **Compression**: Enable time series compression
- **Downsampling**: Implement retention policies with downsampling
- **Federation**: Use Prometheus federation for long-term storage
- **External Storage**: Consider Thanos or Cortex for scale

## 🔗 External Integrations

### Notification Channels
- **Slack**: Real-time alerts to development teams
- **PagerDuty**: Critical incident management
- **Email**: Weekly summary reports
- **Webhook**: Custom notification endpoints

### ITSM Integration
- **ServiceNow**: Automatic ticket creation
- **Jira**: Issue tracking and project management
- **Confluence**: Documentation and runbook management

### Security Integration
- **SIEM Systems**: Security event correlation
- **Compliance Tools**: Audit trail and compliance reporting
- **Threat Intelligence**: Integration with threat detection platforms

---

**Access URLs**:
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093
