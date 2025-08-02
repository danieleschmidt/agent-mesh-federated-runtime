# Monitoring & Observability Guide

Comprehensive monitoring setup for Agent Mesh Federated Runtime with metrics, logging, tracing, and alerting.

## Table of Contents

1. [Overview](#overview)
2. [Metrics Collection](#metrics-collection)
3. [Logging Strategy](#logging-strategy)
4. [Distributed Tracing](#distributed-tracing)
5. [Alerting](#alerting)
6. [Dashboards](#dashboards)
7. [Performance Monitoring](#performance-monitoring)
8. [Troubleshooting](#troubleshooting)

## Overview

Agent Mesh monitoring stack includes:

- **Prometheus** - Metrics collection and alerting
- **Grafana** - Visualization and dashboards
- **OpenTelemetry** - Distributed tracing
- **Structured Logging** - Centralized log aggregation
- **Custom Metrics** - P2P and federated learning specific metrics

## Metrics Collection

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agent-mesh-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Custom Metrics

Agent Mesh exposes the following metrics categories:

#### P2P Network Metrics

```python
# Network connectivity
agent_mesh_p2p_connected_peers
agent_mesh_p2p_connection_duration_seconds
agent_mesh_p2p_messages_total
agent_mesh_p2p_message_bytes_total
agent_mesh_p2p_network_partition_score

# Discovery metrics
agent_mesh_p2p_discovery_peers_found_total
agent_mesh_p2p_discovery_duration_seconds
```

#### Federated Learning Metrics

```python
# Training progress
agent_mesh_federated_current_round
agent_mesh_federated_rounds_completed_total
agent_mesh_federated_round_participants
agent_mesh_federated_round_duration_seconds
agent_mesh_federated_last_round_timestamp

# Model metrics
agent_mesh_federated_model_accuracy
agent_mesh_federated_model_loss
agent_mesh_federated_model_updates_total
agent_mesh_federated_aggregation_duration_seconds
```

#### Consensus Metrics

```python
# Consensus performance
agent_mesh_consensus_proposals_total
agent_mesh_consensus_votes_total
agent_mesh_consensus_failures_total
agent_mesh_consensus_duration_seconds
agent_mesh_consensus_leader_elections_total
```

#### System Metrics

```python
# Process metrics
process_cpu_seconds_total
process_resident_memory_bytes
process_open_fds
process_start_time_seconds

# API metrics
agent_mesh_api_requests_total
agent_mesh_api_request_duration_seconds
agent_mesh_api_active_connections
```

### Metric Implementation

```python
# src/agent_mesh/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Create registry
registry = CollectorRegistry()

# P2P metrics
p2p_connected_peers = Gauge(
    'agent_mesh_p2p_connected_peers',
    'Number of connected P2P peers',
    registry=registry
)

p2p_messages_total = Counter(
    'agent_mesh_p2p_messages_total',
    'Total P2P messages sent/received',
    ['direction', 'message_type'],
    registry=registry
)

# Federated learning metrics
federated_round_participants = Gauge(
    'agent_mesh_federated_round_participants',
    'Number of participants in current federated round',
    registry=registry
)

federated_round_duration = Histogram(
    'agent_mesh_federated_round_duration_seconds',
    'Duration of federated learning rounds',
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=registry
)

# Consensus metrics
consensus_duration = Histogram(
    'agent_mesh_consensus_duration_seconds', 
    'Time to reach consensus',
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    registry=registry
)
```

## Logging Strategy

### Structured Logging

```python
# src/agent_mesh/logging/config.py
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger()
```

### Log Categories

#### P2P Network Logs

```python
logger.info(
    "p2p_peer_connected",
    peer_id=peer_id,
    peer_addr=peer_addr,
    connection_time=duration,
    total_peers=total_count
)

logger.warning(
    "p2p_connection_failed",
    peer_id=peer_id,
    error=str(error),
    retry_count=retry_count
)
```

#### Federated Learning Logs

```python
logger.info(
    "federated_round_started",
    round_id=round_id,
    participants=participant_list,
    expected_duration=estimated_duration
)

logger.info(
    "federated_model_updated",
    round_id=round_id,
    accuracy=model_accuracy,
    loss=model_loss,
    update_size_bytes=update_size
)
```

#### Consensus Logs

```python
logger.info(
    "consensus_proposal_submitted",
    proposal_id=proposal_id,
    proposer=proposer_id,
    proposal_type=proposal_type
)

logger.error(
    "consensus_failed",
    proposal_id=proposal_id,
    error=error_message,
    validators=validator_list
)
```

### Log Aggregation

#### ELK Stack Configuration

```yaml
# docker-compose-logging.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
      
  logstash:
    image: logstash:7.17.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch
      
  kibana:
    image: kibana:7.17.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

#### Logstash Configuration

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "agent-mesh" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      remove_field => [ "host", "agent", "ecs", "log" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "agent-mesh-%{+YYYY.MM.dd}"
  }
}
```

## Distributed Tracing

### OpenTelemetry Setup

```python
# src/agent_mesh/tracing/config.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

# Add span processor
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Trace Instrumentation

```python
# Federated learning trace
@tracer.start_as_current_span("federated_round")
def execute_federated_round(round_id: int):
    span = trace.get_current_span()
    span.set_attribute("round.id", round_id)
    
    with tracer.start_as_current_span("local_training") as training_span:
        # Local training logic
        training_span.set_attribute("training.epochs", epochs)
        training_span.set_attribute("training.batch_size", batch_size)
        
    with tracer.start_as_current_span("model_aggregation") as agg_span:
        # Aggregation logic
        agg_span.set_attribute("aggregation.strategy", "fedavg")
        agg_span.set_attribute("aggregation.participants", len(participants))
```

### Jaeger Deployment

```yaml
# docker-compose-tracing.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
      - "6831:6831/udp"
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
```

## Alerting

### Alert Rules

```yaml
# monitoring/rules/agent_mesh.yml
groups:
  - name: agent_mesh_critical
    rules:
    - alert: AgentMeshNodeDown
      expr: up{job="agent-mesh-nodes"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Agent Mesh node {{ $labels.instance }} is down"
        
    - alert: FederatedLearningStalled
      expr: time() - agent_mesh_federated_last_round_timestamp > 600
      for: 0m
      labels:
        severity: critical
      annotations:
        summary: "Federated learning has been stalled for 10+ minutes"
        
    - alert: P2PNetworkPartition
      expr: agent_mesh_p2p_network_partition_score > 0.5
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "P2P network partition detected (score: {{ $value }})"
```

### Alertmanager Configuration

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@agent-mesh.io'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/'
    
- name: 'critical-alerts'
  email_configs:
  - to: 'ops-team@agent-mesh.io'
    subject: 'CRITICAL: Agent Mesh Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Instance: {{ .Labels.instance }}
      Value: {{ .Value }}
      {{ end }}
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#agent-mesh-alerts'
    title: 'Agent Mesh Critical Alert'
```

## Dashboards

### Grafana Setup

```bash
# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=true \
  --set adminPassword="admin"

# Import dashboards
kubectl create configmap agent-mesh-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n monitoring
```

### Dashboard Categories

1. **Overview Dashboard** - High-level system health
2. **P2P Network Dashboard** - Network topology and performance
3. **Federated Learning Dashboard** - Training progress and metrics
4. **Consensus Dashboard** - Consensus performance and failures
5. **Resource Usage Dashboard** - CPU, memory, storage usage
6. **API Performance Dashboard** - API response times and errors

### Custom Dashboard Panels

```json
{
  "title": "Federated Learning Progress",
  "type": "graph",
  "targets": [
    {
      "expr": "agent_mesh_federated_current_round",
      "legendFormat": "Current Round"
    },
    {
      "expr": "rate(agent_mesh_federated_rounds_completed_total[1h])",
      "legendFormat": "Rounds/Hour"
    }
  ]
}
```

## Performance Monitoring

### Benchmarking

```python
# Run performance benchmarks
python scripts/benchmark.py

# Results saved to benchmark_results/
```

### Load Testing

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class AgentMeshUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def mesh_status(self):
        self.client.get("/mesh/status")
        
    @task(2)
    def federated_status(self):
        self.client.get("/federated/status")
        
    @task(1)
    def consensus_status(self):
        self.client.get("/consensus/status")
```

### Continuous Performance Monitoring

```yaml
# .github/workflows/performance.yml
- name: Run performance tests
  run: |
    pytest tests/performance/ --benchmark-json=benchmark.json
    
- name: Store benchmark results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark.json
```

## Troubleshooting

### Common Monitoring Issues

1. **Missing Metrics**
   ```bash
   # Check metric endpoints
   curl http://localhost:8000/metrics
   
   # Verify Prometheus scraping
   curl http://localhost:9090/api/v1/targets
   ```

2. **High Memory Usage**
   ```bash
   # Check metric cardinality
   curl -s http://localhost:9090/api/v1/label/__name__/values | jq '.data | length'
   
   # Reduce retention or sampling
   prometheus --storage.tsdb.retention.time=15d
   ```

3. **Dashboard Not Loading**
   ```bash
   # Check Grafana logs
   kubectl logs -n monitoring grafana-0
   
   # Verify data source
   curl -u admin:admin http://localhost:3000/api/datasources
   ```

### Debugging Commands

```bash
# Check all monitoring components
kubectl get all -n monitoring

# Test metric collection
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n monitoring

# View Jaeger traces
kubectl port-forward svc/jaeger 16686:16686 -n monitoring
```

For advanced monitoring configurations and custom metrics, see the [Advanced Monitoring Guide](ADVANCED_MONITORING.md).