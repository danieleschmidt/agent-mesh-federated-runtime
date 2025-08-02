# Getting Started Guide

Welcome to the Agent Mesh Federated Runtime! This guide will help you set up and run your first decentralized federated learning experiment.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.9 or higher with pip
- **Memory**: Minimum 2GB RAM, 4GB recommended
- **Storage**: 5GB free disk space
- **Network**: Internet connection for initial setup and peer discovery

### Optional Components

- **Docker**: For containerized deployments
- **Node.js 18+**: For web dashboard development
- **CUDA**: For GPU-accelerated training (optional)

## Installation

### Quick Install (Recommended)

```bash
# Install from PyPI
pip install agent-mesh-federated-runtime

# Verify installation
agent-mesh --version
```

### Development Install

```bash
# Clone repository
git clone https://github.com/your-org/agent-mesh-federated-runtime.git
cd agent-mesh-federated-runtime

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
npm run test:unit
```

### Docker Install

```bash
# Pull official image
docker pull your-org/agent-mesh:latest

# Run basic node
docker run -p 4001:4001 -p 8000:8000 your-org/agent-mesh:latest
```

## Your First Mesh Network

### Step 1: Create a Bootstrap Node

Create a file `bootstrap_node.py`:

```python
import asyncio
from agent_mesh import MeshNode, FederatedLearner
from agent_mesh.examples import SimpleModel, load_sample_data

async def run_bootstrap_node():
    # Create the first node in the mesh
    node = MeshNode(
        node_id="bootstrap-001",
        listen_addr="/ip4/0.0.0.0/tcp/4001",
        role="auto"  # Automatically determine role
    )
    
    # Setup federated learning task
    learner = FederatedLearner(
        model_fn=SimpleModel,
        dataset_fn=lambda: load_sample_data("mnist", partition=0),
        aggregation="fedavg"
    )
    
    # Attach learning task to node
    node.attach_task(learner)
    
    # Start the node
    print("Starting bootstrap node...")
    await node.start()
    
    # Keep running
    try:
        while True:
            metrics = await node.get_metrics()
            print(f"Active peers: {metrics.peer_count}, Training round: {metrics.round}")
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down...")
        await node.stop()

if __name__ == "__main__":
    asyncio.run(run_bootstrap_node())
```

Run the bootstrap node:

```bash
python bootstrap_node.py
```

### Step 2: Add Peer Nodes

Create `peer_node.py`:

```python
import asyncio
from agent_mesh import MeshNode, FederatedLearner
from agent_mesh.examples import SimpleModel, load_sample_data

async def run_peer_node(node_id: str, data_partition: int):
    # Create peer node
    node = MeshNode(
        node_id=node_id,
        bootstrap_peers=[
            "/ip4/127.0.0.1/tcp/4001/p2p/bootstrap-001"
        ]
    )
    
    # Setup federated learning with different data partition
    learner = FederatedLearner(
        model_fn=SimpleModel,
        dataset_fn=lambda: load_sample_data("mnist", partition=data_partition),
        aggregation="fedavg"
    )
    
    node.attach_task(learner)
    
    print(f"Starting peer node {node_id}...")
    await node.start()
    
    # Monitor training progress
    try:
        while True:
            metrics = await node.get_metrics()
            print(f"Node {node_id} - Round: {metrics.round}, Loss: {metrics.loss:.4f}")
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        await node.stop()

if __name__ == "__main__":
    import sys
    node_id = sys.argv[1] if len(sys.argv) > 1 else "peer-001"
    partition = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    asyncio.run(run_peer_node(node_id, partition))
```

Run multiple peer nodes in separate terminals:

```bash
# Terminal 2
python peer_node.py peer-001 1

# Terminal 3
python peer_node.py peer-002 2

# Terminal 4
python peer_node.py peer-003 3
```

### Step 3: Monitor the Network

Access the web dashboard at `http://localhost:8000` to see:

- Network topology and peer connections
- Training progress and model metrics
- Consensus status and Byzantine fault detection
- Resource utilization and performance stats

## Configuration Options

### Node Configuration

```python
node = MeshNode(
    node_id="my-node-001",
    listen_addr="/ip4/0.0.0.0/tcp/4001",
    
    # Network configuration
    bootstrap_peers=[
        "/ip4/192.168.1.100/tcp/4001/p2p/bootstrap-id"
    ],
    discovery_method="mdns",  # or "dht", "gossip"
    
    # Security settings
    enable_encryption=True,
    identity_file="~/.agent-mesh/identity.key",
    
    # Performance tuning
    max_peers=50,
    connection_timeout=30,
    heartbeat_interval=10,
    
    # Role preferences
    role="auto",  # or "trainer", "aggregator", "validator"
    capabilities=["training", "aggregation"],
    resources={"cpu": 4, "memory_gb": 8, "gpu": True}
)
```

### Federated Learning Configuration

```python
learner = FederatedLearner(
    model_fn=create_model,
    dataset_fn=load_data,
    
    # Training parameters
    local_epochs=5,
    batch_size=32,
    learning_rate=0.01,
    
    # Aggregation strategy
    aggregation="fedavg",  # or "scaffold", "fedprox"
    
    # Privacy settings
    differential_privacy={
        "epsilon": 1.0,
        "delta": 1e-5,
        "mechanism": "gaussian"
    },
    
    # Security settings
    secure_aggregation={
        "protocol": "shamir",
        "threshold": 0.5
    },
    
    # Convergence criteria
    max_rounds=100,
    target_accuracy=0.95,
    patience=10
)
```

## Common Use Cases

### Cross-Silo Federated Learning

For organizations collaborating while keeping data private:

```python
# Each organization runs this configuration
node = MeshNode(
    node_id=f"org-{organization_id}",
    role="trainer",
    privacy_level="high",
    data_residency="local"
)

learner = FederatedLearner(
    aggregation="scaffold",  # Better for heterogeneous data
    differential_privacy={"epsilon": 0.1},  # Strong privacy
    secure_aggregation={"protocol": "paillier"}
)
```

### Edge Device Coordination

For resource-constrained IoT deployments:

```python
node = MeshNode(
    node_id=f"edge-{device_id}",
    resource_constraints={
        "max_memory_mb": 512,
        "max_cpu_percent": 50
    },
    connectivity="intermittent",
    battery_aware=True
)

learner = FederatedLearner(
    model_fn=LightweightModel,  # Mobile-optimized model
    local_epochs=1,  # Reduce computation
    compression=True  # Compress updates
)
```

### Multi-Agent Task Coordination

For collaborative AI agents:

```python
from agent_mesh import AgentMesh, CollaborativeTask

mesh = AgentMesh(
    discovery_method="gossip",
    consensus="raft"
)

task = CollaborativeTask(
    name="distributed_planning",
    min_agents=3,
    coordination_protocol="contract_net"
)

async with mesh.join_network() as network:
    role = await network.negotiate_role(
        capabilities=["planning", "execution"],
        resources={"cpu": 4, "memory_gb": 8}
    )
    
    await network.execute_task(task)
```

## Troubleshooting

### Connection Issues

**Problem**: Nodes can't discover each other

**Solution**: Check network connectivity and firewall settings:

```bash
# Test connectivity
telnet <peer-ip> 4001

# Check firewall (Ubuntu/Debian)
sudo ufw allow 4001

# Enable mDNS discovery
sudo systemctl enable avahi-daemon
```

**Problem**: Bootstrap peer not reachable

**Solution**: Verify bootstrap peer address and format:

```python
# Correct format
bootstrap_peers=[
    "/ip4/192.168.1.100/tcp/4001/p2p/QmBootstrapNodeID"
]

# Get node ID from bootstrap node logs
# Look for: "Node ID: QmBootstrapNodeID"
```

### Performance Issues

**Problem**: Slow consensus or high latency

**Solution**: Tune consensus parameters:

```python
node = MeshNode(
    consensus_config={
        "timeout_ms": 1000,  # Reduce for faster networks
        "batch_size": 100,   # Increase for better throughput
        "max_pending": 1000  # Queue size
    }
)
```

**Problem**: Memory usage too high

**Solution**: Enable memory optimization:

```python
node = MeshNode(
    memory_optimization=True,
    cache_size="100MB",
    gc_interval=60  # Force garbage collection
)
```

### Security Issues

**Problem**: Certificate validation errors

**Solution**: Regenerate node identity:

```bash
# Remove old identity
rm ~/.agent-mesh/identity.key

# Node will generate new identity on startup
python your_node.py
```

**Problem**: Suspected Byzantine behavior

**Solution**: Enable enhanced monitoring:

```python
node = MeshNode(
    security_level="paranoid",
    byzantine_detection=True,
    audit_logging=True
)
```

## Next Steps

### Learning Resources

- [Advanced Configuration Guide](../developer/configuration.md)
- [Security Best Practices](../security/best-practices.md)
- [Deployment Strategies](../deployment/strategies.md)
- [API Reference Documentation](../../api/)

### Example Projects

- [Healthcare Federated Learning](../../examples/healthcare/)
- [IoT Edge Computing](../../examples/iot-edge/)
- [Financial Services Collaboration](../../examples/fintech/)
- [Multi-Agent Robotics](../../examples/robotics/)

### Community

- **Discord**: [Join our community](https://discord.gg/agent-mesh)
- **GitHub Discussions**: Ask questions and share experiences
- **Stack Overflow**: Tag questions with `agent-mesh`
- **Monthly Meetups**: Virtual meetups for users and contributors

---

**Need Help?** Check our [FAQ](../user/faq.md) or reach out on [Discord](https://discord.gg/agent-mesh)!