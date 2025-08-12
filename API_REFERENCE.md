# Agent Mesh API Reference

## Overview

This document provides comprehensive API documentation for the Agent Mesh system. The APIs are organized by component and include both synchronous and asynchronous interfaces.

## Core APIs

### MeshNode API

The central coordination component for mesh network participation.

#### Class: `MeshNode`

```python
from agent_mesh import MeshNode

node = MeshNode(
    node_id=UUID("..."),
    host="127.0.0.1",
    port=8080,
    config_path="config.json"
)
```

#### Methods

##### `async initialize() -> None`
Initialize the mesh node with all required components.

**Example:**
```python
await node.initialize()
```

##### `async start() -> None`
Start the node and begin participating in the mesh network.

**Example:**
```python
await node.start()
```

##### `async stop() -> None`
Gracefully shutdown the node and cleanup resources.

**Example:**
```python
await node.stop()
```

##### `async connect_to_peer(host: str, port: int) -> bool`
Connect to another mesh node.

**Parameters:**
- `host`: Target node hostname or IP address
- `port`: Target node port number

**Returns:** Boolean indicating success

**Example:**
```python
success = await node.connect_to_peer("192.168.1.100", 8080)
```

##### `get_connected_peers() -> List[PeerInfo]`
Get list of currently connected peers.

**Returns:** List of PeerInfo objects

**Example:**
```python
peers = node.get_connected_peers()
for peer in peers:
    print(f"Peer: {peer.peer_id} at {peer.address}")
```

##### `get_node_status() -> Dict[str, Any]`
Get current node status and metrics.

**Returns:** Dictionary containing status information

**Example:**
```python
status = node.get_node_status()
print(f"Node {status['node_id']} is {status['state']}")
```

### P2P Network API

Low-level peer-to-peer networking functionality.

#### Class: `P2PNetwork`

```python
from agent_mesh.core.network import P2PNetwork

network = P2PNetwork(
    node_id=uuid4(),
    listen_addr="127.0.0.1:4001"
)
```

#### Methods

##### `async start() -> None`
Start the P2P network listener.

##### `async stop() -> None`
Stop the network and close all connections.

##### `async secure_connect_to_peer(address: str) -> PeerConnection`
Establish secure connection to a peer.

**Parameters:**
- `address`: Peer address in format "host:port"

**Returns:** PeerConnection object

##### `async send_message(peer_id: UUID, message: dict) -> bool`
Send encrypted message to a peer.

**Parameters:**
- `peer_id`: Target peer identifier
- `message`: Message payload

**Returns:** Boolean indicating success

##### `register_message_handler(message_type: str, handler: Callable) -> None`
Register handler for specific message types.

**Parameters:**
- `message_type`: Message type to handle
- `handler`: Async function to handle messages

**Example:**
```python
async def handle_training_request(peer_id: UUID, message: dict):
    print(f"Training request from {peer_id}: {message}")

network.register_message_handler("training_request", handle_training_request)
```

### Security API

Cryptographic operations and security management.

#### Class: `SecurityManager`

```python
from agent_mesh.core.security import SecurityManager

security = SecurityManager(
    node_id=uuid4(),
    key_algorithm=SignatureAlgorithm.ED25519,
    encryption_algorithm=EncryptionAlgorithm.NACL_SECRETBOX
)
```

#### Methods

##### `async initialize() -> None`
Initialize security manager and generate keys.

##### `async get_node_identity() -> NodeIdentity`
Get current node identity with public key.

**Returns:** NodeIdentity object

##### `async encrypt_message(data: bytes, recipient_id: UUID = None) -> EncryptedMessage`
Encrypt data for transmission.

**Parameters:**
- `data`: Raw data to encrypt
- `recipient_id`: Target recipient (None for broadcast)

**Returns:** EncryptedMessage object

##### `async decrypt_message(encrypted_msg: EncryptedMessage) -> bytes`
Decrypt received message.

**Parameters:**
- `encrypted_msg`: Encrypted message to decrypt

**Returns:** Decrypted data

##### `async sign_data(data: bytes) -> bytes`
Sign data with node's private key.

**Parameters:**
- `data`: Data to sign

**Returns:** Digital signature

##### `async verify_signature(data: bytes, signature: bytes, signer_id: UUID) -> bool`
Verify signature from another node.

**Parameters:**
- `data`: Original data
- `signature`: Digital signature
- `signer_id`: ID of signing node

**Returns:** Boolean indicating validity

### Consensus API

Byzantine fault-tolerant consensus implementation.

#### Class: `RaftConsensus`

```python
from agent_mesh.core.consensus import RaftConsensus

consensus = RaftConsensus(
    node_id=uuid4(),
    peers=[peer1_id, peer2_id, peer3_id]
)
```

#### Methods

##### `async initialize() -> None`
Initialize consensus engine.

##### `async start() -> None`
Start participating in consensus.

##### `async submit_proposal(data: dict) -> Optional[dict]`
Submit proposal to the cluster.

**Parameters:**
- `data`: Proposal data

**Returns:** Consensus result or None if failed

**Example:**
```python
proposal = {"action": "update_model", "params": {...}}
result = await consensus.submit_proposal(proposal)
```

##### `is_leader() -> bool`
Check if this node is the current leader.

**Returns:** Boolean indicating leadership status

##### `get_consensus_status() -> Dict[str, Any]`
Get current consensus state and metrics.

**Returns:** Status dictionary

## Federated Learning APIs

### FederatedLearner API

Coordinates distributed machine learning training.

#### Class: `FederatedLearner`

```python
from agent_mesh.federated.learner import FederatedLearner

learner = FederatedLearner(
    node_id=uuid4(),
    model=my_model,
    training_data=train_dataset
)
```

#### Methods

##### `async initialize() -> None`
Initialize federated learning components.

##### `async start_training_round(participants: List[UUID], round_number: int) -> dict`
Initiate a federated learning round.

**Parameters:**
- `participants`: List of participating node IDs
- `round_number`: Training round number

**Returns:** Training round results

##### `async local_train(epochs: int, batch_size: int = 32) -> TrainingResult`
Perform local model training.

**Parameters:**
- `epochs`: Number of training epochs
- `batch_size`: Training batch size

**Returns:** TrainingResult with metrics and model updates

##### `async get_model_parameters() -> dict`
Get current model parameters.

**Returns:** Model parameters dictionary

##### `async update_model_parameters(params: dict) -> None`
Update model with new parameters.

**Parameters:**
- `params`: New model parameters

### SecureAggregator API

Privacy-preserving model aggregation.

#### Class: `SecureAggregator`

```python
from agent_mesh.federated.aggregator import SecureAggregator

aggregator = SecureAggregator(
    node_id=uuid4(),
    algorithm="fedavg",
    security_level="high"
)
```

#### Methods

##### `async aggregate_updates(updates: List[ModelUpdate]) -> dict`
Securely aggregate model updates.

**Parameters:**
- `updates`: List of ModelUpdate objects

**Returns:** Aggregated model weights

##### `async add_update(update: ModelUpdate) -> None`
Add model update to current round.

**Parameters:**
- `update`: Model update from participant

##### `get_aggregation_statistics() -> Dict[str, Any]`
Get aggregation performance metrics.

**Returns:** Statistics dictionary

## Performance and Scaling APIs

### Cache API

Advanced caching with multiple strategies.

#### Class: `CacheManager`

```python
from agent_mesh.core.cache import CacheManager

cache_manager = CacheManager(
    node_id=uuid4(),
    network_manager=network
)
```

#### Methods

##### `async start() -> None`
Start cache manager and all cache levels.

##### `async get(key: str, cache_type: str = "general") -> Optional[Any]`
Retrieve value from cache.

**Parameters:**
- `key`: Cache key
- `cache_type`: Cache type ("general", "model", "data")

**Returns:** Cached value or None

##### `async put(key: str, value: Any, cache_type: str = "general", ttl: float = None) -> bool`
Store value in cache.

**Parameters:**
- `key`: Cache key
- `value`: Value to cache
- `cache_type`: Cache type
- `ttl`: Time-to-live in seconds

**Returns:** Success boolean

##### `get_statistics() -> Dict[str, Any]`
Get comprehensive cache statistics.

**Returns:** Cache statistics

### AutoScaler API

Intelligent resource management and scaling.

#### Class: `ResourceManager`

```python
from agent_mesh.core.autoscaler import ResourceManager

resource_manager = ResourceManager(
    node_id=uuid4(),
    scaling_policy=ScalingPolicy(),
    load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
)
```

#### Methods

##### `async start() -> None`
Start resource management.

##### `async route_request(request: Any, timeout: float = 30.0) -> Any`
Route request to best available node.

**Parameters:**
- `request`: Request to route
- `timeout`: Request timeout

**Returns:** Response from selected node

##### `update_node_metrics(metrics: NodeMetrics) -> None`
Update metrics for a node.

**Parameters:**
- `metrics`: Current node metrics

##### `get_resource_statistics() -> Dict[str, Any]`
Get resource management statistics.

**Returns:** Statistics dictionary

### Monitoring API

Comprehensive system monitoring and observability.

#### Class: `ComprehensiveMonitor`

```python
from agent_mesh.core.monitoring import ComprehensiveMonitor

monitor = ComprehensiveMonitor(node_id=uuid4())
```

#### Methods

##### `async start() -> None`
Start monitoring systems.

##### `get_monitoring_dashboard() -> Dict[str, Any]`
Get comprehensive monitoring data.

**Returns:** Dashboard data dictionary

##### `record_metric(name: str, value: float, labels: Dict[str, str] = None) -> None`
Record custom metric.

**Parameters:**
- `name`: Metric name
- `value`: Metric value
- `labels`: Optional metric labels

## Configuration APIs

### Configuration Management

System configuration and environment management.

#### Class: `ConfigManager`

```python
from agent_mesh.core.config import ConfigManager

config = ConfigManager.load_config("production.json")
```

#### Methods

##### `@staticmethod load_config(config_path: str) -> Dict[str, Any]`
Load configuration from file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:** Configuration dictionary

##### `@staticmethod validate_config(config: dict) -> bool`
Validate configuration structure.

**Parameters:**
- `config`: Configuration to validate

**Returns:** Validation result

## Error Handling

All API methods may raise the following exceptions:

### `AgentMeshException`
Base exception for all Agent Mesh errors.

### `NetworkException`
Network-related errors (connection failures, timeouts).

### `SecurityException`
Security and cryptographic errors.

### `ConsensusException`
Consensus algorithm errors.

### `FederatedLearningException`
Federated learning specific errors.

## Event Handling

### Event Types

The system emits events for monitoring and integration:

- `node_started`: Node initialization complete
- `peer_connected`: New peer connection established
- `peer_disconnected`: Peer connection lost
- `consensus_decided`: Consensus decision reached
- `training_round_started`: FL training round initiated
- `training_round_completed`: FL training round finished
- `model_updated`: Global model parameters updated

### Event Handlers

Register event handlers to respond to system events:

```python
def on_peer_connected(peer_id: UUID, peer_info: PeerInfo):
    print(f"New peer connected: {peer_id}")

node.register_event_handler("peer_connected", on_peer_connected)
```

## Usage Examples

### Basic Node Setup

```python
import asyncio
from uuid import uuid4
from agent_mesh import MeshNode

async def main():
    # Create and initialize node
    node = MeshNode(
        node_id=uuid4(),
        host="0.0.0.0",
        port=8080
    )
    
    await node.initialize()
    await node.start()
    
    # Connect to existing network
    await node.connect_to_peer("192.168.1.100", 8080)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await node.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Federated Learning Example

```python
import torch
import torch.nn as nn
from agent_mesh.federated import FederatedLearner

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.fc(x)

async def federated_training():
    model = SimpleNet()
    
    learner = FederatedLearner(
        node_id=uuid4(),
        model=model,
        training_data=train_loader
    )
    
    await learner.initialize()
    
    # Perform local training
    result = await learner.local_train(epochs=5)
    print(f"Training loss: {result.loss:.4f}")
    
    # Get updated parameters
    params = await learner.get_model_parameters()
    return params
```

This API reference provides comprehensive documentation for integrating with and extending the Agent Mesh system. For additional examples and tutorials, see the examples/ directory in the repository.