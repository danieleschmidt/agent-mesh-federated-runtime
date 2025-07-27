# ADR-002: Consensus Protocol Selection

## Status
Accepted

## Context
The agent mesh system requires consensus protocols for different scenarios: Byzantine fault tolerance for critical operations, leader election for coordination, and efficient agreement for routine decisions. We need protocols that balance security, performance, and complexity.

## Decision
We will implement a **multi-layer consensus architecture**:
- **PBFT (Practical Byzantine Fault Tolerance)** for critical model updates and security decisions
- **Raft** for leader election and coordination tasks
- **Gossip-based eventual consistency** for metrics and status propagation

## Alternatives Considered

### Option 1: Single PBFT for Everything
- **Pros**: Strongest security guarantees, proven Byzantine tolerance
- **Cons**: High overhead for routine operations, complex implementation, scalability limits

### Option 2: Single Raft for Everything
- **Pros**: Simple implementation, good performance, well-understood
- **Cons**: No Byzantine fault tolerance, leader bottleneck, not suitable for adversarial environments

### Option 3: Blockchain-based Consensus (PoW/PoS)
- **Pros**: Decentralized, battle-tested in cryptocurrencies
- **Cons**: Energy intensive, high latency, not suitable for real-time applications

### Option 4: Multi-layer Architecture (Chosen)
- **Pros**: 
  - Right tool for each use case
  - Performance optimization through protocol selection
  - Graceful degradation under different threat models
  - Scalability through protocol specialization
- **Cons**: Implementation complexity, protocol coordination overhead

## Rationale

### Use Case Analysis
1. **Model Update Validation**: Requires Byzantine tolerance due to potential adversarial participants
2. **Task Coordination**: Needs leader election but lower security requirements
3. **Status Sharing**: Best-effort delivery sufficient, eventual consistency acceptable

### Performance Considerations
- PBFT: O(n²) message complexity, suitable for <100 nodes
- Raft: O(n) message complexity, scales to 1000+ nodes
- Gossip: O(log n) message complexity, scales to 10,000+ nodes

### Security Threat Model
- **Byzantine adversaries**: Up to 33% of nodes may be malicious
- **Crash failures**: Nodes may fail-stop without malicious behavior
- **Network partitions**: Temporary isolation of node subsets

## Implementation Strategy

### Protocol Selection Matrix
| Operation Type | Consensus Protocol | Fault Tolerance | Performance |
|---------------|-------------------|-----------------|-------------|
| Model Updates | PBFT | Byzantine (f < n/3) | Low throughput |
| Leader Election | Raft | Crash (f < n/2) | Medium throughput |
| Status Updates | Gossip | Network partition | High throughput |

### Integration Architecture
```python
class ConsensusManager:
    def __init__(self):
        self.pbft_engine = PBFTConsensus()
        self.raft_engine = RaftConsensus()
        self.gossip_engine = GossipProtocol()
    
    async def consensus_request(self, operation_type, data):
        if operation_type in CRITICAL_OPERATIONS:
            return await self.pbft_engine.propose(data)
        elif operation_type in COORDINATION_OPERATIONS:
            return await self.raft_engine.propose(data)
        else:
            return await self.gossip_engine.disseminate(data)
```

## Consequences

### Positive
- Optimized performance for different operation types
- Strong security guarantees where needed
- Scalable architecture supporting large networks
- Fault tolerance appropriate for each use case

### Negative
- Implementation complexity of multiple protocols
- Protocol coordination overhead
- Testing complexity across different consensus scenarios
- Potential for protocol interaction bugs

### Risk Mitigation
- Comprehensive protocol testing with chaos engineering
- Clear API boundaries between consensus layers
- Fallback mechanisms for protocol failures
- Regular security audits of consensus implementations

## Performance Targets

### PBFT Layer
- Finality time: <5 seconds for critical operations
- Throughput: 100 transactions/second
- Network size: Up to 100 nodes

### Raft Layer
- Leader election time: <2 seconds
- Log replication latency: <500ms
- Network size: Up to 1000 nodes

### Gossip Layer
- Propagation time: <10 seconds to 99% of network
- Message overhead: <1% of total bandwidth
- Network size: Up to 10,000 nodes

## Validation Criteria
- Consensus safety: No conflicting decisions under normal operation
- Liveness guarantee: Progress within specified time bounds
- Byzantine tolerance: Correct operation with up to f malicious nodes
- Performance benchmarks: Meet specified latency and throughput targets

## Related Decisions
- ADR-001: P2P Networking Protocol Choice
- ADR-003: Message Serialization Format
- ADR-005: Cryptographic Signature Schemes