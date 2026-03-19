# agent-mesh-federated-runtime

A decentralized peer-to-peer runtime for federated multi-agent systems.  
**Zero external dependencies** — pure Python 3.10+ stdlib.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                     MeshNetwork                         │
│  (adjacency list, broadcast / unicast / gossip)         │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │AgentNode │────│AgentNode │────│AgentNode │          │
│  │COORDINATOR    │ WORKER   │    │ WORKER   │          │
│  └────┬─────┘    └──────────┘    └──────────┘          │
│       │                ╲              /                  │
│       │           ┌──────────┐                          │
│       │           │AgentNode │                          │
│  RoleNegotiator   │ WORKER   │  ← BullyElection         │
│  (Bully alg.)     └──────────┘                          │
└────────────────────────────────────────────────────────┘
         │                     │
  FederatedTask          ByzantineFaultTolerance
  (split → workers       (majority vote, deviant
   → aggregate)           detection & exclusion)
```

---

## Core components

| Module | Class | Responsibility |
|--------|-------|----------------|
| `node.py` | `AgentNode` | Identity, role, capability set, liveness |
| `network.py` | `MeshNetwork` | P2P topology; broadcast / unicast / gossip |
| `negotiator.py` | `RoleNegotiator` | Dynamic role assignment; Bully coordinator election |
| `task.py` | `FederatedTask` | Split payload across workers, aggregate results |
| `bft.py` | `ByzantineFaultTolerance` | Majority-vote consensus, faulty-node exclusion |

---

## Quick start

```python
from agent_mesh import AgentNode, MeshNetwork, RoleNegotiator, FederatedTask, ByzantineFaultTolerance

# Build a mesh
net = MeshNetwork()
nodes = [AgentNode(capabilities=frozenset({"compute"})) for _ in range(5)]
for n in nodes:
    net.add_node(n)
for i, a in enumerate(nodes):
    for b in nodes[i+1:]:
        net.connect(a.node_id, b.node_id)

# Assign roles
neg = RoleNegotiator(net)
neg.assign_initial_roles()

# Crash the coordinator → re-elect
net.coordinator().mark_dead()
new_coord = neg.elect_coordinator()

# Distributed task
task = FederatedTask(payload=list(range(20)), work_fn=lambda shard: sum(x**2 for x in shard))
workers = [n for n in net.alive_nodes]
shards = task.split(len(workers))
for i, (worker, shard) in enumerate(zip(workers, shards)):
    task.execute_shard(i, shard, worker.node_id)
print(task.aggregate())  # sum of squares

# BFT vote
bft = ByzantineFaultTolerance(threshold=0.67)
canonical, faulty = bft.collect_and_decide("round1", [
    ("n1", 42), ("n2", 42), ("n3", 42), ("n4", 99)  # n4 is faulty
])
print(canonical, faulty)
```

---

## Run the demo

```bash
python demo.py
```

Demonstrates:
1. 5-node fully-connected mesh + initial role assignment  
2. Coordinator crash + Bully re-election  
3. Distributed sum-of-squares task across workers  
4. BFT majority vote with one faulty node  
5. Gossip protocol propagation  

---

## Tests

```bash
# with pytest (any modern version)
python -m pytest tests/ -v

# or plain unittest
python -m unittest discover tests
```

---

## Design decisions

**No central coordinator** — the `MeshNetwork` has no master node; every node
has equal status in the topology.  The *coordinator role* is a logical label
negotiated dynamically.

**Bully election** — simple, deterministic, and correct for crash failures.
Higher-ID node wins.  Works in O(n²) messages in the worst case — fine for
small-to-medium meshes.

**Majority BFT** — tolerates up to ⌊(n−1)/3⌋ faulty nodes when threshold ≥ 2/3.
Faulty nodes are excluded from future voting rounds automatically.

**Pluggable aggregation** — `FederatedTask` accepts any `aggregator` function
(element-wise sum by default).  Useful for gradient averaging in federated
learning or custom reduce operations.

**Stdlib only** — no `asyncio`, `numpy`, `cryptography`, or any third-party
package.  Easy to embed in resource-constrained environments.

---

## License

MIT
