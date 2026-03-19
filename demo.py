#!/usr/bin/env python3
"""
demo.py — 5-node mesh demonstration.

Covers:
  1. Build a 5-node fully-connected mesh and assign initial roles.
  2. Simulate coordinator crash → trigger re-election.
  3. Execute a distributed federated task (sum of squares) across workers.
  4. BFT majority vote with one faulty node.

Run:
  python demo.py
"""

from __future__ import annotations

import logging
import sys

from agent_mesh import (
    AgentNode,
    ByzantineFaultTolerance,
    FederatedTask,
    MeshNetwork,
    NodeRole,
    RoleNegotiator,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("demo")

SEP = "─" * 60


def build_full_mesh(n: int = 5) -> tuple[MeshNetwork, list[AgentNode]]:
    """Create n nodes in a fully-connected topology."""
    net = MeshNetwork(gossip_fanout=2)
    nodes = [AgentNode(capabilities=frozenset({"compute"})) for _ in range(n)]
    for node in nodes:
        net.add_node(node)
    # full mesh
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            net.connect(a.node_id, b.node_id)
    return net, nodes


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ──────────────────────────────────────────────
# 1. Initial setup
# ──────────────────────────────────────────────
section("1 · Build 5-node mesh & assign initial roles")

network, nodes = build_full_mesh(5)
negotiator = RoleNegotiator(network)
negotiator.assign_initial_roles()

for node in nodes:
    print(node)

coord = network.coordinator()
assert coord is not None
print(f"\n✓ Coordinator: {coord.node_id[:8]}… ({coord.role.value})")

# ──────────────────────────────────────────────
# 2. Coordinator failure + re-election
# ──────────────────────────────────────────────
section("2 · Simulate coordinator crash → re-election (Bully)")

old_coord_id = coord.node_id
coord.mark_dead()
print(f"✗ Node {old_coord_id[:8]}… crashed (was coordinator)\n")

assert negotiator.detect_failure(), "Expected failure detection"

new_coord = negotiator.elect_coordinator()
assert new_coord is not None
assert new_coord.node_id != old_coord_id
print(f"\n✓ New coordinator elected: {new_coord.node_id[:8]}… ({new_coord.role.value})")

for node in network.alive_nodes:
    print(f"  {node}")

# ──────────────────────────────────────────────
# 3. Distributed federated task
# ──────────────────────────────────────────────
section("3 · Distributed task: sum of squares over [1..20]")

data = list(range(1, 21))  # 1..20  →  sum of squares = 2870

workers = [n for n in network.alive_nodes if n.role == NodeRole.WORKER]
print(f"Workers: {len(workers)}  |  Payload: {data[:5]}…{data[-3:]}")

task = FederatedTask(
    payload=data,
    work_fn=lambda shard: sum(x ** 2 for x in shard),
    aggregator=lambda parts: sum(parts),
)

shards = task.split(len(workers))
for idx, (worker, shard) in enumerate(zip(workers, shards)):
    result = task.execute_shard(idx, shard, worker.node_id)
    print(f"  shard {idx}: worker={worker.node_id[:8]}… data={shard} → {result.value}")

final = task.aggregate()
expected = sum(x ** 2 for x in data)
print(f"\n✓ Aggregated result : {final}")
print(f"  Expected (ground truth): {expected}")
assert final == expected, f"Mismatch! {final} != {expected}"
print("  ✓ Match!")

# ──────────────────────────────────────────────
# 4. BFT majority vote with one faulty node
# ──────────────────────────────────────────────
section("4 · Byzantine fault tolerance (majority vote, 1 faulty node)")

bft = ByzantineFaultTolerance(threshold=0.67)
alive = network.alive_nodes

# 3 honest nodes report "42", 1 faulty node reports "99"
slot = "health_check"
honest_nodes = alive[:3]
faulty_node = alive[3]

for node in honest_nodes:
    bft.report(slot, node.node_id, 42)
    print(f"  {node.node_id[:8]}… reports: 42")

bft.report(slot, faulty_node.node_id, 99)
print(f"  {faulty_node.node_id[:8]}… reports: 99  ← FAULTY")

print(bft.summary(slot))

canonical, deviants = bft.decide(slot)
print(f"\n✓ Canonical value : {canonical}")
print(f"  Faulty nodes    : {[d[:8]+'…' for d in deviants]}")

assert canonical == 42
assert faulty_node.node_id in deviants
print("  ✓ Faulty node correctly identified!")

# ──────────────────────────────────────────────
# 5. Gossip propagation
# ──────────────────────────────────────────────
section("5 · Gossip protocol spread")

events = []
network.register_handler(
    "gossip_test",
    lambda sender, msg: events.append((sender.node_id[:8], msg.get("payload"))),
)

initiator = new_coord
reached = network.gossip(initiator, {"type": "gossip_test", "payload": "hello mesh"})
print(f"✓ Gossip reached {reached} additional nodes from {initiator.node_id[:8]}…")
print(f"  Events received: {len(events)}")

# ──────────────────────────────────────────────
section("All checks passed ✓")
