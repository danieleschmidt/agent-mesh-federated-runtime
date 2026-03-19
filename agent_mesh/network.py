"""MeshNetwork — pure P2P topology with broadcast, unicast, and gossip."""

from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .node import AgentNode, NodeRole


# A message is just a plain dict so we stay stdlib-only.
Message = Dict[str, Any]
MessageHandler = Callable[[AgentNode, Message], None]


class MeshNetwork:
    """Peer-to-peer mesh with no central coordinator.

    Topology is represented as a bidirectional adjacency list.  All
    "network calls" are in-process (synchronous function invocations),
    which makes unit-testing trivial while preserving the structural
    semantics of a real P2P mesh.

    Usage
    -----
    net = MeshNetwork()
    net.add_node(node_a)
    net.add_node(node_b)
    net.connect(node_a.node_id, node_b.node_id)
    net.register_handler("hello", my_handler)
    net.unicast(node_a, node_b.node_id, {"type": "hello", "text": "hi"})
    """

    def __init__(self, gossip_fanout: int = 2) -> None:
        self._nodes: Dict[str, AgentNode] = {}
        # adjacency list: node_id → set of peer node_ids
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        # type-keyed message handlers
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        # in-order message log (for testing / replay)
        self._log: deque[Tuple[float, str, str, Message]] = deque(maxlen=10_000)
        self.gossip_fanout = gossip_fanout

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node: AgentNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._edges:
            self._edges[node.node_id] = set()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges (simulates node departure/crash)."""
        self._nodes.pop(node_id, None)
        # remove outgoing edges
        self._edges.pop(node_id, None)
        # remove incoming edges
        for peers in self._edges.values():
            peers.discard(node_id)

    def connect(self, node_id_a: str, node_id_b: str) -> None:
        """Add a bidirectional edge between two nodes."""
        self._edges[node_id_a].add(node_id_b)
        self._edges[node_id_b].add(node_id_a)

    def disconnect(self, node_id_a: str, node_id_b: str) -> None:
        self._edges[node_id_a].discard(node_id_b)
        self._edges[node_id_b].discard(node_id_a)

    def peers_of(self, node_id: str) -> List[AgentNode]:
        """Return alive peers directly connected to *node_id*."""
        return [
            self._nodes[pid]
            for pid in self._edges.get(node_id, set())
            if pid in self._nodes and self._nodes[pid].alive
        ]

    @property
    def alive_nodes(self) -> List[AgentNode]:
        return [n for n in self._nodes.values() if n.alive]

    def get_node(self, node_id: str) -> Optional[AgentNode]:
        return self._nodes.get(node_id)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def register_handler(self, msg_type: str, handler: MessageHandler) -> None:
        self._handlers[msg_type].append(handler)

    def _dispatch(self, sender: AgentNode, msg: Message) -> None:
        msg_type = msg.get("type", "__unknown__")
        for handler in self._handlers.get(msg_type, []):
            handler(sender, msg)

    # ------------------------------------------------------------------
    # Messaging primitives
    # ------------------------------------------------------------------

    def unicast(self, sender: AgentNode, target_id: str, msg: Message) -> bool:
        """Send a message to a single node.  Returns False if unreachable."""
        target = self._nodes.get(target_id)
        if not target or not target.alive:
            return False
        # check connectivity
        if target_id not in self._edges.get(sender.node_id, set()):
            return False
        self._record(sender.node_id, target_id, msg)
        self._dispatch(sender, {**msg, "_to": target_id, "_from": sender.node_id})
        return True

    def broadcast(self, sender: AgentNode, msg: Message) -> int:
        """Flood message to ALL alive nodes (except sender).

        Returns number of nodes reached.
        """
        reached = 0
        for node in self.alive_nodes:
            if node.node_id == sender.node_id:
                continue
            self._record(sender.node_id, node.node_id, msg)
            self._dispatch(sender, {**msg, "_to": node.node_id, "_from": sender.node_id})
            reached += 1
        return reached

    def gossip(
        self,
        sender: AgentNode,
        msg: Message,
        seen: Optional[Set[str]] = None,
        _depth: int = 0,
        _max_depth: int = 5,
    ) -> int:
        """Epidemic gossip — spread to *fanout* random peers, who re-gossip.

        Uses a *seen* set to prevent cycles.  Returns total nodes reached.
        """
        if seen is None:
            seen = set()
        seen.add(sender.node_id)

        if _depth >= _max_depth:
            return 0

        peers = [p for p in self.peers_of(sender.node_id) if p.node_id not in seen]
        targets = random.sample(peers, min(self.gossip_fanout, len(peers)))

        reached = 0
        for peer in targets:
            self._record(sender.node_id, peer.node_id, msg)
            self._dispatch(
                sender,
                {**msg, "_to": peer.node_id, "_from": sender.node_id, "_gossip": True},
            )
            reached += 1
            # re-gossip from peer
            reached += self.gossip(peer, msg, seen=seen, _depth=_depth + 1, _max_depth=_max_depth)
        return reached

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record(self, src: str, dst: str, msg: Message) -> None:
        self._log.append((time.time(), src, dst, msg))

    def topology_summary(self) -> str:
        lines = ["MeshNetwork topology:"]
        for nid, peers in sorted(self._edges.items()):
            node = self._nodes.get(nid)
            status = "✓" if (node and node.alive) else "✗"
            lines.append(f"  {status} {nid[:8]}… → {[p[:8]+'…' for p in sorted(peers)]}")
        return "\n".join(lines)

    def coordinator(self) -> Optional[AgentNode]:
        """Return the current alive coordinator, or None."""
        for node in self.alive_nodes:
            if node.role == NodeRole.COORDINATOR:
                return node
        return None
