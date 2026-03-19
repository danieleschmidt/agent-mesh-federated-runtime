"""RoleNegotiator — dynamic role negotiation using the Bully election algorithm.

Bully algorithm summary
-----------------------
When a node detects the coordinator is gone it starts an election by sending
ELECTION messages to all nodes with a *higher* ID.  If no one responds it
declares itself coordinator.  If a higher node responds it yields.

In a single-process simulation we collapse the async messaging into a
synchronous pass; each "round" of the election runs deterministically.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .node import AgentNode, NodeRole
from .network import MeshNetwork

logger = logging.getLogger(__name__)


class RoleNegotiator:
    """Manages role assignment and coordinator elections across the mesh.

    Parameters
    ----------
    network : MeshNetwork
        The shared network instance all nodes participate in.
    """

    def __init__(self, network: MeshNetwork) -> None:
        self.network = network

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_initial_roles(self) -> None:
        """Assign roles on startup: highest-ID node becomes coordinator,
        rest become workers."""
        alive = self._alive_sorted()
        if not alive:
            return
        for node in alive:
            node.set_role(NodeRole.WORKER)
        alive[-1].set_role(NodeRole.COORDINATOR)
        logger.info("Initial coordinator: %s", alive[-1].node_id[:8])

    def detect_failure(self) -> bool:
        """Return True if there is no alive coordinator in the mesh."""
        return self.network.coordinator() is None

    def elect_coordinator(self, initiator: Optional[AgentNode] = None) -> Optional[AgentNode]:
        """Run a Bully election and return the newly elected coordinator.

        If *initiator* is None the lowest-ID alive node starts the election
        (any node that detects the coordinator is gone may initiate).
        """
        alive = self._alive_sorted()
        if not alive:
            logger.warning("No alive nodes — election aborted.")
            return None

        if initiator is None:
            initiator = alive[0]

        logger.info(
            "Election started by %s (%d alive nodes)",
            initiator.node_id[:8],
            len(alive),
        )

        # Bully: send ELECTION to every node with a higher ID
        higher = [n for n in alive if n.node_id > initiator.node_id]

        if not higher:
            # No higher node responded → initiator wins
            return self._crown(initiator)

        # In a sync simulation: find the highest node among those
        # reachable from initiator (direct peers + transitive)
        reachable_higher = [
            n for n in higher if self._is_reachable(initiator, n)
        ]

        if not reachable_higher:
            # Higher nodes exist but are partitioned away → initiator wins
            return self._crown(initiator)

        # The highest reachable node wins
        winner = max(reachable_higher, key=lambda n: n.node_id)
        return self._crown(winner)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alive_sorted(self) -> List[AgentNode]:
        """Return alive nodes sorted by node_id (ascending)."""
        return sorted(self.network.alive_nodes, key=lambda n: n.node_id)

    def _is_reachable(self, src: AgentNode, dst: AgentNode) -> bool:
        """BFS reachability check through alive nodes."""
        visited = set()
        queue = [src.node_id]
        while queue:
            current = queue.pop(0)
            if current == dst.node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for peer in self.network.peers_of(current):
                if peer.node_id not in visited:
                    queue.append(peer.node_id)
        return False

    def _crown(self, winner: AgentNode) -> AgentNode:
        """Set *winner* as coordinator and demote all others to worker."""
        for node in self.network.alive_nodes:
            if node.node_id == winner.node_id:
                node.set_role(NodeRole.COORDINATOR)
            elif node.role == NodeRole.COORDINATOR:
                node.set_role(NodeRole.WORKER)
        logger.info("New coordinator elected: %s", winner.node_id[:8])
        # Broadcast the result
        self.network.broadcast(
            winner,
            {
                "type": "coordinator_elected",
                "coordinator_id": winner.node_id,
            },
        )
        return winner
