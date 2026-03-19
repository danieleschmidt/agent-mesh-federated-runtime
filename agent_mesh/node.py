"""AgentNode — a single node in the agent mesh."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, FrozenSet, Optional


class NodeRole(str, Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    OBSERVER = "observer"


class AgentNode:
    """Represents one participant in the mesh.

    Attributes:
        node_id: Globally unique identifier (UUID string).
        role: Current role in the mesh (mutable — can change via negotiation).
        capabilities: Immutable frozenset of capability tags (e.g. {"compute", "storage"}).
        alive: Whether this node is considered reachable by the mesh.
        joined_at: Unix timestamp when the node was created.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        role: NodeRole = NodeRole.WORKER,
        capabilities: Optional[FrozenSet[str]] = None,
    ) -> None:
        self.node_id: str = node_id or str(uuid.uuid4())
        self.role: NodeRole = role
        self.capabilities: FrozenSet[str] = frozenset(capabilities or [])
        self.alive: bool = True
        self.joined_at: float = time.time()
        # arbitrary metadata that protocols may attach
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def mark_dead(self) -> None:
        self.alive = False

    def mark_alive(self) -> None:
        self.alive = True

    def set_role(self, role: NodeRole) -> None:
        self.role = role

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        caps = ",".join(sorted(self.capabilities)) or "—"
        return (
            f"AgentNode(id={self.node_id[:8]}…, role={self.role.value}, "
            f"caps=[{caps}], {status})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)
