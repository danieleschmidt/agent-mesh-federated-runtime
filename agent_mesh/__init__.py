"""
agent_mesh — Decentralized peer-to-peer runtime for federated multi-agent systems.

Stdlib-only. No dependencies.
"""

from .node import AgentNode, NodeRole
from .network import MeshNetwork
from .negotiator import RoleNegotiator
from .task import FederatedTask, TaskResult
from .bft import ByzantineFaultTolerance

__all__ = [
    "AgentNode",
    "NodeRole",
    "MeshNetwork",
    "RoleNegotiator",
    "FederatedTask",
    "TaskResult",
    "ByzantineFaultTolerance",
]
__version__ = "1.0.0"
