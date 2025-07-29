"""Agent Mesh Federated Runtime - Main Package.

A decentralized peer-to-peer runtime for federated learning and multi-agent systems.
Provides true P2P architecture with no single point of failure, automatic role
negotiation, and Byzantine fault tolerance.

Key Components:
    - MeshNode: Core P2P network node
    - FederatedLearner: Federated learning coordinator
    - AgentMesh: Multi-agent coordination system
    - ConsensusEngine: Byzantine fault-tolerant consensus
    - SecureAggregator: Privacy-preserving aggregation
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt <daniel@terragon.ai>"
__license__ = "MIT"

# Core exports
from .core.mesh_node import MeshNode
from .core.network import P2PNetwork
from .core.consensus import ConsensusEngine
from .federated.learner import FederatedLearner
from .federated.aggregator import SecureAggregator
from .coordination.agent_mesh import AgentMesh
from .coordination.task_scheduler import TaskScheduler

__all__ = [
    "__version__",
    "MeshNode", 
    "P2PNetwork",
    "ConsensusEngine",
    "FederatedLearner",
    "SecureAggregator", 
    "AgentMesh",
    "TaskScheduler",
]