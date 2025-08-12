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

# Core exports with graceful degradation
def _safe_import(module_path, class_name):
    """Safely import a class with fallback."""
    try:
        full_path = f"agent_mesh.{module_path}"
        module = __import__(full_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Return a dummy class for missing components
        class DummyComponent:
            def __init__(self, *args, **kwargs):
                import warnings
                warnings.warn(f"Using dummy implementation for {class_name}: {e}")
            
        DummyComponent.__name__ = class_name
        return DummyComponent

try:
    MeshNode = _safe_import("core.mesh_node", "MeshNode")
    P2PNetwork = _safe_import("core.network", "P2PNetwork") 
    ConsensusEngine = _safe_import("core.consensus", "ConsensusEngine")
    FederatedLearner = _safe_import("federated.learner", "FederatedLearner")
    SecureAggregator = _safe_import("federated.aggregator", "SecureAggregator")
    AgentMesh = _safe_import("coordination.agent_mesh", "AgentMesh")
    TaskScheduler = _safe_import("coordination.task_scheduler", "TaskScheduler")
except Exception as e:
    import warnings
    warnings.warn(f"Failed to import Agent Mesh components: {e}")

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