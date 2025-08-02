"""Mock helpers for advanced testing scenarios."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class MockNetworkDelays:
    """Simulate network delays and failures."""
    
    def __init__(self, base_latency_ms: float = 10):
        self.base_latency_ms = base_latency_ms
        self.failure_rate = 0.0
        self.partition_peers = set()
    
    async def add_delay(self, peer_id: Optional[str] = None):
        """Add network delay to operations."""
        delay = self.base_latency_ms / 1000.0
        if peer_id in self.partition_peers:
            raise ConnectionError(f"Network partition: {peer_id}")
        
        if self.failure_rate > 0 and time.time() % 1 < self.failure_rate:
            raise TimeoutError("Network timeout")
        
        await asyncio.sleep(delay)
    
    def set_failure_rate(self, rate: float):
        """Set network failure rate (0.0 to 1.0)."""
        self.failure_rate = max(0.0, min(1.0, rate))
    
    def partition_peer(self, peer_id: str):
        """Simulate network partition for a peer."""
        self.partition_peers.add(peer_id)
    
    def heal_partition(self, peer_id: str):
        """Heal network partition for a peer."""
        self.partition_peers.discard(peer_id)


class MockConsensusOracle:
    """Oracle for consensus testing with controllable outcomes."""
    
    def __init__(self):
        self.proposals = {}
        self.votes = {}
        self.byzantine_nodes = set()
        self.leader_election_results = {}
    
    def add_proposal(self, proposal_id: str, expected_outcome: bool):
        """Add a proposal with expected consensus outcome."""
        self.proposals[proposal_id] = expected_outcome
    
    def add_byzantine_node(self, node_id: str):
        """Mark a node as Byzantine (malicious)."""
        self.byzantine_nodes.add(node_id)
    
    def set_leader(self, term: int, leader_id: str):
        """Set expected leader for a term."""
        self.leader_election_results[term] = leader_id
    
    def should_vote_yes(self, node_id: str, proposal_id: str) -> bool:
        """Determine if a node should vote yes on a proposal."""
        if node_id in self.byzantine_nodes:
            # Byzantine nodes vote randomly or maliciously
            return time.time() % 2 < 1
        
        return self.proposals.get(proposal_id, True)
    
    def get_expected_leader(self, term: int) -> Optional[str]:
        """Get expected leader for a term."""
        return self.leader_election_results.get(term)


class MockByzantineNode:
    """Simulate Byzantine (malicious) node behavior."""
    
    def __init__(self, node_id: str, behavior: str = "random"):
        self.node_id = node_id
        self.behavior = behavior
        self.is_active = True
        self.message_corruption_rate = 0.3
    
    def should_send_message(self) -> bool:
        """Decide if the Byzantine node should send a message."""
        if not self.is_active:
            return False
        
        if self.behavior == "silent":
            return False
        elif self.behavior == "random":
            return time.time() % 2 < 1
        
        return True
    
    def corrupt_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Corrupt a message based on Byzantine behavior."""
        if time.time() % 1 > self.message_corruption_rate:
            return message
        
        corrupted = message.copy()
        
        if self.behavior == "equivocation":
            # Send different messages to different peers
            corrupted["equivocation_marker"] = time.time()
        elif self.behavior == "wrong_values":
            # Modify message content
            if "data" in corrupted:
                corrupted["data"] = "corrupted_data"
        
        return corrupted


class MockResourceManager:
    """Mock resource management for testing resource constraints."""
    
    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.network_bandwidth = 1000.0  # Mbps
        self.constraints = {}
    
    def set_cpu_usage(self, usage: float):
        """Set CPU usage percentage (0.0 to 1.0)."""
        self.cpu_usage = max(0.0, min(1.0, usage))
    
    def set_memory_usage(self, usage: float):
        """Set memory usage percentage (0.0 to 1.0)."""
        self.memory_usage = max(0.0, min(1.0, usage))
    
    def set_constraint(self, resource: str, limit: float):
        """Set resource constraint."""
        self.constraints[resource] = limit
    
    def check_resource_availability(self, resource: str, required: float) -> bool:
        """Check if required resources are available."""
        if resource == "cpu":
            return self.cpu_usage + required <= self.constraints.get("cpu", 1.0)
        elif resource == "memory":
            return self.memory_usage + required <= self.constraints.get("memory", 1.0)
        
        return True
    
    async def allocate_resources(self, resources: Dict[str, float]):
        """Allocate resources (may fail if insufficient)."""
        for resource, amount in resources.items():
            if not self.check_resource_availability(resource, amount):
                raise ResourceError(f"Insufficient {resource}")
        
        # Simulate resource allocation time
        await asyncio.sleep(0.01)


class ResourceError(Exception):
    """Resource allocation error."""
    pass


@asynccontextmanager
async def mock_distributed_environment(
    num_nodes: int = 5,
    byzantine_nodes: int = 1,
    network_delay_ms: float = 10,
    failure_rate: float = 0.0
):
    """Create a mock distributed environment for testing."""
    
    # Create network delay simulator
    network = MockNetworkDelays(network_delay_ms)
    network.set_failure_rate(failure_rate)
    
    # Create consensus oracle
    oracle = MockConsensusOracle()
    
    # Create nodes (some Byzantine)
    nodes = []
    for i in range(num_nodes):
        node_id = f"node-{i:03d}"
        if i < byzantine_nodes:
            node = MockByzantineNode(node_id, behavior="random")
            oracle.add_byzantine_node(node_id)
        else:
            node = MagicMock()
            node.node_id = node_id
        
        nodes.append(node)
    
    # Create resource manager
    resources = MockResourceManager()
    
    try:
        yield {
            "nodes": nodes,
            "network": network,
            "oracle": oracle,
            "resources": resources,
        }
    finally:
        # Cleanup
        pass


class FaultInjector:
    """Inject various types of faults for chaos testing."""
    
    def __init__(self):
        self.active_faults = {}
        self.fault_history = []
    
    async def inject_network_partition(
        self, 
        node_groups: List[List[str]], 
        duration: float = 10.0
    ):
        """Inject network partition between node groups."""
        fault_id = f"partition_{time.time()}"
        
        # Record fault
        self.fault_history.append({
            "type": "network_partition",
            "groups": node_groups,
            "start_time": time.time(),
            "duration": duration,
        })
        
        # Simulate partition
        await asyncio.sleep(duration)
    
    async def inject_node_failure(self, node_id: str, duration: float = 30.0):
        """Inject node failure."""
        fault_id = f"node_failure_{node_id}_{time.time()}"
        
        self.active_faults[fault_id] = {
            "type": "node_failure",
            "node_id": node_id,
            "start_time": time.time(),
            "duration": duration,
        }
        
        # Simulate node failure
        await asyncio.sleep(duration)
        
        # Remove fault
        self.active_faults.pop(fault_id, None)
    
    async def inject_message_corruption(
        self, 
        corruption_rate: float = 0.1, 
        duration: float = 15.0
    ):
        """Inject message corruption."""
        fault_id = f"corruption_{time.time()}"
        
        self.active_faults[fault_id] = {
            "type": "message_corruption",
            "rate": corruption_rate,
            "start_time": time.time(),
            "duration": duration,
        }
        
        await asyncio.sleep(duration)
        self.active_faults.pop(fault_id, None)
    
    def is_node_failed(self, node_id: str) -> bool:
        """Check if a node is currently failed."""
        current_time = time.time()
        
        for fault in self.active_faults.values():
            if (fault["type"] == "node_failure" and 
                fault["node_id"] == node_id and
                current_time - fault["start_time"] < fault["duration"]):
                return True
        
        return False
    
    def get_message_corruption_rate(self) -> float:
        """Get current message corruption rate."""
        current_time = time.time()
        max_rate = 0.0
        
        for fault in self.active_faults.values():
            if (fault["type"] == "message_corruption" and
                current_time - fault["start_time"] < fault["duration"]):
                max_rate = max(max_rate, fault["rate"])
        
        return max_rate


@pytest.fixture
def mock_network_delays():
    """Fixture for network delay simulation."""
    return MockNetworkDelays()


@pytest.fixture
def consensus_oracle():
    """Fixture for consensus testing oracle."""
    return MockConsensusOracle()


@pytest.fixture
def byzantine_node():
    """Fixture for Byzantine node simulation."""
    return MockByzantineNode("byzantine-001")


@pytest.fixture
def resource_manager():
    """Fixture for resource management testing."""
    return MockResourceManager()


@pytest.fixture
def fault_injector():
    """Fixture for fault injection testing."""
    return FaultInjector()


@pytest.fixture
async def distributed_test_env():
    """Fixture for distributed testing environment."""
    async with mock_distributed_environment() as env:
        yield env