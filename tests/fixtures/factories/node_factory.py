"""Factory functions for creating test mesh nodes."""

import secrets
from typing import Optional, Dict, Any
from unittest.mock import AsyncMock, MagicMock


def create_test_node(
    node_id: Optional[str] = None,
    role: str = "auto",
    port_offset: int = 0,
    **kwargs
) -> MagicMock:
    """Create a mock mesh node for testing.
    
    Args:
        node_id: Unique node identifier
        role: Node role (trainer, aggregator, validator, auto)
        port_offset: Port offset for multiple nodes
        **kwargs: Additional node configuration
        
    Returns:
        Mock mesh node with realistic configuration
    """
    if node_id is None:
        node_id = f"test-node-{secrets.token_hex(4)}"
    
    base_config = {
        "node_id": node_id,
        "role": role,
        "listen_addr": f"/ip4/127.0.0.1/tcp/{14001 + port_offset}",
        "grpc_port": 15001 + port_offset,
        "api_port": 18001 + port_offset,
        "metrics_port": 19001 + port_offset,
        "environment": "testing",
        "log_level": "DEBUG",
    }
    
    # Merge with provided kwargs
    config = {**base_config, **kwargs}
    
    # Create mock node
    node = MagicMock()
    
    # Set configuration attributes
    for key, value in config.items():
        setattr(node, key, value)
    
    # Set runtime state
    node.is_running = False
    node.peers = {}
    node.tasks = {}
    node.metrics = {}
    
    # Mock async methods
    node.start = AsyncMock()
    node.stop = AsyncMock()
    node.connect_to_peer = AsyncMock()
    node.disconnect_from_peer = AsyncMock()
    node.broadcast_message = AsyncMock()
    node.send_message = AsyncMock()
    node.join_network = AsyncMock()
    node.leave_network = AsyncMock()
    node.attach_task = AsyncMock()
    node.detach_task = AsyncMock()
    
    # Mock sync methods
    node.get_peer_count = MagicMock(return_value=0)
    node.get_task_count = MagicMock(return_value=0)
    node.get_metrics = MagicMock(return_value={})
    node.is_connected_to = MagicMock(return_value=False)
    
    return node


def create_test_network(
    size: int = 5,
    byzantine_nodes: int = 1,
    **kwargs
) -> list[MagicMock]:
    """Create a network of test nodes.
    
    Args:
        size: Number of nodes in the network
        byzantine_nodes: Number of byzantine (malicious) nodes
        **kwargs: Additional configuration for all nodes
        
    Returns:
        List of connected mock nodes
    """
    nodes = []
    
    for i in range(size):
        # Determine node role
        if i < byzantine_nodes:
            role = "byzantine"
        elif i == 0:
            role = "aggregator"
        else:
            role = "trainer"
        
        node = create_test_node(
            node_id=f"node-{i:03d}",
            role=role,
            port_offset=i,
            **kwargs
        )
        
        # Set network state
        node.network_size = size
        node.node_index = i
        
        nodes.append(node)
    
    # Mock network connections
    for i, node in enumerate(nodes):
        # Connect to all other nodes (full mesh)
        peer_ids = [n.node_id for j, n in enumerate(nodes) if j != i]
        node.peers = {pid: MagicMock() for pid in peer_ids}
        node.get_peer_count.return_value = len(peer_ids)
    
    return nodes


def create_bootstrap_node(**kwargs) -> MagicMock:
    """Create a bootstrap node for network initialization.
    
    Args:
        **kwargs: Additional node configuration
        
    Returns:
        Mock bootstrap node
    """
    config = {
        "node_id": "bootstrap-node",
        "role": "bootstrap",
        "is_bootstrap": True,
        **kwargs
    }
    
    node = create_test_node(**config)
    
    # Bootstrap-specific methods
    node.accept_new_node = AsyncMock()
    node.provide_peer_list = AsyncMock(return_value=[])
    node.announce_new_peer = AsyncMock()
    
    return node


def create_edge_node(**kwargs) -> MagicMock:
    """Create an edge node with resource constraints.
    
    Args:
        **kwargs: Additional node configuration
        
    Returns:
        Mock edge node with limited resources
    """
    config = {
        "role": "trainer",
        "node_type": "edge",
        "max_memory_mb": 512,
        "max_cpu_percent": 50,
        "battery_aware": True,
        "connectivity": "intermittent",
        **kwargs
    }
    
    node = create_test_node(**config)
    
    # Edge-specific methods
    node.get_battery_level = MagicMock(return_value=0.8)
    node.get_memory_usage = MagicMock(return_value=0.6)
    node.get_cpu_usage = MagicMock(return_value=0.4)
    node.enter_power_save_mode = AsyncMock()
    node.exit_power_save_mode = AsyncMock()
    
    return node


def create_gpu_node(**kwargs) -> MagicMock:
    """Create a GPU-enabled node for ML training.
    
    Args:
        **kwargs: Additional node configuration
        
    Returns:
        Mock GPU node
    """
    config = {
        "role": "trainer",
        "node_type": "gpu",
        "gpu_available": True,
        "gpu_memory_gb": 8,
        "gpu_compute_capability": "7.5",
        **kwargs
    }
    
    node = create_test_node(**config)
    
    # GPU-specific methods
    node.get_gpu_utilization = MagicMock(return_value=0.7)
    node.get_gpu_memory_usage = MagicMock(return_value=0.5)
    node.move_model_to_gpu = AsyncMock()
    node.move_model_to_cpu = AsyncMock()
    
    return node
