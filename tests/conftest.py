"""
Pytest configuration and shared fixtures for Agent Mesh tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Test configuration
TEST_CONFIG = {
    "environment": "testing",
    "log_level": "DEBUG",
    "p2p_port_base": 14000,
    "grpc_port_base": 15000,
    "api_port_base": 18000,
    "test_timeout": 30,
    "network_size": 5,
    "byzantine_nodes": 1,
}

# Pytest configuration
pytest_plugins = [
    "pytest_asyncio",
]


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    policy = asyncio.DefaultEventLoopPolicy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def async_client():
    """Create an async HTTP client for API testing."""
    try:
        from httpx import AsyncClient
        async with AsyncClient() as client:
            yield client
    except ImportError:
        pytest.skip("httpx not installed")


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="function")
def mock_environment(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        "P2P_LISTEN_ADDR": "/ip4/127.0.0.1/tcp/14001",
        "GRPC_LISTEN_PORT": "15001",
        "LOCAL_DB_PATH": ":memory:",
        "METRICS_ENABLED": "false",
        "TRACING_ENABLED": "false",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture(scope="function")
def available_ports():
    """Generate available ports for testing."""
    import socket
    
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            return s.getsockname()[1]
    
    return {
        "p2p": get_free_port(),
        "grpc": get_free_port(),
        "api": get_free_port(),
        "metrics": get_free_port(),
    }


@pytest.fixture(scope="function")
def mock_crypto_keys():
    """Provide mock cryptographic keys for testing."""
    return {
        "private_key": "mock_private_key_bytes",
        "public_key": "mock_public_key_bytes",
        "certificate": "mock_certificate_pem",
        "ca_cert": "mock_ca_certificate_pem",
    }


@pytest.fixture(scope="function")
async def mock_mesh_node():
    """Create a mock mesh node for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    node = MagicMock()
    node.node_id = "test-node-001"
    node.listen_addr = "/ip4/127.0.0.1/tcp/14001"
    node.peers = {}
    node.is_running = False
    
    # Mock async methods
    node.start = AsyncMock()
    node.stop = AsyncMock()
    node.connect_to_peer = AsyncMock()
    node.disconnect_from_peer = AsyncMock()
    node.broadcast_message = AsyncMock()
    node.send_message = AsyncMock()
    
    return node


@pytest.fixture(scope="function")
async def mock_consensus_engine():
    """Create a mock consensus engine for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    engine = MagicMock()
    engine.consensus_type = "pbft"
    engine.fault_tolerance = 0.33
    engine.is_leader = False
    
    # Mock async methods
    engine.propose = AsyncMock()
    engine.vote = AsyncMock()
    engine.finalize = AsyncMock()
    engine.reach_consensus = AsyncMock()
    
    return engine


@pytest.fixture(scope="function")
async def mock_federated_learner():
    """Create a mock federated learner for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    learner = MagicMock()
    learner.model = MagicMock()
    learner.dataset = MagicMock()
    learner.aggregation_strategy = "fedavg"
    learner.current_round = 0
    learner.is_training = False
    
    # Mock async methods
    learner.train_round = AsyncMock()
    learner.aggregate_updates = AsyncMock()
    learner.evaluate_model = AsyncMock()
    
    return learner


@pytest.fixture(scope="function")
def sample_model_update():
    """Provide a sample model update for testing."""
    import numpy as np
    
    return {
        "round": 1,
        "node_id": "test-node-001",
        "weights": {
            "layer1": np.random.randn(10, 5).tolist(),
            "layer2": np.random.randn(5, 1).tolist(),
        },
        "metadata": {
            "samples": 1000,
            "loss": 0.5,
            "accuracy": 0.85,
        },
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.fixture(scope="function")
def sample_task():
    """Provide a sample collaborative task for testing."""
    return {
        "task_id": "task-001",
        "name": "distributed_reasoning",
        "type": "collaborative",
        "min_agents": 3,
        "max_agents": 10,
        "coordination_protocol": "contract_net",
        "requirements": {
            "capabilities": ["reasoning", "planning"],
            "resources": {"cpu": 2, "memory_gb": 4},
        },
        "deadline": "2024-01-02T00:00:00Z",
    }


@pytest.fixture(scope="function")
def network_topology():
    """Create a sample network topology for testing."""
    return {
        "nodes": [
            {"id": "node-001", "type": "trainer", "addr": "/ip4/127.0.0.1/tcp/14001"},
            {"id": "node-002", "type": "aggregator", "addr": "/ip4/127.0.0.1/tcp/14002"},
            {"id": "node-003", "type": "validator", "addr": "/ip4/127.0.0.1/tcp/14003"},
            {"id": "node-004", "type": "trainer", "addr": "/ip4/127.0.0.1/tcp/14004"},
            {"id": "node-005", "type": "trainer", "addr": "/ip4/127.0.0.1/tcp/14005"},
        ],
        "connections": [
            ("node-001", "node-002"),
            ("node-002", "node-003"),
            ("node-003", "node-004"),
            ("node-004", "node-005"),
            ("node-005", "node-001"),
        ],
    }


@pytest.fixture(scope="function")
async def test_database():
    """Create an in-memory test database."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import create_async_engine
        
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        yield engine
        await engine.dispose()
    except ImportError:
        pytest.skip("sqlalchemy not installed")


@pytest.fixture(scope="function")
def performance_config():
    """Configuration for performance tests."""
    return {
        "iterations": 100,
        "timeout": 10.0,
        "warmup_rounds": 3,
        "target_latency_ms": 100,
        "target_throughput": 1000,
    }


@pytest.fixture(scope="function")
def integration_network_config():
    """Configuration for integration test networks."""
    return {
        "network_size": 5,
        "byzantine_nodes": 1,
        "fault_injection": True,
        "network_delays": [10, 50, 100],  # milliseconds
        "partition_probability": 0.1,
    }


@pytest.fixture(scope="function")
def security_test_config():
    """Configuration for security tests."""
    return {
        "attack_types": ["sybil", "eclipse", "byzantine"],
        "malicious_nodes": [1, 2],
        "attack_duration": 30,  # seconds
        "recovery_timeout": 60,  # seconds
    }


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance/benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring network access"
    )
    config.addinivalue_line(
        "markers", "security: Security-related tests"
    )


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    for item in items:
        # Add slow marker to integration and e2e tests
        if "integration" in item.keywords or "e2e" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(pytest.mark.skip(reason="GPU not available"))
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))
        
        # Skip network tests if offline
        if "network" in item.keywords:
            import socket
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
            except OSError:
                item.add_marker(pytest.mark.skip(reason="Network not available"))


# Custom assertions
def assert_eventually(condition, timeout=5.0, interval=0.1):
    """Assert that a condition becomes true within a timeout period."""
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError(f"Condition not met within {timeout} seconds")


# Async test utilities
async def wait_for_condition(condition, timeout=5.0, interval=0.1):
    """Wait for an async condition to become true."""
    import asyncio
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if await condition():
            return
        await asyncio.sleep(interval)
    raise asyncio.TimeoutError(f"Condition not met within {timeout} seconds")


# Test data generators
def generate_random_bytes(size=32):
    """Generate random bytes for testing."""
    import secrets
    return secrets.token_bytes(size)


def generate_peer_id():
    """Generate a random peer ID."""
    import secrets
    return f"peer-{secrets.token_hex(8)}"


def generate_node_config(node_id=None, port_offset=0):
    """Generate a node configuration for testing."""
    if node_id is None:
        node_id = generate_peer_id()
    
    return {
        "node_id": node_id,
        "listen_addr": f"/ip4/127.0.0.1/tcp/{14001 + port_offset}",
        "grpc_port": 15001 + port_offset,
        "api_port": 18001 + port_offset,
        "metrics_port": 19001 + port_offset,
        "environment": "testing",
        "log_level": "DEBUG",
    }