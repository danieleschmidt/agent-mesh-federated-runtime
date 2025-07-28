"""Test helper utilities for Agent Mesh Federated Runtime."""

import asyncio
import time
import socket
from typing import Any, Callable, Optional, Dict, List
from contextlib import asynccontextmanager, contextmanager
import threading
import queue


class AsyncTestHelper:
    """Helper class for async testing utilities."""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        error_msg: Optional[str] = None
    ) -> None:
        """Wait for a condition to become true with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return
            await asyncio.sleep(interval)
        
        msg = error_msg or f"Condition not met within {timeout} seconds"
        raise asyncio.TimeoutError(msg)
    
    @staticmethod
    async def wait_for_async_condition(
        condition: Callable[[], Any],
        timeout: float = 5.0,
        interval: float = 0.1,
        error_msg: Optional[str] = None
    ) -> None:
        """Wait for an async condition to become true with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition():
                return
            await asyncio.sleep(interval)
        
        msg = error_msg or f"Async condition not met within {timeout} seconds"
        raise asyncio.TimeoutError(msg)
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 10.0):
        """Run a coroutine with a timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)


class NetworkTestHelper:
    """Helper class for network testing utilities."""
    
    @staticmethod
    def get_free_port() -> int:
        """Get a free port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            return s.getsockname()[1]
    
    @staticmethod
    def get_free_ports(count: int) -> List[int]:
        """Get multiple free ports for testing."""
        return [NetworkTestHelper.get_free_port() for _ in range(count)]
    
    @staticmethod
    def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
        """Check if a port is open."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False
    
    @staticmethod
    async def wait_for_port_open(
        host: str, 
        port: int, 
        timeout: float = 10.0,
        interval: float = 0.1
    ) -> None:
        """Wait for a port to become open."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if NetworkTestHelper.is_port_open(host, port):
                return
            await asyncio.sleep(interval)
        
        raise TimeoutError(f"Port {host}:{port} not open within {timeout} seconds")


class MockNetworkSimulator:
    """Simulate network conditions for testing."""
    
    def __init__(self):
        self.latency_ms = 0
        self.packet_loss_rate = 0.0
        self.bandwidth_kbps = float('inf')
        self.jitter_ms = 0
        self._enabled = False
    
    def set_network_conditions(
        self,
        latency_ms: int = 0,
        packet_loss_rate: float = 0.0,
        bandwidth_kbps: float = float('inf'),
        jitter_ms: int = 0
    ):
        """Set network simulation parameters."""
        self.latency_ms = latency_ms
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth_kbps = bandwidth_kbps
        self.jitter_ms = jitter_ms
    
    async def simulate_delay(self):
        """Simulate network delay."""
        if not self._enabled:
            return
        
        delay = self.latency_ms / 1000.0
        if self.jitter_ms > 0:
            import random
            jitter = random.uniform(-self.jitter_ms, self.jitter_ms) / 1000.0
            delay += jitter
        
        if delay > 0:
            await asyncio.sleep(delay)
    
    def simulate_packet_loss(self) -> bool:
        """Simulate packet loss."""
        if not self._enabled:
            return False
        
        import random
        return random.random() < self.packet_loss_rate
    
    @contextmanager
    def enable_simulation(self):
        """Context manager to enable network simulation."""
        self._enabled = True
        try:
            yield self
        finally:
            self._enabled = False


class PerformanceTracker:
    """Track performance metrics during tests."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        del self.start_times[name]
        return duration
    
    @contextmanager
    def time_operation(self, name: str):
        """Context manager to time an operation."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timed operation."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'total': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'std': (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


class MessageCollector:
    """Collect messages during tests for verification."""
    
    def __init__(self, max_size: int = 1000):
        self.messages = queue.Queue(maxsize=max_size)
        self.collected = []
        self._lock = threading.Lock()
    
    def add_message(self, message: Any):
        """Add a message to the collector."""
        with self._lock:
            self.collected.append({
                'timestamp': time.time(),
                'message': message
            })
            
        if not self.messages.full():
            self.messages.put(message)
    
    def get_messages(self, timeout: float = 1.0) -> List[Any]:
        """Get all collected messages."""
        messages = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                break
        
        return messages
    
    def wait_for_message(self, timeout: float = 5.0) -> Any:
        """Wait for a single message."""
        return self.messages.get(timeout=timeout)
    
    def clear(self):
        """Clear all messages."""
        with self._lock:
            self.collected.clear()
        
        while not self.messages.empty():
            try:
                self.messages.get_nowait()
            except queue.Empty:
                break
    
    def count(self) -> int:
        """Get the count of collected messages."""
        with self._lock:
            return len(self.collected)


class ByzantineTestHelper:
    """Helper for testing Byzantine fault tolerance."""
    
    @staticmethod
    def create_malicious_behavior(attack_type: str) -> Dict[str, Any]:
        """Create configuration for malicious node behavior."""
        behaviors = {
            'silent': {
                'description': 'Node stops responding',
                'actions': ['stop_responding'],
                'probability': 1.0
            },
            'random': {
                'description': 'Node sends random messages',
                'actions': ['send_random_data'],
                'probability': 0.5
            },
            'delay': {
                'description': 'Node introduces delays',
                'actions': ['add_delay'],
                'parameters': {'delay_ms': 1000},
                'probability': 1.0
            },
            'corrupt': {
                'description': 'Node corrupts message data',
                'actions': ['corrupt_data'],
                'probability': 0.3
            },
            'eclipse': {
                'description': 'Node attempts eclipse attack',
                'actions': ['isolate_peers', 'false_routing'],
                'probability': 1.0
            },
            'sybil': {
                'description': 'Node creates multiple identities',
                'actions': ['create_fake_identities'],
                'parameters': {'fake_count': 5},
                'probability': 1.0
            }
        }
        
        return behaviors.get(attack_type, behaviors['random'])
    
    @staticmethod
    def calculate_fault_tolerance(total_nodes: int, byzantine_nodes: int) -> Dict[str, Any]:
        """Calculate fault tolerance metrics."""
        byzantine_ratio = byzantine_nodes / total_nodes
        theoretical_max = total_nodes // 3
        
        return {
            'total_nodes': total_nodes,
            'byzantine_nodes': byzantine_nodes,
            'byzantine_ratio': byzantine_ratio,
            'theoretical_max_byzantine': theoretical_max,
            'within_tolerance': byzantine_nodes <= theoretical_max,
            'safety_margin': theoretical_max - byzantine_nodes,
            'consensus_possible': byzantine_nodes < total_nodes // 2
        }


# Test data generators
def generate_test_crypto_keys() -> Dict[str, bytes]:
    """Generate mock cryptographic keys for testing."""
    import secrets
    
    return {
        'private_key': secrets.token_bytes(32),
        'public_key': secrets.token_bytes(32),
        'shared_secret': secrets.token_bytes(32),
        'signature': secrets.token_bytes(64),
        'certificate': b'-----BEGIN CERTIFICATE-----\nMOCK_CERT_DATA\n-----END CERTIFICATE-----'
    }


def generate_test_peer_info(count: int = 1) -> List[Dict[str, Any]]:
    """Generate test peer information."""
    import secrets
    
    peers = []
    for i in range(count):
        peers.append({
            'peer_id': f'peer-{secrets.token_hex(8)}',
            'address': f'/ip4/127.0.0.1/tcp/{14000 + i}',
            'public_key': secrets.token_bytes(32),
            'capabilities': ['training', 'aggregation', 'validation'],
            'resources': {
                'cpu_cores': 4,
                'memory_gb': 8,
                'gpu_available': i % 2 == 0
            },
            'reputation': 0.8 + (i * 0.05),
            'online': True
        })
    
    return peers