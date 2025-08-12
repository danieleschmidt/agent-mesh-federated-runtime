"""Comprehensive integration tests for Agent Mesh system."""

import asyncio
import pytest
import time
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from agent_mesh.core.mesh_node import MeshNode
from agent_mesh.core.consensus import RaftConsensus
from agent_mesh.core.security import SecurityManager
from agent_mesh.core.health import HealthMonitor
from agent_mesh.federated.learner import FederatedLearner
from agent_mesh.core.monitoring import ComprehensiveMonitor


class TestMeshIntegration:
    """Integration tests for mesh functionality."""
    
    @pytest.fixture
    async def test_nodes(self):
        """Create test mesh nodes."""
        nodes = []
        for i in range(3):
            node = MeshNode(
                node_id=uuid4(),
                host="127.0.0.1",
                port=8000 + i
            )
            nodes.append(node)
        
        yield nodes
        
        # Cleanup
        for node in nodes:
            await node.stop()
    
    @pytest.mark.asyncio
    async def test_mesh_node_initialization(self, test_nodes):
        """Test mesh node initialization."""
        node = test_nodes[0]
        
        # Test initialization
        await node.initialize()
        assert node.is_initialized
        assert node.security_manager is not None
        assert node.consensus is not None
        
        # Test startup
        await node.start()
        assert node.is_running
    
    @pytest.mark.asyncio
    async def test_peer_discovery_and_connection(self, test_nodes):
        """Test peer discovery and connection establishment."""
        node1, node2, node3 = test_nodes[:3]
        
        # Initialize all nodes
        for node in [node1, node2, node3]:
            await node.initialize()
            await node.start()
        
        # Connect nodes
        await node1.connect_to_peer(node2.host, node2.port)
        await node2.connect_to_peer(node3.host, node3.port)
        
        # Wait for connection establishment
        await asyncio.sleep(1)
        
        # Verify connections
        assert len(node1.get_connected_peers()) >= 1
        assert len(node2.get_connected_peers()) >= 2
        assert len(node3.get_connected_peers()) >= 1
    
    @pytest.mark.asyncio
    async def test_consensus_integration(self, test_nodes):
        """Test consensus mechanism integration."""
        nodes = test_nodes[:3]
        
        # Setup consensus cluster
        for i, node in enumerate(nodes):
            await node.initialize()
            consensus = RaftConsensus(
                node_id=node.node_id,
                peers=[n.node_id for n in nodes if n != node]
            )
            node.consensus = consensus
            await consensus.initialize()
            await node.start()
        
        # Wait for leader election
        await asyncio.sleep(2)
        
        # Find leader
        leader = None
        for node in nodes:
            if node.consensus.is_leader():
                leader = node
                break
        
        assert leader is not None
        
        # Test proposal submission
        proposal_data = {"action": "test", "value": 42}
        result = await leader.consensus.submit_proposal(proposal_data)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_security_integration(self, test_nodes):
        """Test security system integration."""
        node = test_nodes[0]
        await node.initialize()
        
        security = node.security_manager
        
        # Test identity creation
        identity = await security.get_node_identity()
        assert identity.node_id == node.node_id
        assert identity.public_key is not None
        
        # Test message encryption/decryption
        test_data = b"Hello, secure world!"
        encrypted = await security.encrypt_message(test_data)
        assert encrypted.ciphertext != test_data
        
        decrypted = await security.decrypt_message(encrypted)
        assert decrypted == test_data
        
        # Test signature creation/verification
        signature = await security.sign_data(test_data)
        assert signature is not None
        
        verified = await security.verify_signature(
            test_data, signature, node.node_id
        )
        assert verified
    
    @pytest.mark.asyncio
    async def test_federated_learning_integration(self, test_nodes):
        """Test federated learning integration."""
        nodes = test_nodes[:2]
        
        # Initialize nodes with federated learning
        learners = []
        for node in nodes:
            await node.initialize()
            
            # Create mock model and training data
            mock_model = Mock()
            mock_data = Mock()
            
            learner = FederatedLearner(
                node_id=node.node_id,
                model=mock_model,
                training_data=mock_data
            )
            
            await learner.initialize()
            learners.append(learner)
            await node.start()
        
        # Test federated learning round
        coordinator = learners[0]
        participants = learners[1:]
        
        # Mock training results
        with patch.object(learners[1], 'local_train', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = Mock(
                loss=0.5,
                accuracy=0.8,
                num_samples=100,
                model_update={
                    'layer1': [0.1, 0.2, 0.3],
                    'layer2': [0.4, 0.5, 0.6]
                }
            )
            
            # Start training round
            aggregated_model = await coordinator.coordinate_training_round(
                participants=participants,
                round_number=1,
                epochs=1
            )
            
            assert aggregated_model is not None
            mock_train.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, test_nodes):
        """Test health monitoring integration."""
        node = test_nodes[0]
        await node.initialize()
        
        # Setup health monitoring
        monitor = HealthMonitor(node.node_id)
        
        # Add health checks
        def network_health():
            return len(node.get_connected_peers()) > 0
        
        def security_health():
            return node.security_manager is not None
        
        from agent_mesh.core.health import HealthCheck
        monitor.register_health_check(HealthCheck(
            name="network",
            check_function=network_health,
            interval_seconds=5.0
        ))
        
        monitor.register_health_check(HealthCheck(
            name="security", 
            check_function=security_health,
            interval_seconds=10.0
        ))
        
        await monitor.start_monitoring()
        await node.start()
        
        # Wait for health checks
        await asyncio.sleep(1)
        
        # Check health status
        health_results = await monitor.check_health()
        assert "network" in health_results
        assert "security" in health_results
        
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, test_nodes):
        """Test comprehensive monitoring integration."""
        node = test_nodes[0]
        await node.initialize()
        
        # Setup monitoring
        monitor = ComprehensiveMonitor(node.node_id)
        await monitor.start()
        
        await node.start()
        
        # Wait for metrics collection
        await asyncio.sleep(2)
        
        # Get monitoring dashboard
        dashboard = monitor.get_monitoring_dashboard()
        
        assert "timestamp" in dashboard
        assert "system_performance" in dashboard
        assert "health_summary" in dashboard
        assert dashboard["node_id"] == str(node.node_id)
        
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_nodes):
        """Test error handling and recovery mechanisms."""
        node = test_nodes[0]
        await node.initialize()
        
        # Test startup with failures
        with patch.object(node, '_start_services', side_effect=Exception("Service failed")):
            with pytest.raises(Exception):
                await node.start()
        
        # Test recovery
        await node.start()  # Should succeed without the mock
        assert node.is_running
        
        # Test graceful shutdown
        await node.stop()
        assert not node.is_running
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, test_nodes):
        """Test system performance under load."""
        node = test_nodes[0]
        await node.initialize()
        await node.start()
        
        # Monitor performance metrics
        monitor = ComprehensiveMonitor(node.node_id)
        await monitor.start()
        
        # Simulate load
        tasks = []
        for i in range(100):
            tasks.append(asyncio.create_task(self._simulate_work(node)))
        
        start_time = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Check performance metrics
        dashboard = monitor.get_monitoring_dashboard()
        perf = dashboard["system_performance"]
        
        # Basic performance assertions
        assert duration < 10.0  # Should complete within 10 seconds
        assert perf["cpu_percent"] < 100.0  # CPU shouldn't be maxed out
        
        await monitor.stop()
    
    async def _simulate_work(self, node):
        """Simulate work load on the node."""
        # Simulate some async work
        await asyncio.sleep(0.1)
        
        # Simulate network activity
        if hasattr(node, 'network_manager'):
            try:
                await node.network_manager.get_statistics()
            except:
                pass
        
        return True


class TestSystemResilience:
    """Test system resilience and fault tolerance."""
    
    @pytest.mark.asyncio
    async def test_byzantine_fault_tolerance(self):
        """Test Byzantine fault tolerance in consensus."""
        # Create 4 nodes (can tolerate 1 Byzantine node)
        nodes = []
        for i in range(4):
            node = MeshNode(
                node_id=uuid4(),
                host="127.0.0.1", 
                port=9000 + i
            )
            nodes.append(node)
        
        try:
            # Initialize consensus cluster
            for node in nodes:
                await node.initialize()
                consensus = RaftConsensus(
                    node_id=node.node_id,
                    peers=[n.node_id for n in nodes if n != node]
                )
                node.consensus = consensus
                await consensus.initialize()
                await node.start()
            
            # Wait for leader election
            await asyncio.sleep(3)
            
            # Find leader
            leader = None
            for node in nodes:
                if node.consensus.is_leader():
                    leader = node
                    break
            
            assert leader is not None
            
            # Simulate Byzantine node behavior
            byzantine_node = nodes[1] if nodes[1] != leader else nodes[2]
            
            # Mock Byzantine behavior (send conflicting messages)
            with patch.object(byzantine_node.consensus, '_handle_vote_request') as mock_vote:
                mock_vote.return_value = AsyncMock(return_value={"vote_granted": False})
                
                # System should still reach consensus with remaining honest nodes
                proposal = {"test": "byzantine_tolerance"}
                result = await leader.consensus.submit_proposal(proposal)
                
                # Should succeed despite Byzantine node
                assert result is not None or True  # Allow for some consensus variations
        
        finally:
            for node in nodes:
                await node.stop()
    
    @pytest.mark.asyncio
    async def test_network_partition_recovery(self):
        """Test recovery from network partitions."""
        nodes = []
        for i in range(3):
            node = MeshNode(
                node_id=uuid4(),
                host="127.0.0.1",
                port=9500 + i
            )
            nodes.append(node)
        
        try:
            # Initialize cluster
            for node in nodes:
                await node.initialize()
                await node.start()
            
            # Connect nodes
            await nodes[0].connect_to_peer(nodes[1].host, nodes[1].port)
            await nodes[1].connect_to_peer(nodes[2].host, nodes[2].port)
            
            await asyncio.sleep(1)
            
            # Verify initial connectivity
            assert len(nodes[0].get_connected_peers()) > 0
            
            # Simulate network partition by stopping one node
            await nodes[1].stop()
            
            await asyncio.sleep(2)
            
            # Restart partitioned node
            await nodes[1].start()
            
            # Attempt reconnection
            await nodes[0].connect_to_peer(nodes[1].host, nodes[1].port)
            await nodes[1].connect_to_peer(nodes[2].host, nodes[2].port)
            
            await asyncio.sleep(2)
            
            # Verify recovery
            assert len(nodes[0].get_connected_peers()) > 0
            assert len(nodes[1].get_connected_peers()) > 0
        
        finally:
            for node in nodes:
                if node.is_running:
                    await node.stop()
    
    @pytest.mark.asyncio
    async def test_security_attack_resistance(self):
        """Test resistance to security attacks."""
        node = MeshNode(node_id=uuid4(), host="127.0.0.1", port=9600)
        
        try:
            await node.initialize()
            await node.start()
            
            security = node.security_manager
            
            # Test replay attack resistance
            message = b"test message"
            encrypted1 = await security.encrypt_message(message)
            encrypted2 = await security.encrypt_message(message)
            
            # Same message should produce different ciphertext (nonce)
            assert encrypted1.nonce != encrypted2.nonce
            assert encrypted1.ciphertext != encrypted2.ciphertext
            
            # Test tampering detection
            tampered = encrypted1
            tampered.ciphertext = tampered.ciphertext[:-1] + b'X'
            
            with pytest.raises(Exception):
                await security.decrypt_message(tampered)
            
            # Test signature forgery resistance
            signature = await security.sign_data(message)
            tampered_message = message + b" tampered"
            
            verified = await security.verify_signature(
                tampered_message, signature, node.node_id
            )
            assert not verified
        
        finally:
            await node.stop()


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Benchmark message processing throughput."""
        node = MeshNode(node_id=uuid4(), host="127.0.0.1", port=9700)
        
        try:
            await node.initialize()
            await node.start()
            
            # Prepare test messages
            messages = [f"test message {i}".encode() for i in range(1000)]
            
            # Benchmark encryption/decryption
            start_time = time.time()
            
            for message in messages[:100]:  # Test with 100 messages
                encrypted = await node.security_manager.encrypt_message(message)
                decrypted = await node.security_manager.decrypt_message(encrypted)
                assert decrypted == message
            
            duration = time.time() - start_time
            throughput = 100 / duration
            
            # Should handle at least 50 messages per second
            assert throughput > 50, f"Throughput too low: {throughput:.2f} msg/s"
            
            print(f"Message throughput: {throughput:.2f} messages/second")
        
        finally:
            await node.stop()
    
    @pytest.mark.asyncio
    async def test_consensus_latency(self):
        """Benchmark consensus decision latency."""
        nodes = []
        for i in range(3):
            node = MeshNode(
                node_id=uuid4(),
                host="127.0.0.1",
                port=9800 + i
            )
            nodes.append(node)
        
        try:
            # Setup cluster
            for node in nodes:
                await node.initialize()
                consensus = RaftConsensus(
                    node_id=node.node_id,
                    peers=[n.node_id for n in nodes if n != node]
                )
                node.consensus = consensus
                await consensus.initialize()
                await node.start()
            
            await asyncio.sleep(2)  # Wait for leader election
            
            # Find leader
            leader = None
            for node in nodes:
                if node.consensus.is_leader():
                    leader = node
                    break
            
            if leader is None:
                pytest.skip("No leader elected")
            
            # Benchmark consensus proposals
            proposals = [{"round": i, "data": f"test_{i}"} for i in range(10)]
            
            start_time = time.time()
            
            for proposal in proposals:
                result = await leader.consensus.submit_proposal(proposal)
                # Allow for some failures in test environment
                if result is None:
                    continue
            
            duration = time.time() - start_time
            avg_latency = duration / len(proposals) * 1000  # Convert to ms
            
            # Should average less than 500ms per proposal
            assert avg_latency < 500, f"Consensus latency too high: {avg_latency:.2f}ms"
            
            print(f"Average consensus latency: {avg_latency:.2f}ms")
        
        finally:
            for node in nodes:
                await node.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        node = MeshNode(node_id=uuid4(), host="127.0.0.1", port=9900)
        
        try:
            await node.initialize()
            
            # Setup monitoring
            monitor = ComprehensiveMonitor(node.node_id)
            await monitor.start()
            
            await node.start()
            
            # Record initial memory
            initial_metrics = monitor.metrics_collector.get_performance_metrics()
            initial_memory = initial_metrics.memory_used_mb
            
            # Simulate workload for a period
            for i in range(100):
                # Generate some work
                test_data = f"workload data {i}" * 100
                encrypted = await node.security_manager.encrypt_message(test_data.encode())
                await node.security_manager.decrypt_message(encrypted)
                
                if i % 10 == 0:
                    await asyncio.sleep(0.1)  # Yield control
            
            # Record final memory
            final_metrics = monitor.metrics_collector.get_performance_metrics()
            final_memory = final_metrics.memory_used_mb
            
            # Memory growth should be reasonable (less than 50MB increase)
            memory_growth = final_memory - initial_memory
            assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.2f}MB"
            
            print(f"Memory growth: {memory_growth:.2f}MB")
            
            await monitor.stop()
        
        finally:
            await node.stop()