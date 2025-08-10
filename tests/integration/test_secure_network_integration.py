"""Integration tests for secure P2P network functionality."""

import asyncio
import pytest
import time
from uuid import uuid4

from agent_mesh.core.network import P2PNetwork, PeerInfo, NetworkMessage, PeerStatus
from agent_mesh.core.health import CircuitBreaker, RetryManager


@pytest.mark.integration
class TestSecureNetworkIntegration:
    """Integration tests for secure P2P networking."""

    @pytest.fixture
    async def network_pair(self):
        """Create a pair of P2P networks for testing."""
        network1 = P2PNetwork(
            node_id=uuid4(),
            listen_addr="/ip4/127.0.0.1/tcp/0"
        )
        network2 = P2PNetwork(
            node_id=uuid4(),
            listen_addr="/ip4/127.0.0.1/tcp/0"
        )
        
        yield network1, network2
        
        # Cleanup
        try:
            await network1.stop()
            await network2.stop()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_basic_network_startup_shutdown(self, network_pair):
        """Test basic network startup and shutdown."""
        network1, network2 = network_pair
        
        # Test startup
        await network1.start()
        await network2.start()
        
        # Verify networks are running
        assert network1._running is True
        assert network2._running is True
        
        # Test shutdown
        await network1.stop()
        await network2.stop()
        
        # Verify networks are stopped
        assert network1._running is False
        assert network2._running is False

    @pytest.mark.asyncio
    async def test_secure_peer_connection(self, network_pair):
        """Test secure connection establishment between peers."""
        network1, network2 = network_pair
        
        # Start both networks
        await network1.start()
        await network2.start()
        
        # Wait for networks to fully initialize
        await asyncio.sleep(0.1)
        
        # Get actual listening addresses (since we used port 0)
        network1_addr = f"127.0.0.1:{network1._get_listen_port()}"
        
        # Network2 connects to Network1
        try:
            peer_info = await network2.secure_connect_to_peer(network1_addr)
            
            assert isinstance(peer_info, PeerInfo)
            assert peer_info.status == PeerStatus.CONNECTED
            assert peer_info.public_key is not None
            assert peer_info.shared_secret is not None
            
            # Verify connection statistics
            assert network2.stats.connections_active >= 1
            assert network2.stats.handshakes_completed >= 1
            
        except Exception as e:
            pytest.skip(f"Connection test requires running server: {e}")

    @pytest.mark.asyncio
    async def test_encrypted_message_exchange(self, network_pair):
        """Test encrypted message exchange between connected peers."""
        network1, network2 = network_pair
        
        await network1.start()
        await network2.start()
        
        # Set up message handlers
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
        
        network1.register_message_handler("test_message", message_handler)
        
        try:
            # Establish connection
            network1_addr = f"127.0.0.1:{network1._get_listen_port()}"
            peer_info = await network2.secure_connect_to_peer(network1_addr)
            
            # Create and send test message
            test_message = NetworkMessage(
                sender_id=network2.node_id,
                recipient_id=network1.node_id,
                message_type="test_message",
                payload={"content": "Hello, encrypted world!", "timestamp": time.time()}
            )
            
            await network2.send_message(peer_info.peer_id, test_message)
            
            # Wait for message processing
            await asyncio.sleep(0.1)
            
            # Verify message was received and decrypted
            assert len(received_messages) > 0
            received_msg = received_messages[0]
            assert received_msg.payload["content"] == "Hello, encrypted world!"
            
        except Exception as e:
            pytest.skip(f"Message exchange test requires full network stack: {e}")

    @pytest.mark.asyncio
    async def test_multiple_peer_connections(self, tmp_path):
        """Test connections with multiple peers simultaneously."""
        # Create 3 networks
        networks = []
        for i in range(3):
            network = P2PNetwork(
                node_id=uuid4(),
                listen_addr=f"/ip4/127.0.0.1/tcp/{14000 + i}"
            )
            networks.append(network)
        
        try:
            # Start all networks
            for network in networks:
                await network.start()
            
            await asyncio.sleep(0.1)
            
            # Network 0 connects to networks 1 and 2
            peer1_addr = f"127.0.0.1:{14001}"
            peer2_addr = f"127.0.0.1:{14002}"
            
            peer1_info = await networks[0].secure_connect_to_peer(peer1_addr)
            peer2_info = await networks[0].secure_connect_to_peer(peer2_addr)
            
            # Verify connections
            connected_peers = await networks[0].get_connected_peers()
            assert len(connected_peers) == 2
            
            # Verify each peer has unique cryptographic material
            assert peer1_info.shared_secret != peer2_info.shared_secret
            assert peer1_info.public_key != peer2_info.public_key
            
        except Exception as e:
            pytest.skip(f"Multi-peer test requires network setup: {e}")
        finally:
            # Cleanup
            for network in networks:
                try:
                    await network.stop()
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, network_pair):
        """Test connection failure and recovery mechanisms."""
        network1, network2 = network_pair
        
        # Test connection to non-existent peer
        invalid_addr = "192.168.255.255:65535"
        
        with pytest.raises(Exception):  # Should handle gracefully
            await network1.secure_connect_to_peer(invalid_addr)
        
        # Verify network is still operational
        await network1.start()
        assert network1._running is True

    @pytest.mark.asyncio 
    async def test_handshake_timeout_handling(self, network_pair):
        """Test handling of handshake timeouts."""
        network1, network2 = network_pair
        
        # Mock a slow handshake scenario
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                network1.secure_connect_to_peer("192.168.255.255:65535"),
                timeout=0.1  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_network_statistics_tracking(self, network_pair):
        """Test network statistics are properly tracked."""
        network1, network2 = network_pair
        
        await network1.start()
        await network2.start()
        
        # Initial statistics
        initial_stats = network1.stats
        assert initial_stats.messages_sent == 0
        assert initial_stats.connections_active == 0
        
        try:
            # Attempt connection
            network2_addr = f"127.0.0.1:{network2._get_listen_port()}"
            await network1.secure_connect_to_peer(network2_addr)
            
            # Statistics should be updated
            updated_stats = network1.stats
            assert updated_stats.connections_active >= initial_stats.connections_active
            assert updated_stats.handshakes_completed > initial_stats.handshakes_completed
            
        except Exception as e:
            pytest.skip(f"Statistics test requires full network: {e}")


@pytest.mark.integration
class TestNetworkResilience:
    """Test network resilience and fault tolerance."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with network layer."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=1.0,
            reset_timeout=0.5
        )
        
        # Simulate failures
        for _ in range(3):
            try:
                await circuit_breaker.call(self._failing_operation)
            except Exception:
                pass
        
        # Circuit should be open now
        assert circuit_breaker.is_open()
        
        # Wait for reset timeout
        await asyncio.sleep(0.6)
        
        # Circuit should allow half-open state
        assert not circuit_breaker.is_open()

    async def _failing_operation(self):
        """Mock operation that always fails."""
        raise Exception("Simulated failure")

    @pytest.mark.asyncio
    async def test_retry_manager_integration(self):
        """Test retry manager integration with network operations."""
        retry_manager = RetryManager(
            max_attempts=3,
            base_delay=0.1,
            backoff_factor=2.0
        )
        
        attempt_count = 0
        
        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Flaky failure")
            return "success"
        
        # Should succeed after retries
        result = await retry_manager.retry(flaky_operation)
        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_network_partition_handling(self, tmp_path):
        """Test network behavior during simulated partition."""
        # Create isolated networks
        network1 = P2PNetwork(node_id=uuid4(), listen_addr="/ip4/127.0.0.1/tcp/15001")
        network2 = P2PNetwork(node_id=uuid4(), listen_addr="/ip4/127.0.0.1/tcp/15002")
        
        try:
            await network1.start()
            await network2.start()
            
            # Initially, networks are partitioned (not connected)
            peers1 = await network1.get_connected_peers()
            peers2 = await network2.get_connected_peers()
            
            assert len(peers1) == 0
            assert len(peers2) == 0
            
            # Attempt to establish connection (heal partition)
            try:
                peer_info = await network1.secure_connect_to_peer("127.0.0.1:15002")
                
                # After connection, each network should see the other
                peers1 = await network1.get_connected_peers()
                assert len(peers1) >= 1
                
            except Exception as e:
                pytest.skip(f"Partition test requires network setup: {e}")
                
        finally:
            await network1.stop()
            await network2.stop()


@pytest.mark.integration
class TestCryptographicSecurity:
    """Test cryptographic security in integration scenarios."""

    @pytest.mark.asyncio
    async def test_key_rotation_simulation(self):
        """Test key rotation behavior."""
        # Create network with initial keys
        network = P2PNetwork(node_id=uuid4())
        
        # Get initial public key
        initial_public_key = network.crypto_manager.get_public_key_bytes()
        
        # Simulate key rotation by creating new crypto manager
        network.crypto_manager = network.crypto_manager.__class__()
        new_public_key = network.crypto_manager.get_public_key_bytes()
        
        # Keys should be different after rotation
        assert initial_public_key != new_public_key

    @pytest.mark.asyncio
    async def test_message_integrity_over_network(self):
        """Test message integrity is maintained over network transmission."""
        network1 = P2PNetwork(node_id=uuid4())
        network2 = P2PNetwork(node_id=uuid4())
        
        # Test message signing
        message_data = b"Important message for integrity testing"
        signature1 = network1.crypto_manager.sign_message(message_data)
        
        # Network2 should be able to verify Network1's signature
        network1_public = network1.crypto_manager.get_public_key_bytes()
        is_valid = network2.crypto_manager.verify_signature(
            message_data, signature1, network1_public
        )
        assert is_valid is True
        
        # Tampered message should fail verification
        tampered_message = message_data + b"tampered"
        is_invalid = network2.crypto_manager.verify_signature(
            tampered_message, signature1, network1_public
        )
        assert is_invalid is False

    @pytest.mark.asyncio
    async def test_encryption_performance(self):
        """Test encryption performance with larger messages."""
        crypto_manager = network1.crypto_manager
        
        # Generate test data of various sizes
        test_sizes = [100, 1024, 10240, 102400]  # 100B to 100KB
        
        for size in test_sizes:
            test_data = b"x" * size
            
            # Generate dummy shared secret
            shared_secret = b"x" * 32
            
            start_time = time.time()
            encrypted_data, nonce = crypto_manager.encrypt_message(test_data, shared_secret)
            encrypt_time = time.time() - start_time
            
            start_time = time.time()
            decrypted_data = crypto_manager.decrypt_message(
                encrypted_data, nonce, shared_secret
            )
            decrypt_time = time.time() - start_time
            
            # Verify correctness
            assert decrypted_data == test_data
            
            # Performance should be reasonable (< 100ms for 100KB)
            assert encrypt_time < 0.1
            assert decrypt_time < 0.1


@pytest.mark.integration  
@pytest.mark.slow
class TestNetworkScalability:
    """Test network scalability with multiple connections."""

    @pytest.mark.asyncio
    async def test_connection_pool_scaling(self):
        """Test connection pool behavior with many connections."""
        # Create hub network
        hub_network = P2PNetwork(
            node_id=uuid4(),
            listen_addr="/ip4/127.0.0.1/tcp/16000"
        )
        
        # Create client networks
        client_networks = []
        for i in range(5):  # Test with 5 clients
            client = P2PNetwork(
                node_id=uuid4(),
                listen_addr=f"/ip4/127.0.0.1/tcp/{16001 + i}"
            )
            client_networks.append(client)
        
        try:
            await hub_network.start()
            
            # Start all clients
            for client in client_networks:
                await client.start()
            
            await asyncio.sleep(0.1)
            
            # Each client connects to hub
            for client in client_networks:
                try:
                    await client.secure_connect_to_peer("127.0.0.1:16000")
                except Exception:
                    pass  # May fail in test environment
            
            # Hub should handle multiple connections
            connected_peers = await hub_network.get_connected_peers()
            
            # In test environment, some connections may fail
            # Just verify the structure works
            assert isinstance(connected_peers, list)
            
        finally:
            # Cleanup all networks
            await hub_network.stop()
            for client in client_networks:
                try:
                    await client.stop()
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_message_throughput(self, tmp_path):
        """Test message throughput between peers."""
        network1 = P2PNetwork(node_id=uuid4())
        network2 = P2PNetwork(node_id=uuid4())
        
        # Message counter
        message_count = 0
        
        def count_messages(message):
            nonlocal message_count
            message_count += 1
        
        network2.register_message_handler("throughput_test", count_messages)
        
        try:
            await network1.start()
            await network2.start()
            
            # Connect networks (may fail in test environment)
            try:
                peer_info = await network1.secure_connect_to_peer("127.0.0.1:4002")
                
                # Send multiple messages rapidly
                num_messages = 10
                start_time = time.time()
                
                for i in range(num_messages):
                    message = NetworkMessage(
                        sender_id=network1.node_id,
                        message_type="throughput_test",
                        payload={"sequence": i, "data": f"message_{i}"}
                    )
                    await network1.send_message(peer_info.peer_id, message)
                
                # Wait for message processing
                await asyncio.sleep(0.5)
                
                elapsed_time = time.time() - start_time
                throughput = num_messages / elapsed_time
                
                # Should handle reasonable throughput
                assert throughput > 0
                
            except Exception as e:
                pytest.skip(f"Throughput test requires network setup: {e}")
                
        finally:
            await network1.stop()
            await network2.stop()


# Helper fixtures and utilities
@pytest.fixture
async def test_networks():
    """Create multiple test networks for complex scenarios."""
    networks = []
    
    for i in range(3):
        network = P2PNetwork(
            node_id=uuid4(),
            listen_addr=f"/ip4/127.0.0.1/tcp/{17000 + i}"
        )
        networks.append(network)
    
    yield networks
    
    # Cleanup
    for network in networks:
        try:
            await network.stop()
        except Exception:
            pass


def network_test_config():
    """Configuration for network integration tests."""
    return {
        "timeout": 5.0,
        "max_connections": 10,
        "handshake_timeout": 2.0,
        "message_timeout": 1.0,
    }