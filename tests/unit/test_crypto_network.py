"""Comprehensive unit tests for cryptographic P2P network layer."""

import asyncio
import pytest
import json
import secrets
import struct
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from agent_mesh.core.network import (
    CryptoManager, P2PNetwork, PeerInfo, NetworkMessage,
    NetworkError, NetworkConnectionError, NetworkProtocolError,
    PeerStatus, TransportProtocol
)


class TestCryptoManager:
    """Test cryptographic operations manager."""

    @pytest.fixture
    def crypto_manager(self):
        """Create crypto manager instance."""
        return CryptoManager()

    @pytest.fixture
    def second_crypto_manager(self):
        """Create second crypto manager for key exchange tests."""
        return CryptoManager()

    @pytest.mark.unit
    def test_crypto_manager_initialization(self, crypto_manager):
        """Test crypto manager initializes with valid keys."""
        # Test that keys are generated
        public_key_bytes = crypto_manager.get_public_key_bytes()
        assert isinstance(public_key_bytes, bytes)
        assert len(public_key_bytes) == 32  # Ed25519 public key size

    @pytest.mark.unit
    def test_message_signing_and_verification(self, crypto_manager, second_crypto_manager):
        """Test message signing and signature verification."""
        message = b"Hello, secure world!"
        
        # Sign message with first manager
        signature = crypto_manager.sign_message(message)
        assert isinstance(signature, bytes)
        assert len(signature) == 64  # Ed25519 signature size
        
        # Verify signature with first manager's public key
        public_key = crypto_manager.get_public_key_bytes()
        is_valid = crypto_manager.verify_signature(message, signature, public_key)
        assert is_valid is True
        
        # Verify that different key can't verify the signature
        other_public_key = second_crypto_manager.get_public_key_bytes()
        is_invalid = crypto_manager.verify_signature(message, signature, other_public_key)
        assert is_invalid is False

    @pytest.mark.unit
    def test_shared_secret_generation(self, crypto_manager, second_crypto_manager):
        """Test shared secret generation between two peers."""
        # Generate shared secrets from both sides
        public_key_1 = crypto_manager.get_public_key_bytes()
        public_key_2 = second_crypto_manager.get_public_key_bytes()
        
        secret_1 = crypto_manager.generate_shared_secret(public_key_2)
        secret_2 = second_crypto_manager.generate_shared_secret(public_key_1)
        
        # Both should generate the same shared secret
        assert secret_1 == secret_2
        assert len(secret_1) == 32  # 256-bit shared secret
        assert isinstance(secret_1, bytes)

    @pytest.mark.unit
    def test_message_encryption_decryption(self, crypto_manager, second_crypto_manager):
        """Test message encryption and decryption."""
        message = b"This is a secret message for testing encryption!"
        
        # Generate shared secret
        public_key_2 = second_crypto_manager.get_public_key_bytes()
        shared_secret = crypto_manager.generate_shared_secret(public_key_2)
        
        # Encrypt message
        encrypted_data, nonce = crypto_manager.encrypt_message(message, shared_secret)
        assert isinstance(encrypted_data, bytes)
        assert isinstance(nonce, bytes)
        assert len(nonce) == 12  # AES-GCM nonce size
        assert encrypted_data != message  # Ensure it's actually encrypted
        
        # Decrypt message
        decrypted_message = crypto_manager.decrypt_message(
            encrypted_data, nonce, shared_secret
        )
        assert decrypted_message == message

    @pytest.mark.unit
    def test_encryption_with_invalid_secret(self, crypto_manager):
        """Test encryption fails with invalid shared secret."""
        message = b"Test message"
        invalid_secret = b"invalid_secret_wrong_length"
        
        with pytest.raises(Exception):  # Should raise due to wrong key length
            crypto_manager.encrypt_message(message, invalid_secret)

    @pytest.mark.unit
    def test_signature_verification_with_corrupted_signature(self, crypto_manager):
        """Test signature verification fails with corrupted signature."""
        message = b"Test message"
        signature = crypto_manager.sign_message(message)
        
        # Corrupt the signature
        corrupted_signature = signature[:-1] + b'\x00'
        
        public_key = crypto_manager.get_public_key_bytes()
        is_valid = crypto_manager.verify_signature(
            message, corrupted_signature, public_key
        )
        assert is_valid is False


class TestP2PNetworkCrypto:
    """Test P2P network with cryptographic enhancements."""

    @pytest.fixture
    def node_id(self):
        """Create test node ID."""
        return uuid4()

    @pytest.fixture
    def p2p_network(self, node_id):
        """Create P2P network instance."""
        return P2PNetwork(
            node_id=node_id,
            listen_addr="/ip4/127.0.0.1/tcp/0"
        )

    @pytest.mark.unit
    def test_p2p_network_initialization(self, p2p_network, node_id):
        """Test P2P network initializes with crypto manager."""
        assert p2p_network.node_id == node_id
        assert hasattr(p2p_network, 'crypto_manager')
        assert isinstance(p2p_network.crypto_manager, CryptoManager)

    @pytest.mark.unit
    def test_generate_peer_id(self, p2p_network):
        """Test peer ID generation from address."""
        peer_addr = "192.168.1.100:4001"
        peer_id = p2p_network._generate_peer_id(peer_addr)
        
        assert isinstance(peer_id, UUID)
        
        # Same address should generate same peer ID
        peer_id_2 = p2p_network._generate_peer_id(peer_addr)
        assert peer_id == peer_id_2
        
        # Different address should generate different peer ID
        different_peer_id = p2p_network._generate_peer_id("192.168.1.101:4001")
        assert peer_id != different_peer_id

    @pytest.mark.unit
    def test_parse_tcp_address(self, p2p_network):
        """Test TCP address parsing."""
        test_cases = [
            ("192.168.1.100:4001", ("192.168.1.100", 4001)),
            ("tcp://localhost:8080", ("localhost", 8080)),
            ("example.com", ("example.com", 4001)),  # Default port
            ("127.0.0.1:5555", ("127.0.0.1", 5555)),
        ]
        
        for addr, expected in test_cases:
            host, port = p2p_network._parse_tcp_address(addr)
            assert (host, port) == expected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_encrypted_message_serialization(self, p2p_network):
        """Test encrypted message serialization and deserialization."""
        # Mock writer for testing
        class MockWriter:
            def __init__(self):
                self.data = bytearray()
                
            def write(self, data):
                self.data.extend(data)
                
            async def drain(self):
                pass
                
            def get_extra_info(self, key):
                if key == 'peername':
                    return ('127.0.0.1', 4001)
                return None
        
        writer = MockWriter()
        message_type = "test_message"
        data = {"content": "Hello, encrypted world!", "timestamp": 12345.67}
        
        # Test unencrypted message
        await p2p_network._send_encrypted_message(
            writer, message_type, data, encrypt=False
        )
        
        # Verify message structure
        assert len(writer.data) > 4  # Should have length prefix
        length_prefix = struct.unpack('!I', writer.data[:4])[0]
        message_bytes = writer.data[4:4+length_prefix]
        message = json.loads(message_bytes.decode('utf-8'))
        
        assert message["type"] == message_type
        assert message["data"] == data
        assert "timestamp" in message

    @pytest.mark.unit
    def test_network_statistics_initialization(self, p2p_network):
        """Test network statistics are properly initialized."""
        stats = p2p_network.stats
        
        assert stats.messages_sent == 0
        assert stats.messages_received == 0
        assert stats.connections_active == 0
        assert stats.handshakes_completed == 0
        assert stats.encryption_enabled is True

    @pytest.mark.unit
    def test_peer_info_with_crypto_fields(self):
        """Test PeerInfo includes cryptographic fields."""
        peer_id = uuid4()
        public_key = secrets.token_bytes(32)
        shared_secret = secrets.token_bytes(32)
        
        peer_info = PeerInfo(
            peer_id=peer_id,
            addresses=["192.168.1.100:4001"],
            protocols={TransportProtocol.TCP},
            status=PeerStatus.CONNECTED,
            public_key=public_key,
            shared_secret=shared_secret
        )
        
        assert peer_info.peer_id == peer_id
        assert peer_info.public_key == public_key
        assert peer_info.shared_secret == shared_secret
        assert peer_info.status == PeerStatus.CONNECTED


class TestNetworkErrorHandling:
    """Test network error handling and edge cases."""

    @pytest.fixture
    def p2p_network(self):
        """Create P2P network for error testing."""
        return P2PNetwork(node_id=uuid4())

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, p2p_network):
        """Test handling of connection failures."""
        # Test connection to non-existent peer
        invalid_peer_addr = "192.168.255.255:65535"
        
        with pytest.raises(NetworkConnectionError):
            await p2p_network.secure_connect_to_peer(invalid_peer_addr)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_receive_message_with_invalid_length(self):
        """Test message receiving with invalid length prefix."""
        # Mock reader that returns invalid data
        class MockInvalidReader:
            async def readexactly(self, n):
                if n == 4:  # Length prefix
                    return struct.pack('!I', 1000000)  # Too large
                else:
                    raise asyncio.IncompleteReadError(b'', n)
        
        p2p_network = P2PNetwork(node_id=uuid4())
        reader = MockInvalidReader()
        
        with pytest.raises(Exception):  # Should raise due to incomplete read
            await p2p_network._receive_message(reader)

    @pytest.mark.unit
    def test_invalid_address_formats(self, p2p_network):
        """Test handling of various invalid address formats."""
        invalid_addresses = [
            "",
            "not_an_address",
            "192.168.1.1",  # Missing port
            "tcp://",
            "192.168.1.1:99999",  # Invalid port
        ]
        
        for addr in invalid_addresses:
            try:
                host, port = p2p_network._parse_tcp_address(addr)
                # Some invalid addresses might still parse with defaults
                assert isinstance(host, str)
                assert isinstance(port, int)
                assert 1 <= port <= 65535
            except (ValueError, AttributeError):
                # Expected for truly invalid addresses
                pass

    @pytest.mark.unit
    def test_crypto_manager_error_handling(self):
        """Test crypto manager error handling."""
        crypto_manager = CryptoManager()
        
        # Test with invalid public key length
        invalid_public_key = b"too_short"
        
        with pytest.raises(Exception):
            crypto_manager.generate_shared_secret(invalid_public_key)
        
        # Test signature verification with invalid key
        message = b"test message"
        signature = crypto_manager.sign_message(message)
        
        is_valid = crypto_manager.verify_signature(
            message, signature, invalid_public_key
        )
        assert is_valid is False  # Should handle gracefully


class TestNetworkMessageHandling:
    """Test network message handling and protocols."""

    @pytest.mark.unit
    def test_network_message_creation(self):
        """Test network message creation and serialization."""
        sender_id = uuid4()
        recipient_id = uuid4()
        
        message = NetworkMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type="test_message",
            payload={"data": "test_payload", "number": 42},
            ttl=5
        )
        
        assert message.sender_id == sender_id
        assert message.recipient_id == recipient_id
        assert message.message_type == "test_message"
        assert message.payload["data"] == "test_payload"
        assert message.ttl == 5
        assert isinstance(message.timestamp, float)

    @pytest.mark.unit
    def test_broadcast_message_creation(self):
        """Test broadcast message creation (no specific recipient)."""
        sender_id = uuid4()
        
        broadcast_msg = NetworkMessage(
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            message_type="broadcast_announcement",
            payload={"announcement": "New node joined the network"}
        )
        
        assert broadcast_msg.recipient_id is None
        assert broadcast_msg.message_type == "broadcast_announcement"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handshake_message_structure(self, p2p_network):
        """Test handshake message structure and content."""
        public_key = p2p_network.crypto_manager.get_public_key_bytes()
        
        # Simulate handshake init message structure
        handshake_data = {
            "node_id": str(p2p_network.node_id),
            "public_key": public_key.hex(),  # Convert to hex for JSON
            "protocol_version": "1.0",
            "capabilities": ["federated_learning", "consensus", "encryption"],
        }
        
        # Validate handshake structure
        assert "node_id" in handshake_data
        assert "public_key" in handshake_data
        assert "protocol_version" in handshake_data
        assert "capabilities" in handshake_data
        assert len(handshake_data["capabilities"]) > 0


@pytest.mark.integration
class TestCryptoIntegration:
    """Integration tests for crypto components."""

    @pytest.mark.asyncio
    async def test_full_cryptographic_handshake_simulation(self):
        """Test full cryptographic handshake between two nodes."""
        # Create two crypto managers (simulating two nodes)
        node1_crypto = CryptoManager()
        node2_crypto = CryptoManager()
        
        # Exchange public keys
        node1_public = node1_crypto.get_public_key_bytes()
        node2_public = node2_crypto.get_public_key_bytes()
        
        # Generate shared secrets
        shared_secret_1 = node1_crypto.generate_shared_secret(node2_public)
        shared_secret_2 = node2_crypto.generate_shared_secret(node1_public)
        
        # Verify shared secrets match
        assert shared_secret_1 == shared_secret_2
        
        # Test encrypted communication
        test_message = b"Secure communication established!"
        
        # Node 1 encrypts and sends to Node 2
        encrypted_data, nonce = node1_crypto.encrypt_message(test_message, shared_secret_1)
        
        # Node 2 receives and decrypts
        decrypted_message = node2_crypto.decrypt_message(
            encrypted_data, nonce, shared_secret_2
        )
        
        assert decrypted_message == test_message

    @pytest.mark.asyncio
    async def test_message_integrity_with_signatures(self):
        """Test message integrity using digital signatures."""
        crypto_manager = CryptoManager()
        
        message_data = {"type": "model_update", "round": 5, "accuracy": 0.95}
        message_bytes = json.dumps(message_data, sort_keys=True).encode('utf-8')
        
        # Sign the message
        signature = crypto_manager.sign_message(message_bytes)
        
        # Verify signature
        public_key = crypto_manager.get_public_key_bytes()
        is_valid = crypto_manager.verify_signature(
            message_bytes, signature, public_key
        )
        assert is_valid is True
        
        # Test with tampered message
        tampered_message = message_bytes + b"tampered"
        is_invalid = crypto_manager.verify_signature(
            tampered_message, signature, public_key
        )
        assert is_invalid is False


# Fixtures for all tests
@pytest.fixture
def mock_mesh_node():
    """Create mock mesh node for testing."""
    mock_node = AsyncMock()
    mock_node.node_id = "test-node-001"
    mock_node.start = AsyncMock()
    mock_node.stop = AsyncMock()
    mock_node.connect_to_peer = AsyncMock()
    mock_node.disconnect_from_peer = AsyncMock()
    mock_node.broadcast_message = AsyncMock()
    return mock_node


@pytest.fixture
def mock_consensus_engine():
    """Create mock consensus engine for testing."""
    mock_consensus = AsyncMock()
    mock_consensus.reach_consensus = AsyncMock()
    mock_consensus.propose = AsyncMock()
    return mock_consensus


@pytest.fixture
def mock_federated_learner():
    """Create mock federated learner for testing."""
    mock_learner = AsyncMock()
    mock_learner.aggregate_updates = AsyncMock()
    mock_learner.train_round = AsyncMock()
    return mock_learner


@pytest.fixture
def sample_model_update():
    """Create sample model update for testing."""
    return {
        "node_id": "test-node-001",
        "round": 1,
        "weights": {"layer1": [[0.1, 0.2]], "layer2": [[0.3, 0.4]]},
        "samples": 100,
        "loss": 0.25,
        "accuracy": 0.87
    }


@pytest.fixture
def mock_crypto_keys():
    """Create mock cryptographic keys for testing."""
    return {
        "private_key": "mock_private_key_data",
        "public_key": "mock_public_key_data", 
        "certificate": "mock_certificate_data"
    }


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "environment": "test",
        "log_level": "INFO",
        "p2p_port_base": 14000,
        "max_peers": 50,
        "consensus_timeout": 5000
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for testing."""
    return tmp_path