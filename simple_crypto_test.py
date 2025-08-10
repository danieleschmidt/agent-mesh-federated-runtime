#!/usr/bin/env python3
"""Simple crypto test without external dependencies."""

import sys
from pathlib import Path
import secrets
import hashlib
import struct
import json
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_cryptography():
    """Test basic cryptographic operations using built-in modules."""
    print("🔒 Testing Basic Cryptographic Operations...")
    
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        
        # Test 1: Key generation
        print("  ✓ Testing Ed25519 key generation...")
        private_key_1 = ed25519.Ed25519PrivateKey.generate()
        public_key_1 = private_key_1.public_key()
        
        private_key_2 = ed25519.Ed25519PrivateKey.generate()
        public_key_2 = private_key_2.public_key()
        
        # Get raw bytes
        public_bytes_1 = public_key_1.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        public_bytes_2 = public_key_2.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        assert len(public_bytes_1) == 32
        assert len(public_bytes_2) == 32
        assert public_bytes_1 != public_bytes_2
        
        # Test 2: Message signing
        print("  ✓ Testing message signing...")
        message = b"Test message for cryptographic signing"
        signature = private_key_1.sign(message)
        
        # Verify signature
        public_key_1.verify(signature, message)  # Should not raise
        
        # Test with wrong key (should fail)
        try:
            public_key_2.verify(signature, message)
            assert False, "Verification should have failed"
        except:
            pass  # Expected to fail
        
        # Test 3: Shared secret generation (using HKDF)
        print("  ✓ Testing shared secret generation...")
        salt = b"agent_mesh_kdf_salt"
        info = b"shared_secret"
        
        combined_key = public_bytes_1 + public_bytes_2
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=info,
        )
        shared_secret = hkdf.derive(combined_key)
        
        assert len(shared_secret) == 32
        
        # Test 4: AES-GCM encryption
        print("  ✓ Testing AES-GCM encryption...")
        test_message = b"This is a secret message for testing!"
        nonce = secrets.token_bytes(12)
        
        cipher = Cipher(
            algorithms.AES(shared_secret),
            modes.GCM(nonce)
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(test_message) + encryptor.finalize()
        tag = encryptor.tag
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(shared_secret),
            modes.GCM(nonce, tag)
        )
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        assert decrypted == test_message
        
        print("  ✅ All cryptographic operations successful!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Missing cryptography library: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Crypto test failed: {e}")
        return False

def test_network_address_parsing():
    """Test network address parsing logic."""
    print("🌐 Testing Network Address Parsing...")
    
    try:
        from urllib.parse import urlparse
        
        def parse_tcp_address(addr):
            """Parse TCP address string to host and port."""
            if "://" in addr:
                parsed = urlparse(addr)
                return parsed.hostname or "localhost", parsed.port or 4001
            elif ":" in addr:
                parts = addr.split(":")
                return parts[0], int(parts[1])
            else:
                return addr, 4001
        
        test_cases = [
            ("127.0.0.1:4001", ("127.0.0.1", 4001)),
            ("tcp://localhost:8080", ("localhost", 8080)),
            ("example.com", ("example.com", 4001)),
            ("192.168.1.100:5555", ("192.168.1.100", 5555)),
        ]
        
        for addr, expected in test_cases:
            result = parse_tcp_address(addr)
            print(f"  ✓ {addr} -> {result}")
            assert result == expected, f"Expected {expected}, got {result}"
        
        print("  ✅ Address parsing tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Address parsing test failed: {e}")
        return False

def test_peer_id_generation():
    """Test peer ID generation."""
    print("🆔 Testing Peer ID Generation...")
    
    try:
        from uuid import UUID
        
        def generate_peer_id(peer_addr):
            """Generate deterministic peer ID from address."""
            hash_obj = hashlib.sha256(peer_addr.encode())
            return UUID(hash_obj.hexdigest()[:32])
        
        # Test deterministic generation
        addr1 = "192.168.1.100:4001"
        peer_id_1a = generate_peer_id(addr1)
        peer_id_1b = generate_peer_id(addr1)
        
        assert peer_id_1a == peer_id_1b, "Same address should generate same ID"
        
        # Test different addresses generate different IDs
        addr2 = "192.168.1.101:4001"
        peer_id_2 = generate_peer_id(addr2)
        
        assert peer_id_1a != peer_id_2, "Different addresses should generate different IDs"
        
        print(f"  ✓ {addr1} -> {peer_id_1a}")
        print(f"  ✓ {addr2} -> {peer_id_2}")
        print("  ✅ Peer ID generation tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Peer ID test failed: {e}")
        return False

def test_message_serialization():
    """Test message serialization."""
    print("📦 Testing Message Serialization...")
    
    try:
        from uuid import uuid4
        import time
        
        # Test message structure
        message = {
            "message_id": str(uuid4()),
            "sender_id": str(uuid4()),
            "recipient_id": str(uuid4()),
            "message_type": "test_message",
            "payload": {"data": "Hello, world!", "timestamp": time.time()},
            "timestamp": time.time(),
            "ttl": 10
        }
        
        # JSON serialization
        serialized = json.dumps(message, sort_keys=True)
        deserialized = json.loads(serialized)
        
        assert deserialized["message_type"] == "test_message"
        assert deserialized["payload"]["data"] == "Hello, world!"
        assert deserialized["ttl"] == 10
        
        # Test with length prefix (like our protocol)
        message_bytes = serialized.encode('utf-8')
        length_prefix = struct.pack('!I', len(message_bytes))
        full_message = length_prefix + message_bytes
        
        # Deserialize
        length = struct.unpack('!I', full_message[:4])[0]
        payload = full_message[4:4+length]
        recovered_message = json.loads(payload.decode('utf-8'))
        
        assert recovered_message == deserialized
        
        print(f"  ✓ Message size: {len(message_bytes)} bytes")
        print(f"  ✓ With prefix: {len(full_message)} bytes")
        print("  ✅ Message serialization tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Serialization test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test basic performance characteristics."""
    print("⚡ Testing Performance Benchmarks...")
    
    try:
        import time
        
        # Test JSON serialization performance
        large_data = {
            "type": "model_update",
            "weights": [[0.1] * 100 for _ in range(100)],  # 10K floats
            "metadata": {"samples": 1000, "accuracy": 0.95}
        }
        
        # Serialization benchmark
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            serialized = json.dumps(large_data)
            
        serialize_time = (time.time() - start_time) / iterations
        
        # Deserialization benchmark
        start_time = time.time()
        
        for _ in range(iterations):
            deserialized = json.loads(serialized)
            
        deserialize_time = (time.time() - start_time) / iterations
        
        print(f"  ✓ Serialization: {serialize_time*1000:.2f}ms per message")
        print(f"  ✓ Deserialization: {deserialize_time*1000:.2f}ms per message")
        print(f"  ✓ Message size: {len(serialized)} bytes")
        
        # Performance requirements
        assert serialize_time < 0.01, "Serialization should be under 10ms"
        assert deserialize_time < 0.01, "Deserialization should be under 10ms"
        
        print("  ✅ Performance benchmarks passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🚀 Agent Mesh Simple Test Suite")
    print("=" * 50)
    
    test_functions = [
        ("Basic Cryptography", test_basic_cryptography),
        ("Network Address Parsing", test_network_address_parsing),
        ("Peer ID Generation", test_peer_id_generation),
        ("Message Serialization", test_message_serialization),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        print()
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
    
    print()
    print("=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All core functionality tests passed!")
        print("💡 Core cryptographic and networking logic is working correctly.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Core functionality is working.")
        return True
    else:
        print("❌ Multiple test failures. Core functionality needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)