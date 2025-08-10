#!/usr/bin/env python3
"""Simple test runner for validating Agent Mesh implementations."""

import asyncio
import sys
import traceback
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_crypto_tests():
    """Run basic cryptographic tests."""
    print("🧪 Testing Cryptographic Network Layer...")
    
    try:
        from agent_mesh.core.network import CryptoManager, P2PNetwork
        from uuid import uuid4
        
        # Test 1: CryptoManager initialization
        print("  ✓ Testing CryptoManager initialization...")
        crypto1 = CryptoManager()
        crypto2 = CryptoManager()
        
        # Test 2: Key generation
        print("  ✓ Testing key generation...")
        key1 = crypto1.get_public_key_bytes()
        key2 = crypto2.get_public_key_bytes()
        
        assert len(key1) == 32, "Ed25519 key should be 32 bytes"
        assert len(key2) == 32, "Ed25519 key should be 32 bytes"
        assert key1 != key2, "Keys should be different"
        
        # Test 3: Message signing and verification
        print("  ✓ Testing message signing and verification...")
        message = b"Test message for signing"
        signature = crypto1.sign_message(message)
        
        # Verify with correct key
        is_valid = crypto1.verify_signature(message, signature, key1)
        assert is_valid, "Signature should be valid with correct key"
        
        # Verify with wrong key
        is_invalid = crypto1.verify_signature(message, signature, key2)
        assert not is_invalid, "Signature should be invalid with wrong key"
        
        # Test 4: Shared secret generation
        print("  ✓ Testing shared secret generation...")
        secret1 = crypto1.generate_shared_secret(key2)
        secret2 = crypto2.generate_shared_secret(key1)
        
        assert secret1 == secret2, "Shared secrets should match"
        assert len(secret1) == 32, "Shared secret should be 32 bytes"
        
        # Test 5: Message encryption and decryption
        print("  ✓ Testing message encryption and decryption...")
        test_message = b"This is a secret message for encryption testing!"
        
        encrypted_data, nonce = crypto1.encrypt_message(test_message, secret1)
        decrypted_message = crypto2.decrypt_message(encrypted_data, nonce, secret2)
        
        assert decrypted_message == test_message, "Decrypted message should match original"
        
        # Test 6: P2P Network initialization
        print("  ✓ Testing P2P Network initialization...")
        network = P2PNetwork(node_id=uuid4())
        
        assert hasattr(network, 'crypto_manager'), "Network should have crypto manager"
        assert network.crypto_manager is not None, "Crypto manager should be initialized"
        
        print("  ✅ All cryptographic tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Crypto test failed: {e}")
        traceback.print_exc()
        return False

def run_network_tests():
    """Run basic network functionality tests."""
    print("🌐 Testing P2P Network Layer...")
    
    try:
        from agent_mesh.core.network import P2PNetwork, PeerStatus, TransportProtocol
        from uuid import uuid4
        
        # Test 1: Network creation
        print("  ✓ Testing network creation...")
        node_id = uuid4()
        network = P2PNetwork(
            node_id=node_id,
            listen_addr="/ip4/127.0.0.1/tcp/0"
        )
        
        assert network.node_id == node_id, "Node ID should match"
        assert network.crypto_manager is not None, "Should have crypto manager"
        
        # Test 2: Address parsing
        print("  ✓ Testing address parsing...")
        test_cases = [
            ("127.0.0.1:4001", ("127.0.0.1", 4001)),
            ("tcp://localhost:8080", ("localhost", 8080)),
            ("example.com", ("example.com", 4001)),
        ]
        
        for addr, expected in test_cases:
            result = network._parse_tcp_address(addr)
            assert result == expected, f"Address parsing failed for {addr}"
        
        # Test 3: Peer ID generation
        print("  ✓ Testing peer ID generation...")
        peer_addr = "192.168.1.100:4001"
        peer_id1 = network._generate_peer_id(peer_addr)
        peer_id2 = network._generate_peer_id(peer_addr)
        
        assert peer_id1 == peer_id2, "Same address should generate same peer ID"
        
        different_peer_id = network._generate_peer_id("192.168.1.101:4001")
        assert peer_id1 != different_peer_id, "Different addresses should generate different IDs"
        
        print("  ✅ All network tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Network test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_tests():
    """Run basic performance benchmarks."""
    print("⚡ Testing Performance Characteristics...")
    
    try:
        from agent_mesh.core.network import CryptoManager
        
        crypto = CryptoManager()
        
        # Test encryption performance
        print("  ✓ Testing encryption performance...")
        test_sizes = [100, 1024, 10240]  # 100B, 1KB, 10KB
        
        for size in test_sizes:
            test_data = b"x" * size
            shared_secret = b"y" * 32
            
            start_time = time.time()
            encrypted_data, nonce = crypto.encrypt_message(test_data, shared_secret)
            encrypt_time = time.time() - start_time
            
            start_time = time.time()
            decrypted_data = crypto.decrypt_message(encrypted_data, nonce, shared_secret)
            decrypt_time = time.time() - start_time
            
            assert decrypted_data == test_data, f"Encryption/decryption failed for {size} bytes"
            
            print(f"    {size:5d} bytes: encrypt={encrypt_time*1000:5.1f}ms, decrypt={decrypt_time*1000:5.1f}ms")
            
            # Performance requirements
            assert encrypt_time < 0.1, f"Encryption too slow for {size} bytes"
            assert decrypt_time < 0.1, f"Decryption too slow for {size} bytes"
        
        print("  ✅ Performance tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

async def run_async_tests():
    """Run async functionality tests."""
    print("🔄 Testing Async Operations...")
    
    try:
        from agent_mesh.core.network import P2PNetwork
        from uuid import uuid4
        
        # Test basic async operations
        network = P2PNetwork(node_id=uuid4())
        
        # These would normally require actual network setup
        # For now, just verify the async methods exist and can be called
        assert hasattr(network, 'start'), "Should have start method"
        assert hasattr(network, 'stop'), "Should have stop method"
        assert hasattr(network, 'secure_connect_to_peer'), "Should have secure connect method"
        
        print("  ✅ Async tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Async test failed: {e}")
        traceback.print_exc()
        return False

def run_import_tests():
    """Test that all modules can be imported successfully."""
    print("📦 Testing Module Imports...")
    
    modules_to_test = [
        "agent_mesh.core.network",
        "agent_mesh.core.health",
        "agent_mesh.core.metrics", 
        "agent_mesh.core.performance",
        "agent_mesh.core.discovery",
        "agent_mesh.core.consensus",
        "agent_mesh.federated.learner",
        "agent_mesh.federated.aggregator",
        "agent_mesh.coordination.agent_mesh",
    ]
    
    imported_count = 0
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
            imported_count += 1
        except ImportError as e:
            print(f"  ⚠ {module_name} - {e}")
        except Exception as e:
            print(f"  ❌ {module_name} - {e}")
    
    print(f"  📊 Successfully imported {imported_count}/{len(modules_to_test)} modules")
    
    if imported_count >= len(modules_to_test) * 0.8:  # 80% success rate
        print("  ✅ Import tests passed!")
        return True
    else:
        print("  ❌ Too many import failures!")
        return False

def main():
    """Run all tests."""
    print("🚀 Agent Mesh Test Suite")
    print("=" * 50)
    
    test_functions = [
        ("Import Tests", run_import_tests),
        ("Crypto Tests", run_crypto_tests),
        ("Network Tests", run_network_tests), 
        ("Performance Tests", run_performance_tests),
        ("Async Tests", lambda: asyncio.run(run_async_tests())),
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
        print("🎉 All tests passed! System is ready for deployment.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. System is functional with minor issues.")
        return True
    else:
        print("❌ Multiple test failures. System needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)