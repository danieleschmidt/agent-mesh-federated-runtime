#!/usr/bin/env python3
"""Simple test to verify core functionality."""

import sys
import os
import asyncio
from uuid import uuid4

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent_mesh.core.mesh_node import MeshNode, NodeCapabilities
from agent_mesh.core.security import SecurityManager
from agent_mesh.federated.algorithms import FedAvgAlgorithm, FedAvgConfig
from agent_mesh.coordination.task_scheduler import TaskScheduler, Task, TaskPriority

async def test_core_functionality():
    """Test basic functionality of core components."""
    print("🧪 Testing Agent Mesh Core Functionality")
    
    # Test 1: Node Creation and Initialization
    print("\n1. Testing Node Creation...")
    try:
        node_id = uuid4()
        capabilities = NodeCapabilities(
            cpu_cores=4,
            memory_gb=8.0,
            skills={"ml", "data_processing"}
        )
        
        node = MeshNode(node_id, capabilities)
        await node.start()
        print("✅ Node creation and initialization successful")
        
        # Test 2: Security Manager
        print("\n2. Testing Security Manager...")
        security_manager = SecurityManager(node_id)
        await security_manager.initialize()
        
        identity = await security_manager.get_node_identity()
        print(f"✅ Security manager initialized with identity: {str(identity.node_id)[:8]}...")
        
        # Test 3: FedAvg Algorithm
        print("\n3. Testing FedAvg Algorithm...")
        config = FedAvgConfig(learning_rate=0.01)
        fedavg = FedAvgAlgorithm(config)
        print("✅ FedAvg algorithm created successfully")
        
        # Test 4: Task Scheduler
        print("\n4. Testing Task Scheduler...")
        scheduler = TaskScheduler(node_id)
        await scheduler.start()
        
        # Create a test task
        task = Task(
            name="test_task",
            description="A simple test task",
            priority=TaskPriority.NORMAL,
            required_skills={"test"}
        )
        
        task_id = await scheduler.submit_task(task)
        status = await scheduler.get_task_status(task_id)
        print(f"✅ Task scheduler working - submitted task with status: {status}")
        
        await scheduler.stop()
        await node.stop()
        await security_manager.cleanup()
        
        print("\n🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_core_functionality())
    sys.exit(0 if success else 1)