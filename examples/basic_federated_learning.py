#!/usr/bin/env python3
"""Basic federated learning example using Agent Mesh.

This example demonstrates how to set up a simple federated learning network
with multiple nodes training a model collaboratively.
"""

import asyncio
import logging
import random
from typing import Any
from uuid import uuid4

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agent_mesh.core.mesh_node import MeshNode, NodeCapabilities
from agent_mesh.federated.learner import FederatedLearner, FederatedConfig


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_synthetic_dataset(num_samples=1000, input_size=10):
    """Create synthetic dataset for training."""
    # Generate random data
    X = torch.randn(num_samples, input_size)
    # Simple linear relationship with noise
    weights = torch.randn(input_size)
    y = torch.matmul(X, weights) + 0.1 * torch.randn(num_samples)
    
    return TensorDataset(X, y.unsqueeze(1))


class FederatedLearningDemo:
    """Demonstration of federated learning with Agent Mesh."""
    
    def __init__(self, num_nodes=3):
        self.num_nodes = num_nodes
        self.nodes = []
        self.learners = []
        
    async def setup_nodes(self):
        """Set up mesh nodes for federated learning."""
        print(f"Setting up {self.num_nodes} nodes for federated learning...")
        
        for i in range(self.num_nodes):
            # Create node with capabilities
            capabilities = NodeCapabilities(
                cpu_cores=2 + random.randint(0, 4),
                memory_gb=4 + random.randint(0, 8),
                gpu_available=random.choice([True, False]),
                skills={"machine_learning", "federated_learning"}
            )
            
            # Create mesh node
            node = MeshNode(
                node_id=uuid4(),
                listen_addr=f"/ip4/127.0.0.1/tcp/{4000 + i}",
                capabilities=capabilities
            )\n            \n            # Model and dataset factories\n            def model_fn():\n                return SimpleModel()\n            \n            def dataset_fn():\n                # Each node gets different data (simulating distributed data)\n                return create_synthetic_dataset(200 + random.randint(0, 300))\n            \n            # Create federated learner\n            config = FederatedConfig(\n                rounds=10,\n                local_epochs=3,\n                batch_size=16,\n                learning_rate=0.01,\n                min_participants=2,\n                aggregation_algorithm=\"fedavg\"\n            )\n            \n            learner = FederatedLearner(\n                node_id=node.node_id,\n                model_fn=model_fn,\n                dataset_fn=dataset_fn,\n                config=config,\n                network_manager=node.network\n            )\n            \n            self.nodes.append(node)\n            self.learners.append(learner)\n        \n        print(f\"Created {len(self.nodes)} nodes with federated learning capability\")\n        \n    async def start_network(self):\n        \"\"\"Start all nodes and connect them in a network.\"\"\"\n        print(\"Starting mesh network...\")\n        \n        # Start all nodes\n        for node in self.nodes:\n            await node.start()\n            print(f\"Node {node.node_id} started on {await node.network.get_listen_address()}\")\n        \n        # Connect nodes to each other (create mesh topology)\n        for i, node in enumerate(self.nodes):\n            # Connect to previous nodes to form mesh\n            for j in range(i):\n                if j != i:\n                    try:\n                        peer_addr = await self.nodes[j].network.get_listen_address()\n                        await node.join_network([peer_addr])\n                        print(f\"Node {i} connected to Node {j}\")\n                    except Exception as e:\n                        print(f\"Failed to connect Node {i} to Node {j}: {e}\")\n        \n        # Wait for network stabilization\n        await asyncio.sleep(2)\n        \n        print(\"Mesh network established\")\n        \n    async def run_federated_learning(self):\n        \"\"\"Run federated learning across the network.\"\"\"\n        print(\"Starting federated learning...\")\n        \n        # Start all learners\n        learning_tasks = []\n        for learner in self.learners:\n            task = asyncio.create_task(learner.start_training())\n            learning_tasks.append(task)\n        \n        # Wait for training to complete (or timeout)\n        try:\n            await asyncio.wait_for(\n                asyncio.gather(*learning_tasks, return_exceptions=True),\n                timeout=300  # 5 minutes timeout\n            )\n        except asyncio.TimeoutError:\n            print(\"Training timeout - stopping...\")\n            for learner in self.learners:\n                await learner.stop_training()\n        \n        print(\"Federated learning completed\")\n        \n    async def display_results(self):\n        \"\"\"Display training results from all nodes.\"\"\"\n        print(\"\\n=== FEDERATED LEARNING RESULTS ===\")\n        \n        for i, learner in enumerate(self.learners):\n            metrics = learner.get_training_metrics()\n            global_model = learner.get_global_model()\n            \n            print(f\"\\nNode {i} (ID: {learner.node_id}):\") \n            print(f\"  - Training rounds completed: {len(metrics)}\")\n            print(f\"  - Training role: {'Coordinator' if learner._is_coordinator else 'Participant'}\")\n            \n            if metrics:\n                final_metrics = metrics[-1]\n                print(f\"  - Final loss: {final_metrics.loss:.4f}\")\n                print(f\"  - Participants in final round: {final_metrics.participants_count}\")\n                print(f\"  - Total communication cost: {sum(m.communication_cost_mb for m in metrics):.2f} MB\")\n            \n            if global_model:\n                # Count model parameters\n                num_params = sum(p.numel() for p in global_model.parameters())\n                print(f\"  - Global model parameters: {num_params:,}\")\n        \n        # Display aggregated results\n        if self.learners and self.learners[0].get_training_metrics():\n            coordinator_metrics = self.learners[0].get_training_metrics()\n            print(f\"\\n=== AGGREGATED RESULTS ===\")\n            print(f\"Total training rounds: {len(coordinator_metrics)}\")\n            \n            if coordinator_metrics:\n                print(\"\\nLoss progression:\")\n                for i, metrics in enumerate(coordinator_metrics[:5]):  # Show first 5 rounds\n                    print(f\"  Round {metrics.round_number}: loss={metrics.loss:.4f}, participants={metrics.participants_count}\")\n                \n                if len(coordinator_metrics) > 5:\n                    print(f\"  ... (and {len(coordinator_metrics) - 5} more rounds)\")\n                    final = coordinator_metrics[-1]\n                    print(f\"  Round {final.round_number}: loss={final.loss:.4f}, participants={final.participants_count}\")\n        \n    async def cleanup(self):\n        \"\"\"Clean up resources.\"\"\"\n        print(\"\\nCleaning up...\")\n        \n        # Stop all learners\n        for learner in self.learners:\n            await learner.stop_training()\n        \n        # Stop all nodes\n        for node in self.nodes:\n            await node.stop()\n        \n        print(\"Cleanup completed\")\n        \n    async def run_demo(self):\n        \"\"\"Run the complete federated learning demonstration.\"\"\"\n        try:\n            await self.setup_nodes()\n            await self.start_network()\n            await self.run_federated_learning()\n            await self.display_results()\n        finally:\n            await self.cleanup()\n\n\nasync def main():\n    \"\"\"Main demonstration function.\"\"\"\n    print(\"🚀 Agent Mesh Federated Learning Demo\")\n    print(\"======================================\")\n    \n    # Set up logging\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n    )\n    \n    # Run demonstration\n    demo = FederatedLearningDemo(num_nodes=3)\n    await demo.run_demo()\n    \n    print(\"\\n✅ Demonstration completed successfully!\")\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())\n"