"""Command-line interface for Agent Mesh.

This module provides a comprehensive CLI for managing and interacting
with the Agent Mesh federated runtime.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import click
import structlog
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.mesh_node import MeshNode, NodeCapabilities, NodeRole
from .core.network import P2PNetwork
from .core.consensus import ConsensusEngine, ConsensusConfig
from .core.security import SecurityManager
from .federated.learner import FederatedLearner, FederatedConfig
from .coordination.agent_mesh import AgentMesh, CoordinationProtocol


console = Console()


class SimpleModel(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_dataset():
    """Create dummy dataset for testing."""
    # Create synthetic data
    import torch.utils.data as data
    
    class DummyDataset(data.Dataset):
        def __init__(self, size=1000):
            self.size = size
            self.data = torch.randn(size, 1, 28, 28)
            self.targets = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    return DummyDataset()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', default='INFO', help='Set log level')
def cli(verbose: bool, log_level: str):
    """Agent Mesh Federated Runtime CLI."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    if verbose:
        console.print("🚀 Agent Mesh Federated Runtime", style="bold green")


@cli.command()
@click.option('--node-id', help='Node ID (UUID)')
@click.option('--listen-addr', default='/ip4/0.0.0.0/tcp/4001', help='Listen address')
@click.option('--bootstrap-peers', multiple=True, help='Bootstrap peer addresses')
@click.option('--auto-role', is_flag=True, default=True, help='Enable automatic role negotiation')
@click.option('--cpu-cores', default=4, help='Number of CPU cores')
@click.option('--memory-gb', default=8.0, help='Memory in GB')
@click.option('--skills', multiple=True, help='Node skills/capabilities')
def start_node(
    node_id: Optional[str],
    listen_addr: str,
    bootstrap_peers: List[str],
    auto_role: bool,
    cpu_cores: int,
    memory_gb: float,
    skills: List[str]
):
    """Start a mesh node."""
    
    async def _start_node():
        # Create node capabilities
        capabilities = NodeCapabilities(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            skills=set(skills) if skills else {"general", "ml"}
        )
        
        # Create mesh node
        mesh_node = MeshNode(
            node_id=UUID(node_id) if node_id else None,
            listen_addr=listen_addr,
            bootstrap_peers=list(bootstrap_peers),
            capabilities=capabilities,
            auto_role=auto_role
        )
        
        console.print(f"🌐 Starting mesh node {mesh_node.node_id}", style="bold blue")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting node...", total=None)
                
                await mesh_node.start()
                
                progress.update(task, description="Node started successfully!")
                
                console.print("✅ Mesh node is running", style="bold green")
                console.print(f"📍 Node ID: {mesh_node.node_id}")
                console.print(f"🎭 Role: {mesh_node.role.value}")
                console.print(f"🔗 Listen Address: {listen_addr}")
                
                if bootstrap_peers:
                    console.print(f"🔄 Connected to {len(bootstrap_peers)} bootstrap peers")
                
                # Keep node running
                console.print("\n⏳ Node is running. Press Ctrl+C to stop...")
                try:
                    await asyncio.Event().wait()  # Wait indefinitely
                except KeyboardInterrupt:
                    console.print("\n🛑 Stopping node...")
                    await mesh_node.stop()
                    console.print("✅ Node stopped gracefully")
        
        except Exception as e:
            console.print(f"❌ Failed to start node: {e}", style="bold red")
            raise
    
    asyncio.run(_start_node())


@cli.command()
@click.option('--node-id', help='Node ID (UUID)')
@click.option('--listen-addr', default='/ip4/0.0.0.0/tcp/4001', help='Listen address')
@click.option('--bootstrap-peers', multiple=True, help='Bootstrap peer addresses')
@click.option('--rounds', default=10, help='Number of training rounds')
@click.option('--local-epochs', default=5, help='Local training epochs')
@click.option('--batch-size', default=32, help='Batch size')
@click.option('--learning-rate', default=0.01, help='Learning rate')
@click.option('--algorithm', default='fedavg', help='Federated algorithm')
def start_federated_training(
    node_id: Optional[str],
    listen_addr: str,
    bootstrap_peers: List[str],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    algorithm: str
):
    """Start federated learning training."""
    
    async def _start_training():
        # Create node capabilities
        capabilities = NodeCapabilities(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_available=torch.cuda.is_available(),
            skills={"machine_learning", "federated_learning", "pytorch"}
        )
        
        # Create mesh node
        mesh_node = MeshNode(
            node_id=UUID(node_id) if node_id else None,
            listen_addr=listen_addr,
            bootstrap_peers=list(bootstrap_peers),
            capabilities=capabilities,
            auto_role=True
        )
        
        # Federated learning configuration
        fed_config = FederatedConfig(
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            aggregation_algorithm=algorithm
        )
        
        # Create federated learner
        federated_learner = FederatedLearner(
            node_id=mesh_node.node_id,
            model_fn=lambda: SimpleModel(),
            dataset_fn=create_dummy_dataset,
            config=fed_config,
            network_manager=mesh_node.network,
            consensus_engine=mesh_node.consensus
        )
        
        console.print(f"🤖 Starting federated learning node {mesh_node.node_id}", style="bold blue")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                startup_task = progress.add_task("Starting node and FL system...", total=None)
                
                # Start mesh node
                await mesh_node.start()
                progress.update(startup_task, description="Mesh node started")
                
                # Wait for network to stabilize
                await asyncio.sleep(2)
                
                progress.update(startup_task, description="Starting federated training...")
                
                # Start federated learning
                training_task = asyncio.create_task(federated_learner.start_training())
                
                progress.update(startup_task, description="Federated training started!")
                
                console.print("✅ Federated learning is running", style="bold green")
                console.print(f"📍 Node ID: {mesh_node.node_id}")
                console.print(f"🎭 Role: {mesh_node.role.value}")
                console.print(f"🧠 Algorithm: {algorithm}")
                console.print(f"🔄 Rounds: {rounds}")
                console.print(f"📊 Local Epochs: {local_epochs}")
                
                # Monitor training progress
                try:
                    while not federated_learner.is_training_complete():
                        await asyncio.sleep(10)
                        
                        metrics = federated_learner.get_training_metrics()
                        if metrics:
                            latest = metrics[-1]
                            console.print(f"📈 Round {latest.round_number}: Loss={latest.loss:.4f}, "
                                        f"Participants={latest.participants_count}")
                    
                    console.print("🎉 Federated training completed!", style="bold green")
                    
                    # Show final results
                    final_metrics = federated_learner.get_training_metrics()
                    if final_metrics:
                        table = Table(title="Training Results")
                        table.add_column("Round", style="cyan")
                        table.add_column("Loss", style="magenta")
                        table.add_column("Participants", style="green")
                        table.add_column("Time (s)", style="yellow")
                        
                        for metric in final_metrics[-5:]:  # Show last 5 rounds
                            table.add_row(
                                str(metric.round_number),
                                f"{metric.loss:.4f}",
                                str(metric.participants_count),
                                f"{metric.aggregation_time_seconds:.2f}"
                            )
                        
                        console.print(table)
                
                except KeyboardInterrupt:
                    console.print("\n🛑 Stopping federated training...")
                    await federated_learner.stop_training()
                    await mesh_node.stop()
                    console.print("✅ Training stopped gracefully")
        
        except Exception as e:
            console.print(f"❌ Failed to start federated training: {e}", style="bold red")
            raise
    
    asyncio.run(_start_training())


@cli.command()
@click.option('--node-id', help='Node ID (UUID)')
@click.option('--listen-addr', default='/ip4/0.0.0.0/tcp/4001', help='Listen address')
@click.option('--bootstrap-peers', multiple=True, help='Bootstrap peer addresses')
@click.option('--protocol', default='contract_net', help='Coordination protocol')
@click.option('--max-agents', default=10, help='Maximum agents in coordination')
def start_coordination(
    node_id: Optional[str],
    listen_addr: str,
    bootstrap_peers: List[str],
    protocol: str,
    max_agents: int
):
    """Start multi-agent coordination."""
    
    async def _start_coordination():
        # Create node capabilities
        capabilities = NodeCapabilities(
            cpu_cores=4,
            memory_gb=8.0,
            skills={"coordination", "task_management", "planning"}
        )
        
        # Create mesh node
        mesh_node = MeshNode(
            node_id=UUID(node_id) if node_id else None,
            listen_addr=listen_addr,
            bootstrap_peers=list(bootstrap_peers),
            capabilities=capabilities,
            auto_role=True
        )
        
        # Create agent mesh
        coord_protocol = CoordinationProtocol(protocol)
        agent_mesh = AgentMesh(
            mesh_node=mesh_node,
            default_protocol=coord_protocol
        )
        
        console.print(f"🤝 Starting coordination node {mesh_node.node_id}", style="bold blue")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting coordination system...", total=None)
                
                # Start mesh node
                await mesh_node.start()
                progress.update(task, description="Mesh node started")
                
                # Start agent mesh
                await agent_mesh.start()
                progress.update(task, description="Agent coordination started!")
                
                console.print("✅ Agent coordination is running", style="bold green")
                console.print(f"📍 Node ID: {mesh_node.node_id}")
                console.print(f"🎭 Role: {mesh_node.role.value}")
                console.print(f"🤝 Protocol: {protocol}")
                console.print(f"👥 Max Agents: {max_agents}")
                
                # Keep running and show periodic status
                console.print("\n⏳ Coordination system is running. Press Ctrl+C to stop...")
                try:
                    while True:
                        await asyncio.sleep(30)
                        
                        # Show coordination metrics
                        metrics = agent_mesh.get_collaboration_metrics()
                        console.print(f"📊 Active agents: {metrics['total_agents']}, "
                                    f"Collaborations: {metrics['active_collaborations']}")
                
                except KeyboardInterrupt:
                    console.print("\n🛑 Stopping coordination system...")
                    await agent_mesh.stop()
                    await mesh_node.stop()
                    console.print("✅ Coordination system stopped gracefully")
        
        except Exception as e:
            console.print(f"❌ Failed to start coordination: {e}", style="bold red")
            raise
    
    asyncio.run(_start_coordination())


@cli.command()
@click.option('--peer-addr', required=True, help='Peer address to connect to')
@click.option('--timeout', default=10, help='Connection timeout in seconds')
def test_connection(peer_addr: str, timeout: int):
    """Test connection to a peer."""
    
    async def _test_connection():
        console.print(f"🔗 Testing connection to {peer_addr}", style="bold blue")
        
        try:
            # Create temporary node for testing
            mesh_node = MeshNode()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Connecting...", total=None)
                
                await mesh_node.start()
                progress.update(task, description="Node started, connecting to peer...")
                
                # Try to connect
                peer_info = await asyncio.wait_for(
                    mesh_node.network.connect_to_peer(peer_addr),
                    timeout=timeout
                )
                
                progress.update(task, description="Connection successful!")
                
                console.print("✅ Connection successful!", style="bold green")
                console.print(f"📍 Peer ID: {peer_info.peer_id}")
                console.print(f"🌐 Address: {peer_addr}")
                console.print(f"⚡ Latency: {peer_info.latency_ms}ms")
                
                await mesh_node.stop()
        
        except asyncio.TimeoutError:
            console.print(f"❌ Connection timeout after {timeout}s", style="bold red")
        except Exception as e:
            console.print(f"❌ Connection failed: {e}", style="bold red")
    
    asyncio.run(_test_connection())


@cli.command()
@click.option('--config-file', type=click.Path(exists=True), help='Configuration file path')
def validate_config(config_file: Optional[str]):
    """Validate configuration file."""
    
    if not config_file:
        console.print("❌ No configuration file provided", style="bold red")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        console.print(f"📄 Validating configuration: {config_file}", style="bold blue")
        
        # Basic validation
        required_fields = ["node_id", "listen_addr"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            console.print(f"❌ Missing required fields: {missing_fields}", style="bold red")
            return
        
        # Validate node_id format
        try:
            UUID(config["node_id"])
        except ValueError:
            console.print("❌ Invalid node_id format (must be UUID)", style="bold red")
            return
        
        console.print("✅ Configuration is valid!", style="bold green")
        
        # Show configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in config.items():
            table.add_row(key, str(value))
        
        console.print(table)
    
    except json.JSONDecodeError as e:
        console.print(f"❌ Invalid JSON: {e}", style="bold red")
    except Exception as e:
        console.print(f"❌ Validation failed: {e}", style="bold red")


@cli.command()
def version():
    """Show version information."""
    from . import __version__
    
    console.print(f"🚀 Agent Mesh Federated Runtime v{__version__}", style="bold green")
    console.print(f"🐍 Python: {sys.version.split()[0]}")
    console.print(f"🔥 PyTorch: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        console.print(f"⚡ CUDA: {torch.version.cuda} (GPU available)", style="green")
    else:
        console.print("💾 CUDA: Not available (CPU only)", style="yellow")


@cli.command()
def example_config():
    """Generate example configuration file."""
    
    config = {
        "node_id": str(uuid4()),
        "listen_addr": "/ip4/0.0.0.0/tcp/4001",
        "bootstrap_peers": [
            "/ip4/192.168.1.100/tcp/4001/p2p/QmExamplePeer1",
            "/ip4/192.168.1.101/tcp/4001/p2p/QmExamplePeer2"
        ],
        "capabilities": {
            "cpu_cores": 4,
            "memory_gb": 8.0,
            "storage_gb": 100.0,
            "gpu_available": False,
            "skills": ["machine_learning", "coordination"]
        },
        "federated_learning": {
            "rounds": 100,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "aggregation_algorithm": "fedavg"
        },
        "coordination": {
            "protocol": "contract_net",
            "max_agents": 10,
            "collaboration_strategy": "cooperative"
        }
    }
    
    config_file = "agent_mesh_config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"✅ Example configuration saved to: {config_file}", style="bold green")
    console.print("📝 Edit the configuration and use with --config-file option")


if __name__ == "__main__":
    cli()