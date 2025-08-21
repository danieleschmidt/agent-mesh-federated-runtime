#!/usr/bin/env python3
"""Autonomous Intelligence System Demonstration.

This demo showcases the complete autonomous intelligence capabilities
including self-healing, adaptive optimization, intelligent routing,
autonomous decision-making, and continuous learning.
"""

import asyncio
import logging
import time
import random
import json
from typing import Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent_mesh.autonomous import (
    SelfHealingManager,
    AdaptiveOptimizer,
    IntelligentRouter,
    AutonomousDecisionEngine,
    ContinuousLearningCoordinator
)
from agent_mesh.autonomous.self_healing import HealthStatus
from agent_mesh.autonomous.adaptive_optimizer import OptimizationStrategy
from agent_mesh.autonomous.intelligent_router import RoutingStrategy, NodeStatus
from agent_mesh.autonomous.decision_engine import DecisionType, ActionType
from agent_mesh.autonomous.learning_coordinator import LearningType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('autonomous_intelligence.log')
    ]
)
logger = logging.getLogger(__name__)

class AutonomousIntelligenceDemo:
    """Demonstration of autonomous intelligence capabilities."""
    
    def __init__(self):
        # Initialize autonomous components
        self.self_healing = SelfHealingManager(check_interval=10.0)
        self.optimizer = AdaptiveOptimizer(
            strategy=OptimizationStrategy.BALANCED,
            optimization_interval=15.0
        )
        self.router = IntelligentRouter(
            strategy=RoutingStrategy.ADAPTIVE,
            topology_update_interval=20.0
        )
        self.decision_engine = AutonomousDecisionEngine(
            decision_interval=20.0
        )
        self.learning_coordinator = ContinuousLearningCoordinator(
            learning_interval=30.0
        )
        
        # Demo state
        self.demo_metrics = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "network_latency": 25.0,
            "request_rate": 100.0,
            "error_rate": 0.02,
            "response_time": 150.0
        }
        
        self.is_running = False
        self._demo_task = None
        
    async def setup_demo_environment(self):
        """Set up the demo environment with initial configuration."""
        logger.info("🚀 Setting up Autonomous Intelligence Demo Environment")
        
        # Set up self-healing health metrics
        self.self_healing.register_health_metric("cpu_usage", 80.0, 95.0)
        self.self_healing.register_health_metric("memory_usage", 85.0, 95.0)
        self.self_healing.register_health_metric("network_connectivity", 50.0, 20.0)
        self.self_healing.register_health_metric("error_rate", 5.0, 10.0)
        
        # Set up network topology for intelligent routing
        nodes = [
            ("node-1", "192.168.1.10", 8080),
            ("node-2", "192.168.1.11", 8080),
            ("node-3", "192.168.1.12", 8080),
            ("node-4", "192.168.1.13", 8080),
            ("node-5", "192.168.1.14", 8080)
        ]
        
        for node_id, address, port in nodes:
            self.router.add_node(node_id, address, port, NodeStatus.HEALTHY)
        
        # Add network links
        links = [
            ("node-1", "node-2", 10.0, 1000.0),
            ("node-1", "node-3", 15.0, 800.0),
            ("node-2", "node-4", 12.0, 900.0),
            ("node-3", "node-4", 8.0, 1200.0),
            ("node-4", "node-5", 20.0, 600.0),
            ("node-2", "node-5", 25.0, 500.0)
        ]
        
        for source, target, latency, bandwidth in links:
            self.router.add_link(source, target, latency, bandwidth)
        
        # Register optimization parameters
        self.optimizer.register_parameter("demo_batch_size", 16.0, 1.0, 64.0, 2.0)
        self.optimizer.register_parameter("demo_timeout", 30.0, 5.0, 120.0, 5.0)
        
        logger.info("✅ Demo environment setup complete")
    
    async def simulate_system_dynamics(self):
        """Simulate realistic system dynamics and events."""
        while self.is_running:
            try:
                # Simulate metric fluctuations
                self._simulate_metric_changes()
                
                # Update all components with current metrics
                await self._update_components()
                
                # Simulate network events
                await self._simulate_network_events()
                
                # Simulate random system events
                await self._simulate_system_events()
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system simulation: {e}")
                await asyncio.sleep(5.0)
    
    def _simulate_metric_changes(self):
        """Simulate realistic metric changes over time."""
        current_time = time.time()
        
        # Add some realistic patterns
        hour_of_day = (current_time / 3600) % 24
        
        # CPU usage with daily pattern
        base_cpu = 30 + 20 * (1 + math.sin((hour_of_day - 6) * math.pi / 12))
        self.demo_metrics["cpu_usage"] = max(0, min(100, 
            base_cpu + random.gauss(0, 5)
        ))
        
        # Memory usage with gradual increase
        self.demo_metrics["memory_usage"] = max(30, min(95,
            self.demo_metrics["memory_usage"] + random.gauss(0, 2)
        ))
        
        # Network latency with occasional spikes
        if random.random() < 0.1:  # 10% chance of spike
            self.demo_metrics["network_latency"] = random.uniform(80, 200)
        else:
            self.demo_metrics["network_latency"] = max(5, min(50,
                25 + random.gauss(0, 8)
            ))
        
        # Request rate with business hours pattern
        if 9 <= hour_of_day <= 17:  # Business hours
            base_rate = 200 + 100 * random.random()
        else:
            base_rate = 50 + 50 * random.random()
        
        self.demo_metrics["request_rate"] = max(0, base_rate + random.gauss(0, 20))
        
        # Error rate with occasional problems
        if random.random() < 0.05:  # 5% chance of error spike
            self.demo_metrics["error_rate"] = random.uniform(0.05, 0.15)
        else:
            self.demo_metrics["error_rate"] = max(0, random.uniform(0.001, 0.03))
        
        # Response time correlated with CPU and latency
        base_response = 50 + (self.demo_metrics["cpu_usage"] / 100) * 100
        base_response += self.demo_metrics["network_latency"]
        self.demo_metrics["response_time"] = max(10, base_response + random.gauss(0, 20))
    
    async def _update_components(self):
        """Update all autonomous components with current metrics."""
        # Update self-healing
        self.self_healing.update_health_metric("cpu_usage", self.demo_metrics["cpu_usage"])
        self.self_healing.update_health_metric("memory_usage", self.demo_metrics["memory_usage"])
        self.self_healing.update_health_metric("network_connectivity", 
            100 - self.demo_metrics["network_latency"])
        self.self_healing.update_health_metric("error_rate", 
            self.demo_metrics["error_rate"] * 100)
        
        # Update optimizer
        self.optimizer.record_metric("cpu_usage", self.demo_metrics["cpu_usage"])
        self.optimizer.record_metric("memory_usage", self.demo_metrics["memory_usage"])
        self.optimizer.record_metric("response_time", self.demo_metrics["response_time"])
        self.optimizer.record_metric("requests_per_second", self.demo_metrics["request_rate"])
        self.optimizer.record_metric("error_rate", self.demo_metrics["error_rate"])
        
        # Update decision engine
        self.decision_engine.update_state(
            cpu_usage=self.demo_metrics["cpu_usage"],
            memory_usage=self.demo_metrics["memory_usage"],
            network_latency=self.demo_metrics["network_latency"],
            request_rate=self.demo_metrics["request_rate"],
            error_rate=self.demo_metrics["error_rate"],
            response_time=self.demo_metrics["response_time"]
        )
        
        # Update learning coordinator
        for metric, value in self.demo_metrics.items():
            self.learning_coordinator.record_metric(metric, value)
    
    async def _simulate_network_events(self):
        """Simulate network topology changes."""
        if random.random() < 0.1:  # 10% chance per cycle
            # Randomly update node status
            node_ids = ["node-1", "node-2", "node-3", "node-4", "node-5"]
            node_id = random.choice(node_ids)
            
            # Random latency and load updates
            latency = random.uniform(5, 50)
            bandwidth = random.uniform(500, 1500)
            load = random.random()
            
            self.router.update_node_metrics(
                node_id=node_id,
                latency=latency,
                bandwidth=bandwidth,
                load=load
            )
            
            # Occasionally simulate node issues
            if random.random() < 0.05:  # 5% chance
                status = random.choice([NodeStatus.DEGRADED, NodeStatus.OVERLOADED])
                self.router.update_node_metrics(node_id=node_id, status=status)
    
    async def _simulate_system_events(self):
        """Simulate various system events."""
        if random.random() < 0.2:  # 20% chance per cycle
            event_types = [
                "deployment_started",
                "cache_cleared",
                "database_backup",
                "security_scan",
                "performance_test"
            ]
            
            event_type = random.choice(event_types)
            event_data = {
                "severity": random.choice(["low", "medium", "high"]),
                "duration": random.uniform(10, 300),
                "component": random.choice(["api", "database", "cache", "network"])
            }
            
            self.learning_coordinator.record_system_event(event_type, event_data)
    
    async def demonstrate_autonomous_features(self):
        """Demonstrate key autonomous features."""
        logger.info("🧠 Demonstrating Autonomous Intelligence Features")
        
        await asyncio.sleep(30)  # Let system collect some data
        
        # 1. Demonstrate self-healing
        logger.info("🩺 Self-Healing Demonstration")
        health_status = self.self_healing.get_system_health()
        logger.info(f"System health: {health_status['status']}")
        
        # Simulate health issues
        self.self_healing.update_health_metric("cpu_usage", 95.0)  # Critical
        await asyncio.sleep(5)
        
        health_status = self.self_healing.get_system_health()
        logger.info(f"Health after CPU spike: {health_status['status']}")
        
        # 2. Demonstrate adaptive optimization
        logger.info("⚡ Adaptive Optimization Demonstration")
        opt_summary = self.optimizer.get_optimization_summary()
        logger.info(f"Optimization summary: strategy={opt_summary['strategy']}, "
                   f"performance={opt_summary['current_performance']:.3f}")
        
        # 3. Demonstrate intelligent routing
        logger.info("🛣️ Intelligent Routing Demonstration")
        route = self.router.find_route("node-1", "node-5")
        if route:
            logger.info(f"Route from node-1 to node-5: {' → '.join(route.nodes)}")
            logger.info(f"Route metrics: latency={route.total_latency:.1f}ms, "
                       f"reliability={route.reliability_score:.3f}")
        
        # Simulate route performance feedback
        if route:
            self.router.record_route_performance(
                route.nodes, 
                success=True, 
                actual_latency=route.total_latency + random.uniform(-5, 10),
                actual_bandwidth=route.total_bandwidth * random.uniform(0.8, 1.2)
            )
        
        # 4. Demonstrate decision engine
        logger.info("🎯 Autonomous Decision Engine Demonstration")
        decision_summary = self.decision_engine.get_decision_summary()
        logger.info(f"Decision engine: {decision_summary['total_decisions']} decisions, "
                   f"success_rate={decision_summary['success_rate']:.2f}")
        
        # 5. Demonstrate learning coordinator
        logger.info("📚 Continuous Learning Demonstration")
        learning_insights = self.learning_coordinator.get_learning_insights()
        logger.info(f"Learning insights: phase={learning_insights['current_phase']}, "
                   f"patterns={learning_insights['total_patterns']}, "
                   f"predictions={learning_insights['active_predictions']}")
        
        # Show discovered patterns
        patterns = self.learning_coordinator.get_pattern_summary()
        if patterns:
            logger.info(f"Top pattern: {patterns[0]['description']} "
                       f"(strength: {patterns[0]['strength']:.3f})")
    
    async def run_performance_stress_test(self):
        """Run a stress test to trigger autonomous responses."""
        logger.info("🔥 Running Performance Stress Test")
        
        # Gradually increase system load
        stress_duration = 60  # 1 minute stress test
        start_time = time.time()
        
        while time.time() - start_time < stress_duration and self.is_running:
            progress = (time.time() - start_time) / stress_duration
            
            # Increase CPU usage
            self.demo_metrics["cpu_usage"] = 40 + progress * 50
            
            # Increase memory usage
            self.demo_metrics["memory_usage"] = 50 + progress * 40
            
            # Increase error rate
            self.demo_metrics["error_rate"] = 0.01 + progress * 0.08
            
            # Increase response time
            self.demo_metrics["response_time"] = 100 + progress * 300
            
            logger.info(f"Stress test progress: {progress*100:.1f}% - "
                       f"CPU: {self.demo_metrics['cpu_usage']:.1f}%, "
                       f"Errors: {self.demo_metrics['error_rate']*100:.2f}%")
            
            await asyncio.sleep(5)
        
        logger.info("✅ Stress test complete - observing autonomous responses")
        
        # Give time for autonomous systems to respond
        await asyncio.sleep(30)
        
        # Check what actions were taken
        recovery_history = self.self_healing.get_recovery_history(10)
        if recovery_history:
            logger.info(f"🩺 Self-healing actions taken: {len(recovery_history)}")
            for action in recovery_history[-3:]:  # Last 3 actions
                logger.info(f"  - {action.get('action', 'unknown')} at "
                           f"{time.ctime(action.get('timestamp', 0))}")
        
        decision_summary = self.decision_engine.get_decision_summary()
        logger.info(f"🎯 Autonomous decisions: {decision_summary['total_decisions']} total")
    
    async def generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("📊 Generating Autonomous Intelligence Report")
        
        report = {
            "demo_duration": time.time(),
            "self_healing": {
                "health_status": self.self_healing.get_system_health(),
                "recovery_actions": len(self.self_healing.get_recovery_history()),
            },
            "optimization": self.optimizer.get_optimization_summary(),
            "routing": {
                "topology": self.router.get_topology_summary(),
                "analytics": self.router.get_route_analytics()
            },
            "decisions": self.decision_engine.get_decision_summary(),
            "learning": {
                "insights": self.learning_coordinator.get_learning_insights(),
                "patterns": len(self.learning_coordinator.get_pattern_summary()),
                "predictions": len(self.learning_coordinator.get_prediction_summary())
            }
        }
        
        # Save report
        with open("autonomous_intelligence_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📄 Report saved to autonomous_intelligence_report.json")
        
        # Print summary
        print("\n" + "="*80)
        print("🤖 AUTONOMOUS INTELLIGENCE DEMONSTRATION COMPLETE")
        print("="*80)
        print(f"🩺 Self-Healing: {report['self_healing']['recovery_actions']} recovery actions")
        print(f"⚡ Optimization: {report['optimization']['optimization_episodes']} optimization episodes")
        print(f"🛣️ Routing: {report['routing']['topology']['nodes']['total']} nodes, "
              f"{report['routing']['topology']['links']['total']} links")
        print(f"🎯 Decisions: {report['decisions']['total_decisions']} autonomous decisions")
        print(f"📚 Learning: {report['learning']['patterns']} patterns, "
              f"{report['learning']['predictions']} active predictions")
        print("="*80)
    
    async def start_demo(self):
        """Start the complete autonomous intelligence demonstration."""
        logger.info("🚀 Starting Autonomous Intelligence Demonstration")
        
        try:
            # Setup
            await self.setup_demo_environment()
            
            # Start all autonomous components
            await self.self_healing.start()
            await self.optimizer.start()
            await self.router.start()
            await self.decision_engine.start()
            await self.learning_coordinator.start()
            
            self.is_running = True
            
            # Start simulation
            simulation_task = asyncio.create_task(self.simulate_system_dynamics())
            
            # Run demonstrations
            await asyncio.sleep(10)  # Initial warm-up
            await self.demonstrate_autonomous_features()
            
            await asyncio.sleep(30)  # Let systems adapt
            await self.run_performance_stress_test()
            
            await asyncio.sleep(60)  # Final observation period
            
            # Generate report
            await self.generate_final_report()
            
            # Cleanup
            self.is_running = False
            simulation_task.cancel()
            
            await self.self_healing.stop()
            await self.optimizer.stop()
            await self.router.stop()
            await self.decision_engine.stop()
            await self.learning_coordinator.stop()
            
            logger.info("✅ Autonomous Intelligence Demonstration Complete")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise

async def main():
    """Main demo function."""
    print("🤖 Agent Mesh Autonomous Intelligence Demonstration")
    print("This demo showcases advanced autonomous capabilities including:")
    print("  🩺 Self-Healing and Recovery")
    print("  ⚡ Adaptive Performance Optimization") 
    print("  🛣️ Intelligent Network Routing")
    print("  🎯 Autonomous Decision Making")
    print("  📚 Continuous Learning and Adaptation")
    print()
    
    demo = AutonomousIntelligenceDemo()
    await demo.start_demo()

if __name__ == "__main__":
    import math  # Add missing import
    asyncio.run(main())