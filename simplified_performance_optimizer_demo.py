"""Simplified Performance Optimization Demo - Dependency-Free Implementation.

Demonstrates autonomous performance optimization capabilities without external dependencies.
"""

import asyncio
import time
import random
import logging
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """Performance optimization targets."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_CONSUMPTION = "energy"
    CONSENSUS_SPEED = "consensus_speed"


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    consensus_success_rate: float = 0.0
    energy_consumption_watts: float = 0.0
    operations_per_watt: float = 0.0
    
    def get_composite_score(self) -> float:
        """Calculate composite performance score (0-1)."""
        # Weighted composite score
        latency_score = max(0, 1 - (self.latency_ms / 1000.0))
        throughput_score = min(1, self.throughput_ops_sec / 1000.0)
        resource_score = 1 - max(self.cpu_usage_percent, self.memory_usage_mb / 1000.0) / 100.0
        consensus_score = self.consensus_success_rate
        energy_score = min(1, self.operations_per_watt / 100.0)
        
        composite = (
            0.25 * latency_score +
            0.25 * throughput_score +
            0.2 * resource_score +
            0.2 * consensus_score +
            0.1 * energy_score
        )
        
        return max(0.0, min(1.0, composite))


@dataclass
class OptimizationAction:
    """Optimization action."""
    action_type: str
    target_component: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence_score: float
    estimated_cost: float
    priority: int = 1


class SimplifiedPerformanceOptimizer:
    """Simplified autonomous performance optimizer."""
    
    def __init__(self, optimization_interval: float = 3.0):
        """Initialize optimizer."""
        self.optimization_interval = optimization_interval
        self.current_metrics = PerformanceMetrics()
        self.metrics_history: deque = deque(maxlen=50)
        self.optimization_cycle_count = 0
        self.completed_optimizations: List[Dict[str, Any]] = []
        
        # Performance thresholds
        self.thresholds = {
            'latency_ms': 100.0,
            'cpu_usage_percent': 80.0,
            'memory_usage_mb': 800.0,
            'consensus_success_rate': 0.9,
            'throughput_ops_sec': 500.0
        }
        
        # ML simulation parameters
        self.learning_history: deque = deque(maxlen=100)
        self.action_success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        logger.info("Simplified Performance Optimizer initialized")
    
    async def start_optimization_loop(self, max_cycles: int = 10) -> None:
        """Start optimization loop for demo."""
        logger.info("Starting autonomous optimization loop")
        
        for cycle in range(max_cycles):
            try:
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Analyze and optimize
                optimization_actions = await self._analyze_and_optimize()
                
                # Execute optimizations
                if optimization_actions:
                    await self._execute_optimizations(optimization_actions)
                
                # Update learning models
                await self._update_learning_models()
                
                # Log cycle results
                await self._log_optimization_cycle()
                
                self.optimization_cycle_count += 1
                
                # Wait for next cycle
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Optimization cycle failed: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _collect_performance_metrics(self) -> None:
        """Simulate performance metrics collection."""
        metrics = PerformanceMetrics()
        
        # Simulate system state with trends
        base_time = time.time()
        cycle_factor = self.optimization_cycle_count
        
        # Base metrics with some variance and trends
        metrics.latency_ms = random.uniform(30, 200) + (cycle_factor * 2)  # Gradually increasing
        metrics.throughput_ops_sec = random.uniform(200, 800) + (cycle_factor * 10)  # Gradually increasing
        metrics.cpu_usage_percent = random.uniform(40, 90) - (cycle_factor * 2)  # Gradually decreasing from optimizations
        metrics.memory_usage_mb = random.uniform(300, 900) - (cycle_factor * 5)  # Gradually decreasing
        
        # Consensus metrics
        metrics.consensus_success_rate = random.uniform(0.85, 0.98) + (cycle_factor * 0.005)  # Gradually improving
        
        # Energy metrics
        metrics.energy_consumption_watts = (
            metrics.cpu_usage_percent * 0.5 + 
            metrics.memory_usage_mb * 0.01
        )
        metrics.operations_per_watt = metrics.throughput_ops_sec / max(metrics.energy_consumption_watts, 1.0)
        
        # Apply historical optimizations (simulate their effects)
        if self.completed_optimizations:
            recent_optimizations = [opt for opt in self.completed_optimizations 
                                  if time.time() - opt['timestamp'] < 30.0]
            
            total_improvement = sum(opt['result']['improvement'] for opt in recent_optimizations)
            improvement_factor = 1.0 + (total_improvement * 0.1)  # 10% of improvement applies
            
            # Apply improvements
            metrics.latency_ms /= improvement_factor
            metrics.throughput_ops_sec *= improvement_factor
            metrics.consensus_success_rate = min(0.99, metrics.consensus_success_rate * improvement_factor)
        
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
    
    async def _analyze_and_optimize(self) -> List[OptimizationAction]:
        """Generate optimization actions."""
        actions = []
        metrics = self.current_metrics
        
        # Latency optimization
        if metrics.latency_ms > self.thresholds['latency_ms']:
            success_rate = statistics.mean(self.action_success_rates.get("optimize_latency", [0.7]))
            
            actions.append(OptimizationAction(
                action_type="optimize_latency",
                target_component="network_layer",
                parameters={"optimization": "connection_pooling"},
                expected_improvement=0.3,
                confidence_score=success_rate,
                estimated_cost=0.2,
                priority=1
            ))
        
        # CPU optimization
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage_percent']:
            success_rate = statistics.mean(self.action_success_rates.get("optimize_cpu", [0.8]))
            
            actions.append(OptimizationAction(
                action_type="optimize_cpu",
                target_component="processing_engine",
                parameters={"optimization": "parallel_processing"},
                expected_improvement=0.4,
                confidence_score=success_rate,
                estimated_cost=0.3,
                priority=1
            ))
        
        # Throughput optimization
        if metrics.throughput_ops_sec < self.thresholds['throughput_ops_sec']:
            success_rate = statistics.mean(self.action_success_rates.get("optimize_throughput", [0.75]))
            
            actions.append(OptimizationAction(
                action_type="optimize_throughput",
                target_component="processing_pipeline",
                parameters={"optimization": "batch_processing"},
                expected_improvement=0.25,
                confidence_score=success_rate,
                estimated_cost=0.25,
                priority=2
            ))
        
        # Consensus optimization
        if metrics.consensus_success_rate < self.thresholds['consensus_success_rate']:
            success_rate = statistics.mean(self.action_success_rates.get("optimize_consensus", [0.85]))
            
            actions.append(OptimizationAction(
                action_type="optimize_consensus",
                target_component="consensus_engine",
                parameters={"optimization": "adaptive_threshold"},
                expected_improvement=0.35,
                confidence_score=success_rate,
                estimated_cost=0.1,
                priority=1
            ))
        
        # Memory optimization
        if metrics.memory_usage_mb > self.thresholds['memory_usage_mb']:
            success_rate = statistics.mean(self.action_success_rates.get("optimize_memory", [0.7]))
            
            actions.append(OptimizationAction(
                action_type="optimize_memory",
                target_component="caching_layer",
                parameters={"optimization": "cache_tuning"},
                expected_improvement=0.2,
                confidence_score=success_rate,
                estimated_cost=0.15,
                priority=2
            ))
        
        # Predictive optimizations based on trends
        if len(self.metrics_history) >= 5:
            recent_metrics = list(self.metrics_history)[-5:]
            latency_trend = self._calculate_trend([m.latency_ms for m in recent_metrics])
            
            if latency_trend > 0.05:  # Latency increasing trend
                actions.append(OptimizationAction(
                    action_type="predictive_cache_warmup",
                    target_component="caching_layer",
                    parameters={"optimization": "predictive_caching"},
                    expected_improvement=0.2,
                    confidence_score=0.6,
                    estimated_cost=0.15,
                    priority=3
                ))
        
        # Sort by priority and confidence
        actions.sort(key=lambda a: (a.priority, -a.confidence_score))
        return actions[:3]  # Top 3 actions per cycle
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple trend slope."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize by mean
        mean_value = sum_y / n
        return slope / mean_value if mean_value != 0 else 0.0
    
    async def _execute_optimizations(self, actions: List[OptimizationAction]) -> None:
        """Execute optimization actions."""
        for action in actions:
            try:
                logger.info(f"Executing: {action.action_type} on {action.target_component}")
                
                # Simulate execution
                execution_result = await self._simulate_optimization_execution(action)
                
                # Update success rates for ML learning
                success = execution_result['success']
                self.action_success_rates[action.action_type].append(1.0 if success else 0.0)
                
                # Record completion
                self.completed_optimizations.append({
                    'action': action.action_type,
                    'target': action.target_component,
                    'result': execution_result,
                    'timestamp': time.time()
                })
                
                logger.info(f"Completed: {action.action_type}, improvement: {execution_result['improvement']:.3f}")
                
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
    
    async def _simulate_optimization_execution(self, action: OptimizationAction) -> Dict[str, Any]:
        """Simulate optimization execution."""
        await asyncio.sleep(random.uniform(0.2, 0.8))  # Simulate execution time
        
        # Success based on confidence score
        success = random.random() < action.confidence_score
        
        if success:
            improvement = action.expected_improvement * random.uniform(0.8, 1.2)
        else:
            improvement = action.expected_improvement * random.uniform(-0.1, 0.3)
        
        return {
            'success': success,
            'improvement': improvement,
            'execution_time': random.uniform(0.2, 0.8)
        }
    
    async def _update_learning_models(self) -> None:
        """Update learning models and thresholds."""
        current_score = self.current_metrics.get_composite_score()
        self.learning_history.append(current_score)
        
        # Adaptive threshold adjustment
        if len(self.learning_history) >= 10:
            recent_scores = list(self.learning_history)[-10:]
            avg_recent_score = statistics.mean(recent_scores)
            
            # If performance is consistently good, tighten thresholds
            if avg_recent_score > 0.8:
                for key in self.thresholds:
                    if key == 'consensus_success_rate':
                        self.thresholds[key] = min(0.95, self.thresholds[key] * 1.02)
                    elif 'percent' in key:
                        self.thresholds[key] *= 0.98  # Lower CPU/memory thresholds
                    else:
                        self.thresholds[key] *= 0.95  # Lower latency threshold, higher throughput
            
            # If performance is poor, relax thresholds
            elif avg_recent_score < 0.6:
                for key in self.thresholds:
                    if key == 'consensus_success_rate':
                        self.thresholds[key] = max(0.85, self.thresholds[key] * 0.98)
                    elif 'percent' in key:
                        self.thresholds[key] *= 1.02  # Higher CPU/memory thresholds
                    else:
                        self.thresholds[key] *= 1.05  # Higher latency threshold, lower throughput
    
    async def _log_optimization_cycle(self) -> None:
        """Log optimization cycle."""
        if self.optimization_cycle_count % 3 == 0:  # Log every 3 cycles
            score = self.current_metrics.get_composite_score()
            logger.info(f"Cycle {self.optimization_cycle_count}: Score {score:.3f}, "
                       f"Latency {self.current_metrics.latency_ms:.1f}ms, "
                       f"Throughput {self.current_metrics.throughput_ops_sec:.1f} ops/sec")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        if not self.metrics_history:
            return {"error": "No data available"}
        
        current_score = self.current_metrics.get_composite_score()
        recent_optimizations = [opt for opt in self.completed_optimizations 
                              if time.time() - opt['timestamp'] < 60.0]
        
        return {
            "current_performance_score": current_score,
            "optimization_cycles": self.optimization_cycle_count,
            "recent_optimizations": len(recent_optimizations),
            "avg_improvement": statistics.mean([opt['result']['improvement'] 
                                              for opt in recent_optimizations]) if recent_optimizations else 0,
            "success_rate": statistics.mean([opt['result']['success'] 
                                           for opt in recent_optimizations]) if recent_optimizations else 0,
            "current_metrics": {
                "latency_ms": self.current_metrics.latency_ms,
                "throughput_ops_sec": self.current_metrics.throughput_ops_sec,
                "cpu_usage_percent": self.current_metrics.cpu_usage_percent,
                "consensus_success_rate": self.current_metrics.consensus_success_rate,
                "operations_per_watt": self.current_metrics.operations_per_watt
            }
        }


async def run_performance_optimization_demo():
    """Run performance optimization demonstration."""
    print("⚡ Autonomous Performance Optimizer - Demo")
    print("=" * 60)
    print("🎯 Self-Optimizing Distributed System")
    print("📊 Monitoring: Latency, Throughput, CPU, Memory, Consensus")
    print("🧠 ML-Driven: Adaptive thresholds and predictive optimization")
    print("🔄 Autonomous: Continuous self-improvement")
    print()
    
    # Initialize optimizer
    optimizer = SimplifiedPerformanceOptimizer(optimization_interval=2.0)
    
    try:
        print("🚀 Starting autonomous optimization system...")
        print("⏱️  Running 10 optimization cycles...")
        print()
        
        # Run optimization cycles
        await optimizer.start_optimization_loop(max_cycles=10)
        
        # Generate final report
        print("📈 AUTONOMOUS OPTIMIZATION REPORT")
        print("=" * 60)
        
        report = optimizer.get_optimization_report()
        
        print(f"🎯 Final Performance Score: {report['current_performance_score']:.3f}")
        print(f"🔄 Optimization Cycles: {report['optimization_cycles']}")
        print(f"⚡ Recent Optimizations: {report['recent_optimizations']}")
        print(f"🎪 Success Rate: {report['success_rate']:.1%}")
        print(f"📈 Avg Improvement: {report['avg_improvement']:.3f}")
        
        print(f"\n💡 Final System Metrics:")
        metrics = report['current_metrics']
        print(f"  • Latency: {metrics['latency_ms']:.1f}ms")
        print(f"  • Throughput: {metrics['throughput_ops_sec']:.1f} ops/sec") 
        print(f"  • CPU Usage: {metrics['cpu_usage_percent']:.1f}%")
        print(f"  • Consensus Success: {metrics['consensus_success_rate']:.2%}")
        print(f"  • Energy Efficiency: {metrics['operations_per_watt']:.1f} ops/watt")
        
        # Show improvement over time
        if len(optimizer.learning_history) >= 5:
            initial_score = optimizer.learning_history[0]
            final_score = optimizer.learning_history[-1]
            improvement = ((final_score - initial_score) / initial_score) * 100
            print(f"\n📈 Performance Improvement: {improvement:+.1f}%")
        
        print("\n✅ Autonomous performance optimization demo completed!")
        print("🚀 System demonstrates continuous self-optimization capabilities!")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Optimization demo failed: {e}")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run performance optimization demo
    asyncio.run(run_performance_optimization_demo())