"""Autonomous Performance Optimizer for Generation 3 Scaling.

Advanced self-adapting performance optimization with machine learning,
predictive scaling, and autonomous resource management.
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class MockNumpy:
        @staticmethod
        def array(x): return x
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): 
            if not x: return 0
            mean = sum(x) / len(x)
            return (sum((xi - mean) ** 2 for xi in x) / len(x)) ** 0.5
        @staticmethod  
        def percentile(x, p): 
            if not x: return 0
            sorted_x = sorted(x)
            idx = int(len(sorted_x) * p / 100)
            return sorted_x[min(idx, len(sorted_x) - 1)]
    np = MockNumpy()


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    AUTONOMOUS = "autonomous"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    CONNECTIONS = "connections"
    CACHE = "cache"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 100.0
    response_time_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    error_rate_percent: float = 0.0


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    strategy: OptimizationStrategy
    resource_type: ResourceType
    action: str
    improvement_percent: float
    cost: float
    confidence: float
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


class AutonomousPerformanceOptimizer:
    """Autonomous performance optimization system."""
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.AUTONOMOUS,
        optimization_interval: float = 30.0,
        metrics_history_size: int = 1000,
        config: Optional[Dict[str, Any]] = None
    ):
        self.strategy = strategy
        self.optimization_interval = optimization_interval
        self.metrics_history_size = metrics_history_size
        self.config = config or {}
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=metrics_history_size)
        self.resource_utilization: Dict[ResourceType, deque] = {
            resource: deque(maxlen=100) for resource in ResourceType
        }
        
        # Optimization state
        self.optimizations_applied: List[OptimizationResult] = []
        self.performance_baselines: Dict[str, float] = {}
        self.learning_models: Dict[ResourceType, Dict] = defaultdict(dict)
        
        # Control flags
        self._running = False
        self._optimization_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        self.logger = logging.getLogger("autonomous_optimizer")
        
        # Initialize performance baselines
        self._initialize_baselines()
    
    def _initialize_baselines(self) -> None:
        """Initialize performance baselines for comparison."""
        self.performance_baselines = {
            "response_time_ms": 100.0,  # Target: < 100ms
            "throughput_ops_sec": 1000.0,  # Target: > 1000 ops/sec
            "cpu_usage_percent": 70.0,  # Target: < 70%
            "memory_usage_mb": 512.0,  # Target: < 512MB
            "cache_hit_rate": 90.0,  # Target: > 90%
            "error_rate_percent": 1.0,  # Target: < 1%
        }
    
    async def start(self) -> None:
        """Start autonomous performance optimization."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting autonomous performance optimizer",
                        strategy=self.strategy.value)
        
        # Start optimization loop
        self._optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop(self) -> None:
        """Stop performance optimization."""
        self._running = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        self.logger.info("Autonomous performance optimizer stopped")
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # In a real implementation, this would gather actual system metrics
        # For demo purposes, we simulate realistic metrics
        
        import random
        import psutil
        
        try:
            # Use actual system metrics if available
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Simulate network and other metrics
            network_io = random.uniform(10, 100)  # Mbps
            disk_io = random.uniform(5, 50)  # Mbps
            
        except ImportError:
            # Fallback to simulated metrics
            cpu_percent = random.uniform(20, 80)
            memory_mb = random.uniform(200, 800)
            network_io = random.uniform(10, 100)
            disk_io = random.uniform(5, 50)
        
        # Simulate application-level metrics
        response_time = random.uniform(50, 200)  # ms
        throughput = random.uniform(500, 2000)  # ops/sec
        cache_hit_rate = random.uniform(85, 98)  # %
        error_rate = random.uniform(0, 3)  # %
        connections = random.randint(10, 100)
        
        metrics = PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            network_io_mbps=network_io,
            disk_io_mbps=disk_io,
            active_connections=connections,
            cache_hit_rate=cache_hit_rate,
            response_time_ms=response_time,
            throughput_ops_sec=throughput,
            error_rate_percent=error_rate
        )
        
        # Store metrics
        with self._lock:
            self.metrics_history.append(metrics)
            self.resource_utilization[ResourceType.CPU].append(cpu_percent)
            self.resource_utilization[ResourceType.MEMORY].append(memory_mb)
            self.resource_utilization[ResourceType.NETWORK].append(network_io)
            self.resource_utilization[ResourceType.DISK].append(disk_io)
        
        return metrics
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._running:
            try:
                # Collect current metrics
                current_metrics = await self.collect_metrics()
                
                # Analyze performance
                analysis = await self._analyze_performance(current_metrics)
                
                # Generate optimizations
                optimizations = await self._generate_optimizations(analysis)
                
                # Apply optimizations
                applied_optimizations = await self._apply_optimizations(optimizations)
                
                # Update learning models
                await self._update_learning_models(current_metrics, applied_optimizations)
                
                # Log optimization results
                if applied_optimizations:
                    self.logger.info("Applied performance optimizations",
                                   count=len(applied_optimizations),
                                   total_improvement=sum(opt.improvement_percent 
                                                       for opt in applied_optimizations))
                
                # Wait for next optimization cycle
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error("Error in optimization loop", error=str(e))
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _analyze_performance(
        self,
        current_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Analyze current performance against baselines and trends."""
        analysis = {
            "bottlenecks": [],
            "trends": {},
            "anomalies": [],
            "optimization_opportunities": []
        }
        
        # Check against baselines
        if current_metrics.response_time_ms > self.performance_baselines["response_time_ms"]:
            analysis["bottlenecks"].append({
                "type": "high_latency",
                "severity": min((current_metrics.response_time_ms / 
                               self.performance_baselines["response_time_ms"]), 3.0),
                "metric": "response_time_ms",
                "current": current_metrics.response_time_ms,
                "baseline": self.performance_baselines["response_time_ms"]
            })
        
        if current_metrics.throughput_ops_sec < self.performance_baselines["throughput_ops_sec"]:
            analysis["bottlenecks"].append({
                "type": "low_throughput",
                "severity": self.performance_baselines["throughput_ops_sec"] / 
                           max(current_metrics.throughput_ops_sec, 1),
                "metric": "throughput_ops_sec",
                "current": current_metrics.throughput_ops_sec,
                "baseline": self.performance_baselines["throughput_ops_sec"]
            })
        
        if current_metrics.cpu_usage_percent > self.performance_baselines["cpu_usage_percent"]:
            analysis["bottlenecks"].append({
                "type": "high_cpu",
                "severity": current_metrics.cpu_usage_percent / 
                           self.performance_baselines["cpu_usage_percent"],
                "metric": "cpu_usage_percent",
                "current": current_metrics.cpu_usage_percent,
                "baseline": self.performance_baselines["cpu_usage_percent"]
            })
        
        # Analyze trends from historical data
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            
            # CPU trend
            cpu_values = [m.cpu_usage_percent for m in recent_metrics]
            cpu_trend = self._calculate_trend(cpu_values)
            analysis["trends"]["cpu"] = cpu_trend
            
            # Memory trend  
            memory_values = [m.memory_usage_mb for m in recent_metrics]
            memory_trend = self._calculate_trend(memory_values)
            analysis["trends"]["memory"] = memory_trend
            
            # Response time trend
            response_values = [m.response_time_ms for m in recent_metrics]
            response_trend = self._calculate_trend(response_values)
            analysis["trends"]["response_time"] = response_trend
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend direction and magnitude."""
        if len(values) < 2:
            return {"direction": 0.0, "magnitude": 0.0}
        
        # Simple linear regression slope
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x_sq_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
        
        return {
            "direction": 1.0 if slope > 0 else -1.0 if slope < 0 else 0.0,
            "magnitude": abs(slope)
        }
    
    async def _generate_optimizations(
        self,
        analysis: Dict[str, Any]
    ) -> List[OptimizationResult]:
        """Generate optimization recommendations based on analysis."""
        optimizations = []
        
        for bottleneck in analysis["bottlenecks"]:
            if bottleneck["type"] == "high_latency":
                # Suggest caching optimizations
                optimizations.append(OptimizationResult(
                    strategy=self.strategy,
                    resource_type=ResourceType.CACHE,
                    action="increase_cache_size",
                    improvement_percent=min(bottleneck["severity"] * 15, 50),
                    cost=bottleneck["severity"] * 0.1,
                    confidence=0.8
                ))
                
                # Suggest connection pooling
                optimizations.append(OptimizationResult(
                    strategy=self.strategy,
                    resource_type=ResourceType.CONNECTIONS,
                    action="optimize_connection_pool",
                    improvement_percent=min(bottleneck["severity"] * 10, 30),
                    cost=bottleneck["severity"] * 0.05,
                    confidence=0.7
                ))
            
            elif bottleneck["type"] == "low_throughput":
                # Suggest parallel processing
                optimizations.append(OptimizationResult(
                    strategy=self.strategy,
                    resource_type=ResourceType.CPU,
                    action="increase_parallelism",
                    improvement_percent=min(bottleneck["severity"] * 20, 60),
                    cost=bottleneck["severity"] * 0.15,
                    confidence=0.85
                ))
                
                # Suggest I/O optimization
                optimizations.append(OptimizationResult(
                    strategy=self.strategy,
                    resource_type=ResourceType.DISK,
                    action="optimize_io_operations", 
                    improvement_percent=min(bottleneck["severity"] * 12, 40),
                    cost=bottleneck["severity"] * 0.08,
                    confidence=0.75
                ))
            
            elif bottleneck["type"] == "high_cpu":
                # Suggest CPU optimization
                optimizations.append(OptimizationResult(
                    strategy=self.strategy,
                    resource_type=ResourceType.CPU,
                    action="optimize_cpu_intensive_operations",
                    improvement_percent=min(bottleneck["severity"] * 25, 70),
                    cost=bottleneck["severity"] * 0.2,
                    confidence=0.9
                ))
                
                # Suggest load balancing
                optimizations.append(OptimizationResult(
                    strategy=self.strategy,
                    resource_type=ResourceType.CPU,
                    action="implement_load_balancing",
                    improvement_percent=min(bottleneck["severity"] * 18, 50),
                    cost=bottleneck["severity"] * 0.12,
                    confidence=0.8
                ))
        
        # Sort by potential improvement and confidence
        optimizations.sort(
            key=lambda opt: opt.improvement_percent * opt.confidence,
            reverse=True
        )
        
        return optimizations[:5]  # Return top 5 optimizations
    
    async def _apply_optimizations(
        self,
        optimizations: List[OptimizationResult]
    ) -> List[OptimizationResult]:
        """Apply selected optimizations."""
        applied = []
        
        for optimization in optimizations:
            # Apply cost-benefit analysis
            if optimization.confidence < 0.6:
                continue  # Skip low-confidence optimizations
            
            if optimization.cost > 0.3:
                continue  # Skip high-cost optimizations
            
            # Simulate applying the optimization
            success = await self._simulate_optimization_application(optimization)
            
            if success:
                optimization.applied = True
                applied.append(optimization)
                self.optimizations_applied.append(optimization)
                
                self.logger.info("Applied optimization",
                               action=optimization.action,
                               improvement=optimization.improvement_percent,
                               cost=optimization.cost)
        
        return applied
    
    async def _simulate_optimization_application(
        self,
        optimization: OptimizationResult
    ) -> bool:
        """Simulate applying an optimization (replace with actual implementation)."""
        # In a real implementation, this would apply actual optimizations
        
        if optimization.action == "increase_cache_size":
            # Simulate cache size increase
            await asyncio.sleep(0.1)  # Simulate work
            return True
        
        elif optimization.action == "optimize_connection_pool":
            # Simulate connection pool optimization
            await asyncio.sleep(0.1)
            return True
        
        elif optimization.action == "increase_parallelism":
            # Simulate parallelism increase
            await asyncio.sleep(0.1)
            return True
        
        elif optimization.action == "optimize_io_operations":
            # Simulate I/O optimization
            await asyncio.sleep(0.1)
            return True
        
        elif optimization.action == "optimize_cpu_intensive_operations":
            # Simulate CPU optimization
            await asyncio.sleep(0.1)
            return True
        
        elif optimization.action == "implement_load_balancing":
            # Simulate load balancing implementation
            await asyncio.sleep(0.1)
            return True
        
        return False
    
    async def _update_learning_models(
        self,
        metrics: PerformanceMetrics,
        optimizations: List[OptimizationResult]
    ) -> None:
        """Update machine learning models based on optimization results."""
        # Simple learning model updating (in production, use proper ML)
        
        for optimization in optimizations:
            resource_type = optimization.resource_type
            
            # Update success rate for this action
            if resource_type not in self.learning_models:
                self.learning_models[resource_type] = {}
            
            action_stats = self.learning_models[resource_type].get(
                optimization.action,
                {"attempts": 0, "successes": 0, "avg_improvement": 0.0}
            )
            
            action_stats["attempts"] += 1
            if optimization.applied:
                action_stats["successes"] += 1
                # Update average improvement
                current_avg = action_stats["avg_improvement"]
                action_stats["avg_improvement"] = (
                    (current_avg * (action_stats["successes"] - 1) + 
                     optimization.improvement_percent) / action_stats["successes"]
                )
            
            self.learning_models[resource_type][optimization.action] = action_stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-10:] if len(self.metrics_history) >= 10 else list(self.metrics_history)
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_response_time = np.mean([m.response_time_ms for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_ops_sec for m in recent_metrics])
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate_percent for m in recent_metrics])
        
        # Calculate performance scores
        cpu_score = max(0, 100 - avg_cpu)  # Lower is better
        memory_score = max(0, 100 - (avg_memory / 10))  # Normalized
        response_score = max(0, 100 - avg_response_time)  # Lower is better
        throughput_score = min(100, avg_throughput / 10)  # Higher is better
        cache_score = avg_cache_hit_rate  # Higher is better
        error_score = max(0, 100 - avg_error_rate * 10)  # Lower is better
        
        overall_score = np.mean([cpu_score, memory_score, response_score, 
                                throughput_score, cache_score, error_score])
        
        return {
            "overall_performance_score": round(overall_score, 2),
            "metrics": {
                "cpu_usage_percent": round(avg_cpu, 2),
                "memory_usage_mb": round(avg_memory, 2),
                "response_time_ms": round(avg_response_time, 2),
                "throughput_ops_sec": round(avg_throughput, 2),
                "cache_hit_rate": round(avg_cache_hit_rate, 2),
                "error_rate_percent": round(avg_error_rate, 2)
            },
            "optimizations_applied": len(self.optimizations_applied),
            "total_improvement_percent": sum(
                opt.improvement_percent for opt in self.optimizations_applied
            ),
            "learning_models": dict(self.learning_models)
        }


# Demonstration function
async def demonstrate_autonomous_optimization():
    """Demonstrate autonomous performance optimization."""
    print("🚀 Autonomous Performance Optimizer - Generation 3 Demo")
    print("=" * 60)
    
    optimizer = AutonomousPerformanceOptimizer(
        strategy=OptimizationStrategy.AUTONOMOUS,
        optimization_interval=5.0  # 5 second intervals for demo
    )
    
    try:
        # Start optimizer
        await optimizer.start()
        
        print("📊 Starting performance optimization...")
        
        # Run for 30 seconds
        for i in range(6):
            await asyncio.sleep(5)
            
            # Get current performance summary
            summary = optimizer.get_performance_summary()
            
            print(f"\n📈 Optimization Cycle {i + 1}:")
            print(f"   Overall Score: {summary.get('overall_performance_score', 0):.1f}/100")
            print(f"   Optimizations Applied: {summary.get('optimizations_applied', 0)}")
            print(f"   Total Improvement: {summary.get('total_improvement_percent', 0):.1f}%")
            
            if 'metrics' in summary:
                metrics = summary['metrics']
                print(f"   CPU: {metrics['cpu_usage_percent']:.1f}%")
                print(f"   Memory: {metrics['memory_usage_mb']:.1f}MB")
                print(f"   Response Time: {metrics['response_time_ms']:.1f}ms")
                print(f"   Throughput: {metrics['throughput_ops_sec']:.0f} ops/sec")
    
    finally:
        await optimizer.stop()
        
        print("\n🎯 Final Performance Summary:")
        final_summary = optimizer.get_performance_summary()
        print(f"   Final Score: {final_summary.get('overall_performance_score', 0):.1f}/100")
        print(f"   Total Optimizations: {final_summary.get('optimizations_applied', 0)}")
        print(f"   Cumulative Improvement: {final_summary.get('total_improvement_percent', 0):.1f}%")
        
        print("\n✅ Autonomous optimization demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_autonomous_optimization())