"""Adaptive Performance Optimizer with ML-driven optimization."""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    BALANCED = "balanced"
    CUSTOM = "custom"

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0  # Importance weight for optimization

@dataclass
class OptimizationParameter:
    """Tunable optimization parameter."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    impact_score: float = 0.0  # Learned impact on performance
    adjustment_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)

class AdaptiveOptimizer:
    """ML-driven adaptive performance optimizer."""
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        optimization_interval: float = 60.0,
        learning_rate: float = 0.1
    ):
        self.strategy = strategy
        self.optimization_interval = optimization_interval
        self.learning_rate = learning_rate
        
        # Performance tracking
        self.metrics: Dict[str, List[PerformanceMetric]] = {}
        self.parameters: Dict[str, OptimizationParameter] = {}
        
        # Learning components
        self.performance_history: List[Dict[str, float]] = []
        self.optimization_episodes: List[Dict[str, Any]] = []
        
        # State management
        self.is_running = False
        self._optimization_task: Optional[asyncio.Task] = None
        self.baseline_performance: Optional[Dict[str, float]] = None
        
        # Register default optimization parameters
        self._register_default_parameters()
        
    def _register_default_parameters(self):
        """Register default optimization parameters."""
        
        # Network parameters
        self.register_parameter(
            "connection_pool_size",
            current_value=50.0,
            min_value=10.0,
            max_value=200.0,
            step_size=5.0
        )
        
        self.register_parameter(
            "request_timeout",
            current_value=30.0,
            min_value=5.0,
            max_value=120.0,
            step_size=5.0
        )
        
        self.register_parameter(
            "batch_size",
            current_value=32.0,
            min_value=1.0,
            max_value=128.0,
            step_size=1.0
        )
        
        # Cache parameters
        self.register_parameter(
            "cache_size_mb",
            current_value=256.0,
            min_value=64.0,
            max_value=2048.0,
            step_size=64.0
        )
        
        self.register_parameter(
            "cache_ttl_seconds",
            current_value=300.0,
            min_value=60.0,
            max_value=3600.0,
            step_size=60.0
        )
        
        # Processing parameters
        self.register_parameter(
            "worker_threads",
            current_value=4.0,
            min_value=1.0,
            max_value=16.0,
            step_size=1.0
        )
        
        self.register_parameter(
            "queue_size",
            current_value=1000.0,
            min_value=100.0,
            max_value=10000.0,
            step_size=100.0
        )
    
    def register_parameter(
        self,
        name: str,
        current_value: float,
        min_value: float,
        max_value: float,
        step_size: float
    ):
        """Register an optimization parameter."""
        parameter = OptimizationParameter(
            name=name,
            current_value=current_value,
            min_value=min_value,
            max_value=max_value,
            step_size=step_size
        )
        self.parameters[name] = parameter
        logger.info(f"Registered optimization parameter: {name}")
    
    def record_metric(self, name: str, value: float, weight: float = 1.0):
        """Record a performance metric."""
        metric = PerformanceMetric(name=name, value=value, weight=weight)
        
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append(metric)
        
        # Keep metrics history manageable
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-500:]
    
    def get_metric_stats(self, name: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
            
        recent_values = [m.value for m in self.metrics[name][-window_size:]]
        
        return {
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "std_dev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            "min": min(recent_values),
            "max": max(recent_values),
            "count": len(recent_values)
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score based on strategy."""
        if not self.metrics:
            return 0.0
            
        scores = []
        weights = []
        
        # Get recent metric values
        for name, metric_list in self.metrics.items():
            if not metric_list:
                continue
                
            recent_values = [m.value for m in metric_list[-10:]]  # Last 10 values
            avg_value = statistics.mean(recent_values)
            weight = metric_list[-1].weight
            
            # Normalize and weight based on strategy
            if self.strategy == OptimizationStrategy.THROUGHPUT:
                if "throughput" in name.lower() or "requests_per_second" in name.lower():
                    weight *= 3.0  # Boost throughput metrics
                elif "latency" in name.lower() or "response_time" in name.lower():
                    avg_value = 1.0 / max(avg_value, 0.001)  # Invert latency (lower is better)
                    
            elif self.strategy == OptimizationStrategy.LATENCY:
                if "latency" in name.lower() or "response_time" in name.lower():
                    avg_value = 1.0 / max(avg_value, 0.001)  # Invert latency
                    weight *= 3.0
                    
            elif self.strategy == OptimizationStrategy.RESOURCE_EFFICIENCY:
                if "cpu" in name.lower() or "memory" in name.lower():
                    avg_value = 1.0 / max(avg_value, 0.001)  # Invert resource usage
                    weight *= 2.0
                    
            scores.append(avg_value)
            weights.append(weight)
        
        if not scores:
            return 0.0
            
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return weighted_score
    
    def _calculate_parameter_impact(self, parameter_name: str) -> float:
        """Calculate the impact of a parameter on performance."""
        if parameter_name not in self.parameters:
            return 0.0
            
        parameter = self.parameters[parameter_name]
        
        if len(parameter.adjustment_history) < 2:
            return 0.0
            
        # Look for correlation between parameter changes and performance
        impacts = []
        
        for i in range(1, min(len(parameter.adjustment_history), 10)):
            prev_timestamp, prev_value = parameter.adjustment_history[i-1]
            curr_timestamp, curr_value = parameter.adjustment_history[i]
            
            # Find performance before and after the change
            before_perf = self._get_performance_at_time(prev_timestamp + 1)
            after_perf = self._get_performance_at_time(curr_timestamp + 30)  # 30s after change
            
            if before_perf and after_perf:
                param_change = curr_value - prev_value
                perf_change = after_perf - before_perf
                
                if abs(param_change) > 0.001:  # Avoid division by zero
                    impact = perf_change / param_change
                    impacts.append(impact)
        
        if impacts:
            return statistics.mean(impacts)
        return 0.0
    
    def _get_performance_at_time(self, timestamp: float) -> Optional[float]:
        """Get performance score around a specific timestamp."""
        # Find metrics around the timestamp (±5 seconds)
        relevant_metrics = {}
        
        for name, metric_list in self.metrics.items():
            values = [
                m.value for m in metric_list 
                if abs(m.timestamp - timestamp) <= 5.0
            ]
            if values:
                relevant_metrics[name] = statistics.mean(values)
        
        if not relevant_metrics:
            return None
            
        # Calculate a simple performance score
        total_score = 0.0
        count = 0
        
        for name, value in relevant_metrics.items():
            # Normalize based on typical ranges
            if "latency" in name.lower() or "response_time" in name.lower():
                # Lower latency is better
                score = max(0, 1000 - value) / 1000
            elif "cpu" in name.lower() or "memory" in name.lower():
                # Lower resource usage is better (up to a point)
                score = max(0, 100 - value) / 100
            else:
                # Higher values generally better for throughput, etc.
                score = min(value / 100, 1.0)
                
            total_score += score
            count += 1
        
        return total_score / max(count, 1)
    
    def _suggest_parameter_adjustment(self, parameter_name: str) -> Optional[float]:
        """Suggest adjustment for a parameter using gradient-based optimization."""
        if parameter_name not in self.parameters:
            return None
            
        parameter = self.parameters[parameter_name]
        
        # Update impact score
        parameter.impact_score = self._calculate_parameter_impact(parameter_name)
        
        # Use gradient-based adjustment
        if abs(parameter.impact_score) < 0.001:
            # Random exploration if no clear impact
            direction = 1 if time.time() % 2 < 1 else -1
            adjustment = direction * parameter.step_size * 0.5
        else:
            # Move in direction of positive impact
            direction = 1 if parameter.impact_score > 0 else -1
            adjustment = direction * parameter.step_size * self.learning_rate
        
        new_value = parameter.current_value + adjustment
        
        # Ensure within bounds
        new_value = max(parameter.min_value, min(parameter.max_value, new_value))
        
        if abs(new_value - parameter.current_value) < 0.001:
            return None  # No meaningful change
            
        return new_value
    
    async def _optimization_cycle(self):
        """Execute one optimization cycle."""
        try:
            current_performance = self._calculate_performance_score()
            
            # Establish baseline if needed
            if self.baseline_performance is None:
                self.baseline_performance = {"score": current_performance, "timestamp": time.time()}
                logger.info(f"Established performance baseline: {current_performance:.4f}")
                return
            
            # Record current state
            episode = {
                "timestamp": time.time(),
                "performance_score": current_performance,
                "parameters": {name: param.current_value for name, param in self.parameters.items()},
                "adjustments": {}
            }
            
            # Try optimizing one parameter per cycle
            parameters_to_try = list(self.parameters.keys())
            
            for param_name in parameters_to_try:
                suggested_value = self._suggest_parameter_adjustment(param_name)
                
                if suggested_value is not None:
                    old_value = self.parameters[param_name].current_value
                    self.parameters[param_name].current_value = suggested_value
                    
                    # Record the adjustment
                    self.parameters[param_name].adjustment_history.append(
                        (time.time(), suggested_value)
                    )
                    
                    # Keep history manageable
                    if len(self.parameters[param_name].adjustment_history) > 50:
                        self.parameters[param_name].adjustment_history = \
                            self.parameters[param_name].adjustment_history[-25:]
                    
                    episode["adjustments"][param_name] = {
                        "old_value": old_value,
                        "new_value": suggested_value,
                        "impact_score": self.parameters[param_name].impact_score
                    }
                    
                    logger.info(f"Adjusted {param_name}: {old_value:.2f} → {suggested_value:.2f}")
                    break  # One adjustment per cycle
            
            # Record episode
            self.optimization_episodes.append(episode)
            
            # Keep episodes history manageable
            if len(self.optimization_episodes) > 200:
                self.optimization_episodes = self.optimization_episodes[-100:]
            
            # Calculate improvement
            improvement = current_performance - self.baseline_performance["score"]
            if improvement > 0.1:  # Significant improvement
                logger.info(f"Performance improved by {improvement:.4f}")
                self.baseline_performance = {"score": current_performance, "timestamp": time.time()}
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        logger.info("Adaptive optimizer started")
        
        while self.is_running:
            try:
                await self._optimization_cycle()
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def start(self):
        """Start the adaptive optimizer."""
        if self.is_running:
            logger.warning("Adaptive optimizer is already running")
            return
            
        self.is_running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Adaptive optimizer started")
    
    async def stop(self):
        """Stop the adaptive optimizer."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Adaptive optimizer stopped")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary and insights."""
        current_performance = self._calculate_performance_score()
        
        summary = {
            "strategy": self.strategy.value,
            "current_performance": current_performance,
            "baseline_performance": self.baseline_performance,
            "improvement": (
                current_performance - self.baseline_performance["score"] 
                if self.baseline_performance else 0.0
            ),
            "parameters": {
                name: {
                    "current_value": param.current_value,
                    "impact_score": param.impact_score,
                    "adjustments_count": len(param.adjustment_history)
                }
                for name, param in self.parameters.items()
            },
            "metrics_summary": {
                name: self.get_metric_stats(name, 50)
                for name in self.metrics.keys()
            },
            "optimization_episodes": len(self.optimization_episodes)
        }
        
        return summary
    
    def set_strategy(self, strategy: OptimizationStrategy):
        """Change optimization strategy."""
        old_strategy = self.strategy
        self.strategy = strategy
        logger.info(f"Changed optimization strategy: {old_strategy.value} → {strategy.value}")
        
        # Reset baseline to re-evaluate with new strategy
        self.baseline_performance = None
    
    def get_parameter_value(self, name: str) -> Optional[float]:
        """Get current value of optimization parameter."""
        if name in self.parameters:
            return self.parameters[name].current_value
        return None
    
    def set_parameter_value(self, name: str, value: float):
        """Manually set parameter value."""
        if name in self.parameters:
            param = self.parameters[name]
            value = max(param.min_value, min(param.max_value, value))
            param.current_value = value
            param.adjustment_history.append((time.time(), value))
            logger.info(f"Manually set {name} to {value}")
    
    def export_optimization_data(self) -> Dict[str, Any]:
        """Export optimization data for analysis."""
        return {
            "parameters": {
                name: {
                    "config": {
                        "min_value": param.min_value,
                        "max_value": param.max_value,
                        "step_size": param.step_size
                    },
                    "current_value": param.current_value,
                    "impact_score": param.impact_score,
                    "history": param.adjustment_history[-50:]  # Last 50 adjustments
                }
                for name, param in self.parameters.items()
            },
            "episodes": self.optimization_episodes[-100:],  # Last 100 episodes
            "metrics": {
                name: [
                    {"value": m.value, "timestamp": m.timestamp, "weight": m.weight}
                    for m in metric_list[-100:]  # Last 100 measurements
                ]
                for name, metric_list in self.metrics.items()
            }
        }