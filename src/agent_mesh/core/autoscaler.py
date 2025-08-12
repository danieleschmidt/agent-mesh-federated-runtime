"""Auto-scaling and load balancing for Agent Mesh system."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from uuid import UUID, uuid4

import structlog


class ScalingDirection(Enum):
    """Scaling direction."""
    
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections" 
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CPU_BASED = "cpu_based"
    RESPONSE_TIME_BASED = "response_time_based"
    ADAPTIVE = "adaptive"


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    
    node_id: UUID
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def get_load_score(self) -> float:
        """Calculate overall load score."""
        # Weighted combination of metrics
        return (
            self.cpu_usage * 0.3 +
            self.memory_usage * 0.2 +
            min(self.active_connections / 100, 1.0) * 0.2 +
            min(self.requests_per_second / 1000, 1.0) * 0.15 +
            min(self.average_response_time / 1000, 1.0) * 0.1 +
            self.error_rate * 0.05
        )
    
    def is_overloaded(self, thresholds: Dict[str, float]) -> bool:
        """Check if node is overloaded."""
        return (
            self.cpu_usage > thresholds.get("cpu", 80.0) or
            self.memory_usage > thresholds.get("memory", 80.0) or
            self.error_rate > thresholds.get("error_rate", 5.0) or
            self.average_response_time > thresholds.get("response_time", 2000.0)
        )
    
    def is_underloaded(self, thresholds: Dict[str, float]) -> bool:
        """Check if node is underloaded."""
        return (
            self.cpu_usage < thresholds.get("cpu_low", 20.0) and
            self.memory_usage < thresholds.get("memory_low", 30.0) and
            self.error_rate < thresholds.get("error_rate_low", 1.0) and
            self.average_response_time < thresholds.get("response_time_low", 100.0)
        )


@dataclass 
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    
    min_nodes: int = 1
    max_nodes: int = 10
    target_cpu: float = 60.0
    target_memory: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 40.0
    cooldown_period: float = 300.0  # 5 minutes
    scale_up_steps: int = 2
    scale_down_steps: int = 1
    evaluation_period: float = 60.0
    
    # Advanced thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 85.0,
        "memory": 85.0,
        "cpu_low": 20.0,
        "memory_low": 30.0,
        "error_rate": 5.0,
        "error_rate_low": 1.0,
        "response_time": 2000.0,
        "response_time_low": 100.0
    })


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        """Initialize load balancer."""
        self.strategy = strategy
        self.logger = structlog.get_logger("load_balancer")
        
        # State for different strategies
        self._round_robin_index = 0
        self._weights: Dict[UUID, float] = {}
        self._connection_counts: Dict[UUID, int] = {}
        self._response_times: Dict[UUID, List[float]] = {}
        
        # Adaptive learning
        self._performance_history: Dict[UUID, List[Tuple[float, float]]] = {}  # (timestamp, score)
        self._strategy_performance: Dict[LoadBalancingStrategy, float] = {}
    
    def select_node(self, available_nodes: List[NodeMetrics]) -> Optional[NodeMetrics]:
        """Select best node based on strategy."""
        if not available_nodes:
            return None
        
        if len(available_nodes) == 1:
            return available_nodes[0]
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.CPU_BASED:
            return self._cpu_based_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME_BASED:
            return self._response_time_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_select(available_nodes)
        else:
            return available_nodes[0]
    
    def record_request_completion(
        self,
        node_id: UUID,
        response_time_ms: float,
        success: bool
    ) -> None:
        """Record request completion for learning."""
        # Update response times
        if node_id not in self._response_times:
            self._response_times[node_id] = []
        
        self._response_times[node_id].append(response_time_ms)
        
        # Keep only recent history
        if len(self._response_times[node_id]) > 100:
            self._response_times[node_id] = self._response_times[node_id][-50:]
        
        # Update performance history for adaptive strategy
        score = 1.0 if success else 0.0
        if response_time_ms > 0:
            score *= max(0.1, 1.0 - (response_time_ms / 1000.0))  # Penalize slow responses
        
        if node_id not in self._performance_history:
            self._performance_history[node_id] = []
        
        self._performance_history[node_id].append((time.time(), score))
        
        # Keep only recent history
        if len(self._performance_history[node_id]) > 100:
            self._performance_history[node_id] = self._performance_history[node_id][-50:]
        
        # Update connection count
        if node_id in self._connection_counts:
            self._connection_counts[node_id] = max(0, self._connection_counts[node_id] - 1)
    
    def _round_robin_select(self, nodes: List[NodeMetrics]) -> NodeMetrics:
        """Round-robin selection."""
        selected = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return selected
    
    def _least_connections_select(self, nodes: List[NodeMetrics]) -> NodeMetrics:
        """Select node with least connections."""
        return min(nodes, key=lambda n: self._connection_counts.get(n.node_id, 0))
    
    def _weighted_round_robin_select(self, nodes: List[NodeMetrics]) -> NodeMetrics:
        """Weighted round-robin based on performance."""
        # Calculate weights based on inverse load score
        total_weight = 0.0
        for node in nodes:
            weight = 1.0 / max(0.1, node.get_load_score())
            self._weights[node.node_id] = weight
            total_weight += weight
        
        if total_weight == 0:
            return nodes[0]
        
        # Select based on weights
        target = (self._round_robin_index % 100) / 100.0 * total_weight
        current_weight = 0.0
        
        for node in nodes:
            current_weight += self._weights.get(node.node_id, 1.0)
            if current_weight >= target:
                self._round_robin_index += 1
                return node
        
        return nodes[-1]
    
    def _cpu_based_select(self, nodes: List[NodeMetrics]) -> NodeMetrics:
        """Select node with lowest CPU usage."""
        return min(nodes, key=lambda n: n.cpu_usage)
    
    def _response_time_select(self, nodes: List[NodeMetrics]) -> NodeMetrics:
        """Select node with best response time."""
        def avg_response_time(node_id: UUID) -> float:
            times = self._response_times.get(node_id, [])
            return sum(times) / len(times) if times else 1000.0
        
        return min(nodes, key=lambda n: avg_response_time(n.node_id))
    
    def _adaptive_select(self, nodes: List[NodeMetrics]) -> NodeMetrics:
        """Adaptive selection based on historical performance."""
        def performance_score(node_id: UUID) -> float:
            history = self._performance_history.get(node_id, [])
            if not history:
                return 0.5  # Neutral score for new nodes
            
            # Weight recent performance more heavily
            total_score = 0.0
            total_weight = 0.0
            current_time = time.time()
            
            for timestamp, score in history[-20:]:  # Last 20 requests
                age = current_time - timestamp
                weight = max(0.1, 1.0 - (age / 3600))  # Decay over 1 hour
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.5
        
        # Combine performance score with current load
        def adaptive_score(node: NodeMetrics) -> float:
            perf = performance_score(node.node_id)
            load = 1.0 - node.get_load_score()  # Invert load (lower is better)
            return perf * 0.6 + load * 0.4
        
        return max(nodes, key=adaptive_score)


class AutoScaler:
    """Auto-scaler for managing node pool size."""
    
    def __init__(
        self,
        policy: ScalingPolicy,
        node_manager: Optional[Any] = None
    ):
        """Initialize auto-scaler."""
        self.policy = policy
        self.node_manager = node_manager
        self.logger = structlog.get_logger("auto_scaler")
        
        # Scaling state
        self._last_scaling_action = 0.0
        self._scaling_history: List[Tuple[float, ScalingDirection, int]] = []
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics collection
        self._metrics_history: List[Dict[str, float]] = []
    
    async def start(self) -> None:
        """Start auto-scaler."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Auto-scaler started")
    
    async def stop(self) -> None:
        """Stop auto-scaler."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Auto-scaler stopped")
    
    async def evaluate_scaling(self, node_metrics: List[NodeMetrics]) -> ScalingDirection:
        """Evaluate if scaling action is needed."""
        if not node_metrics:
            return ScalingDirection.MAINTAIN
        
        # Check cooldown period
        if time.time() - self._last_scaling_action < self.policy.cooldown_period:
            return ScalingDirection.MAINTAIN
        
        # Calculate aggregate metrics
        current_nodes = len(node_metrics)
        avg_cpu = sum(m.cpu_usage for m in node_metrics) / current_nodes
        avg_memory = sum(m.memory_usage for m in node_metrics) / current_nodes
        max_cpu = max(m.cpu_usage for m in node_metrics)
        max_memory = max(m.memory_usage for m in node_metrics)
        
        overloaded_nodes = sum(1 for m in node_metrics if m.is_overloaded(self.policy.thresholds))
        underloaded_nodes = sum(1 for m in node_metrics if m.is_underloaded(self.policy.thresholds))
        
        self.logger.debug("Scaling evaluation",
                         current_nodes=current_nodes,
                         avg_cpu=avg_cpu,
                         avg_memory=avg_memory,
                         overloaded_nodes=overloaded_nodes,
                         underloaded_nodes=underloaded_nodes)
        
        # Scale up conditions
        if (current_nodes < self.policy.max_nodes and
            (avg_cpu > self.policy.scale_up_threshold or
             avg_memory > self.policy.scale_up_threshold or
             max_cpu > 90.0 or max_memory > 90.0 or
             overloaded_nodes > current_nodes * 0.3)):
            return ScalingDirection.UP
        
        # Scale down conditions
        if (current_nodes > self.policy.min_nodes and
            avg_cpu < self.policy.scale_down_threshold and
            avg_memory < self.policy.scale_down_threshold and
            underloaded_nodes > current_nodes * 0.7):
            return ScalingDirection.DOWN
        
        return ScalingDirection.MAINTAIN
    
    async def execute_scaling(
        self,
        direction: ScalingDirection,
        current_nodes: int
    ) -> bool:
        """Execute scaling action."""
        if direction == ScalingDirection.MAINTAIN:
            return True
        
        if not self.node_manager:
            self.logger.warning("Cannot scale: no node manager configured")
            return False
        
        success = False
        
        try:
            if direction == ScalingDirection.UP:
                target_nodes = min(
                    current_nodes + self.policy.scale_up_steps,
                    self.policy.max_nodes
                )
                nodes_to_add = target_nodes - current_nodes
                
                if nodes_to_add > 0:
                    success = await self.node_manager.scale_up(nodes_to_add)
                    if success:
                        self.logger.info("Scaled up successfully",
                                       nodes_added=nodes_to_add,
                                       total_nodes=target_nodes)
            
            elif direction == ScalingDirection.DOWN:
                target_nodes = max(
                    current_nodes - self.policy.scale_down_steps,
                    self.policy.min_nodes
                )
                nodes_to_remove = current_nodes - target_nodes
                
                if nodes_to_remove > 0:
                    success = await self.node_manager.scale_down(nodes_to_remove)
                    if success:
                        self.logger.info("Scaled down successfully",
                                       nodes_removed=nodes_to_remove,
                                       total_nodes=target_nodes)
            
            if success:
                self._last_scaling_action = time.time()
                self._scaling_history.append(
                    (time.time(), direction, current_nodes)
                )
                
                # Keep history limited
                if len(self._scaling_history) > 100:
                    self._scaling_history = self._scaling_history[-50:]
        
        except Exception as e:
            self.logger.error("Scaling action failed",
                            direction=direction.value,
                            error=str(e))
            success = False
        
        return success
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        recent_actions = [
            (timestamp, direction.value, nodes)
            for timestamp, direction, nodes in self._scaling_history[-10:]
        ]
        
        return {
            "policy": {
                "min_nodes": self.policy.min_nodes,
                "max_nodes": self.policy.max_nodes,
                "scale_up_threshold": self.policy.scale_up_threshold,
                "scale_down_threshold": self.policy.scale_down_threshold,
                "cooldown_period": self.policy.cooldown_period
            },
            "last_scaling_action": self._last_scaling_action,
            "total_scaling_actions": len(self._scaling_history),
            "recent_actions": recent_actions,
            "cooldown_remaining": max(
                0, 
                self.policy.cooldown_period - (time.time() - self._last_scaling_action)
            )
        }
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                if self.node_manager:
                    # Get current metrics
                    node_metrics = await self.node_manager.get_node_metrics()
                    
                    if node_metrics:
                        # Evaluate scaling
                        scaling_direction = await self.evaluate_scaling(node_metrics)
                        
                        if scaling_direction != ScalingDirection.MAINTAIN:
                            await self.execute_scaling(scaling_direction, len(node_metrics))
                        
                        # Store metrics for analysis
                        aggregate_metrics = {
                            "timestamp": time.time(),
                            "node_count": len(node_metrics),
                            "avg_cpu": sum(m.cpu_usage for m in node_metrics) / len(node_metrics),
                            "avg_memory": sum(m.memory_usage for m in node_metrics) / len(node_metrics),
                            "max_cpu": max(m.cpu_usage for m in node_metrics),
                            "max_memory": max(m.memory_usage for m in node_metrics)
                        }
                        
                        self._metrics_history.append(aggregate_metrics)
                        
                        # Keep history limited
                        if len(self._metrics_history) > 1000:
                            self._metrics_history = self._metrics_history[-500:]
                
                await asyncio.sleep(self.policy.evaluation_period)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Auto-scaler monitoring error", error=str(e))
                await asyncio.sleep(self.policy.evaluation_period)


class ResourceManager:
    """High-level resource management combining load balancing and auto-scaling."""
    
    def __init__(
        self,
        node_id: UUID,
        scaling_policy: Optional[ScalingPolicy] = None,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        """Initialize resource manager."""
        self.node_id = node_id
        self.logger = structlog.get_logger("resource_manager", node_id=str(node_id))
        
        # Components
        self.load_balancer = LoadBalancer(load_balancing_strategy)
        self.auto_scaler = AutoScaler(scaling_policy or ScalingPolicy())
        
        # Node tracking
        self.active_nodes: Dict[UUID, NodeMetrics] = {}
        self.node_health: Dict[UUID, bool] = {}
        
        # Request routing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self._processing_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    async def start(self) -> None:
        """Start resource manager."""
        await self.auto_scaler.start()
        
        # Start request processing
        for i in range(3):  # 3 processing workers
            task = asyncio.create_task(self._request_processor())
            self._processing_tasks.append(task)
        
        self.logger.info("Resource manager started")
    
    async def stop(self) -> None:
        """Stop resource manager."""
        await self.auto_scaler.stop()
        
        # Stop request processing
        for task in self._processing_tasks:
            task.cancel()
        
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        self._processing_tasks.clear()
        self.logger.info("Resource manager stopped")
    
    async def route_request(
        self,
        request: Any,
        timeout: float = 30.0
    ) -> Any:
        """Route request to best available node."""
        # Get healthy nodes
        healthy_nodes = [
            metrics for node_id, metrics in self.active_nodes.items()
            if self.node_health.get(node_id, False)
        ]
        
        if not healthy_nodes:
            raise RuntimeError("No healthy nodes available")
        
        # Select best node
        selected_node = self.load_balancer.select_node(healthy_nodes)
        if not selected_node:
            raise RuntimeError("Failed to select node")
        
        # Route request
        try:
            start_time = time.time()
            response = await self._send_request_to_node(selected_node, request, timeout)
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Record success
            self.load_balancer.record_request_completion(
                selected_node.node_id, response_time, True
            )
            self.successful_requests += 1
            
            return response
            
        except Exception as e:
            # Record failure
            self.load_balancer.record_request_completion(
                selected_node.node_id, timeout * 1000, False
            )
            self.failed_requests += 1
            raise
        finally:
            self.total_requests += 1
    
    def update_node_metrics(self, metrics: NodeMetrics) -> None:
        """Update metrics for a node."""
        self.active_nodes[metrics.node_id] = metrics
        
        # Simple health check based on metrics
        is_healthy = not metrics.is_overloaded(self.auto_scaler.policy.thresholds)
        self.node_health[metrics.node_id] = is_healthy
    
    def remove_node(self, node_id: UUID) -> None:
        """Remove node from active pool."""
        if node_id in self.active_nodes:
            del self.active_nodes[node_id]
        
        if node_id in self.node_health:
            del self.node_health[node_id]
        
        self.logger.info("Node removed from pool", node_id=str(node_id))
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        healthy_nodes = sum(1 for is_healthy in self.node_health.values() if is_healthy)
        total_nodes = len(self.active_nodes)
        
        success_rate = (
            self.successful_requests / max(self.total_requests, 1) * 100
        )
        
        return {
            "load_balancer": {
                "strategy": self.load_balancer.strategy.value,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate
            },
            "nodes": {
                "total": total_nodes,
                "healthy": healthy_nodes,
                "unhealthy": total_nodes - healthy_nodes,
                "health_percentage": healthy_nodes / max(total_nodes, 1) * 100
            },
            "auto_scaler": self.auto_scaler.get_scaling_statistics()
        }
    
    async def _send_request_to_node(
        self,
        node: NodeMetrics,
        request: Any,
        timeout: float
    ) -> Any:
        """Send request to specific node."""
        # This would integrate with the actual network layer
        # For now, simulate request processing
        await asyncio.sleep(0.1)  # Simulate network latency
        return {"status": "success", "node_id": str(node.node_id)}
    
    async def _request_processor(self) -> None:
        """Background request processor."""
        while True:
            try:
                # This would handle queued requests in a real implementation
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Request processor error", error=str(e))


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager(node_id: Optional[UUID] = None) -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager(node_id or uuid4())
    return _resource_manager


async def initialize_resource_management(
    node_id: UUID,
    scaling_policy: Optional[ScalingPolicy] = None
) -> ResourceManager:
    """Initialize global resource management."""
    global _resource_manager
    _resource_manager = ResourceManager(node_id, scaling_policy)
    await _resource_manager.start()
    return _resource_manager