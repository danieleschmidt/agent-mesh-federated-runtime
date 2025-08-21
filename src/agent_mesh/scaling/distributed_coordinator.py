"""Distributed coordination system for large-scale deployments."""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)

class NodeRole(Enum):
    """Node roles in distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    REPLICA = "replica"
    OBSERVER = "observer"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"

@dataclass
class DistributedNode:
    """Distributed system node."""
    node_id: str
    address: str
    port: int
    role: NodeRole
    capabilities: List[str] = field(default_factory=list)
    current_load: float = 0.0
    max_capacity: float = 100.0
    last_heartbeat: float = field(default_factory=time.time)
    health_score: float = 1.0
    active_tasks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DistributedTask:
    """Distributed task definition."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    requirements: List[str] = field(default_factory=list)
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ClusterMetrics:
    """Cluster-wide metrics."""
    total_nodes: int
    active_nodes: int
    total_capacity: float
    used_capacity: float
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_task_time: float
    throughput: float  # tasks per second
    timestamp: float = field(default_factory=time.time)

class DistributedCoordinator:
    """High-performance distributed coordination system."""
    
    def __init__(
        self,
        node_id: str,
        coordination_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        self.node_id = node_id
        self.coordination_interval = coordination_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.load_balancing_strategy = load_balancing_strategy
        
        # Cluster state
        self.nodes: Dict[str, DistributedNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[DistributedTask] = []
        self.completed_tasks: List[DistributedTask] = []
        
        # Load balancing
        self.round_robin_index = 0
        self.node_weights: Dict[str, float] = {}
        self.load_history: Dict[str, List[Tuple[float, float]]] = {}  # node_id -> [(timestamp, load)]
        
        # Performance optimization
        self.task_processors: Dict[str, Callable] = {}
        self.batch_processing_enabled = True
        self.batch_size = 10
        self.task_parallelism = 5
        
        # Metrics and monitoring
        self.cluster_metrics: List[ClusterMetrics] = []
        self.performance_counters: Dict[str, int] = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "nodes_added": 0,
            "nodes_removed": 0,
            "load_balancing_decisions": 0
        }
        
        # State management
        self.is_running = False
        self._coordination_task: Optional[asyncio.Task] = None
        self._task_processing_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Register self as coordinator node
        self._register_self()
    
    def _register_self(self):
        """Register self as coordinator node."""
        self.nodes[self.node_id] = DistributedNode(
            node_id=self.node_id,
            address="localhost",
            port=8080,
            role=NodeRole.COORDINATOR,
            capabilities=["coordination", "task_scheduling", "load_balancing"],
            max_capacity=1000.0
        )
    
    def register_node(
        self,
        node_id: str,
        address: str,
        port: int,
        role: NodeRole,
        capabilities: List[str],
        max_capacity: float = 100.0
    ) -> bool:
        """Register a new node in the cluster."""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already registered")
            return False
        
        node = DistributedNode(
            node_id=node_id,
            address=address,
            port=port,
            role=role,
            capabilities=capabilities,
            max_capacity=max_capacity
        )
        
        self.nodes[node_id] = node
        self.node_weights[node_id] = 1.0  # Default weight
        self.load_history[node_id] = []
        
        self.performance_counters["nodes_added"] += 1
        logger.info(f"Registered node {node_id} ({role.value}) with capacity {max_capacity}")
        
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the cluster."""
        if node_id not in self.nodes:
            return False
        
        # Reassign tasks from this node
        tasks_to_reassign = [
            task for task in self.tasks.values()
            if task.assigned_node == node_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]
        ]
        
        for task in tasks_to_reassign:
            task.assigned_node = None
            task.status = TaskStatus.PENDING
            self.task_queue.append(task)
            logger.info(f"Reassigned task {task.task_id} due to node removal")
        
        # Remove node
        del self.nodes[node_id]
        if node_id in self.node_weights:
            del self.node_weights[node_id]
        if node_id in self.load_history:
            del self.load_history[node_id]
        
        self.performance_counters["nodes_removed"] += 1
        logger.info(f"Unregistered node {node_id}")
        
        return True
    
    def update_node_heartbeat(
        self,
        node_id: str,
        current_load: float,
        active_tasks: int,
        health_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update node heartbeat and metrics."""
        if node_id not in self.nodes:
            logger.warning(f"Unknown node heartbeat: {node_id}")
            return
        
        node = self.nodes[node_id]
        current_time = time.time()
        
        # Update node state
        node.current_load = current_load
        node.active_tasks = active_tasks
        node.health_score = health_score
        node.last_heartbeat = current_time
        
        if metadata:
            node.metadata.update(metadata)
        
        # Record load history
        self.load_history[node_id].append((current_time, current_load))
        
        # Keep history manageable
        if len(self.load_history[node_id]) > 100:
            self.load_history[node_id] = self.load_history[node_id][-50:]
        
        # Update adaptive weights
        self._update_adaptive_weights(node_id)
    
    def _update_adaptive_weights(self, node_id: str):
        """Update adaptive weights for load balancing."""
        node = self.nodes[node_id]
        
        # Weight based on inverse of load and health score
        load_factor = max(0.1, 1.0 - (node.current_load / node.max_capacity))
        health_factor = node.health_score
        capacity_factor = node.max_capacity / 100.0  # Normalize capacity
        
        self.node_weights[node_id] = load_factor * health_factor * capacity_factor
    
    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        requirements: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> str:
        """Submit a new task for distributed execution."""
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            requirements=requirements or [],
            max_retries=max_retries
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task)
        
        # Sort queue by priority (higher priority first)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        logger.info(f"Submitted task {task_id} ({task_type}) with priority {priority}")
        return task_id
    
    def _select_node_for_task(self, task: DistributedTask) -> Optional[str]:
        """Select best node for task execution using load balancing strategy."""
        available_nodes = [
            node for node in self.nodes.values()
            if (node.role in [NodeRole.WORKER, NodeRole.REPLICA] and
                node.current_load < node.max_capacity * 0.9 and  # 90% capacity limit
                time.time() - node.last_heartbeat < self.heartbeat_timeout and
                all(req in node.capabilities for req in task.requirements))
        ]
        
        if not available_nodes:
            return None
        
        self.performance_counters["load_balancing_decisions"] += 1
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_node = available_nodes[self.round_robin_index % len(available_nodes)]
            self.round_robin_index += 1
            return selected_node.node_id
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_node = min(available_nodes, key=lambda n: n.active_tasks)
            return selected_node.node_id
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_BASED:
            # Select node with most available capacity
            selected_node = min(available_nodes, key=lambda n: n.current_load / n.max_capacity)
            return selected_node.node_id
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted selection based on node weights
            total_weight = sum(self.node_weights.get(n.node_id, 1.0) for n in available_nodes)
            
            if total_weight <= 0:
                selected_node = available_nodes[0]
            else:
                random_value = random.uniform(0, total_weight)
                cumulative_weight = 0
                selected_node = available_nodes[0]
                
                for node in available_nodes:
                    cumulative_weight += self.node_weights.get(node.node_id, 1.0)
                    if random_value <= cumulative_weight:
                        selected_node = node
                        break
            
            return selected_node.node_id
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.ADAPTIVE:
            # Combined scoring approach
            best_score = -1
            selected_node = None
            
            for node in available_nodes:
                # Score based on multiple factors
                load_score = 1.0 - (node.current_load / node.max_capacity)
                health_score = node.health_score
                task_score = 1.0 - (node.active_tasks / max(node.max_capacity / 10, 1))
                
                # Historical performance
                recent_load = self._get_recent_average_load(node.node_id)
                stability_score = 1.0 - min(recent_load / node.max_capacity, 1.0)
                
                combined_score = (
                    load_score * 0.3 +
                    health_score * 0.2 +
                    task_score * 0.2 +
                    stability_score * 0.3
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    selected_node = node
            
            return selected_node.node_id if selected_node else None
        
        else:
            # Default to first available node
            return available_nodes[0].node_id
    
    def _get_recent_average_load(self, node_id: str, window_minutes: int = 5) -> float:
        """Get recent average load for a node."""
        if node_id not in self.load_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        
        recent_loads = [
            load for timestamp, load in self.load_history[node_id]
            if timestamp > cutoff_time
        ]
        
        if not recent_loads:
            return 0.0
        
        return sum(recent_loads) / len(recent_loads)
    
    async def _assign_tasks(self):
        """Assign pending tasks to available nodes."""
        if not self.task_queue:
            return
        
        # Process tasks in batches for better performance
        batch_size = min(self.batch_size, len(self.task_queue))
        tasks_to_process = self.task_queue[:batch_size]
        
        assigned_tasks = []
        
        for task in tasks_to_process:
            node_id = self._select_node_for_task(task)
            
            if node_id:
                # Assign task
                task.assigned_node = node_id
                task.status = TaskStatus.ASSIGNED
                task.started_at = time.time()
                
                # Update node load
                self.nodes[node_id].active_tasks += 1
                
                assigned_tasks.append(task)
                logger.debug(f"Assigned task {task.task_id} to node {node_id}")
            else:
                # No suitable node available, task remains pending
                break
        
        # Remove assigned tasks from queue
        for task in assigned_tasks:
            self.task_queue.remove(task)
        
        # Simulate task execution (in real implementation, send to actual nodes)
        if assigned_tasks:
            asyncio.create_task(self._simulate_task_execution(assigned_tasks))
    
    async def _simulate_task_execution(self, tasks: List[DistributedTask]):
        """Simulate task execution (placeholder for real implementation)."""
        for task in tasks:
            try:
                # Simulate task processing time
                processing_time = random.uniform(1.0, 5.0)
                await asyncio.sleep(processing_time)
                
                # Simulate success/failure
                if random.random() < 0.9:  # 90% success rate
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    task.result = {"status": "success", "processing_time": processing_time}
                    self.performance_counters["tasks_processed"] += 1
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Simulated task failure"
                    task.retry_count += 1
                    self.performance_counters["tasks_failed"] += 1
                    
                    # Retry if not exceeded max retries
                    if task.retry_count < task.max_retries:
                        task.status = TaskStatus.PENDING
                        task.assigned_node = None
                        self.task_queue.append(task)
                
                # Update node load
                if task.assigned_node:
                    self.nodes[task.assigned_node].active_tasks -= 1
                
                # Move completed tasks to history
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    self.completed_tasks.append(task)
                    
                    # Keep completed tasks history manageable
                    if len(self.completed_tasks) > 1000:
                        self.completed_tasks = self.completed_tasks[-500:]
                
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {e}")
                task.status = TaskStatus.FAILED
                task.error = str(e)
    
    def _check_node_health(self):
        """Check health of all nodes and remove stale ones."""
        current_time = time.time()
        stale_nodes = []
        
        for node_id, node in self.nodes.items():
            if (node_id != self.node_id and  # Don't check coordinator node
                current_time - node.last_heartbeat > self.heartbeat_timeout):
                stale_nodes.append(node_id)
        
        for node_id in stale_nodes:
            logger.warning(f"Node {node_id} is stale (no heartbeat), removing...")
            self.unregister_node(node_id)
    
    def _calculate_cluster_metrics(self) -> ClusterMetrics:
        """Calculate current cluster metrics."""
        active_nodes = [
            node for node in self.nodes.values()
            if time.time() - node.last_heartbeat < self.heartbeat_timeout
        ]
        
        total_capacity = sum(node.max_capacity for node in active_nodes)
        used_capacity = sum(node.current_load for node in active_nodes)
        
        pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        completed_tasks = len([t for t in self.completed_tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.completed_tasks if t.status == TaskStatus.FAILED])
        
        # Calculate average task time
        completed_with_time = [
            t for t in self.completed_tasks
            if t.completed_at and t.started_at and t.status == TaskStatus.COMPLETED
        ]
        
        if completed_with_time:
            avg_task_time = sum(
                t.completed_at - t.started_at for t in completed_with_time
            ) / len(completed_with_time)
        else:
            avg_task_time = 0.0
        
        # Calculate throughput (tasks per second)
        recent_completed = [
            t for t in self.completed_tasks
            if t.completed_at and time.time() - t.completed_at <= 60  # Last minute
        ]
        throughput = len(recent_completed) / 60.0  # Tasks per second
        
        return ClusterMetrics(
            total_nodes=len(self.nodes),
            active_nodes=len(active_nodes),
            total_capacity=total_capacity,
            used_capacity=used_capacity,
            pending_tasks=pending_tasks,
            running_tasks=running_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            average_task_time=avg_task_time,
            throughput=throughput
        )
    
    async def _coordination_loop(self):
        """Main coordination loop."""
        logger.info("Distributed coordination started")
        
        while self.is_running:
            try:
                # Check node health
                self._check_node_health()
                
                # Assign pending tasks
                await self._assign_tasks()
                
                await asyncio.sleep(self.coordination_interval)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(self.coordination_interval)
    
    async def _metrics_loop(self):
        """Metrics collection loop."""
        while self.is_running:
            try:
                metrics = self._calculate_cluster_metrics()
                self.cluster_metrics.append(metrics)
                
                # Keep metrics history manageable
                if len(self.cluster_metrics) > 288:  # 24 hours of 5-minute intervals
                    self.cluster_metrics = self.cluster_metrics[-144:]  # Keep 12 hours
                
                await asyncio.sleep(300)  # Collect metrics every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(300)
    
    async def start(self):
        """Start the distributed coordinator."""
        if self.is_running:
            logger.warning("Distributed coordinator is already running")
            return
        
        self.is_running = True
        
        # Start coordination and metrics tasks
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("Distributed coordinator started")
    
    async def stop(self):
        """Stop the distributed coordinator."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all tasks
        for task in [self._coordination_task, self._metrics_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Distributed coordinator stopped")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        current_metrics = self._calculate_cluster_metrics()
        
        return {
            "cluster_id": self.node_id,
            "nodes": {
                node_id: {
                    "role": node.role.value,
                    "address": f"{node.address}:{node.port}",
                    "capabilities": node.capabilities,
                    "current_load": node.current_load,
                    "max_capacity": node.max_capacity,
                    "utilization": node.current_load / node.max_capacity,
                    "active_tasks": node.active_tasks,
                    "health_score": node.health_score,
                    "last_heartbeat": node.last_heartbeat
                }
                for node_id, node in self.nodes.items()
            },
            "metrics": {
                "total_nodes": current_metrics.total_nodes,
                "active_nodes": current_metrics.active_nodes,
                "cluster_utilization": current_metrics.used_capacity / max(current_metrics.total_capacity, 1),
                "pending_tasks": current_metrics.pending_tasks,
                "running_tasks": current_metrics.running_tasks,
                "completed_tasks": current_metrics.completed_tasks,
                "failed_tasks": current_metrics.failed_tasks,
                "average_task_time": current_metrics.average_task_time,
                "throughput": current_metrics.throughput
            },
            "performance": self.performance_counters.copy(),
            "load_balancing_strategy": self.load_balancing_strategy.value
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
        else:
            # Check completed tasks
            task = next((t for t in self.completed_tasks if t.task_id == task_id), None)
        
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status.value,
            "assigned_node": task.assigned_node,
            "priority": task.priority,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "execution_time": (
                (task.completed_at - task.started_at) 
                if task.started_at and task.completed_at 
                else None
            ),
            "retry_count": task.retry_count,
            "result": task.result,
            "error": task.error
        }
    
    def export_cluster_report(self) -> Dict[str, Any]:
        """Export comprehensive cluster report."""
        return {
            "cluster_status": self.get_cluster_status(),
            "recent_metrics": [
                {
                    "timestamp": m.timestamp,
                    "total_nodes": m.total_nodes,
                    "active_nodes": m.active_nodes,
                    "utilization": m.used_capacity / max(m.total_capacity, 1),
                    "throughput": m.throughput,
                    "average_task_time": m.average_task_time
                }
                for m in self.cluster_metrics[-24:]  # Last 24 metric points
            ],
            "node_performance": {
                node_id: {
                    "average_load": self._get_recent_average_load(node_id),
                    "weight": self.node_weights.get(node_id, 1.0),
                    "load_history": self.load_history.get(node_id, [])[-20:]  # Last 20 points
                }
                for node_id in self.nodes.keys()
            },
            "task_summary": {
                "total_submitted": len(self.tasks) + len(self.completed_tasks),
                "currently_pending": len(self.task_queue),
                "currently_running": len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
                "total_completed": len([t for t in self.completed_tasks if t.status == TaskStatus.COMPLETED]),
                "total_failed": len([t for t in self.completed_tasks if t.status == TaskStatus.FAILED])
            }
        }