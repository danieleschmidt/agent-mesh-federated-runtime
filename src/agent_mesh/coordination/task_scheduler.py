"""Task scheduling and management for distributed systems.

This module implements intelligent task scheduling, load balancing,
and resource optimization for the Agent Mesh network.
"""

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Status of tasks in the system."""
    
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""
    
    LOW = 1
    NORMAL = 3
    HIGH = 5
    URGENT = 7
    CRITICAL = 9


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    ROUND_ROBIN = "round_robin"  # Round-robin assignment
    LOAD_BALANCED = "load_balanced"  # Load-aware scheduling
    CAPABILITY_MATCHED = "capability_matched"  # Capability-based matching
    DEADLINE_AWARE = "deadline_aware"  # Deadline-driven scheduling


@dataclass
class ResourceRequirements:
    """Resource requirements for task execution."""
    
    cpu_cores: float = 1.0
    memory_mb: float = 512.0
    storage_mb: float = 100.0
    gpu_memory_mb: float = 0.0
    network_bandwidth_mbps: float = 1.0
    execution_time_estimate: float = 60.0  # seconds


@dataclass
class Task:
    """Task definition for distributed execution."""
    
    task_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    
    # Task properties
    task_type: str = "generic"
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # Requirements and constraints
    required_skills: Set[str] = field(default_factory=set)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Timing
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    estimated_duration: float = 60.0  # seconds
    max_execution_time: float = 300.0  # seconds
    
    # Execution context
    payload: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[UUID] = field(default_factory=list)
    
    # Assignment and execution
    assigned_agent: Optional[UUID] = None
    assigned_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results and metrics
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Retry and recovery
    retry_count: int = 0
    max_retries: int = 3
    
    def dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": str(self.task_id),
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "required_skills": list(self.required_skills),
            "resource_requirements": {
                "cpu_cores": self.resource_requirements.cpu_cores,
                "memory_mb": self.resource_requirements.memory_mb,
                "storage_mb": self.resource_requirements.storage_mb,
                "gpu_memory_mb": self.resource_requirements.gpu_memory_mb,
                "network_bandwidth_mbps": self.resource_requirements.network_bandwidth_mbps,
                "execution_time_estimate": self.resource_requirements.execution_time_estimate
            },
            "created_at": self.created_at,
            "deadline": self.deadline,
            "estimated_duration": self.estimated_duration,
            "max_execution_time": self.max_execution_time,
            "payload": self.payload,
            "dependencies": [str(dep) for dep in self.dependencies],
            "assigned_agent": str(self.assigned_agent) if self.assigned_agent else None,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error_message": self.error_message,
            "execution_metrics": self.execution_metrics,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


@dataclass
class AgentResource:
    """Resource availability of an agent."""
    
    agent_id: UUID
    available_cpu_cores: float
    available_memory_mb: float
    available_storage_mb: float
    available_gpu_memory_mb: float
    network_bandwidth_mbps: float
    current_load: float = 0.0  # 0.0 to 1.0
    skills: Set[str] = field(default_factory=set)
    reliability_score: float = 1.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class SchedulingDecision:
    """Result of scheduling decision."""
    
    task_id: UUID
    assigned_agent: Optional[UUID]
    estimated_start_time: float
    estimated_completion_time: float
    scheduling_score: float
    reasoning: str = ""


class TaskQueue:
    """Priority queue for task management."""
    
    def __init__(self):
        self._heap: List[Tuple[int, float, Task]] = []
        self._task_map: Dict[UUID, Task] = {}
        self._counter = 0  # For tie-breaking
    
    def add_task(self, task: Task) -> None:
        """Add task to queue."""
        # Priority queue with negative priority for max-heap behavior
        priority_score = -task.priority.value
        
        # Add deadline urgency
        if task.deadline:
            time_to_deadline = task.deadline - time.time()
            if time_to_deadline > 0:
                urgency = max(0, 1.0 - (time_to_deadline / 3600))  # 1 hour baseline
                priority_score -= urgency
        
        heapq.heappush(self._heap, (priority_score, self._counter, task))
        self._task_map[task.task_id] = task
        self._counter += 1
    
    def get_next_task(self) -> Optional[Task]:
        """Get next task from queue."""
        while self._heap:
            _, _, task = heapq.heappop(self._heap)
            if task.task_id in self._task_map:
                del self._task_map[task.task_id]
                return task
        return None
    
    def remove_task(self, task_id: UUID) -> Optional[Task]:
        """Remove task from queue."""
        task = self._task_map.pop(task_id, None)
        if task:
            # Mark as removed (will be skipped in get_next_task)
            task.status = TaskStatus.CANCELLED
        return task
    
    def get_task(self, task_id: UUID) -> Optional[Task]:
        """Get task by ID."""
        return self._task_map.get(task_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [task for task in self._task_map.values() 
                if task.status == TaskStatus.PENDING]
    
    def size(self) -> int:
        """Get queue size."""
        return len(self._task_map)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._task_map) == 0


class LoadBalancer:
    """Load balancing for task distribution."""
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.agent_loads: Dict[UUID, float] = {}
        self.round_robin_index = 0
    
    def select_agent(
        self, 
        available_agents: List[AgentResource],
        task: Task
    ) -> Optional[UUID]:
        """Select best agent for task execution."""
        if not available_agents:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(available_agents)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(available_agents)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_selection(available_agents, task)
        elif self.strategy == "capability_match":
            return self._capability_match_selection(available_agents, task)
        else:
            return available_agents[0].agent_id  # Default to first
    
    def _round_robin_selection(self, agents: List[AgentResource]) -> UUID:
        """Simple round-robin selection."""
        selected = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1
        return selected.agent_id
    
    def _least_loaded_selection(self, agents: List[AgentResource]) -> UUID:
        """Select least loaded agent."""
        return min(agents, key=lambda a: a.current_load).agent_id
    
    def _weighted_round_robin_selection(
        self, 
        agents: List[AgentResource], 
        task: Task
    ) -> UUID:
        """Weighted selection based on capacity and current load."""
        scores = []
        
        for agent in agents:
            # Calculate capacity score
            capacity_score = (
                agent.available_cpu_cores * 0.3 +
                agent.available_memory_mb / 1000 * 0.2 +
                agent.network_bandwidth_mbps / 100 * 0.1 +
                agent.reliability_score * 0.4
            )
            
            # Adjust for current load
            load_penalty = agent.current_load * 2.0
            final_score = max(0.1, capacity_score - load_penalty)
            
            scores.append((agent.agent_id, final_score))
        
        # Weighted random selection
        import random
        total_score = sum(score for _, score in scores)
        
        if total_score == 0:
            return agents[0].agent_id
        
        rand_val = random.uniform(0, total_score)
        cumulative = 0
        
        for agent_id, score in scores:
            cumulative += score
            if cumulative >= rand_val:
                return agent_id
        
        return scores[-1][0]  # Fallback
    
    def _capability_match_selection(
        self, 
        agents: List[AgentResource], 
        task: Task
    ) -> UUID:
        """Select agent based on capability matching."""
        if not task.required_skills:
            return self._least_loaded_selection(agents)
        
        # Score agents based on skill match
        scores = []
        
        for agent in agents:
            skill_match = len(task.required_skills & agent.skills) / len(task.required_skills)
            resource_adequacy = self._calculate_resource_adequacy(agent, task)
            
            total_score = skill_match * 0.6 + resource_adequacy * 0.4
            scores.append((agent.agent_id, total_score))
        
        # Select best match
        best_agent, _ = max(scores, key=lambda x: x[1])
        return best_agent
    
    def _calculate_resource_adequacy(
        self, 
        agent: AgentResource, 
        task: Task
    ) -> float:
        """Calculate how well agent resources match task requirements."""
        req = task.resource_requirements
        
        cpu_ratio = min(1.0, agent.available_cpu_cores / req.cpu_cores)
        memory_ratio = min(1.0, agent.available_memory_mb / req.memory_mb)
        storage_ratio = min(1.0, agent.available_storage_mb / req.storage_mb)
        
        if req.gpu_memory_mb > 0:
            gpu_ratio = min(1.0, agent.available_gpu_memory_mb / req.gpu_memory_mb)
            return (cpu_ratio + memory_ratio + storage_ratio + gpu_ratio) / 4.0
        else:
            return (cpu_ratio + memory_ratio + storage_ratio) / 3.0


class TaskScheduler:
    """
    Intelligent task scheduler for distributed agent mesh.
    
    Manages task queuing, scheduling, load balancing, and execution
    coordination across multiple agents in the network.
    """
    
    def __init__(
        self,
        scheduler_id: UUID,
        strategy: SchedulingStrategy = SchedulingStrategy.LOAD_BALANCED,
        max_concurrent_tasks: int = 100
    ):
        """
        Initialize task scheduler.
        
        Args:
            scheduler_id: Unique scheduler identifier
            strategy: Scheduling strategy to use
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.scheduler_id = scheduler_id
        self.strategy = strategy
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.logger = structlog.get_logger("task_scheduler", 
                                         scheduler_id=str(scheduler_id))
        
        # Task management
        self.task_queue = TaskQueue()
        self.running_tasks: Dict[UUID, Task] = {}
        self.completed_tasks: Dict[UUID, Task] = {}
        self.failed_tasks: Dict[UUID, Task] = {}
        
        # Agent and resource management
        self.available_agents: Dict[UUID, AgentResource] = {}
        self.load_balancer = LoadBalancer("weighted_round_robin")
        
        # Scheduling state
        self.scheduling_enabled = False
        self.scheduling_interval = 1.0  # seconds
        self._scheduling_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics and statistics
        self.scheduling_metrics = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_wait_time": 0.0,
            "average_execution_time": 0.0,
            "resource_utilization": 0.0
        }
        
        # Task execution handlers
        self._task_handlers: Dict[str, Callable] = {}
    
    async def start(self) -> None:
        """Start the task scheduler."""
        self.logger.info("Starting task scheduler", strategy=self.strategy.value)
        
        self.scheduling_enabled = True
        
        # Start background tasks
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Task scheduler started")
    
    async def stop(self) -> None:
        """Stop the task scheduler."""
        self.logger.info("Stopping task scheduler")
        
        self.scheduling_enabled = False
        
        # Stop background tasks
        if self._scheduling_task:
            self._scheduling_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Wait for tasks to complete
        background_tasks = [t for t in [self._scheduling_task, self._monitoring_task] if t]
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        
        self.logger.info("Task scheduler stopped")
    
    async def submit_task(self, task: Task) -> UUID:
        """Submit a task for scheduling."""
        task.status = TaskStatus.QUEUED
        self.task_queue.add_task(task)
        
        self.logger.info("Task submitted",
                        task_id=str(task.task_id),
                        task_type=task.task_type,
                        priority=task.priority.value)
        
        return task.task_id
    
    async def cancel_task(self, task_id: UUID) -> bool:
        """Cancel a pending or running task."""
        # Check if task is in queue
        task = self.task_queue.remove_task(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            self.logger.info("Queued task cancelled", task_id=str(task_id))
            return True
        
        # Check if task is running
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            # In a real implementation, we would send cancellation signal to agent
            self.logger.info("Running task cancelled", task_id=str(task_id))
            return True
        
        return False
    
    async def get_task_status(self, task_id: UUID) -> Optional[TaskStatus]:
        """Get status of a task."""
        # Check queue
        task = self.task_queue.get_task(task_id)
        if task:
            return task.status
        
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return TaskStatus.COMPLETED
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return TaskStatus.FAILED
        
        return None
    
    async def register_agent(self, agent_resource: AgentResource) -> None:
        """Register an agent for task execution."""
        self.available_agents[agent_resource.agent_id] = agent_resource
        
        self.logger.info("Agent registered for task execution",
                        agent_id=str(agent_resource.agent_id),
                        cpu_cores=agent_resource.available_cpu_cores,
                        memory_mb=agent_resource.available_memory_mb)
    
    async def unregister_agent(self, agent_id: UUID) -> None:
        """Unregister an agent."""
        if agent_id in self.available_agents:
            del self.available_agents[agent_id]
            self.logger.info("Agent unregistered", agent_id=str(agent_id))
    
    async def update_agent_resources(
        self, 
        agent_id: UUID, 
        resource_update: Dict[str, Any]
    ) -> None:
        """Update agent resource availability."""
        if agent_id in self.available_agents:
            agent = self.available_agents[agent_id]
            
            for key, value in resource_update.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            agent.last_updated = time.time()
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for specific task types."""
        self._task_handlers[task_type] = handler
        self.logger.info("Task handler registered", task_type=task_type)
    
    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Get scheduling performance metrics."""
        # Update resource utilization
        if self.available_agents:
            total_load = sum(agent.current_load for agent in self.available_agents.values())
            self.scheduling_metrics["resource_utilization"] = total_load / len(self.available_agents)
        
        return self.scheduling_metrics.copy()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "pending_tasks": self.task_queue.size(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "available_agents": len(self.available_agents)
        }
    
    # Private methods
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.scheduling_enabled:
            try:
                await self._schedule_pending_tasks()
                await asyncio.sleep(self.scheduling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Scheduling loop error", error=str(e))
                await asyncio.sleep(self.scheduling_interval)
    
    async def _schedule_pending_tasks(self) -> None:
        """Schedule pending tasks to available agents."""
        if self.task_queue.is_empty():
            return
        
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return
        
        # Get available agents
        available_agents = self._get_available_agents()
        if not available_agents:
            return
        
        # Schedule tasks
        while (not self.task_queue.is_empty() and 
               len(self.running_tasks) < self.max_concurrent_tasks and
               available_agents):
            
            task = self.task_queue.get_next_task()
            if not task:
                break
            
            # Check dependencies
            if not self._dependencies_satisfied(task):
                # Re-queue task for later
                task.status = TaskStatus.PENDING
                self.task_queue.add_task(task)
                continue
            
            # Find suitable agent
            suitable_agents = self._filter_suitable_agents(available_agents, task)
            if not suitable_agents:
                # Re-queue task
                task.status = TaskStatus.PENDING
                self.task_queue.add_task(task)
                continue
            
            # Select best agent
            selected_agent_id = self.load_balancer.select_agent(suitable_agents, task)
            
            # Assign task
            await self._assign_task_to_agent(task, selected_agent_id)
            
            # Remove agent from available list if at capacity
            selected_agent = next(a for a in available_agents if a.agent_id == selected_agent_id)
            if selected_agent.current_load >= 0.9:  # 90% capacity
                available_agents.remove(selected_agent)
    
    async def _monitoring_loop(self) -> None:
        """Monitor running tasks for timeouts and completion."""
        while self.scheduling_enabled:
            try:
                current_time = time.time()
                
                # Check for task timeouts
                timed_out_tasks = []
                for task_id, task in self.running_tasks.items():
                    if (task.started_at and 
                        current_time - task.started_at > task.max_execution_time):
                        timed_out_tasks.append(task_id)
                
                # Handle timeouts
                for task_id in timed_out_tasks:
                    await self._handle_task_timeout(task_id)
                
                # Update metrics periodically
                await self._update_metrics()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(10)
    
    def _get_available_agents(self) -> List[AgentResource]:
        """Get list of available agents for task assignment."""
        current_time = time.time()
        available = []
        
        for agent in self.available_agents.values():
            # Check if agent info is fresh (within last 5 minutes)
            if current_time - agent.last_updated > 300:
                continue
            
            # Check if agent has capacity
            if agent.current_load < 0.95:  # 95% capacity threshold
                available.append(agent)
        
        return available
    
    def _filter_suitable_agents(
        self, 
        agents: List[AgentResource], 
        task: Task
    ) -> List[AgentResource]:
        """Filter agents that can execute the task."""
        suitable = []
        
        for agent in agents:
            # Check skill requirements
            if task.required_skills and not task.required_skills.issubset(agent.skills):
                continue
            
            # Check resource requirements
            req = task.resource_requirements
            if (agent.available_cpu_cores < req.cpu_cores or
                agent.available_memory_mb < req.memory_mb or
                agent.available_storage_mb < req.storage_mb):
                continue
            
            # Check GPU requirements
            if req.gpu_memory_mb > 0 and agent.available_gpu_memory_mb < req.gpu_memory_mb:
                continue
            
            suitable.append(agent)
        
        return suitable
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    async def _assign_task_to_agent(self, task: Task, agent_id: UUID) -> None:
        """Assign task to specific agent."""
        task.assigned_agent = agent_id
        task.assigned_at = time.time()
        task.status = TaskStatus.ASSIGNED
        
        # Move to running tasks
        self.running_tasks[task.task_id] = task
        
        # Update agent load
        if agent_id in self.available_agents:
            agent = self.available_agents[agent_id]
            task_load = self._calculate_task_load(task)
            agent.current_load = min(1.0, agent.current_load + task_load)
        
        # Execute task (in real implementation, this would send to agent)
        execution_task = asyncio.create_task(self._execute_task(task))
        
        self.scheduling_metrics["tasks_scheduled"] += 1
        
        self.logger.info("Task assigned to agent",
                        task_id=str(task.task_id),
                        agent_id=str(agent_id))
    
    async def _execute_task(self, task: Task) -> None:
        """Execute task (simulated execution)."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            # Simulate task execution
            execution_time = task.resource_requirements.execution_time_estimate
            await asyncio.sleep(min(execution_time, 5.0))  # Cap simulation time
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = {"status": "success", "execution_time": execution_time}
            
            # Move to completed tasks
            self.running_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            # Update agent load
            if task.assigned_agent and task.assigned_agent in self.available_agents:
                agent = self.available_agents[task.assigned_agent]
                task_load = self._calculate_task_load(task)
                agent.current_load = max(0.0, agent.current_load - task_load)
            
            self.scheduling_metrics["tasks_completed"] += 1
            
            self.logger.info("Task completed",
                           task_id=str(task.task_id),
                           execution_time=execution_time)
            
        except Exception as e:
            await self._handle_task_failure(task, str(e))
    
    async def _handle_task_timeout(self, task_id: UUID) -> None:
        """Handle task timeout."""
        task = self.running_tasks.get(task_id)
        if not task:
            return
        
        task.status = TaskStatus.TIMEOUT
        task.error_message = "Task execution timeout"
        
        # Move to failed tasks
        self.running_tasks.pop(task_id, None)
        self.failed_tasks[task_id] = task
        
        # Update agent load
        if task.assigned_agent and task.assigned_agent in self.available_agents:
            agent = self.available_agents[task.assigned_agent]
            task_load = self._calculate_task_load(task)
            agent.current_load = max(0.0, agent.current_load - task_load)
        
        self.scheduling_metrics["tasks_failed"] += 1
        
        self.logger.warning("Task timeout",
                          task_id=str(task_id),
                          max_execution_time=task.max_execution_time)
    
    async def _handle_task_failure(self, task: Task, error_message: str) -> None:
        """Handle task execution failure."""
        task.status = TaskStatus.FAILED
        task.error_message = error_message
        task.completed_at = time.time()
        
        # Move to failed tasks
        self.running_tasks.pop(task.task_id, None)
        
        # Check if we should retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.assigned_agent = None
            task.assigned_at = None
            task.started_at = None
            
            # Re-queue for retry
            self.task_queue.add_task(task)
            
            self.logger.info("Task queued for retry",
                           task_id=str(task.task_id),
                           retry_count=task.retry_count)
        else:
            self.failed_tasks[task.task_id] = task
            self.scheduling_metrics["tasks_failed"] += 1
            
            self.logger.error("Task failed permanently",
                            task_id=str(task.task_id),
                            error=error_message)
        
        # Update agent load
        if task.assigned_agent and task.assigned_agent in self.available_agents:
            agent = self.available_agents[task.assigned_agent]
            task_load = self._calculate_task_load(task)
            agent.current_load = max(0.0, agent.current_load - task_load)
    
    def _calculate_task_load(self, task: Task) -> float:
        """Calculate relative load of a task."""
        # Simple load calculation based on resource requirements
        req = task.resource_requirements
        
        # Normalize to 0-1 scale
        cpu_load = min(1.0, req.cpu_cores / 8.0)  # Assume 8 core baseline
        memory_load = min(1.0, req.memory_mb / 8192.0)  # 8GB baseline
        
        return max(cpu_load, memory_load) * 0.1  # Scale to reasonable load increment
    
    async def _update_metrics(self) -> None:
        """Update scheduling metrics."""
        # Calculate average wait time
        current_time = time.time()
        wait_times = []
        
        for task in self.running_tasks.values():
            if task.assigned_at:
                wait_time = task.assigned_at - task.created_at
                wait_times.append(wait_time)
        
        for task in self.completed_tasks.values():
            if task.assigned_at:
                wait_time = task.assigned_at - task.created_at
                wait_times.append(wait_time)
        
        if wait_times:
            self.scheduling_metrics["average_wait_time"] = sum(wait_times) / len(wait_times)
        
        # Calculate average execution time
        execution_times = []
        for task in self.completed_tasks.values():
            if task.started_at and task.completed_at:
                exec_time = task.completed_at - task.started_at
                execution_times.append(exec_time)
        
        if execution_times:
            self.scheduling_metrics["average_execution_time"] = sum(execution_times) / len(execution_times)