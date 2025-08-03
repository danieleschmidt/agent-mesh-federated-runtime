"""Data access repositories for Agent Mesh.

Provides repository pattern implementations for data access operations
with caching, validation, and business logic.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

import structlog
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select

from .models import Node, Task, TrainingRound, MetricEntry, ConsensusRound


logger = structlog.get_logger("repositories")


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()
    
    async def refresh(self, instance) -> None:
        """Refresh an instance from the database."""
        await self.session.refresh(instance)


class NodeRepository(BaseRepository):
    """Repository for node data operations."""
    
    async def create_node(self, node_data: Dict[str, Any]) -> Node:
        """Create a new node record."""
        node = Node(**node_data)
        self.session.add(node)
        await self.session.flush()
        await self.session.refresh(node)
        
        self.logger.info("Node created", node_id=str(node.node_id))
        return node
    
    async def get_node_by_id(self, node_id: UUID) -> Optional[Node]:
        """Get node by ID."""
        query = select(Node).where(Node.node_id == node_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_node_by_public_key(self, public_key: str) -> Optional[Node]:
        """Get node by public key."""
        query = select(Node).where(Node.public_key == public_key)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def update_node(self, node_id: UUID, updates: Dict[str, Any]) -> Optional[Node]:
        """Update node information."""
        node = await self.get_node_by_id(node_id)
        if not node:
            return None
        
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        node.updated_at = datetime.utcnow()
        await self.session.flush()
        await self.session.refresh(node)
        
        self.logger.info("Node updated", node_id=str(node_id), updates=list(updates.keys()))
        return node
    
    async def update_node_status(self, node_id: UUID, status: str, last_seen: Optional[datetime] = None) -> Optional[Node]:
        """Update node status and last seen timestamp."""
        updates = {"status": status}
        if last_seen:
            updates["last_seen"] = last_seen
        else:
            updates["last_seen"] = datetime.utcnow()
        
        return await self.update_node(node_id, updates)
    
    async def update_node_metrics(self, node_id: UUID, metrics: Dict[str, Any]) -> Optional[Node]:
        """Update node performance metrics."""
        node = await self.get_node_by_id(node_id)
        if not node:
            return None
        
        # Update specific metric fields
        metric_fields = [
            'reputation', 'reliability_score', 'tasks_completed', 
            'tasks_failed', 'average_response_time', 'uptime_seconds'
        ]
        
        updates = {}
        for field in metric_fields:
            if field in metrics:
                updates[field] = metrics[field]
        
        if updates:
            return await self.update_node(node_id, updates)
        
        return node
    
    async def get_active_nodes(self, roles: Optional[List[str]] = None) -> List[Node]:
        """Get all active nodes, optionally filtered by roles."""
        query = select(Node).where(Node.status == "active")
        
        if roles:
            query = query.where(Node.role.in_(roles))
        
        query = query.order_by(desc(Node.last_seen))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_nodes_by_skills(self, required_skills: List[str]) -> List[Node]:
        """Get nodes that have all required skills."""
        query = select(Node).where(
            and_(
                Node.status == "active",
                Node.skills.op('@>')([required_skills])  # PostgreSQL array contains
            )
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_nodes_with_capacity(
        self, 
        min_cpu_cores: float = 0,
        min_memory_gb: float = 0,
        min_storage_gb: float = 0,
        require_gpu: bool = False
    ) -> List[Node]:
        """Get nodes with specified minimum capacity."""
        conditions = [
            Node.status == "active",
            Node.cpu_cores >= min_cpu_cores,
            Node.memory_gb >= min_memory_gb,
            Node.storage_gb >= min_storage_gb
        ]
        
        if require_gpu:
            conditions.append(Node.gpu_available == True)
        
        query = select(Node).where(and_(*conditions)).order_by(desc(Node.reputation))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def delete_node(self, node_id: UUID) -> bool:
        """Delete a node record."""
        node = await self.get_node_by_id(node_id)
        if not node:
            return False
        
        await self.session.delete(node)
        await self.session.flush()
        
        self.logger.info("Node deleted", node_id=str(node_id))
        return True
    
    async def cleanup_stale_nodes(self, stale_threshold: timedelta = timedelta(hours=1)) -> int:
        """Remove nodes that haven't been seen for a specified time."""
        cutoff_time = datetime.utcnow() - stale_threshold
        
        query = select(Node).where(Node.last_seen < cutoff_time)
        result = await self.session.execute(query)
        stale_nodes = result.scalars().all()
        
        count = 0
        for node in stale_nodes:
            await self.session.delete(node)
            count += 1
        
        if count > 0:
            await self.session.flush()
            self.logger.info("Cleaned up stale nodes", count=count)
        
        return count


class TaskRepository(BaseRepository):
    """Repository for task data operations."""
    
    async def create_task(self, task_data: Dict[str, Any]) -> Task:
        """Create a new task record."""
        task = Task(**task_data)
        self.session.add(task)
        await self.session.flush()
        await self.session.refresh(task)
        
        self.logger.info("Task created", task_id=str(task.task_id))
        return task
    
    async def get_task_by_id(self, task_id: UUID) -> Optional[Task]:
        """Get task by ID."""
        query = select(Task).where(Task.task_id == task_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def update_task(self, task_id: UUID, updates: Dict[str, Any]) -> Optional[Task]:
        """Update task information."""
        task = await self.get_task_by_id(task_id)
        if not task:
            return None
        
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated_at = datetime.utcnow()
        await self.session.flush()
        await self.session.refresh(task)
        
        self.logger.info("Task updated", task_id=str(task_id), updates=list(updates.keys()))
        return task
    
    async def assign_task(self, task_id: UUID, node_id: UUID) -> Optional[Task]:
        """Assign task to a node."""
        updates = {
            "assigned_node_id": node_id,
            "assigned_at": datetime.utcnow(),
            "status": "assigned"
        }
        return await self.update_task(task_id, updates)
    
    async def start_task(self, task_id: UUID) -> Optional[Task]:
        """Mark task as started."""
        updates = {
            "started_at": datetime.utcnow(),
            "status": "running"
        }
        return await self.update_task(task_id, updates)
    
    async def complete_task(self, task_id: UUID, result: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> Optional[Task]:
        """Mark task as completed with results."""
        updates = {
            "completed_at": datetime.utcnow(),
            "status": "completed",
            "result": result
        }
        
        if metrics:
            updates["execution_metrics"] = metrics
        
        return await self.update_task(task_id, updates)
    
    async def fail_task(self, task_id: UUID, error_message: str) -> Optional[Task]:
        """Mark task as failed."""
        updates = {
            "completed_at": datetime.utcnow(),
            "status": "failed",
            "error_message": error_message
        }
        return await self.update_task(task_id, updates)
    
    async def get_pending_tasks(self, limit: int = 100) -> List[Task]:
        """Get pending tasks ordered by priority and creation time."""
        query = (
            select(Task)
            .where(Task.status.in_(["pending", "queued"]))
            .order_by(desc(Task.priority), asc(Task.created_at))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_tasks_by_node(self, node_id: UUID, statuses: Optional[List[str]] = None) -> List[Task]:
        """Get tasks assigned to a specific node."""
        query = select(Task).where(Task.assigned_node_id == node_id)
        
        if statuses:
            query = query.where(Task.status.in_(statuses))
        
        query = query.order_by(desc(Task.created_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_tasks_by_type(self, task_type: str, limit: int = 100) -> List[Task]:
        """Get tasks by type."""
        query = (
            select(Task)
            .where(Task.task_type == task_type)
            .order_by(desc(Task.created_at))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_overdue_tasks(self) -> List[Task]:
        """Get tasks that are past their deadline."""
        now = datetime.utcnow()
        query = select(Task).where(
            and_(
                Task.deadline < now,
                Task.status.in_(["pending", "queued", "assigned", "running"])
            )
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_task_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        # Count tasks by status
        status_query = (
            select(Task.status, func.count(Task.id))
            .group_by(Task.status)
        )
        status_result = await self.session.execute(status_query)
        status_counts = dict(status_result.all())
        
        # Average execution time
        avg_time_query = select(func.avg(
            func.extract('epoch', Task.completed_at - Task.started_at)
        )).where(
            and_(
                Task.status == "completed",
                Task.started_at.isnot(None),
                Task.completed_at.isnot(None)
            )
        )
        avg_time_result = await self.session.execute(avg_time_query)
        avg_execution_time = avg_time_result.scalar() or 0
        
        return {
            "status_counts": status_counts,
            "average_execution_time_seconds": float(avg_execution_time),
            "total_tasks": sum(status_counts.values())
        }


class TrainingRoundRepository(BaseRepository):
    """Repository for federated learning training round data."""
    
    async def create_training_round(self, round_data: Dict[str, Any]) -> TrainingRound:
        """Create a new training round record."""
        training_round = TrainingRound(**round_data)
        self.session.add(training_round)
        await self.session.flush()
        await self.session.refresh(training_round)
        
        self.logger.info("Training round created", 
                        round_id=str(training_round.round_id),
                        session_id=str(training_round.session_id))
        return training_round
    
    async def get_training_round(self, round_id: UUID) -> Optional[TrainingRound]:
        """Get training round by ID."""
        query = select(TrainingRound).where(TrainingRound.round_id == round_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def update_training_round(self, round_id: UUID, updates: Dict[str, Any]) -> Optional[TrainingRound]:
        """Update training round information."""
        training_round = await self.get_training_round(round_id)
        if not training_round:
            return None
        
        for key, value in updates.items():
            if hasattr(training_round, key):
                setattr(training_round, key, value)
        
        await self.session.flush()
        await self.session.refresh(training_round)
        
        self.logger.info("Training round updated", round_id=str(round_id))
        return training_round
    
    async def complete_training_round(
        self, 
        round_id: UUID, 
        loss: float,
        accuracy: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[TrainingRound]:
        """Mark training round as completed."""
        updates = {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "loss": loss
        }
        
        if accuracy is not None:
            updates["accuracy"] = accuracy
        
        if metrics:
            updates["metadata"] = metrics
        
        return await self.update_training_round(round_id, updates)
    
    async def get_session_rounds(self, session_id: UUID) -> List[TrainingRound]:
        """Get all rounds for a training session."""
        query = (
            select(TrainingRound)
            .where(TrainingRound.session_id == session_id)
            .order_by(asc(TrainingRound.round_number))
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_node_training_history(self, node_id: UUID, limit: int = 100) -> List[TrainingRound]:
        """Get training history for a specific node."""
        query = (
            select(TrainingRound)
            .where(TrainingRound.node_id == node_id)
            .order_by(desc(TrainingRound.started_at))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_training_statistics(self, session_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get training statistics."""
        query = select(TrainingRound)
        
        if session_id:
            query = query.where(TrainingRound.session_id == session_id)
        
        # Average metrics
        avg_query = query.with_only_columns([
            func.avg(TrainingRound.loss).label('avg_loss'),
            func.avg(TrainingRound.accuracy).label('avg_accuracy'),
            func.avg(TrainingRound.training_time_seconds).label('avg_training_time'),
            func.count(TrainingRound.id).label('total_rounds')
        ])
        
        result = await self.session.execute(avg_query)
        stats = result.first()
        
        return {
            "average_loss": float(stats.avg_loss) if stats.avg_loss else 0,
            "average_accuracy": float(stats.avg_accuracy) if stats.avg_accuracy else 0,
            "average_training_time_seconds": float(stats.avg_training_time) if stats.avg_training_time else 0,
            "total_rounds": int(stats.total_rounds)
        }


class MetricsRepository(BaseRepository):
    """Repository for metrics and monitoring data."""
    
    async def record_metric(
        self, 
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        node_id: Optional[UUID] = None,
        component: str = "system",
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> MetricEntry:
        """Record a metric entry."""
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "metric_type": metric_type,
            "node_id": node_id,
            "component": component,
            "labels": labels,
            "timestamp": timestamp or datetime.utcnow()
        }
        
        metric = MetricEntry(**metric_data)
        self.session.add(metric)
        await self.session.flush()
        await self.session.refresh(metric)
        
        return metric
    
    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
        node_id: Optional[UUID] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MetricEntry]:
        """Get metrics with filtering."""
        query = select(MetricEntry)
        
        conditions = []
        if metric_name:
            conditions.append(MetricEntry.metric_name == metric_name)
        if node_id:
            conditions.append(MetricEntry.node_id == node_id)
        if component:
            conditions.append(MetricEntry.component == component)
        if start_time:
            conditions.append(MetricEntry.timestamp >= start_time)
        if end_time:
            conditions.append(MetricEntry.timestamp <= end_time)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(MetricEntry.timestamp)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_latest_metric(self, metric_name: str, node_id: Optional[UUID] = None) -> Optional[MetricEntry]:
        """Get the latest value for a metric."""
        query = select(MetricEntry).where(MetricEntry.metric_name == metric_name)
        
        if node_id:
            query = query.where(MetricEntry.node_id == node_id)
        
        query = query.order_by(desc(MetricEntry.timestamp)).limit(1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def cleanup_old_metrics(self, retention_days: int = 30) -> int:
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        query = select(MetricEntry).where(MetricEntry.timestamp < cutoff_time)
        result = await self.session.execute(query)
        old_metrics = result.scalars().all()
        
        count = 0
        for metric in old_metrics:
            await self.session.delete(metric)
            count += 1
        
        if count > 0:
            await self.session.flush()
            self.logger.info("Cleaned up old metrics", count=count)
        
        return count