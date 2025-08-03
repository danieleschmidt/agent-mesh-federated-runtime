"""Database models for Agent Mesh.

Defines SQLAlchemy models for persistent storage of nodes, tasks,
training rounds, metrics, and other system state.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    LargeBinary, ForeignKey, Index, UniqueConstraint, JSON
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.types import TypeDecorator, CHAR


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    
    Uses PostgreSQL's UUID type when available, otherwise uses CHAR(36).
    """
    
    impl = CHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgresUUID())
        else:
            return dialect.type_descriptor(CHAR(36))
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, UUID):
                return str(UUID(value))
            return str(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, UUID):
                return UUID(value)
            return value


Base = declarative_base()


class Node(Base):
    """Node information and state."""
    
    __tablename__ = "nodes"
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid4)
    
    # Node identification
    node_id = Column(GUID(), unique=True, nullable=False, index=True)
    public_key = Column(String(256), unique=True, nullable=False, index=True)
    
    # Network information
    listen_address = Column(String(512), nullable=False)
    external_addresses = Column(JSON, nullable=True)  # List of external addresses
    protocols = Column(JSON, nullable=True)  # Supported protocols
    
    # Node capabilities
    cpu_cores = Column(Integer, nullable=False, default=1)
    memory_gb = Column(Float, nullable=False, default=1.0)
    storage_gb = Column(Float, nullable=False, default=10.0)
    gpu_available = Column(Boolean, nullable=False, default=False)
    gpu_memory_mb = Column(Float, nullable=True)
    bandwidth_mbps = Column(Float, nullable=False, default=10.0)
    skills = Column(JSON, nullable=True)  # List of skills/capabilities
    
    # Node status
    role = Column(String(50), nullable=False, default="participant")
    status = Column(String(50), nullable=False, default="initializing")
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    uptime_seconds = Column(Float, nullable=False, default=0.0)
    
    # Performance metrics
    reputation = Column(Float, nullable=False, default=1.0)
    reliability_score = Column(Float, nullable=False, default=1.0)
    tasks_completed = Column(Integer, nullable=False, default=0)
    tasks_failed = Column(Integer, nullable=False, default=0)
    average_response_time = Column(Float, nullable=False, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="assigned_node")
    training_rounds = relationship("TrainingRound", back_populates="node")
    metrics = relationship("MetricEntry", back_populates="node")
    
    # Indexes
    __table_args__ = (
        Index("idx_node_status_role", "status", "role"),
        Index("idx_node_last_seen", "last_seen"),
        Index("idx_node_reputation", "reputation"),
    )
    
    @validates('role')
    def validate_role(self, key, role):
        valid_roles = ['trainer', 'aggregator', 'validator', 'coordinator', 'observer']
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")
        return role
    
    @validates('status')
    def validate_status(self, key, status):
        valid_statuses = ['initializing', 'connecting', 'active', 'degraded', 'disconnected', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': str(self.id),
            'node_id': str(self.node_id),
            'public_key': self.public_key,
            'listen_address': self.listen_address,
            'external_addresses': self.external_addresses,
            'protocols': self.protocols,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'storage_gb': self.storage_gb,
            'gpu_available': self.gpu_available,
            'gpu_memory_mb': self.gpu_memory_mb,
            'bandwidth_mbps': self.bandwidth_mbps,
            'skills': self.skills,
            'role': self.role,
            'status': self.status,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'uptime_seconds': self.uptime_seconds,
            'reputation': self.reputation,
            'reliability_score': self.reliability_score,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'average_response_time': self.average_response_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }


class Task(Base):
    """Task execution records."""
    
    __tablename__ = "tasks"
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid4)
    
    # Task identification
    task_id = Column(GUID(), unique=True, nullable=False, index=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(100), nullable=False, index=True)
    
    # Task properties
    priority = Column(Integer, nullable=False, default=3)
    status = Column(String(50), nullable=False, default="pending", index=True)
    
    # Requirements
    required_skills = Column(JSON, nullable=True)  # List of required skills
    cpu_cores = Column(Float, nullable=False, default=1.0)
    memory_mb = Column(Float, nullable=False, default=512.0)
    storage_mb = Column(Float, nullable=False, default=100.0)
    gpu_memory_mb = Column(Float, nullable=False, default=0.0)
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deadline = Column(DateTime, nullable=True)
    estimated_duration = Column(Float, nullable=False, default=60.0)  # seconds
    max_execution_time = Column(Float, nullable=False, default=300.0)  # seconds
    
    # Assignment and execution
    assigned_node_id = Column(GUID(), ForeignKey("nodes.node_id"), nullable=True, index=True)
    assigned_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Task data
    payload = Column(JSON, nullable=True)
    dependencies = Column(JSON, nullable=True)  # List of task IDs
    
    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_metrics = Column(JSON, nullable=True)
    
    # Retry handling
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    
    # Metadata
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    assigned_node = relationship("Node", back_populates="tasks")
    
    # Indexes
    __table_args__ = (
        Index("idx_task_status_priority", "status", "priority"),
        Index("idx_task_type_created", "task_type", "created_at"),
        Index("idx_task_assigned_node", "assigned_node_id"),
        Index("idx_task_deadline", "deadline"),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        valid_statuses = ['pending', 'queued', 'assigned', 'running', 'completed', 'failed', 'cancelled', 'timeout']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return status
    
    @validates('priority')
    def validate_priority(self, key, priority):
        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")
        return priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': str(self.id),
            'task_id': str(self.task_id),
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'priority': self.priority,
            'status': self.status,
            'required_skills': self.required_skills,
            'cpu_cores': self.cpu_cores,
            'memory_mb': self.memory_mb,
            'storage_mb': self.storage_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'estimated_duration': self.estimated_duration,
            'max_execution_time': self.max_execution_time,
            'assigned_node_id': str(self.assigned_node_id) if self.assigned_node_id else None,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'payload': self.payload,
            'dependencies': self.dependencies,
            'result': self.result,
            'error_message': self.error_message,
            'execution_metrics': self.execution_metrics,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }


class TrainingRound(Base):
    """Federated learning training round records."""
    
    __tablename__ = "training_rounds"
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid4)
    
    # Round identification
    round_id = Column(GUID(), unique=True, nullable=False, index=True)
    session_id = Column(GUID(), nullable=False, index=True)  # Training session
    round_number = Column(Integer, nullable=False, index=True)
    
    # Participant information
    node_id = Column(GUID(), ForeignKey("nodes.node_id"), nullable=False, index=True)
    coordinator_id = Column(GUID(), nullable=False)
    
    # Training configuration
    algorithm = Column(String(50), nullable=False, default="fedavg")
    local_epochs = Column(Integer, nullable=False, default=5)
    batch_size = Column(Integer, nullable=False, default=32)
    learning_rate = Column(Float, nullable=False, default=0.01)
    
    # Training results
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    samples_count = Column(Integer, nullable=False, default=0)
    training_time_seconds = Column(Float, nullable=True)
    
    # Model information
    model_size_bytes = Column(Integer, nullable=True)
    model_update = Column(LargeBinary, nullable=True)  # Serialized model update
    model_hash = Column(String(64), nullable=True)  # SHA-256 hash
    
    # Communication metrics
    communication_cost_mb = Column(Float, nullable=False, default=0.0)
    aggregation_time_seconds = Column(Float, nullable=True)
    convergence_score = Column(Float, nullable=True)
    
    # Status and timing
    status = Column(String(50), nullable=False, default="pending")
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    node = relationship("Node", back_populates="training_rounds")
    
    # Indexes
    __table_args__ = (
        Index("idx_training_session_round", "session_id", "round_number"),
        Index("idx_training_node_round", "node_id", "round_number"),
        Index("idx_training_algorithm", "algorithm"),
        Index("idx_training_status", "status"),
        UniqueConstraint("session_id", "round_number", "node_id", name="uq_session_round_node"),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        valid_statuses = ['pending', 'training', 'aggregating', 'completed', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': str(self.id),
            'round_id': str(self.round_id),
            'session_id': str(self.session_id),
            'round_number': self.round_number,
            'node_id': str(self.node_id),
            'coordinator_id': str(self.coordinator_id),
            'algorithm': self.algorithm,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'samples_count': self.samples_count,
            'training_time_seconds': self.training_time_seconds,
            'model_size_bytes': self.model_size_bytes,
            'model_hash': self.model_hash,
            'communication_cost_mb': self.communication_cost_mb,
            'aggregation_time_seconds': self.aggregation_time_seconds,
            'convergence_score': self.convergence_score,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata
        }


class MetricEntry(Base):
    """System metrics and monitoring data."""
    
    __tablename__ = "metrics"
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid4)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # counter, gauge, histogram
    
    # Source information
    node_id = Column(GUID(), ForeignKey("nodes.node_id"), nullable=True, index=True)
    component = Column(String(100), nullable=False, index=True)  # consensus, network, federated
    
    # Metric value
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    
    # Additional data
    labels = Column(JSON, nullable=True)  # Metric labels/tags
    dimensions = Column(JSON, nullable=True)  # Additional dimensions
    
    # Timing
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    node = relationship("Node", back_populates="metrics")
    
    # Indexes
    __table_args__ = (
        Index("idx_metric_name_timestamp", "metric_name", "timestamp"),
        Index("idx_metric_node_component", "node_id", "component"),
        Index("idx_metric_type_timestamp", "metric_type", "timestamp"),
    )
    
    @validates('metric_type')
    def validate_metric_type(self, key, metric_type):
        valid_types = ['counter', 'gauge', 'histogram', 'summary']
        if metric_type not in valid_types:
            raise ValueError(f"Invalid metric_type: {metric_type}. Must be one of {valid_types}")
        return metric_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': str(self.id),
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'node_id': str(self.node_id) if self.node_id else None,
            'component': self.component,
            'value': self.value,
            'unit': self.unit,
            'labels': self.labels,
            'dimensions': self.dimensions,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }


class ConsensusRound(Base):
    """Consensus round execution records."""
    
    __tablename__ = "consensus_rounds"
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid4)
    
    # Round identification
    proposal_id = Column(GUID(), unique=True, nullable=False, index=True)
    view_number = Column(Integer, nullable=False, default=0)
    sequence_number = Column(Integer, nullable=False, default=0)
    
    # Proposer information
    proposer_id = Column(GUID(), nullable=False, index=True)
    proposal_type = Column(String(100), nullable=False, index=True)
    
    # Consensus data
    proposal_data = Column(JSON, nullable=True)
    proposal_hash = Column(String(64), nullable=True)
    
    # Participants
    participants = Column(JSON, nullable=True)  # List of participant node IDs
    votes_for = Column(Integer, nullable=False, default=0)
    votes_against = Column(Integer, nullable=False, default=0)
    required_votes = Column(Integer, nullable=False, default=1)
    
    # Result
    accepted = Column(Boolean, nullable=False, default=False)
    finalized_at = Column(DateTime, nullable=True)
    reason = Column(String(256), nullable=True)
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    timeout_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_consensus_proposer_type", "proposer_id", "proposal_type"),
        Index("idx_consensus_view_sequence", "view_number", "sequence_number"),
        Index("idx_consensus_created", "created_at"),
        Index("idx_consensus_accepted", "accepted"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': str(self.id),
            'proposal_id': str(self.proposal_id),
            'view_number': self.view_number,
            'sequence_number': self.sequence_number,
            'proposer_id': str(self.proposer_id),
            'proposal_type': self.proposal_type,
            'proposal_data': self.proposal_data,
            'proposal_hash': self.proposal_hash,
            'participants': self.participants,
            'votes_for': self.votes_for,
            'votes_against': self.votes_against,
            'required_votes': self.required_votes,
            'accepted': self.accepted,
            'finalized_at': self.finalized_at.isoformat() if self.finalized_at else None,
            'reason': self.reason,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'timeout_at': self.timeout_at.isoformat() if self.timeout_at else None,
            'metadata': self.metadata
        }