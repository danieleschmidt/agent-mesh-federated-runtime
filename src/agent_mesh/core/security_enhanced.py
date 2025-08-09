"""Enhanced security system with threat detection and prevention.

This module provides advanced security capabilities including anomaly detection,
intrusion prevention, and comprehensive threat monitoring.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4

import structlog
from .error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
from .monitoring import MeshMonitor, AlertSeverity


class ThreatType(Enum):
    """Types of security threats."""
    
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_SIGNATURE = "invalid_signature"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALFORMED_DATA = "malformed_data"
    REPLAY_ATTACK = "replay_attack"
    SYBIL_ATTACK = "sybil_attack"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"


class SecurityLevel(Enum):
    """Security levels for operations."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event for monitoring and analysis."""
    
    event_id: UUID = field(default_factory=uuid4)
    timestamp: float = field(default_factory=time.time)
    threat_type: ThreatType = ThreatType.SUSPICIOUS_BEHAVIOR
    severity: SecurityLevel = SecurityLevel.MEDIUM
    source_node_id: Optional[UUID] = None
    target_node_id: Optional[UUID] = None
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False


@dataclass
class NodeSecurityProfile:
    """Security profile for a node."""
    
    node_id: UUID
    trust_score: float = 1.0
    reputation_score: float = 1.0
    threat_score: float = 0.0
    
    # Behavioral metrics
    message_rate: float = 0.0
    error_rate: float = 0.0
    failed_authentications: int = 0
    
    # Activity tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_interactions: int = 0
    
    # Threat indicators
    suspicious_activities: List[str] = field(default_factory=list)
    blocked_until: Optional[float] = None
    
    def is_blocked(self) -> bool:
        """Check if node is currently blocked."""
        if self.blocked_until is None:
            return False
        return time.time() < self.blocked_until
    
    def update_trust_score(self, delta: float):
        """Update trust score with bounds checking."""
        self.trust_score = max(0.0, min(1.0, self.trust_score + delta))
        
    def update_threat_score(self, delta: float):
        """Update threat score with bounds checking."""
        self.threat_score = max(0.0, min(1.0, self.threat_score + delta))


class RateLimiter:
    """Rate limiting for security protection."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[UUID, List[float]] = {}
        
    def check_rate_limit(self, node_id: UUID) -> bool:
        """Check if node exceeds rate limit."""
        now = time.time()
        
        if node_id not in self.requests:
            self.requests[node_id] = []
            
        # Remove old requests outside the window
        self.requests[node_id] = [
            req_time for req_time in self.requests[node_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[node_id]) >= self.max_requests:
            return False  # Rate limit exceeded
            
        # Add current request
        self.requests[node_id].append(now)
        return True
        
    def get_current_rate(self, node_id: UUID) -> float:
        """Get current request rate for node."""
        if node_id not in self.requests:
            return 0.0
            
        now = time.time()
        recent_requests = [
            req_time for req_time in self.requests[node_id]
            if now - req_time < self.window_seconds
        ]
        
        return len(recent_requests) / self.window_seconds


class AnomalyDetector:
    """Detects anomalous behavior patterns."""
    
    def __init__(self):
        self.logger = structlog.get_logger("anomaly_detector")
        
        # Baseline metrics
        self.baseline_metrics: Dict[str, float] = {}
        self.metric_history: Dict[str, List[Tuple[float, float]]] = {}  # timestamp, value
        
        # Thresholds
        self.anomaly_threshold = 2.0  # Standard deviations from mean
        self.min_samples = 10
        
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value for anomaly detection."""
        timestamp = time.time()
        
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
            
        self.metric_history[metric_name].append((timestamp, value))
        
        # Keep only recent history (last 24 hours)
        cutoff_time = timestamp - 86400
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts > cutoff_time
        ]
        
        # Update baseline if we have enough samples
        if len(self.metric_history[metric_name]) >= self.min_samples:
            self._update_baseline(metric_name)
            
    def is_anomalous(self, metric_name: str, value: float) -> bool:
        """Check if a value is anomalous."""
        if metric_name not in self.baseline_metrics:
            return False  # No baseline yet
            
        baseline = self.baseline_metrics[metric_name]
        
        # Calculate z-score (simplified)
        if baseline == 0:
            return value > 0  # Any non-zero value is anomalous if baseline is zero
            
        deviation = abs(value - baseline) / max(baseline, 0.1)  # Avoid division by zero
        
        return deviation > self.anomaly_threshold
        
    def _update_baseline(self, metric_name: str):
        """Update baseline metrics."""
        values = [val for _, val in self.metric_history[metric_name]]
        if values:
            self.baseline_metrics[metric_name] = sum(values) / len(values)


class SecurityManager:
    """Enhanced security manager with threat detection."""
    
    def __init__(self, node_id: UUID):
        self.node_id = node_id
        self.logger = structlog.get_logger("security_manager", node_id=str(node_id))
        
        # Security profiles
        self.node_profiles: Dict[UUID, NodeSecurityProfile] = {}
        
        # Security components
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self.anomaly_detector = AnomalyDetector()
        
        # Threat monitoring
        self.security_events: List[SecurityEvent] = []
        self.blocked_nodes: Set[UUID] = set()
        
        # Configuration
        self.max_events_history = 10000
        self.trust_decay_rate = 0.01  # Per hour
        self.threat_threshold = 0.7
        
        # Integration
        self.error_handler: Optional[ErrorHandler] = None
        self.monitor: Optional[MeshMonitor] = None
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def initialize(self):
        """Initialize security manager."""
        self.logger.info("Initializing security manager")
        
        # Add self to node profiles
        self.node_profiles[self.node_id] = NodeSecurityProfile(node_id=self.node_id)
        
        # Start maintenance task
        self._running = True
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
    async def cleanup(self):
        """Cleanup security manager."""
        self.logger.info("Cleaning up security manager")
        self._running = False
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
                
    def set_integrations(self, error_handler: ErrorHandler, monitor: MeshMonitor):
        """Set integration components."""
        self.error_handler = error_handler
        self.monitor = monitor
        
    async def validate_node_access(self, node_id: UUID, operation: str) -> bool:
        """Validate if node has access to perform operation."""
        # Get or create profile
        profile = self.get_node_profile(node_id)
        
        # Check if node is blocked
        if profile.is_blocked():
            await self._record_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.HIGH,
                source_node_id=node_id,
                description=f"Blocked node attempted {operation}"
            )
            return False
            
        # Check rate limit
        if not self.rate_limiter.check_rate_limit(node_id):
            await self._handle_rate_limit_exceeded(node_id, operation)
            return False
            
        # Check trust score
        if profile.trust_score < 0.3:  # Low trust threshold
            await self._record_security_event(
                ThreatType.SUSPICIOUS_BEHAVIOR,
                SecurityLevel.MEDIUM,
                source_node_id=node_id,
                description=f"Low trust node attempted {operation}",
                evidence={"trust_score": profile.trust_score}
            )
            return False
            
        # Update interaction count
        profile.total_interactions += 1
        profile.last_seen = time.time()
        
        return True
        
    async def report_suspicious_behavior(
        self, 
        node_id: UUID, 
        behavior_type: str, 
        evidence: Optional[Dict[str, Any]] = None
    ):
        """Report suspicious behavior from a node."""
        profile = self.get_node_profile(node_id)
        
        # Add to suspicious activities
        profile.suspicious_activities.append(f"{behavior_type}:{time.time()}")
        
        # Update threat score
        threat_increase = 0.1
        if behavior_type in ["invalid_signature", "replay_attack"]:
            threat_increase = 0.3
        elif behavior_type in ["byzantine_behavior", "sybil_attack"]:
            threat_increase = 0.5
            
        profile.update_threat_score(threat_increase)
        profile.update_trust_score(-threat_increase)
        
        # Record security event
        severity = SecurityLevel.HIGH if threat_increase >= 0.3 else SecurityLevel.MEDIUM
        
        await self._record_security_event(
            ThreatType.SUSPICIOUS_BEHAVIOR,
            severity,
            source_node_id=node_id,
            description=f"Suspicious behavior: {behavior_type}",
            evidence=evidence or {}
        )
        
        # Check if node should be blocked
        if profile.threat_score > self.threat_threshold:
            await self._block_node(node_id, duration_seconds=3600)  # Block for 1 hour
            
    async def validate_message_integrity(
        self, 
        message: Dict[str, Any], 
        sender_id: UUID
    ) -> bool:
        """Validate message integrity and detect attacks."""
        
        # Check for replay attacks
        if await self._is_replay_attack(message, sender_id):
            await self.report_suspicious_behavior(sender_id, "replay_attack", {"message_id": message.get("id")})
            return False
            
        # Check message structure
        if not self._validate_message_structure(message):
            await self.report_suspicious_behavior(sender_id, "malformed_data", {"message": message})
            return False
            
        # Check for anomalous message patterns
        message_size = len(str(message))
        self.anomaly_detector.record_metric(f"message_size_{sender_id}", message_size)
        
        if self.anomaly_detector.is_anomalous(f"message_size_{sender_id}", message_size):
            await self.report_suspicious_behavior(
                sender_id, 
                "anomalous_message_size", 
                {"size": message_size}
            )
            
        return True
        
    def get_node_profile(self, node_id: UUID) -> NodeSecurityProfile:
        """Get or create security profile for node."""
        if node_id not in self.node_profiles:
            self.node_profiles[node_id] = NodeSecurityProfile(node_id=node_id)
        return self.node_profiles[node_id]
        
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics summary."""
        total_nodes = len(self.node_profiles)
        blocked_nodes = len(self.blocked_nodes)
        recent_events = len([e for e in self.security_events if time.time() - e.timestamp < 3600])
        
        # Trust distribution
        trust_scores = [p.trust_score for p in self.node_profiles.values()]
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
        
        # Threat level
        threat_scores = [p.threat_score for p in self.node_profiles.values()]
        max_threat = max(threat_scores) if threat_scores else 0.0
        
        return {
            "total_nodes": total_nodes,
            "blocked_nodes": blocked_nodes,
            "recent_security_events": recent_events,
            "average_trust_score": avg_trust,
            "maximum_threat_score": max_threat,
            "security_level": self._calculate_security_level()
        }
        
    async def _handle_rate_limit_exceeded(self, node_id: UUID, operation: str):
        """Handle rate limit exceeded."""
        profile = self.get_node_profile(node_id)
        profile.update_threat_score(0.2)
        profile.update_trust_score(-0.1)
        
        await self._record_security_event(
            ThreatType.RATE_LIMIT_EXCEEDED,
            SecurityLevel.MEDIUM,
            source_node_id=node_id,
            description=f"Rate limit exceeded for {operation}",
            evidence={"current_rate": self.rate_limiter.get_current_rate(node_id)}
        )
        
    async def _is_replay_attack(self, message: Dict[str, Any], sender_id: UUID) -> bool:
        """Check if message is a replay attack."""
        # Simple replay detection based on timestamp
        message_timestamp = message.get("timestamp", 0)
        current_time = time.time()
        
        # Message too old (older than 5 minutes)
        if current_time - message_timestamp > 300:
            return True
            
        # Message from the future (allow 1 minute clock skew)
        if message_timestamp - current_time > 60:
            return True
            
        return False
        
    def _validate_message_structure(self, message: Dict[str, Any]) -> bool:
        """Validate basic message structure."""
        required_fields = ["timestamp", "sender_id", "message_type"]
        
        for field in required_fields:
            if field not in message:
                return False
                
        # Check for reasonable limits
        if len(str(message)) > 1024 * 1024:  # 1MB limit
            return False
            
        return True
        
    async def _block_node(self, node_id: UUID, duration_seconds: int):
        """Block a node for specified duration."""
        profile = self.get_node_profile(node_id)
        profile.blocked_until = time.time() + duration_seconds
        self.blocked_nodes.add(node_id)
        
        await self._record_security_event(
            ThreatType.SUSPICIOUS_BEHAVIOR,
            SecurityLevel.HIGH,
            source_node_id=node_id,
            description=f"Node blocked for {duration_seconds} seconds",
            blocked=True
        )
        
        self.logger.warning("Node blocked for security violation",
                           node_id=str(node_id),
                           duration_seconds=duration_seconds)
                           
    async def _record_security_event(
        self,
        threat_type: ThreatType,
        severity: SecurityLevel,
        source_node_id: Optional[UUID] = None,
        target_node_id: Optional[UUID] = None,
        description: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        blocked: bool = False
    ):
        """Record a security event."""
        event = SecurityEvent(
            threat_type=threat_type,
            severity=severity,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            description=description,
            evidence=evidence or {},
            blocked=blocked
        )
        
        self.security_events.append(event)
        
        # Cleanup old events
        if len(self.security_events) > self.max_events_history:
            self.security_events = self.security_events[-self.max_events_history//2:]
            
        # Report to error handler if available
        if self.error_handler:
            await self.error_handler.handle_error(
                Exception(f"Security event: {description}"),
                ErrorCategory.SECURITY,
                ErrorSeverity.HIGH if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL] else ErrorSeverity.MEDIUM,
                context={"event": event.__dict__}
            )
            
        # Report to monitor if available
        if self.monitor:
            alert_severity = AlertSeverity.CRITICAL if severity == SecurityLevel.CRITICAL else AlertSeverity.WARNING
            self.monitor.trigger_alert(
                f"security_{threat_type.value}",
                alert_severity,
                description
            )
            
    def _calculate_security_level(self) -> str:
        """Calculate overall security level."""
        if not self.node_profiles:
            return "UNKNOWN"
            
        # Check for recent high-severity events
        recent_critical_events = [
            e for e in self.security_events 
            if time.time() - e.timestamp < 3600 and e.severity == SecurityLevel.CRITICAL
        ]
        
        if recent_critical_events:
            return "CRITICAL"
            
        # Check blocked nodes percentage
        blocked_percentage = len(self.blocked_nodes) / len(self.node_profiles)
        if blocked_percentage > 0.1:  # More than 10% blocked
            return "HIGH_RISK"
            
        # Check average trust score
        trust_scores = [p.trust_score for p in self.node_profiles.values()]
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
        
        if avg_trust < 0.5:
            return "MEDIUM_RISK"
        elif avg_trust < 0.8:
            return "LOW_RISK"
        else:
            return "SECURE"
            
    async def _maintenance_loop(self):
        """Background maintenance for security manager."""
        while self._running:
            try:
                await self._decay_trust_scores()
                await self._cleanup_blocked_nodes()
                await self._prune_suspicious_activities()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Security maintenance error", error=str(e))
                await asyncio.sleep(3600)
                
    async def _decay_trust_scores(self):
        """Gradually decay trust scores over time."""
        for profile in self.node_profiles.values():
            # Increase trust for nodes with low threat scores
            if profile.threat_score < 0.1:
                profile.update_trust_score(0.01)
            
            # Decay threat scores for inactive nodes
            time_since_last_seen = time.time() - profile.last_seen
            if time_since_last_seen > 3600:  # 1 hour
                profile.update_threat_score(-0.05)
                
    async def _cleanup_blocked_nodes(self):
        """Remove expired blocks."""
        current_time = time.time()
        
        for node_id in list(self.blocked_nodes):
            profile = self.node_profiles.get(node_id)
            if profile and not profile.is_blocked():
                self.blocked_nodes.discard(node_id)
                self.logger.info("Node unblocked", node_id=str(node_id))
                
    async def _prune_suspicious_activities(self):
        """Prune old suspicious activities."""
        cutoff_time = time.time() - 86400  # 24 hours
        
        for profile in self.node_profiles.values():
            profile.suspicious_activities = [
                activity for activity in profile.suspicious_activities
                if float(activity.split(':')[-1]) > cutoff_time
            ]