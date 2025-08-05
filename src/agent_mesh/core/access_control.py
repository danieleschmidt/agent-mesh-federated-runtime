"""Advanced access control and authorization system for Agent Mesh.

This module provides comprehensive access control, role-based permissions,
attribute-based access control (ABAC), and security policy enforcement.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from .error_handling import SecurityError, AgentMeshError
from .validation import ValidationResult, sanitize_for_logging


class AccessLevel(Enum):
    """Access levels in the system."""
    
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4
    SYSTEM = 5


class ResourceType(Enum):
    """Types of resources that can be protected."""
    
    NODE = "node"
    TASK = "task"
    DATA = "data"
    CONSENSUS = "consensus"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    METRICS = "metrics"
    LOGS = "logs"


class PolicyEffect(Enum):
    """Policy decision effects."""
    
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    
    CRYPTOGRAPHIC_SIGNATURE = "crypto_signature"
    JWT_TOKEN = "jwt_token"
    API_KEY = "api_key"
    MUTUAL_TLS = "mutual_tls"
    MULTI_FACTOR = "multi_factor"


@dataclass
class Permission:
    """Individual permission definition."""
    
    resource_type: ResourceType
    resource_id: Optional[str] = None  # Specific resource or "*" for all
    action: str = ""  # Action like "read", "write", "execute"
    access_level: AccessLevel = AccessLevel.READ
    conditions: Dict[str, Any] = field(default_factory=dict)
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if permission has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def matches(self, resource_type: ResourceType, resource_id: str, action: str) -> bool:
        """Check if permission matches the requested access."""
        # Check resource type
        if self.resource_type != resource_type:
            return False
        
        # Check resource ID (wildcard or exact match)
        if self.resource_id and self.resource_id != "*" and self.resource_id != resource_id:
            return False
        
        # Check action (wildcard or exact match)
        if self.action and self.action != "*" and self.action != action:
            return False
        
        return True


@dataclass
class Role:
    """Role definition with permissions."""
    
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits_from: Set[str] = field(default_factory=set)  # Role inheritance
    created_at: float = field(default_factory=time.time)
    created_by: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subject:
    """Security subject (user, node, service)."""
    
    subject_id: UUID
    subject_type: str  # "node", "user", "service"
    display_name: str
    roles: Set[str] = field(default_factory=set)
    direct_permissions: Set[Permission] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()


@dataclass
class AccessRequest:
    """Access request for authorization."""
    
    subject_id: UUID
    resource_type: ResourceType
    resource_id: str
    action: str
    request_id: UUID = field(default_factory=uuid4)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class AccessDecision:
    """Authorization decision result."""
    
    request_id: UUID
    effect: PolicyEffect
    granted_permissions: List[Permission] = field(default_factory=list)
    denied_reasons: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    decision_time: float = field(default_factory=time.time)
    evaluator: str = "rbac"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    
    policy_id: str
    name: str
    description: str
    effect: PolicyEffect
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100  # Higher number = higher priority
    created_at: float = field(default_factory=time.time)
    created_by: Optional[UUID] = None
    is_active: bool = True


class RoleBasedAccessControl:
    """Role-Based Access Control (RBAC) implementation."""
    
    def __init__(self):
        self.logger = structlog.get_logger("rbac")
        
        # Core RBAC data
        self.roles: Dict[str, Role] = {}
        self.subjects: Dict[UUID, Subject] = {}
        
        # Built-in roles
        self._create_builtin_roles()
    
    def create_role(self, name: str, description: str, created_by: Optional[UUID] = None) -> Role:
        """Create a new role."""
        if name in self.roles:
            raise SecurityError(f"Role '{name}' already exists")
        
        role = Role(
            name=name,
            description=description,
            created_by=created_by
        )
        
        self.roles[name] = role
        
        self.logger.info("Role created", name=name, created_by=str(created_by) if created_by else None)
        return role
    
    def add_permission_to_role(
        self, 
        role_name: str, 
        permission: Permission
    ) -> None:
        """Add permission to role."""
        if role_name not in self.roles:
            raise SecurityError(f"Role '{role_name}' does not exist")
        
        self.roles[role_name].permissions.add(permission)
        
        self.logger.info("Permission added to role", 
                        role=role_name,
                        resource_type=permission.resource_type.value,
                        action=permission.action)
    
    def assign_role_to_subject(self, subject_id: UUID, role_name: str) -> None:
        """Assign role to subject."""
        if role_name not in self.roles:
            raise SecurityError(f"Role '{role_name}' does not exist")
        
        if subject_id not in self.subjects:
            raise SecurityError(f"Subject '{subject_id}' does not exist")
        
        self.subjects[subject_id].roles.add(role_name)
        
        self.logger.info("Role assigned to subject", 
                        subject_id=str(subject_id), role=role_name)
    
    def register_subject(
        self, 
        subject_id: UUID, 
        subject_type: str, 
        display_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Subject:
        """Register a new subject."""
        subject = Subject(
            subject_id=subject_id,
            subject_type=subject_type,
            display_name=display_name,
            attributes=attributes or {}
        )
        
        self.subjects[subject_id] = subject
        
        self.logger.info("Subject registered", 
                        subject_id=str(subject_id),
                        subject_type=subject_type,
                        display_name=display_name)
        
        return subject
    
    def get_subject_permissions(self, subject_id: UUID) -> Set[Permission]:
        """Get all permissions for a subject (from roles and direct)."""
        if subject_id not in self.subjects:
            return set()
        
        subject = self.subjects[subject_id]
        all_permissions = set(subject.direct_permissions)
        
        # Collect permissions from roles
        for role_name in subject.roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                all_permissions.update(role.permissions)
                
                # Handle role inheritance
                inherited_permissions = self._get_inherited_permissions(role)
                all_permissions.update(inherited_permissions)
        
        # Filter out expired permissions
        valid_permissions = {p for p in all_permissions if not p.is_expired()}
        
        return valid_permissions
    
    def check_permission(
        self, 
        subject_id: UUID, 
        resource_type: ResourceType, 
        resource_id: str, 
        action: str
    ) -> bool:
        """Check if subject has permission for specific access."""
        permissions = self.get_subject_permissions(subject_id)
        
        for permission in permissions:
            if permission.matches(resource_type, resource_id, action):
                return True
        
        return False
    
    def _create_builtin_roles(self) -> None:
        """Create built-in system roles."""
        # System Administrator
        admin_role = Role(
            name="system_admin",
            description="Full system administration access"
        )
        admin_role.permissions.add(Permission(
            resource_type=ResourceType.NODE,
            resource_id="*",
            action="*",
            access_level=AccessLevel.SYSTEM
        ))
        self.roles["system_admin"] = admin_role
        
        # Node Operator
        node_operator = Role(
            name="node_operator",
            description="Node operation and management"
        )
        node_operator.permissions.update([
            Permission(ResourceType.NODE, "*", "read", AccessLevel.READ),
            Permission(ResourceType.NODE, "*", "write", AccessLevel.WRITE),
            Permission(ResourceType.TASK, "*", "execute", AccessLevel.EXECUTE),
            Permission(ResourceType.METRICS, "*", "read", AccessLevel.READ)
        ])
        self.roles["node_operator"] = node_operator
        
        # Task Executor
        task_executor = Role(
            name="task_executor",
            description="Task execution permissions"
        )
        task_executor.permissions.update([
            Permission(ResourceType.TASK, "*", "read", AccessLevel.READ),
            Permission(ResourceType.TASK, "*", "execute", AccessLevel.EXECUTE),
            Permission(ResourceType.DATA, "*", "read", AccessLevel.READ)
        ])
        self.roles["task_executor"] = task_executor
        
        # Observer
        observer = Role(
            name="observer",
            description="Read-only access to system"
        )
        observer.permissions.update([
            Permission(ResourceType.NODE, "*", "read", AccessLevel.READ),
            Permission(ResourceType.METRICS, "*", "read", AccessLevel.READ),
            Permission(ResourceType.LOGS, "*", "read", AccessLevel.READ)
        ])
        self.roles["observer"] = observer
    
    def _get_inherited_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions from inherited roles."""
        inherited_permissions = set()
        
        for parent_role_name in role.inherits_from:
            if parent_role_name in self.roles:
                parent_role = self.roles[parent_role_name]
                inherited_permissions.update(parent_role.permissions)
                
                # Recursive inheritance
                nested_inherited = self._get_inherited_permissions(parent_role)
                inherited_permissions.update(nested_inherited)
        
        return inherited_permissions


class AttributeBasedAccessControl:
    """Attribute-Based Access Control (ABAC) implementation."""
    
    def __init__(self):
        self.logger = structlog.get_logger("abac")
        
        # ABAC policies
        self.policies: Dict[str, SecurityPolicy] = {}
        
        # Attribute evaluators
        self.attribute_evaluators: Dict[str, Callable] = {}
        
        # Setup default evaluators
        self._setup_default_evaluators()
    
    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add security policy."""
        self.policies[policy.policy_id] = policy
        
        self.logger.info("Security policy added", 
                        policy_id=policy.policy_id,
                        name=policy.name,
                        effect=policy.effect.value)
    
    def register_attribute_evaluator(self, name: str, evaluator: Callable) -> None:
        """Register custom attribute evaluator."""
        self.attribute_evaluators[name] = evaluator
        self.logger.info("Attribute evaluator registered", name=name)
    
    def evaluate_policies(
        self, 
        request: AccessRequest, 
        subject: Subject,
        context: Dict[str, Any]
    ) -> AccessDecision:
        """Evaluate policies against access request."""
        decision = AccessDecision(
            request_id=request.request_id,
            effect=PolicyEffect.DENY,
            evaluator="abac"
        )
        
        # Sort policies by priority (higher first)
        sorted_policies = sorted(
            [p for p in self.policies.values() if p.is_active],
            key=lambda p: p.priority,
            reverse=True
        )
        
        for policy in sorted_policies:
            policy_result = self._evaluate_policy(policy, request, subject, context)
            
            if policy_result["matches"]:
                decision.effect = policy.effect
                decision.metadata["matched_policy"] = policy.policy_id
                
                if policy.effect == PolicyEffect.ALLOW:
                    decision.granted_permissions.append(
                        Permission(
                            resource_type=request.resource_type,
                            resource_id=request.resource_id,
                            action=request.action,
                            access_level=AccessLevel.EXECUTE
                        )
                    )
                    break  # First allow wins
                elif policy.effect == PolicyEffect.DENY:
                    decision.denied_reasons.append(f"Policy {policy.name} denied access")
                    break  # First deny wins
                elif policy.effect == PolicyEffect.CONDITIONAL:
                    decision.conditions.update(policy_result["conditions"])
        
        return decision
    
    def _evaluate_policy(
        self, 
        policy: SecurityPolicy, 
        request: AccessRequest, 
        subject: Subject,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate individual policy."""
        result = {"matches": False, "conditions": {}}
        
        try:
            # Evaluate all conditions
            for condition_name, condition_value in policy.conditions.items():
                if not self._evaluate_condition(
                    condition_name, condition_value, request, subject, context
                ):
                    return result
            
            result["matches"] = True
            
            # Extract any conditional requirements
            if policy.effect == PolicyEffect.CONDITIONAL:
                result["conditions"] = policy.conditions.get("requirements", {})
            
            return result
            
        except Exception as e:
            self.logger.error("Policy evaluation error", 
                            policy_id=policy.policy_id, error=str(e))
            return result
    
    def _evaluate_condition(
        self, 
        condition_name: str, 
        condition_value: Any, 
        request: AccessRequest,
        subject: Subject, 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate individual condition."""
        try:
            if condition_name == "subject_type":
                return subject.subject_type == condition_value
            
            elif condition_name == "resource_type":
                return request.resource_type.value == condition_value
            
            elif condition_name == "action":
                return request.action == condition_value
            
            elif condition_name == "time_range":
                current_hour = time.gmtime().tm_hour
                start_hour = condition_value.get("start", 0)
                end_hour = condition_value.get("end", 23)
                return start_hour <= current_hour <= end_hour
            
            elif condition_name == "source_network":
                if not request.source_ip:
                    return False
                # Simple network check - in production would use proper IP matching
                return request.source_ip.startswith(condition_value)
            
            elif condition_name == "custom_attribute":
                attr_name = condition_value.get("name")
                attr_value = condition_value.get("value")
                return subject.attributes.get(attr_name) == attr_value
            
            elif condition_name in self.attribute_evaluators:
                evaluator = self.attribute_evaluators[condition_name]
                return evaluator(condition_value, request, subject, context)
            
            else:
                self.logger.warning("Unknown condition type", condition=condition_name)
                return False
                
        except Exception as e:
            self.logger.error("Condition evaluation error", 
                            condition=condition_name, error=str(e))
            return False
    
    def _setup_default_evaluators(self) -> None:
        """Setup default attribute evaluators."""
        def risk_score_evaluator(condition_value, request, subject, context):
            """Evaluate based on risk score."""
            risk_score = context.get("risk_score", 0)
            threshold = condition_value.get("threshold", 50)
            operator = condition_value.get("operator", "less_than")
            
            if operator == "less_than":
                return risk_score < threshold
            elif operator == "greater_than":
                return risk_score > threshold
            else:
                return risk_score == threshold
        
        self.register_attribute_evaluator("risk_score", risk_score_evaluator)


class ComprehensiveAccessControl:
    """Comprehensive access control system combining RBAC and ABAC."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("access_control", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Access control systems
        self.rbac = RoleBasedAccessControl()
        self.abac = AttributeBasedAccessControl()
        
        # Access audit trail
        self.access_history: List[Dict[str, Any]] = []
        self.failed_access_attempts: Dict[UUID, int] = {}
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.locked_subjects: Dict[UUID, float] = {}
        
        # Authentication handlers
        self.auth_handlers: Dict[AuthenticationMethod, Callable] = {}
        
        # Setup default policies
        self._setup_default_policies()
    
    async def authenticate(
        self, 
        subject_id: UUID, 
        credentials: Dict[str, Any],
        method: AuthenticationMethod = AuthenticationMethod.CRYPTOGRAPHIC_SIGNATURE
    ) -> bool:
        """Authenticate a subject."""
        try:
            # Check if subject is locked out
            if self._is_subject_locked(subject_id):
                self.logger.warning("Authentication attempt from locked subject", 
                                  subject_id=str(subject_id))
                return False
            
            # Use appropriate authentication handler
            if method not in self.auth_handlers:
                self.logger.error("Unsupported authentication method", method=method.value)
                return False
            
            auth_handler = self.auth_handlers[method]
            authenticated = await auth_handler(subject_id, credentials)
            
            if authenticated:
                # Reset failed attempts on successful auth
                if subject_id in self.failed_access_attempts:
                    del self.failed_access_attempts[subject_id]
                
                # Update subject activity
                if subject_id in self.rbac.subjects:
                    self.rbac.subjects[subject_id].update_activity()
                
                self.logger.info("Authentication successful", 
                               subject_id=str(subject_id), method=method.value)
            else:
                # Track failed attempts
                self.failed_access_attempts[subject_id] = \
                    self.failed_access_attempts.get(subject_id, 0) + 1
                
                # Lock subject if too many failures
                if self.failed_access_attempts[subject_id] >= self.max_failed_attempts:
                    self.locked_subjects[subject_id] = time.time() + self.lockout_duration
                    self.logger.warning("Subject locked due to failed attempts", 
                                      subject_id=str(subject_id))
                
                self.logger.warning("Authentication failed", 
                                  subject_id=str(subject_id), method=method.value)
            
            return authenticated
            
        except Exception as e:
            self.logger.error("Authentication error", 
                            subject_id=str(subject_id), error=str(e))
            return False
    
    async def authorize(self, request: AccessRequest) -> AccessDecision:
        """Authorize access request using combined RBAC and ABAC."""
        try:
            # Check if subject is locked
            if self._is_subject_locked(request.subject_id):
                return AccessDecision(
                    request_id=request.request_id,
                    effect=PolicyEffect.DENY,
                    denied_reasons=["Subject is locked out"]
                )
            
            # Get subject
            if request.subject_id not in self.rbac.subjects:
                return AccessDecision(
                    request_id=request.request_id,
                    effect=PolicyEffect.DENY,
                    denied_reasons=["Unknown subject"]
                )
            
            subject = self.rbac.subjects[request.subject_id]
            
            # First check RBAC permissions
            rbac_allowed = self.rbac.check_permission(
                request.subject_id,
                request.resource_type,
                request.resource_id,
                request.action
            )
            
            if not rbac_allowed:
                decision = AccessDecision(
                    request_id=request.request_id,
                    effect=PolicyEffect.DENY,
                    denied_reasons=["RBAC: Insufficient permissions"],
                    evaluator="rbac"
                )
            else:
                # RBAC allows, now check ABAC policies
                decision = self.abac.evaluate_policies(request, subject, request.context)
                
                if decision.effect == PolicyEffect.DENY and not decision.denied_reasons:
                    decision.denied_reasons = ["ABAC: No matching allow policy"]
            
            # Audit the decision
            await self._audit_access_decision(request, decision, subject)
            
            return decision
            
        except Exception as e:
            self.logger.error("Authorization error", 
                            request_id=str(request.request_id), error=str(e))
            
            return AccessDecision(
                request_id=request.request_id,
                effect=PolicyEffect.DENY,
                denied_reasons=[f"Authorization error: {str(e)}"]
            )
    
    def register_authentication_handler(
        self, 
        method: AuthenticationMethod, 
        handler: Callable
    ) -> None:
        """Register authentication handler for specific method."""
        self.auth_handlers[method] = handler
        self.logger.info("Authentication handler registered", method=method.value)
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics."""
        total_subjects = len(self.rbac.subjects)
        active_subjects = sum(1 for s in self.rbac.subjects.values() if s.is_active)
        locked_subjects = len([s for s, t in self.locked_subjects.items() if t > time.time()])
        
        recent_access = [a for a in self.access_history if time.time() - a["timestamp"] < 3600]
        allowed_access = sum(1 for a in recent_access if a["allowed"])
        denied_access = len(recent_access) - allowed_access
        
        return {
            "total_subjects": total_subjects,
            "active_subjects": active_subjects,
            "locked_subjects": locked_subjects,
            "total_roles": len(self.rbac.roles),
            "total_policies": len(self.abac.policies),
            "recent_access_attempts": len(recent_access),
            "recent_allowed": allowed_access,
            "recent_denied": denied_access,
            "access_success_rate": allowed_access / max(1, len(recent_access))
        }
    
    # Private methods
    
    def _is_subject_locked(self, subject_id: UUID) -> bool:
        """Check if subject is currently locked out."""
        if subject_id not in self.locked_subjects:
            return False
        
        lockout_time = self.locked_subjects[subject_id]
        if time.time() > lockout_time:
            # Lockout expired
            del self.locked_subjects[subject_id]
            return False
        
        return True
    
    async def _audit_access_decision(
        self, 
        request: AccessRequest, 
        decision: AccessDecision, 
        subject: Subject
    ) -> None:
        """Audit access decision for compliance and monitoring."""
        audit_entry = {
            "timestamp": time.time(),
            "request_id": str(request.request_id),
            "subject_id": str(request.subject_id),
            "subject_type": subject.subject_type,
            "resource_type": request.resource_type.value,
            "resource_id": request.resource_id,
            "action": request.action,
            "allowed": decision.effect == PolicyEffect.ALLOW,
            "evaluator": decision.evaluator,
            "decision_time": decision.decision_time,
            "source_ip": request.source_ip,
            "denied_reasons": decision.denied_reasons
        }
        
        self.access_history.append(audit_entry)
        
        # Keep history manageable
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-5000:]
        
        # Log significant events
        if not audit_entry["allowed"]:
            self.logger.warning("Access denied", 
                              subject_id=str(request.subject_id),
                              resource=f"{request.resource_type.value}:{request.resource_id}",
                              action=request.action,
                              reasons=decision.denied_reasons)
        elif decision.conditions:
            self.logger.info("Conditional access granted", 
                           subject_id=str(request.subject_id),
                           conditions=sanitize_for_logging(decision.conditions))
    
    def _setup_default_policies(self) -> None:
        """Setup default ABAC policies."""
        # Time-based access policy
        time_policy = SecurityPolicy(
            policy_id="time_based_access",
            name="Time-based Access Control",
            description="Restrict access based on time of day",
            effect=PolicyEffect.CONDITIONAL,
            conditions={
                "time_range": {"start": 6, "end": 22},  # 6 AM to 10 PM
                "requirements": {"additional_auth": True}
            },
            priority=200
        )
        self.abac.add_policy(time_policy)
        
        # High-risk operation policy
        high_risk_policy = SecurityPolicy(
            policy_id="high_risk_operations",
            name="High Risk Operations",
            description="Additional controls for high-risk operations",
            effect=PolicyEffect.CONDITIONAL,
            conditions={
                "action": "admin",
                "requirements": {"multi_factor_auth": True, "audit_trail": True}
            },
            priority=300
        )
        self.abac.add_policy(high_risk_policy)
        
        # Emergency access policy
        emergency_policy = SecurityPolicy(
            policy_id="emergency_access",
            name="Emergency Access",
            description="Emergency access during system issues",
            effect=PolicyEffect.ALLOW,
            conditions={
                "custom_attribute": {"name": "emergency_role", "value": "responder"}
            },
            priority=500
        )
        self.abac.add_policy(emergency_policy)


# Global access control instance
_global_access_control: Optional[ComprehensiveAccessControl] = None


def get_access_control(node_id: Optional[UUID] = None) -> ComprehensiveAccessControl:
    """Get global access control instance."""
    global _global_access_control
    if _global_access_control is None:
        _global_access_control = ComprehensiveAccessControl(node_id)
    return _global_access_control


async def check_access(
    subject_id: UUID, 
    resource_type: ResourceType, 
    resource_id: str, 
    action: str,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Check access using global access control."""
    access_control = get_access_control()
    
    request = AccessRequest(
        subject_id=subject_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        context=context or {}
    )
    
    decision = await access_control.authorize(request)
    return decision.effect == PolicyEffect.ALLOW


def require_permission(resource_type: ResourceType, action: str):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract subject_id from function context
            # This would need to be adapted based on your function signature
            subject_id = kwargs.get('subject_id') or getattr(args[0], 'node_id', None)
            resource_id = kwargs.get('resource_id', "*")
            
            if not subject_id:
                raise SecurityError("No subject ID available for permission check")
            
            # Check permission
            has_access = await check_access(subject_id, resource_type, resource_id, action)
            
            if not has_access:
                raise SecurityError(
                    f"Access denied: {action} on {resource_type.value}:{resource_id}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator