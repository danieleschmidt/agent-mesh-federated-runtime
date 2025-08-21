"""Advanced Security Manager with ML-based threat detection."""

import asyncio
import logging
import time
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import base64

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEvent(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE = "brute_force"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTED = "malware_detected"
    ANOMALOUS_TRAFFIC = "anomalous_traffic"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INJECTION_ATTEMPT = "injection_attempt"

@dataclass
class SecurityThreat:
    """Security threat detection."""
    threat_id: str
    threat_type: SecurityEvent
    level: ThreatLevel
    source_ip: str
    description: str
    evidence: Dict[str, Any]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    mitigated: bool = False

@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str  # "warn", "block", "quarantine"
    enabled: bool = True
    created: float = field(default_factory=time.time)

@dataclass
class AccessAttempt:
    """Access attempt record."""
    attempt_id: str
    source_ip: str
    user_agent: str
    endpoint: str
    method: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0

class AdvancedSecurityManager:
    """ML-enhanced security manager with adaptive threat detection."""
    
    def __init__(
        self,
        threat_scan_interval: float = 10.0,
        max_failed_attempts: int = 5,
        lockout_duration: float = 300.0  # 5 minutes
    ):
        self.threat_scan_interval = threat_scan_interval
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        
        # Security state
        self.detected_threats: Dict[str, SecurityThreat] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.access_attempts: List[AccessAttempt] = []
        self.blocked_ips: Dict[str, float] = {}  # IP -> block_until_timestamp
        self.security_events: List[Dict[str, Any]] = []
        
        # ML-based detection
        self.anomaly_baselines: Dict[str, Dict[str, float]] = {}
        self.behavioral_patterns: Dict[str, List[float]] = {}
        self.threat_signatures: Set[str] = set()
        
        # Authentication and encryption
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}  # IP -> timestamps
        self.rate_limit_threshold = 100  # requests per minute
        
        # State management
        self.is_running = False
        self._security_task: Optional[asyncio.Task] = None
        
        # Initialize default policies
        self._initialize_default_policies()
        self._initialize_threat_signatures()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        
        # Brute force protection
        brute_force_policy = SecurityPolicy(
            policy_id="brute_force_protection",
            name="Brute Force Protection",
            description="Detect and prevent brute force attacks",
            rules=[
                {"condition": "failed_attempts > 5", "action": "block_ip"},
                {"condition": "requests_per_minute > 100", "action": "rate_limit"}
            ],
            enforcement_level="block"
        )
        self.security_policies[brute_force_policy.policy_id] = brute_force_policy
        
        # Anomaly detection policy
        anomaly_policy = SecurityPolicy(
            policy_id="anomaly_detection",
            name="Anomaly Detection",
            description="Detect anomalous behavior patterns",
            rules=[
                {"condition": "deviation > 3_sigma", "action": "alert"},
                {"condition": "unusual_access_pattern", "action": "warn"}
            ],
            enforcement_level="warn"
        )
        self.security_policies[anomaly_policy.policy_id] = anomaly_policy
        
        # Data protection policy
        data_protection_policy = SecurityPolicy(
            policy_id="data_protection",
            name="Data Protection",
            description="Protect sensitive data access",
            rules=[
                {"condition": "sensitive_endpoint_access", "action": "log_and_monitor"},
                {"condition": "bulk_data_access", "action": "alert"}
            ],
            enforcement_level="warn"
        )
        self.security_policies[data_protection_policy.policy_id] = data_protection_policy
    
    def _initialize_threat_signatures(self):
        """Initialize known threat signatures."""
        self.threat_signatures.update([
            "SELECT * FROM",  # SQL injection
            "UNION SELECT",
            "<script>",  # XSS
            "javascript:",
            "../../../",  # Path traversal
            "eval(",  # Code injection
            "system(",
            "exec(",
            "/etc/passwd",  # File access
            "/proc/",
        ])
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate secure API key."""
        # Generate random key
        key_data = secrets.token_bytes(32)
        api_key = base64.urlsafe_b64encode(key_data).decode('ascii').rstrip('=')
        
        # Store key metadata
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created": time.time(),
            "last_used": None,
            "usage_count": 0
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        key_info["last_used"] = time.time()
        key_info["usage_count"] += 1
        
        return key_info
    
    def create_secure_session(self, user_id: str, ip_address: str) -> str:
        """Create secure session token."""
        # Generate session token
        session_data = f"{user_id}:{ip_address}:{time.time()}:{secrets.token_hex(16)}"
        session_token = hashlib.sha256(session_data.encode()).hexdigest()
        
        # Store session
        self.active_sessions[session_token] = {
            "user_id": user_id,
            "ip_address": ip_address,
            "created": time.time(),
            "last_activity": time.time(),
            "expires": time.time() + 3600  # 1 hour
        }
        
        return session_token
    
    def validate_session(self, session_token: str, ip_address: str) -> bool:
        """Validate session token."""
        if session_token not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_token]
        current_time = time.time()
        
        # Check expiration
        if current_time > session["expires"]:
            del self.active_sessions[session_token]
            return False
        
        # Check IP consistency (optional security measure)
        if session["ip_address"] != ip_address:
            logger.warning(f"Session IP mismatch: {session['ip_address']} vs {ip_address}")
            # Could be legitimate (mobile users), so just log
        
        # Update last activity
        session["last_activity"] = current_time
        return True
    
    def encrypt_data(self, data: str, key_id: str = "default") -> str:
        """Encrypt sensitive data."""
        if key_id not in self.encryption_keys:
            # Generate new encryption key
            self.encryption_keys[key_id] = secrets.token_bytes(32)
        
        # Simple XOR encryption (in production, use proper encryption like AES)
        key = self.encryption_keys[key_id]
        encrypted = bytearray()
        
        data_bytes = data.encode('utf-8')
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key[i % len(key)])
        
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_data(self, encrypted_data: str, key_id: str = "default") -> Optional[str]:
        """Decrypt sensitive data."""
        if key_id not in self.encryption_keys:
            return None
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            key = self.encryption_keys[key_id]
            
            decrypted = bytearray()
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key[i % len(key)])
            
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None
    
    def record_access_attempt(
        self,
        source_ip: str,
        user_agent: str,
        endpoint: str,
        method: str,
        success: bool,
        response_time: float = 0.0
    ) -> str:
        """Record access attempt for security analysis."""
        attempt_id = f"access_{int(time.time())}_{secrets.token_hex(4)}"
        
        attempt = AccessAttempt(
            attempt_id=attempt_id,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            success=success,
            response_time=response_time
        )
        
        self.access_attempts.append(attempt)
        
        # Keep history manageable
        if len(self.access_attempts) > 10000:
            self.access_attempts = self.access_attempts[-5000:]
        
        # Check for security threats
        asyncio.create_task(self._analyze_access_attempt(attempt))
        
        return attempt_id
    
    async def _analyze_access_attempt(self, attempt: AccessAttempt):
        """Analyze access attempt for security threats."""
        try:
            # Check if IP is already blocked
            if self._is_ip_blocked(attempt.source_ip):
                return
            
            # Analyze for brute force attacks
            await self._check_brute_force(attempt)
            
            # Analyze for rate limiting
            await self._check_rate_limit(attempt)
            
            # Analyze for anomalous patterns
            await self._check_anomalous_behavior(attempt)
            
            # Check for threat signatures
            await self._check_threat_signatures(attempt)
            
        except Exception as e:
            logger.error(f"Error analyzing access attempt: {e}")
    
    async def _check_brute_force(self, attempt: AccessAttempt):
        """Check for brute force attack patterns."""
        if attempt.success:
            return
        
        # Count recent failed attempts from this IP
        recent_window = time.time() - 300  # 5 minutes
        failed_attempts = [
            a for a in self.access_attempts[-100:]  # Last 100 attempts
            if (a.source_ip == attempt.source_ip and 
                not a.success and 
                a.timestamp > recent_window)
        ]
        
        if len(failed_attempts) >= self.max_failed_attempts:
            # Detected brute force attack
            threat = SecurityThreat(
                threat_id=f"brute_force_{attempt.source_ip}_{int(time.time())}",
                threat_type=SecurityEvent.BRUTE_FORCE,
                level=ThreatLevel.HIGH,
                source_ip=attempt.source_ip,
                description=f"Brute force attack detected: {len(failed_attempts)} failed attempts",
                evidence={
                    "failed_attempts": len(failed_attempts),
                    "time_window": "5 minutes",
                    "endpoints": list(set(a.endpoint for a in failed_attempts))
                },
                confidence=0.9
            )
            
            await self._handle_threat(threat)
            
            # Block IP
            self._block_ip(attempt.source_ip, self.lockout_duration)
    
    async def _check_rate_limit(self, attempt: AccessAttempt):
        """Check for rate limit violations."""
        current_time = time.time()
        
        # Track requests from this IP
        if attempt.source_ip not in self.rate_limits:
            self.rate_limits[attempt.source_ip] = []
        
        # Add current request
        self.rate_limits[attempt.source_ip].append(current_time)
        
        # Remove old requests (older than 1 minute)
        self.rate_limits[attempt.source_ip] = [
            t for t in self.rate_limits[attempt.source_ip]
            if current_time - t <= 60
        ]
        
        request_count = len(self.rate_limits[attempt.source_ip])
        
        if request_count > self.rate_limit_threshold:
            # Rate limit violation
            threat = SecurityThreat(
                threat_id=f"rate_limit_{attempt.source_ip}_{int(time.time())}",
                threat_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                level=ThreatLevel.MEDIUM,
                source_ip=attempt.source_ip,
                description=f"Rate limit exceeded: {request_count} requests per minute",
                evidence={
                    "requests_per_minute": request_count,
                    "threshold": self.rate_limit_threshold
                },
                confidence=0.8
            )
            
            await self._handle_threat(threat)
    
    async def _check_anomalous_behavior(self, attempt: AccessAttempt):
        """Check for anomalous behavior patterns."""
        # Build behavioral baseline for this IP
        ip_attempts = [
            a for a in self.access_attempts[-1000:]
            if a.source_ip == attempt.source_ip
        ]
        
        if len(ip_attempts) < 10:
            return  # Not enough data for analysis
        
        # Analyze patterns
        endpoints = [a.endpoint for a in ip_attempts]
        response_times = [a.response_time for a in ip_attempts]
        
        # Check for unusual endpoint access
        endpoint_variety = len(set(endpoints))
        if endpoint_variety > 20:  # Accessing many different endpoints
            threat = SecurityThreat(
                threat_id=f"anomaly_{attempt.source_ip}_{int(time.time())}",
                threat_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                level=ThreatLevel.MEDIUM,
                source_ip=attempt.source_ip,
                description=f"Anomalous behavior: accessing {endpoint_variety} different endpoints",
                evidence={
                    "endpoint_variety": endpoint_variety,
                    "unique_endpoints": list(set(endpoints[-20:]))
                },
                confidence=0.6
            )
            
            await self._handle_threat(threat)
    
    async def _check_threat_signatures(self, attempt: AccessAttempt):
        """Check for known threat signatures in request."""
        suspicious_patterns = []
        
        # Check endpoint for threat signatures
        for signature in self.threat_signatures:
            if signature.lower() in attempt.endpoint.lower():
                suspicious_patterns.append(signature)
        
        # Check user agent for suspicious patterns
        suspicious_user_agents = [
            "sqlmap", "nikto", "nmap", "masscan", "zap", "burp"
        ]
        
        for agent in suspicious_user_agents:
            if agent.lower() in attempt.user_agent.lower():
                suspicious_patterns.append(f"suspicious_user_agent:{agent}")
        
        if suspicious_patterns:
            threat = SecurityThreat(
                threat_id=f"signature_{attempt.source_ip}_{int(time.time())}",
                threat_type=SecurityEvent.INJECTION_ATTEMPT,
                level=ThreatLevel.HIGH,
                source_ip=attempt.source_ip,
                description=f"Threat signatures detected: {', '.join(suspicious_patterns)}",
                evidence={
                    "signatures": suspicious_patterns,
                    "endpoint": attempt.endpoint,
                    "user_agent": attempt.user_agent
                },
                confidence=0.95
            )
            
            await self._handle_threat(threat)
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is currently blocked."""
        if ip_address in self.blocked_ips:
            if time.time() < self.blocked_ips[ip_address]:
                return True
            else:
                # Block expired, remove it
                del self.blocked_ips[ip_address]
                return False
        return False
    
    def _block_ip(self, ip_address: str, duration: float):
        """Block IP address for specified duration."""
        self.blocked_ips[ip_address] = time.time() + duration
        logger.warning(f"Blocked IP {ip_address} for {duration} seconds")
    
    async def _handle_threat(self, threat: SecurityThreat):
        """Handle detected security threat."""
        self.detected_threats[threat.threat_id] = threat
        
        # Log security event
        security_event = {
            "timestamp": threat.timestamp,
            "threat_id": threat.threat_id,
            "type": threat.threat_type.value,
            "level": threat.level.value,
            "source_ip": threat.source_ip,
            "description": threat.description,
            "confidence": threat.confidence
        }
        
        self.security_events.append(security_event)
        
        # Keep events manageable
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
        
        # Take automatic actions based on threat level
        if threat.level == ThreatLevel.CRITICAL:
            self._block_ip(threat.source_ip, 3600)  # Block for 1 hour
            logger.critical(f"CRITICAL THREAT DETECTED: {threat.description}")
        elif threat.level == ThreatLevel.HIGH:
            self._block_ip(threat.source_ip, 900)  # Block for 15 minutes
            logger.error(f"HIGH THREAT DETECTED: {threat.description}")
        elif threat.level == ThreatLevel.MEDIUM:
            logger.warning(f"MEDIUM THREAT DETECTED: {threat.description}")
        else:
            logger.info(f"LOW THREAT DETECTED: {threat.description}")
    
    async def _security_monitoring_loop(self):
        """Main security monitoring loop."""
        logger.info("Advanced security monitoring started")
        
        while self.is_running:
            try:
                # Clean up expired blocks
                current_time = time.time()
                expired_blocks = [
                    ip for ip, block_until in self.blocked_ips.items()
                    if current_time >= block_until
                ]
                
                for ip in expired_blocks:
                    del self.blocked_ips[ip]
                    logger.info(f"Unblocked IP {ip} (block expired)")
                
                # Clean up expired sessions
                expired_sessions = [
                    token for token, session in self.active_sessions.items()
                    if current_time > session["expires"]
                ]
                
                for token in expired_sessions:
                    del self.active_sessions[token]
                
                # Analyze recent security events for patterns
                await self._analyze_security_patterns()
                
                await asyncio.sleep(self.threat_scan_interval)
                
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(self.threat_scan_interval)
    
    async def _analyze_security_patterns(self):
        """Analyze security events for emerging patterns."""
        if len(self.security_events) < 10:
            return
        
        recent_events = self.security_events[-50:]  # Last 50 events
        
        # Analyze by source IP
        ip_counts = {}
        for event in recent_events:
            ip = event.get("source_ip", "unknown")
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        # Look for coordinated attacks (multiple IPs, similar patterns)
        high_activity_ips = [ip for ip, count in ip_counts.items() if count >= 5]
        
        if len(high_activity_ips) >= 3:
            # Possible coordinated attack
            logger.warning(f"Possible coordinated attack detected from {len(high_activity_ips)} IPs")
    
    async def start(self):
        """Start the security manager."""
        if self.is_running:
            logger.warning("Security manager is already running")
            return
        
        self.is_running = True
        self._security_task = asyncio.create_task(self._security_monitoring_loop())
        logger.info("Advanced security manager started")
    
    async def stop(self):
        """Stop the security manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._security_task:
            self._security_task.cancel()
            try:
                await self._security_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced security manager stopped")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary."""
        current_time = time.time()
        
        # Count threats by level
        threat_counts = {}
        for threat in self.detected_threats.values():
            level = threat.level.value
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        # Recent activity
        recent_window = current_time - 3600  # Last hour
        recent_attempts = [
            a for a in self.access_attempts
            if a.timestamp > recent_window
        ]
        
        failed_attempts = [a for a in recent_attempts if not a.success]
        
        return {
            "threats": {
                "total": len(self.detected_threats),
                "by_level": threat_counts,
                "recent_24h": len([
                    t for t in self.detected_threats.values()
                    if current_time - t.timestamp <= 86400
                ])
            },
            "access_control": {
                "blocked_ips": len(self.blocked_ips),
                "active_sessions": len(self.active_sessions),
                "api_keys": len(self.api_keys)
            },
            "activity": {
                "total_attempts": len(self.access_attempts),
                "recent_hour": len(recent_attempts),
                "failed_attempts_hour": len(failed_attempts),
                "success_rate": (
                    (len(recent_attempts) - len(failed_attempts)) / max(len(recent_attempts), 1)
                )
            },
            "policies": {
                "total": len(self.security_policies),
                "enabled": len([p for p in self.security_policies.values() if p.enabled])
            }
        }
    
    def export_security_report(self) -> Dict[str, Any]:
        """Export comprehensive security report."""
        return {
            "summary": self.get_security_summary(),
            "threats": [
                {
                    "threat_id": t.threat_id,
                    "type": t.threat_type.value,
                    "level": t.level.value,
                    "source_ip": t.source_ip,
                    "description": t.description,
                    "confidence": t.confidence,
                    "timestamp": t.timestamp,
                    "mitigated": t.mitigated
                }
                for t in list(self.detected_threats.values())[-100:]  # Last 100 threats
            ],
            "recent_events": self.security_events[-100:],  # Last 100 events
            "blocked_ips": {
                ip: {"blocked_until": block_until, "remaining": max(0, block_until - time.time())}
                for ip, block_until in self.blocked_ips.items()
            },
            "policies": [
                {
                    "policy_id": p.policy_id,
                    "name": p.name,
                    "description": p.description,
                    "enforcement_level": p.enforcement_level,
                    "enabled": p.enabled
                }
                for p in self.security_policies.values()
            ]
        }