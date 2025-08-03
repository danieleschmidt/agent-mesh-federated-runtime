"""Security and cryptographic operations.

This module provides security services for the Agent Mesh system including:
- Identity management and PKI
- End-to-end encryption
- Digital signatures
- Access control and authentication
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, Field
import nacl.secret
import nacl.utils
import nacl.signing
import nacl.encoding


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    
    CHACHA20_POLY1305 = "chacha20_poly1305"
    AES_256_GCM = "aes_256_gcm"
    NACL_SECRETBOX = "nacl_secretbox"


class SignatureAlgorithm(Enum):
    """Supported digital signature algorithms."""
    
    ED25519 = "ed25519"
    RSA_PSS = "rsa_pss"
    ECDSA = "ecdsa"


class AccessLevel(Enum):
    """Access control levels."""
    
    GUEST = "guest"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class NodeIdentity:
    """Node identity and cryptographic keys."""
    
    node_id: UUID
    public_key: bytes
    public_key_hex: str
    certificate: Optional[bytes] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if identity has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class EncryptedMessage:
    """Encrypted message container."""
    
    ciphertext: bytes
    nonce: bytes
    algorithm: EncryptionAlgorithm
    sender_id: UUID
    recipient_id: Optional[UUID] = None
    timestamp: float = field(default_factory=time.time)
    signature: Optional[bytes] = None


@dataclass
class AccessControlEntry:
    """Access control entry for permission management."""
    
    subject_id: UUID  # Node or user ID
    resource: str     # Resource identifier
    access_level: AccessLevel
    granted_by: UUID
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if access has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class SecurityAuditEvent(BaseModel):
    """Security audit event for logging."""
    
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    actor_id: UUID
    resource: str
    action: str
    success: bool
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical


class KeyPair:
    """Cryptographic key pair container."""
    
    def __init__(self, private_key: Any, public_key: Any, algorithm: SignatureAlgorithm):
        self.private_key = private_key
        self.public_key = public_key
        self.algorithm = algorithm
        self.created_at = time.time()
    
    def sign(self, data: bytes) -> bytes:
        """Sign data with private key."""
        if self.algorithm == SignatureAlgorithm.ED25519:
            return self.private_key.sign(data)
        elif self.algorithm == SignatureAlgorithm.RSA_PSS:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            return self.private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        else:
            raise ValueError(f"Unsupported signature algorithm: {self.algorithm}")
    
    def verify(self, data: bytes, signature: bytes, public_key: Any = None) -> bool:
        """Verify signature against data."""
        verify_key = public_key or self.public_key
        
        try:
            if self.algorithm == SignatureAlgorithm.ED25519:
                verify_key.verify(signature, data)
                return True
            elif self.algorithm == SignatureAlgorithm.RSA_PSS:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                verify_key.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
        except Exception:
            return False
        
        return False
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key as bytes."""
        if self.algorithm == SignatureAlgorithm.ED25519:
            return bytes(self.public_key)
        elif self.algorithm == SignatureAlgorithm.RSA_PSS:
            return self.public_key.public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


class SecurityManager:
    """
    Main security manager for the Agent Mesh system.
    
    Handles all cryptographic operations including:
    - Identity management
    - Encryption/decryption
    - Digital signatures
    - Access control
    - Security auditing
    """
    
    def __init__(
        self,
        node_id: Optional[UUID] = None,
        key_algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519,
        encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.NACL_SECRETBOX
    ):
        """
        Initialize security manager.
        
        Args:
            node_id: Node identifier (generated if None)
            key_algorithm: Algorithm for digital signatures
            encryption_algorithm: Algorithm for encryption
        """
        self.node_id = node_id or uuid4()
        self.key_algorithm = key_algorithm
        self.encryption_algorithm = encryption_algorithm
        
        self.logger = structlog.get_logger("security_manager", node_id=str(self.node_id))
        
        # Cryptographic state
        self._identity: Optional[NodeIdentity] = None
        self._key_pair: Optional[KeyPair] = None
        self._session_keys: Dict[UUID, bytes] = {}  # Peer session keys
        
        # Access control
        self._access_control: Dict[str, List[AccessControlEntry]] = {}
        self._permissions_cache: Dict[Tuple[UUID, str], bool] = {}
        
        # Security audit
        self._audit_events: List[SecurityAuditEvent] = []
        self._threat_detection: Dict[str, int] = {}
        
        # Encryption backends
        self._encryption_backends = {
            EncryptionAlgorithm.NACL_SECRETBOX: self._nacl_encrypt,
            EncryptionAlgorithm.AES_256_GCM: self._aes_encrypt,
            EncryptionAlgorithm.CHACHA20_POLY1305: self._chacha20_encrypt
        }
        
        self._decryption_backends = {
            EncryptionAlgorithm.NACL_SECRETBOX: self._nacl_decrypt,
            EncryptionAlgorithm.AES_256_GCM: self._aes_decrypt,
            EncryptionAlgorithm.CHACHA20_POLY1305: self._chacha20_decrypt
        }
    
    async def initialize(self) -> None:
        """Initialize security manager and generate identity."""
        self.logger.info("Initializing security manager")
        
        # Generate key pair
        await self._generate_key_pair()
        
        # Create node identity
        public_key_bytes = self._key_pair.get_public_key_bytes()
        self._identity = NodeIdentity(
            node_id=self.node_id,
            public_key=public_key_bytes,
            public_key_hex=public_key_bytes.hex()
        )
        
        # Set up default permissions
        await self._setup_default_permissions()
        
        self.logger.info("Security manager initialized",
                        public_key=self._identity.public_key_hex[:16] + "...")
    
    async def cleanup(self) -> None:
        """Cleanup security resources."""
        self.logger.info("Cleaning up security manager")
        
        # Clear sensitive data
        self._session_keys.clear()
        self._permissions_cache.clear()
        
        # Audit cleanup
        await self._audit_security_event(
            "security_cleanup",
            self.node_id,
            "system",
            "cleanup",
            True
        )
    
    async def get_node_identity(self) -> NodeIdentity:
        """Get current node identity."""
        if not self._identity:
            raise RuntimeError("Security manager not initialized")
        return self._identity
    
    async def encrypt_message(
        self, 
        data: bytes, 
        recipient_id: Optional[UUID] = None
    ) -> EncryptedMessage:
        """
        Encrypt data for transmission.
        
        Args:
            data: Raw data to encrypt
            recipient_id: Target recipient (None for broadcast)
            
        Returns:
            EncryptedMessage with encrypted data
        """
        # Get or generate session key
        if recipient_id:
            session_key = await self._get_session_key(recipient_id)
        else:
            # Use random key for broadcast (less secure but necessary)
            session_key = nacl.utils.random(32)
        
        # Encrypt using selected algorithm
        encrypt_func = self._encryption_backends[self.encryption_algorithm]
        ciphertext, nonce = await encrypt_func(data, session_key)
        
        encrypted_msg = EncryptedMessage(
            ciphertext=ciphertext,
            nonce=nonce,
            algorithm=self.encryption_algorithm,
            sender_id=self.node_id,
            recipient_id=recipient_id
        )
        
        # Sign the encrypted message
        signature_data = self._prepare_signature_data(encrypted_msg)
        encrypted_msg.signature = self._key_pair.sign(signature_data)
        
        await self._audit_security_event(
            "message_encrypted",
            self.node_id,
            "message",
            "encrypt",
            True,
            {"recipient_id": str(recipient_id) if recipient_id else "broadcast"}
        )
        
        return encrypted_msg
    
    async def decrypt_message(self, encrypted_msg: EncryptedMessage) -> bytes:
        """
        Decrypt received message.
        
        Args:
            encrypted_msg: Encrypted message to decrypt
            
        Returns:
            Decrypted data
        """
        # Verify signature first
        if encrypted_msg.signature:
            signature_data = self._prepare_signature_data(encrypted_msg)
            sender_public_key = await self._get_peer_public_key(encrypted_msg.sender_id)
            
            if not self._verify_signature(signature_data, encrypted_msg.signature, sender_public_key):
                await self._audit_security_event(
                    "signature_verification_failed",
                    encrypted_msg.sender_id,
                    "message",
                    "decrypt",
                    False,
                    risk_level="high"
                )
                raise ValueError("Message signature verification failed")
        
        # Get session key
        session_key = await self._get_session_key(encrypted_msg.sender_id)
        
        # Decrypt using appropriate algorithm
        decrypt_func = self._decryption_backends[encrypted_msg.algorithm]
        plaintext = await decrypt_func(encrypted_msg.ciphertext, encrypted_msg.nonce, session_key)
        
        await self._audit_security_event(
            "message_decrypted",
            encrypted_msg.sender_id,
            "message",
            "decrypt",
            True
        )
        
        return plaintext
    
    async def sign_data(self, data: bytes) -> bytes:
        """Sign data with node's private key."""
        if not self._key_pair:
            raise RuntimeError("Key pair not initialized")
        
        signature = self._key_pair.sign(data)
        
        await self._audit_security_event(
            "data_signed",
            self.node_id,
            "data",
            "sign",
            True
        )
        
        return signature
    
    async def verify_signature(self, data: bytes, signature: bytes, signer_id: UUID) -> bool:
        """Verify signature from another node."""
        try:
            public_key = await self._get_peer_public_key(signer_id)
            result = self._verify_signature(data, signature, public_key)
            
            await self._audit_security_event(
                "signature_verified",
                signer_id,
                "data",
                "verify",
                result,
                risk_level="medium" if not result else "low"
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Signature verification error", error=str(e))
            await self._audit_security_event(
                "signature_verification_error",
                signer_id,
                "data",
                "verify",
                False,
                {"error": str(e)},
                risk_level="high"
            )
            return False
    
    async def establish_session_key(self, peer_id: UUID, peer_public_key: bytes) -> None:
        """Establish session key with a peer."""
        # Simple key derivation - in production would use proper key exchange
        combined_keys = self._identity.public_key + peer_public_key
        session_key = hashlib.sha256(combined_keys).digest()
        
        self._session_keys[peer_id] = session_key
        
        await self._audit_security_event(
            "session_key_established",
            peer_id,
            "session",
            "establish",
            True
        )
        
        self.logger.info("Session key established", peer_id=str(peer_id))
    
    async def check_permission(self, subject_id: UUID, resource: str, action: str) -> bool:
        """Check if subject has permission for action on resource."""
        cache_key = (subject_id, f"{resource}:{action}")
        
        # Check cache first
        if cache_key in self._permissions_cache:
            return self._permissions_cache[cache_key]
        
        # Find applicable access control entries
        resource_entries = self._access_control.get(resource, [])
        
        result = False
        for entry in resource_entries:
            if entry.subject_id == subject_id and not entry.is_expired():
                # Check if access level allows the action
                if self._access_level_allows_action(entry.access_level, action):
                    result = True
                    break
        
        # Cache result
        self._permissions_cache[cache_key] = result
        
        await self._audit_security_event(
            "permission_check",
            subject_id,
            resource,
            action,
            result,
            risk_level="medium" if not result else "low"
        )
        
        return result
    
    async def grant_permission(
        self,
        subject_id: UUID,
        resource: str,
        access_level: AccessLevel,
        expires_at: Optional[float] = None
    ) -> None:
        """Grant permission to a subject."""
        entry = AccessControlEntry(
            subject_id=subject_id,
            resource=resource,
            access_level=access_level,
            granted_by=self.node_id,
            expires_at=expires_at
        )
        
        if resource not in self._access_control:
            self._access_control[resource] = []
        
        self._access_control[resource].append(entry)
        
        # Clear cache for this subject/resource
        cache_keys_to_remove = [
            key for key in self._permissions_cache.keys()
            if key[0] == subject_id and key[1].startswith(resource)
        ]
        for key in cache_keys_to_remove:
            del self._permissions_cache[key]
        
        await self._audit_security_event(
            "permission_granted",
            subject_id,
            resource,
            "grant",
            True,
            {"access_level": access_level.value}
        )
        
        self.logger.info("Permission granted",
                        subject_id=str(subject_id),
                        resource=resource,
                        access_level=access_level.value)
    
    async def revoke_permission(self, subject_id: UUID, resource: str) -> None:
        """Revoke all permissions for subject on resource."""
        if resource in self._access_control:
            # Remove entries for this subject
            self._access_control[resource] = [
                entry for entry in self._access_control[resource]
                if entry.subject_id != subject_id
            ]
        
        # Clear cache
        cache_keys_to_remove = [
            key for key in self._permissions_cache.keys()
            if key[0] == subject_id and key[1].startswith(resource)
        ]
        for key in cache_keys_to_remove:
            del self._permissions_cache[key]
        
        await self._audit_security_event(
            "permission_revoked",
            subject_id,
            resource,
            "revoke",
            True
        )
    
    def get_audit_events(self, limit: int = 100) -> List[SecurityAuditEvent]:
        """Get recent security audit events."""
        return self._audit_events[-limit:]
    
    async def detect_threats(self) -> Dict[str, Any]:
        """Run threat detection analysis."""
        threats = {}
        
        # Analyze failed authentication attempts
        recent_events = self._audit_events[-1000:]  # Last 1000 events
        failed_auths = [e for e in recent_events 
                       if e.event_type in ["signature_verification_failed", "permission_check"]
                       and not e.success]
        
        if len(failed_auths) > 10:  # Threshold
            threats["excessive_failed_attempts"] = len(failed_auths)
        
        # Analyze access patterns
        permission_denials = [e for e in recent_events
                            if e.event_type == "permission_check" and not e.success]
        
        if len(permission_denials) > 20:
            threats["suspicious_access_patterns"] = len(permission_denials)
        
        return threats
    
    # Private methods
    
    async def _generate_key_pair(self) -> None:
        """Generate cryptographic key pair."""
        self.logger.info("Generating key pair", algorithm=self.key_algorithm.value)
        
        if self.key_algorithm == SignatureAlgorithm.ED25519:
            private_key = nacl.signing.SigningKey.generate()
            public_key = private_key.verify_key
            
            self._key_pair = KeyPair(private_key, public_key, self.key_algorithm)
            
        elif self.key_algorithm == SignatureAlgorithm.RSA_PSS:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            self._key_pair = KeyPair(private_key, public_key, self.key_algorithm)
            
        else:
            raise ValueError(f"Unsupported key algorithm: {self.key_algorithm}")
    
    async def _setup_default_permissions(self) -> None:
        """Setup default permissions for the node."""
        # Grant self admin access to all resources
        await self.grant_permission(
            self.node_id,
            "*",  # All resources
            AccessLevel.ADMIN
        )
    
    async def _get_session_key(self, peer_id: UUID) -> bytes:
        """Get or generate session key for peer."""
        if peer_id not in self._session_keys:
            # Generate temporary session key
            # In production, this would use proper key exchange
            self._session_keys[peer_id] = nacl.utils.random(32)
        
        return self._session_keys[peer_id]
    
    async def _get_peer_public_key(self, peer_id: UUID) -> Any:
        """Get public key for a peer."""
        # In production, this would query a key registry or PKI
        # For now, return a placeholder
        if self.key_algorithm == SignatureAlgorithm.ED25519:
            return nacl.signing.VerifyKey(b"placeholder_key_32_bytes_long!")
        else:
            raise NotImplementedError("RSA peer key lookup not implemented")
    
    def _verify_signature(self, data: bytes, signature: bytes, public_key: Any) -> bool:
        """Verify signature using public key."""
        try:
            if self.key_algorithm == SignatureAlgorithm.ED25519:
                public_key.verify(data, signature)
                return True
            else:
                # RSA verification would go here
                return False
        except Exception:
            return False
    
    def _prepare_signature_data(self, encrypted_msg: EncryptedMessage) -> bytes:
        """Prepare data for signing."""
        # Create signature data from message components
        data_to_sign = (
            encrypted_msg.ciphertext +
            encrypted_msg.nonce +
            str(encrypted_msg.sender_id).encode() +
            str(encrypted_msg.timestamp).encode()
        )
        return data_to_sign
    
    async def _nacl_encrypt(self, data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt using NaCl SecretBox."""
        box = nacl.secret.SecretBox(key)
        encrypted = box.encrypt(data)
        return encrypted.ciphertext, encrypted.nonce
    
    async def _nacl_decrypt(self, ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
        """Decrypt using NaCl SecretBox."""
        box = nacl.secret.SecretBox(key)
        encrypted_message = nacl.secret.EncryptedMessage(ciphertext, nonce)
        return box.decrypt(encrypted_message)
    
    async def _aes_encrypt(self, data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt using AES-256-GCM."""
        nonce = nacl.utils.random(12)  # 96-bit nonce for GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return ciphertext + encryptor.tag, nonce
    
    async def _aes_decrypt(self, ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
        """Decrypt using AES-256-GCM."""
        # Split ciphertext and tag
        tag = ciphertext[-16:]
        actual_ciphertext = ciphertext[:-16]
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(actual_ciphertext) + decryptor.finalize()
    
    async def _chacha20_encrypt(self, data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt using ChaCha20-Poly1305."""
        nonce = nacl.utils.random(12)
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return ciphertext, nonce
    
    async def _chacha20_decrypt(self, ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _access_level_allows_action(self, access_level: AccessLevel, action: str) -> bool:
        """Check if access level allows specific action."""
        action_requirements = {
            "read": [AccessLevel.GUEST, AccessLevel.PARTICIPANT, AccessLevel.VALIDATOR, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "write": [AccessLevel.PARTICIPANT, AccessLevel.VALIDATOR, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "validate": [AccessLevel.VALIDATOR, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "admin": [AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "system": [AccessLevel.SYSTEM]
        }
        
        required_levels = action_requirements.get(action, [AccessLevel.ADMIN])
        return access_level in required_levels
    
    async def _audit_security_event(
        self,
        event_type: str,
        actor_id: UUID,
        resource: str,
        action: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        risk_level: str = "low"
    ) -> None:
        """Record security audit event."""
        event = SecurityAuditEvent(
            event_type=event_type,
            actor_id=actor_id,
            resource=resource,
            action=action,
            success=success,
            metadata=metadata or {},
            risk_level=risk_level
        )
        
        self._audit_events.append(event)
        
        # Keep only recent events (memory management)
        if len(self._audit_events) > 10000:
            self._audit_events = self._audit_events[-5000:]
        
        # Log high-risk events
        if risk_level in ["high", "critical"]:
            self.logger.warning("High-risk security event",
                              event_type=event_type,
                              actor_id=str(actor_id),
                              success=success)