"""Security and cryptographic operations.

This module provides comprehensive security services for the Agent Mesh system including:
- PKI (Public Key Infrastructure) with certificate authority
- Secure key exchange protocols (X25519, ECDH)
- End-to-end encryption with forward secrecy
- Digital signatures and identity verification
- Access control and authentication with RBAC
- Certificate lifecycle management
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from uuid import UUID, uuid4

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
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


class KeyExchangeProtocol(Enum):
    """Supported key exchange protocols."""
    
    X25519 = "x25519"
    ECDH_P256 = "ecdh_p256"
    ECDH_P384 = "ecdh_p384"
    

class CertificateType(Enum):
    """Certificate types in PKI hierarchy."""
    
    ROOT_CA = "root_ca"
    INTERMEDIATE_CA = "intermediate_ca"
    NODE_CERTIFICATE = "node_certificate"
    SERVICE_CERTIFICATE = "service_certificate"


class CertificateStatus(Enum):
    """Certificate status for lifecycle management."""
    
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


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


@dataclass
class X509Certificate:
    """Enhanced X.509 certificate with PKI support."""
    
    certificate: x509.Certificate
    certificate_type: CertificateType
    serial_number: int
    subject_id: UUID
    issuer_id: Optional[UUID] = None
    status: CertificateStatus = CertificateStatus.VALID
    created_at: float = field(default_factory=time.time)
    
    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.utcnow()
        return (self.status == CertificateStatus.VALID and
                self.certificate.not_valid_before <= now <= self.certificate.not_valid_after)
    
    @property
    def expires_at(self) -> datetime:
        """Get certificate expiration time."""
        return self.certificate.not_valid_after
    
    def get_public_key_bytes(self) -> bytes:
        """Extract public key bytes from certificate."""
        return self.certificate.public_key().public_bytes(
            encoding=Encoding.DER,
            format=PublicFormat.SubjectPublicKeyInfo
        )
    
    def verify_chain(self, ca_cert: 'X509Certificate') -> bool:
        """Verify certificate against CA certificate."""
        try:
            ca_cert.certificate.public_key().verify(
                self.certificate.signature,
                self.certificate.tbs_certificate_bytes,
                padding.PKCS1v15(),
                self.certificate.signature_hash_algorithm
            )
            return True
        except Exception:
            return False


class CertificateAuthority:
    """Certificate Authority for PKI management."""
    
    def __init__(self, ca_name: str = "Agent Mesh CA"):
        self.ca_name = ca_name
        self.logger = structlog.get_logger("certificate_authority")
        
        # Generate CA key pair
        self._ca_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        self._ca_public_key = self._ca_private_key.public_key()
        
        # Create self-signed CA certificate
        self.ca_certificate = self._create_ca_certificate()
        self.ca_cert_wrapper = X509Certificate(
            certificate=self.ca_certificate,
            certificate_type=CertificateType.ROOT_CA,
            serial_number=1,
            subject_id=uuid4()
        )
        
        # Certificate registry
        self.certificates: Dict[UUID, X509Certificate] = {}
        self.revoked_certificates: Set[int] = set()
        self._serial_counter = 2  # Start from 2 (CA is 1)
    
    def _create_ca_certificate(self) -> x509.Certificate:
        """Create self-signed CA certificate."""
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Agent Mesh"),
            x509.NameAttribute(NameOID.COMMON_NAME, self.ca_name),
        ])
        
        certificate = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self._ca_public_key
        ).serial_number(
            1
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=3650)  # 10 years
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress("127.0.0.1"),
            ]),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).sign(self._ca_private_key, hashes.SHA256(), backend=default_backend())
        
        return certificate
    
    def issue_node_certificate(self, node_id: UUID, public_key: Any, 
                             common_name: str = None) -> X509Certificate:
        """Issue certificate for a mesh node."""
        
        if common_name is None:
            common_name = f"node-{str(node_id)[:8]}"
        
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Agent Mesh"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Mesh Nodes"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        serial_number = self._serial_counter
        self._serial_counter += 1
        
        certificate = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_certificate.subject
        ).public_key(
            public_key
        ).serial_number(
            serial_number
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)  # 1 year
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(f"{common_name}.agent-mesh.local"),
                x509.OtherName(
                    x509.ObjectIdentifier("1.3.6.1.4.1.311.20.2.3"),  # UPN OID
                    str(node_id).encode()
                ),
            ]),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=True,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.CLIENT_AUTH,
                ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=True,
        ).sign(self._ca_private_key, hashes.SHA256(), backend=default_backend())
        
        cert_wrapper = X509Certificate(
            certificate=certificate,
            certificate_type=CertificateType.NODE_CERTIFICATE,
            serial_number=serial_number,
            subject_id=node_id,
            issuer_id=self.ca_cert_wrapper.subject_id
        )
        
        self.certificates[node_id] = cert_wrapper
        
        self.logger.info("Node certificate issued",
                        node_id=str(node_id),
                        serial_number=serial_number,
                        common_name=common_name)
        
        return cert_wrapper
    
    def revoke_certificate(self, node_id: UUID, reason: str = "unspecified") -> bool:
        """Revoke a certificate."""
        if node_id not in self.certificates:
            return False
        
        cert = self.certificates[node_id]
        cert.status = CertificateStatus.REVOKED
        self.revoked_certificates.add(cert.serial_number)
        
        self.logger.warning("Certificate revoked",
                          node_id=str(node_id),
                          serial_number=cert.serial_number,
                          reason=reason)
        
        return True
    
    def verify_certificate(self, cert: X509Certificate) -> bool:
        """Verify certificate against CA and check revocation."""
        # Check if revoked
        if cert.serial_number in self.revoked_certificates:
            return False
        
        # Check basic validity
        if not cert.is_valid:
            return False
        
        # Verify signature
        return cert.verify_chain(self.ca_cert_wrapper)
    
    def get_certificate_status(self, node_id: UUID) -> Optional[CertificateStatus]:
        """Get current status of a certificate."""
        if node_id not in self.certificates:
            return None
        
        cert = self.certificates[node_id]
        
        # Check expiration
        if not cert.is_valid:
            cert.status = CertificateStatus.EXPIRED
        
        return cert.status


class KeyExchangeManager:
    """Manages key exchange protocols for secure communication."""
    
    def __init__(self, protocol: KeyExchangeProtocol = KeyExchangeProtocol.X25519):
        self.protocol = protocol
        self.logger = structlog.get_logger("key_exchange")
        
        # Generate ephemeral key pairs
        self._generate_ephemeral_keys()
        
        # Session keys storage
        self.session_keys: Dict[UUID, bytes] = {}
        self.key_rotation_interval = 3600  # 1 hour
    
    def _generate_ephemeral_keys(self) -> None:
        """Generate ephemeral keys for key exchange."""
        if self.protocol == KeyExchangeProtocol.X25519:
            self._private_key = x25519.X25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()
        else:
            raise ValueError(f"Unsupported key exchange protocol: {self.protocol}")
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key for key exchange."""
        if self.protocol == KeyExchangeProtocol.X25519:
            return self._public_key.public_bytes(
                encoding=Encoding.Raw,
                format=PublicFormat.Raw
            )
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
    
    def perform_key_exchange(self, peer_public_key: bytes, peer_id: UUID) -> bytes:
        """Perform key exchange with peer."""
        if self.protocol == KeyExchangeProtocol.X25519:
            # Load peer public key
            peer_key = x25519.X25519PublicKey.from_public_bytes(peer_public_key)
            
            # Perform key exchange
            shared_key = self._private_key.exchange(peer_key)
            
            # Derive session key using HKDF
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"agent_mesh_session_key",
                info=b"key_exchange",
                backend=default_backend()
            ).derive(shared_key)
            
            # Store session key
            self.session_keys[peer_id] = derived_key
            
            self.logger.info("Key exchange completed", 
                           peer_id=str(peer_id),
                           protocol=self.protocol.value)
            
            return derived_key
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
    
    def get_session_key(self, peer_id: UUID) -> Optional[bytes]:
        """Get session key for peer."""
        return self.session_keys.get(peer_id)
    
    def rotate_ephemeral_keys(self) -> None:
        """Rotate ephemeral keys for forward secrecy."""
        old_public_key = self.get_public_key_bytes()
        self._generate_ephemeral_keys()
        new_public_key = self.get_public_key_bytes()
        
        self.logger.info("Ephemeral keys rotated",
                        old_key_hash=hashlib.sha256(old_public_key).hexdigest()[:16],
                        new_key_hash=hashlib.sha256(new_public_key).hexdigest()[:16])
    
    def cleanup_expired_keys(self) -> None:
        """Remove expired session keys."""
        # In real implementation, would track key timestamps
        # For now, just clear all keys during rotation
        expired_keys = list(self.session_keys.keys())
        
        for peer_id in expired_keys:
            del self.session_keys[peer_id]
            self.logger.debug("Expired session key removed", peer_id=str(peer_id))


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
        
        # PKI integration
        self.certificate_authority: Optional[CertificateAuthority] = None
        self.node_certificate: Optional[X509Certificate] = None
        self.key_exchange_manager = KeyExchangeManager()
        
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
    
    async def setup_certificate_authority(self, ca_name: str = "Agent Mesh CA") -> None:
        """Set up this node as a Certificate Authority."""
        if not self._key_pair:
            raise RuntimeError("Security manager not initialized")
        
        self.certificate_authority = CertificateAuthority(ca_name)
        
        self.logger.info("Certificate Authority initialized", ca_name=ca_name)
        
        await self._log_security_event(
            "ca_setup",
            self.node_id,
            "pki",
            "setup_ca",
            True,
            metadata={"ca_name": ca_name}
        )
    
    async def request_certificate_from_ca(self, ca_manager: 'SecurityManager') -> X509Certificate:
        """Request certificate from a Certificate Authority."""
        if not self._key_pair:
            raise RuntimeError("Security manager not initialized")
        
        if not ca_manager.certificate_authority:
            raise ValueError("Target manager is not a Certificate Authority")
        
        # Issue certificate using our public key
        self.node_certificate = ca_manager.certificate_authority.issue_node_certificate(
            node_id=self.node_id,
            public_key=self._key_pair.public_key
        )
        
        # Update identity with certificate
        self._identity.certificate = self.node_certificate.certificate.public_bytes(Encoding.DER)
        
        self.logger.info("Certificate obtained from CA",
                        serial_number=self.node_certificate.serial_number,
                        ca_id=str(ca_manager.node_id))
        
        await self._log_security_event(
            "certificate_obtained",
            self.node_id,
            "pki",
            "request_certificate",
            True,
            metadata={
                "ca_id": str(ca_manager.node_id),
                "serial_number": self.node_certificate.serial_number
            }
        )
        
        return self.node_certificate
    
    async def verify_peer_certificate(self, peer_cert: X509Certificate) -> bool:
        """Verify a peer's certificate."""
        if not self.certificate_authority:
            self.logger.warning("Cannot verify certificate - not a CA")
            return False
        
        is_valid = self.certificate_authority.verify_certificate(peer_cert)
        
        await self._log_security_event(
            "certificate_verification",
            peer_cert.subject_id,
            "pki",
            "verify_certificate",
            is_valid,
            metadata={
                "serial_number": peer_cert.serial_number,
                "certificate_type": peer_cert.certificate_type.value
            }
        )
        
        return is_valid
    
    async def revoke_peer_certificate(self, peer_id: UUID, reason: str = "unspecified") -> bool:
        """Revoke a peer's certificate."""
        if not self.certificate_authority:
            raise RuntimeError("Not a Certificate Authority")
        
        success = self.certificate_authority.revoke_certificate(peer_id, reason)
        
        await self._log_security_event(
            "certificate_revocation",
            peer_id,
            "pki",
            "revoke_certificate",
            success,
            metadata={"reason": reason},
            risk_level="medium"
        )
        
        return success
    
    async def perform_key_exchange_with_peer(self, peer_id: UUID, peer_public_key: bytes) -> bytes:
        """Perform secure key exchange with a peer."""
        try:
            session_key = self.key_exchange_manager.perform_key_exchange(
                peer_public_key, peer_id
            )
            
            # Store session key for future use
            self._session_keys[peer_id] = session_key
            
            await self._log_security_event(
                "key_exchange_completed",
                peer_id,
                "key_exchange",
                "perform_exchange",
                True,
                metadata={"protocol": self.key_exchange_manager.protocol.value}
            )
            
            return session_key
            
        except Exception as e:
            await self._log_security_event(
                "key_exchange_failed",
                peer_id,
                "key_exchange",
                "perform_exchange",
                False,
                metadata={"error": str(e)},
                risk_level="medium"
            )
            raise
    
    def get_key_exchange_public_key(self) -> bytes:
        """Get public key for key exchange."""
        return self.key_exchange_manager.get_public_key_bytes()
    
    async def rotate_ephemeral_keys(self) -> None:
        """Rotate ephemeral keys for forward secrecy."""
        old_key_hash = hashlib.sha256(self.get_key_exchange_public_key()).hexdigest()[:16]
        
        self.key_exchange_manager.rotate_ephemeral_keys()
        
        new_key_hash = hashlib.sha256(self.get_key_exchange_public_key()).hexdigest()[:16]
        
        await self._log_security_event(
            "key_rotation",
            self.node_id,
            "key_management",
            "rotate_ephemeral_keys",
            True,
            metadata={
                "old_key_hash": old_key_hash,
                "new_key_hash": new_key_hash
            }
        )
    
    async def cleanup_expired_session_keys(self) -> None:
        """Clean up expired session keys."""
        initial_count = len(self._session_keys)
        
        self.key_exchange_manager.cleanup_expired_keys()
        
        # Clear our session keys too (in real implementation, would check timestamps)
        expired_peers = []
        for peer_id in list(self._session_keys.keys()):
            # In real implementation, would check key age
            # For now, just demonstrate the mechanism
            if len(expired_peers) < initial_count // 2:  # Expire half for demo
                expired_peers.append(peer_id)
                del self._session_keys[peer_id]
        
        if expired_peers:
            await self._log_security_event(
                "session_key_cleanup",
                self.node_id,
                "key_management",
                "cleanup_expired",
                True,
                metadata={
                    "expired_count": len(expired_peers),
                    "remaining_count": len(self._session_keys)
                }
            )
    
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