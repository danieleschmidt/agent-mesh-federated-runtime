"""Message serialization and deserialization for network communication.

Provides efficient, secure serialization of network messages with compression,
validation, and versioning support.
"""

import gzip
import json
import pickle
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union
from uuid import UUID

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from pydantic import BaseModel


logger = structlog.get_logger("serialization")


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"


class MessageSerializer:
    """
    High-performance message serializer with compression and encryption.
    
    Handles serialization of complex data structures including dataclasses,
    Pydantic models, and custom types with optional compression and encryption.
    """
    
    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        compression: bool = True,
        encryption_key: Optional[bytes] = None
    ):
        """
        Initialize message serializer.
        
        Args:
            format: Serialization format to use
            compression: Whether to compress messages
            encryption_key: Optional encryption key for secure transport
        """
        self.format = format
        self.compression = compression
        self.encryption_key = encryption_key
        
        # Initialize encryption if key provided
        self._fernet = None
        if encryption_key:
            self._fernet = Fernet(encryption_key)
        
        logger.info("Message serializer initialized", 
                   format=format.value, compression=compression)
    
    def serialize(self, data: Any, metadata: Optional[Dict] = None) -> bytes:
        """
        Serialize data to bytes.
        
        Args:
            data: Data to serialize
            metadata: Optional metadata to include
            
        Returns:
            Serialized bytes
        """
        # Prepare message envelope
        envelope = {
            "version": "1.0",
            "format": self.format.value,
            "timestamp": time.time(),
            "compressed": self.compression,
            "encrypted": self._fernet is not None,
            "metadata": metadata or {},
            "data": self._prepare_for_serialization(data)
        }
        
        # Serialize based on format
        if self.format == SerializationFormat.JSON:
            serialized = json.dumps(envelope, default=self._json_default).encode('utf-8')
        elif self.format == SerializationFormat.PICKLE:
            serialized = pickle.dumps(envelope)
        else:
            # Fallback to JSON
            serialized = json.dumps(envelope, default=self._json_default).encode('utf-8')
        
        # Compress if enabled
        if self.compression:
            serialized = gzip.compress(serialized)
        
        # Encrypt if enabled
        if self._fernet:
            serialized = self._fernet.encrypt(serialized)
        
        logger.debug("Message serialized", 
                    original_size=len(str(data)), 
                    final_size=len(serialized))
        
        return serialized
    
    def deserialize(self, data: bytes) -> tuple[Any, Dict]:
        """
        Deserialize bytes to data.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Tuple of (deserialized_data, metadata)
        """
        try:
            # Decrypt if needed
            if self._fernet:
                data = self._fernet.decrypt(data)
            
            # Decompress if needed
            if self.compression or self._is_compressed(data):
                try:
                    data = gzip.decompress(data)
                except Exception:
                    pass  # Data might not be compressed
            
            # Deserialize envelope
            if self.format == SerializationFormat.JSON:
                envelope = json.loads(data.decode('utf-8'))
            elif self.format == SerializationFormat.PICKLE:
                envelope = pickle.loads(data)
            else:
                # Fallback to JSON
                envelope = json.loads(data.decode('utf-8'))
            
            # Validate envelope
            if not isinstance(envelope, dict) or "data" not in envelope:
                raise ValueError("Invalid message envelope")
            
            # Extract data and metadata
            deserialized_data = self._reconstruct_from_serialization(envelope["data"])
            metadata = envelope.get("metadata", {})
            
            logger.debug("Message deserialized", 
                        format=envelope.get("format", "unknown"),
                        timestamp=envelope.get("timestamp", 0))
            
            return deserialized_data, metadata
            
        except Exception as e:
            logger.error("Deserialization failed", error=str(e))
            raise ValueError(f"Failed to deserialize message: {e}")
    
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """Prepare object for serialization by converting complex types."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_serialization(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._prepare_for_serialization(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, UUID):
            return {"__type__": "UUID", "value": str(obj)}
        elif isinstance(obj, Enum):
            return {"__type__": "Enum", "class": obj.__class__.__name__, "value": obj.value}
        elif is_dataclass(obj):
            return {"__type__": "dataclass", "class": obj.__class__.__name__, 
                   "data": self._prepare_for_serialization(asdict(obj))}
        elif isinstance(obj, BaseModel):
            return {"__type__": "pydantic", "class": obj.__class__.__name__, 
                   "data": self._prepare_for_serialization(obj.dict())}
        else:
            # Fallback to string representation
            logger.warning("Serializing complex object as string", type=type(obj).__name__)
            return {"__type__": "string", "value": str(obj)}
    
    def _reconstruct_from_serialization(self, obj: Any) -> Any:
        """Reconstruct object from serialized representation."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._reconstruct_from_serialization(item) for item in obj]
        elif isinstance(obj, dict):
            if "__type__" in obj:
                # Reconstruct special types
                obj_type = obj["__type__"]
                if obj_type == "UUID":
                    return UUID(obj["value"])
                elif obj_type == "Enum":
                    # Note: This is simplified - in real implementation you'd
                    # need to maintain a registry of enum classes
                    return obj["value"]
                elif obj_type == "dataclass":
                    # Note: This is simplified - in real implementation you'd
                    # need to maintain a registry of dataclass types
                    return self._reconstruct_from_serialization(obj["data"])
                elif obj_type == "pydantic":
                    # Note: This is simplified - in real implementation you'd
                    # need to maintain a registry of pydantic models
                    return self._reconstruct_from_serialization(obj["data"])
                elif obj_type == "string":
                    return obj["value"]
                else:
                    logger.warning("Unknown object type during deserialization", type=obj_type)
                    return obj
            else:
                # Regular dictionary
                return {key: self._reconstruct_from_serialization(value) 
                       for key, value in obj.items()}
        else:
            return obj
    
    def _json_default(self, obj: Any) -> Any:
        """JSON serialization default handler."""
        if isinstance(obj, UUID):
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _is_compressed(self, data: bytes) -> bool:
        """Check if data is gzip compressed."""
        return data.startswith(b'\x1f\x8b')


class SecureMessageSerializer(MessageSerializer):
    """
    Secure message serializer with digital signatures and key exchange.
    
    Extends the basic serializer with cryptographic security features
    including message signing and verification.
    """
    
    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        compression: bool = True,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        public_key: Optional[rsa.RSAPublicKey] = None
    ):
        """
        Initialize secure message serializer.
        
        Args:
            format: Serialization format
            compression: Whether to compress messages
            private_key: RSA private key for signing
            public_key: RSA public key for verification
        """
        super().__init__(format, compression)
        
        self.private_key = private_key
        self.public_key = public_key
        
        # Generate keys if not provided
        if not self.private_key:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
        
        logger.info("Secure message serializer initialized with cryptographic keys")
    
    def serialize_signed(self, data: Any, metadata: Optional[Dict] = None) -> bytes:
        """
        Serialize data with digital signature.
        
        Args:
            data: Data to serialize
            metadata: Optional metadata
            
        Returns:
            Signed serialized bytes
        """
        # First serialize normally
        serialized = self.serialize(data, metadata)
        
        # Create digital signature
        signature = self.private_key.sign(
            serialized,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Create signed envelope
        signed_envelope = {
            "data": serialized,
            "signature": signature,
            "public_key": self.public_key.public_key_pem()
        }
        
        return pickle.dumps(signed_envelope)
    
    def deserialize_verified(self, signed_data: bytes, trusted_public_key: Optional[rsa.RSAPublicKey] = None) -> tuple[Any, Dict]:
        """
        Deserialize and verify signed data.
        
        Args:
            signed_data: Signed serialized bytes
            trusted_public_key: Optional trusted public key
            
        Returns:
            Tuple of (deserialized_data, metadata)
        """
        # Extract signed envelope
        signed_envelope = pickle.loads(signed_data)
        
        serialized_data = signed_envelope["data"]
        signature = signed_envelope["signature"]
        
        # Get public key for verification
        verify_key = trusted_public_key or self.public_key
        
        if not verify_key:
            raise ValueError("No public key available for signature verification")
        
        # Verify signature
        try:
            verify_key.verify(
                signature,
                serialized_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except Exception as e:
            raise ValueError(f"Signature verification failed: {e}")
        
        # Deserialize verified data
        return self.deserialize(serialized_data)
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format."""
        return self.public_key.public_key_pem()
    
    def load_public_key_pem(self, pem_data: bytes) -> rsa.RSAPublicKey:
        """Load public key from PEM data."""
        return serialization.load_pem_public_key(pem_data)


# Global serializer instance
_default_serializer: Optional[MessageSerializer] = None


def get_default_serializer() -> MessageSerializer:
    """Get default message serializer instance."""
    global _default_serializer
    
    if _default_serializer is None:
        _default_serializer = MessageSerializer()
    
    return _default_serializer


def serialize_message(data: Any, metadata: Optional[Dict] = None) -> bytes:
    """Serialize message using default serializer."""
    return get_default_serializer().serialize(data, metadata)


def deserialize_message(data: bytes) -> tuple[Any, Dict]:
    """Deserialize message using default serializer."""
    return get_default_serializer().deserialize(data)