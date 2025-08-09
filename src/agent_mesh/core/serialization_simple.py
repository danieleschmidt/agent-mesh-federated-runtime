"""Simple message serialization without external crypto dependencies."""

import gzip
import json
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union
from uuid import UUID


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"


class MessageSerializer:
    """Simple message serializer for testing."""
    
    def __init__(self, compression: bool = False):
        self.compression = compression
        self.format = SerializationFormat.JSON
    
    def serialize(self, data: Any, metadata: Optional[Dict] = None) -> bytes:
        """Serialize data to bytes."""
        envelope = {
            "version": "1.0",
            "format": self.format.value,
            "timestamp": time.time(),
            "compressed": self.compression,
            "metadata": metadata or {},
            "data": self._prepare_for_serialization(data)
        }
        
        serialized = json.dumps(envelope, default=self._json_default).encode('utf-8')
        
        if self.compression:
            serialized = gzip.compress(serialized)
            
        return serialized
    
    def deserialize(self, data: bytes) -> tuple[Any, Dict]:
        """Deserialize bytes to data."""
        if self.compression or self._is_compressed(data):
            try:
                data = gzip.decompress(data)
            except Exception:
                pass
        
        envelope = json.loads(data.decode('utf-8'))
        deserialized_data = self._reconstruct_from_serialization(envelope["data"])
        metadata = envelope.get("metadata", {})
        
        return deserialized_data, metadata
    
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """Prepare object for serialization."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
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
        else:
            return {"__type__": "string", "value": str(obj)}
    
    def _reconstruct_from_serialization(self, obj: Any) -> Any:
        """Reconstruct object from serialized representation."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._reconstruct_from_serialization(item) for item in obj]
        elif isinstance(obj, dict):
            if "__type__" in obj:
                obj_type = obj["__type__"]
                if obj_type == "UUID":
                    return UUID(obj["value"])
                elif obj_type == "string":
                    return obj["value"]
                else:
                    return obj
            else:
                return {key: self._reconstruct_from_serialization(value) 
                       for key, value in obj.items()}
        else:
            return obj
    
    def _json_default(self, obj: Any) -> Any:
        """JSON serialization default handler."""
        if isinstance(obj, UUID):
            return str(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            return str(obj)
    
    def _is_compressed(self, data: bytes) -> bool:
        """Check if data is gzip compressed."""
        return data.startswith(b'\x1f\x8b')


def serialize_message(data: Any, metadata: Optional[Dict] = None) -> bytes:
    """Serialize message using simple serializer."""
    serializer = MessageSerializer()
    return serializer.serialize(data, metadata)


def deserialize_message(data: bytes) -> tuple[Any, Dict]:
    """Deserialize message using simple serializer."""
    serializer = MessageSerializer()
    return serializer.deserialize(data)