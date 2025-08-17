"""Protocol Optimization Module - Advanced Network Protocol Enhancements.

This module implements advanced protocol optimizations for the Agent Mesh system,
including custom binary protocols, message batching, compression, and network
efficiency improvements for distributed consensus and federated learning.

Features:
- Custom binary message protocols for reduced overhead
- Adaptive message batching and compression
- Network flow control and congestion management
- Protocol-level security optimizations
- Bandwidth usage optimization

Authors: Terragon Labs Research Team
"""

import asyncio
import time
import hashlib
import json
import zlib
import struct
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the optimized protocol."""
    CONSENSUS_PROPOSAL = 0x01
    CONSENSUS_VOTE = 0x02
    CONSENSUS_COMMIT = 0x03
    FEDERATED_UPDATE = 0x04
    FEDERATED_AGGREGATE = 0x05
    HEARTBEAT = 0x06
    TOPOLOGY_UPDATE = 0x07
    SECURITY_CHALLENGE = 0x08
    BATCH_MESSAGE = 0x09
    COMPRESSED_MESSAGE = 0x0A


class CompressionType(Enum):
    """Compression algorithms supported."""
    NONE = 0x00
    ZLIB = 0x01
    GZIP = 0x02
    LZ4 = 0x03
    SNAPPY = 0x04


@dataclass
class MessageHeader:
    """Optimized binary message header."""
    version: int = 1
    message_type: MessageType = MessageType.HEARTBEAT
    compression: CompressionType = CompressionType.NONE
    payload_length: int = 0
    sequence_number: int = 0
    sender_id: str = ""
    timestamp: float = field(default_factory=time.time)
    checksum: int = 0


@dataclass
class NetworkMetrics:
    """Network performance metrics tracking."""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    compression_ratio: float = 1.0
    latency_samples: List[float] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)
    error_count: int = 0
    
    # Protocol-specific metrics
    batch_efficiency: float = 1.0
    protocol_overhead: float = 0.1
    congestion_level: float = 0.0


class BinaryProtocolEncoder:
    """Efficient binary protocol encoder for Agent Mesh messages."""
    
    HEADER_SIZE = 32  # Fixed header size in bytes
    MAX_PAYLOAD_SIZE = 16 * 1024 * 1024  # 16MB max payload
    
    def __init__(self):
        self.sequence_counter = 0
        
    def encode_message(self, message_type: MessageType, payload: Dict[str, Any], 
                      sender_id: str, compression: CompressionType = CompressionType.ZLIB) -> bytes:
        """Encode message to optimized binary format."""
        # Serialize payload to JSON then bytes
        payload_json = json.dumps(payload, separators=(',', ':'))
        payload_bytes = payload_json.encode('utf-8')
        
        # Apply compression if specified
        compressed_payload = self._compress_payload(payload_bytes, compression)
        
        # Create header
        header = MessageHeader(
            message_type=message_type,
            compression=compression,
            payload_length=len(compressed_payload),
            sequence_number=self.sequence_counter,
            sender_id=sender_id,
            timestamp=time.time()
        )
        
        # Calculate checksum
        header.checksum = self._calculate_checksum(compressed_payload)
        
        # Encode header to binary
        header_bytes = self._encode_header(header)
        
        # Increment sequence counter
        self.sequence_counter += 1
        
        return header_bytes + compressed_payload
    
    def _compress_payload(self, payload: bytes, compression: CompressionType) -> bytes:
        """Compress payload using specified algorithm."""
        if compression == CompressionType.NONE:
            return payload
        elif compression == CompressionType.ZLIB:
            return zlib.compress(payload, level=6)
        elif compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(payload, compresslevel=6)
        else:
            # Fallback to zlib for unsupported compression
            return zlib.compress(payload, level=6)
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate CRC32 checksum for data integrity."""
        return zlib.crc32(data) & 0xffffffff
    
    def _encode_header(self, header: MessageHeader) -> bytes:
        """Encode message header to binary format."""
        # Pack header fields into binary format
        sender_id_bytes = header.sender_id.encode('utf-8')[:16]  # Limit sender ID
        sender_id_padded = sender_id_bytes.ljust(16, b'\x00')
        
        # Pack: version(1), type(1), compression(1), reserved(1), length(4), seq(4), 
        #       timestamp(8), checksum(4), sender_id(16)
        header_data = struct.pack(
            '!BBBBIQIB16s',
            header.version,
            header.message_type.value,
            header.compression.value,
            0,  # Reserved byte
            header.payload_length,
            header.sequence_number,
            int(header.timestamp * 1000000),  # Microsecond timestamp
            header.checksum,
            sender_id_padded
        )
        
        return header_data


class BinaryProtocolDecoder:
    """Efficient binary protocol decoder for Agent Mesh messages."""
    
    def __init__(self):
        self.received_messages: Dict[int, MessageHeader] = {}
        
    def decode_message(self, data: bytes) -> Tuple[MessageHeader, Dict[str, Any]]:
        """Decode binary message to header and payload."""
        if len(data) < BinaryProtocolEncoder.HEADER_SIZE:
            raise ValueError(f"Message too short: {len(data)} < {BinaryProtocolEncoder.HEADER_SIZE}")
        
        # Extract and decode header
        header_bytes = data[:BinaryProtocolEncoder.HEADER_SIZE]
        payload_bytes = data[BinaryProtocolEncoder.HEADER_SIZE:]
        
        header = self._decode_header(header_bytes)
        
        # Verify payload length
        if len(payload_bytes) != header.payload_length:
            raise ValueError(f"Payload length mismatch: {len(payload_bytes)} != {header.payload_length}")
        
        # Verify checksum
        calculated_checksum = zlib.crc32(payload_bytes) & 0xffffffff
        if calculated_checksum != header.checksum:
            raise ValueError(f"Checksum mismatch: {calculated_checksum} != {header.checksum}")
        
        # Decompress payload
        decompressed_payload = self._decompress_payload(payload_bytes, header.compression)
        
        # Parse JSON payload
        payload_json = decompressed_payload.decode('utf-8')
        payload = json.loads(payload_json)
        
        # Track received message
        self.received_messages[header.sequence_number] = header
        
        return header, payload
    
    def _decode_header(self, header_bytes: bytes) -> MessageHeader:
        """Decode binary header to MessageHeader object."""
        # Unpack header fields
        unpacked = struct.unpack('!BBBBIQIB16s', header_bytes)
        
        version = unpacked[0]
        message_type = MessageType(unpacked[1])
        compression = CompressionType(unpacked[2])
        # reserved = unpacked[3]
        payload_length = unpacked[4]
        sequence_number = unpacked[5]
        timestamp_us = unpacked[6]
        checksum = unpacked[7]
        sender_id_bytes = unpacked[8]
        
        # Convert timestamp and sender ID
        timestamp = timestamp_us / 1000000.0
        sender_id = sender_id_bytes.rstrip(b'\x00').decode('utf-8')
        
        return MessageHeader(
            version=version,
            message_type=message_type,
            compression=compression,
            payload_length=payload_length,
            sequence_number=sequence_number,
            sender_id=sender_id,
            timestamp=timestamp,
            checksum=checksum
        )
    
    def _decompress_payload(self, payload: bytes, compression: CompressionType) -> bytes:
        """Decompress payload using specified algorithm."""
        if compression == CompressionType.NONE:
            return payload
        elif compression == CompressionType.ZLIB:
            return zlib.decompress(payload)
        elif compression == CompressionType.GZIP:
            import gzip
            return gzip.decompress(payload)
        else:
            # Fallback to zlib
            return zlib.decompress(payload)


class MessageBatcher:
    """Intelligent message batching for improved network efficiency."""
    
    def __init__(self, max_batch_size: int = 10, max_batch_delay: float = 0.01):
        self.max_batch_size = max_batch_size
        self.max_batch_delay = max_batch_delay
        
        self.pending_messages: List[Tuple[MessageType, Dict[str, Any], str]] = []
        self.batch_timer: Optional[asyncio.Task] = None
        self.batch_callbacks: List[callable] = []
        
        # Batching statistics
        self.batch_stats = {
            'batches_sent': 0,
            'messages_batched': 0,
            'average_batch_size': 0.0,
            'compression_savings': 0.0
        }
    
    async def add_message(self, message_type: MessageType, payload: Dict[str, Any], 
                         sender_id: str) -> None:
        """Add message to batch queue."""
        self.pending_messages.append((message_type, payload, sender_id))
        
        # Check if batch is full
        if len(self.pending_messages) >= self.max_batch_size:
            await self._flush_batch()
        elif self.batch_timer is None:
            # Start batch timer for first message
            self.batch_timer = asyncio.create_task(self._batch_timer_expired())
    
    async def _batch_timer_expired(self) -> None:
        """Handle batch timer expiration."""
        await asyncio.sleep(self.max_batch_delay)
        if self.pending_messages:
            await self._flush_batch()
    
    async def _flush_batch(self) -> None:
        """Flush pending messages as a batch."""
        if not self.pending_messages:
            return
        
        # Cancel timer
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Create batch message
        batch_payload = {
            'messages': [
                {
                    'type': msg_type.value,
                    'payload': payload,
                    'sender': sender_id
                }
                for msg_type, payload, sender_id in self.pending_messages
            ],
            'batch_size': len(self.pending_messages),
            'batch_timestamp': time.time()
        }
        
        # Update statistics
        self.batch_stats['batches_sent'] += 1
        self.batch_stats['messages_batched'] += len(self.pending_messages)
        self.batch_stats['average_batch_size'] = (
            self.batch_stats['messages_batched'] / self.batch_stats['batches_sent']
        )
        
        # Clear pending messages
        batch_size = len(self.pending_messages)
        self.pending_messages.clear()
        
        # Notify callbacks
        for callback in self.batch_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(MessageType.BATCH_MESSAGE, batch_payload, "batch_sender")
                else:
                    callback(MessageType.BATCH_MESSAGE, batch_payload, "batch_sender")
            except Exception as e:
                logger.error(f"Batch callback failed: {e}")
        
        logger.debug(f"Flushed batch of {batch_size} messages")
    
    def add_batch_callback(self, callback: callable) -> None:
        """Add callback for when batches are sent."""
        self.batch_callbacks.append(callback)
    
    async def force_flush(self) -> None:
        """Force flush all pending messages."""
        await self._flush_batch()
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return self.batch_stats.copy()


class FlowController:
    """Network flow control and congestion management."""
    
    def __init__(self, max_window_size: int = 1000, initial_window: int = 100):
        self.max_window_size = max_window_size
        self.window_size = initial_window
        self.outstanding_messages = 0
        
        # Congestion control
        self.congestion_threshold = max_window_size // 2
        self.rtt_samples: deque = deque(maxlen=100)
        self.retransmit_count = 0
        
        # Flow control state
        self.flow_control_enabled = True
        self.last_ack_time = time.time()
        
    async def send_message(self, message_data: bytes) -> bool:
        """Check if message can be sent based on flow control."""
        if not self.flow_control_enabled:
            return True
        
        if self.outstanding_messages >= self.window_size:
            # Window is full, apply backpressure
            await self._apply_backpressure()
            return False
        
        self.outstanding_messages += 1
        return True
    
    def receive_ack(self, sequence_number: int, rtt: float) -> None:
        """Process acknowledgment and update flow control."""
        if self.outstanding_messages > 0:
            self.outstanding_messages -= 1
        
        # Update RTT samples
        self.rtt_samples.append(rtt)
        self.last_ack_time = time.time()
        
        # Adaptive window sizing
        self._update_window_size(rtt)
    
    def _update_window_size(self, rtt: float) -> None:
        """Update window size based on network conditions."""
        avg_rtt = sum(self.rtt_samples) / len(self.rtt_samples) if self.rtt_samples else rtt
        
        # Slow start vs congestion avoidance
        if self.window_size < self.congestion_threshold:
            # Slow start: exponential growth
            self.window_size = min(self.window_size * 2, self.max_window_size)
        else:
            # Congestion avoidance: linear growth
            self.window_size = min(self.window_size + 1, self.max_window_size)
        
        # RTT-based adjustment
        if avg_rtt > 0.1:  # High latency detected
            self.window_size = max(self.window_size // 2, 10)
            self.congestion_threshold = self.window_size
    
    async def _apply_backpressure(self) -> None:
        """Apply backpressure when window is full."""
        # Calculate backoff delay based on RTT
        avg_rtt = sum(self.rtt_samples) / len(self.rtt_samples) if self.rtt_samples else 0.01
        backoff_delay = min(avg_rtt * 2, 0.1)  # Max 100ms backoff
        
        await asyncio.sleep(backoff_delay)
    
    def detect_timeout(self, timeout_threshold: float = 5.0) -> bool:
        """Detect if ACKs are timing out."""
        return time.time() - self.last_ack_time > timeout_threshold
    
    def handle_retransmit(self) -> None:
        """Handle retransmission event."""
        self.retransmit_count += 1
        
        # Reduce window size on retransmit
        self.congestion_threshold = max(self.window_size // 2, 10)
        self.window_size = max(self.window_size // 2, 10)
    
    def get_flow_stats(self) -> Dict[str, Any]:
        """Get flow control statistics."""
        avg_rtt = sum(self.rtt_samples) / len(self.rtt_samples) if self.rtt_samples else 0.0
        
        return {
            'window_size': self.window_size,
            'outstanding_messages': self.outstanding_messages,
            'congestion_threshold': self.congestion_threshold,
            'average_rtt': avg_rtt,
            'retransmit_count': self.retransmit_count,
            'window_utilization': self.outstanding_messages / self.window_size if self.window_size > 0 else 0
        }


class ProtocolOptimizer:
    """Comprehensive protocol optimization system."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Protocol components
        self.encoder = BinaryProtocolEncoder()
        self.decoder = BinaryProtocolDecoder()
        self.batcher = MessageBatcher()
        self.flow_controller = FlowController()
        
        # Network metrics
        self.metrics = NetworkMetrics()
        self.connection_metrics: Dict[str, NetworkMetrics] = defaultdict(NetworkMetrics)
        
        # Protocol configuration
        self.compression_enabled = True
        self.batching_enabled = True
        self.flow_control_enabled = True
        
        # Message handlers
        self.message_handlers: Dict[MessageType, callable] = {}
        
        # Setup batch callback
        self.batcher.add_batch_callback(self._handle_batch_ready)
    
    async def send_message(self, message_type: MessageType, payload: Dict[str, Any], 
                          target_node: str) -> bool:
        """Send optimized message to target node."""
        try:
            # Apply batching if enabled
            if self.batching_enabled and message_type != MessageType.BATCH_MESSAGE:
                await self.batcher.add_message(message_type, payload, self.node_id)
                return True
            
            # Encode message
            compression = CompressionType.ZLIB if self.compression_enabled else CompressionType.NONE
            encoded_message = self.encoder.encode_message(message_type, payload, self.node_id, compression)
            
            # Apply flow control
            if self.flow_control_enabled:
                can_send = await self.flow_controller.send_message(encoded_message)
                if not can_send:
                    return False
            
            # Simulate message transmission
            await self._transmit_message(encoded_message, target_node)
            
            # Update metrics
            self.metrics.bytes_sent += len(encoded_message)
            self.metrics.messages_sent += 1
            self.connection_metrics[target_node].bytes_sent += len(encoded_message)
            self.connection_metrics[target_node].messages_sent += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics.error_count += 1
            return False
    
    async def receive_message(self, data: bytes, source_node: str) -> Optional[Tuple[MessageHeader, Dict[str, Any]]]:
        """Receive and decode optimized message."""
        try:
            # Decode message
            header, payload = self.decoder.decode_message(data)
            
            # Update metrics
            self.metrics.bytes_received += len(data)
            self.metrics.messages_received += 1
            self.connection_metrics[source_node].bytes_received += len(data)
            self.connection_metrics[source_node].messages_received += 1
            
            # Calculate compression ratio
            if header.compression != CompressionType.NONE:
                # Estimate original size
                original_size = len(json.dumps(payload).encode('utf-8'))
                compressed_size = header.payload_length
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                self.metrics.compression_ratio = compression_ratio
            
            # Handle acknowledgment for flow control
            if self.flow_control_enabled:
                rtt = time.time() - header.timestamp
                self.flow_controller.receive_ack(header.sequence_number, rtt)
            
            # Process message
            await self._process_message(header, payload, source_node)
            
            return header, payload
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            self.metrics.error_count += 1
            return None
    
    async def _transmit_message(self, data: bytes, target_node: str) -> None:
        """Simulate message transmission with realistic delays."""
        # Simulate network latency (1-50ms)
        base_latency = 0.001 + random.uniform(0, 0.049)
        
        # Add congestion-based delay
        congestion_delay = self.flow_controller.outstanding_messages * 0.0001
        
        total_delay = base_latency + congestion_delay
        await asyncio.sleep(total_delay)
        
        # Track latency
        self.metrics.latency_samples.append(total_delay)
        if len(self.metrics.latency_samples) > 1000:
            self.metrics.latency_samples.pop(0)
    
    async def _process_message(self, header: MessageHeader, payload: Dict[str, Any], 
                             source_node: str) -> None:
        """Process received message based on type."""
        message_type = header.message_type
        
        # Handle batch messages
        if message_type == MessageType.BATCH_MESSAGE:
            await self._process_batch_message(payload, source_node)
            return
        
        # Call registered handler
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(header, payload, source_node)
                else:
                    handler(header, payload, source_node)
            except Exception as e:
                logger.error(f"Message handler failed: {e}")
    
    async def _process_batch_message(self, batch_payload: Dict[str, Any], 
                                   source_node: str) -> None:
        """Process batched messages."""
        messages = batch_payload.get('messages', [])
        
        for message_data in messages:
            try:
                msg_type = MessageType(message_data['type'])
                payload = message_data['payload']
                sender = message_data['sender']
                
                # Create synthetic header for batch message
                header = MessageHeader(
                    message_type=msg_type,
                    sender_id=sender,
                    timestamp=batch_payload.get('batch_timestamp', time.time())
                )
                
                await self._process_message(header, payload, source_node)
                
            except Exception as e:
                logger.error(f"Failed to process batched message: {e}")
    
    async def _handle_batch_ready(self, message_type: MessageType, payload: Dict[str, Any], 
                                sender_id: str) -> None:
        """Handle when a batch is ready to send."""
        # Encode and send batch message
        compression = CompressionType.ZLIB if self.compression_enabled else CompressionType.NONE
        encoded_message = self.encoder.encode_message(message_type, payload, sender_id, compression)
        
        # Send to all connected nodes (simplified)
        target_nodes = ["node_1", "node_2", "node_3"]  # Would be actual connected nodes
        
        for target_node in target_nodes:
            try:
                await self._transmit_message(encoded_message, target_node)
                
                # Update metrics
                self.metrics.bytes_sent += len(encoded_message)
                self.connection_metrics[target_node].bytes_sent += len(encoded_message)
                
            except Exception as e:
                logger.error(f"Failed to send batch to {target_node}: {e}")
    
    def register_message_handler(self, message_type: MessageType, handler: callable) -> None:
        """Register handler for specific message type."""
        self.message_handlers[message_type] = handler
    
    async def optimize_protocol_parameters(self) -> None:
        """Dynamically optimize protocol parameters based on network conditions."""
        # Analyze current metrics
        avg_latency = sum(self.metrics.latency_samples) / len(self.metrics.latency_samples) if self.metrics.latency_samples else 0.01
        
        # Adjust batching parameters
        if avg_latency > 0.05:  # High latency network
            self.batcher.max_batch_delay = 0.02  # Increase batch delay
            self.batcher.max_batch_size = 20     # Increase batch size
        else:  # Low latency network
            self.batcher.max_batch_delay = 0.005  # Decrease batch delay
            self.batcher.max_batch_size = 5       # Decrease batch size
        
        # Adjust compression based on bandwidth utilization
        bandwidth_utilization = self.get_bandwidth_utilization()
        if bandwidth_utilization > 0.8:
            self.compression_enabled = True  # Enable compression on high utilization
        elif bandwidth_utilization < 0.3:
            self.compression_enabled = False  # Disable compression on low utilization
        
        logger.debug(f"Protocol parameters optimized: latency={avg_latency:.4f}s, bandwidth={bandwidth_utilization:.2%}")
    
    def get_bandwidth_utilization(self) -> float:
        """Calculate current bandwidth utilization."""
        # Simplified bandwidth calculation
        total_bytes = self.metrics.bytes_sent + self.metrics.bytes_received
        time_window = 60.0  # 1 minute window
        estimated_bandwidth = 10 * 1024 * 1024  # 10 Mbps estimated capacity
        
        utilization = (total_bytes * 8) / (estimated_bandwidth * time_window)
        return min(utilization, 1.0)
    
    def get_protocol_statistics(self) -> Dict[str, Any]:
        """Get comprehensive protocol statistics."""
        avg_latency = sum(self.metrics.latency_samples) / len(self.metrics.latency_samples) if self.metrics.latency_samples else 0.0
        
        # Calculate throughput
        total_messages = self.metrics.messages_sent + self.metrics.messages_received
        uptime = 3600.0  # Simplified: assume 1 hour uptime
        throughput = total_messages / uptime
        
        return {
            'node_id': self.node_id,
            'protocol_efficiency': {
                'compression_ratio': self.metrics.compression_ratio,
                'average_latency': avg_latency,
                'throughput_msg_per_sec': throughput,
                'bandwidth_utilization': self.get_bandwidth_utilization(),
                'error_rate': self.metrics.error_count / max(total_messages, 1)
            },
            'traffic_stats': {
                'bytes_sent': self.metrics.bytes_sent,
                'bytes_received': self.metrics.bytes_received,
                'messages_sent': self.metrics.messages_sent,
                'messages_received': self.metrics.messages_received,
                'total_errors': self.metrics.error_count
            },
            'optimization_features': {
                'compression_enabled': self.compression_enabled,
                'batching_enabled': self.batching_enabled,
                'flow_control_enabled': self.flow_control_enabled
            },
            'batching_stats': self.batcher.get_batch_stats(),
            'flow_control_stats': self.flow_controller.get_flow_stats(),
            'per_connection_stats': {
                node_id: {
                    'bytes_sent': metrics.bytes_sent,
                    'bytes_received': metrics.bytes_received,
                    'messages_sent': metrics.messages_sent,
                    'messages_received': metrics.messages_received
                }
                for node_id, metrics in self.connection_metrics.items()
            }
        }


async def main():
    """Demonstrate protocol optimization capabilities."""
    print("🚀 Protocol Optimization Module - Performance Demo")
    print("=" * 60)
    
    # Initialize protocol optimizer
    optimizer = ProtocolOptimizer("demo_node")
    
    # Register sample message handlers
    async def consensus_handler(header: MessageHeader, payload: Dict[str, Any], source: str):
        print(f"   Consensus message from {source}: {payload.get('proposal', 'unknown')}")
    
    async def federated_handler(header: MessageHeader, payload: Dict[str, Any], source: str):
        print(f"   Federated update from {source}: {len(payload.get('gradients', []))} gradients")
    
    optimizer.register_message_handler(MessageType.CONSENSUS_PROPOSAL, consensus_handler)
    optimizer.register_message_handler(MessageType.FEDERATED_UPDATE, federated_handler)
    
    # Test message sending and receiving
    print(f"\n📤 Testing Message Protocol:")
    
    # Test consensus messages
    for i in range(5):
        await optimizer.send_message(
            MessageType.CONSENSUS_PROPOSAL,
            {'proposal': f'proposal_{i}', 'value': i * 10, 'timestamp': time.time()},
            f"target_node_{i % 3}"
        )
    
    # Test federated learning messages
    for i in range(3):
        await optimizer.send_message(
            MessageType.FEDERATED_UPDATE,
            {'gradients': [random.uniform(-1, 1) for _ in range(100)], 'participant_id': f"client_{i}"},
            f"aggregator_node"
        )
    
    # Flush any pending batches
    await optimizer.batcher.force_flush()
    
    # Simulate receiving messages
    print(f"\n📥 Simulating Message Reception:")
    
    # Create sample encoded messages
    sample_messages = [
        (MessageType.CONSENSUS_VOTE, {'vote': 'approve', 'proposal_id': 'prop_123'}, 'voter_node'),
        (MessageType.HEARTBEAT, {'status': 'alive', 'load': 0.3}, 'monitor_node'),
        (MessageType.TOPOLOGY_UPDATE, {'new_connections': ['node_a', 'node_b']}, 'topology_node')
    ]
    
    for msg_type, payload, sender in sample_messages:
        # Encode message
        encoded = optimizer.encoder.encode_message(msg_type, payload, sender)
        
        # Simulate receiving it
        result = await optimizer.receive_message(encoded, sender)
        if result:
            header, received_payload = result
            print(f"   Received {msg_type.name} from {sender}")
    
    # Run protocol optimization
    print(f"\n⚙️  Running Protocol Optimization:")
    await optimizer.optimize_protocol_parameters()
    
    # Get statistics
    print(f"\n📊 Protocol Statistics:")
    stats = optimizer.get_protocol_statistics()
    
    efficiency = stats['protocol_efficiency']
    print(f"   Compression ratio: {efficiency['compression_ratio']:.2f}")
    print(f"   Average latency: {efficiency['average_latency']:.4f}s")
    print(f"   Throughput: {efficiency['throughput_msg_per_sec']:.1f} msg/s")
    print(f"   Bandwidth utilization: {efficiency['bandwidth_utilization']:.1%}")
    print(f"   Error rate: {efficiency['error_rate']:.2%}")
    
    traffic = stats['traffic_stats']
    print(f"   Total bytes sent: {traffic['bytes_sent']:,}")
    print(f"   Total messages sent: {traffic['messages_sent']}")
    
    features = stats['optimization_features']
    print(f"   Compression: {'Enabled' if features['compression_enabled'] else 'Disabled'}")
    print(f"   Batching: {'Enabled' if features['batching_enabled'] else 'Disabled'}")
    print(f"   Flow control: {'Enabled' if features['flow_control_enabled'] else 'Disabled'}")
    
    # Batching statistics
    batch_stats = stats['batching_stats']
    print(f"   Batches sent: {batch_stats['batches_sent']}")
    print(f"   Average batch size: {batch_stats['average_batch_size']:.1f}")
    
    # Flow control statistics
    flow_stats = stats['flow_control_stats']
    print(f"   Window size: {flow_stats['window_size']}")
    print(f"   Window utilization: {flow_stats['window_utilization']:.1%}")
    
    print(f"\n🎉 Protocol optimization demonstration completed!")
    print(f"💡 Protocol optimized for low-latency, high-throughput communication")


if __name__ == "__main__":
    asyncio.run(main())