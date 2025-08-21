"""Intelligent routing system with ML-based path optimization."""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import math

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Routing strategy types."""
    SHORTEST_PATH = "shortest_path"
    LOWEST_LATENCY = "lowest_latency"
    HIGHEST_BANDWIDTH = "highest_bandwidth"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"
    ML_OPTIMIZED = "ml_optimized"

class NodeStatus(Enum):
    """Node status for routing decisions."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    UNREACHABLE = "unreachable"

@dataclass
class NetworkNode:
    """Network node representation."""
    node_id: str
    address: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    last_seen: float = field(default_factory=time.time)
    latency: float = 0.0  # milliseconds
    bandwidth: float = 0.0  # Mbps
    load: float = 0.0  # 0.0 to 1.0
    reliability: float = 1.0  # 0.0 to 1.0
    
@dataclass
class NetworkLink:
    """Network link between nodes."""
    source_id: str
    target_id: str
    latency: float = 0.0
    bandwidth: float = 0.0
    packet_loss: float = 0.0
    cost: float = 1.0
    last_updated: float = field(default_factory=time.time)
    
@dataclass
class RoutingPath:
    """Routing path representation."""
    nodes: List[str]
    total_latency: float
    total_bandwidth: float
    reliability_score: float
    cost_score: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class RoutePerformance:
    """Route performance metrics."""
    path_hash: str
    success_rate: float
    avg_latency: float
    avg_bandwidth: float
    packet_loss: float
    last_used: float
    usage_count: int = 0

class IntelligentRouter:
    """ML-enhanced intelligent network router."""
    
    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        learning_rate: float = 0.1,
        topology_update_interval: float = 30.0
    ):
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.topology_update_interval = topology_update_interval
        
        # Network topology
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: Dict[Tuple[str, str], NetworkLink] = {}
        self.adjacency: Dict[str, Set[str]] = {}
        
        # Routing intelligence
        self.route_cache: Dict[str, RoutingPath] = {}
        self.route_performance: Dict[str, RoutePerformance] = {}
        self.routing_history: List[Dict[str, Any]] = []
        
        # ML components
        self.feature_weights: Dict[str, float] = {
            "latency": -1.0,  # Lower is better
            "bandwidth": 1.0,  # Higher is better
            "reliability": 1.0,  # Higher is better
            "load": -0.5,  # Lower is better
            "packet_loss": -1.0  # Lower is better
        }
        
        # State management
        self.is_running = False
        self._topology_task: Optional[asyncio.Task] = None
        
    def add_node(
        self,
        node_id: str,
        address: str,
        port: int,
        status: NodeStatus = NodeStatus.HEALTHY
    ):
        """Add a node to the network topology."""
        node = NetworkNode(
            node_id=node_id,
            address=address,
            port=port,
            status=status
        )
        self.nodes[node_id] = node
        
        if node_id not in self.adjacency:
            self.adjacency[node_id] = set()
            
        logger.info(f"Added node to topology: {node_id} ({address}:{port})")
    
    def add_link(
        self,
        source_id: str,
        target_id: str,
        latency: float = 0.0,
        bandwidth: float = 0.0,
        bidirectional: bool = True
    ):
        """Add a link between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add link: unknown nodes {source_id} or {target_id}")
            return
            
        link = NetworkLink(
            source_id=source_id,
            target_id=target_id,
            latency=latency,
            bandwidth=bandwidth
        )
        
        self.links[(source_id, target_id)] = link
        self.adjacency[source_id].add(target_id)
        
        if bidirectional:
            reverse_link = NetworkLink(
                source_id=target_id,
                target_id=source_id,
                latency=latency,
                bandwidth=bandwidth
            )
            self.links[(target_id, source_id)] = reverse_link
            self.adjacency[target_id].add(source_id)
            
        logger.info(f"Added link: {source_id} → {target_id} (latency: {latency}ms)")
    
    def update_node_metrics(
        self,
        node_id: str,
        latency: Optional[float] = None,
        bandwidth: Optional[float] = None,
        load: Optional[float] = None,
        status: Optional[NodeStatus] = None
    ):
        """Update node performance metrics."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        if latency is not None:
            node.latency = latency
        if bandwidth is not None:
            node.bandwidth = bandwidth
        if load is not None:
            node.load = max(0.0, min(1.0, load))
        if status is not None:
            node.status = status
            
        node.last_seen = time.time()
        
        # Update reliability based on status
        if status == NodeStatus.HEALTHY:
            node.reliability = min(1.0, node.reliability + 0.01)
        elif status == NodeStatus.DEGRADED:
            node.reliability = max(0.5, node.reliability - 0.05)
        elif status == NodeStatus.OVERLOADED:
            node.reliability = max(0.3, node.reliability - 0.02)
        elif status == NodeStatus.UNREACHABLE:
            node.reliability = max(0.0, node.reliability - 0.1)
    
    def update_link_metrics(
        self,
        source_id: str,
        target_id: str,
        latency: Optional[float] = None,
        bandwidth: Optional[float] = None,
        packet_loss: Optional[float] = None
    ):
        """Update link performance metrics."""
        link_key = (source_id, target_id)
        if link_key not in self.links:
            return
            
        link = self.links[link_key]
        
        if latency is not None:
            link.latency = latency
        if bandwidth is not None:
            link.bandwidth = bandwidth
        if packet_loss is not None:
            link.packet_loss = max(0.0, min(1.0, packet_loss))
            
        link.last_updated = time.time()
        
        # Update cost based on metrics
        link.cost = self._calculate_link_cost(link)
    
    def _calculate_link_cost(self, link: NetworkLink) -> float:
        """Calculate dynamic link cost based on metrics."""
        # Base cost (normalized latency)
        latency_cost = link.latency / 100.0  # Normalize to ~1.0 for 100ms
        
        # Bandwidth cost (inverse of bandwidth)
        bandwidth_cost = 1.0 / max(link.bandwidth, 1.0)
        
        # Packet loss penalty
        packet_loss_cost = link.packet_loss * 10.0
        
        # Combine costs with weights
        total_cost = (
            latency_cost * 0.4 +
            bandwidth_cost * 0.3 +
            packet_loss_cost * 0.3
        )
        
        return max(0.1, total_cost)  # Minimum cost to avoid zero
    
    def _calculate_path_score(self, path: List[str]) -> float:
        """Calculate ML-based path score."""
        if len(path) < 2:
            return 0.0
            
        total_score = 0.0
        total_latency = 0.0
        min_bandwidth = float('inf')
        avg_reliability = 0.0
        avg_load = 0.0
        total_packet_loss = 0.0
        
        # Analyze path segments
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # Node metrics
            if source_id in self.nodes:
                node = self.nodes[source_id]
                avg_reliability += node.reliability
                avg_load += node.load
                
            # Link metrics
            link_key = (source_id, target_id)
            if link_key in self.links:
                link = self.links[link_key]
                total_latency += link.latency
                min_bandwidth = min(min_bandwidth, link.bandwidth)
                total_packet_loss += link.packet_loss
        
        # Normalize metrics
        path_length = len(path) - 1
        avg_reliability /= path_length
        avg_load /= path_length
        
        if min_bandwidth == float('inf'):
            min_bandwidth = 0.0
            
        # Calculate weighted score using learned weights
        features = {
            "latency": total_latency,
            "bandwidth": min_bandwidth,
            "reliability": avg_reliability,
            "load": avg_load,
            "packet_loss": total_packet_loss
        }
        
        score = 0.0
        for feature, value in features.items():
            weight = self.feature_weights.get(feature, 0.0)
            
            # Normalize values to 0-1 range
            if feature == "latency":
                normalized_value = max(0, 1.0 - value / 1000.0)  # 1000ms max
            elif feature == "bandwidth":
                normalized_value = min(1.0, value / 1000.0)  # 1000 Mbps max
            elif feature in ["reliability", "load"]:
                normalized_value = value
            elif feature == "packet_loss":
                normalized_value = max(0, 1.0 - value)
            else:
                normalized_value = value
                
            score += weight * normalized_value
        
        return score
    
    def _dijkstra_shortest_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[RoutingPath]:
        """Find shortest path using Dijkstra's algorithm."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
            
        # Priority queue: (cost, node_id, path)
        pq = [(0.0, source_id, [source_id])]
        visited = set()
        
        while pq:
            current_cost, current_node, path = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            if current_node == target_id:
                # Found target, create routing path
                total_latency = sum(
                    self.links.get((path[i], path[i+1]), NetworkLink("", "")).latency
                    for i in range(len(path) - 1)
                )
                min_bandwidth = min(
                    self.links.get((path[i], path[i+1]), NetworkLink("", "")).bandwidth
                    for i in range(len(path) - 1)
                ) if len(path) > 1 else 0.0
                
                reliability_score = self._calculate_path_score(path)
                
                return RoutingPath(
                    nodes=path,
                    total_latency=total_latency,
                    total_bandwidth=min_bandwidth,
                    reliability_score=reliability_score,
                    cost_score=current_cost
                )
            
            # Explore neighbors
            for neighbor_id in self.adjacency.get(current_node, set()):
                if neighbor_id not in visited:
                    link_key = (current_node, neighbor_id)
                    link = self.links.get(link_key)
                    
                    if link:
                        # Check if neighbor node is reachable
                        neighbor_node = self.nodes.get(neighbor_id)
                        if neighbor_node and neighbor_node.status != NodeStatus.UNREACHABLE:
                            new_cost = current_cost + link.cost
                            new_path = path + [neighbor_id]
                            heapq.heappush(pq, (new_cost, neighbor_id, new_path))
        
        return None  # No path found
    
    def _find_ml_optimized_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[RoutingPath]:
        """Find ML-optimized path considering learned preferences."""
        # Use modified Dijkstra with ML-based cost function
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
            
        # Priority queue: (ml_score, node_id, path, metrics)
        pq = [(-1.0, source_id, [source_id], {"latency": 0.0, "bandwidth": float('inf')})]
        visited = set()
        
        while pq:
            neg_score, current_node, path, path_metrics = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            if current_node == target_id:
                # Found target
                return RoutingPath(
                    nodes=path,
                    total_latency=path_metrics["latency"],
                    total_bandwidth=path_metrics["bandwidth"],
                    reliability_score=-neg_score,
                    cost_score=len(path) - 1
                )
            
            # Explore neighbors
            for neighbor_id in self.adjacency.get(current_node, set()):
                if neighbor_id not in visited:
                    link_key = (current_node, neighbor_id)
                    link = self.links.get(link_key)
                    neighbor_node = self.nodes.get(neighbor_id)
                    
                    if link and neighbor_node and neighbor_node.status != NodeStatus.UNREACHABLE:
                        new_path = path + [neighbor_id]
                        new_metrics = {
                            "latency": path_metrics["latency"] + link.latency,
                            "bandwidth": min(path_metrics["bandwidth"], link.bandwidth)
                        }
                        
                        # Calculate ML score for this path
                        ml_score = self._calculate_path_score(new_path)
                        
                        heapq.heappush(pq, (-ml_score, neighbor_id, new_path, new_metrics))
        
        return None
    
    def find_route(
        self,
        source_id: str,
        target_id: str,
        strategy: Optional[RoutingStrategy] = None
    ) -> Optional[RoutingPath]:
        """Find optimal route between nodes."""
        if strategy is None:
            strategy = self.strategy
            
        # Check cache first
        cache_key = f"{source_id}:{target_id}:{strategy.value}"
        if cache_key in self.route_cache:
            cached_route = self.route_cache[cache_key]
            # Check if cache is still valid (5 minutes)
            if time.time() - cached_route.timestamp < 300:
                return cached_route
            else:
                del self.route_cache[cache_key]
        
        # Find route based on strategy
        route = None
        
        if strategy == RoutingStrategy.SHORTEST_PATH:
            route = self._dijkstra_shortest_path(source_id, target_id)
        elif strategy == RoutingStrategy.ML_OPTIMIZED:
            route = self._find_ml_optimized_path(source_id, target_id)
        elif strategy == RoutingStrategy.ADAPTIVE:
            # Try ML-optimized first, fall back to shortest path
            route = self._find_ml_optimized_path(source_id, target_id)
            if not route:
                route = self._dijkstra_shortest_path(source_id, target_id)
        else:
            # Default to shortest path
            route = self._dijkstra_shortest_path(source_id, target_id)
        
        # Cache the route
        if route:
            self.route_cache[cache_key] = route
            
            # Keep cache size manageable
            if len(self.route_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.route_cache.keys(),
                    key=lambda k: self.route_cache[k].timestamp
                )[:500]
                for key in oldest_keys:
                    del self.route_cache[key]
        
        return route
    
    def record_route_performance(
        self,
        path: List[str],
        success: bool,
        actual_latency: float,
        actual_bandwidth: float,
        packet_loss: float = 0.0
    ):
        """Record actual route performance for learning."""
        path_hash = ":".join(path)
        current_time = time.time()
        
        if path_hash not in self.route_performance:
            self.route_performance[path_hash] = RoutePerformance(
                path_hash=path_hash,
                success_rate=1.0 if success else 0.0,
                avg_latency=actual_latency,
                avg_bandwidth=actual_bandwidth,
                packet_loss=packet_loss,
                last_used=current_time,
                usage_count=1
            )
        else:
            perf = self.route_performance[path_hash]
            
            # Update running averages
            alpha = self.learning_rate
            perf.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * perf.success_rate
            perf.avg_latency = alpha * actual_latency + (1 - alpha) * perf.avg_latency
            perf.avg_bandwidth = alpha * actual_bandwidth + (1 - alpha) * perf.avg_bandwidth
            perf.packet_loss = alpha * packet_loss + (1 - alpha) * perf.packet_loss
            perf.last_used = current_time
            perf.usage_count += 1
        
        # Learn and update feature weights
        self._update_feature_weights(path, success, actual_latency, actual_bandwidth, packet_loss)
        
        # Record in history
        history_entry = {
            "timestamp": current_time,
            "path": path,
            "success": success,
            "latency": actual_latency,
            "bandwidth": actual_bandwidth,
            "packet_loss": packet_loss
        }
        
        self.routing_history.append(history_entry)
        
        # Keep history manageable
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]
    
    def _update_feature_weights(
        self,
        path: List[str],
        success: bool,
        actual_latency: float,
        actual_bandwidth: float,
        packet_loss: float
    ):
        """Update feature weights based on route performance."""
        if len(path) < 2:
            return
            
        # Calculate prediction error
        predicted_score = self._calculate_path_score(path)
        actual_score = 1.0 if success else 0.0
        
        # Adjust for actual performance vs predicted
        if actual_latency > 0:
            latency_factor = min(2.0, 100.0 / actual_latency)  # Good latency = 2.0, bad = 0.1
            actual_score *= latency_factor
            
        if actual_bandwidth > 0:
            bandwidth_factor = min(2.0, actual_bandwidth / 100.0)  # Good bandwidth = 2.0
            actual_score *= bandwidth_factor
            
        if packet_loss > 0:
            loss_factor = max(0.1, 1.0 - packet_loss * 10)  # Penalty for packet loss
            actual_score *= loss_factor
        
        # Update weights using gradient descent
        prediction_error = actual_score - predicted_score
        
        if abs(prediction_error) > 0.1:  # Only update for significant errors
            adjustment = self.learning_rate * prediction_error
            
            # Update weights based on path characteristics
            for feature in self.feature_weights:
                if feature == "latency" and actual_latency > 0:
                    self.feature_weights[feature] += adjustment * (0.5 if actual_latency < 50 else -0.5)
                elif feature == "bandwidth" and actual_bandwidth > 0:
                    self.feature_weights[feature] += adjustment * (0.5 if actual_bandwidth > 100 else -0.5)
                elif feature == "packet_loss" and packet_loss > 0:
                    self.feature_weights[feature] += adjustment * -0.5
    
    async def _topology_update_loop(self):
        """Periodically update network topology."""
        logger.info("Network topology monitoring started")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for stale nodes (not seen in 5 minutes)
                stale_nodes = []
                for node_id, node in self.nodes.items():
                    if current_time - node.last_seen > 300:  # 5 minutes
                        stale_nodes.append(node_id)
                        
                # Mark stale nodes as unreachable
                for node_id in stale_nodes:
                    self.nodes[node_id].status = NodeStatus.UNREACHABLE
                    logger.warning(f"Node {node_id} marked as unreachable (stale)")
                
                # Clear route cache for unreachable nodes
                cache_keys_to_remove = []
                for cache_key in self.route_cache:
                    for node_id in stale_nodes:
                        if node_id in cache_key:
                            cache_keys_to_remove.append(cache_key)
                            break
                            
                for cache_key in cache_keys_to_remove:
                    del self.route_cache[cache_key]
                
                await asyncio.sleep(self.topology_update_interval)
                
            except Exception as e:
                logger.error(f"Error in topology update loop: {e}")
                await asyncio.sleep(self.topology_update_interval)
    
    async def start(self):
        """Start the intelligent router."""
        if self.is_running:
            logger.warning("Intelligent router is already running")
            return
            
        self.is_running = True
        self._topology_task = asyncio.create_task(self._topology_update_loop())
        logger.info("Intelligent router started")
    
    async def stop(self):
        """Stop the intelligent router."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self._topology_task:
            self._topology_task.cancel()
            try:
                await self._topology_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Intelligent router stopped")
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """Get network topology summary."""
        healthy_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.HEALTHY)
        total_links = len(self.links)
        
        return {
            "nodes": {
                "total": len(self.nodes),
                "healthy": healthy_nodes,
                "degraded": sum(1 for node in self.nodes.values() if node.status == NodeStatus.DEGRADED),
                "overloaded": sum(1 for node in self.nodes.values() if node.status == NodeStatus.OVERLOADED),
                "unreachable": sum(1 for node in self.nodes.values() if node.status == NodeStatus.UNREACHABLE)
            },
            "links": {
                "total": total_links,
                "avg_latency": statistics.mean([link.latency for link in self.links.values()]) if self.links else 0.0,
                "avg_bandwidth": statistics.mean([link.bandwidth for link in self.links.values()]) if self.links else 0.0
            },
            "routing": {
                "cached_routes": len(self.route_cache),
                "performance_records": len(self.route_performance),
                "strategy": self.strategy.value
            },
            "learning": {
                "feature_weights": self.feature_weights.copy(),
                "routing_history_size": len(self.routing_history)
            }
        }
    
    def get_route_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and insights."""
        if not self.route_performance:
            return {"message": "No routing performance data available"}
            
        # Analyze route performance
        success_rates = [perf.success_rate for perf in self.route_performance.values()]
        latencies = [perf.avg_latency for perf in self.route_performance.values()]
        bandwidths = [perf.avg_bandwidth for perf in self.route_performance.values()]
        
        # Find best and worst performing routes
        best_route = max(self.route_performance.values(), key=lambda p: p.success_rate)
        worst_route = min(self.route_performance.values(), key=lambda p: p.success_rate)
        
        return {
            "performance_summary": {
                "avg_success_rate": statistics.mean(success_rates),
                "avg_latency": statistics.mean(latencies),
                "avg_bandwidth": statistics.mean(bandwidths),
                "total_routes_analyzed": len(self.route_performance)
            },
            "best_route": {
                "path": best_route.path_hash.split(":"),
                "success_rate": best_route.success_rate,
                "avg_latency": best_route.avg_latency,
                "usage_count": best_route.usage_count
            },
            "worst_route": {
                "path": worst_route.path_hash.split(":"),
                "success_rate": worst_route.success_rate,
                "avg_latency": worst_route.avg_latency,
                "usage_count": worst_route.usage_count
            },
            "feature_weights": self.feature_weights.copy()
        }