#!/usr/bin/env python3
"""Performance benchmarking script for Agent Mesh Federated Runtime."""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import statistics

import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BenchmarkResult:
    """Benchmark result data class."""
    name: str
    duration: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    success_rate: float
    metadata: Dict[str, Any]


class MeshBenchmark:
    """Agent Mesh performance benchmark suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
    
    async def benchmark_mesh_join(self, num_nodes: int = 10) -> BenchmarkResult:
        """Benchmark mesh network join latency."""
        print(f"🔄 Benchmarking mesh join with {num_nodes} nodes...")
        
        latencies = []
        successful_joins = 0
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Simulate nodes joining the mesh
            tasks = []
            for i in range(num_nodes):
                task = self._join_mesh_node(session, f"node-{i:03d}")
                tasks.append(task)
            
            # Execute joins concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_duration = time.time() - start_time
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    continue
                if result:
                    latencies.append(result)
                    successful_joins += 1
        
        if not latencies:
            latencies = [0.0]  # Avoid empty list
        
        return BenchmarkResult(
            name="mesh_join",
            duration=total_duration,
            throughput=successful_joins / total_duration,
            latency_p50=statistics.median(latencies),
            latency_p95=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
            latency_p99=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0],
            success_rate=successful_joins / num_nodes,
            metadata={"num_nodes": num_nodes}
        )
    
    async def _join_mesh_node(self, session: aiohttp.ClientSession, node_id: str) -> float:
        """Join a single node to the mesh and measure latency."""
        start_time = time.time()
        
        try:
            payload = {
                "node_id": node_id,
                "role": "auto",
                "capabilities": ["training", "aggregation"]
            }
            
            async with session.post(f"{self.base_url}/mesh/join", json=payload) as response:
                if response.status == 200:
                    return time.time() - start_time
                return None
        except Exception:
            return None
    
    async def benchmark_federated_round(self, num_participants: int = 5) -> BenchmarkResult:
        """Benchmark federated learning round performance."""
        print(f"🔄 Benchmarking federated round with {num_participants} participants...")
        
        round_times = []
        successful_rounds = 0
        
        async with aiohttp.ClientSession() as session:
            # Run multiple federated rounds
            for round_num in range(3):
                start_time = time.time()
                
                # Simulate federated learning round
                round_success = await self._execute_fed_round(session, round_num, num_participants)
                
                if round_success:
                    round_duration = time.time() - start_time
                    round_times.append(round_duration)
                    successful_rounds += 1
        
        if not round_times:
            round_times = [0.0]
        
        total_duration = sum(round_times)
        
        return BenchmarkResult(
            name="federated_round",
            duration=total_duration,
            throughput=successful_rounds / total_duration if total_duration > 0 else 0,
            latency_p50=statistics.median(round_times),
            latency_p95=max(round_times),
            latency_p99=max(round_times),
            success_rate=successful_rounds / 3,
            metadata={"num_participants": num_participants}
        )
    
    async def _execute_fed_round(self, session: aiohttp.ClientSession, round_num: int, num_participants: int) -> bool:
        """Execute a single federated learning round."""
        try:
            payload = {
                "round_id": round_num,
                "participants": [f"node-{i:03d}" for i in range(num_participants)],
                "aggregation_strategy": "fedavg"
            }
            
            async with session.post(f"{self.base_url}/federated/round", json=payload) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def benchmark_consensus(self, num_validators: int = 7) -> BenchmarkResult:
        """Benchmark consensus algorithm performance."""
        print(f"🔄 Benchmarking consensus with {num_validators} validators...")
        
        consensus_times = []
        successful_consensus = 0
        
        async with aiohttp.ClientSession() as session:
            # Run multiple consensus rounds
            for proposal_id in range(5):
                start_time = time.time()
                
                consensus_success = await self._run_consensus(session, proposal_id, num_validators)
                
                if consensus_success:
                    consensus_duration = time.time() - start_time
                    consensus_times.append(consensus_duration)
                    successful_consensus += 1
        
        if not consensus_times:
            consensus_times = [0.0]
        
        total_duration = sum(consensus_times)
        
        return BenchmarkResult(
            name="consensus",
            duration=total_duration,
            throughput=successful_consensus / total_duration if total_duration > 0 else 0,
            latency_p50=statistics.median(consensus_times),
            latency_p95=max(consensus_times),
            latency_p99=max(consensus_times),
            success_rate=successful_consensus / 5,
            metadata={"num_validators": num_validators}
        )
    
    async def _run_consensus(self, session: aiohttp.ClientSession, proposal_id: int, num_validators: int) -> bool:
        """Run a single consensus round."""
        try:
            payload = {
                "proposal_id": proposal_id,
                "proposal_data": {"model_update": "dummy_update"},
                "validators": [f"validator-{i:03d}" for i in range(num_validators)]
            }
            
            async with session.post(f"{self.base_url}/consensus/propose", json=payload) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("🚀 Starting Agent Mesh performance benchmarks...\n")
        
        benchmarks = [
            (self.benchmark_mesh_join, [5, 10, 20]),
            (self.benchmark_federated_round, [3, 5, 10]),
            (self.benchmark_consensus, [3, 5, 7]),
        ]
        
        for benchmark_func, param_list in benchmarks:
            for param in param_list:
                try:
                    result = await benchmark_func(param)
                    self.results.append(result)
                    self._print_result(result)
                except Exception as e:
                    print(f"❌ Benchmark failed: {e}")
        
        self._generate_report()
    
    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result summary."""
        print(f"✅ {result.name} ({result.metadata}):")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Throughput: {result.throughput:.2f} ops/sec")
        print(f"   Latency P50: {result.latency_p50:.3f}s")
        print(f"   Latency P95: {result.latency_p95:.3f}s")
        print(f"   Success Rate: {result.success_rate:.1%}\n")
    
    def _generate_report(self):
        """Generate comprehensive benchmark report."""
        print("📊 Generating benchmark report...")
        
        # Create results directory
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        json_data = []
        for result in self.results:
            json_data.append({
                "name": result.name,
                "duration": result.duration,
                "throughput": result.throughput,
                "latency_p50": result.latency_p50,
                "latency_p95": result.latency_p95,
                "latency_p99": result.latency_p99,
                "success_rate": result.success_rate,
                "metadata": result.metadata,
                "timestamp": time.time()
            })
        
        with open(results_dir / "benchmark_results.json", "w") as f:
            json.dump(json_data, f, indent=2)
        
        # Generate performance charts
        self._create_charts(results_dir)
        
        print(f"📁 Benchmark results saved to {results_dir}/")
    
    def _create_charts(self, results_dir: Path):
        """Create performance visualization charts."""
        plt.style.use('seaborn-v0_8')
        
        # Throughput comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group results by benchmark type
        benchmark_groups = {}
        for result in self.results:
            if result.name not in benchmark_groups:
                benchmark_groups[result.name] = []
            benchmark_groups[result.name].append(result)
        
        # Throughput chart
        for bench_name, bench_results in benchmark_groups.items():
            throughputs = [r.throughput for r in bench_results]
            params = [list(r.metadata.values())[0] for r in bench_results]
            ax1.plot(params, throughputs, marker='o', label=bench_name)
        
        ax1.set_xlabel('Scale Parameter')
        ax1.set_ylabel('Throughput (ops/sec)')
        ax1.set_title('Throughput vs Scale')
        ax1.legend()
        ax1.grid(True)
        
        # Latency chart
        for bench_name, bench_results in benchmark_groups.items():
            latencies = [r.latency_p95 for r in bench_results]
            params = [list(r.metadata.values())[0] for r in bench_results]
            ax2.plot(params, latencies, marker='s', label=bench_name)
        
        ax2.set_xlabel('Scale Parameter')
        ax2.set_ylabel('Latency P95 (seconds)')
        ax2.set_title('Latency vs Scale')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / "performance_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Performance charts saved")


async def main():
    """Main benchmark execution."""
    benchmark = MeshBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())