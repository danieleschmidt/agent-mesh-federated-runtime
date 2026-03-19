"""FederatedTask — distributed task splitting and result aggregation.

A FederatedTask describes work that can be divided across multiple worker
nodes.  Each node processes its shard independently and returns a
TaskResult.  The coordinator then aggregates all partial results into a
final answer.

The aggregation strategy is pluggable; the default merges numeric lists
by summing element-wise (suitable for e.g. gradient aggregation).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional


AggregatorFn = Callable[[List[Any]], Any]


def _default_aggregator(partials: List[Any]) -> Any:
    """Default: numeric element-wise sum, or plain list concatenation."""
    if not partials:
        return None
    first = partials[0]
    if isinstance(first, (int, float)):
        return sum(partials)
    if isinstance(first, list) and all(isinstance(v, (int, float)) for p in partials for v in p):
        length = max(len(p) for p in partials)
        result = [0.0] * length
        for partial in partials:
            for i, v in enumerate(partial):
                result[i] += v
        return result
    # fallback: concatenate
    if isinstance(first, list):
        out: list = []
        for p in partials:
            out.extend(p)
        return out
    return partials


class TaskResult:
    """Holds the partial or final result from a node."""

    def __init__(
        self,
        task_id: str,
        node_id: str,
        shard_index: int,
        value: Any,
        error: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.node_id = node_id
        self.shard_index = shard_index
        self.value = value
        self.error = error
        self.completed_at = time.time()

    def __repr__(self) -> str:
        if self.error:
            return f"TaskResult(shard={self.shard_index}, error={self.error!r})"
        return f"TaskResult(shard={self.shard_index}, value={self.value!r})"


class FederatedTask:
    """A unit of work that can be split and executed across mesh nodes.

    Parameters
    ----------
    payload : Any
        The full input data (list, dict, etc.) to be processed.
    work_fn : Callable[[Any], Any]
        Pure function applied to each shard on a worker node.
    aggregator : AggregatorFn, optional
        Function that merges partial results.  Defaults to element-wise sum.
    task_id : str, optional
        Unique ID; auto-generated if not provided.
    """

    def __init__(
        self,
        payload: Any,
        work_fn: Callable[[Any], Any],
        aggregator: Optional[AggregatorFn] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.task_id: str = task_id or str(uuid.uuid4())
        self.payload = payload
        self.work_fn = work_fn
        self.aggregator: AggregatorFn = aggregator or _default_aggregator
        self.created_at = time.time()
        self._results: Dict[int, TaskResult] = {}

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, n: int) -> List[Any]:
        """Divide *payload* into *n* roughly-equal shards.

        Works for list payloads.  Non-list payloads are replicated as-is
        (useful for tasks like "each node votes on X").
        """
        if isinstance(self.payload, list):
            size = max(1, len(self.payload) // n)
            shards = []
            for i in range(n):
                start = i * size
                end = start + size if i < n - 1 else len(self.payload)
                shards.append(self.payload[start:end])
            return shards
        # non-list: replicate
        return [self.payload] * n

    # ------------------------------------------------------------------
    # Execution (in-process, deterministic)
    # ------------------------------------------------------------------

    def execute_shard(self, shard_index: int, shard_data: Any, node_id: str) -> TaskResult:
        """Run *work_fn* on a shard.  Captures exceptions as error results."""
        try:
            value = self.work_fn(shard_data)
            result = TaskResult(self.task_id, node_id, shard_index, value)
        except Exception as exc:  # noqa: BLE001
            result = TaskResult(self.task_id, node_id, shard_index, None, error=str(exc))
        self._results[shard_index] = result
        return result

    def collect_result(self, result: TaskResult) -> None:
        """Register a partial result returned from a worker node."""
        self._results[result.shard_index] = result

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> Any:
        """Merge all collected partial results into the final answer.

        Ignores error shards (logs them instead).
        """
        good_values = []
        for idx in sorted(self._results):
            r = self._results[idx]
            if r.error:
                continue
            good_values.append(r.value)
        return self.aggregator(good_values)

    @property
    def is_complete(self) -> bool:
        return bool(self._results)

    def __repr__(self) -> str:
        return (
            f"FederatedTask(id={self.task_id[:8]}…, "
            f"shards_done={len(self._results)})"
        )
