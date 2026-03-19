"""ByzantineFaultTolerance — majority-vote based detection and node exclusion.

Strategy
--------
Each node can "report" a value for a given slot (e.g. a computation result
or a health status).  The BFT engine collects all reports and uses a
configurable majority threshold to decide the canonical value and flag
nodes whose reports deviate from it as faulty.

This is a simplified, synchronous implementation of Byzantine voting,
sufficient to tolerate up to ⌊(n-1)/3⌋ faulty nodes when threshold ≥ 2/3.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ByzantineFaultTolerance:
    """Majority-vote BFT engine.

    Parameters
    ----------
    threshold : float
        Fraction of agreeing votes required to accept a value (default 0.67).
    """

    def __init__(self, threshold: float = 0.67) -> None:
        if not (0.5 < threshold <= 1.0):
            raise ValueError("threshold must be in (0.5, 1.0]")
        self.threshold = threshold
        # slot_id → {node_id: reported_value}
        self._reports: Dict[str, Dict[str, Any]] = {}
        # nodes confirmed faulty
        self._faulty_nodes: Set[str] = set()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, slot_id: str, node_id: str, value: Any) -> None:
        """Record that *node_id* reports *value* for *slot_id*."""
        if node_id in self._faulty_nodes:
            logger.debug("Ignoring report from known-faulty node %s", node_id[:8])
            return
        if slot_id not in self._reports:
            self._reports[slot_id] = {}
        self._reports[slot_id][node_id] = value

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, slot_id: str) -> Tuple[Optional[Any], Set[str]]:
        """Compute the majority decision for *slot_id*.

        Returns
        -------
        (canonical_value, newly_faulty_nodes)
            canonical_value is None if no majority was reached.
            newly_faulty_nodes is the set of nodes whose reports differed.
        """
        reports = self._reports.get(slot_id, {})
        if not reports:
            return None, set()

        n = len(reports)
        # Count occurrences of each reported value.
        # Values must be hashable; coerce lists to tuples for counting.
        def _hashable(v: Any) -> Any:
            if isinstance(v, list):
                return tuple(_hashable(x) for x in v)
            if isinstance(v, dict):
                return tuple(sorted((k, _hashable(val)) for k, val in v.items()))
            return v

        counter: Counter = Counter()
        for v in reports.values():
            counter[_hashable(v)] += 1

        canonical_key, count = counter.most_common(1)[0]
        fraction = count / n

        if fraction < self.threshold:
            logger.warning(
                "No BFT majority for slot %s (best=%.0f%%, need=%.0f%%)",
                slot_id,
                fraction * 100,
                self.threshold * 100,
            )
            return None, set()

        # Restore original (unhashable) canonical value
        canonical_value = next(
            v for nid, v in reports.items() if _hashable(v) == canonical_key
        )

        # Identify deviants
        deviants: Set[str] = set()
        for node_id, value in reports.items():
            if _hashable(value) != canonical_key:
                deviants.add(node_id)
                self._faulty_nodes.add(node_id)
                logger.warning("Node %s marked faulty (deviant vote)", node_id[:8])

        return canonical_value, deviants

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def faulty_nodes(self) -> Set[str]:
        return frozenset(self._faulty_nodes)

    def is_faulty(self, node_id: str) -> bool:
        return node_id in self._faulty_nodes

    def clear_slot(self, slot_id: str) -> None:
        """Remove all reports for a slot (used after decision)."""
        self._reports.pop(slot_id, None)

    def reset(self) -> None:
        """Clear all state (useful between test rounds)."""
        self._reports.clear()
        self._faulty_nodes.clear()

    def summary(self, slot_id: str) -> str:
        reports = self._reports.get(slot_id, {})
        lines = [f"BFT slot '{slot_id}' ({len(reports)} reporters):"]
        for nid, val in reports.items():
            tag = " ⚠ FAULTY" if nid in self._faulty_nodes else ""
            lines.append(f"  {nid[:8]}…: {val!r}{tag}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def collect_and_decide(
        self,
        slot_id: str,
        node_values: List[Tuple[str, Any]],
    ) -> Tuple[Optional[Any], Set[str]]:
        """Convenience: report all values then decide in one call."""
        for node_id, value in node_values:
            self.report(slot_id, node_id, value)
        return self.decide(slot_id)
