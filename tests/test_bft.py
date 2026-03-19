"""Tests for ByzantineFaultTolerance."""
import pytest
from agent_mesh import ByzantineFaultTolerance


def test_simple_majority_consensus():
    bft = ByzantineFaultTolerance(threshold=0.67)
    bft.report("slot1", "n1", 42)
    bft.report("slot1", "n2", 42)
    bft.report("slot1", "n3", 42)
    canonical, deviants = bft.decide("slot1")
    assert canonical == 42
    assert deviants == set()


def test_faulty_node_detected():
    bft = ByzantineFaultTolerance(threshold=0.67)
    bft.report("slot1", "n1", 42)
    bft.report("slot1", "n2", 42)
    bft.report("slot1", "n3", 42)
    bft.report("slot1", "n4", 99)  # faulty
    canonical, deviants = bft.decide("slot1")
    assert canonical == 42
    assert "n4" in deviants


def test_no_majority_returns_none():
    bft = ByzantineFaultTolerance(threshold=0.67)
    bft.report("slot1", "n1", 1)
    bft.report("slot1", "n2", 2)
    bft.report("slot1", "n3", 3)
    canonical, deviants = bft.decide("slot1")
    assert canonical is None


def test_faulty_node_reports_ignored_after_detection():
    # Use 4 nodes: 3 honest + 1 faulty → 3/4 = 75% ≥ threshold
    bft = ByzantineFaultTolerance(threshold=0.67)
    bft.report("slot1", "n1", 42)
    bft.report("slot1", "n2", 42)
    bft.report("slot1", "n3", 42)
    bft.report("slot1", "faulty", 99)
    bft.decide("slot1")

    assert bft.is_faulty("faulty")

    bft.clear_slot("slot1")
    bft.report("slot1", "n1", 100)
    bft.report("slot1", "n2", 100)
    bft.report("slot1", "faulty", 0)  # should be ignored
    canonical, _ = bft.decide("slot1")
    assert canonical == 100


def test_collect_and_decide_convenience():
    bft = ByzantineFaultTolerance(threshold=0.6)
    votes = [("n1", "ok"), ("n2", "ok"), ("n3", "ok"), ("n4", "bad")]
    canonical, deviants = bft.collect_and_decide("health", votes)
    assert canonical == "ok"
    assert "n4" in deviants


def test_reset_clears_all_state():
    bft = ByzantineFaultTolerance()
    bft.report("s1", "n1", 1)
    bft.report("s1", "n2", 99)
    bft.decide("s1")
    bft.reset()
    assert len(bft.faulty_nodes) == 0
    canonical, _ = bft.decide("s1")
    assert canonical is None  # no reports after reset


def test_list_values_hashed_correctly():
    bft = ByzantineFaultTolerance(threshold=0.67)
    bft.report("s", "n1", [1, 2, 3])
    bft.report("s", "n2", [1, 2, 3])
    bft.report("s", "n3", [1, 2, 3])
    canonical, deviants = bft.decide("s")
    assert canonical == [1, 2, 3]
    assert deviants == set()


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        ByzantineFaultTolerance(threshold=0.3)
