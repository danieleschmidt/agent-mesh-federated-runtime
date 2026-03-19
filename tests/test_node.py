"""Tests for AgentNode."""
import pytest
from agent_mesh import AgentNode, NodeRole


def test_default_role_is_worker():
    node = AgentNode()
    assert node.role == NodeRole.WORKER


def test_custom_role():
    node = AgentNode(role=NodeRole.COORDINATOR)
    assert node.role == NodeRole.COORDINATOR


def test_capability_check():
    node = AgentNode(capabilities=frozenset({"compute", "storage"}))
    assert node.has_capability("compute")
    assert node.has_capability("storage")
    assert not node.has_capability("gpu")


def test_alive_toggle():
    node = AgentNode()
    assert node.alive
    node.mark_dead()
    assert not node.alive
    node.mark_alive()
    assert node.alive


def test_equality_by_id():
    node_a = AgentNode(node_id="abc")
    node_b = AgentNode(node_id="abc")
    node_c = AgentNode(node_id="xyz")
    assert node_a == node_b
    assert node_a != node_c


def test_repr_contains_role():
    node = AgentNode(role=NodeRole.OBSERVER)
    assert "observer" in repr(node)
