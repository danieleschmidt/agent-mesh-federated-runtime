"""Tests for MeshNetwork."""
import pytest
from agent_mesh import AgentNode, MeshNetwork, NodeRole


def _make_net(n: int = 3):
    """Return a fully-connected mesh of n nodes."""
    net = MeshNetwork()
    nodes = [AgentNode() for _ in range(n)]
    for node in nodes:
        net.add_node(node)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            net.connect(a.node_id, b.node_id)
    return net, nodes


def test_add_and_remove_node():
    net = MeshNetwork()
    node = AgentNode()
    net.add_node(node)
    assert net.get_node(node.node_id) is node
    net.remove_node(node.node_id)
    assert net.get_node(node.node_id) is None


def test_connect_creates_bidirectional_edge():
    net = MeshNetwork()
    a, b = AgentNode(), AgentNode()
    net.add_node(a)
    net.add_node(b)
    net.connect(a.node_id, b.node_id)
    assert b in net.peers_of(a.node_id)
    assert a in net.peers_of(b.node_id)


def test_disconnect():
    net = MeshNetwork()
    a, b = AgentNode(), AgentNode()
    net.add_node(a)
    net.add_node(b)
    net.connect(a.node_id, b.node_id)
    net.disconnect(a.node_id, b.node_id)
    assert b not in net.peers_of(a.node_id)


def test_unicast_delivers_message():
    net, (a, b, c) = _make_net(3)
    received = []
    net.register_handler("ping", lambda sender, msg: received.append(msg))
    net.unicast(a, b.node_id, {"type": "ping", "data": "hello"})
    assert len(received) == 1
    assert received[0]["data"] == "hello"


def test_unicast_fails_for_dead_node():
    net, (a, b, c) = _make_net(3)
    b.mark_dead()
    ok = net.unicast(a, b.node_id, {"type": "ping"})
    assert ok is False


def test_broadcast_reaches_all_others():
    net, nodes = _make_net(5)
    hits = []
    net.register_handler("bcast", lambda sender, msg: hits.append(msg["_to"]))
    sender = nodes[0]
    count = net.broadcast(sender, {"type": "bcast"})
    assert count == 4
    assert len(hits) == 4
    assert sender.node_id not in hits


def test_broadcast_skips_dead_nodes():
    net, nodes = _make_net(4)
    nodes[1].mark_dead()
    count = net.broadcast(nodes[0], {"type": "bcast"})
    assert count == 2  # 4 total - sender - 1 dead


def test_gossip_spreads():
    net, nodes = _make_net(5)
    seen = set()
    net.register_handler("gossip_msg", lambda sender, msg: seen.add(msg["_to"]))
    # fanout=2 from a 5-node mesh should reach multiple nodes
    net.gossip(nodes[0], {"type": "gossip_msg"})
    assert len(seen) >= 1


def test_coordinator_returns_none_when_absent():
    net = MeshNetwork()
    node = AgentNode(role=NodeRole.WORKER)
    net.add_node(node)
    assert net.coordinator() is None


def test_coordinator_found():
    net = MeshNetwork()
    node = AgentNode(role=NodeRole.COORDINATOR)
    net.add_node(node)
    assert net.coordinator() is node
