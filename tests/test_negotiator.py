"""Tests for RoleNegotiator (Bully election)."""
import pytest
from agent_mesh import AgentNode, MeshNetwork, NodeRole
from agent_mesh.negotiator import RoleNegotiator


def _full_mesh(n: int):
    net = MeshNetwork()
    nodes = [AgentNode() for _ in range(n)]
    for node in nodes:
        net.add_node(node)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            net.connect(a.node_id, b.node_id)
    return net, nodes, RoleNegotiator(net)


def test_initial_assignment_creates_one_coordinator():
    net, nodes, neg = _full_mesh(5)
    neg.assign_initial_roles()
    coordinators = [n for n in nodes if n.role == NodeRole.COORDINATOR]
    assert len(coordinators) == 1


def test_initial_coordinator_is_highest_id():
    net, nodes, neg = _full_mesh(5)
    neg.assign_initial_roles()
    coord = net.coordinator()
    highest = max(nodes, key=lambda n: n.node_id)
    assert coord == highest


def test_detect_failure_no_coordinator():
    net, nodes, neg = _full_mesh(3)
    # no coordinator assigned yet
    assert neg.detect_failure()


def test_detect_failure_with_live_coordinator():
    net, nodes, neg = _full_mesh(3)
    neg.assign_initial_roles()
    assert not neg.detect_failure()


def test_election_after_coordinator_crash():
    net, nodes, neg = _full_mesh(5)
    neg.assign_initial_roles()
    old_coord = net.coordinator()

    # crash the coordinator
    old_coord.mark_dead()
    assert neg.detect_failure()

    new_coord = neg.elect_coordinator()
    assert new_coord is not None
    assert new_coord != old_coord
    assert new_coord.role == NodeRole.COORDINATOR


def test_election_with_only_one_node():
    net, nodes, neg = _full_mesh(1)
    new_coord = neg.elect_coordinator()
    assert new_coord is nodes[0]
    assert new_coord.role == NodeRole.COORDINATOR


def test_election_yields_single_coordinator():
    net, nodes, neg = _full_mesh(4)
    neg.assign_initial_roles()
    net.coordinator().mark_dead()
    neg.elect_coordinator()
    alive_coords = [n for n in net.alive_nodes if n.role == NodeRole.COORDINATOR]
    assert len(alive_coords) == 1


def test_no_election_possible_when_all_dead():
    net, nodes, neg = _full_mesh(3)
    for n in nodes:
        n.mark_dead()
    result = neg.elect_coordinator()
    assert result is None
