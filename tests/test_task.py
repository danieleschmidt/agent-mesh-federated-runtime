"""Tests for FederatedTask."""
import pytest
from agent_mesh import FederatedTask, TaskResult


def test_split_list_into_equal_parts():
    task = FederatedTask(payload=list(range(10)), work_fn=sum)
    shards = task.split(5)
    assert len(shards) == 5
    # all elements present
    assert sorted(x for shard in shards for x in shard) == list(range(10))


def test_split_non_list_replicates():
    task = FederatedTask(payload={"key": "value"}, work_fn=lambda x: x)
    shards = task.split(3)
    assert len(shards) == 3
    assert all(s == {"key": "value"} for s in shards)


def test_execute_shard_captures_result():
    task = FederatedTask(payload=[1, 2, 3, 4, 5], work_fn=sum)
    result = task.execute_shard(0, [1, 2, 3], "node-a")
    assert result.value == 6
    assert result.error is None


def test_execute_shard_captures_exception():
    def bad_fn(data):
        raise ValueError("boom")

    task = FederatedTask(payload=[1], work_fn=bad_fn)
    result = task.execute_shard(0, [1], "node-a")
    assert result.error == "boom"
    assert result.value is None


def test_aggregate_sum_of_shards():
    data = list(range(1, 11))  # 1..10 → sum = 55
    task = FederatedTask(payload=data, work_fn=sum)
    shards = task.split(5)
    for i, shard in enumerate(shards):
        task.execute_shard(i, shard, f"node-{i}")
    assert task.aggregate() == 55


def test_aggregate_skips_error_shards():
    data = list(range(1, 11))
    task = FederatedTask(payload=data, work_fn=sum)
    shards = task.split(5)
    # execute first 3, inject error for shard 3
    for i in range(3):
        task.execute_shard(i, shards[i], f"node-{i}")
    task.collect_result(TaskResult(task.task_id, "node-3", 3, None, error="network failure"))
    task.execute_shard(4, shards[4], "node-4")
    # only shards 0-2 and 4 contribute
    partial_sum = sum(sum(shards[i]) for i in [0, 1, 2, 4])
    assert task.aggregate() == partial_sum


def test_default_aggregator_element_wise_list():
    task = FederatedTask(payload=[], work_fn=lambda x: x)
    task.collect_result(TaskResult(task.task_id, "n1", 0, [1.0, 2.0, 3.0]))
    task.collect_result(TaskResult(task.task_id, "n2", 1, [4.0, 5.0, 6.0]))
    result = task.aggregate()
    assert result == [5.0, 7.0, 9.0]


def test_custom_aggregator():
    task = FederatedTask(
        payload=list(range(6)),
        work_fn=lambda shard: max(shard),
        aggregator=max,
    )
    shards = task.split(3)
    for i, shard in enumerate(shards):
        task.execute_shard(i, shard, f"node-{i}")
    assert task.aggregate() == 5  # max of all shard maxima
