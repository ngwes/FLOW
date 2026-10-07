import pytest

from flaw.orchestrator.registry import (
    NodeConflict,
    NodeRegistry,
    NoShardAvailable,
    UnknownNode,
)
from flaw.protocol import Capabilities


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def caps() -> Capabilities:
    return Capabilities(device="cpu", device_name="test-cpu")


def make() -> tuple[NodeRegistry, FakeClock]:
    clock = FakeClock()
    return NodeRegistry(liveness_timeout_s=30.0, clock=clock), clock


def test_register_new_node() -> None:
    reg, _ = make()
    rec = reg.register("a", caps())
    assert rec.alive and rec.shard is None and rec.last_seen == 0.0
    assert reg.is_alive("a")


def test_duplicate_live_register_conflicts() -> None:
    reg, _ = make()
    reg.register("a", caps())
    with pytest.raises(NodeConflict):
        reg.register("a", caps())


def test_reregister_after_death_keeps_shard() -> None:
    reg, clock = make()
    reg.register("a", caps())
    reg.assign_shard("a", 3)
    clock.now = 31.0
    assert reg.sweep() == ["a"]
    rec = reg.register("a", caps())
    assert rec.alive and rec.shard == 0


def test_heartbeat_unknown_raises() -> None:
    reg, _ = make()
    with pytest.raises(UnknownNode):
        reg.heartbeat("ghost")


def test_heartbeat_updates_and_revives() -> None:
    reg, clock = make()
    reg.register("a", caps())
    clock.now = 20.0
    reg.heartbeat("a")
    clock.now = 31.0  # 11 s since heartbeat, timeout is 30 -> still alive
    assert reg.sweep() == []
    assert reg.is_alive("a")
    clock.now = 100.0
    assert reg.sweep() == ["a"]
    reg.heartbeat("a")
    assert reg.is_alive("a")


def test_sweep_reports_dead_once() -> None:
    reg, clock = make()
    reg.register("a", caps())
    clock.now = 31.0
    assert reg.sweep() == ["a"]
    assert reg.sweep() == []


def test_alive_and_dead_lists_sorted() -> None:
    reg, clock = make()
    reg.register("b", caps())
    reg.register("a", caps())
    assert [r.node_id for r in reg.alive_nodes()] == ["a", "b"]
    clock.now = 31.0
    reg.sweep()
    assert [r.node_id for r in reg.dead_nodes()] == ["a", "b"]


def test_shard_assignment_sequential_and_idempotent() -> None:
    reg, _ = make()
    reg.register("a", caps())
    reg.register("b", caps())
    reg.register("c", caps())
    assert reg.assign_shard("a", 3) == 0
    assert reg.assign_shard("b", 3) == 1
    assert reg.assign_shard("a", 3) == 0  # idempotent
    assert reg.assign_shard("c", 3) == 2
    assert reg.shard_map() == {"a": 0, "b": 1, "c": 2}


def test_shard_exhaustion() -> None:
    reg, _ = make()
    reg.register("a", caps())
    reg.register("b", caps())
    reg.assign_shard("a", 1)
    with pytest.raises(NoShardAvailable):
        reg.assign_shard("b", 1)


def test_assign_shard_unknown_raises() -> None:
    reg, _ = make()
    with pytest.raises(UnknownNode):
        reg.assign_shard("ghost", 3)


def test_load_shards_restores() -> None:
    reg, _ = make()
    reg.register("a", caps())
    reg.load_shards({"a": 2})
    assert reg.shard_map() == {"a": 2}
    assert reg.assign_shard("a", 3) == 2
    with pytest.raises(UnknownNode):
        reg.get("b")


def test_load_shards_pending_applied_on_register() -> None:
    reg, _ = make()
    reg.load_shards({"late": 1})
    assert reg.shard_map() == {"late": 1}
    reg.register("new", caps())
    assert reg.assign_shard("new", 3) == 0  # shard 1 is reserved for "late"
    assert reg.register("late", caps()).shard == 1
    assert reg.shard_map() == {"late": 1, "new": 0}
