"""Node registry: liveness tracking and shard assignment.

The orchestrator keeps one :class:`NodeRegistry` for the whole run.  It maps
node ids to :class:`NodeRecord` entries, marks nodes dead when they stop
heartbeating (``clock() - last_seen > liveness_timeout_s``), and hands out
shard indices.  Time comes from an injected ``clock`` (``time.monotonic`` in
production) so liveness can be tested deterministically.

Shard assignments survive an orchestrator restart via
:meth:`NodeRegistry.load_shards`/:meth:`NodeRegistry.shard_map`: assignments
for nodes that have not re-registered yet are held in a pending map — this is
needed because the orchestrator restores the shard map from ``state.json``
before any node connects, and each node picks up its reserved shard when it
registers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from flaw.protocol import Capabilities


class RegistryError(Exception):
    """Raised for node-registry violations."""


class NodeConflict(RegistryError):
    """A *live* record already uses the requested node id."""


class UnknownNode(RegistryError):
    """The requested node id has never been registered."""


class NoShardAvailable(RegistryError):
    """Every shard index is already taken."""


@dataclass
class NodeRecord:
    """Liveness and shard state for one registered node."""

    node_id: str
    capabilities: Capabilities
    shard: int | None = None
    last_seen: float = 0.0
    alive: bool = True


class NodeRegistry:
    """Tracks node liveness and shard assignments for one run."""

    def __init__(self, liveness_timeout_s: float, clock: Callable[[], float]) -> None:
        self.liveness_timeout_s = liveness_timeout_s
        self._clock = clock
        self._records: dict[str, NodeRecord] = {}
        self._pending_shards: dict[str, int] = {}

    def register(self, node_id: str, capabilities: Capabilities) -> NodeRecord:
        """Register *node_id*; a dead record is revived in place keeping its shard.

        Raises :class:`NodeConflict` when a live record already holds *node_id*.
        """
        existing = self._records.get(node_id)
        if existing is not None:
            if existing.alive:
                raise NodeConflict(f"node {node_id!r} is already registered and alive")
            existing.alive = True
            existing.capabilities = capabilities
            existing.last_seen = self._clock()
            return existing
        record = NodeRecord(
            node_id=node_id,
            capabilities=capabilities,
            shard=self._pending_shards.pop(node_id, None),
            last_seen=self._clock(),
        )
        self._records[node_id] = record
        return record

    def heartbeat(self, node_id: str) -> NodeRecord:
        """Update *last_seen* and revive the record; unknown id -> :class:`UnknownNode`."""
        record = self.get(node_id)
        record.last_seen = self._clock()
        record.alive = True
        return record

    def sweep(self) -> list[str]:
        """Mark timed-out records dead; return the ids newly marked dead."""
        now = self._clock()
        # copy: /register runs on a threadpool thread and may insert while we iterate
        snapshot = list(self._records.values())
        newly_dead = sorted(
            record.node_id
            for record in snapshot
            if record.alive and now - record.last_seen > self.liveness_timeout_s
        )
        for node_id in newly_dead:
            self._records[node_id].alive = False
        return newly_dead

    def get(self, node_id: str) -> NodeRecord:
        """Return the record for *node_id*; unknown id -> :class:`UnknownNode`."""
        try:
            return self._records[node_id]
        except KeyError:
            raise UnknownNode(f"unknown node: {node_id!r}") from None

    def is_alive(self, node_id: str) -> bool:
        """Return whether *node_id* is registered and currently alive."""
        record = self._records.get(node_id)
        return record is not None and record.alive

    def alive_nodes(self) -> list[NodeRecord]:
        """Live records sorted by node id."""
        # copy: /register runs on a threadpool thread and may insert while we iterate
        snapshot = list(self._records.values())
        return sorted(
            (record for record in snapshot if record.alive),
            key=lambda record: record.node_id,
        )

    def dead_nodes(self) -> list[NodeRecord]:
        """Dead records sorted by node id."""
        # copy: /register runs on a threadpool thread and may insert while we iterate
        snapshot = list(self._records.values())
        return sorted(
            (record for record in snapshot if not record.alive),
            key=lambda record: record.node_id,
        )

    def assign_shard(self, node_id: str, num_shards: int) -> int:
        """Assign the smallest unused shard to *node_id*.

        Idempotent for a node that already has one; raises
        :class:`NoShardAvailable` when every index is taken.
        """
        record = self.get(node_id)
        if record.shard is not None:
            return record.shard
        used = set(self.shard_map().values())
        free = set(range(num_shards)) - used
        if not free:
            raise NoShardAvailable(f"all {num_shards} shards are taken")
        record.shard = min(free)
        return record.shard

    def shard_map(self) -> dict[str, int]:
        """Current node -> shard mapping, including still-pending reservations."""
        # copy both: /register inserts into _records and pops _pending_shards
        records = list(self._records.values())
        mapping = {record.node_id: record.shard for record in records if record.shard is not None}
        mapping.update(list(self._pending_shards.items()))
        return mapping

    def load_shards(self, shard_map: dict[str, int]) -> None:
        """Restore shard assignments after an orchestrator restart.

        Registered nodes are updated in place; assignments for nodes that have
        not (re-)registered yet are kept pending and applied on their next
        :meth:`register`, so a restored assignment is never double-assigned to
        another node in the meantime.
        """
        for node_id, shard in shard_map.items():
            record = self._records.get(node_id)
            if record is None:
                self._pending_shards[node_id] = shard
            else:
                record.shard = shard
