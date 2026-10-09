"""Run directory layout and crash-safe run state (spec 4.3).

One :class:`RunDir` per run owns the on-disk artifact layout under
``root/<run-name>/``: the config snapshot, ``state.json``, ``metrics.jsonl``,
the ``blobs/`` round directories, ``best.pt``/``final.pt`` (kept outside
``blobs/`` so blob GC can never touch them) and the orchestrator log.

``state.json`` is written atomically (tmp file + :func:`os.replace`) so a
crash mid-save can never leave a truncated state file behind.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Serializes the tmp-write + replace below: the /register threadpool handler and
# the round-loop thread both save, and both share the fixed ``state.json.tmp``
# path, so concurrent saves could interleave and publish a torn state file.
_SAVE_STATE_LOCK = threading.Lock()


@dataclass
class RunState:
    """Mutable run bookkeeping persisted to ``state.json``."""

    run_name: str
    round: int = 0
    phase: str = "idle"
    shard_map: dict[str, int] = field(default_factory=dict)
    best_metric: float | None = None
    best_round: int | None = None
    status: str = "running"
    finished_reason: str | None = None


class RunDir:
    """Paths and persistence for one run's artifacts under ``root/<run_name>``."""

    def __init__(self, root: Path, run_name: str) -> None:
        self.root = root
        self.run_name = run_name

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def blobs_dir(self) -> Path:
        return self.run_dir / "blobs"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def final_path(self) -> Path:
        return self.run_dir / "final.pt"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "orchestrator.log"

    def ensure(self) -> None:
        """Create the run directory and its ``blobs/`` subdirectory."""
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: RunState) -> None:
        """Persist *state* to ``state.json`` atomically (thread-safe)."""
        with _SAVE_STATE_LOCK:
            payload = json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.state_path)

    def load_state(self) -> RunState | None:
        """Return the saved state, or ``None`` when no state file exists yet."""
        if not self.state_path.exists():
            return None
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RunState(**data)

    def snapshot_config(self, text: str) -> None:
        """Write the exact training-description text used for this run."""
        self.config_path.write_text(text, encoding="utf-8")

    def save_best(self, sd: Mapping[str, torch.Tensor]) -> None:
        """Write the best global weights to ``best.pt``."""
        torch.save(dict(sd), self.best_path)

    def save_final(self, sd: Mapping[str, torch.Tensor]) -> None:
        """Write the last global weights to ``final.pt``."""
        torch.save(dict(sd), self.final_path)
