"""Full Docker smoke: compose up (orchestrator + 2 nodes), tiny-sim config, real run.

Requires a Docker daemon and the `docker` compose plugin. Run with: pytest -m docker
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-p", "flaw-smoke"]


def until(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout}s waiting for: {what}")


@pytest.mark.docker
def test_compose_tiny_sim_smoke(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    env = {
        **os.environ,
        "FLAW_RUNS_DIR": str(runs),
        "FLAW_DATA_DIR": str(tmp_path / "data"),
    }
    subprocess.run([*COMPOSE, "build"], cwd=ROOT, env=env, check=True, timeout=1800)
    subprocess.run([*COMPOSE, "up", "-d"], cwd=ROOT, env=env, check=True, timeout=300)
    try:
        state_path = runs / "tiny-sim" / "state.json"

        def finished() -> bool:
            return (
                state_path.exists()
                and json.loads(state_path.read_text(encoding="utf-8"))["status"] == "finished"
            )

        try:
            until(finished, timeout=600, what="compose run to finish")
        except AssertionError:
            logs = subprocess.run(
                [*COMPOSE, "logs", "--no-color"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            raise AssertionError(f"compose run did not finish.\n{logs.stdout[-4000:]}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["finished_reason"] == "max_rounds"
        lines = [
            json.loads(line)
            for line in (runs / "tiny-sim" / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        rounds = [line for line in lines if line["type"] == "round"]
        assert len(rounds) == 4 and all(not line["void"] for line in rounds)
        node_ids = sorted(p["node_id"] for p in rounds[0]["participants"])
        assert node_ids == ["node1", "node2"]  # container identities, not hostnames
    finally:
        subprocess.run([*COMPOSE, "down", "-v"], cwd=ROOT, env=env, check=False, timeout=300)
