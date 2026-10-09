import json
from pathlib import Path

import pytest
import uvicorn

from flaw.simulate import run_simulation

# The federation line is split across two source lines (implicit concatenation) to stay
# under the 100-char ruff limit; the runtime string is still the single mandated YAML line.
SIM_CFG = (
    """
run: {name: sim1, seed: 7}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 32}
federation: {token: secret, rounds: 50, min_participants: 2, round_deadline_s: 60, """
    """heartbeat_s: 0.2, liveness_timeout_s: 5}
"""
)


def write_cfg(tmp_path: Path, text: str = SIM_CFG) -> Path:
    path = tmp_path / "flaw.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_simulate_two_nodes_end_to_end(tmp_path: Path) -> None:
    result = run_simulation(
        write_cfg(tmp_path),
        nodes=2,
        rounds=2,  # overrides the config's 50 rounds to keep the test fast
        run_root=tmp_path / "runs",
        timeout_s=180.0,
    )
    assert result.reason == "max_rounds"
    assert result.rounds == 2
    assert result.best_metric is not None and result.best_round in (1, 2)
    assert (result.run_dir / "final.pt").exists()
    assert (result.run_dir / "config.yaml").exists()  # config snapshot from prepare_run_dir

    lines = [
        json.loads(line)
        for line in (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rounds = [line for line in lines if line["type"] == "round"]
    summary = [line for line in lines if line["type"] == "summary"]
    assert len(rounds) == 2
    assert all(line["void"] is False for line in rounds)
    assert sorted(p["node_id"] for p in rounds[0]["participants"]) == ["sim-0", "sim-1"]
    assert len(summary) == 1 and summary[0]["reason"] == "max_rounds"


def test_simulate_times_out_when_min_participants_unreachable(tmp_path: Path) -> None:
    text = SIM_CFG.replace("min_participants: 2", "min_participants: 3")
    with pytest.raises(TimeoutError, match="min_participants"):
        run_simulation(
            write_cfg(tmp_path, text), nodes=2, run_root=tmp_path / "runs", timeout_s=5.0
        )


def test_simulate_rejects_too_few_shards(tmp_path: Path) -> None:
    text = SIM_CFG.replace("num_shards: 2", "num_shards: 1")
    with pytest.raises(ValueError, match="num_shards"):
        run_simulation(write_cfg(tmp_path, text), nodes=2)


def test_simulate_rejects_zero_nodes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nodes"):
        run_simulation(write_cfg(tmp_path), nodes=0)


def test_simulate_fails_fast_when_server_thread_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A no-op run() returns immediately: the thread dies and `server.started`
    # stays False, so the startup poll must fail fast instead of polling on.
    monkeypatch.setattr(uvicorn.Server, "run", lambda self: None)
    with pytest.raises(RuntimeError, match="failed to start"):
        run_simulation(write_cfg(tmp_path), nodes=1, run_root=tmp_path / "runs", timeout_s=5.0)


def test_simulate_reports_nodes_that_exited_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every node thread returns "fatal_error" immediately; the run can never
    # finish, so the wait loop must notice all threads are dead and say so.
    def fake_run_node(*args: object, **kwargs: object) -> str:
        return "fatal_error"

    monkeypatch.setattr("flaw.simulate.run_node", fake_run_node)
    with pytest.raises(RuntimeError, match="exited early"):
        run_simulation(write_cfg(tmp_path), nodes=2, run_root=tmp_path / "runs", timeout_s=30.0)
