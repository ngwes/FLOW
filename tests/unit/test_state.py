from pathlib import Path

import torch

from flaw.orchestrator.state import RunDir, RunState


def make(tmp_path: Path) -> RunDir:
    rd = RunDir(root=tmp_path / "runs", run_name="exp1")
    rd.ensure()
    return rd


def test_paths_layout(tmp_path: Path) -> None:
    rd = RunDir(root=tmp_path / "runs", run_name="exp1")
    assert rd.run_dir == tmp_path / "runs" / "exp1"
    assert rd.state_path == rd.run_dir / "state.json"
    assert rd.config_path == rd.run_dir / "config.yaml"
    assert rd.metrics_path == rd.run_dir / "metrics.jsonl"
    assert rd.blobs_dir == rd.run_dir / "blobs"
    assert rd.best_path == rd.run_dir / "best.pt"
    assert rd.final_path == rd.run_dir / "final.pt"
    assert rd.log_path == rd.run_dir / "orchestrator.log"


def test_ensure_creates_dirs(tmp_path: Path) -> None:
    rd = make(tmp_path)
    assert rd.run_dir.is_dir()
    assert rd.blobs_dir.is_dir()


def test_state_roundtrip(tmp_path: Path) -> None:
    rd = make(tmp_path)
    state = RunState(
        run_name="exp1",
        round=3,
        phase="open",
        shard_map={"a": 0, "b": 1},
        best_metric=0.9,
        best_round=2,
    )
    rd.save_state(state)
    assert rd.load_state() == state


def test_load_state_missing_returns_none(tmp_path: Path) -> None:
    assert make(tmp_path).load_state() is None


def test_save_state_no_tmp_leftover(tmp_path: Path) -> None:
    rd = make(tmp_path)
    rd.save_state(RunState(run_name="exp1"))
    assert not list(rd.run_dir.glob("*.tmp"))


def test_snapshot_config(tmp_path: Path) -> None:
    rd = make(tmp_path)
    rd.snapshot_config("run:\n  name: exp1\n")
    assert rd.config_path.read_text(encoding="utf-8").startswith("run:")


def test_best_and_final_roundtrip(tmp_path: Path) -> None:
    rd = make(tmp_path)
    sd = {"w": torch.tensor([1.0])}
    rd.save_best(sd)
    rd.save_final(sd)
    assert torch.equal(torch.load(rd.best_path, weights_only=True)["w"], sd["w"])
    assert torch.equal(torch.load(rd.final_path, weights_only=True)["w"], sd["w"])
