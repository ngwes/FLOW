"""Federated (folder-per-node) mode with real files, plus the example data generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

from flaw.simulate import run_simulation

ROOT = Path(__file__).resolve().parents[2]


def write_folder(root: Path, split: str) -> None:
    for label in range(10):
        folder = root / split / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("L", (28, 28), color=(label * 25) % 256).save(folder / "img0.png")


def test_example_data_generator_creates_folders(tmp_path: Path) -> None:
    script = ROOT / "examples" / "federated" / "make_sample_data.py"
    subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path), "--nodes", "2"],
        check=True,
        cwd=ROOT,
    )
    for node in ("node1", "node2"):
        assert sorted(p.name for p in (tmp_path / node / "train").iterdir()) == [
            str(i) for i in range(10)
        ]
        assert sorted(p.name for p in (tmp_path / node / "val").iterdir()) == [
            str(i) for i in range(10)
        ]


def test_federated_folder_mode_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_folder(data, "train")
    write_folder(data, "val")
    cfg_path = tmp_path / "federated.yaml"
    cfg_path.write_text(
        f"""
run: {{name: fed, seed: 5}}
model: {{name: mnist_cnn}}
data: {{path: "{data.as_posix()}"}}
training: {{local_epochs: 1, batch_size: 32}}
federation: {{token: secret, rounds: 2, min_participants: 2, round_deadline_s: 60, """
        """heartbeat_s: 0.2, liveness_timeout_s: 5}
""",
        encoding="utf-8",
    )
    result = run_simulation(cfg_path, nodes=2, run_root=tmp_path / "runs", timeout_s=180.0)
    assert result.reason == "max_rounds"
    assert result.rounds == 2
    assert sorted(p.name for p in (result.run_dir / "blobs").glob("r????"))[-1] == "r0003"
