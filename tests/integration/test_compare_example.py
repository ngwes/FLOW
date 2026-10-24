"""The IID vs non-IID comparison example runs end to end (offline, on the tiny dataset)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compare_example_runs_and_plots(tmp_path: Path) -> None:
    out_png = tmp_path / "compare.png"
    runs = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "compare_iid_non_iid" / "compare.py"),
            "--dataset",
            "tiny",
            "--per-class",
            "5",
            "--rounds",
            "2",
            "--nodes",
            "2",
            "--timeout",
            "180",
            "--run-root",
            str(runs),
            "--out",
            str(out_png),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "IID" in completed.stdout and "non-IID" in completed.stdout
    assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # one shared held-out set: 10 classes x 5 images, scored every round in both runs
    assert len(list((runs / "central-tiny").glob("*/*.png"))) == 50
    for run_name in ("cmp-iid", "cmp-dirichlet"):
        lines = (runs / run_name / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        rounds = [json.loads(line) for line in lines if '"type": "round"' in line]
        assert rounds and all("val_accuracy" in r["central"] for r in rounds), run_name
