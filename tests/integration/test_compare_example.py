"""The IID vs non-IID comparison example runs end to end on tiny configs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compare_example_runs_and_plots(tmp_path: Path) -> None:
    out_png = tmp_path / "compare.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "compare_iid_non_iid" / "compare.py"),
            "--rounds",
            "2",
            "--nodes",
            "2",
            "--timeout",
            "180",
            "--run-root",
            str(tmp_path / "runs"),
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
    assert (tmp_path / "runs" / "cmp-iid" / "metrics.jsonl").exists()
    assert (tmp_path / "runs" / "cmp-dirichlet" / "metrics.jsonl").exists()
