"""Plot run metrics from `metrics.jsonl` files.

matplotlib is an optional dependency: `pip install 'flaw[plot]'`. It is imported
lazily so that everything else keeps working without it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def read_metrics(path: Path) -> list[dict[str, object]]:
    """Read a metrics JSONL file into a list of dicts (blank lines skipped)."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pyplot() -> Any:
    """Import matplotlib lazily, always with a headless backend (we write files)."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting needs matplotlib - install it with: pip install 'flaw[plot]'"
        ) from exc
    return plt


def extract_series(
    lines: Sequence[Mapping[str, object]],
    metric: str,
    *,
    source: str = "aggregated",
) -> tuple[list[int], list[float]]:
    """Pull `(rounds, values)` for one metric, skipping void and non-numeric rows.

    `source` selects which per-round dict to read: `"aggregated"` (federated
    metrics) or `"central"` (central test-set evaluation, when enabled).
    """
    rounds: list[int] = []
    values: list[float] = []
    for line in lines:
        if line.get("type") != "round" or line.get("void"):
            continue
        series = line.get(source)
        round_number = line.get("round")
        if not isinstance(series, dict) or not isinstance(round_number, int):
            continue
        value = series.get(metric)
        if not isinstance(value, (int, float)):
            continue
        rounds.append(round_number)
        values.append(float(value))
    return rounds, values


def plot_runs(
    runs: Mapping[str, Path],
    out_path: Path,
    *,
    metric: str = "val_accuracy",
    include_central: bool = True,
) -> Path:
    """Plot `metric` per round for each labeled run directory; returns the PNG path."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    drawn = 0
    for label, run_dir in runs.items():
        lines = read_metrics(Path(run_dir) / "metrics.jsonl")
        rounds, values = extract_series(lines, metric)
        if rounds:
            drawn += 1
            ax.plot(rounds, values, marker="o", label=f"{label} (federated)")
        if include_central:
            central_rounds, central_values = extract_series(lines, metric, source="central")
            if central_rounds:
                drawn += 1
                ax.plot(
                    central_rounds,
                    central_values,
                    linestyle="--",
                    marker="s",
                    label=f"{label} (central test set)",
                )
    if drawn == 0:
        plt.close(fig)
        raise ValueError(f"no plottable data for metric {metric!r} in any of the given runs")
    ax.set_xlabel("round")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"FL@W — {metric} per round")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
