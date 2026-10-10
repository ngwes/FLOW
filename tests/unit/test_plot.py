import json
import sys
from pathlib import Path

import pytest

from flaw.plot import extract_series, plot_runs, read_metrics


def round_line(
    round_number: int,
    *,
    aggregated: dict[str, object] | None = None,
    central: dict[str, object] | None = None,
    void: bool = False,
) -> dict[str, object]:
    line: dict[str, object] = {
        "type": "round",
        "round": round_number,
        "void": void,
        "void_reason": "insufficient_participants" if void else None,
        "closed_reason": None,
        "participants": [],
        "rejected": [],
        "duration_s": 1.0,
    }
    if aggregated is not None:
        line["aggregated"] = aggregated
    if central is not None:
        line["central"] = central
    return line


def write_metrics(run_dir: Path, lines: list[dict[str, object]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n",
        encoding="utf-8",
    )
    return path


def test_read_metrics_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert read_metrics(path) == [{"a": 1}, {"b": 2}]


def test_read_metrics_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_metrics(tmp_path / "nope.jsonl")


def test_extract_series_skips_void_rows_and_missing_metrics() -> None:
    lines = [
        round_line(1, aggregated={"val_accuracy": 0.5}),
        round_line(2, void=True),  # void: no aggregated dict at all
        round_line(3, aggregated={"val_loss": 0.1}),  # different metric
        round_line(4, aggregated={"val_accuracy": "high"}),  # not a number
        {"type": "summary", "reason": "max_rounds"},  # not a round line
    ]
    assert extract_series(lines, "val_accuracy") == ([1], [0.5])


def test_extract_series_central_source() -> None:
    lines = [
        round_line(1, aggregated={"val_accuracy": 0.5}, central={"val_accuracy": 0.4}),
        round_line(2, void=True),
        round_line(3, aggregated={"val_accuracy": 0.6}, central={"val_accuracy": 0.55}),
    ]
    assert extract_series(lines, "val_accuracy", source="central") == ([1, 3], [0.4, 0.55])


def test_plot_runs_writes_png(tmp_path: Path) -> None:
    run_a = tmp_path / "iid"
    write_metrics(
        run_a,
        [
            round_line(1, aggregated={"val_accuracy": 0.5}, central={"val_accuracy": 0.45}),
            round_line(2, aggregated={"val_accuracy": 0.7}, central={"val_accuracy": 0.6}),
        ],
    )
    run_b = tmp_path / "dirichlet"
    write_metrics(run_b, [round_line(1, aggregated={"val_accuracy": 0.4})])  # no central series
    out = tmp_path / "nested" / "plot.png"
    result = plot_runs({"iid": run_a, "dirichlet": run_b}, out)
    assert result == out
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_runs_without_central_lines(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_metrics(run, [round_line(1, aggregated={"val_loss": 0.5})])
    out = tmp_path / "plot.png"
    assert plot_runs({"run": run}, out, metric="val_loss", include_central=False) == out
    assert out.exists()


def test_plot_runs_no_plottable_data_raises(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_metrics(run, [round_line(1, aggregated={"val_loss": 0.5}), round_line(2, void=True)])
    with pytest.raises(ValueError, match="val_accuracy"):
        plot_runs({"run": run}, tmp_path / "plot.png")


def test_plot_runs_without_matplotlib_raises_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` fail
    with pytest.raises(RuntimeError, match=r"flaw\[plot\]"):
        plot_runs({"run": tmp_path / "run"}, tmp_path / "plot.png")
