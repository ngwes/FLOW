import logging
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from httpx import Response
from typer.testing import CliRunner

from flaw import cli
from flaw.simulate import SimulationResult

runner = CliRunner()

CFG = """
run: {name: exp1, seed: 42}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 16}
federation: {token: secret, rounds: 2, min_participants: 2}
"""


def write_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "flaw.yaml"
    path.write_text(CFG, encoding="utf-8")
    return path


def test_configure_logging_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: calls.update(kw))
    cli._configure_logging(True)
    assert calls["level"] == logging.DEBUG
    cli._configure_logging(False)
    assert calls["level"] == logging.INFO


def test_per_request_http_logs_only_when_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    # httpx logs every request at INFO; at lab scale that buries the round lines.
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: None)
    httpx_logger = logging.getLogger("httpx")
    monkeypatch.setattr(httpx_logger, "level", httpx_logger.level)  # restored afterwards
    cli._configure_logging(False)
    assert httpx_logger.level == logging.WARNING
    cli._configure_logging(True)
    assert httpx_logger.level == logging.DEBUG


@pytest.mark.parametrize(("flags", "access_log"), [([], False), (["-v"], True)])
def test_orchestrator_access_log_only_when_verbose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], access_log: bool
) -> None:
    calls: dict = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: calls.update(kw))
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: None)
    args = [*flags, "orchestrator", str(write_cfg(tmp_path)), "--run-root", str(tmp_path / "r")]
    assert runner.invoke(cli.app, args).exit_code == 0
    assert calls["access_log"] is access_log


def test_orchestrator_command_builds_app_and_snapshots_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: calls.update(kw, app=app))
    result = runner.invoke(
        cli.app,
        [
            "orchestrator",
            str(write_cfg(tmp_path)),
            "--run-root",
            str(tmp_path / "runs"),
            "--port",
            "9999",
        ],
    )
    assert result.exit_code == 0
    assert calls["port"] == 9999 and calls["host"] == "0.0.0.0"
    assert isinstance(calls["app"], FastAPI)
    snapshot = tmp_path / "runs" / "exp1" / "config.yaml"
    assert snapshot.exists() and snapshot.read_text(encoding="utf-8") == CFG


def test_node_command_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    def fake_run_node(cfg, url, **kw):  # type: ignore[no-untyped-def]
        calls.update(url=url)
        return "finished"

    monkeypatch.setattr(cli, "run_node", fake_run_node)
    result = runner.invoke(
        cli.app, ["node", str(write_cfg(tmp_path)), "--orchestrator", "http://hp:1234"]
    )
    assert result.exit_code == 0
    assert calls["url"] == "http://hp:1234"
    assert "finished" in result.output


def test_node_command_fatal_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_node", lambda *a, **kw: "fatal_error")
    result = runner.invoke(cli.app, ["node", str(write_cfg(tmp_path))])
    assert result.exit_code == 1 and "fatal_error" in result.output


def test_simulate_command_prints_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimulationResult(
        run_dir=tmp_path / "runs" / "exp1",
        reason="max_rounds",
        rounds=2,
        best_metric=0.97,
        best_round=1,
    )
    captured: dict = {}

    def fake_sim(path, nodes, **kw):  # type: ignore[no-untyped-def]
        captured.update(nodes=nodes, rounds=kw.get("rounds"))
        return fake

    monkeypatch.setattr(cli, "run_simulation", fake_sim)
    result = runner.invoke(
        cli.app, ["simulate", str(write_cfg(tmp_path)), "--nodes", "3", "--rounds", "2"]
    )
    assert result.exit_code == 0
    assert captured == {"nodes": 3, "rounds": 2}
    assert "max_rounds" in result.output and "0.9700" in result.output


def test_simulate_command_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **kw: object) -> SimulationResult:
        raise TimeoutError("no round opened (federation.min_participants is higher than the ...)")

    monkeypatch.setattr(cli, "run_simulation", boom)
    result = runner.invoke(cli.app, ["simulate", str(write_cfg(tmp_path))])
    assert result.exit_code == 1 and "min_participants" in result.output


def test_invalid_config_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["node", str(bad)])
    assert result.exit_code == 2 and "Invalid config" in result.output


@respx.mock
def test_status_command_renders_nodes() -> None:
    payload = {
        "name": "exp1",
        "phase": "open",
        "round": 3,
        "nodes": [
            {
                "node_id": "a",
                "alive": True,
                "shard": 0,
                "device": "cuda:0",
                "last_heartbeat_age_s": 1.5,
            }
        ],
        "best_metric": 0.9,
        "best_round": 2,
    }
    respx.get("http://orch.test:8000/status").mock(return_value=Response(200, json=payload))
    result = runner.invoke(cli.app, ["status", "--orchestrator", "http://orch.test:8000"])
    assert result.exit_code == 0
    assert "phase=open" in result.output
    assert "cuda:0" in result.output and "1.5" in result.output


@respx.mock
def test_status_command_unreachable_exits_1() -> None:
    respx.get("http://down.test/status").mock(side_effect=httpx.ConnectError("refused"))
    result = runner.invoke(cli.app, ["status", "--orchestrator", "http://down.test"])
    assert result.exit_code == 1 and "Cannot reach" in result.output


def test_plot_command_passes_runs_labels_and_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict = {}

    def fake_plot_runs(runs, out_path, **kw):  # type: ignore[no-untyped-def]
        calls.update(runs=runs, out_path=out_path, **kw)
        return Path(out_path)

    monkeypatch.setattr(cli, "plot_runs", fake_plot_runs)
    result = runner.invoke(
        cli.app,
        [
            "plot",
            str(tmp_path / "runA"),
            str(tmp_path / "runB"),
            "--labels",
            "iid, non-iid",
            "--metric",
            "val_loss",
            "--out",
            str(tmp_path / "plot.png"),
        ],
    )
    assert result.exit_code == 0
    assert calls["runs"] == {"iid": tmp_path / "runA", "non-iid": tmp_path / "runB"}
    assert calls["metric"] == "val_loss"
    assert calls["out_path"] == tmp_path / "plot.png"
    assert "plot.png" in result.output


def test_plot_command_default_labels_are_directory_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_plot_runs(runs, out_path, **kw):  # type: ignore[no-untyped-def]
        captured.update(runs=runs)
        return Path(out_path)

    monkeypatch.setattr(cli, "plot_runs", fake_plot_runs)
    result = runner.invoke(cli.app, ["plot", str(tmp_path / "tiny-sim")])
    assert result.exit_code == 0
    assert captured["runs"] == {"tiny-sim": tmp_path / "tiny-sim"}


def test_plot_command_label_count_mismatch_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "plot_runs", lambda *a, **kw: pytest.fail("must not be called"))
    result = runner.invoke(cli.app, ["plot", str(tmp_path / "a"), "--labels", "x,y"])
    assert result.exit_code == 2
    assert "--labels" in result.output


def test_plot_command_missing_matplotlib_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kw: object) -> None:
        raise RuntimeError("plotting needs matplotlib - install it with: pip install 'flaw[plot]'")

    monkeypatch.setattr(cli, "plot_runs", boom)
    result = runner.invoke(cli.app, ["plot", str(tmp_path / "a")])
    assert result.exit_code == 1
    assert "flaw[plot]" in result.output


def test_plot_command_bad_metric_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kw: object) -> None:
        raise ValueError("no plottable data for metric 'nope'")

    monkeypatch.setattr(cli, "plot_runs", boom)
    result = runner.invoke(cli.app, ["plot", str(tmp_path / "a"), "--metric", "nope"])
    assert result.exit_code == 1
    assert "no plottable data" in result.output


# --- fix wave: rich-markup escapes on interpolated text ---


def test_invalid_config_error_text_survives_rich_markup(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\nbogus: 1\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["node", str(bad)])
    assert result.exit_code == 2
    assert "extra_forbidden" in result.output
    # the bracketed diagnostic is what rich markup would otherwise eat
    assert "[type=extra_forbidden" in result.output


def test_simulation_failure_text_survives_rich_markup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kw: object) -> SimulationResult:
        raise RuntimeError("boom [bold] end")

    monkeypatch.setattr(cli, "run_simulation", boom)
    result = runner.invoke(cli.app, ["simulate", str(write_cfg(tmp_path))])
    assert result.exit_code == 1
    assert "boom [bold] end" in result.output


def test_plot_success_line_escapes_out_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "plot_runs", lambda runs, out, **kw: Path(out))
    weird_path = tmp_path / "o[bold]ut.png"
    result = runner.invoke(cli.app, ["plot", str(tmp_path / "run"), "--out", str(weird_path)])
    assert result.exit_code == 0
    assert str(weird_path) in result.output


def test_plot_command_missing_run_dir_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["plot", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "Plotting failed" in result.output
