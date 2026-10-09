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
