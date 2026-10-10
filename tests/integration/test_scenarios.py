"""End-to-end scenarios over real HTTP: joins, crashes, restarts, early stop, central eval."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from PIL import Image

from flaw.config import FlawConfig, load_config
from flaw.node.client import OrchestratorClient
from flaw.node.runner import run_node
from flaw.orchestrator.server import create_app
from flaw.orchestrator.state import RunDir
from flaw.simulate import free_port

BASE: dict[str, Any] = {
    "run": {"name": "scen", "seed": 3},
    "model": {"name": "mnist_cnn"},
    "data": {"name": "tiny", "num_shards": 3, "scheme": "iid", "val_fraction": 0.25},
    "training": {"local_epochs": 1, "batch_size": 32},
    "federation": {
        "token": "secret",
        "rounds": 2,
        "min_participants": 1,
        "round_deadline_s": 30,
        "heartbeat_s": 0.1,
        "liveness_timeout_s": 1.0,
    },
}


def make_cfg(tmp_path: Path, name: str, **sections: dict[str, Any]) -> FlawConfig:
    payload = copy.deepcopy(BASE)
    payload["run"]["name"] = name
    for key, value in sections.items():
        payload.setdefault(key, {}).update(value)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_config(path)


def until(predicate: Callable[[], bool], timeout: float = 60.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout}s waiting for: {what}")


class NodeHandle:
    """A real node loop in a thread, stoppable on its own."""

    def __init__(
        self,
        client: OrchestratorClient,
        stop: threading.Event,
        thread: threading.Thread,
        results: dict[str, str],
        node_id: str,
    ) -> None:
        self.client = client
        self.stop = stop
        self.thread = thread
        self.results = results
        self.node_id = node_id

    def stop_node(self) -> str:
        self.stop.set()
        self.thread.join(timeout=30.0)
        self.client.close()
        return self.results.get(self.node_id, "still-running")


class LiveOrchestrator:
    """A real uvicorn orchestrator in a thread, for end-to-end scenario tests."""

    def __init__(
        self,
        cfg: FlawConfig,
        run_dir: RunDir,
        long_poll_s: float = 0.5,
        port: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        app = create_app(cfg, run_dir, long_poll_s=long_poll_s)
        self.port = free_port() if port is None else port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="scenario-orchestrator"
        )
        self.url = f"http://127.0.0.1:{self.port}"
        self._nodes: list[NodeHandle] = []

    def start(self) -> LiveOrchestrator:
        self._thread.start()
        deadline = time.monotonic() + 30.0
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("orchestrator failed to start")
            time.sleep(0.01)
        return self

    def stop(self, *, nodes: bool = True) -> None:
        if nodes:
            for node in self._nodes:
                node.stop_node()
        self._server.should_exit = True
        self._thread.join(timeout=15.0)

    def __enter__(self) -> LiveOrchestrator:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def state(self) -> Any:
        return self.run_dir.load_state()

    def finished(self) -> bool:
        state = self.state()
        return state is not None and state.status == "finished"

    def lines(self) -> list[dict[str, Any]]:
        if not self.run_dir.metrics_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.run_dir.metrics_path.read_text(encoding="utf-8").splitlines()
        ]

    def rounds(self) -> list[dict[str, Any]]:
        return [line for line in self.lines() if line["type"] == "round"]

    def summary(self) -> dict[str, Any]:
        summaries = [line for line in self.lines() if line["type"] == "summary"]
        assert summaries, "no summary line written yet"
        return summaries[-1]

    def client(self, node_id: str) -> OrchestratorClient:
        return OrchestratorClient(self.url, self.cfg.federation.token, self.cfg, node_id=node_id)

    def start_node(self, node_id: str, **run_kwargs: Any) -> NodeHandle:
        client = self.client(node_id)
        stop = threading.Event()
        results: dict[str, str] = {}

        def main() -> None:
            results[node_id] = run_node(
                self.cfg, self.url, client=client, device="cpu", stop_event=stop, **run_kwargs
            )

        thread = threading.Thread(target=main, daemon=True, name=f"scenario-node-{node_id}")
        thread.start()
        handle = NodeHandle(client, stop, thread, results, node_id)
        self._nodes.append(handle)
        return handle


def test_node_joins_mid_run(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "join", federation={"rounds": 4, "min_participants": 1})
    with LiveOrchestrator(cfg, RunDir(root=tmp_path / "runs", run_name="join")) as orch:
        orch.start_node("a")
        until(lambda: len(orch.rounds()) >= 1, what="round 1 line")
        orch.start_node("b")
        until(orch.finished, timeout=120.0, what="run finished")

        rounds = orch.rounds()
        assert len(rounds) == 4 and all(not line["void"] for line in rounds)
        assert [p["node_id"] for p in rounds[0]["participants"]] == ["a"]
        later = [line for line in rounds if any(p["node_id"] == "b" for p in line["participants"])]
        assert later, "node b never contributed after joining mid-run"
        assert orch.state().finished_reason == "max_rounds"


def test_crashed_node_round_voids_then_replacement_joins(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "crash", federation={"rounds": 2, "min_participants": 2})
    with LiveOrchestrator(cfg, RunDir(root=tmp_path / "runs", run_name="crash")) as orch:
        orch.start_node("a")
        crashed = orch.client("b")  # registers, then never heartbeats or submits again
        crashed.register()
        try:
            until(
                lambda: any(line["void"] for line in orch.rounds()),
                what="round 1 voided after b was swept",
            )
            first = orch.rounds()[0]
            assert first["round"] == 1
            assert first["void_reason"] == "insufficient_participants"
            assert [p["node_id"] for p in first["participants"]] == ["a"]

            orch.start_node("b2")  # fresh node replaces the crashed one
            until(orch.finished, timeout=120.0, what="run finished")
        finally:
            crashed.close()

        completed = [line for line in orch.rounds() if not line["void"]]
        assert [line["round"] for line in completed] == [1, 2]
        recovered = completed[0]
        assert sorted(p["node_id"] for p in recovered["participants"]) == ["a", "b2"]
        assert orch.state().finished_reason == "max_rounds"


def test_orchestrator_restart_resumes_run(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "restart", federation={"rounds": 3, "min_participants": 1})
    run_dir = RunDir(root=tmp_path / "runs", run_name="restart")

    first = LiveOrchestrator(cfg, run_dir).start()
    handle = first.start_node("a")
    until(lambda: len(first.rounds()) >= 1, what="round 1 line before restart")
    first.stop(nodes=False)  # the server dies; the node keeps running

    second = LiveOrchestrator(cfg, run_dir, port=first.port).start()  # same URL, same run dir
    try:
        until(
            lambda: handle.results.get("a") == "finished",
            timeout=120.0,
            what="node survived the restart and finished",
        )
        until(second.finished, timeout=120.0, what="run finished after restart")
    finally:
        second.stop()
        handle.stop_node()

    completed = [line for line in second.rounds() if not line["void"]]
    assert [line["round"] for line in completed] == [1, 2, 3]  # recovery, not a fresh run
    assert [[p["node_id"] for p in line["participants"]] for line in completed] == [
        ["a"],
        ["a"],
        ["a"],
    ]  # the same node re-registered after the restart
    state = second.state()
    assert state.finished_reason == "max_rounds" and state.round == 3


def test_early_stop_reaches_target(tmp_path: Path) -> None:
    cfg = make_cfg(
        tmp_path,
        "early",
        federation={"rounds": 10, "min_participants": 1},
        early_stopping={"metric": "val_accuracy", "target": 0.0, "patience": 5},
    )
    with LiveOrchestrator(cfg, RunDir(root=tmp_path / "runs", run_name="early")) as orch:
        handle = orch.start_node("a")
        until(orch.finished, timeout=120.0, what="run finished early")
        # wait for the node's own poll to see RunFinished before we stop it
        until(lambda: handle.results.get("a") == "finished", what="node observed RunFinished")
        assert handle.stop_node() == "finished"  # the node left cleanly, not via stop_event

        rounds = orch.rounds()
        assert len(rounds) == 1 and not rounds[0]["void"]
        summary = orch.summary()
        assert summary["reason"] == "target_reached"
        assert summary["best_round"] == 1
        assert orch.run_dir.best_path.exists()


def test_central_eval_final_only_with_real_images(tmp_path: Path) -> None:
    folder = tmp_path / "central" / "0"
    folder.mkdir(parents=True)
    for i in range(8):
        Image.new("L", (28, 28), color=(i * 7) % 256).save(folder / f"img{i}.png")

    cfg = make_cfg(
        tmp_path,
        "central",
        federation={"rounds": 2, "min_participants": 1},
        evaluation={"central_test": {"path": str(tmp_path / "central"), "mode": "final_only"}},
    )
    with LiveOrchestrator(cfg, RunDir(root=tmp_path / "runs", run_name="central")) as orch:
        orch.start_node("a")
        until(orch.finished, timeout=120.0, what="run with central eval finished")

        rounds = orch.rounds()
        assert len(rounds) == 2 and all("central" not in line for line in rounds)
        summary = orch.summary()
        assert summary["reason"] == "max_rounds"
        central = summary["central_final"]
        assert isinstance(central, dict)
        assert isinstance(central["val_accuracy"], float)
