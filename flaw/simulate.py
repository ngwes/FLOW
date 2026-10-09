"""Run a real orchestrator plus N simulated nodes on one machine (tests, demos, CI)."""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from flaw.config import FlawConfig, config_source_text, load_config
from flaw.node.client import OrchestratorClient
from flaw.node.runner import run_node
from flaw.orchestrator.server import create_app
from flaw.orchestrator.state import RunDir

log = logging.getLogger("flaw.simulate")

SERVER_LONG_POLL_S = 1.0  # short on purpose: bounds shutdown latency of pending /assignment calls


@dataclass
class SimulationResult:
    """What a finished simulation produced (mirrors the run's state.json)."""

    run_dir: Path
    reason: str | None
    rounds: int
    best_metric: float | None
    best_round: int | None


def free_port() -> int:
    """Grab an ephemeral local port (small bind/rebind race, fine for lab-scale use)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def prepare_run_dir(
    cfg: FlawConfig, run_root: str | Path, *, snapshot_text: str | None = None
) -> RunDir:
    """Create the run directory and snapshot the config (shared by simulate and CLI)."""
    run_dir = RunDir(root=Path(run_root), run_name=cfg.run.name)
    run_dir.ensure()
    if snapshot_text is not None:
        run_dir.snapshot_config(snapshot_text)
    return run_dir


def run_simulation(
    config_path: str | Path,
    nodes: int,
    *,
    rounds: int | None = None,
    timeout_s: float = 600.0,
    run_root: str | Path = "runs",
    host: str = "127.0.0.1",
    poll_interval_s: float = 0.2,
) -> SimulationResult:
    """Start an orchestrator + `nodes` simulated nodes locally and wait for the run to finish."""
    if nodes < 1:
        raise ValueError(f"nodes must be >= 1, got {nodes}")
    cfg = load_config(Path(config_path))
    if cfg.data.name is not None and cfg.data.num_shards < nodes:
        raise ValueError(
            f"data.num_shards={cfg.data.num_shards} cannot serve {nodes} nodes; "
            f"set data.num_shards >= {nodes} in {config_path}"
        )
    if rounds is not None:
        cfg.federation.rounds = rounds

    run_dir = prepare_run_dir(cfg, run_root, snapshot_text=config_source_text(Path(config_path)))
    app = create_app(cfg, run_dir, long_poll_s=SERVER_LONG_POLL_S)

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    server_thread = threading.Thread(target=server.run, daemon=True, name="flaw-orchestrator")
    server_thread.start()
    deadline = time.monotonic() + timeout_s
    while not server.started:
        if time.monotonic() > deadline or not server_thread.is_alive():
            raise RuntimeError("orchestrator failed to start")
        time.sleep(0.01)
    base_url = f"http://{host}:{port}"
    log.info("orchestrator on %s, run dir %s", base_url, run_dir.run_dir)

    stop = threading.Event()
    clients = [
        OrchestratorClient(base_url, cfg.federation.token, cfg, node_id=f"sim-{i}")
        for i in range(nodes)
    ]
    results: dict[str, str] = {}

    def node_main(index: int) -> None:
        client = clients[index]
        try:
            results[f"sim-{index}"] = run_node(
                cfg, base_url, client=client, device="cpu", stop_event=stop
            )
        finally:
            client.close()

    threads = [
        threading.Thread(target=node_main, args=(i,), daemon=True, name=f"flaw-node-{i}")
        for i in range(nodes)
    ]
    for thread in threads:
        thread.start()

    try:
        state = None
        while time.monotonic() < deadline:
            state = run_dir.load_state()
            if state is not None and state.status == "finished":
                break
            if all(not thread.is_alive() for thread in threads):
                raise RuntimeError(f"all simulated nodes exited early: {results}")
            time.sleep(poll_interval_s)
        else:
            hint = (
                " (federation.min_participants is higher than the number of simulated nodes)"
                if cfg.federation.min_participants > nodes
                else ""
            )
            raise TimeoutError(
                f"simulation did not finish within {timeout_s}s{hint}; node results: {results}"
            )
    finally:
        stop.set()
        server.should_exit = True
        server_thread.join(timeout=15.0)
        for thread in threads:
            thread.join(timeout=5.0)
        for client in clients:
            client.close()

    assert state is not None  # the wait loop only exits via the finished break
    return SimulationResult(
        run_dir=run_dir.run_dir,
        reason=state.finished_reason,
        rounds=state.round,
        best_metric=state.best_metric,
        best_round=state.best_round,
    )
