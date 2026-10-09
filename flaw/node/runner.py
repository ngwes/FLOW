"""Node main loop: register → long-poll → train → submit. Runs in the foreground."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from flaw.blobstore import BlobError
from flaw.config import FlawConfig
from flaw.models import build_model
from flaw.node.client import ClientError, OrchestratorClient
from flaw.node.data import prepare_node_data
from flaw.node.trainer import TrainOutcome, pick_device, train_local
from flaw.protocol import (
    Assignment,
    BlobRef,
    Capabilities,
    HeartbeatResponse,
    RegisterResponse,
    ResultResponse,
    RunFinished,
    serialize_state_dict,
)

log = logging.getLogger("flaw.node")
MAX_BACKOFF_S = 30.0


class NodeClient(Protocol):
    """The client surface the loop needs; :class:`OrchestratorClient` satisfies it."""

    def register(self, capabilities: Capabilities | None = None) -> RegisterResponse: ...

    def get_assignment(self) -> Assignment | RunFinished | None: ...

    def download_blob(self, ref: BlobRef) -> dict[str, torch.Tensor]: ...

    def upload_blob(self, round_no: int, data: bytes) -> BlobRef: ...

    def submit_result(
        self,
        round_no: int,
        update: BlobRef,
        num_samples: int,
        metrics: dict[str, float],
        duration_s: float,
        device: str,
    ) -> ResultResponse: ...

    def heartbeat(self) -> HeartbeatResponse: ...

    def close(self) -> None: ...


def _backoff(attempt: int) -> float:
    return min(MAX_BACKOFF_S, float(2**attempt))  # 1, 2, 4, ... 30


def resolve_node_id(cfg: FlawConfig) -> str:
    return cfg.federation.node_id or socket.gethostname()


def resolve_device(cfg: FlawConfig) -> str | None:
    # gpu may legitimately be 0 — always compare against None, never truthiness
    return f"cuda:{cfg.resources.gpu}" if cfg.resources.gpu is not None else None


def apply_vram_cap(cfg: FlawConfig, device: str) -> None:
    """Soft VRAM cap (CUDA only): allocations past the fraction raise a CUDA OOM."""
    if device.startswith("cuda") and cfg.resources.vram_fraction is not None:
        index = torch.device(device).index
        torch.cuda.set_per_process_memory_fraction(
            cfg.resources.vram_fraction, index if index is not None else 0
        )


def _heartbeat_loop(client: NodeClient, cfg: FlawConfig, stop: threading.Event) -> None:
    while not stop.wait(cfg.federation.heartbeat_s):
        try:
            client.heartbeat()
        except ClientError as exc:
            log.debug("heartbeat failed: %s", exc)


def run_node(
    cfg: FlawConfig,
    server_url: str,
    *,
    client: NodeClient | None = None,
    device: str | None = None,
    prepare_data: Callable[[FlawConfig, int, int], tuple[Dataset[Any], Dataset[Any]]] = (
        prepare_node_data
    ),
    train: Callable[..., TrainOutcome] = train_local,
    sleep: Callable[[float], None] = time.sleep,
    stop_event: threading.Event | None = None,
) -> str:
    """Drive one node through a run; returns ``"finished"``, ``"stopped"`` or ``"fatal_error"``."""
    stop = stop_event if stop_event is not None else threading.Event()
    device, device_name = pick_device(device if device is not None else resolve_device(cfg))
    apply_vram_cap(cfg, device)
    owns_client = client is None
    if client is None:
        client = OrchestratorClient(server_url, cfg.federation.token, cfg, resolve_node_id(cfg))
    heartbeat_thread: threading.Thread | None = None
    try:
        attempt = 0
        while not stop.is_set():
            try:
                registration = client.register(Capabilities(device=device, device_name=device_name))
            except ClientError as exc:
                if exc.fatal:
                    log.error("registration rejected: %s", exc)
                    return "fatal_error"
                sleep(_backoff(attempt))
                attempt += 1
                continue
            break
        else:
            return "stopped"
        shard = registration.shard
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, args=(client, cfg, stop), daemon=True
        )
        heartbeat_thread.start()
        log.info("registered as %s", resolve_node_id(cfg))

        datasets: tuple[Dataset[Any], Dataset[Any]] | None = None
        attempt = 0
        while not stop.is_set():
            try:
                assignment = client.get_assignment()
            except ClientError as exc:
                if exc.fatal:
                    log.error("orchestrator rejected the node: %s", exc)
                    return "fatal_error"
                sleep(_backoff(attempt))
                attempt += 1
                continue
            if assignment is None:
                attempt = 0
                continue
            if isinstance(assignment, RunFinished):
                log.info(
                    "run finished: %s (best round %s, best metric %s)",
                    assignment.reason,
                    assignment.best_round,
                    assignment.best_metric,
                )
                return "finished"
            try:
                if datasets is None:
                    if shard is None and cfg.data.name is not None:
                        log.error("no shard assigned for shared data; cannot select local data")
                        return "fatal_error"
                    datasets = prepare_data(cfg, shard or 0, cfg.run.seed)
                global_sd = client.download_blob(assignment.weights)
                model = build_model(cfg.model)
                model.load_state_dict(global_sd)
                result = train(
                    model,
                    datasets[0],
                    datasets[1],
                    assignment.train_spec,
                    device=device,
                    seed=assignment.train_spec.seed,
                )
                ref = client.upload_blob(assignment.round, serialize_state_dict(result.update))
                response = client.submit_result(
                    assignment.round,
                    ref,
                    result.num_samples,
                    result.metrics,
                    result.duration_s,
                    device,
                )
                attempt = 0
                if response.accepted:
                    log.info("round %d done: %s", assignment.round, result.metrics)
                else:
                    log.info("round %d update rejected: %s", assignment.round, response.reason)
            except ClientError as exc:
                if exc.fatal:
                    log.error("round %d rejected: %s", assignment.round, exc)
                    return "fatal_error"
                log.warning("round %d failed (%s); retrying", assignment.round, exc)
                sleep(_backoff(attempt))
                attempt += 1
            except BlobError as exc:
                log.warning("round %d blob error (%s); retrying", assignment.round, exc)
                sleep(_backoff(attempt))
                attempt += 1
            except Exception:
                log.exception("round %d crashed; retrying", assignment.round)
                sleep(_backoff(attempt))
                attempt += 1
        return "stopped"
    finally:
        stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        if owns_client:
            client.close()
