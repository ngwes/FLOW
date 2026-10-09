"""Orchestrator HTTP API: registration, heartbeats, blobs, status (spec 5.1).

The :class:`Orchestrator` object owns the run's volatile state (registry, blob
store, run bookkeeping) and is kept separate from the FastAPI wiring in
:func:`create_app`, so the round engine (Task 14) can extend the class without
touching the endpoints.  Every rejection is a JSON body ``{"reason": "<text>"}``
built by :func:`_error`, with status 400 (malformed), 403 (token/version),
404 (unknown node/blob) or 409 (conflict).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from flaw.blobstore import BlobNotFound, BlobStore, node_blob_id
from flaw.config import FlawConfig
from flaw.orchestrator.registry import (
    NodeConflict,
    NodeRegistry,
    NoShardAvailable,
    UnknownNode,
)
from flaw.orchestrator.state import RunDir, RunState
from flaw.protocol import (
    BlobUploadResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    NodeStatus,
    ProtocolError,
    RegisterRequest,
    RegisterResponse,
    RunInfo,
    StatusResponse,
    check_protocol_version,
    compute_config_hash,
)


def _error(status: int, reason: str) -> JSONResponse:
    """Build the uniform rejection body ``{"reason": reason}`` with *status*."""
    return JSONResponse(status_code=status, content={"reason": reason})


class Orchestrator:
    """Server-side run state and services shared by all endpoints."""

    def __init__(
        self,
        cfg: FlawConfig,
        run_dir: RunDir,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        long_poll_s: float = 30.0,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.clock = clock
        self.wall_clock = wall_clock
        self.long_poll_s = long_poll_s
        self.lock = asyncio.Lock()
        self.blobs = BlobStore(root=run_dir.blobs_dir)
        self.registry = NodeRegistry(cfg.federation.liveness_timeout_s, clock)
        self.state = run_dir.load_state() or RunState(run_name=cfg.run.name)
        self.expected_config_hash = compute_config_hash(cfg.canonical_hash_payload())
        self.registry.load_shards(self.state.shard_map)


def create_app(
    cfg: FlawConfig,
    run_dir: RunDir,
    *,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    long_poll_s: float = 30.0,
) -> FastAPI:
    """Build the orchestrator app; its :class:`Orchestrator` is at ``app.state``."""
    orch = Orchestrator(cfg, run_dir, clock=clock, wall_clock=wall_clock, long_poll_s=long_poll_s)
    app = FastAPI()
    app.state.orchestrator = orch

    @app.post("/register", response_model=RegisterResponse)
    def register(body: RegisterRequest) -> RegisterResponse | JSONResponse:
        if body.token != orch.cfg.federation.token:
            return _error(403, "invalid token")
        try:
            check_protocol_version(body.protocol_version)
        except ProtocolError as exc:
            return _error(403, str(exc))
        if body.config_hash != orch.expected_config_hash:
            return _error(409, "config mismatch: node config hash does not match the run")
        try:
            orch.registry.register(body.node_id, body.capabilities)
        except NodeConflict:
            return _error(409, f"node id {body.node_id!r} is in use")
        if orch.cfg.data.name is not None:
            try:
                orch.registry.assign_shard(body.node_id, orch.cfg.data.num_shards)
            except NoShardAvailable as exc:
                return _error(409, f"no shards available: {exc}")
        orch.state.shard_map = orch.registry.shard_map()
        orch.run_dir.save_state(orch.state)
        return RegisterResponse(
            accepted=True,
            run=RunInfo(
                name=orch.cfg.run.name,
                current_round=orch.state.round,
                phase=orch.state.phase,
            ),
            shard=orch.registry.get(body.node_id).shard,
        )

    @app.post("/heartbeat", response_model=HeartbeatResponse)
    def heartbeat(body: HeartbeatRequest) -> HeartbeatResponse | JSONResponse:
        try:
            orch.registry.heartbeat(body.node_id)
        except UnknownNode:
            return _error(404, f"unknown node: {body.node_id!r}")
        return HeartbeatResponse(phase=orch.state.phase, round=orch.state.round)

    @app.get("/blob/{blob_id:path}")
    def download_blob(blob_id: str) -> Response:
        try:
            data = orch.blobs.get(blob_id)
        except ProtocolError as exc:
            return _error(400, str(exc))
        except BlobNotFound:
            return _error(404, f"blob not found: {blob_id!r}")
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"X-Blob-Sha256": hashlib.sha256(data).hexdigest()},
        )

    @app.post("/blob", response_model=BlobUploadResponse)
    async def upload_blob(
        request: Request, node_id: str, round: int
    ) -> BlobUploadResponse | JSONResponse:
        if request.headers.get("X-Flaw-Token") != orch.cfg.federation.token:
            return _error(403, "invalid token")
        try:
            orch.registry.get(node_id)
        except UnknownNode:
            return _error(404, f"unknown node: {node_id!r}")
        if round < 1:
            return _error(400, f"round must be >= 1, got {round}")
        try:
            ref = orch.blobs.put(node_blob_id(round, node_id), await request.body())
        except ProtocolError as exc:
            return _error(400, str(exc))
        return BlobUploadResponse(blob_id=ref.blob_id, sha256=ref.sha256, size=ref.size)

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        records = [*orch.registry.alive_nodes(), *orch.registry.dead_nodes()]
        return StatusResponse(
            name=orch.cfg.run.name,
            phase=orch.state.phase,
            round=orch.state.round,
            nodes=[
                NodeStatus(
                    node_id=record.node_id,
                    alive=record.alive,
                    shard=record.shard,
                    device=record.capabilities.device,
                    last_heartbeat_age_s=orch.clock() - record.last_seen,
                )
                for record in records
            ],
            best_metric=orch.state.best_metric,
            best_round=orch.state.best_round,
        )

    return app
