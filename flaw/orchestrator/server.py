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
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from torch.utils.data import Dataset

from flaw.blobstore import BlobNotFound, BlobStore, ChecksumMismatch, global_blob_id, node_blob_id
from flaw.config import FlawConfig
from flaw.config import OptimizerSpec as WireOptimizerSpec
from flaw.models import build_model
from flaw.orchestrator.aggregation import AggregationError, aggregate_weighted
from flaw.orchestrator.centraleval import default_central_factory, run_central_eval
from flaw.orchestrator.metrics import (
    EarlyStopper,
    MissingMetric,
    aggregate_metrics,
    write_metrics_line,
)
from flaw.orchestrator.registry import (
    NodeConflict,
    NodeRegistry,
    NoShardAvailable,
    UnknownNode,
)
from flaw.orchestrator.rounds import (
    ParticipantResult,
    RoundManager,
    RoundPhase,
    RoundResult,
    SubmitOutcome,
)
from flaw.orchestrator.state import RunDir, RunState
from flaw.protocol import (
    Assignment,
    BlobRef,
    BlobUploadResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    NodeStatus,
    ProtocolError,
    RegisterRequest,
    RegisterResponse,
    ResultRequest,
    ResultResponse,
    RunFinished,
    RunInfo,
    StatusResponse,
    TrainSpec,
    check_protocol_version,
    compute_config_hash,
    deserialize_state_dict,
    serialize_state_dict,
)

log = logging.getLogger("flaw.orchestrator")


def _error(status: int, reason: str) -> JSONResponse:
    """Build the uniform rejection body ``{"reason": reason}`` with *status*."""
    return JSONResponse(status_code=status, content={"reason": reason})


def validate_update(
    update: Mapping[str, torch.Tensor], template: Mapping[str, torch.Tensor]
) -> str | None:
    """Return a rejection reason, or None when the update matches the template exactly."""
    for key, ref in template.items():
        if key not in update:
            return "missing_key"
        value = update[key]
        if tuple(value.shape) != tuple(ref.shape) or not torch.is_floating_point(value):
            return "shape_mismatch"
        if not bool(torch.isfinite(value).all()):
            return "non_finite"
    if set(update) - set(template):
        return "extra_key"
    return None


class Orchestrator:
    """Server-side run state and services shared by all endpoints."""

    def __init__(
        self,
        cfg: FlawConfig,
        run_dir: RunDir,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        long_poll_s: float = 30.0,
        central_dataset_factory: Callable[[], Dataset[Any]] | None = None,
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
        self.rounds = RoundManager(
            cfg.federation.min_participants, cfg.federation.round_deadline_s, clock
        )
        self.stopper = EarlyStopper(
            cfg.early_stopping.metric,
            cfg.early_stopping.target,
            cfg.early_stopping.patience,
            cfg.early_stopping.min_delta,
            cfg.federation.rounds,
        )
        mode = cfg.evaluation.central_test.mode
        self.every_round_eval = mode == "every_round"
        self.final_eval = mode == "final_only"
        self.central_factory = (
            None if mode == "off" else (central_dataset_factory or default_central_factory(cfg))
        )
        self.finished: RunFinished | None = None
        self._template: dict[str, torch.Tensor] | None = None
        self._loop_task: asyncio.Task[None] | None = None

    def model_template(self) -> dict[str, torch.Tensor]:
        """Immutable reference state_dict used for validation and aggregation."""
        if self._template is None:
            model = build_model(self.cfg.model)
            self._template = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return self._template

    def _global_rounds(self) -> list[int]:
        out: list[int] = []
        for d in self.run_dir.blobs_dir.glob("r????"):
            if (d / "global.pt").exists():
                out.append(int(d.name[1:]))
        return sorted(out)

    def recover(self) -> None:
        """Restore the round cursor from committed blobs (at-least-once rounds)."""
        if self.state.status == "finished":
            # a restart of a completed run must keep answering /assignment with
            # RunFinished instead of long-polling forever for work that never comes
            self.state.phase = "finished"
            self.finished = RunFinished(
                reason=self.state.finished_reason or "finished",
                best_round=self.state.best_round,
                best_metric=self.state.best_metric,
            )
        else:
            existing = self._global_rounds()
            self.state.round = max(existing) - 1 if existing else 0
            self.state.phase = "idle"
            if not self.blobs.exists(global_blob_id(self.state.round + 1)):
                torch.manual_seed(self.cfg.run.seed)  # orchestrator-level RNG: startup only
                model = build_model(self.cfg.model)
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.blobs.put(global_blob_id(self.state.round + 1), serialize_state_dict(sd))
        self.run_dir.save_state(self.state)

    def _ref_for(self, blob_id: str) -> BlobRef:
        data = self.blobs.get(blob_id)
        return BlobRef(blob_id=blob_id, sha256=hashlib.sha256(data).hexdigest(), size=len(data))

    def try_assignment(self, node_id: str) -> Assignment | RunFinished | None:
        """None -> the node should keep waiting; call under `self.lock`."""
        if self.finished is not None:
            return self.finished
        if self.rounds.phase is not RoundPhase.OPEN:
            return None
        if node_id not in self.rounds.expected() or node_id in self.rounds.submitted_nodes():
            return None
        if not self.registry.is_alive(node_id):
            return None
        o = self.cfg.training.optimizer
        round_no = self.rounds.current_round
        return Assignment(
            round=round_no,
            train_spec=TrainSpec(
                local_epochs=self.cfg.training.local_epochs,
                batch_size=self.cfg.training.batch_size,
                optimizer=WireOptimizerSpec(name=o.name, lr=o.lr, momentum=o.momentum),
                seed=self.cfg.run.seed,
            ),
            weights=self._ref_for(global_blob_id(round_no)),
        )

    def start(self) -> None:
        self._loop_task = asyncio.create_task(self._round_loop())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def _round_loop(self) -> None:
        while True:
            try:
                async with self.lock:
                    if self.state.status == "finished":
                        return
                    self.registry.sweep()
                    alive = [r.node_id for r in self.registry.alive_nodes()]
                if len(alive) < self.cfg.federation.min_participants:
                    await asyncio.sleep(0.05)
                    continue
                round_no = self.state.round + 1
                async with self.lock:
                    self.rounds.open(round_no, alive)
                    self.state.round = round_no
                    self.state.phase = "open"
                    self.run_dir.save_state(self.state)
                    opened_at = self.wall_clock()
                closed: str | None = None
                while closed is None:
                    await asyncio.sleep(0.02)
                    async with self.lock:
                        self.registry.sweep()
                        alive_expected = {
                            r.node_id
                            for r in self.registry.alive_nodes()
                            if r.node_id in self.rounds.expected()
                        }
                        closed = self.rounds.close_reason(alive_expected)
                result = self.rounds.close(closed)
                async with self.lock:
                    self._process_round_result(result, opened_at)
            except Exception:
                # one bad iteration must never wedge the run; CancelledError is a
                # BaseException, so a stop() still propagates and ends the task
                log.exception("round loop iteration failed; retrying")
                await asyncio.sleep(0.05)

    def _process_round_result(self, result: RoundResult, opened_at: float) -> None:
        participants = [
            {
                "node_id": p.node_id,
                "num_samples": p.num_samples,
                "metrics": dict(p.metrics),
                "duration_s": p.duration_s,
                "device": p.device,
                "update_blob_id": p.update_blob.blob_id,
            }
            for p in result.participants
        ]
        line: dict[str, object] = {
            "type": "round",
            "round": result.round,
            "void": result.void,
            "void_reason": result.void_reason,
            "closed_reason": result.closed_reason,
            "participants": participants,
            "rejected": [],
            "duration_s": self.wall_clock() - opened_at,
        }

        def void(reason_value: str) -> None:
            line.update(void=True, void_reason=reason_value, closed_reason=None)
            write_metrics_line(self.run_dir.metrics_path, line)
            self.state.round = result.round - 1  # the same round number opens again
            self.state.phase = "idle"
            self.run_dir.save_state(self.state)

        if result.void:
            void(result.void_reason or "insufficient_participants")
            return

        template = self.model_template()
        updates: list[tuple[Mapping[str, torch.Tensor], int]] = []
        accepted: list[ParticipantResult] = []
        rejected: list[dict[str, str]] = []
        for p in result.participants:
            if p.num_samples <= 0:
                # zero/negative sample weights would divide by zero in
                # aggregate_metrics or skew the mean; never aggregate them.
                rejected.append({"node_id": p.node_id, "reason": "invalid_samples"})
                continue
            try:
                sd = deserialize_state_dict(self.blobs.get(p.update_blob.blob_id))
                reason = validate_update(sd, template)
            except Exception:  # untrusted bytes: decode/validation crashes reject too
                rejected.append({"node_id": p.node_id, "reason": "unreadable"})
                continue
            if reason is not None:
                rejected.append({"node_id": p.node_id, "reason": reason})
                continue
            updates.append((sd, p.num_samples))
            accepted.append(p)
        line["rejected"] = rejected

        if len(updates) < self.cfg.federation.min_participants:
            void("insufficient_valid_updates")
            return
        try:
            aggregated = aggregate_weighted(updates, template)
        except AggregationError:
            void("aggregation_failed")
            return

        ref = self.blobs.put(global_blob_id(result.round + 1), serialize_state_dict(aggregated))
        line["global_blob_id"] = ref.blob_id
        line["aggregated"] = aggregate_metrics([(p.metrics, p.num_samples) for p in accepted])

        if self.every_round_eval and self.central_factory is not None:
            try:
                line["central"] = run_central_eval(self.cfg, aggregated, self.central_factory)
            except Exception as exc:  # a broken central set must not kill the run
                line["central_error"] = str(exc)

        decision = None
        try:
            decision = self.stopper.update(result.round, line["aggregated"])  # type: ignore[arg-type]
        except MissingMetric:
            line["early_stop_metric_missing"] = True
        if self.stopper.best_round == result.round:
            self.run_dir.save_best(aggregated)

        self.state.round = result.round
        self.state.phase = "idle"
        self.state.best_metric = self.stopper.best_metric
        self.state.best_round = self.stopper.best_round
        write_metrics_line(self.run_dir.metrics_path, line)
        self.blobs.gc(
            current_round=result.round, keep_rounds=self.cfg.storage.retention.keep_rounds
        )

        stop_reason = decision.reason if (decision is not None and decision.stop) else None
        if stop_reason is None and result.round >= self.cfg.federation.rounds:
            stop_reason = "max_rounds"
        if stop_reason is not None:
            self._finish(stop_reason, aggregated)
        else:
            self.run_dir.save_state(self.state)

    def _finish(self, reason: str, final_sd: Mapping[str, torch.Tensor]) -> None:
        self.state.status = "finished"
        self.state.finished_reason = reason
        self.state.phase = "finished"
        self.run_dir.save_final(final_sd)
        summary: dict[str, object] = {
            "type": "summary",
            "reason": reason,
            "best_round": self.stopper.best_round,
            "best_metric": self.stopper.best_metric,
        }
        if self.final_eval and self.central_factory is not None:
            try:
                if self.run_dir.best_path.exists():
                    best_sd = torch.load(self.run_dir.best_path, weights_only=True)
                    summary["central_best"] = run_central_eval(
                        self.cfg, best_sd, self.central_factory
                    )
                summary["central_final"] = run_central_eval(
                    self.cfg, final_sd, self.central_factory
                )
            except Exception as exc:  # never let evaluation kill a finished run
                summary["central_error"] = str(exc)
        write_metrics_line(self.run_dir.metrics_path, summary)
        self.run_dir.save_state(self.state)
        self.finished = RunFinished(
            reason=reason, best_round=self.stopper.best_round, best_metric=self.stopper.best_metric
        )
        self.rounds.finish()


def create_app(
    cfg: FlawConfig,
    run_dir: RunDir,
    *,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    long_poll_s: float = 30.0,
    start_loop: bool = True,
    central_dataset_factory: Callable[[], Dataset[Any]] | None = None,
) -> FastAPI:
    """Build the orchestrator app; its :class:`Orchestrator` is at ``app.state``."""
    orch = Orchestrator(
        cfg,
        run_dir,
        clock=clock,
        wall_clock=wall_clock,
        long_poll_s=long_poll_s,
        central_dataset_factory=central_dataset_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start_loop:
            orch.recover()
            orch.start()
        try:
            yield
        finally:
            if start_loop:
                await orch.stop()

    app = FastAPI(title=f"FL@W orchestrator — {cfg.run.name}", lifespan=lifespan)
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

    @app.get("/assignment")
    async def assignment(node_id: str) -> Response:
        try:
            orch.registry.get(node_id)
        except UnknownNode:
            return _error(404, "unknown node")
        deadline = orch.clock() + orch.long_poll_s
        while True:
            async with orch.lock:
                resp = orch.try_assignment(node_id)
            if resp is not None:
                return JSONResponse(content=resp.model_dump(mode="json"))
            if orch.clock() >= deadline:
                return Response(status_code=204)
            await asyncio.sleep(0.05)

    @app.post("/result")
    async def result(body: ResultRequest) -> Response:
        try:
            orch.registry.get(body.node_id)
        except UnknownNode:
            return _error(404, "unknown node")

        def rejected(reason: str) -> JSONResponse:
            return JSONResponse(content=ResultResponse(accepted=False, reason=reason).model_dump())

        expected_id = node_blob_id(body.round, body.node_id)
        if body.update.blob_id != expected_id:
            return rejected("invalid")
        try:
            data = orch.blobs.get(expected_id)
            orch.blobs.verify(body.update, data)
        except (BlobNotFound, ChecksumMismatch):
            return rejected("invalid")

        participant = ParticipantResult(
            node_id=body.node_id,
            num_samples=body.num_samples,
            metrics=dict(body.metrics),
            duration_s=body.duration_s,
            device=body.device,
            update_blob=body.update,
        )
        async with orch.lock:
            outcome = orch.rounds.submit(body.node_id, body.round, participant)
        if outcome is SubmitOutcome.ACCEPTED:
            return JSONResponse(content=ResultResponse(accepted=True).model_dump())
        reasons = {
            SubmitOutcome.DUPLICATE: "duplicate",
            SubmitOutcome.STALE: "stale",
            SubmitOutcome.UNKNOWN_ROUND: "unknown_round",
        }
        return rejected(reasons[outcome])

    return app
