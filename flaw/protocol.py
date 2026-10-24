"""FL@W wire protocol: pydantic schemas, blob ids, config hashing.

Everything that crosses the network between orchestrator and nodes is defined
here, so both sides serialize exactly the same shapes.  ``ProtocolError``
subclasses :class:`Exception` directly on purpose: pydantic v2 wraps
``ValueError`` raised inside validators into ``ValidationError``, and every
call site matches on ``ProtocolError``.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from flaw.config import OptimizerSpec

PROTOCOL_VERSION = 1

BLOB_ID_RE = re.compile(r"^r\d{4}/(global|node-[A-Za-z0-9._-]{1,64})$")


class ProtocolError(Exception):
    """Raised when a message or identifier violates the wire protocol."""


def validate_blob_id(blob_id: str) -> str:
    """Return *blob_id* unchanged if valid, else raise :class:`ProtocolError`.

    Valid ids are ``r<4 digits>/global`` or ``r<4 digits>/node-<name>``.
    """
    if not BLOB_ID_RE.fullmatch(blob_id):
        raise ProtocolError(f"invalid blob id: {blob_id!r}")
    return blob_id


def check_protocol_version(version: int) -> None:
    """Accept *version* or raise :class:`ProtocolError` naming the field."""
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol_version {version} not supported (expected {PROTOCOL_VERSION})"
        )


def compute_config_hash(payload: dict[str, object]) -> str:
    """sha256 hex digest of the canonical (sorted, compact) JSON of *payload*."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Capabilities(BaseModel):
    """What a node can offer: device type and a human-readable device name."""

    model_config = ConfigDict(extra="forbid")

    device: str
    device_name: str


class BlobRef(BaseModel):
    """Pointer to an immutable blob in the store (weights or updates)."""

    model_config = ConfigDict(extra="forbid")

    blob_id: str
    sha256: str
    size: int
    format: str = "torch-state-dict-v1"
    dtype: str = "fp32"

    @field_validator("blob_id")
    @classmethod
    def _check_blob_id(cls, v: str) -> str:
        return validate_blob_id(v)

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, v: str) -> str:
        if len(v) != 64:
            raise ProtocolError(f"sha256 must be 64 hex characters, got {len(v)}")
        return v


class TrainSpec(BaseModel):
    """Hyperparameters for one round of local training on a node."""

    model_config = ConfigDict(extra="forbid")

    local_epochs: int
    batch_size: int
    optimizer: OptimizerSpec
    seed: int


class RunInfo(BaseModel):
    """Identity and progress of the run, as seen by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    current_round: int
    phase: str


class RegisterRequest(BaseModel):
    """Node -> orchestrator: join a run, presenting identity and capabilities."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: int
    token: str
    node_id: str
    capabilities: Capabilities
    config_hash: str


class RegisterResponse(BaseModel):
    """Orchestrator -> node: registration verdict.

    ``shard`` is the data shard the orchestrator assigned; nodes use it to
    select their local slice of the dataset.
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: str | None = None
    run: RunInfo | None = None
    shard: int | None = None


class Assignment(BaseModel):
    """Orchestrator -> node: the round to train and the weights to start from."""

    model_config = ConfigDict(extra="forbid")

    round: int
    train_spec: TrainSpec
    weights: BlobRef


class RunFinished(BaseModel):
    """Orchestrator -> node: the run is over; stop asking for work."""

    model_config = ConfigDict(extra="forbid")

    run_finished: bool = True
    reason: str
    best_round: int | None = None
    best_metric: float | None = None


class ResultRequest(BaseModel):
    """Node -> orchestrator: a finished round's update plus its metrics."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    round: int
    update: BlobRef
    num_samples: int
    metrics: dict[str, float]
    duration_s: float
    device: str


class ResultResponse(BaseModel):
    """Orchestrator -> node: whether the submitted update was accepted."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: str | None = None


class HeartbeatRequest(BaseModel):
    """Node -> orchestrator: liveness ping."""

    model_config = ConfigDict(extra="forbid")

    node_id: str


class HeartbeatResponse(BaseModel):
    """Orchestrator -> node: current phase and round, for display and control flow."""

    model_config = ConfigDict(extra="forbid")

    phase: str
    round: int


class BlobUploadResponse(BaseModel):
    """Orchestrator -> node: server-computed identity of a stored blob."""

    model_config = ConfigDict(extra="forbid")

    blob_id: str
    sha256: str
    size: int


class NodeStatus(BaseModel):
    """One node's row in the status snapshot."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    alive: bool
    shard: int | None
    device: str
    last_heartbeat_age_s: float


class StatusResponse(BaseModel):
    """Orchestrator -> whoever asks: the full run status snapshot."""

    model_config = ConfigDict(extra="forbid")

    name: str
    phase: str
    round: int
    nodes: list[NodeStatus]
    best_metric: float | None
    best_round: int | None


class ErrorResponse(BaseModel):
    """Any server error surfaced to a client."""

    model_config = ConfigDict(extra="forbid")

    reason: str


def serialize_state_dict(sd: Mapping[str, torch.Tensor]) -> bytes:
    """Serialize a state dict to bytes, detaching tensors onto the CPU."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in sd.items()}, buf)
    return buf.getvalue()


def deserialize_state_dict(data: bytes) -> dict[str, torch.Tensor]:
    """Load a :func:`serialize_state_dict` payload back onto the CPU."""
    return cast(
        "dict[str, torch.Tensor]",
        torch.load(io.BytesIO(data), map_location="cpu", weights_only=True),
    )
