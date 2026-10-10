"""Node-side HTTP client for the orchestrator API (spec 5.1).

One :class:`OrchestratorClient` per training node speaks the wire protocol:
register, long-poll for assignments, move weight blobs (verifying sha256 on
download) and report results.  Every failure surfaces as a :class:`ClientError`
with an explicit ``fatal`` flag; nothing here retries -- the node runner owns
backoff.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, TypeVar

import httpx
import torch
from pydantic import BaseModel

from flaw.blobstore import ChecksumMismatch
from flaw.config import FlawConfig
from flaw.node.trainer import pick_device
from flaw.protocol import (
    PROTOCOL_VERSION,
    Assignment,
    BlobRef,
    Capabilities,
    HeartbeatResponse,
    RegisterResponse,
    ResultResponse,
    RunFinished,
    compute_config_hash,
    deserialize_state_dict,
)

# Retrying cannot fix these; everything else (5xx, transport) is transient.
_FATAL_STATUSES = {400, 403, 404, 409}

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ClientError(Exception):
    """A request to the orchestrator failed.

    ``status_code`` is 0 for transport errors, else the HTTP status code;
    ``reason`` is the server's ``{"reason": ...}`` text (or the raw body);
    ``fatal`` is True when a retry cannot help (400/403/404/409) -- callers may
    still override that policy for :attr:`unknown_node`, which re-registration fixes.
    """

    def __init__(self, status_code: int, reason: str, fatal: bool) -> None:
        super().__init__(f"HTTP {status_code}: {reason}" if status_code else reason)
        self.status_code = status_code
        self.reason = reason
        self.fatal = fatal

    @property
    def unknown_node(self) -> bool:
        """True for the orchestrator's 404 "unknown node" (after a restart: re-register)."""
        return self.status_code == 404 and self.reason.startswith("unknown node")


class OrchestratorClient:
    """Synchronous, thread-safe HTTP client for one node's orchestrator session."""

    def __init__(
        self,
        base_url: str,
        token: str,
        cfg: FlawConfig,
        node_id: str,
        timeout_s: float = 10.0,
        long_poll_s: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.cfg = cfg
        self.node_id = node_id
        self.timeout_s = timeout_s
        self.long_poll_s = long_poll_s
        self._http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._http.close()

    def register(self, capabilities: Capabilities | None = None) -> RegisterResponse:
        """Join the run, presenting identity, capabilities and config hash.

        When *capabilities* is None the device is auto-detected via
        :func:`flaw.node.trainer.pick_device`.
        """
        if capabilities is None:
            device, device_name = pick_device(None)
            capabilities = Capabilities(device=device, device_name=device_name)
        payload: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "token": self.token,
            "node_id": self.node_id,
            "capabilities": capabilities.model_dump(),
            "config_hash": compute_config_hash(self.cfg.canonical_hash_payload()),
        }
        return self._post_model("/register", payload, RegisterResponse)

    def get_assignment(self) -> Assignment | RunFinished | None:
        """Long-poll for work: 204 -> None, a ``run_finished`` body ends the run."""
        r = self._request(
            "GET",
            "/assignment",
            params={"node_id": self.node_id},
            timeout=self.long_poll_s + 10.0,
        )
        if r.status_code == 204:
            return None
        self._check(r)
        body = r.json()
        if body.get("run_finished"):
            return RunFinished.model_validate(body)
        return Assignment.model_validate(body)

    def download_blob(self, ref: BlobRef) -> dict[str, torch.Tensor]:
        """Fetch and deserialize a blob, refusing bytes that fail *ref*'s sha256."""
        r = self._request("GET", f"/blob/{ref.blob_id}", timeout=self.timeout_s)
        self._check(r)
        data = r.content
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ChecksumMismatch(f"blob {ref.blob_id} failed sha256 verification")
        return deserialize_state_dict(data)

    def upload_blob(self, round_no: int, data: bytes) -> BlobRef:
        """Upload raw update bytes for *round_no*; return the stored blob's ref."""
        r = self._request(
            "POST",
            "/blob",
            params={"node_id": self.node_id, "round": round_no},
            content=data,
            headers={"X-Flaw-Token": self.token},
        )
        self._check(r)
        body = r.json()
        return BlobRef(blob_id=body["blob_id"], sha256=body["sha256"], size=body["size"])

    def submit_result(
        self,
        round_no: int,
        update: BlobRef,
        num_samples: int,
        metrics: dict[str, float],
        duration_s: float,
        device: str,
    ) -> ResultResponse:
        """Report a finished round's update blob and metrics."""
        payload: dict[str, object] = {
            "node_id": self.node_id,
            "round": round_no,
            "update": update.model_dump(),
            "num_samples": num_samples,
            "metrics": metrics,
            "duration_s": duration_s,
            "device": device,
        }
        return self._post_model("/result", payload, ResultResponse)

    def heartbeat(self) -> HeartbeatResponse:
        """Ping the orchestrator to stay registered as alive."""
        return self._post_model("/heartbeat", {"node_id": self.node_id}, HeartbeatResponse)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, self.base_url + path, **kwargs)
        except httpx.TransportError as exc:
            raise ClientError(0, f"transport error: {exc}", fatal=False) from exc

    def _check(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            reason = str(response.json().get("reason", response.text))
        except ValueError:
            reason = response.text
        raise ClientError(
            response.status_code, reason, fatal=response.status_code in _FATAL_STATUSES
        )

    def _post_model(
        self, path: str, payload: Mapping[str, object], model_cls: type[_ModelT]
    ) -> _ModelT:
        r = self._request("POST", path, json=payload)
        self._check(r)
        return model_cls.model_validate(r.json())
