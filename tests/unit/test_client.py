import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx
import torch

from flaw.blobstore import ChecksumMismatch
from flaw.config import load_config
from flaw.node.client import ClientError, OrchestratorClient
from flaw.protocol import (
    PROTOCOL_VERSION,
    BlobRef,
    Capabilities,
    compute_config_hash,
    serialize_state_dict,
)

BASE_URL = "http://orchestrator.test:8000"

CFG = """
run: {name: exp1}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 16}
federation: {token: secret}
"""


@pytest.fixture
def cfg(tmp_path: Path):  # type: ignore[no-untyped-def]
    p = tmp_path / "flaw.yaml"
    p.write_text(CFG, encoding="utf-8")
    return load_config(p)


@pytest.fixture
def client(cfg):  # type: ignore[no-untyped-def]
    c = OrchestratorClient(BASE_URL, token="secret", cfg=cfg, node_id="lab-1", long_poll_s=0.05)
    yield c
    c.close()


def blob_ref(data: bytes) -> BlobRef:
    return BlobRef(blob_id="r0001/global", sha256=hashlib.sha256(data).hexdigest(), size=len(data))


@respx.mock
def test_register_payload_and_response(client, cfg) -> None:  # type: ignore[no-untyped-def]
    route = respx.post(f"{BASE_URL}/register").mock(
        return_value=httpx.Response(
            200,
            json={
                "accepted": True,
                "run": {"name": "exp1", "current_round": 0, "phase": "idle"},
            },
        )
    )
    resp = client.register()
    assert resp.accepted is True and resp.run.name == "exp1"
    sent = json.loads(route.calls[0].request.content)
    assert sent["protocol_version"] == PROTOCOL_VERSION
    assert sent["token"] == "secret" and sent["node_id"] == "lab-1"
    assert sent["config_hash"] == compute_config_hash(cfg.canonical_hash_payload())
    assert sent["capabilities"]["device"]  # auto-picked (cpu on CI)


@respx.mock
def test_register_explicit_capabilities(client) -> None:  # type: ignore[no-untyped-def]
    route = respx.post(f"{BASE_URL}/register").mock(
        return_value=httpx.Response(200, json={"accepted": True})
    )
    capabilities = Capabilities(device="cuda:0", device_name="Fake GPU 4090")
    client.register(capabilities=capabilities)
    sent = json.loads(route.calls[0].request.content)
    assert sent["capabilities"] == {"device": "cuda:0", "device_name": "Fake GPU 4090"}


@respx.mock
def test_register_conflict_is_fatal(client) -> None:
    respx.post(f"{BASE_URL}/register").mock(
        return_value=httpx.Response(409, json={"reason": "node id in use"})
    )
    with pytest.raises(ClientError) as excinfo:
        client.register()
    assert excinfo.value.fatal is True and excinfo.value.status_code == 409
    assert "in use" in excinfo.value.reason
    assert str(excinfo.value) == "HTTP 409: node id in use"


@respx.mock
def test_server_error_without_json_body(client) -> None:
    respx.post(f"{BASE_URL}/register").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(ClientError) as excinfo:
        client.register()
    assert excinfo.value.status_code == 500 and excinfo.value.fatal is False
    assert "boom" in excinfo.value.reason


@respx.mock
def test_get_assignment_none_and_finished(client) -> None:
    payload = {
        "round": 3,
        "train_spec": {
            "local_epochs": 1,
            "batch_size": 16,
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
            "seed": 42,
        },
        "weights": {"blob_id": "r0003/global", "sha256": "a" * 64, "size": 4},
    }
    route = respx.get(f"{BASE_URL}/assignment").mock(return_value=httpx.Response(200, json=payload))
    assignment = client.get_assignment()
    assert assignment.round == 3
    assert assignment.train_spec.seed == 42
    assert assignment.weights.blob_id == "r0003/global"
    assert route.calls[0].request.url.params["node_id"] == "lab-1"

    respx.get(f"{BASE_URL}/assignment").mock(return_value=httpx.Response(204))
    assert client.get_assignment() is None

    respx.get(f"{BASE_URL}/assignment").mock(
        return_value=httpx.Response(
            200,
            json={
                "run_finished": True,
                "reason": "max_rounds",
                "best_round": 2,
                "best_metric": 0.9,
            },
        )
    )
    finished = client.get_assignment()
    assert finished.run_finished is True and finished.reason == "max_rounds"


@respx.mock
def test_download_blob_verifies_checksum(client) -> None:
    data = serialize_state_dict({"w": torch.tensor([1.0, 2.0])})
    respx.get(f"{BASE_URL}/blob/r0001/global").mock(return_value=httpx.Response(200, content=data))
    sd = client.download_blob(blob_ref(data))
    assert torch.equal(sd["w"], torch.tensor([1.0, 2.0]))

    bad = BlobRef(blob_id="r0001/global", sha256="0" * 64, size=len(data))
    with pytest.raises(ChecksumMismatch):
        client.download_blob(bad)


@respx.mock
def test_upload_blob(client) -> None:
    data = b"weights"
    route = respx.post(f"{BASE_URL}/blob").mock(
        return_value=httpx.Response(
            200,
            json={
                "blob_id": "r0002/node-lab-1",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            },
        )
    )
    ref = client.upload_blob(2, data)
    assert ref.blob_id == "r0002/node-lab-1"
    request = route.calls[0].request
    assert request.url.params["node_id"] == "lab-1" and request.url.params["round"] == "2"
    assert request.headers["X-Flaw-Token"] == "secret"
    assert request.content == data


@respx.mock
def test_submit_result(client) -> None:
    route = respx.post(f"{BASE_URL}/result").mock(
        return_value=httpx.Response(200, json={"accepted": True, "reason": None})
    )
    resp = client.submit_result(
        1, blob_ref(b"x"), num_samples=32, metrics={"loss": 0.5}, duration_s=1.5, device="cpu"
    )
    assert resp.accepted is True
    sent = json.loads(route.calls[0].request.content)
    assert sent["node_id"] == "lab-1" and sent["round"] == 1
    assert sent["update"]["blob_id"] == "r0001/global"
    assert sent["num_samples"] == 32 and sent["device"] == "cpu"


@respx.mock
def test_heartbeat(client) -> None:
    respx.post(f"{BASE_URL}/heartbeat").mock(
        return_value=httpx.Response(200, json={"phase": "open", "round": 4})
    )
    resp = client.heartbeat()
    assert resp.phase == "open" and resp.round == 4


@respx.mock
def test_transport_error_is_not_fatal(client) -> None:
    respx.get(f"{BASE_URL}/assignment").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(ClientError) as excinfo:
        client.get_assignment()
    assert excinfo.value.fatal is False and excinfo.value.status_code == 0
    message = str(excinfo.value)
    assert message.startswith("transport error:")
    assert "HTTP 0" not in message
