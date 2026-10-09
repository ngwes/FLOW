from pathlib import Path

from fastapi.testclient import TestClient

from flaw.config import load_config
from flaw.orchestrator.server import Orchestrator, create_app
from flaw.orchestrator.state import RunDir
from flaw.protocol import PROTOCOL_VERSION, compute_config_hash

CFG = """
run: {name: exp1}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 16}
federation: {token: secret, rounds: 3, min_participants: 1, round_deadline_s: 0.3}
"""

CFG_FEDERATED = CFG.replace(
    "data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}",
    "data: {path: /nonexistent, val_fraction: 0.25}",
)


def make_app(tmp_path: Path, text: str = CFG) -> tuple[TestClient, Orchestrator]:
    cfg_path = tmp_path / "flaw.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    cfg = load_config(cfg_path)
    run_dir = RunDir(root=tmp_path / "runs", run_name=cfg.run.name)
    run_dir.ensure()
    app = create_app(cfg, run_dir, long_poll_s=0.05)
    client = TestClient(app)
    return client, app.state.orchestrator


def register_body(orch: Orchestrator, node_id: str, **overrides: object) -> dict:
    body: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "token": "secret",
        "node_id": node_id,
        "capabilities": {"device": "cpu", "device_name": "test"},
        "config_hash": compute_config_hash(orch.cfg.canonical_hash_payload()),
    }
    body.update(overrides)
    return body


def test_register_assigns_shards_and_persists(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    r1 = client.post("/register", json=register_body(orch, "a"))
    r2 = client.post("/register", json=register_body(orch, "b"))
    assert r1.status_code == 200 and r1.json()["run"]["name"] == "exp1"
    assert r1.json()["run"]["phase"] == "idle"
    assert r1.json()["shard"] == 0 and r2.json()["shard"] == 1
    assert r2.status_code == 200
    assert orch.registry.shard_map() == {"a": 0, "b": 1}
    assert orch.run_dir.load_state().shard_map == {"a": 0, "b": 1}


def test_register_rejections(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    assert client.post("/register", json=register_body(orch, "a")).status_code == 200
    bad_token = client.post("/register", json=register_body(orch, "b", token="nope"))
    assert bad_token.status_code == 403 and "token" in bad_token.json()["reason"]
    bad_version = client.post(
        "/register", json=register_body(orch, "b", protocol_version=PROTOCOL_VERSION + 1)
    )
    assert bad_version.status_code == 403 and "protocol_version" in bad_version.json()["reason"]
    bad_hash = client.post("/register", json=register_body(orch, "b", config_hash="0" * 64))
    assert bad_hash.status_code == 409 and "config mismatch" in bad_hash.json()["reason"]
    dup = client.post("/register", json=register_body(orch, "a"))
    assert dup.status_code == 409 and "in use" in dup.json()["reason"]


def test_register_shard_exhaustion(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)  # num_shards: 2
    assert client.post("/register", json=register_body(orch, "a")).status_code == 200
    assert client.post("/register", json=register_body(orch, "b")).status_code == 200
    full = client.post("/register", json=register_body(orch, "c"))
    assert full.status_code == 409 and "shards" in full.json()["reason"]


def test_register_federated_mode_no_shards(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path, CFG_FEDERATED)
    r = client.post("/register", json=register_body(orch, "a"))
    assert r.status_code == 200 and r.json()["shard"] is None
    assert orch.registry.shard_map() == {}


def test_heartbeat(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    client.post("/register", json=register_body(orch, "a"))
    r = client.post("/heartbeat", json={"node_id": "a"})
    assert r.status_code == 200 and r.json() == {"phase": "idle", "round": 0}
    assert client.post("/heartbeat", json={"node_id": "ghost"}).status_code == 404


def test_status(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    client.post("/register", json=register_body(orch, "a"))
    body = client.get("/status").json()
    assert body["name"] == "exp1" and body["round"] == 0
    node = body["nodes"][0]
    assert node["node_id"] == "a" and node["alive"] is True and node["shard"] == 0
    assert node["last_heartbeat_age_s"] >= 0


def test_blob_upload_and_download(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    client.post("/register", json=register_body(orch, "a"))
    up = client.post(
        "/blob?node_id=a&round=1",
        content=b"weights-bytes",
        headers={"X-Flaw-Token": "secret"},
    )
    assert up.status_code == 200
    body = up.json()
    assert body["blob_id"] == "r0001/node-a" and body["size"] == 13
    assert orch.blobs.exists("r0001/node-a")

    down = client.get("/blob/r0001/node-a")
    assert down.status_code == 200 and down.content == b"weights-bytes"
    assert down.headers["X-Blob-Sha256"] == body["sha256"]

    assert client.get("/blob/r0002/global").status_code == 404
    assert client.get("/blob/r1/global").status_code == 400


def test_blob_upload_rejections(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    client.post("/register", json=register_body(orch, "a"))
    assert (
        client.post(
            "/blob?node_id=a&round=1", content=b"x", headers={"X-Flaw-Token": "wrong"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/blob?node_id=ghost&round=1", content=b"x", headers={"X-Flaw-Token": "secret"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/blob?node_id=a&round=0", content=b"x", headers={"X-Flaw-Token": "secret"}
        ).status_code
        == 400
    )


def test_blob_upload_unsafe_node_id_is_400(tmp_path: Path) -> None:
    client, orch = make_app(tmp_path)
    # register accepts any node id string (docs the API boundary) ...
    assert client.post("/register", json=register_body(orch, "weird/id")).status_code == 200
    # ... but a node id that cannot form a valid blob id must 400, not 500.
    r = client.post(
        "/blob?node_id=weird%2Fid&round=1",
        content=b"x",
        headers={"X-Flaw-Token": "secret"},
    )
    assert r.status_code == 400
    assert "invalid blob id" in r.json()["reason"]
