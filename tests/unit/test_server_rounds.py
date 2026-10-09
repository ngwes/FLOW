import copy
import io
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from flaw.blobstore import global_blob_id
from flaw.config import load_config
from flaw.models import build_model
from flaw.node.data import make_tiny_dataset
from flaw.orchestrator.aggregation import AggregationError
from flaw.orchestrator.rounds import RoundPhase
from flaw.orchestrator.server import Orchestrator, create_app
from flaw.orchestrator.state import RunDir
from flaw.protocol import (
    PROTOCOL_VERSION,
    compute_config_hash,
    deserialize_state_dict,
    serialize_state_dict,
)

BASE: dict = {
    "run": {"name": "exp1"},
    "model": {"name": "mnist_cnn"},
    "data": {"name": "tiny", "num_shards": 2, "scheme": "iid", "val_fraction": 0.25},
    "training": {"local_epochs": 1, "batch_size": 16},
    "federation": {
        "token": "secret",
        "rounds": 2,
        "min_participants": 1,
        "round_deadline_s": 10.0,
        "heartbeat_s": 0.1,
        "liveness_timeout_s": 10.0,
    },
}


def make_env(
    tmp_path: Path,
    federation: dict | None = None,
    evaluation: dict | None = None,
    storage: dict | None = None,
    early_stopping: dict | None = None,
    central_dataset_factory: Callable[[], object] | None = None,
):  # type: ignore[no-untyped-def]
    cfg_dict = copy.deepcopy(BASE)
    if federation:
        cfg_dict["federation"].update(federation)
    if evaluation is not None:
        cfg_dict["evaluation"] = evaluation
    if storage is not None:
        cfg_dict["storage"] = storage
    if early_stopping is not None:
        cfg_dict["early_stopping"] = early_stopping
    cfg_path = tmp_path / "flaw.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    cfg = load_config(cfg_path)
    run_dir = RunDir(root=tmp_path / "runs", run_name=cfg.run.name)
    run_dir.ensure()
    return create_app(
        cfg, run_dir, long_poll_s=0.05, central_dataset_factory=central_dataset_factory
    )


def until(cond: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def read_lines(orch: Orchestrator) -> list[dict]:
    path = orch.run_dir.metrics_path
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def round_lines(orch: Orchestrator) -> list[dict]:
    return [x for x in read_lines(orch) if x["type"] == "round"]


def summary_line(orch: Orchestrator) -> dict:
    # state.status flips to "finished" before the final central evaluation runs
    # and the summary line lands on disk; wait for the line itself.
    def has_summary() -> bool:
        lines = read_lines(orch)
        return bool(lines) and lines[-1]["type"] == "summary"

    assert until(has_summary)
    return read_lines(orch)[-1]


def global_sd(orch: Orchestrator, round_no: int) -> dict:
    return deserialize_state_dict(orch.blobs.get(global_blob_id(round_no)))


class FakeNode:
    """Minimal protocol client: enough to drive a run, no training involved."""

    def __init__(self, client: TestClient, orch: Orchestrator, node_id: str) -> None:
        self.client, self.orch, self.node_id = client, orch, node_id

    def register(self) -> dict:
        r = self.client.post(
            "/register",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "token": "secret",
                "node_id": self.node_id,
                "capabilities": {"device": "cpu", "device_name": "fake"},
                "config_hash": compute_config_hash(self.orch.cfg.canonical_hash_payload()),
            },
        )
        assert r.status_code == 200, r.text
        return r.json()

    def heartbeat(self) -> None:
        assert self.client.post("/heartbeat", json={"node_id": self.node_id}).status_code == 200

    def assignment(self):  # type: ignore[no-untyped-def]
        return self.client.get("/assignment", params={"node_id": self.node_id})

    def wait_assignment(self, round_no: int, timeout: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self.assignment()
            body = r.json() if r.status_code == 200 else {}
            if body and not body.get("run_finished") and body.get("round") == round_no:
                return body
            time.sleep(0.02)
        raise AssertionError(f"{self.node_id}: no assignment for round {round_no}")

    def no_assignment(self, timeout: float = 0.3) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self.assignment()
            if r.status_code == 200 and not r.json().get("run_finished"):
                return False
            time.sleep(0.02)
        return True

    def update_sd(
        self, round_no: int, delta: float, extra_key: bool = False, shape_break: bool = False
    ) -> dict:
        sd = {k: v + delta for k, v in global_sd(self.orch, round_no).items()}
        if extra_key:
            sd["bogus"] = torch.ones(3)
        if shape_break:
            sd["fc2.bias"] = torch.ones(1)
        return sd

    def upload(self, round_no: int, sd: dict) -> dict:
        r = self.client.post(
            f"/blob?node_id={self.node_id}&round={round_no}",
            content=serialize_state_dict(sd),
            headers={"X-Flaw-Token": "secret"},
        )
        assert r.status_code == 200, r.text
        return r.json()

    def submit(
        self,
        round_no: int,
        delta: float = 1.0,
        num_samples: int = 10,
        metrics: dict | None = None,
        extra_key: bool = False,
        shape_break: bool = False,
        corrupt_after_upload: bool = False,
    ):  # type: ignore[no-untyped-def]
        ref = self.upload(
            round_no, self.update_sd(round_no, delta, extra_key=extra_key, shape_break=shape_break)
        )
        if corrupt_after_upload:
            self.orch.blobs.path_for(ref["blob_id"]).write_bytes(b"corrupt")
        return self.client.post(
            "/result",
            json={
                "node_id": self.node_id,
                "round": round_no,
                "update": {
                    "blob_id": ref["blob_id"],
                    "sha256": ref["sha256"],
                    "size": ref["size"],
                },
                "num_samples": num_samples,
                "metrics": metrics if metrics is not None else {"loss": 0.5, "val_accuracy": 0.6},
                "duration_s": 0.1,
                "device": "cpu",
            },
        )


def submit_ref(node: FakeNode, round_no: int, ref: dict, num_samples: int = 1):  # type: ignore[no-untyped-def]
    """Post one /result for an already-uploaded blob reference."""
    return node.client.post(
        "/result",
        json={
            "node_id": node.node_id,
            "round": round_no,
            "update": {
                "blob_id": ref["blob_id"],
                "sha256": ref["sha256"],
                "size": ref["size"],
            },
            "num_samples": num_samples,
            "metrics": {"loss": 0.5, "val_accuracy": 0.6},
            "duration_s": 0.1,
            "device": "cpu",
        },
    )


def submit_blob(node: FakeNode, round_no: int, sd: dict, num_samples: int = 1):  # type: ignore[no-untyped-def]
    return submit_ref(node, round_no, node.upload(round_no, sd), num_samples=num_samples)


def test_two_round_run_completes(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 2, "min_participants": 2})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a, b = FakeNode(client, orch, "a"), FakeNode(client, orch, "b")
        a.register()
        b.register()
        for round_no in (1, 2):
            a.wait_assignment(round_no)
            b.wait_assignment(round_no)
            assert a.submit(round_no, delta=1.0, num_samples=1).json()["accepted"] is True
            assert a.no_assignment(0.15)  # already submitted this round
            assert b.submit(round_no, delta=3.0, num_samples=3).json()["accepted"] is True

        assert until(lambda: orch.state.status == "finished")
        initial, after1, after2 = global_sd(orch, 1), global_sd(orch, 2), global_sd(orch, 3)
        for key in initial:
            # weighted mean: (1*(x+1) + 3*(x+3)) / 4 == x + 2.5
            assert torch.allclose(after1[key], initial[key] + 2.5, atol=1e-5, rtol=1e-4)
            assert torch.allclose(after2[key], after1[key] + 2.5, atol=1e-5, rtol=1e-4)

        rounds = round_lines(orch)
        assert [x["round"] for x in rounds] == [1, 2]
        assert rounds[0]["closed_reason"] == "all_reported" and rounds[0]["void"] is False
        assert [p["node_id"] for p in rounds[0]["participants"]] == ["a", "b"]
        assert rounds[0]["aggregated"]["val_accuracy"] == pytest.approx(0.6)
        summary = summary_line(orch)
        assert summary["type"] == "summary" and summary["reason"] == "max_rounds"
        assert summary["best_round"] == 1 and summary["best_metric"] == pytest.approx(0.6)
        assert orch.run_dir.best_path.exists() and orch.run_dir.final_path.exists()
        best = torch.load(orch.run_dir.best_path, weights_only=True)
        assert torch.allclose(best["fc2.bias"], after1["fc2.bias"], atol=1e-6)

        finished = a.assignment()
        assert finished.status_code == 200
        assert finished.json()["run_finished"] is True
        assert finished.json()["reason"] == "max_rounds"


def test_void_round_reopens_same_number(tmp_path: Path) -> None:
    app = make_env(
        tmp_path, federation={"rounds": 1, "min_participants": 1, "round_deadline_s": 1.0}
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        assert until(lambda: any(x["void"] for x in round_lines(orch)), timeout=5.0)
        void_line = round_lines(orch)[0]
        assert void_line["round"] == 1
        assert void_line["void_reason"] == "insufficient_participants"
        assert void_line["closed_reason"] is None and void_line["participants"] == []
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert orch.state.finished_reason == "max_rounds"
        rounds = round_lines(orch)
        assert [x["round"] for x in rounds] == [1, 1]
        assert rounds[0]["void"] is True and rounds[1]["void"] is False
        assert global_sd(orch, 2)  # the next handout was written


def test_invalid_update_voids_round(tmp_path: Path) -> None:
    app = make_env(
        tmp_path, federation={"rounds": 1, "min_participants": 1, "round_deadline_s": 5.0}
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert (
            a.submit(1, extra_key=True).json()["accepted"] is True
        )  # shape checked at aggregation
        assert until(
            lambda: any(
                x.get("void_reason") == "insufficient_valid_updates" for x in round_lines(orch)
            )
        )
        first = round_lines(orch)[0]
        assert first["rejected"] == [{"node_id": "a", "reason": "extra_key"}]
        assert first["void"] is True and first["closed_reason"] is None
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert round_lines(orch)[-1]["void"] is False


def test_zero_sample_submission_is_rejected(tmp_path: Path) -> None:
    app = make_env(
        tmp_path, federation={"rounds": 1, "min_participants": 1, "round_deadline_s": 5.0}
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        # the submission is accepted on the wire; the zero count drops it at aggregation
        assert a.submit(1, delta=1.0, num_samples=0).json()["accepted"] is True
        assert until(
            lambda: any(
                x.get("void_reason") == "insufficient_valid_updates" for x in round_lines(orch)
            )
        )
        line = round_lines(orch)[0]
        assert line["void"] is True
        assert line["rejected"] == [{"node_id": "a", "reason": "invalid_samples"}]
        a.wait_assignment(1)  # the same round number reopened
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert round_lines(orch)[-1]["void"] is False


def test_zero_sample_metric_key_does_not_break_aggregation(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a, b = FakeNode(client, orch, "a"), FakeNode(client, orch, "b")
        a.register()
        b.register()
        a.wait_assignment(1)
        b.wait_assignment(1)
        # a reports a key nobody else does: weighting it by zero samples would
        # divide by zero inside aggregate_metrics and kill the round-loop task.
        zero_a = a.submit(1, delta=1.0, num_samples=0, metrics={"loss": 0.5, "only_a": 1.0}).json()
        assert zero_a["accepted"] is True
        good_b = b.submit(1, delta=2.0, num_samples=2, metrics={"loss": 0.4}).json()
        assert good_b["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        line = round_lines(orch)[0]
        assert line["void"] is False
        assert line["rejected"] == [{"node_id": "a", "reason": "invalid_samples"}]
        assert line["aggregated"] == {"loss": pytest.approx(0.4)}
        assert summary_line(orch)["reason"] == "max_rounds"


def test_malformed_update_rejections_void_round(tmp_path: Path) -> None:
    app = make_env(
        tmp_path, federation={"rounds": 1, "min_participants": 1, "round_deadline_s": 5.0}
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        shape_break = a.update_sd(1, 1.0, shape_break=True)
        missing_key = a.update_sd(1, 1.0)
        del missing_key["fc1.bias"]
        non_finite = a.update_sd(1, 1.0)
        non_finite["fc1.bias"] = torch.full_like(non_finite["fc1.bias"], float("nan"))
        bad_updates = [
            (shape_break, "shape_mismatch"),
            (missing_key, "missing_key"),
            (non_finite, "non_finite"),
        ]
        for offset, (sd, reason) in enumerate(bad_updates):
            a.wait_assignment(1)
            assert submit_blob(a, 1, sd).json()["accepted"] is True
            assert until(lambda: len(round_lines(orch)) >= offset + 1)
            line = round_lines(orch)[offset]
            assert line["void"] is True
            assert line["rejected"] == [{"node_id": "a", "reason": reason}]
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert round_lines(orch)[-1]["void"] is False


def test_dead_node_does_not_stall_round(tmp_path: Path) -> None:
    app = make_env(
        tmp_path,
        federation={
            "rounds": 1,
            "min_participants": 2,
            "round_deadline_s": 30.0,
            "liveness_timeout_s": 0.3,
        },
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a, b = FakeNode(client, orch, "a"), FakeNode(client, orch, "b")
        a.register()
        b.register()
        assert until(
            lambda: (
                orch.rounds.phase is RoundPhase.OPEN and set(orch.rounds.expected()) == {"a", "b"}
            )
        )
        a.heartbeat()
        assert until(lambda: not orch.registry.is_alive("b"), timeout=3.0)
        a.heartbeat()  # a stays alive; b stays silent
        assert b.no_assignment(0.2)  # dead but still expected: never handed work
        a.heartbeat()  # liveness_timeout_s is 0.3 s: refresh after the 0.2 s dead wait
        a.wait_assignment(1)
        a.heartbeat()  # keep a alive across submit up to the round's close check
        assert a.submit(1, delta=0.5, num_samples=1).json()["accepted"] is True
        # deadline is 30 s, so a round line this fast proves b was dropped from expected
        assert until(lambda: len(round_lines(orch)) == 1, timeout=5.0)
        line = round_lines(orch)[0]
        assert line["void"] is True and line["void_reason"] == "insufficient_participants"
        assert [p["node_id"] for p in line["participants"]] == ["a"]
        assert a.no_assignment(0.2)  # alive {a} < min_participants 2: no round ever reopens


def test_mid_round_join_waits_for_next_round(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 2, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        assert until(lambda: orch.rounds.phase is RoundPhase.OPEN)
        c = FakeNode(client, orch, "c")
        c.register()
        assert orch.registry.get("c").shard == 1
        assert c.no_assignment(0.3)  # joined mid-round -> not expected this round
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        c.wait_assignment(2)
        a.wait_assignment(2)
        assert a.submit(2, delta=1.0, num_samples=1).json()["accepted"] is True
        assert c.submit(2, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        rounds = round_lines(orch)
        assert [tuple(p["node_id"] for p in r["participants"]) for r in rounds] == [
            ("a",),
            ("a", "c"),
        ]


def test_duplicate_and_stale_results(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 2, "min_participants": 2})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a, b = FakeNode(client, orch, "a"), FakeNode(client, orch, "b")
        a.register()
        b.register()
        assert client.get("/assignment", params={"node_id": "ghost"}).status_code == 404
        ghost = client.post(
            "/result",
            json={
                "node_id": "ghost",
                "round": 1,
                "update": {"blob_id": "r0001/node-ghost", "sha256": "0" * 64, "size": 1},
                "num_samples": 1,
                "metrics": {"loss": 0.5},
                "duration_s": 0.1,
                "device": "cpu",
            },
        )
        assert ghost.status_code == 404
        a.wait_assignment(1)
        b.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        dup = a.submit(1, delta=1.0, num_samples=1)
        assert dup.json() == {"accepted": False, "reason": "duplicate"}
        # a blob for a round that has not opened yet: id and sha check out, the round does not
        ref = a.upload(2, a.update_sd(1, 1.0))
        unknown = client.post(
            "/result",
            json={
                "node_id": "a",
                "round": 2,
                "update": {
                    "blob_id": ref["blob_id"],
                    "sha256": ref["sha256"],
                    "size": ref["size"],
                },
                "num_samples": 1,
                "metrics": {"loss": 0.5, "val_accuracy": 0.6},
                "duration_s": 0.1,
                "device": "cpu",
            },
        )
        assert unknown.json() == {"accepted": False, "reason": "unknown_round"}
        assert b.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        a.wait_assignment(2)
        stale = a.submit(1, delta=1.0, num_samples=1)
        assert stale.json() == {"accepted": False, "reason": "stale"}


def test_corrupted_or_foreign_update_blob_rejected(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, corrupt_after_upload=True).json() == {
            "accepted": False,
            "reason": "invalid",
        }
        # a blob whose id does not belong to this node/round is rejected too
        ref = a.upload(1, a.update_sd(1, 1.0))
        foreign = client.post(
            "/result",
            json={
                "node_id": "a",
                "round": 1,
                "update": {
                    "blob_id": "r0001/node-someone-else",
                    "sha256": ref["sha256"],
                    "size": ref["size"],
                },
                "num_samples": 1,
                "metrics": {"loss": 0.5},
                "duration_s": 0.1,
                "device": "cpu",
            },
        )
        assert foreign.json() == {"accepted": False, "reason": "invalid"}
        # never-uploaded blob for a registered node
        c = FakeNode(client, orch, "c")
        c.register()
        missing = client.post(
            "/result",
            json={
                "node_id": "c",
                "round": 1,
                "update": {"blob_id": "r0001/node-c", "sha256": "0" * 64, "size": 1},
                "num_samples": 1,
                "metrics": {"loss": 0.5},
                "duration_s": 0.1,
                "device": "cpu",
            },
        )
        assert missing.json() == {"accepted": False, "reason": "invalid"}


def test_unreadable_update_blob_rejected(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        # arbitrary bytes: the id/sha/size checks pass, deserialization cannot
        up = client.post(
            "/blob?node_id=a&round=1",
            content=b"not-a-state-dict",
            headers={"X-Flaw-Token": "secret"},
        )
        assert up.status_code == 200
        assert submit_ref(a, 1, up.json()).json()["accepted"] is True
        assert until(lambda: len(round_lines(orch)) == 1)
        line = round_lines(orch)[0]
        assert line["rejected"] == [{"node_id": "a", "reason": "unreadable"}]
        assert line["void"] is True and line["void_reason"] == "insufficient_valid_updates"
        a.wait_assignment(1)  # the same round number reopened
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert round_lines(orch)[-1]["void"] is False


def test_torch_payload_with_string_value_does_not_kill_loop(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 2, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        # a decodable torch zip that is malformed *inside*: one template key maps
        # to a string (weights_only permits str primitives), so validation raises
        sd = a.update_sd(1, 1.0)
        sd["fc1.weight"] = "boom"
        buf = io.BytesIO()
        torch.save(sd, buf)
        up = client.post(
            "/blob?node_id=a&round=1",
            content=buf.getvalue(),
            headers={"X-Flaw-Token": "secret"},
        )
        assert up.status_code == 200
        assert submit_ref(a, 1, up.json()).json()["accepted"] is True
        assert until(lambda: len(round_lines(orch)) == 1)
        line = round_lines(orch)[0]
        assert line["void"] is True and line["void_reason"] == "insufficient_valid_updates"
        assert line["rejected"] == [{"node_id": "a", "reason": "unreadable"}]
        assert orch._loop_task is not None and orch._loop_task.done() is False
        # the loop survived: the retried round and the following one both complete
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        a.wait_assignment(2)
        assert a.submit(2, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")


def test_round_loop_survives_iteration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        real_sweep = orch.registry.sweep
        calls = {"n": 0}

        def flaky_sweep() -> list[str]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sweep exploded")
            return real_sweep()

        monkeypatch.setattr(orch.registry, "sweep", flaky_sweep)
        with caplog.at_level(logging.ERROR, logger="flaw.orchestrator"):
            a = FakeNode(client, orch, "a")
            a.register()
            assert until(
                lambda: any("round loop iteration failed" in r.getMessage() for r in caplog.records)
            )
            # a normal round still completes after the swallowed failure
            a.wait_assignment(1)
            assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
            assert until(lambda: orch.state.status == "finished")
            assert round_lines(orch)[-1]["void"] is False


def test_early_stop_target_reached(tmp_path: Path) -> None:
    app = make_env(
        tmp_path,
        federation={"rounds": 10, "min_participants": 1},
        early_stopping={"metric": "val_accuracy", "target": 0.9, "patience": 5, "min_delta": 0.001},
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert (
            a.submit(1, num_samples=4, metrics={"loss": 0.2, "val_accuracy": 0.95}).json()[
                "accepted"
            ]
            is True
        )
        assert until(lambda: orch.state.status == "finished")
        assert orch.state.finished_reason == "target_reached"
        assert round_lines(orch)[0]["aggregated"]["val_accuracy"] == pytest.approx(0.95)
        summary = summary_line(orch)
        assert summary["type"] == "summary" and summary["reason"] == "target_reached"
        assert summary["best_round"] == 1 and summary["best_metric"] == pytest.approx(0.95)
        assert orch.run_dir.best_path.exists() and orch.run_dir.final_path.exists()


def test_missing_early_stop_metric_is_flagged(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, num_samples=2, metrics={"loss": 0.4}).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert orch.state.finished_reason == "max_rounds"  # stopper skipped, round cap applies
        line = round_lines(orch)[0]
        assert line["early_stop_metric_missing"] is True
        assert "val_accuracy" not in line["aggregated"]
        assert summary_line(orch)["best_round"] is None
        assert not orch.run_dir.best_path.exists()


def test_blob_retention_gc_keeps_best_and_final(tmp_path: Path) -> None:
    app = make_env(
        tmp_path,
        federation={"rounds": 3, "min_participants": 1},
        storage={"retention": {"keep_rounds": 1}},
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        for round_no in (1, 2, 3):
            a.wait_assignment(round_no)
            assert (
                a.submit(round_no, delta=float(round_no), num_samples=1).json()["accepted"] is True
            )
        assert until(lambda: orch.state.status == "finished")
        assert not orch.blobs.exists("r0001/global")
        assert not orch.blobs.exists("r0002/global")
        assert orch.blobs.exists("r0003/global") and orch.blobs.exists("r0004/global")
        assert orch.run_dir.best_path.exists() and orch.run_dir.final_path.exists()


def test_recover_uses_existing_blobs(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 5})
    orch = app.state.orchestrator
    for n in (1, 2, 3):
        sd = {k: v + float(n) for k, v in orch.model_template().items()}
        orch.blobs.put(global_blob_id(n), serialize_state_dict(sd))
    orch.state.round = 1  # stale on-disk state: crash after round 2 was committed
    orch.run_dir.save_state(orch.state)
    orch.recover()
    assert orch.state.round == 2  # round 3 was not fully committed -> redo it
    assert orch.blobs.exists(global_blob_id(3))  # existing blobs untouched


def test_recover_writes_seeded_initial_global(tmp_path: Path) -> None:
    app = make_env(tmp_path)
    orch = app.state.orchestrator
    orch.recover()
    assert orch.state.round == 0
    assert orch.blobs.exists("r0001/global")
    torch.manual_seed(orch.cfg.run.seed)
    expected = build_model(orch.cfg.model).state_dict()
    got = global_sd(orch, 1)
    assert all(torch.equal(got[k], expected[k]) for k in expected)


def test_restarted_finished_run_reports_run_finished(tmp_path: Path) -> None:
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=2).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert until(
            lambda: (
                json.loads(orch.run_dir.state_path.read_text(encoding="utf-8"))["status"]
                == "finished"
            )
        )
    # restart over the same run_dir: a fresh app, fresh registry, same state.json
    cfg = load_config(tmp_path / "flaw.yaml")
    run_dir = RunDir(root=tmp_path / "runs", run_name=cfg.run.name)
    restarted = create_app(cfg, run_dir, long_poll_s=0.05)
    with TestClient(restarted) as client:
        orch2 = restarted.state.orchestrator
        b = FakeNode(client, orch2, "b")
        b.register()
        resp = b.assignment()
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_finished"] is True
        assert body["reason"] == "max_rounds"
        assert body["best_round"] == 1
        assert orch2.state.phase == "finished"


def test_aggregation_failure_voids_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import flaw.orchestrator.server as server_module

    def boom(*args: object, **kwargs: object) -> object:
        raise AggregationError("boom")

    monkeypatch.setattr(server_module, "aggregate_weighted", boom)
    app = make_env(tmp_path, federation={"rounds": 1, "min_participants": 1})
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(
            lambda: any(x.get("void_reason") == "aggregation_failed" for x in round_lines(orch))
        )


def test_central_eval_every_round(tmp_path: Path) -> None:
    def factory() -> object:
        return make_tiny_dataset(seed=99, n=64, val_n=32)[1]

    app = make_env(
        tmp_path,
        federation={"rounds": 1, "min_participants": 1},
        evaluation={"federated": False, "central_test": {"mode": "every_round"}},
        central_dataset_factory=factory,
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        central = round_lines(orch)[0]["central"]
        assert set(central) == {"val_loss", "val_accuracy"}


def test_central_eval_final_only(tmp_path: Path) -> None:
    def factory() -> object:
        return make_tiny_dataset(seed=99, n=64, val_n=32)[1]

    app = make_env(
        tmp_path,
        federation={"rounds": 1, "min_participants": 1},
        evaluation={"central_test": {"mode": "final_only"}},
        central_dataset_factory=factory,
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert "central" not in round_lines(orch)[0]
        summary = summary_line(orch)
        assert set(summary["central_best"]) == {"val_loss", "val_accuracy"}
        assert set(summary["central_final"]) == {"val_loss", "val_accuracy"}


def test_central_eval_failure_recorded(tmp_path: Path) -> None:
    def factory() -> object:
        raise RuntimeError("no data here")

    app = make_env(
        tmp_path,
        federation={"rounds": 1, "min_participants": 1},
        evaluation={"central_test": {"mode": "every_round"}},
        central_dataset_factory=factory,
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert "no data here" in round_lines(orch)[0]["central_error"]
        assert orch.state.finished_reason == "max_rounds"


def test_central_final_only_failure_recorded(tmp_path: Path) -> None:
    def factory() -> object:
        raise RuntimeError("no data here")

    app = make_env(
        tmp_path,
        federation={"rounds": 1, "min_participants": 1},
        evaluation={"central_test": {"mode": "final_only"}},
        central_dataset_factory=factory,
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert orch.state.finished_reason == "max_rounds"
        assert "central" not in round_lines(orch)[0]
        summary = summary_line(orch)
        assert summary["type"] == "summary" and summary["reason"] == "max_rounds"
        assert "no data here" in summary["central_error"]
        assert "central_best" not in summary and "central_final" not in summary


def test_central_off_skips_evaluation(tmp_path: Path) -> None:
    def factory() -> object:
        return make_tiny_dataset(seed=99, n=64, val_n=32)[1]

    app = make_env(
        tmp_path,
        federation={"rounds": 1, "min_participants": 1},
        evaluation={"central_test": {"mode": "off"}},
        central_dataset_factory=factory,
    )
    with TestClient(app) as client:
        orch = app.state.orchestrator
        a = FakeNode(client, orch, "a")
        a.register()
        a.wait_assignment(1)
        assert a.submit(1, delta=1.0, num_samples=1).json()["accepted"] is True
        assert until(lambda: orch.state.status == "finished")
        assert "central" not in round_lines(orch)[0]
        assert "central_best" not in summary_line(orch)
