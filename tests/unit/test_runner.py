import logging
import socket
import threading
import time
from pathlib import Path

import pytest
import torch

from flaw.blobstore import ChecksumMismatch
from flaw.config import load_config
from flaw.models import MnistCNN
from flaw.node import runner
from flaw.node.client import ClientError
from flaw.node.runner import run_node
from flaw.node.trainer import TrainOutcome
from flaw.protocol import (
    Assignment,
    BlobRef,
    HeartbeatResponse,
    OptimizerSpec,
    RegisterResponse,
    ResultResponse,
    RunFinished,
    RunInfo,
    TrainSpec,
)

CFG = """
run: {name: exp1, seed: 42}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 16}
federation: {token: secret, heartbeat_s: 0.01, node_id: lab-9}
"""


def cfg(tmp_path: Path, text: str = CFG):  # type: ignore[no-untyped-def]
    p = tmp_path / "flaw.yaml"
    p.write_text(text, encoding="utf-8")
    return load_config(p)


def assignment(round_no: int = 1) -> Assignment:
    return Assignment(
        round=round_no,
        train_spec=TrainSpec(
            local_epochs=1,
            batch_size=8,
            optimizer=OptimizerSpec(name="sgd", lr=0.1, momentum=0.0),
            seed=42,
        ),
        weights=BlobRef(blob_id=f"r{round_no:04d}/global", sha256="0" * 64, size=10),
    )


def outcome() -> TrainOutcome:
    return TrainOutcome(
        update={"w": torch.ones(2)},
        num_samples=7,
        metrics={"loss": 0.1, "val_loss": 0.1, "val_accuracy": 0.9},
        duration_s=0.01,
    )


def registered(shard: int | None = 0) -> RegisterResponse:
    return RegisterResponse(
        accepted=True, run=RunInfo(name="exp1", current_round=0, phase="idle"), shard=shard
    )


class FakeClient:
    def __init__(self, assignments: list) -> None:
        self.assignments = list(assignments)
        self.register_errors: list[Exception] = []
        self.register_responses: list[object] = []
        self.register_shard: int | None = 0
        self.download_errors: list[Exception] = []
        self.heartbeat_error: Exception | None = None
        self.submit_accepted = True
        self.submit_errors: list[Exception] = []
        # real mnist_cnn state dict: the runner builds cfg.model and loads
        # strictly, so a toy {"w": ...} dict cannot load
        self.sd = MnistCNN().state_dict()
        self.node_id = "fake-node"
        self.register_calls = 0
        self.heartbeat_calls = 0
        self.downloaded: list[BlobRef] = []
        self.submitted: list[tuple] = []
        self.capabilities = None
        self.closed = False

    def register(self, capabilities=None):  # type: ignore[no-untyped-def]
        self.register_calls += 1
        self.capabilities = capabilities
        if self.register_responses:
            item = self.register_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self.register_errors:
            raise self.register_errors.pop(0)
        return RegisterResponse(
            accepted=True,
            run=RunInfo(name="exp1", current_round=0, phase="idle"),
            shard=self.register_shard,
        )

    def get_assignment(self):  # type: ignore[no-untyped-def]
        item = self.assignments.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def download_blob(self, ref: BlobRef):  # type: ignore[no-untyped-def]
        self.downloaded.append(ref)
        if self.download_errors:
            raise self.download_errors.pop(0)
        return {k: v.clone() for k, v in self.sd.items()}

    def upload_blob(self, round_no: int, data: bytes) -> BlobRef:
        return BlobRef(blob_id=f"r{round_no:04d}/node-lab-9", sha256="0" * 64, size=len(data))

    def submit_result(self, round_no, update, num_samples, metrics, duration_s, device):  # type: ignore[no-untyped-def]
        self.submitted.append((round_no, update, num_samples, metrics, duration_s, device))
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        return ResultResponse(
            accepted=self.submit_accepted, reason=None if self.submit_accepted else "stale"
        )

    def heartbeat(self) -> HeartbeatResponse:
        self.heartbeat_calls += 1
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        return HeartbeatResponse(phase="open", round=1)

    def close(self) -> None:
        self.closed = True


def test_run_node_full_flow(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient(
        [assignment(1), RunFinished(reason="max_rounds", best_round=1, best_metric=0.9)]
    )
    prepared: list[tuple] = []
    trained: list[tuple] = []

    def prepare(c, shard, seed):  # type: ignore[no-untyped-def]
        prepared.append((shard, seed))
        return ("train", "val")

    def train(model, train_ds, val_ds, spec, device="cpu", seed=42):  # type: ignore[no-untyped-def]
        trained.append((train_ds, val_ds))
        return outcome()

    result = run_node(
        config, "http://x", client=client, prepare_data=prepare, train=train, sleep=lambda s: None
    )
    assert result == "finished"
    assert prepared == [(0, 42)]
    assert trained == [("train", "val")]
    assert client.register_calls == 1
    assert client.capabilities is not None and client.capabilities.device
    assert client.downloaded[0].blob_id == "r0001/global"
    assert len(client.submitted) == 1
    round_no, update, num_samples, metrics, duration_s, device = client.submitted[0]
    assert round_no == 1 and num_samples == 7
    assert metrics["val_accuracy"] == 0.9 and update.blob_id == "r0001/node-lab-9"
    assert client.closed is False  # injected client is owned by the caller


def test_data_prepared_once_for_multiple_rounds(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient(
        [
            assignment(1),
            assignment(2),
            RunFinished(reason="max_rounds", best_round=2, best_metric=0.9),
        ]
    )
    prepared: list[tuple] = []

    def prepare(c, shard, seed):  # type: ignore[no-untyped-def]
        prepared.append((shard, seed))
        return ("t", "v")

    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=prepare,
        train=lambda *a, **kw: outcome(),
        sleep=lambda s: None,
    )
    assert result == "finished" and len(prepared) == 1 and len(client.submitted) == 2


def test_registration_retries_then_succeeds(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([RunFinished(reason="max_rounds")])
    client.register_errors = [ClientError(0, "down", fatal=False)] * 2
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished"
    assert sleeps == [1.0, 2.0] and client.register_calls == 3


def test_logs_the_identity_the_client_registers_with(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # `flaw simulate` names its clients sim-N while the config says lab-9 (or nothing,
    # i.e. the hostname): the log must name who actually registered.
    client = FakeClient([RunFinished(reason="max_rounds")])
    client.node_id = "sim-3"
    with caplog.at_level(logging.INFO, logger="flaw.node"):
        run_node(
            cfg(tmp_path),
            "http://x",
            client=client,
            prepare_data=lambda c, s, seed: ("t", "v"),
            train=lambda *a, **kw: outcome(),
            sleep=lambda s: None,
        )
    assert "registered as sim-3" in caplog.text
    assert "lab-9" not in caplog.text


def test_round_log_line_is_compact(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    client = FakeClient([assignment(1), RunFinished(reason="max_rounds")])
    with caplog.at_level(logging.INFO, logger="flaw.node"):
        run_node(
            cfg(tmp_path),
            "http://x",
            client=client,
            prepare_data=lambda c, s, seed: ("t", "v"),
            train=lambda *a, **kw: outcome(),
            sleep=lambda s: None,
        )
    done = [r.getMessage() for r in caplog.records if "done" in r.getMessage()]
    metrics = outcome().metrics
    expected = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
    assert done == [f"round 1 done: {expected}"]


def test_register_fatal_returns_immediately(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([RunFinished(reason="max_rounds")])
    client.register_errors = [ClientError(403, "bad token", fatal=True)]
    sleeps: list[float] = []
    result = run_node(config, "http://x", client=client, sleep=sleeps.append)
    assert result == "fatal_error" and sleeps == []


def test_transient_assignment_errors_back_off(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient(
        [
            ClientError(0, "down", fatal=False),
            ClientError(500, "boom", fatal=False),
            RunFinished(reason="max_rounds"),
        ]
    )
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished" and sleeps == [1.0, 2.0]


def test_corrupt_blob_and_train_crash_are_retried(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient(
        [assignment(1), assignment(1), assignment(1), RunFinished(reason="max_rounds")]
    )
    client.download_errors = [ChecksumMismatch("bad checksum")]
    trains: list[int] = []

    def train(*a: object, **kw: object) -> TrainOutcome:
        trains.append(1)
        if len(trains) == 1:
            raise RuntimeError("CUDA out of memory")
        return outcome()

    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=train,
        sleep=sleeps.append,
    )
    assert result == "finished"
    assert sleeps == [1.0, 2.0]  # one for the corrupt blob, one for the training crash
    assert len(trains) == 2 and len(client.submitted) == 1


def test_rejected_submission_is_not_an_error(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([assignment(1), RunFinished(reason="max_rounds")])
    client.submit_accepted = False
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished" and sleeps == []


def test_shared_mode_without_shard_is_fatal(tmp_path: Path) -> None:
    config = cfg(tmp_path)  # data.name = tiny (shared mode) needs a shard to select data
    client = FakeClient([assignment(1)])
    client.register_shard = None
    sleeps: list[float] = []
    result = run_node(config, "http://x", client=client, sleep=sleeps.append)
    assert result == "fatal_error" and sleeps == []


def test_stop_event_before_start(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    stop = threading.Event()
    stop.set()
    client = FakeClient([assignment(1)])
    result = run_node(config, "http://x", client=client, stop_event=stop, sleep=lambda s: None)
    assert result == "stopped" and client.register_calls == 0


def test_heartbeat_thread_beats_and_survives_errors(tmp_path: Path) -> None:
    config = cfg(tmp_path)  # federation.heartbeat_s: 0.01
    client = FakeClient([])
    client.heartbeat_error = ClientError(0, "down", fatal=False)
    stop = threading.Event()
    thread = threading.Thread(target=runner._heartbeat_loop, args=(client, config, stop))
    thread.start()
    time.sleep(0.1)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert client.heartbeat_calls >= 1


def test_vram_cap_and_gpu_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = CFG + "\nresources: {gpu: '0', vram_fraction: 0.5}\n"
    config = cfg(tmp_path, text)
    monkeypatch.setattr(runner, "pick_device", lambda explicit: (explicit, "Fake GPU"))
    calls: list[tuple] = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda frac, dev: calls.append((frac, dev))
    )
    client = FakeClient([RunFinished(reason="max_rounds")])
    result = run_node(config, "http://x", client=client, sleep=lambda s: None)
    assert result == "finished"
    assert calls == [(0.5, 0)]  # int device index parsed from "cuda:0"
    assert client.capabilities.device == "cuda:0"


def _cgroup(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"cpu.max": "400000 100000\n"}, 4.0),  # cgroup v2, `--cpus 4`
        ({"cpu.max": "150000 100000\n"}, 1.5),
        ({"cpu.max": "max 100000\n"}, None),  # v2, no limit
        ({"cpu/cpu.cfs_quota_us": "250000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 2.5),  # v1
        ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, None),
        ({}, None),  # not in a cgroup we can read (bare metal, Windows)
        ({"cpu.max": "garbage\n"}, None),
    ],
)
def test_cgroup_cpu_limit(tmp_path: Path, files: dict[str, str], expected: float | None) -> None:
    assert runner.cgroup_cpu_limit(_cgroup(tmp_path, files)) == expected


@pytest.mark.parametrize(
    ("limit", "env", "current", "expected_calls"),
    [
        (4.0, {}, 11, [4]),  # the compose case: 11 threads squeezed into 4 CPUs
        (1.5, {}, 11, [2]),  # partial CPUs round up
        (0.5, {}, 11, [1]),
        (16.0, {}, 11, []),  # quota above the default: leave torch alone
        (None, {}, 11, []),  # no quota
        (4.0, {"OMP_NUM_THREADS": "8"}, 11, []),  # an explicit setting wins
    ],
)
def test_apply_cpu_quota_sizes_torch_threads(
    monkeypatch: pytest.MonkeyPatch,
    limit: float | None,
    env: dict[str, str],
    current: int,
    expected_calls: list[int],
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(runner, "cgroup_cpu_limit", lambda: limit)
    monkeypatch.setattr(torch, "get_num_threads", lambda: current)
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    runner.apply_cpu_quota(env)
    assert calls == expected_calls


def test_run_node_applies_cpu_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    applied: list[bool] = []
    monkeypatch.setattr(runner, "apply_cpu_quota", lambda: applied.append(True))
    client = FakeClient([RunFinished(reason="max_rounds")])
    assert run_node(cfg(tmp_path), "http://x", client=client, sleep=lambda s: None) == "finished"
    assert applied == [True]


def test_device_override_wins_over_auto_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = cfg(tmp_path)
    picked: list[str | None] = []

    def fake_pick(explicit):  # type: ignore[no-untyped-def]
        picked.append(explicit)
        return explicit, "Fake"

    monkeypatch.setattr(runner, "pick_device", fake_pick)
    client = FakeClient([RunFinished(reason="max_rounds")])
    result = run_node(config, "http://x", client=client, device="cpu", sleep=lambda s: None)
    assert result == "finished" and picked == ["cpu"]


def test_default_client_construction_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = cfg(tmp_path)
    captured: dict = {}

    class FakeOrchClient:
        def __init__(self, base_url, token, cfg_, node_id, **kw):  # type: ignore[no-untyped-def]
            captured.update(base_url=base_url, token=token, node_id=node_id)
            self.node_id = node_id

        def register(self, capabilities=None):  # type: ignore[no-untyped-def]
            return RegisterResponse(
                accepted=True, run=RunInfo(name="exp1", current_round=0, phase="idle"), shard=0
            )

        def get_assignment(self):  # type: ignore[no-untyped-def]
            return RunFinished(reason="max_rounds")

        def heartbeat(self):  # type: ignore[no-untyped-def]
            return HeartbeatResponse(phase="open", round=1)

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(runner, "OrchestratorClient", FakeOrchClient)
    result = run_node(config, "http://orch:8000", sleep=lambda s: None)
    assert result == "finished"
    assert captured["base_url"] == "http://orch:8000"
    assert captured["token"] == "secret" and captured["node_id"] == "lab-9"
    assert captured["closed"] is True


def test_resolve_node_id_falls_back_to_hostname(tmp_path: Path) -> None:
    config = cfg(tmp_path, CFG.replace(", node_id: lab-9", ""))
    assert runner.resolve_node_id(config) == socket.gethostname()


# --- controller additions (T16 per-module coverage: error paths) ---


def test_none_assignment_loops_without_sleep(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([None, assignment(1), RunFinished(reason="max_rounds")])
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished" and len(client.submitted) == 1 and sleeps == []


def test_fatal_assignment_error_returns_immediately(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([ClientError(404, "not registered", fatal=True)])
    sleeps: list[float] = []
    result = run_node(config, "http://x", client=client, sleep=sleeps.append)
    assert result == "fatal_error" and sleeps == []


def test_nonfatal_round_client_error_backs_off(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([assignment(1), RunFinished(reason="max_rounds")])
    client.download_errors = [ClientError(500, "boom", fatal=False)]
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished" and sleeps == [1.0] and client.submitted == []


def test_fatal_round_client_error_returns_immediately(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([assignment(1), RunFinished(reason="max_rounds")])
    client.download_errors = [ClientError(404, "gone", fatal=True)]
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "fatal_error" and sleeps == []


def test_stop_event_mid_loop(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    stop = threading.Event()
    client = FakeClient([assignment(1), assignment(2)])

    def train(*a: object, **kw: object) -> TrainOutcome:
        stop.set()
        return outcome()

    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=train,
        stop_event=stop,
        sleep=lambda s: None,
    )
    assert result == "stopped" and len(client.submitted) == 1


# --- fix wave: reconnection after an orchestrator restart ---


def test_unknown_node_on_assignment_reregisters(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient(
        [
            ClientError(404, "unknown node", fatal=True),
            assignment(1),
            RunFinished(reason="max_rounds"),
        ]
    )
    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished"
    assert client.register_calls == 2
    assert len(client.submitted) == 1
    assert sleeps == []


def test_unknown_node_on_submit_reregisters_and_replays_round(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([assignment(1), assignment(1), RunFinished(reason="max_rounds")])
    client.submit_errors = [ClientError(404, "unknown node", fatal=True)]
    prepared: list[tuple] = []

    def prepare(c, shard, seed):  # type: ignore[no-untyped-def]
        prepared.append((shard, seed))
        return ("t", "v")

    sleeps: list[float] = []
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=prepare,
        train=lambda *a, **kw: outcome(),
        sleep=sleeps.append,
    )
    assert result == "finished"
    assert client.register_calls == 2
    assert len(client.submitted) == 2
    assert prepared == [(0, 42), (0, 42)]  # data re-prepared after reconnect
    assert sleeps == []


def test_reregister_rejection_is_fatal(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([ClientError(404, "unknown node", fatal=True)])
    client.register_responses = [registered(), ClientError(403, "bad token", fatal=True)]
    result = run_node(config, "http://x", client=client, sleep=lambda s: None)
    assert result == "fatal_error"
    assert client.register_calls == 2


def test_reregister_rejection_during_round_is_fatal(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([assignment(1)])
    client.submit_errors = [ClientError(404, "unknown node", fatal=True)]
    client.register_responses = [registered(), ClientError(403, "bad token", fatal=True)]
    result = run_node(
        config,
        "http://x",
        client=client,
        prepare_data=lambda c, s, seed: ("t", "v"),
        train=lambda *a, **kw: outcome(),
        sleep=lambda s: None,
    )
    assert result == "fatal_error"
    assert client.register_calls == 2


def test_stop_during_reregister_returns_stopped(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    client = FakeClient([ClientError(404, "unknown node", fatal=True)])
    client.register_responses = [registered(), ClientError(0, "down", fatal=False)]
    stop = threading.Event()
    result = run_node(
        config,
        "http://x",
        client=client,
        stop_event=stop,
        sleep=lambda s: stop.set(),
    )
    assert result == "stopped"
    assert client.register_calls == 2
