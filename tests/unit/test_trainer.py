import copy

import pytest
import torch

from flaw.models import build_model
from flaw.node.data import make_tiny_dataset
from flaw.node.trainer import evaluate, pick_device, train_local
from flaw.protocol import OptimizerSpec, TrainSpec


def spec(epochs: int = 2, batch: int = 16, lr: float = 0.1) -> TrainSpec:
    return TrainSpec(
        local_epochs=epochs,
        batch_size=batch,
        optimizer=OptimizerSpec(name="sgd", lr=lr, momentum=0.9),
        seed=42,
    )


def test_pick_device_explicit() -> None:
    device, name = pick_device("cpu")
    assert device == "cpu" and name


def test_pick_device_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device(None)[0] == "cpu"


def test_pick_device_auto_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    assert pick_device(None) == ("cuda:0", "Fake GPU")


def test_train_local_forward_call_count() -> None:
    train_ds, val_ds = make_tiny_dataset(seed=0, n=32, val_n=8)
    model = build_model_by_name()
    calls = 0
    original = model.forward

    def counting_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.forward = counting_forward  # type: ignore[method-assign]
    outcome = train_local(model, train_ds, val_ds, spec(epochs=2, batch=16), seed=1)
    assert calls == 5  # 4 training forwards (2 batches x 2 epochs) + 1 evaluation forward
    assert outcome.num_samples == 32


def test_train_local_reduces_loss() -> None:
    train_ds, val_ds = make_tiny_dataset(seed=0)
    model = build_model_by_name()
    before = evaluate(model, val_ds, batch_size=64, device="cpu")["val_loss"]
    outcome = train_local(model, train_ds, val_ds, spec(epochs=5, lr=0.2), seed=0)
    assert outcome.metrics["val_loss"] < before
    assert 0.0 <= outcome.metrics["val_accuracy"] <= 1.0
    assert set(outcome.metrics) == {"loss", "val_loss", "val_accuracy"}


def test_train_local_deterministic() -> None:
    train_ds, val_ds = make_tiny_dataset(seed=0)
    model = build_model_by_name()
    a = train_local(copy.deepcopy(model), train_ds, val_ds, spec(), seed=7)
    b = train_local(copy.deepcopy(model), train_ds, val_ds, spec(), seed=7)
    assert torch.equal(a.update["fc2.weight"], b.update["fc2.weight"])


def test_train_local_seed_changes_weights() -> None:
    train_ds, val_ds = make_tiny_dataset(seed=0)
    model = build_model_by_name()
    a = train_local(copy.deepcopy(model), train_ds, val_ds, spec(), seed=1)
    b = train_local(copy.deepcopy(model), train_ds, val_ds, spec(), seed=2)
    assert not torch.equal(a.update["fc2.weight"], b.update["fc2.weight"])


def test_update_is_cpu_detached() -> None:
    train_ds, val_ds = make_tiny_dataset(seed=0, n=16, val_n=8)
    outcome = train_local(build_model_by_name(), train_ds, val_ds, spec(epochs=1), seed=0)
    for tensor in outcome.update.values():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad


def test_train_local_adam() -> None:
    train_ds, val_ds = make_tiny_dataset(seed=0, n=16, val_n=8)
    adam_spec = TrainSpec(
        local_epochs=1,
        batch_size=16,
        optimizer=OptimizerSpec(name="adam", lr=0.1),
        seed=42,
    )
    outcome = train_local(build_model_by_name(), train_ds, val_ds, adam_spec, seed=0)
    assert outcome.num_samples == 16
    assert set(outcome.metrics) == {"loss", "val_loss", "val_accuracy"}
    assert "fc2.weight" in outcome.update


def build_model_by_name():  # type: ignore[no-untyped-def]
    from flaw.config import ModelSection

    return build_model(ModelSection(name="mnist_cnn"))
