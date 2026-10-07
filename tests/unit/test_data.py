from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from flaw.config import load_config
from flaw.node.data import (
    _labels,
    infer_in_channels,
    load_dataset,
    load_folder_dataset,
    make_tiny_dataset,
    partition_indices,
    prepare_node_data,
)

CFG_SHARED = """
run: {name: t}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 3, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 16}
"""


def cfg(tmp_path: Path, text: str = CFG_SHARED):  # type: ignore[no-untyped-def]
    p = tmp_path / "flaw.yaml"
    p.write_text(text, encoding="utf-8")
    return load_config(p)


def test_tiny_dataset_deterministic_and_shaped() -> None:
    train_a, val_a = make_tiny_dataset(seed=0)
    train_b, val_b = make_tiny_dataset(seed=0)
    assert len(train_a) == 256 and len(val_a) == 64
    assert torch.equal(train_a.tensors[0], train_b.tensors[0])
    x, y = train_a[0]
    assert x.shape == (1, 28, 28) and 0 <= int(y) < 10


def test_partition_iid_sizes_and_coverage() -> None:
    labels = [i % 10 for i in range(90)]
    parts = partition_indices(labels, num_shards=3, scheme="iid", alpha=0.5, seed=1)
    sizes = sorted(len(p) for p in parts)
    assert sizes == [30, 30, 30]
    assert sorted(i for p in parts for i in p) == list(range(90))


def test_partition_iid_balance() -> None:
    labels = [i % 10 for i in range(300)]
    parts = partition_indices(labels, num_shards=3, scheme="iid", alpha=0.5, seed=7)
    for part in parts:
        counts = [sum(1 for i in part if labels[i] == c) for c in range(10)]
        assert max(counts) / len(part) < 0.2  # well mixed


def test_partition_deterministic() -> None:
    labels = [i % 10 for i in range(90)]
    a = partition_indices(labels, 3, "dirichlet", 0.5, seed=3)
    b = partition_indices(labels, 3, "dirichlet", 0.5, seed=3)
    assert a == b


def test_partition_dirichlet_skew_monotonic() -> None:
    labels = [i % 10 for i in range(300)]
    low = partition_indices(labels, 3, "dirichlet", 0.1, seed=5)
    high = partition_indices(labels, 3, "dirichlet", 10.0, seed=5)

    def skew(parts: list[list[int]]) -> float:
        shares = []
        for part in parts:
            if not part:
                continue
            counts = [sum(1 for i in part if labels[i] == c) for c in range(10)]
            shares.append(max(counts) / len(part))
        return sum(shares) / len(shares)

    assert skew(low) > skew(high)


def test_load_dataset_tiny_and_unknown(tmp_path: Path) -> None:
    assert len(load_dataset("tiny", tmp_path)) == 256
    with pytest.raises(ValueError, match="unknown"):
        load_dataset("bogus", tmp_path)


def test_prepare_node_data_shared(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    train, val = prepare_node_data(config, shard=0, seed=42)
    assert len(train) + len(val) == 86  # 256/3 rounded down, remainder in last? see impl
    assert abs(len(val) - 0.25 * (len(train) + len(val))) <= 1


def test_prepare_federated_uses_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fed = tmp_path / "localdata"
    (fed / "train").mkdir(parents=True)
    (fed / "val").mkdir()
    text = CFG_SHARED.replace(
        "data: {name: tiny, num_shards: 3, scheme: iid, val_fraction: 0.25}",
        f"data: {{path: {fed.as_posix()}, val_fraction: 0.5}}",
    )
    config = cfg(tmp_path, text)
    train_a, val_a = make_tiny_dataset(seed=0, n=8, val_n=8)
    monkeypatch.setattr(
        "flaw.node.data._load_folder",
        lambda path, channels: train_a if "train" in str(path) else val_a,
    )
    train, val = prepare_node_data(config, shard=0, seed=42)
    assert len(train) == 8 and len(val) == 8


def test_infer_in_channels(tmp_path: Path) -> None:
    assert infer_in_channels(cfg(tmp_path)) == 1  # mnist_cnn
    cifar = CFG_SHARED.replace("model: {name: mnist_cnn}", "model: {name: cifar_cnn}")
    assert infer_in_channels(cfg(tmp_path, cifar)) == 3


def test_load_folder_dataset_grayscale(tmp_path: Path) -> None:
    from PIL import Image

    root = tmp_path / "imgs"
    for cls in ("a", "b"):
        (root / cls).mkdir(parents=True)
        Image.new("L", (28, 28), color=128).save(root / cls / "0.png")
    ds = load_folder_dataset(root, channels=1)  # ImageFolder would give RGB without this
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert 0.0 <= float(x.min()) <= float(x.max()) <= 1.0
    assert int(y) in {0, 1}


# --- Branch-coverage tests for data.py helpers ----------------------------------


class _TargetsStub:
    """Minimal torchvision-like dataset: labels live on a ``.targets`` attribute."""

    def __init__(self, targets: list[int]) -> None:
        self.targets = targets


def test_labels_reads_targets_attribute() -> None:
    assert _labels(_TargetsStub([1, 0, 2])) == [1, 0, 2]


def test_labels_without_targets_raises() -> None:
    with pytest.raises(ValueError):
        _labels(Dataset())  # no .targets and not a TensorDataset


def test_infer_in_channels_without_conv_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flaw.node.data.build_model", lambda cfg: torch.nn.Linear(4, 2))
    with pytest.raises(ValueError):
        infer_in_channels(cfg(tmp_path))


def test_partition_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError):
        partition_indices(
            [i % 2 for i in range(10)], num_shards=2, scheme="bogus", alpha=0.5, seed=0
        )


def _recording_dataset_class(records: list[tuple[object, object, object]]) -> type:
    class RecordingDataset:
        def __init__(self, root: object, train: object, download: object) -> None:
            records.append((root, train, download))

    return RecordingDataset


@pytest.mark.parametrize(("name", "attribute"), [("mnist", "MNIST"), ("cifar10", "CIFAR10")])
def test_load_dataset_torchvision_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, attribute: str
) -> None:
    records: list[tuple[object, object, object]] = []
    stub_cls = _recording_dataset_class(records)
    monkeypatch.setattr(f"torchvision.datasets.{attribute}", stub_cls)
    dataset = load_dataset(name, tmp_path)
    assert isinstance(dataset, stub_cls)
    assert records == [(tmp_path, True, True)]


def test_load_folder_dataset_rgb(tmp_path: Path) -> None:
    from PIL import Image

    root = tmp_path / "imgs_rgb"
    for cls in ("a", "b"):
        (root / cls).mkdir(parents=True)
        Image.new("L", (28, 28), color=128).save(root / cls / "0.png")
    ds = load_folder_dataset(root, channels=3)  # no Grayscale: ImageFolder's RGB stays
    x, y = ds[0]
    assert x.shape == (3, 28, 28)
    assert int(y) in {0, 1}
