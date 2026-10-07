"""Dataset loading and sharding for FL@W nodes.

Shared mode (``data.name``) downloads (or synthesises) one dataset and cuts it
into ``num_shards`` slices with :func:`partition_indices`: ``iid`` for
well-mixed shards, ``dirichlet`` for the deliberately skewed, non-IID case.
Federated mode (``data.path``) reads each node's own ``train/`` and ``val/``
class folders via :func:`load_folder_dataset` and never shards.  ``tiny`` is
generated in-process, so tests and ``flaw simulate`` run offline and fast;
torchvision is imported lazily for the same reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from flaw.config import FlawConfig
from flaw.models import build_model

_TINY_CLASSES = 10
_TINY_IMAGE_SIZE = 28
_TINY_BLOCK = 5  # bright 5x5 block marking each class in the tiny dataset


def partition_indices(
    labels: Sequence[int], num_shards: int, scheme: str, alpha: float, seed: int
) -> list[list[int]]:
    """Split ``range(len(labels))`` into *num_shards* lists of indices.

    ``"iid"`` shuffles a seeded permutation and cuts it into near-equal
    contiguous chunks (earlier shards take the remainder).  ``"dirichlet"``
    draws each class's shard proportions from ``Dirichlet(alpha)``: low alpha
    means each class lands mostly on one shard (extreme skew), high alpha means
    nearly mixed.  Pure and deterministic: every index appears exactly once and
    the same seed always yields the same partition.
    """
    if scheme == "iid":
        return _partition_iid(len(labels), num_shards, seed)
    if scheme == "dirichlet":
        return _partition_dirichlet(labels, num_shards, alpha, seed)
    raise ValueError(f"unknown partition scheme: {scheme!r}")


def _partition_iid(n: int, num_shards: int, seed: int) -> list[list[int]]:
    rng = numpy.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(n)]
    parts: list[list[int]] = []
    start = 0
    for shard in range(num_shards):
        size = n // num_shards + (1 if shard < n % num_shards else 0)
        parts.append(perm[start : start + size])
        start += size
    return parts


def _partition_dirichlet(
    labels: Sequence[int], num_shards: int, alpha: float, seed: int
) -> list[list[int]]:
    rng = numpy.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)

    parts: list[list[int]] = [[] for _ in range(num_shards)]
    for class_id in sorted(by_class):
        indices = by_class[class_id]
        props = [float(p) for p in rng.dirichlet([alpha] * num_shards)]
        exact = [p * len(indices) for p in props]
        counts = [math.floor(value) for value in exact]
        # floor leaves len(indices) - sum(counts) items unassigned (< num_shards);
        # hand them out largest fractional part first so allocation is deterministic.
        remainder = len(indices) - sum(counts)
        if remainder > 0:
            order = sorted(range(num_shards), key=lambda s: exact[s] - counts[s], reverse=True)
            for shard in order[:remainder]:
                counts[shard] += 1
        start = 0
        for shard, count in enumerate(counts):
            parts[shard].extend(indices[start : start + count])
            start += count
    return parts


def make_tiny_dataset(
    seed: int = 0, n: int = 256, val_n: int = 64
) -> tuple[Dataset[Any], Dataset[Any]]:
    """Deterministic synthetic 10-class ``1x28x28`` ``(train, val)`` dataset pair.

    Each class paints a bright block at its own position on a seeded noise
    background, so a small CNN can actually learn to separate the classes.
    Generated in-process: offline, fast, and the stand-in every test and
    ``flaw simulate`` uses instead of MNIST/CIFAR.
    """
    return _tiny_split(seed, n), _tiny_split(seed + 1, val_n)


def _tiny_split(seed: int, n: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n, dtype=torch.long) % _TINY_CLASSES
    images = torch.rand((n, 1, _TINY_IMAGE_SIZE, _TINY_IMAGE_SIZE), generator=generator) * 0.1
    for class_id in range(_TINY_CLASSES):
        row, column = divmod(class_id, 5)
        rows = slice(row * _TINY_BLOCK, (row + 1) * _TINY_BLOCK)
        cols = slice(column * _TINY_BLOCK, (column + 1) * _TINY_BLOCK)
        images[labels == class_id, 0, rows, cols] = 1.0
    return TensorDataset(images, labels)


def load_dataset(name: str, data_dir: Path) -> Dataset[Any]:
    """Built-in dataset *name*: ``tiny`` (offline), else the torchvision train split.

    ``mnist``/``cifar10`` download once into *data_dir*.  torchvision is
    imported lazily so importing this module stays fast and network-free.
    An unknown name raises :class:`ValueError`.
    """
    if name == "tiny":
        return make_tiny_dataset()[0]
    if name == "mnist":
        # torchvision ships no py.typed; mypy reports the first such import per file.
        from torchvision.datasets import MNIST  # type: ignore[import-untyped]

        return cast("Dataset[Any]", MNIST(root=data_dir, train=True, download=True))
    if name == "cifar10":
        from torchvision.datasets import CIFAR10

        return cast("Dataset[Any]", CIFAR10(root=data_dir, train=True, download=True))
    raise ValueError(f"unknown dataset: {name!r}")


def load_folder_dataset(path: Path, channels: int = 3) -> Dataset[Any]:
    """Every image under *path* (one class per subfolder) as tensors.

    ``ImageFolder`` converts images to RGB by default; ``channels == 1`` adds a
    ``Grayscale`` transform first, which a grayscale model such as ``mnist_cnn``
    needs.  The orchestrator's central evaluation uses this too.  v1
    limitation: images must already be model-input-sized (no resize).
    """
    return _load_folder(path, channels)


def _load_folder(path: Path, channels: int) -> Dataset[Any]:
    """torchvision ``ImageFolder`` behind the transform choice; monkeypatch target."""
    from torchvision import transforms  # type: ignore[import-untyped]
    from torchvision.datasets import ImageFolder

    tfms: list[Any] = [transforms.ToTensor()]
    if channels == 1:
        tfms.insert(0, transforms.Grayscale(num_output_channels=1))
    return cast("Dataset[Any]", ImageFolder(path, transform=transforms.Compose(tfms)))


def infer_in_channels(cfg: FlawConfig) -> int:
    """Input channel count of *cfg*'s model: the first ``nn.Conv2d`` in it.

    Federated data folders hold no channel metadata, so the model tells us how
    to load them (a grayscale model needs 1-channel input).  A model without a
    conv layer cannot say what it expects -> :class:`ValueError`.
    """
    import torch.nn as nn

    for module in build_model(cfg.model).modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)
    raise ValueError("model has no Conv2d layer; cannot infer input channels")


def prepare_node_data(cfg: FlawConfig, shard: int, seed: int) -> tuple[Dataset[Any], Dataset[Any]]:
    """The ``(train, val)`` datasets this node trains on, per the config's data mode.

    Shared mode: the full dataset is cut into ``num_shards`` with
    :func:`partition_indices`, this node keeps shard *shard*, and
    ``val_fraction`` of it becomes the validation split (carved with a
    ``seed + shard`` generator, so every node's split differs).  Federated
    mode: ``<path>/train`` and ``<path>/val`` folders; sharding settings are
    ignored because each node already owns its data.
    """
    if cfg.data.path is not None:
        channels = infer_in_channels(cfg)
        root = Path(cfg.data.path)
        return (
            load_folder_dataset(root / "train", channels),
            load_folder_dataset(root / "val", channels),
        )

    name = cfg.data.name
    assert name is not None  # DataSection validates exactly one of name/path
    full = load_dataset(name, Path(cfg.storage.data_dir))
    parts = partition_indices(
        _labels(full), cfg.data.num_shards, cfg.data.scheme, cfg.data.dirichlet_alpha, seed
    )
    shard_indices = parts[shard]
    rng = numpy.random.default_rng(seed + shard)
    # order holds POSITIONS into the shard; the nested Subsets address the full
    # dataset through Subset(full, shard_indices), so positions are what they
    # (and only they) accept -- full-dataset indices would be applied twice.
    order = [int(i) for i in rng.permutation(len(shard_indices))]
    val_size = int(cfg.data.val_fraction * len(shard_indices))
    shard_subset: Dataset[Any] = Subset(full, shard_indices)
    return Subset(shard_subset, order[val_size:]), Subset(shard_subset, order[:val_size])


def _labels(dataset: Dataset[Any]) -> list[int]:
    """Labels of *dataset*: torchvision keeps them on ``.targets``.

    ``TensorDataset`` (our tiny data) has none, so they come from its second
    tensor instead.  Anything else cannot be split by label -> error.
    """
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(label) for label in targets]
    if isinstance(dataset, TensorDataset):
        return [int(value) for value in dataset.tensors[1]]
    raise ValueError("cannot read labels: expected .targets or a TensorDataset")
