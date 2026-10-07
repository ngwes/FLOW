"""Built-in model registry and import-path model loading.

Built-in CNNs are registered under short names (``mnist_cnn``, ``cifar_cnn``);
any other network is loaded from a python import path such as ``my_net.Net``,
with ``model.module_path`` prepended to ``sys.path`` first if set.  Every
built-in model holds only floating-point parameters and buffers, so a plain
mean of state_dicts is well defined (spec 5.3 aggregation v1: no integer
buffers such as BatchNorm's ``num_batches_tracked``).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from flaw.config import ModelSection


class ModelNotFound(Exception):
    """Raised when a model name or import path cannot be resolved."""


class MnistCNN(nn.Module):
    """Small CNN for 1x28x28 grayscale input, returning 10 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a 28x28 batch through two conv/pool stacks and two linears."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # nn.Module.__call__ is typed as returning Any; the cast restores the contract.
        return cast("torch.Tensor", self.fc2(torch.relu(self.fc1(x))))


class CifarCNN(nn.Module):
    """Small CNN for 3x32x32 colour input, returning 10 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a 32x32 batch through two conv/pool stacks and two linears."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # nn.Module.__call__ is typed as returning Any; the cast restores the contract.
        return cast("torch.Tensor", self.fc2(torch.relu(self.fc1(x))))


REGISTRY: dict[str, Callable[[], torch.nn.Module]] = {
    "mnist_cnn": MnistCNN,
    "cifar_cnn": CifarCNN,
}


def build_model(model_cfg: ModelSection) -> torch.nn.Module:
    """Build a fresh model described by *model_cfg* or raise :class:`ModelNotFound`.

    ``model_cfg.name`` looks the model up in :data:`REGISTRY`;
    ``model_cfg.class_path`` (e.g. ``"my_net.Net"``) imports the module and
    class instead, after prepending ``model_cfg.module_path`` to ``sys.path``.
    """
    if model_cfg.name is not None:
        try:
            factory = REGISTRY[model_cfg.name]
        except KeyError as exc:
            raise ModelNotFound(model_cfg.name) from exc
        return factory()

    class_path = model_cfg.class_path
    assert class_path is not None  # ModelSection checks exactly one source is set
    if model_cfg.module_path is not None:
        sys.path.insert(0, model_cfg.module_path)
    try:
        module_name, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = cast("Callable[[], torch.nn.Module]", getattr(module, cls_name))
    except (ImportError, AttributeError, ValueError) as exc:
        raise ModelNotFound(f"{class_path}: {exc}") from exc
    return cls()
