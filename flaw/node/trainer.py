"""Local training on a node: device pick, evaluation, the round's training loop.

A node receives an :class:`~flaw.protocol.TrainSpec` plus the current global
weights, trains locally for ``local_epochs`` and returns a :class:`TrainOutcome`
whose ``update`` is the full state dict, detached on the CPU -- the only thing
that leaves the node.  Seeding comes from the ``seed`` parameter (torch and the
DataLoader's shuffling generator), so the same seed on the same data and
weights reproduces the same update exactly.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Sized
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from flaw.protocol import TrainSpec


@dataclass
class TrainOutcome:
    """Result of one local training round, ready to be sent to the orchestrator.

    ``update`` holds the full state dict as detached CPU tensors, ``metrics``
    exactly ``loss`` (mean training loss), ``val_loss`` and ``val_accuracy``,
    and ``duration_s`` how long the round took.
    """

    update: dict[str, torch.Tensor]
    num_samples: int
    metrics: dict[str, float]
    duration_s: float


def pick_device(explicit: str | None) -> tuple[str, str]:
    """``(device, device_name)`` for training: honor *explicit*, else auto-pick.

    An explicit device string is used as-is, named after itself.  ``None``
    picks ``cuda:0`` when CUDA is available, otherwise the CPU with the
    processor name for display.
    """
    if explicit is not None:
        return explicit, explicit
    if torch.cuda.is_available():
        return "cuda:0", torch.cuda.get_device_name(0)
    return "cpu", platform.processor() or "CPU"


def evaluate(
    model: nn.Module, dataset: Dataset[Any], batch_size: int, device: str
) -> dict[str, float]:
    """No-grad pass over *dataset*: mean cross-entropy and accuracy of *model*.

    ``val_loss`` is the mean loss over all samples (not of the batch means) and
    ``val_accuracy`` the fraction of correctly classified samples.
    """
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss_sum += float(criterion(logits, targets).item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += len(targets)
    return {"val_loss": loss_sum / total, "val_accuracy": correct / total}


def train_local(
    model: nn.Module,
    train_ds: Dataset[Any],
    val_ds: Dataset[Any],
    spec: TrainSpec,
    device: str = "cpu",
    seed: int = 42,
) -> TrainOutcome:
    """Train *model* for ``spec.local_epochs`` epochs and evaluate it on *val_ds*.

    Shuffles ``train_ds`` into batches of ``spec.batch_size`` with the spec's
    optimizer (``sgd`` with momentum, or ``adam``) and cross-entropy loss, then
    returns the trained weights, the training loss and the validation metrics.
    """
    started = time.monotonic()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, generator=generator)
    model.to(device)
    model.train()
    if spec.optimizer.name == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=spec.optimizer.lr, momentum=spec.optimizer.momentum
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=spec.optimizer.lr)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    batches = 0
    for _ in range(spec.local_epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
    metrics = {"loss": loss_sum / batches}
    metrics.update(evaluate(model, val_ds, spec.batch_size, device))
    update = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    # torch's Dataset stub declares no __len__; every concrete dataset has one.
    num_samples = len(cast("Sized", train_ds))
    return TrainOutcome(
        update=update,
        num_samples=num_samples,
        metrics=metrics,
        duration_s=time.monotonic() - started,
    )
