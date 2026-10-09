"""Central test-set evaluation, run on the orchestrator (CPU by design).

The central test set is the orchestrator's held-out check on the aggregated
global weights.  It is optional (``evaluation.central_test.path`` unset means
no central set), runs on the CPU -- the orchestrator container has no GPU --
and never mutates the state dict it is handed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from flaw.config import FlawConfig
from flaw.models import build_model
from flaw.node.data import infer_in_channels, load_folder_dataset
from flaw.node.trainer import evaluate


def default_central_factory(cfg: FlawConfig) -> Callable[[], Dataset[Any]] | None:
    """Dataset factory for ``evaluation.central_test.path`` (None: no central test set)."""
    path = cfg.evaluation.central_test.path
    if path is None:
        return None
    channels = infer_in_channels(cfg)
    return lambda: load_folder_dataset(Path(path), channels)


def run_central_eval(
    cfg: FlawConfig,
    state_dict: Mapping[str, torch.Tensor],
    dataset_factory: Callable[[], Dataset[Any]],
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate ``state_dict`` on the central test set. Pure w.r.t. its inputs."""
    model = build_model(cfg.model)
    model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
    model.eval()
    return evaluate(model, dataset_factory(), batch_size, "cpu")
