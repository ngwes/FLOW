"""Metrics journal and early-stopping rules (spec 4.6/4.7).

Per-round samples from each node are aggregated into one run-level metric
dict, appended as a JSON line to ``metrics.jsonl``, and fed to an
:class:`EarlyStopper` that decides when a run should stop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def aggregate_metrics(entries: Sequence[tuple[Mapping[str, float], int]]) -> dict[str, float]:
    """Sample-weighted mean per metric key.

    Only entries that contain a key contribute to it, so a node that reports
    fewer metrics never dilutes the ones it does report. No entries → ``{}``.
    """
    keys = {key for metrics, _ in entries for key in metrics}
    return {
        key: sum(n * metrics[key] for metrics, n in entries if key in metrics)
        / sum(n for metrics, n in entries if key in metrics)
        for key in keys
    }


def write_metrics_line(path: Path, line: Mapping[str, object]) -> None:
    """Append one JSON line to the metrics journal, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(line), sort_keys=True) + "\n")


class MissingMetric(Exception):
    """Raised when a round's metrics lack the tracked early-stop metric."""


@dataclass(frozen=True)
class EarlyStopDecision:
    """Whether (and why) a run should stop after a round."""

    stop: bool
    reason: str | None = None


class EarlyStopper:
    """Track one metric across rounds and decide when to stop.

    Stops when the tracked metric reaches ``target``, when it fails to improve
    by more than ``min_delta`` for ``patience`` consecutive rounds, when
    ``max_rounds`` is reached, or on ``request_manual_stop``.
    """

    def __init__(
        self,
        metric: str,
        target: float | None,
        patience: int,
        min_delta: float,
        max_rounds: int,
    ) -> None:
        self.metric = metric
        self.target = target
        self.patience = patience
        self.min_delta = min_delta
        self.max_rounds = max_rounds
        self.best_metric: float | None = None
        self.best_round: int | None = None
        self._since_improvement = 0
        self._manual = False

    def request_manual_stop(self) -> None:
        """Ask for a stop at the next :meth:`update` call."""
        self._manual = True

    def update(self, round_no: int, metrics: Mapping[str, float]) -> EarlyStopDecision:
        """Record one round's metrics and decide whether to stop.

        Raises :class:`MissingMetric` when the tracked metric is absent.
        """
        if self._manual:
            return EarlyStopDecision(stop=True, reason="manual")
        if self.metric not in metrics:
            raise MissingMetric(f"metric {self.metric!r} missing from round {round_no}")
        value = metrics[self.metric]
        if self.best_metric is None or value > self.best_metric + self.min_delta:
            self.best_metric = value
            self.best_round = round_no
            self._since_improvement = 0
        else:
            self._since_improvement += 1
        if self.target is not None and value >= self.target:
            return EarlyStopDecision(stop=True, reason="target_reached")
        if self._since_improvement >= self.patience:
            return EarlyStopDecision(stop=True, reason="patience")
        if round_no >= self.max_rounds:
            return EarlyStopDecision(stop=True, reason="max_rounds")
        return EarlyStopDecision(stop=False)
