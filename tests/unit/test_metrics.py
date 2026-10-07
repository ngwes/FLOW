import json
from pathlib import Path

import pytest

from flaw.orchestrator.metrics import (
    EarlyStopper,
    MissingMetric,
    aggregate_metrics,
    write_metrics_line,
)


def test_aggregate_metrics_weighted() -> None:
    out = aggregate_metrics([({"loss": 1.0, "val_accuracy": 0.5}, 1), ({"loss": 3.0}, 3)])
    assert out["loss"] == pytest.approx(2.5)
    assert out["val_accuracy"] == pytest.approx(0.5)  # only the entry that has it


def test_aggregate_metrics_empty() -> None:
    assert aggregate_metrics([]) == {}


def test_write_metrics_line_appends_parseable(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "metrics.jsonl"
    write_metrics_line(path, {"type": "round", "round": 1})
    write_metrics_line(path, {"type": "round", "round": 2})
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [line["round"] for line in lines] == [1, 2]


def make_stopper(**kw: object) -> EarlyStopper:
    base: dict[str, object] = {
        "metric": "val_accuracy",
        "target": None,
        "patience": 3,
        "min_delta": 0.001,
        "max_rounds": 100,
    }
    base.update(kw)
    return EarlyStopper(**base)  # type: ignore[arg-type]


def test_target_reached() -> None:
    stopper = make_stopper(target=0.9)
    decision = stopper.update(1, {"val_accuracy": 0.95})
    assert decision.stop and decision.reason == "target_reached"
    assert stopper.best_metric == 0.95 and stopper.best_round == 1


def test_patience_exhausted() -> None:
    stopper = make_stopper(patience=2)
    assert not stopper.update(1, {"val_accuracy": 0.5}).stop
    assert not stopper.update(2, {"val_accuracy": 0.5}).stop  # no improvement, counter=1
    decision = stopper.update(3, {"val_accuracy": 0.5})  # counter=2 → stop
    assert decision.stop and decision.reason == "patience"


def test_min_delta_boundary_is_not_improvement() -> None:
    stopper = make_stopper(patience=2, min_delta=0.001)
    stopper.update(1, {"val_accuracy": 0.500})
    assert stopper.update(2, {"val_accuracy": 0.501}).reason is None  # exactly +min_delta: no
    decision = stopper.update(3, {"val_accuracy": 0.5001})  # still no → counter=2
    assert decision.stop and decision.reason == "patience"


def test_improvement_resets_counter() -> None:
    stopper = make_stopper(patience=2)
    stopper.update(1, {"val_accuracy": 0.5})
    stopper.update(2, {"val_accuracy": 0.4})  # counter=1
    assert not stopper.update(3, {"val_accuracy": 0.6}).stop  # improves → reset
    assert stopper.best_round == 3


def test_max_rounds() -> None:
    stopper = make_stopper(patience=100, max_rounds=3)
    assert not stopper.update(1, {"val_accuracy": 0.1}).stop
    assert not stopper.update(2, {"val_accuracy": 0.2}).stop
    decision = stopper.update(3, {"val_accuracy": 0.3})
    assert decision.stop and decision.reason == "max_rounds"


def test_missing_metric_raises() -> None:
    with pytest.raises(MissingMetric):
        make_stopper().update(1, {"loss": 0.5})


def test_manual_stop() -> None:
    stopper = make_stopper()
    stopper.request_manual_stop()
    decision = stopper.update(1, {"val_accuracy": 0.9})
    assert decision.stop and decision.reason == "manual"
