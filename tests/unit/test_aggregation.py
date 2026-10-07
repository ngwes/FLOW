import pytest
import torch

from flaw.orchestrator.aggregation import AggregationError, aggregate_weighted


def ref_sd() -> dict[str, torch.Tensor]:
    return {"w": torch.zeros(2), "b": torch.zeros(1)}


def test_exact_weighted_average() -> None:
    u1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    u2 = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([4.0])}
    out = aggregate_weighted([(u1, 1), (u2, 3)], ref_sd())
    assert torch.allclose(out["w"], torch.tensor([2.5, 3.5]))
    assert torch.allclose(out["b"], torch.tensor([3.0]))


def test_single_update_is_identity() -> None:
    u1 = {"w": torch.tensor([1.5, -2.0]), "b": torch.tensor([0.25])}
    out = aggregate_weighted([(u1, 5)], ref_sd())
    assert torch.allclose(out["w"], u1["w"])
    assert torch.allclose(out["b"], u1["b"])


def test_equal_weights_plain_mean() -> None:
    u1 = {"w": torch.tensor([0.0, 0.0]), "b": torch.tensor([0.0])}
    u2 = {"w": torch.tensor([2.0, 4.0]), "b": torch.tensor([6.0])}
    out = aggregate_weighted([(u1, 10), (u2, 10)], ref_sd())
    assert torch.allclose(out["w"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(out["b"], torch.tensor([3.0]))


def test_deterministic_and_reference_unchanged() -> None:
    ref = ref_sd()
    u1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([1.0])}
    a = aggregate_weighted([(u1, 1)], ref)
    b = aggregate_weighted([(u1, 1)], ref)
    assert torch.equal(a["w"], b["w"])
    assert torch.equal(ref["w"], torch.zeros(2))


def test_empty_raises() -> None:
    with pytest.raises(AggregationError, match="no updates"):
        aggregate_weighted([], ref_sd())


def test_zero_total_samples_raises() -> None:
    u1 = {"w": torch.zeros(2), "b": torch.zeros(1)}
    with pytest.raises(AggregationError, match="positive"):
        aggregate_weighted([(u1, 0)], ref_sd())


def test_key_mismatch_raises() -> None:
    missing = {"w": torch.zeros(2)}
    with pytest.raises(AggregationError, match="missing"):
        aggregate_weighted([(missing, 1)], ref_sd())


def test_extra_keys_raise() -> None:
    extra = {"w": torch.zeros(2), "b": torch.zeros(1), "z": torch.zeros(1)}
    with pytest.raises(AggregationError, match="extra"):
        aggregate_weighted([(extra, 1)], ref_sd())


def test_shape_mismatch_raises() -> None:
    bad = {"w": torch.zeros(3), "b": torch.zeros(1)}
    with pytest.raises(AggregationError, match="shape"):
        aggregate_weighted([(bad, 1)], ref_sd())


def test_nan_rejected() -> None:
    bad = {"w": torch.tensor([float("nan"), 0.0]), "b": torch.zeros(1)}
    with pytest.raises(AggregationError, match="non-finite"):
        aggregate_weighted([(bad, 1)], ref_sd())


def test_inf_rejected() -> None:
    bad = {"w": torch.tensor([float("inf"), 0.0]), "b": torch.zeros(1)}
    with pytest.raises(AggregationError, match="non-finite"):
        aggregate_weighted([(bad, 1)], ref_sd())


def test_non_float_rejected() -> None:
    bad = {"w": torch.zeros(2, dtype=torch.int64), "b": torch.zeros(1)}
    with pytest.raises(AggregationError, match="non-float"):
        aggregate_weighted([(bad, 1)], ref_sd())
