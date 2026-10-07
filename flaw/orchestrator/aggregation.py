"""Sample-weighted FedAvg aggregation with safety guards."""

from collections.abc import Mapping, Sequence

import torch


class AggregationError(Exception):
    """Raised when updates cannot be safely aggregated."""


def aggregate_weighted(
    updates: Sequence[tuple[Mapping[str, torch.Tensor], int]],
    reference: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not updates:
        raise AggregationError("no updates to aggregate")
    total = sum(n for _, n in updates)
    if total <= 0:
        raise AggregationError("total sample count must be positive")

    ref_keys = set(reference)
    for sd, _ in updates:
        if set(sd) != ref_keys:
            missing = ref_keys - set(sd)
            extra = set(sd) - ref_keys
            raise AggregationError(
                f"state_dict keys mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
            )
        for key, tensor in sd.items():
            if tuple(tensor.shape) != tuple(reference[key].shape):
                raise AggregationError(
                    f"shape mismatch for '{key}': "
                    f"{tuple(tensor.shape)} != {tuple(reference[key].shape)}"
                )
            if not tensor.is_floating_point():
                raise AggregationError(f"non-float tensor '{key}' not supported")
            if not bool(torch.isfinite(tensor).all()):
                raise AggregationError(f"non-finite values in '{key}'")

    result: dict[str, torch.Tensor] = {}
    for key in reference:
        acc: torch.Tensor | None = None
        for sd, n in updates:
            term = sd[key].to(torch.float64) * (n / total)
            acc = term if acc is None else acc + term
        assert acc is not None
        result[key] = acc.to(reference[key].dtype)
    return result
