import pytest
import torch

from flaw.protocol import (
    PROTOCOL_VERSION,
    Assignment,
    BlobRef,
    ProtocolError,
    RegisterResponse,
    check_protocol_version,
    compute_config_hash,
    deserialize_state_dict,
    serialize_state_dict,
    validate_blob_id,
)

SHA256 = "0" * 64


def test_blob_id_valid() -> None:
    assert validate_blob_id("r0001/global") == "r0001/global"
    assert validate_blob_id("r0042/node-lab-pc.1") == "r0042/node-lab-pc.1"


@pytest.mark.parametrize(
    "bad",
    [
        "r1/global",
        "r0001/../etc",
        "r0001/global/",
        "x0001/global",
        "r0001/node-",
        "r0001/other",
        "r0001/global\n",
    ],
)
def test_blob_id_invalid(bad: str) -> None:
    with pytest.raises(ProtocolError):
        validate_blob_id(bad)


def test_blob_ref_rejects_bad_id() -> None:
    with pytest.raises(ProtocolError):
        BlobRef(blob_id="nope", sha256="0" * 64, size=1)


@pytest.mark.parametrize("bad_len", [0, 63, 65])
def test_blob_ref_rejects_bad_sha256_length(bad_len: int) -> None:
    with pytest.raises(ProtocolError):
        BlobRef(blob_id="r0001/global", sha256="a" * bad_len, size=1)
    with pytest.raises(ProtocolError):
        BlobRef.model_validate({"blob_id": "r0001/global", "sha256": "a" * bad_len, "size": 1})


def test_protocol_error_is_not_a_value_error() -> None:
    # Pydantic v2 wraps ValueError raised in validators into ValidationError; every
    # ProtocolError site relies on the type subclassing Exception directly so the
    # error propagates unwrapped (raises(ProtocolError) tests would otherwise break).
    assert issubclass(ProtocolError, Exception)
    assert not issubclass(ProtocolError, ValueError)


def test_blob_ref_defaults() -> None:
    ref = BlobRef(blob_id="r0001/global", sha256=SHA256, size=10)
    assert ref.format == "torch-state-dict-v1"
    assert ref.dtype == "fp32"


def test_register_response_optional_fields() -> None:
    resp = RegisterResponse(accepted=False)
    assert resp.reason is None
    assert resp.run is None
    assert resp.shard is None
    assigned = RegisterResponse(
        accepted=True, run={"name": "run", "current_round": 0, "phase": "waiting"}, shard=2
    )
    assert assigned.shard == 2


def test_protocol_version_check() -> None:
    check_protocol_version(PROTOCOL_VERSION)  # no raise
    with pytest.raises(ProtocolError, match="protocol_version"):
        check_protocol_version(PROTOCOL_VERSION + 1)


def test_config_hash_stable_and_key_order_independent() -> None:
    p1 = {"model": {"name": "a"}, "training": {"lr": 1}, "data": {"name": "d", "scheme": "iid"}}
    p2 = {"data": {"scheme": "iid", "name": "d"}, "training": {"lr": 1}, "model": {"name": "a"}}
    assert compute_config_hash(p1) == compute_config_hash(p2)
    assert len(compute_config_hash(p1)) == 64


def test_config_hash_changes_on_relevant_fields() -> None:
    base = {"model": {"name": "a"}, "training": {"lr": 1}, "data": {"name": "d", "scheme": "iid"}}
    changed = {
        "model": {"name": "b"},
        "training": {"lr": 1},
        "data": {"name": "d", "scheme": "iid"},
    }
    assert compute_config_hash(base) != compute_config_hash(changed)


def test_state_dict_serialization_roundtrip() -> None:
    sd = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor(3.0)}
    out = deserialize_state_dict(serialize_state_dict(sd))
    assert set(out) == {"w", "b"}
    assert torch.equal(out["w"], sd["w"])
    assert torch.equal(out["b"], sd["b"])


def test_assignment_roundtrip() -> None:
    a = Assignment(
        round=3,
        train_spec={
            "local_epochs": 1,
            "batch_size": 8,
            "optimizer": {"name": "sgd", "lr": 0.1, "momentum": 0.0},
            "seed": 42,
        },
        weights={"blob_id": "r0003/global", "sha256": "0" * 64, "size": 10},
    )
    assert Assignment.model_validate_json(a.model_dump_json()).weights.blob_id == "r0003/global"
