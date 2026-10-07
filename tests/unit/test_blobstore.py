import hashlib
from pathlib import Path

import pytest

from flaw.blobstore import (
    BlobNotFound,
    BlobStore,
    ChecksumMismatch,
    global_blob_id,
    node_blob_id,
    round_prefix,
)
from flaw.protocol import ProtocolError


def test_id_helpers() -> None:
    assert round_prefix(7) == "r0007"
    assert global_blob_id(42) == "r0042/global"
    assert node_blob_id(42, "lab-pc.1") == "r0042/node-lab-pc.1"


def test_put_get_roundtrip(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    ref = store.put("r0001/global", b"weights")
    assert ref.blob_id == "r0001/global"
    assert ref.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert ref.size == 7
    assert ref.format == "torch-state-dict-v1"
    assert store.get("r0001/global") == b"weights"
    assert store.path_for("r0001/global") == tmp_path / "r0001" / "global.pt"


def test_get_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(BlobNotFound):
        BlobStore(tmp_path).get("r0001/global")


def test_exists(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    store.put("r0001/global", b"x")
    assert store.exists("r0001/global")
    assert not store.exists("r0002/global")


def test_verify_detects_corruption(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    ref = store.put("r0001/global", b"weights")
    store.verify(ref, b"weights")  # no raise
    with pytest.raises(ChecksumMismatch):
        store.verify(ref, b"tampered")
    bad_size = ref.model_copy(update={"size": 999})
    with pytest.raises(ChecksumMismatch):
        store.verify(bad_size, b"weights")


@pytest.mark.parametrize("bad", ["r0001/../etc", "r1/global", "/abs/path"])
def test_bad_blob_ids_rejected(tmp_path: Path, bad: str) -> None:
    store = BlobStore(tmp_path)
    with pytest.raises(ProtocolError):
        store.path_for(bad)
    with pytest.raises(ProtocolError):
        store.put(bad, b"x")
    with pytest.raises(ProtocolError):
        store.get(bad)


def test_gc_retention_and_protection(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    ids = [global_blob_id(r) for r in range(1, 6)]
    for blob_id in ids:
        store.put(blob_id, b"x")
    protected = tmp_path / "best.pt"
    protected.write_bytes(b"best")
    (tmp_path / "junk").mkdir()

    deleted = store.gc(current_round=5, keep_rounds=3)  # deletes rounds <= 5-3 = 2
    assert deleted == sorted([ids[0], ids[1]])
    assert not store.exists(ids[0])
    assert not store.exists(ids[1])
    assert store.exists(ids[2]) and store.exists(ids[4])
    assert protected.read_bytes() == b"best"
    assert (tmp_path / "junk").exists()


def test_gc_noop_when_nothing_old(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    store.put("r0004/global", b"x")
    assert store.gc(current_round=5, keep_rounds=3) == []


def test_gc_empty_store(tmp_path: Path) -> None:
    assert BlobStore(tmp_path).gc(current_round=1, keep_rounds=1) == []


def test_gc_missing_root_returns_empty(tmp_path: Path) -> None:
    assert BlobStore(tmp_path / "missing").gc(current_round=1, keep_rounds=1) == []


def test_gc_skips_file_named_like_round_dir(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    stray = tmp_path / "r0001"
    stray.write_bytes(b"not a directory")

    # cutoff is 5 - 3 = 2, so a directory named r0001 would be deleted;
    # a plain file must be skipped instead of crashing with NotADirectoryError.
    assert store.gc(current_round=5, keep_rounds=3) == []
    assert stray.read_bytes() == b"not a directory"
