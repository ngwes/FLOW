"""On-disk blob store: sha256-tracked weight files with retention GC (spec 5.2).

Blobs (global weights and node updates) are plain files under
``root/r<NNNN>/``; the blob id maps to the path by appending ``.pt``, so the
forward slash in e.g. ``r0001/global`` becomes a directory separator.  Every
path derived from a blob id goes through :func:`flaw.protocol.validate_blob_id`
first, which makes directory traversal impossible.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from flaw.protocol import BlobRef, validate_blob_id


class BlobError(Exception):
    """Base class for blob store errors."""


class BlobNotFound(BlobError):
    """Raised when a blob id has no file in the store."""


class ChecksumMismatch(BlobError):
    """Raised when blob bytes do not match the expected size or sha256."""


def round_prefix(round_no: int) -> str:
    """Return the round directory name for *round_no*: ``r0007``."""
    return f"r{round_no:04d}"


def global_blob_id(round_no: int) -> str:
    """Return the global-weights blob id for *round_no*: ``r0007/global``."""
    return f"{round_prefix(round_no)}/global"


def node_blob_id(round_no: int, node_id: str) -> str:
    """Return the update blob id for *node_id* in *round_no*: ``r0007/node-<id>``."""
    return f"{round_prefix(round_no)}/node-{node_id}"


class BlobStore:
    """A local directory of immutable, content-addressed blob files."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, blob_id: str) -> Path:
        """Map a validated *blob_id* to its file path under the root."""
        validate_blob_id(blob_id)
        return self.root / (blob_id + ".pt")

    def put(self, blob_id: str, data: bytes) -> BlobRef:
        """Atomically write *data* under *blob_id* and return its reference."""
        path = self.path_for(blob_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return BlobRef(
            blob_id=blob_id,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )

    def get(self, blob_id: str) -> bytes:
        """Return the stored bytes for *blob_id* or raise :class:`BlobNotFound`."""
        path = self.path_for(blob_id)
        if not path.exists():
            raise BlobNotFound(blob_id)
        return path.read_bytes()

    def exists(self, blob_id: str) -> bool:
        """Return whether a file exists for *blob_id*."""
        return self.path_for(blob_id).exists()

    def verify(self, ref: BlobRef, data: bytes) -> None:
        """Raise :class:`ChecksumMismatch` unless *data* matches *ref*."""
        if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ChecksumMismatch(ref.blob_id)

    def gc(self, current_round: int, keep_rounds: int) -> list[str]:
        """Delete round directories with ``n <= current_round - keep_rounds``.

        Returns the sorted list of deleted blob ids.  Only ``rNNNN``
        directories are touched; loose files at the root (``best.pt``,
        ``final.pt``, ...) are never deleted.
        """
        if not self.root.is_dir():
            return []
        cutoff = current_round - keep_rounds
        deleted: list[str] = []
        for entry in self.root.iterdir():
            match = re.fullmatch(r"r(\d{4})", entry.name)
            if match is None or not entry.is_dir():
                continue
            if int(match.group(1)) <= cutoff:
                for child in entry.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                        continue
                    deleted.append(f"{entry.name}/{child.stem}")
                    child.unlink()
                entry.rmdir()
        return sorted(deleted)
