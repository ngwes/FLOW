"""FL@W configuration: pydantic models, YAML loading, ``FLAW_*`` env overrides.

Every subsystem consumes :class:`FlawConfig` (loaded via :func:`load_config`) and
reports config problems as :class:`ConfigError`.  Unknown keys are rejected
everywhere so typos in YAML surface immediately instead of silently training the
wrong thing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]  # PyYAML ships no py.typed; see task-2 report
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Raised for any invalid config: unreadable file, bad YAML, bad values."""


class RunSection(BaseModel):
    """``run:`` — identity of the run and the global random seed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 42


class ModelSection(BaseModel):
    """``model:`` — built-in registry name or a python import path (class)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = None
    class_path: str | None = Field(default=None, alias="class")
    module_path: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> ModelSection:
        has_name = self.name is not None
        has_class = self.class_path is not None
        if has_name == has_class:
            raise ValueError("model: exactly one of name/class required")
        if self.module_path is not None and not has_class:
            raise ValueError("model: module_path requires class")
        return self


class DataSection(BaseModel):
    """``data:`` — built-in dataset name (shared mode) or a local path (federated)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    path: str | None = None
    scheme: Literal["iid", "dirichlet"] = "iid"
    dirichlet_alpha: float = Field(default=0.5, gt=0)
    num_shards: int = Field(default=3, ge=1)
    shard: Literal["auto"] | int = "auto"
    val_fraction: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def _check_source(self) -> DataSection:
        if (self.name is None) == (self.path is None):
            raise ValueError("data: exactly one of name/path required")
        if isinstance(self.shard, int) and not 0 <= self.shard < self.num_shards:
            raise ValueError(
                f"data: shard must be 'auto' or in [0, num_shards) (num_shards={self.num_shards})"
            )
        return self


class OptimizerSpec(BaseModel):
    """``training.optimizer:`` — the optimizer the nodes use each round."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["sgd", "adam"] = "sgd"
    lr: float = Field(default=0.01, gt=0)
    momentum: float = Field(default=0.0, ge=0)


class TrainingSection(BaseModel):
    """``training:`` — local training hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    local_epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=64, ge=1)
    optimizer: OptimizerSpec = OptimizerSpec()


class FederationSection(BaseModel):
    """``federation:`` — token, round budget, timing and node identity."""

    model_config = ConfigDict(extra="forbid")

    token: str = "change-me"
    rounds: int = Field(default=50, ge=1)
    min_participants: int = Field(default=1, ge=1)
    round_deadline_s: float = Field(default=300.0, gt=0)
    heartbeat_s: float = Field(default=10.0, gt=0)
    liveness_timeout_s: float = Field(default=30.0, gt=0)
    node_id: str | None = None


class EarlyStoppingSection(BaseModel):
    """``early_stopping:`` — stop when the metric reaches target/stops improving."""

    model_config = ConfigDict(extra="forbid")

    metric: str = "val_accuracy"
    target: float | None = None
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.001, ge=0)


class CentralTestSection(BaseModel):
    """``evaluation.central_test:`` — held-out test set on the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    mode: Literal["off", "every_round", "final_only"] = "final_only"


class EvaluationSection(BaseModel):
    """``evaluation:`` — what gets measured, and where."""

    model_config = ConfigDict(extra="forbid")

    federated: bool = True
    central_test: CentralTestSection = CentralTestSection()


class RetentionSection(BaseModel):
    """``storage.retention:`` — how many closed rounds of blobs to keep."""

    model_config = ConfigDict(extra="forbid")

    keep_rounds: int = Field(default=3, ge=1)


class StorageSection(BaseModel):
    """``storage:`` — blob cache location and retention policy."""

    model_config = ConfigDict(extra="forbid")

    blobs: str | None = None
    retention: RetentionSection = RetentionSection()
    data_dir: str = "./data"


class ResourcesSection(BaseModel):
    """``resources:`` — container quotas, enforced when launched via Docker."""

    model_config = ConfigDict(extra="forbid")

    cpus: float | None = None
    memory: str | None = None
    gpu: str | None = None
    vram_fraction: float | None = Field(default=None, gt=0, le=1)


class FlawConfig(BaseModel):
    """The complete, validated FL@W configuration."""

    model_config = ConfigDict(extra="forbid")

    run: RunSection
    model: ModelSection
    data: DataSection
    training: TrainingSection = TrainingSection()
    federation: FederationSection = FederationSection()
    early_stopping: EarlyStoppingSection = EarlyStoppingSection()
    evaluation: EvaluationSection = EvaluationSection()
    storage: StorageSection = StorageSection()
    resources: ResourcesSection = ResourcesSection()

    def canonical_hash_payload(self) -> dict[str, object]:
        """The subset of the config that must match across nodes.

        Per-node paths and sharding choices are deliberately excluded so that
        federated-mode nodes with different local data still match.
        """
        return {
            "model": self.model.model_dump(),
            "training": self.training.model_dump(),
            "data": {"name": self.data.name, "scheme": self.data.scheme},
        }


# Env overrides (spec: container-friendliness).  str var -> (section, field);
# numeric var -> (section, field, parser).
_STR_OVERRIDES: tuple[tuple[str, str, str], ...] = (
    ("FLAW_TOKEN", "federation", "token"),
    ("FLAW_NODE_ID", "federation", "node_id"),
    ("FLAW_BLOBS", "storage", "blobs"),
    ("FLAW_DATA_DIR", "storage", "data_dir"),
)

_NUMERIC_OVERRIDES: tuple[tuple[str, str, str, type[int] | type[float]], ...] = (
    ("FLAW_ROUNDS", "federation", "rounds", int),
    ("FLAW_VRAM_FRACTION", "resources", "vram_fraction", float),
)


def _parse_number(var: str, raw: str, cast: type[int] | type[float]) -> int | float:
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{var}: {raw!r} is not a valid number") from exc


def _set_section_field(
    data: dict[str, Any], var: str, section: str, field: str, value: object
) -> None:
    existing = data.get(section)
    if existing is None:
        data[section] = {field: value}
    elif isinstance(existing, dict):
        existing[field] = value
    else:
        raise ConfigError(f"{var} targets {field!r} but the {section!r} section is not a mapping")


def _apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> None:
    for var, section, field in _STR_OVERRIDES:
        if var in env:
            _set_section_field(data, var, section, field, env[var])
    for var, section, field, cast in _NUMERIC_OVERRIDES:
        if var in env:
            _set_section_field(data, var, section, field, _parse_number(var, env[var], cast))


def _read_config_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc


def _parse_config_mapping(text: str, path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config {path}: top level must be a mapping")
    return loaded


def load_config(path: Path, env: Mapping[str, str] | None = None) -> FlawConfig:
    """Load *path*, apply ``FLAW_*`` overrides (*env*, default ``os.environ``), validate."""
    overrides = os.environ if env is None else env
    data = _parse_config_mapping(_read_config_text(path), path)
    _apply_env_overrides(data, overrides)
    try:
        return FlawConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def config_source_text(path: Path) -> str:
    """Raw config text, kept verbatim for the run-directory snapshot (Task 17)."""
    return path.read_text(encoding="utf-8")
