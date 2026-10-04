from pathlib import Path

import pytest

from flaw.config import ConfigError, FlawConfig, config_source_text, load_config

VALID_YAML = """
run:
  name: test-run
model:
  name: mnist_cnn
data:
  name: tiny
  scheme: iid
  num_shards: 3
training:
  local_epochs: 1
  batch_size: 32
  optimizer: {name: sgd, lr: 0.05, momentum: 0.9}
federation:
  token: secret
  rounds: 4
  min_participants: 1
  round_deadline_s: 30
"""


def write_cfg(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "flaw.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_config(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path, VALID_YAML))
    assert cfg.run.name == "test-run"
    assert cfg.model.name == "mnist_cnn"
    assert cfg.data.name == "tiny"
    assert cfg.training.optimizer.lr == 0.05
    assert cfg.federation.rounds == 4


def test_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path, VALID_YAML))
    assert cfg.federation.min_participants == 1
    assert cfg.federation.round_deadline_s == 30
    assert cfg.federation.heartbeat_s == 10.0
    assert cfg.federation.liveness_timeout_s == 30.0
    assert cfg.early_stopping.metric == "val_accuracy"
    assert cfg.early_stopping.patience == 5
    assert cfg.evaluation.central_test.mode == "final_only"
    assert cfg.storage.retention.keep_rounds == 3
    assert cfg.data.val_fraction == 0.1
    assert cfg.run.seed == 42


def test_env_overrides(tmp_path: Path) -> None:
    env = {"FLAW_TOKEN": "from-env", "FLAW_ROUNDS": "7", "FLAW_NODE_ID": "lab-1"}
    cfg = load_config(write_cfg(tmp_path, VALID_YAML), env=env)
    assert cfg.federation.token == "from-env"
    assert cfg.federation.rounds == 7
    assert cfg.federation.node_id == "lab-1"


@pytest.mark.parametrize(
    "patch, message_part",
    [
        ("model:\n  name: mnist_cnn\n  class: x.Y\n", "exactly one"),
        ("model: {}\n", "model"),
        ("data:\n  scheme: iid\n", "exactly one"),
        ("data:\n  name: tiny\n  scheme: dirichlet\n  dirichlet_alpha: 0\n", "alpha"),
        ("data:\n  name: tiny\n  num_shards: 0\n", "num_shards"),
        ("data:\n  name: tiny\n  shard: 9\n  num_shards: 3\n", "shard"),
        ("data:\n  name: tiny\n  scheme: bogus\n", "scheme"),
        ("data:\n  name: tiny\n  val_fraction: 1.0\n", "val_fraction"),
        (
            "training:\n  local_epochs: 1\n  batch_size: 8\n"
            "  optimizer: {name: rmsprop, lr: 0.1}\n",
            "optimizer",
        ),
        ("training:\n  local_epochs: 0\n  batch_size: 8\n", "local_epochs"),
        ("federation:\n  min_participants: 0\n", "min_participants"),
        ("federation:\n  round_deadline_s: 0\n", "round_deadline_s"),
        ("early_stopping:\n  patience: 0\n", "patience"),
        ("early_stopping:\n  min_delta: -1\n", "min_delta"),
        ("storage:\n  retention: {keep_rounds: 0}\n", "keep_rounds"),
        ("resources:\n  vram_fraction: 1.5\n", "vram_fraction"),
    ],
)
def test_invalid_configs_rejected(tmp_path: Path, patch: str, message_part: str) -> None:
    import yaml

    base = yaml.safe_load(VALID_YAML)
    base.update(yaml.safe_load(patch))  # each patch replaces a whole top-level section
    text = yaml.safe_dump(base)
    with pytest.raises(ConfigError, match=message_part):
        load_config(write_cfg(tmp_path, text))


def test_canonical_hash_payload_shape(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path, VALID_YAML))
    payload = cfg.canonical_hash_payload()
    assert set(payload) == {"model", "training", "data"}
    assert set(payload["data"]) == {"name", "scheme"}  # paths/sharding excluded (spec §5.4)


# --- Tests added for full branch coverage of flaw/config.py and edge semantics ---


def test_config_source_text_returns_raw_text(tmp_path: Path) -> None:
    text = "# leading comment\nrun:\n  name: raw-run  # trailing comment\n"
    assert config_source_text(write_cfg(tmp_path, text)) == text


def test_env_overrides_storage_and_resources(tmp_path: Path) -> None:
    env = {"FLAW_BLOBS": "/tmp/blobs", "FLAW_DATA_DIR": "/tmp/data", "FLAW_VRAM_FRACTION": "0.25"}
    cfg = load_config(write_cfg(tmp_path, VALID_YAML), env=env)
    assert cfg.storage.blobs == "/tmp/blobs"
    assert cfg.storage.data_dir == "/tmp/data"
    assert cfg.resources.vram_fraction == 0.25


@pytest.mark.parametrize("env", [{"FLAW_ROUNDS": "abc"}, {"FLAW_VRAM_FRACTION": "abc"}])
def test_bad_numeric_env_value_rejected(tmp_path: Path, env: dict[str, str]) -> None:
    var = next(iter(env))
    with pytest.raises(ConfigError, match=var):
        load_config(write_cfg(tmp_path, VALID_YAML), env=env)


def test_env_override_value_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="vram_fraction"):
        load_config(write_cfg(tmp_path, VALID_YAML), env={"FLAW_VRAM_FRACTION": "1.5"})


def test_env_override_on_malformed_section_rejected(tmp_path: Path) -> None:
    text = "run: {name: r}\nmodel: {name: m}\ndata: {name: tiny}\nfederation: 3\n"
    with pytest.raises(ConfigError, match=r"FLAW_ROUNDS.*'federation' section"):
        load_config(write_cfg(tmp_path, text), env={"FLAW_ROUNDS": "5"})


def test_missing_config_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_non_utf8_config_rejected(tmp_path: Path) -> None:
    p = tmp_path / "flaw.yaml"
    p.write_bytes(b"run: \xff\xfe\n")
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config(p)


def test_invalid_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write_cfg(tmp_path, "run: [unclosed\n"))


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_cfg(tmp_path, "- 1\n- 2\n"))


def test_empty_config_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="run"):
        load_config(write_cfg(tmp_path, "\n"))


def test_model_class_alias_accepted(tmp_path: Path) -> None:
    text = "run: {name: r}\nmodel: {class: my_code.MyNet, module_path: /code}\ndata: {name: tiny}\n"
    cfg = load_config(write_cfg(tmp_path, text))
    assert cfg.model.name is None
    assert cfg.model.class_path == "my_code.MyNet"
    assert cfg.model.module_path == "/code"


def test_model_module_path_without_class_rejected(tmp_path: Path) -> None:
    text = "run: {name: r}\nmodel: {name: m, module_path: /code}\ndata: {name: tiny}\n"
    with pytest.raises(ConfigError, match="module_path"):
        load_config(write_cfg(tmp_path, text))


def test_data_shard_within_range_accepted(tmp_path: Path) -> None:
    text = "run: {name: r}\nmodel: {name: m}\ndata: {name: tiny, shard: 2, num_shards: 3}\n"
    assert load_config(write_cfg(tmp_path, text)).data.shard == 2


def test_data_name_and_path_both_rejected(tmp_path: Path) -> None:
    text = "run: {name: r}\nmodel: {name: m}\ndata: {name: tiny, path: /data}\n"
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(write_cfg(tmp_path, text))


def test_unknown_top_level_section_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trainning"):
        load_config(write_cfg(tmp_path, VALID_YAML + "trainning: {local_epochs: 1}\n"))


def test_sections_not_shared_between_instances() -> None:
    a = FlawConfig(run={"name": "a"}, model={"name": "m"}, data={"name": "d"})
    b = FlawConfig(run={"name": "b"}, model={"name": "m"}, data={"name": "d"})
    a.training.optimizer.lr = 0.99
    assert b.training.optimizer.lr == 0.01
