from pathlib import Path

import torch

from flaw.config import load_config
from flaw.models import build_model
from flaw.node.data import make_tiny_dataset
from flaw.orchestrator.centraleval import default_central_factory, run_central_eval

CFG = """
run: {name: t}
model: {name: mnist_cnn}
data: {name: tiny, num_shards: 2, scheme: iid, val_fraction: 0.25}
training: {local_epochs: 1, batch_size: 16}
"""


def cfg(tmp_path: Path, text: str = CFG):  # type: ignore[no-untyped-def]
    p = tmp_path / "flaw.yaml"
    p.write_text(text, encoding="utf-8")
    return load_config(p)


def factory():
    return make_tiny_dataset(seed=5, n=64, val_n=32)[1]


def test_run_central_eval_reports_metrics(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    sd = build_model(config.model).state_dict()
    out = run_central_eval(config, sd, factory, batch_size=16)
    assert set(out) == {"val_loss", "val_accuracy"}
    assert 0.0 <= out["val_accuracy"] <= 1.0
    assert run_central_eval(config, sd, factory, batch_size=16) == out  # deterministic


def test_run_central_eval_does_not_mutate_state_dict(tmp_path: Path) -> None:
    config = cfg(tmp_path)
    sd = build_model(config.model).state_dict()
    before = {k: v.clone() for k, v in sd.items()}
    run_central_eval(config, sd, factory, batch_size=16)
    assert all(torch.equal(sd[k], before[k]) for k in before)


def test_default_central_factory(tmp_path: Path) -> None:
    assert default_central_factory(cfg(tmp_path)) is None  # no central_test.path configured

    from PIL import Image

    root = tmp_path / "central"
    for cls in ("a", "b"):
        (root / cls).mkdir(parents=True)
        Image.new("L", (28, 28), color=64).save(root / cls / "0.png")
    text = CFG + f"\nevaluation: {{central_test: {{path: {root.as_posix()}, mode: final_only}}}}\n"
    config = cfg(tmp_path, text)
    make = default_central_factory(config)
    assert make is not None
    assert len(make()) == 2
    out = run_central_eval(config, build_model(config.model).state_dict(), make, batch_size=8)
    assert 0.0 <= out["val_accuracy"] <= 1.0
