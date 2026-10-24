"""Train the same model twice - IID shards vs non-IID (Dirichlet) shards - and plot both.

Usage (from the repository root, after `pip install -e ".[plot]"`):

python examples/compare_iid_non_iid/compare.py [--rounds 10] [--nodes 4] [--out runs/compare.png]

Both runs happen on this machine via `flaw simulate`: one orchestrator plus N node
threads each, no Docker. Both are scored every round on the SAME held-out central
test set (exported once from the dataset's test split), so the two curves share one
yardstick. `--dataset tiny` swaps MNIST for the offline synthetic dataset (fast, no
download; used by the test suite).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from flaw.node.data import make_tiny_dataset
from flaw.plot import plot_runs, read_metrics
from flaw.simulate import run_simulation

HERE = Path(__file__).parent
CONFIGS = (("IID", "flaw_iid.yaml"), ("non-IID (Dirichlet 0.5)", "flaw_dirichlet.yaml"))


def _test_images(dataset: str, data_dir: Path) -> Any:
    """(PIL image or tensor, label) pairs from data the nodes never train on."""
    if dataset == "mnist":
        from torchvision.datasets import MNIST  # type: ignore[import-untyped]

        return MNIST(root=data_dir, train=False, download=True)  # PIL images
    # tiny: a fresh seed, so none of these images appear in the training split
    return make_tiny_dataset(seed=100, val_n=1000)[1]


def export_central_test(dataset: str, out_dir: Path, per_class: int, data_dir: Path) -> Path:
    """Write `per_class` test images per class as `<out_dir>/<label>/<i>.png` (once)."""
    if out_dir.exists():
        return out_dir
    counts: dict[int, int] = {}
    for index, (image, label) in enumerate(_test_images(dataset, data_dir)):
        label = int(label)
        if counts.get(label, 0) >= per_class:
            continue
        if not isinstance(image, Image.Image):  # tiny yields 1x28x28 float tensors
            pixels = (image.squeeze(0).clamp(0.0, 1.0) * 255).numpy().astype("uint8")
            image = Image.fromarray(pixels, mode="L")
        folder = out_dir / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"{index}.png")
        counts[label] = counts.get(label, 0) + 1
    return out_dir


def effective_config(config_name: str, dataset: str, central_dir: Path, out_dir: Path) -> Path:
    """Copy of a shipped config with the dataset and central-test path filled in."""
    cfg = yaml.safe_load((HERE / config_name).read_text(encoding="utf-8"))
    cfg["data"]["name"] = dataset
    cfg["evaluation"]["central_test"]["path"] = central_dir.as_posix()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / config_name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def last_central_accuracy(run_dir: Path) -> float | None:
    rounds = [
        line for line in read_metrics(run_dir / "metrics.jsonl") if line.get("type") == "round"
    ]
    central = [line["central"] for line in rounds if isinstance(line.get("central"), dict)]
    return float(central[-1]["val_accuracy"]) if central else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "tiny"), default="mnist")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=200, help="central test images per class")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/compare.png"))
    args = parser.parse_args()

    central_dir = export_central_test(
        args.dataset,
        args.run_root / f"central-{args.dataset}",
        args.per_class,
        args.data_dir,
    )
    print(f"== central test set: {central_dir}")

    run_dirs: dict[str, Path] = {}
    for label, config_name in CONFIGS:
        print(f"== running {label} ...")
        config = effective_config(
            config_name, args.dataset, central_dir, args.run_root / "_configs"
        )
        result = run_simulation(
            config,
            args.nodes,
            rounds=args.rounds,
            timeout_s=args.timeout,
            run_root=args.run_root,
        )
        central = last_central_accuracy(result.run_dir)
        central_text = f"{central:.4f}" if central is not None else "n/a"
        print(f"   {label}: {result.rounds} rounds, central test accuracy {central_text}")
        run_dirs[label] = result.run_dir

    out_path = plot_runs(run_dirs, args.out)
    print(f"plot written to {out_path}")


if __name__ == "__main__":
    main()
