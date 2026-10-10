"""Train the same model twice - IID shards vs non-IID (Dirichlet) shards - and plot both.

Usage (from the repository root, after `pip install -e ".[plot]"`):

python examples/compare_iid_non_iid/compare.py [--rounds 10] [--nodes 4] [--out runs/compare.png]

Both runs happen on this machine via `flaw simulate`: one orchestrator plus N node
threads each, no Docker and no dataset download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flaw.plot import plot_runs
from flaw.simulate import run_simulation

HERE = Path(__file__).parent
CONFIGS = (("IID", "flaw_iid.yaml"), ("non-IID (Dirichlet 0.5)", "flaw_dirichlet.yaml"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/compare.png"))
    args = parser.parse_args()

    run_dirs: dict[str, Path] = {}
    for label, config_name in CONFIGS:
        print(f"== running {label} ...")
        result = run_simulation(
            HERE / config_name,
            args.nodes,
            rounds=args.rounds,
            timeout_s=args.timeout,
            run_root=args.run_root,
        )
        best = result.best_metric if result.best_metric is not None else float("nan")
        print(
            f"   {label}: {result.rounds} rounds ({result.reason}), "
            f"best val_accuracy {best:.4f} at round {result.best_round}"
        )
        run_dirs[label] = result.run_dir

    out_path = plot_runs(run_dirs, args.out)
    print(f"plot written to {out_path}")


if __name__ == "__main__":
    main()
