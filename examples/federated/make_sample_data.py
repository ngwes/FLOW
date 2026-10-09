"""Write two small synthetic image folders for the federated example (offline, deterministic).

Usage: python examples/federated/make_sample_data.py [--out DIR] [--nodes 2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from flaw.node.data import make_tiny_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--nodes", type=int, default=2)
    args = parser.parse_args()

    train_ds, val_ds = make_tiny_dataset(seed=0)
    for node in range(args.nodes):
        for split_name, dataset in (("train", train_ds), ("val", val_ds)):
            # Contiguous chunks (not a stride split): labels are index % 10, so a stride
            # would leave each node with only half the classes. Last node takes the rest.
            chunk = len(dataset) // args.nodes
            start = node * chunk
            stop = len(dataset) if node == args.nodes - 1 else start + chunk
            for index in range(start, stop):
                image, label = dataset[index]
                folder = args.out / f"node{node + 1}" / split_name / str(int(label))
                folder.mkdir(parents=True, exist_ok=True)
                pixels = (image.squeeze(0).clamp(0.0, 1.0) * 255).numpy().astype("uint8")
                Image.fromarray(pixels, mode="L").save(folder / f"img{index}.png")
    print(f"wrote {args.nodes} node folders under {args.out}")


if __name__ == "__main__":
    main()
