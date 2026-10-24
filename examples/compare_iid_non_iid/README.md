# IID vs non-IID: why sharding matters

**IID** means every node sees the same kind of data. Imagine shuffling one deck of cards
perfectly and dealing four equal hands. **Non-IID** means each node's data is skewed:
hand everyone a slice straight off the top of the deck, and one person gets mostly
hearts while another gets almost no picture cards.

Real labs live in the second world: one PC recorded only morning samples, another
machine saw only one device model. So a federated framework has to work there too.
This example shows what skew does to learning on real MNIST:

```bash
pip install -e ".[plot]"
python examples/compare_iid_non_iid/compare.py            # ~15 min on a CPU
python examples/compare_iid_non_iid/compare.py --dataset tiny --rounds 3   # offline smoke run
```

The two configs train identically except for the `data` line (plus the per-run `run.name`):

| config | `data.scheme` | what each node trains on |
|---|---|---|
| `flaw_iid.yaml` | `iid` | a random mix of all 10 classes |
| `flaw_dirichlet.yaml` | `dirichlet` (`dirichlet_alpha: 0.5`) | a skewed mix: some nodes see few classes |

Lower `dirichlet_alpha` means more skew (0.5 is strong in practice; `1.0` is milder).

## One yardstick for both runs

Each node validates on a held-out slice of *its own* shard. Under skew that slice is
lopsided too, so the nodes' validation accuracy is not comparable between the two runs.
`compare.py` therefore exports 200 images per class from MNIST's **test** split, which no
node ever trains on, into `runs/central-mnist/`. The orchestrator scores the averaged
model on them after every round (`evaluation.central_test.mode: every_round`).

## What it shows

After 10 rounds on 4 nodes (see the plot in the main README):

| | node validation | central test set |
|---|---|---|
| IID | 98.37% | **98.85%** |
| non-IID (Dirichlet 0.5) | 98.65% | **98.35%** |

- Averaging weights across skewed nodes still works. Here the skew costs about half a
  point on the shared test set.
- Under skew the nodes' own validation looks *better* than the truth, because each node
  is graded on the same lopsided classes it trains on. The central test set is the
  honest number.

Both runs write the usual artifacts under `runs/cmp-iid/` and `runs/cmp-dirichlet/`.
Re-plot them with `flaw plot runs/cmp-iid runs/cmp-dirichlet --labels iid,non-iid`.
