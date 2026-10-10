# IID vs non-IID — why sharding matters

**IID** means every node sees the same kind of data: imagine shuffling one deck of cards
perfectly and dealing four equal hands. **Non-IID** means each node's data is skewed:
hand everyone a slice straight off the top of the deck, and one person gets mostly
hearts while another gets almost no picture cards.

Real labs live in the second world — a PC that recorded only morning samples, a machine
that saw only one device model — so a federated framework has to work there too. This
example shows what that does to learning, using the built-in `tiny` dataset (so it runs
offline in under a minute):

```bash
pip install -e ".[plot]"
python examples/compare_iid_non_iid/compare.py --rounds 10 --nodes 4
```

The two configs are identical except for the `data.scheme` line:

| config | `data.scheme` | what each node trains on |
|---|---|---|
| `flaw_iid.yaml` | `iid` | a random mix of all 10 classes |
| `flaw_dirichlet.yaml` | `dirichlet` (`dirichlet_alpha: 0.5`) | a skewed mix — some nodes see few classes |

Lower `dirichlet_alpha` ⇒ more skew (0.5 is strong in practice; `1.0` is milder).

**What to look for in `runs/compare.png`:** the non-IID curve typically starts lower and
is jumpier — each round the average is pulled around by nodes that are locally
over-fitting their own slice. With enough rounds the two curves usually converge,
which is the interesting part: averaging weights across skewed nodes still works, it
just needs a few more rounds.

Both runs write the usual artifacts under `runs/cmp-iid/` and `runs/cmp-dirichlet/` —
inspect them afterwards with `flaw plot runs/cmp-iid runs/cmp-dirichlet --labels iid,non-iid`.
