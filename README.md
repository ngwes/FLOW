# FL@W — Federated Learning at Work

Train one model across the PCs you already have. Each machine trains on its own data,
the orchestrator averages everyone's weights every round, and no raw data ever leaves
its machine. Built from scratch (FedAvg over HTTP plus a blob store), sized for 1–3
lab PCs with consumer GPUs — and it still works on a laptop CPU.

## 60-second demo (no GPU, no download)

```bash
pip install -e ".[dev]"
flaw simulate examples/tiny-sim/flaw.yaml --nodes 2
```

`simulate` starts a real orchestrator plus N simulated nodes on this machine and
prints the result. Artifacts land in `runs/tiny-sim/`.

## Run it for real (one process per machine)

On the orchestrator PC:

```bash
flaw orchestrator examples/mnist/flaw.yaml --host 0.0.0.0
```

On every node PC (same YAML file, same LAN):

```bash
FLAW_NODE_ID=lab-1 flaw node examples/mnist/flaw.yaml --orchestrator http://ORCHESTRATOR_IP:8000
```

Watch progress from any machine:

```bash
flaw status --orchestrator http://ORCHESTRATOR_IP:8000
```

## What you get, in `runs/<name>/`

| file | what it is |
|---|---|
| `metrics.jsonl` | one JSON line per round plus a summary line: losses, accuracy, timings, who took part |
| `best.pt` / `final.pt` | best-by-metric and last global weights (a PyTorch `state_dict`) |
| `state.json` | run status, current round, node-to-shard map |
| `blobs/rNNNN/` | the weight blobs actually moved between machines, addressed by id |
| `config.yaml` | snapshot of the training description that produced the run |

## Two ways to have data

- **Shared** — one dataset, split deterministically across nodes:
  `data: {name: mnist, scheme: iid | dirichlet, num_shards: N}`.
- **Federated** — each node keeps its own image folder and nothing is shared:
  `data: {path: ./data/local}` with `train/<class>/*.png` and `val/<class>/*.png`
  inside. `data.path` is resolved relative to the working directory (run `flaw`
  from the repo root), so every machine runs the same YAML with a `path` pointing
  at its own folder. `python examples/federated/make_sample_data.py` builds toy
  folders so you can try it offline.

## Rules of the road

- Nodes join and leave whenever they like. A round closes when every node that was
  alive at its start has reported, or at `round_deadline_s`.
- If fewer than `min_participants` results make it home, the round is **void and
  re-run under the same number** — round numbers never skip.
- Heavy bytes never travel inside JSON: messages carry blob ids, and every weight
  download is verified against its sha256.
- The central test set is optional (`evaluation.central_test.mode: off | every_round | final_only`).

## Development

```bash
pip install -e ".[dev]"
pytest -m "not docker" --cov=flaw --cov-fail-under=95
```

Roadmap: **Phase 1 (this)** core + CLI · **Phase 2** Docker images, per-node resource
quotas, compose · **Phase 3** docs, plotting, IID vs non-IID comparison · **Phase 4
(later, not built)** secure aggregation, TLS, multi-model support. Details in
`docs/superpowers/specs/2026-10-04-flaw-design.md`.
