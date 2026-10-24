# Changelog

All notable changes to FL@W are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-10-24

Bug fixes found by running FL@W for real (MNIST, Docker), quieter output, honest
experiments, and a documentation site.

### Fixed

- Node containers trained ~100x slower than they should: torch started one thread
  per *host* core inside a 4-CPU quota and was throttled (a tiny round took ~60 s
  instead of ~1 s). Nodes now size torch's thread pool to the cgroup CPU quota
  (v2 and v1); an explicit `OMP_NUM_THREADS` still wins.
- Shared-mode `mnist` and `cifar10` crashed on the first training batch: torchvision
  returned PIL images that a DataLoader cannot batch. Images are now `[0, 1]` tensors,
  the same scaling the folder loader uses.
- Node logs reported the configured id or the hostname instead of the identity the
  node actually registered with (in `flaw simulate`: `sim-0`, `sim-1`, ...).
- Docker images copy `README.md` and `LICENSE`, which `pip install .` now needs.

### Changed

- Quieter logs: httpx's per-request lines and uvicorn's access log appear only with
  `--verbose`; the per-round node line is compact (`loss=… val_accuracy=…`).
- `flaw plot` draws each run in one color: solid for node validation, dashed for the
  central test set.
- `examples/compare_iid_non_iid` runs on MNIST and scores both runs on one shared
  central test set (`--dataset tiny` keeps an offline mode).
- `examples/tiny-sim` uses SGD with momentum, so the demo visibly learns in 4 rounds.

- The deployment guide covers CPU quotas, `OMP_NUM_THREADS` and `-v`.

### Added

- Documentation site at <https://ngwes.github.io/FLOW/> (MkDocs Material, deployed
  to GitHub Pages on every push to `main`).
- MIT license, package metadata, README results and architecture diagrams.

## [0.1.0] — 2026-10-10

First release: the federated core, the container path, and the docs (Phases 1–3).

### Added

- **Federated core** — from-scratch FedAvg over HTTP: sample-weighted aggregation,
  soft-synchronous rounds (all-alive fast path or `round_deadline_s`), partial
  participation, and void rounds that re-run under the same number.
- **Elastic membership** — nodes register, heartbeat, join mid-run, leave, or crash;
  liveness timeouts gate round opening (`min_participants`) and close timing.
- **Crash-safe orchestrator** — at-least-once rounds; `state.json` plus blob-derived
  recovery recomputes the current round after a restart.
- **Blob store** — heavy bytes at well-defined paths (`rNNNN/global`,
  `rNNNN/node-<id>`); messages carry ids + sha256 and downloads are verified before
  deserialization; retention-based GC.
- **Two data modes** — shared (IID / Dirichlet sharding of a built-in dataset) and
  federated (per-machine image folders, nothing shared).
- **Training** — PyTorch models, local training/validation on each node, early
  stopping (`target`, `patience`, `min_delta`), best/last checkpoints.
- **Central test set** — `off | every_round | final_only` evaluation on the
  orchestrator; evaluation failures never kill a run.
- **CLI** — `flaw orchestrator|node|simulate|status|plot`, `--version`; one YAML
  training description shared by every machine, with env overrides (`FLAW_NODE_ID`,
  `FLAW_TOKEN`, `FLAW_ROUNDS`, `FLAW_BLOBS`, `FLAW_DATA_DIR`, `FLAW_VRAM_FRACTION`).
- **Simulation** — `flaw simulate` runs an orchestrator plus N node threads on one
  machine for smoke tests and demos.
- **Containers** — pinned GPU node image with CPU fallback (`FLAW_NODE_BASE`), CPU
  orchestrator image, compose stack with per-node cpus/memory quotas, GPU device
  reservations, and a VRAM soft cap.
- **Examples & docs** — `examples/tiny-sim`, `examples/mnist`, `examples/federated`,
  `examples/compare_iid_non_iid`; `docs/protocol.md`, `docs/deployment.md`; CI with a
  coverage gate and a Docker smoke job.

### Known limitations

- No secure aggregation or TLS; runs assume a trusted network (Phase 4, not built).
- Single model family per run (Phase 4 explores multi-model support).
- Four config knobs are accepted and validated but never read by the framework in
  v0.1.0: `storage.blobs` (including the `FLAW_BLOBS` override), `data.shard`,
  `evaluation.federated`, and `resources.cpus`/`resources.memory` (container quotas
  come from `docker-compose.yml` instead).
