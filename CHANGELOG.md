# Changelog

All notable changes to FL@W are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
