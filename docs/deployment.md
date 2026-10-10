# Deploying FL@W

Two shapes, same config file everywhere. Work through "Bare processes" first — Docker
is a convenience layer on top, not a different system.

## Bare processes on 2+ PCs

On the orchestrator PC (the one that will hold the artifacts):

```bash
flaw orchestrator examples/mnist/flaw.yaml --host 0.0.0.0
```

On each node PC — the same YAML file, committed/shared beforehand:

```bash
FLAW_NODE_ID=lab-1 flaw node examples/mnist/flaw.yaml --orchestrator http://ORCH_IP:8000
```

Checklist:

- All machines run the *same* `flaw.yaml` — the orchestrator checks a hash of the
  training-relevant config (model, training, data name/scheme) at registration
  (`config mismatch: node config hash does not match the run`). `FLAW_TOKEN` must
  match everywhere; `FLAW_NODE_ID` must be unique per machine.
- Each node needs a unique `FLAW_NODE_ID` (or `federation.node_id`).
- The orchestrator port (default 8000) must be reachable from every node — allow it
  through the OS firewall (on Windows, "Windows Defender Firewall → Advanced →
  Inbound Rules").
- Find the orchestrator's IP with `ip a` (Linux) or `ipconfig` (Windows).
- Datasets land in `storage.data_dir` (default `./data`) on the machines that load them.

## Docker on one machine

```bash
# CPU-only machines (also what CI runs):
FLAW_NODE_BASE=python:3.11-slim FLAW_NODE_TORCH_INDEX=https://download.pytorch.org/whl/cpu docker compose up --build -d

# GPU machines: pin one GPU per node in docker-compose.gpu.yml first
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

- The orchestrator publishes `localhost:8000`; nodes talk to it over the compose
  network — nothing else needs to be exposed.
- Config mounting: edit the `./examples/tiny-sim/flaw.yaml:/etc/flaw/flaw.yaml:ro`
  mount in `docker-compose.yml` to point at your own training description.
- Results are visible on the host because `${FLAW_RUNS_DIR:-./runs}` is bind-mounted
  to `/app/runs`. Watch them with `flaw status --orchestrator http://localhost:8000`.
- `FLAW_DATA_DIR` is read twice: compose uses your host-shell value to pick the host folder (`./data/nodeN` by default), while the containers always see it set to `/data`.
- Node containers exit when the run is over; `docker compose logs node1` tells you
  why if one stopped early.
- Resource quota example (already set to 4 CPUs / 8 GB per node):

```yaml
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8g
```

- Split a shared GPU between two node containers with the soft VRAM cap:
  `FLAW_VRAM_FRACTION=0.45` per node (see `flaw.yaml` → `resources.vram_fraction`).

## GPU passthrough

### Linux

1. Install the NVIDIA driver and `nvidia-container-toolkit`.
2. `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
3. Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Windows 11 via WSL2

1. Install a recent NVIDIA driver on Windows (the WSL2-enabled one from NVIDIA,
   not one bundled by Windows Update).
2. Docker Desktop → Settings → General → "Use the WSL 2 based engine"; GPU support is
   automatic on recent versions.
3. Verify from PowerShell: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
4. Caveats: WSL2 exposes a single GPU to Linux containers, so two node containers
   share it — use `FLAW_VRAM_FRACTION` to split VRAM, or run one node container.
   Keep datasets in the Linux filesystem (`~/data`, not `/mnt/c/...`) for IO speed,
   and cap WSL memory in `%UserProfile%\.wslconfig` if Windows and training fight
   over RAM.

## Where things live

| path | created by | content |
|---|---|---|
| `runs/<name>/` | orchestrator | `metrics.jsonl`, `best.pt`, `final.pt`, `state.json`, `blobs/`, `config.yaml` |
| `runs/<name>/orchestrator.log` | orchestrator | the orchestrator's log for that run |
| `data/` (default) | nodes | downloaded datasets (`mnist`, `cifar10`) |
| `./data/nodeN/` (compose) | nodes | per-container dataset cache |

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| node exits `fatal_error` at startup | token or config mismatch — HTTP 403/409 rejections are fatal | same YAML and token everywhere; the node log shows the exact rejection |
| node keeps retrying instead of registering | the orchestrator is unreachable (transport errors are retried with backoff, not fatal) | check the URL and the firewall between the machines |
| orchestrator restarts mid-run | nodes lose their registration but keep retrying | nothing — nodes re-register automatically (with backoff) and keep their shard from `state.json` |
| `config mismatch: node config hash does not match the run` at registration | the training-relevant config differs between machines | re-copy the file; the hash covers model, training, and data name/scheme (env overrides like `FLAW_ROUNDS` do not change it) |
| rounds keep voiding | `min_participants` higher than the number of live nodes | lower it or add nodes |
| "no shards available: all N shards are taken" | `num_shards` < number of nodes | raise `num_shards` |
| CUDA OOM on a node | model/batch too big, or two nodes share one GPU | lower `training.batch_size`, set `resources.vram_fraction` |
| run stalls near the deadline each round | one node is much slower | raise `federation.round_deadline_s` |
| `nvidia-smi` works on the host but not in a container | toolkit/desktop GPU support missing | redo the passthrough section above |

## See also

- `docs/protocol.md` — the wire protocol, blob conventions, and round-loop semantics.
- `README.md` — the 60-second demo and the two data modes.
- `examples/compare_iid_non_iid/` — why sharding affects training quality.
