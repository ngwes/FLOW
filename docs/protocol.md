# FL@W protocol reference — v1

FL@W moves two kinds of things between machines: small JSON control messages and
**weight blobs** — PyTorch `state_dict` files that travel as raw bytes, never inside
JSON. This document is the complete contract between a node and the orchestrator:
enough to implement your own client. Wire types live in `flaw/protocol.py`; the
endpoints and the round loop in `flaw/orchestrator/server.py`.

## Terms

| term | meaning |
|---|---|
| orchestrator | the server: one per run, owns the round loop and the global model |
| node | a training PC: downloads global weights, trains locally, uploads an update |
| round N | one train-and-average cycle, numbered from 1, never skipped |
| global weights | the model handed out at the start of a round: blob `r{N}/global` |
| update blob | a node's trained weights for round N: blob `r{N}/node-<id>` |
| void round | a round that produced no usable update; it re-runs under the same number |

`PROTOCOL_VERSION = 1`. Registration carries it; a mismatch is rejected with 403.

## Blob ids and the handout invariant

Blob ids match `^r\d{4}/(global|node-[A-Za-z0-9._-]{1,64})$`:

- `r0007/global` — global weights for round 7.
- `r0007/node-lab-1` — `lab-1`'s update for round 7.

Three invariants make recovery and late joiners safe:

1. **When round N opens, `r{N}/global` exists.** A fresh run writes `r0001/global`
   (random init) before the first round opens.
2. **Aggregating a non-void round N writes `r{N+1}/global`** — including the final
   round of a run, so restarting a finished run cannot rewrite its history.
3. **Recovery is derived, not stored**: after a crash the orchestrator sets
   `round = max(existing global blob round) - 1`, then ensures the handout for the
   next round exists. Re-running a round is at-least-once and idempotent for nodes —
   a node that already reported gets `stale` or `duplicate` and moves on.

On disk (orchestrator side): `runs/<name>/blobs/rNNNN/global.pt` and
`.../node-<id>.pt`; rounds older than `storage.retention.keep_rounds` are
garbage-collected.

## Authentication (v1)

- `POST /register` — the shared secret is the `token` field of the JSON body.
- `POST /blob` — the secret is the `X-Flaw-Token` header.
- All other endpoints are open on the LAN in v1. TLS and authenticated reads are
  not built yet. Run FL@W on a trusted network only.

## Endpoints

| method | path | request | success responses | errors |
|---|---|---|---|---|
| POST | `/register` | `RegisterRequest` | 200 `RegisterResponse` | 403, 409 |
| POST | `/heartbeat` | `HeartbeatRequest` | 200 `HeartbeatResponse` | 404 |
| GET | `/assignment?node_id=…` | — | 200 `Assignment`, 200 `RunFinished`, 204 (timeout) | 404 |
| POST | `/result` | `ResultRequest` | 200 `ResultResponse` (`accepted` true or false) | 400, 404 |
| GET | `/blob/{blob_id}` | — | 200 raw bytes + `X-Blob-Sha256` | 400, 404 |
| POST | `/blob?node_id=…&round=…` | raw bytes | 200 `BlobUploadResponse` | 400, 403, 404 |
| GET | `/status` | — | 200 `StatusResponse` | — |

FL@W's own error bodies are always `{"reason": "<text>"}`.

### POST /register

```json
{
  "protocol_version": 1,
  "token": "change-me-to-something-shared",
  "node_id": "lab-1",
  "capabilities": {"device": "cuda:0", "device_name": "NVIDIA GeForce RTX 3060"},
  "config_hash": "<sha256 of the training-relevant config>"
}
```

```json
{"accepted": true, "reason": null,
 "run": {"name": "tiny-sim", "current_round": 3, "phase": "idle"},
 "shard": 0}
```

- `config_hash` proves every machine trains the same recipe; a mismatch → **409**
  `config mismatch: node config hash does not match the run` with no partial acceptance.
- `shard` (shared-data mode only) is the data slice this node must train on; it is
  assigned once, persisted in `state.json`, and returned again on re-registration.
  In federated mode it is `null`.
- Rejections: **403** wrong token or wrong `protocol_version`; **409** node id
  already in use, or no shards left.

### POST /heartbeat

```json
{"node_id": "lab-1"}  →  {"phase": "open", "round": 4}
```

A node is *alive* while its last heartbeat is younger than
`federation.liveness_timeout_s`. Dead nodes are swept out of open rounds, which can
trigger the fast close path. Unknown node → **404**.

### GET /assignment?node_id=lab-1

Long poll: the orchestrator holds the request until there is news or `long_poll_s`
(default 30 s; `flaw simulate` uses 1 s) elapses.

- **200 `Assignment`** — a round is open and this node has not reported yet:

```json
{"round": 4,
 "train_spec": {"local_epochs": 2, "batch_size": 64, "seed": 42,
                "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0}},
 "weights": {"blob_id": "r0004/global", "sha256": "<64 hex chars>", "size": 913754,
             "format": "torch-state-dict-v1", "dtype": "fp32"}}
```

  The node downloads `weights` via `GET /blob`, verifies the sha256, then trains.
  (Nobody uses the raw network like this twice: the same assignment is redelivered
  every poll until the node submits, so training is always "safe to retry from scratch".)

- **200 `RunFinished`** — the run is over: `{"run_finished": true, "reason":
  "max_rounds", "best_round": 6, "best_metric": 0.9711}`. The node exits cleanly.
- **204** — nothing yet; poll again.
- **404** — unknown node (register first).

### POST /result

Upload the update blob first (`POST /blob`), then submit:

```json
{"node_id": "lab-1", "round": 4,
 "update": {"blob_id": "r0004/node-lab-1", "sha256": "<64 hex chars>", "size": 913754,
            "format": "torch-state-dict-v1", "dtype": "fp32"},
 "num_samples": 128,
 "metrics": {"loss": 0.21, "val_loss": 0.24, "val_accuracy": 0.93},
 "duration_s": 12.4,
 "device": "cuda:0"}
```

Response: `{"accepted": true, "reason": null}` — or `accepted: false` with one of:

| reason | meaning |
|---|---|
| `duplicate` | this node already reported for this round |
| `stale` | the round closed (or moved on); late results are never aggregated |
| `unknown_round` | the round number is from the future |
| `invalid` | the blob didn't pass the checks below |

Checks run in this order: a malformed `update` ref (bad `blob_id` or bad `sha256`
length) → **400**; unknown node → **404**; submitted `update.blob_id` is not
this node's update id for this round → `invalid`; blob missing on the orchestrator
or sha256 mismatch → `invalid` (untrusted bytes are never deserialized); then
`duplicate` / `stale` / `unknown_round`.

### GET /blob/{blob_id}

**200** — raw bytes, `application/octet-stream`, header
`X-Blob-Sha256: <hex>`. **400** malformed id, **404** unknown blob.
Clients MUST recompute the sha256 and refuse to deserialize on mismatch — that one
check is what protects a node from a corrupted or swapped file.

### POST /blob?node_id=lab-1&round=4

Body: raw weights bytes. Header `X-Flaw-Token` required. The blob is stored under
its canonical id `r0004/node-lab-1`; any other id in the result submission is
rejected. Response: `{"blob_id": "r0004/node-lab-1", "sha256": "<hex>", "size": 913754}`.
Errors: **403** bad token, **404** unknown node, **400** round < 1.

### GET /status

```json
{"name": "tiny-sim", "phase": "open", "round": 4,
 "best_metric": 0.93, "best_round": 3,
 "nodes": [{"node_id": "lab-1", "alive": true, "shard": 0, "device": "cuda:0",
            "last_heartbeat_age_s": 2.1}]}
```

Used by `flaw status` and the Docker healthcheck. Read-only, no auth in v1.

## The round loop

Phases: `idle` (between rounds) → `open` (collecting results) → `idle`, until
`finished`.

1. **Open gate.** The loop opens round N+1 only once at least
   `federation.min_participants` nodes are alive. Opening with fewer would be a
   guaranteed void; waiting also keeps round numbers dense for late joins.
   Nodes that register mid-round are *not* expected in it — they get their first
   assignment the following round.
2. **Opening** snapshots the alive nodes as "expected" and hands out `r{N}/global`.
3. **Closing** happens the moment every still-alive expected node has reported
   (`closed_reason: "all_reported"`), or when `round_deadline_s` has passed since
   opening (`closed_reason: "deadline"`). A crashed node stops being "alive" after
   liveness timeout, which can enable the fast path for the rest.
4. **Void rules**, checked in order (each produces `void: true` and a
   `void_reason`, and the same round number opens again):
   `insufficient_participants` / `insufficient_valid_updates` / `aggregation_failed`.
5. **Aggregation** is the sample-weighted mean `Σ nᵢ·wᵢ / Σ nᵢ`, per weight tensor,
   over the valid updates only.
6. After each non-void round: write `r{N+1}/global`, append the metrics line, and
   track `best.pt`. The run ends with `max_rounds`, `target_reached`, or `patience`
   (see `early_stopping`), then `final.pt` is saved and the summary line written.

## Artifacts

`runs/<name>/state.json` — the orchestrator's progress cache:

```json
{"run_name": "tiny-sim", "round": 4, "phase": "open",
 "shard_map": {"lab-1": 0, "lab-2": 1},
 "best_metric": 0.93, "best_round": 3, "status": "running", "finished_reason": null}
```

`runs/<name>/metrics.jsonl` — one line per round plus a summary line:

- round line: `type`, `round`, `void`, `void_reason`, `closed_reason`,
  `participants` (each: `node_id`, `num_samples`, `metrics`, `duration_s`,
  `device`, `update_blob_id`), `rejected` (`node_id`, `reason`), `duration_s`,
  plus optional `global_blob_id`, `aggregated`, `central`, `central_error`,
  `early_stop_metric_missing`.
- summary line: `type: "summary"`, `reason`, `best_round`, `best_metric`, plus
  optional `central_best`, `central_final`, `central_error`.

`best.pt` / `final.pt` — the two checkpoints; `config.yaml` — snapshot of the
training description; `blobs/` — every weight file that moved, exactly as it moved.

## Failure modes, and what the protocol does about them

| failure | behaviour |
|---|---|
| node crashes mid-round | swept after `liveness_timeout_s`; round closes with or without it |
| node returns late | `stale`; its blob is ignored but retained until GC |
| node starts duplicated (same id) | second registration → 409 `node id '<id>' is in use` |
| orchestrator crashes mid-round | restart recomputes the round from blobs; round N re-runs |
| blob corrupted in transit | sha256 mismatch → rejected before deserialization |
| few nodes left | open gate + `min_participants` voids keep rounds honest |
| bad central test set | `central_error` recorded; the run continues |
