# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](../deployment/docker.md):

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

The following metrics are exposed:

## General Metrics

--8<-- "gen:metrics-general"

## Speculative Decoding Metrics

--8<-- "gen:metrics-spec-decode"

## NIXL KV Connector Metrics

--8<-- "gen:metrics-nixl"

## HiSparse KV Connector Metrics

These metrics are exposed when the `HiSparseConnector` KV connector is
configured. The host-pool gauges are reported by the scheduler once per engine
step, for the host pool shared by all tensor-parallel ranks; evictable cached
blocks count as free. With `--kv-cache-metrics`, the
`vllm:hisparse_host_block_*_seconds` histograms sample host-pool block
residency, like the device `vllm:kv_block_*_seconds` histograms, which cover
only the GPU pool.

--8<-- "gen:metrics-hisparse"

## Simple CPU Offload Connector Metrics

These metrics are exposed when the `SimpleCPUOffloadConnector` KV connector
is configured (e.g. `--kv-transfer-config='{"kv_connector":
"SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config":
{"kv_offload_backend": "disk", "disk_path": "/mnt/nvme/kv"}}'`). They are
updated once per engine step.

Caveats to keep in mind when interpreting them:

- A "completed" store means the write syscalls returned; it is **not**
  fsync-durable, and the disk backend's file is process-lifetime scratch,
  unlinked at startup and shutdown.
- `save_outcomes_total` classifies eager-mode boundary hand-off stores;
  lazy-mode stores are not classified.
- `used_blocks` counts blocks pinned by in-flight transfers or cache hits;
  warm cached blocks that are evictable are not counted. Use the
  `capacity_blocks` label of `simple_kv_offload_info` as the denominator.
- Counters and gauges are quantized to engine steps. They are reported by the
  scheduler process and reflect engine-wide logical block counts, not values
  pooled from individual tensor-parallel workers.

--8<-- "gen:metrics-simple-kv-offload"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "gen:metrics-mfu"

## Native MegaMoE Load Metrics

Native DeepGEMM MegaMoE can report expert-parallel rank imbalance without
enabling EPLB. Collection is disabled by default. Set
`VLLM_MOE_EXPERT_LOAD_CONFIG` to a JSON object before starting the API server
and workers; unset it to disable collection:

```bash
export VLLM_MOE_EXPERT_LOAD_CONFIG='{
  "directory": "/tmp/vllm-moe-load",
  "ep_world": 4,
  "interval_s": 10,
  "rank_metrics": true,
  "max_samples": 360
}'
```

The configuration applies to one EP replica on one host. Its worker processes
and API server must share the directory and hostname; give each replica its
own directory. `ep_world` must match the number of EP ranks. Pipeline-parallel
or multi-host EP replicas are not supported. All recorded target and draft
layers must have the same number of local physical experts. This feature
currently instruments `DeepGemmMegaMoEExperts`; other MoE backends do not
produce these counters.

| Configuration | Meaning |
| --- | --- |
| `directory` | Required writable directory for atomic CPU summaries. |
| `ep_world` | Required number of EP ranks in the replica, from 1 to 256. |
| `interval_s` | Sampling interval, from 1 to 3600 seconds. Default: 10. |
| `stale_after_s` | Maximum observation age for imbalance values. Default: three sampling intervals. Must be at least `interval_s`. |
| `rank_metrics` | Enable optional cumulative counters by layer and rank. Default: `false`. |
| `target_layers` | Number of registered target layers preceding draft layers. Set this when using speculative decoding; otherwise all registered layers are labeled `target`. |
| `max_samples` | Expert-capture budget, from 0 to 1440. Default: 360. Zero disables expert captures. |
| `collection_id` | Optional identifier shared by workers and collector to reject summaries from another collection generation. |

The existing `/metrics` endpoint exposes:

| Metric | Meaning |
| --- | --- |
| `vllm:moe_rank_imbalance_average` | Assignment-weighted layer average of max-rank load divided by mean-rank load. |
| `vllm:moe_rank_imbalance_worst` | Largest active layer max-rank/mean-rank ratio. |
| `vllm:moe_received_assignments_total` | Optional cumulative native receive assignments, with `moe_layer`, `model_part`, and `ep_rank` labels. |
| `vllm:moe_collection_age_seconds` | Age of the oldest rank's observation used for the current window. Absent before observations arrive. |
| `vllm:moe_collection_complete` | One when every EP rank has compatible, fresh observations for a matching window; zero otherwise. |
| `vllm:moe_collection_skew_seconds` | Largest timestamp spread across ranks at either window boundary. |

The two imbalance metrics have a `model_part` label (`target` or `draft`). A
value of 1 means balanced rank assignment counts. Idle windows produce no
imbalance value. Missing, stale, restarted, or incompatible ranks suppress
imbalance values until a complete window is available. The first observation
is a baseline, excluding warmup assignments from subsequent differences.
These counts measure native physical expert receives; they do not include
the shared expert or represent per-request latency. Redundant experts are
physical slots, so counts describe the placement actually executed by EPLB.

Each worker reserves a counter bank and two pinned host buffers during model
initialization, with capacity for 256 layers. It enqueues asynchronous copies
after execution and publishes CPU summaries from a background thread. The
serving thread does not wait for a copy or filesystem operation. A slow
publisher skips samples when both buffers are busy. Four summary
observations per rank bound online storage. Optional rank series add
`registered_layers * ep_world` counters per replica; there are no online
expert labels. Scrape-time aggregation reads only CPU summaries.

### Bounded Expert Captures

Request an expert-level capture in the same configured environment and
filesystem namespace:

```bash
python -m vllm.v1.metrics.moe_load --capture imbalance-check --samples 6
```

The capture ID accepts up to 64 letters, digits, underscores, or hyphens.
`--samples` must not exceed `max_samples`. Each rank writes at most that many
snapshots under `directory/<hostname>/online/diagnostics/rank<rank>/`, and
continues publishing online metrics after the capture finishes. The first
snapshot is a capture baseline. Re-arming with a new ID reuses the numbered
snapshot files; filter by `capture_id` when reading an archive. Compute
per-expert differences modulo 2**32 because native int32 counters can wrap.

## Custom Histogram Buckets

The core engine histograms ship with default bucket boundaries tuned for
typical serving workloads. The `--custom-histogram-buckets` option replaces
the boundaries of one or more *bucket families* — exactly the histograms
listed in the table below — with your own list; histograms owned by other
subsystems (for example, the NIXL connector metrics) are not affected. Use it,
for example, to track sub-300ms latency SLOs with the request-phase
histograms, whose smallest default boundary is 0.3s:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --custom-histogram-buckets '{"request_latency": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 30.0]}'
```

Each family key overrides a group of related histograms:

| Family key | Histograms |
| --- | --- |
| `request_latency` | `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`, `vllm:request_inference_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` |
| `time_to_first_token` | `vllm:time_to_first_token_seconds` |
| `inter_token_latency` | `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds` |
| `iteration_tokens` | `vllm:iteration_tokens_total` |
| `request_params_n` | `vllm:request_params_n` |
| `request_num_preemptions` | `vllm:request_num_preemptions` |
| `request_tokens` | `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, `vllm:request_max_num_generation_tokens`, `vllm:request_params_max_tokens`, `vllm:request_prefill_kv_computed_tokens` |
| `kv_cache_residency` | `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds` |

Bucket values must be positive, finite, and strictly increasing; unknown
family keys are rejected at startup. Families you do not list keep their
default boundaries. The `request_tokens` defaults normally scale with
`--max-model-len`; an override replaces that computed list. The
`kv_cache_residency` family only takes effect when `--kv-cache-metrics` is
enabled.

!!! warning "Bucket cardinality"
    Every bucket boundary creates one extra time series per metric and per
    label combination (model and engine index, multiplied under data-parallel
    deployments). Long bucket lists inflate Prometheus storage, scrape sizes,
    and query costs. Keep custom lists short, and only override the families
    you actively monitor.

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
