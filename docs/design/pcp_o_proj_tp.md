# PCP O-Proj tensor parallelism

PCP O-Proj TP shards the attention output-projection weight across prefill
context-parallel ranks as well as tensor-parallel ranks. It trades additional
communication for a smaller resident O-Proj parameter shard. It is disabled
by default.

## Configuration

Enable the feature with `--enable-pcp-o-proj-tp`:

```bash
vllm serve <model> \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 2 \
  --decode-context-parallel-size 1 \
  --enable-pcp-o-proj-tp \
  --enforce-eager \
  --dtype bfloat16
```

Supported configurations require:

- PCP greater than one and DCP equal to one.
- DeepSeek V2/V3/V3.2, GLM-4 MoE Lite, or GLM-5 MLA checkpoints using
  `GlmMoeDsaForCausalLM`.
- Float16 or bfloat16 activations and a bias-free O-Proj.
- An unquantized O-Proj, or ModelOpt W4A4 NVFP4 with a supported CUTLASS-compatible
  layout and no local input-axis padding. O-Proj can be excluded from
  checkpoint quantization to use the unquantized path.
- Eager execution, with DBO, LoRA, and CPU weight offloading disabled.

These constraints are validated during configuration and weight initialization.
An NVFP4 checkpoint that excludes attention from quantization exercises the
unquantized O-Proj path.

## Execution

For weight `W[N, K]`, TP size `T`, and PCP size `P`, TP rank `t` and PCP rank
`p` load:

```text
W_local[t,p] = W[:, (t*P+p)*K/(T*P) : (t*P+p+1)*K/(T*P)]
```

The ordinary checkpoint loader uses flattened rank `t*P+p` and size `T*P`.
Checkpoint names and formats remain unchanged.

| Logical batch | Weight use | PCP communication |
| --- | --- | --- |
| Prefill or mixed | Reconstruct the original TP weight shard | Gather weights before O-Proj |
| Decode only | Project the matching input-feature slice with the resident shard | Sum partial projections |

For prefill and mixed batches, attention launches an asynchronous weight gather
early and O-Proj waits for it before projecting the local tokens. For decode,
queries are replicated across PCP ranks, so each rank projects its feature
slice and PCP all-reduce combines contributions for the same tokens. Any
existing TP output reduction still runs afterward; layers with
`reduce_results=False` retain their caller-managed TP reduction.

The runner must set `ForwardContext.global_has_prefill` from the logical batch
before PCP partitioning. Local attention metadata cannot select the collective:
a short prefill may be assigned to one PCP rank while others have only decode
tokens. MTP uses the same global decision for its initial pass and selects the
decode path for subsequent replicated drafting passes. Metadata-free profiling
uses the decode path.

## Responsibilities and lifecycle

`PCPOProjTP` in `vllm/model_executor/layers/pcp_o_proj_tp.py` owns the PCP policy:
linear sharding, early prefetch, input-feature selection, and output reductions.
Attention keeps the original linear registered as `o_proj` and the optional
controller as `o_proj_tp`. The generic MLA wrapper and the NVIDIA/ROCm DSA paths
call the controller when enabled.

`GatheredWeights` in `vllm/model_executor/layers/weight_gather.py` owns full-weight
reconstruction and temporary storage aliases. Linear methods declare
`WeightGatherSpec` descriptors and validate their post-load layout. The runtime
uses a weak reference to the owning linear, preserving its checkpoint names
without registering parameters twice. Quantization methods keep their identity
and normal kernel dispatch.

After cold loading or checkpoint reload, the loader initializes registered
`GatheredWeights` modules with one `WeightGatherWorkspace` per model. Compatible
layers share full-weight buffers by layout, shape, dtype, and device. Allocation
occurs before profiling and KV-cache sizing. Gather inputs are refreshed so
in-place weight updates are visible. The full-weight context waits for the
collective and restores local storage on exit, including after an exception.

NVFP4 scale reconstruction in `quantization/utils/nvfp4_utils.py` unswizzles
CUTLASS 128x4 block scales, gathers uint8 aliases to preserve FP8 bits, and
reswizzles the full scales. The generic runtime does not interpret NVFP4
formats. Packed weights concatenate on their input axis, while replicated
global scales remain unchanged.

Attention must call `pre_attention_hook()` once before each projection. Missing
or unconsumed hooks raise an error. Sequential execution is required because
the next projection reuses the shared workspace; concurrent microbatches and
CUDA graphs require additional lifetime management.

## Memory and performance

Each resident parameter shard is `1/P` of its original TP shard. Total memory
savings are smaller: dense gathers also retain a staging copy per layer, and
full-weight and quantization-layout workspaces add storage. Separate target and
draft models own separate workspaces. Measure total memory after initialization
when evaluating capacity.

Prefill gathers may overlap with attention, but collectives on the same PCP
communicator can serialize. Decode adds a PCP reduction and replicates queries
that ordinary PCP execution could distribute. Performance therefore depends on
the model, topology, and batch mix.

## Tests

`tests/model_executor/layers/test_pcp_o_proj_tp.py` uses real process groups to
check dense reconstruction, shared workspace reuse, weight updates, exception
restoration, NVFP4 scale bits and padding, projection-path transitions, and
checkpoint names.

`test_pcp_o_proj_tp_mtp_cached_prefill` in
`tests/distributed/test_context_parallel.py` compares generation with the feature
disabled and enabled. It verifies the prefix-cache hit, accounts for MTP dropping
its trailing matched block, and chunks that recomputation to put a one-token
prefill beside decoding under a fixed synchronous token budget.
Layout tests alone do not establish quantized-kernel or full-model accuracy;
NVFP4 model evaluation and serving benchmarks remain separate validation.
