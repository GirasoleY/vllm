# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct the conservative runtime slice of generic NIXL placement."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import regex as re
import torch

from vllm.distributed.kv_transfer.canonical_mapping import (
    derive_rank_canonical_mappings,
    native_vllm_dcp_rank,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlPlacementMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.placement import (
    build_nixl_placement_metadata,
)
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    ConnectorCapabilities,
    KVFormatManifest,
    KVGroupFormat,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    UniformTypeKVCacheSpecs,
    is_full_attention_spec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class NixlRuntimePlacementUnsupported(ValueError):
    """The local cache is outside the first safe generic NIXL runtime slice."""


_SUPPORTED_RUNTIME_ATTENTION_TYPES = frozenset(
    (FullAttentionSpec, MLAAttentionSpec, SinkFullAttentionSpec)
)

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)(?:layers?|blocks?|h)\.(\d+)(?:\.|$)")


def _model_layer_index(layer_name: str) -> int | None:
    """Best-effort model layer ordinal from vLLM's canonical module names."""
    match = _LAYER_INDEX_PATTERN.search(layer_name)
    if match is not None:
        return int(match.group(1))
    integer_components = [
        int(component) for component in layer_name.split(".") if component.isdigit()
    ]
    return integer_components[0] if len(integer_components) == 1 else None


def finalize_nixl_placement_cohort(
    workers: Sequence[NixlPlacementMetadata],
) -> tuple[NixlPlacementMetadata, ...]:
    """Normalize and bind a gathered cohort to one endpoint generation.

    A PP worker only receives cache metadata for its local layer slice. The
    wire protocol, however, requires every endpoint worker to advertise the
    same full-model format manifest. Merge those stage-local fragments by
    their stable positional group ID, then rewrite every local layer mapping
    against the resulting endpoint-wide semantic IDs and layer indices.
    """
    workers = tuple(workers)
    if not workers or any(
        not isinstance(worker, NixlPlacementMetadata) for worker in workers
    ):
        raise ValueError("workers must contain NixlPlacementMetadata values")

    versions = {worker.format_manifest.version for worker in workers}
    model_fingerprints = {
        worker.format_manifest.model_fingerprint for worker in workers
    }
    if len(versions) != 1 or len(model_fingerprints) != 1:
        raise ValueError(
            "placement workers must share one format version and model fingerprint"
        )

    # ``group_id`` is the scheduler-visible block-list position and is kept in
    # projected PP cache configs even when a stage owns no layer in that group.
    # Empty stage fragments are omitted by the runtime builder, so union the
    # non-empty fragments here. Byte semantics other than layer membership
    # must already agree exactly.
    groups_by_id: dict[int, list[KVGroupFormat]] = {}
    for worker in workers:
        for group in worker.format_manifest.groups:
            groups_by_id.setdefault(group.group_id, []).append(group)
    if set(groups_by_id) != set(range(len(groups_by_id))):
        raise ValueError(
            "gathered placement group IDs must form contiguous positional indices"
        )

    merged_groups: list[KVGroupFormat] = []
    group_id_by_layer: dict[str, int] = {}
    for group_id in sorted(groups_by_id):
        fragments = groups_by_id[group_id]
        first = fragments[0]
        byte_format = (
            first.kind,
            first.canonical_page_token_span,
            first.dtype,
            first.canonical_page_size_bytes,
            first.format_id,
            first.quantization,
            first.scale_dtype,
            first.scale_granularity,
        )
        layer_names: set[str] = set()
        for fragment in fragments:
            if (
                fragment.kind,
                fragment.canonical_page_token_span,
                fragment.dtype,
                fragment.canonical_page_size_bytes,
                fragment.format_id,
                fragment.quantization,
                fragment.scale_dtype,
                fragment.scale_granularity,
            ) != byte_format:
                raise ValueError(
                    f"placement group {group_id} has inconsistent byte formats"
                )
            layer_names.update(fragment.layer_names)
        ordered_layer_names = tuple(sorted(layer_names))
        for layer_name in ordered_layer_names:
            previous_group_id = group_id_by_layer.setdefault(layer_name, group_id)
            if previous_group_id != group_id:
                raise ValueError(
                    f"layer {layer_name!r} appears in multiple placement groups"
                )
        merged_groups.append(
            replace(
                first,
                semantic_id=_semantic_group_id(first.kind, ordered_layer_names),
                layer_names=ordered_layer_names,
            )
        )

    format_manifest = KVFormatManifest(
        version=versions.pop(),
        model_fingerprint=model_fingerprints.pop(),
        groups=tuple(merged_groups),
    )
    group_by_id = {group.group_id: group for group in merged_groups}
    # Preserve real model-layer ordinals emitted by stage builders whenever
    # they agree. A local fallback index restarts at zero on every PP stage;
    # detect that case by an index collision across PP ranks and replace it
    # with a deterministic endpoint-wide ordering. Multiple attention modules
    # in the same model block may legitimately share an index on one stage.
    indices_by_layer: dict[str, set[int]] = {}
    pp_ranks_by_layer: dict[str, set[int]] = {}
    for worker in workers:
        for mapping in worker.rank_placement.mappings:
            indices_by_layer.setdefault(mapping.layer_name, set()).add(
                mapping.layer_index
            )
            pp_ranks_by_layer.setdefault(mapping.layer_name, set()).add(
                worker.rank_placement.pp_rank
            )
    inconsistent_layers = sorted(
        layer_name
        for layer_name, indices in indices_by_layer.items()
        if len(indices) != 1
    )
    if inconsistent_layers:
        raise ValueError(
            f"placement workers disagree on layer indices for {inconsistent_layers}"
        )
    missing_mappings = sorted(group_id_by_layer.keys() - indices_by_layer.keys())
    multiply_staged_layers = sorted(
        layer_name
        for layer_name, pp_ranks in pp_ranks_by_layer.items()
        if len(pp_ranks) != 1
    )
    if missing_mappings or multiply_staged_layers:
        raise ValueError(
            "placement format does not have exactly one owning PP stage per layer: "
            f"missing={missing_mappings}, multi_stage={multiply_staged_layers}"
        )
    layer_indices = {
        layer_name: next(iter(indices))
        for layer_name, indices in indices_by_layer.items()
    }
    layers_by_index: dict[int, list[str]] = {}
    for layer_name, layer_index in layer_indices.items():
        layers_by_index.setdefault(layer_index, []).append(layer_name)
    has_cross_stage_collision = any(
        len(
            {
                pp_rank
                for layer_name in layer_names
                for pp_rank in pp_ranks_by_layer[layer_name]
            }
        )
        > 1
        for layer_names in layers_by_index.values()
    )
    if has_cross_stage_collision:
        if all(_model_layer_index(name) is not None for name in indices_by_layer):
            reindexed_layer_names = sorted(
                indices_by_layer,
                key=lambda name: (cast(int, _model_layer_index(name)), name),
            )
        else:
            reindexed_layer_names = sorted(indices_by_layer)
        layer_indices = {
            layer_name: layer_index
            for layer_index, layer_name in enumerate(reindexed_layer_names)
        }

    normalized_workers: list[NixlPlacementMetadata] = []
    format_fingerprint = format_manifest.fingerprint()
    for worker in workers:
        local_group_id_by_semantic_id = {
            group.semantic_id: group.group_id for group in worker.format_manifest.groups
        }
        mappings = []
        for mapping in worker.rank_placement.mappings:
            try:
                group_id = local_group_id_by_semantic_id[mapping.semantic_group_id]
                merged_group = group_by_id[group_id]
                layer_index = layer_indices[mapping.layer_name]
            except KeyError as error:
                raise ValueError(
                    f"worker {worker.rank_placement.worker_id!r} mapping is not "
                    "covered by the gathered endpoint format"
                ) from error
            if mapping.layer_name not in merged_group.layer_names:
                raise ValueError(
                    f"worker {worker.rank_placement.worker_id!r} maps layer "
                    f"{mapping.layer_name!r} outside placement group {group_id}"
                )
            mappings.append(
                replace(
                    mapping,
                    layer_index=layer_index,
                    semantic_group_id=merged_group.semantic_id,
                )
            )
        local_indices = [mapping.layer_index for mapping in mappings]
        if not local_indices:
            raise ValueError("every placement worker must own an attention layer")
        rank_placement = replace(
            worker.rank_placement,
            format_manifest_fingerprint=format_fingerprint,
            layer_range=(min(local_indices), max(local_indices) + 1),
            mappings=tuple(sorted(mappings, key=lambda mapping: mapping.layer_index)),
        )
        normalized_workers.append(
            replace(
                worker,
                format_manifest=format_manifest,
                rank_placement=rank_placement,
            )
        )

    workers = tuple(normalized_workers)

    cohort = []
    for worker in workers:
        placement = worker.rank_placement
        cohort.append(
            {
                "deployment_id": placement.deployment_id,
                "worker_id": placement.worker_id,
                "worker_incarnation": placement.worker_incarnation,
                "format_manifest": worker.format_manifest.fingerprint(),
                "rank": placement.rank,
                "tp": (placement.tp_size, placement.tp_rank),
                "dcp": (placement.dcp_size, placement.dcp_rank),
                "dcp_group_id": placement.dcp_group_id,
                "pcp": (placement.pcp_size, placement.pcp_rank),
                "pp": (placement.pp_size, placement.pp_rank),
                "dp": (placement.dp_size, placement.dp_rank),
                "dp_group_id": placement.dp_group_id,
                "ep": (placement.ep_size, placement.ep_rank),
                "cp_interleave": placement.cp_interleave,
                "layer_range": placement.layer_range,
            }
        )
    encoded = json.dumps(
        sorted(cohort, key=lambda item: (item["rank"], item["worker_id"])),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    generation = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    if generation == 0:
        generation = 1

    return tuple(
        replace(
            worker,
            rank_placement=replace(
                worker.rank_placement,
                topology_generation=generation,
            ),
        )
        for worker in workers
    )


def _group_layer_specs(group: KVCacheGroupSpec) -> dict[str, AttentionSpec]:
    layer_names = tuple(group.layer_names)
    group_spec = group.kv_cache_spec
    if not is_full_attention_spec(group_spec):
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime placement requires stable full-attention "
            "request-prefix pages"
        )
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        raw_specs: dict[str, KVCacheSpec] = group_spec.kv_cache_specs
    else:
        raw_specs = dict.fromkeys(layer_names, group_spec)
    if set(raw_specs) != set(layer_names) or any(
        type(spec) not in _SUPPORTED_RUNTIME_ATTENTION_TYPES
        for spec in raw_specs.values()
    ):
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime placement currently requires an explicitly "
            "supported, non-recycling full-attention cache semantic"
        )
    return cast(dict[str, AttentionSpec], raw_specs)


def _semantic_group_id(kind: str, layer_names: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\0".join(sorted(layer_names)).encode()).hexdigest()
    return f"{kind}:{digest}"


def _generic_format_id(kind: str, storage_family: str | None = None) -> str:
    if kind == "mla":
        # Certified MLA pages are one contiguous latent row per token. Their
        # semantic canonical coordinates do not depend on the backend's local
        # cache-layout enum.
        return "kv-placement-v1:mla:latent-token-row"
    # Certified MHA mappings use token/head semantic coordinates independent
    # of the local NHD/HND stride order. Packed interleaved K/V and split
    # K/V-half tensors still encode different canonical regions, though, and
    # are not mutually normalized yet.
    if storage_family not in ("packed-kv", "split-kv"):
        raise NixlRuntimePlacementUnsupported(
            "generic MHA placement requires a certified packed or split KV family"
        )
    return f"kv-placement-v1:mha:{storage_family}:token-head-row"


def _build_group_formats(
    kv_cache_config: KVCacheConfig,
    mappings: Mapping[str, CanonicalPageMapping],
    caches: Mapping[str, torch.Tensor],
) -> tuple[KVGroupFormat, ...]:
    formats: list[KVGroupFormat] = []
    for group_id, group in enumerate(kv_cache_config.transfer_groups):
        layer_names = tuple(group.layer_names)
        # PP projection preserves global group positions by leaving empty
        # stage-local groups in the cache config. They have no mapping or
        # registration to advertise on this worker; cohort finalization merges
        # the non-empty fragments from all stages.
        if not layer_names:
            continue
        specs = _group_layer_specs(group)
        if any(layer_name not in mappings for layer_name in layer_names):
            raise NixlRuntimePlacementUnsupported(
                "generic NIXL runtime placement requires a certified mapping "
                "for every transferred attention layer"
            )
        if any(layer_name not in caches for layer_name in layer_names):
            raise NixlRuntimePlacementUnsupported(
                "generic NIXL runtime placement requires an addressable cache "
                "for every transferred attention layer"
            )

        # The certified MLA mapper treats one page as a flat latent byte row.
        # Prove that assumption against the instantiated tensor rather than
        # silently applying it to hidden-state or strided backend storage.
        for layer_name, spec in specs.items():
            if type(spec) is not MLAAttentionSpec:
                continue
            cache = caches[layer_name]
            mapping = mappings[layer_name]
            page = cache[0]
            page_size_bytes = page.numel() * cache.element_size()
            if (
                not page.is_contiguous()
                or page_size_bytes != mapping.local_page_size_bytes
            ):
                raise NixlRuntimePlacementUnsupported(
                    f"MLA cache page for layer {layer_name!r} is not an exact "
                    "contiguous local page"
                )

        kinds = {
            "mla" if isinstance(spec, MLAAttentionSpec) else "mha"
            for spec in specs.values()
        }
        mapping_values = [mappings[layer_name] for layer_name in layer_names]
        page_spans = {mapping.canonical_token_span for mapping in mapping_values}
        page_sizes = {mapping.canonical_page_size_bytes for mapping in mapping_values}
        dtypes = {
            str(caches[name].dtype).removeprefix("torch.") for name in layer_names
        }
        quant_modes = {spec.kv_quant_mode.name.lower() for spec in specs.values()}
        if (
            len(kinds) != 1
            or len(page_spans) != 1
            or None in page_spans
            or len(page_sizes) != 1
            or len(dtypes) != 1
            or len(quant_modes) != 1
        ):
            raise NixlRuntimePlacementUnsupported(
                "a positional NIXL cache group must use one canonical page "
                "geometry, dtype, attention kind, and quantization mode"
            )
        kind = kinds.pop()
        quantization = quant_modes.pop()
        storage_family: str | None = None
        if kind == "mha":
            storage_families = {
                "packed-kv"
                if caches[layer_name].ndim == 4
                else "split-kv"
                if caches[layer_name].ndim == 5
                else "unsupported"
                for layer_name in layer_names
            }
            if len(storage_families) != 1:
                raise NixlRuntimePlacementUnsupported(
                    "a positional NIXL MHA cache group must use one storage family"
                )
            storage_family = storage_families.pop()
        formats.append(
            KVGroupFormat(
                group_id=group_id,
                semantic_id=_semantic_group_id(kind, layer_names),
                kind=kind,
                layer_names=layer_names,
                canonical_page_token_span=cast(int, page_spans.pop()),
                dtype=dtypes.pop(),
                canonical_page_size_bytes=page_sizes.pop(),
                format_id=_generic_format_id(kind, storage_family),
                quantization=quantization if quantization != "none" else None,
            )
        )
    if not formats:
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime placement has no transferable attention groups"
        )
    return tuple(formats)


def build_runtime_nixl_placement(
    *,
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    caches: Mapping[str, torch.Tensor],
    deployment_id: str,
    topology_generation: int,
    worker_id: str,
    worker_incarnation: str,
    tp_rank: int,
    pcp_rank: int = 0,
    pp_rank: int = 0,
    dp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    physical_pages_per_logical: int = 1,
    max_segments_per_batch: int | None = 4096,
) -> NixlPlacementMetadata:
    """Advertise the first runtime-safe generic NIXL READ configuration.

    Stage-local PP manifests are normalized after the full PP x PCP x TP cohort
    is gathered. PCP is supported as a replicated persistent-KV axis, including
    when PP selects each replica's layer ownership. Simultaneous PCP and DCP,
    along with aliased HMA layouts, still needs request-time ownership or
    allocator geometry that this runtime slice cannot advertise.
    """
    parallel = vllm_config.parallel_config
    pcp_size = parallel.prefill_context_parallel_size
    dcp_size = parallel.decode_context_parallel_size
    pp_size = parallel.pipeline_parallel_size
    if pcp_size > 1 and dcp_size > 1:
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime activation does not support PCP and DCP "
            "on the same endpoint"
        )
    if kv_cache_config.num_blocks <= 0:
        raise NixlRuntimePlacementUnsupported("KV cache must contain pages")
    if (
        not isinstance(physical_pages_per_logical, int)
        or isinstance(physical_pages_per_logical, bool)
        or physical_pages_per_logical <= 0
    ):
        raise NixlRuntimePlacementUnsupported(
            "physical_pages_per_logical must be a positive integer"
        )
    transfer_layer_names = {
        layer_name
        for group in kv_cache_config.transfer_groups
        for layer_name in group.layer_names
    }
    missing_transfer_caches = sorted(transfer_layer_names - caches.keys())
    if missing_transfer_caches:
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime activation is missing transferred caches: "
            f"{missing_transfer_caches}"
        )
    transfer_caches = {
        layer_name: caches[layer_name] for layer_name in transfer_layer_names
    }
    if not transfer_caches or any(
        not isinstance(cache, torch.Tensor) for cache in transfer_caches.values()
    ):
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime activation requires direct tensor caches"
        )
    expected_physical_pages = kv_cache_config.num_blocks * physical_pages_per_logical
    if any(
        cache.ndim == 0 or cache.shape[0] != expected_physical_pages
        for cache in transfer_caches.values()
    ):
        raise NixlRuntimePlacementUnsupported(
            "generic NIXL runtime activation requires every transferred cache "
            f"to contain {expected_physical_pages} physical pages"
        )

    try:
        mappings = derive_rank_canonical_mappings(
            vllm_config,
            kv_cache_config,
            dict(caches),
            tp_rank=tp_rank,
            pcp_rank=pcp_rank,
            physical_pages_per_logical=physical_pages_per_logical,
        )
    except (AssertionError, ValueError) as error:
        raise NixlRuntimePlacementUnsupported(
            "canonical KV placement could not certify this cache layout"
        ) from error
    try:
        group_formats = _build_group_formats(
            kv_cache_config,
            mappings,
            transfer_caches,
        )
        layer_names = sorted(
            layer_name for group in group_formats for layer_name in group.layer_names
        )
        # Canonical derivation intentionally sees every cache group so it can
        # validate the instantiated backend layout.  The NIXL wire manifest,
        # however, is positional over ``transfer_groups`` only.  Do not leak
        # disabled/HMA-only groups into the placement builder or their local
        # mappings would be unmatched by the advertised format.
        transfer_mappings = {
            layer_name: mappings[layer_name] for layer_name in layer_names
        }
        parsed_layer_indices = {
            layer_name: _model_layer_index(layer_name) for layer_name in layer_names
        }
        if all(index is not None for index in parsed_layer_indices.values()):
            layer_indices = {
                layer_name: cast(int, index)
                for layer_name, index in parsed_layer_indices.items()
            }
        else:
            # Unconventional model module names still get a valid stage-local
            # candidate. Cohort finalization detects PP-stage collisions and
            # deterministically reindexes the complete endpoint if necessary.
            layer_indices = {
                layer_name: index for index, layer_name in enumerate(layer_names)
            }
        tp_size = parallel.tensor_parallel_size
        dp_size = parallel.data_parallel_size
        dcp_rank = native_vllm_dcp_rank(
            tp_size=tp_size,
            tp_rank=tp_rank,
            dcp_size=dcp_size,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
        )
        dcp_group_id = (
            f"{deployment_id}:dp{dp_rank}:pp{pp_rank}:dcp{tp_rank // dcp_size}"
        )
        if pcp_size > 1:
            dcp_group_id = (
                f"{deployment_id}:dp{dp_rank}:pp{pp_rank}:pcp{pcp_rank}:"
                f"dcp{tp_rank // dcp_size}"
            )
        return build_nixl_placement_metadata(
            model_fingerprint=vllm_config.model_config.compute_hash(),
            group_formats=group_formats,
            layer_indices=layer_indices,
            mappings=transfer_mappings,
            caches=transfer_caches,
            capabilities=ConnectorCapabilities(
                contiguous_copy=True,
                strided_copy=True,
                scatter_gather=True,
                gpu_pack_unpack=False,
                supports_read=True,
                supports_write=False,
                max_segments_per_batch=max_segments_per_batch,
            ),
            deployment_id=deployment_id,
            topology_generation=topology_generation,
            worker_id=worker_id,
            worker_incarnation=worker_incarnation,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            dcp_group_id=dcp_group_id,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_group_id=f"{deployment_id}:dp",
            # EP is diagnostic only: expert sharding does not create another
            # KV-owning coordinate beyond the underlying attention process.
            ep_size=ep_size,
            ep_rank=ep_rank,
            cp_interleave=parallel.cp_kv_cache_interleave_size,
            layer_range=(
                min(layer_indices.values()),
                max(layer_indices.values()) + 1,
            ),
        )
    except NixlRuntimePlacementUnsupported:
        raise
    except (AssertionError, ValueError) as error:
        raise NixlRuntimePlacementUnsupported(
            "static generic NIXL placement validation failed"
        ) from error


__all__ = [
    "NixlRuntimePlacementUnsupported",
    "build_runtime_nixl_placement",
    "finalize_nixl_placement_cohort",
]
