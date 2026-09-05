# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build static NIXL handshake metadata for generic KV placement.

This module deliberately stops at describing registered, token-addressable
attention pages. Request allocation, source/destination plan composition, and
NIXL transfer submission remain separate runtime concerns.
"""

from collections.abc import Mapping, Sequence

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlPageRegistrationTemplate,
    NixlPlacementMetadata,
)
from vllm.distributed.kv_transfer.kv_placement import (
    KV_PLACEMENT_PROTOCOL_VERSION,
    CanonicalPageMapping,
    ConnectorCapabilities,
    KVFormatManifest,
    KVGroupFormat,
    LayerPageMapping,
    RankPlacementManifest,
)

_TOKEN_MAPPED_ATTENTION_KINDS = frozenset({"attention", "mha", "mla"})


def _validate_axis(size: int, rank: int, name: str) -> None:
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{name}_size must be a positive integer")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank >= size:
        raise ValueError(f"{name}_rank must be in [0, {size})")


def flatten_nixl_transfer_rank(
    *,
    tp_size: int,
    tp_rank: int,
    pcp_size: int,
    pcp_rank: int,
    pp_size: int,
    pp_rank: int,
) -> int:
    """Flatten PP x PCP x TP coordinates into a DP-local transport rank.

    DP is a request-routing axis and therefore selects a separate endpoint
    replica before this rank is interpreted. DCP and EP do not add KV-owning
    processes beyond the underlying TP coordinate.
    """
    _validate_axis(tp_size, tp_rank, "tp")
    _validate_axis(pcp_size, pcp_rank, "pcp")
    _validate_axis(pp_size, pp_rank, "pp")
    return (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank


def _validate_certified_mapping(
    layer_name: str,
    group: KVGroupFormat,
    mapping: CanonicalPageMapping,
) -> None:
    if group.kind not in _TOKEN_MAPPED_ATTENTION_KINDS:
        raise ValueError(
            f"layer {layer_name!r} uses unsupported KV kind {group.kind!r}; "
            "the generic NIXL READ slice only certifies token-mapped attention"
        )
    if mapping.is_opaque:
        raise ValueError(
            f"layer {layer_name!r} has an opaque mapping; generic NIXL "
            "placement requires a certified token mapping"
        )
    if mapping.canonical_token_span is None:
        raise ValueError(f"layer {layer_name!r} mapping has no canonical token span")
    region_strides = dict(mapping.canonical_region_token_strides)
    run_regions = {run.canonical_region for run in mapping.runs}
    missing_regions = sorted(run_regions - region_strides.keys())
    if missing_regions:
        raise ValueError(
            f"layer {layer_name!r} mapping lacks token strides for canonical "
            f"regions {missing_regions}"
        )
    if mapping.canonical_token_span != group.canonical_page_token_span:
        raise ValueError(
            f"layer {layer_name!r} canonical token span does not match its "
            "KV group format"
        )
    if mapping.canonical_page_size_bytes != group.canonical_page_size_bytes:
        raise ValueError(
            f"layer {layer_name!r} canonical page size does not match its "
            "KV group format"
        )


def _build_page_registration(
    layer_name: str,
    cache: torch.Tensor,
    mapping: CanonicalPageMapping,
) -> NixlPageRegistrationTemplate:
    if not isinstance(cache, torch.Tensor):
        raise ValueError(f"cache for layer {layer_name!r} must be a tensor")
    if cache.layout is not torch.strided:
        raise ValueError(f"cache for layer {layer_name!r} must use strided layout")
    if cache.ndim == 0 or cache.shape[0] <= 0:
        raise ValueError(
            f"cache for layer {layer_name!r} must contain at least one page"
        )
    if any(stride < 0 for stride in cache.stride()):
        raise ValueError(
            f"cache for layer {layer_name!r} must have non-negative strides"
        )

    page_stride = cache.stride(0) * cache.element_size()
    device_id = cache.device.index if cache.device.index is not None else 0
    template = NixlPageRegistrationTemplate(
        layer_name=layer_name,
        base_address=cache.data_ptr(),
        page_stride=page_stride,
        page_size_bytes=mapping.local_page_size_bytes,
        num_pages=cache.shape[0],
        device_id=device_id,
    )

    storage = cache.untyped_storage()
    storage_start = storage.data_ptr()
    storage_end = storage_start + storage.nbytes()
    if template.base_address < storage_start or (
        template.extent_end_address > storage_end
    ):
        raise ValueError(
            f"layer {layer_name!r} page registration extent "
            f"[{template.base_address}, {template.extent_end_address}) exceeds "
            f"tensor storage [{storage_start}, {storage_end})"
        )
    return template


def build_nixl_placement_metadata(
    *,
    model_fingerprint: str,
    group_formats: Sequence[KVGroupFormat],
    layer_indices: Mapping[str, int],
    mappings: Mapping[str, CanonicalPageMapping],
    caches: Mapping[str, torch.Tensor],
    capabilities: ConnectorCapabilities,
    deployment_id: str,
    topology_generation: int,
    worker_id: str,
    worker_incarnation: str,
    tp_size: int,
    tp_rank: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_group_id: str,
    pcp_size: int,
    pcp_rank: int,
    pp_size: int,
    pp_rank: int,
    dp_size: int,
    dp_rank: int,
    dp_group_id: str,
    ep_size: int,
    ep_rank: int,
    cp_interleave: int,
    layer_range: tuple[int, int],
) -> NixlPlacementMetadata:
    """Build a validated generic READ handshake payload from explicit state.

    ``group_formats`` may describe attention layers owned by other PP stages;
    ``mappings`` selects the layers owned by this worker. Extra cache tensors
    are ignored so hybrid models can keep non-attention state out of this
    safely shippable protocol slice.
    """
    if not isinstance(capabilities, ConnectorCapabilities):
        raise ValueError("capabilities must be ConnectorCapabilities")
    if not capabilities.supports_read:
        raise ValueError("generic NIXL placement currently requires READ support")
    if isinstance(group_formats, (str, bytes, Mapping)):
        raise ValueError("group_formats must be a sequence")
    for name, value in (
        ("layer_indices", layer_indices),
        ("mappings", mappings),
        ("caches", caches),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")

    format_manifest = KVFormatManifest(
        version=KV_PLACEMENT_PROTOCOL_VERSION,
        model_fingerprint=model_fingerprint,
        groups=tuple(group_formats),
    )
    unsupported_groups = sorted(
        group.semantic_id
        for group in format_manifest.groups
        if group.kind not in _TOKEN_MAPPED_ATTENTION_KINDS
    )
    if unsupported_groups:
        raise ValueError(
            "generic NIXL placement only advertises token-mapped attention "
            f"groups; unsupported semantic groups: {unsupported_groups}"
        )
    group_by_layer = {
        layer_name: group
        for group in format_manifest.groups
        for layer_name in group.layer_names
    }
    selected_layers = set(mappings)
    if not selected_layers:
        raise ValueError("mappings must advertise at least one local attention layer")
    unknown_layers = sorted(selected_layers - group_by_layer.keys())
    if unknown_layers:
        raise ValueError(
            f"mappings contain layers absent from the KV format: {unknown_layers}"
        )
    missing_indices = sorted(selected_layers - layer_indices.keys())
    if missing_indices:
        raise ValueError(f"layer_indices are missing layers: {missing_indices}")
    missing_caches = sorted(selected_layers - caches.keys())
    if missing_caches:
        raise ValueError(f"caches are missing layers: {missing_caches}")

    ordered_layers = sorted(
        selected_layers, key=lambda layer_name: (layer_indices[layer_name], layer_name)
    )
    layer_mappings: list[LayerPageMapping] = []
    registrations: list[NixlPageRegistrationTemplate] = []
    for layer_name in ordered_layers:
        group = group_by_layer[layer_name]
        mapping = mappings[layer_name]
        cache = caches[layer_name]
        if not isinstance(mapping, CanonicalPageMapping):
            raise ValueError(
                f"mapping for layer {layer_name!r} must be a CanonicalPageMapping"
            )
        if not isinstance(cache, torch.Tensor):
            raise ValueError(f"cache for layer {layer_name!r} must be a tensor")
        _validate_certified_mapping(layer_name, group, mapping)
        cache_dtype = str(cache.dtype).removeprefix("torch.")
        if cache_dtype != group.dtype:
            raise ValueError(
                f"layer {layer_name!r} cache dtype {cache_dtype!r} does not "
                f"match advertised dtype {group.dtype!r}"
            )
        layer_mappings.append(
            LayerPageMapping(
                layer_name=layer_name,
                layer_index=layer_indices[layer_name],
                semantic_group_id=group.semantic_id,
                mapping=mapping,
            )
        )
        registrations.append(_build_page_registration(layer_name, cache, mapping))

    rank_placement = RankPlacementManifest(
        version=KV_PLACEMENT_PROTOCOL_VERSION,
        deployment_id=deployment_id,
        topology_generation=topology_generation,
        worker_id=worker_id,
        worker_incarnation=worker_incarnation,
        format_manifest_fingerprint=format_manifest.fingerprint(),
        rank=flatten_nixl_transfer_rank(
            tp_size=tp_size,
            tp_rank=tp_rank,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
        ),
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
        dp_group_id=dp_group_id,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cp_interleave=cp_interleave,
        layer_range=layer_range,
        mappings=tuple(layer_mappings),
    )
    return NixlPlacementMetadata(
        format_manifest=format_manifest,
        rank_placement=rank_placement,
        capabilities=capabilities,
        page_registration_templates=tuple(registrations),
    )


__all__ = ["build_nixl_placement_metadata", "flatten_nixl_transfer_rank"]
