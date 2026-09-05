# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    MAX_NIXL_HANDSHAKE_BYTES,
    MAX_NIXL_HANDSHAKE_REGIONS,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NixlPageRegistrationTemplate,
    NixlPlacementMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.placement import (
    build_nixl_placement_metadata,
    flatten_nixl_transfer_rank,
)
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    ConnectorCapabilities,
    CopyRun,
    KVGroupFormat,
)


def _identity_mapping(page_size: int = 4) -> CanonicalPageMapping:
    return CanonicalPageMapping(
        canonical_page_size_bytes=page_size,
        local_page_size_bytes=page_size,
        runs=(CopyRun(0, 0, page_size, 1, page_size, page_size),),
        num_writers=1,
        writer_index=0,
        canonical_token_span=page_size,
        canonical_region_token_strides=((0, 1),),
    )


def _group() -> KVGroupFormat:
    return KVGroupFormat(
        group_id=0,
        semantic_id="decoder-mla",
        kind="mla",
        layer_names=("model.layers.4.self_attn",),
        canonical_page_token_span=4,
        dtype="uint8",
        canonical_page_size_bytes=4,
        format_id="test-byte-per-token",
    )


def _capabilities() -> ConnectorCapabilities:
    return ConnectorCapabilities(
        contiguous_copy=True,
        strided_copy=True,
        scatter_gather=True,
        gpu_pack_unpack=False,
        supports_read=True,
        supports_write=False,
        max_segments_per_batch=4096,
    )


def _build(cache: torch.Tensor) -> NixlPlacementMetadata:
    layer_name = "model.layers.4.self_attn"
    return build_nixl_placement_metadata(
        model_fingerprint="model-v1",
        group_formats=(_group(),),
        layer_indices={layer_name: 4},
        mappings={layer_name: _identity_mapping()},
        caches={layer_name: cache},
        capabilities=_capabilities(),
        deployment_id="prefill",
        topology_generation=7,
        worker_id="prefill-pp1-tp1",
        worker_incarnation="boot-123",
        tp_size=2,
        tp_rank=1,
        dcp_size=2,
        dcp_rank=1,
        dcp_group_id="prefill-dp0-pp1-dcp",
        pcp_size=1,
        pcp_rank=0,
        pp_size=2,
        pp_rank=1,
        dp_size=2,
        dp_rank=0,
        dp_group_id="prefill-dp",
        ep_size=2,
        ep_rank=1,
        cp_interleave=1,
        layer_range=(4, 5),
    )


def _agent(placement: NixlPlacementMetadata) -> NixlAgentMetadata:
    return NixlAgentMetadata(
        engine_id="prefill",
        agent_metadata=b"nixl-agent",
        kv_caches_base_addr=[0x1000],
        device_id=0,
        num_blocks=3,
        block_lens=[4],
        block_strides=[4],
        kv_cache_layout="HND",
        block_size=4,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHINFER_MLA",
        physical_blocks_per_logical_kv_block=1,
        dcp_size=2,
        placement_metadata=placement,
    )


def test_static_placement_wire_metadata_strictly_round_trips():
    placement = _build(torch.empty((3, 4), dtype=torch.uint8))
    template = placement.page_registration_templates[0]

    assert placement.rank_placement.rank == 3
    assert template.page_size_bytes == 4
    assert template.page_address(2) == template.base_address + 8
    assert template.extent_end_address == template.base_address + 12
    assert (
        NixlPlacementMetadata.from_dict(json.loads(json.dumps(placement.to_dict())))
        == placement
    )
    assert (
        NixlPageRegistrationTemplate.from_dict(
            json.loads(json.dumps(template.to_dict()))
        )
        == template
    )

    agent = _agent(placement)
    encoded_agent = msgspec.msgpack.encode(agent)
    assert NixlAgentMetadata.decode(encoded_agent) == agent
    handshake = NixlHandshakePayload(
        compatibility_hash="compatibility-hash",
        placement_compatibility_hash="placement-compatibility-hash",
        agent_metadata_bytes=encoded_agent,
    )
    assert NixlHandshakePayload.decode(msgspec.msgpack.encode(handshake)) == handshake

    unknown = placement.to_dict()
    unknown["unversioned_extension"] = True
    with pytest.raises(ValueError, match="unknown=.*unversioned_extension"):
        NixlPlacementMetadata.from_dict(unknown)

    missing = template.to_dict()
    del missing["page_stride"]
    with pytest.raises(ValueError, match="missing=.*page_stride"):
        NixlPageRegistrationTemplate.from_dict(missing)


@pytest.mark.parametrize(
    ("target", "mutation", "field"),
    (
        ("outer", "unknown", "unversioned_extension"),
        ("outer", "missing", "compatibility_hash"),
        ("agent", "unknown", "unversioned_extension"),
        ("agent", "missing", "engine_id"),
        ("placement", "unknown", "unversioned_extension"),
        ("placement", "missing", "capabilities"),
    ),
)
def test_actual_handshake_wire_decode_rejects_unknown_and_missing_fields(
    target, mutation, field
):
    placement = _build(torch.empty((3, 4), dtype=torch.uint8))
    agent_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(_agent(placement)))
    outer_wire = {
        "compatibility_hash": "compatibility-hash",
        "placement_compatibility_hash": "placement-compatibility-hash",
        "agent_metadata_bytes": msgspec.msgpack.encode(agent_wire),
        "endpoint_incarnation": "endpoint-boot",
    }
    if target == "outer":
        target_wire = outer_wire
    elif target == "agent":
        target_wire = agent_wire
    else:
        target_wire = agent_wire["placement_metadata"]
    if mutation == "unknown":
        target_wire[field] = True
    else:
        del target_wire[field]
    outer_wire["agent_metadata_bytes"] = msgspec.msgpack.encode(agent_wire)

    encoded_outer = msgspec.msgpack.encode(outer_wire)
    if target == "outer":
        with pytest.raises(ValueError, match=field):
            NixlHandshakePayload.decode(encoded_outer)
        return

    decoded_outer = NixlHandshakePayload.decode(encoded_outer)
    with pytest.raises(ValueError, match=field):
        NixlAgentMetadata.decode(decoded_outer.agent_metadata_bytes)


@pytest.mark.parametrize(
    ("target", "field", "wrong_value"),
    (
        ("outer", "compatibility_hash", 7),
        ("agent", "num_blocks", "three"),
        ("placement", "page_size_bytes", "four"),
    ),
)
def test_actual_handshake_wire_decode_preserves_typed_validation(
    target, field, wrong_value
):
    placement = _build(torch.empty((3, 4), dtype=torch.uint8))
    agent_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(_agent(placement)))
    outer_wire = {
        "compatibility_hash": "compatibility-hash",
        "placement_compatibility_hash": "placement-compatibility-hash",
        "agent_metadata_bytes": msgspec.msgpack.encode(agent_wire),
        "endpoint_incarnation": "endpoint-boot",
    }
    if target == "outer":
        outer_wire[field] = wrong_value
    elif target == "agent":
        agent_wire[field] = wrong_value
    else:
        agent_wire["placement_metadata"]["page_registration_templates"][0][field] = (
            wrong_value
        )
    outer_wire["agent_metadata_bytes"] = msgspec.msgpack.encode(agent_wire)

    encoded_outer = msgspec.msgpack.encode(outer_wire)
    if target == "outer":
        with pytest.raises(ValueError, match=field):
            NixlHandshakePayload.decode(encoded_outer)
        return

    decoded_outer = NixlHandshakePayload.decode(encoded_outer)
    with pytest.raises(ValueError, match=field):
        NixlAgentMetadata.decode(decoded_outer.agent_metadata_bytes)


@pytest.mark.parametrize(
    ("decoder", "description"),
    (
        (NixlHandshakePayload.decode, "NIXL handshake"),
        (NixlAgentMetadata.decode, "NIXL agent metadata"),
    ),
)
def test_handshake_decode_rejects_oversized_payload(decoder, description):
    with pytest.raises(ValueError, match=description):
        decoder(b"\x80" + b"\x00" * MAX_NIXL_HANDSHAKE_BYTES)


@pytest.mark.parametrize(
    "field", ("kv_caches_base_addr", "block_lens", "block_strides")
)
def test_agent_decode_bounds_legacy_region_arrays(field):
    placement = _build(torch.empty((3, 4), dtype=torch.uint8))
    agent_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(_agent(placement)))
    agent_wire[field] = [0] * (MAX_NIXL_HANDSHAKE_REGIONS + 1)

    with pytest.raises(ValueError, match=f"{field} must contain at most"):
        NixlAgentMetadata.decode(msgspec.msgpack.encode(agent_wire))


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("page_registration_templates", "must contain at most"),
        ("mappings", "too many layer mappings"),
    ),
)
def test_agent_decode_bounds_placement_arrays(field, error):
    placement = _build(torch.empty((3, 4), dtype=torch.uint8))
    agent_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(_agent(placement)))
    placement_wire = agent_wire["placement_metadata"]
    if field == "page_registration_templates":
        placement_wire[field] = [placement_wire[field][0]] * (
            MAX_NIXL_HANDSHAKE_REGIONS + 1
        )
    else:
        mappings = placement_wire["rank_placement"][field]
        placement_wire["rank_placement"][field] = [mappings[0]] * (
            MAX_NIXL_HANDSHAKE_REGIONS + 1
        )

    with pytest.raises(ValueError, match=error):
        NixlAgentMetadata.decode(msgspec.msgpack.encode(agent_wire))


def test_flattened_transfer_rank_is_unique_across_pp_pcp_and_tp():
    ranks = {
        flatten_nixl_transfer_rank(
            tp_size=4,
            tp_rank=tp_rank,
            pcp_size=2,
            pcp_rank=pcp_rank,
            pp_size=3,
            pp_rank=pp_rank,
        )
        for pp_rank in range(3)
        for pcp_rank in range(2)
        for tp_rank in range(4)
    }

    assert ranks == set(range(24))
    assert (
        flatten_nixl_transfer_rank(
            tp_size=4,
            tp_rank=0,
            pcp_size=2,
            pcp_rank=0,
            pp_size=3,
            pp_rank=1,
        )
        == 8
    )


def test_builder_rejects_page_extent_past_tensor_storage():
    backing = torch.empty(5, dtype=torch.uint8)
    cache = torch.as_strided(backing, size=(2, 1), stride=(4, 1))

    with pytest.raises(ValueError, match="page registration extent.*exceeds"):
        _build(cache)


def test_request_metadata_preserves_remote_tokens_and_pcp_size():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="request-1",
        local_block_ids=([7, 8],),
        kv_transfer_params={
            "remote_block_ids": [[1, 2]],
            "remote_engine_id": "decode",
            "remote_request_id": "decode-request-1",
            "remote_host": "decode-host",
            "remote_port": 8000,
            "remote_num_tokens": 37,
            "pcp_size": 2,
        },
    )

    request = metadata.reqs_to_recv["request-1"]
    assert request.remote_num_tokens == 37
    assert request.pcp_size == 2


def test_request_metadata_defaults_remote_pcp_size_to_one():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_save(
        request_id="request-1",
        local_block_ids=([7, 8],),
        kv_transfer_params={},
    )

    assert metadata.reqs_to_save["request-1"].pcp_size == 1
