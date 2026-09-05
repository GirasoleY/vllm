# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.segmented import (
    NixlDirectDescriptorBatch,
    NixlEphemeralDlistTracker,
    NixlPageRegistration,
    build_nixl_direct_descriptor_batches,
    iter_nixl_direct_descriptor_batches,
    iter_nixl_direct_descriptor_batches_streaming,
    prepare_nixl_direct_batch,
    prepare_nixl_direct_batches,
)
from vllm.distributed.kv_transfer.kv_placement import TransferRun


def _pages(*entries: tuple[int, int, int, int, int]):
    return {
        (rank, page_id): NixlPageRegistration(base, length, device)
        for rank, page_id, base, length, device in entries
    }


class _FakeNixlWrapper:
    def __init__(self):
        self._next_handle = 100
        self.prepared_dlists: list[
            tuple[str, tuple[tuple[int, int, int], ...], int]
        ] = []
        self.prepared_xfers: list[tuple] = []
        self.released_xfers: list[int] = []
        self.released_dlists: list[int] = []
        self.fail_make_xfer = False

    def get_xfer_descs(self, descriptors, memory_type):
        assert memory_type == "VRAM"
        return tuple(descriptors)

    def prep_xfer_dlist(self, agent_name, descriptors):
        handle = self._next_handle
        self._next_handle += 1
        self.prepared_dlists.append((agent_name, tuple(descriptors), handle))
        return handle

    def make_prepped_xfer(
        self,
        operation,
        local_handle,
        local_ids,
        remote_handle,
        remote_ids,
    ):
        if self.fail_make_xfer:
            raise RuntimeError("injected make_prepped_xfer failure")
        handle = self._next_handle
        self._next_handle += 1
        self.prepared_xfers.append(
            (
                operation,
                local_handle,
                local_ids.copy(),
                remote_handle,
                remote_ids.copy(),
                handle,
            )
        )
        return handle

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)

    def release_dlist_handle(self, handle):
        self.released_dlists.append(handle)


def test_affine_runs_expand_to_direct_nixl_descriptor_pairs():
    runs = (
        TransferRun(
            source_rank=2,
            destination_rank=7,
            source_page_id=11,
            destination_page_id=19,
            source_offset=4,
            destination_offset=8,
            fragment_size=2,
            fragment_count=3,
            source_stride=8,
            destination_stride=4,
        ),
    )

    batches = build_nixl_direct_descriptor_batches(
        runs,
        _pages((2, 11, 1000, 24, 3)),
        _pages((7, 19, 2000, 18, 5)),
    )

    assert batches == (
        NixlDirectDescriptorBatch(
            source_rank=2,
            destination_rank=7,
            source_descriptors=(
                (1004, 2, 3),
                (1012, 2, 3),
                (1020, 2, 3),
            ),
            destination_descriptors=(
                (2008, 2, 5),
                (2012, 2, 5),
                (2016, 2, 5),
            ),
        ),
    )
    assert batches[0].segment_count == 3
    assert batches[0].total_bytes == 6
    assert (batches[0].batch_index, batches[0].batch_count) == (0, 1)
    assert not batches[0].requires_aggregate_completion


def test_batches_are_grouped_by_rank_pair_and_oriented_for_read_or_write():
    runs = (
        TransferRun(1, 9, 0, 0, 0, 0, 4, 1, 4, 4),
        TransferRun(0, 9, 0, 0, 4, 4, 4, 1, 4, 4),
    )
    batches = build_nixl_direct_descriptor_batches(
        runs,
        _pages((0, 0, 100, 8, 0), (1, 0, 200, 8, 1)),
        _pages((9, 0, 900, 8, 9)),
    )

    assert [(batch.source_rank, batch.destination_rank) for batch in batches] == [
        (0, 9),
        (1, 9),
    ]
    remote_rank, local, remote = batches[0].transfer_sides("READ", local_rank=9)
    assert remote_rank == 0
    assert local == ((904, 4, 9),)
    assert remote == ((104, 4, 0),)

    remote_rank, local, remote = batches[1].transfer_sides("WRITE", local_rank=1)
    assert remote_rank == 9
    assert local == ((200, 4, 1),)
    assert remote == ((900, 4, 9),)


def test_high_fragmentation_is_split_only_into_more_direct_batches():
    fragment_count = 12_289
    run = TransferRun(
        source_rank=0,
        destination_rank=1,
        source_page_id=4,
        destination_page_id=5,
        source_offset=0,
        destination_offset=0,
        fragment_size=1,
        fragment_count=fragment_count,
        source_stride=2,
        destination_stride=3,
    )

    batches = build_nixl_direct_descriptor_batches(
        [run],
        _pages((0, 4, 10_000, fragment_count * 2, 0)),
        _pages((1, 5, 20_000, fragment_count * 3, 1)),
        max_segments_per_batch=4096,
    )

    assert [batch.segment_count for batch in batches] == [4096, 4096, 4096, 1]
    assert [batch.batch_index for batch in batches] == [0, 1, 2, 3]
    assert all(batch.batch_count == 4 for batch in batches)
    assert all(batch.requires_aggregate_completion for batch in batches)
    assert sum(batch.segment_count for batch in batches) == fragment_count
    assert sum(batch.total_bytes for batch in batches) == fragment_count
    assert batches[-1].source_descriptors[-1] == (
        10_000 + (fragment_count - 1) * 2,
        1,
        0,
    )
    assert batches[-1].destination_descriptors[-1] == (
        20_000 + (fragment_count - 1) * 3,
        1,
        1,
    )

    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    prepared = prepare_nixl_direct_batches(
        wrapper,
        tracker,
        batches,
        operation="WRITE",
        local_rank=0,
        remote_agents={1: "destination-agent-1"},
        memory_type="VRAM",
    )
    assert len(prepared) == 4
    assert [len(xfer[2]) for xfer in wrapper.prepared_xfers] == [4096, 4096, 4096, 1]
    assert tracker.release_all() == 4


def test_streaming_affine_run_emits_bounded_direct_batches():
    fragment_count = 12_289
    run = TransferRun(0, 1, 4, 5, 0, 0, 1, fragment_count, 2, 3)

    batches = tuple(
        iter_nixl_direct_descriptor_batches_streaming(
            iter((run,)),
            _pages((0, 4, 10_000, fragment_count * 2, 0)),
            _pages((1, 5, 20_000, fragment_count * 3, 1)),
        )
    )

    assert [batch.segment_count for batch in batches] == [4096, 4096, 4096, 1]
    assert [batch.batch_index for batch in batches] == [0, 1, 2, 3]
    assert all(batch.batch_count is None for batch in batches)
    assert all(batch.requires_aggregate_completion for batch in batches)
    assert sum(batch.segment_count for batch in batches) == fragment_count


def test_streaming_buffers_are_globally_bounded_across_peers():
    consumed_runs = 0

    def runs():
        nonlocal consumed_runs
        for index in range(12):
            consumed_runs += 1
            source_rank = index % 4
            yield TransferRun(
                source_rank,
                9,
                0,
                0,
                index // 4,
                index,
                1,
                1,
                1,
                1,
            )

    stream = iter_nixl_direct_descriptor_batches_streaming(
        runs(),
        _pages(*((rank, 0, rank * 100, 3, rank) for rank in range(4))),
        _pages((9, 0, 900, 12, 9)),
        max_segments_per_batch=4,
        max_buffered_segments=4,
    )

    first = next(stream)
    # The fifth input descriptor triggers pressure before it is buffered; the
    # remaining run stream has not been consumed merely to derive batch_count.
    assert consumed_runs == 5
    assert (first.source_rank, first.destination_rank) == (0, 9)
    assert first.segment_count == 1
    assert first.batch_count is None

    batches = (first, *stream)
    assert consumed_runs == 12
    assert sum(batch.segment_count for batch in batches) == 12
    assert all(batch.segment_count <= 4 for batch in batches)
    indices_by_peer: dict[tuple[int, int], list[int]] = {}
    for batch in batches:
        indices_by_peer.setdefault(
            (batch.source_rank, batch.destination_rank), []
        ).append(batch.batch_index)
    assert all(
        indices == list(range(len(indices))) for indices in indices_by_peer.values()
    )


def test_no_batch_hint_keeps_arbitrarily_fragmented_peer_plan_direct():
    fragment_count = 4097
    run = TransferRun(0, 1, 0, 0, 0, 0, 1, fragment_count, 1, 1)

    batches = build_nixl_direct_descriptor_batches(
        [run],
        _pages((0, 0, 0, fragment_count, 0)),
        _pages((1, 0, 0, fragment_count, 1)),
    )

    assert len(batches) == 1
    assert batches[0].segment_count == fragment_count


def test_bounded_batches_stream_before_a_later_peer_is_lowered():
    runs = (
        TransferRun(0, 1, 0, 0, 0, 0, 1, 3, 1, 1),
        TransferRun(2, 3, 0, 0, 0, 0, 1, 1, 1, 1),
    )
    batches = iter_nixl_direct_descriptor_batches(
        runs,
        _pages((0, 0, 100, 3, 0)),
        _pages((1, 0, 200, 3, 1)),
        max_segments_per_batch=2,
    )

    first = next(batches)
    assert (first.source_rank, first.destination_rank, first.segment_count) == (
        0,
        1,
        2,
    )
    assert next(batches).segment_count == 1
    with pytest.raises(ValueError, match="missing source page registration"):
        next(batches)


def test_batch_completion_indices_reset_for_each_peer_pair():
    runs = (
        TransferRun(0, 2, 0, 0, 0, 0, 1, 3, 1, 1),
        TransferRun(1, 2, 0, 0, 0, 3, 1, 3, 1, 1),
    )

    batches = build_nixl_direct_descriptor_batches(
        runs,
        _pages((0, 0, 100, 3, 0), (1, 0, 200, 3, 1)),
        _pages((2, 0, 300, 6, 2)),
        max_segments_per_batch=2,
    )

    assert [
        (batch.source_rank, batch.batch_index, batch.batch_count) for batch in batches
    ] == [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]


def test_prepare_read_batch_tracks_ephemeral_descriptor_lists():
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    batch = NixlDirectDescriptorBatch(
        3,
        7,
        ((100, 4, 3), (108, 4, 3)),
        ((200, 4, 7), (204, 4, 7)),
    )

    prepared = prepare_nixl_direct_batch(
        wrapper,
        tracker,
        batch,
        operation="READ",
        local_rank=7,
        remote_agent_name="source-agent-3",
        memory_type="VRAM",
    )

    assert wrapper.prepared_dlists == [
        ("NIXL_INIT_AGENT", batch.destination_descriptors, 100),
        ("source-agent-3", batch.source_descriptors, 101),
    ]
    operation, local_handle, local_ids, remote_handle, remote_ids, handle = (
        wrapper.prepared_xfers[0]
    )
    assert operation == "READ"
    assert (local_handle, remote_handle, handle) == (100, 101, 102)
    assert local_ids.tolist() == [0, 1]
    assert remote_ids.tolist() == [0, 1]
    assert str(local_ids.dtype) == str(remote_ids.dtype) == "int32"
    assert prepared.descriptor_batch is batch
    assert prepared.transfer_handle == 102
    assert tracker.pending_count == 1

    assert tracker.release(prepared.transfer_handle)
    assert not tracker.release(prepared.transfer_handle)
    assert wrapper.released_xfers == [102]
    assert wrapper.released_dlists == [100, 101]


def test_prepare_write_batch_uses_source_as_local_side():
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    batch = NixlDirectDescriptorBatch(2, 8, ((10, 4, 2),), ((20, 4, 8),))

    prepared = prepare_nixl_direct_batch(
        wrapper,
        tracker,
        batch,
        operation="WRITE",
        local_rank=2,
        remote_agent_name="destination-agent-8",
        memory_type="VRAM",
    )

    assert wrapper.prepared_dlists == [
        ("NIXL_INIT_AGENT", batch.source_descriptors, 100),
        ("destination-agent-8", batch.destination_descriptors, 101),
    ]
    assert wrapper.prepared_xfers[0][0] == "WRITE"
    assert tracker.release(prepared.transfer_handle)


def test_prepare_failure_releases_both_untracked_descriptor_lists():
    wrapper = _FakeNixlWrapper()
    wrapper.fail_make_xfer = True
    tracker = NixlEphemeralDlistTracker(wrapper)
    batch = NixlDirectDescriptorBatch(0, 1, ((10, 4, 0),), ((20, 4, 1),))

    with pytest.raises(RuntimeError, match="injected"):
        prepare_nixl_direct_batch(
            wrapper,
            tracker,
            batch,
            operation="WRITE",
            local_rank=0,
            remote_agent_name="destination-agent-1",
            memory_type="VRAM",
        )

    assert tracker.pending_count == 0
    assert wrapper.released_xfers == []
    assert wrapper.released_dlists == [100, 101]


def test_prepare_rejects_tracker_for_another_wrapper():
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(_FakeNixlWrapper())
    batch = NixlDirectDescriptorBatch(0, 1, ((10, 4, 0),), ((20, 4, 1),))

    with pytest.raises(ValueError, match="another wrapper"):
        prepare_nixl_direct_batch(
            wrapper,
            tracker,
            batch,
            operation="WRITE",
            local_rank=0,
            remote_agent_name="destination-agent-1",
            memory_type="VRAM",
        )

    assert not wrapper.prepared_dlists


def test_group_prepare_failure_releases_earlier_peer_resources():
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    batches = (
        NixlDirectDescriptorBatch(0, 1, ((10, 4, 0),), ((20, 4, 1),)),
        NixlDirectDescriptorBatch(0, 2, ((14, 4, 0),), ((30, 4, 2),)),
    )

    with pytest.raises(ValueError, match="remote agent for rank 2"):
        prepare_nixl_direct_batches(
            wrapper,
            tracker,
            batches,
            operation="WRITE",
            local_rank=0,
            remote_agents={1: "destination-agent-1"},
            memory_type="VRAM",
        )

    assert tracker.pending_count == 0
    assert wrapper.released_xfers == [102]
    assert wrapper.released_dlists == [100, 101]


def test_tracker_shutdown_releases_each_prepared_batch_exactly_once():
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    batches = build_nixl_direct_descriptor_batches(
        [TransferRun(0, 1, 0, 0, 0, 0, 1, 5, 1, 1)],
        _pages((0, 0, 1000, 5, 0)),
        _pages((1, 0, 2000, 5, 1)),
        max_segments_per_batch=2,
    )
    prepared = [
        prepare_nixl_direct_batch(
            wrapper,
            tracker,
            batch,
            operation="WRITE",
            local_rank=0,
            remote_agent_name="destination-agent-1",
            memory_type="VRAM",
        )
        for batch in batches
    ]

    assert tracker.pending_count == 3
    assert tracker.release(prepared[0].transfer_handle)
    assert tracker.release_all() == 2
    assert tracker.release_all() == 0
    assert sorted(wrapper.released_xfers) == sorted(
        item.transfer_handle for item in prepared
    )
    assert len(wrapper.released_dlists) == 2 * len(prepared)
    assert len(set(wrapper.released_dlists)) == len(wrapper.released_dlists)
    with pytest.raises(RuntimeError, match="closed"):
        tracker.track(999, 1000, 1001)


@pytest.mark.parametrize(
    ("operation", "local_rank", "match"),
    [
        ("READ", 0, "destination rank"),
        ("WRITE", 1, "source rank"),
        ("write", 0, "canonical value"),
        ("COPY", 0, "canonical value"),
    ],
)
def test_transfer_orientation_fails_closed(operation: str, local_rank: int, match: str):
    batch = NixlDirectDescriptorBatch(0, 1, ((10, 4, 0),), ((20, 4, 1),))

    with pytest.raises(ValueError, match=match):
        batch.transfer_sides(operation, local_rank)


def test_missing_page_and_out_of_bounds_run_fail_closed():
    run = TransferRun(0, 1, 3, 4, 4, 0, 4, 2, 8, 4)

    with pytest.raises(ValueError, match="missing source page registration"):
        build_nixl_direct_descriptor_batches(
            [run],
            {},
            _pages((1, 4, 0, 8, 0)),
        )

    with pytest.raises(ValueError, match="source transfer extent"):
        build_nixl_direct_descriptor_batches(
            [run],
            _pages((0, 3, 0, 15, 0)),
            _pages((1, 4, 0, 8, 0)),
        )


@pytest.mark.parametrize("limit", [0, -1, True])
def test_batch_hint_must_be_positive(limit: int):
    with pytest.raises(ValueError, match="max_segments_per_batch"):
        build_nixl_direct_descriptor_batches([], {}, {}, max_segments_per_batch=limit)
