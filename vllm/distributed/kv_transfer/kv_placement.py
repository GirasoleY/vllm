# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector-independent KV page placement and transfer-plan composition.

Attention backends describe a local page as byte correspondences into a
parallelism-neutral canonical page.  A connector only needs the result of
composing a producer's correspondences with a consumer's correspondences.

The composer deliberately has no descriptor-count or fragmentation limit.
Connector implementations may batch the returned runs, but a segmented direct
copy remains a valid plan regardless of how many runs it contains.
"""

import hashlib
import heapq
import json
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

KV_PLACEMENT_PROTOCOL_VERSION = 1
# Wire-safety ceilings, not transfer-policy thresholds. A valid affine run may
# still describe an unbounded number of direct segments; these limits bound the
# compact static manifest itself while it is decoded and validated.
KV_PLACEMENT_MAX_GROUPS = 1024
KV_PLACEMENT_MAX_LAYERS = 4096
KV_PLACEMENT_MAX_MAPPINGS = 4096
KV_PLACEMENT_MAX_RUNS = 262_144


@dataclass(frozen=True)
class CopyRun:
    """A strided correspondence between local and canonical page bytes.

    For ``i`` in ``range(num_fragments)``, fragment ``i`` maps
    ``[local_offset + i * local_stride, +fragment_size)`` to
    ``[canonical_offset + i * canonical_stride, +fragment_size)``.
    """

    local_offset: int
    canonical_offset: int
    fragment_size: int
    num_fragments: int
    local_stride: int
    canonical_stride: int
    # Canonical coordinates are partitioned into semantic regions (for
    # example, one region per K/V head).  This keeps coordinates stable when
    # two endpoints use different page token spans: head-major pages cannot be
    # represented correctly by concatenating compact page byte offsets.
    canonical_region: int = 0
    # Offloading stores a compact canonical page.  When its physical byte
    # layout differs from the semantic coordinate above, these fields describe
    # that compact destination.  Connectors ignore them.
    canonical_storage_offset: int | None = None
    canonical_storage_stride: int | None = None

    @property
    def storage_offset(self) -> int:
        """Offset in the compact canonical page used by offloading."""
        if self.canonical_storage_offset is None:
            return self.canonical_offset
        return self.canonical_storage_offset

    @property
    def storage_stride(self) -> int:
        """Fragment stride in the compact canonical page used by offloading."""
        if self.canonical_storage_stride is None:
            return self.canonical_stride
        return self.canonical_storage_stride


@dataclass(frozen=True)
class CanonicalPageMapping:
    """How one rank's local page maps into a canonical page.

    ``num_writers`` equivalent replicas use ``writer_index`` to elect exactly
    one source for each canonical page.  Election rotates by canonical page
    index so replicated ranks share transfer work.
    """

    canonical_page_size_bytes: int
    local_page_size_bytes: int
    runs: tuple[CopyRun, ...]
    num_writers: int
    writer_index: int
    parallelism_agnostic: bool = False
    # Optional semantic geometry used by connector composition. Offloading-only
    # opaque mappings may omit it. Region IDs need not be dense.
    canonical_token_span: int | None = None
    canonical_region_token_strides: tuple[tuple[int, int], ...] = ()
    is_opaque: bool = False
    opaque_layout_signature: str | None = None

    def is_writer(self, block_id: int) -> bool:
        """Return whether this replica writes the canonical page ``block_id``."""
        if self.num_writers <= 0:
            raise ValueError("num_writers must be positive")
        if not 0 <= self.writer_index < self.num_writers:
            raise ValueError("writer_index must be in [0, num_writers)")
        return block_id % self.num_writers == self.writer_index

    def token_stride(self, canonical_region: int) -> int:
        """Return bytes per canonical token in a semantic region."""
        for region, stride in self.canonical_region_token_strides:
            if region == canonical_region:
                return stride
        raise ValueError(
            f"canonical region {canonical_region} has no token-stride metadata"
        )


@dataclass(frozen=True)
class PagePlacement:
    """One local page placed in a request-wide canonical byte space.

    ``first_token`` locates the page independently of its local token span.
    ``canonical_base`` remains a low-level byte-coordinate escape hatch for
    mappings without token geometry. ``local_base`` is an offset within the
    registration identified by
    ``(rank, local_page_id)``.  ``canonical_page_index`` is intentionally
    separate from both: it is the stable logical index used for replica writer
    election.
    """

    rank: int
    local_page_id: int
    canonical_page_index: int
    mapping: CanonicalPageMapping
    # Digest/key binding model semantics, dtype, quantization and canonical
    # region definitions.  Naked byte layouts are never composed across
    # incompatible canonical spaces.
    canonical_space_id: str
    first_token: int | None = None
    valid_token_offset: int = 0
    valid_token_count: int | None = None
    canonical_base: int = 0
    local_base: int = 0


@dataclass(frozen=True)
class TransferRun:
    """A connector-neutral strided direct-copy operation."""

    source_rank: int
    destination_rank: int
    source_page_id: int
    destination_page_id: int
    source_offset: int
    destination_offset: int
    fragment_size: int
    fragment_count: int
    source_stride: int
    destination_stride: int


@dataclass(frozen=True)
class _MappedFragment:
    rank: int
    page_id: int
    canonical_region: int
    canonical_start: int
    canonical_end: int
    local_start: int

    @property
    def local_end(self) -> int:
        return self.local_start + self.canonical_end - self.canonical_start


@dataclass(frozen=True)
class _CopyFragment:
    source_rank: int
    destination_rank: int
    source_page_id: int
    destination_page_id: int
    source_offset: int
    destination_offset: int
    size: int

    @property
    def endpoint_key(self) -> tuple[int, int, int, int]:
        return (
            self.source_rank,
            self.source_page_id,
            self.destination_rank,
            self.destination_page_id,
        )


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")


def _wire_fields(
    value: object,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    data = dict(value)
    keys = set(data)
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(keys - fields)
        raise ValueError(
            f"invalid {label} fields: missing={missing}, unknown={unknown}"
        )
    return data


def _require_exact_partition(
    intervals: list[tuple[int, int]],
    start: int,
    end: int,
    label: str,
) -> None:
    """Require intervals to cover ``[start, end)`` exactly once."""
    cursor = start
    for interval_start, interval_end in sorted(intervals):
        if interval_start < cursor:
            raise ValueError(f"{label} has duplicate or overlapping bytes")
        if interval_start > cursor:
            raise ValueError(f"{label} has a gap in [{cursor}, {interval_start})")
        cursor = interval_end
    if cursor != end:
        raise ValueError(f"{label} has a gap in [{cursor}, {end})")


def _require_disjoint(
    fragments: Sequence[_MappedFragment],
    label: str,
) -> None:
    """Reject duplicate canonical coverage while permitting gaps."""
    previous: _MappedFragment | None = None
    for fragment in sorted(
        fragments,
        key=lambda item: (
            item.canonical_region,
            item.canonical_start,
            item.canonical_end,
        ),
    ):
        if (
            previous is not None
            and fragment.canonical_region == previous.canonical_region
            and fragment.canonical_start < previous.canonical_end
        ):
            raise ValueError(f"{label} has duplicate or overlapping canonical bytes")
        previous = fragment


def _validate_and_expand(placement: PagePlacement) -> list[_MappedFragment]:
    """Validate one placement and expand its affine runs into intervals."""
    _require_nonnegative_int(placement.rank, "rank")
    _require_nonnegative_int(placement.local_page_id, "local_page_id")
    _require_nonnegative_int(placement.canonical_page_index, "canonical_page_index")
    _require_string(placement.canonical_space_id, "canonical_space_id")
    if placement.first_token is not None:
        _require_nonnegative_int(placement.first_token, "first_token")
        if placement.canonical_base != 0:
            raise ValueError("first_token and canonical_base cannot both be set")
    if placement.valid_token_count is not None:
        _require_nonnegative_int(placement.valid_token_count, "valid_token_count")
        if placement.first_token is None:
            raise ValueError("valid_token_count requires first_token")
    _require_nonnegative_int(placement.valid_token_offset, "valid_token_offset")
    if placement.valid_token_offset and placement.valid_token_count is None:
        raise ValueError("valid_token_offset requires valid_token_count")
    _require_nonnegative_int(placement.canonical_base, "canonical_base")
    _require_nonnegative_int(placement.local_base, "local_base")

    mapping = placement.mapping
    _require_positive_int(
        mapping.canonical_page_size_bytes, "canonical_page_size_bytes"
    )
    _require_positive_int(mapping.local_page_size_bytes, "local_page_size_bytes")
    _require_positive_int(mapping.num_writers, "num_writers")
    _require_nonnegative_int(mapping.writer_index, "writer_index")
    if mapping.writer_index >= mapping.num_writers:
        raise ValueError("writer_index must be in [0, num_writers)")
    _require_bool(mapping.parallelism_agnostic, "parallelism_agnostic")
    _require_bool(mapping.is_opaque, "is_opaque")
    if mapping.is_opaque:
        if mapping.opaque_layout_signature is None:
            raise ValueError("opaque mapping requires an opaque layout signature")
        _require_string(mapping.opaque_layout_signature, "opaque_layout_signature")
    elif mapping.opaque_layout_signature is not None:
        raise ValueError("certified mapping cannot have an opaque layout signature")
    if mapping.canonical_token_span is not None:
        _require_positive_int(mapping.canonical_token_span, "canonical_token_span")
        if (
            placement.valid_token_count is not None
            and placement.valid_token_offset + placement.valid_token_count
            > mapping.canonical_token_span
        ):
            raise ValueError(
                "valid token coverage cannot exceed the mapping canonical token span"
            )
    region_strides = dict(mapping.canonical_region_token_strides)
    if len(region_strides) != len(mapping.canonical_region_token_strides):
        raise ValueError("canonical region token strides must have unique region IDs")
    if len(region_strides) > KV_PLACEMENT_MAX_RUNS:
        raise ValueError("canonical page mapping has too many semantic regions")
    for region, stride in mapping.canonical_region_token_strides:
        _require_nonnegative_int(region, "canonical region token-stride ID")
        _require_positive_int(stride, "canonical region token stride")
    if not mapping.runs:
        raise ValueError("runs must not be empty")
    if any(not isinstance(run, CopyRun) for run in mapping.runs):
        raise ValueError("runs must contain CopyRun values")
    if len(mapping.runs) > KV_PLACEMENT_MAX_RUNS:
        raise ValueError("canonical page mapping has too many copy runs")

    for run in mapping.runs:
        _require_positive_int(run.num_fragments, "num_fragments")

    local_intervals: list[tuple[int, int]] = []
    fragments: list[_MappedFragment] = []
    for run_index, run in enumerate(mapping.runs):
        prefix = f"runs[{run_index}]"
        _require_nonnegative_int(run.local_offset, f"{prefix}.local_offset")
        _require_nonnegative_int(run.canonical_offset, f"{prefix}.canonical_offset")
        _require_positive_int(run.fragment_size, f"{prefix}.fragment_size")
        _require_positive_int(run.num_fragments, f"{prefix}.num_fragments")
        _require_nonnegative_int(run.local_stride, f"{prefix}.local_stride")
        _require_nonnegative_int(run.canonical_stride, f"{prefix}.canonical_stride")
        _require_nonnegative_int(run.canonical_region, f"{prefix}.canonical_region")
        storage_offset = (
            run.canonical_offset
            if run.canonical_storage_offset is None
            else run.canonical_storage_offset
        )
        storage_stride = (
            run.canonical_stride
            if run.canonical_storage_stride is None
            else run.canonical_storage_stride
        )
        _require_nonnegative_int(storage_offset, f"{prefix}.canonical_storage_offset")
        _require_nonnegative_int(storage_stride, f"{prefix}.canonical_storage_stride")

        for fragment_index in range(run.num_fragments):
            local_start = run.local_offset + fragment_index * run.local_stride
            local_end = local_start + run.fragment_size
            canonical_start = (
                run.canonical_offset + fragment_index * run.canonical_stride
            )
            canonical_end = canonical_start + run.fragment_size
            storage_start = storage_offset + fragment_index * storage_stride
            storage_end = storage_start + run.fragment_size
            if local_end > mapping.local_page_size_bytes:
                raise ValueError(f"{prefix} exceeds the local page")
            if storage_end > mapping.canonical_page_size_bytes:
                raise ValueError(f"{prefix} exceeds canonical page storage")
            local_intervals.append((local_start, local_end))
            fragments.append(
                _MappedFragment(
                    rank=placement.rank,
                    page_id=placement.local_page_id,
                    canonical_region=run.canonical_region,
                    canonical_start=(
                        placement.first_token
                        * mapping.token_stride(run.canonical_region)
                        if placement.first_token is not None
                        else placement.canonical_base
                    )
                    + canonical_start,
                    canonical_end=(
                        placement.first_token
                        * mapping.token_stride(run.canonical_region)
                        if placement.first_token is not None
                        else placement.canonical_base
                    )
                    + canonical_end,
                    local_start=placement.local_base + local_start,
                )
            )

    _require_exact_partition(
        local_intervals,
        0,
        mapping.local_page_size_bytes,
        f"placement ({placement.rank}, {placement.local_page_id}) local page",
    )
    _require_disjoint(
        fragments,
        f"placement ({placement.rank}, {placement.local_page_id})",
    )
    if placement.valid_token_count is not None:
        clipped: list[_MappedFragment] = []
        assert placement.first_token is not None
        for fragment in fragments:
            stride = mapping.token_stride(fragment.canonical_region)
            valid_start = (
                placement.first_token + placement.valid_token_offset
            ) * stride
            valid_end = (
                placement.first_token
                + placement.valid_token_offset
                + placement.valid_token_count
            ) * stride
            start = max(fragment.canonical_start, valid_start)
            end = min(fragment.canonical_end, valid_end)
            if start < end:
                clipped.append(
                    _MappedFragment(
                        rank=fragment.rank,
                        page_id=fragment.page_id,
                        canonical_region=fragment.canonical_region,
                        canonical_start=start,
                        canonical_end=end,
                        local_start=fragment.local_start
                        + start
                        - fragment.canonical_start,
                    )
                )
        fragments = clipped
    return fragments


def _validate_placement_compact(placement: PagePlacement) -> None:
    """Validate one placement with memory proportional to compact run count.

    Unlike :func:`_validate_and_expand`, this never retains one object per
    affine fragment. It is used by the streaming composer so a large but
    compact interleaved mapping can be checked before any direct batch is
    emitted without first materializing the whole page.
    """
    _require_nonnegative_int(placement.rank, "rank")
    _require_nonnegative_int(placement.local_page_id, "local_page_id")
    _require_nonnegative_int(placement.canonical_page_index, "canonical_page_index")
    _require_string(placement.canonical_space_id, "canonical_space_id")
    if placement.first_token is not None:
        _require_nonnegative_int(placement.first_token, "first_token")
        if placement.canonical_base != 0:
            raise ValueError("first_token and canonical_base cannot both be set")
    if placement.valid_token_count is not None:
        _require_nonnegative_int(placement.valid_token_count, "valid_token_count")
        if placement.first_token is None:
            raise ValueError("valid_token_count requires first_token")
    _require_nonnegative_int(placement.valid_token_offset, "valid_token_offset")
    if placement.valid_token_offset and placement.valid_token_count is None:
        raise ValueError("valid_token_offset requires valid_token_count")
    _require_nonnegative_int(placement.canonical_base, "canonical_base")
    _require_nonnegative_int(placement.local_base, "local_base")

    mapping = placement.mapping
    _require_positive_int(
        mapping.canonical_page_size_bytes, "canonical_page_size_bytes"
    )
    _require_positive_int(mapping.local_page_size_bytes, "local_page_size_bytes")
    _require_positive_int(mapping.num_writers, "num_writers")
    _require_nonnegative_int(mapping.writer_index, "writer_index")
    if mapping.writer_index >= mapping.num_writers:
        raise ValueError("writer_index must be in [0, num_writers)")
    _require_bool(mapping.parallelism_agnostic, "parallelism_agnostic")
    _require_bool(mapping.is_opaque, "is_opaque")
    if mapping.is_opaque:
        if mapping.opaque_layout_signature is None:
            raise ValueError("opaque mapping requires an opaque layout signature")
        _require_string(mapping.opaque_layout_signature, "opaque_layout_signature")
    elif mapping.opaque_layout_signature is not None:
        raise ValueError("certified mapping cannot have an opaque layout signature")
    if mapping.canonical_token_span is not None:
        _require_positive_int(mapping.canonical_token_span, "canonical_token_span")
        if (
            placement.valid_token_count is not None
            and placement.valid_token_offset + placement.valid_token_count
            > mapping.canonical_token_span
        ):
            raise ValueError(
                "valid token coverage cannot exceed the mapping canonical token span"
            )
    region_strides = dict(mapping.canonical_region_token_strides)
    if len(region_strides) != len(mapping.canonical_region_token_strides):
        raise ValueError("canonical region token strides must have unique region IDs")
    if len(region_strides) > KV_PLACEMENT_MAX_RUNS:
        raise ValueError("canonical page mapping has too many semantic regions")
    for region, stride in mapping.canonical_region_token_strides:
        _require_nonnegative_int(region, "canonical region token-stride ID")
        _require_positive_int(stride, "canonical region token stride")
    if not mapping.runs:
        raise ValueError("runs must not be empty")
    if any(not isinstance(run, CopyRun) for run in mapping.runs):
        raise ValueError("runs must contain CopyRun values")
    if len(mapping.runs) > KV_PLACEMENT_MAX_RUNS:
        raise ValueError("canonical page mapping has too many copy runs")

    for run_index, run in enumerate(mapping.runs):
        prefix = f"runs[{run_index}]"
        _require_nonnegative_int(run.local_offset, f"{prefix}.local_offset")
        _require_nonnegative_int(run.canonical_offset, f"{prefix}.canonical_offset")
        _require_positive_int(run.fragment_size, f"{prefix}.fragment_size")
        _require_positive_int(run.num_fragments, f"{prefix}.num_fragments")
        _require_nonnegative_int(run.local_stride, f"{prefix}.local_stride")
        _require_nonnegative_int(run.canonical_stride, f"{prefix}.canonical_stride")
        _require_nonnegative_int(run.canonical_region, f"{prefix}.canonical_region")
        storage_offset = (
            run.canonical_offset
            if run.canonical_storage_offset is None
            else run.canonical_storage_offset
        )
        storage_stride = (
            run.canonical_stride
            if run.canonical_storage_stride is None
            else run.canonical_storage_stride
        )
        _require_nonnegative_int(storage_offset, f"{prefix}.canonical_storage_offset")
        _require_nonnegative_int(storage_stride, f"{prefix}.canonical_storage_stride")
        local_end = (
            run.local_offset
            + (run.num_fragments - 1) * run.local_stride
            + run.fragment_size
        )
        storage_end = (
            storage_offset
            + (run.num_fragments - 1) * storage_stride
            + run.fragment_size
        )
        if local_end > mapping.local_page_size_bytes:
            raise ValueError(f"{prefix} exceeds the local page")
        if storage_end > mapping.canonical_page_size_bytes:
            raise ValueError(f"{prefix} exceeds canonical page storage")
        if placement.first_token is not None:
            mapping.token_stride(run.canonical_region)

    # Every certified backend mapping stores its local page densely within each
    # affine run. Collapse those runs to intervals and prove the exact local
    # partition in O(run_count log run_count), independent of fragment count.
    # A sparse local affine run could be valid if other runs fill its holes,
    # but accepting that general union would require expanding attacker-chosen
    # fragment counts during handshake. Fail closed on that uncertified shape.
    local_intervals: list[tuple[int, int]] = []
    for run in mapping.runs:
        if run.num_fragments > 1 and run.local_stride != run.fragment_size:
            raise ValueError(
                f"placement ({placement.rank}, {placement.local_page_id}) has a "
                "non-dense local affine run that cannot be certified compactly"
            )
        local_intervals.append(
            (
                run.local_offset,
                run.local_offset
                + (run.num_fragments - 1) * run.local_stride
                + run.fragment_size,
            )
        )
    _require_exact_partition(
        local_intervals,
        0,
        mapping.local_page_size_bytes,
        f"placement ({placement.rank}, {placement.local_page_id}) local page",
    )

    # Canonical mappings may be sparse, notably one HND lane per local head.
    # First split each semantic region into overlapping affine envelopes. A
    # one-run or disjoint-envelope component is trivially safe. Generated HND
    # components share one stride and occupy disjoint, non-wrapping residue
    # intervals inside each stride cell; certify that periodic structure in
    # O(run_count log run_count) rather than expanding every token fragment.
    runs_by_region: dict[int, list[tuple[int, int, CopyRun]]] = defaultdict(list)
    for run in mapping.runs:
        if run.num_fragments > 1 and run.canonical_stride < run.fragment_size:
            raise ValueError(
                f"placement ({placement.rank}, {placement.local_page_id}) has "
                "duplicate or overlapping canonical bytes"
            )
        canonical_end = (
            run.canonical_offset
            + (run.num_fragments - 1) * run.canonical_stride
            + run.fragment_size
        )
        runs_by_region[run.canonical_region].append(
            (run.canonical_offset, canonical_end, run)
        )

    def certify_component(component: list[CopyRun]) -> None:
        if len(component) <= 1:
            return
        if any(run.num_fragments == 1 for run in component):
            raise ValueError(
                f"placement ({placement.rank}, {placement.local_page_id}) has "
                "canonical affine envelopes that cannot be certified compactly"
            )
        strides = {run.canonical_stride for run in component}
        if len(strides) != 1:
            raise ValueError(
                f"placement ({placement.rank}, {placement.local_page_id}) has "
                "canonical affine envelopes that cannot be certified compactly"
            )
        stride = strides.pop()
        residue_intervals: list[tuple[int, int]] = []
        for run in component:
            residue_start = run.canonical_offset % stride
            residue_end = residue_start + run.fragment_size
            if residue_end > stride:
                raise ValueError(
                    f"placement ({placement.rank}, {placement.local_page_id}) has "
                    "a canonical affine fragment crossing its stride cell"
                )
            residue_intervals.append((residue_start, residue_end))
        previous_end: int | None = None
        for start, end in sorted(residue_intervals):
            if previous_end is not None and start < previous_end:
                raise ValueError(
                    f"placement ({placement.rank}, {placement.local_page_id}) has "
                    "duplicate or overlapping canonical bytes"
                )
            previous_end = end

    for region_runs in runs_by_region.values():
        component: list[CopyRun] = []
        component_end = 0
        for start, end, run in sorted(region_runs, key=lambda item: item[:2]):
            if component and start >= component_end:
                certify_component(component)
                component = []
            component.append(run)
            component_end = max(component_end, end) if len(component) > 1 else end
        certify_component(component)


def _mapped_fragment(
    placement: PagePlacement,
    run: CopyRun,
    fragment_index: int,
) -> _MappedFragment | None:
    """Build and clip one affine fragment without retaining its siblings."""
    mapping = placement.mapping
    local_start = run.local_offset + fragment_index * run.local_stride
    canonical_start = run.canonical_offset + fragment_index * run.canonical_stride
    canonical_end = canonical_start + run.fragment_size
    canonical_base = (
        placement.first_token * mapping.token_stride(run.canonical_region)
        if placement.first_token is not None
        else placement.canonical_base
    )
    fragment = _MappedFragment(
        rank=placement.rank,
        page_id=placement.local_page_id,
        canonical_region=run.canonical_region,
        canonical_start=canonical_base + canonical_start,
        canonical_end=canonical_base + canonical_end,
        local_start=placement.local_base + local_start,
    )
    if placement.valid_token_count is None:
        return fragment
    assert placement.first_token is not None
    stride = mapping.token_stride(fragment.canonical_region)
    valid_start = (placement.first_token + placement.valid_token_offset) * stride
    valid_end = (
        placement.first_token
        + placement.valid_token_offset
        + placement.valid_token_count
    ) * stride
    start = max(fragment.canonical_start, valid_start)
    end = min(fragment.canonical_end, valid_end)
    if start >= end:
        return None
    return _MappedFragment(
        rank=fragment.rank,
        page_id=fragment.page_id,
        canonical_region=fragment.canonical_region,
        canonical_start=start,
        canonical_end=end,
        local_start=fragment.local_start + start - fragment.canonical_start,
    )


def _iter_placement_fragments(
    placement: PagePlacement,
    *,
    order: str,
) -> Iterator[_MappedFragment]:
    """Expand one placement lazily in canonical or local byte order."""
    if order not in ("canonical", "local"):
        raise ValueError("fragment order must be 'canonical' or 'local'")

    def run_fragments(run: CopyRun) -> Iterator[_MappedFragment]:
        first_fragment = 0
        final_fragment = run.num_fragments
        if (
            placement.valid_token_count is not None
            and run.num_fragments > 1
            and run.canonical_stride > 0
        ):
            stride = placement.mapping.token_stride(run.canonical_region)
            valid_start = placement.valid_token_offset * stride
            valid_end = (
                placement.valid_token_offset + placement.valid_token_count
            ) * stride
            first_fragment = max(
                0,
                (valid_start - run.canonical_offset - run.fragment_size)
                // run.canonical_stride
                + 1,
            )
            final_fragment = min(
                run.num_fragments,
                max(
                    0,
                    (valid_end - run.canonical_offset + run.canonical_stride - 1)
                    // run.canonical_stride,
                ),
            )
        for fragment_index in range(first_fragment, final_fragment):
            fragment = _mapped_fragment(placement, run, fragment_index)
            if fragment is not None:
                yield fragment

    iterators = [iter(run_fragments(run)) for run in placement.mapping.runs]
    heap: list[tuple[tuple[int, ...], int, _MappedFragment]] = []

    def key(fragment: _MappedFragment) -> tuple[int, ...]:
        if order == "canonical":
            return (
                fragment.canonical_region,
                fragment.canonical_start,
                fragment.canonical_end,
                fragment.local_start,
            )
        return (
            fragment.local_start,
            fragment.local_end,
            fragment.canonical_region,
            fragment.canonical_start,
        )

    for run_index, iterator in enumerate(iterators):
        try:
            fragment = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (key(fragment), run_index, fragment))
    while heap:
        _, run_index, fragment = heapq.heappop(heap)
        yield fragment
        try:
            next_fragment = next(iterators[run_index])
        except StopIteration:
            continue
        heapq.heappush(heap, (key(next_fragment), run_index, next_fragment))


def _iter_merged_placement_fragments(
    placements: Sequence[PagePlacement],
    *,
    order: str,
) -> Iterator[_MappedFragment]:
    """Merge several compact placement streams with bounded lookahead."""
    iterators = [
        iter(_iter_placement_fragments(placement, order=order))
        for placement in placements
    ]
    heap: list[tuple[tuple[int, ...], int, _MappedFragment]] = []

    def key(fragment: _MappedFragment) -> tuple[int, ...]:
        if order == "canonical":
            return (
                fragment.canonical_region,
                fragment.canonical_start,
                fragment.canonical_end,
                fragment.rank,
                fragment.page_id,
                fragment.local_start,
            )
        return (
            fragment.local_start,
            fragment.local_end,
            fragment.rank,
            fragment.page_id,
            fragment.canonical_region,
            fragment.canonical_start,
        )

    for placement_index, iterator in enumerate(iterators):
        try:
            fragment = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (key(fragment), placement_index, fragment))
    while heap:
        _, placement_index, fragment = heapq.heappop(heap)
        yield fragment
        try:
            next_fragment = next(iterators[placement_index])
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (key(next_fragment), placement_index, next_fragment),
        )


def _placement_sort_key(placement: PagePlacement) -> tuple[int, ...]:
    """Return a deterministic key containing only placement coordinates."""
    return (
        placement.rank,
        placement.local_page_id,
        placement.local_base,
        -1 if placement.first_token is None else placement.first_token,
        placement.canonical_base,
        placement.canonical_page_index,
        placement.valid_token_offset,
        -1 if placement.valid_token_count is None else placement.valid_token_count,
    )


def _placements_by_physical_page(
    placements: Sequence[PagePlacement],
) -> Iterator[tuple[tuple[int, int], tuple[PagePlacement, ...]]]:
    """Group compact placements without retaining expanded fragments."""
    by_page: dict[tuple[int, int], list[PagePlacement]] = defaultdict(list)
    for placement in placements:
        by_page[(placement.rank, placement.local_page_id)].append(placement)
    for key in sorted(by_page):
        yield key, tuple(sorted(by_page[key], key=_placement_sort_key))


def _require_fragment_stream_disjoint(
    fragments: Iterable[_MappedFragment],
    *,
    order: str,
    label: str,
) -> None:
    """Validate a sorted fragment stream with constant interval state."""
    previous_region: int | None = None
    previous_end: int | None = None
    for fragment in fragments:
        if order == "local":
            start = fragment.local_start
            end = fragment.local_end
        elif order == "canonical":
            if fragment.canonical_region != previous_region:
                previous_region = fragment.canonical_region
                previous_end = None
            start = fragment.canonical_start
            end = fragment.canonical_end
        else:
            raise AssertionError("unknown fragment order")
        if previous_end is not None and start < previous_end:
            coordinate = "local" if order == "local" else "canonical"
            raise ValueError(f"{label} has overlapping {coordinate} bytes")
        previous_end = end


def _validate_streaming_pages(
    sources: Sequence[PagePlacement],
    destinations: Sequence[PagePlacement],
) -> tuple[PagePlacement, ...]:
    """Validate page aliases and elected source ownership without expansion.

    The returned tuple contains elected sources in deterministic compact order.
    Every validation pass uses a k-way iterator whose memory is proportional to
    compact page/run metadata, never to the expanded fragment count.
    """
    for placement in (*sources, *destinations):
        _validate_placement_compact(placement)

    for key, page_placements in _placements_by_physical_page(sources):
        _require_fragment_stream_disjoint(
            _iter_merged_placement_fragments(page_placements, order="local"),
            order="local",
            label=f"source page {key}",
        )

    for key, page_placements in _placements_by_physical_page(destinations):
        _require_fragment_stream_disjoint(
            _iter_merged_placement_fragments(page_placements, order="local"),
            order="local",
            label=f"destination page {key}",
        )
        _require_fragment_stream_disjoint(
            _iter_merged_placement_fragments(page_placements, order="canonical"),
            order="canonical",
            label=f"destination page {key}",
        )

    elected_sources = tuple(
        sorted(
            (
                placement
                for placement in sources
                if placement.mapping.is_writer(placement.canonical_page_index)
            ),
            key=_placement_sort_key,
        )
    )
    _require_fragment_stream_disjoint(
        _iter_merged_placement_fragments(elected_sources, order="canonical"),
        order="canonical",
        label="elected source placements",
    )
    return elected_sources


def _iter_intersections(
    sources: Sequence[PagePlacement],
    destinations: Iterable[_MappedFragment],
) -> Iterator[_CopyFragment]:
    """Resolve a canonical-sorted destination stream against source bytes."""
    source_iterator = iter(_iter_merged_placement_fragments(sources, order="canonical"))
    source = next(source_iterator, None)
    for destination in destinations:
        cursor = destination.canonical_start
        while cursor < destination.canonical_end:
            while source is not None and (
                source.canonical_region < destination.canonical_region
                or (
                    source.canonical_region == destination.canonical_region
                    and source.canonical_end <= cursor
                )
            ):
                source = next(source_iterator, None)
            if (
                source is None
                or source.canonical_region > destination.canonical_region
                or source.canonical_start > cursor
            ):
                gap_end = destination.canonical_end
                if (
                    source is not None
                    and source.canonical_region == destination.canonical_region
                ):
                    gap_end = min(gap_end, source.canonical_start)
                raise ValueError(
                    "source placements have a canonical gap in region "
                    f"{destination.canonical_region} [{cursor}, {gap_end}) "
                    "required by destination "
                    f"({destination.rank}, {destination.page_id})"
                )

            copy_end = min(source.canonical_end, destination.canonical_end)
            yield _CopyFragment(
                source_rank=source.rank,
                destination_rank=destination.rank,
                source_page_id=source.page_id,
                destination_page_id=destination.page_id,
                source_offset=source.local_start + cursor - source.canonical_start,
                destination_offset=(
                    destination.local_start + cursor - destination.canonical_start
                ),
                size=copy_end - cursor,
            )
            cursor = copy_end


def _iter_colored_destination_fragments(
    destinations: Sequence[PagePlacement],
) -> Iterator[tuple[int, _MappedFragment]]:
    """Assign canonical fragments to deterministic disjoint lanes.

    Active intervals and reusable lane IDs are heaps, making lane selection
    logarithmic in canonical overlap depth while retaining only one entry per
    overlapping replica.
    """
    current_region: int | None = None
    active_lanes: list[tuple[int, int]] = []
    available_lanes: list[int] = []
    next_lane = 0
    fragments = _iter_merged_placement_fragments(
        tuple(sorted(destinations, key=_placement_sort_key)),
        order="canonical",
    )
    for fragment in fragments:
        if fragment.canonical_region != current_region:
            current_region = fragment.canonical_region
            active_lanes.clear()
            available_lanes.clear()
            next_lane = 0
        while active_lanes and active_lanes[0][0] <= fragment.canonical_start:
            _, lane_index = heapq.heappop(active_lanes)
            heapq.heappush(available_lanes, lane_index)
        if available_lanes:
            assigned_lane = heapq.heappop(available_lanes)
        else:
            assigned_lane = next_lane
            next_lane += 1
        heapq.heappush(active_lanes, (fragment.canonical_end, assigned_lane))
        yield assigned_lane, fragment


def _iter_destination_lane(
    destinations: Sequence[PagePlacement],
    lane_index: int,
) -> Iterator[_MappedFragment]:
    """Replay one disjoint lane of a possibly replicated destination set.

    Replaying the compact iterator for each lane avoids retaining the expanded
    destination, while non-replicated DCP/TP partitions use one lane even when
    rank-level placement bounds overlap.
    """
    for assigned_lane, fragment in _iter_colored_destination_fragments(destinations):
        if assigned_lane == lane_index:
            yield fragment


def _destination_lane_count(destinations: Sequence[PagePlacement]) -> int:
    """Return canonical interval overlap depth with bounded state."""
    max_lanes = 0
    for lane_index, _ in _iter_colored_destination_fragments(destinations):
        max_lanes = max(max_lanes, lane_index + 1)
    return max_lanes


def _validate_streaming_coverage(
    elected_sources: Sequence[PagePlacement],
    destinations: Sequence[PagePlacement],
) -> int:
    """Fail before emission if a later destination would expose a source gap."""
    lane_count = _destination_lane_count(destinations)
    for lane_index in range(lane_count):
        for _ in _iter_intersections(
            elected_sources,
            _iter_destination_lane(destinations, lane_index),
        ):
            pass
    return lane_count


def _validate_destination_pages(
    placements: Sequence[PagePlacement],
    expanded: Sequence[list[_MappedFragment]],
) -> None:
    """Validate each consumer page without forbidding cross-rank replicas."""
    by_page: dict[tuple[int, int], list[_MappedFragment]] = defaultdict(list)
    for placement, fragments in zip(placements, expanded):
        key = (placement.rank, placement.local_page_id)
        by_page[key].extend(fragments)

    for key, fragments in by_page.items():
        previous_end: int | None = None
        for start, end in sorted(
            (fragment.local_start, fragment.local_end) for fragment in fragments
        ):
            if previous_end is not None and start < previous_end:
                raise ValueError(f"destination page {key} has overlapping local bytes")
            previous_end = end
        _require_disjoint(fragments, f"destination page {key}")


def _validate_source_pages(
    placements: Sequence[PagePlacement],
    expanded: Sequence[list[_MappedFragment]],
) -> None:
    """Reject one physical source byte being advertised more than once.

    Separate placements may describe disjoint slices of one registration, but
    aliasing the same local bytes to multiple canonical positions would make a
    plan look complete while reading conflicting logical pages.
    """
    by_page: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for placement, fragments in zip(placements, expanded):
        key = (placement.rank, placement.local_page_id)
        by_page[key].extend(
            (fragment.local_start, fragment.local_end) for fragment in fragments
        )

    for key, intervals in by_page.items():
        previous_end: int | None = None
        for start, end in sorted(intervals):
            if previous_end is not None and start < previous_end:
                raise ValueError(f"source page {key} has overlapping local bytes")
            previous_end = end


def _intersect_destinations(
    sources: list[_MappedFragment],
    destinations: Sequence[_MappedFragment],
) -> list[_CopyFragment]:
    """Resolve every destination byte to exactly one source byte."""
    if not sources and destinations:
        raise ValueError("source placements leave all destination bytes uncovered")

    sources.sort(
        key=lambda item: (
            item.canonical_region,
            item.canonical_start,
            item.canonical_end,
        )
    )
    _require_disjoint(sources, "elected source placements")
    sources_by_region: dict[int, list[_MappedFragment]] = defaultdict(list)
    for source in sources:
        sources_by_region[source.canonical_region].append(source)
    source_starts_by_region = {
        region: [fragment.canonical_start for fragment in fragments]
        for region, fragments in sources_by_region.items()
    }
    copies: list[_CopyFragment] = []

    for destination in destinations:
        region_sources = sources_by_region.get(destination.canonical_region, [])
        source_starts = source_starts_by_region.get(destination.canonical_region, [])
        cursor = destination.canonical_start
        source_index = max(0, bisect_right(source_starts, cursor) - 1)
        while cursor < destination.canonical_end:
            while (
                source_index < len(region_sources)
                and region_sources[source_index].canonical_end <= cursor
            ):
                source_index += 1
            if (
                source_index == len(region_sources)
                or region_sources[source_index].canonical_start > cursor
            ):
                gap_end = destination.canonical_end
                if source_index < len(region_sources):
                    gap_end = min(gap_end, region_sources[source_index].canonical_start)
                raise ValueError(
                    "source placements have a canonical gap in region "
                    f"{destination.canonical_region} "
                    f"[{cursor}, {gap_end}) required by destination "
                    f"({destination.rank}, {destination.page_id})"
                )

            source = region_sources[source_index]
            copy_end = min(source.canonical_end, destination.canonical_end)
            copies.append(
                _CopyFragment(
                    source_rank=source.rank,
                    destination_rank=destination.rank,
                    source_page_id=source.page_id,
                    destination_page_id=destination.page_id,
                    source_offset=source.local_start + cursor - source.canonical_start,
                    destination_offset=destination.local_start
                    + cursor
                    - destination.canonical_start,
                    size=copy_end - cursor,
                )
            )
            cursor = copy_end

    return copies


def _merge_contiguous(
    fragments: list[_CopyFragment],
) -> list[_CopyFragment]:
    merged: list[_CopyFragment] = []
    for fragment in sorted(
        fragments, key=lambda item: (item.destination_offset, item.source_offset)
    ):
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.source_offset + previous.size == fragment.source_offset
            and previous.destination_offset + previous.size
            == fragment.destination_offset
        ):
            merged[-1] = _CopyFragment(
                source_rank=previous.source_rank,
                destination_rank=previous.destination_rank,
                source_page_id=previous.source_page_id,
                destination_page_id=previous.destination_page_id,
                source_offset=previous.source_offset,
                destination_offset=previous.destination_offset,
                size=previous.size + fragment.size,
            )
        else:
            merged.append(fragment)
    return merged


def _compress_copies(copies: Sequence[_CopyFragment]) -> tuple[TransferRun, ...]:
    """Coalesce adjacent bytes, then compress equal affine fragments."""
    by_endpoints: dict[tuple[int, int, int, int], list[_CopyFragment]] = defaultdict(
        list
    )
    for copy in copies:
        by_endpoints[copy.endpoint_key].append(copy)

    runs: list[TransferRun] = []
    for fragments in by_endpoints.values():
        merged = _merge_contiguous(fragments)
        index = 0
        while index < len(merged):
            first = merged[index]
            count = 1
            source_stride = first.size
            destination_stride = first.size
            if index + 1 < len(merged) and merged[index + 1].size == first.size:
                second = merged[index + 1]
                candidate_source_stride = second.source_offset - first.source_offset
                candidate_destination_stride = (
                    second.destination_offset - first.destination_offset
                )
                if (
                    candidate_source_stride >= first.size
                    and candidate_destination_stride >= first.size
                ):
                    source_stride = candidate_source_stride
                    destination_stride = candidate_destination_stride
                    count = 2
                    while index + count < len(merged):
                        candidate = merged[index + count]
                        if (
                            candidate.size != first.size
                            or candidate.source_offset
                            != first.source_offset + count * source_stride
                            or candidate.destination_offset
                            != first.destination_offset + count * destination_stride
                        ):
                            break
                        count += 1

            runs.append(
                TransferRun(
                    source_rank=first.source_rank,
                    destination_rank=first.destination_rank,
                    source_page_id=first.source_page_id,
                    destination_page_id=first.destination_page_id,
                    source_offset=first.source_offset,
                    destination_offset=first.destination_offset,
                    fragment_size=first.size,
                    fragment_count=count,
                    source_stride=source_stride,
                    destination_stride=destination_stride,
                )
            )
            index += count

    return tuple(
        sorted(
            runs,
            key=lambda run: (
                run.destination_rank,
                run.destination_page_id,
                run.destination_offset,
                run.source_rank,
                run.source_page_id,
                run.source_offset,
            ),
        )
    )


def compose_page_placements(
    sources: Sequence[PagePlacement],
    destinations: Sequence[PagePlacement],
) -> tuple[TransferRun, ...]:
    """Compose producer and consumer mappings through canonical byte space.

    Replica writer election is applied to sources only.  Elected sources must
    provide every byte requested by every destination exactly once.  Multiple
    destination pages may request the same canonical bytes, which is necessary
    for replicated consumer layouts.
    """
    canonical_spaces = {
        placement.canonical_space_id for placement in (*sources, *destinations)
    }
    if len(canonical_spaces) > 1:
        raise ValueError(
            "source and destination placements use incompatible canonical spaces: "
            f"{sorted(canonical_spaces)}"
        )
    mappings = [placement.mapping for placement in (*sources, *destinations)]
    if any(mapping.is_opaque for mapping in mappings):
        opaque_signatures = {
            mapping.opaque_layout_signature for mapping in mappings if mapping.is_opaque
        }
        if (
            not all(mapping.is_opaque for mapping in mappings)
            or len(opaque_signatures) != 1
        ):
            raise ValueError(
                "opaque placements require identical opaque layout signatures"
            )
    source_expanded = [_validate_and_expand(placement) for placement in sources]
    destination_expanded = [
        _validate_and_expand(placement) for placement in destinations
    ]
    _validate_source_pages(sources, source_expanded)
    _validate_destination_pages(destinations, destination_expanded)

    elected_sources = [
        fragment
        for placement, fragments in zip(sources, source_expanded)
        if placement.mapping.is_writer(placement.canonical_page_index)
        for fragment in fragments
    ]
    destination_fragments = [
        fragment for fragments in destination_expanded for fragment in fragments
    ]
    copies = _intersect_destinations(elected_sources, destination_fragments)
    return _compress_copies(copies)


def iter_page_placement_transfer_runs(
    sources: Sequence[PagePlacement],
    destinations: Sequence[PagePlacement],
    *,
    max_buffered_copy_fragments: int = 4096,
) -> Iterator[TransferRun]:
    """Compose placements while retaining only a bounded fragment window.

    This is the runtime counterpart of :func:`compose_page_placements`. It
    preserves the eager API's validation and direct-copy semantics, but never
    constructs request-wide expanded-fragment, copy-fragment, or transfer-run
    collections. ``max_buffered_copy_fragments`` controls compression
    lookahead only: reaching the limit emits another set of direct runs and
    never selects packing, staging, or rejection.

    Static page mappings are fully validated, and all destination coverage is
    checked, before the first run is emitted. Those checks replay compact
    affine iterators and therefore bound memory even for a highly fragmented
    single page.
    """
    _require_positive_int(
        max_buffered_copy_fragments,
        "max_buffered_copy_fragments",
    )
    sources = tuple(sources)
    destinations = tuple(destinations)
    if any(not isinstance(item, PagePlacement) for item in sources):
        raise ValueError("sources must contain PagePlacement values")
    if any(not isinstance(item, PagePlacement) for item in destinations):
        raise ValueError("destinations must contain PagePlacement values")

    canonical_spaces = {
        placement.canonical_space_id for placement in (*sources, *destinations)
    }
    if len(canonical_spaces) > 1:
        raise ValueError(
            "source and destination placements use incompatible canonical spaces: "
            f"{sorted(canonical_spaces)}"
        )
    mappings = [placement.mapping for placement in (*sources, *destinations)]
    if any(mapping.is_opaque for mapping in mappings):
        opaque_signatures = {
            mapping.opaque_layout_signature for mapping in mappings if mapping.is_opaque
        }
        if (
            not all(mapping.is_opaque for mapping in mappings)
            or len(opaque_signatures) != 1
        ):
            raise ValueError(
                "opaque placements require identical opaque layout signatures"
            )

    elected_sources = _validate_streaming_pages(sources, destinations)
    destination_lane_count = _validate_streaming_coverage(elected_sources, destinations)

    buffered_copies: list[_CopyFragment] = []
    for lane_index in range(destination_lane_count):
        copies = _iter_intersections(
            elected_sources,
            _iter_destination_lane(destinations, lane_index),
        )
        for copy in copies:
            buffered_copies.append(copy)
            if len(buffered_copies) < max_buffered_copy_fragments:
                continue
            runs = _compress_copies(buffered_copies)
            buffered_copies.clear()
            yield from runs
    if buffered_copies:
        yield from _compress_copies(buffered_copies)


def _copy_run_to_dict(run: CopyRun) -> dict[str, Any]:
    return {
        "local_offset": run.local_offset,
        "canonical_offset": run.canonical_offset,
        "fragment_size": run.fragment_size,
        "num_fragments": run.num_fragments,
        "local_stride": run.local_stride,
        "canonical_stride": run.canonical_stride,
        "canonical_region": run.canonical_region,
        "canonical_storage_offset": run.canonical_storage_offset,
        "canonical_storage_stride": run.canonical_storage_stride,
    }


def _copy_run_from_dict(value: object) -> CopyRun:
    data = _wire_fields(
        value,
        {
            "local_offset",
            "canonical_offset",
            "fragment_size",
            "num_fragments",
            "local_stride",
            "canonical_stride",
            "canonical_region",
            "canonical_storage_offset",
            "canonical_storage_stride",
        },
        "copy run",
    )
    return CopyRun(**data)


_COPY_RUN_WIRE_FIELDS = {
    "local_offset",
    "canonical_offset",
    "fragment_size",
    "num_fragments",
    "local_stride",
    "canonical_stride",
    "canonical_region",
    "canonical_storage_offset",
    "canonical_storage_stride",
}

_CANONICAL_PAGE_MAPPING_WIRE_FIELDS = {
    "canonical_page_size_bytes",
    "local_page_size_bytes",
    "runs",
    "num_writers",
    "writer_index",
    "parallelism_agnostic",
    "canonical_token_span",
    "canonical_region_token_strides",
    "is_opaque",
    "opaque_layout_signature",
}

_LAYER_PAGE_MAPPING_WIRE_FIELDS = {
    "layer_name",
    "layer_index",
    "semantic_group_id",
    "mapping",
}


def _prescan_rank_mapping_budget(mappings: list[object]) -> None:
    """Bound aggregate nested mapping work before constructing mappings."""
    if len(mappings) > KV_PLACEMENT_MAX_MAPPINGS:
        raise ValueError("rank placement has too many layer mappings")

    total_runs = 0
    for raw_layer_mapping in mappings:
        layer_mapping = _wire_fields(
            raw_layer_mapping,
            _LAYER_PAGE_MAPPING_WIRE_FIELDS,
            "layer page mapping",
        )
        mapping = _wire_fields(
            layer_mapping["mapping"],
            _CANONICAL_PAGE_MAPPING_WIRE_FIELDS,
            "canonical page mapping",
        )
        runs = mapping["runs"]
        if not isinstance(runs, list):
            raise ValueError("canonical page mapping runs must be an array")

        total_runs += len(runs)
        if total_runs > KV_PLACEMENT_MAX_RUNS:
            raise ValueError("rank placement has too many copy runs")
        for raw_run in runs:
            run = _wire_fields(raw_run, _COPY_RUN_WIRE_FIELDS, "copy run")
            num_fragments = run["num_fragments"]
            _require_positive_int(num_fragments, "num_fragments")


def _mapping_to_dict(mapping: CanonicalPageMapping) -> dict[str, Any]:
    return {
        "canonical_page_size_bytes": mapping.canonical_page_size_bytes,
        "local_page_size_bytes": mapping.local_page_size_bytes,
        "runs": [_copy_run_to_dict(run) for run in mapping.runs],
        "num_writers": mapping.num_writers,
        "writer_index": mapping.writer_index,
        "parallelism_agnostic": mapping.parallelism_agnostic,
        "canonical_token_span": mapping.canonical_token_span,
        "canonical_region_token_strides": [
            list(item) for item in mapping.canonical_region_token_strides
        ],
        "is_opaque": mapping.is_opaque,
        "opaque_layout_signature": mapping.opaque_layout_signature,
    }


def _mapping_from_dict(value: object) -> CanonicalPageMapping:
    data = _wire_fields(
        value,
        _CANONICAL_PAGE_MAPPING_WIRE_FIELDS,
        "canonical page mapping",
    )
    runs = data["runs"]
    if not isinstance(runs, list):
        raise ValueError("canonical page mapping runs must be an array")
    if len(runs) > KV_PLACEMENT_MAX_RUNS:
        raise ValueError("canonical page mapping has too many copy runs")
    region_strides = data["canonical_region_token_strides"]
    if not isinstance(region_strides, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in region_strides
    ):
        raise ValueError(
            "canonical region token strides must be [region, stride] pairs"
        )
    if len(region_strides) > KV_PLACEMENT_MAX_RUNS:
        raise ValueError("canonical page mapping has too many semantic regions")
    return CanonicalPageMapping(
        canonical_page_size_bytes=data["canonical_page_size_bytes"],
        local_page_size_bytes=data["local_page_size_bytes"],
        runs=tuple(_copy_run_from_dict(run) for run in runs),
        num_writers=data["num_writers"],
        writer_index=data["writer_index"],
        parallelism_agnostic=data["parallelism_agnostic"],
        canonical_token_span=data["canonical_token_span"],
        canonical_region_token_strides=tuple(
            (item[0], item[1]) for item in region_strides
        ),
        is_opaque=data["is_opaque"],
        opaque_layout_signature=data["opaque_layout_signature"],
    )


@dataclass(frozen=True)
class KVGroupFormat:
    """Canonical byte format for one transfer-homogeneous KV group.

    ``group_id`` is the endpoint-local cache-manager index. ``semantic_id`` is
    stable across deployments and is used to match groups that were ordered or
    partitioned differently. Serializers must split heterogeneous local groups
    into homogeneous protocol groups.
    """

    group_id: int
    semantic_id: str
    kind: str
    layer_names: tuple[str, ...]
    canonical_page_token_span: int
    dtype: str
    canonical_page_size_bytes: int
    format_id: str
    quantization: str | None = None
    scale_dtype: str | None = None
    scale_granularity: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.layer_names, (str, bytes)):
            raise ValueError("layer_names must be a sequence of names")
        try:
            layer_names = tuple(self.layer_names)
        except TypeError as error:
            raise ValueError("layer_names must be a sequence of names") from error
        object.__setattr__(self, "layer_names", layer_names)

        _require_nonnegative_int(self.group_id, "group_id")
        _require_string(self.semantic_id, "semantic_id")
        _require_string(self.kind, "kind")
        _require_positive_int(
            self.canonical_page_token_span, "canonical_page_token_span"
        )
        _require_string(self.dtype, "dtype")
        _require_positive_int(
            self.canonical_page_size_bytes, "canonical_page_size_bytes"
        )
        _require_string(self.format_id, "format_id")
        if not layer_names:
            raise ValueError("layer_names must not be empty")
        for layer_name in layer_names:
            _require_string(layer_name, "layer_name")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names must be unique within a group")

        for name in ("quantization", "scale_dtype", "scale_granularity"):
            value = getattr(self, name)
            if value is not None:
                _require_string(value, name)
        if self.quantization is None and (
            self.scale_dtype is not None or self.scale_granularity is not None
        ):
            raise ValueError("scale metadata requires quantization")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible wire representation."""
        return {
            "group_id": self.group_id,
            "semantic_id": self.semantic_id,
            "kind": self.kind,
            "layer_names": list(self.layer_names),
            "canonical_page_token_span": self.canonical_page_token_span,
            "dtype": self.dtype,
            "canonical_page_size_bytes": self.canonical_page_size_bytes,
            "format_id": self.format_id,
            "quantization": self.quantization,
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
        }

    @classmethod
    def from_dict(cls, value: object) -> "KVGroupFormat":
        """Parse a wire representation, rejecting missing or unknown fields."""
        data = _wire_fields(value, set(cls.__dataclass_fields__), "KV group format")
        layer_names = data["layer_names"]
        if not isinstance(layer_names, list):
            raise ValueError("layer_names must be an array")
        if len(layer_names) > KV_PLACEMENT_MAX_LAYERS:
            raise ValueError("KV group format has too many layers")
        data["layer_names"] = tuple(layer_names)
        return cls(**data)


@dataclass(frozen=True)
class KVFormatManifest:
    """Versioned canonical KV format manifest for an endpoint."""

    version: int
    model_fingerprint: str
    groups: tuple[KVGroupFormat, ...]

    def __post_init__(self) -> None:
        if isinstance(self.groups, (str, bytes)):
            raise ValueError("groups must be a sequence")
        try:
            groups = tuple(self.groups)
        except TypeError as error:
            raise ValueError("groups must be a sequence") from error
        object.__setattr__(self, "groups", groups)

        _require_nonnegative_int(self.version, "version")
        if self.version != KV_PLACEMENT_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported KV placement protocol version {self.version}"
            )
        _require_string(self.model_fingerprint, "model_fingerprint")
        if not groups:
            raise ValueError("groups must not be empty")
        if len(groups) > KV_PLACEMENT_MAX_GROUPS:
            raise ValueError("KV format manifest has too many groups")
        if any(not isinstance(group, KVGroupFormat) for group in groups):
            raise ValueError("groups must contain KVGroupFormat values")

        group_ids = [group.group_id for group in groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("group_id values must be unique")
        semantic_ids = [group.semantic_id for group in groups]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("semantic_id values must be unique")
        layer_names = [name for group in groups for name in group.layer_names]
        if len(layer_names) > KV_PLACEMENT_MAX_LAYERS:
            raise ValueError("KV format manifest has too many layers")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("layer names must be unique across groups")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible wire representation."""
        return {
            "version": self.version,
            "model_fingerprint": self.model_fingerprint,
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, value: object) -> "KVFormatManifest":
        """Parse a wire representation, rejecting unknown protocol fields."""
        data = _wire_fields(
            value, {"version", "model_fingerprint", "groups"}, "KV format manifest"
        )
        groups = data["groups"]
        if not isinstance(groups, list):
            raise ValueError("groups must be an array")
        if len(groups) > KV_PLACEMENT_MAX_GROUPS:
            raise ValueError("KV format manifest has too many groups")
        return cls(
            version=data["version"],
            model_fingerprint=data["model_fingerprint"],
            groups=tuple(KVGroupFormat.from_dict(group) for group in groups),
        )

    def group(self, group_id: int) -> KVGroupFormat:
        """Return a group by id, failing closed when it was not advertised."""
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise ValueError(f"unknown KV group {group_id}")

    def semantic_group(self, semantic_id: str) -> KVGroupFormat:
        """Return a group by its cross-deployment semantic identity."""
        for group in self.groups:
            if group.semantic_id == semantic_id:
                return group
        raise ValueError(f"unknown semantic KV group {semantic_id!r}")

    def fingerprint(self) -> str:
        """Digest the exact manifest, including endpoint-local page geometry."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def canonical_space_id(self, semantic_id: str) -> str:
        """Digest semantic bytes while excluding endpoint-local page geometry."""
        group = self.semantic_group(semantic_id)
        semantic_format = {
            "version": self.version,
            "model_fingerprint": self.model_fingerprint,
            "semantic_id": group.semantic_id,
            "kind": group.kind,
            # Group ordering is endpoint-local metadata. Canonical byte
            # identity must survive a scheduler or PP stage enumerating the
            # same semantic layers in a different order.
            "layer_names": sorted(group.layer_names),
            "dtype": group.dtype,
            "format_id": group.format_id,
            "quantization": group.quantization,
            "scale_dtype": group.scale_dtype,
            "scale_granularity": group.scale_granularity,
        }
        encoded = json.dumps(
            semantic_format, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LayerPageMapping:
    """Named per-layer mapping carried by a rank placement manifest."""

    layer_name: str
    layer_index: int
    semantic_group_id: str
    mapping: CanonicalPageMapping

    def __post_init__(self) -> None:
        _require_string(self.layer_name, "layer_name")
        _require_nonnegative_int(self.layer_index, "layer_index")
        _require_string(self.semantic_group_id, "semantic_group_id")
        if not isinstance(self.mapping, CanonicalPageMapping):
            raise ValueError("mapping must be a CanonicalPageMapping")
        _validate_placement_compact(
            PagePlacement(
                rank=0,
                local_page_id=0,
                canonical_page_index=0,
                mapping=self.mapping,
                canonical_space_id="manifest-validation",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "semantic_group_id": self.semantic_group_id,
            "mapping": _mapping_to_dict(self.mapping),
        }

    @classmethod
    def from_dict(cls, value: object) -> "LayerPageMapping":
        data = _wire_fields(
            value,
            _LAYER_PAGE_MAPPING_WIRE_FIELDS,
            "layer page mapping",
        )
        return cls(
            layer_name=data["layer_name"],
            layer_index=data["layer_index"],
            semantic_group_id=data["semantic_group_id"],
            mapping=_mapping_from_dict(data["mapping"]),
        )


@dataclass(frozen=True)
class RankPlacementManifest:
    """A worker's topology coordinates and per-layer canonical mappings.

    DP coordinates select the request/attention replica. EP coordinates are
    diagnostic metadata only and never affect KV byte placement. A deployment
    called "DEP" is represented by DP routing plus EP diagnostics, not by an
    independent KV-placement axis.
    """

    version: int
    deployment_id: str
    topology_generation: int
    worker_id: str
    worker_incarnation: str
    format_manifest_fingerprint: str
    rank: int
    tp_size: int
    tp_rank: int
    dcp_size: int
    dcp_rank: int
    dcp_group_id: str
    pcp_size: int
    pcp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    dp_group_id: str
    ep_size: int
    ep_rank: int
    cp_interleave: int
    layer_range: tuple[int, int]
    mappings: tuple[LayerPageMapping, ...]

    def __post_init__(self) -> None:
        if isinstance(self.layer_range, (str, bytes)):
            raise ValueError("layer_range must be a two-item sequence")
        try:
            layer_range = tuple(self.layer_range)
        except TypeError as error:
            raise ValueError("layer_range must be a two-item sequence") from error
        object.__setattr__(self, "layer_range", layer_range)

        raw_mappings = self.mappings
        if isinstance(raw_mappings, (str, bytes, Mapping)):
            raise ValueError("mappings must be a sequence of LayerPageMapping values")
        try:
            mappings = tuple(raw_mappings)
        except TypeError as error:
            raise ValueError(
                "mappings must be a sequence of LayerPageMapping values"
            ) from error
        object.__setattr__(self, "mappings", mappings)

        _require_nonnegative_int(self.version, "version")
        if self.version != KV_PLACEMENT_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported KV placement protocol version {self.version}"
            )
        _require_string(self.deployment_id, "deployment_id")
        _require_nonnegative_int(self.topology_generation, "topology_generation")
        _require_string(self.worker_id, "worker_id")
        _require_string(self.worker_incarnation, "worker_incarnation")
        _require_string(self.format_manifest_fingerprint, "format_manifest_fingerprint")
        _require_nonnegative_int(self.rank, "rank")
        for axis in ("tp", "dcp", "pcp", "pp", "dp", "ep"):
            size = getattr(self, f"{axis}_size")
            rank = getattr(self, f"{axis}_rank")
            _require_positive_int(size, f"{axis}_size")
            _require_nonnegative_int(rank, f"{axis}_rank")
            if rank >= size:
                raise ValueError(f"{axis}_rank must be in [0, {axis}_size)")
        _require_string(self.dcp_group_id, "dcp_group_id")
        _require_string(self.dp_group_id, "dp_group_id")
        _require_positive_int(self.cp_interleave, "cp_interleave")
        if len(layer_range) != 2:
            raise ValueError("layer_range must contain exactly two values")
        layer_start, layer_end = layer_range
        _require_nonnegative_int(layer_start, "layer_range start")
        _require_nonnegative_int(layer_end, "layer_range end")
        if layer_start > layer_end:
            raise ValueError("layer_range must be half-open with start <= end")
        if any(not isinstance(mapping, LayerPageMapping) for mapping in mappings):
            raise ValueError("mappings must contain LayerPageMapping values")
        if len(mappings) > KV_PLACEMENT_MAX_MAPPINGS:
            raise ValueError("rank placement has too many layer mappings")
        if sum(len(mapping.mapping.runs) for mapping in mappings) > (
            KV_PLACEMENT_MAX_RUNS
        ):
            raise ValueError("rank placement has too many copy runs")
        mapping_names = [mapping.layer_name for mapping in mappings]
        if len(set(mapping_names)) != len(mapping_names):
            raise ValueError("mapping layer names must be unique")
        if any(
            not layer_start <= mapping.layer_index < layer_end for mapping in mappings
        ):
            raise ValueError("mapping layer_index must be inside layer_range")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible wire representation."""
        return {
            "version": self.version,
            "deployment_id": self.deployment_id,
            "topology_generation": self.topology_generation,
            "worker_id": self.worker_id,
            "worker_incarnation": self.worker_incarnation,
            "format_manifest_fingerprint": self.format_manifest_fingerprint,
            "rank": self.rank,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "dcp_size": self.dcp_size,
            "dcp_rank": self.dcp_rank,
            "dcp_group_id": self.dcp_group_id,
            "pcp_size": self.pcp_size,
            "pcp_rank": self.pcp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "dp_size": self.dp_size,
            "dp_rank": self.dp_rank,
            "dp_group_id": self.dp_group_id,
            "ep_size": self.ep_size,
            "ep_rank": self.ep_rank,
            "cp_interleave": self.cp_interleave,
            "layer_range": list(self.layer_range),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
        }

    @classmethod
    def from_dict(cls, value: object) -> "RankPlacementManifest":
        """Parse a wire representation, including nested page mappings."""
        data = _wire_fields(value, set(cls.__dataclass_fields__), "rank placement")
        layer_range = data["layer_range"]
        mappings = data["mappings"]
        if not isinstance(layer_range, list):
            raise ValueError("layer_range must be an array")
        if not isinstance(mappings, list):
            raise ValueError("mappings must be an array")
        _prescan_rank_mapping_budget(mappings)
        data["layer_range"] = tuple(layer_range)
        data["mappings"] = tuple(
            LayerPageMapping.from_dict(mapping) for mapping in mappings
        )
        return cls(**data)

    def mapping_for(self, layer_name: str) -> CanonicalPageMapping:
        """Return a layer mapping, failing closed when it was not advertised."""
        for layer_mapping in self.mappings:
            if layer_mapping.layer_name == layer_name:
                return layer_mapping.mapping
        raise ValueError(f"rank does not advertise layer {layer_name!r}")

    def validate_format(self, manifest: KVFormatManifest) -> None:
        """Bind mappings to the exact advertised format and semantic groups."""
        if self.format_manifest_fingerprint != manifest.fingerprint():
            raise ValueError("rank placement does not match KV format manifest")
        for layer_mapping in self.mappings:
            group = manifest.semantic_group(layer_mapping.semantic_group_id)
            if layer_mapping.layer_name not in group.layer_names:
                raise ValueError(
                    f"layer {layer_mapping.layer_name!r} is not in semantic group "
                    f"{group.semantic_id!r}"
                )
            mapping = layer_mapping.mapping
            if mapping.canonical_token_span != group.canonical_page_token_span:
                raise ValueError(
                    f"layer {layer_mapping.layer_name!r} canonical token span "
                    "does not match its KV group format"
                )
            if mapping.canonical_page_size_bytes != group.canonical_page_size_bytes:
                raise ValueError(
                    f"layer {layer_mapping.layer_name!r} canonical page size "
                    "does not match its KV group format"
                )


@dataclass(frozen=True)
class ConnectorCapabilities:
    """Copy primitives and directions supported by a connector.

    ``max_segments_per_batch`` is only a batching hint.  It never limits a
    composed plan and must not be used to select packing or reject a transfer.
    """

    contiguous_copy: bool
    strided_copy: bool
    scatter_gather: bool
    gpu_pack_unpack: bool
    supports_read: bool
    supports_write: bool
    max_segments_per_batch: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "contiguous_copy",
            "strided_copy",
            "scatter_gather",
            "gpu_pack_unpack",
            "supports_read",
            "supports_write",
        ):
            _require_bool(getattr(self, name), name)
        if not self.supports_read and not self.supports_write:
            raise ValueError("a connector must support reads, writes, or both")
        if self.max_segments_per_batch is not None:
            _require_positive_int(self.max_segments_per_batch, "max_segments_per_batch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contiguous_copy": self.contiguous_copy,
            "strided_copy": self.strided_copy,
            "scatter_gather": self.scatter_gather,
            "gpu_pack_unpack": self.gpu_pack_unpack,
            "supports_read": self.supports_read,
            "supports_write": self.supports_write,
            "max_segments_per_batch": self.max_segments_per_batch,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConnectorCapabilities":
        data = _wire_fields(value, set(cls.__dataclass_fields__), "capabilities")
        return cls(**data)


@dataclass(frozen=True)
class KVRange:
    """Canonical token coverage requested for one KV cache group.

    Allocated coverage is ``[first_token, first_token + token_count)``.  Valid
    (unpadded) tokens are its prefix of length ``valid_token_count``.
    """

    semantic_group_id: str
    first_token: int
    token_count: int
    valid_token_count: int

    def __post_init__(self) -> None:
        _require_string(self.semantic_group_id, "semantic_group_id")
        _require_nonnegative_int(self.first_token, "first_token")
        _require_nonnegative_int(self.token_count, "token_count")
        _require_nonnegative_int(self.valid_token_count, "valid_token_count")
        if self.valid_token_count > self.token_count:
            raise ValueError("valid_token_count must not exceed token_count")

    @property
    def end_token(self) -> int:
        return self.first_token + self.token_count

    @property
    def valid_end_token(self) -> int:
        return self.first_token + self.valid_token_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_group_id": self.semantic_group_id,
            "first_token": self.first_token,
            "token_count": self.token_count,
            "valid_token_count": self.valid_token_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "KVRange":
        data = _wire_fields(value, set(cls.__dataclass_fields__), "KV range")
        return cls(**data)


def validate_kv_ranges(
    ranges: Sequence[KVRange],
    manifest: KVFormatManifest | None = None,
) -> tuple[KVRange, ...]:
    """Validate sparse request coverage in deterministic group/token order.

    A group may contain multiple ranges (for example, prefix-cache holes), but
    ranges within that group must not overlap.  Adjacent ranges are retained so
    request metadata can preserve allocator/page boundaries.
    """
    normalized = tuple(ranges)
    if any(not isinstance(kv_range, KVRange) for kv_range in normalized):
        raise ValueError("ranges must contain KVRange values")
    if manifest is not None:
        known_group_ids = {group.semantic_id for group in manifest.groups}
        unknown = sorted(
            {kv_range.semantic_group_id for kv_range in normalized} - known_group_ids
        )
        if unknown:
            raise ValueError(f"ranges reference unknown KV groups {unknown}")
    ordered = tuple(
        sorted(
            normalized,
            key=lambda kv_range: (
                kv_range.semantic_group_id,
                kv_range.first_token,
            ),
        )
    )
    previous_by_group: dict[str, KVRange] = {}
    for kv_range in ordered:
        previous = previous_by_group.get(kv_range.semantic_group_id)
        if previous is not None and kv_range.first_token < previous.end_token:
            raise ValueError(
                f"KV ranges for group {kv_range.semantic_group_id!r} overlap at token "
                f"{kv_range.first_token}"
            )
        previous_by_group[kv_range.semantic_group_id] = kv_range
    return ordered


__all__ = [
    "CanonicalPageMapping",
    "ConnectorCapabilities",
    "CopyRun",
    "KVFormatManifest",
    "KVGroupFormat",
    "KVRange",
    "KV_PLACEMENT_MAX_GROUPS",
    "KV_PLACEMENT_MAX_LAYERS",
    "KV_PLACEMENT_MAX_MAPPINGS",
    "KV_PLACEMENT_MAX_RUNS",
    "KV_PLACEMENT_PROTOCOL_VERSION",
    "LayerPageMapping",
    "PagePlacement",
    "RankPlacementManifest",
    "TransferRun",
    "compose_page_placements",
    "iter_page_placement_transfer_runs",
    "validate_kv_ranges",
]
