# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
from collections.abc import Iterator
from typing import Any

import regex as re
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


class MultipartFrameLimitError(ValueError):
    """Raised before an inbound multipart message can exceed its frame cap."""


def recv_multipart_bounded(sock: zmq.Socket, max_frames: int) -> list[bytes]:
    """Receive at most ``max_frames`` without allocating an unbounded list.

    The caller must discard or recreate the socket after
    :class:`MultipartFrameLimitError`, because unread tail frames remain in the
    current message.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    parts: list[bytes] = []
    for _ in range(max_frames):
        part = sock.recv()
        if not isinstance(part, bytes):
            raise ValueError("multipart frames must be bytes")
        parts.append(part)
        if not sock.getsockopt(zmq.RCVMORE):
            return parts
    raise MultipartFrameLimitError(
        f"multipart message exceeds the {max_frames}-frame limit"
    )


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(
    socket_type: Any,
    addr: str,
    *,
    max_message_size: int | None = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a size-bounded ZMQ socket."""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx,
            path=addr,
            socket_type=socket_type,
            bind=socket_type == zmq.ROUTER,
            max_message_size=max_message_size,
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)
