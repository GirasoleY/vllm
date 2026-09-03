# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.int8, False),
        (torch.float8_e4m3fn, False),
    ],
)
def test_custom_allreduce_filters_dtype(
    dtype: torch.dtype,
    expected: bool,
) -> None:
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 2
    communicator.max_size = 1024

    assert communicator.should_custom_ar(torch.empty(16, dtype=dtype)) is expected


@pytest.mark.parametrize(
    ("major", "local_multicast", "expected"),
    [
        (8, True, False),
        (9, True, False),
        (10, False, False),
        (10, True, True),
    ],
)
def test_cross_node_mnnvl_gate_checks_generation_and_multicast(
    monkeypatch,
    major,
    local_multicast,
    expected,
):
    def has_device_capability(capability, device_id):
        assert capability == 100
        assert device_id == 3
        return major >= 10

    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        has_device_capability,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: local_multicast,
    )
    monkeypatch.setattr(car.dist, "all_reduce", lambda *_args, **_kwargs: None)

    assert car._group_can_attempt_mnnvl(object(), torch.device("cuda:3")) is expected


def test_cross_node_mnnvl_gate_requires_support_on_every_rank(monkeypatch):
    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: True,
    )

    def report_unsupported_peer(support, **_kwargs):
        support.zero_()

    monkeypatch.setattr(car.dist, "all_reduce", report_unsupported_peer)

    assert not car._group_can_attempt_mnnvl(object(), torch.device("cuda:0"))


def test_local_multicast_support_rejects_non_cuda(monkeypatch):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: False)

    assert not car._has_local_multicast_support(torch.device("cuda:0"))


@pytest.mark.parametrize(
    ("world_size", "device_capability", "expected"),
    [
        (8, (10, 0), True),
        (8, (10, 3), True),
        (4, (10, 3), False),
        (8, (10, 1), False),
        (8, (9, 0), False),
    ],
)
def test_mnnvl_multimem_reduce_scatter_platform_gate(
    monkeypatch,
    world_size,
    device_capability,
    expected,
):
    def is_device_capability(capability, device_id):
        assert capability in ((10, 0), (10, 3))
        assert device_id == 3
        return device_capability == capability

    monkeypatch.setattr(
        car.current_platform,
        "is_device_capability",
        is_device_capability,
    )

    supported = car._supports_mnnvl_multimem_reduce_scatter(
        torch.device("cuda:3"), world_size
    )
    assert supported is expected


class _FakeMultimemBuffer:
    def __init__(self, ptr=0x1000):
        self.ptr = ptr
        self.zeroed_slice = None

    def data_ptr(self):
        return self.ptr

    def __getitem__(self, index):
        self.zeroed_slice = index
        return self

    def zero_(self):
        return self


class _FakeSymmetricMemory:
    def __init__(
        self,
        *,
        allocation_error=False,
        rendezvous_error=False,
        multicast_ptr=0x2000,
    ):
        self.allocation_error = allocation_error
        self.rendezvous_error = rendezvous_error
        self.multicast_ptr = multicast_ptr
        self.empty_calls = 0
        self.rendezvous_calls = 0
        self.storage_size = 0
        self.buffer = None

    def empty(self, size, *_args, **_kwargs):
        self.empty_calls += 1
        if self.allocation_error:
            raise RuntimeError("allocation failed")
        self.storage_size = size
        self.buffer = _FakeMultimemBuffer()
        return self.buffer

    def rendezvous(self, _buffer, group_name):
        assert group_name == "test"
        self.rendezvous_calls += 1
        if self.rendezvous_error:
            raise RuntimeError("rendezvous failed")
        return SimpleNamespace(
            multicast_ptr=self.multicast_ptr,
            get_buffer=lambda peer, *_args, **_kwargs: _FakeMultimemBuffer(
                0x1000 + peer * 0x10000
            ),
        )


def _make_multimem_communicator():
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.group = SimpleNamespace(group_name="test")
    communicator.device = torch.device("cuda:0")
    communicator.rank = 0
    communicator.world_size = 8
    communicator.mnnvl_multimem_rs_supported = True
    communicator.max_mnnvl_multimem_reduce_scatter_size = 64 * 1024 * 1024
    communicator.mnnvl_multimem_rs_init_attempted = False
    communicator._clear_mnnvl_multimem_reduce_scatter_buffer()
    return communicator


def _patch_multimem_init_runtime(monkeypatch, symm_mem):
    registered_ptrs = []
    monkeypatch.setattr(car, "torch_symm_mem", symm_mem)
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car, "_has_local_multicast_support", lambda _device: True)
    monkeypatch.setattr(car.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(car.ops, "meta_size", lambda: 4096)
    monkeypatch.setattr(
        car.ops, "register_buffer", lambda _ptr, ptrs: registered_ptrs.append(ptrs)
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    return registered_ptrs


@pytest.mark.parametrize(
    ("failed_phase", "empty_calls", "rendezvous_calls", "reduction_count"),
    [
        ("eligibility", 0, 0, 1),
        ("capacity", 0, 0, 1),
        ("allocation", 1, 0, 2),
        ("resource", 1, 1, 3),
        ("setup", 1, 1, 4),
    ],
)
def test_mnnvl_multimem_init_requires_group_consensus(
    monkeypatch,
    failed_phase,
    empty_calls,
    rendezvous_calls,
    reduction_count,
):
    symm_mem = _FakeSymmetricMemory()
    _patch_multimem_init_runtime(monkeypatch, symm_mem)
    reductions = 0

    def all_reduce(value, *, op, group):
        nonlocal reductions
        reductions += 1
        assert op == dist.ReduceOp.MIN
        assert group.group_name == "test"
        if reductions == 1 and failed_phase == "eligibility":
            value[0] = 0
        elif reductions == 1 and failed_phase == "capacity":
            value[1] = 32 * 1024 * 1024
        elif (reductions, failed_phase) in (
            (2, "allocation"),
            (3, "resource"),
            (4, "setup"),
        ):
            value.zero_()

    monkeypatch.setattr(car.dist, "all_reduce", all_reduce)
    communicator = _make_multimem_communicator()

    assert not communicator.initialize_mnnvl_multimem_reduce_scatter()
    assert reductions == reduction_count
    assert symm_mem.empty_calls == empty_calls
    assert symm_mem.rendezvous_calls == rendezvous_calls
    assert communicator.mnnvl_multimem_rs_buffer is None
    assert communicator.mnnvl_multimem_rs_handle is None
    assert communicator.mnnvl_multimem_rs_peer_buffers is None
    assert communicator.mnnvl_multimem_rs_buffer_size == 0
    assert communicator.mnnvl_multimem_rs_local_ptr == 0
    assert communicator.mnnvl_multimem_rs_multicast_ptr == 0
    assert (communicator._mnnvl_multimem_rs_setup_keepalive is not None) is (
        failed_phase == "setup"
    )


def test_mnnvl_multimem_init_coordinates_local_allocation_failure(monkeypatch):
    symm_mem = _FakeSymmetricMemory(allocation_error=True)
    _patch_multimem_init_runtime(monkeypatch, symm_mem)
    reductions = 0

    def all_reduce(_value, *, op, group):
        nonlocal reductions
        reductions += 1
        assert op == dist.ReduceOp.MIN
        assert group.group_name == "test"

    monkeypatch.setattr(car.dist, "all_reduce", all_reduce)
    communicator = _make_multimem_communicator()

    assert not communicator.initialize_mnnvl_multimem_reduce_scatter()
    assert reductions == 2
    assert symm_mem.empty_calls == 1
    assert symm_mem.rendezvous_calls == 0


@pytest.mark.parametrize("capacity", [0, True, torch.iinfo(torch.int32).max + 1])
def test_mnnvl_multimem_init_rejects_invalid_capacity(monkeypatch, capacity):
    symm_mem = _FakeSymmetricMemory()
    _patch_multimem_init_runtime(monkeypatch, symm_mem)
    monkeypatch.setattr(car.dist, "all_reduce", lambda *_args, **_kwargs: None)
    communicator = _make_multimem_communicator()
    communicator.max_mnnvl_multimem_reduce_scatter_size = capacity

    assert not communicator.initialize_mnnvl_multimem_reduce_scatter()
    assert symm_mem.empty_calls == 0


def test_mnnvl_multimem_init_propagates_rendezvous_failure(monkeypatch):
    symm_mem = _FakeSymmetricMemory(rendezvous_error=True)
    _patch_multimem_init_runtime(monkeypatch, symm_mem)
    monkeypatch.setattr(car.dist, "all_reduce", lambda *_args, **_kwargs: None)
    communicator = _make_multimem_communicator()

    with pytest.raises(RuntimeError, match="rendezvous failed"):
        communicator.initialize_mnnvl_multimem_reduce_scatter()

    assert symm_mem.empty_calls == 1
    assert symm_mem.rendezvous_calls == 1


def test_mnnvl_multimem_init_coordinates_signal_setup_failure(monkeypatch):
    symm_mem = _FakeSymmetricMemory()
    _patch_multimem_init_runtime(monkeypatch, symm_mem)

    def register_buffer(*_args):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(car.ops, "register_buffer", register_buffer)
    reductions = 0

    def all_reduce(_value, **_kwargs):
        nonlocal reductions
        reductions += 1

    monkeypatch.setattr(car.dist, "all_reduce", all_reduce)
    communicator = _make_multimem_communicator()

    assert not communicator.initialize_mnnvl_multimem_reduce_scatter()
    assert reductions == 4
    assert communicator.mnnvl_multimem_rs_peer_buffers is None
    assert communicator._mnnvl_multimem_rs_setup_keepalive is not None


def test_mnnvl_multimem_init_publishes_after_consensus_and_is_idempotent(
    monkeypatch,
):
    symm_mem = _FakeSymmetricMemory()
    registered_ptrs = _patch_multimem_init_runtime(monkeypatch, symm_mem)
    reductions = 0

    def all_reduce(_value, *, op, group):
        nonlocal reductions
        reductions += 1
        assert op == dist.ReduceOp.MIN
        assert group.group_name == "test"

    monkeypatch.setattr(car.dist, "all_reduce", all_reduce)
    communicator = _make_multimem_communicator()

    assert communicator.initialize_mnnvl_multimem_reduce_scatter()
    assert communicator.initialize_mnnvl_multimem_reduce_scatter()
    assert reductions == 4
    assert symm_mem.empty_calls == 1
    assert symm_mem.rendezvous_calls == 1
    assert communicator.mnnvl_multimem_rs_buffer is not None
    assert communicator.mnnvl_multimem_rs_handle is not None
    assert communicator.mnnvl_multimem_rs_peer_buffers is not None
    assert communicator.mnnvl_multimem_rs_buffer_size == 64 * 1024 * 1024
    assert communicator.mnnvl_multimem_rs_local_ptr == 0x1000
    assert communicator.mnnvl_multimem_rs_multicast_ptr == 0x2000
    assert symm_mem.storage_size == 64 * 1024 * 1024 + 4096
    assert symm_mem.buffer is not None
    assert symm_mem.buffer.zeroed_slice == slice(64 * 1024 * 1024, None)
    assert len(registered_ptrs) == 1
    assert registered_ptrs[0] == [0x1000 + peer * 0x10000 for peer in range(8)]


@pytest.mark.parametrize(
    ("local_available", "remote_available"),
    [(False, True), (True, False), (True, True)],
)
def test_sp_reduce_scatter_init_agrees_on_custom_ar_availability(
    monkeypatch, local_available, remote_available
):
    custom_ar = SimpleNamespace(
        disabled=False,
        initialize_mnnvl_multimem_reduce_scatter=lambda: True,
    )
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.ca_comm = custom_ar if local_available else None
    communicator.cpu_group = SimpleNamespace(group_name="test")
    communicator.is_stateless = False
    communicator._sp_reduce_scatter_init_result = None
    reductions = 0
    initializations = 0

    def initialize():
        nonlocal initializations
        initializations += 1
        return True

    custom_ar.initialize_mnnvl_multimem_reduce_scatter = initialize
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)

    def all_reduce(value, *, op, group):
        nonlocal reductions
        reductions += 1
        assert op == dist.ReduceOp.MIN
        assert group.group_name == "test"
        if not remote_available:
            value.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    expected = local_available and remote_available
    assert communicator.initialize_sp_reduce_scatter() is expected
    assert communicator.initialize_sp_reduce_scatter() is expected
    assert reductions == 1
    assert initializations == int(expected)


def test_sp_reduce_scatter_init_skips_stateless_group(monkeypatch):
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.is_stateless = True
    communicator._sp_reduce_scatter_init_result = None
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("unexpected collective"),
    )

    assert not communicator.initialize_sp_reduce_scatter()


@pytest.mark.parametrize(
    (
        "message_bytes",
        "lamport_ptr",
        "multimem_ptr",
        "batch_invariant",
        "expected",
    ),
    [
        (16 * 1024 * 1024, 1, 1, False, "mnnvl_lamport"),
        (16 * 1024 * 1024 + 128, 1, 1, False, "mnnvl_multimem"),
        (64 * 1024 * 1024, 1, 1, False, "mnnvl_multimem"),
        (64 * 1024 * 1024 + 128, 1, 1, False, None),
        (32 * 1024 * 1024, 1, 0, False, None),
        (8 * 1024 * 1024, 1, 1, True, "mnnvl_lamport"),
        (32 * 1024 * 1024, 1, 1, True, None),
        (8 * 1024 * 1024, 0, 1, False, "legacy"),
        (32 * 1024 * 1024, 0, 1, False, "mnnvl_multimem"),
    ],
)
def test_mnnvl_reduce_scatter_backend_gate(
    monkeypatch,
    message_bytes,
    lamport_ptr,
    multimem_ptr,
    batch_invariant,
    expected,
):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 8
    communicator.mnnvl_only = False
    communicator.fully_connected = True
    communicator.mnnvl_multicast_ptr = lamport_ptr
    communicator.mnnvl_multimem_rs_local_ptr = multimem_ptr
    communicator.mnnvl_multimem_rs_multicast_ptr = multimem_ptr
    communicator.mnnvl_multimem_rs_peer_buffers = (
        [_FakeMultimemBuffer()] if multimem_ptr else None
    )
    communicator.mnnvl_multimem_rs_buffer_size = 64 * 1024 * 1024
    communicator.max_mnnvl_reduce_scatter_size = 16 * 1024 * 1024
    communicator.max_mnnvl_multimem_reduce_scatter_size = 64 * 1024 * 1024
    communicator.max_reduce_scatter_size = 16 * 1024 * 1024
    inp = torch.empty((8, message_bytes // 16), dtype=torch.bfloat16)

    assert inp.nbytes == message_bytes
    assert communicator._select_reduce_scatter_backend(inp) == expected
    assert communicator.should_custom_reduce_scatter(inp) is (expected is not None)
    assert communicator.should_mnnvl_multimem_reduce_scatter(inp) is (
        expected == "mnnvl_multimem"
    )


def test_mnnvl_reduce_scatter_backend_rejects_scalar(monkeypatch):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 8
    communicator.mnnvl_only = False

    assert communicator._select_reduce_scatter_backend(torch.tensor(1.0)) is None


def test_mnnvl_reduce_scatter_backend_allows_cross_node_multimem(monkeypatch):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.envs, "VLLM_BATCH_INVARIANT", False)
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 8
    communicator.mnnvl_only = True
    communicator.fully_connected = False
    communicator.mnnvl_multicast_ptr = 0
    communicator.mnnvl_multimem_rs_local_ptr = 0x1000
    communicator.mnnvl_multimem_rs_multicast_ptr = 0x2000
    communicator.mnnvl_multimem_rs_peer_buffers = [_FakeMultimemBuffer()]
    communicator.mnnvl_multimem_rs_buffer_size = 64 * 1024 * 1024
    communicator.max_mnnvl_reduce_scatter_size = 16 * 1024 * 1024
    communicator.max_reduce_scatter_size = 16 * 1024 * 1024
    inp = torch.empty((8, 64), dtype=torch.bfloat16)

    assert communicator._select_reduce_scatter_backend(inp) == "mnnvl_multimem"


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)
