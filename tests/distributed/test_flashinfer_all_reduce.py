# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import (
    cuda_communicator as cuda_communicator_module,
)
from vllm.distributed.device_communicators import flashinfer_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator


def _mock_sm103_tp8_mnnvl(
    monkeypatch: pytest.MonkeyPatch,
    node_count: int,
) -> tuple[object, Mock]:
    capability = Mock()
    capability.to_int.return_value = 103
    group = object()
    group_node_count = Mock(return_value=node_count)
    monkeypatch.setattr(flashinfer_all_reduce, "fi_ar_available", True)
    monkeypatch.setattr(flashinfer_all_reduce.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        flashinfer_all_reduce.current_platform,
        "get_device_capability",
        lambda: capability,
    )
    monkeypatch.setattr(flashinfer_all_reduce.dist, "get_world_size", lambda _: 8)
    monkeypatch.setattr(flashinfer_all_reduce.dist, "get_rank", lambda _: 0)
    monkeypatch.setattr(flashinfer_all_reduce, "get_node_count", lambda: node_count)
    monkeypatch.setattr(flashinfer_all_reduce, "_node_count", group_node_count)
    monkeypatch.setattr(
        flashinfer_all_reduce, "_resolve_fi_ar_backend", lambda: ("mnnvl", False)
    )
    monkeypatch.setattr(
        flashinfer_all_reduce.PassConfig,
        "default_fi_allreduce_fusion_max_size_mb",
        lambda *, use_mnnvl_tuning=False: {8: 16 if use_mnnvl_tuning else 2},
    )
    return group, group_node_count


@pytest.mark.parametrize(
    ("tp_group_node_count", "expected_size_mb", "expected_exclusive"),
    [(2, 84, True), (1, 2, False), (4, 2, False)],
)
def test_standalone_workspace_tuning_uses_tp_group_topology(
    monkeypatch: pytest.MonkeyPatch,
    tp_group_node_count: int,
    expected_size_mb: int,
    expected_exclusive: bool,
) -> None:
    group, group_node_count = _mock_sm103_tp8_mnnvl(
        monkeypatch,
        tp_group_node_count,
    )

    communicator = flashinfer_all_reduce.FlashInferAllReduce(
        group=group,  # type: ignore[arg-type]
        device="cuda",
    )

    group_node_count.assert_called_once_with(group)
    assert (
        communicator.max_workspace_size == expected_size_mb * flashinfer_all_reduce.MiB
    )
    assert communicator.max_size_is_exclusive is expected_exclusive


@pytest.mark.parametrize(
    ("shape", "dtype", "nbytes", "is_contiguous"),
    [
        ((1, 1), torch.float64, 8, True),
        ((1, 1), torch.bfloat16, 84 * flashinfer_all_reduce.MiB, True),
        ((1, 1), torch.bfloat16, 2, False),
        ((1,), torch.bfloat16, 2, True),
    ],
)
def test_standalone_all_reduce_rejects_unsupported_inputs_without_workspace(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    nbytes: int,
    is_contiguous: bool,
) -> None:
    communicator = flashinfer_all_reduce.FlashInferAllReduce.__new__(
        flashinfer_all_reduce.FlashInferAllReduce
    )
    communicator.disabled = False
    communicator.max_workspace_size = 84 * flashinfer_all_reduce.MiB
    communicator.max_size_is_exclusive = True
    communicator.max_num_tokens = 0
    communicator._ensure_workspace = Mock(return_value=True)  # type: ignore[method-assign]
    input_tensor = Mock(
        is_cuda=True,
        shape=shape,
        dtype=dtype,
        nbytes=nbytes,
    )
    input_tensor.is_contiguous.return_value = is_contiguous

    assert not communicator.should_use_fi_ar(input_tensor)
    communicator._ensure_workspace.assert_not_called()


def test_primary_initialization_promotes_shared_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group, _ = _mock_sm103_tp8_mnnvl(monkeypatch, node_count=2)
    workspace = Mock(backend="mnnvl")
    create_workspace = Mock(return_value=workspace)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace", None)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_quant_workspace", None)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace_groups", {})
    monkeypatch.setattr(flashinfer_all_reduce, "_create_workspace", create_workspace)

    workspace_kwargs = dict(
        world_size=8,
        rank=0,
        max_token_num=128,
        hidden_dim=7168,
        dtype=torch.bfloat16,
        group=group,  # type: ignore[arg-type]
    )
    primary_workspace = flashinfer_all_reduce.get_fi_ar_workspace(**workspace_kwargs)
    quant_workspace = flashinfer_all_reduce.get_fi_ar_quant_workspace(
        **workspace_kwargs
    )
    standalone_workspace = flashinfer_all_reduce.get_fi_ar_workspace(
        world_size=8,
        rank=0,
        max_token_num=6144,
        hidden_dim=7168,
        dtype=torch.bfloat16,
        group=group,  # type: ignore[arg-type]
    )

    assert primary_workspace is workspace
    assert quant_workspace is workspace
    assert standalone_workspace is workspace
    create_workspace.assert_called_once_with(
        "mnnvl",
        8,
        0,
        6144,
        7168,
        torch.bfloat16,
        group,
    )


def test_single_node_fusion_first_promotes_to_mnnvl_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group, _ = _mock_sm103_tp8_mnnvl(monkeypatch, node_count=1)
    workspace = Mock(backend="mnnvl")
    create_workspace = Mock(return_value=workspace)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace", None)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_quant_workspace", None)
    monkeypatch.setattr(flashinfer_all_reduce, "_create_workspace", create_workspace)

    result = flashinfer_all_reduce.get_fi_ar_workspace(
        world_size=8,
        rank=0,
        max_token_num=32,
        hidden_dim=7168,
        dtype=torch.bfloat16,
        group=group,  # type: ignore[arg-type]
    )

    assert result is workspace
    create_workspace.assert_called_once_with(
        "mnnvl",
        8,
        0,
        1170,
        7168,
        torch.bfloat16,
        group,
    )


def test_disabled_communicator_still_destroys_shared_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroy_workspace = Mock()
    monkeypatch.setattr(
        flashinfer_all_reduce, "destroy_fi_ar_workspace", destroy_workspace
    )
    communicator = flashinfer_all_reduce.FlashInferAllReduce.__new__(
        flashinfer_all_reduce.FlashInferAllReduce
    )
    communicator.disabled = True

    communicator.destroy()

    destroy_workspace.assert_called_once_with()


def _cuda_communicator_for_all_reduce(fi_ar_comm: Mock) -> CudaCommunicator:
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.pynccl_comm = Mock(world_size=8, disabled=False)
    communicator.fi_ar_comm = fi_ar_comm
    communicator.qr_comm = None
    communicator.aiter_ar_comm = None
    communicator.ca_comm = None
    communicator.symm_mem_comm = None
    return communicator


def test_cuda_communicator_dispatches_eligible_flashinfer_before_nccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tensor = torch.empty(1)
    output = torch.empty(2)
    fi_ar_comm = Mock(disabled=False)
    fi_ar_comm.should_use_fi_ar.return_value = True
    fi_ar_comm.all_reduce.return_value = output
    communicator = _cuda_communicator_for_all_reduce(fi_ar_comm)
    nccl_selector = Mock(return_value=True)
    monkeypatch.setattr(
        cuda_communicator_module,
        "should_nccl_symm_mem_allreduce",
        nccl_selector,
    )

    assert communicator.all_reduce(input_tensor) is output
    fi_ar_comm.should_use_fi_ar.assert_called_once_with(input_tensor)
    fi_ar_comm.all_reduce.assert_called_once_with(input_tensor)
    nccl_selector.assert_not_called()


def test_cuda_communicator_uses_nccl_when_flashinfer_rejects_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tensor = torch.empty(1)
    output = torch.empty(2)
    fi_ar_comm = Mock(disabled=False)
    fi_ar_comm.should_use_fi_ar.return_value = False
    communicator = _cuda_communicator_for_all_reduce(fi_ar_comm)
    monkeypatch.setattr(
        cuda_communicator_module,
        "should_nccl_symm_mem_allreduce",
        Mock(return_value=True),
    )
    nccl_all_reduce = Mock(return_value=output)
    monkeypatch.setattr(
        torch.ops.vllm,
        "all_reduce_symmetric_with_copy",
        nccl_all_reduce,
        raising=False,
    )

    assert communicator.all_reduce(input_tensor) is output
    nccl_all_reduce.assert_called_once_with(input_tensor)
    fi_ar_comm.should_use_fi_ar.assert_called_once_with(input_tensor)
    fi_ar_comm.all_reduce.assert_not_called()


def test_workspace_creation_failure_falls_through_to_nccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tensor = torch.empty(1)
    output = torch.empty(2)
    fi_ar_comm = Mock(disabled=False)

    def fail_workspace_creation(_: torch.Tensor) -> bool:
        fi_ar_comm.disabled = True
        return False

    fi_ar_comm.should_use_fi_ar.side_effect = fail_workspace_creation
    communicator = _cuda_communicator_for_all_reduce(fi_ar_comm)
    monkeypatch.setattr(
        cuda_communicator_module,
        "should_nccl_symm_mem_allreduce",
        Mock(return_value=True),
    )
    nccl_all_reduce = Mock(return_value=output)
    monkeypatch.setattr(
        torch.ops.vllm,
        "all_reduce_symmetric_with_copy",
        nccl_all_reduce,
        raising=False,
    )

    assert communicator.all_reduce(input_tensor) is output
    nccl_all_reduce.assert_called_once_with(input_tensor)
    fi_ar_comm.all_reduce.assert_not_called()
