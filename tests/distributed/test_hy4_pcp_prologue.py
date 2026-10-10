# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4PCPKVExchange must fill the paged caches exactly as all-gathering the
inputs and running the production kernel chain does."""

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _gather(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    out = t.new_empty((dist.get_world_size() * t.shape[0], *t.shape[1:]))
    dist.all_gather_into_tensor(out, t)
    return out


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    try:
        from types import SimpleNamespace

        import vllm._custom_ops  # noqa: F401  registers torch.ops._C
        from tests.kernels.test_hy4_pcp_prologue import (
            BLOCK,
            _idx_rows,
            _rows,
            make_inputs,
            reference_k,
        )
        from vllm.v1.attention.ops.pcp import HY4PCPKVExchange

        if not hasattr(torch.ops._C, "hy4_pcp_kv_norm_rope_quant_multicast"):
            return
        rank, world = dist.get_rank(), dist.get_world_size()
        kv_exchange = HY4PCPKVExchange(dist.group.WORLD, device, max_local_tokens=2048)
        grids = (148, 16, 37, 296)
        for step, num_tokens in enumerate([1, 7, 1000, 2048, 333, 2048, 64, 512]):
            x = make_inputs(num_tokens, torch.bfloat16, seed=100 * step + rank)
            # Production chain on the gathered inputs, with global slots.
            rows = world * num_tokens
            blocks = (rows + BLOCK - 1) // BLOCK + 4
            g = torch.Generator().manual_seed(step)
            slots = torch.randperm(blocks * BLOCK, generator=g)[:rows].to(device)
            slots[::17] = -1  # padding rows are skipped
            idx_slots = slots.flip(0).contiguous()
            gx = SimpleNamespace(**vars(x))
            gx.T = rows
            gx.kv_lora = _gather(x.kv_lora)
            gx.kw = _gather(x.kw)
            gx.positions = _gather(x.positions)
            mla_ref = torch.zeros(blocks, BLOCK, 352, dtype=torch.uint8, device=device)
            idx_ref = torch.zeros(blocks, BLOCK, 132, dtype=torch.uint8, device=device)
            reference_k(gx, mla_ref, idx_ref, slots)  # negative slots are skipped

            # Fused exchange: the multicast grid differs per rank and per step,
            # and one rank starts late.
            kv_exchange.multicast_blocks = grids[(step + rank) % len(grids)]
            if rank == step % world:
                torch.cuda._sleep(2_000_000)
            kv_exchange.multicast(
                x.kv_lora, x.kw[:, :128], x.positions, x.cos_sin, x.idx_cos_sin,
                x.kv_w, 1e-6, x.norm.weight, x.norm.bias, 1e-6,
            )  # fmt: skip
            torch.matmul(x.q[:, 0].float().T, x.q[:, 0].float())  # unrelated work
            mla = torch.zeros_like(mla_ref)
            idx = torch.zeros_like(idx_ref)
            kv_exchange.wait_cache_insert(slots, mla, idx, idx_slots)
            torch.accelerator.synchronize()

            valid = slots >= 0
            mla_rows = _rows(mla, slots[valid])
            torch.testing.assert_close(
                mla_rows, _rows(mla_ref, slots[valid]), atol=0, rtol=0
            )
            # Row r went to idx_slots[r] here and to slots[r] in the reference.
            both = valid & (idx_slots >= 0)
            v, s = _idx_rows(idx, idx_slots[both])
            v_ref, s_ref = _idx_rows(idx_ref, slots[both])
            torch.testing.assert_close(v, v_ref, atol=0, rtol=0)
            torch.testing.assert_close(s, s_ref, atol=0, rtol=0)
            assert kv_exchange.state[0].item() == step + 1
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _has_multicast_support(), reason="needs NVLS multicast")
@pytest.mark.parametrize("world_size", [2, 4])
def test_hy4_pcp_prologue_matches_all_gather(world_size):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    port = str(get_open_port())
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        p = mp.Process(target=_worker, args=(env,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join(timeout=600)
    for p in processes:
        if p.is_alive():
            p.kill()
            p.join()
        assert p.exitcode == 0
