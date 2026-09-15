# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V2 warmup gates and PP preload ordering."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_kernel_warmup_restores_uncalibrated_adaptive_manager(monkeypatch, fail_warmup):
    """Startup must warm fixed drafts before calibration and retain its manager."""
    manager = SimpleNamespace(cost_tables=None)
    rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner = SimpleNamespace(
        adaptive_verification=manager,
        rejection_sampler=rejection_sampler,
    )

    def run_steps(model_runner, execute, sample):
        assert model_runner.adaptive_verification is None
        assert not model_runner.rejection_sampler.enable_adaptive_verification
        if fail_warmup:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(warmup, "_warmup_kernels", run_steps)
    if fail_warmup:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.warmup_kernels(runner, _fail, _fail)
    else:
        warmup.warmup_kernels(runner, _fail, _fail)
    assert runner.adaptive_verification is manager
    assert manager.cost_tables is None
    assert rejection_sampler.enable_adaptive_verification


@pytest.mark.parametrize("is_last_rank", [False, True])
@pytest.mark.parametrize("has_pp", [False, True])
def test_worker_preloads_before_pp_feedback_warmup(monkeypatch, is_last_rank, has_pp):
    """Each worker must finish loading before posting its own PP feedback."""
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm.config.compilation import CompilationMode
    from vllm.v1.worker import gpu_worker

    handler = SimpleNamespace(disabled=False) if has_pp else None
    if handler is not None:
        handler.set_disabled = lambda disabled: setattr(handler, "disabled", disabled)
    events = []
    runner = SimpleNamespace(
        pp_handler=handler,
        is_last_pp_rank=is_last_rank,
        is_pooling_model=False,
        lora_config=None,
        maybe_remove_all_loras=Mock(),
        warmup_pp_decode_update=lambda: events.append("deferred_update_loaded"),
    )
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.model_runner = runner
    worker.use_v2_model_runner = True
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    worker.model_config = SimpleNamespace(enforce_eager=False)
    worker._get_cudagraph_capture_context = nullcontext

    class CaptureReached(Exception):
        pass

    def capture_model():
        events.append("capture")
        raise CaptureReached

    runner.capture_model = capture_model

    def preload_registered_kernels(worker):
        assert handler is None or handler.disabled
        events.append("registered_kernels_loaded")

    def warmup_steps(model_runner, execute, sample):
        if handler is not None and not handler.disabled:
            assert events[-1] == "startup_shapes_loaded"
            events.append("feedback_warmup")
        else:
            assert "registered_kernels_loaded" in events
            if has_pp and not is_last_rank:
                assert "deferred_update_loaded" in events
            events.append("startup_shapes_loaded")

    monkeypatch.setattr(gpu_worker, "kernel_warmup", preload_registered_kernels)
    monkeypatch.setattr(gpu_worker, "warmup_kernels", warmup_steps)
    with pytest.raises(CaptureReached):
        worker.compile_or_warm_up_model()

    assert events.count("startup_shapes_loaded") == 1
    assert events.count("feedback_warmup") == int(has_pp)
    assert events[-1] == "capture"
    assert handler is None or not handler.disabled


@pytest.mark.parametrize("was_disabled", [False, True])
def test_worker_restores_feedback_state_when_preloading_fails(was_disabled):
    """A failed preload must retain the caller's feedback setting."""
    from vllm.v1.worker.gpu_worker import Worker

    handler = SimpleNamespace(disabled=was_disabled)
    handler.set_disabled = lambda disabled: setattr(handler, "disabled", disabled)
    worker = Worker.__new__(Worker)
    worker.model_runner = SimpleNamespace(pp_handler=handler)

    def fail_preload():
        assert handler.disabled
        raise RuntimeError("preload failed")

    worker._preload_model_kernels = fail_preload
    with pytest.raises(RuntimeError, match="preload failed"):
        worker.compile_or_warm_up_model()
    assert handler.disabled is was_disabled
