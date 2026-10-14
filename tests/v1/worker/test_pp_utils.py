# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from unittest.mock import Mock

import numpy as np

from vllm.v1.worker.gpu import pp_utils


def _batch(
    num_computed, prefill_len, num_scheduled, is_prefilling=None, max_seq_len=None
):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        is_prefilling_np=(
            np.array(is_prefilling, dtype=np.bool_)
            if is_prefilling is not None
            else None
        ),
        max_seq_len_np=(
            np.array(max_seq_len, dtype=np.int32) if max_seq_len is not None else None
        ),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_final_prefill_with_one_output_skips_broadcast():
    batch = _batch([1000], [1004], [4], [True], [1005])
    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_final_prefill_with_decode_budget_keeps_broadcast():
    batch = _batch([1000], [1004], [4], [True], [1260])
    mask = pp_utils.compute_need_sampled_mask(batch)
    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_past_length_cap_keeps_broadcast():
    batch = _batch([14176], [12175], [8], [False], [12176])
    mask = pp_utils.compute_need_sampled_mask(batch)
    assert mask is not None
    assert mask.tolist() == [True]


def test_mixed_batch_keeps_continuing_request():
    batch = _batch([1000, 10], [1004, 8], [4, 1], [True, False], [1005, 1024])
    mask = pp_utils.compute_need_sampled_mask(batch)
    assert mask is not None
    assert mask.tolist() == [False, True]
