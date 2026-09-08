# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

pytestmark = pytest.mark.skip_global_cleanup


def _parallel_config(*, tp: int = 2, pcp: int = 2, dcp: int = 2):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
    )


def test_mrv1_integrated_draft_rejects_unsupported_parallelism():
    with pytest.raises(NotImplementedError, match="to match target parallelism"):
        SpecDecodeBaseProposer._validate_integrated_draft_parallel_config(
            _parallel_config(),
            _parallel_config(pcp=1),
        )


def test_standalone_draft_rejects_context_parallelism():
    proposer = object.__new__(DraftModelProposer)
    proposer.speculative_config = SimpleNamespace(
        target_parallel_config=_parallel_config(),
        draft_parallel_config=_parallel_config(pcp=1),
    )

    with pytest.raises(NotImplementedError, match="currently require"):
        proposer._raise_if_draft_parallelism_unsupported()
