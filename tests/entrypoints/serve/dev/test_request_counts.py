# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from vllm.entrypoints.serve.dev.rlhf.api_router import request_counts


def test_request_counts_returns_authoritative_scheduler_snapshot():
    snapshot = {
        "pause_state": "paused_all",
        "running_count": 0,
        "waiting_count": 2,
        "skipped_waiting_count": 0,
        "running_request_ids": [],
        "waiting_request_ids": ["cmpl-fixed-0", "cmpl-fixed-1"],
        "skipped_waiting_request_ids": [],
    }
    engine = SimpleNamespace(
        get_request_queue_snapshot=AsyncMock(return_value=snapshot)
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine))
    )

    response = asyncio.run(request_counts(request))

    assert response.status_code == 200
    assert json.loads(response.body) == snapshot
