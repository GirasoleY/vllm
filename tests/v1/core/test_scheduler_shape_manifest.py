# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1.core.sched.shape_manifest import (
    hash_shape_records,
    normalize_shape_records,
    resolve_shape_manifest_path,
)

pytestmark = pytest.mark.cpu_test


def _record(request_id: str) -> dict:
    return {
        "schema_version": 1,
        "step_seq": 0,
        "queues_before": {
            "running": [],
            "waiting": [request_id],
            "skipped_waiting": [],
        },
        "scheduled": [
            {
                "request_id": request_id,
                "prompt_len": 120000,
                "computed_before": 113920,
                "scheduled_tokens": 6080,
                "phase": "prefill",
                "admission": "new",
            }
        ],
        "preempted": [],
        "queues_after": {
            "running": [request_id],
            "waiting": [],
            "skipped_waiting": [],
        },
    }


def test_shape_hash_normalizes_request_ids():
    first = _record("cmpl-tp-run-0")
    second = _record("cmpl-dcp-run-0")
    second["step_seq"] = 57

    assert hash_shape_records([first]) == hash_shape_records([second])
    normalized = normalize_shape_records([first])
    assert normalized[0]["scheduled"][0]["request_id"] == "r0"
    assert json.dumps(normalized, sort_keys=True) != json.dumps([first], sort_keys=True)


def test_shape_manifest_path_is_unique_without_placeholders(tmp_path):
    resolved = resolve_shape_manifest_path(
        str(tmp_path / "shapes.jsonl"), host="node/0", pid=17
    )

    assert resolved.name == "shapes.host-node_0.pid-17.jsonl"


def test_shape_manifest_path_supports_explicit_placeholders(tmp_path):
    resolved = resolve_shape_manifest_path(
        str(tmp_path / "{host}" / "shapes-{pid}.jsonl"), host="node0", pid=17
    )

    assert resolved == tmp_path / "node0" / "shapes-17.jsonl"
