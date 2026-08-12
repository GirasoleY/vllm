# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical scheduler shape manifests for deterministic benchmark audits.

The recorder runs only in an engine-core process when
``VLLM_SCHEDULER_SHAPE_MANIFEST_PATH`` is set. TP workers do not emit files;
each DP engine core does. Path templates support ``{host}`` and ``{pid}``, and
both components are inserted automatically when omitted.
"""

import argparse
import hashlib
import json
import os
import socket
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO

import regex as re

SCHEMA_VERSION = 1
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_component(value: str) -> str:
    return _SAFE_COMPONENT_RE.sub("_", value)


def resolve_shape_manifest_path(
    path_template: str,
    *,
    host: str | None = None,
    pid: int | None = None,
) -> Path:
    """Resolve a per-process scheduler shape manifest path.

    ``path_template`` may contain ``{host}`` and ``{pid}``. Missing host or
    PID placeholders are inserted before the suffix so a shared output
    directory remains safe across nodes and engine restarts.
    """
    resolved_host = _safe_component(host or socket.gethostname())
    resolved_pid = os.getpid() if pid is None else pid
    try:
        expanded = path_template.format(host=resolved_host, pid=resolved_pid)
    except KeyError as exc:
        raise ValueError(
            "Scheduler shape manifest path supports only {host} and {pid} "
            f"placeholders; got unknown placeholder {exc}."
        ) from exc

    path = Path(expanded)
    missing_components = []
    if "{host}" not in path_template:
        missing_components.append(f"host-{resolved_host}")
    if "{pid}" not in path_template:
        missing_components.append(f"pid-{resolved_pid}")
    if missing_components:
        suffix = path.suffix
        stem = path.name[: -len(suffix)] if suffix else path.name
        path = path.with_name(".".join((stem, *missing_components)) + suffix)
    return path


def canonical_shape_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


class SchedulerShapeManifestRecorder:
    """Writes canonical, topology-neutral scheduler step records as JSONL."""

    def __init__(self, path_template: str) -> None:
        self.path = resolve_shape_manifest_path(path_template)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._output: TextIO = self.path.open("x", encoding="utf-8", buffering=1)
        self._step_seq = 0

    def record(self, record: Mapping[str, Any]) -> None:
        output_record = dict(record)
        output_record["schema_version"] = SCHEMA_VERSION
        output_record["step_seq"] = self._step_seq
        self._output.write(canonical_shape_json(output_record) + "\n")
        self._step_seq += 1

    def close(self) -> None:
        self._output.close()


def read_shape_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Shape manifest line {line_number} is not a JSON object."
                )
            records.append(record)
    return records


def normalize_shape_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replace process-generated request IDs with encounter-order aliases."""
    aliases: dict[str, str] = {}

    def alias(request_id: str) -> str:
        if request_id not in aliases:
            aliases[request_id] = f"r{len(aliases)}"
        return aliases[request_id]

    def normalize_queue(queue: Sequence[str]) -> list[str]:
        return [alias(request_id) for request_id in queue]

    normalized_records = []
    first_step_seq = None
    for expected_step_seq, source_record in enumerate(records):
        record = deepcopy(dict(source_record))
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported scheduler shape manifest schema version: "
                f"{record.get('schema_version')!r}."
            )
        source_step_seq = record.get("step_seq")
        if first_step_seq is None:
            if not isinstance(source_step_seq, int):
                raise ValueError(
                    "Scheduler shape manifest step sequence must be an integer."
                )
            first_step_seq = source_step_seq
        wanted_source_step_seq = first_step_seq + expected_step_seq
        if source_step_seq != wanted_source_step_seq:
            raise ValueError(
                "Scheduler shape manifest step sequence must be contiguous; "
                f"expected {wanted_source_step_seq}, got {source_step_seq!r}."
            )
        record["step_seq"] = expected_step_seq

        for queue_field in ("queues_before", "queues_after"):
            queues = record[queue_field]
            for name in ("running", "waiting", "skipped_waiting"):
                queues[name] = normalize_queue(queues[name])
        for scheduled_request in record["scheduled"]:
            scheduled_request["request_id"] = alias(scheduled_request["request_id"])
        record["preempted"] = normalize_queue(record["preempted"])
        normalized_records.append(record)
    return normalized_records


def hash_shape_records(records: Iterable[Mapping[str, Any]]) -> str:
    normalized = normalize_shape_records(records)
    payload = "".join(canonical_shape_json(record) + "\n" for record in normalized)
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_shape_manifest(path: str | Path) -> str:
    return hash_shape_records(read_shape_manifest(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash a scheduler shape manifest after normalizing request IDs."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--normalized-output", type=Path)
    args = parser.parse_args()

    normalized = normalize_shape_records(read_shape_manifest(args.manifest))
    if args.normalized_output is not None:
        args.normalized_output.write_text(
            "".join(canonical_shape_json(record) + "\n" for record in normalized),
            encoding="utf-8",
        )
    print(hash_shape_records(normalized))


if __name__ == "__main__":
    main()
