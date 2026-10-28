#!/usr/bin/env python3
"""Validate the public AM5000 task release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "am5000"
PRIVATE_PATTERNS = re.compile(
    r"/Users/|/mnt/shared-storage|/sharedata/|sarah|xieyuejin|pjlab|safevlagent|"
    r"liyu|gpfs|ai4good1|BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key|password\s*[:=]",
    re.IGNORECASE,
)
BASE_FIELDS = {"task_index", "task_type", "instruction", "topic", "tool"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    expected_names = {f"task_{task_type}.json" for task_type in range(1, 6)}
    actual_names = {path.name for path in DATASET.glob("task_*.json")}
    if actual_names != expected_names:
        raise SystemExit(f"unexpected dataset files: {sorted(actual_names ^ expected_names)}")

    all_ids: set[int] = set()
    total = 0
    for entry in manifest["files"]:
        path = DATASET / entry["path"]
        raw = path.read_bytes()
        if len(raw) != entry["bytes"]:
            raise SystemExit(f"byte count mismatch: {path.name}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"sha256 mismatch: {path.name}")
        text = raw.decode("utf-8")
        if PRIVATE_PATTERNS.search(text):
            raise SystemExit(f"private environment string found: {path.name}")

        rows = json.loads(text)
        if len(rows) != entry["records"]:
            raise SystemExit(f"record count mismatch: {path.name}")
        task_type = int(entry["task_type"])
        ids: list[int] = []
        for row in rows:
            if not BASE_FIELDS.issubset(row):
                raise SystemExit(f"missing required field in {path.name}")
            if int(row["task_type"]) != task_type:
                raise SystemExit(f"task type mismatch in {path.name}")
            task_index = int(row["task_index"])
            if task_index in all_ids:
                raise SystemExit(f"duplicate task_index: {task_index}")
            if not isinstance(row["instruction"], str) or not row["instruction"].strip():
                raise SystemExit(f"empty instruction: {task_index}")
            all_ids.add(task_index)
            ids.append(task_index)

        if min(ids) != entry["min_task_index"] or max(ids) != entry["max_task_index"]:
            raise SystemExit(f"identifier range mismatch: {path.name}")
        total += len(rows)

    if total != manifest["total_tasks"] or len(all_ids) != manifest["total_tasks"]:
        raise SystemExit("total task count mismatch")
    print(json.dumps({"status": "ok", "dataset": "AM5000", "tasks": total, "types": 5}, indent=2))


if __name__ == "__main__":
    main()
