#!/usr/bin/env python3
"""Prepare detector-gated Raw CAA intervention specifications."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row.get("task_type", 0)), int(row.get("task_index", 0)), row["case_id"]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Per-case detector score CSV.")
    parser.add_argument("--vectors", required=True, help="NPZ containing layer-wise CAA directions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-shift", default="t")
    parser.add_argument("--direction-shift", default="t-2")
    parser.add_argument("--vector-key", default="raw_caa_all_layer")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--expected-cases", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_path = Path(args.scores)
    vectors_path = Path(args.vectors)
    output_dir = Path(args.output_dir)
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    for path in (scores_path, vectors_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = [
        dict(row)
        for row in read_csv(scores_path)
        if row.get("shift") == args.score_shift and row.get("split") == args.split
    ]
    rows.sort(key=stable_key)
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("score rows are not unique by case_id")
    if args.expected_cases is not None and len(rows) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} cases, found {len(rows)}")

    for row in rows:
        score = float(row["score"])
        row["classifier_shift"] = args.score_shift
        row["classifier_score"] = f"{score:.10g}"
        row["classifier_threshold"] = f"{args.threshold:.10g}"
        row["classifier_pred_label"] = "dishonest" if score >= args.threshold else "honest"
        row["gate_on"] = "1" if score >= args.threshold else "0"
    gated = [row for row in rows if row["gate_on"] == "1"]

    with np.load(vectors_path, allow_pickle=False) as archive:
        if args.vector_key not in archive.files:
            raise KeyError(f"{args.vector_key!r} not found in {vectors_path}")
        vectors = np.asarray(archive[args.vector_key])
    expected_shape = (args.layers, args.hidden_size)
    if vectors.shape != expected_shape or not np.all(np.isfinite(vectors)):
        raise RuntimeError(f"expected finite vector array {expected_shape}, found {vectors.shape}")

    state_dir = output_dir / "state"
    specs_dir = state_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    copied_vectors = state_dir / "steering_vectors.npz"
    shutil.copy2(vectors_path, copied_vectors)
    if sha256(vectors_path) != sha256(copied_vectors):
        raise RuntimeError("steering-vector checksum mismatch")

    shard_counts: list[int] = []
    for shard_id in range(args.shards):
        lines = []
        for row in gated[shard_id :: args.shards]:
            short = f"type{row.get('task_type', '')}_idx{row.get('task_index', '')}_gated_caa"
            lines.append(
                f"{row['case_id']}:{args.vector_key}:{args.alpha}:"
                f"{row.get('task_type', '')}:{row.get('task_index', '')}:{short}"
            )
        (specs_dir / f"shard_{shard_id:02d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        shard_counts.append(len(lines))

    gate_fields = [
        "case_id", "task_type", "task_index", "final_label", "classifier_shift",
        "classifier_score", "classifier_threshold", "classifier_pred_label", "gate_on",
    ]
    write_csv(state_dir / "gate_manifest.csv", rows, gate_fields)
    manifest = {
        "score_source": str(scores_path),
        "score_sha256": sha256(scores_path),
        "vector_source": str(vectors_path),
        "vector_sha256": sha256(vectors_path),
        "score_shift": args.score_shift,
        "direction_shift": args.direction_shift,
        "threshold": args.threshold,
        "alpha": args.alpha,
        "vector_key": args.vector_key,
        "vector_shape": list(vectors.shape),
        "cases": len(rows),
        "intervention_cases": len(gated),
        "no_operation_cases": len(rows) - len(gated),
        "prediction_counts": dict(Counter(row["classifier_pred_label"] for row in rows)),
        "shards": args.shards,
        "shard_counts": shard_counts,
        "injection": "current last token at the layers represented by the vector array",
    }
    (state_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

