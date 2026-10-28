#!/usr/bin/env python3
"""Apply the corrected three-class policy to complete visible final answers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from am_deception.evaluation.label_policy import classify


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-id-column", default="case_id")
    parser.add_argument("--answer-column", default="final_response_text")
    parser.add_argument("--strict-label-column", default="strict_label")
    parser.add_argument("--strict-reason-column", default="strict_label_reason")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_csv(input_path)
    judged: list[dict[str, str | int]] = []
    for row in rows:
        case_id = row.get(args.case_id_column, "").strip()
        if not case_id:
            raise ValueError(f"missing {args.case_id_column!r}")
        answer = row.get(args.answer_column, "")
        meta = {"case_id": case_id, "final_response_text": answer}
        strict = {
            "final_label": row.get(args.strict_label_column, ""),
            "strict_label_reason": row.get(args.strict_reason_column, ""),
        }
        label, reason, source = classify(meta, strict)
        judged.append({
            **row,
            "case_id": case_id,
            "final_label": label,
            "final_label_reason": reason,
            "classification_source": source,
            "answer_chars": len(answer.strip()),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in judged:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "case_judgments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(judged)
    counts = Counter(str(row["final_label"]) for row in judged)
    summary = {
        "cases": len(judged),
        "visible_final_answers": sum(bool(str(row.get(args.answer_column, "")).strip()) for row in judged),
        "label_counts": {label: counts[label] for label in ("honest", "dishonest", "invalid")},
        "policy": "complete visible final answer; runner labels are not formal labels",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

