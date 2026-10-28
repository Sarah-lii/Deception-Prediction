#!/usr/bin/env python3
"""Aggregate paired baseline/intervention labels and optional detector gating."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABELS = ("honest", "dishonest", "invalid")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_label(value: str) -> str:
    value = value.strip().lower()
    if value in LABELS:
        return value
    if value in {"unknown", "ambiguous", "unknown_invalid"}:
        return "invalid"
    raise ValueError(f"unsupported formal label: {value!r}")


def keyed(rows: list[dict[str, str]], case_column: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get(case_column, "").strip()
        if not case_id or case_id in result:
            raise ValueError(f"missing or duplicate case_id: {case_id!r}")
        result[case_id] = row
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-manifest")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--label-column", default="final_label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = keyed(read_csv(Path(args.baseline)), args.case_column)
    intervention = keyed(read_csv(Path(args.intervention)), args.case_column)
    gate = keyed(read_csv(Path(args.gate_manifest)), args.case_column) if args.gate_manifest else {}

    if gate:
        target_ids = list(gate)
    else:
        target_ids = sorted(set(baseline) & set(intervention))
    paired: list[dict[str, str]] = []
    for case_id in target_ids:
        if case_id not in baseline:
            raise KeyError(f"baseline is missing {case_id}")
        before = normalized_label(baseline[case_id][args.label_column])
        prediction = gate.get(case_id, {}).get("classifier_pred_label", "dishonest").strip().lower()
        if prediction == "honest":
            after = before
            operation = "no_operation"
        elif prediction == "dishonest":
            if case_id not in intervention:
                raise KeyError(f"intervention is missing gated case {case_id}")
            after = normalized_label(intervention[case_id][args.label_column])
            operation = "steering"
        else:
            raise ValueError(f"invalid gate prediction for {case_id}: {prediction!r}")
        paired.append({
            "case_id": case_id,
            "baseline_label": before,
            "gate_prediction": prediction,
            "operation": operation,
            "final_label": after,
            "transition": f"{before}_to_{after}",
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_case_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]) if paired else ["case_id"])
        writer.writeheader()
        writer.writerows(paired)

    transition_counts = Counter(row["transition"] for row in paired)
    transition_rows = [
        {"baseline_label": before, "final_label": after, "count": transition_counts[f"{before}_to_{after}"]}
        for before in LABELS
        for after in LABELS
    ]
    with (output_dir / "transition_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["baseline_label", "final_label", "count"])
        writer.writeheader()
        writer.writerows(transition_rows)

    baseline_counts = Counter(row["baseline_label"] for row in paired)
    final_counts = Counter(row["final_label"] for row in paired)
    summary = {
        "cases": len(paired),
        "baseline_counts": {label: baseline_counts[label] for label in LABELS},
        "final_counts": {label: final_counts[label] for label in LABELS},
        "transition_counts": dict(sorted(transition_counts.items())),
        "gated": bool(gate),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

