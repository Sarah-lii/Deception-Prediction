#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import os
import sys
import time
import traceback
import requests
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


def import_base(prework_dir: Path, module_name: str):
    sys.path.insert(0, str(prework_dir))
    return importlib.import_module(module_name)


def parse_specs(path: Path) -> list[tuple[str, str, float, int, int, str]]:
    specs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(":", 5)
        if len(parts) == 4:
            case_id, task_type, task_index, short = parts
            specs.append((case_id, "none", 0.0, int(task_type), int(task_index), short))
        else:
            case_id, method, alpha, task_type, task_index, short = parts
            specs.append((case_id, method, float(alpha), int(task_type), int(task_index), short))
    return specs


def rows_in_csv(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8", errors="ignore", newline="") as f:
        return max(0, sum(1 for _ in csv.DictReader(f)))


def save_direct_index(out_dir: Path, rows: list[dict[str, Any]], base: Any) -> None:
    fields = [
        "case_id",
        "task_type",
        "task_index",
        "branch_index",
        "status",
        "terminal_error_type",
        "label",
        "semantic_label",
        "label_reason",
        "first_decisive_step",
        "decisive_execution_status",
        "trace_decisive_label",
        "trace_decisive_semantic_label",
        "trace_decisive_reason",
        "trace_decisive_step",
        "final_response_mode",
        "final_response_available",
        "normal_final_delivered",
        "final_delivered",
        "final_response_text",
        "artifact_written",
        "artifacts",
        "action_steps_total",
        "continuation_steps_run",
        "parse_error_streak",
        "tool_error_streak",
        "other_error_streak",
        "max_steps_final_generated",
        "max_steps_final_error",
        "stop_error_kind",
        "freeze_reason",
        "prefix_hash",
        "elapsed_sec",
        "branch_dir",
        "log",
        "hidden_dir",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "decision_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            paths = row.get("paths", {})
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "task_type": row.get("task_type"),
                    "task_index": row.get("task_index"),
                    "branch_index": row.get("branch_index", 0),
                    "status": row.get("status"),
                    "terminal_error_type": row.get("terminal_error_type"),
                    "label": row.get("label"),
                    "semantic_label": row.get("semantic_label"),
                    "label_reason": row.get("label_reason"),
                    "first_decisive_step": row.get("first_decisive_step"),
                    "decisive_execution_status": row.get("decisive_execution_status"),
                    "trace_decisive_label": row.get("trace_decisive_label"),
                    "trace_decisive_semantic_label": row.get("trace_decisive_semantic_label"),
                    "trace_decisive_reason": row.get("trace_decisive_reason"),
                    "trace_decisive_step": row.get("trace_decisive_step"),
                    "final_response_mode": row.get("final_response_mode"),
                    "final_response_available": row.get("final_response_available"),
                    "normal_final_delivered": row.get("normal_final_delivered"),
                    "final_delivered": row.get("final_delivered"),
                    "final_response_text": row.get("final_response_text"),
                    "artifact_written": row.get("artifact_written"),
                    "artifacts": json.dumps(row.get("artifacts", []), ensure_ascii=False),
                    "action_steps_total": row.get("action_steps_total"),
                    "continuation_steps_run": row.get("continuation_steps_run"),
                    "parse_error_streak": row.get("parse_error_streak"),
                    "tool_error_streak": row.get("tool_error_streak"),
                    "other_error_streak": row.get("other_error_streak"),
                    "max_steps_final_generated": row.get("max_steps_final_generated"),
                    "max_steps_final_error": row.get("max_steps_final_error"),
                    "stop_error_kind": row.get("stop_error_kind"),
                    "freeze_reason": row.get("freeze_reason", "direct_full_run_no_prefix"),
                    "prefix_hash": "",
                    "elapsed_sec": row.get("elapsed_sec"),
                    "branch_dir": paths.get("branch_dir"),
                    "log": paths.get("log"),
                    "hidden_dir": paths.get("hidden_dir"),
                }
            )


def run_direct_case(args: argparse.Namespace, spec: tuple[str, int, int, str], base: Any) -> dict[str, Any]:
    case_id, steering_method, steering_alpha, task_type, task_index, short = spec
    run_tag = f"qwen3_{short}_{args.run_suffix}"
    run_out = Path(args.output_root) / "outputs" / run_tag
    decision = run_out / "decision_index.csv"
    if rows_in_csv(decision) >= 1:
        return {"case_id": case_id, "status": "skip_existing", "run_tag": run_tag}

    case_dir = run_out / case_id
    branch_dir = case_dir / "branches" / "branch_000"
    branch_work_dir = branch_dir / "workspace"
    branch_dir.mkdir(parents=True, exist_ok=True)
    log_path = branch_dir / "agent_log.log"
    session_id = f"{case_id}__direct_full_run"

    task = base.load_task(Path(args.repo_dir), task_type, task_index)
    instruction, attached, _input_dir, work_dir = base.prepare_original_task_files(task, case_dir)

    agent = base.make_agent(task_type)
    started = time.time()
    status = "ok"
    error = None
    terminal_error_type = None
    buf = io.StringIO()

    if args.control_url:
        payload = {
            "active": True,
            "session_id": session_id,
            "hidden_dir": str(log_path.parent / "agent_hidden"),
            "log_path": str(log_path),
        }
        if steering_method != "none" and float(steering_alpha) != 0.0:
            payload["steering"] = {
                "active": True,
                "method": steering_method,
                "alpha": float(steering_alpha),
                "vector_path": args.steering_vector_path,
            }
        requests.post(args.control_url, json=payload, timeout=10).raise_for_status()
    try:
        with base.pushd(work_dir), redirect_stdout(buf), redirect_stderr(buf):
            generator = agent.run(
                task=instruction,
                additional_args={"attached_path_or_url": attached},
                reset=True,
                stream=True,
                max_steps=args.max_steps,
            )
            try:
                for _event in generator:
                    pass
            finally:
                try:
                    generator.close()
                except Exception:
                    pass
    except Exception as exc:
        status = "error"
        error = traceback.format_exc()
        terminal_error_type = type(exc).__name__
    finally:
        base.notify_hidden_server(args.control_url, log_path, active=False, session_id=session_id)

    log_path.write_text(base.clean_log(buf.getvalue()), encoding="utf-8")
    artifacts = base.written_artifacts(work_dir)
    normal_final_delivered = base.final_was_delivered(agent)
    max_final_text = base.max_steps_final_text(agent)
    final_text = base.final_answer_text(agent) or max_final_text
    if max_final_text and not normal_final_delivered:
        final_response_mode = "max_steps_final_answer"
    elif normal_final_delivered:
        final_response_mode = "normal_final_answer"
    else:
        final_response_mode = "no_final_response"

    label, reason, decisive_step = ("unknown", "direct runner error before semantic classification", None)
    if status == "ok":
        label, reason, decisive_step = base.classify_trajectory(task, agent, work_dir)

    meta = {
        "status": status,
        "error": error,
        "terminal_error_type": terminal_error_type,
        "case_id": case_id,
        "task_type": task_type,
        "task_index": task_index,
        "branch_index": 0,
        "label": label,
        "semantic_label": base.binary_semantic_label(label),
        "label_reason": reason,
        "first_decisive_step": decisive_step,
        "decisive_execution_status": base.decisive_execution_status(agent, decisive_step),
        "trace_decisive_label": label,
        "trace_decisive_semantic_label": base.binary_semantic_label(label),
        "trace_decisive_reason": reason,
        "trace_decisive_step": decisive_step,
        "final_response_mode": final_response_mode,
        "final_response_available": final_response_mode != "no_final_response",
        "normal_final_delivered": normal_final_delivered,
        "final_delivered": final_response_mode != "no_final_response",
        "final_response_text": final_text,
        "artifact_written": bool(artifacts),
        "artifacts": artifacts,
        "elapsed_sec": round(time.time() - started, 3),
        "action_steps_total": base.action_count(agent),
        "continuation_steps_run": base.action_count(agent),
        "parse_error_streak": "",
        "tool_error_streak": "",
        "other_error_streak": "",
        "max_steps_final_generated": final_response_mode == "max_steps_final_answer",
        "max_steps_final_error": "",
        "stop_error_kind": "direct_full_run",
        "freeze_reason": "direct_full_run_no_prefix",
        "paths": {
            "branch_dir": str(branch_dir),
            "workspace": str(work_dir),
            "log": str(log_path),
            "hidden_dir": str(branch_dir / "agent_hidden"),
        },
        "protocol": "direct_full_run_no_frozen_prefix",
    }
    (branch_dir / "branch_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=base.json_default), encoding="utf-8")
    base.save_agent_snapshot(agent, branch_dir / "snapshot", meta)
    save_direct_index(run_out, [meta], base)
    return {"case_id": case_id, "status": status, "run_tag": run_tag, "label": label, "final_response_mode": final_response_mode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--prework-dir", required=True)
    parser.add_argument("--adapter-module", default="agent_runtime_adapter")
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-suffix", default="gated_raw_caa")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--control-url", default=os.environ.get("LOCAL_AGENT_SERVER_CONTROL_URL", ""))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--steering-method", default="none")
    parser.add_argument("--steering-alpha", type=float, default=0.0)
    parser.add_argument("--steering-vector-path", default="")
    args = parser.parse_args()

    base = import_base(Path(args.prework_dir), args.adapter_module)
    status_path = Path(args.status_file)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for spec in parse_specs(Path(args.spec_file)):
        case_id = spec[0]
        started = time.time()
        try:
            result = run_direct_case(args, spec, base)
            line = {
                "time": time.strftime("%F_%T"),
                "case_id": case_id,
                "status": result.get("status", "ok"),
                "run_tag": result.get("run_tag", ""),
                "label": result.get("label", ""),
                "final_response_mode": result.get("final_response_mode", ""),
                "elapsed_sec": round(time.time() - started, 3),
            }
            code = "OK" if result.get("status") in {"ok", "skip_existing"} else "FAILED_DIRECT"
        except Exception:
            line = {
                "time": time.strftime("%F_%T"),
                "case_id": case_id,
                "status": "FAILED_EXCEPTION",
                "run_tag": "",
                "label": "",
                "final_response_mode": "",
                "elapsed_sec": round(time.time() - started, 3),
                "error": traceback.format_exc(),
            }
            code = "FAILED_EXCEPTION"
        with status_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"code": code, **line}, ensure_ascii=False) + "\n")
        print(json.dumps({"code": code, **line}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
