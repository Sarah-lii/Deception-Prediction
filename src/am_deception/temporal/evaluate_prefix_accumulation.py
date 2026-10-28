#!/usr/bin/env python3
"""Evaluate frozen AM5000 IIMMD checkpoints on prefix-mean hidden states.

This script reuses the historical AM5000 v1f checkpoints. It does not train
new parameters. For each evaluation case it reconstructs prefix windows from
the pre-decision hidden call files:

  prefix_1 = mean(call_0)
  prefix_2 = mean(call_0, call_1)
  ...

The frozen model, standardization, train reference pools, kernel parameters,
and validation threshold are taken from the checkpoint's training shift.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


MODEL_ROOT = Path(".")
SHIFTS = ("t", "t-1", "t-2", "t-3", "t-4", "t-5", "t-6", "t-7")
LABELS = ("honest", "dishonest")
NEG_LABEL = "honest"
POS_LABEL = "dishonest"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_key(case_id: str) -> tuple:
    parts = []
    for part in case_id.replace("-", "_").split("_"):
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts)


class LayerTransformerMap(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers_in: int,
        latent_dim: int = 300,
        d_model: int = 192,
        nhead: int = 4,
        depth: int = 2,
        dropout: float = 0.05,
        pooling: str = "mean",
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "global_attention"}:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.attention_temperature = max(float(attention_temperature), 1e-4)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, num_layers_in, d_model))
        if self.pooling == "global_attention":
            self.layer_logits = nn.Parameter(torch.zeros(num_layers_in, dtype=torch.float32))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        encoded = self.encoder(hidden)
        if self.pooling == "global_attention":
            weights = torch.softmax(self.layer_logits[: encoded.shape[1]] / self.attention_temperature, dim=0)
            pooled = (encoded * weights[None, :, None]).sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)
        return self.out(pooled)


def sq_cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp((a * a).sum(1, keepdim=True) + (b * b).sum(1, keepdim=True).T - 2.0 * (a @ b.T), min=0.0)


class DeepKernel(nn.Module):
    def __init__(self, sigma_init: float, sigma0_init: float, epsilon_init: float = 0.05, bandwidth_log_radius: float | None = None) -> None:
        super().__init__()
        self.bandwidth_log_radius = None if bandwidth_log_radius is None else max(float(bandwidth_log_radius), 1e-3)
        self.register_buffer("log_sigma_center", torch.tensor(math.log(max(sigma_init, 1e-6)), dtype=torch.float32), persistent=False)
        self.register_buffer("log_sigma0_center", torch.tensor(math.log(max(sigma0_init, 1e-6)), dtype=torch.float32), persistent=False)
        self.raw_log_sigma = nn.Parameter(self.log_sigma_center.detach().clone())
        self.raw_log_sigma0 = nn.Parameter(self.log_sigma0_center.detach().clone())
        eps = min(max(epsilon_init, 1e-4), 1 - 1e-4)
        self.epsilon_logit = nn.Parameter(torch.tensor(math.log(eps / (1.0 - eps)), dtype=torch.float32))

    def _bounded(self, raw: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if self.bandwidth_log_radius is None:
            return raw
        radius = self.bandwidth_log_radius
        return center + radius * torch.tanh((raw - center) / radius)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self._bounded(self.raw_log_sigma, self.log_sigma_center)).clamp(min=1e-6)

    @property
    def sigma0(self) -> torch.Tensor:
        return torch.exp(self._bounded(self.raw_log_sigma0, self.log_sigma0_center)).clamp(min=1e-6)

    @property
    def epsilon(self) -> torch.Tensor:
        return torch.sigmoid(self.epsilon_logit).clamp(min=1e-4, max=0.5)

    def forward(self, phi_a: torch.Tensor, phi_b: torch.Tensor, raw_a: torch.Tensor, raw_b: torch.Tensor) -> torch.Tensor:
        learned = (1.0 - self.epsilon) * torch.exp(-sq_cdist(phi_a, phi_b) / self.sigma) + self.epsilon
        anchor = torch.exp(-sq_cdist(raw_a, raw_b) / self.sigma0)
        return learned * anchor


def load_npz_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        keys = ["response_mean", "mean_hidden", "hidden_mean", "features", "hidden", "arr_0", "x"]
        for key in keys + [k for k in archive.files if k not in keys]:
            if key not in archive.files:
                continue
            arr = np.asarray(archive[key])
            if not np.issubdtype(arr.dtype, np.number):
                continue
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
            if arr.ndim == 2 and arr.shape[0] >= 40 and arr.shape[1] == 5120:
                return arr[:40].astype(np.float32, copy=False)
    raise ValueError(f"no [40,5120] numeric matrix in {path}")


def load_selected_hidden_files(feature_npz: Path) -> list[str]:
    with np.load(feature_npz, allow_pickle=False) as archive:
        if "selected_hidden_files" not in archive.files:
            return []
        values = archive["selected_hidden_files"]
    out: list[str] = []
    for item in values.tolist():
        text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = feature_npz.parent / candidate
        if candidate.is_file():
            out.append(str(candidate))
    return out


def load_shift_reference_dataset(
    index_rows: list[dict[str, str]],
    split_by_case: dict[str, str],
    shift: str,
) -> tuple[list[dict], np.ndarray, list[dict]]:
    meta: list[dict] = []
    arrays: list[np.ndarray] = []
    errors: list[dict] = []
    selected = [
        row for row in index_rows
        if row.get("shift") == shift
        and row.get("final_label") in LABELS
        and split_by_case.get(row.get("case_id", "")) == "train"
        and row.get("feature_npz", "")
    ]
    selected.sort(key=lambda row: stable_key(row["case_id"]))
    for row in selected:
        try:
            matrix = load_npz_matrix(Path(row["feature_npz"]))
            arrays.append(matrix)
            meta.append({
                "case_id": row["case_id"],
                "task_type": str(row.get("task_type", "")),
                "task_index": str(row.get("task_index", "")),
                "final_label": row["final_label"],
                "split": "train",
                "train_shift": shift,
                "feature_npz": row["feature_npz"],
            })
        except Exception as exc:
            errors.append({"case_id": row.get("case_id", ""), "shift": shift, "error": repr(exc), "feature_npz": row.get("feature_npz", "")})
    if not arrays:
        return meta, np.empty((0, 40, 5120), dtype=np.float32), errors
    return meta, np.stack(arrays, axis=0).astype(np.float32), errors


def search_hidden_limited(case_id: str, roots: Iterable[Path], max_matches: int = 12, max_dirs: int = 2000) -> list[str]:
    matches: list[str] = []
    visited = 0
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirnames, filenames in os.walk(root):
            visited += 1
            if visited > max_dirs:
                return matches
            current_path = Path(current)
            if case_id not in str(current_path):
                dirnames[:] = [name for name in dirnames if case_id in name or name in {"outputs", "branches", "branch_000", "agent_hidden", "hidden"}]
            for name in filenames:
                if name.startswith("call_") and name.endswith(".npz") and case_id in str(current_path):
                    matches.append(str(current_path / name))
                    if len(matches) >= max_matches:
                        return matches
    return matches


def build_case_audit_map(case_audit_path: Path) -> dict[str, dict[str, str]]:
    if not case_audit_path.is_file():
        return {}
    return {row.get("case_id", ""): row for row in read_csv(case_audit_path) if row.get("case_id")}


def hidden_files_from_case_audit(row: dict, audit_map: dict[str, dict[str, str]]) -> list[str]:
    audit = audit_map.get(row["case_id"], {})
    candidates: list[Path] = []
    for key in ("hidden_dir", "agent_hidden_dir", "hidden_path"):
        value = audit.get(key, "")
        if value:
            candidates.append(Path(value))
    branch_dir = audit.get("branch_dir", "")
    if branch_dir:
        branch = Path(branch_dir)
        candidates.extend([branch / "agent_hidden", branch / "hidden"])
    for candidate in candidates:
        if candidate.is_file() and candidate.name.startswith("call_") and candidate.suffix == ".npz":
            return [str(candidate)]
        if candidate.is_dir():
            files = sorted(
                candidate.glob("call_*.npz"),
                key=lambda path: int(re.search(r"call_(\d+)", path.name).group(1)) if re.search(r"call_(\d+)", path.name) else -1,
            )
            if files:
                return [str(path) for path in files]
    return []


def selected_hidden_files_for_row(row: dict, audit_map: dict[str, dict[str, str]]) -> list[str]:
    feature = row.get("t_feature_npz") or row.get("feature_npz") or ""
    files = load_selected_hidden_files(Path(feature)) if feature else []
    if files:
        return files
    return hidden_files_from_case_audit(row, audit_map)


def labels_for(meta: list[dict], indices: np.ndarray) -> np.ndarray:
    return np.asarray([1 if meta[int(i)]["final_label"] == POS_LABEL else 0 for i in indices], dtype=np.int8)


def auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    ok = np.isfinite(scores)
    y, scores = y[ok], scores[ok]
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    honest = y == 0
    dishonest = y == 1
    honest_acc = float((~pred[honest]).mean()) if honest.any() else float("nan")
    dishonest_acc = float(pred[dishonest].mean()) if dishonest.any() else float("nan")
    return {
        "acc": float((pred == (y == 1)).mean()) if len(y) else float("nan"),
        "bal_acc": float(np.nanmean([honest_acc, dishonest_acc])),
        "honest_acc": honest_acc,
        "dishonest_acc": dishonest_acc,
    }


def best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    valid = scores[np.isfinite(scores)]
    if valid.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    values = np.unique(np.sort(valid))
    if values.size == 1:
        thresholds = values
    else:
        eps = max(1e-12, float(np.ptp(values)) * 1e-9)
        thresholds = np.concatenate(([values[0] - eps], (values[:-1] + values[1:]) / 2.0, [values[-1] + eps]))
    best = (-1.0, float(thresholds[0]))
    for threshold in thresholds:
        bal = threshold_metrics(y, scores, float(threshold))["bal_acc"]
        if bal > best[0]:
            best = (bal, float(threshold))
    return best[1], best[0]


def parse_clean_variants(text: str) -> list[tuple[str, str, int]]:
    variants: list[tuple[str, str, int]] = []
    for item in [part.strip() for part in text.split(",") if part.strip()]:
        match = re.fullmatch(r"(exact|ge)(\d+)", item)
        if not match:
            raise ValueError(f"invalid clean variant: {item}")
        mode, steps = match.group(1), int(match.group(2))
        if steps < 1:
            raise ValueError(f"invalid clean variant steps: {item}")
        variants.append((item, mode, steps))
    return variants


def load_checkpoint(train_shift: str, device: torch.device) -> tuple[nn.Module, DeepKernel, dict, np.ndarray, np.ndarray]:
    ckpt_path = MODEL_ROOT / f"best_model_{train_shift.replace('-', 'm')}.pth"
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(checkpoint["model_config"])
    model = LayerTransformerMap(
        input_dim=int(cfg["input_dim"]),
        num_layers_in=int(cfg["num_layers_in"]),
        latent_dim=int(cfg["latent_dim"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
        pooling=str(cfg.get("pooling", "mean")),
        attention_temperature=float(cfg.get("attention_temperature", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    kernel = DeepKernel(
        sigma_init=float(cfg.get("kernel_sigma_init", checkpoint.get("sigma_init", 1.0))),
        sigma0_init=float(cfg.get("kernel_sigma0_init", checkpoint.get("sigma0_init", 1.0))),
        epsilon_init=0.05,
        bandwidth_log_radius=cfg.get("kernel_log_radius"),
    ).to(device)
    kernel.load_state_dict(checkpoint["kernel_state_dict"])
    mean = checkpoint["standardization"]["mean"].detach().cpu().numpy().astype(np.float32)
    std = checkpoint["standardization"]["std"].detach().cpu().numpy().astype(np.float32)
    return model, kernel, cfg, mean, np.maximum(std, 1e-4)


@torch.no_grad()
def score_batches(
    model: nn.Module,
    kernel: DeepKernel,
    x_std: np.ndarray,
    ref_h: np.ndarray,
    ref_d: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    ref_h_t = torch.from_numpy(ref_h).to(device=device, dtype=torch.float32)
    ref_d_t = torch.from_numpy(ref_d).to(device=device, dtype=torch.float32)
    raw_h = ref_h_t.flatten(1)
    raw_d = ref_d_t.flatten(1)
    phi_h = model(ref_h_t)
    phi_d = model(ref_d_t)
    const_h = kernel(phi_h, phi_h, raw_h, raw_h).mean()
    const_d = kernel(phi_d, phi_d, raw_d, raw_d).mean()
    scores = []
    for start in range(0, len(x_std), batch_size):
        xb = torch.from_numpy(x_std[start : start + batch_size]).to(device=device, dtype=torch.float32)
        raw_b = xb.flatten(1)
        phi_b = model(xb)
        mean_d = kernel(phi_b, phi_d, raw_b, raw_d).mean(dim=1)
        mean_h = kernel(phi_b, phi_h, raw_b, raw_h).mean(dim=1)
        scores.append((2.0 * (mean_d - mean_h) + const_h - const_d).detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32)


def build_t_feature_table(index_rows: list[dict[str, str]], split_by_case: dict[str, str]) -> list[dict]:
    rows = []
    for row in index_rows:
        if row.get("shift") != "t" or row.get("final_label") not in LABELS:
            continue
        case_id = row.get("case_id", "")
        if case_id not in split_by_case:
            continue
        feature = row.get("feature_npz", "")
        if not feature:
            continue
        rows.append({
            "case_id": case_id,
            "task_type": str(row.get("task_type", "")),
            "task_index": str(row.get("task_index", "")),
            "final_label": row["final_label"],
            "split": split_by_case[case_id],
            "t_feature_npz": feature,
        })
    rows.sort(key=lambda item: stable_key(item["case_id"]))
    return rows


def load_prefix_dataset(
    feature_rows: list[dict],
    prefix_len: int,
    max_prefix_available: int,
    emit_features: bool,
    feature_out: Path,
    audit_map: dict[str, dict[str, str]],
) -> tuple[list[dict], np.ndarray, list[dict]]:
    meta: list[dict] = []
    arrays: list[np.ndarray] = []
    errors: list[dict] = []
    for seen, row in enumerate(feature_rows, start=1):
        feature_npz = Path(row["t_feature_npz"])
        try:
            hidden_files = selected_hidden_files_for_row(row, audit_map)
            if len(hidden_files) < prefix_len:
                continue
            hidden_files = hidden_files[:max_prefix_available]
            if len(hidden_files) < prefix_len:
                continue
            matrices = [load_npz_matrix(Path(path)) for path in hidden_files[:prefix_len]]
            matrix = np.stack(matrices, axis=0).mean(axis=0).astype(np.float32)
            out_path = ""
            if emit_features:
                out_file = feature_out / f"prefix_{prefix_len:02d}" / f"type{row['task_type']}" / f"{row['case_id']}.npz"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_file,
                    response_mean=matrix,
                    case_id=np.array(row["case_id"]),
                    task_type=np.array(str(row["task_type"])),
                    final_label=np.array(row["final_label"]),
                    split=np.array(row["split"]),
                    prefix_len=np.array(prefix_len, dtype=np.int32),
                    selected_hidden_files=np.array(hidden_files[:prefix_len]),
                )
                out_path = str(out_file)
            arrays.append(matrix)
            meta.append({**row, "prefix_len": prefix_len, "predecision_call_count": len(hidden_files), "prefix_feature_npz": out_path})
        except Exception as exc:
            errors.append({"case_id": row.get("case_id", ""), "prefix_len": prefix_len, "error": repr(exc), "t_feature_npz": row.get("t_feature_npz", "")})
        if seen % 500 == 0:
            print(json.dumps({"event": "prefix_load_progress", "prefix_len": prefix_len, "seen": seen, "loaded": len(arrays), "errors": len(errors)}), flush=True)
    if not arrays:
        return meta, np.empty((0, 40, 5120), dtype=np.float32), errors
    return meta, np.stack(arrays, axis=0).astype(np.float32), errors


def evaluate_prefixes(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_rows = read_csv(Path(args.index))
    split_rows = read_csv(Path(args.split))
    audit_map = build_case_audit_map(Path(args.case_audit))
    split_by_case = {row["case_id"]: row["split"] for row in split_rows}
    t_rows = build_t_feature_table(index_rows, split_by_case)
    if not t_rows:
        raise RuntimeError("no t rows with split labels were found")

    call_counts = []
    call_errors = []
    for row in t_rows:
        try:
            call_counts.append((row["case_id"], row["split"], row["final_label"], row["task_type"], len(selected_hidden_files_for_row(row, audit_map))))
        except Exception as exc:
            call_errors.append({"case_id": row["case_id"], "error": repr(exc), "t_feature_npz": row["t_feature_npz"]})
    write_csv(out / "hidden_file_read_errors.csv", call_errors)
    call_count_rows = [
        {"case_id": case_id, "split": split, "final_label": label, "task_type": task_type, "predecision_call_count": count}
        for case_id, split, label, task_type, count in call_counts
    ]
    write_csv(out / "predecision_call_counts.csv", call_count_rows)
    eligible_by_prefix = Counter()
    for _, split, label, _, count in call_counts:
        for prefix in range(1, min(args.max_prefix, count) + 1):
            eligible_by_prefix[(prefix, split, label)] += 1

    clean_pool = [
        item for item in call_counts
        if 1 <= item[4] <= args.clean_max_steps and item[1] in {"val", "test"}
    ]
    length_counter = Counter(item[4] for item in clean_pool)
    clean_steps = max(length_counter, key=lambda key: (length_counter[key], key)) if length_counter else 0
    clean_cases = {item[0] for item in clean_pool if item[4] == clean_steps}
    clean_t_rows = [row for row in t_rows if row["case_id"] in clean_cases]
    clean_variants = parse_clean_variants(args.clean_variants) if args.clean_variants else []
    clean_variant_rows: dict[str, list[dict]] = {}
    clean_variant_steps: dict[str, int] = {}
    for name, mode, steps in clean_variants:
        if mode == "exact":
            cases = {item[0] for item in clean_pool if item[4] == steps}
        else:
            cases = {item[0] for item in clean_pool if item[4] >= steps}
        clean_variant_rows[name] = [row for row in t_rows if row["case_id"] in cases]
        clean_variant_steps[name] = steps

    summary_by_shift = {row["shift"]: row for row in read_csv(Path(args.summary))}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    print(json.dumps({
        "event": "start",
        "device": str(device),
        "t_cases": len(t_rows),
        "clean_steps": clean_steps,
        "clean_cases": len(clean_t_rows),
        "clean_variants": {name: {"steps": clean_variant_steps[name], "cases": len(rows)} for name, rows in clean_variant_rows.items()},
    }), flush=True)

    all_rows: list[dict] = []
    case_score_rows: list[dict] = []
    load_error_rows: list[dict] = []
    feature_out = out / "prefix_features"
    bundles = []
    for train_shift in [item.strip() for item in args.train_shifts.split(",") if item.strip()]:
        model, kernel, cfg, mean, std = load_checkpoint(train_shift, device)
        train_ref_meta, train_ref_raw, train_ref_errors = load_shift_reference_dataset(index_rows, split_by_case, train_shift)
        load_error_rows.extend({**err, "train_shift": train_shift, "dataset": "train_reference"} for err in train_ref_errors)
        if len(train_ref_meta) == 0:
            raise RuntimeError(f"no train reference prefix data for {train_shift}")
        train_labels = np.asarray([1 if row["final_label"] == POS_LABEL else 0 for row in train_ref_meta], dtype=np.int8)
        ref_h = ((train_ref_raw[train_labels == 0] - mean) / std).astype(np.float32)
        ref_d = ((train_ref_raw[train_labels == 1] - mean) / std).astype(np.float32)
        if min(len(ref_h), len(ref_d)) < 4:
            raise RuntimeError(f"insufficient train references for {train_shift}: H={len(ref_h)} D={len(ref_d)}")
        threshold = float(summary_by_shift.get(train_shift, {}).get("threshold_from_val", "nan"))
        if not np.isfinite(threshold):
            raise RuntimeError(f"missing validation threshold for {train_shift}")
        bundles.append({
            "train_shift": train_shift,
            "model": model,
            "kernel": kernel,
            "mean": mean,
            "std": std,
            "ref_h": ref_h,
            "ref_d": ref_d,
            "threshold": threshold,
            "train_reference_n": int(len(train_ref_meta)),
            "train_reference_honest": int((train_labels == 0).sum()),
            "train_reference_dishonest": int((train_labels == 1).sum()),
            "model_path": str(MODEL_ROOT / f"best_model_{train_shift.replace('-', 'm')}.pth"),
        })

    if not args.skip_prefix_generalization:
        for prefix_len in range(1, args.max_prefix + 1):
            eval_meta, eval_raw, eval_errors = load_prefix_dataset(
                [row for row in t_rows if row["split"] in {"val", "test"}],
                prefix_len=prefix_len,
                max_prefix_available=args.max_prefix,
                emit_features=args.emit_prefix_features,
                feature_out=feature_out,
                audit_map=audit_map,
            )
            load_error_rows.extend({**err, "dataset": f"eval_prefix_{prefix_len:02d}"} for err in eval_errors)
            if len(eval_meta) == 0:
                continue
            for bundle in bundles:
                train_shift = bundle["train_shift"]
                x_std = ((eval_raw - bundle["mean"]) / bundle["std"]).astype(np.float32)
                scores = score_batches(bundle["model"], bundle["kernel"], x_std, bundle["ref_h"], bundle["ref_d"], device, args.eval_batch)
                for split_name in ("val", "test"):
                    idx = np.asarray([i for i, row in enumerate(eval_meta) if row["split"] == split_name], dtype=np.int64)
                    if len(idx) == 0:
                        continue
                    y = labels_for(eval_meta, idx)
                    s = scores[idx]
                    metrics = threshold_metrics(y, s, bundle["threshold"])
                    th2, best_bal = best_threshold(y, s)
                    row = {
                        "experiment": "prefix_generalization",
                        "train_shift": str(train_shift),
                        "eval_prefix_len": prefix_len,
                        "eval_split": split_name,
                        "n": int(len(idx)),
                        "honest": int((y == 0).sum()),
                        "dishonest": int((y == 1).sum()),
                        "auroc": auc_rank(y, s),
                        "acc_at_train_val_threshold": metrics["acc"],
                        "bal_acc_at_train_val_threshold": metrics["bal_acc"],
                        "honest_acc_at_train_val_threshold": metrics["honest_acc"],
                        "dishonest_acc_at_train_val_threshold": metrics["dishonest_acc"],
                        "best_bal_acc_with_eval_threshold": best_bal,
                        "best_eval_threshold": th2,
                        "threshold_from_train_val": bundle["threshold"],
                        "train_reference_shift": str(train_shift),
                        "train_reference_n": bundle["train_reference_n"],
                        "train_reference_honest": bundle["train_reference_honest"],
                        "train_reference_dishonest": bundle["train_reference_dishonest"],
                        "model_path": bundle["model_path"],
                    }
                    all_rows.append(row)
                    for i, score in zip(idx, s):
                        case_score_rows.append({**eval_meta[int(i)], "experiment": "prefix_generalization", "train_shift": str(train_shift), "eval_prefix_len": prefix_len, "score": float(score), "y_dishonest": int(eval_meta[int(i)]["final_label"] == POS_LABEL)})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if clean_steps:
        for prefix_len in range(1, clean_steps + 1):
            eval_meta, eval_raw, eval_errors = load_prefix_dataset(
                clean_t_rows,
                prefix_len=prefix_len,
                max_prefix_available=clean_steps,
                emit_features=False,
                feature_out=feature_out,
                audit_map=audit_map,
            )
            load_error_rows.extend({**err, "dataset": f"clean_prefix_{prefix_len:02d}"} for err in eval_errors)
            if len(eval_meta) == 0:
                continue
            for bundle in bundles:
                train_shift = bundle["train_shift"]
                x_std = ((eval_raw - bundle["mean"]) / bundle["std"]).astype(np.float32)
                scores = score_batches(bundle["model"], bundle["kernel"], x_std, bundle["ref_h"], bundle["ref_d"], device, args.eval_batch)
                idx = np.arange(len(eval_meta), dtype=np.int64)
                y = labels_for(eval_meta, idx)
                metrics = threshold_metrics(y, scores, bundle["threshold"])
                th2, best_bal = best_threshold(y, scores)
                all_rows.append({
                    "experiment": "clean_same_predecision_length",
                    "train_shift": str(train_shift),
                    "eval_prefix_len": prefix_len,
                    "eval_split": "val_test_clean",
                    "n": int(len(idx)),
                    "honest": int((y == 0).sum()),
                    "dishonest": int((y == 1).sum()),
                    "auroc": auc_rank(y, scores),
                    "acc_at_train_val_threshold": metrics["acc"],
                    "bal_acc_at_train_val_threshold": metrics["bal_acc"],
                    "honest_acc_at_train_val_threshold": metrics["honest_acc"],
                    "dishonest_acc_at_train_val_threshold": metrics["dishonest_acc"],
                    "best_bal_acc_with_eval_threshold": best_bal,
                    "best_eval_threshold": th2,
                    "threshold_from_train_val": bundle["threshold"],
                    "train_reference_shift": str(train_shift),
                    "train_reference_n": bundle["train_reference_n"],
                    "train_reference_honest": bundle["train_reference_honest"],
                    "train_reference_dishonest": bundle["train_reference_dishonest"],
                    "model_path": bundle["model_path"],
                })
                for meta_row, score in zip(eval_meta, scores):
                    case_score_rows.append({**meta_row, "experiment": "clean_same_predecision_length", "train_shift": str(train_shift), "eval_prefix_len": prefix_len, "score": float(score), "y_dishonest": int(meta_row["final_label"] == POS_LABEL)})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for variant_name, rows_for_variant in clean_variant_rows.items():
        variant_steps = clean_variant_steps[variant_name]
        if not rows_for_variant:
            continue
        experiment_name = f"clean_{variant_name}"
        for prefix_len in range(1, variant_steps + 1):
            eval_meta, eval_raw, eval_errors = load_prefix_dataset(
                rows_for_variant,
                prefix_len=prefix_len,
                max_prefix_available=variant_steps,
                emit_features=False,
                feature_out=feature_out,
                audit_map=audit_map,
            )
            load_error_rows.extend({**err, "dataset": f"{experiment_name}_prefix_{prefix_len:02d}"} for err in eval_errors)
            if len(eval_meta) == 0:
                continue
            for bundle in bundles:
                train_shift = bundle["train_shift"]
                x_std = ((eval_raw - bundle["mean"]) / bundle["std"]).astype(np.float32)
                scores = score_batches(bundle["model"], bundle["kernel"], x_std, bundle["ref_h"], bundle["ref_d"], device, args.eval_batch)
                idx = np.arange(len(eval_meta), dtype=np.int64)
                y = labels_for(eval_meta, idx)
                metrics = threshold_metrics(y, scores, bundle["threshold"])
                th2, best_bal = best_threshold(y, scores)
                all_rows.append({
                    "experiment": experiment_name,
                    "train_shift": str(train_shift),
                    "eval_prefix_len": prefix_len,
                    "eval_split": "val_test_clean",
                    "n": int(len(idx)),
                    "honest": int((y == 0).sum()),
                    "dishonest": int((y == 1).sum()),
                    "auroc": auc_rank(y, scores),
                    "acc_at_train_val_threshold": metrics["acc"],
                    "bal_acc_at_train_val_threshold": metrics["bal_acc"],
                    "honest_acc_at_train_val_threshold": metrics["honest_acc"],
                    "dishonest_acc_at_train_val_threshold": metrics["dishonest_acc"],
                    "best_bal_acc_with_eval_threshold": best_bal,
                    "best_eval_threshold": th2,
                    "threshold_from_train_val": bundle["threshold"],
                    "train_reference_shift": str(train_shift),
                    "train_reference_n": bundle["train_reference_n"],
                    "train_reference_honest": bundle["train_reference_honest"],
                    "train_reference_dishonest": bundle["train_reference_dishonest"],
                    "model_path": bundle["model_path"],
                })
                for meta_row, score in zip(eval_meta, scores):
                    case_score_rows.append({**meta_row, "experiment": experiment_name, "train_shift": str(train_shift), "eval_prefix_len": prefix_len, "score": float(score), "y_dishonest": int(meta_row["final_label"] == POS_LABEL)})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    write_csv(out / "prefix_generalization_metrics.csv", all_rows)
    write_csv(out / "prefix_case_scores.csv", case_score_rows)
    write_csv(out / "prefix_load_errors.csv", load_error_rows)
    write_csv(
        out / "prefix_availability_by_split_label.csv",
        [
            {"prefix_len": k[0], "split": k[1], "final_label": k[2], "cases": v}
            for k, v in sorted(eligible_by_prefix.items())
        ],
    )
    write_csv(
        out / "clean_length_distribution.csv",
        [{"predecision_call_count": k, "val_test_cases": v} for k, v in sorted(length_counter.items())],
    )
    manifest = {
        "status": "ok",
        "method": "frozen AM5000 v1f IIMMD checkpoints evaluated on prefix-mean hidden states",
        "index": str(args.index),
        "split": str(args.split),
        "summary": str(args.summary),
        "case_audit": str(args.case_audit),
        "model_root": str(MODEL_ROOT),
        "output": str(out),
        "max_prefix": args.max_prefix,
        "train_shifts": [item.strip() for item in args.train_shifts.split(",") if item.strip()],
        "t_cases_with_split": len(t_rows),
        "clean_definition": "val/test cases with a clear finite pre-decision call count; choose the most frequent count <= clean_max_steps, then evaluate prefix 1..that count",
        "clean_steps": clean_steps,
        "clean_cases": len(clean_t_rows),
        "clean_variants": {name: {"steps": clean_variant_steps[name], "cases": len(rows)} for name, rows in clean_variant_rows.items()},
        "skip_prefix_generalization": bool(args.skip_prefix_generalization),
        "rows": len(all_rows),
        "case_score_rows": len(case_score_rows),
        "load_error_rows": len(load_error_rows),
        "device": str(device),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "manifest": manifest}, ensure_ascii=False), flush=True)
    return manifest


def inventory(args: argparse.Namespace) -> dict:
    paths = {
        "index": Path(args.index),
        "case_audit": Path(args.case_audit),
        "split": Path(args.split),
        "summary": Path(args.summary),
        "model_root": MODEL_ROOT,
    }
    result: dict[str, object] = {
        "status": "ok",
        "paths": {name: str(path) for name, path in paths.items()},
        "exists": {name: path.exists() for name, path in paths.items()},
        "model_files": {},
        "sample_t_npz_keys": [],
        "sample_selected_hidden_files": [],
        "sample_selected_calls": [],
        "case_audit_header": [],
        "case_audit_sample": {},
        "selected_hidden_file_length_histogram": {},
        "selected_call_length_histogram": {},
        "fallback_hidden_file_length_histogram": {},
        "case_hidden_search_samples": [],
    }
    for shift in SHIFTS:
        path = MODEL_ROOT / f"best_model_{shift.replace('-', 'm')}.pth"
        result["model_files"][shift] = {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}
    if Path(args.index).is_file():
        rows = read_csv(Path(args.index))
        audit_map = build_case_audit_map(Path(args.case_audit))
        result["index_rows"] = len(rows)
        t_rows = [row for row in rows if row.get("shift") == "t" and row.get("feature_npz", "")]
        result["t_rows"] = len(t_rows)
        hidden_len_counter: Counter[int] = Counter()
        call_len_counter: Counter[int] = Counter()
        fallback_len_counter: Counter[int] = Counter()
        sample_cases: list[dict] = []
        for row in t_rows[:20]:
            p = Path(row["feature_npz"])
            if p.is_file():
                with np.load(p, allow_pickle=False) as archive:
                    result["sample_t_npz_keys"] = list(archive.files)
                    if "selected_calls" in archive.files:
                        result["sample_selected_calls"] = np.asarray(archive["selected_calls"]).astype(int).tolist()
                result["sample_selected_hidden_files"] = load_selected_hidden_files(p)[:5]
                break
        for row in t_rows:
            p = Path(row["feature_npz"])
            if not p.is_file():
                continue
            try:
                with np.load(p, allow_pickle=False) as archive:
                    files_len = len(archive["selected_hidden_files"]) if "selected_hidden_files" in archive.files else 0
                    calls_len = len(archive["selected_calls"]) if "selected_calls" in archive.files else 0
                hidden_len_counter[files_len] += 1
                call_len_counter[calls_len] += 1
                fallback_len_counter[len(selected_hidden_files_for_row(row, audit_map))] += 1
                if len(sample_cases) < 5 and calls_len:
                    sample_cases.append({"case_id": row["case_id"], "feature_npz": row["feature_npz"], "selected_calls_len": calls_len, "selected_hidden_files_len": files_len, "fallback_hidden_files_len": len(selected_hidden_files_for_row(row, audit_map))})
            except Exception:
                continue
        result["selected_hidden_file_length_histogram"] = dict(sorted(hidden_len_counter.items()))
        result["selected_call_length_histogram"] = dict(sorted(call_len_counter.items()))
        result["fallback_hidden_file_length_histogram"] = dict(sorted(fallback_len_counter.items()))
        result["case_hidden_search_samples"] = sample_cases
    if Path(args.split).is_file():
        result["split_rows"] = len(read_csv(Path(args.split)))
    if Path(args.case_audit).is_file():
        audit_rows = read_csv(Path(args.case_audit))
        result["case_audit_rows"] = len(audit_rows)
        if audit_rows:
            result["case_audit_header"] = list(audit_rows[0].keys())
            result["case_audit_sample"] = {key: audit_rows[0].get(key, "") for key in list(audit_rows[0].keys())[:30]}
    if Path(args.summary).is_file():
        result["summary_rows"] = len(read_csv(Path(args.summary)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "inventory", **result}, ensure_ascii=False), flush=True)
    missing = [name for name, ok in result["exists"].items() if not ok]
    missing_models = [shift for shift, item in result["model_files"].items() if not item["exists"]]
    if missing or missing_models:
        raise SystemExit(5)
    return result


def main() -> None:
    global MODEL_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inventory", "evaluate"], default="evaluate")
    parser.add_argument("--index", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--case-audit", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-shifts", default=",".join(SHIFTS))
    parser.add_argument("--max-prefix", type=int, default=20)
    parser.add_argument("--clean-max-steps", type=int, default=20)
    parser.add_argument("--clean-variants", default="")
    parser.add_argument("--skip-prefix-generalization", action="store_true")
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--emit-prefix-features", action="store_true")
    args = parser.parse_args()
    MODEL_ROOT = Path(args.model_root)
    if args.mode == "inventory":
        inventory(args)
    else:
        evaluate_prefixes(args)


if __name__ == "__main__":
    main()
