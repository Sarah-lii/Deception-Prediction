#!/usr/bin/env python3
"""IIMMD-style deep-kernel MMD evaluation for AM hidden states.

The script trains a small neural feature map phi(g) without a 0/1 classifier.
It maximizes an MMD test-power proxy between dishonest and honest hidden-state
distributions, then scores held-out cases by MMD distance to the train
reference pools.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


LABELS = ("honest", "dishonest")
SHIFTS = ("t", "t-1", "t-2", "t-3", "t-4", "t-5", "t-6", "t-7")
POS_LABEL = "dishonest"
NEG_LABEL = "honest"


@dataclass
class SplitConfig:
    train_total: int = 1500
    test_total: int = 500
    seed: int = 20260819
    balanced_labels: bool = True


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def auc_rank(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=np.int8)
    s = np.asarray(list(scores), dtype=np.float64)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    pred = scores >= threshold
    acc = float((pred == (y == 1)).mean()) if len(y) else float("nan")
    h_mask = y == 0
    d_mask = y == 1
    h_acc = float((~pred[h_mask]).mean()) if h_mask.any() else float("nan")
    d_acc = float(pred[d_mask].mean()) if d_mask.any() else float("nan")
    bal = float(np.nanmean([h_acc, d_acc]))
    return acc, bal, h_acc, d_acc


def best_val_threshold(y_val: np.ndarray, s_val: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(s_val)
    y = y_val[ok]
    s = s_val[ok]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    scores = np.unique(np.sort(s))
    if len(scores) == 1:
        thresholds = scores
    else:
        mids = (scores[:-1] + scores[1:]) / 2.0
        eps = max(1e-12, float(np.ptp(scores)) * 1e-9)
        thresholds = np.concatenate(([scores[0] - eps], mids, [scores[-1] + eps]))
    best_bal = -1.0
    best_th = float(thresholds[0])
    for th in thresholds:
        _, bal, _, _ = accuracy_at_threshold(y, s, float(th))
        if bal > best_bal:
            best_bal = bal
            best_th = float(th)
    return best_th, float(best_bal)


def select_metric_value(row: dict, metric: str) -> float:
    if metric == "val_auc":
        value = row.get("val_auc", float("nan"))
    elif metric == "val_direction_free_auc":
        value = row.get("val_direction_free_auc", float("nan"))
    elif metric == "val_accuracy":
        value = row.get("val_accuracy_at_best_bal_threshold", float("nan"))
    elif metric == "val_balanced_accuracy":
        value = row.get("val_balanced_accuracy_at_best_bal_threshold", float("nan"))
    elif metric == "val_composite_auc_balacc":
        auc = row.get("val_direction_free_auc", float("nan"))
        bal = row.get("val_balanced_accuracy_at_best_bal_threshold", float("nan"))
        value = 0.5 * float(auc) + 0.5 * float(bal)
    elif metric == "val_composite_raw_auc_balacc":
        auc = row.get("val_auc", float("nan"))
        bal = row.get("val_balanced_accuracy_at_best_bal_threshold", float("nan"))
        value = 0.5 * float(auc) + 0.5 * float(bal)
    elif metric == "val_macro_type_balanced_accuracy":
        value = row.get("val_macro_type_balanced_accuracy_at_global_threshold", float("nan"))
    elif metric == "val_worst_type_balanced_accuracy":
        value = row.get("val_worst_type_balanced_accuracy_at_global_threshold", float("nan"))
    elif metric == "val_composite_global_macro_balacc":
        global_bal = row.get("val_balanced_accuracy_at_best_bal_threshold", float("nan"))
        macro_bal = row.get("val_macro_type_balanced_accuracy_at_global_threshold", float("nan"))
        value = 0.5 * float(global_bal) + 0.5 * float(macro_bal)
    else:
        raise ValueError(f"unknown selection metric: {metric}")
    try:
        return float(value)
    except Exception:
        return float("nan")


def stable_case_key(case_id: str) -> tuple:
    pieces = []
    for part in case_id.replace("-", "_").split("_"):
        if part.isdigit():
            pieces.append((0, int(part)))
        else:
            pieces.append((1, part))
    return tuple(pieces)


def allocate_quota(group_sizes: dict[str, int], quota: int, rng: random.Random) -> dict[str, int]:
    quota = min(quota, sum(group_sizes.values()))
    if quota <= 0:
        return {k: 0 for k in group_sizes}
    total = sum(group_sizes.values())
    raw = {k: quota * v / total for k, v in group_sizes.items()}
    out = {k: min(group_sizes[k], int(math.floor(raw[k]))) for k in group_sizes}
    remaining = quota - sum(out.values())
    order = sorted(
        group_sizes,
        key=lambda k: (raw[k] - math.floor(raw[k]), group_sizes[k], rng.random()),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for key in order:
            if out[key] < group_sizes[key]:
                out[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return out


def build_case_split(rows: list[dict[str, str]], cfg: SplitConfig) -> list[dict[str, str]]:
    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        label = row.get("final_label", "")
        if label not in LABELS:
            continue
        case_id = row["case_id"]
        if case_id not in by_case:
            by_case[case_id] = {
                "case_id": case_id,
                "task_type": str(row["task_type"]),
                "final_label": label,
                "task_index": str(row.get("task_index", "")),
            }
    by_label_type: dict[str, dict[str, list[dict[str, str]]]] = {
        label: defaultdict(list) for label in LABELS
    }
    for row in by_case.values():
        by_label_type[row["final_label"]][row["task_type"]].append(row)

    rng = random.Random(cfg.seed)
    for label in LABELS:
        for task_type in by_label_type[label]:
            by_label_type[label][task_type].sort(key=lambda r: stable_case_key(r["case_id"]))
            rng.shuffle(by_label_type[label][task_type])

    if cfg.balanced_labels:
        train_label_quota = {label: cfg.train_total // 2 for label in LABELS}
        test_label_quota = {label: cfg.test_total // 2 for label in LABELS}
    else:
        label_sizes = {label: sum(len(v) for v in by_label_type[label].values()) for label in LABELS}
        train_label_quota = allocate_quota(label_sizes, cfg.train_total, rng)
        remaining_sizes = {label: label_sizes[label] - train_label_quota[label] for label in LABELS}
        test_label_quota = allocate_quota(remaining_sizes, cfg.test_total, rng)

    split_rows = []
    for label in LABELS:
        group_sizes = {task_type: len(items) for task_type, items in by_label_type[label].items()}
        train_counts = allocate_quota(group_sizes, train_label_quota[label], rng)
        remaining_sizes = {k: group_sizes[k] - train_counts.get(k, 0) for k in group_sizes}
        test_counts = allocate_quota(remaining_sizes, test_label_quota[label], rng)
        for task_type, items in by_label_type[label].items():
            n_train = train_counts.get(task_type, 0)
            n_test = test_counts.get(task_type, 0)
            for split, selected in (
                ("train", items[:n_train]),
                ("test", items[n_train : n_train + n_test]),
                ("val", items[n_train + n_test :]),
            ):
                for row in selected:
                    split_rows.append({**row, "split": split})
    split_rows.sort(key=lambda r: (r["split"], r["final_label"], r["task_type"], stable_case_key(r["case_id"])))
    return split_rows


def load_npz_feature(path: str, max_layers: int = 40) -> np.ndarray:
    with np.load(path) as z:
        keys = ["response_mean", "mean_hidden", "hidden_mean", "features", "hidden", "arr_0", "x"]
        for key in keys + [k for k in z.files if k not in keys]:
            if key not in z.files:
                continue
            arr = np.asarray(z[key])
            if not np.issubdtype(arr.dtype, np.number):
                continue
            if arr.ndim == 2 and arr.shape[0] >= 8:
                arr = arr[:max_layers]
                return arr.astype(np.float32, copy=False)
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
                if arr.ndim == 2 and arr.shape[0] >= 8:
                    arr = arr[:max_layers]
                    return arr.astype(np.float32, copy=False)
    raise ValueError(f"no usable layer matrix in {path}")


def resolve_feature_path(path: str) -> str:
    # Missing feature paths fail explicitly instead of being silently remapped.
    return path


def load_shift_matrix(
    rows: list[dict[str, str]],
    split_by_case: dict[str, str],
    shift: str,
    max_layers: int,
) -> tuple[list[dict[str, str]], np.ndarray, list[dict[str, str]]]:
    selected = [
        row for row in rows
        if row.get("shift") == shift
        and row.get("final_label") in LABELS
        and row.get("case_id") in split_by_case
    ]
    meta = []
    arrays = []
    errors = []
    for i, row in enumerate(selected, start=1):
        original_path = row.get("feature_npz", "")
        path = resolve_feature_path(original_path)
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            arr = load_npz_feature(path, max_layers=max_layers)
            meta.append({
                "case_id": row["case_id"],
                "task_type": str(row["task_type"]),
                "task_index": str(row.get("task_index", "")),
                "final_label": row["final_label"],
                "shift": shift,
                "split": split_by_case[row["case_id"]],
                "feature_npz": path,
                "source_feature_npz": original_path,
            })
            arrays.append(arr)
        except Exception as exc:
            errors.append({
                "case_id": row.get("case_id", ""),
                "task_type": str(row.get("task_type", "")),
                "final_label": row.get("final_label", ""),
                "shift": shift,
                "feature_npz": path,
                "source_feature_npz": original_path,
                "error": repr(exc),
            })
        if i % 500 == 0:
            print(json.dumps({"event": "load_progress", "shift": shift, "seen": i, "ok": len(arrays), "bad": len(errors)}), flush=True)
    if not arrays:
        return meta, np.empty((0, 0, 0), dtype=np.float32), errors
    min_layers = min(arr.shape[0] for arr in arrays)
    arrays = [arr[:min_layers] for arr in arrays]
    return meta, np.stack(arrays, axis=0).astype(np.float32), errors


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

    def attention_weights(self, num_layers: int | None = None) -> torch.Tensor | None:
        if self.pooling != "global_attention":
            return None
        logits = self.layer_logits if num_layers is None else self.layer_logits[:num_layers]
        return torch.softmax(logits / self.attention_temperature, dim=0)

    def attention_regularization(self, num_layers: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.attention_weights(num_layers)
        if weights is None:
            zero = self.pos.sum() * 0.0
            return zero, zero
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
        if weights.numel() <= 1:
            smooth = weights.sum() * 0.0
        else:
            smooth = torch.mean((weights[1:] - weights[:-1]) ** 2)
        return entropy, smooth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h)
        if self.pooling == "global_attention":
            weights = self.attention_weights(h.shape[1])
            pooled = (h * weights[None, :, None]).sum(dim=1)
        else:
            pooled = h.mean(dim=1)
        return self.out(pooled)


def sq_cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).T
    return torch.clamp(a2 + b2 - 2.0 * (a @ b.T), min=0.0)


def median_positive(d2: torch.Tensor) -> torch.Tensor:
    vals = d2.detach().flatten()
    vals = vals[vals > 1e-8]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=d2.device)
    return torch.median(vals)


class DeepKernel(nn.Module):
    def __init__(
        self,
        sigma_init: float,
        sigma0_init: float,
        epsilon_init: float = 0.05,
        bandwidth_log_radius: float | None = None,
    ) -> None:
        super().__init__()
        self.bandwidth_log_radius = (
            None if bandwidth_log_radius is None else max(float(bandwidth_log_radius), 1e-3)
        )
        self.register_buffer(
            "log_sigma_center",
            torch.tensor(math.log(max(sigma_init, 1e-6)), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "log_sigma0_center",
            torch.tensor(math.log(max(sigma0_init, 1e-6)), dtype=torch.float32),
            persistent=False,
        )
        # Keep the parameterization compatible with the historical checkpoints.
        self.raw_log_sigma = nn.Parameter(self.log_sigma_center.detach().clone())
        self.raw_log_sigma0 = nn.Parameter(self.log_sigma0_center.detach().clone())
        eps = min(max(epsilon_init, 1e-4), 1 - 1e-4)
        self.epsilon_logit = nn.Parameter(torch.tensor(math.log(eps / (1.0 - eps)), dtype=torch.float32))

    def _bounded_log_bandwidth(self, raw: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if self.bandwidth_log_radius is None:
            return raw
        radius = self.bandwidth_log_radius
        return center + radius * torch.tanh((raw - center) / radius)

    def bandwidth_regularization(self) -> torch.Tensor:
        if self.bandwidth_log_radius is None:
            return self.raw_log_sigma.sum() * 0.0
        radius = self.bandwidth_log_radius
        log_sigma = self._bounded_log_bandwidth(self.raw_log_sigma, self.log_sigma_center)
        log_sigma0 = self._bounded_log_bandwidth(self.raw_log_sigma0, self.log_sigma0_center)
        return 0.5 * (
            ((log_sigma - self.log_sigma_center) / radius) ** 2
            + ((log_sigma0 - self.log_sigma0_center) / radius) ** 2
        )

    def set_bandwidth_trainable(self, trainable: bool) -> None:
        self.raw_log_sigma.requires_grad_(trainable)
        self.raw_log_sigma0.requires_grad_(trainable)

    @property
    def sigma(self) -> torch.Tensor:
        log_sigma = self._bounded_log_bandwidth(self.raw_log_sigma, self.log_sigma_center)
        return torch.exp(log_sigma).clamp(min=1e-6)

    @property
    def sigma0(self) -> torch.Tensor:
        log_sigma0 = self._bounded_log_bandwidth(self.raw_log_sigma0, self.log_sigma0_center)
        return torch.exp(log_sigma0).clamp(min=1e-6)

    @property
    def epsilon(self) -> torch.Tensor:
        return torch.sigmoid(self.epsilon_logit).clamp(min=1e-4, max=0.5)

    def forward(
        self,
        phi_a: torch.Tensor,
        phi_b: torch.Tensor,
        raw_a: torch.Tensor,
        raw_b: torch.Tensor,
    ) -> torch.Tensor:
        d2_latent = sq_cdist(phi_a, phi_b)
        d2_raw = sq_cdist(raw_a, raw_b)
        latent_part = (1.0 - self.epsilon) * torch.exp(-d2_latent / self.sigma) + self.epsilon
        raw_part = torch.exp(-d2_raw / self.sigma0)
        return latent_part * raw_part


def mmd_unbiased_and_var(kxx: torch.Tensor, kyy: torch.Tensor, kxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = kxx.shape[0]
    m = kyy.shape[0]
    kxx_off = kxx - torch.diag(torch.diag(kxx))
    kyy_off = kyy - torch.diag(torch.diag(kyy))
    mmd2 = kxx_off.sum() / max(1, n * (n - 1)) + kyy_off.sum() / max(1, m * (m - 1)) - 2.0 * kxy.mean()
    row_x = kxx_off.sum(dim=1) / max(1, n - 1) - kxy.mean(dim=1)
    row_y = kyy_off.sum(dim=1) / max(1, m - 1) - kxy.mean(dim=0)
    h = torch.cat([row_x, row_y], dim=0)
    var_est = torch.var(h, unbiased=False) / max(1, h.numel())
    return mmd2, var_est


def compute_train_stats(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = x[train_mask]
    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = train.std(axis=0, keepdims=True).astype(np.float32)
    sigma = np.maximum(sigma, 1e-4)
    return mu, sigma


def stratified_indices(meta: list[dict[str, str]], mask: np.ndarray, label: str) -> np.ndarray:
    return np.asarray([i for i, row in enumerate(meta) if mask[i] and row["final_label"] == label], dtype=np.int64)


def sample_batch(indices: np.ndarray, batch_each: int, rng: np.random.Generator) -> np.ndarray:
    replace = len(indices) < batch_each
    return rng.choice(indices, size=batch_each, replace=replace)


def group_indices_by_type(meta: list[dict[str, str]], indices: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        groups[str(meta[int(idx)]["task_type"])].append(int(idx))
    return {key: np.asarray(vals, dtype=np.int64) for key, vals in groups.items()}


def sample_type_balanced_batch(groups: dict[str, np.ndarray], batch_each: int, rng: np.random.Generator) -> np.ndarray:
    active = [key for key, vals in sorted(groups.items()) if len(vals)]
    if not active:
        return np.empty((0,), dtype=np.int64)
    base = batch_each // len(active)
    remainder = batch_each % len(active)
    chunks = []
    for pos, key in enumerate(active):
        take = base + (1 if pos < remainder else 0)
        vals = groups[key]
        chunks.append(rng.choice(vals, size=take, replace=len(vals) < take))
    out = np.concatenate(chunks, axis=0)
    rng.shuffle(out)
    return out


def rbf_mmd_biased(a: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return (a.sum() + b.sum()) * 0.0
    sigma = sigma.detach().clamp(min=1e-6)
    kaa = torch.exp(-sq_cdist(a, a) / sigma)
    kbb = torch.exp(-sq_cdist(b, b) / sigma)
    kab = torch.exp(-sq_cdist(a, b) / sigma)
    return kaa.mean() + kbb.mean() - 2.0 * kab.mean()


def conditional_type_mmd_penalty(
    phi: torch.Tensor,
    task_types: list[str],
    sigma: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor, int]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, task_type in enumerate(task_types):
        by_type[str(task_type)].append(i)
    keys = [key for key in sorted(by_type) if len(by_type[key]) >= 2]
    if len(keys) < 2:
        return phi.sum() * 0.0, 0
    penalties = []
    used_pairs = 0
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            ia = torch.as_tensor(by_type[key_a], device=phi.device, dtype=torch.long)
            ib = torch.as_tensor(by_type[key_b], device=phi.device, dtype=torch.long)
            penalties.append(rbf_mmd_biased(phi.index_select(0, ia), phi.index_select(0, ib), sigma))
            used_pairs += 1
            if max_pairs > 0 and used_pairs >= max_pairs:
                return torch.stack(penalties).mean(), used_pairs
    return torch.stack(penalties).mean(), used_pairs


def metrics_by_type(
    meta: list[dict[str, str]],
    eval_idx: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    split_name: str,
    shift: str,
) -> tuple[list[dict], dict[str, float]]:
    rows = []
    bal_values = []
    auc_values = []
    for task_type in sorted({str(meta[int(i)]["task_type"]) for i in eval_idx}):
        mask = np.asarray([str(meta[int(i)]["task_type"]) == task_type for i in eval_idx], dtype=bool)
        yy = y[mask]
        ss = scores[mask]
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            continue
        auc = auc_rank(yy, ss)
        acc, bal, h_acc, d_acc = accuracy_at_threshold(yy, ss, threshold)
        rows.append({
            "shift": shift,
            "split": split_name,
            "task_type": task_type,
            "n": int(len(yy)),
            "honest": int((yy == 0).sum()),
            "dishonest": int((yy == 1).sum()),
            "auc": auc,
            "accuracy_at_global_threshold": acc,
            "balanced_accuracy_at_global_threshold": bal,
            "honest_accuracy_at_global_threshold": h_acc,
            "dishonest_accuracy_at_global_threshold": d_acc,
            "threshold": threshold,
        })
        if np.isfinite(auc):
            auc_values.append(float(auc))
        if np.isfinite(bal):
            bal_values.append(float(bal))
    summary = {
        f"{split_name}_macro_type_auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        f"{split_name}_macro_type_balanced_accuracy_at_global_threshold": float(np.mean(bal_values)) if bal_values else float("nan"),
        f"{split_name}_worst_type_balanced_accuracy_at_global_threshold": float(np.min(bal_values)) if bal_values else float("nan"),
        f"{split_name}_type_count_with_both_labels": float(len(bal_values)),
    }
    return rows, summary


def evaluate_scores(
    model: nn.Module,
    kernel: DeepKernel,
    x_std: np.ndarray,
    meta: list[dict[str, str]],
    ref_h_idx: np.ndarray,
    ref_d_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = np.asarray([1 if meta[i]["final_label"] == POS_LABEL else 0 for i in eval_idx], dtype=np.int8)
    with torch.no_grad():
        ref_h = torch.from_numpy(x_std[ref_h_idx]).to(device=device, dtype=torch.float32)
        ref_d = torch.from_numpy(x_std[ref_d_idx]).to(device=device, dtype=torch.float32)
        raw_h = ref_h.flatten(1)
        raw_d = ref_d.flatten(1)
        phi_h = model(ref_h)
        phi_d = model(ref_d)
        const_h = kernel(phi_h, phi_h, raw_h, raw_h).mean()
        const_d = kernel(phi_d, phi_d, raw_d, raw_d).mean()
        scores = []
        for start in range(0, len(eval_idx), chunk_size):
            idx = eval_idx[start : start + chunk_size]
            xb = torch.from_numpy(x_std[idx]).to(device=device, dtype=torch.float32)
            raw_b = xb.flatten(1)
            phi_b = model(xb)
            mean_d = kernel(phi_b, phi_d, raw_b, raw_d).mean(dim=1)
            mean_h = kernel(phi_b, phi_h, raw_b, raw_h).mean(dim=1)
            score = 2.0 * (mean_d - mean_h) + const_h - const_d
            scores.append(score.detach().cpu().numpy())
    return labels, np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


def init_bandwidths(
    model: nn.Module,
    x_std: np.ndarray,
    h_idx: np.ndarray,
    d_idx: np.ndarray,
    device: torch.device,
    batch_each: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    idx = np.concatenate([sample_batch(h_idx, batch_each, rng), sample_batch(d_idx, batch_each, rng)])
    xb = torch.from_numpy(x_std[idx]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        phi = model(xb)
        raw = xb.flatten(1)
        sigma = float(median_positive(sq_cdist(phi, phi)).detach().cpu())
        sigma0 = float(median_positive(sq_cdist(raw, raw)).detach().cpu())
    return max(sigma, 1e-4), max(sigma0, 1e-4)


def train_one_shift(args, rows, split_rows, shift: str, out_dir: Path, device: torch.device) -> tuple[dict, list[dict], list[dict], list[dict]]:
    split_by_case = {row["case_id"]: row["split"] for row in split_rows}
    print(json.dumps({"event": "load_shift_start", "shift": shift}), flush=True)
    meta, x, errors = load_shift_matrix(rows, split_by_case, shift, args.max_layers)
    if errors:
        write_csv(out_dir / f"load_errors_{shift.replace('-', 'm')}.csv", errors)
    if x.size == 0:
        return {"shift": shift, "status": "no_features_loaded", "load_errors": len(errors)}, [], [], []

    split_arr = np.asarray([row["split"] for row in meta])
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"
    train_h = stratified_indices(meta, train_mask, NEG_LABEL)
    train_d = stratified_indices(meta, train_mask, POS_LABEL)
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    min_needed = max(4, args.batch_each // 4)
    if len(train_h) < min_needed or len(train_d) < min_needed or len(val_idx) < 10 or len(test_idx) < 10:
        return {
            "shift": shift,
            "status": "insufficient_samples",
            "loaded_n": len(meta),
            "train_honest": len(train_h),
            "train_dishonest": len(train_d),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "load_errors": len(errors),
        }, [], [], []

    print(json.dumps({"event": "standardize", "shift": shift, "shape": list(x.shape)}), flush=True)
    mu, sigma = compute_train_stats(x, train_mask)
    x_std = ((x - mu) / sigma).astype(np.float32)
    del x

    torch.manual_seed(args.seed)
    shift_offset = {name: i for i, name in enumerate(SHIFTS)}.get(shift, 0)
    np_rng = np.random.default_rng(args.seed + shift_offset * 1009)
    train_h_by_type = group_indices_by_type(meta, train_h)
    train_d_by_type = group_indices_by_type(meta, train_d)
    model = LayerTransformerMap(
        input_dim=x_std.shape[2],
        num_layers_in=x_std.shape[1],
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        depth=args.depth,
        dropout=args.dropout,
        pooling=args.pooling,
        attention_temperature=args.attention_temperature,
    ).to(device)
    sigma_init, sigma0_init = init_bandwidths(model, x_std, train_h, train_d, device, args.batch_each, np_rng)
    kernel = DeepKernel(
        sigma_init=sigma_init,
        sigma0_init=sigma0_init,
        epsilon_init=args.epsilon_init,
        bandwidth_log_radius=args.kernel_log_radius,
    ).to(device)
    kernel.set_bandwidth_trainable(args.kernel_warmup_epochs <= 0)
    if args.pooling == "global_attention" and args.attention_lr is not None:
        attn_params = [model.layer_logits]
        attn_param_ids = {id(p) for p in attn_params}
        model_params = [p for p in model.parameters() if id(p) not in attn_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": model_params, "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": list(kernel.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": attn_params, "lr": args.attention_lr, "weight_decay": 0.0},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(kernel.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    history = []
    best_state = None
    best_select_score = -1.0
    best_epoch = -1
    patience_left = args.patience
    start_time = time.time()

    epoch = 0
    while args.max_epochs is None or epoch < args.max_epochs:
        epoch += 1
        if epoch == args.kernel_warmup_epochs + 1:
            kernel.set_bandwidth_trainable(True)
        model.train()
        kernel.train()
        losses = []
        mmds = []
        vars_ = []
        type_mmds = []
        type_mmd_pair_counts = []
        for _ in range(args.steps_per_epoch):
            if args.type_balanced_batch:
                h_batch = sample_type_balanced_batch(train_h_by_type, args.batch_each, np_rng)
                d_batch = sample_type_balanced_batch(train_d_by_type, args.batch_each, np_rng)
            else:
                h_batch = sample_batch(train_h, args.batch_each, np_rng)
                d_batch = sample_batch(train_d, args.batch_each, np_rng)
            xh = torch.from_numpy(x_std[h_batch]).to(device=device, dtype=torch.float32)
            xd = torch.from_numpy(x_std[d_batch]).to(device=device, dtype=torch.float32)
            raw_h = xh.flatten(1)
            raw_d = xd.flatten(1)
            phi_h = model(xh)
            phi_d = model(xd)
            kdd = kernel(phi_d, phi_d, raw_d, raw_d)
            khh = kernel(phi_h, phi_h, raw_h, raw_h)
            kdh = kernel(phi_d, phi_h, raw_d, raw_h)
            mmd2, var_est = mmd_unbiased_and_var(kdd, khh, kdh)
            stat = mmd2 / torch.sqrt(var_est + args.var_lambda)
            if args.type_mmd_weight:
                h_types = [meta[int(i)]["task_type"] for i in h_batch]
                d_types = [meta[int(i)]["task_type"] for i in d_batch]
                h_type_mmd, h_type_pairs = conditional_type_mmd_penalty(phi_h, h_types, kernel.sigma, args.type_mmd_max_pairs)
                d_type_mmd, d_type_pairs = conditional_type_mmd_penalty(phi_d, d_types, kernel.sigma, args.type_mmd_max_pairs)
                type_mmd = 0.5 * (h_type_mmd + d_type_mmd)
                type_pair_count = h_type_pairs + d_type_pairs
            else:
                type_mmd = stat * 0.0
                type_pair_count = 0
            attn_entropy, attn_smooth = model.attention_regularization(x_std.shape[1])
            bandwidth_penalty = kernel.bandwidth_regularization()
            loss = (
                -stat
                + args.type_mmd_weight * type_mmd
                + args.attention_entropy_weight * attn_entropy
                + args.attention_smooth_weight * attn_smooth
                + args.kernel_log_reg_weight * bandwidth_penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(kernel.parameters()), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mmds.append(float(mmd2.detach().cpu()))
            vars_.append(float(var_est.detach().cpu()))
            type_mmds.append(float(type_mmd.detach().cpu()))
            type_mmd_pair_counts.append(int(type_pair_count))

        y_val, s_val = evaluate_scores(model, kernel, x_std, meta, train_h, train_d, val_idx, device, args.eval_chunk_size)
        val_auc = auc_rank(y_val, s_val)
        val_auc_df = max(val_auc, 1.0 - val_auc) if np.isfinite(val_auc) else float("nan")
        val_threshold, val_bal = best_val_threshold(y_val, s_val)
        val_acc, val_bal_check, val_h_acc, val_d_acc = accuracy_at_threshold(y_val, s_val, val_threshold)
        _, val_type_summary = metrics_by_type(meta, val_idx, y_val, s_val, val_threshold, "val", shift)
        row = {
            "shift": shift,
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "mmd2": float(np.mean(mmds)),
            "var_est": float(np.mean(vars_)),
            "test_power_proxy": float(-np.mean(losses)),
            "type_mmd_penalty": float(np.mean(type_mmds)),
            "type_mmd_pair_count": float(np.mean(type_mmd_pair_counts)),
            "type_mmd_weight": args.type_mmd_weight,
            "type_balanced_batch": args.type_balanced_batch,
            "val_auc": val_auc,
            "val_direction_free_auc": val_auc_df,
            "val_threshold": val_threshold,
            "val_accuracy_at_best_bal_threshold": val_acc,
            "val_balanced_accuracy_at_best_bal_threshold": val_bal,
            "val_honest_accuracy_at_best_bal_threshold": val_h_acc,
            "val_dishonest_accuracy_at_best_bal_threshold": val_d_acc,
            "sigma": float(kernel.sigma.detach().cpu()),
            "sigma0": float(kernel.sigma0.detach().cpu()),
            "epsilon": float(kernel.epsilon.detach().cpu()),
            "kernel_log_sigma_delta": float(
                (torch.log(kernel.sigma) - kernel.log_sigma_center).detach().cpu()
            ),
            "kernel_log_sigma0_delta": float(
                (torch.log(kernel.sigma0) - kernel.log_sigma0_center).detach().cpu()
            ),
            "kernel_bandwidth_penalty": float(bandwidth_penalty.detach().cpu()),
            "kernel_bandwidth_trainable": bool(kernel.raw_log_sigma.requires_grad),
            "kernel_warmup_epochs": args.kernel_warmup_epochs,
            "kernel_log_radius": args.kernel_log_radius,
            "kernel_log_reg_weight": args.kernel_log_reg_weight,
            "pooling": args.pooling,
            "attention_entropy": float(model.attention_regularization(x_std.shape[1])[0].detach().cpu()),
            "attention_smooth": float(model.attention_regularization(x_std.shape[1])[1].detach().cpu()),
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        row.update(val_type_summary)
        history.append(row)
        print(json.dumps({"event": "epoch", **row}, ensure_ascii=False), flush=True)

        select_score = select_metric_value(row, args.selection_metric)
        if np.isfinite(select_score) and select_score > best_select_score:
            best_select_score = float(select_score)
            best_epoch = epoch
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "kernel": {k: v.detach().cpu() for k, v in kernel.state_dict().items()},
                "epoch": epoch,
                "val_auc": val_auc,
                "val_direction_free_auc": val_auc_df,
                "val_threshold": val_threshold,
                "val_accuracy_at_best_bal_threshold": val_acc,
                "val_balanced_accuracy_at_best_bal_threshold": val_bal_check,
                "val_macro_type_auc": row.get("val_macro_type_auc", float("nan")),
                "val_macro_type_balanced_accuracy_at_global_threshold": row.get("val_macro_type_balanced_accuracy_at_global_threshold", float("nan")),
                "val_worst_type_balanced_accuracy_at_global_threshold": row.get("val_worst_type_balanced_accuracy_at_global_threshold", float("nan")),
                "selection_metric": args.selection_metric,
                "selection_score": best_select_score,
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(json.dumps({"event": "early_stop", "shift": shift, "epoch": epoch}), flush=True)
                break

    if best_state is None:
        return {"shift": shift, "status": "no_valid_model", "loaded_n": len(meta)}, history, [], []
    model.load_state_dict(best_state["model"])
    kernel.load_state_dict(best_state["kernel"])
    model_path = out_dir / f"best_model_{shift.replace('-', 'm')}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kernel_state_dict": kernel.state_dict(),
            "model_config": {
                "input_dim": x_std.shape[2],
                "num_layers_in": x_std.shape[1],
                "latent_dim": args.latent_dim,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "depth": args.depth,
                "dropout": args.dropout,
                "pooling": args.pooling,
                "attention_temperature": args.attention_temperature,
                "attention_lr": args.attention_lr,
                "kernel_log_radius": args.kernel_log_radius,
                "kernel_warmup_epochs": args.kernel_warmup_epochs,
                "kernel_log_reg_weight": args.kernel_log_reg_weight,
                "kernel_sigma_init": sigma_init,
                "kernel_sigma0_init": sigma0_init,
                "type_balanced_batch": args.type_balanced_batch,
                "type_mmd_weight": args.type_mmd_weight,
                "type_mmd_max_pairs": args.type_mmd_max_pairs,
            },
            "standardization": {
                "mean": torch.from_numpy(mu.squeeze(0)),
                "std": torch.from_numpy(sigma.squeeze(0)),
            },
            "shift": shift,
            "best_epoch": best_epoch,
        },
        model_path,
    )
    attention_weights = model.attention_weights(x_std.shape[1])
    if attention_weights is not None:
        attn_path = out_dir / f"layer_attention_{shift.replace('-', 'm')}.csv"
        weights_np = attention_weights.detach().cpu().numpy()
        write_csv(
            attn_path,
            [
                {"shift": shift, "layer_index": i, "attention_weight": float(w)}
                for i, w in enumerate(weights_np)
            ],
        )
    else:
        attn_path = None

    y_val, s_val = evaluate_scores(model, kernel, x_std, meta, train_h, train_d, val_idx, device, args.eval_chunk_size)
    y_test, s_test = evaluate_scores(model, kernel, x_std, meta, train_h, train_d, test_idx, device, args.eval_chunk_size)
    val_auc = auc_rank(y_val, s_val)
    test_auc = auc_rank(y_test, s_test)
    th, val_bal = best_val_threshold(y_val, s_val)
    val_type_rows, val_type_summary = metrics_by_type(meta, val_idx, y_val, s_val, th, "val", shift)
    test_type_rows, test_type_summary = metrics_by_type(meta, test_idx, y_test, s_test, th, "test", shift)
    test_acc, test_bal, test_h_acc, test_d_acc = accuracy_at_threshold(y_test, s_test, th)
    case_scores = []
    for idx, label, score in zip(val_idx, y_val, s_val):
        case_scores.append({**meta[int(idx)], "score": float(score), "y": int(label), "selected_split": "val"})
    for idx, label, score in zip(test_idx, y_test, s_test):
        case_scores.append({**meta[int(idx)], "score": float(score), "y": int(label), "selected_split": "test"})

    summary = {
        "shift": shift,
        "status": "ok",
        "loaded_n": len(meta),
        "loaded_cases": len({row["case_id"] for row in meta}),
        "load_errors": len(errors),
        "feature_shape": "x".join(map(str, x_std.shape[1:])),
        "train_n": int(train_mask.sum()),
        "train_honest": len(train_h),
        "train_dishonest": len(train_d),
        "val_n": len(val_idx),
        "val_honest": int(sum(meta[i]["final_label"] == NEG_LABEL for i in val_idx)),
        "val_dishonest": int(sum(meta[i]["final_label"] == POS_LABEL for i in val_idx)),
        "test_n": len(test_idx),
        "test_honest": int(sum(meta[i]["final_label"] == NEG_LABEL for i in test_idx)),
        "test_dishonest": int(sum(meta[i]["final_label"] == POS_LABEL for i in test_idx)),
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "selection_score": best_state.get("selection_score", float("nan")),
        "best_epoch_val_threshold": best_state.get("val_threshold", float("nan")),
        "best_epoch_val_accuracy": best_state.get("val_accuracy_at_best_bal_threshold", float("nan")),
        "best_epoch_val_balanced_accuracy": best_state.get("val_balanced_accuracy_at_best_bal_threshold", float("nan")),
        "best_epoch_val_macro_type_auc": best_state.get("val_macro_type_auc", float("nan")),
        "best_epoch_val_macro_type_balanced_accuracy": best_state.get("val_macro_type_balanced_accuracy_at_global_threshold", float("nan")),
        "best_epoch_val_worst_type_balanced_accuracy": best_state.get("val_worst_type_balanced_accuracy_at_global_threshold", float("nan")),
        "val_auc": val_auc,
        "val_direction_free_auc": max(val_auc, 1.0 - val_auc) if np.isfinite(val_auc) else float("nan"),
        "val_threshold_balanced_accuracy": val_bal,
        "test_auc": test_auc,
        "test_direction_free_auc": max(test_auc, 1.0 - test_auc) if np.isfinite(test_auc) else float("nan"),
        "test_accuracy_at_val_threshold": test_acc,
        "test_balanced_accuracy_at_val_threshold": test_bal,
        "test_honest_accuracy_at_val_threshold": test_h_acc,
        "test_dishonest_accuracy_at_val_threshold": test_d_acc,
        "threshold_from_val": th,
        "sigma": float(kernel.sigma.detach().cpu()),
        "sigma0": float(kernel.sigma0.detach().cpu()),
        "epsilon": float(kernel.epsilon.detach().cpu()),
        "kernel_log_sigma_delta": float(
            (torch.log(kernel.sigma) - kernel.log_sigma_center).detach().cpu()
        ),
        "kernel_log_sigma0_delta": float(
            (torch.log(kernel.sigma0) - kernel.log_sigma0_center).detach().cpu()
        ),
        "kernel_log_radius": args.kernel_log_radius,
        "kernel_warmup_epochs": args.kernel_warmup_epochs,
        "kernel_log_reg_weight": args.kernel_log_reg_weight,
        "pooling": args.pooling,
        "attention_temperature": args.attention_temperature,
        "attention_lr": args.attention_lr,
        "attention_entropy_weight": args.attention_entropy_weight,
        "attention_smooth_weight": args.attention_smooth_weight,
        "type_balanced_batch": args.type_balanced_batch,
        "type_mmd_weight": args.type_mmd_weight,
        "type_mmd_max_pairs": args.type_mmd_max_pairs,
        "layer_attention_path": str(attn_path) if attn_path is not None else "",
        "model_path": str(model_path),
    }
    summary.update(val_type_summary)
    summary.update(test_type_summary)
    return summary, history, case_scores, val_type_rows + test_type_rows


def write_markdown_report(out_dir: Path, summary_rows: list[dict], manifest: dict) -> None:
    ok = [row for row in summary_rows if row.get("status") == "ok"]
    ok_sorted = sorted(ok, key=lambda r: float(r.get("selection_score", -1)), reverse=True)
    lines = [
        "# AM5000 IIMMD Deep-Kernel Report",
        "",
        "## Method",
        "",
        "This run trains a neural feature map phi(g) and deep kernel for MMD two-sample testing. It does not train a direct honest/dishonest classifier. The optimized objective is a negative test-power proxy:",
        "",
        "`loss = - MMD_u^2(X, Y) / sqrt(var_est + lambda)`",
        "",
        "The kernel is:",
        "",
        "`k(g,g') = ((1 - epsilon) exp(-||phi(g)-phi(g')||^2 / sigma) + epsilon) exp(-||g-g'||^2 / sigma0)`",
        "",
        "Here X is the dishonest train batch and Y is the honest train batch. Validation labels are used only for model/shift selection; test labels are used only after selection.",
        "",
        "## Split",
        "",
        f"- Train cases: {manifest['split_counts'].get('train', 0)}",
        f"- Validation cases: {manifest['split_counts'].get('val', 0)}",
        f"- Test cases: {manifest['split_counts'].get('test', 0)}",
        f"- Split rule: {manifest['split_rule']}",
        "",
        "## Best Shifts",
        "",
        "| shift | status | train H/D | val H/D | test H/D | best epoch | val AUROC | val macro type Bal. ACC | test AUROC | test Bal. ACC | test macro type Bal. ACC | worst type Bal. ACC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda r: str(r.get("shift", ""))):
        def fmt(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return ""
        lines.append(
            "| {shift} | {status} | {trh}/{trd} | {vh}/{vd} | {teh}/{ted} | {ep} | {vauc} | {vmacro} | {tauc} | {tbal} | {tmacro} | {tworst} |".format(
                shift=row.get("shift", ""),
                status=row.get("status", ""),
                trh=row.get("train_honest", ""),
                trd=row.get("train_dishonest", ""),
                vh=row.get("val_honest", ""),
                vd=row.get("val_dishonest", ""),
                teh=row.get("test_honest", ""),
                ted=row.get("test_dishonest", ""),
                ep=row.get("best_epoch", ""),
                vauc=fmt(row.get("val_auc")),
                vmacro=fmt(row.get("val_macro_type_balanced_accuracy_at_global_threshold")),
                tauc=fmt(row.get("test_auc")),
                tbal=fmt(row.get("test_balanced_accuracy_at_val_threshold")),
                tmacro=fmt(row.get("test_macro_type_balanced_accuracy_at_global_threshold")),
                tworst=fmt(row.get("test_worst_type_balanced_accuracy_at_global_threshold")),
            )
        )
    lines.extend(["", "## Selected Model", ""])
    if ok_sorted:
        best = ok_sorted[0]
        lines.extend([
        f"- Selected by `{best.get('selection_metric', 'val_direction_free_auc')}`: shift `{best['shift']}`, epoch `{best['best_epoch']}`.",
            f"- Raw test AUROC: `{float(best['test_auc']):.4f}`.",
            f"- Direction-free test AUROC: `{float(best['test_direction_free_auc']):.4f}`.",
            f"- Test balanced accuracy using validation threshold: `{float(best['test_balanced_accuracy_at_val_threshold']):.4f}`.",
        ])
    else:
        lines.append("- No shift produced a valid model.")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--split",
        default=None,
        help="Optional fixed case split CSV with case_id and split columns. "
        "When provided, reuse it instead of rebuilding a split from train/test totals.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--shifts", default=",".join(SHIFTS))
    parser.add_argument("--train-total", type=int, default=1500)
    parser.add_argument("--test-total", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pooling", choices=["mean", "global_attention"], default="mean")
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--attention-lr", type=float, default=None)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.0)
    parser.add_argument("--attention-smooth-weight", type=float, default=0.0)
    parser.add_argument("--type-balanced-batch", action="store_true")
    parser.add_argument("--type-mmd-weight", type=float, default=0.0)
    parser.add_argument("--type-mmd-max-pairs", type=int, default=0)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional hard upper bound. Omit to train until patience-based early stopping.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--batch-each", type=int, default=64)
    parser.add_argument("--eval-chunk-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--var-lambda", type=float, default=1e-6)
    parser.add_argument("--epsilon-init", type=float, default=0.05)
    parser.add_argument(
        "--kernel-log-radius",
        type=float,
        default=0.5,
        help="Maximum absolute log-bandwidth movement after tanh bounding; 0.5 is about x0.61 to x1.65.",
    )
    parser.add_argument(
        "--kernel-warmup-epochs",
        type=int,
        default=10,
        help="Keep sigma and sigma0 fixed for this many epochs before unfreezing them.",
    )
    parser.add_argument(
        "--kernel-log-reg-weight",
        type=float,
        default=0.05,
        help="Penalty weight for movement of effective log bandwidths from initialization.",
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--direction-free-select", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=[
            "val_auc",
            "val_direction_free_auc",
            "val_accuracy",
            "val_balanced_accuracy",
            "val_composite_auc_balacc",
            "val_composite_raw_auc_balacc",
            "val_macro_type_balanced_accuracy",
            "val_worst_type_balanced_accuracy",
            "val_composite_global_macro_balacc",
        ],
        default=None,
    )
    parser.add_argument("--emit-tgz", action="store_true")
    args = parser.parse_args()
    if args.selection_metric is None:
        args.selection_metric = "val_direction_free_auc" if args.direction_free_select else "val_auc"

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(Path(args.index))
    rows = [row for row in rows if row.get("final_label") in LABELS and row.get("feature_npz")]
    split_cfg = SplitConfig(train_total=args.train_total, test_total=args.test_total, seed=args.seed)
    if args.split:
        fixed_rows = read_csv_rows(Path(args.split))
        split_by_case = {row["case_id"]: row for row in fixed_rows if row.get("case_id") and row.get("split")}
        row_by_case: dict[str, dict[str, str]] = {}
        for row in rows:
            case_id = row.get("case_id", "")
            if case_id and case_id not in row_by_case:
                row_by_case[case_id] = row
        split_rows = []
        for case_id, fixed in split_by_case.items():
            source = row_by_case.get(case_id)
            if source is None or source.get("final_label") not in LABELS:
                continue
            split_rows.append({
                "case_id": case_id,
                "task_type": str(source.get("task_type", fixed.get("task_type", ""))),
                "final_label": source["final_label"],
                "task_index": str(source.get("task_index", fixed.get("task_index", ""))),
                "split": fixed["split"],
            })
    else:
        split_rows = build_case_split(rows, split_cfg)
    write_csv(out_dir / "case_split.csv", split_rows)
    split_counts = Counter(row["split"] for row in split_rows)
    split_by_type_label = Counter((row["split"], row["task_type"], row["final_label"]) for row in split_rows)
    write_csv(
        out_dir / "case_split_by_type_label.csv",
        [
            {"split": k[0], "task_type": k[1], "final_label": k[2], "cases": v}
            for k, v in sorted(split_by_type_label.items())
        ],
    )

    shifts = [s.strip() for s in args.shifts.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "device": str(device), "rows": len(rows), "case_splits": dict(split_counts)}, ensure_ascii=False), flush=True)

    summary_rows = []
    history_rows = []
    all_score_rows = []
    all_type_metric_rows = []
    for shift in shifts:
        shift_out = out_dir / "models"
        shift_out.mkdir(parents=True, exist_ok=True)
        summary, history, case_scores, type_metric_rows = train_one_shift(args, rows, split_rows, shift, shift_out, device)
        summary_rows.append(summary)
        history_rows.extend(history)
        all_score_rows.extend(case_scores)
        all_type_metric_rows.extend(type_metric_rows)
        write_csv(out_dir / "iimmd_summary_by_shift.partial.csv", summary_rows)
        write_csv(out_dir / "iimmd_epoch_history.partial.csv", history_rows)
        write_csv(out_dir / "iimmd_case_scores.partial.csv", all_score_rows)
        write_csv(out_dir / "iimmd_type_metrics.partial.csv", all_type_metric_rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "iimmd_summary_by_shift.csv", summary_rows)
    write_csv(out_dir / "iimmd_epoch_history.csv", history_rows)
    write_csv(out_dir / "iimmd_case_scores.csv", all_score_rows)
    write_csv(out_dir / "iimmd_type_metrics.csv", all_type_metric_rows)
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get("selection_score", float("nan"))))
    manifest = {
        "method": "IIMMD-style deep-kernel MMD test-power training",
        "paper_basis": [
            "Liu et al., Learning Deep Kernels for Non-Parametric Two-Sample Tests, ICML 2020",
            "User-requested PAA/IIMMD-style objective: maximize MMD_u^2 / sqrt(var_est + lambda)",
        ],
        "input_index": str(Path(args.index)),
        "input_split": str(Path(args.split)) if args.split else "",
        "output_dir": str(out_dir),
        "labels": {"positive": POS_LABEL, "negative": NEG_LABEL},
        "split_config": asdict(split_cfg),
        "split_counts": dict(split_counts),
        "split_rule": (
            f"fixed split from {args.split}"
            if args.split
            else (
                f"case-level split; label-balanced train={args.train_total // 2}/"
                f"{args.train_total // 2} and test={args.test_total // 2}/"
                f"{args.test_total // 2}; remaining labeled cases are validation"
            )
        ),
        "model": {
            "type": "LayerTransformerMap",
            "input": "one AM task shift-window hidden-state matrix, shape layers x hidden_dim",
            "latent_dim": args.latent_dim,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "depth": args.depth,
            "pooling": args.pooling,
            "attention_temperature": args.attention_temperature,
            "attention_lr": args.attention_lr,
            "attention_entropy_weight": args.attention_entropy_weight,
            "attention_smooth_weight": args.attention_smooth_weight,
            "type_balanced_batch": args.type_balanced_batch,
            "type_mmd_weight": args.type_mmd_weight,
            "type_mmd_max_pairs": args.type_mmd_max_pairs,
        },
        "kernel": "((1-epsilon)*exp(-||phi(g)-phi(g')||^2/sigma)+epsilon)*exp(-||g-g'||^2/sigma0)",
        "kernel_config": {
            "bandwidth_log_radius": args.kernel_log_radius,
            "warmup_epochs": args.kernel_warmup_epochs,
            "log_bandwidth_regularization_weight": args.kernel_log_reg_weight,
            "effective_range_multiple": [
                math.exp(-args.kernel_log_radius),
                math.exp(args.kernel_log_radius),
            ],
        },
        "objective": "- MMD_u^2(X_dishonest, Y_honest) / sqrt(var_est + lambda) + alpha * conditional within-label type MMD",
        "selection": f"best epoch per shift by {args.selection_metric}; best shift can be selected by validation only",
        "device": str(device),
        "summary_rows": len(summary_rows),
        "history_rows": len(history_rows),
        "case_score_rows": len(all_score_rows),
        "type_metric_rows": len(all_type_metric_rows),
        "best_by_selection_metric": best,
        "elapsed_sec": round(time.time() - t0, 2),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(out_dir, summary_rows, manifest)
    if args.emit_tgz:
        tgz = out_dir.parent / f"{out_dir.name}_tables_models.tgz"
        with tarfile.open(tgz, "w:gz") as tf:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    tf.add(path, arcname=str(Path(out_dir.name) / path.relative_to(out_dir)))
        print(json.dumps({"event": "packed", "tgz": str(tgz), "bytes": tgz.stat().st_size}), flush=True)
    print(json.dumps({"event": "complete", "out": str(out_dir), "best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
