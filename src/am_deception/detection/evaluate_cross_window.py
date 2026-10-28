#!/usr/bin/env python3
"""Cross-shift evaluation for frozen mean-pooling IIMMD/v1f checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from am_deception.detection import train_deep_kernel_mmd as iimmd


SHIFTS = ("t", "t-1", "t-2", "t-3", "t-4", "t-5", "t-6", "t-7")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shift_tag(shift: str) -> str:
    return shift.replace("-", "m")


def split_indices(meta: list[dict[str, str]], split: str) -> np.ndarray:
    return np.asarray([i for i, row in enumerate(meta) if row["split"] == split], dtype=np.int64)


def eval_splits(meta: list[dict[str, str]]) -> list[tuple[str, np.ndarray]]:
    val_idx = split_indices(meta, "val")
    test_idx = split_indices(meta, "test")
    heldout_idx = np.concatenate([val_idx, test_idx])
    return [("val", val_idx), ("test", test_idx), ("heldout", heldout_idx)]


def labels_for(meta: list[dict[str, str]], indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [1 if meta[int(i)]["final_label"] == iimmd.POS_LABEL else 0 for i in indices],
        dtype=np.int8,
    )


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw - mean[None, :, :]) / std[None, :, :]).astype(np.float32, copy=False)


def load_checkpoint(model_root: Path, train_shift: str, device: torch.device) -> tuple[dict, torch.nn.Module, iimmd.DeepKernel, np.ndarray, np.ndarray]:
    path = model_root / f"best_model_{shift_tag(train_shift)}.pth"
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["model_config"]
    model = iimmd.LayerTransformerMap(
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
    model.load_state_dict(ckpt["model_state_dict"])
    kernel = iimmd.DeepKernel(
        float(cfg["kernel_sigma_init"]),
        float(cfg["kernel_sigma0_init"]),
        0.05,
        cfg.get("kernel_log_radius"),
    ).to(device)
    kernel.load_state_dict(ckpt["kernel_state_dict"])
    mean = ckpt["standardization"]["mean"].detach().cpu().numpy().astype(np.float32)
    std = ckpt["standardization"]["std"].detach().cpu().numpy().astype(np.float32)
    return ckpt, model, kernel, mean, std


@torch.no_grad()
def score_against_reference(
    model: torch.nn.Module,
    kernel: iimmd.DeepKernel,
    x_eval: np.ndarray,
    ref_h: np.ndarray,
    ref_d: np.ndarray,
    device: torch.device,
    chunk_size: int,
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
    pieces = []
    for start in range(0, len(x_eval), chunk_size):
        xb = torch.from_numpy(x_eval[start : start + chunk_size]).to(device=device, dtype=torch.float32)
        raw_b = xb.flatten(1)
        phi_b = model(xb)
        mean_d = kernel(phi_b, phi_d, raw_b, raw_d).mean(dim=1)
        mean_h = kernel(phi_b, phi_h, raw_b, raw_h).mean(dim=1)
        pieces.append((2.0 * (mean_d - mean_h) + const_h - const_d).detach().cpu().numpy())
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0,), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--eval-chunk-size", type=int, default=128)
    args = parser.parse_args()

    paths = {
        "index": Path(args.index),
        "split": Path(args.split),
        "model_root": Path(args.model_root),
        "summary": Path(args.summary),
        "out": Path(args.out),
    }
    for name in ("index", "split", "model_root", "summary"):
        if not paths[name].exists():
            raise FileNotFoundError(f"{name} does not exist: {paths[name]}")
    paths["out"].mkdir(parents=True, exist_ok=True)

    rows = iimmd.read_csv_rows(paths["index"])
    split_rows = iimmd.read_csv_rows(paths["split"])
    split_by_case = {row["case_id"]: row["split"] for row in split_rows}
    summary_rows = {row["shift"]: row for row in iimmd.read_csv_rows(paths["summary"])}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "device": str(device), "out": str(paths["out"])}, ensure_ascii=False), flush=True)

    bundles = []
    for train_shift in SHIFTS:
        ckpt, model, kernel, mean, std = load_checkpoint(paths["model_root"], train_shift, device)
        meta_train, raw_train, errors = iimmd.load_shift_matrix(rows, split_by_case, train_shift, args.max_layers)
        train_idx = split_indices(meta_train, "train")
        train_h = np.asarray([i for i in train_idx if meta_train[int(i)]["final_label"] == iimmd.NEG_LABEL], dtype=np.int64)
        train_d = np.asarray([i for i in train_idx if meta_train[int(i)]["final_label"] == iimmd.POS_LABEL], dtype=np.int64)
        x_train = standardize(raw_train, mean, std)
        bundles.append({
            "shift": train_shift,
            "model": model,
            "kernel": kernel,
            "mean": mean,
            "std": std,
            "ref_h": x_train[train_h],
            "ref_d": x_train[train_d],
            "threshold": float(summary_rows[train_shift]["threshold_from_val"]),
            "train_h": len(train_h),
            "train_d": len(train_d),
            "train_n": len(train_idx),
            "load_errors_train": len(errors),
            "checkpoint": str(paths["model_root"] / f"best_model_{shift_tag(train_shift)}.pth"),
        })
        del raw_train, x_train, meta_train
        gc.collect()

    metric_rows: list[dict] = []
    for eval_shift in SHIFTS:
        print(json.dumps({"event": "load_eval_shift", "eval_shift": eval_shift}, ensure_ascii=False), flush=True)
        meta_eval, raw_eval, eval_errors = iimmd.load_shift_matrix(rows, split_by_case, eval_shift, args.max_layers)
        for bundle in bundles:
            x_eval = standardize(raw_eval, bundle["mean"], bundle["std"])
            scores = score_against_reference(
                bundle["model"],
                bundle["kernel"],
                x_eval,
                bundle["ref_h"],
                bundle["ref_d"],
                device,
                args.eval_chunk_size,
            )
            for split_name, idx in eval_splits(meta_eval):
                y = labels_for(meta_eval, idx)
                s = scores[idx]
                acc, bal, h_acc, d_acc = iimmd.accuracy_at_threshold(y, s, bundle["threshold"])
                metric_rows.append({
                    "train_shift": bundle["shift"],
                    "eval_shift": eval_shift,
                    "eval_split": split_name,
                    "same_shift": int(bundle["shift"] == eval_shift),
                    "n": len(idx),
                    "honest": int((y == 0).sum()),
                    "dishonest": int((y == 1).sum()),
                    "auroc": iimmd.auc_rank(y, s),
                    "acc": acc,
                    "bal_acc": bal,
                    "honest_acc": h_acc,
                    "dishonest_acc": d_acc,
                    "threshold_from_train_val": bundle["threshold"],
                    "reference_train_honest": bundle["train_h"],
                    "reference_train_dishonest": bundle["train_d"],
                    "train_checkpoint": bundle["checkpoint"],
                    "load_errors_train": bundle["load_errors_train"],
                    "load_errors_eval": len(eval_errors),
                })
            del x_eval, scores
        del raw_eval, meta_eval
        gc.collect()

    fields = [
        "train_shift", "eval_shift", "eval_split", "same_shift", "n", "honest", "dishonest",
        "auroc", "acc", "bal_acc", "honest_acc", "dishonest_acc", "threshold_from_train_val",
        "reference_train_honest", "reference_train_dishonest", "train_checkpoint",
        "load_errors_train", "load_errors_eval",
    ]
    write_csv(paths["out"] / "cross_shift_metrics.csv", metric_rows, fields)
    write_csv(paths["out"] / "same_shift_test_metrics.csv", [
        row for row in metric_rows if row["eval_split"] == "test" and row["same_shift"]
    ], fields)
    for split_name in ("val", "test", "heldout"):
        split_rows_out = [row for row in metric_rows if row["eval_split"] == split_name]
        for metric in ("auroc", "acc", "bal_acc"):
            matrix_rows = []
            for train_shift in SHIFTS:
                values = {row["eval_shift"]: row[metric] for row in split_rows_out if row["train_shift"] == train_shift}
                matrix_rows.append({"train_shift": train_shift, **{shift: values.get(shift, "") for shift in SHIFTS}})
            write_csv(paths["out"] / f"{split_name}_{metric}_matrix.csv", matrix_rows, ["train_shift", *SHIFTS])
    manifest = {
        "status": "ok",
        "method": "frozen mean-pooling IIMMD/v1f cross-shift evaluation",
        "index": str(paths["index"]),
        "split": str(paths["split"]),
        "model_root": str(paths["model_root"]),
        "summary": str(paths["summary"]),
        "out": str(paths["out"]),
        "rows": len(metric_rows),
        "shifts": list(SHIFTS),
    }
    (paths["out"] / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
