#!/usr/bin/env python3
"""Fail if the public release contains private paths or excluded artifacts."""

from __future__ import annotations

import json
import py_compile
import re
import tempfile
from pathlib import Path

from validate_dataset import main as validate_dataset


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {
    "datasets", "outputs", "results", "artifacts", "checkpoints",
    "models", "logs", "figures", "paper",
}
FORBIDDEN_SUFFIXES = {
    ".csv", ".jsonl", ".npz", ".npy", ".pt", ".pth", ".safetensors",
    ".log", ".tgz", ".zip", ".png", ".jpg", ".jpeg", ".pdf", ".docx",
}
PRIVATE_PATTERNS = re.compile(
    r"/Users/|/mnt/shared-storage|/sharedata/|sarah|xieyuejin|pjlab|safevlagent|"
    r"liyu|gpfs|ai4good1|original_flow_freeze_resample"
)
ENCODED_SOURCE = re.compile(r"base64\s+-d|\.b64\b|\.base64\b", re.IGNORECASE)


def main() -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.relative_to(ROOT).parts]
    python_files: list[Path] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            raise SystemExit(f"forbidden data/output directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"forbidden data/output file: {relative}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise SystemExit(f"unexpected large file: {relative}")
        validator_paths = {
            Path(__file__).resolve(),
            (ROOT / "scripts" / "validate_dataset.py").resolve(),
        }
        if path.resolve() not in validator_paths and (
            path.suffix.lower() in {".py", ".md", ".toml", ".json", ".txt"}
            or path.name == ".gitignore"
        ):
            text = path.read_text(encoding="utf-8")
            if PRIVATE_PATTERNS.search(text):
                raise SystemExit(f"private environment path found: {relative}")
            if ENCODED_SOURCE.search(text):
                raise SystemExit(f"encoded source bootstrap found: {relative}")
        if path.suffix == ".py":
            python_files.append(path)

    with tempfile.TemporaryDirectory(prefix="am_deception_compile_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, path in enumerate(python_files):
            py_compile.compile(str(path), cfile=str(temp_root / f"{index}.pyc"), doraise=True)

    config = json.loads((ROOT / "configs/am5000.example.json").read_text(encoding="utf-8"))
    if config["labels"] != ["honest", "dishonest", "invalid"]:
        raise SystemExit("formal labels are inconsistent")
    validate_dataset()
    print(json.dumps({
        "status": "ok",
        "root": str(ROOT),
        "files": len(files),
        "python_files": len(python_files),
        "contains_dataset": True,
        "contains_model_outputs": False,
    }, indent=2))


if __name__ == "__main__":
    main()
