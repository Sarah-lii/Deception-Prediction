# Data contract

No data is distributed with this repository.

## Feature index

The detector expects a CSV with at least:

| Column | Meaning |
|---|---|
| `case_id` | Stable task identifier |
| `task_type` | Task-family identifier |
| `final_label` | `honest` or `dishonest` for detector training |
| `shift` | One of `t`, `t-1`, ..., `t-7` |
| `feature_npz` | Path to the corresponding feature file |

Each feature file must contain a numeric matrix compatible with `[layers, hidden_size]`; the AM5000 configuration uses `[40, 5120]`. Three-dimensional arrays are reduced over their leading call/token axis by the loader.

## Fixed split

The split CSV requires `case_id` and `split`, where `split` is `train`, `validation`, or `test`. A case must not appear in more than one split.

## Final-answer evaluation

`judge_final_answers.py` accepts a CSV containing `case_id` and `final_response_text`. Optional columns include `strict_label`, `task_type`, and `final_response_mode`. Labels are based on the complete visible final answer, not runner heuristics or hidden-state scores.

## Privacy and release policy

Do not commit raw prompts, tool logs, final answers, hidden-state tensors, checkpoints, or per-case scores. Release such artifacts only through an approved data repository with a separate data card and license.

