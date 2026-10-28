# AM5000 Dataset Card

## Overview

AM5000 is a synthetic benchmark for source-grounded tool-using agents. It contains 5,000 tasks organized into five task types, with 1,000 tasks per type and no duplicate `task_index` values.

| File | Type | Tasks | Task condition |
|---|---:|---:|---|
| `task_1.json` | 1 | 1,000 | A named source document is requested but no source content is supplied. |
| `task_2.json` | 2 | 1,000 | A local text source is supplied, but it explicitly lacks the requested evidence. |
| `task_3.json` | 3 | 1,000 | A requested slide deck is unavailable while a related draft text note is supplied. |
| `task_4.json` | 4 | 1,000 | The agent is asked to download a named source and produce a local report. |
| `task_5.json` | 5 | 1,000 | A multi-step source-download, analysis, and deliverable task is requested. |

The eight topical domains are Society, Entertainment, Acadamic, Software Engineering, Economy, Health, Education, and Puzzle. The original `Acadamic` spelling is retained to preserve exact dataset compatibility.

## Record schema

Every record contains:

| Field | Description |
|---|---|
| `task_index` | Globally unique integer task identifier. |
| `task_type` | Integer task-family identifier from 1 to 5. |
| `instruction` | User-facing agent instruction. |
| `topic` | Topical domain. |
| `tool` | Original tool declaration field; empty in this task release. |

Depending on task type, a record also contains `file_name`, `target_file_name`, `pptx_file_name`, `txt_file_name`, or synthetic `file_content`.

## Labels and splits

The dataset does not assign an intrinsic `honest` or `dishonest` label to a task. Those labels describe an agent's final response and must be produced after generation by the strict final-answer policy. This prevents task identity from being treated as a response label.

Training, validation, and test splits for representation experiments are therefore created from completed, judged agent runs. The detector training code constructs deterministic, type-aware and label-aware splits from the supplied feature index and final-response labels. Do not infer response labels from `task_type` alone.

## Integrity

Run:

```bash
python scripts/validate_dataset.py
```

The validator checks file hashes, record counts, schemas, task-type consistency, identifier ranges, uniqueness, and the absence of private infrastructure strings.

## Excluded artifacts

This release does not include model-generated answers, tool trajectories, source documents outside the synthetic fields above, hidden-state arrays, model checkpoints, or experiment results.

