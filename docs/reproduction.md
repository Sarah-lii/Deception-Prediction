# Reproduction guide

## Stage 1: decision-relative detector

Train one frozen detector for each window `t` through `t-7`. The training command writes checkpoints and metrics to the user-supplied output directory. These generated files are ignored by Git.

## Stage 2: cross-window and prefix evaluation

Cross-window evaluation loads the frozen checkpoint, standardization statistics, kernel parameters, and reference pools from the training stage. Prefix evaluation applies those same frozen components to start-relative prefixes; it does not retrain or recalibrate thresholds.

## Stage 3: gated Raw CAA intervention

`prepare_gated_caa.py` combines detector scores with a layer-wise CAA vector archive. Cases predicted honest remain no-ops; cases predicted dishonest receive an intervention specification. The steering server adds the selected layer vector to the current last-token residual stream on each generation forward pass.

`run_agent.py` expects `--prework-dir` to provide the module selected by `--adapter-module` (default: `agent_runtime_adapter`). The adapter must expose the task-loading, agent-construction, final-answer extraction, and snapshot functions referenced by the runner. This adapter is deployment-specific and is not bundled with the public repository.

## Stage 4: final behavior

Run `judge_final_answers.py` on complete final answers, then `aggregate_transitions.py` on matched baseline and intervention judgments. The final analysis uses `honest`, `dishonest`, and `invalid`; `ambiguous` is never treated as a formal result label.

## Compute policy

Detector training and model generation may require GPUs. Prefix postprocessing, deterministic final-answer auditing, and aggregation should use CPU unless a model-based judge is explicitly configured. Job scheduling and storage mounts are deployment concerns and are intentionally absent from this repository.
