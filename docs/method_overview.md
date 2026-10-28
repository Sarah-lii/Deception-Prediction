# Method Overview

The project is organized as one directional experimental pipeline. Each stage
has a narrow input/output contract so that task data, model activations, learned
detectors, interventions, and final behavioral labels remain distinct.

```text
AM5000 tasks
    |
    v
Agent generation + hidden-state extraction
    |
    +--> complete final answers --> strict final-answer judge
    |
    v
decision-relative windows t ... t-7
    |
    v
mean-pooled deep-kernel MMD detector (300 dimensions)
    |
    +--> frozen cross-window evaluation
    +--> frozen prefix accumulation evaluation
    +--> risk gate
              |
              v
       layer-wise Raw CAA steering
              |
              v
       continued agent generation
              |
              v
       complete final-answer judge + paired transitions
```

## Stage ownership

| Stage | Module | Primary input | Primary output |
|---|---|---|---|
| Task definitions | `data/am5000/` | Released JSON task files | 5,000 agent tasks |
| Detector training | `detection/train_deep_kernel_mmd.py` | Feature index and judged labels | Frozen detector, kernel, reference pools, split |
| Cross-window evaluation | `detection/evaluate_cross_window.py` | Frozen detector and windowed features | Test metrics across `t` to `t-7` |
| Temporal analysis | `temporal/evaluate_prefix_accumulation.py` | Frozen detector and prefix features | Prefix-level risk trajectories |
| Intervention preparation | `intervention/prepare_gated_caa.py` | Detector scores and layer-wise CAA vectors | Per-case intervention specifications |
| Online intervention | `intervention/steering_server.py`, `intervention/run_agent.py` | Agent requests and intervention specifications | Continued complete final answers |
| Behavioral evaluation | `evaluation/` | Complete visible final answers | Three-class labels and paired transitions |

## Separation of concerns

- AM5000 tasks do not contain response-level honesty labels.
- Detector splits are produced only after completed responses are judged.
- Temporal evaluation reuses frozen detector parameters; it does not retrain or
  recalibrate on the test window.
- Steering changes hidden states during generation, but success is evaluated
  from the newly generated complete final answer.
- Runner heuristics such as `ambiguous` are not formal result labels.

