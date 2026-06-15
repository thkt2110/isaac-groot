# Official-Mini Plan Update

This note adds two required items to the final GR00T official-mini pipeline plan.

## 1. Notebook 04: Denormalized Evaluation

Notebook 04 must report the main evaluation metrics on raw action space, not only normalized action space.

Required computation:

```python
pred_raw = pred_norm * action_std + action_mean
target_raw = target_norm * action_std + action_mean
raw_mse = mse(pred_raw, target_raw)
raw_mae = mae(pred_raw, target_raw)
raw_rmse = sqrt(raw_mse)
```

Report priority:

- Main table: `raw_mse`, `raw_mae`, `raw_rmse`
- Optional/debug table: normalized MSE/MAE/RMSE

Reason:

- Raw-space metrics are easier to explain in the seminar.
- They are closer to the open-loop evaluation style used by the original GR00T pipeline.
- Normalized-space metrics are useful for debugging but should not be the only reported result.

## 2. Notebook 05: H Ablation Required

Notebook 05 must include action horizon ablation:

```text
H=8 vs H=16
```

Implementation requirement:

- Re-run Notebook 02 with `ACTION_HORIZON = 8`.
- Re-run Notebook 02 with `ACTION_HORIZON = 16`.
- No model class change is required.
- The model output remains `[B, H, action_dim]`; only `H` changes through config/data.

Fair comparison requirements:

- Same episode split
- Same DiT config
- Same optimizer and learning rate
- Same number of epochs
- Same K-step inference setting
- Same denormalized raw action metrics

Interpretation:

- `H=16` is the official-mini baseline because GR00T predicts action chunks.
- `H=8` is the ablation run.
- Expected trade-off: shorter horizon can be easier to learn, while longer horizon is closer to the original GR00T action chunking objective.

## Mapping To Notebooks

| Requirement | Notebook | Status in new official-mini notebooks |
|---|---|---|
| Denormalized raw action metrics | 04 | Implemented in `04_evaluate_official_mini_denormalized.ipynb` |
| H=8 vs H=16 ablation | 05 | Implemented in `05_strong_ablation_official_mini.ipynb`; requires H8 prepared/trained output |

