# Budgeter implementation

BUTR makes the number of decoder queries input dependent while retaining standard Top-K initialization. The budgeter introduces no learned parameters or additional supervision.

For encoder classification logits with shape `[B, L, C]` and a valid-proposal mask with shape `[B, L, 1]` or `[1, L, 1]`, the paper's estimator and mapping are:

```python
confidence = logits.sigmoid().amax(dim=-1)
density_proxy = (confidence * valid_mask.squeeze(-1)).sum(dim=-1)
budget = (safety_factor * density_proxy).ceil().long()
budget = budget.clamp(min_queries, max_queries)
```

The production implementation additionally suppresses scores below `budget_min_prob`. The default paper configuration uses `budget_score_method: max_sigmoid` and `budget_map: linear`. Other supported scoring methods (`max`, `sum_prob`, `union_prob`) and the `bins` mapping are retained for ablations.

See [`budget.py`](../src/zoo/models/budget.py) for the estimator and [`butr_decoder.py`](../src/zoo/models/butr_decoder.py) for Top-K query initialization. Training and evaluation safety factors are configured separately and should follow the selected experiment configuration.

Budgets are estimated independently for each image. The tensorized decoder uses the maximum budget within a batch, so every image in that batch receives the same number of decoded queries. Use evaluation batch size 1 when measuring per-image query usage.
