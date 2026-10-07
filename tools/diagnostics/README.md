# Optional diagnostics

`proposal_stats.py` preserves the original proposal scatter, box, and heatmap helpers outside the training loop. It requires the plotting dependencies in `requirements.txt` and a decoder exposing `proposal_stats_buffer`; the default BUTR decoder does not collect this buffer.

The normal train/evaluate path does not import these helpers or generate proposal PDFs. For final prediction overlays, use `tools/inference/torch_inf.py`. Existing feature visualization and dataset analysis scripts remain in `tools/` for manual use.

`criterion_comparison.py` is the historical verbose loss-debugging implementation previously stored in `src/zoo/models/compare.py`. It is kept here for manual comparisons and does not register or replace the production criterion.
