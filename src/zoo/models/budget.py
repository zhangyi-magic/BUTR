"""Non-learned query budgeting from encoder classification confidences."""

import torch


def bucketize_budget(budget, bins):
    """Round each budget up to the nearest available bin."""
    boundaries = budget.new_tensor(bins)
    indices = torch.bucketize(budget, boundaries, right=False)
    return boundaries[indices.clamp(max=len(bins) - 1)]


@torch.no_grad()
def compute_query_budget(
    logits,
    valid_mask,
    min_queries=300,
    max_queries=1500,
    safety=1.0,
    min_prob=1e-6,
    score_method="max_sigmoid",
    budget_map="linear",
    bins=(600, 900, 1200, 1500),
):
    """Return one integer budget per image, without gradients or learned parameters.

    The paper uses max-class sigmoid confidence and a linear mapping:
    ceil(safety * sum(confidence)), clipped to the feasible query range.
    Other scoring methods and discrete bins support the ablations.
    logits has shape [B, L, C]; valid_mask has shape [B, L, 1] or [1, L, 1].
    """
    if logits.ndim != 3:
        raise ValueError("Expected encoder logits with shape [B, L, C].")
    batch, locations, _ = logits.shape
    valid = valid_mask.squeeze(-1) > 0.5
    if valid.shape[0] == 1 and batch > 1:
        valid = valid.expand(batch, -1)
    if valid.shape != (batch, locations):
        raise ValueError("Valid mask does not match the encoder logits.")
    probabilities = logits.sigmoid() * valid.unsqueeze(-1).to(logits.dtype)
    if score_method == "max":
        scores = torch.relu(logits.amax(dim=-1))
    elif score_method == "max_sigmoid":
        scores = probabilities.amax(dim=-1)
    elif score_method == "sum_prob":
        scores = probabilities.sum(dim=-1)
    elif score_method == "union_prob":
        log_product = torch.log1p(-probabilities.clamp(max=1 - 1e-6)).sum(dim=-1)
        scores = 1.0 - torch.exp(log_product)
    else:
        raise ValueError(f"Unknown score_method: {score_method}")
    scores = scores * valid.to(scores.dtype)
    scores = torch.where(scores > min_prob, scores, scores.new_zeros(()))
    budget = torch.ceil(scores.sum(dim=1) * safety).to(torch.long)
    budget = budget.clamp(min=min_queries, max=max_queries)
    if budget_map == "bins":
        budget = bucketize_budget(budget, bins).clamp(min=min_queries, max=max_queries)
    elif budget_map != "linear":
        raise ValueError(f"Unknown budget_map: {budget_map}")
    return budget
