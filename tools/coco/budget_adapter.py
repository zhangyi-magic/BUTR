"""Attach BUTR budgeting to an official D-FINE/DEIM decoder for inference."""

import importlib.util
import inspect
import math
from collections import Counter
from pathlib import Path
from types import MethodType

import torch


def bridge_convert_pil_image(converter):
    """Call legacy DEIM's image conversion through the current torchvision API."""
    from torchvision.transforms import v2

    if (
        hasattr(v2.Transform, "transform")
        and converter.transform is v2.Transform.transform
        and "_transform" in converter.__dict__
    ):
        converter.transform = converter._transform
        return True
    return False


def load_budget_function():
    # Load the shared estimator without importing BUTR's model registry.
    path = Path(__file__).resolve().parents[2] / "src/zoo/models/budget.py"
    spec = importlib.util.spec_from_file_location("_butr_budget_function", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_query_budget


class BudgetAdapter:
    """Keep upstream Top-K ranking and model state; change only its query count."""

    def __init__(self, decoder, *, min_queries, max_queries, safety, min_prob=1e-6):
        if getattr(decoder, "learn_query_content", False):
            raise ValueError("The plug-in requires proposal-derived query content.")
        if getattr(decoder, "query_select_method", None) != "default":
            raise ValueError("The plug-in requires standard confidence-ranked Top-K selection.")
        if not 1 <= min_queries <= max_queries:
            raise ValueError("Require 1 <= min_queries <= max_queries.")
        if not math.isfinite(safety) or safety <= 0:
            raise ValueError("safety must be finite and positive.")
        if not math.isfinite(min_prob) or min_prob < 0:
            raise ValueError("min_prob must be finite and nonnegative.")
        if "_select_topk" in decoder.__dict__:
            raise ValueError("Decoder selection is already overridden.")
        original = decoder._select_topk
        parameters = tuple(inspect.signature(original).parameters)
        if parameters != ("memory", "outputs_logits", "outputs_anchors_unact", "topk"):
            raise ValueError(f"Unsupported upstream Top-K signature: {parameters}")
        compute_budget = load_budget_function()
        self.settings = dict(
            min_queries=int(min_queries),
            max_queries=int(max_queries),
            safety=float(safety),
            min_prob=float(min_prob),
        )
        self.enabled = True
        self.requested_counts = []
        self.decoded_counts = []
        self.decoder = decoder

        def select(module, memory, outputs_logits, outputs_anchors_unact, topk):
            if self.enabled:
                if module.training:
                    raise RuntimeError("The official-model budget plug-in is inference-only.")
                # Upstream marks invalid proposal anchors with infinity.
                valid_mask = torch.isfinite(outputs_anchors_unact).all(-1, keepdim=True)
                budgets = compute_budget(outputs_logits, valid_mask, **self.settings)
                topk = min(int(budgets.max().item()), outputs_logits.shape[1])
                self.requested_counts.extend(budgets.cpu().tolist())
            self.decoded_counts.extend([int(topk)] * outputs_logits.shape[0])
            return original(memory, outputs_logits, outputs_anchors_unact, topk)

        decoder._select_topk = MethodType(select, decoder)

    def reset(self, *, enabled):
        self.enabled = enabled
        self.requested_counts.clear()
        self.decoded_counts.clear()

    def summary(self):
        if not self.decoded_counts:
            return dict(images=0)
        counts = self.decoded_counts
        result = dict(
            images=len(counts),
            mean=sum(counts) / len(counts),
            minimum=min(counts),
            maximum=max(counts),
            histogram=dict(sorted(Counter(counts).items())),
        )
        if self.requested_counts:
            result["requested_mean"] = sum(self.requested_counts) / len(self.requested_counts)
        return result

    def remove(self):
        """Restore class-method selection without changing checkpoint keys."""
        del self.decoder._select_topk
