"""CPU/Gloo integration test for evaluation metric sharing and query reductions."""

import datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from src.misc.qnum_stats import qnum_report
from src.solver.det_engine import evaluate
from src.zoo.models.postprocessor import DomePostProcessor


class Detector(torch.nn.Module):
    def forward(self, samples, targets=None, epoch=None):
        queries = 3 + 2 * dist.get_rank()
        return dict(
            pred_logits=torch.zeros(1, queries, 2),
            pred_boxes=torch.full((1, queries, 4), 0.1),
            batch_queries_num=[queries],
        )


class Evaluator:
    """A rank-zero evaluator, matching the production COCO metric ownership."""

    def __init__(self):
        self.coco_eval = {"bbox": SimpleNamespace(stats=torch.zeros(12))}

    def cleanup(self):
        pass

    def update(self, results):
        assert len(results) == 1

    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        if dist.get_rank() == 0:
            self.coco_eval["bbox"].stats[0] = 0.35

    def summarize(self):
        pass


def evaluate_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        loader = [
            (
                torch.zeros(1, 3, 32, 32),
                [dict(image_id=torch.tensor(rank), orig_size=torch.tensor([40, 20]))],
            )
        ]
        result, _ = evaluate(
            Detector(),
            torch.nn.Identity(),
            DomePostProcessor(num_classes=2),
            loader,
            Evaluator(),
            torch.device("cpu"),
        )
        assert abs(result["coco_eval_bbox"][0] - 0.35) < 1e-6, result
        counts = qnum_report()
        assert counts["count"] == 2 and counts["mean"] == 4.0 and counts["std"] == 1.0
    finally:
        dist.destroy_process_group()


class DistributedEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_every_rank_receives_identical_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            torch.multiprocessing.spawn(evaluate_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
