"""Aggregate decoded query counts across evaluation workers."""

import torch


QNUM_SUM = 0.0
QNUM_SUMSQ = 0.0
QNUM_COUNT = 0


def qnum_reset():
    global QNUM_SUM, QNUM_SUMSQ, QNUM_COUNT
    QNUM_SUM = 0.0
    QNUM_SUMSQ = 0.0
    QNUM_COUNT = 0


def qnum_update(batch_queries_num):
    global QNUM_SUM, QNUM_SUMSQ, QNUM_COUNT

    values = [int(x) for x in batch_queries_num]
    QNUM_COUNT += len(values)
    QNUM_SUM += float(sum(values))
    QNUM_SUMSQ += float(sum(x * x for x in values))


def qnum_report(prefix="[QUERY NUM]", device=None):
    global QNUM_SUM, QNUM_SUMSQ, QNUM_COUNT

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed and torch.distributed.get_backend() == "nccl":
        device = torch.device(device or "cuda")
    else:
        device = torch.device("cpu")
    stats = torch.tensor(
        [QNUM_SUM, QNUM_SUMSQ, float(QNUM_COUNT)],
        device=device,
        dtype=torch.float64,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    else:
        is_main_process = True

    q_sum, q_sumsq, count = stats.tolist()
    if count <= 0:
        if is_main_process:
            print(f"{prefix} no samples collected.")
        return {"count": 0, "mean": 0.0, "variance": 0.0, "std": 0.0}

    q_mean = q_sum / count
    q_var = max(0.0, q_sumsq / count - q_mean * q_mean)
    q_std = q_var ** 0.5

    if is_main_process:
        print(f"{prefix} N={int(count)} | #Q: mean={q_mean:.3f} | var={q_var:.3f} | std={q_std:.3f}")
    return {"count": int(count), "mean": q_mean, "variance": q_var, "std": q_std}
