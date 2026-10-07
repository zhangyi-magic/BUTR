"""CUDA latency measurement, invoked explicitly outside training."""

import time

import torch


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    postprocessor,
    data_loader,
    device,
    warmup_iters: int = 50,
    measure_iters: int = None,
    measure_postprocess: bool = True,
):
    """
    Benchmark per-image inference latency on a single GPU.

    Args:
        model: detection model
        postprocessor: postprocess function/module
        data_loader: evaluation dataloader (recommended bs=1)
        device: cuda device
        warmup_iters: number of warmup iterations
        measure_iters: number of measured iterations after warmup.
                       If None, measure all remaining iterations in the dataloader.
        measure_postprocess: if True, measure model + postprocess;
                             otherwise measure model forward only

    Returns:
        stats: dict with latency/fps results
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Latency benchmarking requires a CUDA device.")
    if warmup_iters < 0 or (measure_iters is not None and measure_iters < 1):
        raise ValueError("Invalid warmup or measurement iteration count.")
    if getattr(data_loader, "batch_size", None) != 1:
        raise ValueError("Per-image latency requires batch_size=1.")
    model.eval()

    latency_list = []
    measured_images = 0
    measured_iters = 0

    print(
        f"[Latency] warmup_iters={warmup_iters}, measure_iters={measure_iters}, "
        f"measure_postprocess={measure_postprocess}"
    )

    for it, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)

        if hasattr(samples, "tensors"):
            batch_size = samples.tensors.shape[0]
        else:
            batch_size = len(targets)

        # warmup stage
        if it < warmup_iters:
            outputs = model(samples)
            if measure_postprocess:
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
                _ = postprocessor(outputs, orig_target_sizes)
            continue

        # stop if a fixed number of measured iters is requested
        if measure_iters is not None and measured_iters >= measure_iters:
            break

        torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        outputs = model(samples)
        if measure_postprocess:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            _ = postprocessor(outputs, orig_target_sizes)

        torch.cuda.synchronize(device)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        latency_list.append(elapsed / batch_size)
        measured_images += batch_size
        measured_iters += 1

    assert measured_iters > 0, (
        "No iterations were measured. Check data_loader size or warmup_iters."
    )

    avg_latency = sum(latency_list) / len(latency_list)
    avg_latency_ms = avg_latency * 1000.0
    fps = 1.0 / avg_latency

    print(f"[Latency] Measured iterations: {measured_iters}")
    print(f"[Latency] Measured images: {measured_images}")
    print(f"[Latency] Average latency: {avg_latency_ms:.3f} ms/image")
    print(f"[Latency] FPS: {fps:.3f}")

    stats = {
        "latency_ms_per_image": avg_latency_ms,
        "fps": fps,
        "num_measured_iters": measured_iters,
        "num_measured_images": measured_images,
        "measure_postprocess": measure_postprocess,
    }
    return stats
