"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import contextlib
import copy
import io

import torch


def stats(cfg) -> tuple[int, str]:
    """Profile a deployed copy with a synthetic input, preserving training RNG state."""
    from calflops import calculate_flops

    base_size = cfg.train_dataloader.collate_fn.base_size
    if isinstance(base_size, (list, tuple)):
        input_shape = (1, 3, base_size[0], base_size[1])
    else:
        input_shape = (1, 3, base_size, base_size)

    parameter = next(cfg.model.parameters())
    device = parameter.device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        model_for_info = copy.deepcopy(cfg.model).deploy()
        # calflops generates an uninitialized tensor when given only input_shape.
        # Supply a defined input so dynamic query budgeting receives finite values.
        inputs = torch.zeros(input_shape, device=device, dtype=parameter.dtype)
        query_counts = []

        def record_queries(module, args, output):
            if isinstance(output, dict) and torch.is_tensor(output.get("pred_logits")):
                query_counts.append(int(output["pred_logits"].shape[1]))

        handle = model_for_info.register_forward_hook(record_queries)
        profile_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(profile_output):
                flops, macs, _ = calculate_flops(
                    model=model_for_info,
                    args=[inputs],
                    output_as_string=True,
                    output_precision=4,
                    print_results=True,
                    print_detailed=False,
                    include_backPropagation=False,
                )
        finally:
            handle.remove()
        params = sum(p.numel() for p in model_for_info.parameters())

    # calflops uses a fixed training label even when profiling a deployment model.
    print(profile_output.getvalue().replace("Total Training Params:", "Deployment parameters:"), end="")
    queries = ", ".join(str(n) for n in sorted(set(query_counts))) or "not exposed by model"
    return params, (
        f"Deployment parameters: {params:,}\n"
        f"Synthetic-input forward FLOPs: {flops}   MACs: {macs}\n"
        f"Profile: eval/deploy mode; zero input {input_shape}; decoder queries: {queries}.\n"
        "This is a startup sanity check, not dataset-average FLOPs or mean queries.\n"
        "Dataset averages require a trained checkpoint and real evaluation images."
    )
