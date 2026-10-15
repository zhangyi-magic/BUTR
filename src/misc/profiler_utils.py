"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import copy

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
        flops, macs, _ = calculate_flops(
            model=model_for_info,
            args=[inputs],
            output_as_string=True,
            output_precision=4,
            print_detailed=False,
        )
        params = sum(p.numel() for p in model_for_info.parameters())

    return params, (
        f"Model FLOPs: {flops}   MACs: {macs}   Params: {params:,}\n"
        f"Profile input: {input_shape} (synthetic input; dynamic-query FLOPs depend on the image)."
    )
