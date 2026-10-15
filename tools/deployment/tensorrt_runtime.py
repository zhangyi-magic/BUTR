"""TensorRT runtime for fixed detectors and exact dynamic-budget BUTR pipelines."""

import json
from pathlib import Path

import numpy as np
import torch


class TensorRTRunner:
    def __init__(
        self, engine_path, device="cuda:0", max_queries=1500, *, runtime_mode="sync", _shared=None
    ):
        import tensorrt as trt

        if runtime_mode not in ("sync", "async"):
            raise ValueError("TensorRTRunner requires sync or async execution.")
        self.runtime_mode = runtime_mode
        self.runtime_api = "execute_v2" if runtime_mode == "sync" else "execute_async_v3"
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        if _shared is None:
            self.logger = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(self.logger, "")
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        else:
            self.runtime, self.engine, self.logger = _shared
        if self.engine is None:
            raise RuntimeError(f"Cannot load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        # Detailed engine metadata must not add NVTX instrumentation to inference.
        # Apply the same choice to fixed-query and split dynamic-query engines.
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.names = list(self.engine)
        self.input_names = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.input_shapes = {
            name: tuple(self.engine.get_tensor_shape(name)) for name in self.input_names
        }
        self.dtypes = {
            name: torch.from_numpy(
                np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            for name in self.names
        }
        self.host_names = {
            name
            for name in self.names
            if self.engine.get_tensor_location(name) == trt.TensorLocation.HOST
        }
        self.shape_input_names = [
            name for name in self.input_names if self.engine.is_shape_inference_io(name)
        ]
        if any(name not in self.host_names for name in self.shape_input_names):
            raise ValueError("This runtime requires shape inputs in host memory.")
        self.buffers, self.shapes = {}, {}
        self.dynamic_outputs = []
        for name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            capacity = tuple(
                max_queries if dim == -1 and axis == 1 else dim for axis, dim in enumerate(shape)
            )
            if any(dim < 0 for dim in capacity):
                raise ValueError(f"Unbounded output {name}: {shape}")
            self.buffers[name] = torch.empty(
                capacity,
                dtype=self.dtypes[name],
                device="cpu" if name in self.host_names else self.device,
            )
            if -1 in shape:
                self.dynamic_outputs.append(name)
            else:
                self.shapes[name] = shape
        self.addresses = {name: buffer.data_ptr() for name, buffer in self.buffers.items()}
        self.bound_addresses = {}
        self.bindings = [self.addresses.get(name, 0) for name in self.names]
        self.last_shape_values = None
        self.output_views = None
        self.stream = torch.cuda.Stream(device=self.device) if runtime_mode == "async" else None

    def input_shape(self, name):
        return self.input_shapes[name]

    def input_dtype(self, name):
        return self.dtypes[name]

    def _validate_inputs(self, inputs):
        for name in self.input_names:
            tensor = inputs[name]
            device = torch.device("cpu") if name in self.host_names else self.device
            if (
                tensor.device != device
                or not tensor.is_contiguous()
                or tensor.dtype != self.dtypes[name]
            ):
                raise ValueError(
                    f"Input {name} must be contiguous, {self.dtypes[name]}, on {device}."
                )
            if tuple(tensor.shape) != self.input_shape(name):
                raise ValueError(
                    f"Input {name} has wrong shape: {tensor.shape} != {self.input_shape(name)}"
                )

    def _shape_values(self, inputs):
        return tuple(
            (name, tuple(inputs[name].flatten().tolist())) for name in self.shape_input_names
        )

    def __call__(self, inputs):
        self._validate_inputs(inputs)
        for name in self.input_names:
            tensor = inputs[name]
            self.addresses[name] = tensor.data_ptr()
        for index, name in enumerate(self.names):
            address = self.addresses[name]
            # Notify TensorRT of host shape inputs even when their addresses
            # remain stable: their VALUES can change between consecutive calls.
            if self.bound_addresses.get(name) != address or name in self.shape_input_names:
                self.context.set_tensor_address(name, address)
                self.bound_addresses[name] = address
                self.bindings[index] = address
        shape_values = self._shape_values(inputs)
        if self.dynamic_outputs and shape_values != self.last_shape_values:
            missing = self.context.infer_shapes()
            if missing:
                raise RuntimeError(f"TensorRT shape inputs are not specified: {missing}")
            for name in self.dynamic_outputs:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(
                        f"Unresolved data-dependent output {name}; use the split BUTR pipeline."
                    )
                if int(np.prod(shape)) > self.buffers[name].numel():
                    raise RuntimeError(f"Output {name} exceeds --max-queries capacity.")
                self.shapes[name] = shape
            self.last_shape_values = shape_values
            self.output_views = None
        if self.stream is not None:
            current = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(current)
            succeeded = self.context.execute_async_v3(self.stream.cuda_stream)
            current.wait_stream(self.stream)
        else:
            succeeded = self.context.execute_v2(self.bindings)
        if not succeeded:
            raise RuntimeError("TensorRT execution failed.")

    def outputs(self):
        if self.output_views is None:
            self.output_views = {
                name: self.buffers[name]
                .flatten()[: int(np.prod(self.shapes[name]))]
                .view(self.shapes[name])
                for name in self.output_names
            }
        return self.output_views


class CUDAGraphRunner(TensorRTRunner):
    """Capture a fixed shape; use a separate context for other exact budgets."""

    def __init__(self, *args, shape_values=None, **kwargs):
        super().__init__(*args, runtime_mode="async", **kwargs)
        self.runtime_mode = "cuda_graph"
        self.runtime_api = "CUDA Graph replay + execute_async_v3 for other exact query counts"
        self.capture_shape_values = shape_values
        self.graph = None
        self.static_inputs = None
        self.active = self
        # A captured context must never be reshaped: its workspace addresses
        # are embedded in the graph. The fallback shares weights, not workspace.
        self.fallback = (
            TensorRTRunner(
                *args,
                runtime_mode="async",
                _shared=(self.runtime, self.engine, self.logger),
                **kwargs,
            )
            if self.dynamic_outputs
            else None
        )

    def __call__(self, inputs):
        self._validate_inputs(inputs)
        shape_values = self._shape_values(inputs)
        if self.capture_shape_values is None:
            self.capture_shape_values = shape_values
        if self.fallback is not None and shape_values != self.capture_shape_values:
            self.active = self.fallback
            self.fallback(inputs)
            return
        self.active = self
        if self.static_inputs is None:
            self.static_inputs = {}
            for name, tensor in inputs.items():
                # Encoder outputs already have stable storage. Image and
                # target-size inputs supplied by callers need private buffers.
                self.static_inputs[name] = (
                    tensor
                    if name not in ("images", "orig_target_sizes") and name not in self.host_names
                    else tensor.clone()
                )
        for name, tensor in inputs.items():
            if self.static_inputs[name].data_ptr() != tensor.data_ptr():
                self.static_inputs[name].copy_(tensor)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        if self.graph is None:
            # The first enqueue after setting a shape performs work that must
            # happen outside stream capture. It is part of benchmark warmup.
            super().__call__(self.static_inputs)
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT CUDA Graph capture failed.")
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        current.wait_stream(self.stream)

    def outputs(self):
        return super().outputs() if self.active is self else self.active.outputs()


def _runner(path, device, maximum, runtime_mode, shape_values=None):
    if runtime_mode == "cuda_graph":
        return CUDAGraphRunner(path, device, maximum, shape_values=shape_values)
    return TensorRTRunner(path, device, maximum, runtime_mode=runtime_mode)


class BUTRTensorRTPipeline:
    def __init__(self, manifest_path, device="cuda:0", max_queries=None, *, runtime_mode="sync"):
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("format") != "butr-tensorrt-split-v1":
            raise ValueError("Unsupported TensorRT pipeline manifest.")
        maximum = self.manifest["query_profile"][-1]
        if max_queries is not None and max_queries < maximum:
            raise ValueError("max-queries is smaller than the engine query profile.")
        self.runtime_mode = runtime_mode
        self.encoder = _runner(
            path.parent / self.manifest["encoder"], device, maximum, runtime_mode
        )
        common_shape = ((self.manifest["budget_tensor"], (self.manifest["query_profile"][0],)),)
        self.decoder = _runner(
            path.parent / self.manifest["decoder"], device, maximum, runtime_mode, common_shape
        )
        self.runtime_api = self.decoder.runtime_api
        self.host_buffers = {
            name: torch.empty(
                self.decoder.input_shape(name),
                dtype=self.decoder.input_dtype(name),
                pin_memory=True,
            )
            for name in self.decoder.input_names
            if name in self.decoder.host_names
        }

    def input_shape(self, name):
        return (self.encoder if name in self.encoder.input_names else self.decoder).input_shape(
            name
        )

    def input_dtype(self, name):
        return (self.encoder if name in self.encoder.input_names else self.decoder).input_dtype(
            name
        )

    def __call__(self, inputs):
        self.encoder({name: inputs[name] for name in self.encoder.input_names})
        available = {**inputs, **self.encoder.outputs()}
        decoder_inputs = {}
        for name in self.decoder.input_names:
            tensor = available[name]
            if name in self.host_buffers:
                self.host_buffers[name].copy_(tensor)
                tensor = self.host_buffers[name]
            decoder_inputs[name] = tensor.to(dtype=self.decoder.input_dtype(name)).contiguous()
        self.decoder(decoder_inputs)

    def outputs(self):
        return self.decoder.outputs()


def load_runner(path, device="cuda:0", max_queries=1500, *, runtime_mode="sync"):
    if Path(path).suffix.lower() == ".json":
        return BUTRTensorRTPipeline(path, device, max_queries, runtime_mode=runtime_mode)
    return _runner(path, device, max_queries, runtime_mode)
