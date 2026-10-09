# TensorRT FP16 deployment and latency

Run these commands on Linux with a CUDA GPU, from the BUTR repository root. Use the existing training environment and install the optional deployment dependencies:

```bash
python -m pip install -r tools/deployment/requirements.txt \
  -r tools/deployment/requirements-tensorrt.txt
```

The benchmark targets TensorRT **10.4.0**, batch size **1**, and FP16-enabled engines, following the [D-FINE](https://github.com/Peterande/D-FINE/blob/master/tools/benchmark/trt_benchmark.py) and [DEIM](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/tools/benchmark/trt_benchmark.py) timing protocol. The TensorRT pip package supplies its own CUDA libraries; these can differ from PyTorch's CUDA build. Use a driver compatible with both.

## Build a dynamic BUTR pipeline

Export the trained checkpoint with the matching YAML:

```bash
python tools/deployment/export_onnx.py -c configs/models/BUTR-L-AITOD.yml \
  -r output/butr_l_aitodv2/best_stg2.pth -o output/butr_l_aitodv2/model.onnx

python tools/deployment/split_tensorrt.py \
  --onnx output/butr_l_aitodv2/model.onnx \
  --output-dir output/butr_l_aitodv2/trt --query-profile 300 600 1500

CUDA_VISIBLE_DEVICES=0 python tools/deployment/build_tensorrt.py \
  --onnx output/butr_l_aitodv2/trt/encoder.onnx \
  -o output/butr_l_aitodv2/trt/encoder.engine --fp32-sensitive

CUDA_VISIBLE_DEVICES=0 python tools/deployment/build_tensorrt.py \
  --onnx output/butr_l_aitodv2/trt/decoder.onnx \
  -o output/butr_l_aitodv2/trt/decoder.engine \
  --pipeline output/butr_l_aitodv2/trt/pipeline.json --fp32-sensitive
```

The query profile is `MIN OPT MAX`; its endpoints must cover the YAML's minimum and maximum budgets. `OPT` selects a build optimization point without rounding runtime budgets. Adjust these values for other configurations. The current TensorRT TopK limit is 3840; a model requiring more queries needs another implementation.

ONNX retains the input-dependent budget, but TensorRT 10.4 cannot construct a single engine when the proposal TopK count depends on floating-point image computation. The splitter preserves the graph's operations in two engines: the encoder computes features and the exact budget; the decoder receives that budget as a host shape input. Feature tensors stay on the GPU. Neither the query count nor the decoder is padded to the maximum budget.

`--fp32-sensitive` preserves normalization, decoder operations other than convolution/matrix multiplication, and detection post-processing in FP32. Convolution and matrix multiplication remain eligible for FP16. Tiny-object boxes can be sensitive to rounding; an FP16-enabled engine can use mixed precision. The build sidecar records every precision constraint, GPU, versions, profile, and graph hash. `--workspace-gib` optionally limits builder workspace; keep this setting consistent across comparisons.

Rebuild both engines on the target GPU. An engine built on an RTX 4090 is not a portable T4 artifact.

## Verify detections

```bash
CUDA_VISIBLE_DEVICES=0 python tools/deployment/verify_tensorrt.py \
  -c configs/models/BUTR-L-AITOD.yml -r output/butr_l_aitodv2/best_stg2.pth \
  --engine output/butr_l_aitodv2/trt/pipeline.json \
  --images /path/to/AI-TOD/images/test --limit 32 \
  --output output/butr_l_aitodv2/trt/verification.json
```

This compares budgets and confident detection sets against deployed PyTorch FP32, allowing permutations and numerical differences within the reported tolerances. Use images of varying density. The default budget tolerance is one query because FP16 rounding near an integer can change `ceil`; it does not permit fixed query counts. The check records failures and returns a nonzero exit code if any image exceeds the tolerances. FP16 can change individual selected proposals and scores even when AP remains close; use the paired AP evaluation below to assess detection accuracy. This spot check does not measure dataset AP.

For a paired AP evaluation using the configuration's validation transforms and evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/deployment/verify_tensorrt.py \
  -c configs/models/BUTR-L-AITOD.yml -r output/butr_l_aitodv2/best_stg2.pth \
  --engine output/butr_l_aitodv2/trt/pipeline.json \
  --images /path/to/AI-TOD/images/test \
  --annotations /path/to/AI-TOD/annotations/aitodv2_test.json --limit 0 \
  --output output/butr_l_aitodv2/trt/ap_comparison.json
```

`--limit 0` evaluates the full dataset. A positive limit selects the same first images for both runtimes and is useful for a smoke test. AP uses all returned detections; numerical spot-check thresholds are not applied. Check both AP and query usage for each checkpoint and target GPU before reporting deployment results.

## Measure latency

For an annotated image (the same command accepts video input):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inference/trt_inf.py \
  --engine output/butr_l_aitodv2/trt/pipeline.json \
  -i /path/to/image.png -o output/trt_detections.png
```

Video detection runs independently per frame; the saved video's frame rate does not measure inference speed. Use the benchmark for latency:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark/trt_benchmark.py \
  --engine output/butr_l_aitodv2/trt/pipeline.json \
  --infer-dir /path/to/AI-TOD/images/test --preprocess resize \
  --output output/butr_l_aitodv2/trt/latency.json
```

The defaults perform 1000 warmup calls on a synthetic tensor, followed by 1000 real images. CUDA synchronization brackets each `time.time()` measurement. Timing includes both engine calls, host budget transfer, shape resolution, and exported detection post-processing. Image loading, decoding, resizing, and input transfer occur outside the measured interval, matching the upstream timing boundary. The reported mean uses all measured images; there is no fastest-subset filter.

### Optimized execution

The default `--runtime sync` retains the upstream `execute_v2` API. The runtime reuses output views, bindings, and a pinned host budget buffer, and resolves output shapes when the budget changes. Shape input values remain dynamic even when their host addresses are reused.

For CUDA Graph execution, add `--runtime cuda_graph` to the benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark/trt_benchmark.py \
  --engine output/butr_l_aitodv2/trt/pipeline.json \
  --infer-dir /path/to/AI-TOD/images/test --preprocess resize \
  --runtime cuda_graph --output output/butr_l_aitodv2/trt/latency_graph.json
```

The encoder runs through a captured CUDA Graph. The decoder captures the exact minimum budget specified by the pipeline's query profile; other budgets use asynchronous TensorRT execution in a separate context. This preserves every query count without rounding or padding. Both contexts share engine weights; the additional context and stable graph buffers consume extra GPU memory. Graph capture occurs on first use and is normally covered by warmup.

`--runtime async` uses asynchronous execution without graph capture. Both optimized modes use non-default CUDA streams with explicit dependencies on the caller's stream. The benchmark still synchronizes before and after each measured image. Its timing includes copies into stable graph buffers, budget transfer, fallback execution, and post-processing. The JSON records the execution mode and API.

Use the same execution mode for BUTR and its fixed-query latency baseline. CUDA Graph results have the same timing boundary as the upstream protocol, but use a different execution API; report that choice explicitly. Fixed-query engines can also use `--runtime cuda_graph`. These runtime optimizations reuse existing engines and do not change their precision settings.

The inference and verification commands also accept `--runtime cuda_graph` or `--runtime async`. Validate the chosen mode on the target GPU before reporting results.

BUTR uses its configuration's fixed input resolution and direct resize (`--preprocess resize`). `--preprocess official`, the default, uses the upstream aspect-preserving resize and padding for D-FINE/DEIM. For reproducible image selection, provide `--image-list /path/to/list.txt`, with one path relative to `--infer-dir` per line. The JSON records the selected paths, per-image latency and query count, mean/median/p95 latency, GPU, versions, and engine build metadata.

Use an otherwise idle GPU for reported latency. Keep model size, input resolution, preprocessing, workspace, precision, and image selection explicit; measurements made alongside other GPU jobs only validate the workflow.

## Official D-FINE / DEIM checkpoints

DEIM contains both model families. Export the matching official checkpoint at fixed batch size 1, in the same environment:

```bash
python tools/deployment/export_onnx.py --upstream-root /path/to/DEIM \
  -c configs/deim_dfine/dfine_hgnetv2_s_coco.yml \
  -r /path/to/dfine_s_coco.pth -o output/dfine_s.onnx
```

Build and benchmark the exported ONNX:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/deployment/build_tensorrt.py \
  --onnx output/dfine_s.onnx -o output/dfine_s.engine

CUDA_VISIBLE_DEVICES=0 python tools/benchmark/trt_benchmark.py \
  --engine output/dfine_s.engine --infer-dir /path/to/COCO/val2017 \
  --preprocess official --output output/dfine_s_latency.json
```

Use `configs/deim_dfine/deim_hgnetv2_s_coco.yml` and its matching checkpoint for DEIM-S. These fixed-query baseline commands use a single engine. BUTR's two-engine adaptation retains the same timing boundary and includes its additional runtime costs.
