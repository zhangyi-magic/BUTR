# TensorRT deployment benchmarks

## Reproduction environment and hardware

The deployment benchmarks run on a single NVIDIA Tesla T4 (16 GB) allocated by Kaggle, using an isolated Python 3.10.14 environment with PyTorch 2.4.0+cu118 and TensorRT 10.4.0. Batch size is 1. The two available T4 GPUs are not combined: each trial uses one GPU. Within each dataset, all model comparisons run sequentially on the same physical GPU; AI-TOD-v2 and VisDrone were measured in separate sessions.

COCO calibration uses 640 × 640 inputs, three feature levels, and 300 fixed queries. AI-TOD-v2 and VisDrone use 800 × 800 inputs, four feature levels for BUTR/D-FINE/DEIM/Dome-DETR, and five for DQ-DETR. These are deployment runtime measurements, separate from the paper accuracy results in the [README](../README.md). The [deployment guide](tensorrt.md) explains export and execution.

D-FINE's deployed decoder depths are 3/4/4 layers for S/M/L on AI-TOD-v2 and 3/4/6 on VisDrone, verified in the measured ONNX graphs. Fixed query counts are 600 and 400 respectively. Cross-dataset latency differences therefore reflect both decoder depth and query count, alongside dataset-specific heads and separately built engines.

## COCO latency calibration

We first remeasured official D-FINE-S/M/L COCO checkpoints with the official plain-FP16 TensorRT build settings and `TRTInference.speed` / `TimeProfiler` / `execute_v2` timing logic. Both the [D-FINE model zoo](https://github.com/Peterande/D-FINE#model-zoo) and [DEIM-D-FINE model zoo](https://github.com/Intellindust-AI-Lab/DEIM#deim-d-fine) publish the same S/M/L latency references shown below; the local calibration measurements use D-FINE checkpoints.

| COCO model | Published D-FINE/DEIM latency | Measured latency | Difference |
| --- | ---: | ---: | ---: |
| D-FINE-S | 3.49 ms | 3.614 ms | +3.57% |
| D-FINE-M | 5.62 ms | 5.802 ms | +3.24% |
| D-FINE-L | 8.07 ms | 8.448 ms | +4.69% |

Each value averages three rounds, each with 1,000 warmup calls and the same 1,000 COCO val2017 images. All images and rounds are retained. Our measured latencies are slightly higher than the published references, by approximately 3.2%–4.7%; the subsequent tiny-object comparisons use the same software environment and input-initialization procedure.

Before importing the GPU inference stack, every worker reads and hash-checks all input JPEGs on the CPU. This shared file pre-read is outside the timed interval and is an additional initialization step beyond the original public script. The official engine construction and inference timing logic are retained. COCO calibration uses plain FP16 builds; tiny-object deployments use the sensitive-FP32 constraints described below.

## Measurement protocol

- Hardware and software: one NVIDIA T4, TensorRT 10.4.0, PyTorch 2.4.0+cu118, batch size 1, and 800 × 800 inputs. Trials execute sequentially on the same physical GPU.
- Image selection: a fixed 1,000-image test subset per dataset, with identical images and ordering for every model within that dataset. AI-TOD-v2 has 14,018 test images. VisDrone uses 1,000 images uniformly selected from the full 1,610-image test split, sorted by image ID, at indices `floor(i * 1609 / 999)` for `i=0..999`; it does not use the 120-image moderate subset. These are timing subsets, not full-test AP evaluations.
- Input initialization: every worker performs the same CPU JPEG pre-read and hash check described above before GPU inference initialization. This step is not included in latency.
- Timing: three rounds per model, each with 1,000 synthetic-input warmup calls followed by 1,000 real images. The table reports the mean of the three round means and their population standard deviation. Every image and round is retained.
- Timer and execution: the upstream D-FINE/DEIM `TRTInference.speed` and synchronized `TimeProfiler` boundary, using synchronous TensorRT `execute_v2`. CUDA Graph execution and asynchronous execution are not used. Runtime NVTX verbosity is disabled uniformly.
- Included work: the complete exported model pipeline, all engine executions, budget readback and synchronization, shape handling, and exported detection post-processing. Image loading, decoding, resizing, normalization, and input transfer are outside the timed interval. This is model-pipeline latency, not file-to-detection wall-clock latency.
- Precision: FP16-enabled engines with normalization, sensitive decoder operations, and post-processing constrained to FP32. DQ-DETR additionally protects bounding-box and sampling-coordinate operations with its validated coordinate precision policy. These are mixed-precision deployments, not all-operator FP16 engines. TF32 is disabled; builder timing iterations are 8.

## AI-TOD-v2 latency and memory

| Model | Latency (ms) | Round SD (ms) | Mean queries | Sampled process GPU peak (MiB) |
| --- | ---: | ---: | ---: | ---: |
| BUTR-S | 13.054 | 0.040 | 325.768 | 672 |
| BUTR-M | 18.780 | 0.105 | 319.542 | 710 |
| BUTR-L | 22.343 | 0.117 | 316.845 | 730 |
| D-FINE-S | 12.195 | 0.016 | 600 | 450 |
| D-FINE-M | 18.252 | 0.124 | 600 | 478 |
| D-FINE-L | 22.021 | 0.087 | 600 | 498 |
| DEIM-S† | 12.218 | 0.010 | 600 | 450 |
| DEIM-M | 18.353 | 0.138 | 600 | 478 |
| DEIM-L | 22.061 | 0.048 | 600 | 498 |
| Dome-DETR-S | 24.680 | 0.017 | 637.523 | 594 |
| Dome-DETR-M | 31.165 | 0.114 | 627.621 | 628 |
| Dome-DETR-L | 34.584 | 0.018 | 644.144 | 648 |
| DQ-DETR | 205.701 | 0.203 | 404 | 1168 |

DQ-DETR has a single model configuration rather than S/M/L variants and is included as its own row.

D-FINE and DEIM use 600 fixed queries. BUTR and Dome-DETR retain dynamic query counts within `[300, 1500]`; DQ-DETR retains its `[300, 500, 900, 1500]` query buckets. DQ-DETR uses five feature levels and ImageNet input normalization; the other families use four feature levels. These configurations differ from the official 640 × 640, three-feature-level, 300-query D-FINE/DEIM COCO models, so their published COCO latency numbers are not directly comparable to this table.

Memory is measured in a separate process per model, after 20 warmup calls and during the same 1,000-image sequence, using 20 ms NVML sampling. It covers whole-process GPU residency, including CUDA/library state, TensorRT weights, contexts, workspace, I/O, and GPU preprocessing buffers. Engine construction and load-time peaks are excluded. It is a sampled inference peak, not an exact allocator high-water mark or a measurement of model weights alone. Memory-phase timings are excluded from the latency table.

## VisDrone latency and memory

The same timing and memory protocols apply to the following VisDrone deployments. BUTR uses the split-before-transpose layout; the other models retain their original layouts.

| Model | Latency (ms) | Round SD (ms) | Mean queries | Sampled process GPU peak (MiB) |
| --- | ---: | ---: | ---: | ---: |
| BUTR-S | 14.101 | 0.004 | 444.523 | 708 |
| BUTR-M | 20.537 | 0.101 | 416.195 | 746 |
| BUTR-L | 25.274 | 0.117 | 400.061 | 776 |
| D-FINE-S | 12.168 | 0.054 | 400 | 508 |
| D-FINE-M | 18.541 | 0.170 | 400 | 536 |
| D-FINE-L | 23.576 | 0.123 | 400 | 562 |
| DEIM-S | 12.030 | 0.031 | 400 | 508 |
| DEIM-M | 18.633 | 0.096 | 400 | 536 |
| DEIM-L | 23.957 | 0.069 | 400 | 562 |
| Dome-DETR-S | 24.176 | 0.084 | 385.231 | 630 |
| Dome-DETR-M | 30.864 | 0.087 | 375.738 | 664 |
| Dome-DETR-L | 35.840 | 0.075 | 372.186 | 690 |
| DQ-DETR | 210.886 | 0.341 | 508.8 | 1208 |

D-FINE and DEIM use 400 fixed queries; BUTR and Dome-DETR use budgets within `[250, 500]`. DQ-DETR uses its original `[300, 500, 900, 1500]` query buckets and the recovered VisDrone checkpoint. For DEIM-S/M, same-scale D-FINE weights are used to benchmark the DEIM inference architecture with its SiLU activation settings; those rows are architecture timing measurements, not DEIM training or accuracy reproduction.

The complete prescribed three-round suite is used for the reported means. An earlier interrupted partial suite is retained separately and is not pooled into those means. All models in the completed suite and the separate DQ-DETR run use the same physical T4 and identical image ordering.

### BUTR layout comparison

Original and split-before-transpose deployments retain the same per-image budgets and share the same precision policy and timing boundary.

| Model | Original latency | Split-before-transpose latency | Original sampled GPU peak | Split-before-transpose sampled GPU peak |
| --- | ---: | ---: | ---: | ---: |
| BUTR-S | 19.558 ms | 14.101 ms | 690 MiB | 708 MiB |
| BUTR-M | 25.964 ms | 20.537 ms | 730 MiB | 746 MiB |
| BUTR-L | 30.657 ms | 25.274 ms | 758 MiB | 776 MiB |

On these VisDrone deployments, the layout rewrite reduces latency by approximately 5.4 ms and increases the sampled inference memory peak by 16–18 MiB. These are deployment-specific measurements, not a universal memory cost of the mathematical rewrite.

## Deployment layout and accuracy scope

BUTR uses two engines with the original per-image budget passed to the decoder as a host shape input. Its decoder uses a split-before-transpose layout: split query-independent value features by spatial scale first, then transpose each scale, instead of transposing the entire concatenated feature tensor before splitting. The budget calculation, query counts, weights, and decoder mathematics are preserved. This changes TensorRT fusion and tactic selection; it is a graph-layout optimization rather than query rounding or padding.

D-FINE/DEIM retain their original single-engine layouts. Dome-DETR and DQ-DETR use dynamic front/back deployments, with both engine calls and budget synchronization included in latency. Dome-DETR's density/NMS and query-initialization work remains included. The public [splitter](../tools/deployment/split_tensorrt.py) now applies the measured BUTR layout rewrite by default, with `--decoder-layout original` available for controls. The [standalone optimizer](../tools/deployment/optimize_value_layout.py) applies the same rewrite to full fixed-query graphs.

## Layout fairness controls

We applied the same split-before-transpose operator rewrite to the complete D-FINE-S and DEIM-S single-engine graphs on AI-TOD-v2, preserving weights, 600 fixed queries, shape-control expressions and graph I/O. These controls use the same T4, precision policy, workspace, builder timing iterations, image order and inference timer. Each result averages three complete rounds; an independent original-graph rebuild checks ordinary engine-build variation.

| Model | Original engine | Rebuilt original graph | Split-before-transpose |
| --- | ---: | ---: | ---: |
| D-FINE-S | 12.097 ms | 12.138 ms | 15.173 ms |
| DEIM-S† | 12.209 ms | 12.226 ms | 15.321 ms |

The rewritten versions are approximately 25% slower than the rebuilt original graphs. These are a separate control suite, so their round means differ slightly from the main timing table. We retain the original baseline layouts because they perform better in these controls; applying the same graph rewrite universally would slow the tested baselines. This evidence covers AI-TOD-v2 S-scale controls and does not establish the outcome for every scale or dataset.

### Which feature is rewritten?

The operation is the decoder cross-attention value layout, not the budgeter's confidence summation. At 800 × 800 with four feature levels, raw encoder memory has shape `[1, 53125, 256]`. The layout first reshapes it to `[1, 53125, 8, 32]`. The original graph transposes the whole tensor to `[1, 8, 32, 53125]` before slicing the four spatial scales; split-first slices the spatial-token axis first and transposes each scale afterward. Both versions yield the same per-scale values.

The split BUTR boundary also retains normalized/projected encoder memory of shape `[1, 53125, 256]` for Top-K query-content initialization. The budgeter consumes its classification confidences; it does not perform this value-layout transpose. The additional large boundary buffer is therefore not a feature created exclusively for confidence summation. Fixed-query D-FINE also computes raw and projected encoder features, but its single engine can keep them internal, whereas this BUTR split requires both as explicit encoder outputs/decoder inputs.

Each full FP32 feature tensor is about 51.9 MiB. Keeping both at the engine boundary affects residency and buffer lifetime; whole-process memory additionally includes contexts, workspace, weights, I/O and allocator behavior. This boundary cost already exists in the original split BUTR deployment. It is distinct from the 16–18 MiB increase measured when applying split-first in the VisDrone controls.

### Why does the rewrite help BUTR but hurt these baselines?

An equivalent ONNX expression need not compile into an equally fast TensorRT implementation. Fusion, tensor layouts, intermediate materialization and tactic selection depend on the surrounding graph, precision and engine I/O; see the [TensorRT 10.4 Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1040/pdf/TensorRT-Developer-Guide.pdf) for fusion conditions. A two-engine dynamic decoder receiving explicit feature buffers is a different compilation context from a complete fixed-query single-engine graph whose feature producers are internal.

The larger, explicit BUTR boundary buffers help explain why its compilation context and memory lifetimes differ, but do not by themselves prove the source of the speed difference. The rewrite experiments and rebuild controls establish the observed performance reversal; they do not isolate a unique compiler-level cause. The optimization should be selected by measurement on each deployment, with detection accuracy checked after rebuilding engines.

On AI-TOD-v2, paired original/split-before-transpose BUTR inference on the same 1,000-image subset retained identical query counts for every image. Subset AP changes were −0.014, −0.032, and +0.002 points for S/M/L. Strict numerical matching did not pass for all images; these subset checks do not establish full-test deployment AP equivalence. The paper AP table should not be interpreted as an FP16 accuracy evaluation of these measured engines.

On the VisDrone timing subset, paired AP500 changes were +0.00006, +0.01683, and −0.00538 points for S/M/L. AP500 uses the original category IDs and `maxDets=[1,10,100,500]`, with mean precision over IoU .50:.95 and area=all at maxDet=500. Budgets agree for every image; original and candidate predictions each repeat bitwise. Strict numerical checks still differ on 11, 1,000, and 1,000 images respectively, so the subset AP evidence is not a claim of exact output equality or full-test AP reproduction.

† The AI-TOD-v2 DEIM-S engine has an unresolved deployment-accuracy mismatch. Its latency and memory describe that engine's execution and remain provisional for an accuracy-validated model comparison. The other baseline deployment APs were not re-evaluated in this benchmark.
