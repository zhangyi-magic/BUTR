# Training environment and installation checks

The installation commands in the [README](../README.md#installation) use the following training environment.

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 22.04.5 LTS, x86-64 |
| Python | 3.10.14 |
| PyTorch | 2.4.0, Conda CUDA 11.8 build |
| torchvision | 0.19.0, Conda CUDA 11.8 build |
| NumPy | 2.2.6, installed through Conda |
| faster-coco-eval | 1.7.1 |
| faster-coco-eval-aitod | 1.0.2 |
| GCC | 12.3.0 |
| Training hardware | 8 × NVIDIA GeForce RTX 4090 |

The PyTorch build uses CUDA 11.8 libraries, independently of the system CUDA toolkit. Use a compatible NVIDIA driver. The PyTorch/torchvision pairing follows the [official PyTorch 2.4.0 instructions](https://pytorch.org/get-started/previous-versions/), with explicit channel and CUDA build selection to prevent mixed CPU/CUDA installations.

## Dependency scope

The Conda commands use `--override-channels` to select only the listed public channels, independently of machine-specific defaults. This also avoids pulling in Anaconda default channels that may require a separate Terms of Service acceptance step.

PyTorch and torchvision are selected from the `pytorch` channel with CUDA build constraints. Installing `pytorch-cuda=11.8` alone does not guarantee that the solver selects a CUDA PyTorch build; keep the build constraints in the README installation command.

The single root [requirements.txt](../requirements.txt) pins direct project dependencies. It covers the main training and evaluation pipeline, PyTorch image/video inference, Grain preprocessing, profiling, and the declared dependencies of the DEIM checkout used by the COCO plug-in.

Transitive dependencies, Conda build identifiers, system libraries, and the NVIDIA driver are not fully pinned.

Torchaudio and packages for unrelated or optional workflows are not required by the main HGNetv2 configurations. For example, the optional `TimmModel` backbone imports `timm` only when selected. Legacy ONNX, TensorRT, OpenVINO, and FiftyOne tools require their own dependencies and have not been validated for BUTR's dynamic query count.

## Build prerequisites

For Python 3.10, the published [faster-coco-eval-aitod 1.0.2 files](https://pypi.org/project/faster-coco-eval-aitod/1.0.2/#files) require building the extension from source. A working C++ compiler and Python development headers must be available. On Ubuntu, `build-essential` provides the compiler; a Conda Python installation normally includes its Python headers. Verify that the compiler is accessible before installing the requirements:

```bash
g++ --version
```

The primary BUTR configurations use PyTorch deformable attention and do not require building `src/zoo/models/ops/`. That optional extension has a separate CUDA compilation requirement.

## Check the installed environment

After following the README installation commands, run the following from the BUTR root, with the BUTR environment active:

```bash
python -m pip check
python - <<'PY'
import numpy as np
import torch
import torchvision
import faster_coco_eval
import faster_coco_eval_aitod
from torchvision.ops import nms
from src.core import YAMLConfig

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("NumPy:", np.__version__)
assert torch.cuda.is_available(), "CUDA is unavailable in this environment"
print("GPU:", torch.cuda.get_device_name(0))

# Check the NumPy/PyTorch interface and compiled CUDA torchvision operator.
array = np.zeros((2, 4), dtype=np.float32)
assert np.array_equal(torch.from_numpy(array).numpy(), array)
boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
assert nms(boxes, scores, 0.5).cpu().tolist() == [0]

config = YAMLConfig("configs/models/BUTR-S-AITOD.yml")
config.yaml_cfg["HGNetv2"]["pretrained"] = False
assert config.model is not None
print("Environment checks passed.")
PY
```

These checks cover dependency metadata, project and evaluator imports, NumPy interoperability, a compiled CUDA operator, and model construction.
