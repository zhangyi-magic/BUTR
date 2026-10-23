# Training environment and installation checks

The [README](../README.md#installation) provides a minimal installation entry. Core dependency bounds follow official Dome-DETR: PyTorch >= 2.0.1, torchvision >= 0.15.2, and faster-coco-eval >= 1.6.5. The table below records one tested environment, rather than minimum required versions. A compiler with C++17 support is required.

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

The single root [requirements.txt](../requirements.txt) follows Dome-DETR's published version constraints for shared dependencies and includes the additional libraries used by this repository. Install a compatible CUDA-enabled PyTorch/torchvision pair; lower bounds do not mean that arbitrary combinations of those packages are compatible.

To reproduce the reference environment above, use this expanded installation instead of the minimal README commands:

```bash
conda create -n BUTR python=3.10.14 --override-channels \
  -c https://conda.anaconda.org/conda-forge -y &&
conda activate BUTR &&
conda install "pytorch=2.4.0=*cuda11.8*" "torchvision=0.19.0=*cu118*" pytorch-cuda=11.8 \
  numpy=2.2.6 pyyaml=6.0.3 --override-channels \
  -c https://conda.anaconda.org/pytorch \
  -c https://conda.anaconda.org/nvidia \
  -c https://conda.anaconda.org/conda-forge -y &&
python -m pip install -r requirements.txt &&
python -m pip check
```

Full channel URLs together with `--override-channels` bypass unavailable mirror mappings without changing global Conda configuration. Explicit CUDA build constraints prevent CPU-only PyTorch installations. If the environment already exists, skip `conda create` and activate it. Compiler setup is described below.

Transitive dependencies, Conda build identifiers, system libraries, and the NVIDIA driver are not fully pinned.

Torchaudio and packages for unrelated or optional workflows are not required by the main HGNetv2 configurations. For example, the optional `TimmModel` backbone imports `timm` only when selected. ONNX export and CPU correctness checks use the optional dependencies in `tools/deployment/requirements.txt`; see [ONNX usage](usage.md#onnx-export-and-inference). [TensorRT deployment](tensorrt.md) additionally uses `tools/deployment/requirements-tensorrt.txt` and a compatible NVIDIA driver. OpenVINO and FiftyOne tools require their own dependencies; OpenVINO deployment remains unvalidated.

## Build prerequisites

For Python 3.10, the published [faster-coco-eval-aitod 1.0.2 files](https://pypi.org/project/faster-coco-eval-aitod/1.0.2/#files) require building the extension from source. A working C++17 compiler and Python development headers must be available. A Conda Python installation normally includes its Python headers. On Ubuntu, install the compiler if `g++ --version` fails:

```bash
sudo apt-get update && sudo apt-get install -y build-essential &&
g++ --version
```

Without administrator privileges, install a compiler in the existing BUTR environment instead. Conda's [compiler activation scripts](https://conda-forge.org/docs/user/faq/) set `CC` and `CXX`; reactivate the environment after installing the toolchain:

```bash
conda activate BUTR &&
conda install "gxx_linux-64=12" --override-channels \
  -c https://conda.anaconda.org/conda-forge -y &&
conda deactivate &&
conda activate BUTR &&
"$CXX" --version
```

The Conda compiler may have a prefixed executable name, so use `"$CXX"` to check it. Once either compiler installation succeeds, continue from the repository root:

```bash
python -m pip install -r requirements.txt &&
python -m pip check
```

An error such as `No such file or directory: 'g++'` indicates that the build could not find the compiler. Fixing the compiler setup and rerunning pip is sufficient; already installed dependencies do not need to be removed.

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
